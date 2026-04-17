/**
 * SessionRelay — Durable Object that acts as a WebSocket relay between
 * a browser client and a GPU node running PTZBrain.
 *
 * Connection flow:
 *   Browser → CF Worker /api/sessions/:id/ws?token=XXX
 *           → Worker routes to SessionRelay DO (keyed by session_id)
 *           → DO accepts the browser WebSocket
 *           → DO opens an outbound WebSocket to the GPU node
 *           → Messages relay bidirectionally until either side closes
 *
 * The DO stays alive for the duration of the session (no hibernation).
 * Multiple browser clients (e.g. dashboard tabs) can connect to the same
 * session — all receive the same GPU node events.
 */
import type { Env } from "../types.js";

export class SessionRelay implements DurableObject {
  private env: Env;

  /** All accepted browser WebSocket server-sides, keyed by a numeric ID. */
  private clients = new Map<number, WebSocket>();
  private nextClientId = 0;

  /** Outbound WebSocket to the GPU node (one per session). */
  private nodeWS:    WebSocket | null = null;
  private nodeReady  = false;

  /** Messages buffered while the node WS is connecting. */
  private sendBuffer: (string | ArrayBuffer)[] = [];

  constructor(_state: DurableObjectState, env: Env) {
    this.env = env;
  }

  async fetch(request: Request): Promise<Response> {
    const url           = new URL(request.url);
    const upgradeHeader = request.headers.get("Upgrade");

    if (upgradeHeader?.toLowerCase() !== "websocket") {
      return new Response("WebSocket upgrade required", { status: 426 });
    }

    const sessionId = url.searchParams.get("session_id");
    const token     = url.searchParams.get("token");

    if (!sessionId || !token) {
      return new Response("session_id and token are required", { status: 400 });
    }

    // Look up the GPU node endpoint for this session
    const nodeEndpoint = await this.resolveNodeEndpoint(sessionId);
    if (!nodeEndpoint) {
      return new Response("Session not found or no node assigned", { status: 404 });
    }

    // Create the WebSocket pair: clientWS goes to browser, serverWS stays in DO
    const { 0: clientWS, 1: serverWS } = new WebSocketPair();
    serverWS.accept();

    const clientId = this.nextClientId++;
    this.clients.set(clientId, serverWS);

    // Wire up browser → node
    serverWS.addEventListener("message", (event) => {
      this.onClientMessage(event.data);
    });
    serverWS.addEventListener("close", () => {
      this.clients.delete(clientId);
      // If last client disconnected, close the node WS
      if (this.clients.size === 0 && this.nodeWS) {
        this.nodeWS.close(1000, "All clients disconnected");
        this.nodeWS = null;
      }
    });
    serverWS.addEventListener("error", () => {
      this.clients.delete(clientId);
    });

    // Connect to the GPU node if not already connected
    if (!this.nodeWS && !this.nodeReady) {
      this.connectToNode(nodeEndpoint, sessionId, token).catch(() => {
        // Notify all connected clients that the node is unreachable
        for (const ws of this.clients.values()) {
          try {
            ws.send(JSON.stringify({ type: "error", message: "Cannot reach GPU node" }));
            ws.close(1014, "GPU node unreachable");
          } catch { /* ignore */ }
        }
        this.clients.clear();
      });
    }

    return new Response(null, { status: 101, webSocket: clientWS });
  }

  // ── Node connection ─────────────────────────────────────────────────────────

  private async connectToNode(
    apiEndpoint: string,
    sessionId:   string,
    token:       string,
  ): Promise<void> {
    // Convert http(s) to ws(s)
    const wsUrl = apiEndpoint
      .replace(/^https:/, "wss:")
      .replace(/^http:/, "ws:")
      .replace(/\/$/, "");

    const nodeWsUrl = `${wsUrl}/api/sessions/${sessionId}/ws?token=${encodeURIComponent(token)}`;

    const resp = await fetch(nodeWsUrl, {
      headers: { Upgrade: "websocket" },
    });

    if (!resp.webSocket) {
      throw new Error(`Node returned ${resp.status} — no WebSocket`);
    }

    const nodeWS = resp.webSocket;
    nodeWS.accept();
    this.nodeWS    = nodeWS;
    this.nodeReady = true;

    // Flush buffered messages
    for (const msg of this.sendBuffer) {
      try { nodeWS.send(msg); } catch { /* ignore */ }
    }
    this.sendBuffer = [];

    // Node → all clients
    nodeWS.addEventListener("message", (event) => {
      this.broadcast(event.data);
    });

    nodeWS.addEventListener("close", (event) => {
      this.nodeWS    = null;
      this.nodeReady = false;
      // Notify clients and close
      for (const ws of this.clients.values()) {
        try {
          ws.send(JSON.stringify({ type: "node_disconnected", code: event.code }));
          ws.close(1001, "GPU node disconnected");
        } catch { /* ignore */ }
      }
      this.clients.clear();
    });

    nodeWS.addEventListener("error", () => {
      this.nodeWS    = null;
      this.nodeReady = false;
      for (const ws of this.clients.values()) {
        try { ws.close(1014, "GPU node error"); } catch { /* ignore */ }
      }
      this.clients.clear();
    });
  }

  // ── Message helpers ─────────────────────────────────────────────────────────

  private onClientMessage(data: string | ArrayBuffer): void {
    if (!this.nodeReady || !this.nodeWS) {
      this.sendBuffer.push(data);
      return;
    }
    try { this.nodeWS.send(data); } catch { /* ignore */ }
  }

  private broadcast(data: string | ArrayBuffer): void {
    for (const ws of this.clients.values()) {
      try { ws.send(data); } catch { /* ignore */ }
    }
  }

  // ── Node endpoint resolution ────────────────────────────────────────────────

  private async resolveNodeEndpoint(sessionId: string): Promise<string | null> {
    // Fast path: KV stores the api_endpoint directly
    const cached = await this.env.ACAS_ROUTING.get(`session:${sessionId}`);
    if (cached) return cached;

    // Slow path: D1 join
    const row = await this.env.DB.prepare(`
      SELECT n.api_endpoint
      FROM session_routing sr
      JOIN nodes n ON n.node_id = sr.node_id
      WHERE sr.session_id = ? AND n.status = 'ONLINE'
    `)
      .bind(sessionId)
      .first<{ api_endpoint: string }>();

    if (row?.api_endpoint) {
      await this.env.ACAS_ROUTING.put(`session:${sessionId}`, row.api_endpoint, {
        expirationTtl: 7_200,
      });
      return row.api_endpoint;
    }

    return null;
  }
}
