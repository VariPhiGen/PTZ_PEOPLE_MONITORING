/**
 * Core routing / proxy logic.
 *
 * Pattern                               Strategy          Notes
 * ─────────────────────────────────────────────────────────────────────────────
 * ALL /api/cameras                      fan-out union     client's nodes
 * ALL /api/sessions/start               any client node   creates routing entry
 * ALL /api/sessions/:id/*               single node       lookup session→node
 * POST /api/search/face                 least-loaded      GPU-heavy
 * GET  /api/search/:id/journey          fan-out merge     cross-node merge
 * ALL  /api/search/*                    any client node
 * ALL  /api/analytics/*                 D1 direct         served by CP, not nodes
 * PUT  /api/settings/*                  KV + broadcast    write + fan-out
 * ALL  /api/enrollment/*                any client node   + post-enroll sync
 * ALL  /api/admin/*                     any node          SUPER_ADMIN only
 * ALL  /api/auth/*                      any node          no client filter
 */
import { Hono } from "hono";
import type { Env } from "../types.js";
import { extractBearer, verifyJWT } from "../lib/jwt.js";
import {
  anyClientNode,
  anyNode,
  fanOutToNodes,
  leastLoadedNode,
  proxyToNode,
} from "../lib/fanout.js";

export const proxyRouter = new Hono<{ Bindings: Env }>();

// ── Auth middleware ───────────────────────────────────────────────────────────

/**
 * Verify the JWT and return the payload, or return a 401 Response.
 */
async function auth(
  authHeader: string | null | undefined,
  secret: string,
) {
  const token = extractBearer(authHeader);
  if (!token) return null;
  return verifyJWT(token, secret);
}

/**
 * Build a forwarded RequestInit, preserving the original headers + body.
 * Body is cloned so the original stream is not consumed.
 */
async function forwardInit(req: Request): Promise<RequestInit> {
  const headers = new Headers(req.headers);
  // Remove hop-by-hop headers
  headers.delete("host");
  headers.delete("connection");
  headers.delete("transfer-encoding");

  const hasBody = !["GET", "HEAD"].includes(req.method);
  const body    = hasBody ? await req.arrayBuffer() : undefined;

  return {
    method:  req.method,
    headers,
    body,
    redirect: "follow",
  };
}

function pathWithQuery(url: URL): string {
  return url.pathname + url.search;
}

function jsonError(message: string, status: number): Response {
  return new Response(JSON.stringify({ error: message }), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

// ── /api/cameras/** ───────────────────────────────────────────────────────────

proxyRouter.all("/cameras/*", async (c) => {
  const payload = await auth(c.req.header("Authorization"), c.env.JWT_SECRET);
  if (!payload) return jsonError("Unauthorized", 401);

  const clientId = payload.client_id;
  if (!clientId && payload.role !== "SUPER_ADMIN") {
    return jsonError("No tenant context", 403);
  }

  // SUPER_ADMIN with an explicit ?client_id= param
  const effectiveClientId =
    clientId ?? new URL(c.req.url).searchParams.get("client_id");
  if (!effectiveClientId) return jsonError("client_id required", 400);

  const init = await forwardInit(c.req.raw);
  const path = pathWithQuery(new URL(c.req.url));

  // For write operations (POST/PUT/DELETE), route to a specific node based on
  // KV camera→node routing. For GETs we fan-out to get a complete view.
  if (c.req.method === "GET") {
    const result = await fanOutToNodes(
      c.env, effectiveClientId, path, init, "union",
    );
    return c.json({ ...result.data as object, _cp: { nodeCount: result.nodeCount, partial: result.partial } });
  }

  // Mutating operations: route to the camera's assigned node
  const camId = c.req.url.match(/\/api\/cameras\/([^/]+)/)?.[1];
  if (camId) {
    const assigned = await c.env.ACAS_ROUTING.get(`cam:${camId}`);
    if (assigned) {
      const node = await c.env.DB.prepare("SELECT * FROM nodes WHERE node_id = ?")
        .bind(assigned)
        .first<{ api_endpoint: string }>();
      if (node) {
        const resp = await proxyToNode(node.api_endpoint, path, init);
        return new Response(resp.body, { status: resp.status, headers: resp.headers });
      }
    }
  }

  // No specific node found — route to any client node
  const node = await anyClientNode(c.env.DB, effectiveClientId);
  if (!node) return jsonError("No nodes available for this client", 503);

  const resp = await proxyToNode(node.api_endpoint, path, init);

  // On successful camera creation, sync assignment to D1+KV
  if (c.req.method === "POST" && resp.status === 201) {
    syncCameraAssignment(c.env, resp.clone(), node.node_id, effectiveClientId).catch(() => {});
  }

  return new Response(resp.body, { status: resp.status, headers: resp.headers });
});

// ── /api/sessions/start ───────────────────────────────────────────────────────

proxyRouter.post("/sessions/start", async (c) => {
  const payload = await auth(c.req.header("Authorization"), c.env.JWT_SECRET);
  if (!payload) return jsonError("Unauthorized", 401);

  const clientId = payload.client_id;
  if (!clientId && payload.role !== "SUPER_ADMIN") {
    return jsonError("No tenant context", 403);
  }

  const body = await c.req.json<{ camera_id?: string; client_id?: string }>();
  const effectiveClientId = clientId ?? body.client_id;
  if (!effectiveClientId) return jsonError("client_id required", 400);

  // Determine which node owns this camera
  let node;
  if (body.camera_id) {
    const nodeId = await c.env.ACAS_ROUTING.get(`cam:${body.camera_id}`);
    if (nodeId) {
      node = await c.env.DB.prepare("SELECT * FROM nodes WHERE node_id = ?")
        .bind(nodeId)
        .first<{ node_id: string; api_endpoint: string }>();
    }
  }
  node ??= await anyClientNode(c.env.DB, effectiveClientId);
  if (!node) return jsonError("No nodes available", 503);

  const init = await forwardInit(c.req.raw);
  // Re-inject the parsed body (it was consumed above)
  const bodyWithClientId = { ...body, client_id: effectiveClientId };
  init.body = JSON.stringify(bodyWithClientId);

  const resp = await proxyToNode(node.api_endpoint, "/api/sessions/start", init);

  // Write session→node routing on success
  if (resp.status === 201) {
    syncSessionRouting(c.env, resp.clone(), node.node_id, effectiveClientId).catch(() => {});
  }

  return new Response(resp.body, { status: resp.status, headers: resp.headers });
});

// ── /api/sessions/:id/** ──────────────────────────────────────────────────────

proxyRouter.all("/sessions/:id/*", async (c) => {
  const payload = await auth(c.req.header("Authorization"), c.env.JWT_SECRET);
  if (!payload) return jsonError("Unauthorized", 401);

  const sessionId = c.req.param("id");
  const route     = await lookupSession(c.env, sessionId);
  if (!route) return jsonError("Session not found or not routable", 404);

  const init = await forwardInit(c.req.raw);
  const path = pathWithQuery(new URL(c.req.url));
  const resp = await proxyToNode(route.api_endpoint, path, init);
  return new Response(resp.body, { status: resp.status, headers: resp.headers });
});

// ── WebSocket relay: /api/sessions/:id/ws ─────────────────────────────────────

proxyRouter.get("/sessions/:id/ws", async (c) => {
  const sessionId = c.req.param("id");
  const token     = new URL(c.req.url).searchParams.get("token");
  if (!token) return jsonError("token query param required", 400);

  const payload = await verifyJWT(token, c.env.JWT_SECRET);
  if (!payload) return jsonError("Invalid token", 401);

  const upgradeHeader = c.req.header("Upgrade");
  if (upgradeHeader?.toLowerCase() !== "websocket") {
    return jsonError("WebSocket upgrade required", 426);
  }

  // Route through SessionRelay Durable Object
  const doId = c.env.SESSION_RELAY.idFromName(sessionId);
  const stub  = c.env.SESSION_RELAY.get(doId);

  const relayUrl = new URL(c.req.url);
  relayUrl.searchParams.set("session_id", sessionId);

  return stub.fetch(relayUrl.toString(), c.req.raw);
});

// ── /api/search/face ──────────────────────────────────────────────────────────

proxyRouter.post("/search/face", async (c) => {
  const payload = await auth(c.req.header("Authorization"), c.env.JWT_SECRET);
  if (!payload) return jsonError("Unauthorized", 401);

  const clientId = payload.client_id;
  if (!clientId) return jsonError("No tenant context", 403);

  const node = await leastLoadedNode(c.env.DB, clientId);
  if (!node) return jsonError("No nodes available", 503);

  const init = await forwardInit(c.req.raw);
  const resp = await proxyToNode(node.api_endpoint, "/api/search/face", init);
  return new Response(resp.body, { status: resp.status, headers: resp.headers });
});

proxyRouter.post("/search/face/base64", async (c) => {
  const payload = await auth(c.req.header("Authorization"), c.env.JWT_SECRET);
  if (!payload) return jsonError("Unauthorized", 401);

  const clientId = payload.client_id;
  if (!clientId) return jsonError("No tenant context", 403);

  const node = await leastLoadedNode(c.env.DB, clientId);
  if (!node) return jsonError("No nodes available", 503);

  const init = await forwardInit(c.req.raw);
  const resp = await proxyToNode(node.api_endpoint, "/api/search/face/base64", init);
  return new Response(resp.body, { status: resp.status, headers: resp.headers });
});

// ── /api/search/:id/journey — fan-out across all client nodes ─────────────────

proxyRouter.get("/search/:id/journey", async (c) => {
  const payload = await auth(c.req.header("Authorization"), c.env.JWT_SECRET);
  if (!payload) return jsonError("Unauthorized", 401);

  const clientId = payload.client_id;
  if (!clientId) return jsonError("No tenant context", 403);

  const init   = await forwardInit(c.req.raw);
  const path   = pathWithQuery(new URL(c.req.url));
  const result = await fanOutToNodes(c.env, clientId, path, init, "merge_journey");

  return c.json({
    ...result.data as object,
    _cp: { nodeCount: result.nodeCount, partial: result.partial },
  });
});

// ── /api/search/* (other) ─────────────────────────────────────────────────────

proxyRouter.all("/search/*", async (c) => {
  const payload = await auth(c.req.header("Authorization"), c.env.JWT_SECRET);
  if (!payload) return jsonError("Unauthorized", 401);

  const clientId = payload.client_id;
  if (!clientId) return jsonError("No tenant context", 403);

  const node = await anyClientNode(c.env.DB, clientId);
  if (!node) return jsonError("No nodes available", 503);

  const init = await forwardInit(c.req.raw);
  const path = pathWithQuery(new URL(c.req.url));
  const resp = await proxyToNode(node.api_endpoint, path, init);
  return new Response(resp.body, { status: resp.status, headers: resp.headers });
});

// ── /api/enrollment/** ────────────────────────────────────────────────────────

proxyRouter.all("/enrollment/*", async (c) => {
  const payload = await auth(c.req.header("Authorization"), c.env.JWT_SECRET);
  if (!payload) return jsonError("Unauthorized", 401);

  const clientId = payload.client_id;
  if (!clientId && payload.role !== "SUPER_ADMIN") {
    return jsonError("No tenant context", 403);
  }

  const url              = new URL(c.req.url);
  const effectiveClientId = clientId ?? url.searchParams.get("client_id");
  if (!effectiveClientId) return jsonError("client_id required", 400);

  const node = await anyClientNode(c.env.DB, effectiveClientId);
  if (!node) return jsonError("No nodes available", 503);

  const init = await forwardInit(c.req.raw);
  const path = pathWithQuery(url);
  const resp = await proxyToNode(node.api_endpoint, path, init);

  // After successful enrollment, broadcast embedding sync to other nodes
  if (resp.ok && c.req.method === "POST") {
    broadcastEnrollmentSync(c.env, effectiveClientId, node.node_id, path).catch(
      () => {},
    );
  }

  return new Response(resp.body, { status: resp.status, headers: resp.headers });
});

// ── /api/admin/** — SUPER_ADMIN, proxy to any node ────────────────────────────

proxyRouter.all("/admin/*", async (c) => {
  const payload = await auth(c.req.header("Authorization"), c.env.JWT_SECRET);
  if (!payload) return jsonError("Unauthorized", 401);
  if (payload.role !== "SUPER_ADMIN") return jsonError("Forbidden", 403);

  const node = await anyNode(c.env.DB);
  if (!node) return jsonError("No nodes available", 503);

  const init = await forwardInit(c.req.raw);
  const path = pathWithQuery(new URL(c.req.url));
  const resp = await proxyToNode(node.api_endpoint, path, init);
  return new Response(resp.body, { status: resp.status, headers: resp.headers });
});

// ── /api/auth/** — proxy to any node (login, refresh, etc.) ──────────────────

proxyRouter.all("/auth/*", async (c) => {
  const node = await anyNode(c.env.DB);
  if (!node) return jsonError("No nodes available", 503);

  const init = await forwardInit(c.req.raw);
  const path = pathWithQuery(new URL(c.req.url));
  const resp = await proxyToNode(node.api_endpoint, path, init);
  return new Response(resp.body, { status: resp.status, headers: resp.headers });
});

// ── Background helpers ─────────────────────────────────────────────────────────

/** Parse a created camera from the node response and write to D1+KV. */
async function syncCameraAssignment(
  env:       Env,
  resp:      Response,
  nodeId:    string,
  clientId:  string,
): Promise<void> {
  try {
    const data = await resp.json<{ camera_id?: string; name?: string; room_name?: string }>();
    if (!data.camera_id) return;

    await env.DB.prepare(`
      INSERT INTO camera_assignments (camera_id, node_id, client_id, camera_name, room_name)
      VALUES (?, ?, ?, ?, ?)
      ON CONFLICT (camera_id) DO UPDATE SET node_id = excluded.node_id
    `)
      .bind(data.camera_id, nodeId, clientId, data.name ?? null, data.room_name ?? null)
      .run();

    await env.ACAS_ROUTING.put(`cam:${data.camera_id}`, nodeId, {
      expirationTtl: 86_400,
    });
  } catch {
    // Best-effort
  }
}

/** Write session_id → node_id routing after a successful session start. */
async function syncSessionRouting(
  env:      Env,
  resp:     Response,
  nodeId:   string,
  clientId: string,
): Promise<void> {
  try {
    const data = await resp.json<{ session_id?: string; camera_id?: string }>();
    if (!data.session_id) return;

    const node = await env.DB.prepare("SELECT api_endpoint FROM nodes WHERE node_id = ?")
      .bind(nodeId)
      .first<{ api_endpoint: string }>();

    await env.DB.prepare(`
      INSERT INTO session_routing (session_id, camera_id, node_id, client_id)
      VALUES (?, ?, ?, ?)
      ON CONFLICT (session_id) DO UPDATE SET node_id = excluded.node_id
    `)
      .bind(data.session_id, data.camera_id ?? null, nodeId, clientId)
      .run();

    if (node) {
      await env.ACAS_ROUTING.put(`session:${data.session_id}`, node.api_endpoint, {
        expirationTtl: 7_200,
      });
    }
  } catch {
    // Best-effort
  }
}

/** After enrollment, trigger FAISS index sync on all other client nodes. */
async function broadcastEnrollmentSync(
  env:        Env,
  clientId:   string,
  primaryNode: string,
  path:       string,
): Promise<void> {
  const { results } = await env.DB.prepare(`
    SELECT DISTINCT n.node_id, n.api_endpoint
    FROM nodes n
    JOIN camera_assignments ca ON ca.node_id = n.node_id
    WHERE ca.client_id = ?
      AND n.node_id   != ?
      AND n.status     = 'ONLINE'
  `)
    .bind(clientId, primaryNode)
    .all<{ node_id: string; api_endpoint: string }>();

  // Best-effort notify — extract person_id from path if possible
  const personId = path.match(/\/enrollment\/([^/]+)/)?.[1];
  if (!personId || personId === "enroll" || personId === "upload") return;

  await Promise.allSettled(
    results.map((n) =>
      fetch(`${n.api_endpoint}/api/enrollment/${personId}/sync`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ client_id: clientId }),
      }),
    ),
  );
}

/** Look up session routing from KV (fast) then D1 (fallback). */
async function lookupSession(
  env:       Env,
  sessionId: string,
): Promise<{ api_endpoint: string } | null> {
  // KV stores the api_endpoint directly for speed
  const cached = await env.ACAS_ROUTING.get(`session:${sessionId}`);
  if (cached) return { api_endpoint: cached };

  const row = await env.DB.prepare(`
    SELECT n.api_endpoint
    FROM session_routing sr
    JOIN nodes n ON n.node_id = sr.node_id
    WHERE sr.session_id = ? AND n.status = 'ONLINE'
  `)
    .bind(sessionId)
    .first<{ api_endpoint: string }>();

  if (row?.api_endpoint) {
    await env.ACAS_ROUTING.put(`session:${sessionId}`, row.api_endpoint, {
      expirationTtl: 7_200,
    });
  }

  return row ?? null;
}
