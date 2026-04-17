/**
 * VGI ACFR WebSocket client — wraps a native WebSocket with:
 *  • Automatic exponential-backoff reconnect
 *  • Typed event emitter
 *  • Heartbeat (client ping → server pong)
 *  • Clean lifecycle management (connect / close / destroy)
 *
 * Connects to the Control Plane SessionRelay Durable Object at:
 *   ws(s)://{CP_HOST}/api/sessions/{sessionId}/ws?token={jwt}
 */

type Listener<T = unknown> = (data: T) => void;

export interface WsOptions {
  /** Initial reconnect delay in ms (doubles on each retry, capped at maxDelay) */
  minDelay?:    number;
  maxDelay?:    number;
  maxRetries?:  number;
  /** Client→server ping interval in ms (0 = disabled) */
  pingInterval?: number;
}

const DEFAULT_OPTS: Required<WsOptions> = {
  minDelay:     1_000,
  maxDelay:     30_000,
  maxRetries:   20,
  pingInterval: 25_000,
};

export class AcasWebSocket {
  private ws:          WebSocket | null = null;
  private url:         string;
  private opts:        Required<WsOptions>;
  private retries      = 0;
  private destroyed    = false;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private pingTimer:      ReturnType<typeof setInterval> | null = null;

  // Event listeners keyed by message type (or "*" for all)
  private listeners = new Map<string, Set<Listener>>();

  /** Fired when connection state changes */
  onStatus?: (status: "connecting" | "open" | "closed" | "error") => void;

  constructor(url: string, opts: WsOptions = {}) {
    this.url  = url;
    this.opts = { ...DEFAULT_OPTS, ...opts };
  }

  // ── Public API ──────────────────────────────────────────────────────────────

  connect(): this {
    if (!this.destroyed) this._connect();
    return this;
  }

  close(): void {
    this.destroyed = true;
    this._clearTimers();
    this.ws?.close(1000, "client closed");
    this.ws = null;
  }

  send(data: unknown): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(typeof data === "string" ? data : JSON.stringify(data));
    }
  }

  on<T = unknown>(event: string, listener: Listener<T>): () => void {
    if (!this.listeners.has(event)) this.listeners.set(event, new Set());
    this.listeners.get(event)!.add(listener as Listener);
    return () => this.listeners.get(event)?.delete(listener as Listener);
  }

  get readyState(): number {
    return this.ws?.readyState ?? WebSocket.CLOSED;
  }

  get isOpen(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  // ── Internals ───────────────────────────────────────────────────────────────

  private _connect(): void {
    if (this.destroyed) return;

    this.onStatus?.("connecting");

    try {
      this.ws = new WebSocket(this.url);
    } catch (err) {
      this._scheduleReconnect();
      return;
    }

    this.ws.onopen = () => {
      this.retries = 0;
      this.onStatus?.("open");
      this._emit("__open__", null);
      this._startPing();
    };

    this.ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data as string) as { type?: string };
        const type = msg.type ?? "__message__";
        this._emit(type, msg);
        this._emit("*", msg);
      } catch {
        // Non-JSON frames (raw strings) emitted as-is
        this._emit("__raw__", event.data);
        this._emit("*", event.data);
      }
    };

    this.ws.onclose = (event) => {
      this._stopPing();
      this.onStatus?.("closed");
      this._emit("__close__", { code: event.code, reason: event.reason });
      if (!this.destroyed && event.code !== 1000) {
        this._scheduleReconnect();
      }
    };

    this.ws.onerror = () => {
      this.onStatus?.("error");
      this._emit("__error__", null);
    };
  }

  private _scheduleReconnect(): void {
    if (this.destroyed) return;
    if (this.retries >= this.opts.maxRetries) {
      this._emit("__max_retries__", null);
      return;
    }

    const delay = Math.min(
      this.opts.minDelay * 2 ** this.retries,
      this.opts.maxDelay,
    );
    this.retries++;

    this.reconnectTimer = setTimeout(() => {
      if (!this.destroyed) this._connect();
    }, delay);
  }

  private _startPing(): void {
    if (!this.opts.pingInterval) return;
    this.pingTimer = setInterval(() => {
      this.send({ type: "ping", ts: Date.now() });
    }, this.opts.pingInterval);
  }

  private _stopPing(): void {
    if (this.pingTimer) {
      clearInterval(this.pingTimer);
      this.pingTimer = null;
    }
  }

  private _clearTimers(): void {
    this._stopPing();
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  private _emit(event: string, data: unknown): void {
    this.listeners.get(event)?.forEach((fn) => fn(data));
  }
}

// ── Factory ───────────────────────────────────────────────────────────────────

const CP_URL = process.env.NEXT_PUBLIC_API_URL ?? "";

/**
 * Create a WebSocket connection to a live session.
 * The Control Plane upgrades the request and relays to the GPU node.
 */
export function createSessionWs(
  sessionId: string,
  token:     string,
  opts?:     WsOptions,
): AcasWebSocket {
  const wsBase = CP_URL.replace(/^http/, "ws");
  const url    = `${wsBase}/api/sessions/${sessionId}/ws?token=${encodeURIComponent(token)}`;
  return new AcasWebSocket(url, opts);
}
