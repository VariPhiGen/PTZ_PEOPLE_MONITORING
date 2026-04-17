/**
 * Settings routes — write to KV and broadcast to all client nodes.
 *
 * GET /api/settings/:key      — read from KV
 * PUT /api/settings/:key      — write to KV + broadcast PUT to all client nodes
 * GET /api/settings           — list all keys (prefix scan)
 * DELETE /api/settings/:key   — remove from KV + broadcast DELETE
 */
import { Hono } from "hono";
import type { Env, NodeRow } from "../types.js";
import { extractBearer, verifyJWT } from "../lib/jwt.js";

export const settingsRouter = new Hono<{ Bindings: Env }>();

// ── Auth helper ───────────────────────────────────────────────────────────────

async function authPayload(authHeader: string | null | undefined, secret: string) {
  const token = extractBearer(authHeader);
  if (!token) return null;
  return verifyJWT(token, secret);
}

function requireAdmin(role: string): boolean {
  return role === "SUPER_ADMIN" || role === "CLIENT_ADMIN";
}

// ── GET /api/settings ─────────────────────────────────────────────────────────

settingsRouter.get("/", async (c) => {
  const payload = await authPayload(c.req.header("Authorization"), c.env.JWT_SECRET);
  if (!payload) return c.json({ error: "Unauthorized" }, 401);

  const prefix = payload.role === "SUPER_ADMIN"
    ? "global:"
    : `client:${payload.client_id}:`;

  const list = await c.env.ACAS_CONFIG.list({ prefix });
  const keys = list.keys.map((k) => ({ key: k.name, metadata: k.metadata }));

  return c.json({ prefix, keys });
});

// ── GET /api/settings/:key ────────────────────────────────────────────────────

settingsRouter.get("/:key", async (c) => {
  const payload = await authPayload(c.req.header("Authorization"), c.env.JWT_SECRET);
  if (!payload) return c.json({ error: "Unauthorized" }, 401);

  const key = resolveKey(c.req.param("key"), payload.role, payload.client_id);
  const raw = await c.env.ACAS_CONFIG.get(key);
  if (raw === null) return c.json({ error: "Not found" }, 404);

  let value: unknown;
  try { value = JSON.parse(raw); } catch { value = raw; }

  return c.json({ key, value });
});

// ── PUT /api/settings/:key ────────────────────────────────────────────────────

settingsRouter.put("/:key", async (c) => {
  const payload = await authPayload(c.req.header("Authorization"), c.env.JWT_SECRET);
  if (!payload) return c.json({ error: "Unauthorized" }, 401);
  if (!requireAdmin(payload.role)) return c.json({ error: "Forbidden" }, 403);

  const paramKey = c.req.param("key");
  const key      = resolveKey(paramKey, payload.role, payload.client_id);
  const body     = await c.req.json<{ value: unknown; ttl?: number }>();

  const serialized = JSON.stringify(body.value);
  const opts       = body.ttl ? { expirationTtl: body.ttl } : undefined;

  await c.env.ACAS_CONFIG.put(key, serialized, opts);

  // Broadcast to all relevant nodes
  const nodes = await getRelevantNodes(c.env.DB, payload.role, payload.client_id);
  await broadcastSetting(nodes, paramKey, "PUT", body.value);

  return c.json({ key, saved: true, broadcast: nodes.length });
});

// ── DELETE /api/settings/:key ─────────────────────────────────────────────────

settingsRouter.delete("/:key", async (c) => {
  const payload = await authPayload(c.req.header("Authorization"), c.env.JWT_SECRET);
  if (!payload) return c.json({ error: "Unauthorized" }, 401);
  if (payload.role !== "SUPER_ADMIN") return c.json({ error: "Forbidden" }, 403);

  const paramKey = c.req.param("key");
  const key      = resolveKey(paramKey, payload.role, payload.client_id);

  await c.env.ACAS_CONFIG.delete(key);

  const nodes = await getRelevantNodes(c.env.DB, payload.role, payload.client_id);
  await broadcastSetting(nodes, paramKey, "DELETE", null);

  return c.json({ key, deleted: true, broadcast: nodes.length });
});

// ── Helpers ───────────────────────────────────────────────────────────────────

function resolveKey(
  rawKey:   string,
  role:     string,
  clientId: string | null,
): string {
  if (role === "SUPER_ADMIN") return `global:${rawKey}`;
  return `client:${clientId}:${rawKey}`;
}

async function getRelevantNodes(
  db:       D1Database,
  role:     string,
  clientId: string | null,
): Promise<NodeRow[]> {
  if (role === "SUPER_ADMIN") {
    const { results } = await db
      .prepare("SELECT * FROM nodes WHERE status = 'ONLINE'")
      .all<NodeRow>();
    return results;
  }

  if (!clientId) return [];

  const { results } = await db
    .prepare(`
      SELECT DISTINCT n.*
      FROM nodes n
      JOIN camera_assignments ca ON ca.node_id = n.node_id
      WHERE ca.client_id = ? AND n.status = 'ONLINE'
    `)
    .bind(clientId)
    .all<NodeRow>();

  return results;
}

async function broadcastSetting(
  nodes:  NodeRow[],
  key:    string,
  method: "PUT" | "DELETE",
  value:  unknown,
): Promise<void> {
  if (nodes.length === 0) return;

  await Promise.allSettled(
    nodes.map((n) =>
      fetch(`${n.api_endpoint}/api/node/config/setting/${encodeURIComponent(key)}`, {
        method,
        headers: { "Content-Type": "application/json" },
        body:    method === "PUT" ? JSON.stringify({ value }) : undefined,
      }),
    ),
  );
}
