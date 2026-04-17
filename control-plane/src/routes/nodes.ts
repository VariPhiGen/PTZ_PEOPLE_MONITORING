/**
 * Node management routes
 *
 * POST /api/nodes/register          — register a new GPU node (NODE_AUTH_TOKEN)
 * POST /api/nodes/:id/heartbeat     — receive health updates
 * GET  /api/nodes                   — list nodes (SUPER_ADMIN JWT)
 * PUT  /api/nodes/:id/drain         — gracefully drain a node (SUPER_ADMIN)
 * DELETE /api/nodes/:id             — remove a node (SUPER_ADMIN)
 */
import { Hono } from "hono";
import type { Env, HeartbeatPayload, NodeRow } from "../types.js";
import { extractBearer, verifyJWT } from "../lib/jwt.js";

export const nodesRouter = new Hono<{ Bindings: Env }>();

// ── Auth helpers ──────────────────────────────────────────────────────────────

/** Verify the shared NODE_AUTH_TOKEN for node→CP communication. */
function requireNodeAuth(
  authHeader: string | null | undefined,
  token: string,
): boolean {
  if (!authHeader) return false;
  const provided = authHeader.startsWith("Bearer ")
    ? authHeader.slice(7)
    : authHeader;
  return provided === token;
}

/** Require SUPER_ADMIN JWT. */
async function requireSuperAdmin(
  authHeader: string | null | undefined,
  secret: string,
): Promise<boolean> {
  const token = extractBearer(authHeader);
  if (!token) return false;
  const payload = await verifyJWT(token, secret);
  return payload?.role === "SUPER_ADMIN";
}

// ── POST /api/nodes/register ──────────────────────────────────────────────────

nodesRouter.post("/register", async (c) => {
  if (!requireNodeAuth(c.req.header("Authorization"), c.env.NODE_AUTH_TOKEN)) {
    return c.json({ error: "Unauthorized" }, 401);
  }

  const body = await c.req.json<{
    node_id:      string;
    node_name:    string;
    location?:    string;
    connectivity?: "PUBLIC_IP" | "CLOUDFLARE_TUNNEL";
    api_endpoint: string;
    gpu_model?:   string;
    max_cameras?: number;
  }>();

  if (!body.node_id || !body.api_endpoint) {
    return c.json({ error: "node_id and api_endpoint are required" }, 400);
  }

  const now = Math.floor(Date.now() / 1000);

  await c.env.DB.prepare(`
    INSERT INTO nodes (node_id, node_name, location, connectivity, api_endpoint,
                       gpu_model, max_cameras, active_cameras, status, last_heartbeat)
    VALUES (?, ?, ?, ?, ?, ?, ?, 0, 'ONLINE', ?)
    ON CONFLICT (node_id) DO UPDATE SET
      node_name      = excluded.node_name,
      location       = excluded.location,
      connectivity   = excluded.connectivity,
      api_endpoint   = excluded.api_endpoint,
      gpu_model      = excluded.gpu_model,
      max_cameras    = excluded.max_cameras,
      status         = 'ONLINE',
      last_heartbeat = excluded.last_heartbeat
  `)
    .bind(
      body.node_id,
      body.node_name ?? body.node_id,
      body.location ?? null,
      body.connectivity ?? "PUBLIC_IP",
      body.api_endpoint,
      body.gpu_model ?? null,
      body.max_cameras ?? 10,
      now,
    )
    .run();

  return c.json({ registered: true, node_id: body.node_id, ts: now }, 201);
});

// ── POST /api/nodes/:id/heartbeat ─────────────────────────────────────────────

nodesRouter.post("/:id/heartbeat", async (c) => {
  if (!requireNodeAuth(c.req.header("Authorization"), c.env.NODE_AUTH_TOKEN)) {
    return c.json({ error: "Unauthorized" }, 401);
  }

  const nodeId = c.req.param("id");
  const body   = await c.req.json<HeartbeatPayload>();
  const now    = Math.floor(Date.now() / 1000);

  // Update heartbeat, active_cameras, and health snapshot
  const result = await c.env.DB.prepare(`
    UPDATE nodes
    SET last_heartbeat = ?,
        active_cameras = ?,
        health_json    = ?,
        status         = CASE WHEN status = 'OFFLINE' THEN 'ONLINE' ELSE status END
    WHERE node_id = ?
  `)
    .bind(
      now,
      body.active_cameras ?? 0,
      JSON.stringify(body.health ?? {}),
      nodeId,
    )
    .run();

  if (result.meta.changes === 0) {
    return c.json({ error: "Node not found" }, 404);
  }

  // Sync session routing from the node's active session list
  if (body.session_ids && body.session_ids.length > 0) {
    for (const sid of body.session_ids) {
      await c.env.ACAS_ROUTING.put(`session:${sid}`, nodeId, {
        expirationTtl: 7_200,   // 2 h
      });
      // Upsert into D1 (best-effort; don't fail heartbeat if this errors)
      await c.env.DB.prepare(`
        INSERT INTO session_routing (session_id, node_id, client_id)
        VALUES (?, ?, '')
        ON CONFLICT (session_id) DO UPDATE SET node_id = excluded.node_id
      `)
        .bind(sid, nodeId)
        .run()
        .catch(() => {});
    }
  }

  // Store per-node analytics deltas (pushed by node in heartbeat)
  if (body.analytics) {
    const today = new Date().toISOString().slice(0, 10);
    for (const [metric, value] of Object.entries(body.analytics)) {
      await c.env.DB.prepare(`
        INSERT INTO analytics_aggregate (date, metric, scope, value)
        VALUES (?, ?, ?, ?)
        ON CONFLICT (date, metric, scope) DO UPDATE SET value = excluded.value
      `)
        .bind(today, metric, `node:${nodeId}`, value)
        .run()
        .catch(() => {});
    }
  }

  return c.json({ ok: true, ts: now });
});

// ── GET /api/nodes ─────────────────────────────────────────────────────────────

nodesRouter.get("/", async (c) => {
  if (!(await requireSuperAdmin(c.req.header("Authorization"), c.env.JWT_SECRET))) {
    return c.json({ error: "Forbidden" }, 403);
  }

  const { results } = await c.env.DB.prepare(
    "SELECT * FROM nodes ORDER BY status, node_name",
  ).all<NodeRow>();

  return c.json({ total: results.length, nodes: results });
});

// ── PUT /api/nodes/:id/drain ──────────────────────────────────────────────────

nodesRouter.put("/:id/drain", async (c) => {
  if (!(await requireSuperAdmin(c.req.header("Authorization"), c.env.JWT_SECRET))) {
    return c.json({ error: "Forbidden" }, 403);
  }

  const nodeId = c.req.param("id");
  const result = await c.env.DB.prepare(
    "UPDATE nodes SET status = 'DRAINING' WHERE node_id = ?",
  )
    .bind(nodeId)
    .run();

  if (result.meta.changes === 0) {
    return c.json({ error: "Node not found" }, 404);
  }

  return c.json({ node_id: nodeId, status: "DRAINING" });
});

// ── DELETE /api/nodes/:id ─────────────────────────────────────────────────────

nodesRouter.delete("/:id", async (c) => {
  if (!(await requireSuperAdmin(c.req.header("Authorization"), c.env.JWT_SECRET))) {
    return c.json({ error: "Forbidden" }, 403);
  }

  const nodeId = c.req.param("id");

  // Block deletion if node has active cameras
  const node = await c.env.DB.prepare(
    "SELECT active_cameras FROM nodes WHERE node_id = ?",
  )
    .bind(nodeId)
    .first<{ active_cameras: number }>();

  if (!node) return c.json({ error: "Node not found" }, 404);

  if (node.active_cameras > 0) {
    return c.json(
      { error: "Cannot delete node with active cameras — drain first" },
      409,
    );
  }

  // Remove camera assignments
  await c.env.DB.prepare(
    "DELETE FROM camera_assignments WHERE node_id = ?",
  )
    .bind(nodeId)
    .run();

  await c.env.DB.prepare("DELETE FROM nodes WHERE node_id = ?")
    .bind(nodeId)
    .run();

  return c.body(null, 204);
});
