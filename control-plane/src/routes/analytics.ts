/**
 * Analytics routes — served directly from D1 aggregates.
 * No backend node proxy needed; the CP is the analytics source of truth.
 *
 * GET /api/analytics/attendance-trends
 * GET /api/analytics/recognition-accuracy
 * GET /api/analytics/system-health
 * GET /api/analytics/camera-uptime
 * GET /api/analytics/faculty-report
 * GET /api/analytics/student-report/:id
 * GET /api/analytics/flow-matrix
 * GET /api/analytics/occupancy-forecast
 */
import { Hono } from "hono";
import type { AnalyticsRow, Env } from "../types.js";
import { extractBearer, verifyJWT } from "../lib/jwt.js";

export const analyticsRouter = new Hono<{ Bindings: Env }>();

// ── Auth helper ───────────────────────────────────────────────────────────────

async function authPayload(authHeader: string | null | undefined, secret: string) {
  const token = extractBearer(authHeader);
  if (!token) return null;
  return verifyJWT(token, secret);
}

function scopeFilter(
  clientId: string | null,
  role: string,
): { filter: string; scope: string } {
  if (role === "SUPER_ADMIN") {
    return { filter: "", scope: "global" };
  }
  return {
    filter: `AND scope IN ('global', 'client:${clientId}', 'node:${clientId}')`,
    scope: `client:${clientId}`,
  };
}

// ── Attendance trends ─────────────────────────────────────────────────────────

analyticsRouter.get("/attendance-trends", async (c) => {
  const payload = await authPayload(c.req.header("Authorization"), c.env.JWT_SECRET);
  if (!payload) return c.json({ error: "Unauthorized" }, 401);

  const url       = new URL(c.req.url);
  const dateFrom  = url.searchParams.get("date_from") ?? thirtyDaysAgo();
  const dateTo    = url.searchParams.get("date_to")   ?? today();
  const { filter } = scopeFilter(payload.client_id, payload.role);

  const { results } = await c.env.DB.prepare(`
    SELECT date, metric, scope, value
    FROM analytics_aggregate
    WHERE metric LIKE 'status_%'
      AND date >= ? AND date <= ?
      ${filter}
    ORDER BY date, metric
  `)
    .bind(dateFrom, dateTo)
    .all<AnalyticsRow>();

  return c.json({ date_from: dateFrom, date_to: dateTo, data: results });
});

// ── Recognition accuracy ──────────────────────────────────────────────────────

analyticsRouter.get("/recognition-accuracy", async (c) => {
  const payload = await authPayload(c.req.header("Authorization"), c.env.JWT_SECRET);
  if (!payload) return c.json({ error: "Unauthorized" }, 401);

  const { filter } = scopeFilter(payload.client_id, payload.role);
  const dateFrom   = new URL(c.req.url).searchParams.get("date_from") ?? thirtyDaysAgo();

  const { results } = await c.env.DB.prepare(`
    SELECT date, AVG(value) as avg_value, MIN(value) as min_value, MAX(value) as max_value
    FROM analytics_aggregate
    WHERE metric = 'avg_recognition_rate'
      AND date >= ?
      ${filter}
    GROUP BY date
    ORDER BY date
  `)
    .bind(dateFrom)
    .all<{ date: string; avg_value: number; min_value: number; max_value: number }>();

  return c.json({ data: results });
});

// ── System health ─────────────────────────────────────────────────────────────

analyticsRouter.get("/system-health", async (c) => {
  const payload = await authPayload(c.req.header("Authorization"), c.env.JWT_SECRET);
  if (!payload) return c.json({ error: "Unauthorized" }, 401);

  const now = Math.floor(Date.now() / 1000);

  // Node status summary
  const { results: nodes } = await c.env.DB.prepare(`
    SELECT status, COUNT(*) as cnt
    FROM nodes
    GROUP BY status
  `).all<{ status: string; cnt: number }>();

  // Latest health metrics (last 24h)
  const { results: healthMetrics } = await c.env.DB.prepare(`
    SELECT metric, scope, value
    FROM analytics_aggregate
    WHERE date = ?
      AND metric IN ('gpu_utilization', 'cpu_percent', 'gpu_memory_used', 'active_cameras')
    ORDER BY scope, metric
  `)
    .bind(today())
    .all<AnalyticsRow>();

  const nodeStatus: Record<string, number> = {};
  for (const n of nodes) nodeStatus[n.status] = n.cnt;

  return c.json({ ts: now, node_status: nodeStatus, health_metrics: healthMetrics });
});

// ── Camera uptime ─────────────────────────────────────────────────────────────

analyticsRouter.get("/camera-uptime", async (c) => {
  const payload = await authPayload(c.req.header("Authorization"), c.env.JWT_SECRET);
  if (!payload) return c.json({ error: "Unauthorized" }, 401);

  const { results } = await c.env.DB.prepare(`
    SELECT ca.camera_id, ca.camera_name, ca.room_name, n.node_id,
           n.node_name, n.status AS node_status, n.active_cameras
    FROM camera_assignments ca
    JOIN nodes n ON n.node_id = ca.node_id
    WHERE ca.client_id = ?
    ORDER BY ca.camera_name
  `)
    .bind(payload.client_id ?? "")
    .all<{
      camera_id: string; camera_name: string | null; room_name: string | null;
      node_id: string; node_name: string; node_status: string; active_cameras: number;
    }>();

  return c.json({ total: results.length, cameras: results });
});

// ── Faculty report ────────────────────────────────────────────────────────────

analyticsRouter.get("/faculty-report", async (c) => {
  const payload = await authPayload(c.req.header("Authorization"), c.env.JWT_SECRET);
  if (!payload) return c.json({ error: "Unauthorized" }, 401);

  const dateFrom = new URL(c.req.url).searchParams.get("date_from") ?? thirtyDaysAgo();
  const { filter } = scopeFilter(payload.client_id, payload.role);

  const { results } = await c.env.DB.prepare(`
    SELECT scope, metric, date, value
    FROM analytics_aggregate
    WHERE metric LIKE 'faculty_%'
      AND date >= ?
      ${filter}
    ORDER BY date DESC, scope
  `)
    .bind(dateFrom)
    .all<AnalyticsRow>();

  return c.json({ data: results });
});

// ── Student report ────────────────────────────────────────────────────────────

analyticsRouter.get("/student-report/:id", async (c) => {
  const payload = await authPayload(c.req.header("Authorization"), c.env.JWT_SECRET);
  if (!payload) return c.json({ error: "Unauthorized" }, 401);

  // Student report is too fine-grained for D1 aggregates alone;
  // forward to any client node that has the full attendance DB.
  return c.json({ error: "Route to backend node for per-student data" }, 307);
});

// ── Flow matrix ───────────────────────────────────────────────────────────────

analyticsRouter.get("/flow-matrix", async (c) => {
  const payload = await authPayload(c.req.header("Authorization"), c.env.JWT_SECRET);
  if (!payload) return c.json({ error: "Unauthorized" }, 401);

  // Camera transitions are stored per-node; forward to backend proxy
  return c.json({ error: "Route to backend node for flow matrix" }, 307);
});

// ── Occupancy forecast ────────────────────────────────────────────────────────

analyticsRouter.get("/occupancy-forecast", async (c) => {
  const payload = await authPayload(c.req.header("Authorization"), c.env.JWT_SECRET);
  if (!payload) return c.json({ error: "Unauthorized" }, 401);

  const dateFrom = new URL(c.req.url).searchParams.get("date_from") ?? thirtyDaysAgo();
  const { filter } = scopeFilter(payload.client_id, payload.role);

  const { results } = await c.env.DB.prepare(`
    SELECT scope, date, AVG(value) as avg_occupancy
    FROM analytics_aggregate
    WHERE metric = 'active_cameras'
      AND date >= ?
      ${filter}
    GROUP BY scope, date
    ORDER BY scope, date
  `)
    .bind(dateFrom)
    .all<{ scope: string; date: string; avg_occupancy: number }>();

  return c.json({ data: results });
});

// ── Date helpers ──────────────────────────────────────────────────────────────

function today(): string {
  return new Date().toISOString().slice(0, 10);
}

function thirtyDaysAgo(): string {
  const d = new Date();
  d.setDate(d.getDate() - 30);
  return d.toISOString().slice(0, 10);
}
