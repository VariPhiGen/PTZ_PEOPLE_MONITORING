/**
 * ACAS Control-Plane — Cloudflare Worker entry point
 *
 * Architecture:
 *   ┌──────────────────────────────────────────────────────────────┐
 *   │  Cloudflare Worker (this file)                               │
 *   │                                                              │
 *   │  /api/nodes/**        → nodesRouter   (D1, KV)              │
 *   │  /api/analytics/**    → analyticsRouter (D1)                 │
 *   │  /api/settings/**     → settingsRouter (KV + node broadcast) │
 *   │  /api/sessions/:id/ws → SessionRelay DO (WebSocket relay)    │
 *   │  /api/**              → proxyRouter   (fan-out to nodes)     │
 *   └──────────────────────────────────────────────────────────────┘
 *
 * Cron (every 5 minutes, configured in wrangler.toml):
 *   - Aggregate analytics from node heartbeats stored in D1
 *   - Detect stale nodes (over 15 min since last heartbeat)
 *   - Migrate cameras from offline nodes to healthy ones
 */
import { Hono }            from "hono";
import { cors }            from "hono/cors";
import { logger }          from "hono/logger";
import type { Env }        from "./types.js";
import { nodesRouter }     from "./routes/nodes.js";
import { proxyRouter }     from "./routes/proxy.js";
import { analyticsRouter } from "./routes/analytics.js";
import { settingsRouter }  from "./routes/settings.js";
import { aggregateAnalytics, runMigration } from "./lib/migration.js";

// Re-export the Durable Object class so wrangler can find it
export { SessionRelay } from "./objects/SessionRelay.js";

// ── Application ───────────────────────────────────────────────────────────────

const app = new Hono<{ Bindings: Env }>();

app.use("*", logger());
app.use(
  "*",
  cors({
    origin:      ["*"],
    allowMethods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allowHeaders: ["Authorization", "Content-Type"],
    maxAge:       86_400,
  }),
);

// ── Health ────────────────────────────────────────────────────────────────────

app.get("/health", (c) =>
  c.json({ service: "acas-control-plane", status: "ok", ts: Date.now() }),
);

// ── Subrouters ────────────────────────────────────────────────────────────────

app.route("/api/nodes",     nodesRouter);
app.route("/api/analytics", analyticsRouter);
app.route("/api/settings",  settingsRouter);

// The proxy router must come LAST — it has catch-all routes
app.route("/api",           proxyRouter);

// ── 404 fallback ──────────────────────────────────────────────────────────────

app.notFound((c) => c.json({ error: "Not found" }, 404));

app.onError((err, c) => {
  console.error("Unhandled error:", err);
  return c.json({ error: "Internal server error" }, 500);
});

// ── Cloudflare Worker export ──────────────────────────────────────────────────

export default {
  /**
   * Handle HTTP requests (and WebSocket upgrades).
   */
  async fetch(
    request: Request,
    env:     Env,
    _ctx:    ExecutionContext,
  ): Promise<Response> {
    return app.fetch(request, env, _ctx);
  },

  /**
   * Cron trigger — runs every 5 minutes.
   * 1. Detect stale nodes and mark OFFLINE.
   * 2. Migrate cameras from offline nodes.
   * 3. Aggregate analytics into D1.
   */
  async scheduled(
    _event:      ScheduledEvent,
    env:         Env,
    ctx:         ExecutionContext,
  ): Promise<void> {
    ctx.waitUntil(
      (async () => {
        try {
          const migrationResult = await runMigration(env);
          if (migrationResult.offlineNodes.length > 0) {
            console.log(
              `[cron] Node migration: offline=${migrationResult.offlineNodes.join(",")}, ` +
              `migrated=${migrationResult.migratedCameras} cameras`,
            );
          }
        } catch (err) {
          console.error("[cron] Migration error:", err);
        }

        try {
          await aggregateAnalytics(env);
        } catch (err) {
          console.error("[cron] Analytics error:", err);
        }
      })(),
    );
  },
};
