/**
 * Camera migration — detect offline nodes and reassign their cameras
 * to healthy nodes belonging to the same client.
 *
 * Called by the cron handler every 5 minutes.
 * A node is marked OFFLINE when last_heartbeat is older than
 * HEARTBEAT_TIMEOUT_S (default 900 s = 3 × 5-min intervals).
 */
import type { CameraAssignmentRow, Env, MigrationResult, NodeRow } from "../types.js";

export async function runMigration(env: Env): Promise<MigrationResult> {
  const timeoutS = parseInt(env.HEARTBEAT_TIMEOUT_S || "900", 10);
  const cutoff   = Math.floor(Date.now() / 1000) - timeoutS;

  // Find nodes that missed their heartbeats but are still marked ONLINE
  const { results: staleNodes } = await env.DB.prepare(`
    SELECT * FROM nodes
    WHERE status = 'ONLINE'
      AND last_heartbeat < ?
  `)
    .bind(cutoff)
    .all<NodeRow>();

  const offlineNodes:    string[] = [];
  let   migratedCameras          = 0;
  const errors:          string[] = [];

  for (const node of staleNodes) {
    // Mark node OFFLINE
    await env.DB.prepare(
      "UPDATE nodes SET status = 'OFFLINE' WHERE node_id = ?",
    ).bind(node.node_id).run();

    offlineNodes.push(node.node_id);

    // Get all camera assignments for this node
    const { results: cameras } = await env.DB.prepare(
      "SELECT * FROM camera_assignments WHERE node_id = ?",
    )
      .bind(node.node_id)
      .all<CameraAssignmentRow>();

    if (cameras.length === 0) continue;

    // Group cameras by client_id
    const byClient = new Map<string, CameraAssignmentRow[]>();
    for (const cam of cameras) {
      const list = byClient.get(cam.client_id) ?? [];
      list.push(cam);
      byClient.set(cam.client_id, list);
    }

    // For each client, find a healthy target node and reassign
    for (const [clientId, cams] of byClient) {
      const target = await pickTargetNode(env.DB, clientId, node.node_id, cams.length);
      if (!target) {
        errors.push(`no_target_node:client=${clientId}`);
        continue;
      }

      for (const cam of cams) {
        try {
          await env.DB.prepare(
            "UPDATE camera_assignments SET node_id = ? WHERE camera_id = ?",
          )
            .bind(target.node_id, cam.camera_id)
            .run();

          // Update KV routing cache
          await env.ACAS_ROUTING.put(`cam:${cam.camera_id}`, target.node_id, {
            expirationTtl: 86_400,
          });

          migratedCameras++;
        } catch (err) {
          errors.push(`migrate_failed:${cam.camera_id}:${String(err)}`);
        }
      }

      // Notify target node about the new camera assignments
      await notifyNodeOfMigration(target.api_endpoint, clientId, cams, node.node_id);
    }
  }

  // Aggregate analytics: count migrations
  if (migratedCameras > 0) {
    const today = new Date().toISOString().slice(0, 10);
    await env.DB.prepare(`
      INSERT INTO analytics_aggregate (date, metric, scope, value)
      VALUES (?, 'camera_migrations', 'global', ?)
      ON CONFLICT (date, metric, scope) DO UPDATE SET value = value + excluded.value
    `)
      .bind(today, migratedCameras)
      .run();
  }

  return { offlineNodes, migratedCameras, errors };
}

// ── Helpers ───────────────────────────────────────────────────────────────────

async function pickTargetNode(
  db:           D1Database,
  clientId:     string,
  excludeNode:  string,
  cameraCount:  number,
): Promise<NodeRow | null> {
  // Find ONLINE node for this client that has capacity
  const row = await db
    .prepare(`
      SELECT DISTINCT n.*
      FROM nodes n
      JOIN camera_assignments ca ON ca.node_id = n.node_id
      WHERE ca.client_id = ?
        AND n.node_id   != ?
        AND n.status     = 'ONLINE'
        AND (n.max_cameras - n.active_cameras) >= ?
      ORDER BY n.active_cameras ASC
      LIMIT 1
    `)
    .bind(clientId, excludeNode, cameraCount)
    .first<NodeRow>();

  return row ?? null;
}

async function notifyNodeOfMigration(
  apiEndpoint:  string,
  clientId:     string,
  cameras:      CameraAssignmentRow[],
  fromNodeId:   string,
): Promise<void> {
  try {
    await fetch(`${apiEndpoint}/api/node/cameras/migrate`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({
        client_id:  clientId,
        from_node:  fromNodeId,
        cameras:    cameras.map((c) => ({
          camera_id:   c.camera_id,
          camera_name: c.camera_name,
          room_name:   c.room_name,
        })),
      }),
    });
  } catch {
    // Best-effort; node will resync on next heartbeat
  }
}

// ── Analytics aggregation (called from cron) ──────────────────────────────────

export async function aggregateAnalytics(env: Env): Promise<void> {
  const today = new Date().toISOString().slice(0, 10);

  // Global: count ONLINE nodes
  const { results: onlineNodes } = await env.DB.prepare(
    "SELECT COUNT(*) as cnt FROM nodes WHERE status = 'ONLINE'",
  ).all<{ cnt: number }>();

  if (onlineNodes[0]) {
    await upsertAggregate(env.DB, today, "online_nodes", "global", onlineNodes[0].cnt);
  }

  // Global: total active_cameras
  const { results: camRows } = await env.DB.prepare(
    "SELECT SUM(active_cameras) as total FROM nodes WHERE status = 'ONLINE'",
  ).all<{ total: number | null }>();

  if (camRows[0]) {
    await upsertAggregate(
      env.DB, today, "active_cameras", "global", camRows[0].total ?? 0,
    );
  }

  // Per-node health (last heartbeat health_json)
  const { results: nodes } = await env.DB.prepare(
    "SELECT node_id, health_json FROM nodes WHERE status = 'ONLINE'",
  ).all<{ node_id: string; health_json: string | null }>();

  for (const n of nodes) {
    if (!n.health_json) continue;
    try {
      const h = JSON.parse(n.health_json) as Record<string, number>;
      for (const [metric, value] of Object.entries(h)) {
        if (typeof value === "number") {
          await upsertAggregate(env.DB, today, metric, `node:${n.node_id}`, value);
        }
      }
    } catch {
      // ignore parse errors
    }
  }
}

async function upsertAggregate(
  db:     D1Database,
  date:   string,
  metric: string,
  scope:  string,
  value:  number,
): Promise<void> {
  await db
    .prepare(`
      INSERT INTO analytics_aggregate (date, metric, scope, value)
      VALUES (?, ?, ?, ?)
      ON CONFLICT (date, metric, scope) DO UPDATE SET value = excluded.value
    `)
    .bind(date, metric, scope, value)
    .run();
}
