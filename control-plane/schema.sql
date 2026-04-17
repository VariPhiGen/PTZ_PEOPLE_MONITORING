-- ACAS Control-Plane D1 Schema
-- Run: wrangler d1 execute ACAS_DB --file=schema.sql

-- ── Nodes ──────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS nodes (
  node_id        TEXT    PRIMARY KEY,
  node_name      TEXT    NOT NULL,
  location       TEXT,
  -- How the CP reaches this node
  connectivity   TEXT    NOT NULL DEFAULT 'PUBLIC_IP',  -- PUBLIC_IP | CLOUDFLARE_TUNNEL
  api_endpoint   TEXT    NOT NULL,                       -- https://node-hostname:8000
  gpu_model      TEXT,
  max_cameras    INTEGER NOT NULL DEFAULT 10,
  active_cameras INTEGER NOT NULL DEFAULT 0,
  status         TEXT    NOT NULL DEFAULT 'ONLINE',      -- ONLINE | OFFLINE | DRAINING
  last_heartbeat INTEGER NOT NULL DEFAULT 0,             -- unix epoch
  health_json    TEXT                                    -- latest health snapshot JSON
);

CREATE INDEX IF NOT EXISTS idx_nodes_status   ON nodes(status);
CREATE INDEX IF NOT EXISTS idx_nodes_heartbeat ON nodes(last_heartbeat);

-- ── Camera assignments ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS camera_assignments (
  camera_id   TEXT PRIMARY KEY,
  node_id     TEXT NOT NULL REFERENCES nodes(node_id),
  client_id   TEXT NOT NULL,
  camera_name TEXT,
  room_name   TEXT
);

CREATE INDEX IF NOT EXISTS idx_cam_node      ON camera_assignments(node_id);
CREATE INDEX IF NOT EXISTS idx_cam_client    ON camera_assignments(client_id);

-- ── Session routing ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS session_routing (
  session_id TEXT PRIMARY KEY,
  camera_id  TEXT,
  node_id    TEXT NOT NULL REFERENCES nodes(node_id),
  client_id  TEXT NOT NULL,
  started_at INTEGER NOT NULL DEFAULT (strftime('%s','now'))
);

CREATE INDEX IF NOT EXISTS idx_sess_node   ON session_routing(node_id);
CREATE INDEX IF NOT EXISTS idx_sess_client ON session_routing(client_id);

-- ── Analytics aggregates ───────────────────────────────────────────────────────
-- scope: global | node:{node_id} | client:{client_id}
CREATE TABLE IF NOT EXISTS analytics_aggregate (
  date   TEXT    NOT NULL,   -- YYYY-MM-DD
  metric TEXT    NOT NULL,   -- e.g. sessions_completed, avg_recognition_rate
  scope  TEXT    NOT NULL,   -- global | node:xxx | client:xxx
  value  REAL    NOT NULL,
  PRIMARY KEY (date, metric, scope)
);

-- ── Clients (mirror of backend, kept in sync via heartbeat) ────────────────────
CREATE TABLE IF NOT EXISTS cp_clients (
  client_id TEXT PRIMARY KEY,
  name      TEXT NOT NULL,
  slug      TEXT NOT NULL UNIQUE,
  status    TEXT NOT NULL DEFAULT 'ACTIVE'
);

CREATE INDEX IF NOT EXISTS idx_cp_clients_slug ON cp_clients(slug);
