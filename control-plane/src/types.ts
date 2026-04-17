/**
 * Shared types for the ACAS Control-Plane Cloudflare Worker.
 */

// ── Environment bindings ───────────────────────────────────────────────────────

export interface Env {
  // D1 — persistent relational data
  DB: D1Database;
  // KV — fast lookups & config
  ACAS_CONFIG:  KVNamespace;   // global settings
  ACAS_ROUTING: KVNamespace;   // camera_id → node_id  |  session_id → api_endpoint
  // Durable Object
  SESSION_RELAY: DurableObjectNamespace;
  // Secrets (set via `wrangler secret put`)
  JWT_SECRET:       string;
  NODE_AUTH_TOKEN:  string;
  // Vars
  ENVIRONMENT:         string;
  HEARTBEAT_TIMEOUT_S: string;   // default "900"
}

// ── D1 row shapes ──────────────────────────────────────────────────────────────

export interface NodeRow {
  node_id:        string;
  node_name:      string;
  location:       string | null;
  connectivity:   "PUBLIC_IP" | "CLOUDFLARE_TUNNEL";
  api_endpoint:   string;
  gpu_model:      string | null;
  max_cameras:    number;
  active_cameras: number;
  status:         "ONLINE" | "OFFLINE" | "DRAINING";
  last_heartbeat: number;
  health_json:    string | null;
}

export interface CameraAssignmentRow {
  camera_id:   string;
  node_id:     string;
  client_id:   string;
  camera_name: string | null;
  room_name:   string | null;
}

export interface SessionRoutingRow {
  session_id: string;
  camera_id:  string | null;
  node_id:    string;
  client_id:  string;
  started_at: number;
}

export interface AnalyticsRow {
  date:   string;
  metric: string;
  scope:  string;
  value:  number;
}

export interface CpClientRow {
  client_id: string;
  name:      string;
  slug:      string;
  status:    string;
}

// ── JWT payload ────────────────────────────────────────────────────────────────

export interface JWTPayload {
  user_id:     string;
  email:       string;
  name:        string;
  role:        "SUPER_ADMIN" | "CLIENT_ADMIN" | "VIEWER";
  client_id:   string | null;
  client_slug: string | null;
  permissions: string[];
  iat:         number;
  exp:         number;
  type:        "access" | "refresh";
}

// ── Fan-out ────────────────────────────────────────────────────────────────────

export type MergeStrategy =
  | "union"          // concatenate `.items` arrays from all responses
  | "first"          // return first successful response verbatim
  | "merge_journey"  // merge + sort journey events by first_seen
  | "sum"            // add numeric `.total` / `.count` fields
  | "latest";        // return the response with the largest `.updated_at`

export interface FanOutResult {
  data:       unknown;
  nodeCount:  number;
  errors:     string[];
  partial:    boolean;
}

// ── Node heartbeat payload ─────────────────────────────────────────────────────

export interface HeartbeatPayload {
  active_cameras: number;
  health: {
    gpu_utilization:   number;
    gpu_memory_used:   number;   // MB
    gpu_memory_total:  number;   // MB
    cpu_percent:       number;
    ram_used_mb:       number;
  };
  analytics?: {
    [metric: string]: number;
  };
  // Sync helpers — node pushes its current camera list and active sessions
  camera_ids?:    string[];
  session_ids?:   string[];
}

// ── Migration ─────────────────────────────────────────────────────────────────

export interface MigrationResult {
  offlineNodes:    string[];
  migratedCameras: number;
  errors:          string[];
}
