// ── Auth ──────────────────────────────────────────────────────────────────────

export type Role = "SUPER_ADMIN" | "CLIENT_ADMIN" | "VIEWER";

export interface JWTPayload {
  user_id:     string;
  email:       string;
  name:        string;
  role:        Role;
  client_id:   string | null;
  client_slug: string | null;
  permissions: string[];
  iat:         number;
  exp:         number;
  type:        "access" | "refresh";
}

export interface LoginResponse {
  access_token:  string;
  refresh_token: string;
  user: {
    user_id:   string;
    email:     string;
    name:      string;
    role:      Role;
    client_id: string | null;
    mfa_enabled: boolean;
  };
}

// ── Client / Node ─────────────────────────────────────────────────────────────

export interface Client {
  client_id:    string;
  name:         string;
  slug:         string;
  logo_url:     string | null;
  status:       "ACTIVE" | "SUSPENDED" | "ARCHIVED";
  max_cameras:  number;
  max_persons:  number;
  camera_count: number;
  person_count: number;
  active_sessions: number;
  settings:     Record<string, unknown>;
}

export interface GpuNode {
  node_id:        string;
  node_name:      string;
  location:       string | null;
  connectivity:   string;
  api_endpoint:   string;
  gpu_model:      string | null;
  max_cameras:    number;
  active_cameras: number;
  status:         "ONLINE" | "OFFLINE" | "DRAINING";
  last_heartbeat: number;
  health_json:    Record<string, unknown> | null;
}

// ── Cameras ───────────────────────────────────────────────────────────────────

export type CameraStatus = "ONLINE" | "OFFLINE" | "DEGRADED";
export type CameraMode   = "ATTENDANCE" | "MONITORING" | "BOTH";

/** Legacy rectangle zone (kept for backward compat with old DB rows) */
export interface RoiRect { x: number; y: number; w: number; h: number }
/** New polygon zone — normalized [0,1] coordinate pairs */
export type ZonePoint   = { x: number; y: number };
export type ZonePolygon = ZonePoint[];
/** A zone value may be either legacy rect or new polygon */
export type ZoneValue   = RoiRect | ZonePolygon | null;

export interface MonitoringHours {
  start: string;           // "HH:MM"
  end:   string;           // "HH:MM"
  days:  string[];         // ["MON","TUE","WED","THU","FRI"]
}

export interface Camera {
  camera_id:           string;
  client_id:           string;
  name:                string;
  room_name:           string;
  building:            string | null;
  floor:               string | null;
  rtsp_url:            string;
  onvif_host:          string;
  onvif_port:          number;
  onvif_username:      string;
  status:              CameraStatus;
  mode:                CameraMode;
  node_id:             string | null;
  /** API base URL of the GPU node this camera is assigned to.
   *  Used to build correct stream/snapshot/ptz URLs in multi-node deployments. */
  node_api_endpoint:   string | null;
  roi_rect:            ZoneValue;
  faculty_zone:        ZoneValue;
  fov_h:               number | null;
  fov_v:               number | null;
  pan_speed:           number | null;
  tilt_speed:          number | null;
  zoom_speed:          number | null;
  alert_on_unknown:    boolean;
  restricted_zone:     boolean;
  monitoring_hours:    MonitoringHours | null;
  dataset_id:          string | null;
  camera_distance_m:   number;          // metres from camera to monitored area (default 7)
  scan_cell_meters:    number | null;   // scan cell width in metres; null = auto
  learned_params:      Record<string, unknown> | null;
  timetable_id:        string | null;
  camera_type:         string | null;
  created_at:          string;
  updated_at:          string;
}

// ── Persons / Enrollment ──────────────────────────────────────────────────────

export type PersonRole   = "STUDENT" | "FACULTY" | "ADMIN";
export type PersonStatus = "ACTIVE" | "INACTIVE" | "SUSPENDED";

export interface Person {
  person_id:   string;
  client_id:   string;
  external_id: string;
  name:        string;
  role:        PersonRole;
  department:  string | null;
  email:       string | null;
  status:      PersonStatus;
  enrolled:    boolean;
  thumbnail:   string | null;
  last_seen:   string | null;
}

// ── Sessions & Attendance ─────────────────────────────────────────────────────

export type SyncStatus = "PENDING" | "SYNCED" | "HELD" | "FAILED" | "DISCARDED";
export type AttendanceStatus = "P" | "L" | "EE" | "A" | "ND" | "EX";

export interface Session {
  session_id:       string;
  client_id:        string;
  camera_id:        string;
  camera_name:      string;
  course_id:        string | null;
  course_name:      string | null;
  faculty_id:       string | null;
  faculty_name:     string | null;
  scheduled_start:  string;
  scheduled_end:    string;
  actual_start:     string | null;
  actual_end:       string | null;
  faculty_status:   string | null;
  sync_status:      SyncStatus;
  held_reason:      string | null;
  cycle_count:      number;
  recognition_rate: number | null;
  created_at:       string;
}

export interface AttendanceRecord {
  record_id:        string;
  session_id:       string;
  person_id:        string;
  person_name:      string;
  person_external_id: string;
  total_duration:   string;
  detection_count:  number;
  first_seen:       string | null;
  last_seen:        string | null;
  status:           AttendanceStatus;
  confidence_avg:   number;
  override_by:      string | null;
  override_reason:  string | null;
}

// ── Live session state ────────────────────────────────────────────────────────

export type BrainState =
  | "OVERVIEW_SCAN"
  | "PATH_PLAN"
  | "CELL_TRANSIT"
  | "CELL_RECOGNIZE"
  | "FACE_HUNT"
  | "CELL_COMPLETE"
  | "CYCLE_COMPLETE"
  | "STOPPED"
  | "ERROR";

export interface ScanCell {
  cell_id:           string;
  center_pan:        number;
  center_tilt:       number;
  required_zoom:     number;
  expected_faces:    number;
  unrecognized_count: number;
  priority:          number;
}

export interface SessionState {
  session_id:         string;
  camera_id:          string;
  state:              BrainState;
  mode:               "ATTENDANCE" | "MONITORING";
  current_cell_id:    string | null;
  path_index:         number;
  cycle_count:        number;
  recognition_rate:   number;
  recognized_count:   number;
  unrecognized_count: number;
  current_ptz:        { pan: number; tilt: number; zoom: number };
  scan_cells:         ScanCell[];
  duration_trackers:  Record<string, { total_seconds: number; state: string }>;
  ts:                 number;
}

export interface RecognitionEvent {
  type:          "RECOGNIZED" | "UNKNOWN" | "LIVENESS_FAIL";
  person_id:     string | null;
  person_name:   string | null;
  confidence:    number;
  liveness:      number;
  cell_id:       string;
  ts:            number;
}

// ── Analytics ─────────────────────────────────────────────────────────────────

export interface AttendanceTrend {
  date:     string;
  present:  number;
  late:     number;
  early_exit: number;
  absent:   number;
}

export interface SystemHealth {
  ts:          number;
  node_status: Record<string, number>;
  health_metrics: Array<{ metric: string; scope: string; value: number }>;
}

// ── Search ────────────────────────────────────────────────────────────────────

export interface SearchHit {
  person_id:    string;
  name:         string;
  role:         PersonRole;
  department:   string | null;
  external_id:  string;
  thumbnail:    string | null;
  last_seen:    string | null;
  match_score:  number;
}

export interface JourneyEvent {
  type:         "ATTENDANCE" | "SIGHTING";
  source_id:    string;
  camera_id:    string;
  camera_name:  string;
  area:         string | null;
  first_seen:   string;
  last_seen:    string;
  duration_s:   number;
  status:       string | null;
  transit_time: number | null;
}

// ── Pagination ────────────────────────────────────────────────────────────────

export interface PaginatedResponse<T> {
  items:  T[];
  total:  number;
  limit:  number;
  offset: number;
}

// ── API Error ─────────────────────────────────────────────────────────────────

export interface ApiError {
  detail: string | { msg: string; type: string }[];
  status: number;
}
