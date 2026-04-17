/**
 * Axios API client — points at the Control Plane (NEXT_PUBLIC_API_URL).
 *
 * Features:
 *  • Attaches JWT from the acas_token cookie to every request.
 *  • On 401: attempts one silent refresh using acas_refresh cookie.
 *  • On 403: rejects with the original error (callers show a toast).
 *  • On unrecoverable 401: clears tokens and redirects to /auth/login.
 */
import axios, {
  AxiosError,
  AxiosInstance,
  InternalAxiosRequestConfig,
} from "axios";
import { getToken, getRefreshToken, setTokens, clearTokens, getPayload } from "@/lib/auth";
import type { LoginResponse } from "@/types";

// Empty string = relative paths → works behind nginx, Cloudflare, or Next.js rewrites.
// Set NEXT_PUBLIC_API_URL only to target a remote backend directly.
const BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "";

export const api: AxiosInstance = axios.create({
  baseURL:         BASE_URL,
  timeout:         30_000,
  headers:         { "Content-Type": "application/json" },
  withCredentials: false,
});

// ── Request interceptor — attach JWT ─────────────────────────────────────────

api.interceptors.request.use((config: InternalAxiosRequestConfig) => {
  const token = getToken();
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// ── Response interceptor — refresh / error handling ──────────────────────────

type RetryableConfig = InternalAxiosRequestConfig & { _retry?: boolean };

let _refreshPromise: Promise<string> | null = null;

api.interceptors.response.use(
  (response) => response,
  async (error: AxiosError) => {
    const originalRequest = error.config as RetryableConfig | undefined;

    if (error.response?.status === 401 && originalRequest && !originalRequest._retry) {
      originalRequest._retry = true;

      const refreshToken = getRefreshToken();
      if (!refreshToken) {
        redirect("/auth/login");
        return Promise.reject(error);
      }

      // Deduplicate concurrent refresh calls
      if (!_refreshPromise) {
        _refreshPromise = axios
          .post<LoginResponse>(`${BASE_URL}/api/auth/refresh`, {
            refresh_token: refreshToken,
          })
          .then(({ data }) => {
            setTokens(data.access_token, data.refresh_token);
            return data.access_token;
          })
          .catch((err) => {
            clearTokens();
            redirect("/auth/login");
            throw err;
          })
          .finally(() => {
            _refreshPromise = null;
          });
      }

      try {
        const newToken = await _refreshPromise;
        originalRequest.headers.Authorization = `Bearer ${newToken}`;
        return api(originalRequest);
      } catch {
        return Promise.reject(error);
      }
    }

    return Promise.reject(error);
  },
);

/** Safe browser redirect (no-op in SSR) */
function redirect(path: string): void {
  if (typeof window !== "undefined") {
    window.location.href = path;
  }
}

/**
 * Append client_id from the JWT to a FormData if not already present.
 * Required for SUPER_ADMIN users whose JWT has no tenant scope — the backend's
 * resolve_client_id falls back to the form body when jwt_cid is null.
 */
function withClientId(form: FormData): FormData {
  if (!form.has("client_id")) {
    const cid = getPayload()?.client_id;
    if (cid) form.append("client_id", cid);
  }
  return form;
}

// ── Typed API helpers ─────────────────────────────────────────────────────────

export const authApi = {
  login: (email: string, password: string, mfa_code?: string) =>
    api.post<LoginResponse>("/api/auth/login", { email, password, mfa_code }),
  refresh: (refresh_token: string) =>
    api.post<LoginResponse>("/api/auth/refresh", { refresh_token }),
  me: () => api.get("/api/auth/me"),
  logout: () => { clearTokens(); redirect("/auth/login"); },
};

// ── Camera helpers ────────────────────────────────────────────────────────────

/**
 * Resolve the correct API base URL for a specific camera.
 *
 * Priority order:
 *  1. camera.node_api_endpoint   — the registered API URL for the GPU node
 *     this camera lives on (e.g. http://192.168.1.5:18000 for LAN,
 *     https://node1.yourdomain.com for Cloudflare/cloud).
 *  2. NEXT_PUBLIC_API_URL        — global fallback (single-node or dev mode).
 *
 * This lets every camera operation (stream, snapshot, ptz, roi) hit
 * the correct GPU node regardless of whether you're on localhost,
 * behind a Cloudflare tunnel, or on a cloud VM.
 */
function _nodeBase(camera: { node_api_endpoint?: string | null }): string {
  return (
    camera.node_api_endpoint?.replace(/\/$/, "") ||
    (process.env.NEXT_PUBLIC_API_URL ?? "").replace(/\/$/, "")
  );
}

/** Axios instance that targets a specific node URL (not the global BASE_URL). */
function _nodeApi(base: string) {
  const inst = axios.create({ baseURL: base, timeout: 30_000 });
  inst.interceptors.request.use((c) => {
    const t = getToken();
    if (t) c.headers.Authorization = `Bearer ${t}`;
    return c;
  });
  return inst;
}

export const camerasApi = {
  // ── Catalogue ops (hit the dashboard's own backend — any node works) ─────
  list:           (params?: Record<string, unknown>) => api.get("/api/cameras",                   { params }),
  get:            (id: string)                        => api.get(`/api/cameras/${id}`),
  create:         (data: unknown)                     => api.post("/api/cameras",                 data),
  update:         (id: string, data: unknown)         => api.put(`/api/cameras/${id}`,            data),
  delete:         (id: string)                        => api.delete(`/api/cameras/${id}`),
  testConnection: (data: unknown)                     => api.post("/api/cameras/test-connection", data),
  nodes:          ()                                  => api.get("/api/cameras/available-nodes"),

  // ── Per-camera ops — route to the camera's specific node ────────────────
  /** Real-time snapshot JPEG blob from the camera's GPU node. */
  snapshot:  (id: string, camera?: { node_api_endpoint?: string | null }) => {
    const base = camera ? _nodeBase(camera) : (process.env.NEXT_PUBLIC_API_URL ?? "");
    return _nodeApi(base).get(`/api/cameras/${id}/snapshot`, { responseType: "blob" });
  },

  /** Test ONVIF connectivity. Routes to the camera's node. */
  test:      (id: string, camera?: { node_api_endpoint?: string | null }) => {
    const base = camera ? _nodeBase(camera) : (process.env.NEXT_PUBLIC_API_URL ?? "");
    return _nodeApi(base).post(`/api/cameras/${id}/test`);
  },

  /** Test PTZ protocol (ONVIF/ISAPI/CGI_DAHUA). Only for PTZ/PTZ_ZOOM cameras. Routes to the camera's node. */
  testProtocol: (id: string, protocol: string, camera?: { node_api_endpoint?: string | null }) => {
    const base = camera ? _nodeBase(camera) : (process.env.NEXT_PUBLIC_API_URL ?? "");
    return _nodeApi(base).post(`/api/cameras/${id}/test-protocol`, { protocol });
  },

  /** Auto-detect camera brand and best-supported PTZ protocol. Routes to the camera's node. */
  detectProtocol: (id: string, camera?: { node_api_endpoint?: string | null }) => {
    const base = camera ? _nodeBase(camera) : (process.env.NEXT_PUBLIC_API_URL ?? "");
    return _nodeApi(base).post(`/api/cameras/${id}/detect-protocol`);
  },

  /** Save ROI + faculty zone. Routes to the camera's node. */
  saveRoi:   (id: string, data: unknown, camera?: { node_api_endpoint?: string | null }) => {
    const base = camera ? _nodeBase(camera) : (process.env.NEXT_PUBLIC_API_URL ?? "");
    return _nodeApi(base).post(`/api/cameras/${id}/roi`, data);
  },

  /** Delete all zones (roi_rect + faculty_zone) for a camera. Routes to the camera's node. */
  deleteRoi: (id: string, camera?: { node_api_endpoint?: string | null }) => {
    const base = camera ? _nodeBase(camera) : (process.env.NEXT_PUBLIC_API_URL ?? "");
    return _nodeApi(base).delete(`/api/cameras/${id}/roi`);
  },

  /** PTZ move. Routes to the camera's node. */
  ptzMove:   (id: string, data: unknown, camera?: { node_api_endpoint?: string | null }) => {
    const base = camera ? _nodeBase(camera) : (process.env.NEXT_PUBLIC_API_URL ?? "");
    return _nodeApi(base).post(`/api/cameras/${id}/ptz-move`, data);
  },

  /** PTZ status poll. Routes to the camera's node. */
  ptzStatus: (id: string, camera?: { node_api_endpoint?: string | null }) => {
    const base = camera ? _nodeBase(camera) : (process.env.NEXT_PUBLIC_API_URL ?? "");
    return _nodeApi(base).get(`/api/cameras/${id}/ptz-status`);
  },

  /**
   * Build the MJPEG live-stream URL for a camera.
   *
   * The URL uses ?token= instead of an Authorization header because
   * browsers cannot send custom headers with <img src="...">.
   *
   * It points to the camera's specific GPU node endpoint so the
   * stream works correctly across localhost / Cloudflare / cloud deployments.
   */
  streamUrl: (id: string, camera?: { node_api_endpoint?: string | null }): string => {
    const token = getToken() ?? "";
    const base  = camera ? _nodeBase(camera) : (process.env.NEXT_PUBLIC_API_URL ?? "");
    return `${base}/api/cameras/${id}/stream?token=${encodeURIComponent(token)}`;
  },

  /** Annotated MJPEG stream — overlays detection boxes, face names, brain state, PTZ values. */
  annotatedStreamUrl: (id: string, camera?: { node_api_endpoint?: string | null }): string => {
    const token = getToken() ?? "";
    const base  = camera ? _nodeBase(camera) : (process.env.NEXT_PUBLIC_API_URL ?? "");
    return `${base}/api/cameras/${id}/annotated-stream?token=${encodeURIComponent(token)}`;
  },

  /** Get / set AI brain mode for a camera. Routes to the camera's node. */
  getAiMode: (id: string, camera?: { node_api_endpoint?: string | null }) => {
    const base = camera ? _nodeBase(camera) : (process.env.NEXT_PUBLIC_API_URL ?? "");
    return _nodeApi(base).get(`/api/cameras/${id}/ai-mode`);
  },
  setAiMode: (id: string, mode: "AI" | "MANUAL", camera?: { node_api_endpoint?: string | null }) => {
    const base = camera ? _nodeBase(camera) : (process.env.NEXT_PUBLIC_API_URL ?? "");
    return _nodeApi(base).post(`/api/cameras/${id}/ai-mode`, { mode });
  },

  // ── Scan Preset API ─────────────────────────────────────────────────────────
  listPresets: (id: string, camera?: { node_api_endpoint?: string | null }) => {
    const base = camera ? _nodeBase(camera) : (process.env.NEXT_PUBLIC_API_URL ?? "");
    return _nodeApi(base).get(`/api/cameras/${id}/presets`);
  },
  createPreset: (id: string, data: unknown, camera?: { node_api_endpoint?: string | null }) => {
    const base = camera ? _nodeBase(camera) : (process.env.NEXT_PUBLIC_API_URL ?? "");
    return _nodeApi(base).post(`/api/cameras/${id}/presets`, data);
  },
  updatePreset: (id: string, presetId: string, data: unknown, camera?: { node_api_endpoint?: string | null }) => {
    const base = camera ? _nodeBase(camera) : (process.env.NEXT_PUBLIC_API_URL ?? "");
    return _nodeApi(base).put(`/api/cameras/${id}/presets/${presetId}`, data);
  },
  deletePreset: (id: string, presetId: string, camera?: { node_api_endpoint?: string | null }) => {
    const base = camera ? _nodeBase(camera) : (process.env.NEXT_PUBLIC_API_URL ?? "");
    return _nodeApi(base).delete(`/api/cameras/${id}/presets/${presetId}`);
  },
  reorderPresets: (id: string, presetIds: string[], camera?: { node_api_endpoint?: string | null }) => {
    const base = camera ? _nodeBase(camera) : (process.env.NEXT_PUBLIC_API_URL ?? "");
    return _nodeApi(base).post(`/api/cameras/${id}/presets/reorder`, presetIds);
  },
  gotoPreset: (id: string, presetId: string, camera?: { node_api_endpoint?: string | null }) => {
    const base = camera ? _nodeBase(camera) : (process.env.NEXT_PUBLIC_API_URL ?? "");
    return _nodeApi(base).post(`/api/cameras/${id}/presets/${presetId}/goto`);
  },

  /** Persist move speed (and optional settle delay) in camera learned_params. */
  setMoveSpeed: (
    id: string,
    data: { move_speed: number; settle_s?: number },
    camera?: { node_api_endpoint?: string | null },
  ) => {
    const base = camera ? _nodeBase(camera) : (process.env.NEXT_PUBLIC_API_URL ?? "");
    return _nodeApi(base).put(`/api/cameras/${id}/move-speed`, data);
  },

  /**
   * Run FOV auto-calibration — measures optical-flow pixel displacement for
   * known ONVIF movements at wide (zoom=0) and narrow (zoom=1) and persists
   * the resulting K constants to camera.learned_params. Takes ~30 s.
   * Routes to the camera's node.
   */
  calibrateFov: (id: string, camera?: { node_api_endpoint?: string | null }) => {
    const base = camera ? _nodeBase(camera) : (process.env.NEXT_PUBLIC_API_URL ?? "");
    const inst = axios.create({ baseURL: base, timeout: 120_000 });
    inst.interceptors.request.use((c) => {
      const t = getToken();
      if (t) c.headers.Authorization = `Bearer ${t}`;
      return c;
    });
    return inst.post(`/api/cameras/${id}/calibrate-fov`);
  },
};

export const sessionsApi = {
  start:   (data: unknown) => api.post("/api/sessions/start",   data),
  stop:    (id: string)    => api.post(`/api/sessions/${id}/stop`),
  active:  ()              => api.get("/api/sessions/active"),
  state:   (id: string)    => api.get(`/api/sessions/${id}/state`),
};

export const attendanceApi = {
  sessions: (params?: Record<string, unknown>) => api.get("/api/attendance/sessions",           { params }),
  session:  (id: string)                        => api.get(`/api/attendance/sessions/${id}`),
  records:  (sessionId: string, params?: Record<string, unknown>) =>
    api.get(`/api/attendance/sessions/${sessionId}/records`, { params }),
  held:     (params?: Record<string, unknown>) => api.get("/api/attendance/held",               { params }),
  export:   (params?: Record<string, unknown>) => api.get("/api/attendance/export",             { params, responseType: "blob" }),
  forceSync:(id: string, data: unknown)         => api.post(`/api/attendance/held/${id}/force-sync`, data),
  discard:  (id: string, data: unknown)         => api.post(`/api/attendance/held/${id}/discard`,    data),
  override:       (id: string, data: unknown)         => api.post(`/api/attendance/records/${id}/override`, data),
  crossRecords:   (params?: Record<string, unknown>) => api.get("/api/attendance/records",               { params }),
  unknown:        (params?: Record<string, unknown>) => api.get("/api/attendance/unknown",               { params }),
  assignUnknown:  (id: string, personId: string)     => api.post(`/api/attendance/unknown/${id}/assign`, { person_id: personId }),
  dismissUnknown: (id: string)                        => api.post(`/api/attendance/unknown/${id}/dismiss`, {}),
};

export const enrollmentApi = {
  list:      (params?: Record<string, unknown>) => api.get("/api/enrollment/list",            { params }),
  get:       (id: string)                        => api.get(`/api/enrollment/${id}`),
  upload:    (form: FormData)                    => api.post("/api/enrollment/upload",         withClientId(form), { headers: { "Content-Type": undefined } }),
  enroll:    (data: unknown)                     => api.post("/api/enrollment/enroll",         data),
  bulkImport:(form: FormData)                    => api.post("/api/enrollment/bulk-import",    withClientId(form), { headers: { "Content-Type": undefined } }),
  reEnroll:  (id: string, data: FormData | Record<string, unknown>) => api.put(`/api/enrollment/${id}/re-enroll`, data instanceof FormData ? withClientId(data) : data, { headers: { "Content-Type": undefined } }),
  delete:    (id: string)                        => api.delete(`/api/enrollment/${id}`),
  images:    (id: string)                        => api.get(`/api/enrollment/${id}/images`),
  guidelines:()                                  => api.get("/api/enrollment/guidelines"),
};

export const searchApi = {
  person:     (q: string, params?: Record<string, unknown>) => api.get("/api/search/person",         { params: { q, ...params } }),
  face:       (form: FormData)                               => api.post("/api/search/face",          withClientId(form), { headers: { "Content-Type": undefined } }),
  faceBase64: (data: unknown)                                => api.post("/api/search/face/base64",   data),
  journey:    (id: string, params?: Record<string, unknown>) => api.get(`/api/search/${id}/journey`,  { params }),
  crossCamera:(id: string, params?: Record<string, unknown>) => api.get(`/api/search/${id}/cross-camera`, { params }),
  area:       (params?: Record<string, unknown>)             => api.get("/api/search/area",           { params }),
  heatmap:    (id: string, params?: Record<string, unknown>) => api.get(`/api/search/${id}/heatmap`,  { params }),
};

export const analyticsApi = {
  trends:           (params?: Record<string, unknown>) => api.get("/api/analytics/attendance-trends",    { params }),
  accuracy:         (params?: Record<string, unknown>) => api.get("/api/analytics/recognition-accuracy", { params }),
  health:           ()                                  => api.get("/api/analytics/system-health"),
  uptime:           (params?: Record<string, unknown>) => api.get("/api/analytics/camera-uptime",        { params }),
  faculty:          (params?: Record<string, unknown>) => api.get("/api/analytics/faculty-report",       { params }),
  student:          (id: string)                        => api.get(`/api/analytics/student-report/${id}`),
  flowMatrix:       (params?: Record<string, unknown>) => api.get("/api/analytics/flow-matrix",          { params }),
  occupancy:        (params?: Record<string, unknown>) => api.get("/api/analytics/occupancy-forecast",   { params }),
  ptzStats:         ()                                  => api.get("/api/analytics/ptz-stats"),
  dwellDistrib:     (params?: Record<string, unknown>) => api.get("/api/analytics/dwell-distribution",   { params }),
  frequentVisitors: (params?: Record<string, unknown>) => api.get("/api/analytics/frequent-visitors",    { params }),
  hourlyOccupancy:   (params?: Record<string, unknown>) => api.get("/api/analytics/hourly-occupancy",     { params }),
  courseAttendance:  (params?: Record<string, unknown>) => api.get("/api/analytics/course-attendance",   { params }),
};

export const adminApi = {
  // Clients
  clients:        (params?: Record<string, unknown>) => api.get("/api/admin/clients",                      { params }),
  client:         (id: string)                        => api.get(`/api/admin/clients/${id}`),
  createClient:   (data: unknown)                     => api.post("/api/admin/clients",                    data),
  updateClient:   (id: string, data: unknown)         => api.put(`/api/admin/clients/${id}`,               data),
  clientStatus:   (id: string, data: unknown)         => api.put(`/api/admin/clients/${id}/status`,        data),
  assignNode:     (id: string, data: unknown)         => api.post(`/api/admin/clients/${id}/nodes`,        data),
  removeNode:     (id: string, nodeId: string)        => api.delete(`/api/admin/clients/${id}/nodes/${nodeId}`),
  // GPU Nodes
  listNodes:      ()                                  => api.get("/api/admin/nodes"),
  registerNode:   (data: unknown)                     => api.post("/api/admin/nodes",                      data),
  drainNode:      (id: string)                        => api.put(`/api/admin/nodes/${id}/drain`, {}),
  deleteNode:     (id: string)                        => api.delete(`/api/admin/nodes/${id}`),
  // Users
  users:          (params?: Record<string, unknown>) => api.get("/api/admin/users",                        { params }),
  createUser:     (data: unknown)                     => api.post("/api/admin/users",                      data),
  updateUser:     (id: string, data: unknown)         => api.put(`/api/admin/users/${id}`,                 data),
  userStatus:     (id: string, data: unknown)         => api.put(`/api/admin/users/${id}/status`,          data),
  resetPassword:  (id: string)                        => api.post(`/api/admin/users/${id}/reset-password`),
  // Analytics
  usage:          ()                                  => api.get("/api/admin/analytics/usage"),
  platform:       ()                                  => api.get("/api/admin/analytics/platform"),
};

export const datasetsApi = {
  list:        (params?: Record<string, unknown>)  => api.get("/api/datasets",                              { params }),
  get:         (id: string)                         => api.get(`/api/datasets/${id}`),
  create:      (data: unknown)                      => api.post("/api/datasets",                            data),
  update:      (id: string, data: unknown)          => api.put(`/api/datasets/${id}`,                      data),
  archive:     (id: string)                         => api.delete(`/api/datasets/${id}`),
  persons:     (id: string, params?: Record<string, unknown>) => api.get(`/api/datasets/${id}/persons`,    { params }),
  movePerson:  (datasetId: string, personId: string, targetDatasetId: string) =>
    api.post(`/api/datasets/${datasetId}/persons/${personId}/move`, { target_dataset_id: targetDatasetId }),
  ensureDefault: ()                                 => api.post("/api/datasets/ensure-default"),
};

export const timetableApi = {
  list:         ()                                    => api.get("/api/timetables"),
  get:          (id: string)                          => api.get(`/api/timetables/${id}`),
  create:       (data: unknown)                       => api.post("/api/timetables",             data),
  update:       (id: string, data: unknown)           => api.patch(`/api/timetables/${id}`,      data),
  remove:       (id: string)                          => api.delete(`/api/timetables/${id}`),
  seedDefault:  ()                                    => api.post("/api/timetables/seed-default", {}),
  addEntry:     (id: string, data: unknown)           => api.post(`/api/timetables/${id}/entries`,              data),
  updateEntry:  (id: string, eid: string, data: unknown) => api.put(`/api/timetables/${id}/entries/${eid}`,     data),
  removeEntry:  (id: string, eid: string)             => api.delete(`/api/timetables/${id}/entries/${eid}`),
};

export const nodeApi = {
  info:   ()                                                    => api.get("/api/node/info"),
  reload: ()                                                    => api.post("/api/node/config/reload"),
  /** Start a PTZ brain session — routes to the camera's own GPU node. */
  start:  (id: string, camera?: { node_api_endpoint?: string | null }) => {
    const base = camera ? _nodeBase(camera) : (process.env.NEXT_PUBLIC_API_URL ?? "");
    // Must send a JSON body ({}) so FastAPI can parse StartCameraRequest (all fields optional)
    return _nodeApi(base).post(`/api/node/cameras/${id}/start`, {});
  },
  /** Stop a PTZ brain session — routes to the camera's own GPU node. */
  stop:   (id: string, camera?: { node_api_endpoint?: string | null }) => {
    const base = camera ? _nodeBase(camera) : (process.env.NEXT_PUBLIC_API_URL ?? "");
    return _nodeApi(base).post(`/api/node/cameras/${id}/stop`, {});
  },
  /** Real-time monitoring status for a camera. */
  status: (id: string, camera?: { node_api_endpoint?: string | null }) => {
    const base = camera ? _nodeBase(camera) : (process.env.NEXT_PUBLIC_API_URL ?? "");
    return _nodeApi(base).get(`/api/node/cameras/${id}/status`);
  },
};
