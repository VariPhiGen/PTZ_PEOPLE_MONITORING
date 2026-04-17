"use client";

import {
  useCallback, useEffect, useMemo, useRef, useState,
} from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  ArrowDown, ArrowDownLeft, ArrowDownRight, ArrowLeft, ArrowRight,
  ArrowUp, ArrowUpLeft, ArrowUpRight,
  Bot, BotOff, Brain,
  Camera as CameraIcon, Check, ChevronDown, Home, Loader2,
  MapPin, MousePointer2, Move3d, Navigation, Pencil, Play, Plus, RefreshCw, Search,
  Square, Trash2, Users, Wifi, WifiOff, ZoomIn, ZoomOut,
} from "lucide-react";
import { toast } from "sonner";
import { camerasApi, nodeApi, sessionsApi, datasetsApi, adminApi, timetableApi } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from "@/components/ui/dialog";
import { Card, CardContent } from "@/components/ui/card";
import { cn, CAMERA_STATUS_COLOR } from "@/lib/utils";
import { useAuth } from "@/hooks/useAuth";
import type { Camera, ZonePoint, ZonePolygon } from "@/types";

// ─────────────────────────────────────────────────────────────────────────────
// Shared helpers
// ─────────────────────────────────────────────────────────────────────────────

const WEEKDAYS = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"];

const MODE_COLORS: Record<string, string> = {
  ATTENDANCE:  "bg-blue-500/15 text-blue-400",
  MONITORING:  "bg-violet-500/15 text-violet-400",
  BOTH:        "bg-cyan-500/15 text-cyan-400",
};

const STATUS_INDICATOR: Record<string, string> = {
  ONLINE:   "bg-green-500",
  OFFLINE:  "bg-zinc-500",
  DEGRADED: "bg-amber-500",
};

interface AvailableNode {
  node_id:   string;
  node_name: string;
  location:  string | null;
  active_cameras: number;
  max_cameras:    number;
}

// ─────────────────────────────────────────────────────────────────────────────
// PTZ ↔ pixel zone transform  (mirrors ONVIFController exactly)
// ─────────────────────────────────────────────────────────────────────────────

/** Log-linear FOV model — matches backend ONVIFController defaults. */
const _DEF_FOV_H_WIDE   = 60.0;   // degrees at zoom = 0
const _DEF_FOV_H_NARROW = 3.0;    // degrees at zoom = 1
const _DEF_FOV_V_WIDE   = 34.0;
const _DEF_FOV_V_NARROW = 1.7;
/** Default ONVIF pan/tilt range (pan_max - pan_min). */
const _PAN_RANGE  = 2.0;   // -1 … +1
const _TILT_RANGE = 2.0;

type PtzPos = { pan: number; tilt: number; zoom: number };

/**
 * FOV helpers — per-camera wide AND narrow overrides eliminate zoom-axis drift.
 * The narrow value matters most: a wrong narrow FOV causes exponential projection
 * error at high zoom, making the zone appear to shrink / drift away as zoom increases.
 */
function _fovH(zoom: number, fovWide?: number | null, fovNarrow?: number | null) {
  const wide   = fovWide   ?? _DEF_FOV_H_WIDE;
  const narrow = fovNarrow ?? _DEF_FOV_H_NARROW;
  return wide * Math.exp(zoom * Math.log(narrow / wide));
}
function _fovV(zoom: number, fovWide?: number | null, fovNarrow?: number | null) {
  const wide   = fovWide   ?? _DEF_FOV_V_WIDE;
  const narrow = fovNarrow ?? _DEF_FOV_V_NARROW;
  return wide * Math.exp(zoom * Math.log(narrow / wide));
}

type FovOverride = {
  fovH?:       number | null;  // wide horizontal FOV (degrees at zoom=0)
  fovV?:       number | null;  // wide vertical   FOV
  fovHNarrow?: number | null;  // narrow horizontal FOV (degrees at zoom=1)
  fovVNarrow?: number | null;  // narrow vertical   FOV
  panRange?:   number;         // ONVIF pan range (pan_max - pan_min)
  tiltRange?:  number;
  panScale?:   number;         // physical_pan_range / 180° — corrects pan/tilt drift
  tiltScale?:  number;         // physical_tilt_range / 90°
};

/**
 * Convert pixel coordinate to ONVIF pan offset.
 * panScale corrects for cameras whose physical pan range ≠ 180°.
 * Formula assumes: 1 ONVIF unit = (180° / panScale) physical degrees.
 * Default panScale=1 → 180° range (ONVIF -1..+1 = 180° physical).
 * For 350° cameras set panScale = 350/180 ≈ 1.944.
 */
function _pixToONVIF(cx: number, fovDeg: number, range: number, scale: number) {
  return cx * (fovDeg / 360.0) * range / scale;
}
function _ONVIFToPix(delta: number, fovDeg: number, range: number, scale: number) {
  return delta * scale / ((fovDeg / 360.0) * range);
}

// ─────────────────────────────────────────────────────────────────────────────
// Auto-calibration math — one-click exact correction from observed drift
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Auto-compute pan scale from a two-point observation.
 * 1. Anchor was placed at (anchorPix, anchorPtz).
 * 2. Camera panned to curPtz. User clicks the actual landmark position (truePix).
 * 3. Algebra (derived from projection equations) gives exact pan_scale.
 */
function autoComputePanScale(
  anchorPix: ZonePoint, truePix: ZonePoint,
  anchorPtz: PtzPos, curPtz: PtzPos, fov: FovOverride,
): number | null {
  const panDelta = curPtz.pan - anchorPtz.pan;
  if (Math.abs(panDelta) < 0.005) return null;          // not enough pan movement
  const cx0    = anchorPix.x * 2 - 1;                  // -1..+1
  const cxTrue = truePix.x    * 2 - 1;
  const fovH   = _fovH(anchorPtz.zoom, fov.fovH, fov.fovHNarrow);
  const pr     = fov.panRange ?? _PAN_RANGE;
  // cx_true = cx0 + (p0-p1)*ps*360/(fovH*pr)  ⇒  ps = (cxTrue-cx0)*fovH*pr / (360*(p0-p1))
  const ps = (cxTrue - cx0) * fovH * pr / (360 * (anchorPtz.pan - curPtz.pan));
  return ps > 0.3 && ps < 5.0 ? ps : null;
}

/**
 * Auto-compute narrow FOV from a zoom-drift observation.
 * Anchor at zoom≈0; camera zoomed to curPtz.zoom. User clicks true position.
 */
function autoComputeFovNarrow(
  anchorPix: ZonePoint, truePix: ZonePoint,
  anchorPtz: PtzPos, curPtz: PtzPos, fov: FovOverride,
): number | null {
  if (Math.abs(curPtz.zoom - anchorPtz.zoom) < 0.05) return null;  // not enough zoom change
  const cx0    = anchorPix.x * 2 - 1;
  const cxTrue = truePix.x    * 2 - 1;
  if (Math.abs(cxTrue) < 0.05 || Math.abs(cx0) < 0.05) return null; // too close to center
  const fovAtZ0  = _fovH(anchorPtz.zoom, fov.fovH, fov.fovHNarrow);
  // fov_at_z1 = cx0 * fov_at_z0 / cx_true
  const fovAtZ1  = cx0 * fovAtZ0 / cxTrue;
  const fovWide  = fov.fovH ?? _DEF_FOV_H_WIDE;
  if (fovAtZ1 <= 0 || fovAtZ1 >= fovWide) return null;
  const z1 = curPtz.zoom;
  // Invert log-linear: fov_narrow = fov_wide * (fov_at_z1/fov_wide)^(1/z1)
  const fovNarrow = fovWide * Math.exp(Math.log(fovAtZ1 / fovWide) / z1);
  return fovNarrow > 0.2 && fovNarrow < fovWide ? fovNarrow : null;
}

/**
 * Auto-compute tilt scale from tilt-drift observation.
 */
function autoComputeTiltScale(
  anchorPix: ZonePoint, truePix: ZonePoint,
  anchorPtz: PtzPos, curPtz: PtzPos, fov: FovOverride,
): number | null {
  const tiltDelta = curPtz.tilt - anchorPtz.tilt;
  if (Math.abs(tiltDelta) < 0.005) return null;
  const cy0    = -(anchorPix.y * 2 - 1);   // invert Y: canvas↓ = camera↑
  const cyTrue = -(truePix.y    * 2 - 1);
  const fovV   = _fovV(anchorPtz.zoom, fov.fovV, fov.fovVNarrow);
  const tr     = fov.tiltRange ?? _TILT_RANGE;
  const ts = (cyTrue - cy0) * fovV * tr / (180 * (anchorPtz.tilt - curPtz.tilt));
  return ts > 0.3 && ts < 5.0 ? ts : null;
}

/** World-space PTZ point — absolute pan/tilt in ONVIF units, zoom-independent. */
type WorldPoint = { pan: number; tilt: number };

function pixelPolyToWorld(pts: ZonePolygon, ref: PtzPos, fov?: FovOverride): WorldPoint[] {
  const fhRef = _fovH(ref.zoom, fov?.fovH, fov?.fovHNarrow);
  const fvRef = _fovV(ref.zoom, fov?.fovV, fov?.fovVNarrow);
  const pr = fov?.panRange  ?? _PAN_RANGE;
  const tr = fov?.tiltRange ?? _TILT_RANGE;
  const ps = fov?.panScale  ?? 1.0;
  const ts = fov?.tiltScale ?? 1.0;
  return pts.map(({ x, y }) => ({
    pan:  ref.pan  + _pixToONVIF( x * 2 - 1,    fhRef, pr, ps),
    tilt: ref.tilt + _pixToONVIF(-(y * 2 - 1),  fvRef, tr, ts),
  }));
}

function worldPolyToPixel(world: WorldPoint[], cur: PtzPos, fov?: FovOverride): ZonePolygon {
  const fhCur = _fovH(cur.zoom, fov?.fovH, fov?.fovHNarrow);
  const fvCur = _fovV(cur.zoom, fov?.fovV, fov?.fovVNarrow);
  const pr = fov?.panRange  ?? _PAN_RANGE;
  const tr = fov?.tiltRange ?? _TILT_RANGE;
  const ps = fov?.panScale  ?? 1.0;
  const ts = fov?.tiltScale ?? 1.0;
  return world.map(({ pan, tilt }) => {
    const nx =  _ONVIFToPix(pan  - cur.pan,  fhCur, pr, ps);
    const ny = -_ONVIFToPix(tilt - cur.tilt, fvCur, tr, ts);
    return { x: nx / 2 + 0.5, y: ny / 2 + 0.5 };
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Camera Card
// ─────────────────────────────────────────────────────────────────────────────

function CameraCard({
  camera, onOpen, onStart, onStop, onTest, onDelete, canEdit, monitoring,
}: {
  camera:     Camera;
  onOpen:     () => void;
  onStart:    () => void;
  onStop:     () => void;
  onTest:     () => void;
  onDelete:   () => void;
  canEdit:    boolean;
  monitoring: boolean;
}) {
  return (
    <div
      className={cn(
        "group relative flex flex-col rounded-xl border bg-card overflow-hidden cursor-pointer transition-all",
        monitoring ? "border-emerald-500/40 hover:border-emerald-500/60" : "border-border/60 hover:border-zinc-600",
      )}
      onClick={onOpen}
    >
      {/* Status stripe */}
      <div className={cn("absolute top-0 left-0 right-0 h-0.5", monitoring ? "bg-emerald-500" : STATUS_INDICATOR[camera.status])} />

      {/* Thumbnail area */}
      <div className="relative h-32 bg-zinc-900 flex items-center justify-center border-b border-border/40">
        <CameraIcon className="h-10 w-10 text-zinc-700" />
        <div className={cn(
          "absolute top-2.5 right-2.5 h-2 w-2 rounded-full",
          STATUS_INDICATOR[camera.status],
          camera.status === "ONLINE" && "animate-pulse-dot",
        )} />
        {monitoring && (
          <div className="absolute top-2 left-2 flex items-center gap-1 rounded-full bg-emerald-500/20 border border-emerald-500/30 px-2 py-0.5">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
            <span className="text-[9px] font-semibold text-emerald-400">AI ACTIVE</span>
          </div>
        )}
      </div>

      <div className="p-3 flex-1">
        <div className="flex items-start justify-between gap-2 mb-1.5">
          <p className="text-sm font-semibold text-zinc-100 truncate leading-tight">{camera.name}</p>
          <span className={cn("shrink-0 rounded-full px-2 py-0.5 text-[10px] font-semibold", MODE_COLORS[camera.mode])}>
            {camera.mode}
          </span>
        </div>
        <p className="text-xs text-zinc-500">
          {camera.room_name}
          {camera.building && ` · ${camera.building}`}
          {camera.floor && ` F${camera.floor}`}
        </p>
        <p className="text-[10px] text-zinc-600 mt-0.5 font-mono truncate">{camera.onvif_host}</p>

        {canEdit && (
          <div
            className="flex items-center gap-1 mt-3"
            onClick={(e) => e.stopPropagation()}
          >
            {monitoring ? (
              <Button
                size="sm" variant="outline"
                className="flex-1 h-7 text-xs text-red-400 border-red-500/30 hover:bg-red-500/10"
                onClick={onStop}
              >
                <Square className="h-3 w-3 mr-1" />Stop
              </Button>
            ) : (
              <Button
                size="sm" variant="outline" className="flex-1 h-7 text-xs"
                title="Start AI scanning"
                onClick={onStart}
              >
                <Play className="h-3 w-3 mr-1" />Start
              </Button>
            )}
            <Button size="sm" variant="outline" className="h-7 text-xs px-2" title="Test ONVIF" onClick={onTest}>
              <Wifi className="h-3 w-3" />
            </Button>
            <Button
              size="sm" variant="outline"
              className="h-7 text-xs px-2 text-red-400 hover:text-red-300 hover:border-red-400"
              title="Delete camera"
              onClick={onDelete}
            >
              <Trash2 className="h-3 w-3" />
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Add Camera Dialog
// ─────────────────────────────────────────────────────────────────────────────

const CAMERA_TYPES = [
  { value: "PTZ",         label: "PTZ (Full — Pan · Tilt · Zoom)",       hasPresets: true,  hasZoom: true  },
  { value: "PTZ_ZOOM",    label: "PTZ Zoom Only (Zoom face-hunt only)",   hasPresets: true,  hasZoom: true  },
  { value: "BULLET_ZOOM", label: "Bullet Camera (Fixed + Zoom)",          hasPresets: false, hasZoom: true  },
  { value: "BULLET",      label: "Bullet Camera (Fixed, No Zoom)",        hasPresets: false, hasZoom: false },
];

interface AddCameraForm {
  name:             string;
  room_name:        string;
  building:         string;
  floor:            string;
  rtsp_url:         string;
  onvif_host:       string;
  onvif_port:       string;
  onvif_username:   string;
  onvif_password:   string;
  mode:             string;
  camera_type:      string;
  dataset_ids:      string[];
  timetable_id:     string;
  node_id:          string;
  alert_on_unknown: boolean;
  restricted_zone:  boolean;
  monitoring_hours: { start: string; end: string; days: string[] };
  ptz_protocol:     string;   // "ONVIF" | "ISAPI" | "CGI_DAHUA"
}

const PTZ_PROTOCOLS = [
  { value: "ONVIF",      label: "ONVIF (Standard / Universal)" },
  { value: "ISAPI",      label: "ISAPI (Hikvision Direct HTTP)" },
  { value: "CGI_DAHUA",  label: "CGI (Dahua / Amcrest Direct HTTP)" },
  { value: "CGI_CPPLUS", label: "CGI (CPPlus / Dahua OEM)" },
];

const EMPTY_CAM_FORM = (): AddCameraForm => ({
  name: "", room_name: "", building: "", floor: "",
  rtsp_url: "", onvif_host: "", onvif_port: "80",
  onvif_username: "admin", onvif_password: "",
  mode: "ATTENDANCE", camera_type: "PTZ", dataset_ids: [], timetable_id: "", node_id: "",
  alert_on_unknown: false, restricted_zone: false,
  monitoring_hours: { start: "08:00", end: "18:00", days: ["MON","TUE","WED","THU","FRI"] },
  ptz_protocol: "ONVIF",
});

function ToggleSwitch({
  checked, onChange, label, description,
}: {
  checked: boolean; onChange: (v: boolean) => void; label: string; description?: string;
}) {
  return (
    <div className="flex items-start justify-between gap-4">
      <div>
        <p className="text-sm font-medium text-zinc-200">{label}</p>
        {description && <p className="text-xs text-muted-foreground mt-0.5">{description}</p>}
      </div>
      <button
        type="button"
        onClick={() => onChange(!checked)}
        className={cn(
          "relative h-5 w-9 shrink-0 rounded-full border-2 border-transparent transition-colors",
          checked ? "bg-zinc-300" : "bg-zinc-700",
        )}
      >
        <span className={cn(
          "inline-block h-4 w-4 rounded-full bg-white shadow transition-transform",
          checked ? "translate-x-4" : "translate-x-0",
        )} />
      </button>
    </div>
  );
}

function AddCameraDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const qc  = useQueryClient();
  const { clientId, isSuperAdmin } = useAuth();
  const [form, setForm]   = useState<AddCameraForm>(EMPTY_CAM_FORM());
  const [error, setError] = useState<string | null>(null);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<"ok" | "fail" | null>(null);
  // SUPER_ADMIN picks which client this camera belongs to
  const [adminClientId, setAdminClientId] = useState<string>("");
  const effectiveClientId = isSuperAdmin ? adminClientId : (clientId ?? "");

  const set = <K extends keyof AddCameraForm>(k: K, v: AddCameraForm[K]) =>
    setForm((s) => ({ ...s, [k]: v }));

  const { data: clientsRaw } = useQuery({
    queryKey: ["admin-clients-picker"],
    queryFn:  () => adminApi.clients().then((r) => r.data),
    enabled:  open && isSuperAdmin,
    staleTime: 120_000,
  });
  const clientOptions: Array<{ client_id: string; name: string }> =
    ((clientsRaw as { items?: any[] })?.items ?? (Array.isArray(clientsRaw) ? clientsRaw : []))
      .map((c: any) => ({ client_id: c.client_id, name: c.name }));

  const { data: nodesRaw } = useQuery({
    queryKey: ["available-nodes"],
    queryFn:  () => camerasApi.nodes().then((r) => r.data),
    enabled:  open,
  });
  const nodes: AvailableNode[] =
    (nodesRaw as { nodes?: AvailableNode[] })?.nodes ??
    (Array.isArray(nodesRaw) ? (nodesRaw as AvailableNode[]) : []);

  const { data: datasetsRaw } = useQuery({
    queryKey: ["datasets", effectiveClientId],
    queryFn:  () => datasetsApi.list(effectiveClientId ? { client_id: effectiveClientId } : undefined).then((r) => r.data),
    enabled:  open && !!effectiveClientId,
  });
  const datasets: Array<{ dataset_id: string; name: string; color: string }> =
    (datasetsRaw as { items?: Array<{ dataset_id: string; name: string; color: string }> })?.items ?? [];

  const { data: timetablesRaw } = useQuery({
    queryKey: ["timetables"],
    queryFn:  () => timetableApi.list().then((r) => r.data),
    enabled:  open,
  });
  const timetables: Array<{ timetable_id: string; name: string }> =
    Array.isArray(timetablesRaw) ? timetablesRaw : [];

  const mutation = useMutation({
    mutationFn: () => camerasApi.create({
      name:             form.name,
      room_name:        form.room_name,
      building:         form.building  || undefined,
      floor:            form.floor     || undefined,
      rtsp_url:         form.rtsp_url,
      onvif_host:       form.onvif_host,
      onvif_port:       parseInt(form.onvif_port) || 80,
      onvif_username:   form.onvif_username,
      onvif_password:   form.onvif_password,
      mode:             form.mode,
      camera_type:      form.camera_type,
      ptz_protocol:     ["PTZ","PTZ_ZOOM"].includes(form.camera_type) ? form.ptz_protocol || undefined : undefined,
      dataset_ids:      form.dataset_ids.length ? form.dataset_ids : undefined,
      timetable_id:     form.timetable_id || undefined,
      node_id:          form.node_id     || undefined,
      alert_on_unknown: form.alert_on_unknown,
      restricted_zone:  form.restricted_zone,
      monitoring_hours: ["MONITORING", "BOTH"].includes(form.mode) ? form.monitoring_hours : undefined,
      ...(isSuperAdmin && adminClientId ? { client_id: adminClientId } : {}),
    }),
    onSuccess: () => {
      toast.success(`Camera "${form.name}" added`);
      qc.invalidateQueries({ queryKey: ["cameras"] });
      handleClose();
    },
    onError: (err: unknown) => {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setError(typeof detail === "string" ? detail : "Failed to add camera");
    },
  });

  async function handleTest() {
    if (!form.rtsp_url && !form.onvif_host) {
      setError("Enter RTSP URL or ONVIF host before testing");
      return;
    }
    setTesting(true);
    setTestResult(null);
    try {
      await camerasApi.testConnection({
        rtsp_url:       form.rtsp_url  || undefined,
        onvif_host:     form.onvif_host,
        onvif_port:     parseInt(form.onvif_port) || 80,
        onvif_username: form.onvif_username,
        onvif_password: form.onvif_password,
      });
      setTestResult("ok");
      toast.success("Connection successful");
    } catch {
      setTestResult("fail");
      toast.error("Connection failed — check RTSP URL / ONVIF credentials");
    } finally {
      setTesting(false);
    }
  }

  function handleClose() {
    setForm(EMPTY_CAM_FORM());
    setError(null);
    setTestResult(null);
    setAdminClientId("");
    onClose();
  }

  function handleSubmit() {
    setError(null);
    if (isSuperAdmin && !adminClientId) { setError("Select a client first"); return; }
    if (!form.name.trim())      { setError("Camera name is required");  return; }
    if (!form.room_name.trim()) { setError("Room name is required");     return; }
    if (!form.rtsp_url.trim())  { setError("RTSP URL is required");      return; }
    if (!form.onvif_host.trim()){ setError("ONVIF host is required");    return; }
    mutation.mutate();
  }

  const isMonitoring = ["MONITORING", "BOTH"].includes(form.mode);

  return (
    <Dialog open={open} onOpenChange={(o) => !o && handleClose()}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle>Add Camera</DialogTitle>
          <DialogDescription>Register an ONVIF/RTSP PTZ camera on a GPU node.</DialogDescription>
        </DialogHeader>

        <div className="flex-1 overflow-y-auto space-y-5 py-1 pr-1">
          {/* ── Client picker (SUPER_ADMIN only) ──────────── */}
          {isSuperAdmin && (
            <section>
              <p className="text-xs font-semibold uppercase tracking-wider text-zinc-500 mb-3">Client</p>
              <div className="space-y-1.5">
                <Label className="text-xs text-muted-foreground">Assign to client *</Label>
                <select
                  value={adminClientId}
                  onChange={e => { setAdminClientId(e.target.value); set("dataset_ids", []); }}
                  className="w-full h-9 px-3 rounded-md border border-zinc-700 bg-zinc-900 text-sm text-zinc-200"
                >
                  <option value="">Select client…</option>
                  {clientOptions.map(c => (
                    <option key={c.client_id} value={c.client_id}>{c.name}</option>
                  ))}
                </select>
              </div>
            </section>
          )}

          {/* ── Location ──────────────────────────────────── */}
          <section>
            <p className="text-xs font-semibold uppercase tracking-wider text-zinc-500 mb-3">Location</p>
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1.5 col-span-2">
                <Label className="text-xs text-muted-foreground">Camera name *</Label>
                <Input value={form.name} onChange={(e) => set("name", e.target.value)}
                  placeholder="Main Entrance PTZ" className="bg-zinc-900 border-zinc-700" />
              </div>
              <div className="space-y-1.5">
                <Label className="text-xs text-muted-foreground">Room / area *</Label>
                <Input value={form.room_name} onChange={(e) => set("room_name", e.target.value)}
                  placeholder="Lecture Hall A" className="bg-zinc-900 border-zinc-700" />
              </div>
              <div className="space-y-1.5">
                <Label className="text-xs text-muted-foreground">Building</Label>
                <Input value={form.building} onChange={(e) => set("building", e.target.value)}
                  placeholder="Block C" className="bg-zinc-900 border-zinc-700" />
              </div>
              <div className="space-y-1.5">
                <Label className="text-xs text-muted-foreground">Floor</Label>
                <Input value={form.floor} onChange={(e) => set("floor", e.target.value)}
                  placeholder="1" className="bg-zinc-900 border-zinc-700" />
              </div>
            </div>
          </section>

          {/* ── Connection ─────────────────────────────────── */}
          <section>
            <p className="text-xs font-semibold uppercase tracking-wider text-zinc-500 mb-3">Connection</p>
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1.5 col-span-2">
                <Label className="text-xs text-muted-foreground">RTSP URL *</Label>
                <Input value={form.rtsp_url} onChange={(e) => set("rtsp_url", e.target.value)}
                  placeholder="rtsp://camera-ip:554/stream" className="bg-zinc-900 border-zinc-700 font-mono text-xs" />
              </div>
              <div className="space-y-1.5">
                <Label className="text-xs text-muted-foreground">ONVIF host *</Label>
                <Input value={form.onvif_host} onChange={(e) => set("onvif_host", e.target.value)}
                  placeholder="192.168.1.100" className="bg-zinc-900 border-zinc-700 font-mono text-xs" />
              </div>
              <div className="space-y-1.5">
                <Label className="text-xs text-muted-foreground">ONVIF port</Label>
                <Input type="number" value={form.onvif_port} onChange={(e) => set("onvif_port", e.target.value)}
                  placeholder="80" className="bg-zinc-900 border-zinc-700" />
              </div>
              <div className="space-y-1.5">
                <Label className="text-xs text-muted-foreground">ONVIF username</Label>
                <Input value={form.onvif_username} onChange={(e) => set("onvif_username", e.target.value)}
                  className="bg-zinc-900 border-zinc-700" />
              </div>
              <div className="space-y-1.5">
                <Label className="text-xs text-muted-foreground">ONVIF password</Label>
                <Input type="password" value={form.onvif_password} onChange={(e) => set("onvif_password", e.target.value)}
                  className="bg-zinc-900 border-zinc-700" />
              </div>
            </div>

            {/* Test Connection */}
            <div className="mt-3 flex items-center gap-3">
              <Button
                type="button" size="sm" variant="outline"
                onClick={handleTest} disabled={testing}
                className={cn(
                  testResult === "ok"   && "border-green-500/50 text-green-400",
                  testResult === "fail" && "border-red-500/50 text-red-400",
                )}
              >
                {testing ? (
                  <><Loader2 className="h-3.5 w-3.5 animate-spin" />Testing…</>
                ) : testResult === "ok" ? (
                  <><Check className="h-3.5 w-3.5" />Connected</>
                ) : (
                  "Test Connection"
                )}
              </Button>
              {testResult && (
                <span className={cn("text-xs", testResult === "ok" ? "text-green-400" : "text-red-400")}>
                  {testResult === "ok" ? "RTSP + ONVIF reachable" : "Connection failed"}
                </span>
              )}
            </div>
          </section>

          {/* ── Mode + Node ─────────────────────────────────── */}
          <section>
            <p className="text-xs font-semibold uppercase tracking-wider text-zinc-500 mb-3">Mode & Assignment</p>
            <div className="grid grid-cols-2 gap-3">
              {/* Mode */}
              <div className="space-y-1.5">
                <Label className="text-xs text-muted-foreground">Camera mode *</Label>
                <div className="grid grid-cols-3 gap-1.5">
                  {(["ATTENDANCE", "MONITORING", "BOTH"] as const).map((m) => (
                    <button
                      key={m} type="button"
                      onClick={() => set("mode", m)}
                      className={cn(
                        "rounded-lg border py-2 text-xs font-medium transition-colors",
                        form.mode === m
                          ? "border-zinc-500 bg-zinc-800 text-zinc-100"
                          : "border-zinc-800 text-zinc-500 hover:border-zinc-600",
                      )}
                    >
                      {m === "ATTENDANCE" ? "Attend." : m === "MONITORING" ? "Monitor" : "Both"}
                    </button>
                  ))}
                </div>
              </div>

              {/* Face Datasets — multi-select */}
              <div className="space-y-1.5">
                <Label className="text-xs text-muted-foreground">
                  Face Datasets
                  <span className="text-zinc-600 ml-1">— recognition scope (none = client-wide)</span>
                </Label>
                {datasets.length === 0 ? (
                  <p className="text-xs text-zinc-500 italic">
                    {effectiveClientId ? "No datasets found for this client" : "Select a client first"}
                  </p>
                ) : (
                  <div className="rounded-md border border-zinc-700 bg-zinc-900 divide-y divide-zinc-800 max-h-36 overflow-y-auto">
                    {datasets.map((d) => {
                      const checked = form.dataset_ids.includes(d.dataset_id);
                      return (
                        <label key={d.dataset_id}
                          className="flex items-center gap-2.5 px-3 py-2 cursor-pointer hover:bg-zinc-800 transition-colors">
                          <input
                            type="checkbox"
                            className="h-3.5 w-3.5 accent-indigo-500"
                            checked={checked}
                            onChange={() => {
                              const next = checked
                                ? form.dataset_ids.filter((x) => x !== d.dataset_id)
                                : [...form.dataset_ids, d.dataset_id];
                              set("dataset_ids", next);
                            }}
                          />
                          <span className="h-2 w-2 rounded-full flex-shrink-0"
                            style={{ backgroundColor: d.color ?? "#6366f1" }} />
                          <span className="text-sm text-zinc-200">{d.name}</span>
                        </label>
                      );
                    })}
                  </div>
                )}
                {form.dataset_ids.length > 0 && (
                  <p className="text-xs text-indigo-400">{form.dataset_ids.length} dataset{form.dataset_ids.length > 1 ? "s" : ""} selected</p>
                )}
              </div>

              {/* Node */}
              <div className="space-y-1.5">
                <Label className="text-xs text-muted-foreground">Target GPU node</Label>
                <div className="relative">
                  <select
                    value={form.node_id}
                    onChange={(e) => set("node_id", e.target.value)}
                    className="w-full appearance-none rounded-md border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-zinc-100 focus:outline-none focus:ring-2 focus:ring-ring"
                  >
                    <option value="">Auto-assign</option>
                    {nodes.map((n) => (
                      <option key={n.node_id} value={n.node_id}>
                        {n.node_name} ({n.active_cameras}/{n.max_cameras})
                      </option>
                    ))}
                  </select>
                  <ChevronDown className="pointer-events-none absolute right-2.5 top-2.5 h-4 w-4 text-zinc-500" />
                </div>
              </div>

              {/* Timetable */}
              <div className="space-y-1.5">
                <Label className="text-xs text-muted-foreground">Timetable</Label>
                <div className="relative">
                  <select
                    value={form.timetable_id}
                    onChange={(e) => set("timetable_id", e.target.value)}
                    className="w-full appearance-none rounded-md border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-zinc-100 focus:outline-none focus:ring-2 focus:ring-ring"
                  >
                    <option value="">No timetable</option>
                    {timetables.map((t) => (
                      <option key={t.timetable_id} value={t.timetable_id}>{t.name}</option>
                    ))}
                  </select>
                  <ChevronDown className="pointer-events-none absolute right-2.5 top-2.5 h-4 w-4 text-zinc-500" />
                </div>
              </div>

              {/* Camera Type */}
              <div className="space-y-1.5 col-span-2">
                <Label className="text-xs text-muted-foreground">Camera Type</Label>
                <div className="relative">
                  <select
                    value={form.camera_type}
                    onChange={(e) => set("camera_type", e.target.value)}
                    className="w-full appearance-none rounded-md border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-zinc-100 focus:outline-none focus:ring-2 focus:ring-ring"
                  >
                    {CAMERA_TYPES.map((t) => (
                      <option key={t.value} value={t.value}>{t.label}</option>
                    ))}
                  </select>
                  <ChevronDown className="pointer-events-none absolute right-2.5 top-2.5 h-4 w-4 text-zinc-500" />
                </div>
              </div>

              {/* PTZ Protocol — only for PTZ / PTZ_ZOOM */}
              {["PTZ","PTZ_ZOOM"].includes(form.camera_type) && (
                <div className="space-y-1.5 col-span-2">
                  <Label className="text-xs text-muted-foreground">PTZ Control Protocol</Label>
                  <div className="relative">
                    <select
                      value={form.ptz_protocol}
                      onChange={(e) => set("ptz_protocol", e.target.value)}
                      className="w-full appearance-none rounded-md border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-zinc-100 focus:outline-none focus:ring-2 focus:ring-ring"
                    >
                      {PTZ_PROTOCOLS.map((p) => (
                        <option key={p.value} value={p.value}>{p.label}</option>
                      ))}
                    </select>
                    <ChevronDown className="pointer-events-none absolute right-2.5 top-2.5 h-4 w-4 text-zinc-500" />
                  </div>
                  <p className="text-[11px] text-zinc-500">ISAPI / CGI offer ~2 ms latency vs ONVIF's 50–200 ms. Save the camera first, then test the protocol from the Edit dialog.</p>
                </div>
              )}
            </div>
          </section>

          {/* ── Monitoring options ─────────────────────────── */}
          {isMonitoring && (
            <section className="rounded-xl border border-violet-500/20 bg-violet-500/5 p-4 space-y-4">
              <p className="text-xs font-semibold uppercase tracking-wider text-violet-400">Monitoring Settings</p>

              {/* Monitoring hours */}
              <div>
                <Label className="text-xs text-muted-foreground mb-2 block">Active hours</Label>
                <div className="flex items-center gap-3">
                  <Input type="time" value={form.monitoring_hours.start}
                    onChange={(e) => set("monitoring_hours", { ...form.monitoring_hours, start: e.target.value })}
                    className="bg-zinc-900 border-zinc-700 w-32" />
                  <span className="text-zinc-500 text-sm">to</span>
                  <Input type="time" value={form.monitoring_hours.end}
                    onChange={(e) => set("monitoring_hours", { ...form.monitoring_hours, end: e.target.value })}
                    className="bg-zinc-900 border-zinc-700 w-32" />
                </div>
                <div className="flex gap-1.5 mt-2">
                  {WEEKDAYS.map((d) => {
                    const active = form.monitoring_hours.days.includes(d);
                    return (
                      <button
                        key={d} type="button"
                        onClick={() => set("monitoring_hours", {
                          ...form.monitoring_hours,
                          days: active
                            ? form.monitoring_hours.days.filter((x) => x !== d)
                            : [...form.monitoring_hours.days, d],
                        })}
                        className={cn(
                          "rounded px-1.5 py-0.5 text-[10px] font-medium border transition-colors",
                          active ? "border-violet-400 bg-violet-400/15 text-violet-300" : "border-zinc-700 text-zinc-600",
                        )}
                      >
                        {d.slice(0, 2)}
                      </button>
                    );
                  })}
                </div>
              </div>

              <ToggleSwitch
                checked={form.restricted_zone}
                onChange={(v) => set("restricted_zone", v)}
                label="Restricted zone"
                description="Trigger alert when anyone enters this zone outside hours"
              />
              <ToggleSwitch
                checked={form.alert_on_unknown}
                onChange={(v) => set("alert_on_unknown", v)}
                label="Alert on unknown person"
                description="Send Kafka notification when an unrecognised face is detected"
              />
            </section>
          )}

          {/* Alert on unknown (for non-monitoring modes) */}
          {!isMonitoring && (
            <ToggleSwitch
              checked={form.alert_on_unknown}
              onChange={(v) => set("alert_on_unknown", v)}
              label="Alert on unknown person"
              description="Send Kafka notification when an unrecognised face is detected"
            />
          )}

          {error && (
            <p className="text-sm text-red-400 bg-red-400/10 rounded-lg px-3 py-2">{error}</p>
          )}
        </div>

        <DialogFooter className="mt-2 border-t border-border/50 pt-4">
          <Button variant="ghost" size="sm" onClick={handleClose}>Cancel</Button>
          <Button size="sm" onClick={handleSubmit} disabled={mutation.isPending}>
            {mutation.isPending ? (
              <><Loader2 className="h-4 w-4 animate-spin" />Adding…</>
            ) : (
              <><CameraIcon className="h-4 w-4" />Add Camera</>
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Edit Camera Dialog
// ─────────────────────────────────────────────────────────────────────────────

function EditCameraDialog({
  camera, open, onClose,
}: {
  camera: Camera | null;
  open:   boolean;
  onClose: () => void;
}) {
  const qc = useQueryClient();
  const { clientId } = useAuth();
  // Prefer the camera's own client_id (works for SUPER_ADMIN editing any client's camera)
  const effectiveClientId = (camera as any)?.client_id ?? clientId;
  const [form, setForm] = useState<AddCameraForm>(EMPTY_CAM_FORM());
  const [error, setError] = useState<string | null>(null);
  const [testingProto, setTestingProto] = useState(false);
  const [detectingProto, setDetectingProto] = useState(false);
  const [protoResult, setProtoResult] = useState<{ ok: boolean; latency_ms: number; detail: string } | null>(null);
  const [detectResult, setDetectResult] = useState<{ brand: string | null; model: string | null; recommended: string; detail: string; isapi_ok: boolean; cgi_ok: boolean } | null>(null);

  const set = <K extends keyof AddCameraForm>(k: K, v: AddCameraForm[K]) => {
    setForm((s) => ({ ...s, [k]: v }));
    // Reset protocol test result when protocol selection changes
    if (k === "ptz_protocol") setProtoResult(null);
  };

  // Pre-populate when camera changes
  useEffect(() => {
    if (!camera) return;
    setForm({
      name:             camera.name ?? "",
      room_name:        (camera as any).room_name ?? "",
      building:         (camera as any).building ?? "",
      floor:            (camera as any).floor ?? "",
      rtsp_url:         camera.rtsp_url ?? "",
      onvif_host:       camera.onvif_host ?? "",
      onvif_port:       String(camera.onvif_port ?? 80),
      onvif_username:   camera.onvif_username ?? "admin",
      onvif_password:   "",          // never pre-fill password
      mode:             camera.mode ?? "ATTENDANCE",
      camera_type:      (camera as any).camera_type ?? "PTZ",
      dataset_ids:      (camera as any).dataset_ids ?? ((camera as any).dataset_id ? [(camera as any).dataset_id] : []),
      timetable_id:     (camera as any).timetable_id ?? "",
      node_id:          camera.node_id ?? "",
      alert_on_unknown: (camera as any).alert_on_unknown ?? false,
      restricted_zone:  (camera as any).restricted_zone ?? false,
      monitoring_hours: (camera as any).monitoring_hours ?? {
        start: "08:00", end: "18:00",
        days: ["MON","TUE","WED","THU","FRI"],
      },
      ptz_protocol: (camera as any).learned_params?.ptz_protocol ?? "ONVIF",
    });
    setError(null);
  }, [camera]);

  const { data: nodesRaw } = useQuery({
    queryKey: ["available-nodes"],
    queryFn:  () => camerasApi.nodes().then((r) => r.data),
    enabled:  open,
  });
  const nodes: AvailableNode[] =
    (nodesRaw as { nodes?: AvailableNode[] })?.nodes ??
    (Array.isArray(nodesRaw) ? (nodesRaw as AvailableNode[]) : []);

  const { data: datasetsRaw } = useQuery({
    queryKey: ["datasets", effectiveClientId],
    queryFn:  () => datasetsApi.list(effectiveClientId ? { client_id: effectiveClientId } : undefined).then((r) => r.data),
    enabled:  open && !!effectiveClientId,
  });
  const datasets: Array<{ dataset_id: string; name: string; color?: string }> =
    (datasetsRaw as { items?: Array<{ dataset_id: string; name: string; color?: string }> })?.items ?? [];

  const { data: editTimetablesRaw } = useQuery({
    queryKey: ["timetables"],
    queryFn:  () => timetableApi.list().then((r) => r.data),
    enabled:  open,
  });
  const editTimetables: Array<{ timetable_id: string; name: string }> =
    Array.isArray(editTimetablesRaw) ? editTimetablesRaw : [];

  const mutation = useMutation({
    mutationFn: () => {
      if (!camera) throw new Error("No camera");
      const payload: Record<string, unknown> = {
        name:             form.name,
        room_name:        form.room_name,
        building:         form.building  || undefined,
        floor:            form.floor     || undefined,
        rtsp_url:         form.rtsp_url,
        onvif_host:       form.onvif_host,
        onvif_port:       parseInt(form.onvif_port) || 80,
        onvif_username:   form.onvif_username,
        mode:             form.mode,
        camera_type:      form.camera_type || null,
        ptz_protocol:     ["PTZ","PTZ_ZOOM"].includes(form.camera_type) ? form.ptz_protocol || null : null,
        dataset_ids:      form.dataset_ids.length ? form.dataset_ids : null,
        timetable_id:     form.timetable_id || null,
        node_id:          form.node_id     || null,
        alert_on_unknown: form.alert_on_unknown,
        restricted_zone:  form.restricted_zone,
        monitoring_hours: ["MONITORING","BOTH"].includes(form.mode) ? form.monitoring_hours : undefined,
      };
      if (form.onvif_password) payload.onvif_password = form.onvif_password;
      return camerasApi.update(camera.camera_id, payload);
    },
    onSuccess: () => {
      toast.success(`Camera "${form.name}" updated`);
      qc.invalidateQueries({ queryKey: ["cameras"] });
      onClose();
    },
    onError: (err: unknown) => {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setError(typeof detail === "string" ? detail : "Failed to update camera");
    },
  });

  function handleSubmit() {
    setError(null);
    if (!form.name.trim())      { setError("Camera name is required");  return; }
    if (!form.room_name.trim()) { setError("Room name is required");     return; }
    if (!form.rtsp_url.trim())  { setError("RTSP URL is required");      return; }
    if (!form.onvif_host.trim()){ setError("ONVIF host is required");    return; }
    mutation.mutate();
  }

  async function handleTestProtocol() {
    if (!camera) return;
    setTestingProto(true);
    setProtoResult(null);
    try {
      const res = await camerasApi.testProtocol(camera.camera_id, form.ptz_protocol, camera as any);
      setProtoResult(res.data as { ok: boolean; latency_ms: number; detail: string });
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setProtoResult({ ok: false, latency_ms: 0, detail: detail ?? "Request failed" });
    } finally {
      setTestingProto(false);
    }
  }

  async function handleDetectProtocol() {
    if (!camera) return;
    setDetectingProto(true);
    setDetectResult(null);
    setProtoResult(null);
    try {
      const res = await camerasApi.detectProtocol(camera.camera_id, camera as any);
      const d = res.data as { brand: string | null; model: string | null; recommended: string; detail: string; isapi_ok: boolean; cgi_ok: boolean };
      setDetectResult(d);
      // Auto-select the recommended protocol
      set("ptz_protocol", d.recommended);
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setDetectResult({ brand: null, model: null, recommended: "ONVIF", detail: detail ?? "Detection failed", isapi_ok: false, cgi_ok: false });
    } finally {
      setDetectingProto(false);
    }
  }

  const isMonitoring = ["MONITORING", "BOTH"].includes(form.mode);

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle>Edit Camera</DialogTitle>
          <DialogDescription>Update camera settings, dataset, or GPU node assignment.</DialogDescription>
        </DialogHeader>

        <div className="flex-1 overflow-y-auto space-y-5 py-1 pr-1">
          {/* Location */}
          <section>
            <p className="text-xs font-semibold uppercase tracking-wider text-zinc-500 mb-3">Location</p>
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1.5 col-span-2">
                <Label className="text-xs text-muted-foreground">Camera name *</Label>
                <Input value={form.name} onChange={(e) => set("name", e.target.value)}
                  className="bg-zinc-900 border-zinc-700" />
              </div>
              <div className="space-y-1.5">
                <Label className="text-xs text-muted-foreground">Room / area *</Label>
                <Input value={form.room_name} onChange={(e) => set("room_name", e.target.value)}
                  className="bg-zinc-900 border-zinc-700" />
              </div>
              <div className="space-y-1.5">
                <Label className="text-xs text-muted-foreground">Building</Label>
                <Input value={form.building} onChange={(e) => set("building", e.target.value)}
                  className="bg-zinc-900 border-zinc-700" />
              </div>
              <div className="space-y-1.5">
                <Label className="text-xs text-muted-foreground">Floor</Label>
                <Input value={form.floor} onChange={(e) => set("floor", e.target.value)}
                  className="bg-zinc-900 border-zinc-700" />
              </div>
            </div>
          </section>

          {/* Connection */}
          <section>
            <p className="text-xs font-semibold uppercase tracking-wider text-zinc-500 mb-3">Connection</p>
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1.5 col-span-2">
                <Label className="text-xs text-muted-foreground">RTSP URL *</Label>
                <Input value={form.rtsp_url} onChange={(e) => set("rtsp_url", e.target.value)}
                  className="bg-zinc-900 border-zinc-700 font-mono text-xs" />
              </div>
              <div className="space-y-1.5">
                <Label className="text-xs text-muted-foreground">ONVIF host *</Label>
                <Input value={form.onvif_host} onChange={(e) => set("onvif_host", e.target.value)}
                  className="bg-zinc-900 border-zinc-700 font-mono text-xs" />
              </div>
              <div className="space-y-1.5">
                <Label className="text-xs text-muted-foreground">ONVIF port</Label>
                <Input type="number" value={form.onvif_port} onChange={(e) => set("onvif_port", e.target.value)}
                  className="bg-zinc-900 border-zinc-700" />
              </div>
              <div className="space-y-1.5">
                <Label className="text-xs text-muted-foreground">ONVIF username</Label>
                <Input value={form.onvif_username} onChange={(e) => set("onvif_username", e.target.value)}
                  className="bg-zinc-900 border-zinc-700" />
              </div>
              <div className="space-y-1.5">
                <Label className="text-xs text-muted-foreground">ONVIF password <span className="text-zinc-600">(leave blank to keep current)</span></Label>
                <Input type="password" value={form.onvif_password} onChange={(e) => set("onvif_password", e.target.value)}
                  placeholder="••••••••" className="bg-zinc-900 border-zinc-700" />
              </div>
            </div>
          </section>

          {/* Mode, Dataset, Node */}
          <section>
            <p className="text-xs font-semibold uppercase tracking-wider text-zinc-500 mb-3">Mode &amp; Assignment</p>
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1.5">
                <Label className="text-xs text-muted-foreground">Camera mode *</Label>
                <div className="grid grid-cols-3 gap-1.5">
                  {(["ATTENDANCE", "MONITORING", "BOTH"] as const).map((m) => (
                    <button key={m} type="button" onClick={() => set("mode", m)}
                      className={cn(
                        "rounded-lg border py-2 text-xs font-medium transition-colors",
                        form.mode === m
                          ? "border-zinc-500 bg-zinc-800 text-zinc-100"
                          : "border-zinc-800 text-zinc-500 hover:border-zinc-600",
                      )}>
                      {m === "ATTENDANCE" ? "Attend." : m === "MONITORING" ? "Monitor" : "Both"}
                    </button>
                  ))}
                </div>
              </div>

              {/* Face Datasets — multi-select */}
              <div className="space-y-1.5">
                <Label className="text-xs text-muted-foreground">
                  Face Datasets
                  <span className="text-zinc-600 ml-1">— recognition scope (none = client-wide)</span>
                </Label>
                {datasets.length === 0 ? (
                  <p className="text-xs text-zinc-500 italic">No datasets available</p>
                ) : (
                  <div className="rounded-md border border-zinc-700 bg-zinc-900 divide-y divide-zinc-800 max-h-36 overflow-y-auto">
                    {datasets.map((d) => {
                      const checked = form.dataset_ids.includes(d.dataset_id);
                      return (
                        <label key={d.dataset_id}
                          className="flex items-center gap-2.5 px-3 py-2 cursor-pointer hover:bg-zinc-800 transition-colors">
                          <input
                            type="checkbox"
                            className="h-3.5 w-3.5 accent-indigo-500"
                            checked={checked}
                            onChange={() => {
                              const next = checked
                                ? form.dataset_ids.filter((x) => x !== d.dataset_id)
                                : [...form.dataset_ids, d.dataset_id];
                              set("dataset_ids", next);
                            }}
                          />
                          <span className="h-2 w-2 rounded-full flex-shrink-0"
                            style={{ backgroundColor: (d as any).color ?? "#6366f1" }} />
                          <span className="text-sm text-zinc-200">{d.name}</span>
                        </label>
                      );
                    })}
                  </div>
                )}
                {form.dataset_ids.length > 0 && (
                  <p className="text-xs text-indigo-400">{form.dataset_ids.length} dataset{form.dataset_ids.length > 1 ? "s" : ""} selected</p>
                )}
              </div>

              {/* GPU Node */}
              <div className="space-y-1.5">
                <Label className="text-xs text-muted-foreground">Target GPU node</Label>
                <div className="relative">
                  <select value={form.node_id} onChange={(e) => set("node_id", e.target.value)}
                    className="w-full appearance-none rounded-md border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-zinc-100 focus:outline-none focus:ring-2 focus:ring-ring">
                    <option value="">Auto-assign</option>
                    {nodes.map((n) => (
                      <option key={n.node_id} value={n.node_id}>
                        {n.node_name} ({n.active_cameras}/{n.max_cameras})
                      </option>
                    ))}
                  </select>
                  <ChevronDown className="pointer-events-none absolute right-2.5 top-2.5 h-4 w-4 text-zinc-500" />
                </div>
              </div>

              {/* Timetable */}
              <div className="space-y-1.5">
                <Label className="text-xs text-muted-foreground">Timetable</Label>
                <div className="relative">
                  <select value={form.timetable_id} onChange={(e) => set("timetable_id", e.target.value)}
                    className="w-full appearance-none rounded-md border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-zinc-100 focus:outline-none focus:ring-2 focus:ring-ring">
                    <option value="">No timetable</option>
                    {editTimetables.map((t) => (
                      <option key={t.timetable_id} value={t.timetable_id}>{t.name}</option>
                    ))}
                  </select>
                  <ChevronDown className="pointer-events-none absolute right-2.5 top-2.5 h-4 w-4 text-zinc-500" />
                </div>
              </div>

              {/* Camera Type */}
              <div className="space-y-1.5 col-span-2">
                <Label className="text-xs text-muted-foreground">Camera Type</Label>
                <div className="relative">
                  <select value={form.camera_type} onChange={(e) => set("camera_type", e.target.value)}
                    className="w-full appearance-none rounded-md border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-zinc-100 focus:outline-none focus:ring-2 focus:ring-ring">
                    {CAMERA_TYPES.map((t) => (
                      <option key={t.value} value={t.value}>{t.label}</option>
                    ))}
                  </select>
                  <ChevronDown className="pointer-events-none absolute right-2.5 top-2.5 h-4 w-4 text-zinc-500" />
                </div>
              </div>

              {/* PTZ Protocol — only for PTZ / PTZ_ZOOM */}
              {["PTZ","PTZ_ZOOM"].includes(form.camera_type) && (
                <div className="space-y-2 col-span-2">
                  <Label className="text-xs text-muted-foreground">PTZ Control Protocol</Label>
                  <div className="flex gap-2">
                    <div className="relative flex-1">
                      <select
                        value={form.ptz_protocol}
                        onChange={(e) => set("ptz_protocol", e.target.value)}
                        className="w-full appearance-none rounded-md border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-zinc-100 focus:outline-none focus:ring-2 focus:ring-ring"
                      >
                        {PTZ_PROTOCOLS.map((p) => (
                          <option key={p.value} value={p.value}>{p.label}</option>
                        ))}
                      </select>
                      <ChevronDown className="pointer-events-none absolute right-2.5 top-2.5 h-4 w-4 text-zinc-500" />
                    </div>
                    <Button
                      type="button"
                      size="sm"
                      variant="outline"
                      className="shrink-0 border-zinc-700 text-zinc-300 hover:bg-zinc-800"
                      disabled={testingProto || detectingProto || !camera?.onvif_host}
                      onClick={handleTestProtocol}
                    >
                      {testingProto ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : "Test"}
                    </Button>
                    <Button
                      type="button"
                      size="sm"
                      variant="outline"
                      className="shrink-0 border-zinc-700 text-zinc-400 hover:bg-zinc-800 text-xs"
                      disabled={testingProto || detectingProto || !camera?.onvif_host}
                      onClick={handleDetectProtocol}
                      title="Auto-detect camera brand and best protocol"
                    >
                      {detectingProto ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : "Auto-detect"}
                    </Button>
                  </div>
                  {/* Auto-detect result */}
                  {detectResult && (
                    <div className="rounded-md px-3 py-2 text-xs space-y-0.5 bg-zinc-800/60 border border-zinc-700/50">
                      <div className="flex items-center gap-2">
                        <span className="text-zinc-400">Brand:</span>
                        <span className="text-zinc-200 font-medium">{detectResult.brand ?? "Unknown"}{detectResult.model ? ` — ${detectResult.model}` : ""}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-zinc-400">Recommended:</span>
                        <span className="text-emerald-400 font-semibold">{detectResult.recommended}</span>
                        <span className="text-zinc-500">(auto-selected above)</span>
                      </div>
                      <div className="text-zinc-500 text-[11px] mt-1">{detectResult.detail}</div>
                    </div>
                  )}
                  {/* Manual test result */}
                  {protoResult && (
                    <div className={cn(
                      "rounded-md px-3 py-2 text-xs font-mono",
                      protoResult.ok ? "bg-emerald-900/30 text-emerald-400 border border-emerald-700/40" : "bg-red-900/30 text-red-400 border border-red-700/40",
                    )}>
                      {protoResult.ok ? "✓" : "✗"} {protoResult.detail}
                      {protoResult.ok && protoResult.latency_ms > 0 && (
                        <span className="ml-2 text-zinc-400">({protoResult.latency_ms} ms)</span>
                      )}
                    </div>
                  )}
                  <p className="text-[11px] text-zinc-500">ISAPI / CGI offer ~2 ms latency vs ONVIF's 50–200 ms. Use Auto-detect to identify your camera brand, or Test to verify a specific protocol.</p>
                </div>
              )}
            </div>
          </section>

          {/* Monitoring options */}
          {isMonitoring && (
            <section className="rounded-xl border border-violet-500/20 bg-violet-500/5 p-4 space-y-4">
              <p className="text-xs font-semibold uppercase tracking-wider text-violet-400">Monitoring Settings</p>
              <div>
                <Label className="text-xs text-muted-foreground mb-2 block">Active hours</Label>
                <div className="flex items-center gap-3">
                  <Input type="time" value={form.monitoring_hours.start}
                    onChange={(e) => set("monitoring_hours", { ...form.monitoring_hours, start: e.target.value })}
                    className="bg-zinc-900 border-zinc-700 w-32" />
                  <span className="text-zinc-500 text-sm">to</span>
                  <Input type="time" value={form.monitoring_hours.end}
                    onChange={(e) => set("monitoring_hours", { ...form.monitoring_hours, end: e.target.value })}
                    className="bg-zinc-900 border-zinc-700 w-32" />
                </div>
                <div className="flex gap-1.5 mt-2">
                  {WEEKDAYS.map((d) => {
                    const active = form.monitoring_hours.days.includes(d);
                    return (
                      <button key={d} type="button"
                        onClick={() => set("monitoring_hours", {
                          ...form.monitoring_hours,
                          days: active
                            ? form.monitoring_hours.days.filter((x) => x !== d)
                            : [...form.monitoring_hours.days, d],
                        })}
                        className={cn(
                          "rounded px-1.5 py-0.5 text-[10px] font-medium border transition-colors",
                          active ? "border-violet-400 bg-violet-400/15 text-violet-300" : "border-zinc-700 text-zinc-600",
                        )}>
                        {d.slice(0, 2)}
                      </button>
                    );
                  })}
                </div>
              </div>
              <ToggleSwitch checked={form.restricted_zone} onChange={(v) => set("restricted_zone", v)}
                label="Restricted zone" description="Trigger alert when anyone enters this zone outside hours" />
              <ToggleSwitch checked={form.alert_on_unknown} onChange={(v) => set("alert_on_unknown", v)}
                label="Alert on unknown person" description="Send Kafka notification when an unrecognised face is detected" />
            </section>
          )}
          {!isMonitoring && (
            <ToggleSwitch checked={form.alert_on_unknown} onChange={(v) => set("alert_on_unknown", v)}
              label="Alert on unknown person" description="Send Kafka notification when an unrecognised face is detected" />
          )}

          {error && (
            <p className="text-sm text-red-400 bg-red-400/10 rounded-lg px-3 py-2">{error}</p>
          )}
        </div>

        <DialogFooter className="mt-2 border-t border-border/50 pt-4">
          <Button variant="ghost" size="sm" onClick={onClose}>Cancel</Button>
          <Button size="sm" onClick={handleSubmit} disabled={mutation.isPending}>
            {mutation.isPending ? (
              <><Loader2 className="h-4 w-4 animate-spin" />Saving…</>
            ) : (
              <><Check className="h-4 w-4" />Save Changes</>
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared: live MJPEG stream <img> + AI control badge
// ─────────────────────────────────────────────────────────────────────────────

type AiMode = "AI" | "MANUAL";

function AiControlBar({ camera, canEdit, inline = false }: { camera: Camera; canEdit: boolean; inline?: boolean }) {
  const [mode,    setMode]    = useState<AiMode>("AI");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    camerasApi.getAiMode(camera.camera_id, camera)
      .then((r) => setMode((r.data as { mode: AiMode }).mode))
      .catch(() => {});
  }, [camera]);

  async function toggle() {
    if (!canEdit) return;
    const next: AiMode = mode === "AI" ? "MANUAL" : "AI";
    setLoading(true);
    try {
      await camerasApi.setAiMode(camera.camera_id, next, camera);
      setMode(next);
      toast.success(next === "MANUAL" ? "AI paused — manual control active" : "AI resumed");
    } catch {
      toast.error("Failed to update AI mode");
    } finally {
      setLoading(false);
    }
  }

  // Inline mode — just the compact toggle button, used inside LiveViewTab control strip
  if (inline) {
    if (!canEdit) return null;
    return (
      <button
        onClick={toggle}
        disabled={loading}
        className={cn(
          "flex items-center gap-1 rounded-md px-2 py-1 text-[11px] font-medium border transition-colors disabled:opacity-50",
          mode === "AI"
            ? "border-amber-500/30 bg-amber-500/10 text-amber-300 hover:bg-amber-500/20"
            : "border-emerald-500/30 bg-emerald-500/10 text-emerald-300 hover:bg-emerald-500/20",
        )}
      >
        {loading
          ? <Loader2 className="h-3 w-3 animate-spin" />
          : mode === "AI"
            ? <><BotOff className="h-3 w-3" />Pause</>
            : <><Play className="h-3 w-3" />Resume</>}
      </button>
    );
  }

  return (
    <div className={cn(
      "flex items-center justify-between rounded-lg px-3 py-2 text-xs",
      mode === "AI"
        ? "bg-emerald-500/10 border border-emerald-500/30"
        : "bg-amber-500/10 border border-amber-500/30",
    )}>
      <div className="flex items-center gap-2">
        {mode === "AI"
          ? <Bot className="h-3.5 w-3.5 text-emerald-400" />
          : <BotOff className="h-3.5 w-3.5 text-amber-400" />}
        <span className={mode === "AI" ? "text-emerald-300" : "text-amber-300"}>
          {mode === "AI" ? "AI control active — camera is scanning autonomously" : "Manual control — AI paused"}
        </span>
      </div>
      {canEdit && (
        <button
          onClick={toggle}
          disabled={loading}
          className={cn(
            "flex items-center gap-1.5 rounded-md px-2.5 py-1 text-[11px] font-medium transition-colors disabled:opacity-50",
            mode === "AI"
              ? "bg-amber-500/20 text-amber-300 hover:bg-amber-500/30"
              : "bg-emerald-500/20 text-emerald-300 hover:bg-emerald-500/30",
          )}
        >
          {loading
            ? <Loader2 className="h-3 w-3 animate-spin" />
            : mode === "AI" ? <><BotOff className="h-3 w-3" />Pause AI</> : <><Play className="h-3 w-3" />Resume AI</>}
        </button>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// LiveStream component — renders the MJPEG <img> with error fallback
// ─────────────────────────────────────────────────────────────────────────────

function LiveStream({ camera, className }: { camera: Camera; className?: string }) {
  const [error,   setError]   = useState(false);
  const [loading, setLoading] = useState(true);
  const streamUrl = camerasApi.streamUrl(camera.camera_id, camera);

  // Reset state and start 4 s timeout whenever the stream URL changes.
  // MJPEG multipart streams don't reliably fire onLoad in all browsers,
  // so the timeout acts as a fallback to dismiss the spinner.
  useEffect(() => {
    setError(false);
    setLoading(true);
    const t = setTimeout(() => setLoading(false), 4000);
    return () => clearTimeout(t);
  }, [streamUrl]);

  return (
    <div className={cn("relative bg-zinc-950", className)}>
      {loading && !error && (
        <div className="absolute inset-0 flex items-center justify-center">
          <Loader2 className="h-8 w-8 animate-spin text-zinc-600" />
        </div>
      )}
      {error ? (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2">
          <WifiOff className="h-8 w-8 text-zinc-600" />
          <p className="text-sm text-zinc-500">Stream unavailable</p>
          <Button size="sm" variant="outline" onClick={() => { setError(false); setLoading(true); }}>
            Retry
          </Button>
        </div>
      ) : (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          key={streamUrl}
          src={streamUrl}
          alt="Live stream"
          className="w-full h-full object-contain"
          onLoad={() => setLoading(false)}
          onError={() => { setError(true); setLoading(false); }}
        />
      )}
      {/* Live badge */}
      {!error && (
        <div className="absolute top-2 left-2 flex items-center gap-1.5 rounded-full bg-black/60 px-2 py-0.5 pointer-events-none">
          <span className="h-1.5 w-1.5 rounded-full bg-red-500 animate-pulse" />
          <span className="text-[10px] font-medium text-white">LIVE</span>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Live stream with PTZ position readout
// ─────────────────────────────────────────────────────────────────────────────

function LiveStreamWithZone({
  camera,
  className,
  isAiActive = false,
}: {
  camera: Camera;
  className?: string;
  isAiActive?: boolean;
}) {
  const [error,   setError]   = useState(false);
  const [loading, setLoading] = useState(true);

  // BULLET cameras have no PTZ overlay — always use plain stream.
  // PTZ cameras use annotated stream when AI is active (shows detection boxes + brain state).
  const isBullet  = camera.camera_type === "BULLET";
  const streamUrl = (isAiActive && !isBullet)
    ? camerasApi.annotatedStreamUrl(camera.camera_id, camera)
    : camerasApi.streamUrl(camera.camera_id, camera);

  // Reset state and start 4 s fallback timeout whenever the stream URL changes.
  // Covers both AI mode switches (isAiActive) and camera changes.
  useEffect(() => {
    setError(false);
    setLoading(true);
    const t = setTimeout(() => setLoading(false), 4000);
    return () => clearTimeout(t);
  }, [streamUrl]);

  return (
    <div className={cn("relative bg-zinc-950", className)}>
      {loading && !error && (
        <div className="absolute inset-0 flex items-center justify-center">
          <Loader2 className="h-8 w-8 animate-spin text-zinc-600" />
        </div>
      )}
      {error ? (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2">
          <WifiOff className="h-8 w-8 text-zinc-600" />
          <p className="text-sm text-zinc-500">Stream unavailable</p>
          <Button size="sm" variant="outline" onClick={() => { setError(false); setLoading(true); }}>
            Retry
          </Button>
        </div>
      ) : (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          key={streamUrl}
          src={streamUrl}
          alt="Live stream"
          className="w-full h-full object-contain"
          onLoad={() => setLoading(false)}
          onError={() => { setError(true); setLoading(false); }}
        />
      )}

      {/* Live badge — amber when AI annotated stream is active */}
      {!error && (
        <div className={cn(
          "absolute top-2 left-2 flex items-center gap-1.5 rounded-full px-2 py-0.5 pointer-events-none",
          isAiActive ? "bg-amber-500/80" : "bg-black/60",
        )}>
          <span className={cn("h-1.5 w-1.5 rounded-full animate-pulse", isAiActive ? "bg-white" : "bg-red-500")} />
          <span className="text-[10px] font-medium text-white">{isAiActive ? "AI" : "LIVE"}</span>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Monitoring Status Panel — real-time poll of camera brain state
// ─────────────────────────────────────────────────────────────────────────────

interface MonitoringState {
  monitoring:       boolean;
  camera_id:        string;
  session_id?:      string;
  brain_state?:     string;
  cycle_count?:     number;
  total_cells?:     number;
  path_index?:      number;
  fps_actual?:      number;
  cycle_time_s?:    number;
  recognition_rate?: number;
  present_count?:   number;
  absent_count?:    number;
  unknown_count?:   number;
  current_ptz?:     { pan: number; tilt: number; zoom: number } | null;
  scan_cells?:      Array<{ cell_id: string; center_pan: number; center_tilt: number; zoom: number; faces: number }>;
}

const BRAIN_STATE_META: Record<string, { label: string; labelFixed: string; color: string }> = {
  PRESET_TRANSIT:   { label: "Moving",      labelFixed: "Scanning",    color: "text-amber-400" },
  PRESET_RECOGNIZE: { label: "Recognizing", labelFixed: "Recognizing", color: "text-emerald-400" },
  PRESET_COMPLETE:  { label: "Preset Done", labelFixed: "Scan Done",   color: "text-blue-400" },
  FACE_HUNT:        { label: "Face Hunt",   labelFixed: "Face Hunt",   color: "text-cyan-400" },
  CYCLE_COMPLETE:   { label: "Cycle Done",  labelFixed: "Cycle Done",  color: "text-green-400" },
  IDLE:             { label: "Idle",        labelFixed: "Idle",        color: "text-zinc-400" },
  STOPPED:          { label: "Stopped",     labelFixed: "Stopped",     color: "text-zinc-500" },
  ERROR:            { label: "Error",       labelFixed: "Error",       color: "text-red-400" },
};

function MonitoringStatusPanel({ camera }: { camera: Camera }) {
  const [state, setState] = useState<MonitoringState | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const r = await nodeApi.status(camera.camera_id, camera);
        if (!cancelled) { setState(r.data as MonitoringState); setError(false); }
      } catch {
        if (!cancelled) setError(true);
      }
    };
    poll();
    const id = setInterval(poll, 3000);
    return () => { cancelled = true; clearInterval(id); };
  }, [camera.camera_id, camera]);

  if (error || !state || !state.monitoring) return null;

  const isFixed = (camera as any).camera_type === "BULLET";
  const bsMeta  = BRAIN_STATE_META[state.brain_state ?? ""];
  const bs      = bsMeta
    ? { label: isFixed ? bsMeta.labelFixed : bsMeta.label, color: bsMeta.color }
    : { label: state.brain_state ?? "Unknown", color: "text-zinc-400" };
  const ptz   = state.current_ptz;
  const rate  = state.recognition_rate != null ? `${(state.recognition_rate * 100).toFixed(0)}%` : "—";

  return (
    <div className="rounded-xl border border-emerald-500/20 bg-emerald-500/5 overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-emerald-500/10">
        <div className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
          <span className="text-xs font-semibold text-emerald-400">AI Monitoring Active</span>
        </div>
        <span className={cn("text-xs font-medium", bs.color)}>{bs.label}</span>
      </div>

      {/* Three metric groups */}
      <div className="grid grid-cols-3 divide-x divide-border/30">
        {/* Camera Movement */}
        <div className="px-3 py-2.5 space-y-1.5">
          <div className="flex items-center gap-1.5 text-[10px] font-semibold text-zinc-400 uppercase tracking-wide">
            <Move3d className="h-3 w-3" />{isFixed ? "Camera" : "Camera Movement"}
          </div>
          <div className="space-y-1 text-xs">
            <div className="flex justify-between">
              <span className="text-zinc-500">Cycle</span>
              <span className="text-zinc-200 font-medium">{state.cycle_count ?? 0}</span>
            </div>
            {isFixed ? (
              <div className="flex justify-between">
                <span className="text-zinc-500">Mode</span>
                <span className="text-zinc-400 font-medium">Fixed Position</span>
              </div>
            ) : (
              <div className="flex justify-between">
                <span className="text-zinc-500">Presets</span>
                <span className="text-zinc-200 font-medium">{state.path_index ?? 0}/{state.total_cells ?? 0}</span>
              </div>
            )}
            {!isFixed && ptz && (
              <div className="flex justify-between">
                <span className="text-zinc-500">PTZ</span>
                <span className="text-zinc-300 font-mono text-[10px]">
                  {ptz.pan.toFixed(2)} / {ptz.tilt.toFixed(2)}
                </span>
              </div>
            )}
            <div className="flex justify-between">
              <span className="text-zinc-500">Speed</span>
              <span className="text-zinc-200 font-medium">
                {state.cycle_time_s ? `${state.cycle_time_s}s/cycle` : "—"}
              </span>
            </div>
          </div>
        </div>

        {/* Face Dataset Access */}
        <div className="px-3 py-2.5 space-y-1.5">
          <div className="flex items-center gap-1.5 text-[10px] font-semibold text-zinc-400 uppercase tracking-wide">
            <Users className="h-3 w-3" />Face Dataset
          </div>
          <div className="space-y-1 text-xs">
            <div className="flex justify-between">
              <span className="text-zinc-500">Recognized</span>
              <span className="text-emerald-400 font-medium">{state.present_count ?? 0}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-zinc-500">Unknown</span>
              <span className="text-amber-400 font-medium">{state.unknown_count ?? 0}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-zinc-500">Match Rate</span>
              <span className="text-zinc-200 font-medium">{rate}</span>
            </div>
          </div>
        </div>

        {/* AI Processing */}
        <div className="px-3 py-2.5 space-y-1.5">
          <div className="flex items-center gap-1.5 text-[10px] font-semibold text-zinc-400 uppercase tracking-wide">
            <Brain className="h-3 w-3" />AI Processing
          </div>
          <div className="space-y-1 text-xs">
            <div className="flex justify-between">
              <span className="text-zinc-500">FPS</span>
              <span className={cn("font-medium",
                (state.fps_actual ?? 0) >= 5 ? "text-emerald-400" :
                (state.fps_actual ?? 0) >= 2 ? "text-amber-400"   : "text-red-400"
              )}>
                {state.fps_actual ?? 0}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-zinc-500">State</span>
              <span className={cn("font-medium", bs.color)}>{bs.label}</span>
            </div>
            {!isFixed && (
              <div className="flex justify-between">
                <span className="text-zinc-500">Presets</span>
                <span className="text-zinc-200 font-medium">{state.total_cells ?? 0}</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Tab 1 — Live View
// ─────────────────────────────────────────────────────────────────────────────

function LiveViewTab({
  camera, canEdit, monitoring, onStart, onStop,
}: {
  camera:     Camera;
  canEdit:    boolean;
  monitoring: boolean;
  onStart:    () => void;
  onStop:     () => void;
}) {
  return (
    <div className="space-y-3">
      {/* Metrics panel — only visible when AI session is running */}
      <MonitoringStatusPanel camera={camera} />

      {/* Live stream — annotated when AI is active */}
      <LiveStreamWithZone
        camera={camera}
        className="rounded-xl overflow-hidden border border-border/40 aspect-video"
        isAiActive={monitoring}
      />

      {/* Control strip — shown always, style changes with AI state */}
      {monitoring ? (
        <div className="flex items-center justify-between rounded-lg px-3 py-2 border bg-emerald-500/5 border-emerald-500/20">
          <div className="flex items-center gap-2 text-xs">
            <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
            <span className="text-emerald-400 font-medium">AI Session Active</span>
          </div>
          {canEdit && (
            <div className="flex items-center gap-2">
              <AiControlBar camera={camera} canEdit={canEdit} inline />
              <Button
                size="sm" variant="outline"
                className="h-7 text-xs text-red-400 border-red-500/30 hover:bg-red-500/10"
                onClick={onStop}
              >
                <Square className="h-3 w-3 mr-1" />Stop
              </Button>
            </div>
          )}
        </div>
      ) : canEdit ? (
        <Button size="sm" className="w-full" onClick={onStart}>
          <Play className="h-3.5 w-3.5 mr-1.5" />Start AI Session
        </Button>
      ) : null}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// FOV Auto-Calibration Panel — PTZ & Presets → FOV sub-tab
// ─────────────────────────────────────────────────────────────────────────────
//
// One-click optical-flow calibration of pan/tilt/zoom geometry.  Invokes
// POST /api/cameras/{id}/calibrate-fov on the camera's GPU node, which
// measures pixel displacement for known ONVIF moves at wide (zoom=0) and
// narrow (zoom=1) and persists the resulting K constants to learned_params.
// Only rendered for full PTZ cameras (not PTZ_ZOOM / BULLET*).

interface FovCalibrationResult {
  K_pan_wide?:    number;
  K_pan_narrow?:  number;
  K_tilt_wide?:   number;
  K_tilt_narrow?: number;
  fov_h_wide?:    number;
  fov_h_narrow?:  number;
  fov_v_wide?:    number;
  fov_v_narrow?:  number;
}

function FovCalibrationPanel({
  camera,
  canEdit,
  isMoving,
  onCalibrated,
}: {
  camera: Camera;
  canEdit: boolean;
  isMoving: boolean;
  onCalibrated: () => void;
}) {
  const [running, setRunning]   = useState(false);
  const [progress, setProgress] = useState<string>("");
  // Read current stored calibration straight from camera.learned_params so we
  // always show the persisted values (not cached local state).
  const lp = (camera.learned_params ?? {}) as Record<string, unknown>;
  const stored: FovCalibrationResult = {
    K_pan_wide:    lp.K_pan_wide    as number | undefined,
    K_pan_narrow:  lp.K_pan_narrow  as number | undefined,
    K_tilt_wide:   lp.K_tilt_wide   as number | undefined,
    K_tilt_narrow: lp.K_tilt_narrow as number | undefined,
    fov_h_wide:    lp.fov_h_wide    as number | undefined,
    fov_h_narrow:  lp.fov_h_narrow  as number | undefined,
    fov_v_wide:    lp.fov_v_wide    as number | undefined,
    fov_v_narrow:  lp.fov_v_narrow  as number | undefined,
  };
  const hasCalibration =
    stored.K_pan_wide  != null ||
    stored.K_tilt_wide != null ||
    stored.fov_h_wide  != null;

  async function handleRunCalibration() {
    if (!canEdit || running) return;
    // Auto-calibration drives the camera through a 30 s sequence of moves.
    // Warn before blocking any active AI session / manual motion.
    if (!window.confirm(
      "Run FOV auto-calibration?\n\n" +
      "The camera will move through pan/tilt/zoom positions for ~30 seconds. " +
      "Any active AI session will be interrupted. Make sure a textured scene " +
      "is visible (not a blank wall)."
    )) return;

    setRunning(true);
    setProgress("Starting calibration…");
    // UX progress hints — the backend endpoint is synchronous so we can't
    // stream real progress; these approximate the phases based on the server
    // routine (wide pan → wide tilt → narrow pan → narrow tilt).
    const phases: [number, string][] = [
      [2_000,  "Measuring wide pan…"],
      [8_000,  "Measuring wide tilt…"],
      [14_000, "Measuring narrow pan…"],
      [22_000, "Measuring narrow tilt…"],
      [28_000, "Restoring camera position…"],
    ];
    const timers = phases.map(([t, msg]) => setTimeout(() => setProgress(msg), t));

    try {
      const res = await camerasApi.calibrateFov(camera.camera_id, camera);
      const calib: FovCalibrationResult = (res.data as { calibration?: FovCalibrationResult })?.calibration ?? {};
      timers.forEach(clearTimeout);
      setProgress("");
      toast.success(
        `Calibration saved — K_pan_wide=${(calib.K_pan_wide ?? 0).toFixed(4)}, ` +
        `K_tilt_wide=${(calib.K_tilt_wide ?? 0).toFixed(4)}`
      );
      onCalibrated();
    } catch (err: unknown) {
      timers.forEach(clearTimeout);
      setProgress("");
      const msg = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail
        ?? "Calibration failed. Check camera connectivity and scene visibility.";
      toast.error(msg);
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="flex flex-col gap-3 text-xs">
      <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 px-3 py-2.5 space-y-1">
        <p className="text-amber-400 font-semibold text-[11px]">FOV Auto-Calibration</p>
        <p className="text-zinc-400 text-[10px] leading-4">
          Measures the camera&apos;s true pan/tilt/zoom geometry using optical
          flow. Runs once per camera; required for accurate face tracking.
          Takes ~30 s.
        </p>
      </div>

      {/* Pre-check warnings */}
      <ul className="text-[10px] text-zinc-500 space-y-1 list-disc list-inside">
        <li>Camera must be online and have a textured scene in view (no blank walls).</li>
        <li>Active AI scanning will be interrupted for ~30 s.</li>
        <li>Re-run after any physical remount or firmware update.</li>
      </ul>

      {/* Run button */}
      <Button
        size="sm"
        onClick={handleRunCalibration}
        disabled={running || isMoving || !canEdit}
        className={cn(
          "w-full text-xs text-white",
          hasCalibration
            ? "bg-zinc-700 hover:bg-zinc-600"
            : "bg-emerald-600 hover:bg-emerald-500",
        )}
      >
        {running
          ? <><Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" />{progress || "Calibrating…"}</>
          : hasCalibration
            ? <><RefreshCw className="h-3.5 w-3.5 mr-1" />Re-run calibration</>
            : <><Play className="h-3.5 w-3.5 mr-1" />Run FOV calibration</>}
      </Button>

      {/* Read-only gate */}
      {!canEdit && (
        <p className="text-[10px] text-amber-400 bg-amber-400/10 rounded px-2 py-1">
          Read-only for Viewer role.
        </p>
      )}

      {/* Stored calibration values */}
      <div className="space-y-1.5 border-t border-zinc-800 pt-2">
        <p className="text-[10px] font-medium text-zinc-300">
          Stored calibration {hasCalibration ? "" : "— not yet calibrated"}
        </p>
        {hasCalibration ? (
          <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-[10px] font-mono">
            <span className="text-zinc-500">K_pan_wide</span>
            <span className="text-zinc-300 text-right">
              {stored.K_pan_wide?.toFixed(5) ?? "—"}
            </span>
            <span className="text-zinc-500">K_pan_narrow</span>
            <span className="text-zinc-300 text-right">
              {stored.K_pan_narrow?.toFixed(5) ?? "—"}
            </span>
            <span className="text-zinc-500">K_tilt_wide</span>
            <span className="text-zinc-300 text-right">
              {stored.K_tilt_wide?.toFixed(5) ?? "—"}
            </span>
            <span className="text-zinc-500">K_tilt_narrow</span>
            <span className="text-zinc-300 text-right">
              {stored.K_tilt_narrow?.toFixed(5) ?? "—"}
            </span>
            <span className="text-zinc-500">FOV wide (H/V)</span>
            <span className="text-zinc-300 text-right">
              {stored.fov_h_wide != null ? `${stored.fov_h_wide.toFixed(1)}°` : "—"}
              {" / "}
              {stored.fov_v_wide != null ? `${stored.fov_v_wide.toFixed(1)}°` : "—"}
            </span>
            <span className="text-zinc-500">FOV narrow (H/V)</span>
            <span className="text-zinc-300 text-right">
              {stored.fov_h_narrow != null ? `${stored.fov_h_narrow.toFixed(1)}°` : "—"}
              {" / "}
              {stored.fov_v_narrow != null ? `${stored.fov_v_narrow.toFixed(1)}°` : "—"}
            </span>
          </div>
        ) : (
          <p className="text-[10px] text-zinc-500 leading-4">
            Face tracking will use conservative geometry estimates until this
            camera is calibrated, which reduces tracking accuracy.
          </p>
        )}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Tab 2 — PTZ & Presets  (live stream + PTZ controls + calibration + presets)
// ─────────────────────────────────────────────────────────────────────────────

// ─── ZonePtzTab ────────────────────────────────────────────────────────────────

function ZonePtzTab({ camera, canEdit }: { camera: Camera; canEdit: boolean }) {
  const qc        = useQueryClient();
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // PTZ
  const [pan,  setPan]  = useState(0);
  const [tilt, setTilt] = useState(0);
  const [zoom, setZoom] = useState(0);
  const [moving, setMoving] = useState(false);
  // Actual pan/tilt range from camera ONVIF (default -1..+1 = range 2)
  const [ptzLimits, setPtzLimits] = useState({ panRange: _PAN_RANGE, tiltRange: _TILT_RANGE });
  // Smooth PTZ animation — target written by poll, display smoothed by rAF
  const ptzTarget  = useRef({ pan: 0, tilt: 0, zoom: 0 });
  const ptzSmooth  = useRef({ pan: 0, tilt: 0, zoom: 0 });
  // Always holds the latest redraw fn so the rAF loop can call it without deps
  const redrawRef  = useRef<(dp?: number, dt?: number, dz?: number) => void>(() => {});
  // Ref kept in sync with `moving` state — readable inside the poll closure
  // without stale-closure issues.  Poll skips ONVIF calls while a move is
  // in flight so we never saturate the camera's ONVIF connection limit.
  const movingRef  = useRef(false);
  const [subTab, setSubTab] = useState<"ptz" | "presets" | "calibrate">("ptz");
  // Track AI mode so we can auto-pause before manual moves
  const [ptzAiMode, setPtzAiMode] = useState<AiMode>("AI");

  // ── Manual calibration state ─────────────────────────────────────────────
  // Narrow-zoom FOV + per-axis scale corrections refined by click-anchor drift
  // measurements on the Cal sub-tab. Seeded from the backend's learned_params
  // on first PTZ poll and persisted via saveRoi.
  const [fovNarrow,  setFovNarrow]  = useState<number>(_DEF_FOV_H_NARROW);
  const [fovVNarrow, setFovVNarrow] = useState<number>(_DEF_FOV_V_NARROW);
  const [panScale,   setPanScale]   = useState<number>(1.0);
  const [tiltScale,  setTiltScale]  = useState<number>(1.0);
  // Click-anchor: pixel+world coords of the landmark placed on the canvas,
  // and the PTZ position at which it was placed. Drift from this anchor
  // after a PTZ move drives the pan/tilt/FOV auto-correction math.
  const [calAnchorPixel, setCalAnchorPixel] = useState<ZonePoint | null>(null);
  const [calAnchorWorld, setCalAnchorWorld] = useState<WorldPoint | null>(null);
  const [calAnchorPtz,   setCalAnchorPtz]   = useState<PtzPos | null>(null);
  const [saving,      setSaving]      = useState(false);
  const [calZoomTest, setCalZoomTest] = useState(false);

  useEffect(() => {
    camerasApi.getAiMode(camera.camera_id, camera)
      .then((r) => setPtzAiMode((r.data as { mode: AiMode }).mode))
      .catch(() => {});
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [camera.camera_id]);

  // Recursive PTZ poll — waits for each response before scheduling the next,
  // preventing request pile-up on slow cameras (~400–600 ms ONVIF round-trip).
  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      if (cancelled) return;
      // Skip ONVIF status call while a move is in flight — camera has a limited
      // number of simultaneous ONVIF connections; saturating them causes move failures.
      if (movingRef.current) {
        setTimeout(poll, 300);
        return;
      }
      try {
        const res = await camerasApi.ptzStatus(camera.camera_id, camera);
        if (cancelled) return;
        const d = res.data as {
          pan?: number; tilt?: number; zoom?: number;
          pan_min?: number; pan_max?: number; tilt_min?: number; tilt_max?: number;
          fov_h_narrow?: number; fov_v_narrow?: number; pan_scale?: number; tilt_scale?: number;
        };
        if (d.pan  != null) { setPan(d.pan);   ptzTarget.current.pan  = d.pan;  }
        if (d.tilt != null) { setTilt(d.tilt); ptzTarget.current.tilt = d.tilt; }
        if (d.zoom != null) { setZoom(d.zoom); ptzTarget.current.zoom = d.zoom; }
        // Update actual PTZ limits
        if (d.pan_min != null && d.pan_max != null) {
          const pr = d.pan_max  - d.pan_min;
          const tr = (d.tilt_max ?? 1) - (d.tilt_min ?? -1);
          if (pr > 0 && tr > 0) setPtzLimits({ panRange: pr, tiltRange: tr });
        }
        // Seed fovNarrow / panScale from the backend's calibrated values (only if at default)
        if (d.fov_h_narrow != null) {
          setFovNarrow((prev) => prev === _DEF_FOV_H_NARROW ? d.fov_h_narrow! : prev);
        }
        if (d.pan_scale != null) {
          setPanScale((prev) => prev === 1.0 ? d.pan_scale! : prev);
        }
        if (d.tilt_scale != null) {
          setTiltScale((prev) => prev === 1.0 ? d.tilt_scale! : prev);
        }
      } catch {}
      // Schedule next poll 200 ms after this one completes (camera ONVIF ~100-300ms round-trip)
      if (!cancelled) setTimeout(poll, 200);
    };
    poll();
    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [camera.camera_id]);

  // ── Canvas rendering ───────────────────────────────────────────────────────
  // dp/dt/dz optional: when called from rAF use smooth values; from effects use state defaults
  const redraw = useCallback((dp = pan, dt = tilt, dz = zoom) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const W = canvas.width, H = canvas.height;

    const curPtzSnap: PtzPos = { pan: dp, tilt: dt, zoom: dz };
    const camFov: FovOverride = { fovH: camera.fov_h, fovV: camera.fov_v, fovHNarrow: fovNarrow, fovVNarrow: fovVNarrow, panScale, tiltScale, ...ptzLimits };

    // ── Calibration anchor crosshair + drift indicator ───────────────────────
    if (calAnchorWorld) {
      const projected = worldPolyToPixel([calAnchorWorld], curPtzSnap, camFov);
      const ap = projected[0];
      if (ap.x >= 0 && ap.x <= 1 && ap.y >= 0 && ap.y <= 1) {
        const ax = ap.x * W, ay = ap.y * H;
        const sz = 14;
        ctx.save();
        ctx.strokeStyle = "rgba(255,80,80,1)";
        ctx.lineWidth   = 2;
        ctx.setLineDash([]);
        // crosshair lines
        ctx.beginPath(); ctx.moveTo(ax - sz, ay); ctx.lineTo(ax + sz, ay); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(ax, ay - sz); ctx.lineTo(ax, ay + sz); ctx.stroke();
        // outer ring
        ctx.beginPath(); ctx.arc(ax, ay, 9, 0, Math.PI * 2); ctx.stroke();
        ctx.fillStyle = "rgba(255,80,80,1)";
        ctx.font = "bold 10px monospace";
        ctx.fillText("⊕ CAL", ax + 13, ay - 8);

        // ── Drift indicator — show original click position & distance ──────
        if (calAnchorPixel && calAnchorPtz &&
            (Math.abs(dp - calAnchorPtz.pan) > 0.005 ||
             Math.abs(dt - calAnchorPtz.tilt) > 0.005 ||
             Math.abs(dz - calAnchorPtz.zoom) > 0.005)) {
          const ox = calAnchorPixel.x * W, oy = calAnchorPixel.y * H;
          const driftPx = Math.sqrt((ax - ox) ** 2 + (ay - oy) ** 2);
          // Original position: small white cross
          ctx.strokeStyle = "rgba(255,255,255,0.7)";
          ctx.lineWidth = 1.5;
          const hs = 6;
          ctx.beginPath(); ctx.moveTo(ox - hs, oy); ctx.lineTo(ox + hs, oy); ctx.stroke();
          ctx.beginPath(); ctx.moveTo(ox, oy - hs); ctx.lineTo(ox, oy + hs); ctx.stroke();
          // Line from original to projected
          ctx.strokeStyle = driftPx < 5 ? "rgba(80,255,120,0.8)" : "rgba(255,200,60,0.8)";
          ctx.lineWidth = 1.5;
          ctx.setLineDash([4, 3]);
          ctx.beginPath(); ctx.moveTo(ox, oy); ctx.lineTo(ax, ay); ctx.stroke();
          ctx.setLineDash([]);
          // Drift distance label
          ctx.fillStyle = driftPx < 5 ? "rgba(80,255,120,1)" : "rgba(255,200,60,1)";
          ctx.font = "bold 11px monospace";
          const midX = (ox + ax) / 2, midY = (oy + ay) / 2;
          ctx.fillText(`${Math.round(driftPx)}px`, midX + 4, midY - 4);
        }
        ctx.restore();
      } else {
        // Anchor off-screen — draw arrow at edge
        const ex = Math.max(0.02, Math.min(0.98, ap.x));
        const ey = Math.max(0.02, Math.min(0.98, ap.y));
        ctx.save();
        ctx.fillStyle = "rgba(255,80,80,0.9)";
        ctx.font = "10px monospace";
        ctx.fillText("⊕ (off-screen)", ex * W, ey * H);
        ctx.restore();
      }
    }
  }, [pan, tilt, zoom, ptzLimits, fovNarrow, fovVNarrow, panScale, tiltScale, calAnchorWorld, calAnchorPixel, calAnchorPtz]);

  // Keep redrawRef current so the rAF loop always calls the latest version
  useEffect(() => { redrawRef.current = redraw; }, [redraw]);

  // rAF loop — smoothly interpolates the canvas display position toward the polled target
  useEffect(() => {
    let raf: number;
    const tick = () => {
      const t = ptzTarget.current, s = ptzSmooth.current;
      const k = 0.18; // ~8 rad/s decay — smooth over ~5 frames
      s.pan  += (t.pan  - s.pan)  * k;
      s.tilt += (t.tilt - s.tilt) * k;
      s.zoom += (t.zoom - s.zoom) * k;
      redrawRef.current(s.pan, s.tilt, s.zoom);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // intentionally runs once

  // Sync canvas size + redraw on resize
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const obs = new ResizeObserver(() => {
      canvas.width  = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
      redraw();
    });
    obs.observe(canvas);
    return () => obs.disconnect();
  }, [redraw]);
  useEffect(() => { redraw(); }, [redraw]);

  // ── Calibration canvas click handler ──────────────────────────────────────
  function getCanvasCoords(e: React.MouseEvent<HTMLCanvasElement>): ZonePoint {
    const r = canvasRef.current!.getBoundingClientRect();
    return { x: (e.clientX - r.left) / r.width, y: (e.clientY - r.top) / r.height };
  }

  function onCanvasClick(e: React.MouseEvent<HTMLCanvasElement>) {
    if (subTab !== "calibrate") return;
    if (e.detail > 1) return;
    e.preventDefault();
    const pt = getCanvasCoords(e);
    const camFovNow: FovOverride = {
      fovH: camera.fov_h, fovV: camera.fov_v,
      fovHNarrow: fovNarrow, fovVNarrow: fovVNarrow,
      panScale, tiltScale, ...ptzLimits,
    };
    const curPos = { pan, tilt, zoom };

    // If anchor already placed AND camera has moved — this click is a correction.
    if (calAnchorPixel && calAnchorWorld && calAnchorPtz) {
      const dPan  = Math.abs(curPos.pan  - calAnchorPtz.pan);
      const dTilt = Math.abs(curPos.tilt - calAnchorPtz.tilt);
      const dZoom = Math.abs(curPos.zoom - calAnchorPtz.zoom);
      const moved = dPan > 0.008 || dTilt > 0.008 || dZoom > 0.04;
      if (moved) {
        if (dZoom > dPan && dZoom > dTilt) {
          const v = autoComputeFovNarrow(calAnchorPixel, pt, calAnchorPtz, curPos, camFovNow);
          if (v != null) {
            setFovNarrow(+v.toFixed(3));
            toast.success(`Narrow FOV auto-corrected → ${v.toFixed(2)}°`);
          } else { toast.error("Zoom correction failed — place anchor further from centre"); }
        } else if (dPan >= dTilt) {
          const v = autoComputePanScale(calAnchorPixel, pt, calAnchorPtz, curPos, camFovNow);
          if (v != null) {
            setPanScale(+v.toFixed(4));
            toast.success(`Pan scale auto-corrected → ×${v.toFixed(3)}`);
          } else { toast.error("Pan correction failed — pan further from anchor"); }
        } else {
          const v = autoComputeTiltScale(calAnchorPixel, pt, calAnchorPtz, curPos, camFovNow);
          if (v != null) {
            setTiltScale(+v.toFixed(4));
            toast.success(`Tilt scale auto-corrected → ×${v.toFixed(3)}`);
          } else { toast.error("Tilt correction failed — tilt further from anchor"); }
        }
        const newFov: FovOverride = { ...camFovNow, panScale, tiltScale };
        const newWorld = pixelPolyToWorld([pt], curPos, newFov)[0];
        setCalAnchorPixel(pt);
        setCalAnchorWorld(newWorld);
        setCalAnchorPtz(curPos);
        return;
      }
    }

    // No anchor yet (or camera hasn't moved) — place / replace anchor
    const world = pixelPolyToWorld([pt], curPos, camFovNow)[0];
    setCalAnchorPixel(pt);
    setCalAnchorWorld(world);
    setCalAnchorPtz(curPos);
  }

  // ── Save calibration ──────────────────────────────────────────────────────
  async function handleSaveCalibration() {
    setSaving(true);
    try {
      await camerasApi.saveRoi(camera.camera_id, {
        fov_h_narrow: fovNarrow,
        fov_v_narrow: fovVNarrow,
        pan_scale:    panScale,
        tilt_scale:   tiltScale,
      }, camera);
      qc.invalidateQueries({ queryKey: ["cameras"] });
      toast.success("Calibration saved");
    } catch {
      toast.error("Failed to save calibration");
    } finally {
      setSaving(false);
    }
  }

  // ── PTZ helpers ─────────────────────────────────────────────────────────────

  /** Ensure AI is paused before manual PTZ use — auto-pauses silently if still in AI mode. */
  async function ensureManualMode() {
    if (ptzAiMode === "AI") {
      try {
        await camerasApi.setAiMode(camera.camera_id, "MANUAL", camera);
        setPtzAiMode("MANUAL");
        toast.info("AI paused — manual PTZ control active");
      } catch {
        // Non-fatal; proceed with move anyway
      }
    }
  }

  async function move(type: "absolute" | "relative", data: Partial<{pan:number;tilt:number;zoom:number}>, speed = 0.5) {
    if (!canEdit) return;
    await ensureManualMode();
    movingRef.current = true; setMoving(true);
    try {
      await camerasApi.ptzMove(camera.camera_id, { type, speed, free: true, ...data }, camera);
    }
    catch { toast.error("PTZ move failed"); }
    finally { movingRef.current = false; setMoving(false); }
  }

  async function gotoHome() {
    if (!canEdit) return;
    await ensureManualMode();
    movingRef.current = true; setMoving(true);
    try { await camerasApi.ptzMove(camera.camera_id, { type: "home" }, camera); setPan(0); setTilt(0); setZoom(0); }
    catch { toast.error("Failed to go home"); }
    finally { movingRef.current = false; setMoving(false); }
  }

  // ── Calibration zoom sweep ────────────────────────────────────────────────
  async function handleZoomTest() {
    if (calZoomTest) return;
    movingRef.current = true; setCalZoomTest(true);
    try {
      const steps = [0.0, 0.3, 0.6, 1.0, 0.6, 0.3, 0.0];
      for (const z of steps) {
        await camerasApi.ptzMove(camera.camera_id, { type: "absolute", zoom: z, pan, tilt, speed: 0.4, free: true }, camera);
        await new Promise((r) => setTimeout(r, 2000));
      }
    } catch {
      toast.error("Zoom test failed");
    } finally {
      movingRef.current = false; setCalZoomTest(false);
    }
  }

  // Sensitivity: how much PTZ movement per D-pad press (ONVIF normalised units)
  const [sensitivity, setSensitivity] = useState(0.05);
  const dpad: Array<{ icon: React.ElementType; dp: {pan:number;tilt:number}; pos: string; label: string }> = [
    { icon: ArrowUpLeft,    dp: { pan: -sensitivity, tilt:  sensitivity }, pos: "col-start-1 row-start-1", label: "NW" },
    { icon: ArrowUp,        dp: { pan:            0, tilt:  sensitivity }, pos: "col-start-2 row-start-1", label: "N"  },
    { icon: ArrowUpRight,   dp: { pan:  sensitivity, tilt:  sensitivity }, pos: "col-start-3 row-start-1", label: "NE" },
    { icon: ArrowLeft,      dp: { pan: -sensitivity, tilt:            0 }, pos: "col-start-1 row-start-2", label: "W"  },
    { icon: Home,           dp: { pan:            0, tilt:            0 }, pos: "col-start-2 row-start-2", label: "HOME"},
    { icon: ArrowRight,     dp: { pan:  sensitivity, tilt:            0 }, pos: "col-start-3 row-start-2", label: "E"  },
    { icon: ArrowDownLeft,  dp: { pan: -sensitivity, tilt: -sensitivity }, pos: "col-start-1 row-start-3", label: "SW" },
    { icon: ArrowDown,      dp: { pan:            0, tilt: -sensitivity }, pos: "col-start-2 row-start-3", label: "S"  },
    { icon: ArrowDownRight, dp: { pan:  sensitivity, tilt: -sensitivity }, pos: "col-start-3 row-start-3", label: "SE" },
  ];

  return (
    <div className="flex flex-col gap-3">
      <AiControlBar camera={camera} canEdit={canEdit} />

      <div className="flex gap-3" style={{ minHeight: 340 }}>
        {/* Left: live stream with calibration canvas overlay */}
        <div className="flex-1 flex flex-col gap-2">
          <div className="relative rounded-xl overflow-hidden border border-border/40 bg-zinc-950 aspect-video">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              key={camerasApi.streamUrl(camera.camera_id, camera)}
              src={camerasApi.streamUrl(camera.camera_id, camera)}
              alt="Live stream"
              className="absolute inset-0 w-full h-full object-contain"
              onError={(e) => {
                // Retry stream after 3s on failure
                const img = e.currentTarget;
                setTimeout(() => { img.src = camerasApi.streamUrl(camera.camera_id, camera) + "&t=" + Date.now(); }, 10000);
              }}
            />
            {/* Calibration overlay canvas — only active in Cal sub-tab */}
            <canvas
              ref={canvasRef}
              className={cn(
                "absolute inset-0 w-full h-full",
                subTab === "calibrate" ? "cursor-crosshair" : "cursor-default",
              )}
              onClick={onCanvasClick}
            />
            {/* LIVE badge */}
            <div className="absolute top-2 left-2 flex items-center gap-1.5 rounded-full bg-black/60 px-2 py-0.5 pointer-events-none">
              <span className="h-1.5 w-1.5 rounded-full bg-red-500 animate-pulse" />
              <span className="text-[10px] font-medium text-white">LIVE</span>
            </div>
          </div>
        </div>

        {/* Right: control panel */}
        <div className="w-56 flex flex-col gap-3">
          {(() => {
            const camType = (camera as any).camera_type ?? "PTZ";
            const camTypeHasPresets = CAMERA_TYPES.find((t) => t.value === camType)?.hasPresets !== false;
            // FOV auto-calibration measures pan + tilt + zoom via optical flow,
            // so it only makes sense for full PTZ cameras (not PTZ_ZOOM / BULLET).
            const camSupportsFovCal = camType === "PTZ";
            const availableSubTabs: readonly ("ptz" | "presets" | "calibrate")[] = camTypeHasPresets
              ? (camSupportsFovCal ? ["ptz", "presets", "calibrate"] : ["ptz", "presets"])
              : (camSupportsFovCal ? ["ptz", "calibrate"] : ["ptz"]);
            // If the current sub-tab isn't available for this camera type, snap back to "ptz".
            if (!availableSubTabs.includes(subTab) && subTab !== "ptz") {
              setSubTab("ptz");
            }
            return (
              <>
                <div className="flex rounded-lg border border-zinc-700 overflow-hidden text-xs">
                  {availableSubTabs.map((t) => (
                    <button
                      key={t}
                      onClick={() => setSubTab(t as typeof subTab)}
                      className={cn(
                        "flex-1 py-1.5 font-medium transition-colors capitalize",
                        subTab === t ? "bg-zinc-700 text-white" : "text-zinc-500 hover:text-zinc-300",
                      )}
                    >{t === "calibrate" ? "FOV" : t}</button>
                  ))}
                </div>

                {subTab === "presets" && camTypeHasPresets && (
                  <div className="overflow-y-auto" style={{ maxHeight: 420 }}>
                    <PresetsTab camera={camera} canEdit={canEdit} />
                  </div>
                )}
              </>
            );
          })()}

          {subTab === "ptz" && (
            <div className="flex flex-col gap-3">
              {!canEdit && (
                <p className="text-[10px] text-amber-400 bg-amber-400/10 rounded px-2 py-1">
                  Read-only for Viewer role.
                </p>
              )}
              {canEdit && ptzAiMode === "AI" && (
                <p className="text-[10px] text-blue-300 bg-blue-500/10 border border-blue-500/20 rounded px-2 py-1.5 leading-4">
                  AI scanning is active. Moving the camera will automatically pause AI.
                </p>
              )}
              {canEdit && ptzAiMode === "MANUAL" && (
                <div className="flex items-center justify-between gap-2 text-[10px] text-amber-300 bg-amber-500/10 border border-amber-500/20 rounded px-2 py-1.5">
                  <span>Manual mode — AI paused.</span>
                  <button
                    className="underline hover:text-amber-100"
                    onClick={async () => {
                      try {
                        await camerasApi.setAiMode(camera.camera_id, "AI", camera);
                        setPtzAiMode("AI");
                        toast.success("AI resumed");
                      } catch { toast.error("Failed to resume AI"); }
                    }}
                  >Resume AI</button>
                </div>
              )}
              {/* Sensitivity slider */}
              <div className="space-y-0.5">
                <div className="flex justify-between text-[10px]">
                  <span className="text-zinc-500">Sensitivity</span>
                  <span className="text-zinc-400 font-mono">{sensitivity.toFixed(3)}</span>
                </div>
                <input type="range" min={0.005} max={0.2} step={0.005} value={sensitivity}
                  onChange={(e) => setSensitivity(parseFloat(e.target.value))}
                  className="w-full h-1 appearance-none rounded-full bg-zinc-700 accent-zinc-400 cursor-pointer"
                />
              </div>

              {/* D-pad */}
              <div className="grid grid-cols-3 grid-rows-3 gap-1 w-fit mx-auto">
                {dpad.map(({ icon: Icon, dp, pos, label }) => (
                  <button
                    key={label}
                    disabled={moving || !canEdit}
                    onClick={() => label === "HOME" ? gotoHome() : move("relative", dp)}
                    className={cn(
                      "flex h-8 w-8 items-center justify-center rounded-lg border border-zinc-700 bg-zinc-900",
                      "hover:bg-zinc-800 transition-colors disabled:opacity-40", pos,
                      label === "HOME" && "border-zinc-600 bg-zinc-800",
                    )}
                    title={label}
                  >
                    <Icon className="h-3.5 w-3.5 text-zinc-300" />
                  </button>
                ))}
              </div>
              {/* Zoom */}
              <div className="flex items-center gap-2">
                <button disabled={moving || !canEdit} onClick={() => move("relative", { zoom: -0.1 })}
                  className="flex h-8 w-8 items-center justify-center rounded-lg border border-zinc-700 bg-zinc-900 hover:bg-zinc-800 disabled:opacity-40">
                  <ZoomOut className="h-3.5 w-3.5 text-zinc-300" />
                </button>
                <div className="flex-1 text-center text-[10px] text-zinc-500">Zoom</div>
                <button disabled={moving || !canEdit} onClick={() => move("relative", { zoom: 0.1 })}
                  className="flex h-8 w-8 items-center justify-center rounded-lg border border-zinc-700 bg-zinc-900 hover:bg-zinc-800 disabled:opacity-40">
                  <ZoomIn className="h-3.5 w-3.5 text-zinc-300" />
                </button>
              </div>
              {/* Sliders */}
              {[
                { label: "Pan",  min: -1, max: 1, step: 0.01, val: pan,  set: setPan  },
                { label: "Tilt", min: -1, max: 1, step: 0.01, val: tilt, set: setTilt },
                { label: "Zoom", min:  0, max: 1, step: 0.01, val: zoom, set: setZoom },
              ].map(({ label, min, max, step, val, set: setter }) => (
                <div key={label} className="space-y-0.5">
                  <div className="flex justify-between text-[10px]">
                    <span className="text-zinc-500">{label}</span>
                    <span className="text-zinc-400 font-mono">{val.toFixed(2)}</span>
                  </div>
                  <input type="range" min={min} max={max} step={step} value={val}
                    disabled={!canEdit}
                    onChange={(e) => setter(parseFloat(e.target.value))}
                    className="w-full h-1 appearance-none rounded-full bg-zinc-700 accent-zinc-400 cursor-pointer disabled:opacity-40"
                  />
                </div>
              ))}
              <Button size="sm" className="w-full text-xs" disabled={moving || !canEdit}
                onClick={() => move("absolute", { pan, tilt, zoom }, 0.3)}>
                {moving ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : "Move to Position"}
              </Button>
            </div>
          )}

          {subTab === "calibrate" && (
            <FovCalibrationPanel
              camera={camera}
              canEdit={canEdit}
              isMoving={moving}
              onCalibrated={() => qc.invalidateQueries({ queryKey: ["cameras"] })}
            />
          )}
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Tab 4 — Scan Presets
// ─────────────────────────────────────────────────────────────────────────────

interface ScanPreset {
  preset_id: string;
  name: string;
  pan: number;
  tilt: number;
  zoom: number;
  dwell_s: number;
  order_idx: number;
}

function PresetsTab({ camera, canEdit }: { camera: Camera; canEdit: boolean }) {
  const { data, isLoading, refetch } = useQuery<ScanPreset[]>({
    queryKey: ["presets", camera.camera_id],
    queryFn:  () => camerasApi.listPresets(camera.camera_id, camera).then((r) => r.data),
  });
  const presets: ScanPreset[] = data ?? [];

  const [adding, setAdding]     = useState(false);
  const [editId, setEditId]     = useState<string | null>(null);
  const [form, setForm]         = useState({ name: "", pan: 0, tilt: 0, zoom: 0, dwell_s: 3 });
  const [saving, setSaving]     = useState(false);
  const [goingTo, setGoingTo]   = useState<string | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);

  // Move speed / settle state (camera-level, not per-preset)
  const initMoveSpeed = ((camera as any).learned_params?.camera_speeds?.move_speed as number | undefined) ?? 0.5;
  const initSettleS   = ((camera as any).learned_params?.camera_speeds?.settle_s   as number | undefined) ?? 0.35;
  const [moveSpeed, setMoveSpeed] = useState<number>(initMoveSpeed);
  const [settleS,   setSettleS]   = useState<number>(initSettleS);
  const [speedSaving, setSpeedSaving] = useState(false);

  async function handleSaveSpeed() {
    setSpeedSaving(true);
    try {
      await camerasApi.setMoveSpeed(camera.camera_id, { move_speed: moveSpeed, settle_s: settleS }, camera);
      toast.success("Move speed saved");
    } catch { toast.error("Failed to save move speed"); }
    finally { setSpeedSaving(false); }
  }

  // Seed form with current PTZ position for "capture" feature
  const [ptzNow, setPtzNow] = useState<{ pan: number; tilt: number; zoom: number } | null>(null);
  useEffect(() => {
    camerasApi.ptzStatus(camera.camera_id, camera)
      .then((r) => {
        const d = r.data as { pan?: number; tilt?: number; zoom?: number };
        if (d.pan != null) setPtzNow({ pan: d.pan, tilt: d.tilt ?? 0, zoom: d.zoom ?? 0 });
      })
      .catch(() => {});
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [camera.camera_id]);

  function openAdd(capturePos = false) {
    setEditId(null);
    setForm(capturePos && ptzNow
      ? { name: `Preset ${presets.length + 1}`, ...ptzNow, dwell_s: 3 }
      : { name: `Preset ${presets.length + 1}`, pan: 0, tilt: 0, zoom: 0, dwell_s: 3 });
    setAdding(true);
  }

  function openEdit(p: ScanPreset) {
    setAdding(false);
    setEditId(p.preset_id);
    setForm({ name: p.name, pan: p.pan, tilt: p.tilt, zoom: p.zoom, dwell_s: p.dwell_s });
  }

  async function handleSave() {
    setSaving(true);
    try {
      if (editId) {
        await camerasApi.updatePreset(camera.camera_id, editId, form, camera);
        toast.success("Preset updated");
      } else {
        await camerasApi.createPreset(camera.camera_id, form, camera);
        toast.success("Preset created");
      }
      setAdding(false);
      setEditId(null);
      refetch();
    } catch { toast.error("Failed to save preset"); }
    finally { setSaving(false); }
  }

  async function handleDelete(presetId: string) {
    setDeleting(presetId);
    try {
      await camerasApi.deletePreset(camera.camera_id, presetId, camera);
      toast.success("Preset deleted");
      refetch();
    } catch { toast.error("Failed to delete preset"); }
    finally { setDeleting(null); }
  }

  async function handleGoto(presetId: string) {
    setGoingTo(presetId);
    try {
      await camerasApi.gotoPreset(camera.camera_id, presetId, camera);
      toast.success("Camera moved to preset");
      // Refresh PTZ status after move
      setTimeout(() => {
        camerasApi.ptzStatus(camera.camera_id, camera)
          .then((r) => {
            const d = r.data as { pan?: number; tilt?: number; zoom?: number };
            if (d.pan != null) setPtzNow({ pan: d.pan, tilt: d.tilt ?? 0, zoom: d.zoom ?? 0 });
          }).catch(() => {});
      }, 1500);
    } catch { toast.error("Failed to move to preset"); }
    finally { setGoingTo(null); }
  }

  async function handleMoveUp(idx: number) {
    if (idx === 0) return;
    const ids = presets.map((p) => p.preset_id);
    [ids[idx - 1], ids[idx]] = [ids[idx], ids[idx - 1]];
    try {
      await camerasApi.reorderPresets(camera.camera_id, ids, camera);
      refetch();
    } catch { toast.error("Reorder failed"); }
  }

  async function handleMoveDown(idx: number) {
    if (idx >= presets.length - 1) return;
    const ids = presets.map((p) => p.preset_id);
    [ids[idx], ids[idx + 1]] = [ids[idx + 1], ids[idx]];
    try {
      await camerasApi.reorderPresets(camera.camera_id, ids, camera);
      refetch();
    } catch { toast.error("Reorder failed"); }
  }

  const formPanel = (
    <div className="rounded-lg border border-zinc-700 bg-zinc-900/50 p-3 flex flex-col gap-2.5">
      <p className="text-xs font-medium text-zinc-300">{editId ? "Edit Preset" : "New Preset"}</p>
      <div className="flex flex-col gap-1.5">
        <Label className="text-[10px] text-zinc-500">Name</Label>
        <input
          className="w-full rounded bg-zinc-800 border border-zinc-700 px-2 py-1 text-xs text-zinc-200 focus:outline-none focus:border-zinc-500"
          value={form.name}
          onChange={(e) => setForm((f) => ({ ...f, name: e.target.value }))}
        />
      </div>
      {[
        { key: "pan",  label: "Pan",    min: -1, max: 1, step: 0.001 },
        { key: "tilt", label: "Tilt",   min: -1, max: 1, step: 0.001 },
        { key: "zoom", label: "Zoom",   min:  0, max: 1, step: 0.001 },
        { key: "dwell_s", label: "Dwell (s)", min: 1, max: 60, step: 0.5 },
      ].map(({ key, label, min, max, step }) => (
        <div key={key} className="flex items-center gap-2">
          <span className="text-[10px] text-zinc-500 w-14 shrink-0">{label}</span>
          <input
            type="range" min={min} max={max} step={step}
            value={form[key as keyof typeof form]}
            onChange={(e) => setForm((f) => ({ ...f, [key]: parseFloat(e.target.value) }))}
            className="flex-1 h-1 appearance-none rounded-full bg-zinc-700 accent-zinc-400 cursor-pointer"
          />
          <span className="text-[10px] font-mono text-zinc-400 w-10 text-right">
            {(form[key as keyof typeof form] as number).toFixed(key === "dwell_s" ? 1 : 3)}
          </span>
        </div>
      ))}
      <div className="flex gap-2 mt-1">
        <Button size="sm" className="flex-1 text-xs" disabled={saving} onClick={handleSave}>
          {saving ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Check className="h-3.5 w-3.5 mr-1" />}
          Save
        </Button>
        <Button size="sm" variant="outline" className="text-xs border-zinc-700"
          onClick={() => { setAdding(false); setEditId(null); }}>
          Cancel
        </Button>
      </div>
    </div>
  );

  return (
    <div className="flex flex-col gap-3 p-1">
      {/* ── Camera Move Speed ─────────────────────────────────── */}
      <div className="rounded-lg border border-zinc-700 bg-zinc-900/50 p-3 flex flex-col gap-2">
        <p className="text-[11px] font-medium text-zinc-300">Transition Speed</p>
        {[
          { key: "moveSpeed", label: "Move speed", min: 0.05, max: 1.0, step: 0.01, value: moveSpeed, set: setMoveSpeed, decimals: 2 },
          { key: "settleS",   label: "Settle (s)",  min: 0.0,  max: 5.0, step: 0.05, value: settleS,   set: setSettleS,   decimals: 2 },
        ].map(({ key, label, min, max, step, value, set, decimals }) => (
          <div key={key} className="flex items-center gap-2">
            <span className="text-[10px] text-zinc-500 w-16 shrink-0">{label}</span>
            <input
              type="range" min={min} max={max} step={step}
              value={value}
              onChange={(e) => set(parseFloat(e.target.value))}
              disabled={!canEdit}
              className="flex-1 h-1 appearance-none rounded-full bg-zinc-700 accent-zinc-400 cursor-pointer disabled:opacity-50"
            />
            <span className="text-[10px] font-mono text-zinc-400 w-8 text-right">{value.toFixed(decimals)}</span>
          </div>
        ))}
        {canEdit && (
          <Button size="sm" className="h-6 text-[10px] w-fit self-end mt-0.5" disabled={speedSaving} onClick={handleSaveSpeed}>
            {speedSaving ? <Loader2 className="h-3 w-3 animate-spin" /> : <Check className="h-3 w-3 mr-1" />}
            Apply
          </Button>
        )}
      </div>

      <div className="flex items-center justify-between">
        <p className="text-xs text-zinc-400">
          {presets.length === 0
            ? "No presets yet. Add positions for the camera to visit during scanning."
            : `${presets.length} preset${presets.length !== 1 ? "s" : ""} — camera cycles through these in order.`}
        </p>
        {canEdit && (
          <div className="flex gap-1.5">
            {ptzNow && (
              <Button size="sm" variant="outline"
                className="h-7 text-[10px] border-emerald-700 text-emerald-400 hover:bg-emerald-900/30"
                onClick={() => openAdd(true)}>
                <Navigation className="h-3 w-3 mr-1" />
                Capture
              </Button>
            )}
            <Button size="sm" className="h-7 text-[10px]" onClick={() => openAdd(false)}>
              <Plus className="h-3 w-3 mr-1" />
              Add
            </Button>
          </div>
        )}
      </div>

      {isLoading && <Loader2 className="h-4 w-4 animate-spin text-zinc-500 mx-auto" />}

      {(adding && !editId) && formPanel}

      {presets.length === 0 && !isLoading && !adding && (
        <div className="flex flex-col items-center gap-2 py-8 text-zinc-600">
          <MapPin className="h-8 w-8" />
          <p className="text-xs">Add presets to enable scanning</p>
        </div>
      )}

      <div className="flex flex-col gap-2">
        {presets.map((preset, idx) => (
          <div key={preset.preset_id}>
            {editId === preset.preset_id && formPanel}
            {editId !== preset.preset_id && (
              <div className="rounded-lg border border-zinc-700 bg-zinc-900/40 px-3 py-2.5 flex items-start gap-2.5">
                {/* Order controls */}
                <div className="flex flex-col gap-0.5 mt-0.5">
                  <button
                    className="text-zinc-600 hover:text-zinc-400 disabled:opacity-30"
                    disabled={idx === 0 || !canEdit}
                    onClick={() => handleMoveUp(idx)}
                    title="Move up"
                  >
                    <ArrowUp className="h-3 w-3" />
                  </button>
                  <button
                    className="text-zinc-600 hover:text-zinc-400 disabled:opacity-30"
                    disabled={idx === presets.length - 1 || !canEdit}
                    onClick={() => handleMoveDown(idx)}
                    title="Move down"
                  >
                    <ArrowDown className="h-3 w-3" />
                  </button>
                </div>

                {/* Content */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-1.5">
                    <span className="text-[10px] font-medium text-zinc-500">#{idx + 1}</span>
                    <span className="text-xs font-medium text-zinc-200 truncate">{preset.name}</span>
                  </div>
                  <div className="flex flex-wrap gap-x-3 gap-y-0.5 mt-1">
                    {[
                      { l: "P", v: preset.pan.toFixed(3) },
                      { l: "T", v: preset.tilt.toFixed(3) },
                      { l: "Z", v: preset.zoom.toFixed(3) },
                      { l: "Dwell", v: `${preset.dwell_s}s` },
                    ].map(({ l, v }) => (
                      <span key={l} className="text-[10px] text-zinc-500 font-mono">
                        <span className="text-zinc-600">{l}:</span> {v}
                      </span>
                    ))}
                  </div>
                </div>

                {/* Actions */}
                <div className="flex items-center gap-1 shrink-0">
                  <button
                    className="text-zinc-500 hover:text-blue-400 transition-colors disabled:opacity-40"
                    title="Go to preset"
                    disabled={goingTo === preset.preset_id}
                    onClick={() => handleGoto(preset.preset_id)}
                  >
                    {goingTo === preset.preset_id
                      ? <Loader2 className="h-3.5 w-3.5 animate-spin" />
                      : <Navigation className="h-3.5 w-3.5" />}
                  </button>
                  {canEdit && (
                    <>
                      <button
                        className="text-zinc-500 hover:text-zinc-300 transition-colors"
                        title="Edit"
                        onClick={() => openEdit(preset)}
                      >
                        <MousePointer2 className="h-3.5 w-3.5" />
                      </button>
                      <button
                        className="text-zinc-500 hover:text-red-400 transition-colors disabled:opacity-40"
                        title="Delete"
                        disabled={deleting === preset.preset_id}
                        onClick={() => handleDelete(preset.preset_id)}
                      >
                        {deleting === preset.preset_id
                          ? <Loader2 className="h-3.5 w-3.5 animate-spin" />
                          : <Trash2 className="h-3.5 w-3.5" />}
                      </button>
                    </>
                  )}
                </div>
              </div>
            )}
          </div>
        ))}
      </div>

      {presets.length > 0 && (
        <p className="text-[10px] text-zinc-600 leading-4">
          Camera moves smoothly between presets during scanning. AI pipeline runs in-transit and at each position.
          Dwell time controls how long the camera recognizes faces at each stop.
        </p>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Tab 5 — Camera Info
// ─────────────────────────────────────────────────────────────────────────────

function InfoTab({ camera }: { camera: Camera }) {
  const rows: Array<{ label: string; value: string | null | undefined }> = [
    { label: "ONVIF Host",    value: `${camera.onvif_host}:${camera.onvif_port}` },
    { label: "ONVIF User",    value: camera.onvif_username },
    { label: "RTSP URL",      value: camera.rtsp_url },
    { label: "Node",          value: camera.node_id ?? "Unassigned" },
    { label: "Mode",          value: camera.mode },
    { label: "FOV (H × V)",   value: camera.fov_h != null ? `${camera.fov_h.toFixed(1)}° × ${camera.fov_v?.toFixed(1)}°` : null },
    { label: "Pan speed",     value: camera.pan_speed   != null ? String(camera.pan_speed)   : null },
    { label: "Tilt speed",    value: camera.tilt_speed  != null ? String(camera.tilt_speed)  : null },
    { label: "Zoom speed",    value: camera.zoom_speed  != null ? String(camera.zoom_speed)  : null },
    { label: "Added",         value: camera.created_at },
    { label: "Updated",       value: camera.updated_at },
  ];

  return (
    <div className="divide-y divide-border/40">
      {rows.map(({ label, value }) => (
        <div key={label} className="flex items-start justify-between py-2.5 gap-4">
          <span className="text-xs text-muted-foreground shrink-0">{label}</span>
          <span className="text-xs text-zinc-300 text-right font-mono break-all">{value ?? "—"}</span>
        </div>
      ))}

      {camera.monitoring_hours && (
        <div className="py-2.5">
          <p className="text-xs text-muted-foreground mb-1">Monitoring hours</p>
          {/* Support both {start,end,days} frontend format and legacy day-keyed format */}
          {(camera.monitoring_hours as any).start != null ? (
            <p className="text-xs text-zinc-300 font-mono">
              {(camera.monitoring_hours as any).start}–{(camera.monitoring_hours as any).end}
              {Array.isArray((camera.monitoring_hours as any).days) && (
                <> · {(camera.monitoring_hours as any).days.join(", ")}</>
              )}
            </p>
          ) : (
            <pre className="text-[10px] text-zinc-400 bg-zinc-900 rounded-lg p-2 overflow-x-auto">
              {JSON.stringify(camera.monitoring_hours, null, 2)}
            </pre>
          )}
        </div>
      )}

      {camera.learned_params && Object.keys(camera.learned_params).length > 0 && (
        <div className="py-2.5">
          <p className="text-xs text-muted-foreground mb-1.5">Learned ONVIF params</p>
          <pre className="text-[10px] text-zinc-400 bg-zinc-900 rounded-lg p-3 overflow-x-auto">
            {JSON.stringify(camera.learned_params, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Camera Detail Dialog
// ─────────────────────────────────────────────────────────────────────────────

function CameraDetailDialog({
  camera, onClose, canEdit, monitoring, onStart, onStop,
}: {
  camera:     Camera | null;
  onClose:    () => void;
  canEdit:    boolean;
  monitoring: boolean;
  onStart:    () => void;
  onStop:     () => void;
}) {
  const qc = useQueryClient();
  const [deleting, setDeleting] = useState(false);
  const [editOpen, setEditOpen] = useState(false);

  if (!camera) return null;

  async function handleDelete() {
    if (!confirm(`Delete camera "${camera!.name}"? This cannot be undone.`)) return;
    setDeleting(true);
    try {
      await camerasApi.delete(camera!.camera_id);
      toast.success("Camera deleted");
      qc.invalidateQueries({ queryKey: ["cameras"] });
      onClose();
    } catch {
      toast.error("Delete failed");
    } finally {
      setDeleting(false);
    }
  }

  return (
    <Dialog open={!!camera} onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-w-4xl max-h-[92vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <div className="flex items-center gap-3 pr-6">
            <div className={cn(
              "h-2.5 w-2.5 rounded-full shrink-0",
              STATUS_INDICATOR[camera.status],
            )} />
            <DialogTitle className="text-base">{camera.name}</DialogTitle>
            <span className="text-xs text-zinc-500">{camera.room_name}</span>
            <span className={cn("ml-auto rounded-full px-2.5 py-0.5 text-[10px] font-semibold shrink-0", MODE_COLORS[camera.mode])}>
              {camera.mode}
            </span>
            {canEdit && (
              <div className="flex items-center gap-2 shrink-0">
                <Button
                  size="sm" variant="outline"
                  className="h-7 text-xs px-2"
                  onClick={() => setEditOpen(true)}
                  title="Edit camera settings"
                >
                  <Pencil className="h-3.5 w-3.5 mr-1" />Edit
                </Button>
                <Button
                  size="sm" variant="outline"
                  className="h-7 text-xs px-2 text-red-400 hover:text-red-300 hover:border-red-400"
                  onClick={handleDelete}
                  disabled={deleting}
                  title="Delete this camera"
                >
                  {deleting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <><Trash2 className="h-3.5 w-3.5 mr-1" />Delete</>}
                </Button>
              </div>
            )}
          </div>
        </DialogHeader>

        <div className="flex-1 overflow-y-auto">
          <Tabs defaultValue="live">
            {(() => {
              const ct = (camera as any).camera_type ?? "PTZ";
              const showPtz = ct === "PTZ" || ct === "PTZ_ZOOM";
              return (
                <>
                  <TabsList className="w-full justify-start mb-1">
                    <TabsTrigger value="live">Live View</TabsTrigger>
                    {showPtz && <TabsTrigger value="zone-ptz">PTZ &amp; Presets</TabsTrigger>}
                    <TabsTrigger value="info">Info</TabsTrigger>
                  </TabsList>

                  <TabsContent value="live"><LiveViewTab camera={camera} canEdit={canEdit} monitoring={monitoring} onStart={onStart} onStop={onStop} /></TabsContent>
                  {showPtz && <TabsContent value="zone-ptz"><ZonePtzTab camera={camera} canEdit={canEdit} /></TabsContent>}
                  <TabsContent value="info"><InfoTab camera={camera} /></TabsContent>
                </>
              );
            })()}
          </Tabs>
        </div>
      </DialogContent>

      <EditCameraDialog
        camera={camera}
        open={editOpen}
        onClose={() => setEditOpen(false)}
      />
    </Dialog>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Page
// ─────────────────────────────────────────────────────────────────────────────

export default function CamerasPage() {
  const qc          = useQueryClient();
  const { role }    = useAuth();
  const canEdit     = role !== "VIEWER";

  const [search,    setSearch]    = useState("");
  const [statusF,   setStatusF]   = useState("ALL");
  const [showAdd,   setShowAdd]   = useState(false);
  const [selected,  setSelected]  = useState<Camera | null>(null);
  const [offScheduleCam, setOffScheduleCam] = useState<Camera | null>(null);

  const { data: camerasData, isLoading, refetch } = useQuery({
    queryKey: ["cameras", search, statusF],
    queryFn:  () => camerasApi.list({
      q:      search  || undefined,
      status: statusF !== "ALL" ? statusF : undefined,
      limit:  200,
    }).then((r) => r.data),
  });

  const cameras: Camera[] = (camerasData as { items?: Camera[]; cameras?: Camera[] })?.items
    ?? (camerasData as { cameras?: Camera[] })?.cameras
    ?? [];

  const total     = (camerasData as { total?: number })?.total ?? cameras.length;
  const maxLimit  = (camerasData as { limit?: number; max_cameras?: number })?.max_cameras ?? null;

  // Keep the dialog's camera object in sync whenever the list re-fetches
  useEffect(() => {
    if (!selected) return;
    const fresh = cameras.find((c) => c.camera_id === selected.camera_id);
    if (fresh) setSelected(fresh);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cameras]);

  // Poll active sessions to highlight monitoring cameras
  const { data: activeData } = useQuery({
    queryKey: ["active-sessions"],
    queryFn:  () => sessionsApi.active().then((r) => r.data as { items?: Array<{ camera_id: string }> }),
    refetchInterval: 15000,
  });
  const monitoringCameraIds = new Set(
    (activeData?.items ?? []).map((s) => s.camera_id),
  );

  // Fetch all timetables so we can check schedule before starting a session
  const { data: allTimetablesRaw } = useQuery({
    queryKey: ["timetables"],
    queryFn:  () => timetableApi.list().then((r) => r.data),
  });
  const allTimetables: Array<{
    timetable_id: string;
    entries: Array<{ day_of_week: number; start_time: string; end_time: string }>;
  }> = Array.isArray(allTimetablesRaw) ? allTimetablesRaw : [];

  const startMutation = useMutation({
    mutationFn: (cam: Camera) => nodeApi.start(cam.camera_id, cam),
    onSuccess:  () => { toast.success("AI session started"); qc.invalidateQueries({ queryKey: ["cameras"] }); qc.invalidateQueries({ queryKey: ["active-sessions"] }); },
    onError:    (err: any) => {
      const raw = err?.response?.data?.detail;
      const detail = typeof raw === "string"
        ? raw
        : Array.isArray(raw)
          ? raw.map((d: any) => d?.msg ?? JSON.stringify(d)).join("; ")
          : typeof raw === "object" && raw !== null
            ? JSON.stringify(raw)
            : "Start failed";
      toast.error(String(detail));
    },
  });

  /** Check timetable before starting; show warning dialog if outside schedule. */
  const handleStart = (cam: Camera) => {
    if (!cam.timetable_id) {
      startMutation.mutate(cam);
      return;
    }
    const tt = allTimetables.find((t) => t.timetable_id === cam.timetable_id);
    if (!tt) {
      startMutation.mutate(cam);
      return;
    }
    const now   = new Date();
    // JS getDay(): 0=Sun … 6=Sat → convert to 0=Mon … 6=Sun
    const dow   = (now.getDay() + 6) % 7;
    const hhmm  = now.toTimeString().slice(0, 5); // "HH:MM"
    const inSchedule = tt.entries.some(
      (e) => e.day_of_week === dow && e.start_time <= hhmm && hhmm <= e.end_time,
    );
    if (inSchedule) {
      startMutation.mutate(cam);
    } else {
      setOffScheduleCam(cam);
    }
  };

  const stopMutation = useMutation({
    mutationFn: (cam: Camera) => nodeApi.stop(cam.camera_id, cam),
    onSuccess:  () => { toast.success("Session stopped"); qc.invalidateQueries({ queryKey: ["cameras"] }); qc.invalidateQueries({ queryKey: ["active-sessions"] }); },
    onError:    (err: any) => {
      const raw = err?.response?.data?.detail;
      const detail = typeof raw === "string"
        ? raw
        : Array.isArray(raw)
          ? raw.map((d: any) => d?.msg ?? JSON.stringify(d)).join("; ")
          : typeof raw === "object" && raw !== null
            ? JSON.stringify(raw)
            : "Stop failed";
      toast.error(String(detail));
    },
  });

  const testMutation = useMutation({
    mutationFn: (cam: Camera) => camerasApi.test(cam.camera_id, cam),
    onSuccess:  () => toast.success("Camera reachable"),
    onError:    () => toast.error("Camera unreachable"),
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) => camerasApi.delete(id),
    onSuccess:  () => { toast.success("Camera deleted"); qc.invalidateQueries({ queryKey: ["cameras"] }); },
    onError:    () => toast.error("Delete failed"),
  });

  const statusCounts = {
    ALL:      cameras.length,
    ONLINE:   cameras.filter((c) => c.status === "ONLINE").length,
    OFFLINE:  cameras.filter((c) => c.status === "OFFLINE").length,
    DEGRADED: cameras.filter((c) => c.status === "DEGRADED").length,
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold text-zinc-50">Cameras</h1>
          <p className="text-sm text-muted-foreground mt-0.5">
            {statusCounts.ONLINE} online · {statusCounts.DEGRADED > 0 ? `${statusCounts.DEGRADED} degraded · ` : ""}
            {statusCounts.OFFLINE} offline
            {maxLimit != null && (
              <span className="ml-2 text-zinc-500">
                · <span className={cn(cameras.length >= maxLimit ? "text-amber-400" : "text-zinc-400")}>
                  {cameras.length}/{maxLimit} cameras (client limit)
                </span>
              </span>
            )}
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={() => refetch()}>
            <RefreshCw className="h-4 w-4" />
          </Button>
          {canEdit && (
            <Button size="sm" onClick={() => setShowAdd(true)}>
              <Plus className="h-4 w-4" />
              Add Camera
            </Button>
          )}
        </div>
      </div>

      {/* Filters */}
      <div className="flex items-center gap-3 flex-wrap">
        <div className="relative">
          <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-zinc-500" />
          <Input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search cameras…"
            className="pl-8 bg-zinc-800 border-zinc-700 w-52"
          />
        </div>
        <div className="flex gap-1 rounded-lg border border-border/50 bg-zinc-900 p-1">
          {(["ALL", "ONLINE", "DEGRADED", "OFFLINE"] as const).map((s) => (
            <button
              key={s}
              onClick={() => setStatusF(s)}
              className={cn(
                "flex items-center gap-1.5 rounded px-3 py-1.5 text-xs font-medium transition-colors",
                statusF === s ? "bg-zinc-700 text-zinc-100" : "text-zinc-500 hover:text-zinc-300",
              )}
            >
              <span className={cn("h-1.5 w-1.5 rounded-full",
                s === "ONLINE"   ? "bg-green-500" :
                s === "DEGRADED" ? "bg-amber-500" :
                s === "OFFLINE"  ? "bg-zinc-500"  : "bg-zinc-600"
              )} />
              {s === "ALL" ? "All" : s}
              <span className={cn(
                "rounded-full px-1 text-[10px]",
                statusF === s ? "bg-zinc-600" : "bg-zinc-800",
              )}>
                {statusCounts[s]}
              </span>
            </button>
          ))}
        </div>
      </div>

      {/* Grid */}
      {isLoading ? (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
          {Array.from({ length: 10 }).map((_, i) => (
            <Skeleton key={i} className="h-52 rounded-xl" />
          ))}
        </div>
      ) : cameras.length === 0 ? (
        <div className="flex h-60 items-center justify-center rounded-xl border border-dashed border-border/50">
          <div className="text-center">
            <CameraIcon className="mx-auto h-9 w-9 text-zinc-700 mb-2" />
            <p className="text-sm text-muted-foreground">No cameras found</p>
            {canEdit && (
              <Button size="sm" className="mt-3" onClick={() => setShowAdd(true)}>
                <Plus className="h-3.5 w-3.5" />Add your first camera
              </Button>
            )}
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
          {cameras.map((c) => (
            <CameraCard
              key={c.camera_id}
              camera={c}
              canEdit={canEdit}
              monitoring={monitoringCameraIds.has(c.camera_id)}
              onOpen={() => setSelected(c)}
              onStart={() => handleStart(c)}
              onStop={() => stopMutation.mutate(c)}
              onTest={() => testMutation.mutate(c)}
              onDelete={() => {
                if (confirm(`Delete camera "${c.name}"? This cannot be undone.`))
                  deleteMutation.mutate(c.camera_id);
              }}
            />
          ))}
        </div>
      )}

      {/* Dialogs */}
      {canEdit && <AddCameraDialog open={showAdd} onClose={() => setShowAdd(false)} />}
      <CameraDetailDialog
        camera={selected}
        onClose={() => setSelected(null)}
        canEdit={canEdit}
        monitoring={selected ? monitoringCameraIds.has(selected.camera_id) : false}
        onStart={() => selected && handleStart(selected)}
        onStop={() => selected && stopMutation.mutate(selected)}
      />

      {/* Off-schedule warning dialog */}
      <Dialog open={!!offScheduleCam} onOpenChange={(o) => { if (!o) setOffScheduleCam(null); }}>
        <DialogContent className="max-w-sm">
          <DialogHeader>
            <DialogTitle>Outside Scheduled Hours</DialogTitle>
            <DialogDescription>
              <strong>{offScheduleCam?.name}</strong> is not scheduled for surveillance at
              this time. Starting a session outside timetable hours may not be intended.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="gap-2 sm:gap-0">
            <Button variant="outline" onClick={() => setOffScheduleCam(null)}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={() => {
                if (offScheduleCam) startMutation.mutate(offScheduleCam);
                setOffScheduleCam(null);
              }}
            >
              Start Anyway
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
