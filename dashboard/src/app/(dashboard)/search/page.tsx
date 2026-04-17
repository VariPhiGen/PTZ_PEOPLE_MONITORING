"use client";

import {
  useState, useEffect, useRef, useCallback, useMemo, Fragment,
} from "react";
import { useQuery } from "@tanstack/react-query";
import { format, parseISO, subDays } from "date-fns";
import {
  Camera, ChevronLeft, ChevronRight, Clock, Eye, ImageOff, MapPin,
  Search, Upload, User, Video, VideoOff, X, ZoomIn,
  Calendar, ArrowRight, Fingerprint, Circle,
} from "lucide-react";
import type { FaceDetection as FDClass, Results as FDResults, Detection } from "@mediapipe/face_detection";
import { searchApi } from "@/lib/api";
import { cn, fmtDatetime } from "@/lib/utils";
import { Badge }  from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input }  from "@/components/ui/input";
import type { SearchHit, JourneyEvent, PersonRole } from "@/types";

// ── Utility types ─────────────────────────────────────────────────────────────

interface ExtJourneyEvent extends JourneyEvent {
  course_name?:  string | null;
  building?:     string | null;
  evidence_refs?: string[];
  confidence?:   number;
}

interface JourneyResponse {
  events:   ExtJourneyEvent[];
  summary?: {
    total_locations: number;
    total_duration_s: number;
    attendance_rate: number;
  };
}

interface HeatmapResponse {
  areas:  string[];
  matrix: number[][];  // [areaIdx][0..23] = minutes_present
}

interface CrossCameraEntry {
  camera_id:    string;
  camera_name:  string;
  area:         string | null;
  building:     string | null;
  first_seen:   string;
  last_seen:    string;
  transit_time: number | null;
}

interface EvidenceFrame {
  url:        string;
  timestamp:  string;
  confidence: number;
  location:   string;
}

type FaceQuality = "bad" | "fair" | "good";

// ── Constants ─────────────────────────────────────────────────────────────────

const ROLE_COLOR: Record<PersonRole, string> = {
  STUDENT: "text-blue-400   bg-blue-900/30",
  FACULTY: "text-violet-400 bg-violet-900/30",
  ADMIN:   "text-amber-400  bg-amber-900/30",
};

const EVT_COLOR = {
  ATTENDANCE: { fill: "rgba(59,130,246,0.75)",  stroke: "#3b82f6", bg: "bg-blue-900/20"   },
  SIGHTING:   { fill: "rgba(168,85,247,0.75)",  stroke: "#a855f7", bg: "bg-purple-900/20" },
};

const STATUS_COLOR: Record<string, string> = {
  P: "bg-green-900/40 text-green-400", L: "bg-amber-900/40 text-amber-400",
  EE: "bg-blue-900/40 text-blue-400",  A: "bg-red-900/40 text-red-400",
  ND: "bg-zinc-800 text-zinc-500",     EX: "bg-purple-900/40 text-purple-400",
};

// ── Quality helpers ───────────────────────────────────────────────────────────

function assessQuality(detections: Detection[], vw: number, vh: number): FaceQuality {
  if (detections.length !== 1) return "bad";
  const bb = detections[0].boundingBox;
  const area = bb.width * bb.height;
  const cx   = bb.xCenter, cy = bb.yCenter;
  const lms  = detections[0].landmarks;

  // Levelled face check (eye y-coords close)
  let level = true;
  if (lms.length >= 2) {
    level = Math.abs(lms[0].y - lms[1].y) < 0.06;
  }
  // Enough size + centred
  if (area > 0.06 && cx > 0.22 && cx < 0.78 && cy > 0.15 && cy < 0.85 && level) return "good";
  if (area > 0.02 && cx > 0.1  && cx < 0.9)                                        return "fair";
  return "bad";
  void vw; void vh;
}

function drawDetections(
  ctx: CanvasRenderingContext2D,
  W: number, H: number,
  detections: Detection[],
  quality: FaceQuality,
) {
  ctx.clearRect(0, 0, W, H);
  const color = quality === "good" ? "#22c55e" : quality === "fair" ? "#f59e0b" : "#ef4444";

  for (const d of detections) {
    const bb = d.boundingBox;
    const x  = (bb.xCenter - bb.width  / 2) * W;
    const y  = (bb.yCenter - bb.height / 2) * H;
    const w  = bb.width  * W;
    const h  = bb.height * H;

    // Glow
    ctx.shadowBlur  = 12;
    ctx.shadowColor = color;
    ctx.strokeStyle = color;
    ctx.lineWidth   = 2.5;
    ctx.strokeRect(x, y, w, h);
    ctx.shadowBlur  = 0;

    // Corner accents
    const cs = 14;
    ctx.beginPath();
    [
      [x, y, cs, 0, 0, cs], [x + w, y, -cs, 0, 0, cs],
      [x, y + h, cs, 0, 0, -cs], [x + w, y + h, -cs, 0, 0, -cs],
    ].forEach(([ox, oy, dx1, dy1, dx2, dy2]) => {
      ctx.moveTo(ox + dx1, oy); ctx.lineTo(ox, oy); ctx.lineTo(ox, oy + dy2);
      ctx.moveTo(ox + dx2, oy + (dy2 ? 0 : dy1)); void dy1; void dx1; void dx2; void oy; void ox;
    });
    // Simplified corners
    [[x, y, 1, 1], [x + w, y, -1, 1], [x, y + h, 1, -1], [x + w, y + h, -1, -1]].forEach(
      ([cx2, cy2, sx, sy]) => {
        ctx.beginPath();
        ctx.moveTo(cx2 + sx * cs, cy2);
        ctx.lineTo(cx2, cy2);
        ctx.lineTo(cx2, cy2 + sy * cs);
        ctx.stroke();
      }
    );

    // Landmark dots (eyes, nose)
    if (d.landmarks.length >= 3) {
      for (let i = 0; i < Math.min(3, d.landmarks.length); i++) {
        const lm = d.landmarks[i];
        ctx.beginPath();
        ctx.arc(lm.x * W, lm.y * H, 2.5, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
      }
    }
  }
}

// ── FaceSearch component ──────────────────────────────────────────────────────

function FaceSearch({ onSelect }: { onSelect: (h: SearchHit) => void }) {
  const [mode,         setMode]         = useState<"idle" | "webcam" | "done">("idle");
  const [quality,      setQuality]      = useState<FaceQuality>("bad");
  const [goodProgress, setGoodProgress] = useState(0);
  const [uploading,    setUploading]    = useState(false);
  const [results,      setResults]      = useState<SearchHit[] | null>(null);
  const [error,        setError]        = useState<string | null>(null);

  const videoRef    = useRef<HTMLVideoElement>(null);
  const overlayRef  = useRef<HTMLCanvasElement>(null);
  const fdRef       = useRef<FDClass | null>(null);
  const streamRef   = useRef<MediaStream | null>(null);
  const loopRef     = useRef<number>(0);
  const goodRef     = useRef<number | null>(null);
  const capturedRef  = useRef(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const stopWebcam = useCallback(() => {
    cancelAnimationFrame(loopRef.current);
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
    setMode("done");
  }, []);

  const postImage = useCallback(async (blob: Blob) => {
    const fd = new FormData();
    fd.append("file", blob, "face.jpg");
    try {
      const res = await searchApi.face(fd);
      const data = res.data as { items?: SearchHit[]; results?: SearchHit[] };
      setResults(data.items ?? data.results ?? []);
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } })
        ?.response?.data?.detail;
      setError(detail ? `Search failed: ${detail}` : "Face search failed — please try again.");
    }
    stopWebcam();
  }, [stopWebcam]);

  const captureAndSearch = useCallback(() => {
    if (capturedRef.current) return;
    capturedRef.current = true;
    const vid = videoRef.current;
    const cvs = overlayRef.current;
    if (!vid || !cvs) return;
    const tmp = document.createElement("canvas");
    tmp.width  = vid.videoWidth;
    tmp.height = vid.videoHeight;
    tmp.getContext("2d")?.drawImage(vid, 0, 0);
    tmp.toBlob(b => b && postImage(b), "image/jpeg", 0.92);
    cancelAnimationFrame(loopRef.current);
  }, [postImage]);

  // MediaPipe detection loop
  const runLoop = useCallback(() => {
    const vid = videoRef.current;
    const cvs = overlayRef.current;
    if (!vid || !cvs || vid.readyState < 2) {
      loopRef.current = requestAnimationFrame(runLoop);
      return;
    }
    if (cvs.width !== vid.videoWidth) {
      cvs.width  = vid.videoWidth;
      cvs.height = vid.videoHeight;
    }

    fdRef.current?.send({ image: vid }).catch(() => {});
    loopRef.current = requestAnimationFrame(runLoop);
  }, []);

  const handleFDResults = useCallback((res: FDResults) => {
    const cvs = overlayRef.current;
    if (!cvs) return;
    const ctx = cvs.getContext("2d");
    if (!ctx) return;

    const dets = res.detections ?? [];
    const q    = assessQuality(dets, cvs.width, cvs.height);
    setQuality(q);
    drawDetections(ctx, cvs.width, cvs.height, dets, q);

    if (q === "good") {
      if (!goodRef.current) goodRef.current = Date.now();
      const elapsed = Date.now() - goodRef.current;
      setGoodProgress(Math.min(100, (elapsed / 1000) * 100));
      if (elapsed >= 1000) captureAndSearch();
    } else {
      goodRef.current = null;
      setGoodProgress(0);
    }
  }, [captureAndSearch]);

  // Start webcam + load FaceDetection
  const startWebcam = useCallback(async () => {
    setMode("webcam");
    setResults(null);
    setError(null);
    capturedRef.current = false;
    goodRef.current     = null;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: "user" } });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      // Lazy-load FaceDetection (avoids SSR issues)
      if (!fdRef.current) {
        const { FaceDetection } = await import("@mediapipe/face_detection");
        const fd = new FaceDetection({
          locateFile: (f) => `/node_modules/@mediapipe/face_detection/${f}`,
        });
        fd.setOptions({ model: "short", minDetectionConfidence: 0.55 });
        fd.onResults(handleFDResults);
        await fd.initialize();
        fdRef.current = fd;
      } else {
        fdRef.current.onResults(handleFDResults);
      }

      loopRef.current = requestAnimationFrame(runLoop);
    } catch (err) {
      setError(`Webcam error: ${(err as Error).message}`);
      setMode("idle");
    }
  }, [handleFDResults, runLoop]);

  // File upload path
  const handleFileUpload = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setResults(null);
    setError(null);
    try {
      const fd = new FormData();
      fd.append("file", file);  // backend expects field name "file"
      const res = await searchApi.face(fd);
      const data = res.data as { items?: SearchHit[]; results?: SearchHit[] };
      setResults(data.items ?? data.results ?? []);
      setMode("done");
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } })
        ?.response?.data?.detail;
      setError(detail
        ? `Search failed: ${detail}`
        : "Upload failed — ensure exactly one face is visible in the photo.");
    } finally {
      setUploading(false);
    }
    e.target.value = "";
  }, []);

  useEffect(() => () => {
    cancelAnimationFrame(loopRef.current);
    streamRef.current?.getTracks().forEach(t => t.stop());
  }, []);

  const QUAL_COLOR = { good: "#22c55e", fair: "#f59e0b", bad: "#ef4444" } as const;

  return (
    <div className="space-y-4">
      {/* Webcam / Upload controls */}
      {mode === "idle" || mode === "done" ? (
        <div className="flex flex-col items-center gap-4">
          <div className="flex gap-3">
            <Button onClick={startWebcam} className="gap-2 bg-blue-600 hover:bg-blue-700">
              <Video className="h-4 w-4" /> Use Webcam
            </Button>
            <Button
              variant="outline"
              className="gap-2"
              disabled={uploading}
              onClick={() => fileInputRef.current?.click()}
            >
              <Upload className="h-4 w-4" /> {uploading ? "Searching…" : "Upload Photo"}
            </Button>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="sr-only"
              onChange={handleFileUpload}
            />
          </div>
          {error && <p className="text-red-400 text-sm text-center">{error}</p>}
        </div>
      ) : (
        /* Webcam feed */
        <div className="space-y-2">
          <div className="relative rounded-xl overflow-hidden bg-zinc-900 mx-auto" style={{ maxWidth: 560 }}>
            <video
              ref={videoRef}
              className="w-full block"
              playsInline muted
              style={{ transform: "scaleX(-1)" }}  // mirror
            />
            <canvas
              ref={overlayRef}
              className="absolute inset-0 w-full h-full"
              style={{ transform: "scaleX(-1)" }}
            />
            {/* Quality indicator overlay */}
            <div className="absolute top-3 left-3 flex items-center gap-2 bg-black/70 rounded-full px-3 py-1.5">
              <div
                className="h-2.5 w-2.5 rounded-full"
                style={{ background: QUAL_COLOR[quality], boxShadow: `0 0 8px ${QUAL_COLOR[quality]}` }}
              />
              <span className="text-xs font-mono font-semibold" style={{ color: QUAL_COLOR[quality] }}>
                {quality.toUpperCase()}
              </span>
            </div>
            {/* Progress bar */}
            {quality === "good" && (
              <div className="absolute bottom-0 left-0 right-0 h-1 bg-black/50">
                <div
                  className="h-full bg-green-500 transition-none"
                  style={{ width: `${goodProgress}%` }}
                />
              </div>
            )}
            <Button
              size="icon"
              variant="ghost"
              className="absolute top-2 right-2 h-7 w-7 bg-black/50 hover:bg-black/80"
              onClick={stopWebcam}
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
          <p className="text-center text-xs text-zinc-500">
            {quality === "good"
              ? `Hold still — capturing in ${(1 - goodProgress / 100).toFixed(1)}s`
              : quality === "fair"
              ? "Move closer and centre your face"
              : "Position your face in the frame"}
          </p>
        </div>
      )}

      {/* Results */}
      {results && results.length > 0 && (
        <div className="space-y-2">
          <p className="text-xs text-zinc-500 text-center">Top {results.length} matches</p>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-3">
            {results.slice(0, 5).map(hit => (
              <button
                key={hit.person_id}
                onClick={() => onSelect(hit)}
                className="flex flex-col items-center gap-2 p-3 rounded-xl border border-zinc-800 bg-zinc-900 hover:bg-zinc-800 hover:border-blue-700/50 transition-all text-center group"
              >
                {hit.thumbnail
                  ? <img src={hit.thumbnail} alt={hit.name} className="h-16 w-16 rounded-full object-cover ring-2 ring-zinc-700 group-hover:ring-blue-600" />
                  : <div className="h-16 w-16 rounded-full bg-zinc-800 flex items-center justify-center ring-2 ring-zinc-700">
                      <User className="h-7 w-7 text-zinc-600" />
                    </div>
                }
                <div>
                  <p className="text-xs font-semibold text-zinc-200 truncate">{hit.name}</p>
                  <p className="text-[10px] text-zinc-500">{hit.external_id}</p>
                </div>
                {/* Match score bar */}
                <div className="w-full space-y-1">
                  <div className="h-1 bg-zinc-800 rounded-full overflow-hidden">
                    <div
                      className={cn("h-full rounded-full transition-all",
                        hit.match_score > 0.8 ? "bg-green-500" :
                        hit.match_score > 0.6 ? "bg-amber-500" : "bg-red-500"
                      )}
                      style={{ width: `${hit.match_score * 100}%` }}
                    />
                  </div>
                  <p className={cn("text-[10px] font-bold",
                    hit.match_score > 0.8 ? "text-green-400" :
                    hit.match_score > 0.6 ? "text-amber-400" : "text-red-400"
                  )}>
                    {(hit.match_score * 100).toFixed(1)}%
                  </p>
                </div>
              </button>
            ))}
          </div>
        </div>
      )}
      {results && results.length === 0 && (
        <p className="text-center text-zinc-500 text-sm">No match found</p>
      )}
    </div>
  );
}

// ── TextSearch component ──────────────────────────────────────────────────────

function TextSearch({ onSelect }: { onSelect: (h: SearchHit) => void }) {
  const [query,  setQuery]   = useState("");
  const [dbq,    setDbq]     = useState("");
  const [open,   setOpen]    = useState(false);
  const [cursor, setCursor]  = useState(-1);
  const inputRef             = useRef<HTMLInputElement>(null);

  // Debounce 300ms
  useEffect(() => {
    const id = setTimeout(() => setDbq(query), 300);
    return () => clearTimeout(id);
  }, [query]);

  const { data: hits, isFetching } = useQuery({
    queryKey:  ["search-text", dbq],
    queryFn:   () => dbq.length >= 2
      ? searchApi.person(dbq, { limit: 8 }).then(r => { const d = r.data as { items?: SearchHit[]; results?: SearchHit[] }; return d.items ?? d.results ?? []; })
      : Promise.resolve([]),
    enabled:   dbq.length >= 2,
    staleTime: 5_000,
  });

  const results = hits ?? [];

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (!open || results.length === 0) return;
    if (e.key === "ArrowDown") { e.preventDefault(); setCursor(c => Math.min(c + 1, results.length - 1)); }
    if (e.key === "ArrowUp")   { e.preventDefault(); setCursor(c => Math.max(c - 1, -1)); }
    if (e.key === "Enter" && cursor >= 0) { onSelect(results[cursor]); setOpen(false); setQuery(""); }
    if (e.key === "Escape") setOpen(false);
  };

  return (
    <div className="relative">
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-zinc-500" />
        <Input
          ref={inputRef}
          value={query}
          onChange={e => { setQuery(e.target.value); setOpen(true); setCursor(-1); }}
          onFocus={() => query.length >= 2 && setOpen(true)}
          onBlur={() => setTimeout(() => setOpen(false), 180)}
          onKeyDown={handleKeyDown}
          placeholder="Search by name or ID…"
          className="pl-9 pr-4 h-11 bg-zinc-900 border-zinc-700 text-sm"
        />
        {isFetching && <div className="absolute right-3 top-1/2 -translate-y-1/2 h-4 w-4 rounded-full border-2 border-zinc-600 border-t-blue-500 animate-spin" />}
      </div>

      {open && results.length > 0 && (
        <div className="absolute z-40 top-full left-0 right-0 mt-1 bg-zinc-900 border border-zinc-700 rounded-xl shadow-2xl overflow-hidden">
          {results.map((hit, i) => (
            <button
              key={hit.person_id}
              onMouseDown={() => { onSelect(hit); setQuery(""); setOpen(false); }}
              className={cn(
                "w-full flex items-center gap-3 px-3 py-2.5 hover:bg-zinc-800 text-left transition-colors",
                cursor === i && "bg-zinc-800",
              )}
            >
              {hit.thumbnail
                ? <img src={hit.thumbnail} alt="" className="h-8 w-8 rounded-full object-cover shrink-0" />
                : <div className="h-8 w-8 rounded-full bg-zinc-700 flex items-center justify-center shrink-0">
                    <User className="h-4 w-4 text-zinc-500" />
                  </div>
              }
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-zinc-200 truncate">{hit.name}</p>
                <p className="text-xs text-zinc-500 truncate">{hit.external_id} · {hit.department}</p>
              </div>
              <span className={cn("text-[10px] font-bold px-1.5 py-0.5 rounded shrink-0", ROLE_COLOR[hit.role])}>
                {hit.role}
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Journey Timeline ──────────────────────────────────────────────────────────

function JourneyTimeline({
  events,
  onEventClick,
  selectedEventId,
}: {
  events:          ExtJourneyEvent[];
  onEventClick:    (ev: ExtJourneyEvent) => void;
  selectedEventId: string | null;
}) {
  const [tooltip, setTooltip] = useState<{ ev: ExtJourneyEvent; x: number; y: number } | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const sorted = useMemo(() =>
    [...events].sort((a, b) => a.first_seen.localeCompare(b.first_seen)),
    [events]
  );

  const { minMs, maxMs, ppm, svgW } = useMemo(() => {
    if (sorted.length === 0) return { minMs: 0, maxMs: 0, ppm: 2, svgW: 600 };
    const mi = new Date(sorted[0].first_seen).getTime();
    const ma = new Date(sorted[sorted.length - 1].last_seen).getTime();
    const totalMin = Math.max((ma - mi) / 60000, 30);
    const p = totalMin < 120 ? 5 : totalMin < 360 ? 3.5 : totalMin < 720 ? 2.2 : 1.4;
    return { minMs: mi, maxMs: ma, ppm: p, svgW: Math.max(totalMin * p, 600) };
  }, [sorted]);

  const toX   = (iso: string) => ((new Date(iso).getTime() - minMs) / 60000) * ppm;
  const toW   = (ev: ExtJourneyEvent) => Math.max((ev.duration_s / 60) * ppm, 6);
  const SVG_H = 88;
  const BLK_Y = 26, BLK_H = 36;

  // Hour markers every 30 min
  const markers = useMemo(() => {
    const marks: { x: number; label: string }[] = [];
    const step = 30 * 60000;
    const start = Math.floor(minMs / step) * step;
    for (let t = start; t <= maxMs + step; t += step) {
      const x = ((t - minMs) / 60000) * ppm;
      if (x >= -30 && x <= svgW + 30) marks.push({ x, label: format(new Date(t), "HH:mm") });
    }
    return marks;
  }, [minMs, maxMs, ppm, svgW]);

  if (sorted.length === 0) return (
    <div className="flex items-center justify-center h-24 text-zinc-600 text-sm">
      No journey data for this date range
    </div>
  );

  return (
    <div className="relative">
      <div ref={containerRef} className="overflow-x-auto rounded-lg border border-zinc-800 bg-zinc-950">
        <svg
          viewBox={`0 0 ${svgW} ${SVG_H}`}
          style={{ width: svgW, height: SVG_H, display: "block" }}
        >
          {/* Grid lines */}
          {markers.map(({ x }, i) => (
            <line key={i} x1={x} y1={16} x2={x} y2={SVG_H - 4} stroke="#27272a" strokeWidth={0.5} />
          ))}

          {/* Hour marker labels */}
          {markers.map(({ x, label }, i) => (
            <text key={i} x={x + 2} y={13} fill="#52525b" fontSize={8} fontFamily="monospace">{label}</text>
          ))}

          {/* Transit arrows between events */}
          {sorted.map((ev, i) => {
            if (i === 0 || !ev.transit_time) return null;
            const prev = sorted[i - 1];
            const x1   = toX(prev.last_seen);
            const x2   = toX(ev.first_seen);
            const gap  = x2 - x1;
            if (gap < 4) return null;
            const midX = x1 + gap / 2;
            const midY = BLK_Y + BLK_H / 2;
            return (
              <g key={`transit-${i}`}>
                <line x1={x1} y1={midY} x2={x2} y2={midY} stroke="#3f3f46" strokeWidth={1} strokeDasharray="3,3" />
                <polygon points={`${x2},${midY} ${x2 - 5},${midY - 3} ${x2 - 5},${midY + 3}`} fill="#3f3f46" />
                {gap > 40 && (
                  <text x={midX} y={midY - 4} fill="#52525b" fontSize={7} textAnchor="middle" fontFamily="monospace">
                    {ev.transit_time < 60
                      ? `${ev.transit_time}s`
                      : `${Math.round(ev.transit_time / 60)}m`}
                  </text>
                )}
              </g>
            );
          })}

          {/* Event blocks */}
          {sorted.map(ev => {
            const x   = toX(ev.first_seen);
            const w   = toW(ev);
            const col = EVT_COLOR[ev.type];
            const isSel = ev.source_id === selectedEventId;
            return (
              <g
                key={ev.source_id}
                style={{ cursor: "pointer" }}
                onClick={() => onEventClick(ev)}
                onMouseEnter={e => {
                  const rect = containerRef.current?.getBoundingClientRect();
                  if (rect) setTooltip({ ev, x: e.clientX - rect.left, y: e.clientY - rect.top - 10 });
                }}
                onMouseLeave={() => setTooltip(null)}
              >
                <rect
                  x={x} y={BLK_Y}
                  width={w} height={BLK_H}
                  rx={4}
                  fill={col.fill}
                  stroke={isSel ? "#fff" : col.stroke}
                  strokeWidth={isSel ? 1.5 : 0.75}
                  opacity={isSel ? 1 : 0.85}
                />
                {w > 30 && (
                  <text
                    x={x + 4} y={BLK_Y + 12}
                    fill="#fff" fontSize={7.5}
                    fontFamily="monospace" fontWeight="600"
                    style={{ pointerEvents: "none" }}
                  >
                    {(ev.area ?? ev.camera_name ?? "").slice(0, Math.floor(w / 7))}
                  </text>
                )}
                {ev.status && w > 18 && (
                  <text
                    x={x + 4} y={BLK_Y + 24}
                    fill="rgba(255,255,255,0.7)" fontSize={7}
                    fontFamily="monospace"
                    style={{ pointerEvents: "none" }}
                  >
                    {ev.status}
                  </text>
                )}
              </g>
            );
          })}
        </svg>
      </div>

      {/* Tooltip */}
      {tooltip && (
        <div
          className="absolute z-50 pointer-events-none bg-zinc-800 border border-zinc-700 rounded-lg p-2.5 text-xs shadow-xl min-w-[160px]"
          style={{ left: Math.min(tooltip.x, (containerRef.current?.offsetWidth ?? 400) - 180), top: tooltip.y - 80 }}
        >
          <p className="font-semibold text-zinc-200 mb-1">{tooltip.ev.area ?? tooltip.ev.camera_name}</p>
          <p className="text-zinc-400">
            {format(parseISO(tooltip.ev.first_seen), "MMM d, HH:mm:ss")}
            {tooltip.ev.duration_s > 0 && <> – {format(parseISO(tooltip.ev.last_seen), "MMM d, HH:mm:ss")}</>}
          </p>
          {tooltip.ev.duration_s > 0 && <p className="text-zinc-400">{(() => { const s = Math.round(tooltip.ev.duration_s); if (s < 60) return `${s}s`; if (s < 3600) return `${Math.floor(s/60)}m ${s%60}s`; return `${Math.floor(s/3600)}h ${Math.floor((s%3600)/60)}m`; })()}</p>}
          {tooltip.ev.course_name && <p className="text-blue-400 mt-0.5">{tooltip.ev.course_name}</p>}
          {tooltip.ev.status && <span className={cn("inline-block px-1.5 py-0.5 rounded text-[10px] mt-1", STATUS_COLOR[tooltip.ev.status] ?? "")}>{tooltip.ev.status}</span>}
          <p className="text-[10px] text-zinc-600 mt-1">Click for evidence</p>
        </div>
      )}
    </div>
  );
}

// ── Journey Table ─────────────────────────────────────────────────────────────

function fmtDur(s: number) {
  const r = Math.round(s);
  if (r < 60)   return `${r}s`;
  if (r < 3600) return `${Math.floor(r/60)}m ${r%60}s`;
  return `${Math.floor(r/3600)}h ${Math.floor((r%3600)/60)}m`;
}

function JourneyTable({
  events,
  onEventClick,
  selectedId,
}: {
  events:       ExtJourneyEvent[];
  onEventClick: (ev: ExtJourneyEvent) => void;
  selectedId:   string | null;
}) {
  const [expandedDays, setExpandedDays] = useState<Set<string>>(new Set());

  // Group events by UTC calendar date (YYYY-MM-DD of first_seen)
  const groups = useMemo(() => {
    const sorted = [...events].sort((a, b) => a.first_seen.localeCompare(b.first_seen));
    const map = new Map<string, ExtJourneyEvent[]>();
    for (const ev of sorted) {
      const day = ev.first_seen.slice(0, 10); // "YYYY-MM-DD"
      if (!map.has(day)) map.set(day, []);
      map.get(day)!.push(ev);
    }
    // For each group compute day-level in/out/duration
    return Array.from(map.entries()).map(([day, evs]) => {
      const inTime  = evs[0].first_seen;                     // earliest first_seen
      // last out = latest last_seen among events that have a real out (duration > 0)
      const withOut = evs.filter(e => e.duration_s > 0);
      const outTime = withOut.length > 0
        ? withOut.reduce((best, e) => e.last_seen > best ? e.last_seen : best, withOut[0].last_seen)
        : null;
      const durS    = outTime ? (new Date(outTime).getTime() - new Date(inTime).getTime()) / 1000 : 0;
      return { day, evs, inTime, outTime, durS };
    });
  }, [events]);

  const toggleDay = (day: string) =>
    setExpandedDays(prev => {
      const n = new Set(prev);
      n.has(day) ? n.delete(day) : n.add(day);
      return n;
    });

  return (
    <div className="overflow-x-auto rounded-lg border border-zinc-800">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-zinc-800 bg-zinc-900/60">
            {["Date", "In", "Out", "Duration", "Locations", ""].map(h => (
              <th key={h} className="px-3 py-2 text-left font-semibold text-zinc-500 whitespace-nowrap">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {groups.length === 0 && (
            <tr><td colSpan={6} className="px-3 py-6 text-center text-zinc-600">No records</td></tr>
          )}
          {groups.map(({ day, evs, inTime, outTime, durS }) => {
            const expanded = expandedDays.has(day);
            const locations = Array.from(new Set(evs.map(e => e.area ?? e.camera_name).filter(Boolean)));
            return (
              <Fragment key={day}>
                {/* ── Day summary row ── */}
                <tr
                  className="border-b border-zinc-800/50 hover:bg-zinc-900/40 cursor-pointer transition-colors bg-zinc-900/20"
                  onClick={() => toggleDay(day)}
                >
                  <td className="px-3 py-2.5 font-semibold text-zinc-300 whitespace-nowrap">
                    <div className="flex items-center gap-1.5">
                      <ChevronRight className={cn("h-3.5 w-3.5 text-zinc-500 transition-transform", expanded && "rotate-90")} />
                      {format(parseISO(day), "EEE, MMM d yyyy")}
                    </div>
                  </td>
                  <td className="px-3 py-2.5 font-mono text-green-400 whitespace-nowrap">
                    {format(parseISO(inTime), "HH:mm:ss")}
                  </td>
                  <td className="px-3 py-2.5 font-mono text-red-400 whitespace-nowrap">
                    {outTime ? format(parseISO(outTime), "HH:mm:ss") : <span className="text-zinc-600">—</span>}
                  </td>
                  <td className="px-3 py-2.5 font-mono text-zinc-300">
                    {durS > 0 ? fmtDur(durS) : <span className="text-zinc-600">—</span>}
                  </td>
                  <td className="px-3 py-2.5 text-zinc-400 max-w-[220px]">
                    <span className="truncate block">{locations.join(", ") || "—"}</span>
                  </td>
                  <td className="px-3 py-2.5 text-zinc-600 text-[10px]">{evs.length} event{evs.length !== 1 ? "s" : ""}</td>
                </tr>

                {/* ── Individual events (expanded) ── */}
                {expanded && evs.map(ev => {
                  const isSel = ev.source_id === selectedId;
                  const col   = EVT_COLOR[ev.type];
                  return (
                    <tr
                      key={ev.source_id}
                      className={cn(
                        "border-b border-zinc-800/30 hover:bg-zinc-900/50 cursor-pointer transition-colors",
                        isSel && "bg-zinc-900",
                      )}
                      onClick={() => onEventClick(ev)}
                    >
                      <td className="px-3 py-2 pl-8 text-zinc-500 whitespace-nowrap">
                        <span className="text-[10px]">{ev.area ?? ev.camera_name ?? "—"}</span>
                      </td>
                      <td className="px-3 py-2 font-mono text-zinc-400 whitespace-nowrap">
                        {format(parseISO(ev.first_seen), "HH:mm:ss")}
                      </td>
                      <td className="px-3 py-2 font-mono text-zinc-500 whitespace-nowrap">
                        {ev.duration_s > 0 ? format(parseISO(ev.last_seen), "HH:mm:ss") : <span className="text-zinc-600">—</span>}
                      </td>
                      <td className="px-3 py-2 font-mono text-zinc-500">
                        {ev.duration_s > 0 ? fmtDur(ev.duration_s) : <span className="text-zinc-600">—</span>}
                      </td>
                      <td className="px-3 py-2 text-zinc-500">
                        <div className="flex items-center gap-2 flex-wrap">
                          <span
                            className="text-[10px] font-semibold px-1.5 py-0.5 rounded"
                            style={{ color: col.stroke, background: col.fill.replace("0.75", "0.2") }}
                          >
                            {ev.type === "ATTENDANCE" ? "ATTEND" : "SIGHT"}
                          </span>
                          {ev.course_name && <span className="text-zinc-400 truncate max-w-[100px]">{ev.course_name}</span>}
                          {ev.status && (
                            <span className={cn("px-1.5 py-0.5 rounded text-[10px] font-bold", STATUS_COLOR[ev.status] ?? "text-zinc-500")}>{ev.status}</span>
                          )}
                          {ev.confidence != null && <span className="text-zinc-600">{(ev.confidence * 100).toFixed(0)}%</span>}
                        </div>
                      </td>
                      <td className="px-3 py-2">
                        <Eye className="h-3.5 w-3.5 text-zinc-600 hover:text-blue-400" />
                      </td>
                    </tr>
                  );
                })}
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ── Evidence Gallery + Lightbox ───────────────────────────────────────────────

function EvidenceGallery({
  frames,
  loading,
}: {
  frames:  EvidenceFrame[];
  loading: boolean;
}) {
  const [lbIdx, setLbIdx] = useState<number | null>(null);

  if (loading) return (
    <div className="flex items-center justify-center h-48 text-zinc-600 text-sm gap-2">
      <div className="h-4 w-4 rounded-full border-2 border-zinc-700 border-t-blue-500 animate-spin" />
      Loading evidence…
    </div>
  );

  if (frames.length === 0) return (
    <div className="flex flex-col items-center justify-center h-48 text-zinc-600 text-sm gap-2">
      <ImageOff className="h-8 w-8" />
      <span>Select a journey event to view evidence</span>
    </div>
  );

  return (
    <>
      <div className="grid grid-cols-3 gap-2">
        {frames.map((f, i) => (
          <button
            key={i}
            onClick={() => setLbIdx(i)}
            className="relative group rounded-lg overflow-hidden bg-zinc-900 border border-zinc-800 hover:border-blue-700 transition-colors aspect-square"
          >
            <img src={f.url} alt="" className="w-full h-full object-cover" />
            <div className="absolute inset-0 bg-black/0 group-hover:bg-black/30 transition-all flex items-center justify-center">
              <ZoomIn className="h-5 w-5 text-white opacity-0 group-hover:opacity-100 transition-opacity" />
            </div>
            <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent px-1.5 py-1">
              <p className="text-[9px] text-zinc-300 font-mono truncate">
                {format(parseISO(f.timestamp), "HH:mm:ss")}
              </p>
              <p className="text-[9px] text-green-400 font-mono">
                {(f.confidence * 100).toFixed(0)}%
              </p>
            </div>
          </button>
        ))}
      </div>

      {/* Lightbox */}
      {lbIdx !== null && (
        <div className="fixed inset-0 z-[100] bg-black/95 flex items-center justify-center" onClick={() => setLbIdx(null)}>
          <div className="relative max-w-2xl w-full mx-4" onClick={e => e.stopPropagation()}>
            <img src={frames[lbIdx].url} alt="" className="w-full rounded-xl object-contain max-h-[80vh]" />
            <div className="mt-3 flex items-center justify-between text-sm text-zinc-400">
              <span>{frames[lbIdx].location}</span>
              <span className="font-mono">{fmtDatetime(frames[lbIdx].timestamp)}</span>
              <span className="text-green-400 font-mono">{(frames[lbIdx].confidence * 100).toFixed(1)}% conf</span>
            </div>
            {/* Arrows */}
            {lbIdx > 0 && (
              <button className="absolute -left-14 top-1/2 -translate-y-1/2 h-10 w-10 rounded-full bg-zinc-800 flex items-center justify-center hover:bg-zinc-700" onClick={() => setLbIdx(i => i! - 1)}>
                <ChevronLeft className="h-5 w-5" />
              </button>
            )}
            {lbIdx < frames.length - 1 && (
              <button className="absolute -right-14 top-1/2 -translate-y-1/2 h-10 w-10 rounded-full bg-zinc-800 flex items-center justify-center hover:bg-zinc-700" onClick={() => setLbIdx(i => i! + 1)}>
                <ChevronRight className="h-5 w-5" />
              </button>
            )}
            <button className="absolute -top-12 right-0 h-9 w-9 rounded-full bg-zinc-800 flex items-center justify-center" onClick={() => setLbIdx(null)}>
              <X className="h-4 w-4" />
            </button>
            <p className="text-center text-xs text-zinc-600 mt-2">{lbIdx + 1} / {frames.length}</p>
          </div>
        </div>
      )}
    </>
  );
}

// ── Presence Heatmap ──────────────────────────────────────────────────────────

function PresenceHeatmap({ data, loading }: { data: HeatmapResponse | null; loading: boolean }) {
  if (loading) return (
    <div className="flex items-center justify-center h-40 text-zinc-600 text-sm">
      <div className="h-4 w-4 rounded-full border-2 border-zinc-700 border-t-blue-500 animate-spin mr-2" />
      Loading heatmap…
    </div>
  );
  if (!data || data.areas.length === 0) return (
    <div className="flex items-center justify-center h-40 text-zinc-600 text-sm">No heatmap data</div>
  );

  const maxVal = Math.max(...data.matrix.flat(), 1);
  const HOURS  = Array.from({ length: 24 }, (_, i) => i);

  return (
    <div className="overflow-x-auto">
      <div style={{ minWidth: 520 }}>
        {/* Hour header */}
        <div className="flex items-center mb-1">
          <div className="w-28 shrink-0" />
          {HOURS.map(h => (
            <div key={h} className="flex-1 text-center text-[9px] text-zinc-600 font-mono">
              {h % 3 === 0 ? `${String(h).padStart(2, "0")}h` : ""}
            </div>
          ))}
        </div>
        {/* Rows */}
        {data.areas.map((area, ai) => (
          <div key={area} className="flex items-center gap-0 mb-0.5">
            <div className="w-28 shrink-0 text-[10px] text-zinc-400 truncate pr-2 text-right">{area}</div>
            {HOURS.map(h => {
              const v    = data.matrix[ai]?.[h] ?? 0;
              const pct  = v / maxVal;
              const bg   = pct === 0 ? "bg-zinc-900"
                : pct < 0.15 ? "bg-blue-950"
                : pct < 0.35 ? "bg-blue-900"
                : pct < 0.6  ? "bg-blue-700"
                : pct < 0.85 ? "bg-blue-500"
                : "bg-blue-400";
              return (
                <div
                  key={h}
                  title={v > 0 ? `${v}min @ ${String(h).padStart(2,"0")}:00` : undefined}
                  className={cn("flex-1 h-5 border-r border-b border-zinc-900/50 transition-colors", bg)}
                />
              );
            })}
          </div>
        ))}
        {/* Legend */}
        <div className="flex items-center gap-1.5 mt-2 ml-28 pl-1">
          {["bg-zinc-900","bg-blue-950","bg-blue-900","bg-blue-700","bg-blue-500","bg-blue-400"].map((c,i) => (
            <div key={i} className={cn("h-3 w-4 rounded-sm", c)} />
          ))}
          <span className="text-[9px] text-zinc-600 ml-1">0 → {maxVal} min</span>
        </div>
      </div>
    </div>
  );
}

// ── Movement Map ──────────────────────────────────────────────────────────────

function MovementMap({ trail, loading }: { trail: CrossCameraEntry[]; loading: boolean }) {
  if (loading) return (
    <div className="flex items-center justify-center h-56 text-zinc-600 text-sm">
      <div className="h-4 w-4 rounded-full border-2 border-zinc-700 border-t-blue-500 animate-spin mr-2" />
      Loading movement…
    </div>
  );

  // Build graph
  const nodeMap   = new Map<string, { id: string; name: string; visits: number }>();
  const edgeCount = new Map<string, number>();

  for (let i = 0; i < trail.length; i++) {
    const t = trail[i];
    if (!nodeMap.has(t.camera_id)) nodeMap.set(t.camera_id, { id: t.camera_id, name: t.camera_name ?? t.area ?? t.camera_id.slice(0, 8), visits: 0 });
    nodeMap.get(t.camera_id)!.visits++;
    if (i > 0) {
      const key = `${trail[i-1].camera_id}→${t.camera_id}`;
      edgeCount.set(key, (edgeCount.get(key) ?? 0) + 1);
    }
  }

  const nodes  = Array.from(nodeMap.values());
  const edges  = Array.from(edgeCount.entries()).map(([k, count]) => {
    const [from, to] = k.split("→");
    return { from, to, count };
  });

  if (nodes.length === 0) return (
    <div className="flex items-center justify-center h-56 text-zinc-600 text-sm">No movement data</div>
  );

  const SVG_W = 320, SVG_H = 220;
  const cx = SVG_W / 2, cy = SVG_H / 2;
  const R  = Math.min(SVG_W, SVG_H) / 2 - 32;

  const positioned = nodes.map((n, i) => ({
    ...n,
    x: nodes.length === 1 ? cx : cx + R * Math.cos((2 * Math.PI * i / nodes.length) - Math.PI / 2),
    y: nodes.length === 1 ? cy : cy + R * Math.sin((2 * Math.PI * i / nodes.length) - Math.PI / 2),
  }));
  const posMap = new Map(positioned.map(n => [n.id, n]));
  const maxEdge = Math.max(...edges.map(e => e.count), 1);
  const maxVisit = Math.max(...nodes.map(n => n.visits), 1);

  return (
    <svg viewBox={`0 0 ${SVG_W} ${SVG_H}`} className="w-full" style={{ height: SVG_H }}>
      {/* Edges */}
      {edges.map(({ from, to, count }, i) => {
        const a = posMap.get(from), b = posMap.get(to);
        if (!a || !b) return null;
        const t      = count / maxEdge;
        const midX   = (a.x + b.x) / 2 + (b.y - a.y) * 0.25;
        const midY   = (a.y + b.y) / 2 + (a.x - b.x) * 0.25;
        const opacity = 0.2 + t * 0.65;
        const sw     = 0.8 + t * 3;
        return (
          <g key={i}>
            <path
              d={`M${a.x},${a.y} Q${midX},${midY} ${b.x},${b.y}`}
              fill="none"
              stroke={`rgba(59,130,246,${opacity})`}
              strokeWidth={sw}
            />
            {/* Arrowhead */}
            <circle cx={b.x} cy={b.y} r={3.5} fill={`rgba(59,130,246,${opacity})`} />
          </g>
        );
      })}
      {/* Nodes */}
      {positioned.map(n => {
        const t   = n.visits / maxVisit;
        const r   = 8 + t * 10;
        const col = "#3b82f6";
        return (
          <g key={n.id}>
            <circle cx={n.x} cy={n.y} r={r + 4} fill={`rgba(59,130,246,0.08)`} />
            <circle cx={n.x} cy={n.y} r={r} fill={`rgba(59,130,246,0.25)`} stroke={col} strokeWidth={1.5} />
            <text x={n.x} y={n.y + 1} textAnchor="middle" dominantBaseline="middle" fill="#93c5fd" fontSize={8.5} fontFamily="monospace" fontWeight="600">
              {n.visits}
            </text>
            <text x={n.x} y={n.y + r + 10} textAnchor="middle" fill="#71717a" fontSize={7.5} fontFamily="monospace">
              {n.name.slice(0, 14)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// ── Person Header ─────────────────────────────────────────────────────────────

function PersonHeader({
  person,
  dateFrom,
  dateTo,
  onDateFrom,
  onDateTo,
  onClear,
}: {
  person:     SearchHit;
  dateFrom:   string;
  dateTo:     string;
  onDateFrom: (v: string) => void;
  onDateTo:   (v: string) => void;
  onClear:    () => void;
}) {
  const isLive = person.last_seen
    ? (Date.now() - new Date(person.last_seen).getTime()) < 5 * 60 * 1000
    : false;

  return (
    <div className="flex items-center gap-4 p-4 border-b border-zinc-800 bg-zinc-950 shrink-0">
      {/* Back */}
      <button onClick={onClear} className="h-8 w-8 rounded-full bg-zinc-900 hover:bg-zinc-800 flex items-center justify-center shrink-0 transition-colors">
        <ChevronLeft className="h-4 w-4 text-zinc-400" />
      </button>

      {/* Photo */}
      <div className="relative shrink-0">
        {person.thumbnail
          ? <img src={person.thumbnail} alt={person.name} className="h-14 w-14 rounded-full object-cover ring-2 ring-zinc-700" />
          : <div className="h-14 w-14 rounded-full bg-zinc-800 flex items-center justify-center ring-2 ring-zinc-700">
              <User className="h-7 w-7 text-zinc-600" />
            </div>
        }
        {isLive && (
          <span className="absolute -bottom-0.5 -right-0.5 h-4 w-4 rounded-full bg-green-500 border-2 border-zinc-950 animate-pulse" />
        )}
      </div>

      {/* Info */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <h2 className="text-base font-bold text-zinc-100 truncate">{person.name}</h2>
          {isLive && (
            <span className="text-[9px] font-bold text-green-400 border border-green-700 bg-green-950 px-1.5 py-0.5 rounded animate-pulse">
              LIVE
            </span>
          )}
        </div>
        <div className="flex items-center gap-2 mt-0.5 flex-wrap">
          <span className="text-xs text-zinc-500 font-mono">{person.external_id}</span>
          <span className={cn("text-[10px] font-bold px-1.5 py-0.5 rounded", ROLE_COLOR[person.role])}>{person.role}</span>
          {person.department && (
            <span className="text-xs text-zinc-500">{person.department}</span>
          )}
          {person.last_seen && (
            <span className="text-xs text-zinc-600 flex items-center gap-1">
              <Clock className="h-3 w-3" />
              Last seen {fmtDatetime(person.last_seen)}
            </span>
          )}
        </div>
      </div>

      {/* Date range */}
      <div className="flex items-center gap-2 shrink-0">
        <Calendar className="h-4 w-4 text-zinc-500" />
        <input
          type="date"
          value={dateFrom}
          onChange={e => onDateFrom(e.target.value)}
          className="bg-zinc-900 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-300 h-8"
        />
        <ArrowRight className="h-3 w-3 text-zinc-600" />
        <input
          type="date"
          value={dateTo}
          onChange={e => onDateTo(e.target.value)}
          className="bg-zinc-900 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-300 h-8"
        />
      </div>
    </div>
  );
}

// ── Section card wrapper ──────────────────────────────────────────────────────

function Section({ title, icon: Icon, children }: {
  title: string;
  icon:  React.ComponentType<{ className?: string }>;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-950 overflow-hidden">
      <div className="flex items-center gap-2 px-4 py-2.5 border-b border-zinc-800 bg-zinc-900/50">
        <Icon className="h-4 w-4 text-zinc-500" />
        <h3 className="text-sm font-semibold text-zinc-300">{title}</h3>
      </div>
      <div className="p-4">{children}</div>
    </div>
  );
}

// ── Person Detail view ────────────────────────────────────────────────────────

function PersonDetail({ person, onClear }: { person: SearchHit; onClear: () => void }) {
  const today  = format(new Date(), "yyyy-MM-dd");
  const [dateFrom,      setDateFrom]      = useState(format(subDays(new Date(), 7), "yyyy-MM-dd"));
  const [dateTo,        setDateTo]        = useState(today);
  const [selectedEvt,   setSelectedEvt]   = useState<ExtJourneyEvent | null>(null);
  const [evidenceFrames, setEvidenceFrames] = useState<EvidenceFrame[]>([]);
  const [evidenceLoading, setEvidenceLoading] = useState(false);

  const { data: journeyData, isLoading: journeyLoading } = useQuery({
    queryKey: ["journey", person.person_id, dateFrom, dateTo],
    queryFn:  () => searchApi.journey(person.person_id, { date_from: dateFrom, date_to: dateTo })
                    .then(r => r.data as JourneyResponse),
    staleTime: 30_000,
  });

  const { data: heatmapData, isLoading: heatmapLoading } = useQuery({
    queryKey: ["heatmap", person.person_id, dateFrom, dateTo],
    queryFn:  () => searchApi.heatmap(person.person_id, { date_from: dateFrom, date_to: dateTo })
                    .then(r => r.data as HeatmapResponse),
    staleTime: 30_000,
  });

  const { data: trailData, isLoading: trailLoading } = useQuery({
    queryKey: ["cross-camera", person.person_id, dateTo],
    queryFn:  () => searchApi.crossCamera(person.person_id, { date: dateTo })
                    .then(r => r.data as CrossCameraEntry[]),
    staleTime: 30_000,
  });

  const handleEventClick = useCallback(async (ev: ExtJourneyEvent) => {
    setSelectedEvt(ev);
    setEvidenceLoading(true);
    try {
      if (ev.evidence_refs && ev.evidence_refs.length > 0) {
        setEvidenceFrames(ev.evidence_refs.map((url, i) => ({
          url,
          timestamp: ev.first_seen,
          confidence: ev.confidence ?? 0.9,
          location: ev.area ?? ev.camera_name,
        })));
      } else {
        // Attempt to fetch from sighting/attendance source
        setEvidenceFrames([]);
      }
    } finally {
      setEvidenceLoading(false);
    }
  }, []);

  const events = journeyData?.events ?? [];

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      <PersonHeader
        person={person}
        dateFrom={dateFrom}
        dateTo={dateTo}
        onDateFrom={setDateFrom}
        onDateTo={setDateTo}
        onClear={onClear}
      />

      <div className="flex-1 overflow-y-auto">
        <div className="p-4 space-y-4 max-w-[1600px] mx-auto">
          {/* Summary stats */}
          {journeyData?.summary && (
            <div className="grid grid-cols-3 gap-3">
              {[
                { label: "Locations visited", value: journeyData.summary.total_locations, icon: MapPin },
                { label: "Total time",         value: (() => { const s = Math.round(journeyData.summary.total_duration_s); if (s < 3600) return `${Math.floor(s/60)}m`; return `${Math.floor(s/3600)}h ${Math.floor((s%3600)/60)}m`; })(), icon: Clock },
                { label: "Attendance rate",    value: `${(journeyData.summary.attendance_rate * 100).toFixed(0)}%`, icon: Circle },
              ].map(({ label, value, icon: I }) => (
                <div key={label} className="flex items-center gap-3 p-3 rounded-xl border border-zinc-800 bg-zinc-900/50">
                  <I className="h-5 w-5 text-blue-400 shrink-0" />
                  <div>
                    <p className="text-lg font-bold text-zinc-100">{value}</p>
                    <p className="text-xs text-zinc-500">{label}</p>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Journey Timeline */}
          <Section title="Journey Timeline" icon={Clock}>
            {journeyLoading
              ? <div className="h-20 flex items-center justify-center text-zinc-600 text-sm">
                  <div className="h-4 w-4 rounded-full border-2 border-zinc-700 border-t-blue-500 animate-spin mr-2" />
                  Loading journey…
                </div>
              : <JourneyTimeline
                  events={events}
                  onEventClick={handleEventClick}
                  selectedEventId={selectedEvt?.source_id ?? null}
                />
            }
          </Section>

          {/* Table + Evidence row */}
          <div className="grid grid-cols-1 xl:grid-cols-5 gap-4">
            <div className="xl:col-span-3">
              <Section title="Journey Records" icon={MapPin}>
                <JourneyTable
                  events={events}
                  onEventClick={handleEventClick}
                  selectedId={selectedEvt?.source_id ?? null}
                />
              </Section>
            </div>
            <div className="xl:col-span-2">
              <Section title="Evidence Gallery" icon={Camera}>
                <EvidenceGallery frames={evidenceFrames} loading={evidenceLoading} />
              </Section>
            </div>
          </div>

          {/* Heatmap + Movement row */}
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
            <Section title="Presence Heatmap" icon={Eye}>
              <PresenceHeatmap data={heatmapData ?? null} loading={heatmapLoading} />
            </Section>
            <Section title="Movement Map" icon={ArrowRight}>
              <MovementMap trail={trailData ?? []} loading={trailLoading} />
            </Section>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Main search page ──────────────────────────────────────────────────────────

type SearchTab = "text" | "face";

export default function SearchPage() {
  const [tab,    setTab]    = useState<SearchTab>("text");
  const [person, setPerson] = useState<SearchHit | null>(null);

  const handleSelect = useCallback((h: SearchHit) => {
    setPerson(h);
  }, []);

  const handleClear = useCallback(() => {
    setPerson(null);
  }, []);

  return (
    <div className="flex flex-col h-[calc(100vh-64px)] overflow-hidden bg-zinc-950">
      {/* Search panel — always visible but compact when person selected */}
      {!person && (
        <div className="shrink-0 p-6 border-b border-zinc-800 bg-zinc-950">
          <div className="max-w-2xl mx-auto space-y-5">
            <div className="text-center space-y-1">
              <h1 className="text-xl font-bold text-zinc-100">Person Search</h1>
              <p className="text-sm text-zinc-500">Search by name/ID or use your webcam / an uploaded photo</p>
            </div>

            {/* Tab switcher */}
            <div className="flex rounded-xl border border-zinc-800 p-1 bg-zinc-900 gap-1">
              {(["text", "face"] as const).map(t => (
                <button
                  key={t}
                  onClick={() => setTab(t)}
                  className={cn(
                    "flex-1 flex items-center justify-center gap-2 py-2 rounded-lg text-sm font-medium transition-colors",
                    tab === t ? "bg-zinc-700 text-zinc-100 shadow-sm" : "text-zinc-500 hover:text-zinc-300",
                  )}
                >
                  {t === "text" ? <><Search className="h-4 w-4" /> Name / ID</> : <><Fingerprint className="h-4 w-4" /> Face Search</>}
                </button>
              ))}
            </div>

            {tab === "text"
              ? <TextSearch  onSelect={handleSelect} />
              : <FaceSearch  onSelect={handleSelect} />
            }
          </div>
        </div>
      )}

      {/* Person detail */}
      {person ? (
        <PersonDetail person={person} onClear={handleClear} />
      ) : (
        <div className="flex-1 flex flex-col items-center justify-center text-zinc-700 gap-3">
          <User className="h-12 w-12 opacity-30" />
          <p className="text-sm">Select a person to view their journey analytics</p>
        </div>
      )}
    </div>
  );
}
