"use client";

import { useState, useMemo, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { format, parseISO, isValid } from "date-fns";
import {
  ChevronDown,
  ChevronRight,
  Download,
  RefreshCw,
  Search,
  Calendar,
  Clock,
  Users,
  Eye,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Shield,
  BarChart2,
  FileText,
  Table,
  FileDown,
  Filter,
  X,
  Loader2,
  TrendingUp,
  UserX,
  UserCheck2,
} from "lucide-react";
import { toast } from "sonner";
import { attendanceApi, api, enrollmentApi } from "@/lib/api";
import { useAuth } from "@/hooks/useAuth";
import { cn, fmtDatetime, ATTENDANCE_LABEL, ATTENDANCE_COLOR } from "@/lib/utils";
import type { Session, AttendanceRecord, AttendanceStatus, SyncStatus, PaginatedResponse } from "@/types";
import { Button }   from "@/components/ui/button";
import { Badge }    from "@/components/ui/badge";
import { Input }    from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";

// ── Extended types ────────────────────────────────────────────────────────────

interface ExtAttendanceRecord extends AttendanceRecord {
  liveness_avg?:   number;
  evidence_refs?:  string[];
  thumbnail?:      string | null;
}

interface ExtSession extends Session {
  student_count?: number;
  room_name?:     string;
  building?:      string;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/** HH:MM:SS from ISO datetime */
function fmtTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  try {
    const d = parseISO(iso);
    return isValid(d) ? format(d, "HH:mm:ss") : "—";
  } catch {
    return "—";
  }
}

/** "07 Apr 2026  14:32:08" from ISO datetime */
function fmtTimestamp(iso: string | null | undefined): string {
  if (!iso) return "—";
  try {
    const d = parseISO(iso);
    return isValid(d) ? format(d, "dd MMM yyyy  HH:mm:ss") : "—";
  } catch {
    return "—";
  }
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "";

/** PostgreSQL interval "H:MM:SS" → "Xm Ys" */
function fmtInterval(interval: string | null | undefined): string {
  if (!interval) return "—";
  const m = interval.match(/(\d+):(\d+):(\d+)/);
  if (!m) return interval;
  const h = parseInt(m[1]);
  const mn = parseInt(m[2]);
  const s  = parseInt(m[3]);
  if (h > 0) return `${h}h ${mn.toString().padStart(2, "0")}m`;
  return `${mn.toString().padStart(2, "0")}m ${s.toString().padStart(2, "0")}s`;
}

function todayISO(): string {
  return format(new Date(), "yyyy-MM-dd");
}

// ── Constants ─────────────────────────────────────────────────────────────────

const SYNC_COLOR: Record<SyncStatus, string> = {
  PENDING:   "text-blue-400   bg-blue-900/30",
  SYNCED:    "text-green-400  bg-green-900/30",
  HELD:      "text-amber-400  bg-amber-900/30",
  FAILED:    "text-red-400    bg-red-900/30",
  DISCARDED: "text-zinc-400   bg-zinc-800",
};

const FACULTY_COLOR: Record<string, string> = {
  PRESENT:    "text-green-400",
  LATE:       "text-amber-400",
  ABSENT:     "text-red-400",
  SUBSTITUTE: "text-purple-400",
};

const HELD_PRIORITY: Record<string, { label: string; color: string }> = {
  FACULTY_ABSENT:    { label: "High",   color: "text-red-400   bg-red-900/30"   },
  FACULTY_LATE:      { label: "Medium", color: "text-amber-400 bg-amber-900/30" },
  SUBSTITUTE_FACULTY:{ label: "Medium", color: "text-amber-400 bg-amber-900/30" },
  SYSTEM_HOLD:       { label: "Low",    color: "text-blue-400  bg-blue-900/30"  },
};

const PAGE_SIZE = 25;

// ── Status badges ─────────────────────────────────────────────────────────────

function AttBadge({ status }: { status: AttendanceStatus }) {
  return (
    <span className={cn("text-[10px] font-bold px-1.5 py-0.5 rounded uppercase tabular-nums",
      ATTENDANCE_COLOR[status],
    )}>
      {status}
    </span>
  );
}

function SyncBadge({ status }: { status: SyncStatus }) {
  return (
    <span className={cn("text-[10px] font-semibold px-1.5 py-0.5 rounded uppercase",
      SYNC_COLOR[status],
    )}>
      {status}
    </span>
  );
}

// ── Timeline SVG ──────────────────────────────────────────────────────────────

function SessionTimeline({
  session, records,
}: {
  session: ExtSession;
  records: ExtAttendanceRecord[];
}) {
  const BUCKETS = 24;
  const _tsToMs = (ts: string | null | undefined) => ts ? parseISO(ts).getTime() : null;
  const startMs = _tsToMs(session.actual_start ?? session.scheduled_start) ?? Date.now();
  const endMs   = _tsToMs(session.actual_end   ?? session.scheduled_end)   ?? (startMs + 3600_000);
  const duration = Math.max(endMs - startMs, 1);

  const density = useMemo(() => {
    const d = new Array(BUCKETS).fill(0);
    records.forEach(r => {
      if (!r.first_seen) return;
      const offset = parseISO(r.first_seen as string).getTime() - startMs;
      const bucket = Math.min(BUCKETS - 1, Math.max(0, Math.floor(offset / (duration / BUCKETS))));
      d[bucket]++;
    });
    const max = Math.max(...d, 1);
    return d.map(v => v / max);
  }, [records, startMs, duration]);

  const barW  = 600 / BUCKETS;
  const barH  = 36;
  const barY0 = 18;

  const fStatus = session.faculty_status ?? "";
  const fColor  =
    fStatus === "PRESENT"    ? "#22c55e" :
    fStatus === "SUBSTITUTE" ? "#a855f7" :
    fStatus === "ABSENT"     ? "#52525b" : "#f59e0b";

  const timeMarkers = [0, 0.25, 0.5, 0.75, 1].map(p => {
    const ts = startMs + p * duration;
    let label = "—";
    try { label = isFinite(ts) ? format(new Date(ts), "HH:mm") : "—"; } catch { /* noop */ }
    return { x: p * 600, label };
  });

  return (
    <div className="rounded-lg overflow-hidden border border-zinc-800 bg-zinc-950">
      <div className="flex items-center justify-between px-3 py-1.5 text-[10px] text-zinc-500 border-b border-zinc-800">
        <span className="flex items-center gap-1">
          <span className="h-2.5 w-2.5 rounded-sm" style={{ background: fColor }} />
          Faculty: <span className={cn("font-medium", FACULTY_COLOR[fStatus] ?? "text-zinc-400")}>{fStatus || "Unknown"}</span>
        </span>
        <span className="flex items-center gap-3">
          <span className="flex items-center gap-1">
            <span className="h-2.5 w-2.5 rounded-sm bg-blue-500 opacity-70" />
            Detection density
          </span>
          <span>{records.length} students</span>
        </span>
      </div>
      <svg viewBox="0 0 600 64" className="w-full h-16" preserveAspectRatio="none">
        <rect x={0} y={0} width={600} height={64} fill="#09090b" />
        {/* Faculty band */}
        {fStatus === "LATE" ? (
          <>
            <rect x={0} y={2} width={200} height={12} fill="#52525b" opacity={0.5} />
            <rect x={200} y={2} width={400} height={12} fill="#22c55e" opacity={0.5} />
          </>
        ) : (
          <rect x={0} y={2} width={600} height={12} fill={fColor} opacity={0.4} />
        )}
        {/* Grid lines */}
        {[0.25, 0.5, 0.75].map(p => (
          <line key={p} x1={p * 600} y1={barY0} x2={p * 600} y2={64} stroke="#27272a" strokeWidth={0.5} />
        ))}
        {/* Density bars */}
        {density.map((d, i) => {
          const h = Math.max(d * barH, d > 0 ? 2 : 0);
          return (
            <rect
              key={i}
              x={i * barW + 0.5}
              y={barY0 + barH - h}
              width={barW - 1}
              height={h}
              fill={`rgba(59,130,246,${0.3 + d * 0.7})`}
              rx={1}
            />
          );
        })}
        {/* Time markers */}
        {timeMarkers.map(({ x, label }) => (
          <text key={x} x={Math.min(x + 2, 570)} y={62} fill="#52525b" fontSize={7}>{label}</text>
        ))}
      </svg>
    </div>
  );
}

// ── Evidence dialog ───────────────────────────────────────────────────────────

function EvidenceDialog({
  open,
  onClose,
  evidenceRefs,
  title,
}: {
  open:         boolean;
  onClose:      () => void;
  evidenceRefs: string[];
  title:        string;
}) {
  const [lightbox, setLightbox] = useState<string | null>(null);

  return (
    <Dialog open={open} onOpenChange={open => !open && onClose()}>
      <DialogContent className="max-w-3xl max-h-[85vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Eye className="h-4 w-4 text-zinc-400" />
            Evidence — {title}
          </DialogTitle>
        </DialogHeader>
        {evidenceRefs.length === 0 ? (
          <div className="py-10 text-center text-zinc-600">
            <Eye className="h-8 w-8 mx-auto mb-2 text-zinc-700" />
            No evidence frames recorded for this entry.
          </div>
        ) : (
          <>
            <p className="text-xs text-zinc-500 -mt-2">{evidenceRefs.length} frame{evidenceRefs.length !== 1 ? "s" : ""}</p>
            <div className="overflow-y-auto flex-1">
              <div className="grid grid-cols-4 gap-2 pt-1">
                {evidenceRefs.map((url, i) => (
                  <button
                    key={i}
                    onClick={() => setLightbox(url)}
                    className="aspect-square rounded-lg overflow-hidden bg-zinc-900 border border-zinc-800 hover:border-zinc-600 transition-colors relative group"
                  >
                    <img src={url} alt={`Frame ${i + 1}`} className="w-full h-full object-cover" />
                    <div className="absolute inset-0 bg-black/0 group-hover:bg-black/30 transition-colors flex items-center justify-center">
                      <Eye className="h-5 w-5 text-white opacity-0 group-hover:opacity-100 transition-opacity" />
                    </div>
                    <span className="absolute bottom-1 right-1 text-[9px] text-white/60 bg-black/50 px-1 rounded">
                      #{i + 1}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          </>
        )}
      </DialogContent>

      {/* Lightbox */}
      {lightbox && (
        <Dialog open onOpenChange={() => setLightbox(null)}>
          <DialogContent className="max-w-xl p-2 bg-zinc-950">
            <img src={lightbox} alt="Evidence frame" className="w-full rounded-lg" />
          </DialogContent>
        </Dialog>
      )}
    </Dialog>
  );
}

// ── Override dialog ───────────────────────────────────────────────────────────

function OverrideDialog({
  record,
  onClose,
}: {
  record:  ExtAttendanceRecord | null;
  onClose: () => void;
}) {
  const qc = useQueryClient();
  const [newStatus, setNewStatus] = useState<AttendanceStatus>("P");
  const [reason,    setReason]    = useState("");
  const [saving,    setSaving]    = useState(false);

  const handleSubmit = async () => {
    if (!record || !reason.trim()) return;
    setSaving(true);
    try {
      await attendanceApi.override(record.record_id, { status: newStatus, reason: reason.trim() });
      toast.success(`Override applied — ${record.person_name} set to ${newStatus}`);
      qc.invalidateQueries({ queryKey: ["session-records"] });
      qc.invalidateQueries({ queryKey: ["attendance-records"] });
      onClose();
    } catch {
      toast.error("Override failed");
    } finally {
      setSaving(false);
    }
  };

  if (!record) return null;

  return (
    <Dialog open={!!record} onOpenChange={open => !open && onClose()}>
      <DialogContent className="max-w-sm">
        <DialogHeader>
          <DialogTitle>Override Attendance</DialogTitle>
        </DialogHeader>
        <div className="space-y-4 pt-1">
          <div className="flex items-center gap-3">
            <Avatar className="h-9 w-9">
              <AvatarImage src={record.thumbnail ?? undefined} />
              <AvatarFallback className="bg-zinc-800 text-xs font-semibold text-zinc-300">
                {(record.person_name ?? "?").split(" ").map((n: string) => n[0]).join("").slice(0, 2)}
              </AvatarFallback>
            </Avatar>
            <div>
              <p className="text-sm font-medium text-zinc-200">{record.person_name}</p>
              <p className="text-xs text-zinc-500 font-mono">{record.person_external_id}</p>
            </div>
            <div className="ml-auto">
              <AttBadge status={record.status} />
              <p className="text-[10px] text-zinc-500 text-right mt-0.5">current</p>
            </div>
          </div>

          <div>
            <label className="text-xs text-zinc-400 mb-1.5 block">New status</label>
            <div className="grid grid-cols-3 gap-1.5">
              {(["P", "L", "EE", "A", "ND", "EX"] as AttendanceStatus[]).map(s => (
                <button
                  key={s}
                  onClick={() => setNewStatus(s)}
                  className={cn(
                    "py-1.5 rounded-md text-xs font-semibold border transition-colors",
                    newStatus === s
                      ? cn(ATTENDANCE_COLOR[s], "border-current")
                      : "border-zinc-800 text-zinc-500 hover:border-zinc-600",
                  )}
                >
                  {s} — {ATTENDANCE_LABEL[s]}
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="text-xs text-zinc-400 mb-1 block">
              Reason <span className="text-red-400">*</span>
            </label>
            <textarea
              value={reason}
              onChange={e => setReason(e.target.value)}
              placeholder="Enter reason for override…"
              rows={2}
              className="w-full rounded-md border border-zinc-800 bg-zinc-950 text-sm text-zinc-200 px-3 py-2 placeholder:text-zinc-600 focus:outline-none focus:ring-1 focus:ring-primary resize-none"
            />
          </div>

          <div className="flex gap-2 justify-end">
            <Button variant="outline" size="sm" onClick={onClose}>Cancel</Button>
            <Button
              size="sm"
              disabled={!reason.trim() || saving || newStatus === record.status}
              onClick={handleSubmit}
            >
              {saving ? <><Loader2 className="h-3.5 w-3.5 animate-spin mr-1.5" />Saving…</> : "Apply Override"}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

// ── Student row ───────────────────────────────────────────────────────────────

function StudentRow({
  record,
  canOverride,
  onOverride,
  onEvidence,
}: {
  record:      ExtAttendanceRecord;
  canOverride: boolean;
  onOverride:  (r: ExtAttendanceRecord) => void;
  onEvidence:  (refs: string[], name: string) => void;
}) {
  const enrollUrl    = `${API_BASE}/api/enrollment/thumbnail/${record.person_id}`;
  const evidenceCount = record.evidence_refs?.length ?? 0;

  return (
    <tr className="border-t border-zinc-800/60 hover:bg-zinc-900/40 transition-colors group">
      <td className="px-3 py-2">
        <div className="flex items-center gap-3">
          {/* Enrollment photo */}
          <div className="flex flex-col items-center gap-0.5">
            <div className="relative shrink-0 h-12 w-12 rounded-lg overflow-hidden bg-zinc-800 border border-zinc-700">
              <img
                src={enrollUrl}
                alt={record.person_name ?? ""}
                className="h-full w-full object-cover"
                onError={(e) => { (e.currentTarget as HTMLImageElement).style.display = "none"; }}
              />
              <div className="absolute inset-0 flex items-center justify-center text-sm font-bold text-zinc-400 select-none"
                   style={{ zIndex: -1 }}>
                {(record.person_name ?? "?").split(" ").map((n: string) => n[0]).join("").slice(0, 2).toUpperCase()}
              </div>
            </div>
            <span className="text-[9px] text-zinc-600">enrolled</span>
          </div>
          {/* All captured faces (one per sampled detection) */}
          {evidenceCount > 0 && (
            <div className="flex flex-col gap-0.5">
              <div className="flex gap-1">
                {Array.from({ length: evidenceCount }).map((_, i) => (
                  <div key={i} className="shrink-0 h-12 w-12 rounded-lg overflow-hidden bg-zinc-800 border border-zinc-700">
                    <img
                      src={`${API_BASE}/api/attendance/records/${record.record_id}/evidence?index=${i}`}
                      alt={`Capture ${i + 1}`}
                      className="h-full w-full object-cover"
                      onError={(e) => { (e.currentTarget as HTMLImageElement).style.display = "none"; }}
                    />
                  </div>
                ))}
              </div>
              <span className="text-[9px] text-zinc-600">captured ({evidenceCount})</span>
            </div>
          )}
          <div className="min-w-0">
            <p className="text-xs font-semibold text-zinc-100 truncate max-w-[140px]">{record.person_name}</p>
            <p className="text-[10px] text-zinc-500 font-mono mt-0.5">{record.person_external_id || "—"}</p>
          </div>
        </div>
      </td>
      <td className="px-3 py-2">
        <div className="flex items-center gap-1">
          <AttBadge status={record.status} />
          {record.override_by && (
            <span title={`Overridden: ${record.override_reason}`}>
              <Shield className="h-3 w-3 text-blue-400" />
            </span>
          )}
        </div>
      </td>
      <td className="px-3 py-2 text-xs tabular-nums">
        {(record as any).total_cycles > 0 ? (
          <span className={cn("font-semibold",
            (record as any).cycles_present > 0 ? "text-green-400" : "text-zinc-500",
          )}>
            {(record as any).cycles_present}/{(record as any).total_cycles}
          </span>
        ) : "—"}
      </td>
      <td className="px-3 py-2">
        <p className="text-xs text-zinc-300 tabular-nums whitespace-nowrap">{fmtTimestamp(record.last_seen)}</p>
      </td>
      <td className="px-3 py-2">
        <div className="flex items-center gap-1.5">
          {record.confidence_avg != null ? (
            <>
              <div className="h-1 w-12 rounded-full bg-zinc-800 overflow-hidden">
                <div
                  className={cn("h-full rounded-full",
                    record.confidence_avg >= 0.85 ? "bg-green-500" :
                    record.confidence_avg >= 0.70 ? "bg-amber-500" : "bg-red-500",
                  )}
                  style={{ width: `${record.confidence_avg * 100}%` }}
                />
              </div>
              <span className="text-[10px] text-zinc-400 tabular-nums">{(record.confidence_avg * 100).toFixed(0)}%</span>
            </>
          ) : <span className="text-[10px] text-zinc-600">—</span>}
        </div>
      </td>
      <td className="px-3 py-2">
        <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
          {(record.evidence_refs?.length ?? 0) > 0 && (
            <Button
              variant="ghost"
              size="sm"
              className="h-6 px-1.5 text-[10px] gap-0.5"
              onClick={() => onEvidence(record.evidence_refs!, record.person_name)}
            >
              <Eye className="h-3 w-3" />
              {record.evidence_refs!.length}
            </Button>
          )}
          {canOverride && (
            <Button
              variant="ghost"
              size="sm"
              className="h-6 px-1.5 text-[10px] gap-0.5 text-amber-400 hover:text-amber-300"
              onClick={() => onOverride(record)}
            >
              <Shield className="h-3 w-3" />
              Override
            </Button>
          )}
        </div>
      </td>
    </tr>
  );
}

// ── Session card (expandable) ─────────────────────────────────────────────────

function SessionCard({
  session,
  canOverride,
}: {
  session:     ExtSession;
  canOverride: boolean;
}) {
  const [expanded,        setExpanded]        = useState(false);
  const [overrideRecord,  setOverrideRecord]  = useState<ExtAttendanceRecord | null>(null);
  const [evidenceOpen,    setEvidenceOpen]    = useState(false);
  const [evidenceRefs,    setEvidenceRefs]    = useState<string[]>([]);
  const [evidenceTitle,   setEvidenceTitle]   = useState("");

  const { data: records, isLoading: loadingRecords } = useQuery({
    queryKey: ["session-records", session.session_id],
    queryFn:  () => attendanceApi.records(session.session_id).then(r => ((r.data as any).items ?? r.data) as ExtAttendanceRecord[]),
    enabled:  expanded,
  });

  const openEvidence = (refs: string[], name: string) => {
    setEvidenceRefs(refs);
    setEvidenceTitle(name);
    setEvidenceOpen(true);
  };

  const presentCount  = (records ?? []).filter(r => r.status === "P"  || r.status === "L" || r.status === "EE").length;
  const absentCount   = (records ?? []).filter(r => r.status === "A").length;
  const studentCount  = session.student_count ?? (session as any).record_count ?? (records?.length ?? 0);
  const recRate       = session.recognition_rate != null ? Math.round(session.recognition_rate * 100) : null;

  const _fmtSched = (ts: string | null | undefined) => { try { if (!ts) return "—"; const d = parseISO(ts); return isValid(d) ? format(d, "HH:mm") : "—"; } catch { return "—"; } };
  const scheduled = `${_fmtSched(session.scheduled_start)} – ${_fmtSched(session.scheduled_end)}`;

  return (
    <>
      <div className="border border-zinc-800 rounded-xl overflow-hidden bg-zinc-950">
        {/* Collapsed header */}
        <button
          onClick={() => setExpanded(v => !v)}
          className="w-full text-left px-5 py-4 hover:bg-zinc-900/50 transition-colors"
        >
          <div className="flex items-start justify-between gap-3">
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-sm font-semibold text-zinc-100 truncate">
                  {session.course_name ?? "Unnamed session"}
                </span>
                <SyncBadge status={session.sync_status} />
                {session.faculty_status && (
                  <span className={cn("text-[10px] font-medium", FACULTY_COLOR[session.faculty_status] ?? "text-zinc-400")}>
                    Faculty: {session.faculty_status}
                  </span>
                )}
              </div>
              <div className="flex items-center gap-4 mt-1.5 text-xs text-zinc-500 flex-wrap">
                <span className="flex items-center gap-1">
                  <BarChart2 className="h-3 w-3" />
                  {session.camera_name}
                </span>
                <span className="flex items-center gap-1">
                  <Clock className="h-3 w-3" />
                  {scheduled}
                </span>
                {session.faculty_name && (
                  <span className="flex items-center gap-1">
                    <Users className="h-3 w-3" />
                    {session.faculty_name}
                  </span>
                )}
                {session.course_id && (
                  <span className="font-mono text-[10px] text-zinc-600">{session.course_id}</span>
                )}
              </div>
            </div>

            <div className="flex items-center gap-4 shrink-0">
              {/* Stats */}
              <div className="hidden sm:flex items-center gap-4 text-right">
                <div>
                  <p className="text-sm font-semibold text-zinc-200 tabular-nums">{studentCount}</p>
                  <p className="text-[10px] text-zinc-600">students</p>
                </div>
                {recRate != null && (
                  <div>
                    <p className={cn("text-sm font-semibold tabular-nums",
                      recRate >= 80 ? "text-green-400" : recRate >= 60 ? "text-amber-400" : "text-red-400",
                    )}>
                      {recRate}%
                    </p>
                    <p className="text-[10px] text-zinc-600">recognition</p>
                  </div>
                )}
              </div>
              {expanded
                ? <ChevronDown  className="h-4 w-4 text-zinc-500 shrink-0" />
                : <ChevronRight className="h-4 w-4 text-zinc-500 shrink-0" />
              }
            </div>
          </div>

          {/* Recognition progress bar */}
          {recRate != null && (
            <div className="flex items-center gap-2 mt-2">
              <Progress
                value={recRate}
                className={cn("h-1 flex-1",
                  recRate >= 80 ? "[&>div]:bg-green-500" :
                  recRate >= 60 ? "[&>div]:bg-amber-500" : "[&>div]:bg-red-500",
                )}
              />
              <span className="text-[10px] text-zinc-500 tabular-nums w-8 text-right">{recRate}%</span>
            </div>
          )}
        </button>

        {/* Expanded content */}
        {expanded && (
          <div className="border-t border-zinc-800 px-5 py-4 space-y-4">
            {/* Timeline */}
            <div>
              <p className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider mb-2">
                Timeline
              </p>
              {loadingRecords ? (
                <div className="h-16 bg-zinc-900 rounded-lg animate-pulse" />
              ) : (
                <SessionTimeline session={session} records={records ?? []} />
              )}
            </div>

            {/* Student table */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <p className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider">
                  Students
                  {records && (
                    <span className="ml-2 font-normal text-zinc-600">
                      {presentCount} present · {absentCount} absent
                    </span>
                  )}
                </p>
                {records && records.some(r => (r.evidence_refs?.length ?? 0) > 0) && (
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-6 text-xs gap-1"
                    onClick={() => openEvidence(
                      records.flatMap(r => r.evidence_refs ?? []),
                      session.course_name ?? "Session",
                    )}
                  >
                    <Eye className="h-3 w-3" />
                    All evidence
                  </Button>
                )}
              </div>

              <div className="rounded-lg border border-zinc-800 overflow-hidden">
                <div className="overflow-x-auto max-h-72 overflow-y-auto">
                  <table className="w-full">
                    <thead className="bg-zinc-900 sticky top-0">
                      <tr>
                        {["Student", "Status", "Cycles", "Last Seen", "Confidence", ""].map(h => (
                          <th key={h} className="px-3 py-2 text-left text-[10px] font-semibold text-zinc-500 uppercase tracking-wider whitespace-nowrap">
                            {h}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {loadingRecords
                        ? [...Array(5)].map((_, i) => (
                            <tr key={i} className="border-t border-zinc-800">
                              {[...Array(6)].map((_, j) => (
                                <td key={j} className="px-3 py-2">
                                  <div className="h-3 bg-zinc-800 rounded animate-pulse" style={{ width: `${40 + (i * j + 1) % 50}%` }} />
                                </td>
                              ))}
                            </tr>
                          ))
                        : (records ?? []).length === 0
                        ? (
                            <tr>
                              <td colSpan={6} className="text-center py-6 text-zinc-600 text-sm">
                                No attendance records for this session.
                              </td>
                            </tr>
                          )
                        : (records ?? []).map(r => (
                            <StudentRow
                              key={r.record_id}
                              record={r}
                              canOverride={canOverride}
                              onOverride={setOverrideRecord}
                              onEvidence={openEvidence}
                            />
                          ))
                      }
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      <OverrideDialog  record={overrideRecord}   onClose={() => setOverrideRecord(null)} />
      <EvidenceDialog  open={evidenceOpen}        onClose={() => setEvidenceOpen(false)} evidenceRefs={evidenceRefs} title={evidenceTitle} />
    </>
  );
}

// ── Sessions tab ──────────────────────────────────────────────────────────────

function SessionsTab({ canOverride }: { canOverride: boolean }) {
  const [date,       setDate]       = useState(todayISO());
  const [statusFilter, setStatusFilter] = useState<SyncStatus | "ALL">("ALL");

  const { data, isLoading, isFetching, refetch } = useQuery({
    queryKey: ["attendance-sessions", date, statusFilter],
    queryFn:  () => attendanceApi.sessions({
      date,
      sync_status: statusFilter !== "ALL" ? statusFilter : undefined,
    }).then(r => r.data as { items: ExtSession[]; total: number }),
  });

  const sessions = data?.items ?? [];

  return (
    <div className="space-y-4">
      {/* Filters */}
      <div className="flex flex-wrap gap-2">
        <div className="relative">
          <Calendar className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-zinc-500" />
          <input
            type="date"
            value={date}
            onChange={e => setDate(e.target.value)}
            className="pl-8 h-9 rounded-md border border-zinc-800 bg-zinc-950 text-sm text-zinc-200 focus:outline-none focus:ring-1 focus:ring-primary pr-2"
          />
        </div>
        <select
          value={statusFilter}
          onChange={e => setStatusFilter(e.target.value as SyncStatus | "ALL")}
          className="h-9 rounded-md border border-zinc-800 bg-zinc-950 text-sm px-2.5 text-zinc-300 focus:outline-none focus:ring-1 focus:ring-primary"
        >
          <option value="ALL">All statuses</option>
          <option value="SYNCED">Synced</option>
          <option value="HELD">Held</option>
          <option value="PENDING">Pending</option>
          <option value="FAILED">Failed</option>
          <option value="DISCARDED">Discarded</option>
        </select>
        <Button variant="outline" size="sm" className="h-9 gap-1" onClick={() => refetch()}>
          <RefreshCw className={cn("h-3.5 w-3.5", isFetching && "animate-spin")} />
          Refresh
        </Button>
        <span className="ml-auto self-center text-xs text-zinc-500">
          {data?.total ?? 0} sessions on {date}
        </span>
      </div>

      {/* Session cards */}
      {isLoading ? (
        <div className="space-y-3">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="border border-zinc-800 rounded-xl p-5">
              <div className="h-4 bg-zinc-800 rounded animate-pulse w-48 mb-2" />
              <div className="h-3 bg-zinc-800 rounded animate-pulse w-64" />
            </div>
          ))}
        </div>
      ) : sessions.length === 0 ? (
        <div className="text-center py-16 text-zinc-600">
          <Calendar className="h-8 w-8 mx-auto mb-2 text-zinc-700" />
          No sessions found for {date}.
        </div>
      ) : (
        <div className="space-y-3">
          {sessions.map(s => (
            <SessionCard key={s.session_id} session={s} canOverride={canOverride} />
          ))}
        </div>
      )}
    </div>
  );
}

// ── Records tab ───────────────────────────────────────────────────────────────

function RecordsTab() {
  const [search,       setSearch]       = useState("");
  const [statusFilter, setStatusFilter] = useState<AttendanceStatus | "ALL">("ALL");
  const [dateFrom,     setDateFrom]     = useState("");
  const [dateTo,       setDateTo]       = useState("");
  const [page,         setPage]         = useState(1);
  const [exporting,    setExporting]    = useState<string | null>(null);

  const { data, isLoading, isFetching } = useQuery({
    queryKey: ["attendance-records", page, search, statusFilter, dateFrom, dateTo],
    queryFn:  () => api.get("/api/attendance/records", {
      params: {
        limit:     PAGE_SIZE,
        offset:    (page - 1) * PAGE_SIZE,
        search:    search   || undefined,
        status:    statusFilter !== "ALL" ? statusFilter : undefined,
        date_from: dateFrom || undefined,
        date_to:   dateTo   || undefined,
      },
    }).then(r => r.data as PaginatedResponse<ExtAttendanceRecord>),
  });

  const records    = data?.items ?? [];
  const total      = data?.total ?? 0;
  const totalPages = Math.ceil(total / PAGE_SIZE);

  const handleExport = async (fmt: "csv" | "excel" | "pdf") => {
    setExporting(fmt);
    try {
      const res = await attendanceApi.export({
        format:    fmt,
        search:    search   || undefined,
        status:    statusFilter !== "ALL" ? statusFilter : undefined,
        date_from: dateFrom || undefined,
        date_to:   dateTo   || undefined,
      });
      const ext = fmt === "excel" ? "xlsx" : fmt;
      const url = URL.createObjectURL(new Blob([res.data]));
      const a   = document.createElement("a");
      a.href     = url;
      a.download = `attendance-${new Date().toISOString().slice(0, 10)}.${ext}`;
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      toast.error("Export failed");
    } finally {
      setExporting(null);
    }
  };

  return (
    <div className="space-y-4">
      {/* Filters */}
      <div className="flex flex-wrap gap-2">
        <div className="relative flex-1 min-w-48">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-zinc-500" />
          <Input
            placeholder="Search name, ID, course…"
            value={search}
            onChange={e => { setSearch(e.target.value); setPage(1); }}
            className="pl-9 h-9"
          />
        </div>

        <select
          value={statusFilter}
          onChange={e => { setStatusFilter(e.target.value as AttendanceStatus | "ALL"); setPage(1); }}
          className="h-9 rounded-md border border-zinc-800 bg-zinc-950 text-sm px-2.5 text-zinc-300 focus:outline-none focus:ring-1 focus:ring-primary"
        >
          <option value="ALL">All statuses</option>
          {(["P", "L", "EE", "A", "ND", "EX"] as AttendanceStatus[]).map(s => (
            <option key={s} value={s}>{s} — {ATTENDANCE_LABEL[s]}</option>
          ))}
        </select>

        <div className="flex items-center gap-1">
          <Calendar className="h-3.5 w-3.5 text-zinc-500" />
          <input
            type="date"
            value={dateFrom}
            onChange={e => { setDateFrom(e.target.value); setPage(1); }}
            className="h-9 rounded-md border border-zinc-800 bg-zinc-950 text-sm text-zinc-300 px-2 focus:outline-none focus:ring-1 focus:ring-primary"
          />
          <span className="text-zinc-600 text-xs">–</span>
          <input
            type="date"
            value={dateTo}
            onChange={e => { setDateTo(e.target.value); setPage(1); }}
            className="h-9 rounded-md border border-zinc-800 bg-zinc-950 text-sm text-zinc-300 px-2 focus:outline-none focus:ring-1 focus:ring-primary"
          />
        </div>

        {(search || statusFilter !== "ALL" || dateFrom || dateTo) && (
          <Button variant="ghost" size="sm" className="h-9 gap-1 text-zinc-500" onClick={() => {
            setSearch(""); setStatusFilter("ALL"); setDateFrom(""); setDateTo(""); setPage(1);
          }}>
            <X className="h-3.5 w-3.5" />
            Clear
          </Button>
        )}
      </div>

      {/* Export + count row */}
      <div className="flex items-center justify-between">
        <span className="text-xs text-zinc-500">
          {total.toLocaleString()} record{total !== 1 ? "s" : ""}
          {isFetching && <Loader2 className="inline h-3 w-3 ml-1.5 animate-spin" />}
        </span>
        <div className="flex gap-1.5">
          {([
            { fmt: "csv",   icon: <FileText   className="h-3.5 w-3.5" />, label: "CSV"   },
            { fmt: "excel", icon: <Table      className="h-3.5 w-3.5" />, label: "Excel" },
            { fmt: "pdf",   icon: <FileDown   className="h-3.5 w-3.5" />, label: "PDF"   },
          ] as const).map(({ fmt, icon, label }) => (
            <Button
              key={fmt}
              variant="outline"
              size="sm"
              className="h-7 text-xs gap-1"
              disabled={exporting !== null}
              onClick={() => handleExport(fmt)}
            >
              {exporting === fmt ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : icon}
              {label}
            </Button>
          ))}
        </div>
      </div>

      {/* Table */}
      <div className="border border-zinc-800 rounded-xl overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-zinc-900 border-b border-zinc-800">
              <tr>
                {["Student", "Course", "Date", "Status", "Cycles Detected", "Confidence", "Overridden"].map(h => (
                  <th key={h} className="px-4 py-3 text-left text-[11px] font-semibold text-zinc-500 uppercase tracking-wider whitespace-nowrap">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-800/60">
              {isLoading ? (
                [...Array(8)].map((_, i) => (
                  <tr key={i}>
                    {[...Array(7)].map((_, j) => (
                      <td key={j} className="px-4 py-3">
                        <div className="h-3.5 bg-zinc-800 rounded animate-pulse" style={{ width: `${50 + (i * j) % 45}%` }} />
                      </td>
                    ))}
                  </tr>
                ))
              ) : records.length === 0 ? (
                <tr>
                  <td colSpan={7} className="text-center py-14 text-zinc-600">
                    <Search className="h-7 w-7 mx-auto mb-2 text-zinc-700" />
                    No records match your filters.
                  </td>
                </tr>
              ) : records.map(r => (
                <tr key={r.record_id} className="hover:bg-zinc-900/40 transition-colors">
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-3">
                      {/* Enrollment photo */}
                      <div className="flex flex-col items-center gap-0.5">
                        <img
                          src={`${API_BASE}/api/enrollment/thumbnail/${r.person_id}`}
                          alt={r.person_name ?? "Person"}
                          className="w-12 h-12 rounded-lg object-cover flex-shrink-0 bg-zinc-800 border border-zinc-700"
                          onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
                        />
                        <span className="text-[9px] text-zinc-600">enrolled</span>
                      </div>
                      {/* All captured faces */}
                      {(r.evidence_refs?.length ?? 0) > 0 && (
                        <div className="flex flex-col gap-0.5">
                          <div className="flex gap-1">
                            {Array.from({ length: r.evidence_refs!.length }).map((_, i) => (
                              <div key={i} className="shrink-0 h-12 w-12 rounded-lg overflow-hidden bg-zinc-800 border border-zinc-700">
                                <img
                                  src={`${API_BASE}/api/attendance/records/${r.record_id}/evidence?index=${i}`}
                                  alt={`Capture ${i + 1}`}
                                  className="w-full h-full object-cover"
                                  onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
                                />
                              </div>
                            ))}
                          </div>
                          <span className="text-[9px] text-zinc-600">captured ({r.evidence_refs!.length})</span>
                        </div>
                      )}
                      <div>
                        <p className="text-sm font-medium text-zinc-200">{r.person_name ?? "Unknown"}</p>
                        <p className="text-[10px] text-zinc-500 font-mono">{r.person_external_id}</p>
                      </div>
                    </div>
                  </td>
                  <td className="px-4 py-3 text-xs text-zinc-400 max-w-[150px] truncate">
                    {(r as ExtAttendanceRecord & { course_name?: string }).course_name || "—"}
                  </td>
                  <td className="px-4 py-3 text-xs text-zinc-400 whitespace-nowrap">
                    {fmtTimestamp(r.first_seen)}
                  </td>
                  <td className="px-4 py-3"><AttBadge status={r.status} /></td>
                  <td className="px-4 py-3 text-xs tabular-nums">
                    {(r as any).total_cycles > 0 ? (
                      <span className={cn("font-semibold",
                        (r as any).cycles_present > 0 ? "text-green-400" : "text-zinc-500",
                      )}>
                        {(r as any).cycles_present}/{(r as any).total_cycles}
                      </span>
                    ) : "—"}
                  </td>
                  <td className="px-4 py-3 text-xs text-zinc-400 tabular-nums">
                    {r.confidence_avg != null ? `${(r.confidence_avg * 100).toFixed(1)}%` : "—"}
                  </td>
                  <td className="px-4 py-3">
                    {r.override_by ? (
                      <span className="flex items-center gap-1 text-[10px] text-blue-400" title={r.override_reason ?? undefined}>
                        <Shield className="h-3 w-3" />
                        Yes
                      </span>
                    ) : (
                      <span className="text-[10px] text-zinc-600">—</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between px-4 py-3 border-t border-zinc-800 bg-zinc-900/40">
            <span className="text-xs text-zinc-500">
              {((page - 1) * PAGE_SIZE) + 1}–{Math.min(page * PAGE_SIZE, total)} of {total.toLocaleString()}
            </span>
            <div className="flex items-center gap-1">
              <Button variant="outline" size="sm" className="h-7 w-7 p-0" disabled={page <= 1} onClick={() => setPage(p => p - 1)}>
                <ChevronRight className="h-3.5 w-3.5 rotate-180" />
              </Button>
              <span className="text-xs text-zinc-400 px-2 tabular-nums">{page} / {totalPages}</span>
              <Button variant="outline" size="sm" className="h-7 w-7 p-0" disabled={page >= totalPages} onClick={() => setPage(p => p + 1)}>
                <ChevronRight className="h-3.5 w-3.5" />
              </Button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Held batch card ───────────────────────────────────────────────────────────

function HeldBatchCard({
  session,
  canAct,
}: {
  session: ExtSession;
  canAct:  boolean;
}) {
  const qc = useQueryClient();
  const [confirming,  setConfirming]  = useState<"sync" | "discard" | null>(null);
  const [note,        setNote]        = useState("");
  const [evidenceOpen, setEvidenceOpen] = useState(false);
  const [allEvidence, setAllEvidence] = useState<string[]>([]);
  const [submitting,  setSubmitting]  = useState(false);

  const { data: records } = useQuery({
    queryKey:  ["session-records", session.session_id],
    queryFn:   () => attendanceApi.records(session.session_id).then(r => ((r.data as any).items ?? r.data) as ExtAttendanceRecord[]),
    enabled:   evidenceOpen,
  });

  const handleViewEvidence = async () => {
    // Preload records for evidence
    setEvidenceOpen(true);
  };

  useEffect(() => {
    if (records) {
      setAllEvidence(records.flatMap(r => (r as ExtAttendanceRecord).evidence_refs ?? []));
    }
  }, [records]);

  const handleAction = async (action: "sync" | "discard") => {
    setSubmitting(true);
    try {
      if (action === "sync") {
        await attendanceApi.forceSync(session.session_id, { note: note.trim() || undefined });
        toast.success("Session force-synced to ERP");
      } else {
        await attendanceApi.discard(session.session_id, { reason: note.trim() });
        toast.success("Session discarded");
      }
      qc.invalidateQueries({ queryKey: ["attendance-held"] });
      setConfirming(null);
      setNote("");
    } catch {
      toast.error(`${action === "sync" ? "Force sync" : "Discard"} failed`);
    } finally {
      setSubmitting(false);
    }
  };

  const priority = HELD_PRIORITY[session.held_reason ?? ""] ?? { label: "Low", color: "text-blue-400 bg-blue-900/30" };
  const studentCount = session.student_count ?? 0;
  const _effStart = session.scheduled_start ?? session.actual_start;
  const _effEnd   = session.scheduled_end   ?? session.actual_end;
  const sessionDate = _effStart ? format(parseISO(_effStart), "dd MMM yyyy") : "—";
  const sessionTime = `${_effStart ? format(parseISO(_effStart), "HH:mm") : "—"} – ${_effEnd ? format(parseISO(_effEnd), "HH:mm") : "—"}`;

  return (
    <>
      <div className="border border-zinc-800 rounded-xl overflow-hidden bg-zinc-950">
        <div className="px-5 py-4">
          {/* Header row */}
          <div className="flex items-start justify-between gap-3">
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 flex-wrap mb-1">
                <span className="text-sm font-semibold text-zinc-100">
                  {session.course_name ?? "Unnamed session"}
                </span>
                <span className={cn("text-[10px] font-semibold px-1.5 py-0.5 rounded", priority.color)}>
                  {priority.label} priority
                </span>
                <SyncBadge status="HELD" />
              </div>
              <div className="flex items-center gap-4 text-xs text-zinc-500 flex-wrap">
                <span>{session.camera_name}</span>
                <span className="flex items-center gap-1"><Calendar className="h-3 w-3" />{sessionDate}</span>
                <span className="flex items-center gap-1"><Clock className="h-3 w-3" />{sessionTime}</span>
                <span className="flex items-center gap-1"><Users className="h-3 w-3" />{studentCount} students</span>
              </div>
            </div>
          </div>

          {/* Held reason */}
          {session.held_reason && (
            <div className="mt-3 flex items-start gap-2 text-xs text-amber-300 bg-amber-900/20 border border-amber-800/40 rounded-lg px-3 py-2">
              <AlertTriangle className="h-3.5 w-3.5 shrink-0 mt-0.5 text-amber-400" />
              <span>
                <strong>Held reason:</strong>{" "}
                {session.held_reason.replace(/_/g, " ").toLowerCase()}
              </span>
            </div>
          )}

          {/* Actions */}
          <div className="flex items-center gap-2 mt-4">
            <Button variant="outline" size="sm" className="h-7 text-xs gap-1" onClick={handleViewEvidence}>
              <Eye className="h-3.5 w-3.5" />
              View Evidence
            </Button>
            {canAct && !confirming && (
              <>
                <Button
                  variant="outline"
                  size="sm"
                  className="h-7 text-xs gap-1 text-green-400 border-green-800 hover:bg-green-900/20"
                  onClick={() => setConfirming("sync")}
                >
                  <CheckCircle2 className="h-3.5 w-3.5" />
                  Force Sync
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  className="h-7 text-xs gap-1 text-red-400 border-red-900 hover:bg-red-900/20"
                  onClick={() => setConfirming("discard")}
                >
                  <XCircle className="h-3.5 w-3.5" />
                  Discard
                </Button>
              </>
            )}
          </div>

          {/* Inline confirmation */}
          {canAct && confirming && (
            <div className={cn(
              "mt-3 rounded-lg border p-3 space-y-2",
              confirming === "sync"
                ? "border-green-800 bg-green-900/10"
                : "border-red-900 bg-red-900/10",
            )}>
              <p className="text-xs font-medium text-zinc-300">
                {confirming === "sync"
                  ? "This will mark the session as synced and push records to ERP."
                  : "This will permanently discard the session. This action cannot be undone."}
              </p>
              <textarea
                value={note}
                onChange={e => setNote(e.target.value)}
                placeholder={confirming === "sync" ? "Optional note…" : "Reason for discard (required)"}
                rows={2}
                className="w-full rounded-md border border-zinc-800 bg-zinc-950 text-xs text-zinc-200 px-2.5 py-1.5 placeholder:text-zinc-600 focus:outline-none focus:ring-1 focus:ring-primary resize-none"
              />
              <div className="flex gap-2">
                <Button
                  size="sm"
                  className={cn("h-7 text-xs",
                    confirming === "sync"
                      ? "bg-green-700 hover:bg-green-600 text-white"
                      : "bg-red-800 hover:bg-red-700 text-white",
                  )}
                  disabled={submitting || (confirming === "discard" && !note.trim())}
                  onClick={() => handleAction(confirming)}
                >
                  {submitting
                    ? <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    : confirming === "sync" ? "Confirm Sync" : "Confirm Discard"}
                </Button>
                <Button variant="ghost" size="sm" className="h-7 text-xs" onClick={() => { setConfirming(null); setNote(""); }}>
                  Cancel
                </Button>
              </div>
            </div>
          )}
        </div>
      </div>

      <EvidenceDialog
        open={evidenceOpen}
        onClose={() => setEvidenceOpen(false)}
        evidenceRefs={allEvidence}
        title={session.course_name ?? "Session"}
      />
    </>
  );
}

// ── Held tab ──────────────────────────────────────────────────────────────────

function HeldTab({ canAct }: { canAct: boolean }) {
  const [search, setSearch] = useState("");

  const { data, isLoading, isFetching, refetch } = useQuery({
    queryKey: ["attendance-held", search],
    queryFn:  () => attendanceApi.held({ search: search || undefined }).then(
      r => r.data as { items: ExtSession[]; total: number },
    ),
  });

  const sessions = (data?.items ?? []);
  const filtered = search
    ? sessions.filter(s =>
        (s.course_name ?? "").toLowerCase().includes(search.toLowerCase()) ||
        (s.camera_name ?? "").toLowerCase().includes(search.toLowerCase()),
      )
    : sessions;

  return (
    <div className="space-y-4">
      <div className="flex gap-2">
        <div className="relative flex-1 max-w-xs">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-zinc-500" />
          <Input
            placeholder="Search course or room…"
            value={search}
            onChange={e => setSearch(e.target.value)}
            className="pl-9 h-9"
          />
        </div>
        <Button variant="outline" size="sm" className="h-9 gap-1" onClick={() => refetch()}>
          <RefreshCw className={cn("h-3.5 w-3.5", isFetching && "animate-spin")} />
          Refresh
        </Button>
        <span className="ml-auto self-center text-xs text-zinc-500">
          {filtered.length} held session{filtered.length !== 1 ? "s" : ""}
        </span>
      </div>

      {!canAct && (
        <div className="flex items-center gap-2 text-xs text-zinc-500 bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-2">
          <Eye className="h-3.5 w-3.5 shrink-0" />
          You have read-only access. Contact a Client Admin to sync or discard held sessions.
        </div>
      )}

      {isLoading ? (
        <div className="space-y-3">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="border border-zinc-800 rounded-xl p-5 space-y-2">
              <div className="h-4 bg-zinc-800 rounded animate-pulse w-40" />
              <div className="h-3 bg-zinc-800 rounded animate-pulse w-60" />
            </div>
          ))}
        </div>
      ) : filtered.length === 0 ? (
        <div className="text-center py-16 text-zinc-600">
          <CheckCircle2 className="h-8 w-8 mx-auto mb-2 text-zinc-700" />
          {search ? "No held sessions match your search." : "No held sessions — all clear!"}
        </div>
      ) : (
        <div className="space-y-3">
          {filtered.map(s => (
            <HeldBatchCard key={s.session_id} session={s} canAct={canAct} />
          ))}
        </div>
      )}
    </div>
  );
}

// ── Unknown persons tab ───────────────────────────────────────────────────────

function UnknownTab() {
  const qc = useQueryClient();
  const [sessionFilter, setSessionFilter] = useState("");
  const [assignDialog, setAssignDialog] = useState<{ detectionId: string } | null>(null);
  const [personQuery, setPersonQuery] = useState("");
  const [personResults, setPersonResults] = useState<any[]>([]);
  const [searchLoading, setSearchLoading] = useState(false);

  const { data: unknownRaw, isLoading } = useQuery({
    queryKey: ["unknown-detections", sessionFilter],
    queryFn: () => attendanceApi.unknown(
      sessionFilter ? { session_id: sessionFilter, status: "UNASSIGNED" } : { status: "UNASSIGNED" }
    ).then(r => r.data),
    refetchInterval: 30_000,
  });
  const detections: any[] = Array.isArray(unknownRaw) ? unknownRaw : [];

  async function searchPersons(q: string) {
    setPersonQuery(q);
    if (q.length < 2) { setPersonResults([]); return; }
    setSearchLoading(true);
    try {
      const res = await enrollmentApi.list({ q, limit: 10 });
      const items = (res.data as any)?.items ?? [];
      setPersonResults(items);
    } catch { setPersonResults([]); }
    finally { setSearchLoading(false); }
  }

  async function assign(detectionId: string, personId: string) {
    try {
      await attendanceApi.assignUnknown(detectionId, personId);
      toast.success("Assigned and marked present");
      qc.invalidateQueries({ queryKey: ["unknown-detections"] });
      setAssignDialog(null);
    } catch {
      toast.error("Failed to assign");
    }
  }

  async function dismiss(detectionId: string) {
    try {
      await attendanceApi.dismissUnknown(detectionId);
      toast.success("Dismissed");
      qc.invalidateQueries({ queryKey: ["unknown-detections"] });
    } catch {
      toast.error("Failed to dismiss");
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <Input
          placeholder="Filter by session ID…"
          value={sessionFilter}
          onChange={(e) => setSessionFilter(e.target.value)}
          className="max-w-xs h-8 bg-zinc-900 border-zinc-700 text-sm"
        />
        <span className="text-xs text-zinc-500">
          Unrecognized faces captured during scanning. Assign them to a known person to mark attendance.
        </span>
      </div>

      {isLoading ? (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
          {[...Array(8)].map((_, i) => (
            <div key={i} className="bg-zinc-900 border border-zinc-800 rounded-xl h-48 animate-pulse" />
          ))}
        </div>
      ) : detections.length === 0 ? (
        <div className="text-center py-20 text-zinc-600">
          <UserX className="h-10 w-10 mx-auto mb-2 text-zinc-700" />
          No unassigned unknown detections.
        </div>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
          {detections.map((det) => (
            <div key={det.detection_id} className="bg-zinc-900 border border-zinc-700 rounded-xl overflow-hidden flex flex-col">
              <div className="bg-zinc-800 h-36 flex items-center justify-center">
                <img
                  src={`${API_BASE}/api/attendance/unknown/${det.detection_id}/image`}
                  alt="Unknown face"
                  className="h-full w-full object-cover"
                  onError={(e) => {
                    (e.currentTarget as HTMLImageElement).style.display = "none";
                  }}
                />
              </div>
              <div className="p-3 space-y-2 flex-1 flex flex-col">
                <div className="text-[10px] text-zinc-500 space-y-0.5">
                  {det.confidence != null && (
                    <p>Conf: <span className="text-zinc-300">{(det.confidence * 100).toFixed(0)}%</span></p>
                  )}
                  {det.liveness_score != null && (
                    <p>Live: <span className="text-zinc-300">{(det.liveness_score * 100).toFixed(0)}%</span></p>
                  )}
                  <p className="text-zinc-600 truncate" title={det.session_id}>
                    Session: {det.session_id?.slice(0, 8)}…
                  </p>
                </div>
                <div className="flex gap-1.5 mt-auto pt-1">
                  <button
                    onClick={() => { setAssignDialog({ detectionId: det.detection_id }); setPersonQuery(""); setPersonResults([]); }}
                    className="flex-1 flex items-center justify-center gap-1 bg-blue-600 hover:bg-blue-500 text-white text-[11px] font-medium py-1.5 rounded-lg transition-colors"
                  >
                    <UserCheck2 className="h-3 w-3" /> Assign
                  </button>
                  <button
                    onClick={() => dismiss(det.detection_id)}
                    className="flex-1 flex items-center justify-center gap-1 bg-red-900/50 hover:bg-red-800/70 text-red-400 hover:text-red-300 text-[11px] font-medium py-1.5 rounded-lg transition-colors"
                  >
                    <X className="h-3 w-3" /> Reject
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Assign dialog */}
      {assignDialog && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-zinc-900 border border-zinc-700 rounded-2xl w-full max-w-md p-6 space-y-4">
            <h2 className="text-lg font-semibold text-white">Assign to Person</h2>
            <p className="text-sm text-zinc-400">
              Search for a person to assign this detection to. They will be marked <span className="text-green-400 font-medium">Present</span> for the session.
            </p>
            <div className="relative">
              <Search className="absolute left-3 top-2.5 h-4 w-4 text-zinc-500" />
              <Input
                className="pl-9 bg-zinc-800 border-zinc-600 text-white"
                placeholder="Search by name or ID…"
                value={personQuery}
                onChange={(e) => searchPersons(e.target.value)}
              />
            </div>
            {searchLoading && <p className="text-xs text-zinc-500">Searching…</p>}
            {personResults.length > 0 && (
              <div className="space-y-1 max-h-48 overflow-y-auto">
                {personResults.map((p: any) => (
                  <button
                    key={p.person_id}
                    onClick={() => assign(assignDialog.detectionId, p.person_id)}
                    className="w-full flex items-center gap-3 p-2 rounded-lg hover:bg-zinc-800 transition-colors text-left"
                  >
                    <Avatar className="h-8 w-8 shrink-0">
                      <AvatarImage src={`${API_BASE}/api/enrollment/thumbnail/${p.external_id || p.person_id}`} />
                      <AvatarFallback className="bg-zinc-700 text-xs">{(p.name || "?")[0]}</AvatarFallback>
                    </Avatar>
                    <div>
                      <p className="text-sm text-white font-medium">{p.name}</p>
                      <p className="text-[10px] text-zinc-500">{p.external_id}</p>
                    </div>
                  </button>
                ))}
              </div>
            )}
            <div className="flex justify-end">
              <button
                onClick={() => setAssignDialog(null)}
                className="px-4 py-2 text-sm text-zinc-400 hover:text-white transition-colors"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function AttendancePage() {
  const { isClientAdmin } = useAuth();
  const canOverride = isClientAdmin;

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <div className="mb-6">
        <h1 className="text-xl font-bold text-zinc-100">Attendance</h1>
        <p className="text-sm text-zinc-500 mt-0.5">
          Session records, cross-session analytics, and held batch management.
        </p>
      </div>

      <Tabs defaultValue="sessions">
        <TabsList className="mb-6">
          <TabsTrigger value="sessions" className="gap-1.5">
            <BarChart2 className="h-3.5 w-3.5" />
            Sessions
          </TabsTrigger>
          <TabsTrigger value="records" className="gap-1.5">
            <Table className="h-3.5 w-3.5" />
            Records
          </TabsTrigger>
          <TabsTrigger value="unknown" className="gap-1.5">
            <UserX className="h-3.5 w-3.5" />
            Unknown Faces
          </TabsTrigger>
        </TabsList>

        <TabsContent value="sessions">
          <SessionsTab canOverride={canOverride} />
        </TabsContent>

        <TabsContent value="records">
          <RecordsTab />
        </TabsContent>

        <TabsContent value="unknown">
          <UnknownTab />
        </TabsContent>
      </Tabs>
    </div>
  );
}
