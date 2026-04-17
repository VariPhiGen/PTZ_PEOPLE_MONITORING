"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useDropzone } from "react-dropzone";
import {
  ChevronDown,
  ChevronRight,
  ChevronLeft,
  User,
  Upload,
  X,
  CheckCircle2,
  XCircle,
  Search,
  Plus,
  MoreVertical,
  RefreshCw,
  Download,
  Image as ImageIcon,
  Trash2,
  RotateCcw,
  UserX,
  FileSpreadsheet,
  Camera,
  Lightbulb,
  Eye,
  TrendingUp,
  BarChart3,
  Clock,
} from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";
import { toast } from "sonner";
import { cn, fmtDatetime } from "@/lib/utils";
import { api, enrollmentApi, analyticsApi, datasetsApi, adminApi } from "@/lib/api";
import { useAuth } from "@/hooks/useAuth";
import type { PersonRole, PersonStatus, PaginatedResponse } from "@/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";

// ── Local types ───────────────────────────────────────────────────────────────

interface EnrolledPerson {
  person_id:     string;
  external_id:   string;
  name:          string;
  role:          PersonRole;
  department:    string | null;
  email:         string | null;
  phone:         string | null;
  status:        PersonStatus;
  enrolled:      boolean;
  thumbnail:     string | null;
  last_seen:     string | null;
  image_count:   number;
  quality_score: number | null;
}

interface ImageItem {
  id:      string;
  file:    File;
  preview: string;
  status:  "pending" | "analyzing" | "pass" | "fail";
  score:   number | null;
  reason:  string | null;
  tempKey: string | null;   // returned by /upload — passed to /enroll
}

interface PersonForm {
  name:        string;
  external_id: string;
  role:        PersonRole;
  department:  string;
  email:       string;
  phone:       string;
  dataset_id:  string;  // empty = use default dataset
}

const EMPTY_FORM: PersonForm = {
  name: "", external_id: "", role: "STUDENT",
  department: "", email: "", phone: "", dataset_id: "",
};

type SortKey = "name" | "quality_score" | "last_seen" | "role" | "image_count";

// ── Face angle SVG ────────────────────────────────────────────────────────────

function FaceSVG({
  pan = 0,
  tilt = 0,
  good = true,
  label = "",
  sunglasses = false,
  blur = false,
}: {
  pan?: number;
  tilt?: number;
  good?: boolean;
  label?: string;
  sunglasses?: boolean;
  blur?: boolean;
}) {
  const cx = 20, cy = 23;
  const eyeLX = cx - 6 + pan * 3;
  const eyeRX = cx + 6 + pan * 3;
  const eyeY  = cy - 6 - tilt * 5;
  const noseX = cx + pan * 9;
  const noseY = cy + 1 - tilt * 6;
  const stroke = good ? "#22c55e" : "#ef4444";
  const fill   = good ? "#4ade80" : "#52525b";

  return (
    <div className="flex flex-col items-center gap-1 min-w-[44px]">
      <svg
        width="40"
        height="46"
        viewBox="0 0 40 46"
        style={{ filter: blur ? "blur(1.8px)" : undefined }}
      >
        <ellipse cx="20" cy="24" rx="15" ry="19" fill="#27272a" stroke={stroke} strokeWidth="1.5" />
        {!sunglasses ? (
          <>
            <circle cx={eyeLX} cy={eyeY} r="2"   fill={fill} />
            <circle cx={eyeRX} cy={eyeY} r="2"   fill={fill} />
          </>
        ) : (
          <>
            <rect x={eyeLX - 5} y={eyeY - 3} width="10" height="6" rx="2" fill="#111" stroke="#374151" strokeWidth="0.5" />
            <rect x={eyeRX - 5} y={eyeY - 3} width="10" height="6" rx="2" fill="#111" stroke="#374151" strokeWidth="0.5" />
            <line x1={eyeLX + 5} y1={eyeY} x2={eyeRX - 5} y2={eyeY} stroke="#374151" strokeWidth="0.5" />
          </>
        )}
        <circle cx={noseX} cy={noseY} r="1.5" fill={stroke} />
        {!good && (
          <line x1="4" y1="5" x2="36" y2="43" stroke="#ef444455" strokeWidth="1.5" strokeLinecap="round" />
        )}
      </svg>
      <span className={cn("text-[9px] text-center leading-tight w-12",
        good ? "text-green-400" : "text-zinc-500",
      )}>
        {label}
      </span>
    </div>
  );
}

// ── Guidelines panel ──────────────────────────────────────────────────────────

function GuidelinesPanel() {
  const [open, setOpen] = useState(false);

  return (
    <div className="border border-zinc-800 rounded-xl overflow-hidden mb-6">
      <button
        onClick={() => setOpen(v => !v)}
        className="w-full flex items-center justify-between px-5 py-3.5 bg-zinc-900 hover:bg-zinc-800/70 transition-colors"
      >
        <div className="flex items-center gap-2.5 text-sm font-medium">
          <Lightbulb className="h-4 w-4 text-amber-400" />
          Enrollment Guidelines
          <Badge variant="secondary" className="text-[10px] py-0">Quick Reference</Badge>
        </div>
        {open
          ? <ChevronDown className="h-4 w-4 text-zinc-500" />
          : <ChevronRight className="h-4 w-4 text-zinc-500" />
        }
      </button>

      {open && (
        <div className="bg-zinc-950/80 p-5 space-y-5 border-t border-zinc-800">
          {/* Angle guide */}
          <div>
            <p className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider mb-3 flex items-center gap-1.5">
              <CheckCircle2 className="h-3.5 w-3.5 text-green-400" /> Acceptable angles
            </p>
            <div className="flex gap-3 flex-wrap">
              <FaceSVG pan={0}     tilt={0}    good label="Frontal"    />
              <FaceSVG pan={-0.45} tilt={0}    good label="Left 15°"   />
              <FaceSVG pan={0.45}  tilt={0}    good label="Right 15°"  />
              <FaceSVG pan={0}     tilt={0.3}  good label="Slight up"  />
              <FaceSVG pan={0}     tilt={-0.3} good label="Slight down"/>
            </div>
            <p className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider mt-4 mb-3 flex items-center gap-1.5">
              <XCircle className="h-3.5 w-3.5 text-red-400" /> Rejected poses
            </p>
            <div className="flex gap-3 flex-wrap">
              <FaceSVG pan={-1}   tilt={0}   label="Side profile" />
              <FaceSVG pan={0.8}  tilt={0}   label="Heavy turn"   />
              <FaceSVG pan={0}    tilt={0.8} label="Looking up"   />
              <FaceSVG sunglasses            label="Sunglasses"   />
              <FaceSVG blur                  label="Blurred"      />
            </div>
          </div>

          {/* Tips grid */}
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            {[
              { icon: "🖼️", title: "8–12 images",    text: "Varied lighting & slight angle changes improve recognition accuracy." },
              { icon: "💡", title: "Good lighting",  text: "Even, natural light. Avoid harsh shadows or direct backlighting." },
              { icon: "📐", title: "Min 200×200 px", text: "Face must occupy at least ⅓ of the frame for inter-ocular ≥ 80 px." },
              { icon: "🚫", title: "No occlusion",   text: "No sunglasses, masks, hats or heavy makeup obscuring facial features." },
            ].map(t => (
              <div key={t.title} className="bg-zinc-900 rounded-lg p-3 text-xs">
                <div className="text-lg mb-1">{t.icon}</div>
                <div className="font-medium text-zinc-200 mb-0.5">{t.title}</div>
                <div className="text-zinc-500 leading-relaxed">{t.text}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Quality helpers ───────────────────────────────────────────────────────────

function QualityBadge({
  status, score, reason,
}: {
  status: ImageItem["status"];
  score:  number | null;
  reason: string | null;
}) {
  if (status === "pending")
    return <span className="text-[10px] text-zinc-500 bg-zinc-800 px-1.5 py-0.5 rounded">Pending</span>;
  if (status === "analyzing")
    return <span className="text-[10px] text-blue-400 bg-blue-900/30 px-1.5 py-0.5 rounded animate-pulse">Checking…</span>;
  if (status === "fail")
    return (
      <span
        className="flex items-center gap-0.5 text-[10px] text-red-400 bg-red-900/30 px-1.5 py-0.5 rounded"
        title={reason ?? undefined}
      >
        <XCircle className="h-3 w-3 shrink-0" />
        {reason ? reason.slice(0, 16) : "FAIL"}
      </span>
    );
  const pct = Math.round((score ?? 0) * 100);
  return (
    <span className={cn("flex items-center gap-0.5 text-[10px] px-1.5 py-0.5 rounded",
      pct >= 80 ? "text-green-400 bg-green-900/30" : "text-amber-400 bg-amber-900/30",
    )}>
      <CheckCircle2 className="h-3 w-3 shrink-0" />
      {pct}%
    </span>
  );
}

function QualityBar({ score }: { score: number | null }) {
  if (score === null) return <span className="text-zinc-600 text-xs">—</span>;
  const pct = score * 100;
  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 w-20 rounded-full bg-zinc-800 overflow-hidden">
        <div
          className={cn("h-full rounded-full transition-all",
            pct >= 80 ? "bg-green-500" : pct >= 65 ? "bg-amber-500" : "bg-red-500",
          )}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-[11px] text-zinc-400 tabular-nums">{pct.toFixed(0)}%</span>
    </div>
  );
}

const ROLE_COLOR: Record<PersonRole, string> = {
  STUDENT: "text-blue-400 bg-blue-900/30",
  FACULTY: "text-purple-400 bg-purple-900/30",
  ADMIN:   "text-amber-400 bg-amber-900/30",
};

const STATUS_COLOR: Record<PersonStatus, string> = {
  ACTIVE:    "text-green-400  bg-green-900/30",
  INACTIVE:  "text-zinc-400   bg-zinc-800",
  SUSPENDED: "text-red-400    bg-red-900/30",
};

// ── Sortable column header ────────────────────────────────────────────────────

function Th({
  label, sortKey, current, dir, onClick,
}: {
  label: string;
  sortKey: SortKey;
  current: SortKey;
  dir: "asc" | "desc";
  onClick: (k: SortKey) => void;
}) {
  const active = current === sortKey;
  return (
    <th
      onClick={() => onClick(sortKey)}
      className="px-4 py-3 text-left text-[11px] font-semibold text-zinc-500 uppercase tracking-wider cursor-pointer hover:text-zinc-300 select-none whitespace-nowrap"
    >
      <span className="inline-flex items-center gap-1">
        {label}
        {active && (
          <ChevronDown className={cn("h-3 w-3 transition-transform", dir === "desc" && "rotate-180")} />
        )}
      </span>
    </th>
  );
}

// ── Right-side Sheet ──────────────────────────────────────────────────────────

function Sheet({
  open, onClose, title, subtitle, children, footer,
}: {
  open:     boolean;
  onClose:  () => void;
  title:    string;
  subtitle?: string;
  children: React.ReactNode;
  footer?:  React.ReactNode;
}) {
  // Close on Escape
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [open, onClose]);

  return (
    <>
      <div
        className={cn(
          "fixed inset-0 z-50 bg-black/70 transition-opacity duration-300",
          open ? "opacity-100" : "opacity-0 pointer-events-none",
        )}
        onClick={onClose}
      />
      <div className={cn(
        "fixed inset-y-0 right-0 z-50 flex w-full max-w-2xl flex-col",
        "bg-zinc-950 border-l border-zinc-800 shadow-2xl",
        "transition-transform duration-300 ease-in-out",
        open ? "translate-x-0" : "translate-x-full",
      )}>
        <div className="flex items-start justify-between px-6 py-4 border-b border-zinc-800 shrink-0">
          <div>
            <h2 className="text-base font-semibold text-zinc-100">{title}</h2>
            {subtitle && <p className="text-xs text-zinc-500 mt-0.5">{subtitle}</p>}
          </div>
          <button onClick={onClose} className="text-zinc-500 hover:text-zinc-200 mt-0.5 transition-colors">
            <X className="h-5 w-5" />
          </button>
        </div>
        <div className="flex-1 overflow-y-auto">{children}</div>
        {footer && (
          <div className="shrink-0 border-t border-zinc-800 px-6 py-4 bg-zinc-950">{footer}</div>
        )}
      </div>
    </>
  );
}

// ── Step indicator ────────────────────────────────────────────────────────────

function StepIndicator({ current, steps }: { current: number; steps: string[] }) {
  return (
    <div className="flex items-center px-6 py-3 border-b border-zinc-800 bg-zinc-900/60 shrink-0">
      {steps.map((label, i) => {
        const idx    = i + 1;
        const done   = idx < current;
        const active = idx === current;
        return (
          <div key={label} className="flex items-center gap-1 flex-1 last:flex-none">
            <div className={cn(
              "h-5 w-5 rounded-full text-[10px] font-bold flex items-center justify-center shrink-0",
              done   ? "bg-green-500 text-white"        : "",
              active ? "bg-primary text-white"          : "",
              !done && !active ? "bg-zinc-800 text-zinc-500" : "",
            )}>
              {done ? "✓" : idx}
            </div>
            <span className={cn("text-[11px] font-medium",
              active ? "text-zinc-200" : done ? "text-green-400" : "text-zinc-600",
            )}>
              {label}
            </span>
            {i < steps.length - 1 && (
              <div className={cn("h-px flex-1 mx-2", done ? "bg-green-500/40" : "bg-zinc-800")} />
            )}
          </div>
        );
      })}
    </div>
  );
}

// ── Step 1: Person details / CSV import ───────────────────────────────────────

function parseCSV(text: string): Record<string, string>[] {
  const lines = text.split(/\r?\n/).filter(Boolean);
  if (lines.length < 2) return [];
  const headers = lines[0].split(",").map(h => h.trim());
  return lines.slice(1).map(line => {
    const vals = line.split(",").map(v => v.trim());
    return Object.fromEntries(headers.map((h, i) => [h, vals[i] ?? ""]));
  });
}

function Step1({
  form, onChange, bulkMode, onToggleBulk, csvPreview, onCSVDrop, csvFileName, datasets,
}: {
  form:         PersonForm;
  onChange:     (k: keyof PersonForm, v: string) => void;
  bulkMode:     boolean;
  onToggleBulk: () => void;
  csvPreview:   Record<string, string>[];
  onCSVDrop:    (files: File[]) => void;
  csvFileName:  string;
  datasets:     Array<{ dataset_id: string; name: string }>;
}) {
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop:    onCSVDrop,
    accept:    { "text/csv": [".csv"] },
    maxFiles:  1,
    disabled:  !bulkMode,
  });

  return (
    <div className="p-6 space-y-5">
      {/* Mode toggle */}
      <div className="flex gap-2 p-1 bg-zinc-900 rounded-lg border border-zinc-800">
        {[
          { bulk: false, label: "Single Person",       icon: <User className="h-3.5 w-3.5" /> },
          { bulk: true,  label: "CSV Bulk Import",     icon: <FileSpreadsheet className="h-3.5 w-3.5" /> },
        ].map(opt => (
          <button
            key={String(opt.bulk)}
            onClick={() => { if (opt.bulk !== bulkMode) onToggleBulk(); }}
            className={cn("flex-1 flex items-center justify-center gap-1.5 py-1.5 text-sm rounded-md transition-colors",
              opt.bulk === bulkMode
                ? "bg-primary text-white shadow-sm"
                : "text-zinc-500 hover:text-zinc-300",
            )}
          >
            {opt.icon}
            {opt.label}
          </button>
        ))}
      </div>

      {!bulkMode ? (
        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-xs text-zinc-400 mb-1 block">Full Name <span className="text-red-400">*</span></label>
              <Input
                placeholder="e.g. Priya Sharma"
                value={form.name}
                onChange={e => onChange("name", e.target.value)}
                className="h-9"
              />
            </div>
            <div>
              <label className="text-xs text-zinc-400 mb-1 block">ID / Roll No. <span className="text-red-400">*</span></label>
              <Input
                placeholder="e.g. STU2024001"
                value={form.external_id}
                onChange={e => onChange("external_id", e.target.value)}
                className="h-9"
              />
            </div>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-xs text-zinc-400 mb-1 block">Role <span className="text-red-400">*</span></label>
              <select
                value={form.role}
                onChange={e => onChange("role", e.target.value)}
                className="w-full h-9 rounded-md border border-zinc-800 bg-zinc-950 text-sm px-2.5 text-zinc-200 focus:outline-none focus:ring-1 focus:ring-primary"
              >
                <option value="STUDENT">Student</option>
                <option value="FACULTY">Faculty</option>
                <option value="ADMIN">Admin</option>
              </select>
            </div>
            <div>
              <label className="text-xs text-zinc-400 mb-1 block">Department</label>
              <Input
                placeholder="e.g. Computer Science"
                value={form.department}
                onChange={e => onChange("department", e.target.value)}
                className="h-9"
              />
            </div>
          </div>
          {/* Dataset selector — shown only when multiple datasets exist */}
          {datasets && datasets.length > 1 && (
            <div>
              <label className="text-xs text-zinc-400 mb-1 block">
                Face Dataset
                <span className="text-zinc-600 ml-1">— which group this person belongs to</span>
              </label>
              <select
                value={form.dataset_id}
                onChange={e => onChange("dataset_id", e.target.value)}
                className="w-full h-9 rounded-md border border-zinc-800 bg-zinc-950 text-sm px-2.5 text-zinc-200 focus:outline-none focus:ring-1 focus:ring-primary"
              >
                <option value="">Default dataset</option>
                {datasets.map((d: { dataset_id: string; name: string }) => (
                  <option key={d.dataset_id} value={d.dataset_id}>{d.name}</option>
                ))}
              </select>
            </div>
          )}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-xs text-zinc-400 mb-1 block">Email</label>
              <Input
                type="email"
                placeholder="person@institution.edu"
                value={form.email}
                onChange={e => onChange("email", e.target.value)}
                className="h-9"
              />
            </div>
            <div>
              <label className="text-xs text-zinc-400 mb-1 block">Phone</label>
              <Input
                placeholder="+91 98765 43210"
                value={form.phone}
                onChange={e => onChange("phone", e.target.value)}
                className="h-9"
              />
            </div>
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          <div
            {...getRootProps()}
            className={cn(
              "border-2 border-dashed rounded-xl py-10 text-center cursor-pointer transition-colors",
              isDragActive ? "border-primary bg-primary/5" : "border-zinc-800 hover:border-zinc-600",
            )}
          >
            <input {...getInputProps()} />
            <FileSpreadsheet className="h-8 w-8 text-zinc-600 mx-auto mb-2" />
            {csvFileName ? (
              <p className="text-sm text-green-400 font-medium">{csvFileName}</p>
            ) : (
              <>
                <p className="text-sm text-zinc-400">Drop a CSV file here, or click to browse</p>
                <p className="text-xs text-zinc-600 mt-1">Required columns: name, external_id, role, department, email, phone</p>
              </>
            )}
          </div>

          {csvPreview.length > 0 && (
            <div>
              <p className="text-xs text-zinc-400 mb-2">{csvPreview.length} rows found — preview:</p>
              <div className="max-h-52 overflow-auto rounded-lg border border-zinc-800">
                <table className="w-full text-xs">
                  <thead className="bg-zinc-900 sticky top-0">
                    <tr>
                      {Object.keys(csvPreview[0]).map(h => (
                        <th key={h} className="px-3 py-2 text-left text-zinc-400 font-medium whitespace-nowrap">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {csvPreview.slice(0, 30).map((row, i) => (
                      <tr key={i} className="border-t border-zinc-800/60 hover:bg-zinc-900/40">
                        {Object.values(row).map((v, j) => (
                          <td key={j} className="px-3 py-1.5 text-zinc-300">{v}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
                {csvPreview.length > 30 && (
                  <p className="text-center text-[10px] text-zinc-600 py-2">
                    …and {csvPreview.length - 30} more rows
                  </p>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Step 2: Image upload + quality analysis ───────────────────────────────────

function Step2({
  images, onAddImages, onRemoveImage, analyzing, onAnalyze,
}: {
  images:        ImageItem[];
  onAddImages:   (files: File[]) => void;
  onRemoveImage: (id: string) => void;
  analyzing:     boolean;
  onAnalyze:     () => void;
}) {
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop:    onAddImages,
    accept:    { "image/*": [".jpg", ".jpeg", ".png", ".webp"] },
    maxFiles:  20,
    multiple:  true,
  });

  const passCount    = images.filter(i => i.status === "pass").length;
  const failCount    = images.filter(i => i.status === "fail").length;
  const pendingCount = images.filter(i => i.status === "pending").length;
  const checked      = images.length > 0 && pendingCount === 0 && !analyzing;

  return (
    <div className="p-6 space-y-4">
      <div
        {...getRootProps()}
        className={cn(
          "border-2 border-dashed rounded-xl text-center cursor-pointer transition-colors",
          isDragActive ? "border-primary bg-primary/5" : "border-zinc-800 hover:border-zinc-600",
          images.length > 0 ? "py-5" : "py-10",
        )}
      >
        <input {...getInputProps()} />
        <Camera className="h-7 w-7 text-zinc-600 mx-auto mb-2" />
        <p className="text-sm text-zinc-400">
          {images.length > 0 ? "Drop more photos here" : "Drop face photos here, or click to browse"}
        </p>
        <p className="text-xs text-zinc-600 mt-0.5">JPG, PNG, WebP · up to 20 files</p>
      </div>

      {images.length > 0 && (
        <>
          {/* Stats bar */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3 text-xs">
              <span className="text-zinc-400">{images.length} images</span>
              {checked && (
                <>
                  <span className="text-green-400 font-medium">{passCount} passing</span>
                  {failCount > 0 && <span className="text-red-400">{failCount} failed</span>}
                </>
              )}
            </div>
            {pendingCount > 0 && !analyzing && (
              <Button size="sm" onClick={onAnalyze} className="h-7 text-xs gap-1">
                <CheckCircle2 className="h-3 w-3" />
                Check Quality ({pendingCount})
              </Button>
            )}
          </div>

          {analyzing && (
            <div className="space-y-1.5">
              <p className="text-xs text-blue-400 animate-pulse">Running quality gate…</p>
              <Progress className="h-1" />
            </div>
          )}

          {checked && images.length > 0 && (
            <div className="flex items-center gap-3">
              <Progress
                value={(passCount / images.length) * 100}
                className="h-2 flex-1"
              />
              <span className="text-xs text-zinc-400 tabular-nums whitespace-nowrap">
                {passCount}/{images.length} passing
              </span>
            </div>
          )}

          {/* Image grid */}
          <div className="grid grid-cols-4 gap-2">
            {images.map(img => (
              <div
                key={img.id}
                className={cn(
                  "relative group aspect-square rounded-lg overflow-hidden bg-zinc-900 border",
                  img.status === "pass" ? "border-green-500/40" :
                  img.status === "fail" ? "border-red-500/30"   : "border-zinc-800",
                )}
              >
                <img
                  src={img.preview}
                  alt=""
                  className={cn("w-full h-full object-cover",
                    img.status === "fail" && "opacity-40",
                  )}
                />
                {/* Quality badge */}
                <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/75 to-transparent px-1 pb-1 pt-3">
                  <QualityBadge status={img.status} score={img.score} reason={img.reason} />
                </div>
                {/* Pass tick */}
                {img.status === "pass" && (
                  <div className="absolute top-1 left-1 h-4 w-4 rounded-full bg-green-500/80 flex items-center justify-center">
                    <CheckCircle2 className="h-2.5 w-2.5 text-white" />
                  </div>
                )}
                {/* Remove button */}
                <button
                  onClick={() => onRemoveImage(img.id)}
                  className="absolute top-1 right-1 h-5 w-5 rounded-full bg-black/60 text-zinc-400 hover:text-white hidden group-hover:flex items-center justify-center"
                >
                  <X className="h-3 w-3" />
                </button>
              </div>
            ))}
          </div>
        </>
      )}

      {images.length === 0 && (
        <p className="text-center text-xs text-zinc-600 py-2">
          Add 8–12 face photos for best recognition accuracy.
        </p>
      )}
    </div>
  );
}

// ── Step 3: Review ────────────────────────────────────────────────────────────

function Step3({
  form, images, enrolling,
}: {
  form:      PersonForm;
  images:    ImageItem[];
  enrolling: boolean;
}) {
  const passing = images.filter(i => i.status === "pass");
  const failing = images.filter(i => i.status === "fail");

  return (
    <div className="p-6 space-y-5">
      <div className="bg-zinc-900 rounded-xl p-4">
        <h3 className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider mb-3">Person Details</h3>
        <dl className="grid grid-cols-2 gap-x-6 gap-y-2.5 text-sm">
          {([
            ["Name",       form.name       ],
            ["ID",         form.external_id],
            ["Role",       form.role       ],
            ["Department", form.department ],
            ["Email",      form.email      ],
            ["Phone",      form.phone      ],
          ] as [string, string][]).filter(([, v]) => v).map(([k, v]) => (
            <div key={k} className="flex justify-between gap-2">
              <dt className="text-zinc-500 shrink-0">{k}</dt>
              <dd className={cn("text-zinc-200 text-right truncate", k === "ID" && "font-mono text-xs")}>{v}</dd>
            </div>
          ))}
        </dl>
      </div>

      <div>
        <div className="flex items-center gap-2 mb-3">
          <h3 className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider">Passing Images</h3>
          <span className={cn("text-xs font-semibold px-1.5 py-0.5 rounded",
            passing.length >= 1
              ? "text-green-400 bg-green-900/30"
              : "text-red-400 bg-red-900/30",
          )}>
            {passing.length} / {images.length}
          </span>
          {passing.length < 1 && images.length > 0 && (
            <span className="text-xs text-red-400">Need at least 1</span>
          )}
        </div>

        {passing.length > 0 ? (
          <div className="grid grid-cols-6 gap-1.5">
            {passing.map(img => (
              <div key={img.id} className="aspect-square rounded-md overflow-hidden border border-green-500/30 bg-zinc-900">
                <img src={img.preview} alt="" className="w-full h-full object-cover" />
              </div>
            ))}
          </div>
        ) : (
          <p className="text-center text-sm text-zinc-600 py-6">
            No images passed quality gate. Go back and add better photos.
          </p>
        )}

        {failing.length > 0 && (
          <p className="text-xs text-zinc-600 mt-2">
            {failing.length} image{failing.length > 1 ? "s" : ""} skipped (quality failures).
          </p>
        )}
      </div>

      {enrolling && (
        <div className="flex items-center gap-2 text-sm text-blue-400">
          <RefreshCw className="h-4 w-4 animate-spin" />
          Enrolling — building embeddings…
        </div>
      )}
    </div>
  );
}

// ── New Enrollment Sheet ──────────────────────────────────────────────────────

const SHEET_STEPS = ["Details", "Images", "Review"];

function NewEnrollmentSheet({
  open, onClose, onSuccess, clientId,
}: {
  open:      boolean;
  onClose:   () => void;
  onSuccess: () => void;
  clientId?: string;
}) {
  const qc = useQueryClient();
  const [step,       setStep]       = useState<1 | 2 | 3>(1);
  const [form,       setForm]       = useState<PersonForm>(EMPTY_FORM);

  const { data: datasetsData } = useQuery({
    queryKey: ["datasets", clientId],
    queryFn:  () => datasetsApi.list(clientId ? { client_id: clientId } : undefined).then(r => r.data),
    enabled:  open && !!clientId,
    staleTime: 60_000,
  });
  const datasets: Array<{ dataset_id: string; name: string }> =
    (datasetsData as { items?: Array<{ dataset_id: string; name: string }> })?.items ?? [];
  const [bulkMode,   setBulkMode]   = useState(false);
  const [csvFile,    setCsvFile]    = useState("");
  const [csvPreview, setCsvPreview] = useState<Record<string, string>[]>([]);
  const [images,     setImages]     = useState<ImageItem[]>([]);
  const [analyzing,  setAnalyzing]  = useState(false);
  const [enrolling,  setEnrolling]  = useState(false);

  const reset = useCallback(() => {
    setStep(1); setForm(EMPTY_FORM); setBulkMode(false);
    setCsvFile(""); setCsvPreview([]);
    setImages(prev => { prev.forEach(i => URL.revokeObjectURL(i.preview)); return []; });
  }, []);

  const handleClose = () => { reset(); onClose(); };

  const updateForm = useCallback((k: keyof PersonForm, v: string) =>
    setForm(f => ({ ...f, [k]: v })), []);

  const handleCSVDrop = useCallback((files: File[]) => {
    const file = files[0];
    if (!file) return;
    setCsvFile(file.name);
    file.text().then(text => setCsvPreview(parseCSV(text)));
  }, []);

  const addImages = useCallback((files: File[]) => {
    setImages(prev => [
      ...prev,
      ...files.map(f => ({
        id:      crypto.randomUUID(),
        file:    f,
        preview: URL.createObjectURL(f),
        status:  "pending" as const,
        score:   null,
        reason:  null,
        tempKey: null,
      })),
    ]);
  }, []);

  const removeImage = useCallback((id: string) => {
    setImages(prev => {
      const item = prev.find(i => i.id === id);
      if (item) URL.revokeObjectURL(item.preview);
      return prev.filter(i => i.id !== id);
    });
  }, []);

  const analyzeImages = useCallback(async () => {
    const pending = images.filter(i => i.status === "pending");
    if (!pending.length) return;
    setAnalyzing(true);
    setImages(prev => prev.map(i => i.status === "pending" ? { ...i, status: "analyzing" as const } : i));
    try {
      const fd = new FormData();
      pending.forEach(i => fd.append("images", i.file, i.file.name));
      const { data } = await enrollmentApi.upload(fd);

      // Backend returns a list of results (one per file)
      type QResult = { filename: string; passed: boolean; reason: string | null; temp_key?: string | null; quality?: { iod_px?: number; sharpness?: number; confidence?: number } };
      const raw: QResult[] = Array.isArray(data) ? data : (data.results ?? [data]);
      const byName: Record<string, QResult> = raw.reduce(
        (acc: Record<string, QResult>, r: QResult) => { acc[r.filename] = r; return acc; }, {},
      );
      setImages(prev => prev.map(i => {
        if (i.status !== "analyzing") return i;
        const r = byName[i.file.name];
        if (!r) return { ...i, status: "pass" as ImageItem["status"], score: null, reason: null, tempKey: null };
        // Compute quality score from iod_px and sharpness (not raw detection confidence)
        // IOD: 20px=0 → 80px+=1.0;  Sharpness: 10=0 → 400+=1.0
        const q = r.quality;
        const iodScore      = q?.iod_px    != null ? Math.min(Math.max((q.iod_px - 20) / 60, 0), 1) : null;
        const sharpScore    = q?.sharpness  != null ? Math.min(q.sharpness / 400, 1)                 : null;
        const score = (iodScore != null && sharpScore != null)
          ? Math.round((0.5 * iodScore + 0.5 * sharpScore) * 100) / 100
          : iodScore ?? sharpScore ?? null;
        return { ...i, status: (r.passed ? "pass" : "fail") as ImageItem["status"], score, reason: r.reason, tempKey: r.temp_key ?? null };
      }));
    } catch {
      setImages(prev => prev.map(i => i.status === "analyzing" ? { ...i, status: "pending" as ImageItem["status"] } : i));
      toast.error("Quality check failed. Please try again.");
    } finally {
      setAnalyzing(false);
    }
  }, [images]);

  const handleEnroll = useCallback(async () => {
    const passing = images.filter(i => i.status === "pass");
    if (passing.length < 1) { toast.error("Need at least 1 passing image"); return; }
    setEnrolling(true);
    try {
      // Send image_keys (base64 temp keys from /upload) + person metadata as JSON
      const imageKeys = passing.map(i => i.tempKey).filter(Boolean) as string[];
      if (imageKeys.length === 0) {
        toast.error("Please run quality check first (Analyze Images)");
        setEnrolling(false);
        return;
      }
      await enrollmentApi.enroll({
        ...form,
        image_keys: imageKeys,
        ...(clientId ? { client_id: clientId } : {}),
      });
      toast.success(`${form.name} enrolled successfully`);
      qc.invalidateQueries({ queryKey: ["enrollment-list"] });
      handleClose();
      onSuccess();
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      toast.error(detail ?? "Enrollment failed");
    } finally {
      setEnrolling(false);
    }
  }, [form, images, qc, onSuccess, handleClose]);

  const handleBulkImport = useCallback(async () => {
    if (!csvPreview.length) return;
    setEnrolling(true);
    try {
      const fd = new FormData();
      fd.append("data", JSON.stringify(csvPreview));
      await enrollmentApi.bulkImport(fd);
      toast.success(`${csvPreview.length} persons queued for enrollment`);
      qc.invalidateQueries({ queryKey: ["enrollment-list"] });
      handleClose();
      onSuccess();
    } catch {
      toast.error("Bulk import failed");
    } finally {
      setEnrolling(false);
    }
  }, [csvPreview, qc, onSuccess, handleClose]);

  const canNext = (): boolean => {
    if (step === 1) return bulkMode ? csvPreview.length > 0 : !!(form.name.trim() && form.external_id.trim());
    if (step === 2) return images.filter(i => i.status === "pass").length >= 1;
    return false;
  };

  return (
    <Sheet
      open={open}
      onClose={handleClose}
      title="New Enrollment"
      subtitle="Add a person to the face recognition roster"
      footer={
        <div className="flex items-center justify-between">
          <Button
            variant="outline"
            size="sm"
            onClick={() => { if (step > 1) setStep(s => (s - 1) as 1 | 2 | 3); else handleClose(); }}
          >
            {step === 1 ? "Cancel" : <><ChevronLeft className="h-3.5 w-3.5 mr-1" />Back</>}
          </Button>
          <div className="flex gap-2">
            {bulkMode && step === 1 ? (
              <Button size="sm" disabled={!canNext() || enrolling} onClick={handleBulkImport}>
                {enrolling ? "Importing…" : `Import ${csvPreview.length} persons`}
              </Button>
            ) : step < 3 ? (
              <Button size="sm" disabled={!canNext()} onClick={() => setStep(s => (s + 1) as 2 | 3)}>
                Next
              </Button>
            ) : (
              <Button
                size="sm"
                disabled={images.filter(i => i.status === "pass").length < 1 || enrolling}
                onClick={handleEnroll}
              >
                {enrolling ? <><RefreshCw className="h-3.5 w-3.5 animate-spin mr-1.5" />Enrolling…</> : "Enroll"}
              </Button>
            )}
          </div>
        </div>
      }
    >
      {!bulkMode && <StepIndicator current={step} steps={SHEET_STEPS} />}
      {step === 1 && (
        <Step1
          form={form} onChange={updateForm}
          bulkMode={bulkMode} onToggleBulk={() => setBulkMode(v => !v)}
          csvPreview={csvPreview} onCSVDrop={handleCSVDrop}
          csvFileName={csvFile}
          datasets={datasets}
        />
      )}
      {step === 2 && (
        <Step2
          images={images}
          onAddImages={addImages}
          onRemoveImage={removeImage}
          analyzing={analyzing}
          onAnalyze={analyzeImages}
        />
      )}
      {step === 3 && <Step3 form={form} images={images} enrolling={enrolling} />}
    </Sheet>
  );
}

// ── Re-enroll dialog ──────────────────────────────────────────────────────────

function ReEnrollDialog({
  person, onClose,
}: {
  person: EnrolledPerson | null;
  onClose: () => void;
}) {
  const qc       = useQueryClient();
  const [images, setImages]    = useState<ImageItem[]>([]);
  const [analyzing, setAnalyzing] = useState(false);
  const [loading,   setLoading]   = useState(false);

  useEffect(() => {
    if (!person) {
      setImages(prev => { prev.forEach(i => URL.revokeObjectURL(i.preview)); return []; });
    }
  }, [person]);

  const addImages = useCallback((files: File[]) =>
    setImages(prev => [
      ...prev,
      ...files.map(f => ({
        id: crypto.randomUUID(), file: f,
        preview: URL.createObjectURL(f),
        status: "pending" as const, score: null, reason: null, tempKey: null,
      })),
    ]), []);

  const removeImage = useCallback((id: string) =>
    setImages(prev => {
      const it = prev.find(i => i.id === id);
      if (it) URL.revokeObjectURL(it.preview);
      return prev.filter(i => i.id !== id);
    }), []);

  const analyze = useCallback(async () => {
    const pending = images.filter(i => i.status === "pending");
    if (!pending.length) return;
    setAnalyzing(true);
    setImages(prev => prev.map(i => i.status === "pending" ? { ...i, status: "analyzing" as const } : i));
    try {
      const fd = new FormData();
      pending.forEach(i => fd.append("images", i.file, i.file.name));
      const { data } = await enrollmentApi.upload(fd);
      type R = { filename: string; passed: boolean; reason: string | null; temp_key?: string | null; quality?: { iod_px?: number; sharpness?: number; confidence?: number } };
      const raw: R[] = Array.isArray(data) ? data : (data.results ?? [data]);
      const byName: Record<string, R> = raw.reduce(
        (acc: Record<string, R>, r: R) => { acc[r.filename] = r; return acc; }, {},
      );
      setImages(prev => prev.map(i => {
        if (i.status !== "analyzing") return i;
        const r = byName[i.file.name];
        if (!r) return { ...i, status: "pass" as ImageItem["status"], score: null, reason: null, tempKey: null };
        const q = r.quality;
        const iodScore   = q?.iod_px    != null ? Math.min(Math.max((q.iod_px - 20) / 60, 0), 1) : null;
        const sharpScore = q?.sharpness  != null ? Math.min(q.sharpness / 400, 1)                 : null;
        const score = (iodScore != null && sharpScore != null)
          ? Math.round((0.5 * iodScore + 0.5 * sharpScore) * 100) / 100
          : iodScore ?? sharpScore ?? null;
        return { ...i, status: (r.passed ? "pass" : "fail") as ImageItem["status"], score, reason: r.reason, tempKey: r.temp_key ?? null };
      }));
    } catch {
      setImages(prev => prev.map(i => i.status === "analyzing" ? { ...i, status: "pending" as ImageItem["status"] } : i));
      toast.error("Quality check failed");
    } finally {
      setAnalyzing(false);
    }
  }, [images]);

  const handleReEnroll = async () => {
    if (!person) return;
    const passing = images.filter(i => i.status === "pass");
    if (passing.length < 1) { toast.error("Need at least 1 passing image"); return; }
    const imageKeys = passing.map(i => i.tempKey).filter(Boolean) as string[];
    if (imageKeys.length === 0) { toast.error("Please run quality check first"); return; }
    setLoading(true);
    try {
      await enrollmentApi.reEnroll(person.person_id, { image_keys: imageKeys });
      toast.success(`${person.name} re-enrolled`);
      qc.invalidateQueries({ queryKey: ["enrollment-list"] });
      qc.invalidateQueries({ queryKey: ["person-detail", person.person_id] });
      onClose();
    } catch {
      toast.error("Re-enrollment failed");
    } finally {
      setLoading(false);
    }
  };

  const passCount = images.filter(i => i.status === "pass").length;
  const pendingCount = images.filter(i => i.status === "pending").length;

  return (
    <Dialog open={!!person} onOpenChange={open => !open && onClose()}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>Re-enroll — {person?.name}</DialogTitle>
        </DialogHeader>
        <p className="text-xs text-zinc-500 -mt-2">
          Upload new face photos to replace the current embedding template.
        </p>

        <Step2
          images={images}
          onAddImages={addImages}
          onRemoveImage={removeImage}
          analyzing={analyzing}
          onAnalyze={analyze}
        />

        <div className="flex justify-end gap-2 pt-1">
          <Button variant="outline" size="sm" onClick={onClose}>Cancel</Button>
          <Button
            size="sm"
            disabled={passCount < 1 || loading}
            onClick={handleReEnroll}
          >
            {loading ? <><RefreshCw className="h-3.5 w-3.5 animate-spin mr-1.5" />Re-enrolling…</> : `Re-enroll (${passCount} images)`}
          </Button>
        </div>

        {pendingCount > 0 && passCount === 0 && (
          <p className="text-xs text-amber-400 text-center">
            Click "Check Quality" to analyse your images before re-enrolling. At least 1 image must pass.
          </p>
        )}
      </DialogContent>
    </Dialog>
  );
}

// ── Person Detail Dialog ──────────────────────────────────────────────────────

interface PersonDetailDialogProps {
  person:  EnrolledPerson | null;
  onClose: () => void;
  canEdit: boolean;
  onReEnroll: (p: EnrolledPerson) => void;
}

function PersonDetailDialog({ person, onClose, canEdit, onReEnroll }: PersonDetailDialogProps) {
  const qc      = useQueryClient();
  const [editForm, setEditForm] = useState<Partial<PersonForm>>({});
  const [saving,   setSaving]   = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);

  const { data: detail, isLoading } = useQuery({
    queryKey: ["person-detail", person?.person_id],
    queryFn:  () => enrollmentApi.get(person!.person_id).then(r => r.data),
    enabled:  !!person,
  });

  const { data: rawEnrolledImages } = useQuery({
    queryKey: ["person-images", person?.person_id],
    queryFn:  () => enrollmentApi.images(person!.person_id).then(r => r.data),
    enabled:  !!person,
  });
  const enrolledImages = (() => {
    if (!rawEnrolledImages) return undefined;
    const apiBase = process.env.NEXT_PUBLIC_API_URL ?? "";
    const items = rawEnrolledImages.items ?? rawEnrolledImages.urls?.map((u: string, i: number) => ({
      url: u, image_id: `img_${i}`, quality_score: null, created_at: null,
    })) ?? [];
    return items.map((img: any) => ({
      ...img,
      url: img.url?.startsWith("/") ? `${apiBase}${img.url}` : img.url,
    }));
  })();

  const { data: stats } = useQuery({
    queryKey: ["person-stats", person?.person_id],
    queryFn:  () => analyticsApi.student(person!.person_id).then(r => r.data),
    enabled:  !!person,
  });

  useEffect(() => {
    if (person) {
      setEditForm({
        name:       person.name,
        department: person.department ?? "",
        email:      person.email      ?? "",
        phone:      person.phone      ?? "",
      });
    }
  }, [person]);

  const handleSave = async () => {
    if (!person) return;
    setSaving(true);
    try {
      await api.put(`/api/enrollment/${person.person_id}`, editForm);
      qc.invalidateQueries({ queryKey: ["enrollment-list"] });
      qc.invalidateQueries({ queryKey: ["person-detail", person.person_id] });
      toast.success("Profile updated");
    } catch {
      toast.error("Save failed");
    } finally {
      setSaving(false);
    }
  };

  const handleStatusToggle = async () => {
    if (!person) return;
    const newStatus = person.status === "ACTIVE" ? "INACTIVE" : "ACTIVE";
    try {
      await api.put(`/api/enrollment/${person.person_id}`, { status: newStatus });
      qc.invalidateQueries({ queryKey: ["enrollment-list"] });
      toast.success(`Status set to ${newStatus}`);
      onClose();
    } catch {
      toast.error("Status change failed");
    }
  };

  const handleDelete = async () => {
    if (!person) return;
    setDeleting(true);
    try {
      await enrollmentApi.delete(person.person_id);
      qc.invalidateQueries({ queryKey: ["enrollment-list"] });
      toast.success(`${person.name} deleted`);
      onClose();
    } catch {
      toast.error("Delete failed");
    } finally {
      setDeleting(false);
    }
  };

  if (!person) return null;

  const embeddingHistory: Array<{ version: number; quality_score: number; confidence_avg: number; created_at: string }> =
    detail?.embedding_history ?? (detail?.embedding ? [{
      version: detail.embedding.version ?? 1,
      quality_score: detail.embedding.quality_score ?? 0,
      confidence_avg: detail.embedding.confidence_avg ?? 0,
      created_at: detail.embedding.created_at,
    }] : []);

  const statCards = [
    { label: "Total Sightings",  value: stats?.total_sightings ?? "—",  icon: <Eye      className="h-4 w-4" /> },
    { label: "Sessions",         value: stats?.sessions_count  ?? "—",  icon: <BarChart3 className="h-4 w-4" /> },
    { label: "Avg Confidence",   value: stats?.avg_confidence  != null  ? `${(stats.avg_confidence * 100).toFixed(1)}%` : "—", icon: <TrendingUp className="h-4 w-4" /> },
    { label: "Last Seen",        value: person.last_seen ? fmtDatetime(person.last_seen) : "Never",  icon: <Clock      className="h-4 w-4" /> },
  ];

  return (
    <Dialog open={!!person} onOpenChange={open => !open && onClose()}>
      <DialogContent className="max-w-3xl max-h-[90vh] overflow-hidden flex flex-col p-0">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-800 shrink-0">
          <div className="flex items-center gap-3">
            <Avatar className="h-11 w-11">
              <AvatarImage src={person.thumbnail ?? undefined} />
              <AvatarFallback className="bg-zinc-800 text-zinc-300 font-semibold">
                {person.name.split(" ").map(n => n[0]).join("").slice(0, 2)}
              </AvatarFallback>
            </Avatar>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-base font-semibold text-zinc-100">{person.name}</h2>
                <span className={cn("text-[10px] font-semibold px-1.5 py-0.5 rounded", ROLE_COLOR[person.role])}>
                  {person.role}
                </span>
                <span className={cn("text-[10px] font-semibold px-1.5 py-0.5 rounded", STATUS_COLOR[person.status])}>
                  {person.status}
                </span>
              </div>
              <p className="text-xs text-zinc-500 mt-0.5">
                ID: <span className="font-mono text-zinc-400">{person.external_id}</span>
                {person.department && <> · {person.department}</>}
              </p>
            </div>
          </div>

          {canEdit && (
            <div className="flex items-center gap-2">
              <Button variant="outline" size="sm" className="h-7 text-xs gap-1" onClick={() => onReEnroll(person)}>
                <RotateCcw className="h-3 w-3" />
                Re-enroll
              </Button>
              <Button
                variant="outline"
                size="sm"
                className="h-7 text-xs gap-1 text-amber-400 border-amber-800 hover:bg-amber-950/40"
                onClick={handleStatusToggle}
              >
                <UserX className="h-3 w-3" />
                {person.status === "ACTIVE" ? "Deactivate" : "Activate"}
              </Button>
              <Button
                variant="outline"
                size="sm"
                className="h-7 text-xs gap-1 text-red-400 border-red-800 hover:bg-red-950/40"
                onClick={() => setConfirmDelete(true)}
              >
                <Trash2 className="h-3 w-3" />
                Delete
              </Button>
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="ghost" size="sm" className="h-7 w-7 p-0">
                    <MoreVertical className="h-4 w-4" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuItem onClick={handleStatusToggle}>
                    <UserX className="h-3.5 w-3.5 mr-2" />
                    {person.status === "ACTIVE" ? "Deactivate" : "Activate"}
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem
                    className="text-red-400 focus:text-red-400"
                    onClick={() => setConfirmDelete(true)}
                  >
                    <Trash2 className="h-3.5 w-3.5 mr-2" />
                    Delete person
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          )}
        </div>

        {/* Confirm delete */}
        {confirmDelete && (
          <div className="mx-6 mt-4 p-3 bg-red-950/40 border border-red-800 rounded-lg flex items-center justify-between text-sm">
            <span className="text-red-300">Delete <strong>{person.name}</strong> and all their data permanently?</span>
            <div className="flex gap-2 ml-4">
              <Button variant="outline" size="sm" className="h-6 text-xs" onClick={() => setConfirmDelete(false)}>Cancel</Button>
              <Button variant="destructive" size="sm" className="h-6 text-xs" disabled={deleting} onClick={handleDelete}>
                {deleting ? "Deleting…" : "Delete"}
              </Button>
            </div>
          </div>
        )}

        {/* Tabs */}
        <Tabs defaultValue="profile" className="flex-1 overflow-hidden flex flex-col">
          <TabsList className="mx-6 mt-3 shrink-0 self-start">
            <TabsTrigger value="profile">Profile</TabsTrigger>
            <TabsTrigger value="images">Images ({person.image_count})</TabsTrigger>
            <TabsTrigger value="embeddings">Embeddings</TabsTrigger>
            <TabsTrigger value="stats">Stats</TabsTrigger>
          </TabsList>

          <div className="flex-1 overflow-y-auto px-6 pb-6 pt-4">
            {/* Profile tab */}
            <TabsContent value="profile" className="mt-0">
              {isLoading ? (
                <div className="space-y-3">
                  {[...Array(4)].map((_, i) => (
                    <div key={i} className="h-9 bg-zinc-800 rounded-md animate-pulse" />
                  ))}
                </div>
              ) : (
                <div className="space-y-3 max-w-md">
                  {([
                    ["Full Name",   "name",       "text",  "e.g. Priya Sharma"       ],
                    ["Department",  "department",  "text",  "e.g. Computer Science"   ],
                    ["Email",       "email",       "email", "person@institution.edu"  ],
                    ["Phone",       "phone",       "tel",   "+91 98765 43210"         ],
                  ] as [string, keyof PersonForm, string, string][]).map(([label, key, type, placeholder]) => (
                    <div key={key}>
                      <label className="text-xs text-zinc-400 mb-1 block">{label}</label>
                      <Input
                        type={type}
                        placeholder={placeholder}
                        value={(editForm[key] as string) ?? ""}
                        onChange={e => setEditForm(f => ({ ...f, [key]: e.target.value }))}
                        disabled={!canEdit}
                        className="h-9"
                      />
                    </div>
                  ))}
                  {canEdit && (
                    <Button size="sm" onClick={handleSave} disabled={saving} className="mt-1">
                      {saving ? "Saving…" : "Save Changes"}
                    </Button>
                  )}
                </div>
              )}
            </TabsContent>

            {/* Images tab */}
            <TabsContent value="images" className="mt-0">
              {!enrolledImages ? (
                <div className="grid grid-cols-5 gap-2">
                  {[...Array(10)].map((_, i) => (
                    <div key={i} className="aspect-square bg-zinc-800 rounded-lg animate-pulse" />
                  ))}
                </div>
              ) : (enrolledImages as Array<{ url: string; image_id: string; quality_score: number | null; created_at: string }>).length === 0 ? (
                <div className="text-center py-10 text-zinc-600">
                  <ImageIcon className="h-8 w-8 mx-auto mb-2 text-zinc-700" />
                  No enrolled images found.
                </div>
              ) : (
                <div className="grid grid-cols-5 gap-2">
                  {(enrolledImages as Array<{ url: string; image_id: string; quality_score: number | null; created_at: string }>).map(img => (
                    <div key={img.image_id} className="group relative aspect-square rounded-lg overflow-hidden bg-zinc-900 border border-zinc-800">
                      <img src={img.url} alt="" className="w-full h-full object-cover" />
                      {img.quality_score != null && (
                        <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/70 to-transparent p-1">
                          <span className={cn("text-[9px] font-medium",
                            img.quality_score >= 0.8 ? "text-green-400" :
                            img.quality_score >= 0.65 ? "text-amber-400" : "text-red-400",
                          )}>
                            {(img.quality_score * 100).toFixed(0)}%
                          </span>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </TabsContent>

            {/* Embeddings tab */}
            <TabsContent value="embeddings" className="mt-0">
              {embeddingHistory.length === 0 ? (
                <div className="text-center py-10 text-zinc-600">
                  No embedding history available.
                </div>
              ) : (
                <div className="space-y-4">
                  <p className="text-xs text-zinc-500">
                    Embedding quality drift over {embeddingHistory.length} update{embeddingHistory.length !== 1 ? "s" : ""}.
                    EMA update threshold: 0.85 confidence · drift guard: 0.15.
                  </p>
                  <div className="h-44">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={embeddingHistory} margin={{ top: 4, right: 8, left: -16, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#3f3f46" />
                        <XAxis dataKey="version" tick={{ fontSize: 10, fill: "#71717a" }} label={{ value: "Version", position: "insideBottom", offset: -2, fontSize: 10, fill: "#52525b" }} />
                        <YAxis domain={[0.5, 1]} tickCount={4} tick={{ fontSize: 10, fill: "#71717a" }} tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
                        <Tooltip
                          contentStyle={{ background: "#18181b", border: "1px solid #3f3f46", borderRadius: 6, fontSize: 11 }}
                          labelFormatter={v => `Version ${v}`}
                          formatter={(v: number, n: string) => [`${(v * 100).toFixed(1)}%`, n === "quality_score" ? "Quality" : "Confidence"]}
                        />
                        <Line type="monotone" dataKey="quality_score"  stroke="#22c55e" strokeWidth={2} dot={{ r: 3 }} name="quality_score"  />
                        <Line type="monotone" dataKey="confidence_avg" stroke="#818cf8" strokeWidth={2} dot={{ r: 3 }} name="confidence_avg" strokeDasharray="4 2" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="space-y-2">
                    {embeddingHistory.map((e) => (
                      <div key={e.version} className="flex items-center gap-3 text-xs py-2 border-b border-zinc-800/60 last:border-0">
                        <span className="text-zinc-500 w-16 shrink-0">v{e.version}</span>
                        <QualityBar score={e.quality_score} />
                        <span className="text-zinc-500 ml-auto">{fmtDatetime(e.created_at)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </TabsContent>

            {/* Stats tab */}
            <TabsContent value="stats" className="mt-0">
              <div className="grid grid-cols-2 gap-3 mb-5">
                {statCards.map(s => (
                  <div key={s.label} className="bg-zinc-900 rounded-xl p-3.5 flex items-center gap-3">
                    <div className="h-8 w-8 rounded-lg bg-zinc-800 flex items-center justify-center text-zinc-400 shrink-0">
                      {s.icon}
                    </div>
                    <div>
                      <p className="text-lg font-semibold text-zinc-100 leading-none">{String(s.value)}</p>
                      <p className="text-[10px] text-zinc-500 mt-0.5">{s.label}</p>
                    </div>
                  </div>
                ))}
              </div>

              {stats?.recent_sessions && (
                <div>
                  <p className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider mb-2">Recent Session Attendance</p>
                  <div className="space-y-1.5">
                    {(stats.recent_sessions as Array<{ course_name: string; date: string; status: string }>).map((s, i) => (
                      <div key={i} className="flex items-center justify-between text-xs py-1.5 border-b border-zinc-800/60 last:border-0">
                        <span className="text-zinc-400 truncate max-w-[60%]">{s.course_name}</span>
                        <span className="text-zinc-500">{s.date}</span>
                        <span className={cn("font-semibold",
                          s.status === "P"  ? "text-green-400" :
                          s.status === "L"  ? "text-amber-400" :
                          s.status === "EE" ? "text-orange-400" :
                          s.status === "A"  ? "text-red-400"   : "text-zinc-500",
                        )}>
                          {s.status}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </TabsContent>
          </div>
        </Tabs>
      </DialogContent>
    </Dialog>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

const PAGE_SIZE = 25;

export default function EnrollmentPage() {
  const { isViewer, isSuperAdmin, clientId: jwtClientId } = useAuth();
  const qc = useQueryClient();
  const canEdit = !isViewer;

  const [selectedClientId, setSelectedClientId] = useState<string>("");
  const effectiveClientId = isSuperAdmin ? selectedClientId : (jwtClientId ?? "");

  const [search,         setSearch]         = useState("");
  const [roleFilter,     setRoleFilter]     = useState<PersonRole | "ALL">("ALL");
  const [statusFilter,   setStatusFilter]   = useState<PersonStatus | "ALL">("ALL");
  const [datasetFilter,  setDatasetFilter]  = useState<string>("");
  const [sortBy,         setSortBy]         = useState<SortKey>("name");
  const [sortDir,        setSortDir]        = useState<"asc" | "desc">("asc");
  const [page,           setPage]           = useState(1);
  const [sheetOpen,      setSheetOpen]      = useState(false);
  const [selected,       setSelected]       = useState<EnrolledPerson | null>(null);
  const [reEnrollPerson, setReEnrollPerson] = useState<EnrolledPerson | null>(null);

  // SUPER_ADMIN: load clients for the picker
  const { data: clientsData } = useQuery({
    queryKey: ["admin-clients"],
    queryFn:  () => adminApi.clients().then(r => r.data),
    enabled:  isSuperAdmin,
    staleTime: 120_000,
  });
  const clients: Array<{ client_id: string; name: string; slug: string }> =
    (clientsData?.items ?? clientsData ?? []).map((c: any) => ({ client_id: c.client_id, name: c.name, slug: c.slug }));

  // Auto-select first client
  useEffect(() => {
    if (isSuperAdmin && !selectedClientId && clients.length > 0) {
      setSelectedClientId(clients[0].client_id);
    }
  }, [isSuperAdmin, selectedClientId, clients]);

  const { data: datasetsData } = useQuery({
    queryKey: ["datasets", effectiveClientId],
    queryFn:  () => datasetsApi.list(
      effectiveClientId ? { client_id: effectiveClientId } : undefined,
    ).then(r => r.data),
    enabled: !!effectiveClientId,
  });
  const datasets: Array<{ dataset_id: string; name: string; color: string }> =
    (datasetsData as { items?: Array<{ dataset_id: string; name: string; color: string }> })?.items ?? [];

  const { data, isLoading, isFetching, refetch } = useQuery({
    queryKey: ["enrollment-list", page, search, roleFilter, statusFilter, datasetFilter, sortBy, sortDir, effectiveClientId],
    queryFn:  () => enrollmentApi.list({
      limit:      PAGE_SIZE,
      offset:     (page - 1) * PAGE_SIZE,
      q:          search        || undefined,
      role:       roleFilter    !== "ALL" ? roleFilter    : undefined,
      status:     statusFilter  !== "ALL" ? statusFilter  : undefined,
      dataset_id: datasetFilter || undefined,
      sort_by:    sortBy,
      sort_dir:   sortDir,
      ...(isSuperAdmin && effectiveClientId ? { client_id: effectiveClientId } : {}),
    }).then(r => r.data as PaginatedResponse<EnrolledPerson> & { max_persons: number }),
    enabled: !!effectiveClientId,
  });

  const persons    = data?.items ?? [];
  const total      = data?.total ?? 0;
  const maxPersons = (data as (typeof data & { max_persons?: number }))?.max_persons ?? 10_000;
  const totalPages = Math.ceil(total / PAGE_SIZE);

  const toggleSort = (key: SortKey) => {
    setSortBy(key);
    setSortDir(d => sortBy === key ? (d === "asc" ? "desc" : "asc") : "asc");
    setPage(1);
  };

  const handleSearchChange = (v: string) => { setSearch(v); setPage(1); };

  return (
    <div className="p-6 max-w-7xl mx-auto">
      {/* SUPER_ADMIN client picker */}
      {isSuperAdmin && (
        <div className="flex items-center gap-3 mb-4 p-3 rounded-lg border border-border bg-muted/30">
          <span className="text-xs text-muted-foreground whitespace-nowrap">Client:</span>
          <select
            className="h-8 rounded-md border border-border bg-background px-3 text-sm flex-1 max-w-xs"
            value={selectedClientId}
            onChange={e => { setSelectedClientId(e.target.value); setPage(1); }}
          >
            <option value="">Select a client…</option>
            {clients.map(c => (
              <option key={c.client_id} value={c.client_id}>
                {c.name} ({c.slug})
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Page header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-xl font-bold text-zinc-100">Enrollment</h1>
          <p className="text-sm text-zinc-500 mt-0.5">
            {total.toLocaleString()}&thinsp;/&thinsp;{maxPersons.toLocaleString()} persons
            <span className="text-zinc-600"> (limit)</span>
            {isFetching && <RefreshCw className="inline h-3 w-3 ml-2 animate-spin text-zinc-600" />}
          </p>
        </div>
        {canEdit && (
          <Button onClick={() => setSheetOpen(true)} className="gap-1.5">
            <Plus className="h-4 w-4" />
            New Enrollment
          </Button>
        )}
      </div>

      {/* Guidelines */}
      <GuidelinesPanel />

      {/* Filters */}
      <div className="flex flex-wrap gap-2 mb-4">
        <div className="relative flex-1 min-w-48">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-zinc-500" />
          <Input
            placeholder="Search name, ID, department…"
            value={search}
            onChange={e => handleSearchChange(e.target.value)}
            className="pl-9 h-9"
          />
        </div>

        <select
          value={roleFilter}
          onChange={e => { setRoleFilter(e.target.value as PersonRole | "ALL"); setPage(1); }}
          className="h-9 rounded-md border border-zinc-800 bg-zinc-950 text-sm px-2.5 text-zinc-300 focus:outline-none focus:ring-1 focus:ring-primary"
        >
          <option value="ALL">All roles</option>
          <option value="STUDENT">Student</option>
          <option value="FACULTY">Faculty</option>
          <option value="ADMIN">Admin</option>
        </select>

        <select
          value={statusFilter}
          onChange={e => { setStatusFilter(e.target.value as PersonStatus | "ALL"); setPage(1); }}
          className="h-9 rounded-md border border-zinc-800 bg-zinc-950 text-sm px-2.5 text-zinc-300 focus:outline-none focus:ring-1 focus:ring-primary"
        >
          <option value="ALL">All statuses</option>
          <option value="ACTIVE">Active</option>
          <option value="INACTIVE">Inactive</option>
          <option value="SUSPENDED">Suspended</option>
        </select>

        {datasets.length > 0 && (
          <select
            value={datasetFilter}
            onChange={e => { setDatasetFilter(e.target.value); setPage(1); }}
            className="h-9 rounded-md border border-zinc-800 bg-zinc-950 text-sm px-2.5 text-zinc-300 focus:outline-none focus:ring-1 focus:ring-primary"
          >
            <option value="">All datasets</option>
            {datasets.map(d => (
              <option key={d.dataset_id} value={d.dataset_id}>{d.name}</option>
            ))}
          </select>
        )}

        <Button variant="outline" size="sm" className="h-9 gap-1" onClick={() => refetch()}>
          <RefreshCw className="h-3.5 w-3.5" />
          Refresh
        </Button>

        <Button variant="outline" size="sm" className="h-9 gap-1">
          <Download className="h-3.5 w-3.5" />
          Export
        </Button>
      </div>

      {/* Table */}
      <div className="border border-zinc-800 rounded-xl overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-zinc-900 border-b border-zinc-800">
              <tr>
                <th className="px-4 py-3 text-left w-10" />
                <Th label="Name"       sortKey="name"         current={sortBy} dir={sortDir} onClick={toggleSort} />
                <Th label="Role"       sortKey="role"         current={sortBy} dir={sortDir} onClick={toggleSort} />
                <th className="px-4 py-3 text-left text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">
                  Department
                </th>
                <Th label="Quality"    sortKey="quality_score" current={sortBy} dir={sortDir} onClick={toggleSort} />
                <Th label="Images"     sortKey="image_count"  current={sortBy} dir={sortDir} onClick={toggleSort} />
                <th className="px-4 py-3 text-left text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">
                  Status
                </th>
                <Th label="Last Seen"  sortKey="last_seen"    current={sortBy} dir={sortDir} onClick={toggleSort} />
                <th className="px-4 py-3 w-10" />
              </tr>
            </thead>

            <tbody className="divide-y divide-zinc-800/60">
              {isLoading
                ? [...Array(8)].map((_, i) => (
                    <tr key={i}>
                      {[...Array(9)].map((__, j) => (
                        <td key={j} className="px-4 py-3">
                          <div className="h-4 bg-zinc-800 rounded animate-pulse" style={{ width: `${50 + (i * j) % 40}%` }} />
                        </td>
                      ))}
                    </tr>
                  ))
                : persons.length === 0
                ? (
                    <tr>
                      <td colSpan={9} className="text-center py-16 text-zinc-600">
                        <User className="h-8 w-8 mx-auto mb-2 text-zinc-700" />
                        {search || roleFilter !== "ALL" || statusFilter !== "ALL"
                          ? "No persons match your filters."
                          : "No persons enrolled yet. Click \u201cNew Enrollment\u201d to get started."}
                      </td>
                    </tr>
                  )
                : persons.map(p => (
                    <tr
                      key={p.person_id}
                      onClick={() => setSelected(p)}
                      className="hover:bg-zinc-900/50 cursor-pointer transition-colors group"
                    >
                      {/* Thumbnail */}
                      <td className="px-4 py-3">
                        <Avatar className="h-8 w-8">
                          <AvatarImage src={p.thumbnail ?? undefined} />
                          <AvatarFallback className="bg-zinc-800 text-zinc-400 text-[10px] font-semibold">
                            {p.name.split(" ").map(n => n[0]).join("").slice(0, 2)}
                          </AvatarFallback>
                        </Avatar>
                      </td>
                      {/* Name + ID */}
                      <td className="px-4 py-3">
                        <p className="text-sm font-medium text-zinc-200">{p.name}</p>
                        <p className="text-[10px] text-zinc-500 font-mono mt-0.5">{p.external_id}</p>
                      </td>
                      {/* Role */}
                      <td className="px-4 py-3">
                        <span className={cn("text-[10px] font-semibold px-1.5 py-0.5 rounded", ROLE_COLOR[p.role])}>
                          {p.role}
                        </span>
                      </td>
                      {/* Dept */}
                      <td className="px-4 py-3 text-sm text-zinc-400 max-w-[150px] truncate">
                        {p.department ?? <span className="text-zinc-700">—</span>}
                      </td>
                      {/* Quality */}
                      <td className="px-4 py-3">
                        <QualityBar score={p.quality_score} />
                      </td>
                      {/* Image count */}
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-1.5 text-sm text-zinc-400">
                          <ImageIcon className="h-3.5 w-3.5 text-zinc-600" />
                          {p.image_count}
                        </div>
                      </td>
                      {/* Status */}
                      <td className="px-4 py-3">
                        <span className={cn("text-[10px] font-semibold px-1.5 py-0.5 rounded", STATUS_COLOR[p.status])}>
                          {p.status}
                        </span>
                      </td>
                      {/* Last seen */}
                      <td className="px-4 py-3 text-xs text-zinc-500 whitespace-nowrap">
                        {p.last_seen ? fmtDatetime(p.last_seen) : <span className="text-zinc-700">Never</span>}
                      </td>
                      {/* Actions */}
                      <td className="px-4 py-3" onClick={e => e.stopPropagation()}>
                        {canEdit && (
                          <DropdownMenu>
                            <DropdownMenuTrigger asChild>
                              <Button
                                variant="ghost"
                                size="sm"
                                className="h-7 w-7 p-0 opacity-0 group-hover:opacity-100 transition-opacity"
                              >
                                <MoreVertical className="h-4 w-4" />
                              </Button>
                            </DropdownMenuTrigger>
                            <DropdownMenuContent align="end">
                              <DropdownMenuItem onClick={() => setSelected(p)}>
                                <Eye className="h-3.5 w-3.5 mr-2" />
                                View details
                              </DropdownMenuItem>
                              <DropdownMenuItem onClick={() => setReEnrollPerson(p)}>
                                <RotateCcw className="h-3.5 w-3.5 mr-2" />
                                Re-enroll
                              </DropdownMenuItem>
                              <DropdownMenuSeparator />
                              <DropdownMenuItem
                                className="text-red-400 focus:text-red-400"
                                onClick={async () => {
                                  if (confirm(`Delete "${p.name}" and all their data permanently?`)) {
                                    try {
                                      await enrollmentApi.delete(p.person_id);
                                      qc.invalidateQueries({ queryKey: ["enrollment-list"] });
                                      toast.success(`${p.name} deleted`);
                                    } catch { toast.error("Delete failed"); }
                                  }
                                }}
                              >
                                <Trash2 className="h-3.5 w-3.5 mr-2" />
                                Delete
                              </DropdownMenuItem>
                            </DropdownMenuContent>
                          </DropdownMenu>
                        )}
                      </td>
                    </tr>
                  ))
              }
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
              <Button
                variant="outline"
                size="sm"
                className="h-7 w-7 p-0"
                disabled={page <= 1}
                onClick={() => setPage(p => p - 1)}
              >
                <ChevronLeft className="h-3.5 w-3.5" />
              </Button>
              <span className="text-xs text-zinc-400 px-2 tabular-nums">
                {page} / {totalPages}
              </span>
              <Button
                variant="outline"
                size="sm"
                className="h-7 w-7 p-0"
                disabled={page >= totalPages}
                onClick={() => setPage(p => p + 1)}
              >
                <ChevronDown className="h-3.5 w-3.5 -rotate-90" />
              </Button>
            </div>
          </div>
        )}
      </div>

      {/* New enrollment sheet */}
      <NewEnrollmentSheet
        open={sheetOpen}
        onClose={() => setSheetOpen(false)}
        onSuccess={() => setSheetOpen(false)}
        clientId={effectiveClientId}
      />

      {/* Person detail dialog */}
      <PersonDetailDialog
        person={selected}
        onClose={() => setSelected(null)}
        canEdit={canEdit}
        onReEnroll={p => { setSelected(null); setReEnrollPerson(p); }}
      />

      {/* Re-enroll dialog */}
      <ReEnrollDialog
        person={reEnrollPerson}
        onClose={() => setReEnrollPerson(null)}
      />
    </div>
  );
}
