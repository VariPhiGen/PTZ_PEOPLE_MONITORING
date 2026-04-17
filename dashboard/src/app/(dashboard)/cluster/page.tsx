"use client";

import {
  useState, useEffect, useRef, useCallback, useMemo,
} from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { format, formatDistanceToNow } from "date-fns";
import { toast } from "sonner";
import {
  AlertTriangle, ArrowRight, BarChart3,
  Check, ChevronDown, ChevronRight, Clock, Cpu,
  Database, Download, Grip, Layers, Loader2, Lock,
  MapPin, Plus, RefreshCw, Save, Server, Settings2,
  Trash2, TrendingDown, TrendingUp, Minus, X, Zap,
} from "lucide-react";
import { api } from "@/lib/api";
import { useAuth } from "@/hooks/useAuth";
import { cn, fmtDatetime } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Badge }  from "@/components/ui/badge";
import type { GpuNode, Camera } from "@/types";

// ── Extended types ────────────────────────────────────────────────────────────

interface RichNode extends GpuNode {
  connectivity:      "PUBLIC_IP" | "CLOUDFLARE_TUNNEL";
  latency_ms:        number;
  uptime_pct:        number;
  gpu_pct:           number;
  cpu_pct:           number;
  ram_pct:           number;
  vram_used_gb:      number;
  vram_total_gb:     number;
  faces_db_size:     number;
  throughput_fps:    number;
  assigned_clients:  Array<{ client_id: string; name: string; slug: string }>;
}

interface MigrationEvent {
  migration_id:   string;
  camera_id:      string;
  camera_name:    string;
  from_node_id:   string;
  from_node_name: string;
  to_node_id:     string;
  to_node_name:   string;
  reason:         "MANUAL" | "NODE_FAILURE" | "AUTO_BALANCE" | "DRAIN";
  status:         "COMPLETED" | "FAILED" | "IN_PROGRESS";
  created_at:     string;
}

interface AffinityRule {
  id:                  string;
  type:                "CAMERA" | "CLIENT";
  entity_id:           string;
  entity_name:         string;
  preferred_node_id:   string;
  preferred_node_name: string;
  hard:                boolean;
}

interface BalanceChange {
  camera_id:   string;
  camera_name: string;
  from_node:   string;
  to_node:     string;
  reason:      string;
}

interface AutoBalancePreview {
  changes:          BalanceChange[];
  active_sessions:  number;
  estimated_seconds: number;
}

// ── Constants ─────────────────────────────────────────────────────────────────

const NODE_STATUS = {
  ONLINE:   { label: "Online",   cls: "text-green-400 bg-green-900/30 border-green-800"   },
  OFFLINE:  { label: "Offline",  cls: "text-red-400   bg-red-900/30   border-red-800"     },
  DRAINING: { label: "Draining", cls: "text-amber-400 bg-amber-900/30 border-amber-800 animate-pulse" },
};

const MIGRATION_STATUS = {
  COMPLETED:   { cls: "text-green-400 bg-green-900/30"  },
  FAILED:      { cls: "text-red-400   bg-red-900/30"    },
  IN_PROGRESS: { cls: "text-blue-400  bg-blue-900/30"   },
};

const MIGRATION_REASON_LABEL: Record<string, string> = {
  MANUAL:       "Manual move",
  NODE_FAILURE: "Node failure",
  AUTO_BALANCE: "Auto-balance",
  DRAIN:        "Node drain",
};

const HISTORY_LEN = 20;

// ── Helpers ───────────────────────────────────────────────────────────────────

function trend(history: number[]): "up" | "down" | "flat" {
  if (history.length < 4) return "flat";
  const recent = history.slice(-4);
  const delta  = recent[recent.length - 1] - recent[0];
  if (delta > 3)  return "up";
  if (delta < -3) return "down";
  return "flat";
}

function TrendIcon({ dir }: { dir: "up" | "down" | "flat" }) {
  if (dir === "up")   return <TrendingUp   className="h-2.5 w-2.5 text-red-400"    />;
  if (dir === "down") return <TrendingDown className="h-2.5 w-2.5 text-green-400"  />;
  return                      <Minus        className="h-2.5 w-2.5 text-zinc-600"  />;
}

// ── GPU bar ───────────────────────────────────────────────────────────────────

function GpuBar({ pct, label }: { pct: number; label?: string }) {
  const color = pct >= 85 ? "bg-red-500" : pct >= 65 ? "bg-amber-500" : "bg-green-500";
  const text  = pct >= 85 ? "text-red-400" : pct >= 65 ? "text-amber-400" : "text-green-400";
  return (
    <div className="space-y-0.5">
      {label && (
        <div className="flex justify-between text-[9px]">
          <span className="text-zinc-500">{label}</span>
          <span className={cn("font-bold tabular-nums", text)}>{pct.toFixed(0)}%</span>
        </div>
      )}
      <div className="h-1.5 rounded-full bg-zinc-800 overflow-hidden">
        <div className={cn("h-full rounded-full transition-all duration-700", color)} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

// ── Real-time Sparkline ───────────────────────────────────────────────────────

function Spark({
  data, color = "#3b82f6", w = 80, h = 26, fill = false,
}: {
  data: number[]; color?: string; w?: number; h?: number; fill?: boolean;
}) {
  if (data.length < 2) return <div style={{ width: w, height: h }} />;
  const max  = Math.max(...data, 1);
  const min  = Math.min(...data);
  const span = max - min || 1;
  const pts  = data
    .map((v, i) => `${(i / (data.length - 1)) * w},${h - ((v - min) / span) * (h - 2) - 1}`)
    .join(" ");
  const fillPath = fill
    ? `M0,${h} ` + data.map((v, i) =>
        `L${(i / (data.length - 1)) * w},${h - ((v - min) / span) * (h - 2) - 1}`
      ).join(" ") + ` L${w},${h} Z`
    : undefined;

  return (
    <svg width={w} height={h} className="overflow-visible">
      {fill && fillPath && (
        <path d={fillPath} fill={color} fillOpacity={0.12} />
      )}
      <polyline points={pts} fill="none" stroke={color} strokeWidth={1.5} strokeLinecap="round" />
      {/* Current-value dot */}
      <circle
        cx={(data.length - 1) / (data.length - 1) * w}
        cy={h - ((data[data.length - 1] - min) / span) * (h - 2) - 1}
        r={2.5}
        fill={color}
      />
    </svg>
  );
}

// ── Node Health Sparklines (expanded panel) ────────────────────────────────

function NodeHealthSparklines({
  gpuH, cpuH, ramH, latH,
  gpu, cpu, ram, lat,
}: {
  gpuH: number[]; cpuH: number[]; ramH: number[]; latH: number[];
  gpu: number; cpu: number; ram: number; lat: number;
}) {
  const metrics = [
    { label: "GPU", value: `${gpu.toFixed(0)}%`,   data: gpuH, color: gpu >= 85 ? "#ef4444" : gpu >= 65 ? "#f59e0b" : "#22c55e", fill: true },
    { label: "CPU", value: `${cpu.toFixed(0)}%`,   data: cpuH, color: "#3b82f6", fill: true },
    { label: "RAM", value: `${ram.toFixed(0)}%`,   data: ramH, color: "#a855f7", fill: true },
    { label: "RTT", value: `${lat.toFixed(0)}ms`,  data: latH, color: "#f59e0b", fill: false },
  ];

  return (
    <div className="grid grid-cols-4 gap-3 pb-3 border-b border-zinc-800/60">
      {metrics.map(({ label, value, data, color, fill }) => (
        <div key={label} className="space-y-1">
          <div className="flex items-center justify-between text-[9px]">
            <span className="text-zinc-600">{label}</span>
            <div className="flex items-center gap-0.5">
              <TrendIcon dir={trend(data)} />
              <span className="font-bold tabular-nums" style={{ color }}>{value}</span>
            </div>
          </div>
          <Spark data={data} color={color} w={56} h={22} fill={fill} />
        </div>
      ))}
    </div>
  );
}

// ── Migration progress stepper ────────────────────────────────────────────────

const MIGRATION_STEPS = [
  "Stopping session",
  "Migrating FAISS",
  "Starting on node",
  "Verifying health",
];

function MigrationStepper({ createdAt }: { createdAt: string }) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    const start = new Date(createdAt).getTime();
    const id    = setInterval(() => setElapsed((Date.now() - start) / 1000), 1000);
    return () => clearInterval(id);
  }, [createdAt]);

  // Estimate current step based on elapsed time
  // Steps take roughly: 0-15s, 15-45s, 45-75s, 75s+
  const stepBoundaries = [0, 15, 45, 75];
  const currentStep    = stepBoundaries.findLastIndex(b => elapsed >= b);
  const progress       = Math.min(100, (elapsed / 90) * 100);

  return (
    <div className="mt-1.5 space-y-2">
      {/* Overall progress bar */}
      <div className="flex items-center gap-2">
        <div className="flex-1 h-1 rounded-full bg-zinc-800 overflow-hidden">
          <div
            className="h-full bg-blue-500 rounded-full transition-all duration-1000"
            style={{ width: `${progress}%` }}
          />
        </div>
        <span className="text-[9px] font-mono text-zinc-500 tabular-nums">{elapsed.toFixed(0)}s</span>
      </div>
      {/* Step indicators */}
      <div className="flex items-center gap-0">
        {MIGRATION_STEPS.map((step, i) => {
          const done    = i < currentStep;
          const active  = i === currentStep;
          const pending = i > currentStep;
          return (
            <div key={step} className="flex items-center flex-1">
              <div className="flex flex-col items-center gap-0.5 flex-1">
                <div className={cn(
                  "h-4 w-4 rounded-full flex items-center justify-center text-[8px] font-bold",
                  done    && "bg-blue-600 text-white",
                  active  && "bg-blue-500 text-white animate-pulse ring-2 ring-blue-500/30",
                  pending && "bg-zinc-800 text-zinc-600",
                )}>
                  {done ? <Check className="h-2.5 w-2.5" /> : i + 1}
                </div>
                <span className={cn(
                  "text-[8px] text-center leading-tight max-w-[48px] hidden sm:block",
                  active  && "text-blue-400",
                  done    && "text-zinc-500",
                  pending && "text-zinc-700",
                )}>
                  {step}
                </span>
              </div>
              {i < MIGRATION_STEPS.length - 1 && (
                <div className={cn(
                  "flex-1 h-px mb-3",
                  i < currentStep ? "bg-blue-600" : "bg-zinc-800",
                )} />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Summary bar ───────────────────────────────────────────────────────────────

function SummaryBar({ nodes, activeMigrations }: { nodes: RichNode[]; activeMigrations: number }) {
  const online  = nodes.filter(n => n.status === "ONLINE").length;
  const cameras = nodes.reduce((s, n) => s + n.active_cameras, 0);
  const vram    = nodes.reduce((s, n) => s + n.vram_total_gb, 0);
  const fps     = nodes.reduce((s, n) => s + n.throughput_fps, 0);
  const faceDB  = nodes.reduce((s, n) => s + n.faces_db_size, 0);

  const items = [
    { label: "Nodes Online",    value: `${online}/${nodes.length}`, icon: Server,   color: "text-green-400" },
    { label: "Active Cameras",  value: cameras,                      icon: Layers,   color: "text-blue-400"  },
    { label: "Total GPU VRAM",  value: `${vram.toFixed(0)} GB`,      icon: Cpu,      color: "text-violet-400"},
    { label: "Throughput",      value: `${fps.toFixed(0)} fps`,      icon: Zap,      color: "text-amber-400" },
    { label: "Face DB",         value: faceDB > 1_000 ? `${(faceDB / 1_000).toFixed(1)}K` : String(faceDB), icon: Database, color: "text-teal-400" },
  ];

  return (
    <div className="space-y-2 shrink-0">
      {activeMigrations > 0 && (
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg border border-blue-800 bg-blue-900/15 text-xs text-blue-300">
          <Loader2 className="h-3.5 w-3.5 animate-spin shrink-0" />
          <span className="font-semibold">{activeMigrations} camera migration{activeMigrations > 1 ? "s" : ""} in progress</span>
          <span className="text-blue-500">— sessions will resume automatically</span>
        </div>
      )}
      <div className="grid grid-cols-5 gap-3">
        {items.map(({ label, value, icon: Icon, color }) => (
          <div key={label} className="rounded-xl border border-zinc-800 bg-zinc-900 p-3.5 flex items-center gap-3">
            <Icon className={cn("h-5 w-5 shrink-0", color)} />
            <div>
              <p className={cn("text-xl font-black tabular-nums leading-none", color)}>{value}</p>
              <p className="text-[10px] text-zinc-500 mt-0.5">{label}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Node card ─────────────────────────────────────────────────────────────────

function NodeCard({
  node, cameras, activeMigrations, readOnly, onRefresh,
}: {
  node:             RichNode;
  cameras:          Camera[];
  activeMigrations: MigrationEvent[];
  readOnly:         boolean;
  onRefresh:        () => void;
}) {
  const qc = useQueryClient();
  const [expanded,     setExpanded]     = useState(false);
  const [draining,     setDraining]     = useState(false);
  const [removing,     setRemoving]     = useState(false);
  const [assigning,    setAssigning]    = useState(false);
  const [assignSelect, setAssignSelect] = useState("");

  // ── Real rolling metric histories ──────────────────────────────────────────
  // Seeded with HISTORY_LEN copies of the initial value so sparklines look good
  // immediately. Each history grows by one point whenever the query refetches
  // and produces a new node.gpu_pct / cpu_pct / ram_pct / latency_ms value.
  const [gpuH, setGpuH] = useState<number[]>(() => Array(HISTORY_LEN).fill(node.gpu_pct));
  const [cpuH, setCpuH] = useState<number[]>(() => Array(HISTORY_LEN).fill(node.cpu_pct));
  const [ramH, setRamH] = useState<number[]>(() => Array(HISTORY_LEN).fill(node.ram_pct));
  const [latH, setLatH] = useState<number[]>(() => Array(HISTORY_LEN).fill(node.latency_ms));

  // Append a new sample whenever the node health values change (parent query
  // refetches every 15 s).
  const prevKey = useRef(`${node.gpu_pct}-${node.cpu_pct}-${node.ram_pct}-${node.latency_ms}`);
  useEffect(() => {
    const key = `${node.gpu_pct}-${node.cpu_pct}-${node.ram_pct}-${node.latency_ms}`;
    if (key === prevKey.current) return;
    prevKey.current = key;
    setGpuH(h => [...h.slice(1 - HISTORY_LEN), node.gpu_pct]);
    setCpuH(h => [...h.slice(1 - HISTORY_LEN), node.cpu_pct]);
    setRamH(h => [...h.slice(1 - HISTORY_LEN), node.ram_pct]);
    setLatH(h => [...h.slice(1 - HISTORY_LEN), node.latency_ms]);
  }, [node.gpu_pct, node.cpu_pct, node.ram_pct, node.latency_ms]);

  const nodeCameras    = cameras.filter(c => c.node_id === node.node_id);
  const unassigned     = cameras.filter(c => !c.node_id);
  const cfg            = NODE_STATUS[node.status] ?? NODE_STATUS.OFFLINE;
  const inMigration    = activeMigrations.filter(
    m => m.to_node_id === node.node_id || m.from_node_id === node.node_id,
  );
  const heartbeatAge   = (Date.now() - node.last_heartbeat * 1000) / 1000;
  const heartbeatOk    = heartbeatAge < 60;

  const handleDrain = async () => {
    if (!confirm(`Drain node "${node.node_name}"? All cameras will be migrated.`)) return;
    setDraining(true);
    try {
      await api.put(`/api/admin/nodes/${node.node_id}/drain`);
      toast.success("Node set to draining — cameras migrating…");
      qc.invalidateQueries({ queryKey: ["cluster-nodes"] });
      onRefresh();
    } catch { toast.error("Drain failed"); }
    finally { setDraining(false); }
  };

  const handleRemove = async () => {
    if (nodeCameras.length > 0) { toast.error("Cannot remove node with active cameras"); return; }
    if (!confirm(`Remove node "${node.node_name}" from the cluster?`)) return;
    setRemoving(true);
    try {
      await api.delete(`/api/admin/nodes/${node.node_id}`);
      toast.success("Node removed");
      qc.invalidateQueries({ queryKey: ["cluster-nodes"] });
    } catch { toast.error("Remove failed"); }
    finally { setRemoving(false); }
  };

  const handleAssign = async () => {
    if (!assignSelect) return;
    setAssigning(true);
    try {
      await api.post(`/api/node/cameras/${assignSelect}/assign`, { node_id: node.node_id });
      toast.success("Camera assigned");
      qc.invalidateQueries({ queryKey: ["all-cameras"] });
      setAssignSelect("");
    } catch { toast.error("Assignment failed"); }
    finally { setAssigning(false); }
  };

  const handleRemoveCamera = async (cameraId: string) => {
    try {
      await api.post(`/api/node/cameras/${cameraId}/unassign`, { node_id: node.node_id });
      qc.invalidateQueries({ queryKey: ["all-cameras"] });
      toast.success("Camera removed from node");
    } catch { toast.error("Removal failed"); }
  };

  return (
    <div className={cn(
      "rounded-xl border transition-all",
      node.status === "ONLINE"   ? "border-zinc-800 bg-zinc-950" :
      node.status === "DRAINING" ? "border-amber-800/40 bg-amber-950/10" :
      "border-zinc-800 bg-zinc-950 opacity-60",
      expanded && "border-zinc-700",
      inMigration.length > 0 && "ring-1 ring-blue-800/50",
    )}>
      {/* Card header */}
      <button className="w-full text-left p-4 space-y-3" onClick={() => setExpanded(p => !p)}>
        {/* Row 1: name + status */}
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-center gap-2 min-w-0">
            <span className={cn("h-2 w-2 rounded-full shrink-0 mt-0.5",
              node.status === "ONLINE"   ? "bg-green-400 animate-pulse" :
              node.status === "DRAINING" ? "bg-amber-400 animate-pulse" : "bg-zinc-600",
            )} />
            <div className="min-w-0">
              <p className="text-sm font-bold text-zinc-100 truncate">{node.node_name}</p>
              {node.location && (
                <p className="text-[10px] text-zinc-500 flex items-center gap-1 mt-0.5">
                  <MapPin className="h-2.5 w-2.5" />{node.location}
                </p>
              )}
            </div>
          </div>
          <div className="flex items-center gap-1.5 shrink-0">
            {inMigration.length > 0 && (
              <span className="flex items-center gap-0.5 text-[9px] font-bold px-1.5 py-0.5 rounded text-blue-400 bg-blue-900/30 border border-blue-800">
                <Loader2 className="h-2.5 w-2.5 animate-spin" />
                {inMigration.length}
              </span>
            )}
            <span className={cn("text-[9px] font-bold px-1.5 py-0.5 rounded border", cfg.cls)}>
              {cfg.label}
            </span>
            <span className={cn("text-[9px] font-semibold px-1 py-0.5 rounded",
              node.connectivity === "CLOUDFLARE_TUNNEL"
                ? "text-orange-400 bg-orange-900/30"
                : "text-blue-400 bg-blue-900/30",
            )}>
              {node.connectivity === "CLOUDFLARE_TUNNEL" ? "CF Tunnel" : "Public IP"}
            </span>
            {expanded
              ? <ChevronDown className="h-3.5 w-3.5 text-zinc-500" />
              : <ChevronRight className="h-3.5 w-3.5 text-zinc-500" />}
          </div>
        </div>

        {/* Inline mini sparklines — always visible */}
        <div className="grid grid-cols-3 gap-2">
          {[
            { label: "GPU", data: gpuH, color: node.gpu_pct >= 85 ? "#ef4444" : node.gpu_pct >= 65 ? "#f59e0b" : "#22c55e", pct: node.gpu_pct },
            { label: "CPU", data: cpuH, color: "#3b82f6", pct: node.cpu_pct },
            { label: "RAM", data: ramH, color: "#a855f7", pct: node.ram_pct },
          ].map(({ label, data, color, pct }) => (
            <div key={label} className="space-y-0.5">
              <div className="flex justify-between text-[9px]">
                <span className="text-zinc-600">{label}</span>
                <span className="font-bold tabular-nums" style={{ color }}>{pct.toFixed(0)}%</span>
              </div>
              <Spark data={data} color={color} w={70} h={16} />
            </div>
          ))}
        </div>

        {/* Stats row */}
        <div className="grid grid-cols-3 gap-2 text-center">
          <div>
            <p className="text-xs font-bold text-zinc-200">{node.active_cameras}/{node.max_cameras}</p>
            <p className="text-[9px] text-zinc-600">cameras</p>
          </div>
          <div>
            <p className={cn("text-xs font-bold tabular-nums",
              node.latency_ms < 50 ? "text-green-400" : node.latency_ms < 100 ? "text-amber-400" : "text-red-400",
            )}>
              {node.latency_ms}ms
            </p>
            <p className="text-[9px] text-zinc-600">latency</p>
          </div>
          <div>
            <p className="text-xs font-bold text-zinc-200">{node.uptime_pct.toFixed(0)}%</p>
            <p className="text-[9px] text-zinc-600">uptime</p>
          </div>
        </div>

        {/* Assigned clients */}
        {node.assigned_clients.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {node.assigned_clients.slice(0, 3).map(c => (
              <span key={c.client_id} className="text-[9px] px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400 border border-zinc-700">
                {c.name}
              </span>
            ))}
            {node.assigned_clients.length > 3 && (
              <span className="text-[9px] text-zinc-600">+{node.assigned_clients.length - 3}</span>
            )}
          </div>
        )}
      </button>

      {/* Expanded panel */}
      {expanded && (
        <div className="border-t border-zinc-800 p-4 space-y-4">
          {/* Full health sparklines (4 metrics) */}
          <NodeHealthSparklines
            gpuH={gpuH} cpuH={cpuH} ramH={ramH} latH={latH}
            gpu={node.gpu_pct} cpu={node.cpu_pct} ram={node.ram_pct} lat={node.latency_ms}
          />

          {/* In-progress migrations on this node */}
          {inMigration.length > 0 && (
            <div className="space-y-2">
              <p className="text-[10px] font-semibold text-blue-400 flex items-center gap-1">
                <Loader2 className="h-3 w-3 animate-spin" />
                Active Migrations ({inMigration.length})
              </p>
              {inMigration.map(m => (
                <div key={m.migration_id} className="px-3 py-2 rounded-lg bg-blue-950/20 border border-blue-900/40">
                  <div className="flex items-center justify-between text-[10px] mb-1">
                    <span className="text-zinc-300 font-medium">{m.camera_name}</span>
                    <span className="text-zinc-500">
                      {m.from_node_name} → {m.to_node_name}
                    </span>
                  </div>
                  <MigrationStepper createdAt={m.created_at} />
                </div>
              ))}
            </div>
          )}

          {/* Heartbeat */}
          <div className="flex items-center gap-1.5 text-[10px]">
            <span className={cn("h-1.5 w-1.5 rounded-full", heartbeatOk ? "bg-green-400 animate-pulse" : "bg-red-400")} />
            <span className={cn(heartbeatOk ? "text-zinc-500" : "text-red-400")}>
              Heartbeat {heartbeatOk
                ? formatDistanceToNow(new Date(node.last_heartbeat * 1000), { addSuffix: true })
                : `missed (${heartbeatAge.toFixed(0)}s ago)`}
            </span>
            <span className="ml-auto text-zinc-600 font-mono">{node.gpu_model}</span>
          </div>

          {/* VRAM */}
          <div className="space-y-1">
            <div className="flex justify-between text-[9px] text-zinc-500">
              <span>VRAM</span>
              <span className="font-mono">{node.vram_used_gb.toFixed(1)} / {node.vram_total_gb.toFixed(0)} GB</span>
            </div>
            <div className="h-1 rounded-full bg-zinc-800 overflow-hidden">
              <div className="h-full bg-violet-500 rounded-full" style={{ width: `${(node.vram_used_gb / Math.max(node.vram_total_gb, 1)) * 100}%` }} />
            </div>
          </div>

          {/* Assigned cameras */}
          <div>
            <p className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wider mb-2">
              Assigned Cameras ({nodeCameras.length})
            </p>
            <div className="space-y-1.5">
              {nodeCameras.map(cam => {
                const migrating = inMigration.some(m => m.camera_id === cam.camera_id);
                return (
                  <div key={cam.camera_id} className={cn(
                    "flex items-center gap-2 px-2.5 py-1.5 rounded-lg bg-zinc-900 border",
                    migrating ? "border-blue-800/50" : "border-zinc-800",
                  )}>
                    <span className={cn("h-1.5 w-1.5 rounded-full shrink-0",
                      migrating      ? "bg-blue-400 animate-pulse" :
                      cam.status === "ONLINE"   ? "bg-green-400" :
                      cam.status === "DEGRADED" ? "bg-amber-400" : "bg-zinc-600",
                    )} />
                    <span className="text-xs text-zinc-300 truncate flex-1">{cam.room_name ?? cam.name}</span>
                    {migrating && <span className="text-[9px] text-blue-400">migrating…</span>}
                    <span className="text-[9px] text-zinc-600">{cam.building ?? ""}</span>
                    {!readOnly && !migrating && (
                      <button onClick={() => handleRemoveCamera(cam.camera_id)}
                        className="text-zinc-600 hover:text-red-400 transition-colors shrink-0">
                        <X className="h-3 w-3" />
                      </button>
                    )}
                  </div>
                );
              })}
              {nodeCameras.length === 0 && (
                <p className="text-[10px] text-zinc-700 text-center py-2">No cameras assigned</p>
              )}
            </div>
          </div>

          {/* Assign camera */}
          {!readOnly && (
            <div className="flex gap-2">
              <select value={assignSelect} onChange={e => setAssignSelect(e.target.value)}
                className="flex-1 h-8 px-2 rounded-md border border-zinc-700 bg-zinc-900 text-xs text-zinc-300">
                <option value="">— Assign a camera —</option>
                {unassigned.map(c => (
                  <option key={c.camera_id} value={c.camera_id}>{c.room_name ?? c.name}</option>
                ))}
                {unassigned.length === 0 && <option disabled>All cameras assigned</option>}
              </select>
              <Button size="sm" onClick={handleAssign} disabled={!assignSelect || assigning}
                className="h-8 text-xs gap-1 bg-blue-600 hover:bg-blue-700">
                {assigning ? <Loader2 className="h-3 w-3 animate-spin" /> : <Plus className="h-3 w-3" />}
                Assign
              </Button>
            </div>
          )}

          {/* Face DB */}
          <div className="text-[9px] text-zinc-500 border-t border-zinc-800/60 pt-2">
            <span>Face DB: </span>
            <span className="text-zinc-300 font-mono">{node.faces_db_size.toLocaleString()}</span>
            <span className="ml-4">Throughput: </span>
            <span className="text-zinc-300 font-mono">{node.throughput_fps.toFixed(1)} fps</span>
          </div>

          {/* Danger actions */}
          {!readOnly && (
            <div className="flex gap-2 pt-1 border-t border-zinc-800">
              <Button size="sm" variant="outline" onClick={handleDrain}
                disabled={draining || node.status === "DRAINING"}
                className="flex-1 h-7 text-xs gap-1 border-amber-800 text-amber-400 hover:bg-amber-950">
                {draining ? <Loader2 className="h-3 w-3 animate-spin" /> : <Zap className="h-3 w-3" />}
                {node.status === "DRAINING" ? "Draining…" : "Drain Node"}
              </Button>
              <Button size="sm" variant="ghost" onClick={handleRemove}
                disabled={removing || nodeCameras.length > 0}
                className="flex-1 h-7 text-xs gap-1 text-red-400 hover:text-red-300 hover:bg-red-950/30">
                {removing ? <Loader2 className="h-3 w-3 animate-spin" /> : <Trash2 className="h-3 w-3" />}
                Remove Node
              </Button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Camera Kanban ─────────────────────────────────────────────────────────────

function CameraItem({
  cam, readOnly, onDragStart,
}: {
  cam:         Camera;
  readOnly:    boolean;
  onDragStart: (e: React.DragEvent<HTMLDivElement>, cameraId: string, fromNodeId: string) => void;
}) {
  return (
    <div
      draggable={!readOnly}
      onDragStart={e => onDragStart(e, cam.camera_id, cam.node_id ?? "")}
      className={cn(
        "flex items-center gap-2 px-2.5 py-2 rounded-lg border bg-zinc-900 border-zinc-800 text-xs select-none",
        !readOnly && "cursor-grab active:cursor-grabbing hover:border-zinc-600",
      )}
    >
      {!readOnly && <Grip className="h-3 w-3 text-zinc-600 shrink-0" />}
      <span className={cn("h-1.5 w-1.5 rounded-full shrink-0",
        cam.status === "ONLINE" ? "bg-green-400" : cam.status === "DEGRADED" ? "bg-amber-400" : "bg-zinc-600",
      )} />
      <span className="text-zinc-300 truncate flex-1">{cam.room_name ?? cam.name}</span>
      <span className="text-[9px] text-zinc-600 shrink-0">{cam.mode.slice(0, 3)}</span>
    </div>
  );
}

function CameraKanban({
  nodes, cameras, readOnly,
}: {
  nodes:    RichNode[];
  cameras:  Camera[];
  readOnly: boolean;
}) {
  const qc = useQueryClient();
  const [dragOver,      setDragOver]      = useState<string | null>(null);
  const [balancing,     setBalancing]     = useState(false);
  const [preview,       setPreview]       = useState<AutoBalancePreview | null>(null);
  const [applying,      setApplying]      = useState(false);
  const [applyDelay,    setApplyDelay]    = useState(false);
  const [showAffinity,  setShowAffinity]  = useState(false);
  const [affinityRules, setAffinityRules] = useState<AffinityRule[]>([]);
  const [newRule, setNewRule] = useState({
    type: "CAMERA" as "CAMERA" | "CLIENT",
    entity_id: "", entity_name: "", preferred_node_id: "", hard: false,
  });

  const { data: affinityData } = useQuery({
    queryKey: ["affinity-rules"],
    queryFn:  () => api.get<{ rules: AffinityRule[] }>("/api/admin/nodes/affinity").then(r => r.data.rules ?? []),
    staleTime: 60_000,
  });
  useEffect(() => { if (affinityData) setAffinityRules(affinityData); }, [affinityData]);

  // Compute per-node before/after load impact from preview changes
  const nodeImpact = useMemo(() => {
    if (!preview) return [];
    const delta: Record<string, number> = {};
    preview.changes.forEach(c => {
      delta[c.from_node] = (delta[c.from_node] ?? 0) - 1;
      delta[c.to_node]   = (delta[c.to_node]   ?? 0) + 1;
    });
    return nodes
      .map(n => ({
        nodeId:   n.node_id,
        nodeName: n.node_name,
        before:   cameras.filter(c => c.node_id === n.node_id).length,
        delta:    delta[n.node_name] ?? 0,
        gpuPct:   n.gpu_pct,
        maxCam:   n.max_cameras,
      }))
      .filter(n => n.before > 0 || Math.abs(n.delta) > 0);
  }, [preview, nodes, cameras]);

  const handleDragStart = (e: React.DragEvent<HTMLDivElement>, cameraId: string, fromNodeId: string) => {
    e.dataTransfer.setData("cameraId",   cameraId);
    e.dataTransfer.setData("fromNodeId", fromNodeId);
    e.dataTransfer.effectAllowed = "move";
  };

  const handleDrop = async (e: React.DragEvent<HTMLDivElement>, toNodeId: string) => {
    e.preventDefault();
    setDragOver(null);
    const cameraId   = e.dataTransfer.getData("cameraId");
    const fromNodeId = e.dataTransfer.getData("fromNodeId");
    if (!cameraId || fromNodeId === toNodeId) return;
    try {
      await api.post(`/api/node/cameras/${cameraId}/assign`, { node_id: toNodeId || null });
      qc.invalidateQueries({ queryKey: ["all-cameras"] });
      toast.success("Camera reassigned");
    } catch { toast.error("Assignment failed"); }
  };

  const handleAutoBalance = async () => {
    setBalancing(true);
    try {
      const res = await api.post<AutoBalancePreview>("/api/admin/nodes/auto-balance?dry_run=true");
      setPreview(res.data);
    } catch { toast.error("Auto-balance preview failed"); }
    finally { setBalancing(false); }
  };

  const applyBalance = async () => {
    setApplying(true);
    try {
      await api.post("/api/admin/nodes/auto-balance", { delay: applyDelay });
      setPreview(null);
      qc.invalidateQueries({ queryKey: ["all-cameras"] });
      toast.success(applyDelay
        ? "Auto-balance queued — will apply after active sessions complete"
        : "Auto-balance applied",
      );
    } catch { toast.error("Apply failed"); }
    finally { setApplying(false); }
  };

  const saveAffinity = async () => {
    try {
      await api.put("/api/admin/nodes/affinity", { rules: affinityRules });
      toast.success("Affinity rules saved");
    } catch { toast.error("Save failed"); }
  };

  const addRule = () => {
    if (!newRule.entity_id || !newRule.preferred_node_id) return;
    const targetNode = nodes.find(n => n.node_id === newRule.preferred_node_id);
    setAffinityRules(p => [...p, {
      ...newRule,
      id: crypto.randomUUID(),
      preferred_node_name: targetNode?.node_name ?? newRule.preferred_node_id,
    }]);
    setNewRule({ type: "CAMERA", entity_id: "", entity_name: "", preferred_node_id: "", hard: false });
  };

  const allColumns = [
    { nodeId: null as string | null, label: "Unassigned", cams: cameras.filter(c => !c.node_id) },
    ...nodes.map(n => ({
      nodeId: n.node_id,
      label:  `${n.node_name} (${cameras.filter(c => c.node_id === n.node_id).length}/${n.max_cameras})`,
      cams:   cameras.filter(c => c.node_id === n.node_id),
    })),
  ];

  return (
    <div className="space-y-4">
      {/* Toolbar */}
      {!readOnly && (
        <div className="flex items-center gap-2 flex-wrap">
          <Button size="sm" variant="outline" onClick={handleAutoBalance} disabled={balancing}
            className="h-8 text-xs gap-1.5"
            title="Redistribute cameras evenly across all healthy nodes, respecting affinity rules">
            {balancing ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <BarChart3 className="h-3.5 w-3.5" />}
            Auto-Balance
          </Button>
          <Button size="sm" variant="ghost" onClick={() => setShowAffinity(p => !p)}
            className="h-8 text-xs gap-1.5 text-zinc-400">
            <Settings2 className="h-3.5 w-3.5" />
            Affinity Rules
            {showAffinity ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
          </Button>
          <span className="ml-auto text-[10px] text-zinc-500">
            {cameras.filter(c => !c.node_id).length} unassigned · drag cameras between columns
          </span>
        </div>
      )}

      {/* Auto-balance confirmation with preview ─────────────────────────── */}
      {preview && (
        <div className="rounded-xl border border-blue-800 bg-blue-900/10 p-4 space-y-4">
          {/* Header */}
          <div className="flex items-center justify-between">
            <p className="text-sm font-semibold text-blue-400 flex items-center gap-1.5">
              <BarChart3 className="h-4 w-4" />
              Auto-Balance Preview — {preview.changes.length} move{preview.changes.length !== 1 ? "s" : ""}
            </p>
            <button onClick={() => setPreview(null)} className="text-zinc-500 hover:text-zinc-300">
              <X className="h-4 w-4" />
            </button>
          </div>

          <div className="grid grid-cols-2 gap-4">
            {/* Left: camera moves */}
            <div className="space-y-1.5">
              <p className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wider">Camera Moves</p>
              <div className="space-y-1 max-h-36 overflow-y-auto">
                {preview.changes.map((c, i) => (
                  <div key={i} className="flex items-center gap-1.5 text-xs text-zinc-300 py-0.5">
                    <span className="w-32 truncate text-zinc-300">{c.camera_name}</span>
                    <span className="text-zinc-600 truncate max-w-[60px]">{c.from_node}</span>
                    <ArrowRight className="h-3 w-3 text-blue-500 shrink-0" />
                    <span className="text-blue-300 truncate max-w-[60px]">{c.to_node}</span>
                    <span className="text-zinc-600 text-[9px] ml-auto">{c.reason}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Right: node load impact */}
            <div className="space-y-1.5">
              <p className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wider">Node Load Impact</p>
              <div className="rounded-lg border border-zinc-800 overflow-hidden">
                <table className="w-full text-[10px]">
                  <thead>
                    <tr className="bg-zinc-800/50 text-zinc-500">
                      <th className="px-2 py-1.5 text-left font-semibold">Node</th>
                      <th className="px-2 py-1.5 text-right font-semibold">Cams</th>
                      <th className="px-2 py-1.5 text-right font-semibold">Change</th>
                      <th className="px-2 py-1.5 text-right font-semibold">GPU %</th>
                    </tr>
                  </thead>
                  <tbody>
                    {nodeImpact.map(ni => (
                      <tr key={ni.nodeId} className="border-t border-zinc-800/50">
                        <td className="px-2 py-1.5 text-zinc-300 truncate max-w-[80px]">{ni.nodeName}</td>
                        <td className="px-2 py-1.5 text-right font-mono text-zinc-400">
                          {ni.before} → {ni.before + ni.delta}
                          <span className="text-zinc-700">/{ni.maxCam}</span>
                        </td>
                        <td className={cn("px-2 py-1.5 text-right font-bold tabular-nums",
                          ni.delta > 0 ? "text-amber-400" : ni.delta < 0 ? "text-green-400" : "text-zinc-600",
                        )}>
                          {ni.delta > 0 ? `+${ni.delta}` : ni.delta < 0 ? ni.delta : "—"}
                        </td>
                        <td className={cn("px-2 py-1.5 text-right font-mono",
                          ni.gpuPct >= 85 ? "text-red-400" : ni.gpuPct >= 65 ? "text-amber-400" : "text-green-400",
                        )}>
                          {ni.gpuPct.toFixed(0)}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Active sessions warning */}
          {(preview.active_sessions ?? 0) > 0 && (
            <div className="flex items-start gap-2 px-3 py-2 rounded-lg bg-amber-900/15 border border-amber-800/40 text-xs text-amber-300">
              <AlertTriangle className="h-3.5 w-3.5 mt-0.5 shrink-0" />
              <div>
                <span className="font-semibold">{preview.active_sessions} active session{preview.active_sessions > 1 ? "s" : ""}</span>
                {" "}will be interrupted during migration. Each migration takes ~{(preview.estimated_seconds ?? 90)}s.
              </div>
            </div>
          )}

          {/* Options + actions */}
          <div className="flex items-center gap-3 pt-1 border-t border-zinc-800/60">
            {(preview.active_sessions ?? 0) > 0 && (
              <label className="flex items-center gap-1.5 text-xs text-zinc-400 cursor-pointer">
                <input type="checkbox" checked={applyDelay} onChange={e => setApplyDelay(e.target.checked)}
                  className="accent-blue-600 h-3 w-3" />
                Wait for sessions to complete before migrating
              </label>
            )}
            <div className="ml-auto flex gap-2">
              <Button size="sm" variant="ghost" onClick={() => setPreview(null)}
                className="h-7 text-xs">Cancel</Button>
              <Button size="sm" onClick={applyBalance} disabled={applying}
                className="h-7 text-xs bg-blue-600 hover:bg-blue-700 gap-1">
                {applying ? <Loader2 className="h-3 w-3 animate-spin" /> : <Check className="h-3 w-3" />}
                {applyDelay ? "Queue" : "Apply"} {preview.changes.length} Moves
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Affinity rules */}
      {showAffinity && !readOnly && (
        <div className="rounded-xl border border-zinc-700 bg-zinc-900 p-4 space-y-3">
          <div className="flex items-center justify-between">
            <p className="text-xs font-semibold text-zinc-300">Affinity Rules</p>
            <Button size="sm" variant="ghost" onClick={saveAffinity} className="h-6 text-xs gap-1 text-blue-400">
              <Save className="h-3 w-3" /> Save
            </Button>
          </div>
          <div className="grid grid-cols-5 gap-2">
            <select value={newRule.type} onChange={e => setNewRule(p => ({ ...p, type: e.target.value as "CAMERA"|"CLIENT" }))}
              className="h-7 px-2 rounded border border-zinc-700 bg-zinc-800 text-xs text-zinc-300">
              <option value="CAMERA">Camera</option>
              <option value="CLIENT">Client</option>
            </select>
            <input value={newRule.entity_id} onChange={e => setNewRule(p => ({ ...p, entity_id: e.target.value }))}
              placeholder="Entity ID" className="h-7 px-2 rounded border border-zinc-700 bg-zinc-800 text-xs text-zinc-300" />
            <input value={newRule.entity_name} onChange={e => setNewRule(p => ({ ...p, entity_name: e.target.value }))}
              placeholder="Name (display)" className="h-7 px-2 rounded border border-zinc-700 bg-zinc-800 text-xs text-zinc-300" />
            <select value={newRule.preferred_node_id} onChange={e => setNewRule(p => ({ ...p, preferred_node_id: e.target.value }))}
              className="h-7 px-2 rounded border border-zinc-700 bg-zinc-800 text-xs text-zinc-300">
              <option value="">— Preferred Node —</option>
              {nodes.map(n => <option key={n.node_id} value={n.node_id}>{n.node_name}</option>)}
            </select>
            <div className="flex items-center gap-2">
              <label className="flex items-center gap-1 text-[10px] text-zinc-400">
                <input type="checkbox" checked={newRule.hard} onChange={e => setNewRule(p => ({ ...p, hard: e.target.checked }))}
                  className="accent-blue-600" />
                Hard
              </label>
              <Button size="sm" onClick={addRule} className="h-7 text-xs gap-1 bg-blue-600">
                <Plus className="h-3 w-3" />
              </Button>
            </div>
          </div>
          {affinityRules.length > 0 && (
            <div className="space-y-1.5">
              {affinityRules.map((r, i) => (
                <div key={r.id} className="flex items-center gap-2 text-xs text-zinc-400 py-1 border-b border-zinc-800/60">
                  <span className="text-[9px] font-bold text-zinc-600 w-12">{r.type}</span>
                  <span className="flex-1 truncate">{r.entity_name || r.entity_id}</span>
                  <ArrowRight className="h-3 w-3 text-zinc-600 shrink-0" />
                  <span className="text-zinc-300">{r.preferred_node_name}</span>
                  {r.hard && <span className="text-[9px] text-amber-400 bg-amber-900/20 px-1 rounded">HARD</span>}
                  <button onClick={() => setAffinityRules(p => p.filter((_, j) => j !== i))}
                    className="text-zinc-600 hover:text-red-400"><Trash2 className="h-3 w-3" /></button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Kanban columns */}
      <div className="flex gap-3 overflow-x-auto pb-2" style={{ minHeight: 300 }}>
        {allColumns.map(({ nodeId, label, cams }) => {
          const isOver = dragOver === (nodeId ?? "__unassigned__");
          return (
            <div
              key={nodeId ?? "__unassigned__"}
              className={cn(
                "flex flex-col min-w-[220px] rounded-xl border p-3 transition-colors",
                isOver ? "border-blue-600 bg-blue-950/20" : "border-zinc-800 bg-zinc-900/40",
                !nodeId && "border-dashed",
              )}
              onDragOver={e => { e.preventDefault(); setDragOver(nodeId ?? "__unassigned__"); }}
              onDragLeave={() => setDragOver(null)}
              onDrop={e => handleDrop(e, nodeId ?? "")}
            >
              <p className={cn("text-[10px] font-bold uppercase tracking-wider mb-2",
                nodeId ? "text-zinc-400" : "text-zinc-600 italic",
              )}>
                {label}
              </p>
              <div className="flex-1 space-y-1.5">
                {cams.map(cam => (
                  <CameraItem key={cam.camera_id} cam={cam} readOnly={readOnly} onDragStart={handleDragStart} />
                ))}
                {cams.length === 0 && (
                  <p className="text-[9px] text-zinc-700 text-center py-4">
                    {isOver ? "Drop here" : "Empty"}
                  </p>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Migration log ─────────────────────────────────────────────────────────────

function MigrationLog({ nodes }: { nodes: RichNode[] }) {
  const [filterNode,   setFilterNode]   = useState("");
  const [filterStatus, setFilterStatus] = useState("");
  const [filterReason, setFilterReason] = useState("");

  const hasInProgress = useCallback((events: MigrationEvent[]) =>
    events.some(e => e.status === "IN_PROGRESS"), []);

  const { data, isLoading } = useQuery({
    queryKey:        ["migration-log"],
    queryFn:         () => api.get<{ events: MigrationEvent[] }>("/api/admin/nodes/migration-log")
      .then(r => r.data.events ?? []),
    // Poll more aggressively when migrations are active
    refetchInterval: (d) => hasInProgress(d?.state?.data ?? []) ? 5_000 : 30_000,
    staleTime:       4_000,
  });

  const allEvents  = data ?? [];
  const inProgress = allEvents.filter(e => e.status === "IN_PROGRESS");
  const events = allEvents.filter(e =>
    (!filterNode   || e.from_node_id === filterNode || e.to_node_id === filterNode) &&
    (!filterStatus || e.status === filterStatus) &&
    (!filterReason || e.reason === filterReason),
  );

  const REASON_COLOR: Record<string, string> = {
    MANUAL:       "text-blue-400",
    NODE_FAILURE: "text-red-400",
    AUTO_BALANCE: "text-teal-400",
    DRAIN:        "text-amber-400",
  };

  const exportCSV = () => {
    const rows = [["Camera","From","To","Reason","Timestamp","Status"], ...events.map(e => [
      e.camera_name, e.from_node_name, e.to_node_name,
      e.reason, e.created_at, e.status,
    ])];
    const csv  = rows.map(r => r.map(c => `"${c}"`).join(",")).join("\n");
    const url  = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    Object.assign(document.createElement("a"), { href: url, download: "migration-log.csv" }).click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-3">
      {/* Live migrations banner */}
      {inProgress.length > 0 && (
        <div className="rounded-xl border border-blue-800 bg-blue-900/10 p-4 space-y-3">
          <p className="text-xs font-semibold text-blue-400 flex items-center gap-1.5">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            {inProgress.length} migration{inProgress.length > 1 ? "s" : ""} in progress
          </p>
          {inProgress.map(m => (
            <div key={m.migration_id} className="space-y-1">
              <div className="flex items-center gap-3 text-xs text-zinc-300">
                <span className="font-medium w-40 truncate">{m.camera_name}</span>
                <span className="text-zinc-500">{m.from_node_name}</span>
                <ArrowRight className="h-3 w-3 text-blue-400 shrink-0" />
                <span className="text-blue-300">{m.to_node_name}</span>
                <span className={cn("ml-auto text-[9px] font-semibold", REASON_COLOR[m.reason])}>
                  {MIGRATION_REASON_LABEL[m.reason]}
                </span>
              </div>
              <MigrationStepper createdAt={m.created_at} />
            </div>
          ))}
        </div>
      )}

      {/* Filters */}
      <div className="flex gap-3 flex-wrap items-center">
        <select value={filterNode} onChange={e => setFilterNode(e.target.value)}
          className="h-8 px-2 rounded-md border border-zinc-700 bg-zinc-900 text-xs text-zinc-300">
          <option value="">All Nodes</option>
          {nodes.map(n => <option key={n.node_id} value={n.node_id}>{n.node_name}</option>)}
        </select>
        <select value={filterStatus} onChange={e => setFilterStatus(e.target.value)}
          className="h-8 px-2 rounded-md border border-zinc-700 bg-zinc-900 text-xs text-zinc-300">
          <option value="">All Statuses</option>
          {["IN_PROGRESS", "COMPLETED", "FAILED"].map(s => (
            <option key={s} value={s}>{s.replace("_", " ")}</option>
          ))}
        </select>
        <select value={filterReason} onChange={e => setFilterReason(e.target.value)}
          className="h-8 px-2 rounded-md border border-zinc-700 bg-zinc-900 text-xs text-zinc-300">
          <option value="">All Reasons</option>
          {Object.entries(MIGRATION_REASON_LABEL).map(([k, v]) => (
            <option key={k} value={k}>{v}</option>
          ))}
        </select>
        <span className="text-[10px] text-zinc-600 self-center">{events.length} events</span>
        <Button size="sm" variant="ghost" onClick={exportCSV}
          className="ml-auto h-7 text-xs gap-1 text-zinc-500 hover:text-zinc-300">
          <Download className="h-3.5 w-3.5" /> CSV
        </Button>
      </div>

      {/* Table */}
      <div className="rounded-xl border border-zinc-800 overflow-hidden">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-zinc-800 bg-zinc-900/50">
              {["Camera", "From Node", "To Node", "Reason", "Timestamp", "Status"].map(h => (
                <th key={h} className="px-3 py-2.5 text-left text-zinc-500 font-semibold whitespace-nowrap">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {isLoading && (
              <tr><td colSpan={6} className="px-3 py-8 text-center text-zinc-600">
                <Loader2 className="h-4 w-4 animate-spin inline mr-2" />Loading…
              </td></tr>
            )}
            {!isLoading && events.map(e => (
              <>
                <tr key={e.migration_id}
                  className={cn(
                    "border-b border-zinc-800/50 hover:bg-zinc-900/30",
                    e.status === "IN_PROGRESS" && "bg-blue-950/5",
                  )}>
                  <td className="px-3 py-2.5 text-zinc-300 font-medium">{e.camera_name}</td>
                  <td className="px-3 py-2.5 text-zinc-400">{e.from_node_name}</td>
                  <td className="px-3 py-2.5 text-zinc-400">{e.to_node_name}</td>
                  <td className="px-3 py-2.5">
                    <span className={cn("font-semibold", REASON_COLOR[e.reason])}>
                      {MIGRATION_REASON_LABEL[e.reason]}
                    </span>
                  </td>
                  <td className="px-3 py-2.5 font-mono text-zinc-500">{fmtDatetime(e.created_at)}</td>
                  <td className="px-3 py-2.5">
                    <span className={cn(
                      "text-[10px] font-bold px-1.5 py-0.5 rounded",
                      MIGRATION_STATUS[e.status].cls,
                    )}>
                      {e.status.replace("_", " ")}
                    </span>
                  </td>
                </tr>
                {/* Inline progress stepper for in-progress rows */}
                {e.status === "IN_PROGRESS" && (
                  <tr className="border-b border-zinc-800/50 bg-blue-950/5">
                    <td colSpan={6} className="px-4 pb-3 pt-0">
                      <MigrationStepper createdAt={e.created_at} />
                    </td>
                  </tr>
                )}
              </>
            ))}
            {!isLoading && events.length === 0 && (
              <tr><td colSpan={6} className="px-3 py-8 text-center text-zinc-600">No migration events</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

type ClusterTab = "nodes" | "assignment" | "migrations";

const num = (h: Record<string, unknown>, k: string, def = 0) =>
  typeof h[k] === "number" ? (h[k] as number) : def;

const enrichNode = (n: GpuNode): RichNode => {
  const h = (n.health_json ?? {}) as Record<string, unknown>;
  return {
    ...n,
    connectivity:     ((h["connectivity"] as unknown) as "PUBLIC_IP" | "CLOUDFLARE_TUNNEL") ?? "PUBLIC_IP",
    latency_ms:       num(h, "latency_ms", 0),
    uptime_pct:       num(h, "uptime_pct", 100),
    gpu_pct:          num(h, "gpu_util", 0),
    cpu_pct:          num(h, "cpu_util", 0),
    ram_pct:          num(h, "ram_util", 0),
    vram_used_gb:     num(h, "vram_used_gb", 0),
    vram_total_gb:    num(h, "vram_total_gb", 24),
    faces_db_size:    num(h, "faces_db_size", 0),
    throughput_fps:   num(h, "throughput_fps", 0),
    assigned_clients: ((h["assigned_clients"] as unknown) as Array<{ client_id: string; name: string; slug: string }>) ?? [],
  };
};

export default function ClusterPage() {
  const { isViewer, isClientAdmin, isSuperAdmin } = useAuth();
  const router = useRouter();
  const [tab, setTab] = useState<ClusterTab>("nodes");

  useEffect(() => {
    if (isViewer) router.replace("/attendance");
  }, [isViewer, router]);

  if (isViewer) return null;

  const readOnly = isClientAdmin;

  const { data: rawNodes, isLoading: nodesLoading, refetch } = useQuery({
    queryKey:        ["cluster-nodes"],
    queryFn:         () => isSuperAdmin
      ? api.get<GpuNode[]>("/api/admin/nodes").then(r => r.data)
      : api.get<GpuNode[]>("/api/cameras/available-nodes").then(r => r.data),
    refetchInterval:  15_000,
    staleTime:         5_000,
  });

  const nodes: RichNode[] = useMemo(() => (rawNodes ?? []).map(enrichNode), [rawNodes]);

  const { data: allCams } = useQuery({
    queryKey: ["all-cameras"],
    queryFn:  () => api.get<{ items?: Camera[] } | Camera[]>("/api/cameras").then(r =>
      (r.data as { items?: Camera[] }).items ?? (r.data as Camera[])),
    staleTime: 30_000,
  });
  const cameras: Camera[] = allCams ?? [];

  // Migration events (shared between NodeCard badges and MigrationLog)
  const { data: migrationData } = useQuery({
    queryKey:        ["migration-log"],
    queryFn:         () => api.get<{ events: MigrationEvent[] }>("/api/admin/nodes/migration-log")
      .then(r => r.data.events ?? []),
    refetchInterval:  15_000,
    staleTime:         5_000,
  });
  const allMigrations      = migrationData ?? [];
  const activeMigrations   = allMigrations.filter(m => m.status === "IN_PROGRESS");

  const TABS: Array<{ id: ClusterTab; label: string; icon: React.ComponentType<{ className?: string }> }> = [
    { id: "nodes",      label: "Node Grid",         icon: Server  },
    { id: "assignment", label: "Camera Assignment",  icon: Layers  },
    { id: "migrations", label: `Migration Log${activeMigrations.length > 0 ? ` (${activeMigrations.length})` : ""}`, icon: Clock },
  ];

  return (
    <div className="flex flex-col h-[calc(100vh-64px)] overflow-hidden bg-zinc-950">
      {/* Header */}
      <div className="shrink-0 border-b border-zinc-800 px-5 pt-4 pb-0">
        <div className="flex items-center gap-3 mb-3">
          <Server className="h-5 w-5 text-blue-400" />
          <h1 className="text-base font-bold text-zinc-100">GPU Cluster</h1>
          {readOnly && (
            <span className="flex items-center gap-1 text-[10px] font-semibold text-amber-400 bg-amber-900/20 border border-amber-800 px-1.5 py-0.5 rounded">
              <Lock className="h-2.5 w-2.5" /> Read-only — Your assigned nodes
            </span>
          )}
          <div className="ml-auto flex items-center gap-2">
            {nodesLoading && <Loader2 className="h-3.5 w-3.5 text-zinc-600 animate-spin" />}
            <Button variant="ghost" size="sm" className="h-7 gap-1 text-xs" onClick={() => refetch()}>
              <RefreshCw className="h-3.5 w-3.5" />
            </Button>
          </div>
        </div>
        <div className="flex gap-0 -mb-px">
          {TABS.map(({ id, label, icon: Icon }) => (
            <button key={id} onClick={() => setTab(id)}
              className={cn(
                "flex items-center gap-1.5 px-4 py-2.5 text-xs font-medium border-b-2 transition-colors whitespace-nowrap",
                tab === id ? "border-blue-500 text-blue-400" : "border-transparent text-zinc-500 hover:text-zinc-300",
              )}>
              <Icon className="h-3.5 w-3.5" />{label}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-5 space-y-5">
        <SummaryBar nodes={nodes} activeMigrations={activeMigrations.length} />

        {tab === "nodes" && (
          <div className="grid gap-4" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(300px, 1fr))" }}>
            {nodesLoading && Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="h-52 rounded-xl border border-zinc-800 bg-zinc-900 animate-pulse" />
            ))}
            {nodes.map(n => (
              <NodeCard
                key={n.node_id}
                node={n}
                cameras={cameras}
                activeMigrations={activeMigrations.filter(m =>
                  m.from_node_id === n.node_id || m.to_node_id === n.node_id,
                )}
                readOnly={readOnly}
                onRefresh={refetch}
              />
            ))}
            {!nodesLoading && nodes.length === 0 && (
              <div className="col-span-full py-16 flex flex-col items-center gap-3 text-zinc-600">
                <Server className="h-10 w-10 opacity-30" />
                <p className="text-sm">No nodes registered</p>
              </div>
            )}
          </div>
        )}

        {tab === "assignment" && (
          <CameraKanban nodes={nodes} cameras={cameras} readOnly={readOnly} />
        )}

        {tab === "migrations" && (
          <MigrationLog nodes={nodes} />
        )}
      </div>
    </div>
  );
}
