"use client";

import { useCallback, useEffect, useState } from "react";
import { useDropzone } from "react-dropzone";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Building2, Camera, ChevronRight, Eye, EyeOff, ImagePlus,
  Loader2, Plus, Search, Server, Trash2, Users, X, Check,
} from "lucide-react";
import { toast } from "sonner";
import { adminApi, api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Progress } from "@/components/ui/progress";
import { Textarea } from "@/components/ui/textarea";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription,
} from "@/components/ui/dialog";
import { cn, fmtDatetime } from "@/lib/utils";
import type { Client, GpuNode } from "@/types";

// ─────────────────────────────────────────────────────────────────────────────
// Shared primitives
// ─────────────────────────────────────────────────────────────────────────────

function RangeSlider({
  label, min, max, value, step = 1,
  onChange, format,
}: {
  label: string; min: number; max: number; value: number; step?: number;
  onChange: (v: number) => void; format?: (v: number) => string;
}) {
  const fmt  = format ?? ((v: number) => v.toLocaleString());
  const pct  = ((value - min) / (max - min)) * 100;
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-mono font-semibold text-zinc-200 tabular-nums">{fmt(value)}</span>
      </div>
      <div className="relative py-2">
        {/* Track */}
        <div className="h-1.5 w-full rounded-full bg-zinc-800">
          <div className="h-full rounded-full bg-zinc-400 transition-all" style={{ width: `${pct}%` }} />
        </div>
        <input
          type="range" min={min} max={max} step={step} value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          className="absolute inset-0 h-full w-full cursor-pointer opacity-0"
        />
        {/* Thumb visualisation */}
        <div
          className="pointer-events-none absolute top-1/2 h-4 w-4 -translate-y-1/2 -translate-x-1/2 rounded-full border-2 border-zinc-300 bg-zinc-950 shadow transition-all"
          style={{ left: `${pct}%` }}
        />
      </div>
      <div className="flex justify-between text-[10px] text-zinc-600">
        <span>{fmt(min)}</span>
        <span>{fmt(max)}</span>
      </div>
    </div>
  );
}

function FieldError({ msg }: { msg?: string }) {
  return msg ? <p className="text-xs text-red-400 mt-1">{msg}</p> : null;
}

// ─────────────────────────────────────────────────────────────────────────────
// Step indicator
// ─────────────────────────────────────────────────────────────────────────────

const STEPS = ["Info", "Limits", "GPU Nodes", "Admin", "Review"];

function StepIndicator({ current }: { current: number }) {
  return (
    <div className="flex items-center gap-0 mb-6">
      {STEPS.map((label, i) => {
        const done   = i < current;
        const active = i === current;
        return (
          <div key={i} className="flex items-center flex-1 last:flex-none">
            <div className="flex flex-col items-center gap-1">
              <div className={cn(
                "flex h-7 w-7 items-center justify-center rounded-full text-xs font-semibold border-2 transition-colors",
                done   && "bg-zinc-300 border-zinc-300 text-zinc-900",
                active && "bg-zinc-900 border-zinc-400 text-zinc-100",
                !done && !active && "border-zinc-700 text-zinc-600",
              )}>
                {done ? <Check className="h-3.5 w-3.5" /> : i + 1}
              </div>
              <span className={cn(
                "text-[10px] whitespace-nowrap",
                active ? "text-zinc-200" : done ? "text-zinc-400" : "text-zinc-600",
              )}>{label}</span>
            </div>
            {i < STEPS.length - 1 && (
              <div className={cn("h-px flex-1 mx-2 mb-4", i < current ? "bg-zinc-500" : "bg-zinc-800")} />
            )}
          </div>
        );
      })}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Create Client Dialog — state
// ─────────────────────────────────────────────────────────────────────────────

interface NodeAssignment {
  node_id:             string;
  max_cameras_on_node: number;
}

const EMPTY_STATE = () => ({
  name:          "",
  slug:          "",
  contact_name:  "",
  contact_email: "",
  contact_phone: "",
  address:       "",
  logo_file:     null as File | null,
  max_cameras:   50,
  max_persons:   10000,
  selectedNodes: [] as NodeAssignment[],
  admin_name:    "",
  admin_email:   "",
  admin_password:"",
  showPass:      false,
});

function slugify(s: string) {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 1 — Organisation Info
// ─────────────────────────────────────────────────────────────────────────────

function Step1Info({
  state, setState,
}: {
  state: ReturnType<typeof EMPTY_STATE>;
  setState: React.Dispatch<React.SetStateAction<ReturnType<typeof EMPTY_STATE>>>;
}) {
  const set = (k: string, v: string) => setState((s) => ({ ...s, [k]: v }));

  const onDrop = useCallback((files: File[]) => {
    if (files[0]) setState((s) => ({ ...s, logo_file: files[0] }));
  }, [setState]);
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop, accept: { "image/*": [] }, multiple: false,
  });

  const logoPreview = state.logo_file ? URL.createObjectURL(state.logo_file) : null;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        {/* Logo */}
        <div className="col-span-2">
          <Label className="text-xs text-muted-foreground">Logo</Label>
          <div className="mt-1.5 flex items-center gap-3">
            <div
              {...getRootProps()}
              className={cn(
                "flex h-16 w-16 cursor-pointer items-center justify-center rounded-xl border-2 border-dashed transition-colors overflow-hidden shrink-0",
                isDragActive ? "border-zinc-500 bg-zinc-800" : "border-zinc-700 hover:border-zinc-500",
              )}
            >
              <input {...getInputProps()} />
              {logoPreview ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={logoPreview} alt="logo" className="h-full w-full object-contain" />
              ) : (
                <ImagePlus className="h-5 w-5 text-zinc-600" />
              )}
            </div>
            <p className="text-xs text-muted-foreground">
              {state.logo_file ? state.logo_file.name : "Drop a PNG/SVG logo, or click to browse"}
            </p>
          </div>
        </div>

        {/* Name */}
        <div className="space-y-1.5">
          <Label className="text-xs text-muted-foreground">Organisation name *</Label>
          <Input
            value={state.name}
            onChange={(e) => {
              set("name", e.target.value);
              if (!state.slug || state.slug === slugify(state.name)) {
                set("slug", slugify(e.target.value));
              }
            }}
            placeholder="Arresto Institute"
            className="bg-zinc-900 border-zinc-700"
          />
        </div>

        {/* Slug */}
        <div className="space-y-1.5">
          <Label className="text-xs text-muted-foreground">Slug (URL identifier) *</Label>
          <Input
            value={state.slug}
            onChange={(e) => set("slug", e.target.value)}
            placeholder="arresto-institute"
            className="bg-zinc-900 border-zinc-700 font-mono text-sm"
          />
        </div>

        {/* Contact */}
        <div className="space-y-1.5">
          <Label className="text-xs text-muted-foreground">Contact name *</Label>
          <Input value={state.contact_name} onChange={(e) => set("contact_name", e.target.value)}
            placeholder="Dr. Jane Smith" className="bg-zinc-900 border-zinc-700" />
        </div>
        <div className="space-y-1.5">
          <Label className="text-xs text-muted-foreground">Contact email *</Label>
          <Input type="email" value={state.contact_email} onChange={(e) => set("contact_email", e.target.value)}
            placeholder="admin@institute.edu" className="bg-zinc-900 border-zinc-700" />
        </div>
        <div className="space-y-1.5">
          <Label className="text-xs text-muted-foreground">Contact phone</Label>
          <Input value={state.contact_phone} onChange={(e) => set("contact_phone", e.target.value)}
            placeholder="+91 98765 43210" className="bg-zinc-900 border-zinc-700" />
        </div>

        {/* Address */}
        <div className="col-span-2 space-y-1.5">
          <Label className="text-xs text-muted-foreground">Address</Label>
          <Textarea value={state.address} onChange={(e) => set("address", e.target.value)}
            rows={2} placeholder="123 Campus Road, City, State"
            className="bg-zinc-900 border-zinc-700 resize-none" />
        </div>
      </div>
    </div>
  );
}

function validateStep1(s: ReturnType<typeof EMPTY_STATE>): string | null {
  if (!s.name.trim())          return "Organisation name is required";
  if (!s.slug.trim())          return "Slug is required";
  if (!s.contact_name.trim())  return "Contact name is required";
  if (!s.contact_email.trim()) return "Contact email is required";
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(s.contact_email)) return "Invalid email address";
  return null;
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 2 — Limits
// ─────────────────────────────────────────────────────────────────────────────

function Step2Limits({
  state, setState,
}: {
  state: ReturnType<typeof EMPTY_STATE>;
  setState: React.Dispatch<React.SetStateAction<ReturnType<typeof EMPTY_STATE>>>;
}) {
  return (
    <div className="space-y-8 py-2">
      <div className="rounded-xl border border-border/50 bg-zinc-900/50 p-5 space-y-2">
        <p className="text-sm font-medium text-zinc-200">Cameras</p>
        <p className="text-xs text-muted-foreground mb-4">
          Maximum number of cameras this client can register across all nodes.
        </p>
        <RangeSlider
          label="Max cameras" min={1} max={500} value={state.max_cameras}
          onChange={(v) => setState((s) => ({ ...s, max_cameras: v }))}
        />
        <div className="grid grid-cols-3 gap-2 mt-3 text-xs text-zinc-500">
          {[10, 50, 100, 200, 500].map((v) => (
            <button key={v} onClick={() => setState((s) => ({ ...s, max_cameras: v }))}
              className={cn("rounded px-2 py-1 border transition-colors",
                state.max_cameras === v ? "border-zinc-500 text-zinc-200 bg-zinc-800" : "border-zinc-800 hover:border-zinc-600"
              )}
            >{v}</button>
          ))}
        </div>
      </div>

      <div className="rounded-xl border border-border/50 bg-zinc-900/50 p-5 space-y-2">
        <p className="text-sm font-medium text-zinc-200">Enrolled Persons</p>
        <p className="text-xs text-muted-foreground mb-4">
          Maximum face enrollments (students + faculty + staff) for this client.
        </p>
        <RangeSlider
          label="Max persons" min={100} max={100000} step={100} value={state.max_persons}
          onChange={(v) => setState((s) => ({ ...s, max_persons: v }))}
          format={(v) => v >= 1000 ? `${(v / 1000).toFixed(1)}K` : String(v)}
        />
        <div className="grid grid-cols-3 gap-2 mt-3 text-xs text-zinc-500">
          {[500, 2000, 10000, 25000, 100000].map((v) => (
            <button key={v} onClick={() => setState((s) => ({ ...s, max_persons: v }))}
              className={cn("rounded px-2 py-1 border transition-colors",
                state.max_persons === v ? "border-zinc-500 text-zinc-200 bg-zinc-800" : "border-zinc-800 hover:border-zinc-600"
              )}
            >{v >= 1000 ? `${v / 1000}K` : v}</button>
          ))}
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 3 — GPU Node Assignment
// ─────────────────────────────────────────────────────────────────────────────

function NodeSelectCard({
  node, selected, onToggle, maxCams, onMaxCamsChange,
}: {
  node:            GpuNode;
  selected:        boolean;
  onToggle:        () => void;
  maxCams:         number;
  onMaxCamsChange: (v: number) => void;
}) {
  const h       = node.health_json ?? {};
  const gpuPct  = typeof h["gpu0_utilization"] === "number" ? h["gpu0_utilization"] as number : null;
  const camLoad = node.max_cameras > 0 ? (node.active_cameras / node.max_cameras) * 100 : 0;

  return (
    <div
      className={cn(
        "rounded-xl border-2 p-4 cursor-pointer transition-all",
        selected ? "border-zinc-400 bg-zinc-800/60" : "border-zinc-800 hover:border-zinc-600 bg-card",
      )}
      onClick={onToggle}
    >
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className={cn(
            "flex h-6 w-6 items-center justify-center rounded-full border-2 shrink-0 transition-colors",
            selected ? "border-zinc-300 bg-zinc-300" : "border-zinc-600",
          )}>
            {selected && <Check className="h-3.5 w-3.5 text-zinc-900" />}
          </div>
          <div>
            <p className="text-sm font-semibold text-zinc-100">{node.node_name}</p>
            <p className="text-xs text-muted-foreground">{node.location ?? "—"}</p>
          </div>
        </div>
        <Badge
          variant={node.status === "ONLINE" ? "success" : "secondary"}
          className="text-[10px]"
        >
          {node.status}
        </Badge>
      </div>

      {/* GPU model */}
      {node.gpu_model && (
        <p className="text-xs text-zinc-500 mb-2 flex items-center gap-1">
          <Server className="h-3 w-3" />{node.gpu_model}
        </p>
      )}

      {/* Utilisation bars */}
      <div className="space-y-1.5">
        <div className="flex items-center justify-between text-[10px] text-zinc-500">
          <span>GPU {gpuPct != null ? `${gpuPct.toFixed(0)}%` : "—"}</span>
          <span>Cameras {node.active_cameras}/{node.max_cameras}</span>
        </div>
        {gpuPct != null && (
          <Progress
            value={gpuPct}
            className={cn("h-1", gpuPct > 85 ? "[&>div]:bg-red-400" : gpuPct > 65 ? "[&>div]:bg-amber-400" : "")}
          />
        )}
        <Progress
          value={camLoad}
          className={cn("h-1", camLoad > 85 ? "[&>div]:bg-red-400" : "")}
        />
      </div>

      {/* Per-node limit (only shown when selected) */}
      {selected && (
        <div className="mt-3 pt-3 border-t border-zinc-700" onClick={(e) => e.stopPropagation()}>
          <RangeSlider
            label="Cameras on this node"
            min={1} max={Math.min(node.max_cameras, 50)} value={maxCams}
            onChange={onMaxCamsChange}
          />
        </div>
      )}
    </div>
  );
}

function Step3Nodes({
  state, setState,
}: {
  state: ReturnType<typeof EMPTY_STATE>;
  setState: React.Dispatch<React.SetStateAction<ReturnType<typeof EMPTY_STATE>>>;
}) {
  const { data: nodesRaw, isLoading } = useQuery({
    queryKey: ["admin-all-nodes"],
    queryFn:  () => api.get("/api/nodes").then((r) => {
      const d = r.data as { nodes?: GpuNode[] } | GpuNode[];
      return Array.isArray(d) ? d : (d as { nodes?: GpuNode[] }).nodes ?? [];
    }),
  });
  const nodes: GpuNode[] = (nodesRaw as GpuNode[] | undefined) ?? [];

  function toggleNode(nodeId: string, maxCams: number) {
    setState((s) => {
      const exists = s.selectedNodes.find((n) => n.node_id === nodeId);
      return {
        ...s,
        selectedNodes: exists
          ? s.selectedNodes.filter((n) => n.node_id !== nodeId)
          : [...s.selectedNodes, { node_id: nodeId, max_cameras_on_node: maxCams }],
      };
    });
  }

  function updateNodeCams(nodeId: string, v: number) {
    setState((s) => ({
      ...s,
      selectedNodes: s.selectedNodes.map((n) =>
        n.node_id === nodeId ? { ...n, max_cameras_on_node: v } : n,
      ),
    }));
  }

  if (isLoading) {
    return (
      <div className="grid grid-cols-2 gap-3">
        {[1, 2, 3, 4].map((i) => <Skeleton key={i} className="h-40 rounded-xl" />)}
      </div>
    );
  }

  if (nodes.length === 0) {
    return (
      <div className="flex h-40 items-center justify-center rounded-xl border border-dashed border-border/50">
        <div className="text-center">
          <Server className="mx-auto h-6 w-6 text-zinc-600 mb-2" />
          <p className="text-sm text-muted-foreground">No GPU nodes registered</p>
          <p className="text-xs text-zinc-600 mt-0.5">Nodes will appear here once they register with the Control Plane</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <p className="text-xs text-muted-foreground">
        Select nodes this client can use. Set a per-node camera limit after selecting.
        <span className="ml-2 text-zinc-400">
          {state.selectedNodes.length} node{state.selectedNodes.length !== 1 ? "s" : ""} selected
        </span>
      </p>
      <div className="grid grid-cols-2 gap-3 max-h-[420px] overflow-y-auto pr-1">
        {nodes.map((n) => {
          const assignment = state.selectedNodes.find((a) => a.node_id === n.node_id);
          return (
            <NodeSelectCard
              key={n.node_id}
              node={n}
              selected={!!assignment}
              onToggle={() => toggleNode(n.node_id, Math.min(10, n.max_cameras))}
              maxCams={assignment?.max_cameras_on_node ?? 10}
              onMaxCamsChange={(v) => updateNodeCams(n.node_id, v)}
            />
          );
        })}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 4 — Initial Admin
// ─────────────────────────────────────────────────────────────────────────────

function Step4Admin({
  state, setState,
}: {
  state: ReturnType<typeof EMPTY_STATE>;
  setState: React.Dispatch<React.SetStateAction<ReturnType<typeof EMPTY_STATE>>>;
}) {
  const set = (k: string, v: string | boolean) => setState((s) => ({ ...s, [k]: v }));
  return (
    <div className="space-y-4">
      <p className="text-xs text-muted-foreground">
        An initial CLIENT_ADMIN account will be created and can change their password on first login.
      </p>
      <div className="space-y-1.5">
        <Label className="text-xs text-muted-foreground">Admin full name *</Label>
        <Input value={state.admin_name} onChange={(e) => set("admin_name", e.target.value)}
          placeholder="Dr. Jane Smith" className="bg-zinc-900 border-zinc-700" />
      </div>
      <div className="space-y-1.5">
        <Label className="text-xs text-muted-foreground">Admin email *</Label>
        <Input type="email" value={state.admin_email} onChange={(e) => set("admin_email", e.target.value)}
          placeholder="jane@institute.edu" className="bg-zinc-900 border-zinc-700" />
      </div>
      <div className="space-y-1.5">
        <Label className="text-xs text-muted-foreground">Temporary password *</Label>
        <div className="relative">
          <Input
            type={state.showPass ? "text" : "password"}
            value={state.admin_password}
            onChange={(e) => set("admin_password", e.target.value)}
            placeholder="Min. 10 chars, 1 upper, 1 number, 1 special"
            className="bg-zinc-900 border-zinc-700 pr-10"
          />
          <button type="button" tabIndex={-1}
            onClick={() => set("showPass", !state.showPass)}
            className="absolute inset-y-0 right-0 flex items-center px-3 text-zinc-500 hover:text-zinc-300">
            {state.showPass ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
          </button>
        </div>
        <div className="flex gap-3 mt-1.5">
          {[
            { label: "10+ chars", ok: state.admin_password.length >= 10 },
            { label: "Uppercase",  ok: /[A-Z]/.test(state.admin_password) },
            { label: "Number",     ok: /\d/.test(state.admin_password) },
            { label: "Special",    ok: /[!@#$%^&*]/.test(state.admin_password) },
          ].map(({ label, ok }) => (
            <span key={label} className={cn("flex items-center gap-1 text-[10px]", ok ? "text-green-400" : "text-zinc-600")}>
              <span className={cn("h-1.5 w-1.5 rounded-full", ok ? "bg-green-400" : "bg-zinc-700")} />
              {label}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

function validateStep4(s: ReturnType<typeof EMPTY_STATE>): string | null {
  if (!s.admin_name.trim())    return "Admin name required";
  if (!s.admin_email.trim())   return "Admin email required";
  if (s.admin_password.length < 10) return "Password must be at least 10 characters";
  if (!/[A-Z]/.test(s.admin_password)) return "Password needs an uppercase letter";
  if (!/\d/.test(s.admin_password))    return "Password needs a number";
  if (!/[!@#$%^&*]/.test(s.admin_password)) return "Password needs a special character";
  return null;
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 5 — Review
// ─────────────────────────────────────────────────────────────────────────────

function Step5Review({
  state, nodes,
}: {
  state: ReturnType<typeof EMPTY_STATE>;
  nodes: GpuNode[];
}) {
  const Row = ({ label, value }: { label: string; value: string }) => (
    <div className="flex justify-between text-sm py-1 border-b border-border/30 last:border-0">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-medium text-zinc-200 text-right max-w-[60%] truncate">{value || "—"}</span>
    </div>
  );
  return (
    <div className="space-y-4 max-h-[420px] overflow-y-auto pr-1">
      <div className="rounded-xl bg-zinc-900/60 p-4 space-y-0.5">
        <p className="text-xs font-semibold uppercase tracking-wider text-zinc-500 mb-2">Organisation</p>
        <Row label="Name"          value={state.name} />
        <Row label="Slug"          value={state.slug} />
        <Row label="Contact"       value={`${state.contact_name} · ${state.contact_email}`} />
        <Row label="Phone"         value={state.contact_phone} />
        <Row label="Address"       value={state.address} />
      </div>
      <div className="rounded-xl bg-zinc-900/60 p-4 space-y-0.5">
        <p className="text-xs font-semibold uppercase tracking-wider text-zinc-500 mb-2">Limits</p>
        <Row label="Max Cameras"  value={state.max_cameras.toLocaleString()} />
        <Row label="Max Persons"  value={state.max_persons.toLocaleString()} />
      </div>
      <div className="rounded-xl bg-zinc-900/60 p-4">
        <p className="text-xs font-semibold uppercase tracking-wider text-zinc-500 mb-2">
          GPU Nodes ({state.selectedNodes.length})
        </p>
        {state.selectedNodes.length === 0 ? (
          <p className="text-xs text-zinc-600">No nodes assigned (can add later)</p>
        ) : (
          state.selectedNodes.map((a) => {
            const n = nodes.find((nd) => nd.node_id === a.node_id);
            return (
              <div key={a.node_id} className="flex justify-between text-sm py-1 border-b border-border/30 last:border-0">
                <span className="text-muted-foreground">{n?.node_name ?? a.node_id}</span>
                <span className="font-medium text-zinc-300">{a.max_cameras_on_node} cam limit</span>
              </div>
            );
          })
        )}
      </div>
      <div className="rounded-xl bg-zinc-900/60 p-4 space-y-0.5">
        <p className="text-xs font-semibold uppercase tracking-wider text-zinc-500 mb-2">Initial Admin</p>
        <Row label="Name"  value={state.admin_name} />
        <Row label="Email" value={state.admin_email} />
        <Row label="Pass"  value={"•".repeat(state.admin_password.length)} />
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Create Client Dialog
// ─────────────────────────────────────────────────────────────────────────────

function CreateClientDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const qc = useQueryClient();
  const [step,  setStep]  = useState(0);
  const [state, setState] = useState<ReturnType<typeof EMPTY_STATE>>(EMPTY_STATE);
  const [error, setError] = useState<string | null>(null);

  // Fetch nodes for review step
  const { data: nodesRaw } = useQuery({
    queryKey: ["admin-all-nodes"],
    queryFn:  () => api.get("/api/nodes").then((r) => {
      const d = r.data as { nodes?: GpuNode[] } | GpuNode[];
      return Array.isArray(d) ? d : (d as { nodes?: GpuNode[] }).nodes ?? [];
    }),
    enabled: open,
  });
  const nodes: GpuNode[] = (nodesRaw as GpuNode[] | undefined) ?? [];

  const mutation = useMutation({
    mutationFn: async () => {
      // Build payload
      const fd = new FormData();
      if (state.logo_file) fd.append("logo", state.logo_file);

      const payload = {
        name:          state.name,
        slug:          state.slug,
        contact_name:  state.contact_name,
        contact_email: state.contact_email,
        contact_phone: state.contact_phone || undefined,
        address:       state.address || undefined,
        max_cameras:   state.max_cameras,
        max_persons:   state.max_persons,
        node_assignments: state.selectedNodes,
        initial_admin: {
          name:          state.admin_name,
          email:         state.admin_email,
          temp_password: state.admin_password,
        },
      };
      return adminApi.createClient(payload);
    },
    onSuccess: () => {
      toast.success(`Client "${state.name}" created`);
      qc.invalidateQueries({ queryKey: ["clients"] });
      handleClose();
    },
    onError: (err: unknown) => {
      const data = (err as { response?: { data?: { detail?: unknown } } })?.response?.data;
      const detail = data?.detail;
      let msg = "Failed to create client";
      if (typeof detail === "string") {
        msg = detail;
      } else if (Array.isArray(detail) && detail.length > 0) {
        // Pydantic validation error — pick the first message
        const first = detail[0] as { msg?: string; loc?: string[] };
        const field = first.loc ? first.loc.slice(-1)[0] : "";
        msg = field ? `${field}: ${first.msg ?? "invalid"}` : (first.msg ?? msg);
      }
      toast.error(msg);
    },
  });

  function handleClose() {
    setStep(0);
    setState(EMPTY_STATE());
    setError(null);
    onClose();
  }

  function next() {
    setError(null);
    if (step === 0) {
      const err = validateStep1(state);
      if (err) { setError(err); return; }
    }
    if (step === 3) {
      const err = validateStep4(state);
      if (err) { setError(err); return; }
    }
    if (step < 4) setStep((s) => s + 1);
    else mutation.mutate();
  }

  return (
    <Dialog open={open} onOpenChange={(o) => !o && handleClose()}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle>Create Client</DialogTitle>
          <DialogDescription>
            Set up a new organisation in VGI ACFR with GPU node assignments.
          </DialogDescription>
        </DialogHeader>

        <div className="flex-1 overflow-y-auto px-1 py-2">
          <StepIndicator current={step} />

          {step === 0 && <Step1Info state={state} setState={setState} />}
          {step === 1 && <Step2Limits state={state} setState={setState} />}
          {step === 2 && <Step3Nodes state={state} setState={setState} />}
          {step === 3 && <Step4Admin state={state} setState={setState} />}
          {step === 4 && <Step5Review state={state} nodes={nodes} />}

          {error && (
            <p className="mt-3 text-sm text-red-400 bg-red-400/10 rounded-lg px-3 py-2">{error}</p>
          )}
        </div>

        <div className="flex items-center justify-between border-t border-border/50 pt-4 mt-2">
          <Button
            variant="ghost" size="sm"
            onClick={() => step === 0 ? handleClose() : setStep((s) => s - 1)}
          >
            {step === 0 ? "Cancel" : "Back"}
          </Button>
          <Button onClick={next} disabled={mutation.isPending}>
            {mutation.isPending ? (
              <><Loader2 className="h-4 w-4 animate-spin" />Creating…</>
            ) : step === 4 ? (
              <><Check className="h-4 w-4" />Create Client</>
            ) : (
              "Continue →"
            )}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Client Detail Dialog
// ─────────────────────────────────────────────────────────────────────────────

function ClientDetailDialog({ client, onClose }: { client: Client | null; onClose: () => void }) {
  const qc = useQueryClient();

  const { data: detailRaw, isLoading } = useQuery({
    queryKey: ["client-detail", client?.client_id],
    queryFn:  () => adminApi.client(client!.client_id).then((r) => r.data),
    enabled:  !!client,
    staleTime: 30_000,
    refetchOnWindowFocus: false,
  });
  const detail = detailRaw as (Client & {
    camera_count?: number;
    person_count?: number;
    active_sessions?: number;
    node_assignments?: Array<{ node_id: string; node_name?: string; max_cameras_on_node: number; active: number }>;
    users?: Array<{ user_id: string; name: string; email: string; role: string; status: string }>;
  }) | undefined;

  const statusMutation = useMutation({
    mutationFn: (status: string) => adminApi.clientStatus(client!.client_id, { status }),
    onSuccess: () => {
      toast.success("Status updated");
      qc.invalidateQueries({ queryKey: ["clients"] });
      qc.invalidateQueries({ queryKey: ["client-detail", client?.client_id] });
    },
    onError: () => toast.error("Update failed"),
  });

  const removeNodeMutation = useMutation({
    mutationFn: (nodeId: string) => adminApi.removeNode(client!.client_id, nodeId),
    onSuccess: () => {
      toast.success("Node removed");
      qc.invalidateQueries({ queryKey: ["client-detail", client?.client_id] });
    },
    onError: () => toast.error("Failed to remove node"),
  });

  // ── Edit-Limits panel state ─────────────────────────────────────────────────
  const [showEditLimits, setShowEditLimits]   = useState(false);
  const [editMaxCams,    setEditMaxCams]       = useState(50);
  const [editMaxPersons, setEditMaxPersons]   = useState(10000);

  const editLimitsMutation = useMutation({
    mutationFn: () => adminApi.updateClient(client!.client_id, {
      max_cameras: editMaxCams,
      max_persons: editMaxPersons,
    }),
    onSuccess: () => {
      toast.success("Limits updated");
      qc.invalidateQueries({ queryKey: ["clients"] });
      qc.invalidateQueries({ queryKey: ["client-detail", client?.client_id] });
      setShowEditLimits(false);
    },
    onError: () => toast.error("Failed to update limits"),
  });

  // ── Add-Node panel state ────────────────────────────────────────────────────
  const [showAddNode, setShowAddNode]         = useState(false);
  const [selectedNodeId, setSelectedNodeId]   = useState<string>("");
  const [addNodeMaxCams, setAddNodeMaxCams]   = useState(10);

  const { data: allNodesRaw } = useQuery({
    queryKey:  ["admin-nodes"],
    queryFn:   () => adminApi.listNodes().then((r) => r.data),
    enabled:   showAddNode,
    staleTime: 30_000,
  });
  const allNodes = (Array.isArray(allNodesRaw) ? allNodesRaw : []) as Array<{
    node_id: string; node_name: string; location: string | null;
    status: string; gpu_model: string | null; active_cameras: number; max_cameras: number;
  }>;
  const assignedIds = new Set((detail?.node_assignments ?? []).map((a) => a.node_id));
  const availableNodes = allNodes.filter((n) => !assignedIds.has(n.node_id));

  const addNodeMutation = useMutation({
    mutationFn: () => adminApi.assignNode(client!.client_id, {
      node_id: selectedNodeId,
      max_cameras_on_node: addNodeMaxCams,
    }),
    onSuccess: () => {
      toast.success("Node assigned");
      qc.invalidateQueries({ queryKey: ["client-detail", client?.client_id] });
      setShowAddNode(false);
      setSelectedNodeId("");
      setAddNodeMaxCams(10);
    },
    onError: (err: unknown) => {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      toast.error(detail ?? "Failed to assign node");
    },
  });

  if (!client) return null;

  const c = detail ?? client;
  const cameraCount   = detail?.camera_count   ?? (client as Client & { camera_count?: number })?.camera_count  ?? 0;
  const personCount   = detail?.person_count   ?? (client as Client & { person_count?: number })?.person_count  ?? 0;
  const activeSessions = detail?.active_sessions ?? (client as Client & { active_sessions?: number })?.active_sessions ?? 0;
  const cameraUsePct  = c.max_cameras > 0 ? (cameraCount / c.max_cameras) * 100 : 0;
  const personUsePct  = c.max_persons > 0 ? (personCount / c.max_persons) * 100 : 0;

  return (
    <Dialog open={!!client} onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-w-3xl max-h-[90vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <div className="flex items-center gap-3 pr-6">
            {c.logo_url ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={c.logo_url} alt={c.name} className="h-9 w-9 rounded-lg object-contain border border-border/50" />
            ) : (
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-zinc-800 font-bold text-sm text-zinc-300">
                {c.name[0]}
              </div>
            )}
            <div>
              <DialogTitle>{c.name}</DialogTitle>
              <p className="text-xs text-zinc-500 font-mono mt-0.5">{c.slug}</p>
            </div>
            <Badge
              variant={c.status === "ACTIVE" ? "success" : c.status === "SUSPENDED" ? "warning" : "secondary"}
              className="ml-auto mr-2 text-xs"
            >
              {c.status}
            </Badge>
          </div>
        </DialogHeader>

        <div className="flex-1 overflow-y-auto">
          <Tabs defaultValue="overview">
            <TabsList className="w-full justify-start">
              <TabsTrigger value="overview">Overview</TabsTrigger>
              <TabsTrigger value="nodes">GPU Nodes</TabsTrigger>
              <TabsTrigger value="users">Users</TabsTrigger>
            </TabsList>

            {/* ── Overview ───────────────────────────────── */}
            <TabsContent value="overview" className="space-y-4 pt-2">
              {isLoading ? (
                <div className="grid grid-cols-2 gap-3">
                  {[1,2,3,4].map((i) => <Skeleton key={i} className="h-20 rounded-xl" />)}
                </div>
              ) : (
                <>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="card-stat">
                      <p className="text-xs text-muted-foreground">Cameras</p>
                      <p className="text-2xl font-bold tabular-nums text-zinc-100 mt-1">
                        {cameraCount} <span className="text-sm text-zinc-500">/ {c.max_cameras}</span>
                      </p>
                      <Progress value={cameraUsePct} className="mt-2 h-1.5" />
                    </div>
                    <div className="card-stat">
                      <p className="text-xs text-muted-foreground">Enrolled Persons</p>
                      <p className="text-2xl font-bold tabular-nums text-zinc-100 mt-1">
                        {personCount.toLocaleString()} <span className="text-sm text-zinc-500">/ {c.max_persons.toLocaleString()}</span>
                      </p>
                      <Progress value={personUsePct} className="mt-2 h-1.5" />
                    </div>
                    <div className="card-stat">
                      <p className="text-xs text-muted-foreground">Active Sessions</p>
                      <p className="text-2xl font-bold tabular-nums text-zinc-100 mt-1">{activeSessions}</p>
                    </div>
                    <div className="card-stat">
                      <p className="text-xs text-muted-foreground">Assigned Nodes</p>
                      <p className="text-2xl font-bold tabular-nums text-zinc-100 mt-1">
                        {detail?.node_assignments?.length ?? 0}
                      </p>
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex flex-wrap gap-2 pt-2">
                    {c.status === "ACTIVE" ? (
                      <>
                        <Button size="sm" variant="outline"
                          className="text-amber-400 border-amber-400/30 hover:bg-amber-400/10"
                          onClick={() => statusMutation.mutate("SUSPENDED")}
                          disabled={statusMutation.isPending}
                        >
                          Suspend
                        </Button>
                        <Button size="sm" variant="outline"
                          className="text-red-400 border-red-400/30 hover:bg-red-400/10"
                          onClick={() => {
                            if (confirm(`Archive "${c.name}"? The client will be permanently deactivated (data is kept but logins are blocked).`)) {
                              statusMutation.mutate("ARCHIVED");
                            }
                          }}
                          disabled={statusMutation.isPending}
                        >
                          Archive (Deactivate)
                        </Button>
                      </>
                    ) : (
                      <Button size="sm" variant="outline"
                        className="text-green-400 border-green-400/30 hover:bg-green-400/10"
                        onClick={() => statusMutation.mutate("ACTIVE")}
                        disabled={statusMutation.isPending}
                      >
                        Re-Activate
                      </Button>
                    )}
                    <Button size="sm" variant="outline"
                      onClick={() => {
                        setEditMaxCams(c.max_cameras);
                        setEditMaxPersons(c.max_persons);
                        setShowEditLimits((v) => !v);
                      }}
                    >
                      Edit Limits
                    </Button>
                  </div>

                  {/* ── Inline Edit-Limits panel ── */}
                  {showEditLimits && (
                    <div className="rounded-xl border border-zinc-700/60 bg-zinc-900/60 p-4 space-y-4">
                      <p className="text-xs font-semibold text-zinc-300 uppercase tracking-wide">Edit Limits</p>
                      <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-1.5">
                          <Label className="text-xs text-zinc-400">Max Cameras</Label>
                          <Input
                            type="number" min={1} max={500}
                            value={editMaxCams}
                            onChange={(e) => setEditMaxCams(Math.max(1, Number(e.target.value)))}
                            className="h-8 bg-zinc-800 border-zinc-700 text-sm"
                          />
                          <p className="text-[10px] text-zinc-600">Currently using {cameraCount}</p>
                        </div>
                        <div className="space-y-1.5">
                          <Label className="text-xs text-zinc-400">Max Persons</Label>
                          <Input
                            type="number" min={100} max={1000000} step={100}
                            value={editMaxPersons}
                            onChange={(e) => setEditMaxPersons(Math.max(100, Number(e.target.value)))}
                            className="h-8 bg-zinc-800 border-zinc-700 text-sm"
                          />
                          <p className="text-[10px] text-zinc-600">Currently enrolled {personCount.toLocaleString()}</p>
                        </div>
                      </div>
                      <div className="flex gap-2 justify-end">
                        <Button size="sm" variant="ghost" onClick={() => setShowEditLimits(false)}>Cancel</Button>
                        <Button
                          size="sm"
                          disabled={editLimitsMutation.isPending}
                          onClick={() => editLimitsMutation.mutate()}
                        >
                          {editLimitsMutation.isPending ? "Saving…" : "Save Limits"}
                        </Button>
                      </div>
                    </div>
                  )}
                </>
              )}
            </TabsContent>

            {/* ── Nodes ─────────────────────────────────── */}
            <TabsContent value="nodes" className="space-y-3 pt-2">
              {isLoading ? (
                <Skeleton className="h-40 w-full rounded-xl" />
              ) : (
                <>
                  <div className="flex items-center justify-between">
                    <p className="text-xs text-muted-foreground">
                      {detail?.node_assignments?.length ?? 0} node{(detail?.node_assignments?.length ?? 0) !== 1 ? "s" : ""} assigned
                    </p>
                    <Button size="sm" variant="outline" onClick={() => setShowAddNode((v) => !v)}>
                      <Plus className="h-3.5 w-3.5" />Add Node
                    </Button>
                  </div>

                  {/* ── Inline Add-Node panel ── */}
                  {showAddNode && (
                    <div className="rounded-xl border border-zinc-700/60 bg-zinc-900/60 p-3 space-y-3">
                      <p className="text-xs font-medium text-zinc-300">Select a registered node to assign</p>
                      {availableNodes.length === 0 ? (
                        <p className="text-xs text-zinc-500 py-2 text-center">
                          {allNodes.length === 0
                            ? "No GPU nodes registered. Register one from the GPU Nodes page first."
                            : "All registered nodes are already assigned to this client."}
                        </p>
                      ) : (
                        <div className="space-y-1.5 max-h-48 overflow-y-auto">
                          {availableNodes.map((n) => (
                            <button
                              key={n.node_id}
                              onClick={() => setSelectedNodeId(n.node_id)}
                              className={cn(
                                "w-full flex items-center gap-3 rounded-lg border px-3 py-2 text-left transition-colors",
                                selectedNodeId === n.node_id
                                  ? "border-zinc-400 bg-zinc-800"
                                  : "border-zinc-700/50 bg-zinc-800/30 hover:border-zinc-600",
                              )}
                            >
                              <div className={cn(
                                "h-2 w-2 rounded-full shrink-0",
                                n.status === "ONLINE" ? "bg-green-500" : "bg-zinc-600"
                              )} />
                              <div className="flex-1 min-w-0">
                                <p className="text-sm font-medium text-zinc-200 truncate">{n.node_name}</p>
                                <p className="text-xs text-zinc-500">
                                  {n.location ?? "Unknown location"} · {n.active_cameras}/{n.max_cameras} cams
                                  {n.gpu_model ? ` · ${n.gpu_model}` : ""}
                                </p>
                              </div>
                              <span className={cn(
                                "text-[10px] rounded-full px-1.5 py-0.5",
                                n.status === "ONLINE" ? "bg-green-500/15 text-green-400" : "bg-zinc-700 text-zinc-500"
                              )}>{n.status}</span>
                            </button>
                          ))}
                        </div>
                      )}
                      {selectedNodeId && (
                        <div className="flex items-center gap-3 pt-1 border-t border-zinc-700/40">
                          <Label className="text-xs text-zinc-400 shrink-0">Max cameras on this node</Label>
                          <Input
                            type="number" min={1} max={100}
                            value={addNodeMaxCams}
                            onChange={(e) => setAddNodeMaxCams(Math.max(1, Number(e.target.value)))}
                            className="w-20 h-7 bg-zinc-800 border-zinc-700 text-xs"
                          />
                        </div>
                      )}
                      <div className="flex gap-2 justify-end">
                        <Button size="sm" variant="ghost" onClick={() => { setShowAddNode(false); setSelectedNodeId(""); }}>
                          Cancel
                        </Button>
                        <Button
                          size="sm"
                          disabled={!selectedNodeId || addNodeMutation.isPending}
                          onClick={() => addNodeMutation.mutate()}
                        >
                          {addNodeMutation.isPending ? "Assigning…" : "Assign Node"}
                        </Button>
                      </div>
                    </div>
                  )}

                  <div className="space-y-2">
                    {(detail?.node_assignments ?? []).map((a) => (
                      <div key={a.node_id}
                        className="flex items-center gap-3 rounded-lg border border-border/50 bg-card px-3 py-2.5">
                        <Server className="h-4 w-4 text-zinc-500 shrink-0" />
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium text-zinc-200">{a.node_name ?? a.node_id}</p>
                          <p className="text-xs text-zinc-500">{a.active ?? 0} active / {a.max_cameras_on_node} limit</p>
                        </div>
                        <Progress value={a.max_cameras_on_node > 0 ? ((a.active ?? 0) / a.max_cameras_on_node) * 100 : 0}
                          className="w-24 h-1.5" />
                        <button
                          onClick={() => removeNodeMutation.mutate(a.node_id)}
                          className="ml-2 text-zinc-600 hover:text-red-400 transition-colors"
                          title="Remove node"
                        >
                          <Trash2 className="h-4 w-4" />
                        </button>
                      </div>
                    ))}
                    {(detail?.node_assignments?.length ?? 0) === 0 && !showAddNode && (
                      <p className="text-sm text-muted-foreground py-4 text-center">No nodes assigned</p>
                    )}
                  </div>
                </>
              )}
            </TabsContent>

            {/* ── Users ─────────────────────────────────── */}
            <TabsContent value="users" className="pt-2">
              {isLoading ? (
                <Skeleton className="h-40 w-full rounded-xl" />
              ) : (
                <div className="space-y-1.5">
                  {(detail?.users ?? []).map((u) => (
                    <div key={u.user_id}
                      className="flex items-center gap-3 rounded-lg border border-border/40 bg-card px-3 py-2.5">
                      <div className="flex h-7 w-7 items-center justify-center rounded-full bg-zinc-800 text-xs font-medium text-zinc-300 shrink-0">
                        {u.name.split(" ").map((w: string) => w[0]).join("").slice(0, 2).toUpperCase()}
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-zinc-200">{u.name}</p>
                        <p className="text-xs text-zinc-500">{u.email}</p>
                      </div>
                      <Badge variant={u.role === "CLIENT_ADMIN" ? "info" : "secondary"} className="text-[10px]">
                        {u.role.replace("_", " ")}
                      </Badge>
                      <Badge variant={u.status === "ACTIVE" ? "success" : "secondary"} className="text-[10px]">
                        {u.status}
                      </Badge>
                    </div>
                  ))}
                  {(detail?.users?.length ?? 0) === 0 && (
                    <p className="text-sm text-muted-foreground py-4 text-center">No users</p>
                  )}
                </div>
              )}
            </TabsContent>
          </Tabs>
        </div>
      </DialogContent>
    </Dialog>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Clients Table
// ─────────────────────────────────────────────────────────────────────────────

const STATUS_COLOR: Record<string, string> = {
  ACTIVE:   "text-green-400 bg-green-400/10",
  SUSPENDED:"text-amber-400 bg-amber-400/10",
  ARCHIVED: "text-zinc-500 bg-zinc-500/10",
};

function UsageBar({ used, max, warn = 80 }: { used: number; max: number; warn?: number }) {
  const pct = max > 0 ? (used / max) * 100 : 0;
  return (
    <div className="space-y-0.5">
      <div className="flex justify-between text-[10px] text-zinc-500">
        <span>{used.toLocaleString()}</span>
        <span>{max.toLocaleString()}</span>
      </div>
      <div className="h-1 w-20 rounded-full bg-zinc-800 overflow-hidden">
        <div
          className={cn("h-full rounded-full transition-all", pct >= warn ? "bg-amber-400" : "bg-zinc-400")}
          style={{ width: `${Math.min(pct, 100)}%` }}
        />
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Page
// ─────────────────────────────────────────────────────────────────────────────

export default function ClientsPage() {
  const [search,       setSearch]       = useState("");
  const [statusFilter, setStatusFilter] = useState<string>("ALL");
  const [showCreate,   setShowCreate]   = useState(false);
  const [selected,     setSelected]     = useState<Client | null>(null);

  const { data, isLoading } = useQuery({
    queryKey: ["clients", search, statusFilter],
    queryFn:  () =>
      adminApi.clients({
        q:      search || undefined,
        status: statusFilter !== "ALL" ? statusFilter : undefined,
        limit:  200,
      }).then((r) => r.data),
  });

  const clients: Client[] = (data as { items?: Client[] })?.items ?? [];

  const counts = {
    all:       clients.length,
    active:    clients.filter((c) => c.status === "ACTIVE").length,
    suspended: clients.filter((c) => c.status === "SUSPENDED").length,
    archived:  clients.filter((c) => c.status === "ARCHIVED").length,
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold text-zinc-50">Clients</h1>
          <p className="text-sm text-muted-foreground mt-0.5">
            {counts.active} active · {counts.suspended} suspended · {counts.archived} archived
          </p>
        </div>
        <Button size="sm" onClick={() => setShowCreate(true)}>
          <Plus className="h-4 w-4" />
          Create Client
        </Button>
      </div>

      {/* Filters */}
      <div className="flex items-center gap-3 flex-wrap">
        <div className="relative">
          <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-zinc-500" />
          <Input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search by name or slug…"
            className="pl-8 bg-zinc-800 border-zinc-700 w-56"
          />
        </div>
        <div className="flex gap-1 rounded-lg border border-border/50 bg-zinc-900 p-1">
          {[
            { key: "ALL",       label: "All",       count: counts.all },
            { key: "ACTIVE",    label: "Active",    count: counts.active },
            { key: "SUSPENDED", label: "Suspended", count: counts.suspended },
          ].map(({ key, label, count }) => (
            <button
              key={key}
              onClick={() => setStatusFilter(key)}
              className={cn(
                "flex items-center gap-1.5 rounded px-3 py-1.5 text-xs font-medium transition-colors",
                statusFilter === key ? "bg-zinc-700 text-zinc-100" : "text-zinc-500 hover:text-zinc-300",
              )}
            >
              {label}
              <span className={cn(
                "rounded-full px-1.5 py-0 text-[10px]",
                statusFilter === key ? "bg-zinc-600" : "bg-zinc-800",
              )}>{count}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Table */}
      {isLoading ? (
        <div className="space-y-1.5">
          {Array.from({ length: 6 }).map((_, i) => <Skeleton key={i} className="h-14 rounded-lg" />)}
        </div>
      ) : (
        <div className="overflow-hidden rounded-xl border border-border/50">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border/50 bg-zinc-900/50">
                <th className="py-3 px-4 text-left text-xs font-medium text-muted-foreground">Organisation</th>
                <th className="py-3 px-3 text-left text-xs font-medium text-muted-foreground">Status</th>
                <th className="py-3 px-3 text-left text-xs font-medium text-muted-foreground">Cameras</th>
                <th className="py-3 px-3 text-left text-xs font-medium text-muted-foreground">Persons</th>
                <th className="py-3 px-3 text-center text-xs font-medium text-muted-foreground">Sessions</th>
                <th className="py-3 px-3 text-left text-xs font-medium text-muted-foreground">Created</th>
                <th className="py-3 px-3 text-right text-xs font-medium text-muted-foreground">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/40">
              {clients.map((c) => (
                <tr
                  key={c.client_id}
                  className="hover:bg-zinc-800/30 transition-colors cursor-pointer"
                  onClick={() => setSelected(c)}
                >
                  {/* Name */}
                  <td className="py-3 px-4">
                    <div className="flex items-center gap-3">
                      {c.logo_url ? (
                        // eslint-disable-next-line @next/next/no-img-element
                        <img src={c.logo_url} alt={c.name} className="h-7 w-7 rounded object-contain border border-border/40 shrink-0" />
                      ) : (
                        <div className="flex h-7 w-7 items-center justify-center rounded bg-zinc-700 text-xs font-bold text-zinc-300 shrink-0">
                          {c.name[0]}
                        </div>
                      )}
                      <div className="min-w-0">
                        <p className="font-medium text-zinc-100 truncate">{c.name}</p>
                        <p className="text-xs text-zinc-500 font-mono">{c.slug}</p>
                      </div>
                    </div>
                  </td>

                  {/* Status */}
                  <td className="py-3 px-3">
                    <span className={cn("rounded-full px-2 py-0.5 text-[11px] font-medium", STATUS_COLOR[c.status])}>
                      {c.status}
                    </span>
                  </td>

                  {/* Cameras */}
                  <td className="py-3 px-3">
                    <UsageBar used={(c as Client & { camera_count?: number }).camera_count ?? 0} max={c.max_cameras} />
                  </td>

                  {/* Persons */}
                  <td className="py-3 px-3">
                    <UsageBar used={(c as Client & { person_count?: number }).person_count ?? 0} max={c.max_persons} />
                  </td>

                  {/* Active sessions */}
                  <td className="py-3 px-3 text-center">
                    {((c as Client & { active_sessions?: number }).active_sessions ?? 0) > 0 ? (
                      <Badge variant="success" className="text-[10px]">{(c as Client & { active_sessions?: number }).active_sessions} active</Badge>
                    ) : (
                      <span className="text-xs text-zinc-600">—</span>
                    )}
                  </td>

                  {/* Created */}
                  <td className="py-3 px-3 text-xs text-zinc-500">
                    {/* created_at not in Client type; show slug as fallback indicator */}
                    {fmtDatetime((c as unknown as { created_at?: string }).created_at)}
                  </td>

                  {/* Actions */}
                  <td className="py-3 px-3 text-right">
                    <button
                      onClick={(e) => { e.stopPropagation(); setSelected(c); }}
                      className="inline-flex items-center gap-1 rounded px-2 py-1 text-xs text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800 transition-colors"
                    >
                      <Eye className="h-3.5 w-3.5" />View
                    </button>
                  </td>
                </tr>
              ))}
              {clients.length === 0 && (
                <tr>
                  <td colSpan={7} className="py-12 text-center">
                    <Building2 className="mx-auto h-7 w-7 text-zinc-700 mb-2" />
                    <p className="text-sm text-muted-foreground">No clients found</p>
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}

      {/* Dialogs */}
      <CreateClientDialog open={showCreate} onClose={() => setShowCreate(false)} />
      <ClientDetailDialog client={selected} onClose={() => setSelected(null)} />
    </div>
  );
}
