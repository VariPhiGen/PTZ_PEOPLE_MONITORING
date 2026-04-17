"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Camera,
  Cpu,
  HardDrive,
  MemoryStick,
  Plus,
  RefreshCw,
  Server,
  Thermometer,
  Trash2,
  Wifi,
  WifiOff,
  AlertTriangle,
} from "lucide-react";
import { adminApi } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import type { GpuNode } from "@/types";
import { toast } from "sonner";

// ── helpers ───────────────────────────────────────────────────────────────────

function pct(h: Record<string, unknown>, k: string) {
  return typeof h[k] === "number" ? `${(h[k] as number).toFixed(0)}%` : "—";
}
function gb(h: Record<string, unknown>, k: string) {
  return typeof h[k] === "number" ? `${(h[k] as number).toFixed(1)} GB` : "—";
}
function mbToGb(h: Record<string, unknown>, k: string) {
  return typeof h[k] === "number" ? `${((h[k] as number) / 1024).toFixed(1)} GB` : "—";
}

function Metric({ icon: Icon, label, value, warn }: {
  icon: React.ElementType; label: string; value: string; warn?: boolean;
}) {
  return (
    <div className="flex items-center gap-1.5">
      <Icon className={cn("h-3 w-3 shrink-0", warn ? "text-amber-400" : "text-zinc-500")} />
      <span className={cn("text-xs tabular-nums", warn ? "text-amber-300" : "text-zinc-400")}>
        {label}: {value}
      </span>
    </div>
  );
}

// ── Node card ─────────────────────────────────────────────────────────────────

function NodeCard({ node, onDrain, onDelete }: {
  node: GpuNode;
  onDrain: () => void;
  onDelete: () => void;
}) {
  const h           = (node.health_json ?? {}) as Record<string, unknown>;
  const minutesAgo  = Math.round((Date.now() / 1000 - node.last_heartbeat) / 60);
  const isOnline    = node.status === "ONLINE";
  const isDraining  = node.status === "DRAINING";
  const temp        = typeof h["gpu0_temp_c"] === "number" ? `${(h["gpu0_temp_c"] as number).toFixed(0)}°C` : "—";
  const vram        = gb(h, "vram_used_gb") !== "—" ? gb(h, "vram_used_gb") : mbToGb(h, "gpu0_mem_used_mb");
  const connectivity = (h["connectivity"] as string) ?? node.connectivity ?? "—";
  const assignedClients = (h["assigned_clients"] as Array<{ name: string }> | undefined) ?? [];

  return (
    <Card className="relative overflow-hidden">
      <div className={cn(
        "absolute top-0 left-0 right-0 h-0.5",
        isOnline ? "bg-green-500" : isDraining ? "bg-amber-500" : "bg-zinc-700"
      )} />
      <CardHeader className="pb-2 pt-4">
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-center gap-2.5 min-w-0">
            <div className="h-8 w-8 rounded-lg bg-zinc-800 flex items-center justify-center shrink-0">
              <Server className="h-4 w-4 text-zinc-400" />
            </div>
            <div className="min-w-0">
              <CardTitle className="text-sm truncate">{node.node_name}</CardTitle>
              <p className="text-xs text-muted-foreground mt-0.5 truncate">{node.location ?? "Unknown location"}</p>
            </div>
          </div>
          <Badge
            variant={isOnline ? "success" : isDraining ? "warning" : "secondary"}
            className="text-[10px] shrink-0"
          >
            {node.status}
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="space-y-3">
        {/* Connectivity + GPU */}
        <div className="flex flex-wrap gap-1.5">
          <span className="rounded-full bg-zinc-800 px-2 py-0.5 text-[10px] text-zinc-400 font-mono">
            {connectivity}
          </span>
          {node.gpu_model && (
            <span className="rounded-full bg-zinc-800 px-2 py-0.5 text-[10px] text-zinc-400 flex items-center gap-1">
              <Cpu className="h-2.5 w-2.5" />{node.gpu_model}
            </span>
          )}
        </div>

        {/* Health metrics */}
        {isOnline || isDraining ? (
          <div className="grid grid-cols-2 gap-1">
            <Metric icon={Cpu}         label="CPU"  value={pct(h, "cpu_percent") !== "—" ? pct(h, "cpu_percent") : pct(h, "cpu_util")}  warn={(h["cpu_percent"] as number) > 90} />
            <Metric icon={MemoryStick} label="RAM"  value={pct(h, "ram_percent") !== "—" ? pct(h, "ram_percent") : pct(h, "ram_util")}  warn={(h["ram_percent"] as number) > 90} />
            <Metric icon={Cpu}         label="GPU"  value={pct(h, "gpu0_utilization") !== "—" ? pct(h, "gpu0_utilization") : pct(h, "gpu_util")} />
            <Metric icon={MemoryStick} label="VRAM" value={vram} />
            <Metric icon={HardDrive}   label="Disk" value={pct(h, "disk_percent")} warn={(h["disk_percent"] as number) > 85} />
            <Metric icon={Thermometer} label="Temp" value={temp}                  warn={(h["gpu0_temp_c"] as number) > 80} />
          </div>
        ) : (
          <div className="flex items-center gap-1.5 text-xs text-zinc-600 py-1">
            <WifiOff className="h-3 w-3" />
            Last seen {minutesAgo}m ago — heartbeat stopped
          </div>
        )}

        {/* Assigned clients */}
        {assignedClients.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {assignedClients.map((c) => (
              <span key={c.name} className="rounded-full bg-zinc-800/60 border border-zinc-700/50 px-2 py-0.5 text-[10px] text-zinc-400">
                {c.name}
              </span>
            ))}
          </div>
        )}

        {/* Footer */}
        <div className="flex items-center justify-between pt-1 border-t border-border/40">
          <div className="flex items-center gap-1.5 text-xs text-zinc-400">
            <Camera className="h-3 w-3" />
            {node.active_cameras} / {node.max_cameras} cameras
          </div>
          <div className="flex items-center gap-1">
            {isOnline && (
              <div className="flex items-center gap-1 text-[10px] text-zinc-600">
                <Wifi className="h-3 w-3 text-green-500" />{minutesAgo}m ago
              </div>
            )}
            {!node.last_heartbeat || node.status === "OFFLINE" ? null : (
              <Button size="icon" variant="ghost" className="h-6 w-6 text-zinc-600 hover:text-amber-400"
                title="Drain node" onClick={onDrain}>
                <AlertTriangle className="h-3 w-3" />
              </Button>
            )}
            <Button size="icon" variant="ghost" className="h-6 w-6 text-zinc-600 hover:text-red-400"
              title="Remove from registry" onClick={onDelete}>
              <Trash2 className="h-3 w-3" />
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// ── Register Node dialog ───────────────────────────────────────────────────────

interface RegisterForm {
  node_name: string;
  location: string;
  connectivity: string;
  api_endpoint: string;
  gpu_model: string;
  max_cameras: number;
}

function RegisterNodeDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const qc = useQueryClient();
  const [form, setForm] = useState<RegisterForm>({
    node_name:    "",
    location:     "",
    connectivity: "LOCAL",
    api_endpoint: "http://",
    gpu_model:    "",
    max_cameras:  10,
  });

  const set = (k: keyof RegisterForm, v: string | number) =>
    setForm((f) => ({ ...f, [k]: v }));

  const mutation = useMutation({
    mutationFn: () => adminApi.registerNode({
      node_name:    form.node_name,
      location:     form.location || null,
      connectivity: form.connectivity,
      api_endpoint: form.api_endpoint,
      gpu_model:    form.gpu_model || null,
      max_cameras:  form.max_cameras,
    }),
    onSuccess: () => {
      toast.success("Node registered. It will appear ONLINE once the backend is started.");
      qc.invalidateQueries({ queryKey: ["admin-nodes"] });
      onClose();
    },
    onError: (err: unknown) => {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      toast.error(detail ?? "Failed to register node");
    },
  });

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>Register GPU Node</DialogTitle>
        </DialogHeader>

        {/* How auto-registration works */}
        <div className="rounded-lg bg-zinc-900 border border-zinc-700/50 p-3 text-xs text-zinc-400 space-y-1">
          <p className="text-zinc-300 font-medium flex items-center gap-1.5">
            <Wifi className="h-3.5 w-3.5 text-green-400" />How nodes register automatically
          </p>
          <p>Any machine running the ACAS backend registers itself on startup. Set these variables in its <code className="bg-zinc-800 px-1 rounded">.env</code>:</p>
          <pre className="bg-zinc-800 rounded p-2 text-[11px] text-green-300 leading-relaxed overflow-x-auto">{`NODE_ID=<stable-uuid>          # keep fixed across restarts
NODE_NAME=my-gpu-node
NODE_LOCATION=Server Room A
NODE_API_ENDPOINT=http://<ip>:18000
                   # or https://xxx.trycloudflare.com`}</pre>
          <p>Then restart the backend — it will upsert into the DB and start sending heartbeats every 30s.</p>
        </div>

        <div className="space-y-3 pt-1">
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1.5">
              <Label>Node Name *</Label>
              <Input value={form.node_name} onChange={(e) => set("node_name", e.target.value)}
                placeholder="gpu-node-2" className="bg-zinc-800 border-zinc-700" />
            </div>
            <div className="space-y-1.5">
              <Label>Location</Label>
              <Input value={form.location} onChange={(e) => set("location", e.target.value)}
                placeholder="Server Room B" className="bg-zinc-800 border-zinc-700" />
            </div>
          </div>

          <div className="space-y-1.5">
            <Label>API Endpoint *</Label>
            <Input value={form.api_endpoint} onChange={(e) => set("api_endpoint", e.target.value)}
              placeholder="http://192.168.1.100:18000  or  https://xxx.trycloudflare.com"
              className="bg-zinc-800 border-zinc-700 font-mono text-xs" />
            <p className="text-[11px] text-zinc-500">
              LAN: <code>http://&lt;ip&gt;:18000</code> · Cloudflare Tunnel: <code>https://&lt;subdomain&gt;.trycloudflare.com</code> · Public IP: <code>https://&lt;domain&gt;</code>
            </p>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1.5">
              <Label>Connectivity</Label>
              <select
                value={form.connectivity}
                onChange={(e) => set("connectivity", e.target.value)}
                className="w-full rounded-md border border-zinc-700 bg-zinc-800 px-3 py-2 text-sm text-zinc-100 outline-none focus:ring-1 focus:ring-zinc-500"
              >
                <option value="LOCAL">Local / LAN</option>
                <option value="PUBLIC_IP">Public IP / Domain</option>
                <option value="CLOUDFLARE_TUNNEL">Cloudflare Tunnel</option>
              </select>
            </div>
            <div className="space-y-1.5">
              <Label>GPU Model</Label>
              <Input value={form.gpu_model} onChange={(e) => set("gpu_model", e.target.value)}
                placeholder="NVIDIA RTX 4090" className="bg-zinc-800 border-zinc-700" />
            </div>
          </div>

          <div className="space-y-1.5">
            <Label>Max Cameras</Label>
            <Input type="number" min={1} max={100} value={form.max_cameras}
              onChange={(e) => set("max_cameras", Number(e.target.value))}
              className="bg-zinc-800 border-zinc-700 w-28" />
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={onClose}>Cancel</Button>
          <Button
            disabled={!form.node_name || !form.api_endpoint || mutation.isPending}
            onClick={() => mutation.mutate()}
          >
            {mutation.isPending ? "Registering…" : "Register Node"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function AdminNodesPage() {
  const qc = useQueryClient();
  const [showRegister, setShowRegister] = useState(false);

  const { data, isLoading, isFetching } = useQuery<GpuNode[]>({
    queryKey:        ["admin-nodes"],
    queryFn:         () => adminApi.listNodes().then((r) => r.data as GpuNode[]),
    refetchInterval: 30_000,
    staleTime:       15_000,
  });

  const nodes: GpuNode[] = Array.isArray(data) ? data : [];

  const drainMutation = useMutation({
    mutationFn: (id: string) => adminApi.drainNode(id),
    onSuccess: () => { toast.success("Node set to DRAINING"); qc.invalidateQueries({ queryKey: ["admin-nodes"] }); },
    onError: () => toast.error("Failed to drain node"),
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) => adminApi.deleteNode(id),
    onSuccess: () => { toast.success("Node removed from registry"); qc.invalidateQueries({ queryKey: ["admin-nodes"] }); },
    onError: () => toast.error("Failed to remove node"),
  });

  const online   = nodes.filter((n) => n.status === "ONLINE").length;
  const draining = nodes.filter((n) => n.status === "DRAINING").length;
  const offline  = nodes.filter((n) => n.status === "OFFLINE").length;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold text-zinc-50">GPU Nodes</h1>
          <p className="text-sm text-muted-foreground mt-0.5">
            {online} online
            {draining > 0 ? ` · ${draining} draining` : ""}
            {offline > 0 ? ` · ${offline} offline` : ""}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button size="sm" variant="outline" onClick={() => qc.invalidateQueries({ queryKey: ["admin-nodes"] })}
            disabled={isFetching}>
            <RefreshCw className={cn("h-3.5 w-3.5", isFetching && "animate-spin")} />
          </Button>
          <Button size="sm" onClick={() => setShowRegister(true)}>
            <Plus className="h-4 w-4" />Register Node
          </Button>
        </div>
      </div>

      {isLoading ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {Array.from({ length: 3 }).map((_, i) => <Skeleton key={i} className="h-52 rounded-xl" />)}
        </div>
      ) : nodes.length === 0 ? (
        <div className="flex h-56 items-center justify-center rounded-xl border border-dashed border-border/50">
          <div className="text-center space-y-2">
            <Server className="mx-auto h-8 w-8 text-zinc-600" />
            <p className="text-sm text-muted-foreground">No GPU nodes registered yet</p>
            <p className="text-xs text-zinc-600 max-w-xs">
              Nodes register automatically when the backend starts. You can also add a remote node manually.
            </p>
            <Button size="sm" variant="outline" onClick={() => setShowRegister(true)} className="mt-2">
              <Plus className="h-3.5 w-3.5" />Register Node Manually
            </Button>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {nodes.map((n) => (
            <NodeCard
              key={n.node_id}
              node={n}
              onDrain={() => drainMutation.mutate(n.node_id)}
              onDelete={() => {
                if (confirm(`Remove "${n.node_name}" from registry?`)) {
                  deleteMutation.mutate(n.node_id);
                }
              }}
            />
          ))}
        </div>
      )}

      <RegisterNodeDialog open={showRegister} onClose={() => setShowRegister(false)} />
    </div>
  );
}
