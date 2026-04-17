"use client";

import { useQuery } from "@tanstack/react-query";
import {
  BarChart, Bar, AreaChart, Area, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from "recharts";
import {
  Camera, Cpu, MemoryStick, Server, TrendingUp, Users, Zap,
  CheckCircle2, Clock,
} from "lucide-react";
import { adminApi, analyticsApi } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import type { GpuNode } from "@/types";

// ─────────────────────────────────────────────────────────────────────────────
// Chart style
// ─────────────────────────────────────────────────────────────────────────────

const CS = {
  tooltip: {
    contentStyle: { backgroundColor: "#18181b", border: "1px solid #3f3f46", borderRadius: 8, fontSize: 11 },
    labelStyle:   { color: "#a1a1aa" },
    itemStyle:    { fontSize: 11 },
  },
  tick:    { fontSize: 10, fill: "#52525b" },
  grid:    "#27272a",
};

const PIE_COLORS = ["#6366f1", "#22c55e", "#f59e0b", "#ef4444", "#06b6d4", "#a855f7", "#fb923c", "#34d399"];

// ─────────────────────────────────────────────────────────────────────────────
// Summary stat card
// ─────────────────────────────────────────────────────────────────────────────

function StatCard({
  icon: Icon, label, value, sub, trend, color = "text-zinc-100",
}: {
  icon:   React.ElementType;
  label:  string;
  value:  string | number;
  sub?:   string;
  trend?: number;    // positive = up, negative = down
  color?: string;
}) {
  return (
    <div className="card-stat flex flex-col gap-1">
      <div className="flex items-center justify-between">
        <p className="text-xs text-muted-foreground">{label}</p>
        <Icon className="h-4 w-4 text-zinc-600" />
      </div>
      <p className={cn("text-2xl font-bold tabular-nums mt-0.5", color)}>
        {typeof value === "number" ? value.toLocaleString() : value}
      </p>
      {(sub || trend != null) && (
        <div className="flex items-center gap-2 mt-auto">
          {sub && <p className="text-[10px] text-muted-foreground">{sub}</p>}
          {trend != null && (
            <span className={cn(
              "text-[10px] font-medium ml-auto",
              trend > 0 ? "text-green-400" : trend < 0 ? "text-red-400" : "text-zinc-500",
            )}>
              {trend > 0 ? "↑" : trend < 0 ? "↓" : "—"} {Math.abs(trend)}%
            </span>
          )}
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// GPU node utilisation card
// ─────────────────────────────────────────────────────────────────────────────

function GpuNodeCard({ node }: { node: GpuNode }) {
  const h    = node.health_json ?? {};
  const gpu  = typeof h["gpu0_utilization"] === "number" ? h["gpu0_utilization"] as number : null;
  const mem  = typeof h["gpu0_mem_used_mb"] === "number" && typeof h["gpu0_mem_total_mb"] === "number"
    ? ((h["gpu0_mem_used_mb"] as number) / (h["gpu0_mem_total_mb"] as number)) * 100
    : null;
  const cpu  = typeof h["cpu_percent"] === "number" ? h["cpu_percent"] as number : null;
  const ram  = typeof h["ram_percent"] === "number" ? h["ram_percent"] as number : null;
  const temp = typeof h["gpu0_temp_c"] === "number" ? `${(h["gpu0_temp_c"] as number).toFixed(0)}°C` : "—";

  const camPct = node.max_cameras > 0 ? (node.active_cameras / node.max_cameras) * 100 : 0;
  const minutesAgo = Math.round((Date.now() / 1000 - node.last_heartbeat) / 60);

  function MetricRow({ label, pct, value, warn }: { label: string; pct: number | null; value: string; warn?: boolean }) {
    return (
      <div className="space-y-0.5">
        <div className="flex justify-between text-[10px]">
          <span className="text-zinc-500">{label}</span>
          <span className={cn("font-mono", warn ? "text-amber-400" : "text-zinc-400")}>{value}</span>
        </div>
        {pct != null && (
          <Progress
            value={pct}
            className={cn("h-1", pct > 85 ? "[&>div]:bg-amber-400" : "[&>div]:bg-zinc-500")}
          />
        )}
      </div>
    );
  }

  return (
    <div className={cn(
      "rounded-xl border p-4 space-y-3",
      node.status === "ONLINE"   ? "border-green-500/20 bg-green-500/5"
      : node.status === "DRAINING" ? "border-amber-500/20 bg-amber-500/5"
      : "border-zinc-800 bg-card",
    )}>
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm font-semibold text-zinc-100">{node.node_name}</p>
          <p className="text-xs text-muted-foreground">{node.location ?? "—"}</p>
          {node.gpu_model && (
            <p className="text-[10px] text-zinc-500 mt-0.5 flex items-center gap-1">
              <Cpu className="h-2.5 w-2.5" />{node.gpu_model}
            </p>
          )}
        </div>
        <div className="flex flex-col items-end gap-1">
          <Badge
            variant={node.status === "ONLINE" ? "success" : node.status === "DRAINING" ? "warning" : "secondary"}
            className="text-[10px]"
          >
            {node.status}
          </Badge>
          <span className="text-[10px] text-zinc-600">{minutesAgo}m ago</span>
        </div>
      </div>

      <div className="space-y-2">
        {gpu   != null && <MetricRow label="GPU"   pct={gpu}   value={`${gpu.toFixed(0)}%`}  warn={gpu > 85} />}
        {mem   != null && <MetricRow label="VRAM"  pct={mem}   value={`${mem.toFixed(0)}%`}  warn={mem > 85} />}
        {cpu   != null && <MetricRow label="CPU"   pct={cpu}   value={`${cpu.toFixed(0)}%`}  warn={cpu > 90} />}
        {ram   != null && <MetricRow label="RAM"   pct={ram}   value={`${ram.toFixed(0)}%`}  warn={ram > 90} />}
        <MetricRow label="Cameras" pct={camPct}
          value={`${node.active_cameras}/${node.max_cameras}`}
          warn={camPct > 85}
        />
      </div>

      <div className="flex items-center justify-between text-[10px] border-t border-zinc-800 pt-2">
        <span className="text-zinc-500">GPU temp</span>
        <span className={cn("font-mono", parseFloat(temp) > 80 ? "text-amber-400" : "text-zinc-400")}>{temp}</span>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-client usage table
// ─────────────────────────────────────────────────────────────────────────────

interface ClientUsage {
  client_name:      string;
  sessions:         number;
  cameras:          number;
  persons:          number;
  recognition_rate: number;
  gpu_hours:        number;
}

function ClientUsageTable({ data }: { data: ClientUsage[] }) {
  if (data.length === 0) {
    return <p className="text-sm text-muted-foreground text-center py-6">No usage data</p>;
  }
  const maxSessions = Math.max(...data.map((d) => d.sessions), 1);

  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="border-b border-border/50">
          <th className="py-2 px-3 text-left text-xs font-medium text-muted-foreground">Client</th>
          <th className="py-2 px-3 text-right text-xs font-medium text-muted-foreground">Sessions</th>
          <th className="py-2 px-3 text-left text-xs font-medium text-muted-foreground w-32">Volume</th>
          <th className="py-2 px-3 text-right text-xs font-medium text-muted-foreground">Cameras</th>
          <th className="py-2 px-3 text-right text-xs font-medium text-muted-foreground">Persons</th>
          <th className="py-2 px-3 text-right text-xs font-medium text-muted-foreground">Recog Rate</th>
          <th className="py-2 px-3 text-right text-xs font-medium text-muted-foreground">GPU Hours</th>
        </tr>
      </thead>
      <tbody className="divide-y divide-border/40">
        {data.map((row, i) => (
          <tr key={i} className="hover:bg-zinc-800/20 transition-colors">
            <td className="py-2.5 px-3 font-medium text-zinc-200">{row.client_name}</td>
            <td className="py-2.5 px-3 text-right tabular-nums text-zinc-300">{row.sessions}</td>
            <td className="py-2.5 px-3">
              <div className="h-1.5 rounded-full bg-zinc-800 overflow-hidden w-full">
                <div
                  className="h-full rounded-full bg-indigo-500 transition-all"
                  style={{ width: `${(row.sessions / maxSessions) * 100}%` }}
                />
              </div>
            </td>
            <td className="py-2.5 px-3 text-right text-zinc-400 text-xs">{row.cameras}</td>
            <td className="py-2.5 px-3 text-right text-zinc-400 text-xs">{row.persons.toLocaleString()}</td>
            <td className="py-2.5 px-3 text-right">
              <span className={cn("text-xs font-medium tabular-nums",
                row.recognition_rate > 0.9 ? "text-green-400"
                : row.recognition_rate > 0.75 ? "text-amber-400"
                : "text-red-400"
              )}>
                {(row.recognition_rate * 100).toFixed(1)}%
              </span>
            </td>
            <td className="py-2.5 px-3 text-right text-zinc-400 text-xs tabular-nums">
              {row.gpu_hours?.toFixed(1) ?? "—"}h
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Page
// ─────────────────────────────────────────────────────────────────────────────

export default function PlatformAnalyticsPage() {
  const { data: platformRaw, isLoading: platformLoading } = useQuery({
    queryKey: ["admin-platform"],
    queryFn:  () => adminApi.platform().then((r) => r.data),
    refetchInterval: 60_000,
  });

  const { data: usageRaw, isLoading: usageLoading } = useQuery({
    queryKey: ["admin-usage"],
    queryFn:  () => adminApi.usage().then((r) => r.data),
  });

  const { data: trendsRaw, isLoading: trendsLoading } = useQuery({
    queryKey: ["analytics-trends-30"],
    queryFn:  () => analyticsApi.trends({ days: 30 }).then((r) => r.data),
  });

  const { data: accuracyRaw } = useQuery({
    queryKey: ["recognition-accuracy-30"],
    queryFn:  () => analyticsApi.accuracy({ days: 14 }).then((r) => r.data),
  });

  const platform = platformRaw as Record<string, number | GpuNode[]> | undefined;
  const nodes: GpuNode[] = Array.isArray(platform?.nodes) ? platform!.nodes as GpuNode[] : [];
  const clientUsage: ClientUsage[] = (usageRaw as { clients?: ClientUsage[] })?.clients ?? [];
  const trends   = (trendsRaw   as { data?: unknown[] })?.data   ?? [];
  const accuracy = (accuracyRaw as { data?: unknown[] })?.data   ?? [];

  const onlineNodes   = nodes.filter((n) => n.status === "ONLINE").length;
  const totalCameras  = (platform?.total_cameras  as number | undefined)  ?? 0;
  const totalPersons  = (platform?.total_persons  as number | undefined)  ?? 0;
  const sessionsToday = (platform?.sessions_today as number | undefined)  ?? 0;
  const gpuHoursToday = (platform?.gpu_hours_today as number | undefined) ?? 0;
  const avgAccuracy   = (platform?.avg_recognition_rate as number | undefined) ?? 0;
  const activeSessions = (platform?.active_sessions as number | undefined) ?? 0;

  // Aggregate recognition rate across clients for pie chart
  const recognitionPie = clientUsage.map((c, i) => ({
    name:  c.client_name,
    value: Math.round(c.recognition_rate * 100),
    fill:  PIE_COLORS[i % PIE_COLORS.length],
  }));

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-xl font-semibold text-zinc-50">Platform Analytics</h1>
        <p className="text-sm text-muted-foreground mt-0.5">
          Global metrics across all clients, nodes, and sessions
        </p>
      </div>

      {/* ── Summary stat cards ──────────────────────────────────────── */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        {platformLoading
          ? Array.from({ length: 6 }).map((_, i) => <Skeleton key={i} className="h-24 rounded-xl" />)
          : <>
              <StatCard icon={Server}     label="Nodes Online"    value={`${onlineNodes}/${nodes.length}`}
                color={onlineNodes === nodes.length ? "text-green-400" : "text-amber-400"} />
              <StatCard icon={Camera}     label="Total Cameras"   value={totalCameras} />
              <StatCard icon={Users}      label="Enrolled Persons" value={totalPersons} />
              <StatCard icon={CheckCircle2} label="Sessions Today" value={sessionsToday}
                color={activeSessions > 0 ? "text-green-400" : "text-zinc-100"}
                sub={activeSessions > 0 ? `${activeSessions} active` : undefined} />
              <StatCard icon={TrendingUp} label="Avg Recognition" value={`${(avgAccuracy * 100).toFixed(1)}%`}
                color={avgAccuracy > 0.9 ? "text-green-400" : avgAccuracy > 0.75 ? "text-amber-400" : "text-red-400"} />
              <StatCard icon={Zap}        label="GPU Hours Today" value={gpuHoursToday.toFixed(1) + "h"} />
            </>
        }
      </div>

      {/* ── Trend charts ─────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 gap-6">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-2">
              <Users className="h-4 w-4 text-zinc-400" />
              30-Day Attendance Trend
            </CardTitle>
          </CardHeader>
          <CardContent>
            {trendsLoading ? <Skeleton className="h-44 w-full" /> : (
              <ResponsiveContainer width="100%" height={176}>
                <AreaChart data={trends} margin={{ top: 0, right: 0, left: -22, bottom: 0 }}>
                  <defs>
                    <linearGradient id="gP" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#22c55e" stopOpacity={0.15} />
                      <stop offset="95%" stopColor="#22c55e" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke={CS.grid} />
                  <XAxis dataKey="date" tick={CS.tick} tickLine={false} />
                  <YAxis tick={CS.tick} tickLine={false} />
                  <Tooltip {...CS.tooltip} />
                  <Legend wrapperStyle={{ fontSize: 10 }} />
                  <Area type="monotone" dataKey="present" stroke="#22c55e" fill="url(#gP)" strokeWidth={1.5} name="Present" />
                  <Area type="monotone" dataKey="late"    stroke="#eab308" fill="none"       strokeWidth={1.5} name="Late"    strokeDasharray="4 2" />
                  <Area type="monotone" dataKey="absent"  stroke="#ef4444" fill="none"       strokeWidth={1.5} name="Absent"  strokeDasharray="4 2" />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-2">
              <TrendingUp className="h-4 w-4 text-zinc-400" />
              Recognition Accuracy (14 days)
            </CardTitle>
          </CardHeader>
          <CardContent>
            {!accuracy.length ? <Skeleton className="h-44 w-full" /> : (
              <ResponsiveContainer width="100%" height={176}>
                <LineChart data={accuracy} margin={{ top: 0, right: 0, left: -22, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={CS.grid} />
                  <XAxis dataKey="date" tick={CS.tick} tickLine={false} />
                  <YAxis domain={[75, 100]} tick={CS.tick} tickLine={false} unit="%" />
                  <Tooltip {...CS.tooltip} formatter={(v: number) => [`${v.toFixed(1)}%`, "Accuracy"]} />
                  <Line type="monotone" dataKey="accuracy_pct" stroke="#60a5fa" strokeWidth={2} dot={false} name="Accuracy" />
                </LineChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>
      </div>

      {/* ── Per-client usage ─────────────────────────────────────────── */}
      <div className="grid grid-cols-3 gap-6">
        {/* Bar chart */}
        <Card className="col-span-2">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Sessions by Client (30 days)</CardTitle>
          </CardHeader>
          <CardContent>
            {usageLoading ? <Skeleton className="h-44 w-full" /> : (
              <ResponsiveContainer width="100%" height={176}>
                <BarChart data={clientUsage} margin={{ top: 0, right: 0, left: -22, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={CS.grid} />
                  <XAxis dataKey="client_name" tick={CS.tick} tickLine={false} />
                  <YAxis tick={CS.tick} tickLine={false} />
                  <Tooltip {...CS.tooltip} />
                  <Bar dataKey="sessions"  name="Sessions"   fill="#6366f1" radius={[3,3,0,0]} />
                  <Bar dataKey="gpu_hours" name="GPU Hours"  fill="#22c55e" radius={[3,3,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>

        {/* Recognition pie */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Avg Recognition Rate</CardTitle>
          </CardHeader>
          <CardContent>
            {usageLoading ? <Skeleton className="h-44 w-full" /> : recognitionPie.length > 0 ? (
              <ResponsiveContainer width="100%" height={176}>
                <PieChart>
                  <Pie data={recognitionPie} dataKey="value" cx="50%" cy="50%"
                    innerRadius={40} outerRadius={65}
                    label={({ name, value }: { name: string; value: number }) => `${value}%`}
                    labelLine={false}
                  >
                    {recognitionPie.map((entry, i) => (
                      <Cell key={i} fill={entry.fill} />
                    ))}
                  </Pie>
                  <Tooltip {...CS.tooltip} formatter={(v: number) => [`${v}%`, "Recognition"]} />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <p className="text-sm text-muted-foreground text-center pt-6">No data</p>
            )}
          </CardContent>
        </Card>
      </div>

      {/* ── Per-client usage table ──────────────────────────────────── */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Per-Client Usage</CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          {usageLoading
            ? <div className="p-4 space-y-2">{Array.from({ length: 5 }).map((_, i) => <Skeleton key={i} className="h-8 rounded" />)}</div>
            : <ClientUsageTable data={clientUsage} />
          }
        </CardContent>
      </Card>

      {/* ── GPU node cards ───────────────────────────────────────────── */}
      {nodes.length > 0 && (
        <div>
          <h2 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider mb-3">
            GPU Nodes — Live Utilisation
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {nodes.map((n) => <GpuNodeCard key={n.node_id} node={n} />)}
          </div>
        </div>
      )}
    </div>
  );
}
