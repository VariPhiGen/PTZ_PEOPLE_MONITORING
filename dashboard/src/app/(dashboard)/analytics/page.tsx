"use client";

import { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { subDays, format, parseISO } from "date-fns";
import {
  AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";
import {
  Users, TrendingUp, Camera, CheckCircle2, Clock,
  AlertTriangle, Activity, BarChart2, Cpu, Wifi, Shield, BookOpen,
} from "lucide-react";
import { analyticsApi, enrollmentApi } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Badge }    from "@/components/ui/badge";
import { Button }   from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "";

// ── Helpers ────────────────────────────────────────────────────────────────────

function fmtMs(ms: number) {
  return ms < 1000 ? `${ms.toFixed(0)} ms` : `${(ms / 1000).toFixed(1)} s`;
}

// ── Stat card ──────────────────────────────────────────────────────────────────

function StatCard({
  icon: Icon, label, value, sub, color = "blue",
}: {
  icon: React.ElementType;
  label: string;
  value: string | number;
  sub?: string;
  color?: "blue" | "green" | "amber" | "red" | "violet";
}) {
  const colors = {
    blue:   "bg-blue-500/10 text-blue-400",
    green:  "bg-green-500/10 text-green-400",
    amber:  "bg-amber-500/10 text-amber-400",
    red:    "bg-red-500/10 text-red-400",
    violet: "bg-violet-500/10 text-violet-400",
  };
  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-4 flex gap-4 items-start">
      <div className={cn("rounded-lg p-2.5 shrink-0", colors[color])}>
        <Icon className="h-5 w-5" />
      </div>
      <div className="min-w-0">
        <p className="text-xs text-zinc-500 font-medium">{label}</p>
        <p className="text-2xl font-semibold text-white mt-0.5">{value}</p>
        {sub && <p className="text-xs text-zinc-500 mt-0.5">{sub}</p>}
      </div>
    </div>
  );
}

function Section({ title, children, className }: { title: string; children: React.ReactNode; className?: string }) {
  return (
    <div className={cn("bg-zinc-900 border border-zinc-800 rounded-xl p-4 space-y-4", className)}>
      <h3 className="text-sm font-semibold text-zinc-300">{title}</h3>
      {children}
    </div>
  );
}

// ── Range selector ─────────────────────────────────────────────────────────────

const RANGES = [
  { label: "7 d",  days: 7  },
  { label: "14 d", days: 14 },
  { label: "30 d", days: 30 },
];

// ── Main page ──────────────────────────────────────────────────────────────────

export default function AnalyticsPage() {
  const [rangeDays, setRangeDays] = useState(14);
  const [courseFilter, setCourseFilter] = useState("");
  const [courseDate,   setCourseDate]   = useState("");   // YYYY-MM-DD

  const now      = Math.floor(Date.now() / 1000);
  const dateFrom = Math.floor(subDays(new Date(), rangeDays).getTime() / 1000);

  // ── Data fetches ─────────────────────────────────────────────────────────────

  const { data: trendsRaw, isLoading: loadTrends } = useQuery({
    queryKey: ["analytics-trends", rangeDays],
    queryFn:  () => analyticsApi.trends({ date_from: dateFrom, date_to: now, granularity: "day" })
                      .then(r => r.data as { data?: { bucket: string; P: number; L: number; EE: number; A: number; enrolled: number }[] }),
    staleTime: 60_000,
  });

  const { data: accuracyRaw, isLoading: loadAcc } = useQuery({
    queryKey: ["analytics-accuracy", rangeDays],
    queryFn:  () => analyticsApi.accuracy({ date_from: dateFrom, date_to: now })
                      .then(r => r.data as {
                        data: { date: string; accuracy_pct: number; session_count: number }[];
                        avg_rate: number;
                        session_count: number;
                      }),
    staleTime: 60_000,
  });

  const { data: uptimeRaw, isLoading: loadUptime } = useQuery({
    queryKey: ["analytics-uptime", rangeDays],
    queryFn:  () => analyticsApi.uptime({ date_from: dateFrom, date_to: now })
                      .then(r => r.data as {
                        items: { camera_id: string; name: string; status: string; session_count: number; uptime_pct: number }[];
                      }),
    staleTime: 60_000,
  });

  const { data: healthRaw, isLoading: loadHealth } = useQuery({
    queryKey: ["analytics-health"],
    queryFn:  () => analyticsApi.health()
                      .then(r => r.data as {
                        db_latency_ms: number;
                        redis_latency_ms: number;
                        kafka_ok: boolean;
                        active_sessions: number;
                        camera_status: Record<string, number>;
                      }),
    staleTime: 30_000,
    refetchInterval: 30_000,
  });

  const { data: dwellRaw, isLoading: loadDwell } = useQuery({
    queryKey: ["analytics-dwell", rangeDays],
    queryFn:  () => analyticsApi.dwellDistrib({ days: rangeDays })
                      .then(r => r.data as { items: { bucket: string; count: number }[] }),
    staleTime: 60_000,
  });

  const { data: visitorsRaw, isLoading: loadVisitors } = useQuery({
    queryKey: ["analytics-visitors", rangeDays],
    queryFn:  () => analyticsApi.frequentVisitors({ days: rangeDays, limit: 8 })
                      .then(r => r.data as { items: { name: string; external_id: string; sighting_count: number }[] }),
    staleTime: 60_000,
  });

  const { data: enrollRaw } = useQuery({
    queryKey: ["enroll-count"],
    queryFn:  () => enrollmentApi.list({ limit: 1, offset: 0 })
                      .then(r => r.data as { total?: number }),
    staleTime: 300_000,
  });

  const { data: ptzRaw } = useQuery({
    queryKey: ["analytics-ptz"],
    queryFn:  () => analyticsApi.ptzStats()
                      .then(r => r.data as {
                        cameras: { camera_name: string; avg_cycle_ms: number; recognition_rate: number; cycles_today: number }[];
                      }),
    staleTime: 30_000,
    refetchInterval: 30_000,
  });

  // Course-date filter params (computed only when user has set a date)
  const courseDateFrom = courseDate
    ? Math.floor(new Date(courseDate + "T00:00:00").getTime() / 1000)
    : dateFrom;
  const courseDateTo = courseDate
    ? Math.floor(new Date(courseDate + "T23:59:59").getTime() / 1000)
    : now;

  type CourseRow = {
    course_id: string | null; course_name: string;
    date: string;                // YYYY-MM-DD
    session_count: number;
    camera_name: string | null;
    enrolled: number; present: number; absent: number; rate: number;
  };
  const { data: courseRaw, isLoading: loadCourse } = useQuery({
    // Use courseDate string + rangeDays as key (stable), not epoch values that change every render
    queryKey: ["analytics-course", courseDate || `range-${rangeDays}`, courseFilter],
    queryFn:  () => analyticsApi.courseAttendance({
                      date_from: courseDateFrom,
                      date_to:   courseDateTo,
                      ...(courseFilter ? { course_name: courseFilter } : {}),
                    }).then(r => r.data as { items: CourseRow[]; courses: string[] }),
    staleTime: 60_000,
  });

  // ── Derived data ─────────────────────────────────────────────────────────────

  const trendData = useMemo(() => {
    const rows = trendsRaw?.data ?? [];
    return rows.map(b => ({
      date:     format(parseISO(b.bucket.slice(0, 10)), "MMM d"),
      present:  b.P        ?? 0,
      late:     b.L        ?? 0,
      absent:   b.A        ?? 0,
      enrolled: b.enrolled ?? 0,
    }));
  }, [trendsRaw]);

  const totals = useMemo(() => {
    const present  = trendData.reduce((s, d) => s + d.present,  0);
    const late     = trendData.reduce((s, d) => s + d.late,     0);
    const absent   = trendData.reduce((s, d) => s + d.absent,   0);
    const enrolled = trendData.reduce((s, d) => s + d.enrolled, 0);
    // Use enrolled as denominator; fall back to sum of records if enrolled not populated
    const total    = Math.max(enrolled, present + late + absent);
    const rate     = total > 0 ? Math.round(((present + late) / total) * 100) : 0;
    return { present, late, absent, enrolled, total, rate };
  }, [trendData]);

  const pieData = useMemo(() => [
    { name: "Present", value: totals.present, color: "#22c55e" },
    { name: "Late",    value: totals.late,    color: "#f59e0b" },
    { name: "Absent",  value: totals.absent,  color: "#ef4444" },
  ].filter(d => d.value > 0), [totals]);

  const accuracyData = useMemo(() =>
    (accuracyRaw?.data ?? []).map(d => ({
      date:     format(parseISO(d.date), "MMM d"),
      accuracy: d.accuracy_pct,
      sessions: d.session_count,
    })),
  [accuracyRaw]);

  const avgAccuracy = useMemo(() =>
    accuracyRaw?.avg_rate != null
      ? `${(accuracyRaw.avg_rate * 100).toFixed(1)}%`
      : "—",
  [accuracyRaw]);

  const cameraItems = uptimeRaw?.items ?? [];
  const avgUptime   = cameraItems.length
    ? (cameraItems.reduce((s, c) => s + c.uptime_pct, 0) / cameraItems.length).toFixed(1)
    : "—";

  const dwellItems  = dwellRaw?.items ?? [];
  const visitorItems = visitorsRaw?.items ?? [];
  const enrollCount = enrollRaw?.total ?? "—";

  const noData = !loadTrends && trendData.length === 0;

  return (
    <div className="space-y-6 p-1">

      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold text-white flex items-center gap-2">
            <BarChart2 className="h-5 w-5 text-blue-400" /> Analytics
          </h1>
          <p className="text-xs text-zinc-500 mt-0.5">Attendance, recognition and camera performance</p>
        </div>
        <div className="flex gap-1">
          {RANGES.map(r => (
            <Button
              key={r.days}
              size="sm"
              variant={rangeDays === r.days ? "default" : "outline"}
              className="text-xs h-7 px-3"
              onClick={() => setRangeDays(r.days)}
            >
              {r.label}
            </Button>
          ))}
        </div>
      </div>

      {noData && (
        <div className="flex items-center gap-2 text-xs text-amber-400 bg-amber-900/20 border border-amber-800/40 rounded-lg px-3 py-2">
          <AlertTriangle className="h-3.5 w-3.5 shrink-0" />
          No attendance records found for this period.
        </div>
      )}

      {/* Top stat cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <StatCard icon={Users}        label="Total Records"     value={totals.total.toLocaleString()} sub={`Last ${rangeDays} days`}       color="blue"   />
        <StatCard icon={CheckCircle2} label="Present"           value={totals.present.toLocaleString()} sub={`Late: ${totals.late}`}       color="green"  />
        <StatCard icon={TrendingUp}   label="Attendance Rate"   value={`${totals.rate}%`}             sub="Present + Late / Total"         color="violet" />
        <StatCard icon={Users}        label="Enrolled Persons"  value={enrollCount.toLocaleString()}  sub="With active face profile"       color="amber"  />
      </div>

      {/* Trend + Pie */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2">
          <Section title={`Daily Attendance Trend — last ${rangeDays} days`}>
            {loadTrends ? <Skeleton className="h-52 w-full" /> : trendData.length === 0 ? (
              <div className="h-52 flex items-center justify-center text-zinc-600 text-sm">No records in this period</div>
            ) : (
              <ResponsiveContainer width="100%" height={210}>
                <AreaChart data={trendData} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
                  <defs>
                    <linearGradient id="gPresent" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor="#22c55e" stopOpacity={0.35} />
                      <stop offset="95%" stopColor="#22c55e" stopOpacity={0}    />
                    </linearGradient>
                    <linearGradient id="gLate" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor="#f59e0b" stopOpacity={0.35} />
                      <stop offset="95%" stopColor="#f59e0b" stopOpacity={0}    />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                  <XAxis dataKey="date" tick={{ fill: "#71717a", fontSize: 11 }} tickLine={false} axisLine={false} />
                  <YAxis tick={{ fill: "#71717a", fontSize: 11 }} tickLine={false} axisLine={false} />
                  <Tooltip contentStyle={{ background: "#18181b", border: "1px solid #3f3f46", borderRadius: 8, fontSize: 12 }} labelStyle={{ color: "#a1a1aa" }} />
                  <Area type="monotone" dataKey="present" name="Present" stroke="#22c55e" fill="url(#gPresent)" strokeWidth={2} dot={false} />
                  <Area type="monotone" dataKey="late"    name="Late"    stroke="#f59e0b" fill="url(#gLate)"    strokeWidth={2} dot={false} />
                  <Area type="monotone" dataKey="absent"  name="Absent"  stroke="#ef4444" fill="none"           strokeWidth={1.5} dot={false} strokeDasharray="4 2" />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </Section>
        </div>

        <Section title="Status Distribution">
          {loadTrends ? <Skeleton className="h-52 w-full" /> : pieData.length === 0 ? (
            <div className="h-52 flex items-center justify-center text-zinc-600 text-sm">No data</div>
          ) : (
            <div className="flex flex-col items-center gap-3 pt-2">
              <ResponsiveContainer width="100%" height={160}>
                <PieChart>
                  <Pie data={pieData} cx="50%" cy="50%" innerRadius={48} outerRadius={72} paddingAngle={3} dataKey="value">
                    {pieData.map((e, i) => <Cell key={i} fill={e.color} />)}
                  </Pie>
                  <Tooltip contentStyle={{ background: "#18181b", border: "1px solid #3f3f46", borderRadius: 8, fontSize: 12 }} />
                </PieChart>
              </ResponsiveContainer>
              <div className="flex flex-col gap-1 w-full px-2">
                {pieData.map(e => (
                  <div key={e.name} className="flex items-center justify-between text-xs">
                    <div className="flex items-center gap-2 text-zinc-400">
                      <span className="h-2 w-2 rounded-full shrink-0" style={{ background: e.color }} />
                      {e.name}
                    </div>
                    <span className="text-zinc-300 font-medium tabular-nums">{e.value.toLocaleString()}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </Section>
      </div>

      {/* Recognition accuracy + Dwell distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Section title="Recognition Accuracy by Day">
          {loadAcc ? <Skeleton className="h-52 w-full" /> : accuracyData.length === 0 ? (
            <div className="h-52 flex items-center justify-center text-zinc-600 text-sm">No session data</div>
          ) : (
            <>
              <div className="flex items-center gap-4 text-xs text-zinc-500 -mt-2">
                <span>Overall avg: <span className="text-blue-400 font-semibold">{avgAccuracy}</span></span>
                <span>Sessions: <span className="text-zinc-300">{accuracyRaw?.session_count ?? 0}</span></span>
              </div>
              <ResponsiveContainer width="100%" height={190}>
                <AreaChart data={accuracyData} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
                  <defs>
                    <linearGradient id="gAcc" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor="#3b82f6" stopOpacity={0.4} />
                      <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}   />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                  <XAxis dataKey="date" tick={{ fill: "#71717a", fontSize: 11 }} tickLine={false} axisLine={false} />
                  <YAxis domain={[0, 100]} tick={{ fill: "#71717a", fontSize: 11 }} tickLine={false} axisLine={false} unit="%" />
                  <Tooltip
                    contentStyle={{ background: "#18181b", border: "1px solid #3f3f46", borderRadius: 8, fontSize: 12 }}
                    formatter={(v: number, _: string, p: { payload?: { sessions: number } }) => [
                      [`${v.toFixed(1)}%`, "Accuracy"],
                      [`${p.payload?.sessions ?? 0}`, "Sessions"],
                    ].flat()}
                  />
                  <Area type="monotone" dataKey="accuracy" name="Accuracy" stroke="#3b82f6" fill="url(#gAcc)" strokeWidth={2} dot={false} />
                </AreaChart>
              </ResponsiveContainer>
            </>
          )}
        </Section>

        <Section title="Sighting Dwell-Time Distribution">
          {loadDwell ? <Skeleton className="h-52 w-full" /> : dwellItems.length === 0 ? (
            <div className="h-52 flex items-center justify-center text-zinc-600 text-sm">No sighting data</div>
          ) : (
            <ResponsiveContainer width="100%" height={210}>
              <BarChart data={dwellItems} margin={{ top: 4, right: 4, bottom: 0, left: -20 }} barSize={28}>
                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                <XAxis dataKey="bucket" tick={{ fill: "#71717a", fontSize: 11 }} tickLine={false} axisLine={false} />
                <YAxis tick={{ fill: "#71717a", fontSize: 11 }} tickLine={false} axisLine={false} />
                <Tooltip contentStyle={{ background: "#18181b", border: "1px solid #3f3f46", borderRadius: 8, fontSize: 12 }} />
                <Bar dataKey="count" name="Sightings" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </Section>
      </div>

      {/* Course / Timetable Attendance */}
      <Section title="Course Attendance — Timetable View" className="">
        {/* Filters */}
        <div className="flex flex-wrap gap-3 items-end -mt-1">
          <div className="flex flex-col gap-1">
            <label className="text-[10px] text-zinc-500 font-medium uppercase tracking-wide">Date</label>
            <input
              type="date"
              value={courseDate}
              onChange={e => setCourseDate(e.target.value)}
              className="h-7 text-xs bg-zinc-800 border border-zinc-700 rounded px-2 text-zinc-200 focus:outline-none focus:border-blue-500"
            />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-[10px] text-zinc-500 font-medium uppercase tracking-wide">Course</label>
            <select
              value={courseFilter}
              onChange={e => setCourseFilter(e.target.value)}
              className="h-7 text-xs bg-zinc-800 border border-zinc-700 rounded px-2 text-zinc-200 focus:outline-none focus:border-blue-500"
            >
              <option value="">All courses</option>
              {(courseRaw?.courses ?? []).map(c => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </div>
          {(courseDate || courseFilter) && (
            <Button
              size="sm" variant="ghost"
              className="h-7 text-xs text-zinc-500 hover:text-zinc-200 px-2 self-end"
              onClick={() => { setCourseDate(""); setCourseFilter(""); }}
            >
              Clear
            </Button>
          )}
        </div>

        {loadCourse ? (
          <Skeleton className="h-40 w-full" />
        ) : !courseRaw?.items.length ? (
          <div className="h-24 flex items-center justify-center text-zinc-600 text-sm">
            No session data for this filter
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm min-w-[640px]">
              <thead>
                <tr className="text-left text-[10px] text-zinc-500 border-b border-zinc-800 uppercase tracking-wide">
                  <th className="pb-2 pr-3 font-medium">Course</th>
                  <th className="pb-2 pr-3 font-medium">Date</th>
                  <th className="pb-2 pr-3 font-medium text-right">Sessions</th>
                  <th className="pb-2 pr-3 font-medium text-right">Enrolled</th>
                  <th className="pb-2 pr-3 font-medium text-right">Present</th>
                  <th className="pb-2 pr-3 font-medium text-right">Absent</th>
                  <th className="pb-2 font-medium text-right">Rate</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-800/60">
                {courseRaw.items.map((row, i) => {
                  const dateStr  = row.date
                    ? format(parseISO(row.date), "MMM d, yyyy")
                    : "—";
                  const ratePct  = Math.round(row.rate * 100);
                  return (
                    <tr key={`${row.course_id}-${row.date}-${i}`} className="text-zinc-300 hover:bg-zinc-800/30 transition-colors">
                      <td className="py-2.5 pr-3">
                        <div className="flex items-center gap-1.5">
                          <BookOpen className="h-3.5 w-3.5 text-zinc-500 shrink-0" />
                          <span className="font-medium truncate max-w-[160px]">{row.course_name}</span>
                        </div>
                      </td>
                      <td className="py-2.5 pr-3 text-zinc-400 text-xs">{dateStr}</td>
                      <td className="py-2.5 pr-3 text-right tabular-nums text-zinc-500 text-xs">{row.session_count}</td>
                      <td className="py-2.5 pr-3 text-right tabular-nums">{row.enrolled}</td>
                      <td className="py-2.5 pr-3 text-right tabular-nums text-green-400">{row.present}</td>
                      <td className="py-2.5 pr-3 text-right tabular-nums text-red-400">{row.absent}</td>
                      <td className="py-2.5 text-right">
                        <span className={cn(
                          "text-xs font-semibold tabular-nums",
                          ratePct >= 75 ? "text-green-400" : ratePct >= 50 ? "text-amber-400" : "text-red-400",
                        )}>
                          {ratePct}%
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </Section>

      {/* Camera uptime + Frequent visitors */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Section title="Camera Performance">
          {loadUptime ? <Skeleton className="h-40 w-full" /> : cameraItems.length === 0 ? (
            <div className="h-24 flex items-center justify-center text-zinc-600 text-sm">No camera data</div>
          ) : (
            <div className="overflow-x-auto">
              <div className="text-xs text-zinc-500 mb-2">Avg uptime: <span className="text-green-400 font-semibold">{avgUptime}%</span></div>
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-xs text-zinc-500 border-b border-zinc-800">
                    <th className="pb-2 pr-4 font-medium">Camera</th>
                    <th className="pb-2 pr-4 font-medium">Uptime</th>
                    <th className="pb-2 pr-4 font-medium">Sessions</th>
                    <th className="pb-2 font-medium">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-800/60">
                  {cameraItems.map((cam) => (
                    <tr key={cam.camera_id} className="text-zinc-300 hover:bg-zinc-800/30 transition-colors">
                      <td className="py-2.5 pr-4 font-medium">
                        <div className="flex items-center gap-2">
                          <Camera className="h-3.5 w-3.5 text-zinc-500 shrink-0" />
                          <span className="truncate max-w-[140px]">{cam.name}</span>
                        </div>
                      </td>
                      <td className="py-2.5 pr-4">
                        <div className="flex items-center gap-2">
                          <div className="h-1.5 w-16 bg-zinc-800 rounded-full overflow-hidden">
                            <div
                              className={cn("h-full rounded-full",
                                cam.uptime_pct >= 80 ? "bg-green-500" :
                                cam.uptime_pct >= 50 ? "bg-amber-500" : "bg-red-500"
                              )}
                              style={{ width: `${cam.uptime_pct}%` }}
                            />
                          </div>
                          <span className="text-xs tabular-nums">{cam.uptime_pct.toFixed(1)}%</span>
                        </div>
                      </td>
                      <td className="py-2.5 pr-4 text-zinc-400 tabular-nums">{cam.session_count}</td>
                      <td className="py-2.5">
                        <Badge className={cn("text-[10px] h-5",
                          cam.status === "ACTIVE" ? "bg-green-900/40 text-green-400 border-green-800/40"
                          : cam.status === "OFFLINE" ? "bg-red-900/40 text-red-400 border-red-800/40"
                          : "bg-zinc-800 text-zinc-400 border-zinc-700"
                        )}>
                          {cam.status}
                        </Badge>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Section>

        <Section title={`Frequent Visitors — last ${rangeDays} days`}>
          {loadVisitors ? <Skeleton className="h-40 w-full" /> : visitorItems.length === 0 ? (
            <div className="h-24 flex items-center justify-center text-zinc-600 text-sm">No sighting data</div>
          ) : (
            <div className="space-y-2">
              {visitorItems.map((v, i) => (
                <div key={i} className="flex items-center gap-3">
                  <img
                    src={`${API_BASE}/api/enrollment/thumbnail/${v.external_id}`}
                    alt={v.name}
                    className="h-8 w-8 rounded-full object-cover bg-zinc-800 border border-zinc-700 shrink-0"
                    onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
                  />
                  <div className="flex-1 min-w-0">
                    <p className="text-xs font-medium text-zinc-200 truncate">{v.name}</p>
                    <p className="text-[10px] text-zinc-500 font-mono">{v.external_id}</p>
                  </div>
                  <div className="text-right shrink-0">
                    <p className="text-xs font-semibold text-zinc-200 tabular-nums">{v.sighting_count}</p>
                    <p className="text-[10px] text-zinc-500">sightings</p>
                  </div>
                  <div className="w-20">
                    <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-violet-500 rounded-full"
                        style={{ width: `${Math.min((v.sighting_count / (visitorItems[0]?.sighting_count || 1)) * 100, 100)}%` }}
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </Section>
      </div>

      {/* System health + PTZ stats */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Section title="System Health">
          {loadHealth ? <Skeleton className="h-32 w-full" /> : !healthRaw ? (
            <div className="h-24 flex items-center justify-center text-zinc-600 text-sm">Health data unavailable</div>
          ) : (
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-zinc-800/50 rounded-lg p-3 flex items-center gap-3">
                <Cpu className="h-4 w-4 text-blue-400 shrink-0" />
                <div>
                  <p className="text-[10px] text-zinc-500">DB Latency</p>
                  <p className="text-sm font-semibold text-white">{fmtMs(healthRaw.db_latency_ms)}</p>
                </div>
              </div>
              <div className="bg-zinc-800/50 rounded-lg p-3 flex items-center gap-3">
                <Wifi className="h-4 w-4 text-green-400 shrink-0" />
                <div>
                  <p className="text-[10px] text-zinc-500">Redis Latency</p>
                  <p className="text-sm font-semibold text-white">{fmtMs(healthRaw.redis_latency_ms)}</p>
                </div>
              </div>
              <div className="bg-zinc-800/50 rounded-lg p-3 flex items-center gap-3">
                <Activity className="h-4 w-4 text-violet-400 shrink-0" />
                <div>
                  <p className="text-[10px] text-zinc-500">Active Sessions</p>
                  <p className="text-sm font-semibold text-white">{healthRaw.active_sessions}</p>
                </div>
              </div>
              <div className="bg-zinc-800/50 rounded-lg p-3 flex items-center gap-3">
                <Shield className={cn("h-4 w-4 shrink-0", healthRaw.kafka_ok ? "text-green-400" : "text-red-400")} />
                <div>
                  <p className="text-[10px] text-zinc-500">Kafka</p>
                  <p className={cn("text-sm font-semibold", healthRaw.kafka_ok ? "text-green-400" : "text-red-400")}>
                    {healthRaw.kafka_ok ? "Online" : "Offline"}
                  </p>
                </div>
              </div>
              {Object.entries(healthRaw.camera_status ?? {}).length > 0 && (
                <div className="col-span-2 bg-zinc-800/50 rounded-lg p-3">
                  <p className="text-[10px] text-zinc-500 mb-2">Camera Status</p>
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(healthRaw.camera_status).map(([status, count]) => (
                      <Badge key={status} className={cn("text-[10px]",
                        status === "ACTIVE"  ? "bg-green-900/40 text-green-400 border-green-800/40"
                        : status === "OFFLINE" ? "bg-red-900/40 text-red-400 border-red-800/40"
                        : "bg-zinc-800 text-zinc-400"
                      )}>
                        {status}: {count}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </Section>

        <Section title="PTZ Scan Performance (Live)">
          {!ptzRaw ? (
            <div className="h-24 flex items-center justify-center text-zinc-600 text-sm">No active PTZ sessions</div>
          ) : ptzRaw.cameras.length === 0 ? (
            <div className="h-24 flex items-center justify-center text-zinc-600 text-sm">No active PTZ sessions</div>
          ) : (
            <div className="space-y-3">
              {ptzRaw.cameras.map((cam, i) => (
                <div key={i} className="bg-zinc-800/50 rounded-lg p-3">
                  <div className="flex items-center justify-between mb-2">
                    <p className="text-xs font-medium text-zinc-200 flex items-center gap-1.5">
                      <Camera className="h-3.5 w-3.5 text-zinc-500" />
                      {cam.camera_name}
                    </p>
                    <Badge className="text-[10px] bg-green-900/30 text-green-400 border-green-800/30">Live</Badge>
                  </div>
                  <div className="grid grid-cols-3 gap-2 text-center">
                    <div>
                      <p className="text-xs font-semibold text-white tabular-nums">{cam.cycles_today}</p>
                      <p className="text-[10px] text-zinc-500">Cycles</p>
                    </div>
                    <div>
                      <p className="text-xs font-semibold text-white tabular-nums">{(cam.recognition_rate * 100).toFixed(0)}%</p>
                      <p className="text-[10px] text-zinc-500">Recog. Rate</p>
                    </div>
                    <div>
                      <p className="text-xs font-semibold text-white tabular-nums">{cam.avg_cycle_ms.toLocaleString()} ms</p>
                      <p className="text-[10px] text-zinc-500">Cycle Time</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </Section>
      </div>

    </div>
  );
}
