"use client";

import { useQuery } from "@tanstack/react-query";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, PieChart, Pie, Cell, Legend,
} from "recharts";
import { adminApi } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

const COLORS = ["#6366f1", "#22c55e", "#f59e0b", "#ef4444", "#06b6d4", "#a855f7"];

const CHART_STYLE = {
  contentStyle: { backgroundColor: "#18181b", border: "1px solid #3f3f46", borderRadius: 8, fontSize: 11 },
  tickProps:    { fontSize: 10, fill: "#52525b" },
  gridStroke:   "#27272a",
};

export default function AdminAnalyticsPage() {
  const { data: usageRaw, isLoading: usageLoading } = useQuery({
    queryKey: ["admin-usage"],
    queryFn:  () => adminApi.usage().then((r) => r.data),
  });

  const { data: platformRaw, isLoading: platformLoading } = useQuery({
    queryKey: ["admin-platform"],
    queryFn:  () => adminApi.platform().then((r) => r.data),
  });

  const usageData: { client_name: string; sessions: number; recognition_rate: number }[] =
    (usageRaw as { clients?: { client_name: string; sessions: number; recognition_rate: number }[] })?.clients ?? [];

  const platform = platformRaw as Record<string, number | undefined> | undefined;

  const platformStats = [
    { label: "Total Cameras",   value: platform?.total_cameras   ?? 0 },
    { label: "Enrolled Persons",value: platform?.total_persons   ?? 0 },
    { label: "Sessions Today",  value: platform?.sessions_today  ?? 0 },
    { label: "GPU Hours",       value: platform?.gpu_hours_today ?? 0 },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-zinc-50">Platform Analytics</h1>
        <p className="text-sm text-muted-foreground mt-0.5">Global usage and performance metrics</p>
      </div>

      {/* Platform stat cards */}
      <div className="grid grid-cols-4 gap-4">
        {platformLoading
          ? Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-20 rounded-xl" />)
          : platformStats.map((s) => (
              <div key={s.label} className="card-stat">
                <p className="text-xs text-muted-foreground">{s.label}</p>
                <p className="mt-1 text-2xl font-bold tabular-nums text-zinc-100">
                  {typeof s.value === "number" ? s.value.toLocaleString() : s.value}
                </p>
              </div>
            ))}
      </div>

      <div className="grid grid-cols-2 gap-6">
        {/* Sessions per client */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Sessions by Client</CardTitle>
          </CardHeader>
          <CardContent>
            {usageLoading ? (
              <Skeleton className="h-48 w-full" />
            ) : (
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={usageData} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={CHART_STYLE.gridStroke} />
                  <XAxis dataKey="client_name" tick={CHART_STYLE.tickProps} tickLine={false} />
                  <YAxis tick={CHART_STYLE.tickProps} tickLine={false} />
                  <Tooltip contentStyle={CHART_STYLE.contentStyle} />
                  <Bar dataKey="sessions" fill="#6366f1" radius={[3, 3, 0, 0]} name="Sessions" />
                </BarChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>

        {/* Recognition rate per client */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Avg Recognition Rate by Client</CardTitle>
          </CardHeader>
          <CardContent>
            {usageLoading ? (
              <Skeleton className="h-48 w-full" />
            ) : (
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie
                    data={usageData}
                    dataKey="recognition_rate"
                    nameKey="client_name"
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    label={({ name, value }: { name: string; value: number }) => `${name}: ${(value * 100).toFixed(0)}%`}
                  >
                    {usageData.map((_, index) => (
                      <Cell key={index} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={CHART_STYLE.contentStyle}
                    formatter={(v: number) => [`${(v * 100).toFixed(1)}%`, "Recognition"]}
                  />
                </PieChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
