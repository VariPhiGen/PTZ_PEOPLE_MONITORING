"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Building2,
  Camera,
  ChevronRight,
  CircleDot,
  Plus,
  Search,
  Users,
} from "lucide-react";
import { toast } from "sonner";
import { adminApi } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import type { Client } from "@/types";

// ── Client row ────────────────────────────────────────────────────────────────

function ClientRow({ client, onSelect }: { client: Client; onSelect: (c: Client) => void }) {
  const statusColor: Record<string, string> = {
    ACTIVE:   "text-green-400 bg-green-400/10",
    SUSPENDED:"text-amber-400 bg-amber-400/10",
    ARCHIVED: "text-zinc-500 bg-zinc-500/10",
  };

  return (
    <tr
      className="border-b border-border/40 hover:bg-zinc-800/30 cursor-pointer transition-colors"
      onClick={() => onSelect(client)}
    >
      <td className="py-3 px-4">
        <div className="flex items-center gap-3">
          {client.logo_url ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={client.logo_url} alt={client.name} className="h-7 w-7 rounded object-contain" />
          ) : (
            <div className="flex h-7 w-7 items-center justify-center rounded bg-zinc-700 text-xs font-bold text-zinc-300">
              {client.name[0]}
            </div>
          )}
          <div>
            <p className="text-sm font-medium text-zinc-100">{client.name}</p>
            <p className="text-xs text-zinc-500 font-mono">{client.slug}</p>
          </div>
        </div>
      </td>
      <td className="py-3 px-3">
        <span className={cn("text-xs rounded-full px-2 py-0.5 font-medium", statusColor[client.status])}>
          {client.status}
        </span>
      </td>
      <td className="py-3 px-3 text-right">
        <div className="flex items-center justify-end gap-1 text-xs text-zinc-400">
          <Camera className="h-3 w-3" />
          {client.camera_count}/{client.max_cameras}
        </div>
      </td>
      <td className="py-3 px-3 text-right">
        <div className="flex items-center justify-end gap-1 text-xs text-zinc-400">
          <Users className="h-3 w-3" />
          {client.person_count.toLocaleString()}/{client.max_persons.toLocaleString()}
        </div>
      </td>
      <td className="py-3 px-3 text-center">
        <Badge
          variant={client.active_sessions > 0 ? "success" : "secondary"}
          className="text-[10px]"
        >
          {client.active_sessions} active
        </Badge>
      </td>
      <td className="py-3 px-3 text-right">
        <ChevronRight className="h-4 w-4 text-zinc-600 inline-block" />
      </td>
    </tr>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function AdminClientsPage() {
  const [search,   setSearch]   = useState("");
  const [selected, setSelected] = useState<Client | null>(null);
  const qc = useQueryClient();

  const { data: clientsData, isLoading } = useQuery({
    queryKey: ["admin-clients", search],
    queryFn:  () => adminApi.clients({ q: search || undefined, limit: 100 }).then((r) => r.data),
  });

  const clients: Client[] = (clientsData as { items?: Client[] })?.items ?? [];

  const statusMutation = useMutation({
    mutationFn: ({ id, status }: { id: string; status: string }) =>
      adminApi.clientStatus(id, { status }),
    onSuccess: () => {
      toast.success("Client status updated");
      qc.invalidateQueries({ queryKey: ["admin-clients"] });
      setSelected(null);
    },
    onError: () => toast.error("Update failed"),
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold text-zinc-50">Clients</h1>
          <p className="text-sm text-muted-foreground mt-0.5">
            {clients.length} registered organisations
          </p>
        </div>
        <Button size="sm">
          <Plus className="h-4 w-4" />
          New Client
        </Button>
      </div>

      <div className="grid grid-cols-12 gap-6">
        {/* Table */}
        <div className={cn("space-y-3", selected ? "col-span-7" : "col-span-12")}>
          <div className="relative max-w-xs">
            <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-zinc-500" />
            <Input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search clients…"
              className="pl-8 bg-zinc-800 border-zinc-700"
            />
          </div>

          {isLoading ? (
            <div className="space-y-1.5">
              {Array.from({ length: 6 }).map((_, i) => <Skeleton key={i} className="h-12 rounded-lg" />)}
            </div>
          ) : (
            <div className="overflow-hidden rounded-xl border border-border/50">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border/50 bg-zinc-900/50">
                    <th className="py-2.5 px-4 text-left text-xs font-medium text-muted-foreground">Organisation</th>
                    <th className="py-2.5 px-3 text-left text-xs font-medium text-muted-foreground">Status</th>
                    <th className="py-2.5 px-3 text-right text-xs font-medium text-muted-foreground">Cameras</th>
                    <th className="py-2.5 px-3 text-right text-xs font-medium text-muted-foreground">Persons</th>
                    <th className="py-2.5 px-3 text-center text-xs font-medium text-muted-foreground">Sessions</th>
                    <th className="py-2.5 px-3" />
                  </tr>
                </thead>
                <tbody>
                  {clients.map((c) => (
                    <ClientRow key={c.client_id} client={c} onSelect={setSelected} />
                  ))}
                  {clients.length === 0 && (
                    <tr><td colSpan={6} className="py-10 text-center text-sm text-muted-foreground">No clients</td></tr>
                  )}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Detail panel */}
        {selected && (
          <div className="col-span-5">
            <Card>
              <CardContent className="pt-5 space-y-4">
                <div className="flex items-start justify-between">
                  <div>
                    <h2 className="font-semibold text-zinc-100">{selected.name}</h2>
                    <p className="text-xs text-zinc-500 font-mono mt-0.5">{selected.slug}</p>
                  </div>
                  <button
                    onClick={() => setSelected(null)}
                    className="text-zinc-600 hover:text-zinc-400 text-lg leading-none"
                  >
                    ×
                  </button>
                </div>

                <div className="grid grid-cols-2 gap-2 text-xs">
                  {[
                    { label: "Cameras",  value: `${selected.camera_count} / ${selected.max_cameras}` },
                    { label: "Persons",  value: `${selected.person_count.toLocaleString()} / ${selected.max_persons.toLocaleString()}` },
                    { label: "Sessions", value: selected.active_sessions.toString() },
                    { label: "Status",   value: selected.status },
                  ].map(({ label, value }) => (
                    <div key={label} className="rounded-lg bg-zinc-800/60 px-3 py-2">
                      <p className="text-zinc-500">{label}</p>
                      <p className="font-medium text-zinc-200 mt-0.5">{value}</p>
                    </div>
                  ))}
                </div>

                <div className="flex gap-2 flex-wrap">
                  {selected.status === "ACTIVE" ? (
                    <Button
                      size="sm"
                      variant="outline"
                      className="text-amber-400 border-amber-400/30 hover:bg-amber-400/10"
                      onClick={() => statusMutation.mutate({ id: selected.client_id, status: "SUSPENDED" })}
                      disabled={statusMutation.isPending}
                    >
                      Suspend
                    </Button>
                  ) : (
                    <Button
                      size="sm"
                      variant="outline"
                      className="text-green-400 border-green-400/30 hover:bg-green-400/10"
                      onClick={() => statusMutation.mutate({ id: selected.client_id, status: "ACTIVE" })}
                      disabled={statusMutation.isPending}
                    >
                      Activate
                    </Button>
                  )}
                  <Button size="sm" variant="outline">Edit Limits</Button>
                  <Button size="sm" variant="outline">Manage Nodes</Button>
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
}
