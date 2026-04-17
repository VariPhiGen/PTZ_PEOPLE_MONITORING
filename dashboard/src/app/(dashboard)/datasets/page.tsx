"use client";

import { useState, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Archive, ChevronRight, Database, Edit2, Plus, Search, Users,
} from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { datasetsApi, adminApi } from "@/lib/api";
import { useAuth } from "@/hooks/useAuth";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter,
} from "@/components/ui/dialog";

// ── Types ──────────────────────────────────────────────────────────────────────

interface FaceDataset {
  dataset_id:   string;
  client_id:    string;
  name:         string;
  description:  string | null;
  color:        string;
  is_default:   boolean;
  status:       string;
  person_count: number;
  camera_count?: number;
  created_at:   number;
  updated_at:   number;
}

interface DatasetPerson {
  person_id:       string;
  name:            string;
  role:            string;
  department:      string | null;
  external_id:     string | null;
  status:          string;
  embedding_count: number;
}

interface ClientOption {
  client_id: string;
  name:      string;
  slug:      string;
}

// ── Color picker presets ───────────────────────────────────────────────────────

const COLOR_PRESETS = [
  "#6366f1", "#8b5cf6", "#ec4899", "#f43f5e",
  "#f97316", "#eab308", "#22c55e", "#14b8a6",
  "#3b82f6", "#06b6d4",
];

// ── Main Component ─────────────────────────────────────────────────────────────

export default function DatasetsPage() {
  const { role, clientId, isSuperAdmin } = useAuth();
  const qc       = useQueryClient();
  const isAdmin  = role === "CLIENT_ADMIN" || role === "SUPER_ADMIN";

  // SUPER_ADMIN picks a client; CLIENT_ADMIN uses their own
  const [selectedClientId, setSelectedClientId] = useState<string>("");
  const effectiveClientId = isSuperAdmin ? selectedClientId : (clientId ?? "");

  const [search,          setSearch]          = useState("");
  const [showCreate,      setShowCreate]      = useState(false);
  const [editingDataset,  setEditingDataset]  = useState<FaceDataset | null>(null);
  const [selectedDataset, setSelectedDataset] = useState<FaceDataset | null>(null);
  const [personSearch,    setPersonSearch]    = useState("");
  const [form, setForm] = useState({ name: "", description: "", color: "#6366f1" });

  // ── Clients list (SUPER_ADMIN only) ──────────────────────────────────────
  const { data: clientsData } = useQuery({
    queryKey: ["admin-clients"],
    queryFn:  () => adminApi.clients().then(r => r.data),
    enabled:  isSuperAdmin,
    staleTime: 120_000,
  });
  const clients: ClientOption[] = (clientsData?.items ?? clientsData ?? [])
    .map((c: any) => ({ client_id: c.client_id, name: c.name, slug: c.slug }));

  // Auto-select first client for SUPER_ADMIN
  useEffect(() => {
    if (isSuperAdmin && !selectedClientId && clients.length > 0) {
      setSelectedClientId(clients[0].client_id);
    }
  }, [isSuperAdmin, selectedClientId, clients]);

  // Reset dataset selection when client changes
  useEffect(() => {
    setSelectedDataset(null);
  }, [effectiveClientId]);

  // ── Data ───────────────────────────────────────────────────────────────────

  const { data: datasetsData, isLoading } = useQuery({
    queryKey: ["datasets", effectiveClientId],
    queryFn:  () => datasetsApi.list(
      effectiveClientId ? { client_id: effectiveClientId } : undefined,
    ).then(r => r.data),
    enabled: !!effectiveClientId,
    refetchInterval: 30_000,
  });

  const { data: personsData } = useQuery({
    queryKey: ["dataset-persons", selectedDataset?.dataset_id, personSearch, effectiveClientId],
    queryFn:  () =>
      datasetsApi.persons(selectedDataset!.dataset_id, {
        q: personSearch || undefined,
        limit: 100,
        client_id: isSuperAdmin ? effectiveClientId : undefined,
      }).then(r => r.data),
    enabled: !!selectedDataset,
  });

  // ── Mutations ──────────────────────────────────────────────────────────────

  const createMut = useMutation({
    mutationFn: (data: typeof form) =>
      datasetsApi.create({
        ...data,
        ...(isSuperAdmin && effectiveClientId ? { client_id: effectiveClientId } : {}),
      }),
    onSuccess: () => {
      toast.success("Dataset created");
      qc.invalidateQueries({ queryKey: ["datasets"] });
      setShowCreate(false);
      setForm({ name: "", description: "", color: "#6366f1" });
    },
    onError: (e: any) => toast.error(e.response?.data?.detail ?? "Failed to create dataset"),
  });

  const updateMut = useMutation({
    mutationFn: ({ id, data }: { id: string; data: Partial<typeof form> }) =>
      datasetsApi.update(id, {
        ...data,
        ...(isSuperAdmin && effectiveClientId ? { client_id: effectiveClientId } : {}),
      }),
    onSuccess: () => {
      toast.success("Dataset updated");
      qc.invalidateQueries({ queryKey: ["datasets"] });
      setEditingDataset(null);
    },
    onError: (e: any) => toast.error(e.response?.data?.detail ?? "Failed to update dataset"),
  });

  const archiveMut = useMutation({
    mutationFn: (id: string) => datasetsApi.archive(id),
    onSuccess: () => {
      toast.success("Dataset archived");
      qc.invalidateQueries({ queryKey: ["datasets"] });
    },
    onError: (e: any) => toast.error(e.response?.data?.detail ?? "Failed to archive"),
  });

  const moveMut = useMutation({
    mutationFn: ({ personId, targetId }: { personId: string; targetId: string }) =>
      datasetsApi.movePerson(selectedDataset!.dataset_id, personId, targetId),
    onSuccess: () => {
      toast.success("Person moved");
      qc.invalidateQueries({ queryKey: ["dataset-persons"] });
      qc.invalidateQueries({ queryKey: ["datasets"] });
    },
    onError: (e: any) => toast.error(e.response?.data?.detail ?? "Failed to move person"),
  });

  // ── Filtered datasets ──────────────────────────────────────────────────────

  const datasets: FaceDataset[] = (datasetsData?.items ?? []).filter(
    (d: FaceDataset) =>
      !search || d.name.toLowerCase().includes(search.toLowerCase()),
  );

  function openEdit(ds: FaceDataset) {
    setEditingDataset(ds);
    setForm({ name: ds.name, description: ds.description ?? "", color: ds.color });
  }

  // ── Render ───────────────────────────────────────────────────────────────────

  return (
    <div className="flex h-full flex-col">
      {/* SUPER_ADMIN client picker */}
      {isSuperAdmin && (
        <div className="flex items-center gap-3 px-4 py-2.5 border-b border-border bg-muted/30">
          <span className="text-xs text-muted-foreground whitespace-nowrap">Client:</span>
          <select
            className="h-8 rounded-md border border-border bg-background px-3 text-sm flex-1 max-w-xs"
            value={selectedClientId}
            onChange={e => setSelectedClientId(e.target.value)}
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

      {/* CLIENT_ADMIN header */}
      {!isSuperAdmin && (
        <div className="flex items-center gap-3 px-4 py-3 border-b border-border">
          <Database className="h-5 w-5 text-primary" />
          <div>
            <h1 className="text-base font-semibold">My Face Datasets</h1>
            <p className="text-xs text-muted-foreground">Manage recognition groups for your cameras</p>
          </div>
        </div>
      )}

      {!effectiveClientId && isSuperAdmin ? (
        <div className="flex flex-col items-center justify-center flex-1 text-muted-foreground gap-3 p-8">
          <Database className="h-10 w-10 opacity-30" />
          <p className="text-sm">Select a client above to manage their face datasets</p>
        </div>
      ) : (
        <div className="flex flex-1 gap-0 min-h-0">
          {/* Left panel — dataset list */}
          <div className="flex flex-col w-72 min-w-72 border-r border-border bg-card">
            <div className="flex items-center justify-between px-4 py-3 border-b border-border">
              <div className="flex items-center gap-2">
                <Database className="h-4 w-4 text-primary" />
                <span className="text-sm font-semibold">Face Datasets</span>
              </div>
              {isAdmin && (
                <Button size="sm" variant="ghost" className="h-7 w-7 p-0" onClick={() => setShowCreate(true)}>
                  <Plus className="h-4 w-4" />
                </Button>
              )}
            </div>

            <div className="px-3 py-2 border-b border-border">
              <div className="relative">
                <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
                <Input
                  placeholder="Search datasets…"
                  value={search}
                  onChange={e => setSearch(e.target.value)}
                  className="h-7 pl-7 text-xs"
                />
              </div>
            </div>

            <div className="flex-1 overflow-y-auto py-1">
              {isLoading && (
                <div className="space-y-1 px-2 py-2">
                  {[1, 2, 3].map(i => (
                    <div key={i} className="h-14 rounded-md bg-muted/40 animate-pulse" />
                  ))}
                </div>
              )}
              {datasets.map(ds => (
                <button
                  key={ds.dataset_id}
                  onClick={() => setSelectedDataset(ds)}
                  className={cn(
                    "w-full flex items-start gap-3 px-3 py-2.5 text-left hover:bg-accent/50 transition-colors",
                    selectedDataset?.dataset_id === ds.dataset_id && "bg-accent",
                  )}
                >
                  <span
                    className="mt-0.5 h-3 w-3 rounded-full shrink-0"
                    style={{ backgroundColor: ds.color }}
                  />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-1.5">
                      <span className="text-sm font-medium truncate">{ds.name}</span>
                      {ds.is_default && (
                        <Badge variant="outline" className="text-[10px] px-1 py-0 h-4">Default</Badge>
                      )}
                    </div>
                    <span className="text-xs text-muted-foreground">
                      {ds.person_count.toLocaleString()} person{ds.person_count !== 1 ? "s" : ""}
                    </span>
                  </div>
                  {selectedDataset?.dataset_id === ds.dataset_id && (
                    <ChevronRight className="h-3.5 w-3.5 text-muted-foreground shrink-0 mt-1" />
                  )}
                </button>
              ))}
              {!isLoading && datasets.length === 0 && (
                <p className="text-xs text-muted-foreground text-center py-8">No datasets yet</p>
              )}
            </div>
          </div>

          {/* Right panel — dataset detail */}
          <div className="flex-1 overflow-y-auto p-6">
            {!selectedDataset ? (
              <div className="flex flex-col items-center justify-center h-full text-muted-foreground gap-3">
                <Database className="h-10 w-10 opacity-30" />
                <p className="text-sm">Select a dataset to view its enrolled persons</p>
                {isAdmin && (
                  <Button size="sm" variant="outline" onClick={() => setShowCreate(true)}>
                    <Plus className="h-4 w-4 mr-1.5" /> Create Dataset
                  </Button>
                )}
              </div>
            ) : (
              <div className="max-w-4xl mx-auto space-y-6">
                {/* Dataset header */}
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-3">
                    <span
                      className="h-5 w-5 rounded-full"
                      style={{ backgroundColor: selectedDataset.color }}
                    />
                    <div>
                      <h1 className="text-xl font-semibold flex items-center gap-2">
                        {selectedDataset.name}
                        {selectedDataset.is_default && (
                          <Badge variant="secondary" className="text-xs">Default</Badge>
                        )}
                      </h1>
                      {selectedDataset.description && (
                        <p className="text-sm text-muted-foreground mt-0.5">{selectedDataset.description}</p>
                      )}
                    </div>
                  </div>
                  {isAdmin && (
                    <div className="flex items-center gap-2">
                      <Button size="sm" variant="outline" onClick={() => openEdit(selectedDataset)}>
                        <Edit2 className="h-3.5 w-3.5 mr-1.5" /> Edit
                      </Button>
                      {!selectedDataset.is_default && (
                        <Button
                          size="sm"
                          variant="outline"
                          className="text-destructive hover:text-destructive"
                          onClick={() => {
                            if (confirm(`Archive dataset "${selectedDataset.name}"? All persons must be moved first.`)) {
                              archiveMut.mutate(selectedDataset.dataset_id);
                              setSelectedDataset(null);
                            }
                          }}
                        >
                          <Archive className="h-3.5 w-3.5 mr-1.5" /> Archive
                        </Button>
                      )}
                    </div>
                  )}
                </div>

                {/* Stats row */}
                <div className="grid grid-cols-3 gap-4">
                  {[
                    { label: "Enrolled Persons", value: selectedDataset.person_count.toLocaleString(), icon: Users },
                    { label: "Cameras Attached", value: (selectedDataset.camera_count ?? "—").toString(), icon: Database },
                    { label: "Status",            value: selectedDataset.status, icon: null },
                  ].map(stat => (
                    <div key={stat.label} className="rounded-lg border border-border bg-card p-4">
                      <p className="text-xs text-muted-foreground">{stat.label}</p>
                      <p className="text-2xl font-semibold mt-1">{stat.value}</p>
                    </div>
                  ))}
                </div>

                {/* Persons list */}
                <div className="rounded-lg border border-border bg-card overflow-hidden">
                  <div className="flex items-center justify-between px-4 py-3 border-b border-border">
                    <span className="text-sm font-medium">Enrolled Persons</span>
                    <div className="relative w-52">
                      <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
                      <Input
                        placeholder="Search persons…"
                        value={personSearch}
                        onChange={e => setPersonSearch(e.target.value)}
                        className="h-7 pl-7 text-xs"
                      />
                    </div>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-border bg-muted/30">
                          <th className="text-left px-4 py-2 text-xs text-muted-foreground font-medium">Name</th>
                          <th className="text-left px-4 py-2 text-xs text-muted-foreground font-medium">ID</th>
                          <th className="text-left px-4 py-2 text-xs text-muted-foreground font-medium">Role</th>
                          <th className="text-left px-4 py-2 text-xs text-muted-foreground font-medium">Department</th>
                          <th className="text-left px-4 py-2 text-xs text-muted-foreground font-medium">Templates</th>
                          {isAdmin && (
                            <th className="text-left px-4 py-2 text-xs text-muted-foreground font-medium">Move To</th>
                          )}
                        </tr>
                      </thead>
                      <tbody>
                        {(personsData?.items ?? []).map((p: DatasetPerson) => (
                          <tr key={p.person_id} className="border-b border-border/50 hover:bg-muted/20">
                            <td className="px-4 py-2.5 font-medium">{p.name}</td>
                            <td className="px-4 py-2.5 text-muted-foreground text-xs font-mono">
                              {p.external_id ?? "—"}
                            </td>
                            <td className="px-4 py-2.5">
                              <Badge variant="outline" className="text-xs capitalize">
                                {p.role.toLowerCase()}
                              </Badge>
                            </td>
                            <td className="px-4 py-2.5 text-muted-foreground text-xs">
                              {p.department ?? "—"}
                            </td>
                            <td className="px-4 py-2.5">
                              <span className={cn(
                                "text-xs px-1.5 py-0.5 rounded",
                                p.embedding_count > 0
                                  ? "bg-green-500/15 text-green-400"
                                  : "bg-zinc-500/15 text-zinc-400",
                              )}>
                                {p.embedding_count > 0 ? `${p.embedding_count} / 5` : "none"}
                              </span>
                            </td>
                            {isAdmin && (
                              <td className="px-4 py-2.5">
                                <select
                                  className="text-xs bg-muted border border-border rounded px-2 py-1"
                                  defaultValue=""
                                  onChange={e => {
                                    if (e.target.value) {
                                      moveMut.mutate({ personId: p.person_id, targetId: e.target.value });
                                      e.target.value = "";
                                    }
                                  }}
                                >
                                  <option value="">Move to…</option>
                                  {datasets
                                    .filter(d => d.dataset_id !== selectedDataset.dataset_id)
                                    .map(d => (
                                      <option key={d.dataset_id} value={d.dataset_id}>{d.name}</option>
                                    ))
                                  }
                                </select>
                              </td>
                            )}
                          </tr>
                        ))}
                        {!personsData?.items?.length && (
                          <tr>
                            <td colSpan={6} className="text-center py-8 text-sm text-muted-foreground">
                              No persons enrolled in this dataset yet.
                              Assign a dataset when enrolling faces.
                            </td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Create / Edit dataset dialog */}
      <Dialog open={showCreate || !!editingDataset} onOpenChange={open => {
        if (!open) { setShowCreate(false); setEditingDataset(null); }
      }}>
        <DialogContent className="max-w-sm">
          <DialogHeader>
            <DialogTitle>{editingDataset ? "Edit Dataset" : "New Face Dataset"}</DialogTitle>
          </DialogHeader>
          <div className="space-y-4 py-2">
            <div className="space-y-1.5">
              <Label>Name</Label>
              <Input
                placeholder="e.g. Employees, Visitors, VIPs…"
                value={form.name}
                onChange={e => setForm(f => ({ ...f, name: e.target.value }))}
              />
            </div>
            <div className="space-y-1.5">
              <Label>Description <span className="text-muted-foreground">(optional)</span></Label>
              <Input
                placeholder="Short description"
                value={form.description}
                onChange={e => setForm(f => ({ ...f, description: e.target.value }))}
              />
            </div>
            <div className="space-y-1.5">
              <Label>Color</Label>
              <div className="flex flex-wrap gap-2">
                {COLOR_PRESETS.map(c => (
                  <button
                    key={c}
                    onClick={() => setForm(f => ({ ...f, color: c }))}
                    className={cn(
                      "h-6 w-6 rounded-full border-2 transition-all",
                      form.color === c ? "border-white scale-110" : "border-transparent",
                    )}
                    style={{ backgroundColor: c }}
                  />
                ))}
                <input
                  type="color"
                  value={form.color}
                  onChange={e => setForm(f => ({ ...f, color: e.target.value }))}
                  className="h-6 w-6 rounded cursor-pointer border-0 bg-transparent"
                  title="Custom color"
                />
              </div>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => { setShowCreate(false); setEditingDataset(null); }}>
              Cancel
            </Button>
            <Button
              disabled={!form.name.trim() || createMut.isPending || updateMut.isPending}
              onClick={() => {
                if (editingDataset) {
                  updateMut.mutate({ id: editingDataset.dataset_id, data: form });
                } else {
                  createMut.mutate(form);
                }
              }}
            >
              {editingDataset ? "Save Changes" : "Create Dataset"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
