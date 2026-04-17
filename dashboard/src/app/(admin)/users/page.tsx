"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Check, ChevronDown, Eye, EyeOff, Loader2, Plus,
  Search, Shield, ShieldCheck, User as UserIcon,
} from "lucide-react";
import { toast } from "sonner";
import { adminApi } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { cn, fmtDatetime } from "@/lib/utils";

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

interface AdminUser {
  user_id:     string;
  email:       string;
  name:        string;
  role:        string;
  client_id:   string | null;
  client_name: string | null;
  status:      string;
  mfa_enabled: boolean;
  last_login:  string | null;
  created_at:  string | null;
}

interface AdminClient {
  client_id: string;
  name:      string;
  slug:      string;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

const ROLE_STYLES: Record<string, { label: string; color: string; icon: React.ElementType }> = {
  SUPER_ADMIN:  { label: "Super Admin",  color: "bg-violet-500/15 text-violet-400",  icon: Shield      },
  CLIENT_ADMIN: { label: "Client Admin", color: "bg-blue-500/15 text-blue-400",       icon: ShieldCheck },
  VIEWER:       { label: "Viewer",       color: "bg-zinc-500/15 text-zinc-400",        icon: UserIcon    },
};

function RolePill({ role }: { role: string }) {
  const { label, color } = ROLE_STYLES[role] ?? ROLE_STYLES.VIEWER;
  return <span className={cn("rounded-full px-2 py-0.5 text-[10px] font-semibold", color)}>{label}</span>;
}

function initials(name: string) {
  return name.split(" ").map((w) => w[0]).join("").slice(0, 2).toUpperCase();
}

function validatePassword(p: string): string | null {
  if (p.length < 10)              return "At least 10 characters";
  if (!/[A-Z]/.test(p))          return "One uppercase letter required";
  if (!/\d/.test(p))             return "One number required";
  if (!/[!@#$%^&*]/.test(p))    return "One special character (!@#$%^&*) required";
  return null;
}

// ─────────────────────────────────────────────────────────────────────────────
// Password strength meter
// ─────────────────────────────────────────────────────────────────────────────

function PasswordRules({ password }: { password: string }) {
  const rules = [
    { label: "10+ chars",  ok: password.length >= 10 },
    { label: "Uppercase",  ok: /[A-Z]/.test(password) },
    { label: "Number",     ok: /\d/.test(password) },
    { label: "Special",    ok: /[!@#$%^&*]/.test(password) },
  ];
  return (
    <div className="flex gap-3 mt-1">
      {rules.map(({ label, ok }) => (
        <span key={label} className={cn("flex items-center gap-1 text-[10px]", ok ? "text-green-400" : "text-zinc-600")}>
          <span className={cn("h-1.5 w-1.5 rounded-full", ok ? "bg-green-400" : "bg-zinc-700")} />
          {label}
        </span>
      ))}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Create User Dialog
// ─────────────────────────────────────────────────────────────────────────────

interface CreateUserForm {
  name:      string;
  email:     string;
  password:  string;
  role:      string;
  client_id: string;
}

function CreateUserDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const qc   = useQueryClient();
  const [form, setForm]   = useState<CreateUserForm>({
    name: "", email: "", password: "", role: "CLIENT_ADMIN", client_id: "",
  });
  const [showPass, setShowPass] = useState(false);
  const [error,    setError]    = useState<string | null>(null);

  const set = (k: keyof CreateUserForm, v: string) => setForm((s) => ({ ...s, [k]: v }));

  // Fetch clients for the selector
  const { data: clientsRaw } = useQuery({
    queryKey: ["clients-simple"],
    queryFn:  () => adminApi.clients({ limit: 200 }).then((r) => r.data),
    enabled:  open,
  });
  const clients: AdminClient[] = (clientsRaw as { items?: AdminClient[] })?.items ?? [];

  const mutation = useMutation({
    mutationFn: () => adminApi.createUser({
      name:      form.name,
      email:     form.email,
      password:  form.password,
      role:      form.role,
      client_id: form.role === "SUPER_ADMIN" ? undefined : form.client_id || undefined,
    }),
    onSuccess: () => {
      toast.success(`User "${form.name}" created`);
      qc.invalidateQueries({ queryKey: ["admin-users"] });
      handleClose();
    },
    onError: (err: unknown) => {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      toast.error(typeof detail === "string" ? detail : "Failed to create user");
    },
  });

  function handleClose() {
    setForm({ name: "", email: "", password: "", role: "CLIENT_ADMIN", client_id: "" });
    setError(null);
    onClose();
  }

  function handleSubmit() {
    setError(null);
    if (!form.name.trim())  { setError("Name is required"); return; }
    if (!form.email.trim()) { setError("Email is required"); return; }
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(form.email)) { setError("Invalid email"); return; }
    const passErr = validatePassword(form.password);
    if (passErr) { setError(passErr); return; }
    if (form.role !== "SUPER_ADMIN" && !form.client_id) {
      setError("Select a client for non-super-admin users");
      return;
    }
    mutation.mutate();
  }

  return (
    <Dialog open={open} onOpenChange={(o) => !o && handleClose()}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Create User</DialogTitle>
          <DialogDescription>Add a new user account to the platform.</DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-2">
          {/* Name */}
          <div className="space-y-1.5">
            <Label className="text-xs text-muted-foreground">Full name *</Label>
            <Input value={form.name} onChange={(e) => set("name", e.target.value)}
              placeholder="Jane Smith" className="bg-zinc-900 border-zinc-700" />
          </div>

          {/* Email */}
          <div className="space-y-1.5">
            <Label className="text-xs text-muted-foreground">Email *</Label>
            <Input type="email" value={form.email} onChange={(e) => set("email", e.target.value)}
              placeholder="jane@institute.edu" className="bg-zinc-900 border-zinc-700" />
          </div>

          {/* Password */}
          <div className="space-y-1.5">
            <Label className="text-xs text-muted-foreground">Password *</Label>
            <div className="relative">
              <Input
                type={showPass ? "text" : "password"}
                value={form.password}
                onChange={(e) => set("password", e.target.value)}
                placeholder="Min. 10 chars"
                className="bg-zinc-900 border-zinc-700 pr-10"
              />
              <button type="button" tabIndex={-1}
                onClick={() => setShowPass((s) => !s)}
                className="absolute inset-y-0 right-0 flex items-center px-3 text-zinc-500 hover:text-zinc-300">
                {showPass ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </button>
            </div>
            <PasswordRules password={form.password} />
          </div>

          {/* Role */}
          <div className="space-y-1.5">
            <Label className="text-xs text-muted-foreground">Role *</Label>
            <div className="grid grid-cols-3 gap-2">
              {(["SUPER_ADMIN", "CLIENT_ADMIN", "VIEWER"] as const).map((r) => {
                const { label, icon: Icon } = ROLE_STYLES[r];
                return (
                  <button
                    key={r}
                    type="button"
                    onClick={() => set("role", r)}
                    className={cn(
                      "flex flex-col items-center gap-1 rounded-lg border py-2.5 text-xs font-medium transition-colors",
                      form.role === r
                        ? "border-zinc-500 bg-zinc-800 text-zinc-100"
                        : "border-zinc-800 text-zinc-500 hover:border-zinc-600 hover:text-zinc-400",
                    )}
                  >
                    <Icon className="h-4 w-4" />
                    {label}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Client selector (hidden for SUPER_ADMIN) */}
          {form.role !== "SUPER_ADMIN" && (
            <div className="space-y-1.5">
              <Label className="text-xs text-muted-foreground">Client *</Label>
              <div className="relative">
                <select
                  value={form.client_id}
                  onChange={(e) => set("client_id", e.target.value)}
                  className="w-full appearance-none rounded-md border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-zinc-100 focus:outline-none focus:ring-2 focus:ring-ring"
                >
                  <option value="">Select a client…</option>
                  {clients.map((c) => (
                    <option key={c.client_id} value={c.client_id}>{c.name}</option>
                  ))}
                </select>
                <ChevronDown className="pointer-events-none absolute right-3 top-2.5 h-4 w-4 text-zinc-500" />
              </div>
            </div>
          )}

          {error && (
            <p className="text-sm text-red-400 bg-red-400/10 rounded-lg px-3 py-2">{error}</p>
          )}
        </div>

        <DialogFooter>
          <Button variant="ghost" size="sm" onClick={handleClose}>Cancel</Button>
          <Button size="sm" onClick={handleSubmit} disabled={mutation.isPending}>
            {mutation.isPending ? <><Loader2 className="h-4 w-4 animate-spin" />Creating…</> : "Create User"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Edit User Dialog
// ─────────────────────────────────────────────────────────────────────────────

function EditUserDialog({ user, onClose }: { user: AdminUser | null; onClose: () => void }) {
  const qc = useQueryClient();
  const [name,  setName]  = useState(user?.name  ?? "");
  const [email, setEmail] = useState(user?.email ?? "");

  // Sync when user changes
  const key = user?.user_id;

  const mutation = useMutation({
    mutationFn: () => adminApi.updateUser(user!.user_id, { name, email }),
    onSuccess: () => {
      toast.success("User updated");
      qc.invalidateQueries({ queryKey: ["admin-users"] });
      onClose();
    },
    onError: () => toast.error("Update failed"),
  });

  if (!user) return null;

  return (
    <Dialog open={!!user} onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-w-sm">
        <DialogHeader>
          <DialogTitle>Edit User</DialogTitle>
        </DialogHeader>
        <div className="space-y-4 py-2">
          <div className="space-y-1.5">
            <Label className="text-xs text-muted-foreground">Name</Label>
            <Input value={name} onChange={(e) => setName(e.target.value)}
              className="bg-zinc-900 border-zinc-700" />
          </div>
          <div className="space-y-1.5">
            <Label className="text-xs text-muted-foreground">Email</Label>
            <Input type="email" value={email} onChange={(e) => setEmail(e.target.value)}
              className="bg-zinc-900 border-zinc-700" />
          </div>
        </div>
        <DialogFooter>
          <Button variant="ghost" size="sm" onClick={onClose}>Cancel</Button>
          <Button size="sm" onClick={() => mutation.mutate()} disabled={mutation.isPending}>
            {mutation.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Check className="h-4 w-4" />}
            Save
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Page
// ─────────────────────────────────────────────────────────────────────────────

export default function UsersPage() {
  const qc = useQueryClient();
  const [search,      setSearch]      = useState("");
  const [roleFilter,  setRoleFilter]  = useState("ALL");
  const [showCreate,  setShowCreate]  = useState(false);
  const [editTarget,  setEditTarget]  = useState<AdminUser | null>(null);

  const { data, isLoading } = useQuery({
    queryKey: ["admin-users", search, roleFilter],
    queryFn:  () =>
      adminApi.users({
        q:     search || undefined,
        role:  roleFilter !== "ALL" ? roleFilter : undefined,
        limit: 200,
      }).then((r) => r.data),
  });

  const users: AdminUser[] = (data as { items?: AdminUser[] })?.items ?? [];

  const statusMutation = useMutation({
    mutationFn: ({ id, status }: { id: string; status: string }) =>
      adminApi.userStatus(id, { status }),
    onSuccess: () => {
      toast.success("Status updated");
      qc.invalidateQueries({ queryKey: ["admin-users"] });
    },
    onError: () => toast.error("Failed"),
  });

  const resetMutation = useMutation({
    mutationFn: (id: string) => adminApi.resetPassword(id),
    onSuccess:  () => toast.success("Password reset initiated"),
    onError:    () => toast.error("Reset failed"),
  });

  const roleCounts: Record<string, number> = users.reduce((acc, u) => {
    acc[u.role] = (acc[u.role] ?? 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold text-zinc-50">Users</h1>
          <p className="text-sm text-muted-foreground mt-0.5">
            {users.length} accounts ·{" "}
            {roleCounts.SUPER_ADMIN ?? 0} super admin ·{" "}
            {roleCounts.CLIENT_ADMIN ?? 0} client admin ·{" "}
            {roleCounts.VIEWER ?? 0} viewer
          </p>
        </div>
        <Button size="sm" onClick={() => setShowCreate(true)}>
          <Plus className="h-4 w-4" />
          Create User
        </Button>
      </div>

      {/* Filters */}
      <div className="flex items-center gap-3 flex-wrap">
        <div className="relative">
          <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-zinc-500" />
          <Input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search name or email…"
            className="pl-8 bg-zinc-800 border-zinc-700 w-56"
          />
        </div>
        <div className="flex gap-1 rounded-lg border border-border/50 bg-zinc-900 p-1">
          {[
            { key: "ALL",         label: "All" },
            { key: "SUPER_ADMIN", label: "Super Admin" },
            { key: "CLIENT_ADMIN",label: "Client Admin" },
            { key: "VIEWER",      label: "Viewer" },
          ].map(({ key, label }) => (
            <button
              key={key}
              onClick={() => setRoleFilter(key)}
              className={cn(
                "rounded px-3 py-1.5 text-xs font-medium transition-colors",
                roleFilter === key ? "bg-zinc-700 text-zinc-100" : "text-zinc-500 hover:text-zinc-300",
              )}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Table */}
      {isLoading ? (
        <div className="space-y-1.5">
          {Array.from({ length: 8 }).map((_, i) => <Skeleton key={i} className="h-14 rounded-lg" />)}
        </div>
      ) : (
        <div className="overflow-hidden rounded-xl border border-border/50">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border/50 bg-zinc-900/50">
                <th className="py-3 px-4 text-left text-xs font-medium text-muted-foreground">User</th>
                <th className="py-3 px-3 text-left text-xs font-medium text-muted-foreground">Role</th>
                <th className="py-3 px-3 text-left text-xs font-medium text-muted-foreground">Client</th>
                <th className="py-3 px-3 text-left text-xs font-medium text-muted-foreground">Status</th>
                <th className="py-3 px-3 text-center text-xs font-medium text-muted-foreground">MFA</th>
                <th className="py-3 px-3 text-left text-xs font-medium text-muted-foreground">Last Login</th>
                <th className="py-3 px-3 text-right text-xs font-medium text-muted-foreground">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/40">
              {users.map((u) => (
                <tr key={u.user_id} className="hover:bg-zinc-800/30 transition-colors group">
                  {/* User */}
                  <td className="py-3 px-4">
                    <div className="flex items-center gap-3">
                      <Avatar className="h-8 w-8 shrink-0">
                        <AvatarFallback className={cn("text-[11px]",
                          u.role === "SUPER_ADMIN" ? "bg-violet-800" : "bg-zinc-800"
                        )}>
                          {initials(u.name)}
                        </AvatarFallback>
                      </Avatar>
                      <div className="min-w-0">
                        <p className="font-medium text-zinc-200 truncate">{u.name}</p>
                        <p className="text-[11px] text-zinc-500 truncate">{u.email}</p>
                      </div>
                    </div>
                  </td>

                  {/* Role */}
                  <td className="py-3 px-3"><RolePill role={u.role} /></td>

                  {/* Client */}
                  <td className="py-3 px-3 text-xs text-zinc-500 max-w-[120px] truncate">
                    {u.client_name ?? (u.role === "SUPER_ADMIN" ? <span className="text-violet-500">Platform</span> : "—")}
                  </td>

                  {/* Status */}
                  <td className="py-3 px-3">
                    <Badge
                      variant={u.status === "ACTIVE" ? "success" : u.status === "SUSPENDED" ? "warning" : "secondary"}
                      className="text-[10px]"
                    >
                      {u.status}
                    </Badge>
                  </td>

                  {/* MFA */}
                  <td className="py-3 px-3 text-center">
                    {u.mfa_enabled ? (
                      <span className="text-green-400 text-[10px] font-medium">ON</span>
                    ) : (
                      <span className="text-zinc-600 text-[10px]">off</span>
                    )}
                  </td>

                  {/* Last login */}
                  <td className="py-3 px-3 text-xs text-zinc-500 whitespace-nowrap">
                    {fmtDatetime(u.last_login)}
                  </td>

                  {/* Actions */}
                  <td className="py-3 px-3 text-right">
                    <div className="flex items-center justify-end gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                      <Button
                        size="sm" variant="ghost" className="h-7 px-2 text-xs"
                        onClick={() => setEditTarget(u)}
                      >
                        <Eye className="h-3.5 w-3.5" />
                      </Button>
                      <Button
                        size="sm" variant="ghost"
                        className={cn("h-7 px-2 text-xs",
                          u.status === "ACTIVE" ? "text-amber-400 hover:text-amber-300" : "text-green-400 hover:text-green-300"
                        )}
                        onClick={() => statusMutation.mutate({
                          id:     u.user_id,
                          status: u.status === "ACTIVE" ? "SUSPENDED" : "ACTIVE",
                        })}
                        disabled={statusMutation.isPending}
                      >
                        {u.status === "ACTIVE" ? "Suspend" : "Activate"}
                      </Button>
                      <Button
                        size="sm" variant="ghost"
                        className="h-7 px-2 text-xs text-zinc-500 hover:text-zinc-300"
                        onClick={() => resetMutation.mutate(u.user_id)}
                        disabled={resetMutation.isPending}
                      >
                        Reset pw
                      </Button>
                    </div>
                  </td>
                </tr>
              ))}
              {users.length === 0 && (
                <tr>
                  <td colSpan={7} className="py-12 text-center">
                    <UserIcon className="mx-auto h-7 w-7 text-zinc-700 mb-2" />
                    <p className="text-sm text-muted-foreground">No users found</p>
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}

      {/* Dialogs */}
      <CreateUserDialog open={showCreate} onClose={() => setShowCreate(false)} />
      <EditUserDialog user={editTarget} onClose={() => setEditTarget(null)} />
    </div>
  );
}
