"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Filter, Plus, Search } from "lucide-react";
import { toast } from "sonner";
import { adminApi } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { cn, fmtDatetime } from "@/lib/utils";

interface AdminUser {
  user_id:    string;
  email:      string;
  name:       string;
  role:       string;
  client_id:  string | null;
  status:     string;
  last_login: string | null;
}

const ROLE_COLOR: Record<string, string> = {
  SUPER_ADMIN:  "bg-violet-500/15 text-violet-400",
  CLIENT_ADMIN: "bg-blue-500/15 text-blue-400",
  VIEWER:       "bg-zinc-500/15 text-zinc-400",
};

export default function AdminUsersPage() {
  const [search, setSearch] = useState("");
  const [role,   setRole]   = useState<string>("ALL");
  const qc = useQueryClient();

  const { data, isLoading } = useQuery({
    queryKey: ["admin-users", search, role],
    queryFn:  () =>
      adminApi.users({
        q:     search || undefined,
        role:  role !== "ALL" ? role : undefined,
        limit: 100,
      }).then((r) => r.data),
  });

  const users: AdminUser[] = (data as { items?: AdminUser[] })?.items ?? [];

  const statusMutation = useMutation({
    mutationFn: ({ id, status }: { id: string; status: string }) =>
      adminApi.userStatus(id, { status }),
    onSuccess: () => {
      toast.success("User status updated");
      qc.invalidateQueries({ queryKey: ["admin-users"] });
    },
    onError: () => toast.error("Update failed"),
  });

  const resetMutation = useMutation({
    mutationFn: (id: string) => adminApi.resetPassword(id),
    onSuccess:  () => toast.success("Password reset email sent"),
    onError:    () => toast.error("Reset failed"),
  });

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold text-zinc-50">Users</h1>
          <p className="text-sm text-muted-foreground mt-0.5">{users.length} accounts</p>
        </div>
        <Button size="sm">
          <Plus className="h-4 w-4" />
          New User
        </Button>
      </div>

      {/* Filters */}
      <div className="flex items-center gap-3">
        <div className="relative max-w-xs">
          <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-zinc-500" />
          <Input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search users…"
            className="pl-8 bg-zinc-800 border-zinc-700"
          />
        </div>
        <div className="flex gap-1 rounded-lg border border-border/50 bg-zinc-900 p-1">
          {["ALL", "SUPER_ADMIN", "CLIENT_ADMIN", "VIEWER"].map((r) => (
            <button
              key={r}
              onClick={() => setRole(r)}
              className={cn(
                "rounded px-3 py-1 text-xs font-medium transition-colors",
                role === r ? "bg-zinc-700 text-zinc-100" : "text-zinc-500 hover:text-zinc-300",
              )}
            >
              {r === "ALL" ? "All" : r.replace("_", " ")}
            </button>
          ))}
        </div>
      </div>

      {isLoading ? (
        <div className="space-y-1.5">
          {Array.from({ length: 8 }).map((_, i) => <Skeleton key={i} className="h-14 rounded-lg" />)}
        </div>
      ) : (
        <div className="overflow-hidden rounded-xl border border-border/50">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border/50 bg-zinc-900/50">
                <th className="py-2.5 px-4 text-left text-xs font-medium text-muted-foreground">User</th>
                <th className="py-2.5 px-3 text-left text-xs font-medium text-muted-foreground">Role</th>
                <th className="py-2.5 px-3 text-left text-xs font-medium text-muted-foreground">Status</th>
                <th className="py-2.5 px-3 text-left text-xs font-medium text-muted-foreground">Last Login</th>
                <th className="py-2.5 px-3 text-right text-xs font-medium text-muted-foreground">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/40">
              {users.map((u) => (
                <tr key={u.user_id} className="hover:bg-zinc-800/30 transition-colors">
                  <td className="py-2.5 px-4">
                    <div className="flex items-center gap-2.5">
                      <Avatar className="h-7 w-7">
                        <AvatarFallback className="text-[10px]">
                          {u.name.split(" ").map((w) => w[0]).join("").slice(0, 2).toUpperCase()}
                        </AvatarFallback>
                      </Avatar>
                      <div>
                        <p className="font-medium text-zinc-200 text-xs">{u.name}</p>
                        <p className="text-zinc-500 text-[10px]">{u.email}</p>
                      </div>
                    </div>
                  </td>
                  <td className="py-2.5 px-3">
                    <span className={cn("text-[10px] font-medium rounded-full px-2 py-0.5", ROLE_COLOR[u.role])}>
                      {u.role.replace("_", " ")}
                    </span>
                  </td>
                  <td className="py-2.5 px-3">
                    <Badge
                      variant={u.status === "ACTIVE" ? "success" : "secondary"}
                      className="text-[10px]"
                    >
                      {u.status}
                    </Badge>
                  </td>
                  <td className="py-2.5 px-3 text-xs text-zinc-500">{fmtDatetime(u.last_login)}</td>
                  <td className="py-2.5 px-3 text-right">
                    <div className="flex items-center justify-end gap-1">
                      <Button
                        size="sm"
                        variant="ghost"
                        className="h-7 text-xs"
                        onClick={() =>
                          statusMutation.mutate({
                            id:     u.user_id,
                            status: u.status === "ACTIVE" ? "SUSPENDED" : "ACTIVE",
                          })
                        }
                      >
                        {u.status === "ACTIVE" ? "Suspend" : "Activate"}
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        className="h-7 text-xs text-amber-400 hover:text-amber-300"
                        onClick={() => resetMutation.mutate(u.user_id)}
                      >
                        Reset pw
                      </Button>
                    </div>
                  </td>
                </tr>
              ))}
              {users.length === 0 && (
                <tr><td colSpan={5} className="py-10 text-center text-sm text-muted-foreground">No users</td></tr>
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
