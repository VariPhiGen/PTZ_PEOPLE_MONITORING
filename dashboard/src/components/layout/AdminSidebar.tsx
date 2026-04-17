"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import {
  ArrowLeft,
  BarChart3,
  Building2,
  LogOut,
  Server,
  Settings,
  Users,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Separator } from "@/components/ui/separator";
import { useAuth } from "@/hooks/useAuth";

const ADMIN_NAV = [
  { label: "Clients",            href: "/clients",              icon: Building2  },
  { label: "Users",              href: "/users",                icon: Users      },
  { label: "GPU Nodes",          href: "/admin/nodes",          icon: Server     },
  { label: "Platform Analytics", href: "/platform-analytics",   icon: BarChart3  },
  { label: "Platform Settings",  href: "/admin/settings",       icon: Settings   },
];

export function AdminSidebar() {
  const pathname = usePathname();
  const { user, logout } = useAuth();

  const initials = user?.name
    ? user.name.split(" ").map((w) => w[0]).join("").slice(0, 2).toUpperCase()
    : "SA";

  return (
    <aside className="fixed inset-y-0 left-0 z-40 flex w-60 flex-col border-r border-sidebar-border bg-sidebar">
      {/* ── Brand ───────────────────────────────────────────── */}
      <div className="flex h-14 items-center gap-2 px-3 border-b border-sidebar-border">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src="/vnet.svg"
          alt="VGI ACFR"
          className="h-8 w-auto object-contain shrink-0"
        />
        <span className="text-[10px] text-violet-400 uppercase tracking-wide ml-auto">
          Super Admin
        </span>
      </div>

      {/* ── Back to dashboard ────────────────────────────────── */}
      <div className="px-2 pt-3">
        <Link
          href="/attendance"
          className="sidebar-nav-item text-zinc-500 hover:text-zinc-300"
        >
          <ArrowLeft className="h-4 w-4" />
          <span>Back to Dashboard</span>
        </Link>
        <Separator className="my-2 bg-sidebar-border/60" />
      </div>

      {/* ── Nav ─────────────────────────────────────────────── */}
      <nav className="flex-1 overflow-y-auto px-2 space-y-0.5">
        {ADMIN_NAV.map((item) => {
          const Icon   = item.icon;
          const active = pathname === item.href || pathname.startsWith(item.href + "/");
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn("sidebar-nav-item", active && "active")}
            >
              <Icon className="h-4 w-4 shrink-0" />
              {item.label}
            </Link>
          );
        })}
      </nav>

      {/* ── User footer ─────────────────────────────────────── */}
      <div className="border-t border-sidebar-border p-3">
        <div className="flex items-center gap-3">
          <Avatar className="h-8 w-8">
            <AvatarFallback className="bg-violet-700 text-[11px]">{initials}</AvatarFallback>
          </Avatar>
          <div className="flex flex-col min-w-0 flex-1">
            <span className="truncate text-xs font-medium text-sidebar-accent-foreground">
              {user?.name ?? "Super Admin"}
            </span>
            <span className="truncate text-[10px] text-sidebar-muted">{user?.email ?? ""}</span>
          </div>
          <button
            onClick={logout}
            title="Sign out"
            className="rounded p-1 text-sidebar-muted hover:text-sidebar-accent-foreground hover:bg-sidebar-accent transition-colors"
          >
            <LogOut className="h-4 w-4" />
          </button>
        </div>
      </div>
    </aside>
  );
}
