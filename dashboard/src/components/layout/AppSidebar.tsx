"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  BarChart2,
  CalendarCheck2,
  CalendarDays,
  Camera,
  ChevronRight,
  Database,
  ExternalLink,
  Link2,
  LogOut,
  Network,
  Search,
  Settings,
  Shield,
  UserCheck,
  Users,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Separator } from "@/components/ui/separator";
import { useAuth } from "@/hooks/useAuth";
import type { Client } from "@/types";

// ── Nav item definitions ─────────────────────────────────────────────────────

interface NavItem {
  label:    string;
  href:     string;
  icon:     React.ElementType;
  roles?:   ("SUPER_ADMIN" | "CLIENT_ADMIN" | "VIEWER")[];
  external?: boolean;
}

const NAV_ITEMS: NavItem[] = [
  { label: "Attendance",    href: "/attendance",  icon: CalendarCheck2 },
  { label: "Analytics",     href: "/analytics",   icon: BarChart2 },
  { label: "Search",        href: "/search",       icon: Search },
  // CLIENT_ADMIN+ items
  { label: "Cameras",       href: "/cameras",      icon: Camera,      roles: ["SUPER_ADMIN", "CLIENT_ADMIN"] },
  { label: "Face Datasets", href: "/datasets",     icon: Database,    roles: ["SUPER_ADMIN", "CLIENT_ADMIN"] },
  { label: "Enrollment",    href: "/enrollment",   icon: UserCheck,   roles: ["SUPER_ADMIN", "CLIENT_ADMIN"] },
  { label: "Enroll Links",  href: "/enrollment-tokens", icon: Link2, roles: ["SUPER_ADMIN", "CLIENT_ADMIN"] },
  { label: "Timetables",    href: "/timetables",   icon: CalendarDays, roles: ["SUPER_ADMIN", "CLIENT_ADMIN"] },
  { label: "Settings",      href: "/settings",     icon: Settings,    roles: ["SUPER_ADMIN", "CLIENT_ADMIN"] },
  { label: "Cluster",       href: "/cluster",      icon: Network,     roles: ["SUPER_ADMIN", "CLIENT_ADMIN"] },
  // SUPER_ADMIN shortcut to admin panel
  { label: "Admin Panel",   href: "/admin/clients",icon: Shield,      roles: ["SUPER_ADMIN"], external: false },
];

// ── Props ─────────────────────────────────────────────────────────────────────

interface AppSidebarProps {
  client?: Pick<Client, "name" | "logo_url"> | null;
}

// ── Component ─────────────────────────────────────────────────────────────────

export function AppSidebar({ client }: AppSidebarProps) {
  const pathname = usePathname();
  const { user, role, logout } = useAuth();

  const visibleItems = NAV_ITEMS.filter(
    (item) => !item.roles || (role && item.roles.includes(role)),
  );

  const initials = user?.name
    ? user.name.split(" ").map((w) => w[0]).join("").slice(0, 2).toUpperCase()
    : "?";

  return (
    <aside className="fixed inset-y-0 left-0 z-40 flex w-60 flex-col border-r border-sidebar-border bg-sidebar">
      {/* ── Brand / client header ─────────────────────────────── */}
      <div className="flex h-14 items-center gap-2 px-3 border-b border-sidebar-border">
        {/* Platform logo — always shown */}
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src="/Logowhite.png"
          alt="VGI ACFR"
          className="h-8 w-auto object-contain shrink-0"
        />
        {/* Tenant override: show client name when logged in as CLIENT_ADMIN/VIEWER */}
        {client?.name && role !== "SUPER_ADMIN" && (
          <div className="flex flex-col min-w-0 border-l border-sidebar-border pl-2 ml-1">
            <span className="truncate text-xs font-medium text-sidebar-accent-foreground leading-tight">
              {client.name}
            </span>
            <span className="text-[10px] text-sidebar-muted uppercase tracking-wide">
              {role === "CLIENT_ADMIN" ? "Admin" : "Viewer"}
            </span>
          </div>
        )}
        {(!client?.name || role === "SUPER_ADMIN") && (
          <span className="text-[10px] text-sidebar-muted uppercase tracking-wide ml-auto">
            {role === "SUPER_ADMIN" ? "Super Admin" : ""}
          </span>
        )}
      </div>

      {/* ── Navigation ───────────────────────────────────────── */}
      <nav className="flex-1 overflow-y-auto px-2 py-3 space-y-0.5">
        {visibleItems.map((item) => {
          const Icon    = item.icon;
          const active  = pathname === item.href || pathname.startsWith(item.href + "/");
          const isAdmin = item.href.startsWith("/admin");

          if (isAdmin && role !== "SUPER_ADMIN") return null;

          // Separator before admin link
          const showDivider = item.href === "/admin/clients";

          return (
            <div key={item.href}>
              {showDivider && <Separator className="my-2 bg-sidebar-border/60" />}
              <Link
                href={item.href}
                className={cn(
                  "sidebar-nav-item",
                  active && "active",
                  isAdmin && "text-violet-400 hover:text-violet-300",
                )}
              >
                <Icon className="h-4 w-4 shrink-0" />
                <span className="flex-1">{item.label}</span>
                {item.external && <ExternalLink className="h-3 w-3 opacity-50" />}
                {active && !item.external && (
                  <ChevronRight className="h-3 w-3 opacity-40" />
                )}
              </Link>
            </div>
          );
        })}
      </nav>

      {/* ── User footer ──────────────────────────────────────── */}
      <div className="border-t border-sidebar-border p-3">
        <div className="flex items-center gap-3">
          <Avatar className="h-8 w-8">
            <AvatarImage src="" alt={user?.name ?? ""} />
            <AvatarFallback className="text-[11px]">{initials}</AvatarFallback>
          </Avatar>
          <div className="flex flex-col min-w-0 flex-1">
            <span className="truncate text-xs font-medium text-sidebar-accent-foreground">
              {user?.name ?? "—"}
            </span>
            <span className="truncate text-[10px] text-sidebar-muted">
              {user?.email ?? ""}
            </span>
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
