"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { AppSidebar } from "@/components/layout/AppSidebar";
import { isAuthenticated, getPayload } from "@/lib/auth";
import { api } from "@/lib/api";
import type { Client } from "@/types";

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const router   = useRouter();
  const [ready, setReady] = useState(false);

  useEffect(() => {
    if (!isAuthenticated()) {
      router.replace("/auth/login");
    } else {
      setReady(true);
    }
  }, [router]);

  const payload = getPayload();

  // Fetch client info for CLIENT_ADMIN / VIEWER via the non-admin auth endpoint
  const { data: client } = useQuery<Client>({
    queryKey: ["my-client", payload?.client_id],
    queryFn:  () =>
      api.get("/api/auth/my-client").then((r) => r.data),
    enabled:  !!payload?.client_id && payload.role !== "SUPER_ADMIN",
    staleTime: 5 * 60 * 1000,
  });

  if (!ready) return null;

  return (
    <div className="flex min-h-screen">
      <AppSidebar client={client ?? null} />
      {/* Main area with left margin = sidebar width */}
      <main className="ml-60 flex-1 min-h-screen overflow-x-hidden">
        <div className="p-6">{children}</div>
      </main>
    </div>
  );
}
