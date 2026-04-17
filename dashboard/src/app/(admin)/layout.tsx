"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { AdminSidebar } from "@/components/layout/AdminSidebar";
import { isAuthenticated, getRole } from "@/lib/auth";

export default function AdminLayout({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const [ready, setReady] = useState(false);

  useEffect(() => {
    if (!isAuthenticated()) {
      router.replace("/auth/login");
    } else if (getRole() !== "SUPER_ADMIN") {
      router.replace("/attendance");
    } else {
      setReady(true);
    }
  }, [router]);

  if (!ready) return null;

  return (
    <div className="flex min-h-screen">
      <AdminSidebar />
      <main className="ml-60 flex-1 min-h-screen overflow-x-hidden">
        <div className="p-6">{children}</div>
      </main>
    </div>
  );
}
