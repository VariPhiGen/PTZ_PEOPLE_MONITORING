"use client";
import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { isAuthenticated, getRole, homeForRole } from "@/lib/auth";

export default function RootPage() {
  const router = useRouter();
  useEffect(() => {
    if (isAuthenticated()) {
      const role = getRole();
      router.replace(role ? homeForRole(role) : "/attendance");
    } else {
      router.replace("/auth/login");
    }
  }, [router]);
  return null;
}
