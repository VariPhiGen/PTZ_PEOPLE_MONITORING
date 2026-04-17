"use client";

import { useCallback, useEffect, useState } from "react";
import { getPayload, isAuthenticated, clearTokens } from "@/lib/auth";
import type { JWTPayload, Role } from "@/types";

export interface AuthState {
  user:            JWTPayload | null;
  role:            Role | null;
  clientId:        string | null;
  isAuthenticated: boolean;
  isSuperAdmin:    boolean;
  isClientAdmin:   boolean;
  isViewer:        boolean;
  logout:          () => void;
}

export function useAuth(): AuthState {
  const [user, setUser] = useState<JWTPayload | null>(null);

  useEffect(() => {
    if (isAuthenticated()) setUser(getPayload());
  }, []);

  const logout = useCallback(() => {
    clearTokens();
    window.location.href = "/auth/login";
  }, []);

  const role = user?.role ?? null;

  return {
    user,
    role,
    clientId:        user?.client_id ?? null,
    isAuthenticated: !!user,
    isSuperAdmin:    role === "SUPER_ADMIN",
    isClientAdmin:   role === "CLIENT_ADMIN",
    isViewer:        role === "VIEWER",
    logout,
  };
}
