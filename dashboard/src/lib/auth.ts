import Cookies from "js-cookie";
import { decodeJwt } from "@/lib/utils";
import type { JWTPayload, Role } from "@/types";

export const TOKEN_KEY   = "acas_token";
export const REFRESH_KEY = "acas_refresh";

export function getToken(): string | undefined {
  return Cookies.get(TOKEN_KEY);
}

export function getRefreshToken(): string | undefined {
  return Cookies.get(REFRESH_KEY);
}

export function setTokens(accessToken: string, refreshToken: string): void {
  Cookies.set(TOKEN_KEY,   accessToken,  { expires: 1,  sameSite: "strict" });
  Cookies.set(REFRESH_KEY, refreshToken, { expires: 7,  sameSite: "strict" });
}

export function clearTokens(): void {
  Cookies.remove(TOKEN_KEY);
  Cookies.remove(REFRESH_KEY);
}

export function getPayload(): JWTPayload | null {
  const token = getToken();
  if (!token) return null;
  return decodeJwt<JWTPayload>(token);
}

export function isAuthenticated(): boolean {
  const token = getToken();
  if (!token) return false;
  const payload = decodeJwt<{ exp?: number }>(token);
  if (!payload?.exp) return false;
  return payload.exp > Math.floor(Date.now() / 1000);
}

export function getRole(): Role | null {
  return getPayload()?.role ?? null;
}

/** Redirect target after successful login, based on role */
export function homeForRole(role: Role): string {
  switch (role) {
    case "SUPER_ADMIN":  return "/clients";
    case "CLIENT_ADMIN": return "/cameras";
    case "VIEWER":       return "/attendance";
  }
}
