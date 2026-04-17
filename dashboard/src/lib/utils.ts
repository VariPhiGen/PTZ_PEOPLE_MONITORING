import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

/** Format ISO datetime as "dd MMM, HH:mm" */
export function fmtDatetime(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleString("en-GB", {
    day:    "2-digit",
    month:  "short",
    hour:   "2-digit",
    minute: "2-digit",
    hour12: false,
  });
}

/** Format seconds as "Xh Ym" or "Ym Zs" */
export function fmtDuration(seconds: number): string {
  if (seconds < 60)   return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return `${h}h ${m}m`;
}

/** Attendance status → human label */
export const ATTENDANCE_LABEL: Record<string, string> = {
  P:  "Present",
  L:  "Late",
  EE: "Early Exit",
  A:  "Absent",
  ND: "No Data",
  EX: "Excused",
};

/** Attendance status → Tailwind colour class */
export const ATTENDANCE_COLOR: Record<string, string> = {
  P:  "text-green-400 bg-green-400/10",
  L:  "text-yellow-400 bg-yellow-400/10",
  EE: "text-orange-400 bg-orange-400/10",
  A:  "text-red-400 bg-red-400/10",
  ND: "text-zinc-400 bg-zinc-400/10",
  EX: "text-blue-400 bg-blue-400/10",
};

export const CAMERA_STATUS_COLOR: Record<string, string> = {
  ONLINE:   "text-green-400",
  OFFLINE:  "text-zinc-500",
  DEGRADED: "text-amber-400",
};

export const NODE_STATUS_COLOR: Record<string, string> = {
  ONLINE:   "bg-green-500",
  OFFLINE:  "bg-zinc-500",
  DRAINING: "bg-amber-500",
};

/** Decode a JWT payload without verification (client-side display only). */
export function decodeJwt<T = Record<string, unknown>>(token: string): T | null {
  try {
    const b64 = token.split(".")[1].replace(/-/g, "+").replace(/_/g, "/");
    return JSON.parse(atob(b64)) as T;
  } catch {
    return null;
  }
}
