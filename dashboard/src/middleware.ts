import { NextRequest, NextResponse } from "next/server";

const PUBLIC_PATHS  = ["/auth/login", "/auth/forgot-password", "/auth/reset-password", "/enroll"];
// All routes that require SUPER_ADMIN
const ADMIN_PATHS = ["/admin", "/clients", "/users", "/platform-analytics"];
const ADMIN_PATTERN = (p: string) => ADMIN_PATHS.some((a) => p === a || p.startsWith(a + "/"));

function decodeJwtRole(token: string): string | null {
  try {
    const b64 = token.split(".")[1].replace(/-/g, "+").replace(/_/g, "/");
    const json = JSON.parse(atob(b64));
    return (json as { role?: string }).role ?? null;
  } catch {
    return null;
  }
}

function isExpired(token: string): boolean {
  try {
    const b64  = token.split(".")[1].replace(/-/g, "+").replace(/_/g, "/");
    const json = JSON.parse(atob(b64)) as { exp?: number };
    return typeof json.exp === "number" && json.exp < Math.floor(Date.now() / 1000);
  } catch {
    return true;
  }
}

export function middleware(request: NextRequest): NextResponse {
  const { pathname } = request.nextUrl;

  // Static assets, API routes and Next internals pass through
  if (
    pathname.startsWith("/_next") ||
    pathname.startsWith("/api")   ||
    pathname.includes(".")
  ) {
    return NextResponse.next();
  }

  // Public pages always allowed — enrollment pages skip the auth-redirect-to-home logic
  if (PUBLIC_PATHS.some((p) => pathname.startsWith(p))) {
    if (pathname.startsWith("/enroll")) {
      return NextResponse.next();   // never redirect enrollment links regardless of auth state
    }
    // Auth pages: if already logged in, bounce to home
    const token = request.cookies.get("acas_token")?.value;
    if (token && !isExpired(token)) {
      const role = decodeJwtRole(token);
      const dest = role === "SUPER_ADMIN" ? "/admin" : role === "CLIENT_ADMIN" ? "/cameras" : "/attendance";
      return NextResponse.redirect(new URL(dest, request.url));
    }
    return NextResponse.next();
  }

  const token = request.cookies.get("acas_token")?.value;

  // No token → login
  if (!token || isExpired(token)) {
    const loginUrl = new URL("/auth/login", request.url);
    loginUrl.searchParams.set("redirect", pathname);
    return NextResponse.redirect(loginUrl);
  }

  const role = decodeJwtRole(token);
  if (!role) {
    return NextResponse.redirect(new URL("/auth/login", request.url));
  }

  // Admin routes require SUPER_ADMIN
  if (ADMIN_PATTERN(pathname) && role !== "SUPER_ADMIN") {
    return NextResponse.redirect(
      new URL(role === "CLIENT_ADMIN" ? "/cameras" : "/attendance", request.url)
    );
  }

  // VIEWER cannot access write-action routes
  const VIEWER_BLOCKED = ["/cameras/new", "/enrollment/enroll", "/settings"];
  if (role === "VIEWER" && VIEWER_BLOCKED.some((p) => pathname.startsWith(p))) {
    return NextResponse.redirect(new URL("/attendance", request.url));
  }

  return NextResponse.next();
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
