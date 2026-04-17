from __future__ import annotations

import logging
from typing import Callable

from fastapi import Depends, HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.models.users import UserRole
from app.utils.jwt import TokenError, decode_token

logger = logging.getLogger(__name__)

# Public endpoints that never require a token
_PUBLIC_PREFIXES = (
    "/api/auth/login",
    "/api/auth/refresh",
    "/api/auth/forgot-password",
    "/api/auth/reset-password",
    "/api/health",
    "/api/public/",     # public self-enrollment — token-validated, no JWT
    "/docs",
    "/openapi.json",
    "/redoc",
)


def _is_public(path: str) -> bool:
    return any(path.startswith(p) for p in _PUBLIC_PREFIXES)


def _extract_bearer(request: Request) -> str | None:
    header = request.headers.get("Authorization", "")
    if header.startswith("Bearer "):
        return header[7:]
    return None


def _set_unauthenticated(request: Request) -> None:
    request.state.authenticated = False
    request.state.user_id = None
    request.state.email = None
    request.state.name = None
    request.state.role = None
    request.state.client_id = None
    request.state.client_slug = None
    request.state.permissions = []


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Extracts and verifies the JWT from Authorization: Bearer <token>.
    Populates request.state with user context for downstream use.
    Does NOT reject unauthenticated requests — that is the job of
    require_permission / require_role dependencies on specific routes.
    """

    async def dispatch(self, request: Request, call_next):
        token = _extract_bearer(request)
        if not token:
            _set_unauthenticated(request)
            return await call_next(request)

        settings = request.app.state.settings
        try:
            payload = decode_token(token, settings)
        except TokenError as exc:
            _set_unauthenticated(request)
            if not _is_public(request.url.path):
                return JSONResponse(
                    status_code=401,
                    content={"detail": f"Invalid token: {exc}"},
                )
            return await call_next(request)

        if payload.get("type") != "access":
            _set_unauthenticated(request)
            return JSONResponse(
                status_code=401,
                content={"detail": "Refresh tokens cannot be used as bearer tokens"},
            )

        request.state.authenticated = True
        request.state.user_id = payload.get("user_id")
        request.state.email = payload.get("email")
        request.state.name = payload.get("name")
        request.state.role = payload.get("role")
        request.state.client_id = payload.get("client_id")
        request.state.client_slug = payload.get("client_slug")
        request.state.permissions = payload.get("permissions", [])

        return await call_next(request)


# ── Permission / role guards ───────────────────────────────────────────────────

def require_permission(*perms: str) -> Depends:
    """
    FastAPI dependency. Raises 403 if the authenticated user lacks any of the
    listed permissions. SUPER_ADMIN bypasses all permission checks.

    Usage:
        @router.get("/cameras", dependencies=[require_permission("cameras:read")])
    """
    async def _guard(request: Request) -> None:
        if not getattr(request.state, "authenticated", False):
            raise HTTPException(status_code=401, detail="Not authenticated")
        role = request.state.role
        if role == UserRole.SUPER_ADMIN:
            return
        user_perms: list[str] = request.state.permissions
        missing = [p for p in perms if p not in user_perms]
        if missing:
            raise HTTPException(
                status_code=403,
                detail=f"Missing permissions: {', '.join(missing)}",
            )

    return Depends(_guard)


def require_role(*roles: str) -> Depends:
    """
    FastAPI dependency. Raises 403 if the user's role is not in `roles`.

    Usage:
        @router.delete("/clients/{id}", dependencies=[require_role("SUPER_ADMIN")])
    """
    async def _guard(request: Request) -> None:
        if not getattr(request.state, "authenticated", False):
            raise HTTPException(status_code=401, detail="Not authenticated")
        if request.state.role not in roles:
            raise HTTPException(
                status_code=403,
                detail=f"Requires role: {', '.join(roles)}",
            )

    return Depends(_guard)
