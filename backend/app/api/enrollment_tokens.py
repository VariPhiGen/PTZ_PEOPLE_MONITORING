"""
Enrollment Token management — authenticated admin endpoints.

Admins create shareable enrollment tokens that allow anyone with the link
to self-enroll their face without needing a system account.

Routes:
    POST   /api/enrollment-tokens            — create a new token (returns URL)
    GET    /api/enrollment-tokens            — list tokens for the current client
    DELETE /api/enrollment-tokens/{token_id} — deactivate a token
"""
from __future__ import annotations

import secrets
import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import text

from app.api._shared import resolve_client_id
from app.deps import CurrentUser, DBSession
from app.middleware.auth import require_permission

router = APIRouter(prefix="/api/enrollment-tokens", tags=["enrollment-tokens"])

_RD  = require_permission("face_embeddings:read")
_WR  = require_permission("face_embeddings:create")
_DEL = require_permission("face_embeddings:delete")


# ── Schemas ───────────────────────────────────────────────────────────────────

class CreateTokenRequest(BaseModel):
    label:        str | None = None      # human-readable name for this link
    role_default: str        = "STUDENT" # default role for enrolled persons
    dataset_id:   str | None = None
    client_id:    str | None = None      # SUPER_ADMIN only
    expires_in_days: int | None = None   # None = never expires
    max_uses:     int | None = None      # None = unlimited


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/", dependencies=[_WR], status_code=201)
async def create_token(
    body:    CreateTokenRequest,
    request: Request,
    db:      DBSession,
    user:    CurrentUser,
) -> dict:
    """
    Create a new enrollment token and return the shareable enrollment URL.
    The URL can be opened on any device with a camera.
    """
    cid = resolve_client_id(request, body.client_id)

    token  = secrets.token_urlsafe(32)   # 43-char URL-safe token
    now    = int(time.time())
    exp    = now + body.expires_in_days * 86400 if body.expires_in_days else None
    tok_id = uuid.uuid4()

    await db.execute(
        text("""
            INSERT INTO enrollment_tokens
                (token_id, client_id, token, label, role_default,
                 dataset_id, expires_at, max_uses, use_count, is_active,
                 created_by, created_at)
            VALUES
                ((:tid)::uuid, (:cid)::uuid, :token, :label, :role,
                 (:did)::uuid, :exp, :max_uses, 0, TRUE,
                 (:uid)::uuid, :now)
        """),
        {
            "tid":      str(tok_id),
            "cid":      str(cid),
            "token":    token,
            "label":    body.label,
            "role":     body.role_default,
            "did":      body.dataset_id,
            "exp":      exp,
            "max_uses": body.max_uses,
            "uid":      str(user.get("user_id") or ""),
            "now":      now,
        },
    )
    await db.commit()

    # Build enrollment URL from the request host
    scheme = request.headers.get("X-Forwarded-Proto", request.url.scheme)
    host   = request.headers.get("X-Forwarded-Host", request.url.netloc)
    # Strip the API port and substitute the dashboard port if needed
    # The enrollment page is served by the Next.js dashboard, not the backend.
    # We infer the dashboard host from the DASHBOARD_URL setting if available,
    # otherwise just use the current host.
    settings    = getattr(request.app.state, "settings", None)
    dashboard_url = getattr(settings, "DASHBOARD_URL", None) if settings else None
    if dashboard_url:
        enroll_base = dashboard_url.rstrip("/")
    else:
        enroll_base = f"{scheme}://{host}"

    enroll_url = f"{enroll_base}/enroll/{token}"

    return {
        "token_id":   str(tok_id),
        "token":      token,
        "enroll_url": enroll_url,
        "label":      body.label,
        "role_default": body.role_default,
        "expires_at": exp,
        "max_uses":   body.max_uses,
        "created_at": now,
    }


@router.get("/", dependencies=[_RD])
async def list_tokens(
    request: Request,
    db:      DBSession,
) -> list[dict]:
    """List all enrollment tokens for the current client."""
    cid = resolve_client_id(request, None, require=False)

    rows = (
        await db.execute(
            text("""
                SELECT token_id::text, token, label, role_default,
                       dataset_id::text, expires_at, max_uses, use_count,
                       is_active, created_at
                FROM enrollment_tokens
                WHERE client_id = (:cid)::uuid
                ORDER BY created_at DESC
            """),
            {"cid": str(cid)},
        )
    ).mappings().all()

    now = int(time.time())

    result = []
    for r in rows:
        r = dict(r)
        r["is_expired"] = bool(r["expires_at"] and now > r["expires_at"])
        r["is_exhausted"] = bool(r["max_uses"] and r["use_count"] >= r["max_uses"])
        result.append(r)

    return result


@router.delete("/{token_id}", dependencies=[_DEL])
async def deactivate_token(
    token_id: str,
    request:  Request,
    db:       DBSession,
) -> dict:
    """Deactivate an enrollment token so it can no longer be used."""
    cid = resolve_client_id(request, None, require=False)

    result = await db.execute(
        text("""
            UPDATE enrollment_tokens
            SET is_active = FALSE
            WHERE token_id = (:tid)::uuid AND client_id = (:cid)::uuid
            RETURNING token_id::text
        """),
        {"tid": token_id, "cid": str(cid)},
    )
    row = result.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Token not found")

    await db.commit()
    return {"deactivated": True, "token_id": token_id}
