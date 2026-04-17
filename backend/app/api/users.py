from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, field_validator
from sqlalchemy import func, select

from app.deps import CurrentUser, DBSession
from app.middleware.auth import require_permission, require_role
from app.models.users import User, UserRole, UserStatus
from app.utils.security import validate_password, hash_password

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# CLIENT_ADMIN only — viewers are always scoped to the caller's own client_id.
_CA = require_role("CLIENT_ADMIN", "SUPER_ADMIN")
router = APIRouter(prefix="/api/users", tags=["users"], dependencies=[_CA])


def _now() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def _viewer_dict(u: User) -> dict:
    return {
        "user_id": str(u.user_id),
        "email": u.email,
        "name": u.name,
        "role": u.role,
        "client_id": str(u.client_id) if u.client_id else None,
        "status": u.status,
        "avatar_url": u.avatar_url,
        "last_login": u.last_login,
        "created_at": u.created_at,
    }


# ── Schemas ────────────────────────────────────────────────────────────────────

class CreateViewerBody(BaseModel):
    email: str
    password: str
    name: str

    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str:
        if not _EMAIL_RE.match(v):
            raise ValueError("Invalid email")
        return v.lower().strip()

    @field_validator("password")
    @classmethod
    def _pw(cls, v: str) -> str:
        validate_password(v)
        return v


class UpdateViewerBody(BaseModel):
    name: str | None = None
    avatar_url: str | None = None

    @field_validator("name")
    @classmethod
    def _name(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            raise ValueError("name cannot be blank")
        return v


class ViewerStatusBody(BaseModel):
    status: str

    @field_validator("status")
    @classmethod
    def _valid(cls, v: str) -> str:
        if v not in {UserStatus.ACTIVE, UserStatus.SUSPENDED}:
            raise ValueError("status must be ACTIVE or SUSPENDED")
        return v


# ── Helpers ────────────────────────────────────────────────────────────────────

def _resolve_client_id(current_user: dict) -> str:
    """Return the caller's client_id; SUPER_ADMIN must pass it as a query param instead."""
    cid = current_user.get("client_id")
    if not cid:
        raise HTTPException(422, "client_id required (super admins should use /api/admin/users)")
    return str(cid)


async def _get_viewer_or_404(db, viewer_id: str, client_id: str) -> User:
    result = await db.execute(
        select(User).where(
            User.user_id == viewer_id,
            User.client_id == client_id,
            User.role == UserRole.VIEWER,
        )
    )
    viewer = result.scalar_one_or_none()
    if not viewer:
        raise HTTPException(404, "Viewer not found")
    return viewer


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/viewers", status_code=201)
async def create_viewer(
    body: CreateViewerBody,
    db: DBSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    client_id = _resolve_client_id(current_user)

    dup = await db.execute(select(User).where(User.email == body.email))
    if dup.scalar_one_or_none():
        raise HTTPException(409, "Email already registered")

    now = _now()
    viewer = User(
        email=body.email,
        password_hash=hash_password(body.password),
        name=body.name,
        role=UserRole.VIEWER,
        client_id=UUID(client_id),
        status=UserStatus.ACTIVE,
        created_by=UUID(current_user["user_id"]),
        created_at=now,
        updated_at=now,
    )
    db.add(viewer)
    await db.flush()
    return _viewer_dict(viewer)


@router.get("/viewers")
async def list_viewers(
    db: DBSession,
    current_user: CurrentUser,
    status: str | None = Query(None),
    limit: int = Query(100, le=500),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    client_id = _resolve_client_id(current_user)

    q = (
        select(User)
        .where(User.client_id == client_id, User.role == UserRole.VIEWER)
    )
    if status:
        q = q.where(User.status == status)
    q = q.order_by(User.created_at.desc()).limit(limit).offset(offset)

    rows = await db.execute(q)
    viewers = rows.scalars().all()

    total_q = select(func.count()).where(
        User.client_id == client_id, User.role == UserRole.VIEWER
    )
    if status:
        total_q = total_q.where(User.status == status)
    total = (await db.execute(total_q)).scalar_one()

    return {"total": total, "viewers": [_viewer_dict(v) for v in viewers]}


@router.put("/viewers/{viewer_id}")
async def update_viewer(
    viewer_id: str,
    body: UpdateViewerBody,
    db: DBSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    client_id = _resolve_client_id(current_user)
    viewer = await _get_viewer_or_404(db, viewer_id, client_id)

    if body.name is not None:
        viewer.name = body.name.strip()
    if body.avatar_url is not None:
        viewer.avatar_url = body.avatar_url
    viewer.updated_at = _now()

    return _viewer_dict(viewer)


@router.put("/viewers/{viewer_id}/status")
async def set_viewer_status(
    viewer_id: str,
    body: ViewerStatusBody,
    db: DBSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    client_id = _resolve_client_id(current_user)
    viewer = await _get_viewer_or_404(db, viewer_id, client_id)

    viewer.status = body.status
    viewer.updated_at = _now()
    return {"user_id": viewer_id, "status": viewer.status}
