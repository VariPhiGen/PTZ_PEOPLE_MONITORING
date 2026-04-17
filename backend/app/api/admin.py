from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, field_validator
from sqlalchemy import func, select, update
from sqlalchemy.orm import selectinload

from app.deps import CurrentUser, DBSession
from app.middleware.auth import require_role
from app.models.cameras import Camera, CameraStatus
from app.models.client_node_assignments import ClientNodeAssignment
from app.models.clients import Client, ClientStatus
from app.models.persons import Person
from app.models.sessions import Session, SyncStatus
from app.models.users import User, UserRole, UserStatus
from app.services.auth_service import AuthError, AuthService
from app.utils.security import hash_password, validate_password

_SA = require_role("SUPER_ADMIN")
router = APIRouter(prefix="/api/admin", tags=["admin"], dependencies=[_SA])

# ── Shared helpers ─────────────────────────────────────────────────────────────

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9\-]{0,98}[a-z0-9]?$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _now_epoch() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def _client_dict(c: Client) -> dict:
    return {
        "client_id": str(c.client_id),
        "name": c.name,
        "slug": c.slug,
        "logo_url": c.logo_url,
        "contact_name": c.contact_name,
        "contact_email": c.contact_email,
        "contact_phone": c.contact_phone,
        "address": c.address,
        "status": c.status,
        "max_cameras": c.max_cameras,
        "max_persons": c.max_persons,
        "settings": c.settings,
        "created_at": c.created_at,
        "updated_at": c.updated_at,
    }


def _user_dict(u: User) -> dict:
    return {
        "user_id": str(u.user_id),
        "email": u.email,
        "name": u.name,
        "role": u.role,
        "client_id": str(u.client_id) if u.client_id else None,
        "status": u.status,
        "avatar_url": u.avatar_url,
        "mfa_enabled": u.mfa_enabled,
        "last_login": u.last_login,
        "created_at": u.created_at,
    }


def _node_dict(n: ClientNodeAssignment, node_name: str | None = None, active: int = 0) -> dict:
    return {
        "assignment_id": str(n.assignment_id),
        "client_id": str(n.client_id),
        "node_id": n.node_id,
        "node_name": node_name or getattr(n, "node_name", None) or n.node_id,
        "max_cameras_on_node": n.max_cameras_on_node,
        "active": active,
        "assigned_by": str(n.assigned_by) if n.assigned_by else None,
        "assigned_at": n.assigned_at,
    }


async def _get_client_or_404(db, client_id: str) -> Client:
    result = await db.execute(select(Client).where(Client.client_id == client_id))
    c = result.scalar_one_or_none()
    if not c:
        raise HTTPException(404, "Client not found")
    return c


# ── ① Create client ────────────────────────────────────────────────────────────

class NodeAssignmentIn(BaseModel):
    node_id: str
    max_cameras_on_node: int = 10

    @field_validator("max_cameras_on_node")
    @classmethod
    def _pos(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_cameras_on_node must be ≥ 1")
        return v


class InitialAdminIn(BaseModel):
    name: str
    email: str
    temp_password: str

    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str:
        if not _EMAIL_RE.match(v):
            raise ValueError("Invalid email")
        return v.lower().strip()

    @field_validator("temp_password")
    @classmethod
    def _pw(cls, v: str) -> str:
        validate_password(v)
        return v


class CreateClientBody(BaseModel):
    name: str
    slug: str
    contact_name: str | None = None
    contact_email: str | None = None
    contact_phone: str | None = None
    address: str | None = None
    logo_url: str | None = None
    max_cameras: int = 50
    max_persons: int = 10000
    settings: dict | None = None
    node_assignments: list[NodeAssignmentIn] = []
    initial_admin: InitialAdminIn

    @field_validator("slug")
    @classmethod
    def _slug(cls, v: str) -> str:
        if not _SLUG_RE.match(v):
            raise ValueError(
                "slug must be 3–100 lowercase alphanumeric chars and hyphens"
            )
        return v

    @field_validator("max_cameras", "max_persons")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("must be ≥ 1")
        return v


@router.post("/clients", status_code=201)
async def create_client(
    body: CreateClientBody,
    db: DBSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    # Slug uniqueness — auto-suffix with incrementing counter (acme → acme-2 → acme-3 …)
    slug = body.slug
    candidate = slug
    _suffix = 2
    while True:
        dup = await db.execute(select(Client).where(Client.slug == candidate))
        if not dup.scalar_one_or_none():
            slug = candidate
            break
        if _suffix > 9999:
            raise HTTPException(409, f"Slug '{body.slug}' is exhausted — choose a different organisation name")
        candidate = f"{slug}-{_suffix}"
        _suffix += 1

    # Admin email uniqueness
    dup_email = await db.execute(
        select(User).where(User.email == body.initial_admin.email)
    )
    if dup_email.scalar_one_or_none():
        raise HTTPException(409, f"Email '{body.initial_admin.email}' already registered")

    now = _now_epoch()
    client = Client(
        name=body.name,
        slug=slug,
        logo_url=body.logo_url,
        contact_name=body.contact_name,
        contact_email=body.contact_email,
        contact_phone=body.contact_phone,
        address=body.address,
        status=ClientStatus.ACTIVE,
        max_cameras=body.max_cameras,
        max_persons=body.max_persons,
        settings=body.settings,
        created_at=now,
        updated_at=now,
    )
    db.add(client)
    await db.flush()  # populate client.client_id

    # Node assignments
    for na in body.node_assignments:
        db.add(ClientNodeAssignment(
            client_id=client.client_id,
            node_id=na.node_id,
            assigned_by=UUID(current_user["user_id"]),
            max_cameras_on_node=na.max_cameras_on_node,
            assigned_at=now,
        ))

    # Initial CLIENT_ADMIN
    admin = User(
        email=body.initial_admin.email,
        password_hash=hash_password(body.initial_admin.temp_password),
        name=body.initial_admin.name,
        role=UserRole.CLIENT_ADMIN,
        client_id=client.client_id,
        status=UserStatus.ACTIVE,
        created_by=UUID(current_user["user_id"]),
        created_at=now,
        updated_at=now,
    )
    db.add(admin)
    await db.flush()

    return {
        "client": _client_dict(client),
        "initial_admin": {
            "user_id": str(admin.user_id),
            "email": admin.email,
            "name": admin.name,
            "temp_password": body.initial_admin.temp_password,  # shown once
        },
        "node_assignments": [
            {"node_id": na.node_id, "max_cameras_on_node": na.max_cameras_on_node}
            for na in body.node_assignments
        ],
    }


# ── ② List clients (with stats) ────────────────────────────────────────────────

@router.get("/clients")
async def list_clients(
    db:     DBSession,
    q:      str | None = Query(None, description="Search by name or slug"),
    status: str | None = Query(None),
    limit:  int        = Query(200, ge=1, le=500),
    offset: int        = Query(0, ge=0),
) -> dict:
    cam_q = (
        select(Camera.client_id, func.count().label("camera_count"))
        .group_by(Camera.client_id).subquery()
    )
    person_q = (
        select(Person.client_id, func.count().label("person_count"))
        .group_by(Person.client_id).subquery()
    )
    session_q = (
        select(Session.client_id, func.count().label("active_sessions"))
        .where(Session.actual_start.isnot(None), Session.actual_end.is_(None))
        .group_by(Session.client_id).subquery()
    )
    node_q = (
        select(ClientNodeAssignment.client_id, func.count().label("node_count"))
        .group_by(ClientNodeAssignment.client_id).subquery()
    )

    base = (
        select(
            Client,
            func.coalesce(cam_q.c.camera_count, 0).label("camera_count"),
            func.coalesce(person_q.c.person_count, 0).label("person_count"),
            func.coalesce(session_q.c.active_sessions, 0).label("active_sessions"),
            func.coalesce(node_q.c.node_count, 0).label("node_count"),
        )
        .outerjoin(cam_q,     cam_q.c.client_id     == Client.client_id)
        .outerjoin(person_q,  person_q.c.client_id  == Client.client_id)
        .outerjoin(session_q, session_q.c.client_id == Client.client_id)
        .outerjoin(node_q,    node_q.c.client_id    == Client.client_id)
    )
    if q:
        base = base.where(
            Client.name.ilike(f"%{q}%") | Client.slug.ilike(f"%{q}%")
        )
    if status and status != "ALL":
        base = base.where(Client.status == status)

    total = (await db.scalar(
        select(func.count()).select_from(base.subquery())
    )) or 0

    rows = await db.execute(base.order_by(Client.name).offset(offset).limit(limit))
    items = [
        {**_client_dict(c), "camera_count": cam, "person_count": per,
         "active_sessions": act, "node_count": nc}
        for c, cam, per, act, nc in rows
    ]
    return {"items": items, "total": total, "offset": offset, "limit": limit}


# ── ③ Client detail ────────────────────────────────────────────────────────────

@router.get("/clients/{client_id}")
async def get_client(client_id: str, db: DBSession) -> dict[str, Any]:
    result = await db.execute(
        select(Client)
        .options(
            selectinload(Client.node_assignments),
            selectinload(Client.users),
        )
        .where(Client.client_id == client_id)
    )
    c = result.scalar_one_or_none()
    if not c:
        raise HTTPException(404, "Client not found")

    # Count cameras and persons for this client
    cam_r = await db.execute(
        select(func.count()).select_from(Camera).where(Camera.client_id == c.client_id)
    )
    per_r = await db.execute(
        select(func.count()).select_from(Person).where(Person.client_id == c.client_id)
    )
    sess_r = await db.execute(
        select(func.count()).select_from(Session)
        .where(Session.client_id == c.client_id, Session.actual_end.is_(None))
    )
    camera_count = cam_r.scalar_one() or 0
    person_count = per_r.scalar_one() or 0
    active_sessions = sess_r.scalar_one() or 0

    # Fetch node names and active camera counts per node for this client
    node_ids = [n.node_id for n in c.node_assignments]
    node_info: dict[str, dict] = {}
    if node_ids:
        from sqlalchemy import text as sa_text
        node_rows = await db.execute(
            sa_text("SELECT node_id, node_name, active_cameras FROM nodes WHERE node_id = ANY(:ids)"),
            {"ids": node_ids},
        )
        for row in node_rows:
            node_info[row.node_id] = {"name": row.node_name, "active": row.active_cameras or 0}

    return {
        **_client_dict(c),
        "camera_count": camera_count,
        "person_count": person_count,
        "active_sessions": active_sessions,
        "node_assignments": [
            _node_dict(
                n,
                node_name=node_info.get(n.node_id, {}).get("name"),
                active=node_info.get(n.node_id, {}).get("active", 0),
            )
            for n in c.node_assignments
        ],
        "users": [_user_dict(u) for u in c.users],
    }


# ── ④ Edit client ──────────────────────────────────────────────────────────────

class UpdateClientBody(BaseModel):
    name: str | None = None
    contact_name: str | None = None
    contact_email: str | None = None
    contact_phone: str | None = None
    address: str | None = None
    logo_url: str | None = None
    max_cameras: int | None = None
    max_persons: int | None = None
    settings: dict | None = None

    @field_validator("max_cameras", "max_persons", mode="before")
    @classmethod
    def _positive(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            raise ValueError("must be ≥ 1")
        return v


@router.put("/clients/{client_id}")
async def update_client(
    client_id: str, body: UpdateClientBody, db: DBSession
) -> dict[str, Any]:
    c = await _get_client_or_404(db, client_id)

    for field, value in body.model_dump(exclude_none=True).items():
        setattr(c, field, value)
    c.updated_at = _now_epoch()

    return _client_dict(c)


# ── ⑤ Client status ────────────────────────────────────────────────────────────

class StatusBody(BaseModel):
    status: str

    @field_validator("status")
    @classmethod
    def _valid(cls, v: str) -> str:
        allowed = {ClientStatus.ACTIVE, ClientStatus.SUSPENDED, ClientStatus.ARCHIVED}
        if v not in allowed:
            raise ValueError(f"status must be one of {allowed}")
        return v


@router.put("/clients/{client_id}/status")
async def set_client_status(
    client_id: str, body: StatusBody, db: DBSession
) -> dict[str, Any]:
    c = await _get_client_or_404(db, client_id)
    c.status = body.status
    c.updated_at = _now_epoch()
    return {"client_id": client_id, "status": c.status}


# ── ⑥ Node assignments ─────────────────────────────────────────────────────────

class NodeBody(BaseModel):
    node_id: str
    max_cameras_on_node: int = 10

    @field_validator("max_cameras_on_node")
    @classmethod
    def _pos(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_cameras_on_node must be ≥ 1")
        return v


@router.post("/clients/{client_id}/nodes", status_code=201)
async def assign_node(
    client_id: str,
    body: NodeBody,
    db: DBSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    await _get_client_or_404(db, client_id)

    dup = await db.execute(
        select(ClientNodeAssignment).where(
            ClientNodeAssignment.client_id == client_id,
            ClientNodeAssignment.node_id == body.node_id,
        )
    )
    if dup.scalar_one_or_none():
        raise HTTPException(409, f"Node '{body.node_id}' already assigned to this client")

    na = ClientNodeAssignment(
        client_id=UUID(client_id),
        node_id=body.node_id,
        assigned_by=UUID(current_user["user_id"]),
        max_cameras_on_node=body.max_cameras_on_node,
        assigned_at=_now_epoch(),
    )
    db.add(na)
    await db.flush()

    # Look up node name and active camera count for the response
    from sqlalchemy import text as _t
    node_row = (await db.execute(
        _t("SELECT node_name, active_cameras FROM nodes WHERE node_id=:nid"),
        {"nid": body.node_id},
    )).fetchone()
    node_name   = node_row.node_name   if node_row else body.node_id
    active_cams = node_row.active_cameras if node_row else 0

    return _node_dict(na, node_name=node_name, active=active_cams)


class UpdateNodeBody(BaseModel):
    max_cameras_on_node: int

    @field_validator("max_cameras_on_node")
    @classmethod
    def _pos(cls, v: int) -> int:
        if v < 1:
            raise ValueError("must be ≥ 1")
        return v


@router.put("/clients/{client_id}/nodes/{node_id}")
async def update_node(
    client_id: str, node_id: str, body: UpdateNodeBody, db: DBSession
) -> dict[str, Any]:
    result = await db.execute(
        select(ClientNodeAssignment).where(
            ClientNodeAssignment.client_id == client_id,
            ClientNodeAssignment.node_id == node_id,
        )
    )
    na = result.scalar_one_or_none()
    if not na:
        raise HTTPException(404, "Node assignment not found")

    na.max_cameras_on_node = body.max_cameras_on_node
    return _node_dict(na)


@router.delete("/clients/{client_id}/nodes/{node_id}", status_code=204)
async def remove_node(client_id: str, node_id: str, db: DBSession) -> None:
    # Block removal if any camera on this node is ONLINE or DEGRADED
    running = await db.execute(
        select(func.count()).where(
            Camera.client_id == client_id,
            Camera.node_id == node_id,
            Camera.status.in_([CameraStatus.ONLINE, CameraStatus.DEGRADED]),
        )
    )
    if running.scalar_one() > 0:
        raise HTTPException(
            409,
            f"Cannot remove node '{node_id}': it has cameras currently ONLINE or DEGRADED.",
        )

    result = await db.execute(
        select(ClientNodeAssignment).where(
            ClientNodeAssignment.client_id == client_id,
            ClientNodeAssignment.node_id == node_id,
        )
    )
    na = result.scalar_one_or_none()
    if not na:
        raise HTTPException(404, "Node assignment not found")

    await db.delete(na)


# ── ⑦ User management (super-admin) ───────────────────────────────────────────

class CreateUserBody(BaseModel):
    email: str
    password: str
    name: str
    role: str
    client_id: str | None = None

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

    @field_validator("role")
    @classmethod
    def _role(cls, v: str) -> str:
        if v not in {UserRole.SUPER_ADMIN, UserRole.CLIENT_ADMIN, UserRole.VIEWER}:
            raise ValueError(f"role must be one of SUPER_ADMIN, CLIENT_ADMIN, VIEWER")
        return v


@router.post("/users", status_code=201)
async def create_user(
    body: CreateUserBody,
    db: DBSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    dup = await db.execute(select(User).where(User.email == body.email))
    if dup.scalar_one_or_none():
        raise HTTPException(409, "Email already registered")

    if body.role != UserRole.SUPER_ADMIN and not body.client_id:
        raise HTTPException(422, "client_id required for non-super-admin roles")

    if body.client_id:
        await _get_client_or_404(db, body.client_id)

    now = _now_epoch()
    user = User(
        email=body.email,
        password_hash=hash_password(body.password),
        name=body.name,
        role=body.role,
        client_id=UUID(body.client_id) if body.client_id else None,
        status=UserStatus.ACTIVE,
        created_by=UUID(current_user["user_id"]),
        created_at=now,
        updated_at=now,
    )
    db.add(user)
    await db.flush()
    return _user_dict(user)


@router.get("/users")
async def list_users(
    db: DBSession,
    role: str | None = Query(None),
    client_id: str | None = Query(None),
    status: str | None = Query(None),
    limit: int = Query(100, le=500),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    q = select(User)
    if role:
        q = q.where(User.role == role)
    if client_id:
        q = q.where(User.client_id == client_id)
    if status:
        q = q.where(User.status == status)
    q = q.order_by(User.created_at.desc()).limit(limit).offset(offset)

    rows = await db.execute(q)
    users = rows.scalars().all()

    total_q = select(func.count()).select_from(User)
    if role:
        total_q = total_q.where(User.role == role)
    if client_id:
        total_q = total_q.where(User.client_id == client_id)
    if status:
        total_q = total_q.where(User.status == status)
    total = (await db.execute(total_q)).scalar_one()

    return {"items": [_user_dict(u) for u in users], "total": total, "offset": offset, "limit": limit}


class UpdateUserBody(BaseModel):
    name: str | None = None
    avatar_url: str | None = None
    role: str | None = None
    client_id: str | None = None

    @field_validator("role")
    @classmethod
    def _role(cls, v: str | None) -> str | None:
        if v and v not in {UserRole.SUPER_ADMIN, UserRole.CLIENT_ADMIN, UserRole.VIEWER}:
            raise ValueError("Invalid role")
        return v


@router.put("/users/{user_id}")
async def update_user(
    user_id: str, body: UpdateUserBody, db: DBSession
) -> dict[str, Any]:
    result = await db.execute(select(User).where(User.user_id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(404, "User not found")

    for field, value in body.model_dump(exclude_none=True).items():
        if field == "client_id":
            setattr(user, field, UUID(value) if value else None)
        else:
            setattr(user, field, value)
    user.updated_at = _now_epoch()

    return _user_dict(user)


class UserStatusBody(BaseModel):
    status: str

    @field_validator("status")
    @classmethod
    def _valid(cls, v: str) -> str:
        allowed = {UserStatus.ACTIVE, UserStatus.INACTIVE, UserStatus.SUSPENDED}
        if v not in allowed:
            raise ValueError(f"status must be one of {allowed}")
        return v


@router.put("/users/{user_id}/status")
async def set_user_status(
    user_id: str, body: UserStatusBody, db: DBSession
) -> dict[str, Any]:
    result = await db.execute(select(User).where(User.user_id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(404, "User not found")

    user.status = body.status
    user.updated_at = _now_epoch()
    return {"user_id": user_id, "status": user.status}


@router.post("/users/{user_id}/reset-password")
async def force_reset_password(
    user_id: str,
    request: Request,
    db: DBSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    result = await db.execute(select(User).where(User.user_id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(404, "User not found")

    svc = AuthService(
        session=db,
        redis=request.app.state.redis,
        kafka_producer=request.app.state.kafka_producer,
        settings=request.app.state.settings,
    )
    await svc.forgot_password(user.email)
    return {"detail": "Password reset email queued.", "email": user.email}


# ── ⑧ Analytics ───────────────────────────────────────────────────────────────

@router.get("/analytics/usage")
async def usage_analytics(db: DBSession) -> dict:
    """Per-client usage stats in dashboard-ready format."""
    cam_q = (
        select(Camera.client_id, func.count().label("cameras"))
        .group_by(Camera.client_id).subquery()
    )
    person_q = (
        select(Person.client_id, func.count().label("persons"))
        .group_by(Person.client_id).subquery()
    )
    total_sess_q = (
        select(Session.client_id, func.count().label("total_sessions"))
        .group_by(Session.client_id).subquery()
    )
    # Avg recognition rate per client
    recog_q = (
        select(Session.client_id,
               func.avg(Session.recognition_rate).label("avg_recog"))
        .where(Session.recognition_rate.isnot(None))
        .group_by(Session.client_id).subquery()
    )
    # GPU hours: sum of (actual_end - actual_start) / 3600
    gpu_q = (
        select(Session.client_id,
               (func.sum(Session.actual_end - Session.actual_start) / 3600.0).label("gpu_hours"))
        .where(Session.actual_start.isnot(None), Session.actual_end.isnot(None))
        .group_by(Session.client_id).subquery()
    )

    rows = await db.execute(
        select(
            Client.client_id, Client.name,
            func.coalesce(cam_q.c.cameras, 0),
            func.coalesce(person_q.c.persons, 0),
            func.coalesce(total_sess_q.c.total_sessions, 0),
            func.coalesce(recog_q.c.avg_recog, 0.0),
            func.coalesce(gpu_q.c.gpu_hours, 0.0),
        )
        .outerjoin(cam_q,        cam_q.c.client_id        == Client.client_id)
        .outerjoin(person_q,     person_q.c.client_id     == Client.client_id)
        .outerjoin(total_sess_q, total_sess_q.c.client_id == Client.client_id)
        .outerjoin(recog_q,      recog_q.c.client_id      == Client.client_id)
        .outerjoin(gpu_q,        gpu_q.c.client_id        == Client.client_id)
        .order_by(Client.name)
    )

    clients = [
        {
            "client_id":        str(r[0]),
            "client_name":      r[1],
            "cameras":          int(r[2]),
            "persons":          int(r[3]),
            "sessions":         int(r[4]),
            "recognition_rate": round(float(r[5]), 4),
            "gpu_hours":        round(float(r[6]), 2),
        }
        for r in rows
    ]
    return {"clients": clients}


@router.get("/analytics/platform")
async def platform_analytics(db: DBSession) -> dict[str, Any]:
    """Global platform metrics — flattened for dashboard consumption."""
    from sqlalchemy import text as _t

    total_clients  = (await db.execute(select(func.count()).select_from(Client))).scalar_one()
    active_clients = (await db.execute(select(func.count()).where(Client.status == ClientStatus.ACTIVE))).scalar_one()
    total_cameras  = (await db.execute(select(func.count()).select_from(Camera))).scalar_one()
    online_cameras = (await db.execute(select(func.count()).where(Camera.status == CameraStatus.ONLINE))).scalar_one()
    total_persons  = (await db.execute(select(func.count()).select_from(Person))).scalar_one()
    total_users    = (await db.execute(select(func.count()).select_from(User))).scalar_one()
    total_sessions = (await db.execute(select(func.count()).select_from(Session))).scalar_one()
    active_sessions = (await db.execute(
        select(func.count()).where(Session.actual_start.isnot(None), Session.actual_end.is_(None))
    )).scalar_one()

    # Sessions started today (epoch)
    today_start = int(datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
    sessions_today = (await db.execute(
        select(func.count()).where(Session.actual_start >= today_start)
    )).scalar_one()

    # GPU hours today
    gpu_sec_today = (await db.execute(
        select(func.sum(Session.actual_end - Session.actual_start))
        .where(Session.actual_start >= today_start,
               Session.actual_end.isnot(None))
    )).scalar_one() or 0

    # Avg recognition rate (all time)
    avg_recog = (await db.execute(
        select(func.avg(Session.recognition_rate)).where(Session.recognition_rate.isnot(None))
    )).scalar_one() or 0.0

    # Load nodes from nodes table
    stale = int(time.time()) - 90
    node_rows = (await db.execute(_t("""
        SELECT node_id, node_name, location, connectivity, api_endpoint,
               gpu_model, max_cameras, active_cameras, status, last_heartbeat, health_json
        FROM nodes ORDER BY last_heartbeat DESC
    """))).fetchall()

    nodes = []
    for n in node_rows:
        h = dict(n.health_json) if isinstance(n.health_json, dict) else {}
        nodes.append({
            "node_id":        n.node_id,
            "node_name":      n.node_name,
            "location":       n.location,
            "connectivity":   n.connectivity,
            "api_endpoint":   n.api_endpoint,
            "gpu_model":      n.gpu_model,
            "max_cameras":    n.max_cameras,
            "active_cameras": n.active_cameras,
            "status":         n.status if n.last_heartbeat >= stale else "OFFLINE",
            "last_heartbeat": n.last_heartbeat,
            "health_json":    h,
        })

    return {
        # Flat fields used by platform-analytics page
        "nodes":                nodes,
        "total_cameras":        total_cameras,
        "cameras_online":       online_cameras,
        "total_persons":        total_persons,
        "active_sessions":      active_sessions,
        "sessions_today":       sessions_today,
        "gpu_hours_today":      round(gpu_sec_today / 3600, 2),
        "avg_recognition_rate": round(float(avg_recog), 4),
        # Legacy nested format (kept for backward compat)
        "clients":  {"total": total_clients,  "active": active_clients},
        "cameras":  {"total": total_cameras,  "online": online_cameras},
        "persons":  {"total": total_persons},
        "users":    {"total": total_users},
        "sessions": {"total": total_sessions, "active": active_sessions,
                     "total_hours": round(gpu_sec_today / 3600, 2)},
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Node Management  (/api/admin/nodes/*)
# ═══════════════════════════════════════════════════════════════════════════════

class RegisterNodeBody(BaseModel):
    node_id:      str | None = None   # auto-generated if omitted
    node_name:    str
    location:     str | None = None
    connectivity: str = "PUBLIC_IP"   # PUBLIC_IP | CLOUDFLARE_TUNNEL | LOCAL
    api_endpoint: str
    gpu_model:    str | None = None
    max_cameras:  int = 10


@router.post("/nodes", status_code=201)
async def register_node_manually(body: RegisterNodeBody, db: DBSession) -> dict:
    """Manually register a GPU node (e.g. a remote machine)."""
    from sqlalchemy import text as _t
    import uuid as _uuid

    nid = body.node_id or str(_uuid.uuid4())
    now = int(time.time())

    await db.execute(_t("""
        INSERT INTO nodes
            (node_id, node_name, location, connectivity, api_endpoint,
             gpu_model, max_cameras, active_cameras, status,
             last_heartbeat, registered_at, updated_at)
        VALUES
            (:nid, :name, :loc, :conn, :ep,
             :gpu, :max_c, 0, 'OFFLINE',
             0, :now, :now)
        ON CONFLICT (node_id) DO UPDATE
            SET node_name    = EXCLUDED.node_name,
                location     = EXCLUDED.location,
                connectivity = EXCLUDED.connectivity,
                api_endpoint = EXCLUDED.api_endpoint,
                gpu_model    = EXCLUDED.gpu_model,
                max_cameras  = EXCLUDED.max_cameras,
                updated_at   = EXCLUDED.updated_at
    """), {
        "nid": nid, "name": body.node_name, "loc": body.location,
        "conn": body.connectivity, "ep": body.api_endpoint,
        "gpu": body.gpu_model, "max_c": body.max_cameras, "now": now,
    })

    return {
        "node_id":      nid,
        "node_name":    body.node_name,
        "location":     body.location,
        "connectivity": body.connectivity,
        "api_endpoint": body.api_endpoint,
        "gpu_model":    body.gpu_model,
        "max_cameras":  body.max_cameras,
        "active_cameras": 0,
        "status":       "OFFLINE",
        "last_heartbeat": 0,
    }


@router.get("/nodes")
async def admin_list_nodes(db: DBSession) -> list:
    """All registered nodes enriched with assigned clients embedded in health_json."""
    from sqlalchemy import text as _t
    stale = int(time.time()) - 90

    rows = await db.execute(_t("""
        SELECT n.node_id, n.node_name, n.location, n.connectivity,
               n.api_endpoint, n.gpu_model, n.max_cameras, n.active_cameras,
               n.status, n.last_heartbeat, n.health_json
        FROM nodes n ORDER BY n.last_heartbeat DESC
    """))

    result = []
    for r in rows.fetchall():
        cna_rows = await db.execute(_t("""
            SELECT c.client_id::text, c.name, c.slug
            FROM client_node_assignments cna
            JOIN clients c ON c.client_id = cna.client_id
            WHERE cna.node_id = :nid ORDER BY c.name
        """), {"nid": r.node_id})
        assigned = [{"client_id": cr[0], "name": cr[1], "slug": cr[2]}
                    for cr in cna_rows.fetchall()]

        health = dict(r.health_json) if isinstance(r.health_json, dict) else {}

        # Normalise NodeManager key names → dashboard expected names
        health["gpu_util"]      = health.get("gpu_util")      or health.get("gpu0_utilization", 0)
        health["cpu_util"]      = health.get("cpu_util")      or health.get("cpu_percent", 0)
        health["ram_util"]      = health.get("ram_util")      or health.get("ram_percent", 0)
        health["vram_used_gb"]  = health.get("vram_used_gb")  or round(health.get("gpu0_mem_used_mb", 0) / 1024, 1)
        health["vram_total_gb"] = health.get("vram_total_gb") or round(health.get("gpu0_mem_total_mb", 24576) / 1024, 1)

        health["connectivity"]      = r.connectivity
        health["assigned_clients"]  = assigned
        health.setdefault("latency_ms",     0)
        health.setdefault("uptime_pct",     100)
        health.setdefault("faces_db_size",  0)
        health.setdefault("throughput_fps", 0)

        result.append({
            "node_id":        r.node_id,
            "node_name":      r.node_name,
            "location":       r.location,
            "connectivity":   r.connectivity,
            "api_endpoint":   r.api_endpoint,
            "gpu_model":      r.gpu_model,
            "max_cameras":    r.max_cameras,
            "active_cameras": r.active_cameras,
            "status":         r.status if r.last_heartbeat >= stale else "OFFLINE",
            "last_heartbeat": r.last_heartbeat,
            "health_json":    health,
        })
    return result


@router.put("/nodes/{node_id}/drain")
async def drain_node(node_id: str, db: DBSession) -> dict:
    """Mark a node DRAINING so no new cameras are assigned to it."""
    from sqlalchemy import text as _t
    await db.execute(
        _t("UPDATE nodes SET status='DRAINING', updated_at=:now WHERE node_id=:nid"),
        {"nid": node_id, "now": int(time.time())},
    )
    return {"node_id": node_id, "status": "DRAINING"}


@router.delete("/nodes/{node_id}", status_code=204)
async def remove_node_from_registry(node_id: str, db: DBSession) -> None:
    """Remove a node and its client assignments from the registry."""
    from sqlalchemy import text as _t
    await db.execute(_t("DELETE FROM client_node_assignments WHERE node_id=:nid"), {"nid": node_id})
    await db.execute(_t("DELETE FROM nodes WHERE node_id=:nid"),                  {"nid": node_id})


@router.get("/nodes/affinity")
async def get_affinity_rules(db: DBSession) -> dict:
    """Return camera-to-node affinity rules."""
    from sqlalchemy import text as _t
    import json as _json
    row = (await db.execute(_t(
        "SELECT value FROM system_config WHERE key='node_affinity_rules' LIMIT 1"
    ))).fetchone()
    return {"rules": _json.loads(row[0]) if row else []}


@router.put("/nodes/affinity")
async def save_affinity_rules(body: dict, db: DBSession) -> dict:
    """Persist affinity rules."""
    from sqlalchemy import text as _t
    import json as _json
    await db.execute(_t("""
        INSERT INTO system_config (key, value) VALUES ('node_affinity_rules', :v)
        ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value
    """), {"v": _json.dumps(body.get("rules", []))})
    return {"ok": True}


@router.post("/nodes/auto-balance")
async def auto_balance_nodes(request: Request, db: DBSession) -> dict:
    """Preview or execute camera rebalancing across nodes."""
    import json as _json
    from sqlalchemy import text as _t
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    dry_run = request.query_params.get("dry_run", "false").lower() == "true"

    rows = await db.execute(_t("""
        SELECT n.node_id, n.node_name, n.max_cameras,
               COUNT(c.camera_id) AS assigned_count
        FROM nodes n
        LEFT JOIN cameras c ON c.node_id = n.node_id AND c.status != 'OFFLINE'
        WHERE n.status IN ('ONLINE', 'DRAINING')
        GROUP BY n.node_id, n.node_name, n.max_cameras
        ORDER BY assigned_count DESC
    """))
    nd = rows.fetchall()
    if len(nd) < 2:
        return {"moves": [], "message": "Need at least 2 active nodes to balance"}

    total = sum(r[3] for r in nd)
    avg   = total / len(nd)
    moves = [{"node_id": r[0], "node_name": r[1], "current": r[3], "target": round(avg), "delta": round(avg) - r[3]}
             for r in nd if abs(r[3] - avg) > 1]
    return {
        "dry_run":         dry_run,
        "total_cameras":   total,
        "target_per_node": round(avg, 1),
        "moves":           moves,
        "message":         f"{'Preview — ' if dry_run else ''}{len(moves)} node(s) need rebalancing",
    }


@router.get("/nodes/migration-log")
async def migration_log(db: DBSession, limit: int = 50) -> dict:
    """Recent camera migration events."""
    from sqlalchemy import text as _t
    try:
        rows = await db.execute(_t("""
            SELECT migration_id::text, camera_id::text, camera_name,
                   from_node_id, from_node_name, to_node_id, to_node_name,
                   reason, status,
                   to_char(to_timestamp(created_at),'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS created_at
            FROM camera_migrations ORDER BY created_at DESC LIMIT :lim
        """), {"lim": limit})
        events = [dict(zip(rows.keys(), r)) for r in rows.fetchall()]
    except Exception:
        events = []
    return {"events": events}
