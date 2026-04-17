from __future__ import annotations

import re
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, field_validator

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _validate_email_fmt(v: str) -> str:
    if not _EMAIL_RE.match(v):
        raise ValueError("Invalid email address")
    return v.lower().strip()

from app.deps import CurrentUser, DBSession
from app.services.auth_service import AuthError, AuthService
from app.models.users import UserRole

router = APIRouter(prefix="/api/auth", tags=["auth"])


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    email: str
    password: str
    mfa_code: str | None = None

    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str:
        return _validate_email_fmt(v)


class RefreshRequest(BaseModel):
    refresh_token: str


class ForgotPasswordRequest(BaseModel):
    email: str

    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str:
        return _validate_email_fmt(v)


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str


class UpdateMeRequest(BaseModel):
    name: str | None = None
    avatar_url: str | None = None

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            raise ValueError("name cannot be blank")
        return v


class MfaVerifyRequest(BaseModel):
    code: str


# ── Helper ────────────────────────────────────────────────────────────────────

def _svc(request: Request, db: DBSession) -> AuthService:
    return AuthService(
        session=db,
        redis=request.app.state.redis,
        kafka_producer=request.app.state.kafka_producer,
        settings=request.app.state.settings,
    )


def _auth_error_to_http(exc: AuthError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=str(exc))


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/login")
async def login(
    body: LoginRequest,
    request: Request,
    db: DBSession,
) -> dict[str, Any]:
    try:
        return await _svc(request, db).login(
            email=body.email,
            password=body.password,
            mfa_code=body.mfa_code,
        )
    except AuthError as exc:
        raise _auth_error_to_http(exc)


@router.post("/refresh")
async def refresh(
    body: RefreshRequest,
    request: Request,
    db: DBSession,
) -> dict[str, Any]:
    try:
        return await _svc(request, db).refresh_token(body.refresh_token)
    except AuthError as exc:
        raise _auth_error_to_http(exc)


@router.post("/forgot-password", status_code=202)
async def forgot_password(
    body: ForgotPasswordRequest,
    request: Request,
    db: DBSession,
) -> dict[str, str]:
    # Always 202 — never reveal whether the email exists.
    await _svc(request, db).forgot_password(body.email)
    return {"detail": "If that email is registered, a reset link has been sent."}


@router.post("/reset-password")
async def reset_password(
    body: ResetPasswordRequest,
    request: Request,
    db: DBSession,
) -> dict[str, str]:
    try:
        await _svc(request, db).reset_password(body.token, body.new_password)
        return {"detail": "Password updated successfully."}
    except AuthError as exc:
        raise _auth_error_to_http(exc)


@router.get("/me")
async def get_me(current_user: CurrentUser) -> dict[str, Any]:
    return current_user


@router.put("/me")
async def update_me(
    body: UpdateMeRequest,
    request: Request,
    db: DBSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    from sqlalchemy import select
    from app.models.users import User

    result = await db.execute(
        select(User).where(User.user_id == current_user["user_id"])
    )
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(404, "User not found")

    if body.name is not None:
        user.name = body.name.strip()
    if body.avatar_url is not None:
        user.avatar_url = body.avatar_url

    return {
        "user_id": str(user.user_id),
        "email": user.email,
        "name": user.name,
        "avatar_url": user.avatar_url,
        "role": user.role,
    }


@router.post("/mfa/enable")
async def mfa_enable(
    request: Request,
    db: DBSession,
    current_user: CurrentUser,
) -> dict[str, str]:
    try:
        return await _svc(request, db).enable_mfa(current_user["user_id"])
    except AuthError as exc:
        raise _auth_error_to_http(exc)


@router.post("/mfa/verify")
async def mfa_verify(
    body: MfaVerifyRequest,
    request: Request,
    db: DBSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    ok = await _svc(request, db).verify_mfa(current_user["user_id"], body.code)
    if not ok:
        raise HTTPException(400, "Invalid or expired MFA code")
    return {"detail": "MFA verified and enabled successfully.", "mfa_enabled": True}


@router.get("/my-client")
async def get_my_client(
    request: Request,
    db: DBSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    """Return the caller's own client record. Works for CLIENT_ADMIN and VIEWER."""
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload
    from app.models.clients import Client
    from app.models.client_node_assignments import ClientNodeAssignment

    cid = getattr(request.state, "client_id", None) or current_user.get("client_id")
    if not cid:
        raise HTTPException(400, "No client context (SUPER_ADMIN has no own client)")

    result = await db.execute(
        select(Client)
        .options(selectinload(Client.node_assignments))
        .where(Client.client_id == cid)
    )
    c = result.scalar_one_or_none()
    if not c:
        raise HTTPException(404, "Client not found")

    return {
        "client_id":     str(c.client_id),
        "name":          c.name,
        "slug":          c.slug,
        "logo_url":      c.logo_url,
        "contact_name":  c.contact_name,
        "contact_email": c.contact_email,
        "contact_phone": c.contact_phone,
        "address":       c.address,
        "status":        c.status.value if hasattr(c.status, "value") else c.status,
        "max_cameras":   c.max_cameras,
        "max_persons":   c.max_persons,
        "settings":      c.settings,
        "created_at":    c.created_at,
        "updated_at":    c.updated_at,
    }
