from __future__ import annotations

import logging
from typing import AsyncGenerator

from fastapi import Depends, HTTPException, Request
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.deps import DBSession
from app.models.cameras import Camera
from app.models.client_node_assignments import ClientNodeAssignment
from app.models.clients import Client
from app.models.persons import Person
from app.models.users import UserRole

logger = logging.getLogger(__name__)


class TenantMiddleware:
    """
    Starlette-compatible dependency class that enforces tenant boundaries.

    For CLIENT_ADMIN / VIEWER:
      - Validates that any explicit client_id in path/query matches the JWT.
      - Provides FastAPI dependency helpers for quota and node checks.

    SUPER_ADMIN bypasses all checks.

    Usage: add `TenantMiddleware()` to the middleware stack; use the
    individual Depends helpers on specific routes.
    """

    # Header / query param name that routes may expose
    CLIENT_ID_QUERY_PARAM = "client_id"

    @staticmethod
    def _is_super_admin(request: Request) -> bool:
        return getattr(request.state, "role", None) == UserRole.SUPER_ADMIN

    @staticmethod
    def _jwt_client_id(request: Request) -> str | None:
        return getattr(request.state, "client_id", None)

    # ── Path / query guard ────────────────────────────────────────────────────

    @staticmethod
    async def validate_client_id(request: Request) -> None:
        """
        Dependency: ensure `client_id` path/query param equals the JWT claim.
        Attach as a router-level dependency for all client-scoped routes.
        """
        if TenantMiddleware._is_super_admin(request):
            return

        jwt_cid = TenantMiddleware._jwt_client_id(request)
        if not jwt_cid:
            raise HTTPException(403, "No client association in token")

        # Check path params first, then query params
        param_cid = (
            request.path_params.get(TenantMiddleware.CLIENT_ID_QUERY_PARAM)
            or request.query_params.get(TenantMiddleware.CLIENT_ID_QUERY_PARAM)
        )
        if param_cid and str(param_cid) != str(jwt_cid):
            raise HTTPException(403, "client_id does not match your token")


# ── Quota / assignment dependencies (used on specific routes) ──────────────────


async def check_camera_quota(request: Request, db: DBSession) -> None:
    """
    Dependency for camera-creation routes.
    Verifies the client has not reached max_cameras.
    """
    if getattr(request.state, "role", None) == UserRole.SUPER_ADMIN:
        return

    client_id = getattr(request.state, "client_id", None)
    if not client_id:
        raise HTTPException(403, "No client association")

    result = await db.execute(
        select(Client.max_cameras).where(Client.client_id == client_id)
    )
    max_cameras = result.scalar_one_or_none()
    if max_cameras is None:
        raise HTTPException(404, "Client not found")

    count_result = await db.execute(
        select(func.count()).where(Camera.client_id == client_id)
    )
    current = count_result.scalar_one()
    if current >= max_cameras:
        raise HTTPException(
            422,
            f"Camera quota reached ({current}/{max_cameras}). "
            "Contact your administrator to increase the limit.",
        )


async def check_person_quota(request: Request, db: DBSession) -> None:
    """
    Dependency for person-enrollment routes.
    Verifies the client has not reached max_persons.
    """
    if getattr(request.state, "role", None) == UserRole.SUPER_ADMIN:
        return

    client_id = getattr(request.state, "client_id", None)
    if not client_id:
        raise HTTPException(403, "No client association")

    result = await db.execute(
        select(Client.max_persons).where(Client.client_id == client_id)
    )
    max_persons = result.scalar_one_or_none()
    if max_persons is None:
        raise HTTPException(404, "Client not found")

    count_result = await db.execute(
        select(func.count()).where(Person.client_id == client_id)
    )
    current = count_result.scalar_one()
    if current >= max_persons:
        raise HTTPException(
            422,
            f"Person quota reached ({current}/{max_persons}). "
            "Contact your administrator to increase the limit.",
        )


async def check_node_assignment(request: Request, db: DBSession) -> None:
    """
    Dependency for camera-creation routes that include a `node_id` body field.
    Verifies the node is assigned to the requesting client.
    Reads `node_id` from a previously-parsed body stored in request.state.node_id,
    or from query params (set by your route handler before this dependency fires).
    """
    if getattr(request.state, "role", None) == UserRole.SUPER_ADMIN:
        return

    client_id = getattr(request.state, "client_id", None)
    node_id = getattr(request.state, "node_id", None) or request.query_params.get(
        "node_id"
    )
    if not node_id:
        return  # node_id optional; assignment check skipped if not provided

    result = await db.execute(
        select(ClientNodeAssignment).where(
            ClientNodeAssignment.client_id == client_id,
            ClientNodeAssignment.node_id == node_id,
        )
    )
    if result.scalar_one_or_none() is None:
        raise HTTPException(
            422,
            f"Node '{node_id}' is not assigned to your account.",
        )
