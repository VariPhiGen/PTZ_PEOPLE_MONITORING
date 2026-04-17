from __future__ import annotations

from typing import Annotated, AsyncGenerator

from fastapi import Depends, HTTPException, Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


async def get_db(request: Request) -> AsyncGenerator[AsyncSession, None]:
    """Yield an AsyncSession with the tenant RLS setting applied for the request."""
    client_id = getattr(request.state, "client_id", None)
    rls_value = str(client_id) if client_id else ""

    async with request.app.state.session_factory() as session:
        async with session.begin():
            # SET LOCAL scoped to this transaction; resets after commit/rollback.
            # Using set_config avoids f-string injection risk.
            await session.execute(
                text("SELECT set_config('app.current_client_id', :val, TRUE)"),
                {"val": rls_value},
            )
            yield session


DBSession = Annotated[AsyncSession, Depends(get_db)]


def get_current_user(request: Request) -> dict:
    """Extract authenticated user info from request.state (set by AuthMiddleware)."""
    if not getattr(request.state, "authenticated", False):
        raise HTTPException(status_code=401, detail="Not authenticated")
    return {
        "user_id": request.state.user_id,
        "email": request.state.email,
        "name": request.state.name,
        "role": request.state.role,
        "client_id": request.state.client_id,
        "client_slug": request.state.client_slug,
        "permissions": request.state.permissions,
    }


CurrentUser = Annotated[dict, Depends(get_current_user)]
