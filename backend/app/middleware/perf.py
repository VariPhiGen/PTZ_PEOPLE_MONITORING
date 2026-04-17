"""
ACAS Performance Middleware
============================
• Adds X-Response-Time-Ms header to every response.
• Logs responses slower than a configurable threshold.
• Caches GET analytics responses in Redis for 30 s using the cache service.
• Provides CursorPage — a keyset-cursor pagination helper for any API route.

Mounting
────────
    from app.middleware.perf import PerfMiddleware, CursorPage
    app.add_middleware(PerfMiddleware)

Cursor pagination usage
────────────────────────
    @router.get("/cameras")
    async def list_cameras(
        limit:  int = 50,
        cursor: str | None = None,
        ...
    ):
        page = await CursorPage.decode(cursor, cache)
        rows = await db.execute(
            select(Camera)
            .where(Camera.client_id == client_id)
            .where(Camera.created_at < page.created_at_lt)   # keyset
            .order_by(Camera.created_at.desc())
            .limit(limit + 1)
        )
        items = rows.scalars().all()
        next_cur = await CursorPage.encode(items, limit, cache) if len(items) > limit else None
        return {"items": items[:limit], "next_cursor": next_cur}
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

if TYPE_CHECKING:
    from app.core.cache import CacheService

log = logging.getLogger(__name__)

# Endpoints whose GET responses are eligible for caching
_CACHEABLE_PREFIXES = (
    "/api/analytics/",
    "/api/attendance/sessions",   # list — not individual session detail
    "/api/node/info",
)

# Warn on responses slower than this
_SLOW_MS = 200.0


class PerfMiddleware(BaseHTTPMiddleware):
    """
    Starlette middleware that:
      1. Times every request and adds X-Response-Time-Ms.
      2. Warns on slow responses.
      3. Caches GET analytics responses in Redis (TTL 30 s).
    """

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        t0 = time.perf_counter()

        # ── Cache check for GET analytics endpoints ────────────────────────────
        cache: CacheService | None = getattr(
            getattr(request, "app", None), "state", {}
        )
        cache = getattr(request.app.state, "cache", None) if hasattr(request, "app") else None

        path = request.url.path
        is_cacheable = (
            request.method == "GET"
            and any(path.startswith(p) for p in _CACHEABLE_PREFIXES)
        )

        if is_cacheable and cache:
            params = str(request.query_params)
            cached = await cache.get_analytics(path, params)
            if cached is not None:
                elapsed = (time.perf_counter() - t0) * 1000
                resp = JSONResponse(cached)
                resp.headers["X-Response-Time-Ms"] = f"{elapsed:.1f}"
                resp.headers["X-Cache"] = "HIT"
                return resp

        # ── Execute the actual handler ────────────────────────────────────────
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"

        if elapsed_ms > _SLOW_MS:
            log.warning(
                "SLOW %s %s → %d  %.0fms",
                request.method, path, response.status_code, elapsed_ms,
            )

        # ── Cache successful GET analytics responses ───────────────────────────
        if is_cacheable and cache and response.status_code == 200:
            # We need to read the body to cache it; wrap in a capturing response
            body = b""
            async for chunk in response.body_iterator:  # type: ignore[attr-defined]
                body += chunk
            try:
                data = json.loads(body)
                params = str(request.query_params)
                await cache.set_analytics(path, params, data)
            except Exception:
                pass
            from starlette.responses import Response as StarletteResponse
            return StarletteResponse(
                content=body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type,
            )

        return response


# ── Cursor-keyset pagination helper ──────────────────────────────────────────

class CursorPage:
    """
    Opaque cursor tokens for keyset pagination.

    The cursor encodes a (created_at, id) pair that acts as the exclusive lower
    or upper bound for the next page.  Tokens are stored in Redis for 5 min so
    they stay short (a random UUID), hiding internal state from clients.

    Simple usage when Redis is not available: fall back to base64 JSON.
    """

    def __init__(self, created_at: float, record_id: str, direction: str = "next") -> None:
        self.created_at    = created_at
        self.record_id     = record_id
        self.direction     = direction

    # ── Encode (end of page → produce next cursor) ────────────────────────────

    @classmethod
    async def encode(
        cls,
        last_item: Any,
        cache: "CacheService | None" = None,
    ) -> str:
        payload = {
            "ts":  getattr(last_item, "created_at", time.time()),
            "id":  str(getattr(last_item, "id", "") or getattr(last_item, "session_id", "")),
            "dir": "next",
        }
        if cache:
            token = uuid.uuid4().hex
            await cache.store_cursor(token, payload)
            return token
        # Fallback: embed payload directly in token (no Redis needed)
        raw = json.dumps(payload, separators=(",", ":"))
        return base64.urlsafe_b64encode(raw.encode()).decode()

    # ── Decode (incoming cursor → extract bounds) ─────────────────────────────

    @classmethod
    async def decode(
        cls,
        token: str | None,
        cache: "CacheService | None" = None,
    ) -> "CursorPage":
        if not token:
            # No cursor → first page; use a far-future sentinel
            return cls(created_at=float("inf"), record_id="", direction="first")

        payload: dict | None = None

        if cache:
            payload = await cache.load_cursor(token)

        if payload is None:
            # Try base64 fallback
            try:
                raw = base64.urlsafe_b64decode(token.encode() + b"==").decode()
                payload = json.loads(raw)
            except Exception:
                pass

        if not payload:
            return cls(created_at=float("inf"), record_id="", direction="first")

        # ts may be an ISO string (from SQLAlchemy) or a unix float
        ts = payload.get("ts", float("inf"))
        if isinstance(ts, str):
            from datetime import datetime
            try:
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
            except ValueError:
                ts = float("inf")

        return cls(
            created_at=float(ts),
            record_id=str(payload.get("id", "")),
            direction=str(payload.get("dir", "next")),
        )

    # ── SQLAlchemy filter helpers ─────────────────────────────────────────────

    def created_at_filter(self, model: Any) -> Any:
        """
        Return a SQLAlchemy WHERE clause for keyset pagination.

        Usage:
            .where(cursor.created_at_filter(Session))
        """
        from sqlalchemy import and_, or_
        import datetime as _dt

        if self.direction == "first" or self.created_at == float("inf"):
            return True  # no filter needed for first page

        col_ts = model.created_at
        col_id = getattr(model, "id",
                 getattr(model, "session_id",
                 getattr(model, "record_id", None)))

        ts = _dt.datetime.fromtimestamp(self.created_at, tz=_dt.timezone.utc)

        if col_id is not None:
            return or_(
                col_ts < ts,
                and_(col_ts == ts, col_id < self.record_id),
            )
        return col_ts < ts
