"""Shared helpers used across all API routers."""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)


def now_epoch() -> int:
    return int(datetime.now(timezone.utc).timestamp())


async def rate_limit(
    redis: Any,
    key: str,
    max_calls: int,
    window_s: int,
) -> None:
    """Raise HTTP 429 if the key has been hit more than max_calls times in window_s seconds."""
    count = await redis.incr(key)
    if count == 1:
        await redis.expire(key, window_s)
    if count > max_calls:
        raise HTTPException(status_code=429, detail="Rate limit exceeded — try again later")


def audit(
    request: Request,
    action: str,
    target_type: str,
    target_id: str,
    detail: dict | None = None,
) -> None:
    """Fire-and-forget audit event via the raw Kafka producer on app.state."""
    try:
        producer = getattr(request.app.state, "kafka_producer", None)
        user_id  = getattr(request.state, "user_id",  "system")
        client_id = getattr(request.state, "client_id", None)
        if producer is None:
            return
        payload = {
            "audit_id":    __import__("uuid").uuid4().hex,
            "client_id":   str(client_id) if client_id else None,
            "actor_id":    str(user_id) if user_id else "system",
            "actor_type":  "ADMIN" if request.state.role in ("SUPER_ADMIN", "CLIENT_ADMIN") else "USER",
            "action":      action,
            "target_type": target_type,
            "target_id":   str(target_id),
            "detail_json": json.dumps(detail) if detail else None,
            "ts":          time.time(),
        }
        producer.produce(
            "audit.events",
            value=json.dumps(payload).encode(),
            key=str(target_id).encode(),
        )
        producer.poll(0)
    except Exception as exc:
        logger.debug("audit produce skipped: %s", exc)


def resolve_client_id(
    request: Request,
    body_client_id: str | None = None,
    *,
    require: bool = True,
) -> str | None:
    """
    Return the effective client_id for a request.
    SUPER_ADMIN may supply an explicit client_id; CLIENT_ADMIN/VIEWER are always
    scoped to their own client from the JWT.

    When ``require=False`` and the caller is SUPER_ADMIN with no client_id
    context, returns ``None`` (caller should interpret as "all clients").
    """
    role      = getattr(request.state, "role", None)
    jwt_cid   = getattr(request.state, "client_id", None)

    if role == "SUPER_ADMIN":
        cid = body_client_id or jwt_cid
        if not cid:
            if require:
                raise HTTPException(status_code=400, detail="client_id required for SUPER_ADMIN")
            return None
        return str(cid)

    if jwt_cid is None:
        raise HTTPException(status_code=403, detail="No tenant context")
    return str(jwt_cid)
