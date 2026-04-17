"""
ACAS Redis Cache Service
========================
Centralised caching layer built on the shared aioredis connection pool.

Features
────────
• Analytics cache  — TTL 30 s; keyed by endpoint + query params
• Session state    — Redis HSET; TTL 1 h; fast hgetall for WS snapshots
• Pub/Sub channels — one channel per session, one for monitoring broadcasts
• API rate limiter — sliding-window (used by auth middleware)
• Hot-spot helper  — cursor-keyset encoding for pagination

All keys are namespaced under "acas:" to avoid collisions with other services
sharing the same Redis instance.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

log = logging.getLogger(__name__)

# Key prefixes
_NS          = "acas:"
_ANALYTICS   = _NS + "analytics:"    # analytics:<sha8>
_SESSION_H   = _NS + "sess:"         # sess:<session_id>  (HASH)
_FACE_REPO   = _NS + "faiss:"        # faiss:<client_id>  (metadata HASH)
_WS_CHAN     = _NS + "ws:"           # ws:<session_id>    (pub/sub channel)
_MON_CHAN    = _NS + "mon"           # monitoring broadcast channel
_RATE        = _NS + "rl:"           # rl:<key>           (rate-limit)
_CURSOR      = _NS + "cur:"          # cur:<token>        (cursor cache)
_USER_ONLINE = _NS + "online:"       # online:<user_id>   (presence)

# TTLs (seconds)
TTL_ANALYTICS   = 30
TTL_SESSION     = 3_600    # 1 hour
TTL_CURSOR      = 300      # 5 minutes — cursor tokens
TTL_ONLINE      = 90       # presence heartbeat window
TTL_RATE_WINDOW = 900      # rate-limit window (15 min)


class CacheService:
    """
    Thin async wrapper around the shared Redis client that encodes ACAS-specific
    cache policies.

    Instantiate once at startup, pass the redis.asyncio.Redis client:

        from redis.asyncio import Redis
        cache = CacheService(redis_client)
        app.state.cache = cache
    """

    def __init__(self, redis: Any) -> None:
        self._r = redis

    # ── Analytics cache ───────────────────────────────────────────────────────

    @staticmethod
    def _analytics_key(path: str, params: str = "") -> str:
        digest = hashlib.sha256(f"{path}?{params}".encode()).hexdigest()[:12]
        return _ANALYTICS + digest

    async def get_analytics(self, path: str, params: str = "") -> dict | None:
        raw = await self._r.get(self._analytics_key(path, params))
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    async def set_analytics(self, path: str, params: str, data: dict, ttl: int = TTL_ANALYTICS) -> None:
        key = self._analytics_key(path, params)
        await self._r.setex(key, ttl, json.dumps(data, default=str))

    async def invalidate_analytics(self, prefix: str = "") -> int:
        """Delete all analytics cache keys matching an optional prefix."""
        pattern = _ANALYTICS + ("*" if not prefix else f"*{prefix}*")
        keys = await self._r.keys(pattern)
        if keys:
            return await self._r.delete(*keys)
        return 0

    # ── Session state (HSET) ──────────────────────────────────────────────────

    @staticmethod
    def _sess_key(session_id: str) -> str:
        return _SESSION_H + session_id

    async def get_session_state(self, session_id: str) -> dict | None:
        data = await self._r.hgetall(self._sess_key(session_id))
        if not data:
            return None
        # Redis returns bytes keys/values; decode them
        return {
            (k.decode() if isinstance(k, bytes) else k):
            (v.decode() if isinstance(v, bytes) else v)
            for k, v in data.items()
        }

    async def set_session_field(self, session_id: str, field: str, value: Any) -> None:
        key = self._sess_key(session_id)
        pipe = self._r.pipeline()
        pipe.hset(key, field, json.dumps(value, default=str) if not isinstance(value, str) else value)
        pipe.expire(key, TTL_SESSION)
        await pipe.execute()

    async def set_session_state(self, session_id: str, state: dict, ttl: int = TTL_SESSION) -> None:
        key = self._sess_key(session_id)
        # Flatten all values to strings for HSET
        flat = {k: (json.dumps(v, default=str) if not isinstance(v, str) else v) for k, v in state.items()}
        pipe = self._r.pipeline()
        pipe.hset(key, mapping=flat)
        pipe.expire(key, ttl)
        await pipe.execute()

    async def delete_session_state(self, session_id: str) -> None:
        await self._r.delete(self._sess_key(session_id))

    # ── Pub/Sub helpers ───────────────────────────────────────────────────────

    @staticmethod
    def session_channel(session_id: str) -> str:
        """Redis pub/sub channel name for a specific session."""
        return _WS_CHAN + session_id

    @staticmethod
    def monitoring_channel() -> str:
        """Redis pub/sub channel for node-wide monitoring broadcasts."""
        return _MON_CHAN

    async def publish_session_event(self, session_id: str, event: dict) -> None:
        channel = self.session_channel(session_id)
        payload = json.dumps(event, default=str)
        await self._r.publish(channel, payload)

    async def publish_monitoring_event(self, event: dict) -> None:
        payload = json.dumps(event, default=str)
        await self._r.publish(self.monitoring_channel(), payload)

    # ── FAISS index metadata ──────────────────────────────────────────────────

    async def set_faiss_meta(self, client_id: str, total_vecs: int, built_at: float) -> None:
        key = _FACE_REPO + client_id
        await self._r.hset(key, mapping={
            "total_vectors": str(total_vecs),
            "built_at":      str(built_at),
        })

    async def get_faiss_meta(self, client_id: str) -> dict | None:
        data = await self._r.hgetall(_FACE_REPO + client_id)
        return {k.decode(): v.decode() for k, v in data.items()} if data else None

    # ── Cursor-keyset pagination ───────────────────────────────────────────────

    async def store_cursor(self, cursor_token: str, payload: dict) -> None:
        await self._r.setex(_CURSOR + cursor_token, TTL_CURSOR, json.dumps(payload))

    async def load_cursor(self, cursor_token: str) -> dict | None:
        raw = await self._r.get(_CURSOR + cursor_token)
        return json.loads(raw) if raw else None

    # ── Sliding-window rate limiter ────────────────────────────────────────────

    async def rate_limit_check(
        self,
        key:        str,
        limit:      int,
        window_s:   int = TTL_RATE_WINDOW,
    ) -> tuple[bool, int]:
        """
        Sliding window rate limiter using a sorted set.
        Returns (allowed: bool, remaining: int).
        """
        now_ms = int(time.time() * 1000)
        rk     = _RATE + key
        pipe   = self._r.pipeline()
        # Remove old entries outside the window
        pipe.zremrangebyscore(rk, "-inf", now_ms - window_s * 1000)
        pipe.zcard(rk)
        pipe.zadd(rk, {str(now_ms): now_ms})
        pipe.expire(rk, window_s)
        _, count_before, *_ = await pipe.execute()
        count = int(count_before)
        allowed = count < limit
        remaining = max(0, limit - count - 1)
        return allowed, remaining

    # ── User presence ─────────────────────────────────────────────────────────

    async def heartbeat_online(self, user_id: str) -> None:
        await self._r.setex(_USER_ONLINE + user_id, TTL_ONLINE, "1")

    async def is_online(self, user_id: str) -> bool:
        return await self._r.exists(_USER_ONLINE + user_id) > 0

    # ── Health ────────────────────────────────────────────────────────────────

    async def ping(self) -> bool:
        try:
            return await self._r.ping()
        except Exception:
            return False
