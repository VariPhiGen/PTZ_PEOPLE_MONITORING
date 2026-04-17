from __future__ import annotations

import asyncio
import json
import logging
import secrets
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import pyotp
import qrcode
import io
import base64
from confluent_kafka import Producer
from redis import asyncio as redis_async
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.constants import (
    ACCESS_TOKEN_TTL_HOURS,
    LOGIN_MAX_ATTEMPTS,
    LOGIN_WINDOW_SECONDS,
    MFA_PENDING_KEY,
    PWD_RESET_KEY,
    PWD_RESET_TTL_SECONDS,
    RATELIMIT_LOGIN_KEY,
    REFRESH_TOKEN_KEY,
    REFRESH_TOKEN_TTL_DAYS,
    ROLE_PERMISSIONS,
)
from app.models.users import User, UserRole, UserStatus
from app.utils.jwt import TokenError, create_access_token, create_refresh_token, decode_token
from app.utils.security import hash_password, validate_password, verify_password

logger = logging.getLogger(__name__)


class AuthError(Exception):
    """Raised for expected auth failures (wrong creds, rate limit, etc.)."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


class AuthService:
    def __init__(
        self,
        session: AsyncSession,
        redis: redis_async.Redis,
        kafka_producer: Producer,
        settings: Settings,
    ) -> None:
        self._db = session
        self._redis = redis
        self._kafka = kafka_producer
        self._settings = settings

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_token_payload(self, user: User, client_slug: str | None = None) -> dict:
        return {
            "user_id": str(user.user_id),
            "email": user.email,
            "name": user.name,
            "role": user.role,
            "client_id": str(user.client_id) if user.client_id else None,
            "client_slug": client_slug,
            "permissions": ROLE_PERMISSIONS.get(user.role, []),
        }

    async def _get_user_by_email(self, email: str) -> User | None:
        result = await self._db.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()

    async def _get_user_by_id(self, user_id: str | UUID) -> User | None:
        result = await self._db.execute(
            select(User).where(User.user_id == str(user_id))
        )
        return result.scalar_one_or_none()

    def _kafka_publish(self, topic: str, key: str, value: dict) -> None:
        try:
            self._kafka.produce(
                topic,
                key=key.encode(),
                value=json.dumps(value).encode(),
            )
            self._kafka.poll(0)
        except Exception:
            logger.exception("Kafka publish failed: topic=%s key=%s", topic, key)

    # ── Rate limiting ─────────────────────────────────────────────────────────

    async def _check_rate_limit(self, email: str) -> None:
        key = RATELIMIT_LOGIN_KEY.format(email=email)
        count = await self._redis.incr(key)
        if count == 1:
            await self._redis.expire(key, LOGIN_WINDOW_SECONDS)
        if count > LOGIN_MAX_ATTEMPTS:
            ttl = await self._redis.ttl(key)
            raise AuthError(
                f"Too many login attempts. Try again in {ttl} seconds.",
                status_code=429,
            )

    async def _clear_rate_limit(self, email: str) -> None:
        await self._redis.delete(RATELIMIT_LOGIN_KEY.format(email=email))

    # ── Token store (refresh token revocation) ────────────────────────────────

    async def _store_refresh_token(self, jti: str, user_id: str) -> None:
        key = REFRESH_TOKEN_KEY.format(jti=jti)
        await self._redis.setex(key, REFRESH_TOKEN_TTL_DAYS * 86400, user_id)

    async def _consume_refresh_token(self, jti: str) -> str | None:
        """Return user_id and delete the token atomically."""
        key = REFRESH_TOKEN_KEY.format(jti=jti)
        user_id = await self._redis.getdel(key)
        return user_id

    # ── Public API ────────────────────────────────────────────────────────────

    async def register_user(
        self,
        email: str,
        password: str,
        name: str,
        role: str,
        client_id: str | None = None,
        created_by: str | None = None,
    ) -> User:
        validate_password(password)

        existing = await self._get_user_by_email(email)
        if existing:
            raise AuthError("Email already registered", status_code=409)

        user = User(
            email=email.lower().strip(),
            password_hash=hash_password(password),
            name=name,
            role=role,
            client_id=client_id,
            created_by=created_by,
        )
        self._db.add(user)
        await self._db.flush()  # populate user_id without committing
        logger.info("Registered user %s (role=%s)", user.user_id, role)
        return user

    async def login(
        self,
        email: str,
        password: str,
        mfa_code: str | None = None,
    ) -> dict[str, Any]:
        await self._check_rate_limit(email)

        user = await self._get_user_by_email(email.lower().strip())
        pwd_ok = user is not None and await asyncio.to_thread(
            verify_password, password, user.password_hash
        )
        if not pwd_ok:
            raise AuthError("Invalid email or password", status_code=401)

        if user.status != UserStatus.ACTIVE:
            raise AuthError(f"Account is {user.status.lower()}", status_code=403)

        if user.mfa_enabled:
            if not mfa_code:
                raise AuthError("MFA code required", status_code=422)
            if not pyotp.TOTP(user.mfa_secret).verify(mfa_code, valid_window=1):
                raise AuthError("Invalid MFA code", status_code=401)

        await self._clear_rate_limit(email)

        # Resolve client slug
        client_slug: str | None = None
        if user.client_id:
            from app.models.clients import Client
            result = await self._db.execute(
                select(Client.slug).where(Client.client_id == user.client_id)
            )
            client_slug = result.scalar_one_or_none()

        payload = self._build_token_payload(user, client_slug)
        access_token = create_access_token(payload, self._settings)
        refresh_token, jti = create_refresh_token(payload, self._settings)
        await self._store_refresh_token(jti, str(user.user_id))

        # Update last login (epoch seconds)
        user.last_login = int(datetime.now(timezone.utc).timestamp())

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "user": {
                "user_id": str(user.user_id),
                "email": user.email,
                "name": user.name,
                "role": user.role,
                "client_id": str(user.client_id) if user.client_id else None,
                "client_slug": client_slug,
                "permissions": payload["permissions"],
            },
        }

    async def refresh_token(self, token: str) -> dict[str, Any]:
        try:
            payload = decode_token(token, self._settings)
        except TokenError as exc:
            raise AuthError(str(exc), status_code=401)

        if payload.get("type") != "refresh":
            raise AuthError("Not a refresh token", status_code=401)

        jti = payload.get("jti")
        if not jti:
            raise AuthError("Malformed token", status_code=401)

        user_id = await self._consume_refresh_token(jti)
        if not user_id:
            raise AuthError("Token revoked or expired", status_code=401)

        user = await self._get_user_by_id(user_id)
        if not user or user.status != UserStatus.ACTIVE:
            raise AuthError("User unavailable", status_code=401)

        client_slug: str | None = None
        if user.client_id:
            from app.models.clients import Client
            result = await self._db.execute(
                select(Client.slug).where(Client.client_id == user.client_id)
            )
            client_slug = result.scalar_one_or_none()

        new_payload = self._build_token_payload(user, client_slug)
        access_token = create_access_token(new_payload, self._settings)
        new_refresh_token, new_jti = create_refresh_token(new_payload, self._settings)
        await self._store_refresh_token(new_jti, str(user.user_id))

        return {
            "access_token": access_token,
            "refresh_token": new_refresh_token,
            "token_type": "bearer",
        }

    async def forgot_password(self, email: str) -> None:
        """Always succeeds — never reveals whether an email exists."""
        user = await self._get_user_by_email(email.lower().strip())
        if not user:
            return

        token = secrets.token_urlsafe(32)
        key = PWD_RESET_KEY.format(token=token)
        await self._redis.setex(key, PWD_RESET_TTL_SECONDS, str(user.user_id))

        self._kafka_publish(
            "notifications.outbound",
            key=email,
            value={
                "event": "password_reset",
                "email": user.email,
                "data": {
                    "name": user.name,
                    "reset_token": token,
                    "expires_in_seconds": PWD_RESET_TTL_SECONDS,
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    async def reset_password(self, token: str, new_password: str) -> None:
        validate_password(new_password)

        key = PWD_RESET_KEY.format(token=token)
        user_id = await self._redis.getdel(key)
        if not user_id:
            raise AuthError("Invalid or expired reset token", status_code=400)

        user = await self._get_user_by_id(user_id)
        if not user:
            raise AuthError("User not found", status_code=404)

        user.password_hash = hash_password(new_password)

    async def enable_mfa(self, user_id: str) -> dict[str, str]:
        """Generate a TOTP secret and QR provisioning URI; user must verify before it's active."""
        user = await self._get_user_by_id(user_id)
        if not user:
            raise AuthError("User not found", status_code=404)

        secret = pyotp.random_base32()
        # Store pending secret in Redis until verified
        pending_key = MFA_PENDING_KEY.format(user_id=user_id)
        await self._redis.setex(pending_key, 600, secret)  # 10-min window to verify

        totp = pyotp.TOTP(secret)
        provisioning_uri = totp.provisioning_uri(name=user.email, issuer_name="ACAS")

        # Build a base64-encoded PNG QR code
        img = qrcode.make(provisioning_uri)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        qr_data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

        return {"secret": secret, "qr_url": provisioning_uri, "qr_image": qr_data_url}

    async def verify_mfa(self, user_id: str, code: str) -> bool:
        """
        Confirm a TOTP code. If a pending secret is in Redis, this completes MFA enrollment.
        Otherwise it validates the already-active secret (used as a standalone check).
        """
        pending_key = MFA_PENDING_KEY.format(user_id=user_id)
        pending_secret = await self._redis.get(pending_key)

        if pending_secret:
            # Enrollment flow: verify pending secret, then persist it
            if not pyotp.TOTP(pending_secret).verify(code, valid_window=1):
                return False
            user = await self._get_user_by_id(user_id)
            if not user:
                return False
            user.mfa_secret = pending_secret
            user.mfa_enabled = True
            await self._redis.delete(pending_key)
            return True

        # Active MFA check (used externally, e.g., re-auth flows)
        user = await self._get_user_by_id(user_id)
        if not user or not user.mfa_secret:
            return False
        return pyotp.TOTP(user.mfa_secret).verify(code, valid_window=1)

    @staticmethod
    def get_permissions(role: str) -> list[str]:
        return ROLE_PERMISSIONS.get(role, [])
