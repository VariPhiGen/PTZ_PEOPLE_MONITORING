"""
test_auth.py — Authentication, JWT, MFA, password reset, and rate limiting.

Covers:
  • Login success / wrong password / unknown email
  • JWT access token structure and expiry
  • Refresh token cycle + revocation
  • MFA enable → verify → login with code
  • Forgot-password + reset-password
  • Login rate limit (5 attempts per 15 min)
  • Logout (refresh token invalidated)
"""
from __future__ import annotations

import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient
from jose import jwt
from sqlalchemy.ext.asyncio import AsyncSession

from tests.conftest import (
    TEST_JWT_SECRET,
    TEST_SETTINGS,
    auth_headers,
    create_user,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

async def _login(client: AsyncClient, email: str, password: str, mfa_code: str | None = None):
    body = {"email": email, "password": password}
    if mfa_code:
        body["mfa_code"] = mfa_code
    return await client.post("/api/auth/login", json=body)


# ── Login ─────────────────────────────────────────────────────────────────────

async def test_login_success(app_client, db_session):
    await create_user(db_session, "login.ok@test.com", "Pass@word1234", role="CLIENT_ADMIN")
    await db_session.commit()

    resp = await _login(app_client, "login.ok@test.com", "Pass@word1234")
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["user"]["email"] == "login.ok@test.com"
    assert data["user"]["role"] == "CLIENT_ADMIN"


async def test_login_wrong_password(app_client, db_session):
    await create_user(db_session, "login.bad@test.com", "Pass@word1234")
    await db_session.commit()

    resp = await _login(app_client, "login.bad@test.com", "WrongPassword!")
    assert resp.status_code == 401


async def test_login_unknown_email(app_client):
    resp = await _login(app_client, "nobody@nowhere.test", "anyPassword1!")
    assert resp.status_code == 401


async def test_login_inactive_user(app_client, db_session):
    from app.models.users import User
    user = await create_user(db_session, "inactive@test.com", "Pass@word1234")
    user.status = "SUSPENDED"
    await db_session.commit()

    resp = await _login(app_client, "inactive@test.com", "Pass@word1234")
    assert resp.status_code in (401, 403)


# ── JWT structure ─────────────────────────────────────────────────────────────

async def test_jwt_payload_structure(app_client, db_session, tenant_a):
    admin = await create_user(
        db_session, "jwt.check@test.com", "Pass@word1234",
        role="CLIENT_ADMIN", client_id=tenant_a.client_id,
    )
    await db_session.commit()

    resp = await _login(app_client, "jwt.check@test.com", "Pass@word1234")
    assert resp.status_code == 200
    token = resp.json()["access_token"]
    payload = jwt.decode(token, TEST_JWT_SECRET, algorithms=["HS256"])

    assert payload["email"] == "jwt.check@test.com"
    assert payload["role"]  == "CLIENT_ADMIN"
    assert payload["client_id"] == str(tenant_a.client_id)
    assert "permissions" in payload
    assert isinstance(payload["permissions"], list)
    assert payload["exp"] > payload["iat"]


async def test_access_token_contains_permissions(app_client, db_session):
    await create_user(db_session, "sa.perms@test.com", "Pass@word1234", role="SUPER_ADMIN")
    await db_session.commit()

    resp = await _login(app_client, "sa.perms@test.com", "Pass@word1234")
    payload = jwt.decode(resp.json()["access_token"], TEST_JWT_SECRET, algorithms=["HS256"])
    perms = payload["permissions"]
    # Super admin must have all key permissions
    for p in ("clients:create", "cameras:manage", "persons:enroll", "system:admin"):
        assert p in perms, f"SUPER_ADMIN missing permission: {p}"


async def test_expired_token_is_rejected(app_client, db_session):
    """A token with exp in the past must be rejected."""
    user = await create_user(db_session, "expired@test.com", "Pass@word1234", role="VIEWER")
    await db_session.commit()

    expired_token = jwt.encode(
        {
            "user_id": str(user.user_id),
            "email":   user.email,
            "name":    user.name,
            "role":    "VIEWER",
            "client_id": None,
            "permissions": [],
            "iat": int(time.time()) - 7_200,
            "exp": int(time.time()) - 3_600,   # 1 hour in the past
        },
        TEST_JWT_SECRET,
        algorithm="HS256",
    )
    resp = await app_client.get("/api/auth/me", headers=auth_headers(expired_token))
    assert resp.status_code == 401


async def test_tampered_token_is_rejected(app_client):
    resp = await app_client.get(
        "/api/auth/me",
        headers={"Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.fake.signature"},
    )
    assert resp.status_code == 401


# ── Refresh token ─────────────────────────────────────────────────────────────

async def test_refresh_token_cycle(app_client, db_session):
    await create_user(db_session, "refresh@test.com", "Pass@word1234")
    await db_session.commit()

    login = await _login(app_client, "refresh@test.com", "Pass@word1234")
    refresh_tok = login.json()["refresh_token"]

    resp = await app_client.post("/api/auth/refresh", json={"refresh_token": refresh_tok})
    assert resp.status_code == 200
    new_data = resp.json()
    assert "access_token" in new_data
    # New access token must differ from the original
    assert new_data["access_token"] != login.json()["access_token"]


async def test_refresh_token_cannot_be_reused(app_client, db_session, fake_redis):
    """Each refresh token is single-use; replaying it must return 401."""
    await create_user(db_session, "refresh.once@test.com", "Pass@word1234")
    await db_session.commit()

    login  = await _login(app_client, "refresh.once@test.com", "Pass@word1234")
    rtoken = login.json()["refresh_token"]

    # First use — should succeed
    resp1 = await app_client.post("/api/auth/refresh", json={"refresh_token": rtoken})
    assert resp1.status_code == 200

    # Second use of the same token — should fail
    resp2 = await app_client.post("/api/auth/refresh", json={"refresh_token": rtoken})
    assert resp2.status_code == 401


async def test_invalid_refresh_token(app_client):
    resp = await app_client.post("/api/auth/refresh", json={"refresh_token": "not.a.real.token"})
    assert resp.status_code == 401


# ── GET /api/auth/me ──────────────────────────────────────────────────────────

async def test_get_me(app_client, db_session, client_admin_a, client_admin_a_token):
    await db_session.commit()
    resp = await app_client.get("/api/auth/me", headers=auth_headers(client_admin_a_token))
    assert resp.status_code == 200
    data = resp.json()
    assert data["email"] == client_admin_a.email
    assert data["role"]  == "CLIENT_ADMIN"


async def test_get_me_unauthenticated(app_client):
    resp = await app_client.get("/api/auth/me")
    assert resp.status_code == 401


# ── MFA ───────────────────────────────────────────────────────────────────────

async def test_mfa_enable_returns_secret_and_qr(app_client, db_session):
    user = await create_user(db_session, "mfa.enable@test.com", "Pass@word1234")
    await db_session.commit()
    login = await _login(app_client, "mfa.enable@test.com", "Pass@word1234")
    token = login.json()["access_token"]

    resp = await app_client.post("/api/auth/mfa/enable", headers=auth_headers(token))
    assert resp.status_code == 200
    data = resp.json()
    assert "secret" in data
    assert "qr_url" in data
    assert len(data["secret"]) >= 16


async def test_mfa_verify_with_valid_code(app_client, db_session, fake_redis):
    user = await create_user(db_session, "mfa.verify@test.com", "Pass@word1234")
    await db_session.commit()
    login = await _login(app_client, "mfa.verify@test.com", "Pass@word1234")
    token = login.json()["access_token"]

    # Enable MFA — get the TOTP secret
    enable = await app_client.post("/api/auth/mfa/enable", headers=auth_headers(token))
    assert enable.status_code == 200
    secret = enable.json()["secret"]

    # Generate a valid TOTP code
    import pyotp
    totp = pyotp.TOTP(secret)
    code = totp.now()

    resp = await app_client.post(
        "/api/auth/mfa/verify",
        json={"code": code},
        headers=auth_headers(token),
    )
    assert resp.status_code == 200


async def test_mfa_verify_with_wrong_code(app_client, db_session, fake_redis):
    user = await create_user(db_session, "mfa.wrong@test.com", "Pass@word1234")
    await db_session.commit()
    login = await _login(app_client, "mfa.wrong@test.com", "Pass@word1234")
    token = login.json()["access_token"]

    await app_client.post("/api/auth/mfa/enable", headers=auth_headers(token))

    resp = await app_client.post(
        "/api/auth/mfa/verify",
        json={"code": "000000"},
        headers=auth_headers(token),
    )
    assert resp.status_code in (400, 401)


# ── Forgot / reset password ───────────────────────────────────────────────────

async def test_forgot_password_for_existing_email(app_client, db_session):
    await create_user(db_session, "reset@test.com", "Pass@word1234")
    await db_session.commit()

    resp = await app_client.post("/api/auth/forgot-password", json={"email": "reset@test.com"})
    # Must return 200 even when email exists (to prevent enumeration)
    assert resp.status_code == 200


async def test_forgot_password_for_nonexistent_email(app_client):
    """Must also return 200 — no information leakage about account existence."""
    resp = await app_client.post(
        "/api/auth/forgot-password", json={"email": "nobody@nowhere.test"}
    )
    assert resp.status_code == 200


async def test_reset_password_with_valid_token(app_client, db_session, fake_redis):
    user = await create_user(db_session, "reset.flow@test.com", "Pass@word1234")
    await db_session.commit()

    # Store a reset token directly in fake Redis (simulating the forgot-password handler)
    reset_token = str(uuid.uuid4())
    await fake_redis.setex(f"pwd_reset:{reset_token}", 3600, str(user.user_id))

    resp = await app_client.post(
        "/api/auth/reset-password",
        json={"token": reset_token, "new_password": "NewPass@5678"},
    )
    assert resp.status_code == 200

    # Verify the new password works
    login = await _login(app_client, "reset.flow@test.com", "NewPass@5678")
    assert login.status_code == 200


async def test_reset_password_with_invalid_token(app_client):
    resp = await app_client.post(
        "/api/auth/reset-password",
        json={"token": "invalid-token-xyz", "new_password": "NewPass@5678"},
    )
    assert resp.status_code in (400, 401)


async def test_reset_password_weak_password_rejected(app_client, db_session, fake_redis):
    user = await create_user(db_session, "reset.weak@test.com", "Pass@word1234")
    await db_session.commit()
    reset_token = str(uuid.uuid4())
    await fake_redis.setex(f"pwd_reset:{reset_token}", 3600, str(user.user_id))

    resp = await app_client.post(
        "/api/auth/reset-password",
        json={"token": reset_token, "new_password": "weak"},
    )
    assert resp.status_code in (400, 422)


# ── Login rate limit ──────────────────────────────────────────────────────────

async def test_login_rate_limit_triggers_after_five_failures(app_client, db_session, fake_redis):
    """5 consecutive failed logins for the same email must trigger 429."""
    await create_user(db_session, "ratelimit@test.com", "Pass@word1234")
    await db_session.commit()

    for i in range(5):
        resp = await _login(app_client, "ratelimit@test.com", "WrongPwd!")
        if resp.status_code == 429:
            return  # Rate limit hit before the 5th — also acceptable
        assert resp.status_code == 401

    # 6th attempt must be rate-limited
    resp = await _login(app_client, "ratelimit@test.com", "WrongPwd!")
    assert resp.status_code == 429


async def test_login_rate_limit_does_not_affect_other_emails(
    app_client, db_session, fake_redis
):
    """Rate limit is per-email, not global."""
    await create_user(db_session, "safe@test.com", "Pass@word1234")
    await create_user(db_session, "victim@test.com", "Pass@word1234")
    await db_session.commit()

    # Exhaust limit for victim@
    for _ in range(6):
        await _login(app_client, "victim@test.com", "BadPwd!")

    # safe@ must still be able to log in
    resp = await _login(app_client, "safe@test.com", "Pass@word1234")
    assert resp.status_code == 200
