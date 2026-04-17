from __future__ import annotations

from app.models.users import UserRole

# ── Permission catalogue ──────────────────────────────────────────────────────
# Each string is "<resource>:<action>". Routes declare which perms they require.
# SUPER_ADMIN bypasses all checks at the middleware level.

ROLE_PERMISSIONS: dict[str, list[str]] = {
    UserRole.SUPER_ADMIN: [
        "clients:create", "clients:read", "clients:update", "clients:delete",
        "cameras:create", "cameras:read", "cameras:update", "cameras:delete",
        "persons:create", "persons:read", "persons:update", "persons:delete",
        "face_embeddings:create", "face_embeddings:read", "face_embeddings:delete",
        "sessions:create", "sessions:read", "sessions:update", "sessions:delete",
        "attendance:read", "attendance:update",
        "users:create", "users:read", "users:update", "users:delete",
        "node_assignments:create", "node_assignments:read", "node_assignments:delete",
        "audit:read",
        "system:admin",
    ],
    UserRole.CLIENT_ADMIN: [
        "clients:read", "clients:update",
        "cameras:create", "cameras:read", "cameras:update", "cameras:delete",
        "persons:create", "persons:read", "persons:update", "persons:delete",
        "face_embeddings:create", "face_embeddings:read", "face_embeddings:delete",
        "sessions:create", "sessions:read", "sessions:update",
        "attendance:read", "attendance:update",
        "users:create", "users:read", "users:update",
        "node_assignments:read",
        "audit:read",
    ],
    UserRole.VIEWER: [
        "clients:read",
        "cameras:read",
        "persons:read",
        "sessions:read",
        "attendance:read",
        "users:read",
    ],
}

# Redis key patterns
RATELIMIT_LOGIN_KEY = "ratelimit:login:{email}"
REFRESH_TOKEN_KEY = "refresh_token:{jti}"
PWD_RESET_KEY = "pwd_reset:{token}"
MFA_PENDING_KEY = "mfa_pending:{user_id}"

LOGIN_MAX_ATTEMPTS = 5
LOGIN_WINDOW_SECONDS = 900   # 15 minutes
ACCESS_TOKEN_TTL_HOURS = 24
REFRESH_TOKEN_TTL_DAYS = 7
PWD_RESET_TTL_SECONDS = 3600  # 1 hour
