from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from functools import lru_cache

from jose import JWTError, jwt

from app.config import Settings


class TokenError(Exception):
    pass


@lru_cache(maxsize=1)
def _get_verify_key(secret_key: str, algorithm: str) -> str:
    """For RS256, derive the public key from the private PEM at startup."""
    if algorithm == "RS256":
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.serialization import load_pem_private_key

            priv = load_pem_private_key(secret_key.encode(), password=None)
            return priv.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            ).decode()
        except Exception as exc:
            raise ValueError(
                "JWT_SECRET_KEY must be a valid RSA PEM private key when JWT_ALGORITHM=RS256"
            ) from exc
    # HS256 — same key for sign and verify
    return secret_key


def create_access_token(payload: dict, settings: Settings) -> str:
    now = datetime.now(timezone.utc)
    data = {
        **payload,
        "iat": now,
        "exp": now + timedelta(hours=24),
        "type": "access",
    }
    return jwt.encode(data, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def create_refresh_token(payload: dict, settings: Settings) -> tuple[str, str]:
    """Return (encoded_token, jti). The jti is stored in Redis for revocation."""
    now = datetime.now(timezone.utc)
    jti = str(uuid.uuid4())
    data = {
        **payload,
        "iat": now,
        "exp": now + timedelta(days=7),
        "type": "refresh",
        "jti": jti,
    }
    return jwt.encode(data, settings.jwt_secret_key, algorithm=settings.jwt_algorithm), jti


def decode_token(token: str, settings: Settings) -> dict:
    verify_key = _get_verify_key(settings.jwt_secret_key, settings.jwt_algorithm)
    try:
        return jwt.decode(token, verify_key, algorithms=[settings.jwt_algorithm])
    except JWTError as exc:
        raise TokenError(str(exc)) from exc
