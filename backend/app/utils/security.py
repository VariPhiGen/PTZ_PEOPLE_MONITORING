from __future__ import annotations

import re

import bcrypt

# Characters that count as "special" for the password policy
_SPECIAL_RE = re.compile(r'[!@#$%^&*()\-_=+\[\]{};:\'",.<>?/\\|`~]')
_UPPER_RE = re.compile(r'[A-Z]')
_DIGIT_RE = re.compile(r'[0-9]')


def validate_password(password: str) -> None:
    """Raise ValueError if the password does not meet the policy."""
    errors: list[str] = []
    if len(password) < 10:
        errors.append("at least 10 characters")
    if not _UPPER_RE.search(password):
        errors.append("at least one uppercase letter")
    if not _DIGIT_RE.search(password):
        errors.append("at least one digit")
    if not _SPECIAL_RE.search(password):
        errors.append("at least one special character")
    if errors:
        raise ValueError("Password must contain: " + ", ".join(errors))


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=12)).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())
