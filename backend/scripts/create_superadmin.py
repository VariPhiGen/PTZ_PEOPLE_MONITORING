#!/usr/bin/env python3
"""
Create the first SUPER_ADMIN user.

Usage (inside the container):
  python scripts/create_superadmin.py \
      --email admin@acas.local \
      --password "MyP@ssw0rd!2024" \
      --name "Platform Admin"

Or from host with docker exec:
  docker exec acas-backend python /app/scripts/create_superadmin.py \
      --email admin@acas.local --password "..." --name "Platform Admin"
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

# Allow imports from the app package when running as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import get_settings
from app.models.users import User, UserRole, UserStatus
from app.utils.security import hash_password, validate_password


async def create_superadmin(email: str, password: str, name: str) -> None:
    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=False)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async with factory() as session:
        async with session.begin():
            # Check for existing super admin
            result = await session.execute(
                select(User).where(User.role == UserRole.SUPER_ADMIN)
            )
            existing_admins = result.scalars().all()
            if existing_admins:
                print(
                    f"[warn] {len(existing_admins)} SUPER_ADMIN(s) already exist:\n"
                    + "\n".join(f"  - {u.email}" for u in existing_admins)
                )
                print("Creating an additional super admin anyway …")

            # Check email uniqueness
            dup = await session.execute(
                select(User).where(User.email == email.lower().strip())
            )
            if dup.scalar_one_or_none():
                print(f"[error] A user with email '{email}' already exists.")
                sys.exit(1)

            validate_password(password)

            user = User(
                email=email.lower().strip(),
                password_hash=hash_password(password),
                name=name,
                role=UserRole.SUPER_ADMIN,
                status=UserStatus.ACTIVE,
                client_id=None,
            )
            session.add(user)

    await engine.dispose()
    print(f"[ok] SUPER_ADMIN created: {email} (id={user.user_id})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the first ACAS super admin user.")
    parser.add_argument("--email", required=True, help="Admin email address")
    parser.add_argument("--password", required=True, help="Admin password (must meet policy)")
    parser.add_argument("--name", required=True, help="Display name")
    args = parser.parse_args()

    asyncio.run(create_superadmin(args.email, args.password, args.name))


if __name__ == "__main__":
    main()
