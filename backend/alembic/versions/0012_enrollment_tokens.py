"""enrollment_tokens — shareable public enrollment links

Revision ID: 0012
Revises: 0011
Create Date: 2026-04-15
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS enrollment_tokens (
            token_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            client_id       UUID NOT NULL REFERENCES clients(client_id) ON DELETE CASCADE,
            token           VARCHAR(64) UNIQUE NOT NULL,
            label           VARCHAR(255),
            role_default    VARCHAR(20) NOT NULL DEFAULT 'STUDENT',
            dataset_id      UUID REFERENCES face_datasets(dataset_id) ON DELETE SET NULL,
            expires_at      BIGINT,
            max_uses        INTEGER,
            use_count       INTEGER NOT NULL DEFAULT 0,
            is_active       BOOLEAN NOT NULL DEFAULT TRUE,
            created_by      UUID REFERENCES users(user_id) ON DELETE SET NULL,
            created_at      BIGINT NOT NULL DEFAULT EXTRACT(EPOCH FROM NOW())::BIGINT
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_enrollment_tokens_token
            ON enrollment_tokens(token)
            WHERE is_active = TRUE
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_enrollment_tokens_client
            ON enrollment_tokens(client_id)
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS enrollment_tokens")
