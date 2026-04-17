"""Backfill cameras.dataset_id — assign cameras without a dataset to their client's default dataset.

Revision ID: 0007
Revises: 0006
Create Date: 2026-04-05
"""
from __future__ import annotations

from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        UPDATE cameras c
        SET dataset_id = (
            SELECT fd.dataset_id
            FROM face_datasets fd
            WHERE fd.client_id = c.client_id
              AND fd.is_default = true
              AND fd.status = 'ACTIVE'
            LIMIT 1
        )
        WHERE c.dataset_id IS NULL
    """)


def downgrade() -> None:
    # Cannot safely reverse — we don't know which cameras had NULL before
    pass
