"""Add cameras.dataset_ids JSONB array for multi-dataset face recognition.

cameras.dataset_id  — primary FK (backward compat, used for camera_count on datasets page)
cameras.dataset_ids — JSONB list[str] of ALL dataset UUIDs this camera recognises from.
                      Backfilled from dataset_id; updated by the API on create/update.

Revision ID: 0008
Revises: 0007
Create Date: 2026-04-06
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "cameras",
        sa.Column("dataset_ids", JSONB, nullable=True,
                  comment="List of dataset UUIDs (strings) this camera recognises from. "
                          "Superset of dataset_id; used for multi-dataset recognition."),
    )
    # Backfill: carry the primary dataset_id into the new array column
    op.execute("""
        UPDATE cameras
        SET dataset_ids = jsonb_build_array(dataset_id::text)
        WHERE dataset_id IS NOT NULL AND dataset_ids IS NULL
    """)


def downgrade() -> None:
    op.drop_column("cameras", "dataset_ids")
