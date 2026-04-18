"""Per-dataset adaptive 1:N thresholds

Revision: 0013
Previous: 0012

Stores dataset-specific Tier-1 / Tier-2 identification thresholds calibrated
from genuine / impostor similarity distributions.  When a dataset has no
calibration yet, the columns are NULL and FaceRepository falls back to its
global defaults (_TIER1_THRESHOLD / _TIER2_THRESHOLD).

Calibration metadata (operating-point stats, sample sizes, embedding count,
timestamp) lives in `calibration_stats` so the admin UI can surface it
without a separate table.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE face_datasets
            ADD COLUMN IF NOT EXISTS tier1_threshold   DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS tier2_threshold   DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS calibration_stats JSONB,
            ADD COLUMN IF NOT EXISTS calibrated_at     BIGINT
    """)


def downgrade() -> None:
    op.execute("""
        ALTER TABLE face_datasets
            DROP COLUMN IF EXISTS calibrated_at,
            DROP COLUMN IF EXISTS calibration_stats,
            DROP COLUMN IF EXISTS tier2_threshold,
            DROP COLUMN IF EXISTS tier1_threshold
    """)
