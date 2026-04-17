"""detection_log hypertable (TimescaleDB)

Revision ID: ts0001
Revises:
Create Date: 2026-03-21
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "ts0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()

    conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE"))

    op.create_table(
        "detection_log",
        sa.Column("time", sa.DateTime(timezone=True), primary_key=True, nullable=False),
        sa.Column("client_id", postgresql.UUID(as_uuid=True)),
        sa.Column("session_id", postgresql.UUID(as_uuid=True)),
        sa.Column("person_id", postgresql.UUID(as_uuid=True)),
        sa.Column("track_id", sa.String(255)),
        sa.Column("zone", sa.String(255)),
        sa.Column("confidence", sa.Float),
        sa.Column("liveness_score", sa.Float),
        sa.Column("bbox", postgresql.JSONB),
        sa.Column("frame_ref", sa.Text),
    )

    conn.execute(sa.text(
        "SELECT public.create_hypertable('detection_log', 'time', if_not_exists => TRUE)"
    ))

    op.create_index("ix_detection_log_client_id", "detection_log", ["client_id"])
    op.create_index("ix_detection_log_person_id", "detection_log", ["person_id"])
    op.create_index("ix_detection_log_session_id", "detection_log", ["session_id"])


def downgrade() -> None:
    op.drop_table("detection_log")
