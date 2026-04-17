"""Add camera_scan_presets table for preset-based PTZ scanning.

Revision ID: 0006
Revises: 0005
Create Date: 2026-04-04
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "camera_scan_presets",
        sa.Column("preset_id",  UUID(as_uuid=False), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("camera_id",  UUID(as_uuid=False), sa.ForeignKey(
                  "cameras.camera_id", ondelete="CASCADE"), nullable=False),
        sa.Column("client_id",  UUID(as_uuid=False), sa.ForeignKey(
                  "clients.client_id", ondelete="CASCADE"), nullable=False),
        sa.Column("name",       sa.String(255), nullable=False),
        sa.Column("order_idx",  sa.Integer,     nullable=False, server_default="0"),
        sa.Column("pan",        sa.Float,       nullable=False),
        sa.Column("tilt",       sa.Float,       nullable=False),
        sa.Column("zoom",       sa.Float,       nullable=False, server_default="0.0"),
        sa.Column("dwell_s",    sa.Float,       nullable=False, server_default="3.0"),
        sa.Column("created_at", sa.BigInteger,  nullable=False,
                  server_default=sa.text("EXTRACT(EPOCH FROM NOW())::BIGINT")),
    )
    op.create_index("ix_scan_presets_camera_order",
                    "camera_scan_presets", ["camera_id", "order_idx"])

    # Row-Level Security — same pattern as other tenant tables
    op.execute("ALTER TABLE camera_scan_presets ENABLE ROW LEVEL SECURITY")
    op.execute("""
        CREATE POLICY client_isolation ON camera_scan_presets
            USING (
                client_id::text = current_setting('app.current_client_id', true)
                OR current_setting('app.current_client_id', true) IS NULL
                OR current_setting('app.current_client_id', true) = ''
            )
    """)


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS client_isolation ON camera_scan_presets")
    op.drop_table("camera_scan_presets")
