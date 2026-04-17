"""Add camera_distance_m and scan_cell_meters to cameras table.

Revision ID: 0005
Revises: 0004
Create Date: 2026-03-24
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "cameras",
        sa.Column(
            "camera_distance_m",
            sa.Float(),
            nullable=False,
            server_default="7.0",
            comment="Estimated distance from camera to monitored area in metres.",
        ),
    )
    op.add_column(
        "cameras",
        sa.Column(
            "scan_cell_meters",
            sa.Float(),
            nullable=True,
            comment="Physical width of one PTZ scan cell in metres. NULL = auto-derive from FOV.",
        ),
    )


def downgrade() -> None:
    op.drop_column("cameras", "scan_cell_meters")
    op.drop_column("cameras", "camera_distance_m")
