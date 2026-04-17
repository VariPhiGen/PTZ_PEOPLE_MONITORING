"""add camera_type column

Revision ID: 0010
Revises: 0009
Create Date: 2026-04-08
"""
from alembic import op
import sqlalchemy as sa

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "cameras",
        sa.Column("camera_type", sa.String(20), nullable=False, server_default="PTZ"),
    )


def downgrade() -> None:
    op.drop_column("cameras", "camera_type")
