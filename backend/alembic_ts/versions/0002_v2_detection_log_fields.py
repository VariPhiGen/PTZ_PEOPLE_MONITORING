"""v2 tracking fields on detection_log

Revision ID: 0002
Revises: 0001
Create Date: 2026-04-14
"""
from alembic import op
import sqlalchemy as sa

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("detection_log", sa.Column(
        "tracking_method", sa.String(20), nullable=True
    ))
    op.add_column("detection_log", sa.Column(
        "reid_confidence", sa.Float(), nullable=True
    ))
    op.add_column("detection_log", sa.Column(
        "face_quality_score", sa.Float(), nullable=True
    ))
    op.add_column("detection_log", sa.Column(
        "mot_track_id", sa.Integer(), nullable=True
    ))


def downgrade() -> None:
    for col in ["mot_track_id", "face_quality_score", "reid_confidence", "tracking_method"]:
        op.drop_column("detection_log", col)
