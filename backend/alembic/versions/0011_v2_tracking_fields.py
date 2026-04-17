"""v2 tracking: RE-ID bridging fields on sightings + best_shots table

Revision ID: 0011
Revises: 0010
Create Date: 2026-04-14
"""
from alembic import op
import sqlalchemy as sa

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── Sightings: RE-ID tracking metadata ────────────────────────────────────
    op.add_column("sightings", sa.Column(
        "tracking_method", sa.String(20), nullable=False, server_default="face"
    ))
    op.add_column("sightings", sa.Column(
        "reid_bridged_cycles", sa.Integer(), nullable=False, server_default="0"
    ))
    op.add_column("sightings", sa.Column(
        "reid_bridge_start", sa.TIMESTAMP(timezone=True), nullable=True
    ))
    op.add_column("sightings", sa.Column(
        "reid_bridge_end", sa.TIMESTAMP(timezone=True), nullable=True
    ))
    op.add_column("sightings", sa.Column(
        "face_quality_avg", sa.Float(), nullable=True
    ))
    op.add_column("sightings", sa.Column(
        "face_quality_best", sa.Float(), nullable=True
    ))
    op.add_column("sightings", sa.Column(
        "activity_labels", sa.JSON(), nullable=True, server_default="[]"
    ))

    # ── Best-shot gallery table ────────────────────────────────────────────────
    op.create_table(
        "best_shots",
        sa.Column("id",           sa.dialects.postgresql.UUID(as_uuid=True),
                  primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("person_id",    sa.dialects.postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("persons.person_id", ondelete="CASCADE"), nullable=False),
        sa.Column("session_id",   sa.dialects.postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("client_id",    sa.dialects.postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("quality_score", sa.Float(), nullable=False),
        sa.Column("face_crop_url", sa.Text(), nullable=False),
        sa.Column("captured_at",  sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("camera_id",    sa.dialects.postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("cameras.camera_id", ondelete="SET NULL"), nullable=True),
        sa.Column("preset_id",    sa.String(50), nullable=True),
        sa.Column("yaw_degrees",  sa.Float(), nullable=True),
        sa.Column("pitch_degrees", sa.Float(), nullable=True),
        sa.Column("created_at",   sa.TIMESTAMP(timezone=True),
                  server_default=sa.text("NOW()"), nullable=False),
    )
    op.create_index(
        "idx_bestshots_person_quality",
        "best_shots",
        ["person_id", sa.text("quality_score DESC")],
    )


def downgrade() -> None:
    op.drop_index("idx_bestshots_person_quality", table_name="best_shots")
    op.drop_table("best_shots")

    for col in [
        "activity_labels", "face_quality_best", "face_quality_avg",
        "reid_bridge_end", "reid_bridge_start",
        "reid_bridged_cycles", "tracking_method",
    ]:
        op.drop_column("sightings", col)
