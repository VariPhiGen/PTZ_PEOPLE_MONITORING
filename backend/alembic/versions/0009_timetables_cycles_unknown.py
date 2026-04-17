"""Timetables, cycle-based attendance, unknown detections

Revision ID: 0009
Revises: 0008
Create Date: 2026-04-07

Changes:
  - attendance_records: add cycles_present (int), total_cycles (int)
  - cameras: add timetable_id FK (nullable)
  - New table: timetables
  - New table: timetable_entries
  - New table: unknown_detections
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── 1. attendance_records: cycle columns ───────────────────────────────────
    op.add_column(
        "attendance_records",
        sa.Column("cycles_present", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "attendance_records",
        sa.Column("total_cycles", sa.Integer(), nullable=False, server_default="0"),
    )

    # ── 2. timetables ─────────────────────────────────────────────────────────
    op.create_table(
        "timetables",
        sa.Column("timetable_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("client_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("clients.client_id", ondelete="CASCADE"),
                  nullable=False, index=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("timezone", sa.String(64), nullable=False, server_default="UTC"),
        sa.Column("created_at", sa.BigInteger(), nullable=False,
                  server_default=sa.text("extract(epoch from now())::bigint")),
        sa.Column("updated_at", sa.BigInteger(), nullable=False,
                  server_default=sa.text("extract(epoch from now())::bigint")),
    )

    # ── 3. timetable_entries ───────────────────────────────────────────────────
    op.create_table(
        "timetable_entries",
        sa.Column("entry_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("timetable_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("timetables.timetable_id", ondelete="CASCADE"),
                  nullable=False, index=True),
        sa.Column("client_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("clients.client_id", ondelete="CASCADE"),
                  nullable=False),
        # day_of_week: 0=Monday … 6=Sunday
        sa.Column("day_of_week", sa.Integer(), nullable=False),
        sa.Column("start_time", sa.String(5), nullable=False),   # "HH:MM"
        sa.Column("end_time",   sa.String(5), nullable=False),   # "HH:MM"
        sa.Column("course_id",   sa.String(255), nullable=True),
        sa.Column("course_name", sa.String(255), nullable=True),
        sa.Column("faculty_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("persons.person_id", ondelete="SET NULL"),
                  nullable=True),
        # dataset_ids: JSON array of dataset UUID strings used to build the roster
        sa.Column("dataset_ids", postgresql.JSONB(), nullable=True),
        # explicit roster override (JSON array of person UUID strings)
        sa.Column("roster_ids",  postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.BigInteger(), nullable=False,
                  server_default=sa.text("extract(epoch from now())::bigint")),
    )

    # ── 4. cameras: timetable FK ───────────────────────────────────────────────
    op.add_column(
        "cameras",
        sa.Column("timetable_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("timetables.timetable_id", ondelete="SET NULL"),
                  nullable=True, index=True),
    )

    # ── 5. unknown_detections ──────────────────────────────────────────────────
    op.create_table(
        "unknown_detections",
        sa.Column("detection_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("client_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("clients.client_id", ondelete="CASCADE"),
                  nullable=False, index=True),
        sa.Column("session_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("sessions.session_id", ondelete="CASCADE"),
                  nullable=True, index=True),
        sa.Column("camera_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("cameras.camera_id", ondelete="SET NULL"),
                  nullable=True),
        # MinIO path to the face crop
        sa.Column("image_ref", sa.Text(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("liveness_score", sa.Float(), nullable=True),
        sa.Column("detected_at", sa.BigInteger(), nullable=False,
                  server_default=sa.text("extract(epoch from now())::bigint")),
        # set when a human assigns this to a known person
        sa.Column("assigned_person_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("persons.person_id", ondelete="SET NULL"),
                  nullable=True),
        sa.Column("assigned_by", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("users.user_id", ondelete="SET NULL"),
                  nullable=True),
        sa.Column("assigned_at", sa.BigInteger(), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="UNASSIGNED"),
    )
    op.create_index(
        "ix_unknown_detections_session_status",
        "unknown_detections", ["session_id", "status"],
    )


def downgrade() -> None:
    op.drop_index("ix_unknown_detections_session_status", table_name="unknown_detections")
    op.drop_table("unknown_detections")
    op.drop_column("cameras", "timetable_id")
    op.drop_table("timetable_entries")
    op.drop_table("timetables")
    op.drop_column("attendance_records", "total_cycles")
    op.drop_column("attendance_records", "cycles_present")
