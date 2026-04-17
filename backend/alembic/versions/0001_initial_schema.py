"""initial schema — all tables, RLS, indexes, hypertable, materialized view

Revision ID: 0001
Revises:
Create Date: 2026-03-21
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Tables that carry client_id and need Row-Level Security
_RLS_TABLES = [
    "cameras",
    "persons",
    "face_embeddings",
    "sessions",
    "attendance_records",
    "sightings",
]


def upgrade() -> None:
    conn = op.get_bind()

    # ── 1. Extensions ────────────────────────────────────────────────────────
    conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))
    conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))

    # ── 2. clients ───────────────────────────────────────────────────────────
    op.create_table(
        "clients",
        sa.Column("client_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("slug", sa.String(100), nullable=False, unique=True),
        sa.Column("logo_url", sa.Text),
        sa.Column("contact_name", sa.String(255)),
        sa.Column("contact_email", sa.String(255)),
        sa.Column("contact_phone", sa.String(50)),
        sa.Column("address", sa.Text),
        sa.Column("status", sa.String(20), nullable=False, server_default="ACTIVE"),
        sa.Column("max_cameras", sa.Integer, nullable=False, server_default="50"),
        sa.Column("max_persons", sa.Integer, nullable=False, server_default="10000"),
        sa.Column("settings", postgresql.JSONB),
        sa.Column("created_at", sa.BigInteger, nullable=False,
                  server_default=sa.text("EXTRACT(EPOCH FROM NOW())::BIGINT")),
        sa.Column("updated_at", sa.BigInteger, nullable=False,
                  server_default=sa.text("EXTRACT(EPOCH FROM NOW())::BIGINT")),
    )

    # ── 3. users ─────────────────────────────────────────────────────────────
    op.create_table(
        "users",
        sa.Column("user_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("email", sa.String(255), nullable=False, unique=True),
        sa.Column("password_hash", sa.Text, nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("client_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("clients.client_id", ondelete="RESTRICT")),
        sa.Column("avatar_url", sa.Text),
        sa.Column("status", sa.String(20), nullable=False, server_default="ACTIVE"),
        sa.Column("last_login", sa.BigInteger),
        sa.Column("mfa_enabled", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("mfa_secret", sa.Text),
        sa.Column("created_by", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("users.user_id", ondelete="SET NULL")),
        sa.Column("created_at", sa.BigInteger, nullable=False,
                  server_default=sa.text("EXTRACT(EPOCH FROM NOW())::BIGINT")),
        sa.Column("updated_at", sa.BigInteger, nullable=False,
                  server_default=sa.text("EXTRACT(EPOCH FROM NOW())::BIGINT")),
    )
    op.create_index("ix_users_client_id", "users", ["client_id"])

    # ── 4. cameras ───────────────────────────────────────────────────────────
    op.create_table(
        "cameras",
        sa.Column("camera_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("client_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("clients.client_id", ondelete="RESTRICT"), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("room_name", sa.String(255)),
        sa.Column("building", sa.String(255)),
        sa.Column("floor", sa.String(50)),
        sa.Column("rtsp_url", sa.Text),
        sa.Column("onvif_host", sa.String(255)),
        sa.Column("onvif_port", sa.Integer),
        sa.Column("onvif_username", sa.String(255)),
        sa.Column("onvif_password_encrypted", sa.Text),
        sa.Column("status", sa.String(20), nullable=False, server_default="OFFLINE"),
        sa.Column("roi_rect", postgresql.JSONB),
        sa.Column("faculty_zone", postgresql.JSONB),
        sa.Column("fov_h", sa.Float),
        sa.Column("fov_v", sa.Float),
        sa.Column("pan_speed", sa.Float),
        sa.Column("tilt_speed", sa.Float),
        sa.Column("zoom_speed", sa.Float),
        sa.Column("mode", sa.String(20), nullable=False, server_default="MONITORING"),
        sa.Column("monitoring_hours", postgresql.JSONB),
        sa.Column("restricted_zone", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("alert_on_unknown", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("learned_params", postgresql.JSONB),
        sa.Column("node_id", sa.Text),
        sa.Column("created_at", sa.BigInteger, nullable=False,
                  server_default=sa.text("EXTRACT(EPOCH FROM NOW())::BIGINT")),
        sa.Column("updated_at", sa.BigInteger, nullable=False,
                  server_default=sa.text("EXTRACT(EPOCH FROM NOW())::BIGINT")),
    )
    op.create_index("ix_cameras_client_id", "cameras", ["client_id"])

    # ── 5. persons ───────────────────────────────────────────────────────────
    op.create_table(
        "persons",
        sa.Column("person_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("client_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("clients.client_id", ondelete="RESTRICT"), nullable=False),
        sa.Column("external_id", sa.String(255)),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("department", sa.String(255)),
        sa.Column("email", sa.String(255)),
        sa.Column("phone", sa.String(50)),
        sa.Column("status", sa.String(20), nullable=False, server_default="ACTIVE"),
        sa.Column("consent_at", sa.BigInteger),
        sa.Column("consent_hash", sa.Text),
        sa.Column("created_at", sa.BigInteger, nullable=False,
                  server_default=sa.text("EXTRACT(EPOCH FROM NOW())::BIGINT")),
        sa.Column("updated_at", sa.BigInteger, nullable=False,
                  server_default=sa.text("EXTRACT(EPOCH FROM NOW())::BIGINT")),
        sa.UniqueConstraint("client_id", "external_id", name="uq_persons_client_external"),
    )
    op.create_index("ix_persons_client_id", "persons", ["client_id"])
    # Trigram index for fuzzy name search
    op.create_index(
        "ix_persons_name_trgm", "persons", ["name"],
        postgresql_using="gin",
        postgresql_ops={"name": "gin_trgm_ops"},
    )

    # ── 6. face_embeddings ───────────────────────────────────────────────────
    op.create_table(
        "face_embeddings",
        sa.Column("embedding_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("client_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("clients.client_id", ondelete="RESTRICT"), nullable=False),
        sa.Column("person_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("persons.person_id", ondelete="CASCADE"), nullable=False),
        sa.Column("embedding", Vector(512), nullable=False),
        sa.Column("version", sa.Integer, nullable=False, server_default="1"),
        sa.Column("source", sa.String(20), nullable=False, server_default="ENROLLMENT"),
        sa.Column("confidence_avg", sa.Float),
        sa.Column("image_refs", postgresql.ARRAY(sa.Text)),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("quality_score", sa.Float),
        sa.Column("created_at", sa.BigInteger, nullable=False,
                  server_default=sa.text("EXTRACT(EPOCH FROM NOW())::BIGINT")),
    )
    op.create_index("ix_face_embeddings_client_id", "face_embeddings", ["client_id"])
    op.create_index("ix_face_embeddings_person_id", "face_embeddings", ["person_id"])
    # HNSW index for approximate nearest-neighbour search (cosine distance)
    conn.execute(sa.text(
        "CREATE INDEX ix_face_embeddings_hnsw ON face_embeddings "
        "USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    ))

    # ── 7. sessions ──────────────────────────────────────────────────────────
    op.create_table(
        "sessions",
        sa.Column("session_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("client_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("clients.client_id", ondelete="RESTRICT"), nullable=False),
        sa.Column("camera_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("cameras.camera_id", ondelete="SET NULL")),
        sa.Column("course_id", sa.String(255)),
        sa.Column("course_name", sa.String(255)),
        sa.Column("faculty_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("persons.person_id", ondelete="SET NULL")),
        sa.Column("scheduled_start", sa.BigInteger),
        sa.Column("scheduled_end", sa.BigInteger),
        sa.Column("actual_start", sa.BigInteger),
        sa.Column("actual_end", sa.BigInteger),
        sa.Column("faculty_status", sa.String(50)),
        sa.Column("sync_status", sa.String(20), nullable=False, server_default="PENDING"),
        sa.Column("held_reason", sa.Text),
        sa.Column("scan_map", postgresql.JSONB),
        sa.Column("cycle_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("recognition_rate", sa.Float),
        sa.Column("synced_at", sa.BigInteger),
        sa.Column("created_at", sa.BigInteger, nullable=False,
                  server_default=sa.text("EXTRACT(EPOCH FROM NOW())::BIGINT")),
    )
    op.create_index("ix_sessions_client_id", "sessions", ["client_id"])
    op.create_index("ix_sessions_camera_id", "sessions", ["camera_id"])
    op.create_index("ix_sessions_faculty_id", "sessions", ["faculty_id"])

    # ── 8. attendance_records ────────────────────────────────────────────────
    op.create_table(
        "attendance_records",
        sa.Column("record_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("client_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("clients.client_id", ondelete="RESTRICT"), nullable=False),
        sa.Column("session_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False),
        sa.Column("person_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("persons.person_id", ondelete="CASCADE"), nullable=False),
        sa.Column("total_duration", sa.Interval),
        sa.Column("detection_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("first_seen", sa.BigInteger),
        sa.Column("last_seen", sa.BigInteger),
        sa.Column("status", sa.String(5), nullable=False, server_default="ND"),
        sa.Column("confidence_avg", sa.Float),
        sa.Column("liveness_avg", sa.Float),
        sa.Column("evidence_refs", postgresql.ARRAY(sa.Text)),
        sa.Column("override_by", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("users.user_id", ondelete="SET NULL")),
        sa.Column("override_reason", sa.Text),
        sa.Column("created_at", sa.BigInteger, nullable=False,
                  server_default=sa.text("EXTRACT(EPOCH FROM NOW())::BIGINT")),
    )
    op.create_index("ix_attendance_records_client_id", "attendance_records", ["client_id"])
    op.create_index("ix_attendance_records_session_id", "attendance_records", ["session_id"])
    op.create_index("ix_attendance_records_person_id", "attendance_records", ["person_id"])

    # ── 9. sightings ─────────────────────────────────────────────────────────
    op.create_table(
        "sightings",
        sa.Column("sighting_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("client_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("clients.client_id", ondelete="RESTRICT"), nullable=False),
        sa.Column("person_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("persons.person_id", ondelete="CASCADE"), nullable=False),
        sa.Column("camera_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("cameras.camera_id", ondelete="CASCADE"), nullable=False),
        sa.Column("zone", sa.String(255)),
        sa.Column("first_seen", sa.BigInteger, nullable=False),
        sa.Column("last_seen", sa.BigInteger, nullable=False),
        sa.Column("duration_seconds", sa.Integer),
        sa.Column("confidence_avg", sa.Float),
        sa.Column("frame_refs", postgresql.ARRAY(sa.Text)),
        sa.Column("created_at", sa.BigInteger, nullable=False,
                  server_default=sa.text("EXTRACT(EPOCH FROM NOW())::BIGINT")),
    )
    op.create_index("ix_sightings_client_id", "sightings", ["client_id"])
    op.create_index("ix_sightings_person_first_seen", "sightings", ["person_id", "first_seen"])
    op.create_index("ix_sightings_camera_first_seen", "sightings", ["camera_id", "first_seen"])

    # NOTE: detection_log lives in TimescaleDB (TIMESCALE_URL / acas_ts database).
    # It is bootstrapped by scripts/timescaledb/init-detection-log.sql and managed
    # by the separate alembic_ts Alembic environment, not this migration.

    # ── 10. audit_log ────────────────────────────────────────────────────────
    op.create_table(
        "audit_log",
        sa.Column("audit_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("client_id", postgresql.UUID(as_uuid=True)),
        sa.Column("actor_id", postgresql.UUID(as_uuid=True)),
        sa.Column("actor_type", sa.String(20), nullable=False),
        sa.Column("action", sa.String(255), nullable=False),
        sa.Column("target_type", sa.String(100)),
        sa.Column("target_id", sa.String(255)),
        sa.Column("detail", postgresql.JSONB),
        sa.Column("created_at", sa.BigInteger, nullable=False,
                  server_default=sa.text("EXTRACT(EPOCH FROM NOW())::BIGINT")),
    )
    op.create_index("ix_audit_log_client_id", "audit_log", ["client_id"])
    op.create_index("ix_audit_log_actor_id", "audit_log", ["actor_id"])

    # ── 12. client_node_assignments ──────────────────────────────────────────
    op.create_table(
        "client_node_assignments",
        sa.Column("assignment_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("client_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("clients.client_id", ondelete="CASCADE"), nullable=False),
        sa.Column("node_id", sa.Text, nullable=False),
        sa.Column("assigned_by", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("users.user_id", ondelete="SET NULL")),
        sa.Column("max_cameras_on_node", sa.Integer, nullable=False, server_default="10"),
        sa.Column("assigned_at", sa.BigInteger, nullable=False,
                  server_default=sa.text("EXTRACT(EPOCH FROM NOW())::BIGINT")),
        sa.UniqueConstraint("client_id", "node_id", name="uq_cna_client_node"),
    )
    op.create_index("ix_cna_client_id", "client_node_assignments", ["client_id"])

    # ── 13. Row-Level Security ───────────────────────────────────────────────
    # Policy: row is visible if its client_id matches the session-local setting,
    # OR if the setting is an empty string (used by SUPER_ADMIN / background workers).
    for table in _RLS_TABLES:
        conn.execute(sa.text(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY"))
        conn.execute(sa.text(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY"))
        conn.execute(sa.text(
            f"CREATE POLICY tenant_isolation ON {table} "
            f"USING ("
            f"  client_id = current_setting('app.current_client_id')::uuid "
            f"  OR current_setting('app.current_client_id', true) IS NULL "
            f"  OR current_setting('app.current_client_id', true) = ''"
            f")"
        ))

    # ── 14. person_journeys materialized view ────────────────────────────────
    conn.execute(sa.text("""
        CREATE MATERIALIZED VIEW IF NOT EXISTS person_journeys AS
        SELECT
            s.sighting_id,
            s.client_id,
            s.person_id,
            s.camera_id,
            c.name          AS camera_name,
            c.room_name,
            c.building,
            c.floor,
            s.zone,
            s.first_seen,
            s.last_seen,
            s.duration_seconds,
            LAG(s.camera_id)    OVER w AS prev_camera_id,
            LAG(c.name)         OVER w AS prev_camera_name,
            LAG(s.last_seen)    OVER w AS prev_last_seen,
            CASE
                WHEN LAG(s.last_seen) OVER w IS NOT NULL
                THEN s.first_seen - LAG(s.last_seen) OVER w
            END                         AS transit_time_seconds
        FROM sightings s
        JOIN cameras c ON c.camera_id = s.camera_id
        WINDOW w AS (
            PARTITION BY s.client_id, s.person_id
            ORDER BY s.first_seen
        )
        WITH NO DATA
    """))
    conn.execute(sa.text(
        "CREATE UNIQUE INDEX ix_person_journeys_pk "
        "ON person_journeys (sighting_id)"
    ))
    conn.execute(sa.text(
        "CREATE INDEX ix_person_journeys_person_seen "
        "ON person_journeys (client_id, person_id, first_seen)"
    ))


def downgrade() -> None:
    conn = op.get_bind()

    conn.execute(sa.text("DROP MATERIALIZED VIEW IF EXISTS person_journeys"))

    for table in reversed(_RLS_TABLES):
        conn.execute(sa.text(f"DROP POLICY IF EXISTS tenant_isolation ON {table}"))
        conn.execute(sa.text(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY"))

    op.drop_table("client_node_assignments")
    op.drop_table("audit_log")
    op.drop_table("detection_log")
    op.drop_table("sightings")
    op.drop_table("attendance_records")
    op.drop_table("sessions")
    op.drop_table("face_embeddings")
    op.drop_table("persons")
    op.drop_table("cameras")
    op.drop_table("users")
    op.drop_table("clients")

    conn.execute(sa.text("DROP EXTENSION IF EXISTS pg_trgm"))
    conn.execute(sa.text("DROP EXTENSION IF EXISTS vector"))
