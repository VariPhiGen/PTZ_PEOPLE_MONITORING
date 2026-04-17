"""Performance tuning: HNSW upgrade + composite/partial indexes

Revision: 0002
Previous: 0001

Changes
───────
1. Upgrade HNSW index on face_embeddings.embedding
   ef_construction: 64 → 200  (better recall at query time, ~20% slower build)
   m stays at 16 (already optimal for 512-D cosine)
   ef_search hint set to 100 via GUC in the connection string.

2. Composite indexes for the hottest query patterns:
   • sessions (client_id, sync_status, scheduled_start DESC)   → session list
   • sessions (client_id, camera_id) WHERE active              → live camera lookup
   • attendance_records (session_id, status)                   → session detail
   • attendance_records (person_id, created_at DESC)           → journey timeline
   • sightings (client_id, person_id, first_seen DESC)         → per-person trail
   • cameras (client_id, status)                               → camera list
   • audit_log (client_id, created_at DESC)                    → audit trail
   • client_node_assignments (node_id)                         → node lookup

3. Partial indexes (lower cardinality, skip irrelevant rows):
   • face_embeddings WHERE is_active = true                    → identify()
   • cameras WHERE status = 'ONLINE'                           → live cams
   • sessions WHERE actual_end IS NULL                         → active sessions

4. BRIN index on audit_log.created_at for time-range scans on large tables.

5. Set hnsw.ef_search = 100 at session level (see connection pool config).
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision:      str                  = "0002"
down_revision: Union[str, None]     = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on:    Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Rebuild HNSW index with ef_construction = 200
    #    CONCURRENTLY cannot run inside a transaction; Alembic uses a transaction
    #    by default, so we use a regular DROP + CREATE here.  On production, run
    #    this migration during a low-traffic window (<10 min for 1 M vectors).
    # ─────────────────────────────────────────────────────────────────────────
    conn.execute(sa.text("DROP INDEX IF EXISTS ix_face_embeddings_hnsw"))
    conn.execute(sa.text(
        "CREATE INDEX ix_face_embeddings_hnsw "
        "ON face_embeddings "
        "USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 200)"
    ))

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Composite indexes — hot query patterns
    # ─────────────────────────────────────────────────────────────────────────

    # Sessions list (primary sort column = scheduled_start DESC)
    conn.execute(sa.text(
        "CREATE INDEX IF NOT EXISTS ix_sessions_client_status_ts "
        "ON sessions (client_id, sync_status, scheduled_start DESC)"
    ))

    # Attendance records per session (with status filter)
    conn.execute(sa.text(
        "CREATE INDEX IF NOT EXISTS ix_ar_session_status "
        "ON attendance_records (session_id, status)"
    ))

    # Journey timeline: all attendance for a person, newest first
    conn.execute(sa.text(
        "CREATE INDEX IF NOT EXISTS ix_ar_person_ts "
        "ON attendance_records (person_id, created_at DESC)"
    ))

    # Sighting trail: per-client, per-person, newest first
    conn.execute(sa.text(
        "CREATE INDEX IF NOT EXISTS ix_sightings_client_person_ts "
        "ON sightings (client_id, person_id, first_seen DESC)"
    ))

    # Camera list per client
    conn.execute(sa.text(
        "CREATE INDEX IF NOT EXISTS ix_cameras_client_status "
        "ON cameras (client_id, status)"
    ))

    # Audit log: most recent events per client (used by admin panel)
    conn.execute(sa.text(
        "CREATE INDEX IF NOT EXISTS ix_audit_client_ts "
        "ON audit_log (client_id, created_at DESC)"
    ))

    # Node assignment reverse lookup (node → cameras)
    conn.execute(sa.text(
        "CREATE INDEX IF NOT EXISTS ix_cna_node_id "
        "ON client_node_assignments (node_id)"
    ))

    # Sessions by camera + status (live dashboard: which camera is active?)
    conn.execute(sa.text(
        "CREATE INDEX IF NOT EXISTS ix_sessions_camera_id "
        "ON sessions (camera_id, sync_status)"
    ))

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Partial indexes — skip irrelevant rows entirely
    # ─────────────────────────────────────────────────────────────────────────

    # identify(): only search active embeddings
    conn.execute(sa.text(
        "CREATE INDEX IF NOT EXISTS ix_face_emb_active "
        "ON face_embeddings (client_id, person_id) "
        "WHERE is_active = true"
    ))

    # live camera query: skip offline/degraded cameras
    conn.execute(sa.text(
        "CREATE INDEX IF NOT EXISTS ix_cameras_online "
        "ON cameras (client_id, node_id) "
        "WHERE status = 'ONLINE'"
    ))

    # active sessions (no actual_end → still running)
    conn.execute(sa.text(
        "CREATE INDEX IF NOT EXISTS ix_sessions_active "
        "ON sessions (client_id, camera_id, created_at DESC) "
        "WHERE actual_end IS NULL"
    ))

    # pending/held sync batches
    conn.execute(sa.text(
        "CREATE INDEX IF NOT EXISTS ix_sessions_pending_sync "
        "ON sessions (client_id, sync_status, created_at DESC) "
        "WHERE sync_status IN ('PENDING', 'HELD')"
    ))

    # active persons only (enrollment lookup)
    conn.execute(sa.text(
        "CREATE INDEX IF NOT EXISTS ix_persons_active "
        "ON persons (client_id, department) "
        "WHERE status = 'ACTIVE'"
    ))

    # ─────────────────────────────────────────────────────────────────────────
    # 4. BRIN index on audit_log.created_at
    #    Very cheap to build; efficient for sequential time-range scans
    #    on append-only tables.  pages_per_range=64 ≈ 1 row/64 heap pages.
    # ─────────────────────────────────────────────────────────────────────────
    conn.execute(sa.text(
        "CREATE INDEX IF NOT EXISTS ix_audit_brin "
        "ON audit_log USING brin (created_at) "
        "WITH (pages_per_range = 64)"
    ))

    # Same for sightings (time-range queries in monitoring view)
    conn.execute(sa.text(
        "CREATE INDEX IF NOT EXISTS ix_sightings_brin "
        "ON sightings USING brin (first_seen) "
        "WITH (pages_per_range = 64)"
    ))

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Set hnsw.ef_search default (affects all queries in this DB)
    #    Higher value → better recall, slightly slower; 100 is the sweet spot
    #    for 512-D embeddings with m=16.
    # ─────────────────────────────────────────────────────────────────────────
    conn.execute(sa.text("ALTER DATABASE acas SET hnsw.ef_search = 100"))


def downgrade() -> None:
    conn = op.get_bind()

    # Drop new indexes
    for idx in [
        "ix_sessions_client_status_ts",
        "ix_ar_session_status",
        "ix_ar_person_ts",
        "ix_sightings_client_person_ts",
        "ix_cameras_client_status",
        "ix_audit_client_ts",
        "ix_cna_node_id",
        "ix_sessions_camera_id",
        "ix_face_emb_active",
        "ix_cameras_online",
        "ix_sessions_active",
        "ix_sessions_pending_sync",
        "ix_persons_active",
        "ix_audit_brin",
        "ix_sightings_brin",
    ]:
        conn.execute(sa.text(f"DROP INDEX IF EXISTS {idx}"))

    # Restore original HNSW with ef_construction=64
    conn.execute(sa.text("DROP INDEX IF EXISTS ix_face_embeddings_hnsw"))
    conn.execute(sa.text(
        "CREATE INDEX ix_face_embeddings_hnsw "
        "ON face_embeddings "
        "USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    ))

    conn.execute(sa.text("ALTER DATABASE acas RESET hnsw.ef_search"))
