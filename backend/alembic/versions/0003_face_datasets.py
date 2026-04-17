"""Face Datasets — named groups of enrolled faces per client

Revision: 0003
Previous: 0002

Changes
───────
1. Create face_datasets table.
2. Add dataset_id FK to persons, cameras, face_embeddings.
3. Auto-create one default dataset for every existing client so that all
   pre-existing persons and embeddings are assigned to it.
4. Indexes for hot lookup patterns.
5. Extend RLS on face_datasets.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── 1. face_datasets table ────────────────────────────────────────────────
    op.create_table(
        "face_datasets",
        sa.Column("dataset_id",   UUID(as_uuid=False), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("client_id",    UUID(as_uuid=False), sa.ForeignKey(
                  "clients.client_id", ondelete="CASCADE"), nullable=False),
        sa.Column("name",         sa.String(255), nullable=False),
        sa.Column("description",  sa.Text,        nullable=True),
        sa.Column("color",        sa.String(7),   nullable=False, server_default="#6366f1"),
        sa.Column("is_default",   sa.Boolean,     nullable=False, server_default="false"),
        sa.Column("status",       sa.String(20),  nullable=False, server_default="ACTIVE"),
        sa.Column("person_count", sa.BigInteger,  nullable=False, server_default="0"),
        sa.Column("created_at",   sa.BigInteger,  nullable=False,
                  server_default=sa.text("EXTRACT(EPOCH FROM NOW())::BIGINT")),
        sa.Column("updated_at",   sa.BigInteger,  nullable=False,
                  server_default=sa.text("EXTRACT(EPOCH FROM NOW())::BIGINT")),
    )
    op.create_index("ix_face_datasets_client_id", "face_datasets", ["client_id"])
    op.create_index(
        "ix_face_datasets_client_status",
        "face_datasets", ["client_id", "status"],
    )

    # ── 2. Add dataset_id to persons, cameras, face_embeddings ───────────────
    op.add_column("persons", sa.Column(
        "dataset_id", UUID(as_uuid=False),
        sa.ForeignKey("face_datasets.dataset_id", ondelete="SET NULL"),
        nullable=True,
    ))
    op.create_index("ix_persons_dataset_id", "persons", ["dataset_id"])

    op.add_column("cameras", sa.Column(
        "dataset_id", UUID(as_uuid=False),
        sa.ForeignKey("face_datasets.dataset_id", ondelete="SET NULL"),
        nullable=True,
    ))
    op.create_index("ix_cameras_dataset_id", "cameras", ["dataset_id"])

    op.add_column("face_embeddings", sa.Column(
        "dataset_id", UUID(as_uuid=False),
        sa.ForeignKey("face_datasets.dataset_id", ondelete="SET NULL"),
        nullable=True,
    ))
    op.create_index("ix_face_embeddings_dataset_id", "face_embeddings", ["dataset_id"])

    # ── 3. Create one default dataset per existing client and assign persons ──
    op.execute(sa.text("""
        INSERT INTO face_datasets (dataset_id, client_id, name, is_default, status, person_count)
        SELECT gen_random_uuid(), client_id, 'Default', true, 'ACTIVE',
               (SELECT COUNT(*) FROM persons p WHERE p.client_id = c.client_id)
        FROM clients c;
    """))

    op.execute(sa.text("""
        UPDATE persons p
        SET dataset_id = (
            SELECT dataset_id FROM face_datasets d
            WHERE d.client_id = p.client_id AND d.is_default = true
            LIMIT 1
        )
        WHERE p.dataset_id IS NULL;
    """))

    op.execute(sa.text("""
        UPDATE face_embeddings fe
        SET dataset_id = (
            SELECT p.dataset_id FROM persons p WHERE p.person_id = fe.person_id
        )
        WHERE fe.dataset_id IS NULL;
    """))

    # ── 4. RLS on face_datasets ───────────────────────────────────────────────
    op.execute("ALTER TABLE face_datasets ENABLE ROW LEVEL SECURITY;")
    op.execute("ALTER TABLE face_datasets FORCE ROW LEVEL SECURITY;")
    op.execute("""
        CREATE POLICY tenant_isolation ON face_datasets
        USING (
            client_id::text = current_setting('app.current_client_id', true)
            OR current_setting('app.current_client_id', true) IS NULL
            OR current_setting('app.current_client_id', true) = ''
        );
    """)


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS tenant_isolation ON face_datasets;")
    op.execute("ALTER TABLE face_datasets DISABLE ROW LEVEL SECURITY;")
    op.drop_index("ix_face_embeddings_dataset_id", "face_embeddings")
    op.drop_column("face_embeddings", "dataset_id")
    op.drop_index("ix_cameras_dataset_id", "cameras")
    op.drop_column("cameras", "dataset_id")
    op.drop_index("ix_persons_dataset_id", "persons")
    op.drop_column("persons", "dataset_id")
    op.drop_index("ix_face_datasets_client_status", "face_datasets")
    op.drop_index("ix_face_datasets_client_id", "face_datasets")
    op.drop_table("face_datasets")
