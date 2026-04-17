"""FK and cascade fixes

Revision: 0004
Previous: 0003

Changes
───────
1. cameras.dataset_id FK: change ON DELETE from SET NULL → RESTRICT.
   Prevents silently orphaning cameras when a dataset is deleted.
2. face_embeddings.person_id FK: add ON DELETE CASCADE.
   Deleting a Person now automatically removes their embeddings from DB,
   keeping FAISS and pgvector in sync.
"""

from alembic import op


revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── 1. cameras.dataset_id: SET NULL → RESTRICT ────────────────────────────
    op.drop_constraint(
        "cameras_dataset_id_fkey", "cameras", type_="foreignkey"
    )
    op.create_foreign_key(
        "cameras_dataset_id_fkey",
        "cameras", "face_datasets",
        ["dataset_id"], ["dataset_id"],
        ondelete="RESTRICT",
    )

    # ── 2. face_embeddings.person_id: add ON DELETE CASCADE ──────────────────
    op.drop_constraint(
        "face_embeddings_person_id_fkey", "face_embeddings", type_="foreignkey"
    )
    op.create_foreign_key(
        "face_embeddings_person_id_fkey",
        "face_embeddings", "persons",
        ["person_id"], ["person_id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    # ── 2. Revert face_embeddings.person_id CASCADE → NO ACTION ──────────────
    op.drop_constraint(
        "face_embeddings_person_id_fkey", "face_embeddings", type_="foreignkey"
    )
    op.create_foreign_key(
        "face_embeddings_person_id_fkey",
        "face_embeddings", "persons",
        ["person_id"], ["person_id"],
    )

    # ── 1. Revert cameras.dataset_id: RESTRICT → SET NULL ────────────────────
    op.drop_constraint(
        "cameras_dataset_id_fkey", "cameras", type_="foreignkey"
    )
    op.create_foreign_key(
        "cameras_dataset_id_fkey",
        "cameras", "face_datasets",
        ["dataset_id"], ["dataset_id"],
        ondelete="SET NULL",
    )
