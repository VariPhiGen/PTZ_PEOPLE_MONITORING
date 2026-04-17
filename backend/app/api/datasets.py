"""
Face Datasets API — named groups of enrolled faces within a client tenant.

Endpoints:
  GET    /api/datasets              — list datasets for the current client
  POST   /api/datasets              — create a new dataset
  GET    /api/datasets/{id}         — dataset detail + stats
  PUT    /api/datasets/{id}         — rename / update description / color
  DELETE /api/datasets/{id}         — archive (no hard delete while persons enrolled)
  GET    /api/datasets/{id}/persons — persons belonging to this dataset (paginated)
  POST   /api/datasets/{id}/persons/{person_id}/move — move person to another dataset
  POST   /api/datasets/ensure-default — ensure client has at least one default dataset
"""

from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import text

from app.deps import DBSession
from app.middleware.auth import require_role

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

# require_role() returns a Depends object — use it directly in the list
_RW = [require_role("CLIENT_ADMIN", "SUPER_ADMIN")]
_RO = [require_role("VIEWER", "CLIENT_ADMIN", "SUPER_ADMIN")]


# ── Schemas ────────────────────────────────────────────────────────────────────

class DatasetCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    color: str = "#6366f1"
    client_id: str | None = None

class DatasetUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = None
    color: str | None = None

class MovePersonRequest(BaseModel):
    target_dataset_id: str


# ── Helpers ────────────────────────────────────────────────────────────────────

def _resolve_client(request: Request, body_client_id: str | None = None) -> str:
    state = request.state
    if getattr(state, "role", None) == "SUPER_ADMIN" and body_client_id:
        return body_client_id
    cid = getattr(state, "client_id", None)
    if not cid:
        if getattr(state, "role", None) == "SUPER_ADMIN":
            raise HTTPException(status_code=400, detail="client_id required for SUPER_ADMIN — pass client_id param or use the client picker")
        raise HTTPException(status_code=400, detail="client_id required")
    return str(cid)


async def _resolve_client_from_dataset(request: Request, db, dataset_id: str) -> str:
    """For update/archive/move: derive client_id from the dataset row itself."""
    cid = getattr(request.state, "client_id", None)
    if cid:
        return str(cid)
    row = (await db.execute(
        text("SELECT client_id::text FROM face_datasets WHERE dataset_id = (:did)::uuid"),
        {"did": dataset_id},
    )).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return row.client_id


def _dataset_row(r) -> dict:
    return {
        "dataset_id":   str(r.dataset_id),
        "client_id":    str(r.client_id),
        "name":         r.name,
        "description":  r.description,
        "color":        r.color,
        "is_default":   r.is_default,
        "status":       r.status,
        "person_count": r.person_count,
        "camera_count": getattr(r, "camera_count", 0),
        "created_at":   r.created_at,
        "updated_at":   r.updated_at,
    }


# ── List ───────────────────────────────────────────────────────────────────────

@router.get("", dependencies=_RO)
async def list_datasets(request: Request, db: DBSession, client_id: str | None = None) -> dict:
    client_id = _resolve_client(request, body_client_id=client_id)
    rows = (await db.execute(
        text("""
            SELECT d.*,
                   (SELECT COUNT(*) FROM persons p
                    WHERE p.dataset_id = d.dataset_id AND p.status = 'ACTIVE') AS person_count,
                   (SELECT COUNT(*) FROM cameras  c
                    WHERE c.dataset_id = d.dataset_id) AS camera_count
            FROM face_datasets d
            WHERE d.client_id = (:cid)::uuid
              AND d.status     = 'ACTIVE'
            ORDER BY d.is_default DESC, d.created_at ASC
        """),
        {"cid": client_id},
    )).fetchall()
    return {"items": [_dataset_row(r) for r in rows], "total": len(rows)}


# ── Create ─────────────────────────────────────────────────────────────────────

@router.post("", status_code=201, dependencies=_RW)
async def create_dataset(body: DatasetCreate, request: Request, db: DBSession) -> dict:
    client_id = _resolve_client(request, body_client_id=body.client_id)
    now = int(time.time())
    dataset_id = str(uuid.uuid4())

    result = await db.execute(
        text("""
            INSERT INTO face_datasets
                (dataset_id, client_id, name, description, color,
                 is_default, status, person_count, created_at, updated_at)
            VALUES
                ((:did)::uuid, (:cid)::uuid, :name, :desc, :color,
                 false, 'ACTIVE', 0, :now, :now)
            RETURNING *
        """),
        {
            "did": dataset_id, "cid": client_id,
            "name": body.name, "desc": body.description,
            "color": body.color, "now": now,
        },
    )
    row = result.fetchone()
    await db.commit()
    return _dataset_row(row)


# ── Detail ─────────────────────────────────────────────────────────────────────

@router.get("/{dataset_id}", dependencies=_RO)
async def get_dataset(dataset_id: str, request: Request, db: DBSession, client_id: str | None = None) -> dict:
    client_id = _resolve_client(request, body_client_id=client_id)
    row = (await db.execute(
        text("""
            SELECT d.*,
                   (SELECT COUNT(*) FROM persons p WHERE p.dataset_id = d.dataset_id AND p.status = 'ACTIVE') AS person_count,
                   (SELECT COUNT(*) FROM cameras c WHERE c.dataset_id = d.dataset_id) AS camera_count
            FROM face_datasets d
            WHERE d.dataset_id = (:did)::uuid
              AND d.client_id   = (:cid)::uuid
        """),
        {"did": dataset_id, "cid": client_id},
    )).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return _dataset_row(row)


# ── Update ─────────────────────────────────────────────────────────────────────

@router.put("/{dataset_id}", dependencies=_RW)
async def update_dataset(
    dataset_id: str, body: DatasetUpdate, request: Request, db: DBSession
) -> dict:
    client_id = await _resolve_client_from_dataset(request, db, dataset_id)
    now = int(time.time())

    # Only set provided fields
    sets, params = ["updated_at = :now"], {"did": dataset_id, "cid": client_id, "now": now}
    if body.name is not None:
        sets.append("name = :name"); params["name"] = body.name
    if body.description is not None:
        sets.append("description = :desc"); params["desc"] = body.description
    if body.color is not None:
        sets.append("color = :color"); params["color"] = body.color

    result = await db.execute(
        text(f"""
            UPDATE face_datasets
            SET {', '.join(sets)}
            WHERE dataset_id = (:did)::uuid AND client_id = (:cid)::uuid
            RETURNING *
        """),
        params,
    )
    row = result.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Dataset not found")
    await db.commit()
    return _dataset_row(row)


# ── Archive ────────────────────────────────────────────────────────────────────

@router.delete("/{dataset_id}", status_code=200, dependencies=_RW)
async def archive_dataset(dataset_id: str, request: Request, db: DBSession) -> dict:
    client_id = await _resolve_client_from_dataset(request, db, dataset_id)

    ds = (await db.execute(
        text("SELECT * FROM face_datasets WHERE dataset_id = (:did)::uuid AND client_id = (:cid)::uuid"),
        {"did": dataset_id, "cid": client_id},
    )).fetchone()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if ds.is_default:
        raise HTTPException(status_code=400, detail="Cannot archive the default dataset")

    person_count = (await db.execute(
        text("SELECT COUNT(*) FROM persons WHERE dataset_id = (:did)::uuid AND status = 'ACTIVE'"),
        {"did": dataset_id},
    )).scalar()
    if person_count and person_count > 0:
        raise HTTPException(
            status_code=409,
            detail=f"Dataset has {person_count} enrolled person(s). Move them to another dataset first.",
        )

    await db.execute(
        text("""
            UPDATE face_datasets SET status = 'ARCHIVED', updated_at = :now
            WHERE dataset_id = (:did)::uuid AND client_id = (:cid)::uuid
        """),
        {"did": dataset_id, "cid": client_id, "now": int(time.time())},
    )
    await db.commit()
    return {"archived": True, "dataset_id": dataset_id}


# ── Persons in dataset ─────────────────────────────────────────────────────────

@router.get("/{dataset_id}/persons", dependencies=_RO)
async def list_dataset_persons(
    dataset_id: str,
    request: Request,
    db: DBSession,
    limit: int = 50,
    offset: int = 0,
    q: str | None = None,
    client_id: str | None = None,
) -> dict:
    client_id = _resolve_client(request, body_client_id=client_id)
    params: dict = {"did": dataset_id, "cid": client_id, "limit": limit, "offset": offset}
    search_clause = ""
    if q:
        search_clause = "AND (p.name ILIKE :q OR p.external_id ILIKE :q)"
        params["q"] = f"%{q}%"

    rows = (await db.execute(
        text(f"""
            SELECT p.person_id::text, p.name, p.role, p.department,
                   p.external_id, p.status, p.created_at,
                   (SELECT COUNT(*) FROM face_embeddings fe
                    WHERE fe.person_id = p.person_id AND fe.is_active = true) AS embedding_count
            FROM persons p
            WHERE p.dataset_id = (:did)::uuid
              AND p.client_id  = (:cid)::uuid
              AND p.status     = 'ACTIVE'
              {search_clause}
            ORDER BY p.name
            LIMIT :limit OFFSET :offset
        """),
        params,
    )).fetchall()

    total = (await db.execute(
        text(f"""
            SELECT COUNT(*) FROM persons p
            WHERE p.dataset_id = (:did)::uuid AND p.client_id = (:cid)::uuid
              AND p.status = 'ACTIVE'
            {search_clause}
        """),
        params,
    )).scalar()

    return {
        "items": [dict(r._mapping) for r in rows],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


# ── Move person between datasets ───────────────────────────────────────────────

@router.post("/{dataset_id}/persons/{person_id}/move", dependencies=_RW)
async def move_person(
    dataset_id: str,
    person_id: str,
    body: MovePersonRequest,
    request: Request,
    db: DBSession,
) -> dict:
    client_id = await _resolve_client_from_dataset(request, db, dataset_id)

    # Verify target dataset exists and belongs to the same client
    target = (await db.execute(
        text("SELECT dataset_id FROM face_datasets WHERE dataset_id = (:tdid)::uuid AND client_id = (:cid)::uuid AND status = 'ACTIVE'"),
        {"tdid": body.target_dataset_id, "cid": client_id},
    )).fetchone()
    if not target:
        raise HTTPException(status_code=404, detail="Target dataset not found")

    now = int(time.time())
    result = await db.execute(
        text("""
            UPDATE persons
            SET dataset_id = (:tdid)::uuid, updated_at = :now
            WHERE person_id = (:pid)::uuid AND client_id = (:cid)::uuid
              AND dataset_id = (:sdid)::uuid
        """),
        {
            "tdid": body.target_dataset_id, "pid": person_id,
            "cid": client_id, "sdid": dataset_id, "now": now,
        },
    )
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Person not found in source dataset")

    # Keep face_embeddings.dataset_id in sync
    await db.execute(
        text("""
            UPDATE face_embeddings
            SET dataset_id = (:tdid)::uuid
            WHERE person_id = (:pid)::uuid AND client_id = (:cid)::uuid
        """),
        {"tdid": body.target_dataset_id, "pid": person_id, "cid": client_id},
    )

    # Update cached person_count on both datasets
    await db.execute(
        text("""
            UPDATE face_datasets SET person_count = (
                SELECT COUNT(*) FROM persons WHERE dataset_id = face_datasets.dataset_id
            ), updated_at = :now
            WHERE dataset_id IN ((:sdid)::uuid, (:tdid)::uuid)
        """),
        {"sdid": dataset_id, "tdid": body.target_dataset_id, "now": now},
    )
    await db.commit()
    return {"moved": True, "person_id": person_id, "target_dataset_id": body.target_dataset_id}


# ── Ensure default dataset ─────────────────────────────────────────────────────

@router.post("/ensure-default", dependencies=_RW)
async def ensure_default_dataset(request: Request, db: DBSession) -> dict:
    """
    Idempotently ensure the client has at least one default dataset.
    Called automatically on first login / client creation.
    """
    client_id = _resolve_client(request)
    existing = (await db.execute(
        text("SELECT dataset_id::text FROM face_datasets WHERE client_id = (:cid)::uuid AND is_default = true LIMIT 1"),
        {"cid": client_id},
    )).fetchone()
    if existing:
        return {"dataset_id": existing.dataset_id, "created": False}

    now = int(time.time())
    dataset_id = str(uuid.uuid4())
    await db.execute(
        text("""
            INSERT INTO face_datasets
                (dataset_id, client_id, name, is_default, status, person_count, created_at, updated_at)
            VALUES
                ((:did)::uuid, (:cid)::uuid, 'Default', true, 'ACTIVE', 0, :now, :now)
        """),
        {"did": dataset_id, "cid": client_id, "now": now},
    )
    await db.commit()
    return {"dataset_id": dataset_id, "created": True}
