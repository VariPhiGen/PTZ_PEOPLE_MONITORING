"""
Settings API — per-client configurable thresholds, AI/PTZ parameters, and ERP adapters.

GET  /api/settings/thresholds      — read attendance thresholds
PUT  /api/settings/thresholds      — save attendance thresholds
GET  /api/settings/ai-ptz          — read AI + PTZ tunables
PUT  /api/settings/ai-ptz          — save AI + PTZ tunables
GET  /api/settings/erp             — list ERP adapter stubs (read-only for now)

All settings are stored in clients.settings JSONB column under namespaced keys
so they are isolated per client and won't clash with future columns.
"""
from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm.attributes import flag_modified

from app.api._shared import resolve_client_id, now_epoch
from app.deps import CurrentUser, DBSession
from app.middleware.auth import require_permission
from app.models.clients import Client

router = APIRouter(prefix="/api/settings", tags=["settings"])

_RD = require_permission("cameras:read")
_WR = require_permission("cameras:write")


# ── Default values ────────────────────────────────────────────────────────────

_DEFAULT_THRESHOLDS: dict[str, Any] = {
    "min_presence_pct":   75.0,
    "grace_minutes":      5,
    "early_exit_minutes": 10,
    "min_detections":     2,
    "dept_overrides":     [],
}

_DEFAULT_AI_PTZ: dict[str, Any] = {
    "confidence_floor":         0.60,
    "liveness_threshold":       0.40,
    "template_alpha":           0.85,
    "drift_limit":              0.05,
    "max_hunts_per_cell":       3,
    "zoom_budget_s":            15.0,
    "dbscan_eps_deg":           5.0,
    "path_replan_s":            300.0,
    "settle_ms":                350,
    "target_inter_ocular_px":   100,
    # Face recognition thresholds (applied per-client at runtime)
    "face_tier1_threshold":     0.78,   # FAISS match — targets <0.01 % FAR
    "face_tier2_threshold":     0.85,   # pgvector fallback — higher confidence gate
    "face_max_templates":       5,      # gallery templates per person
}


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class DeptOverride(BaseModel):
    department:          str
    min_presence_pct:    float | None = None
    grace_minutes:       int   | None = None
    early_exit_minutes:  int   | None = None
    min_detections:      int   | None = None


class ThresholdsBody(BaseModel):
    min_presence_pct:    float        = 75.0
    grace_minutes:       int          = 5
    early_exit_minutes:  int          = 10
    min_detections:      int          = 2
    dept_overrides:      list[DeptOverride] = []


class AIPTZBody(BaseModel):
    confidence_floor:        float = 0.60
    liveness_threshold:      float = 0.40
    template_alpha:          float = 0.85
    drift_limit:             float = 0.05
    max_hunts_per_cell:      int   = 3
    zoom_budget_s:           float = 15.0
    dbscan_eps_deg:          float = 5.0
    path_replan_s:           float = 300.0
    settle_ms:               int   = 350
    target_inter_ocular_px:  int   = 100
    # Face recognition thresholds
    face_tier1_threshold:    float = 0.78
    face_tier2_threshold:    float = 0.85
    face_max_templates:      int   = 5


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _load_client_settings(db: DBSession, client_id: str) -> dict:
    row = await db.execute(
        select(Client).where(Client.client_id == uuid.UUID(client_id))
    )
    client = row.scalar_one_or_none()
    if client is None:
        raise HTTPException(status_code=404, detail="Client not found")
    return client.settings or {}


async def _save_client_settings_key(
    db: DBSession,
    client_id: str,
    key: str,
    value: dict,
) -> None:
    row = await db.execute(
        select(Client).where(Client.client_id == uuid.UUID(client_id))
    )
    client = row.scalar_one_or_none()
    if client is None:
        raise HTTPException(status_code=404, detail="Client not found")
    existing = dict(client.settings or {})
    existing[key] = value
    client.settings = existing
    # mark column dirty explicitly (JSONB in-place mutation not auto-detected)
    flag_modified(client, "settings")


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/thresholds", dependencies=[_RD])
async def get_thresholds(
    request: Request,
    db: DBSession,
    client_id: str | None = None,
) -> dict:
    cid      = resolve_client_id(request, client_id)
    stored   = await _load_client_settings(db, cid)
    result   = {**_DEFAULT_THRESHOLDS, **(stored.get("thresholds") or {})}
    return result


@router.put("/thresholds", dependencies=[_WR])
async def put_thresholds(
    body: ThresholdsBody,
    request: Request,
    db: DBSession,
    client_id: str | None = None,
) -> dict:
    cid   = resolve_client_id(request, client_id)
    value = body.model_dump()
    await _save_client_settings_key(db, cid, "thresholds", value)
    return {"status": "ok", "ts": now_epoch()}


@router.get("/ai-ptz", dependencies=[_RD])
async def get_ai_ptz(
    request: Request,
    db: DBSession,
    client_id: str | None = None,
) -> dict:
    cid    = resolve_client_id(request, client_id)
    stored = await _load_client_settings(db, cid)
    result = {**_DEFAULT_AI_PTZ, **(stored.get("ai_ptz") or {})}
    return result


@router.put("/ai-ptz", dependencies=[_WR])
async def put_ai_ptz(
    body: AIPTZBody,
    request: Request,
    db: DBSession,
    client_id: str | None = None,
) -> dict:
    cid   = resolve_client_id(request, client_id)
    value = body.model_dump()
    await _save_client_settings_key(db, cid, "ai_ptz", value)
    return {"status": "ok", "ts": now_epoch()}


@router.get("/erp", dependencies=[_RD])
async def get_erp(request: Request) -> list:
    """Return ERP adapter stubs. Not yet configurable — returns empty list."""
    return []
