"""
Monitoring API — real-time occupancy and live detection feeds.

GET /api/monitoring/occupancy               — per-camera known/unknown/total counts
GET /api/cameras/{camera_id}/live-detections — tracked persons for one camera

These endpoints aggregate data directly from the in-process PTZBrain instances
stored in app.state.ptz_brains, so they reflect the current frame without any
database round-trip.  Person names are looked up from DB only for recognised IDs.
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import select, text

from app.api._shared import resolve_client_id
from app.deps import CurrentUser, DBSession
from app.middleware.auth import require_permission
from app.models.persons import Person

router = APIRouter(tags=["monitoring"])

_RD = require_permission("cameras:read")


def _get_ptz_brains(request: Request) -> dict[str, Any]:
    return getattr(request.app.state, "ptz_brains", {})


# ── /api/monitoring/occupancy ─────────────────────────────────────────────────

@router.get("/api/monitoring/occupancy", dependencies=[_RD])
async def monitoring_occupancy(request: Request) -> dict:
    """
    Return a map of camera_id → {known, unknown, total} for every active brain
    that belongs to the requesting client.
    """
    cid    = resolve_client_id(request, require=False)
    brains = _get_ptz_brains(request)

    result: dict[str, dict] = {}
    for brain in brains.values():
        try:
            state = brain.get_state()
        except Exception:
            continue
        # Filter by client when a cid is available
        if cid and state.client_id != cid:
            continue
        result[state.camera_id] = {
            "known":   state.present_count,
            "unknown": state.unknown_count,
            "total":   state.present_count + state.unknown_count,
        }
    return result


# ── /api/cameras/{camera_id}/live-detections ──────────────────────────────────

@router.get("/api/cameras/{camera_id}/live-detections", dependencies=[_RD])
async def live_detections(camera_id: str, request: Request, db: DBSession) -> list:
    """
    Return the list of persons currently tracked by the PTZBrain for *camera_id*.
    Recognised persons have their DB name resolved; unknown tracks appear as
    is_unknown=True.
    """
    cid    = resolve_client_id(request, require=False)
    brains = _get_ptz_brains(request)

    # Find the brain for this camera
    target_brain = None
    for brain in brains.values():
        try:
            state = brain.get_state()
        except Exception:
            continue
        if state.camera_id != camera_id:
            continue
        if cid and state.client_id != cid:
            continue
        target_brain = brain
        break

    if target_brain is None:
        # No active session for this camera — return empty list (not 404)
        return []

    # Pull live tracks from the SimplePersonTracker
    try:
        tracks = list(target_brain._person_tracker._tracks)
    except Exception:
        tracks = []

    now = time.time()

    # Collect recognised person_ids so we can batch-resolve names
    recognised_ids = {
        str(t.person_id) for t in tracks if t.person_id
    }

    # Batch-fetch names from DB
    name_map: dict[str, str] = {}
    if recognised_ids:
        try:
            rows = await db.execute(
                select(Person.person_id, Person.name).where(
                    Person.person_id.in_([uuid.UUID(pid) for pid in recognised_ids])
                )
            )
            for row in rows:
                name_map[str(row.person_id)] = row.name
        except Exception:
            pass

    result = []
    for track in tracks:
        pid  = str(track.person_id) if track.person_id else None
        name = name_map.get(pid) if pid else None
        result.append({
            "person_id":   pid,
            "person_name": name,
            "is_unknown":  pid is None,
            "thumbnail":   None,
            "first_seen":  time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(track.last_seen - 30),  # approximate first_seen
            ),
            "confidence":  0.0,
        })

    return result
