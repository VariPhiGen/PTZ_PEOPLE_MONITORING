"""
Search API — text search, face image search, journey analytics, area queries.

GET  /api/search/person?q=         — pg_trgm name search
POST /api/search/face               — face image → top-10 identity matches
GET  /api/search/{person_id}/journey      — chronological event timeline
GET  /api/search/{person_id}/cross-camera — per-day camera transition trail
GET  /api/search/area               — who was in an area during a time window
GET  /api/search/{person_id}/heatmap      — area × hour presence heatmap

Authentication: all routes require persons:read.
Face search additionally requires face_embeddings:read.
Rate limiting: search/face is limited to 20 req/min per user.
"""
from __future__ import annotations

import base64
import uuid
from typing import Any

from datetime import datetime, timezone, date as _date_type

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel

from app.api._shared import audit, now_epoch, rate_limit, resolve_client_id
from app.deps import CurrentUser, DBSession
from app.middleware.auth import require_permission

router = APIRouter(prefix="/api/search", tags=["search"])

_RD    = require_permission("persons:read")
_FACE  = require_permission("face_embeddings:read")

_FACE_RL_KEY = "rl:search:face:{user_id}"
_FACE_RL_MAX = 20
_FACE_RL_WIN = 60


# ── Schemas ───────────────────────────────────────────────────────────────────

class FaceSearchRequest(BaseModel):
    """Alternative to multipart upload — accept base64-encoded image."""
    image_b64: str
    client_id: str | None = None   # SUPER_ADMIN only


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_epoch(value: str) -> float:
    """
    Accept either an epoch float string ('1744761600.0') or a calendar date
    string ('2026-04-16' or '2026-04-16T00:00:00').  Returns epoch in seconds.
    """
    try:
        return float(value)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    raise HTTPException(status_code=422, detail=f"Cannot parse date: {value!r}")


def _get_search_engine(request: Request) -> Any:
    eng = getattr(request.app.state, "face_search_engine", None)
    if eng is None:
        raise HTTPException(status_code=503, detail="Search service not initialised")
    return eng


def _hit_dict(h: Any) -> dict:
    return {
        "person_id":    h.person_id,
        "name":         h.name,
        "role":         h.role,
        "department":   h.department,
        "thumbnail_url": h.thumbnail_url,
        "last_seen":    h.last_seen,
        "match_score":  h.match_score,
    }


def _face_hit_dict(h: Any) -> dict:
    d = _hit_dict(h)
    d.update({"similarity": h.similarity, "tier": h.tier, "embedding_id": h.embedding_id})
    return d


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/person", dependencies=[_RD])
async def search_person(
    request: Request,
    db:      DBSession,
    q:       str       = Query(..., min_length=2, description="Name / department / external_id"),
    limit:   int       = Query(10, ge=1, le=50),
    client_id: str | None = Query(None),
) -> dict:
    """Full-text person search using pg_trgm similarity."""
    cid    = resolve_client_id(request, client_id)
    engine = _get_search_engine(request)
    hits   = await engine.search_by_text(cid, q, limit=limit)
    return {"query": q, "total": len(hits), "items": [_hit_dict(h) for h in hits]}


@router.post("/face", dependencies=[_FACE])
async def search_by_face_multipart(
    request:   Request,
    db:        DBSession,
    file:      UploadFile = File(...),
    client_id: str | None = Form(None),
) -> dict:
    """
    Upload a face image (JPEG/PNG) and return the top-10 identity matches.
    Enforces quality gate: exactly 1 face, IOD >= 60 px, sharpness >= 30.
    """
    await rate_limit(
        request.app.state.redis,
        _FACE_RL_KEY.format(user_id=request.state.user_id),
        _FACE_RL_MAX,
        _FACE_RL_WIN,
    )
    cid    = resolve_client_id(request, client_id)
    engine = _get_search_engine(request)

    image_bytes = await file.read()
    try:
        hits = await engine.search_by_face(cid, image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    audit(request, "FACE_SEARCH", "search", cid)
    return {"total": len(hits), "items": [_face_hit_dict(h) for h in hits]}


@router.post("/face/base64", dependencies=[_FACE])
async def search_by_face_base64(body: FaceSearchRequest, request: Request, db: DBSession) -> dict:
    """Accept base64-encoded image instead of multipart."""
    await rate_limit(
        request.app.state.redis,
        _FACE_RL_KEY.format(user_id=request.state.user_id),
        _FACE_RL_MAX,
        _FACE_RL_WIN,
    )
    cid = resolve_client_id(request, body.client_id)
    try:
        image_bytes = base64.b64decode(body.image_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    engine = _get_search_engine(request)
    try:
        hits = await engine.search_by_face(cid, image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    audit(request, "FACE_SEARCH_B64", "search", cid)
    return {"total": len(hits), "items": [_face_hit_dict(h) for h in hits]}


@router.get("/area", dependencies=[_RD])
async def search_area(
    request:   Request,
    db:        DBSession,
    camera_id: str       = Query(...),
    time_from: float     = Query(..., description="Seconds since midnight (or epoch if date omitted)"),
    time_to:   float     = Query(..., description="Seconds since midnight (or epoch if date omitted)"),
    date:      float | None = Query(None, description="Epoch of any moment on the target day (UTC)"),
    client_id: str | None = Query(None),
) -> dict:
    """Return all persons seen at a camera during a time window."""
    cid    = resolve_client_id(request, client_id)
    engine = _get_search_engine(request)
    occupants = await engine.search_area(cid, camera_id, time_from, time_to, date)
    return {
        "camera_id": camera_id,
        "time_from": time_from,
        "time_to":   time_to,
        "total":     len(occupants),
        "items": [
            {
                "person_id":      o.person_id,
                "name":           o.name,
                "role":           o.role,
                "department":     o.department,
                "first_seen":     o.first_seen,
                "last_seen":      o.last_seen,
                "duration_s":     o.duration_s,
                "confidence_avg": o.confidence_avg,
            }
            for o in occupants
        ],
    }


@router.get("/{person_id}/journey", dependencies=[_RD])
async def get_journey(
    person_id: str,
    request:   Request,
    db:        DBSession,
    date_from: str       = Query(..., description="Epoch float or YYYY-MM-DD"),
    date_to:   str       = Query(..., description="Epoch float or YYYY-MM-DD"),
    client_id: str | None = Query(None),
) -> dict:
    """
    Build a chronological journey for a person across attendance records
    and sightings, with transit times and a presence summary.
    """
    cid     = resolve_client_id(request, client_id)
    engine  = _get_search_engine(request)
    ep_from = _to_epoch(date_from)
    # date_to is end-of-day: add 86399s when a bare date string is given
    ep_to   = _to_epoch(date_to)
    try:
        float(date_to)          # was it already an epoch?
    except ValueError:
        ep_to += 86_399         # advance to 23:59:59 of the given date

    try:
        journey = await engine.build_journey(cid, person_id, ep_from, ep_to)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    # Compute attendance_rate from attendance_stats dict
    stats      = journey.summary.attendance_stats or {}
    present_n  = stats.get("P", 0) + stats.get("EE", 0) + stats.get("L", 0)
    total_att  = sum(stats.values()) or 1
    att_rate   = present_n / total_att

    return {
        "person_id":  journey.person_id,
        "name":       journey.name,
        "role":       journey.role,
        "department": journey.department,
        "date_from":  journey.date_from,
        "date_to":    journey.date_to,
        "summary": {
            "total_events":      journey.summary.total_events,
            "total_duration_s":  journey.summary.total_duration_s,
            "total_locations":   len(journey.summary.locations_visited),
            "locations_visited": journey.summary.locations_visited,
            "cameras_visited":   journey.summary.cameras_visited,
            "attendance_rate":   att_rate,
            "attendance_stats":  stats,
            "first_appearance":  journey.summary.first_appearance,
            "last_appearance":   journey.summary.last_appearance,
        },
        "events": [
            {
                # type/transit_time match the frontend JourneyEvent interface
                "type":                e.event_type,
                "event_type":          e.event_type,
                "source_id":           e.source_id,
                "camera_id":           e.camera_id,
                "camera_name":         e.camera_name or str(e.camera_id),
                "area":                e.room_name or e.zone,
                "room_name":           e.room_name,
                "building":            e.building,
                "floor":               e.floor,
                "zone":                e.zone,
                "first_seen":          datetime.fromtimestamp(e.first_seen, tz=timezone.utc).isoformat()
                                       if isinstance(e.first_seen, (int, float)) else e.first_seen,
                "last_seen":           datetime.fromtimestamp(e.last_seen, tz=timezone.utc).isoformat()
                                       if isinstance(e.last_seen, (int, float)) else e.last_seen,
                "duration_s":          e.duration_s,
                "transit_time":        e.transit_from_prev_s,
                "transit_from_prev_s": e.transit_from_prev_s,
                "status":              e.status,
                "confidence_avg":      e.confidence_avg,
            }
            for e in journey.events
        ],
    }


@router.get("/{person_id}/cross-camera", dependencies=[_RD])
async def cross_camera_trail(
    person_id: str,
    request:   Request,
    db:        DBSession,
    date:      str       = Query(..., description="Epoch float or YYYY-MM-DD"),
    client_id: str | None = Query(None),
) -> list:
    """
    Return ordered camera visits for a person on a given calendar day.
    Returns a flat list of per-camera visit entries (not from/to transitions).
    """
    cid    = resolve_client_id(request, client_id)
    engine = _get_search_engine(request)
    ep     = _to_epoch(date)
    trail  = await engine.get_cross_camera_trail(cid, person_id, ep)

    # Convert CameraTransition (from→to pairs) to flat per-camera visit list.
    # Each transition's "to_*" fields describe the camera the person arrived at.
    result = []
    for t in trail:
        first_seen_iso = (
            datetime.fromtimestamp(t.arrived_at, tz=timezone.utc).isoformat()
            if t.arrived_at else None
        )
        last_seen_ts = t.arrived_at + t.duration_s if (t.arrived_at and t.duration_s) else None
        last_seen_iso = (
            datetime.fromtimestamp(last_seen_ts, tz=timezone.utc).isoformat()
            if last_seen_ts else None
        )
        result.append({
            "camera_id":    t.to_camera_id,
            "camera_name":  t.to_camera_name,
            "area":         t.to_location,
            "building":     None,
            "first_seen":   first_seen_iso,
            "last_seen":    last_seen_iso,
            "transit_time": t.transit_time_s,
            "duration_s":   t.duration_s,
            "confidence_avg": t.confidence_avg,
        })
    return result


@router.get("/{person_id}/heatmap", dependencies=[_RD])
async def person_heatmap(
    person_id: str,
    request:   Request,
    db:        DBSession,
    date_from: str       = Query(..., description="Epoch float or YYYY-MM-DD"),
    date_to:   str       = Query(..., description="Epoch float or YYYY-MM-DD"),
    client_id: str | None = Query(None),
) -> dict:
    """
    Return a presence heatmap for a person: areas list + 24-column matrix (minutes).
    """
    cid     = resolve_client_id(request, client_id)
    engine  = _get_search_engine(request)
    ep_from = _to_epoch(date_from)
    ep_to   = _to_epoch(date_to)
    try:
        float(date_to)
    except ValueError:
        ep_to += 86_399

    try:
        journey = await engine.build_journey(cid, person_id, ep_from, ep_to)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    # Convert {area: {hour: seconds}} → {areas: [...], matrix: [[minutes×24], ...]}
    raw_heatmap: dict = journey.heatmap or {}
    areas  = sorted(raw_heatmap.keys())
    matrix = [
        [round(raw_heatmap[a].get(h, 0.0) / 60, 2) for h in range(24)]
        for a in areas
    ]

    return {
        "person_id": person_id,
        "date_from": ep_from,
        "date_to":   ep_to,
        "areas":     areas,
        "matrix":    matrix,
    }
