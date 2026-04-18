"""
Attendance API — session records, held sessions, export, and overrides.

GET  /api/attendance/sessions                 — list sessions with filters
GET  /api/attendance/sessions/{id}            — session detail
GET  /api/attendance/sessions/{id}/records    — per-student records
GET  /api/attendance/records                  — cross-session record query
GET  /api/attendance/held                     — held sessions awaiting review
GET  /api/attendance/export                   — CSV export
POST /api/attendance/held/{id}/force-sync     — approve & sync a held session
POST /api/attendance/held/{id}/discard        — discard a held session
POST /api/attendance/records/{id}/override    — manual status override by admin
"""
from __future__ import annotations

import asyncio
import csv
import io
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import func, select, text, update

from app.api._shared import audit, now_epoch, resolve_client_id


def _epoch_to_iso(ts: int | None) -> str | None:
    """Convert Unix epoch seconds → ISO-8601 string, or None."""
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")
from app.deps import CurrentUser, DBSession
from app.middleware.auth import require_permission
from app.models.attendance_records import AttendanceRecord, AttendanceStatus
from app.models.cameras import Camera
from app.models.persons import Person
from app.models.sessions import Session, SyncStatus
from app.models.unknown_detections import UnknownDetection

router = APIRouter(prefix="/api/attendance", tags=["attendance"])

_RD = require_permission("attendance:read")
_UP = require_permission("attendance:update")

_OVERRIDE_RL_KEY = "rl:att:override:{client_id}"
_OVERRIDE_MAX    = 30
_OVERRIDE_WIN_S  = 60


# ── Schemas ───────────────────────────────────────────────────────────────────

class OverrideRequest(BaseModel):
    status: str
    reason: str

    def validate_status(self) -> None:
        valid = {s.value for s in AttendanceStatus}
        if self.status not in valid:
            raise HTTPException(status_code=400, detail=f"status must be one of {valid}")


class HeldActionRequest(BaseModel):
    reason: str | None = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _session_dict(s: Session, record_count: int | None = None) -> dict:
    d: dict[str, Any] = {
        "session_id":       str(s.session_id),
        "client_id":        str(s.client_id),
        "camera_id":        str(s.camera_id) if s.camera_id else None,
        "course_id":        s.course_id,
        "course_name":      s.course_name,
        "faculty_id":       str(s.faculty_id) if s.faculty_id else None,
        "scheduled_start":  _epoch_to_iso(s.scheduled_start),
        "scheduled_end":    _epoch_to_iso(s.scheduled_end),
        "actual_start":     _epoch_to_iso(s.actual_start),
        "actual_end":       _epoch_to_iso(s.actual_end),
        "faculty_status":   s.faculty_status,
        "sync_status":      s.sync_status,
        "held_reason":      s.held_reason,
        "cycle_count":      s.cycle_count,
        "recognition_rate": s.recognition_rate,
        "synced_at":        s.synced_at,
        "created_at":       s.created_at,
    }
    if record_count is not None:
        d["record_count"] = record_count
    return d


def _record_dict(r: AttendanceRecord) -> dict:
    return {
        "record_id":       str(r.record_id),
        "client_id":       str(r.client_id),
        "session_id":      str(r.session_id),
        "person_id":       str(r.person_id),
        "status":          r.status,
        "total_duration":  str(r.total_duration) if r.total_duration else None,
        "detection_count":  r.detection_count,
        "cycles_present":   r.cycles_present,
        "total_cycles":     r.total_cycles,
        "first_seen":       _epoch_to_iso(r.first_seen),
        "last_seen":        _epoch_to_iso(r.last_seen),
        "confidence_avg":  r.confidence_avg,
        "liveness_avg":    r.liveness_avg,
        "evidence_refs":   r.evidence_refs,
        "override_by":     str(r.override_by) if r.override_by else None,
        "override_reason": r.override_reason,
        "created_at":      r.created_at,
    }


async def _enrich_records(records: list[dict], db) -> list[dict]:
    """
    Add person_name, person_external_id, thumbnail, and course_name to each record.
    Uses bulk SQL queries to avoid N+1.
    """
    if not records:
        return records

    person_ids  = list({r["person_id"] for r in records})
    session_ids = list({r["session_id"] for r in records if r.get("session_id")})

    person_rows = (await db.execute(
        text("""
            SELECT p.person_id::text, p.name, p.external_id,
                   (SELECT fe.image_refs[1]
                    FROM face_embeddings fe
                    WHERE fe.person_id = p.person_id AND fe.is_active = true
                      AND array_length(fe.image_refs, 1) > 0
                    ORDER BY fe.created_at ASC LIMIT 1) AS first_image_ref
            FROM persons p
            WHERE p.person_id = ANY((:ids)::uuid[])
        """),
        {"ids": person_ids},
    )).fetchall()

    persons: dict[str, dict] = {}
    for row in person_rows:
        thumbnail = f"/api/enrollment/thumbnail/{row.person_id}" if row.first_image_ref else None
        persons[row.person_id] = {
            "person_name":        row.name or "Unknown",
            "person_external_id": row.external_id or "",
            "thumbnail":          thumbnail,
        }

    # Bulk-fetch course_name from sessions
    course_map: dict[str, str | None] = {}
    if session_ids:
        sess_rows = (await db.execute(
            text("SELECT session_id::text, course_name FROM sessions WHERE session_id = ANY((:ids)::uuid[])"),
            {"ids": session_ids},
        )).fetchall()
        for row in sess_rows:
            course_map[row.session_id] = row.course_name

    enriched = []
    for r in records:
        info = persons.get(r["person_id"], {
            "person_name": "Unknown", "person_external_id": "", "thumbnail": None
        })
        enriched.append({
            **r,
            **info,
            "course_name": course_map.get(r.get("session_id", ""), None),
        })
    return enriched


async def _get_session_or_404(db: DBSession, session_id: str) -> Session:
    s = await db.get(Session, uuid.UUID(session_id))
    if s is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return s


async def _get_record_or_404(db: DBSession, record_id: str) -> AttendanceRecord:
    r = await db.get(AttendanceRecord, uuid.UUID(record_id))
    if r is None:
        raise HTTPException(status_code=404, detail="Record not found")
    return r


# ── Sessions list ─────────────────────────────────────────────────────────────

@router.get("/sessions", dependencies=[_RD])
async def list_sessions(
    request:      Request,
    db:           DBSession,
    camera_id:    str | None = Query(None),
    faculty_id:   str | None = Query(None),
    sync_status:  str | None = Query(None),
    date:         str | None = Query(None, description="YYYY-MM-DD calendar day filter"),
    date_from:    int | None = Query(None, description="epoch"),
    date_to:      int | None = Query(None, description="epoch"),
    limit:        int        = Query(50, ge=1, le=200),
    offset:       int        = Query(0, ge=0),
) -> dict:
    # Convert calendar day to epoch range if provided
    if date and not date_from and not date_to:
        from datetime import datetime, timezone
        try:
            day = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            date_from = int(day.timestamp())
            date_to   = int(day.timestamp()) + 86400 - 1
        except ValueError:
            pass

    cid = resolve_client_id(request, require=False)
    q   = select(Session)
    if cid:
        q = q.where(Session.client_id == uuid.UUID(cid))
    if camera_id:
        q = q.where(Session.camera_id == uuid.UUID(camera_id))
    if faculty_id:
        q = q.where(Session.faculty_id == uuid.UUID(faculty_id))
    if sync_status:
        q = q.where(Session.sync_status == sync_status)
    if date_from:
        q = q.where(func.coalesce(Session.scheduled_start, Session.actual_start) >= date_from)
    if date_to:
        q = q.where(func.coalesce(Session.scheduled_start, Session.actual_start) <= date_to)

    total  = await db.scalar(select(func.count()).select_from(q.subquery())) or 0
    result = await db.execute(q.order_by(Session.created_at.desc()).offset(offset).limit(limit))
    sessions = result.scalars().all()

    # Attach record counts + camera/faculty names in bulk
    if sessions:
        session_ids = [s.session_id for s in sessions]
        cam_ids     = list({s.camera_id  for s in sessions if s.camera_id})
        fac_ids     = list({s.faculty_id for s in sessions if s.faculty_id})

        cnt_rows = (await db.execute(
            text("SELECT session_id::text, COUNT(*) FROM attendance_records WHERE session_id = ANY((:ids)::uuid[]) GROUP BY session_id"),
            {"ids": [str(x) for x in session_ids]},
        )).fetchall()
        cnt_map: dict[str, int] = {r[0]: r[1] for r in cnt_rows}

        cam_rows = (await db.execute(
            text("SELECT camera_id::text, name FROM cameras WHERE camera_id = ANY((:ids)::uuid[])"),
            {"ids": [str(x) for x in cam_ids]},
        )).fetchall() if cam_ids else []
        cam_map: dict[str, str] = {r[0]: r[1] for r in cam_rows}

        fac_rows = (await db.execute(
            text("SELECT person_id::text, name FROM persons WHERE person_id = ANY((:ids)::uuid[])"),
            {"ids": [str(x) for x in fac_ids]},
        )).fetchall() if fac_ids else []
        fac_map: dict[str, str] = {r[0]: r[1] for r in fac_rows}
    else:
        cnt_map, cam_map, fac_map = {}, {}, {}

    items = []
    for s in sessions:
        d = _session_dict(s, cnt_map.get(str(s.session_id), 0))
        d["student_count"]  = d.pop("record_count", 0)
        d["camera_name"]    = cam_map.get(str(s.camera_id),  "") if s.camera_id  else ""
        d["faculty_name"]   = fac_map.get(str(s.faculty_id), "") if s.faculty_id else ""
        items.append(d)

    return {"total": total, "offset": offset, "limit": limit, "items": items}


@router.get("/sessions/{session_id}", dependencies=[_RD])
async def get_session(session_id: str, db: DBSession) -> dict:
    s = await _get_session_or_404(db, session_id)
    cnt = await db.scalar(
        select(func.count()).select_from(AttendanceRecord).where(
            AttendanceRecord.session_id == s.session_id
        )
    ) or 0
    d = _session_dict(s, cnt)
    d["student_count"] = d.pop("record_count", 0)
    if s.camera_id:
        cam = await db.get(Camera, s.camera_id)
        d["camera_name"] = cam.name if cam else ""
    else:
        d["camera_name"] = ""
    if s.faculty_id:
        fac = await db.get(Person, s.faculty_id)
        d["faculty_name"] = fac.name if fac else ""
    else:
        d["faculty_name"] = ""
    return d


@router.get("/sessions/{session_id}/records", dependencies=[_RD])
async def session_records(
    session_id: str,
    request:    Request,
    db:         DBSession,
    status:     str | None = Query(None),
    limit:      int        = Query(100, ge=1, le=500),
    offset:     int        = Query(0, ge=0),
) -> dict:
    await _get_session_or_404(db, session_id)
    q = select(AttendanceRecord).where(
        AttendanceRecord.session_id == uuid.UUID(session_id)
    )
    if status:
        q = q.where(AttendanceRecord.status == status)
    total   = await db.scalar(select(func.count()).select_from(q.subquery())) or 0
    result  = await db.execute(q.order_by(AttendanceRecord.created_at).offset(offset).limit(limit))
    records = result.scalars().all()
    items   = await _enrich_records([_record_dict(r) for r in records], db)
    return {"total": total, "offset": offset, "limit": limit, "items": items}


# ── Cross-session records ─────────────────────────────────────────────────────

@router.get("/records", dependencies=[_RD])
async def list_records(
    request:   Request,
    db:        DBSession,
    person_id: str | None = Query(None),
    status:    str | None = Query(None),
    date_from: int | None = Query(None),
    date_to:   int | None = Query(None),
    limit:     int        = Query(50, ge=1, le=200),
    offset:    int        = Query(0, ge=0),
) -> dict:
    cid = resolve_client_id(request, require=False)
    q   = select(AttendanceRecord)
    if cid:
        q = q.where(AttendanceRecord.client_id == uuid.UUID(cid))
    if person_id:
        q = q.where(AttendanceRecord.person_id == uuid.UUID(person_id))
    if status:
        q = q.where(AttendanceRecord.status == status)
    if date_from:
        q = q.where(AttendanceRecord.first_seen >= date_from)
    if date_to:
        q = q.where(AttendanceRecord.first_seen <= date_to)
    total  = await db.scalar(select(func.count()).select_from(q.subquery())) or 0
    result = await db.execute(q.order_by(AttendanceRecord.created_at.desc()).offset(offset).limit(limit))
    items  = await _enrich_records([_record_dict(r) for r in result.scalars().all()], db)
    return {
        "total":  total,
        "offset": offset,
        "limit":  limit,
        "items":  items,
    }


# ── Held sessions ─────────────────────────────────────────────────────────────

@router.get("/held", dependencies=[_RD])
async def list_held(
    request: Request,
    db:      DBSession,
    reason:  str | None = Query(None, description="Filter by held_reason substring"),
    limit:   int        = Query(50, ge=1, le=200),
    offset:  int        = Query(0, ge=0),
) -> dict:
    cid = resolve_client_id(request, require=False)
    q   = select(Session).where(Session.sync_status == SyncStatus.HELD)
    if cid:
        q = q.where(Session.client_id == uuid.UUID(cid))
    if reason:
        q = q.where(Session.held_reason.ilike(f"%{reason}%"))
    total    = await db.scalar(select(func.count()).select_from(q.subquery())) or 0
    result   = await db.execute(q.order_by(Session.created_at.desc()).offset(offset).limit(limit))
    sessions = result.scalars().all()

    if sessions:
        cam_ids = list({s.camera_id  for s in sessions if s.camera_id})
        fac_ids = list({s.faculty_id for s in sessions if s.faculty_id})
        sid_list = [str(s.session_id) for s in sessions]

        cnt_rows = (await db.execute(
            text("SELECT session_id::text, COUNT(*) FROM attendance_records WHERE session_id = ANY((:ids)::uuid[]) GROUP BY session_id"),
            {"ids": sid_list},
        )).fetchall()
        cnt_map: dict[str, int] = {r[0]: r[1] for r in cnt_rows}

        cam_rows = (await db.execute(
            text("SELECT camera_id::text, name FROM cameras WHERE camera_id = ANY((:ids)::uuid[])"),
            {"ids": [str(x) for x in cam_ids]},
        )).fetchall() if cam_ids else []
        cam_map: dict[str, str] = {r[0]: r[1] for r in cam_rows}

        fac_rows = (await db.execute(
            text("SELECT person_id::text, name FROM persons WHERE person_id = ANY((:ids)::uuid[])"),
            {"ids": [str(x) for x in fac_ids]},
        )).fetchall() if fac_ids else []
        fac_map: dict[str, str] = {r[0]: r[1] for r in fac_rows}
    else:
        cnt_map, cam_map, fac_map = {}, {}, {}

    items = []
    for s in sessions:
        d = _session_dict(s, cnt_map.get(str(s.session_id), 0))
        d["student_count"] = d.pop("record_count", 0)
        d["camera_name"]   = cam_map.get(str(s.camera_id),  "") if s.camera_id  else ""
        d["faculty_name"]  = fac_map.get(str(s.faculty_id), "") if s.faculty_id else ""
        items.append(d)

    return {"total": total, "offset": offset, "limit": limit, "items": items}


# ── Export ────────────────────────────────────────────────────────────────────

@router.get("/export", dependencies=[_RD])
async def export_attendance(
    request:    Request,
    db:         DBSession,
    session_id: str | None = Query(None),
    date_from:  int | None = Query(None),
    date_to:    int | None = Query(None),
    fmt:        str        = Query("csv", pattern="^(csv|json)$"),
) -> Any:
    """Export attendance records as CSV or JSON."""
    cid = resolve_client_id(request)
    q   = (
        select(
            AttendanceRecord.record_id,
            AttendanceRecord.session_id,
            AttendanceRecord.person_id,
            AttendanceRecord.status,
            AttendanceRecord.detection_count,
            AttendanceRecord.first_seen,
            AttendanceRecord.last_seen,
            AttendanceRecord.confidence_avg,
            AttendanceRecord.override_reason,
            Person.name.label("person_name"),
            Person.role.label("person_role"),
            Person.external_id,
        )
        .join(Person, Person.person_id == AttendanceRecord.person_id)
        .where(AttendanceRecord.client_id == uuid.UUID(cid))
    )
    if session_id:
        q = q.where(AttendanceRecord.session_id == uuid.UUID(session_id))
    if date_from:
        q = q.where(AttendanceRecord.first_seen >= date_from)
    if date_to:
        q = q.where(AttendanceRecord.first_seen <= date_to)

    result = await db.execute(q.order_by(AttendanceRecord.first_seen))
    rows   = result.fetchall()

    if fmt == "json":
        data = [dict(r._mapping) for r in rows]
        # Cast non-serialisable types
        for d in data:
            for k, v in d.items():
                if hasattr(v, "hex"):  # UUID
                    d[k] = str(v)
        return {"total": len(data), "records": data}

    # CSV
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "record_id", "session_id", "person_id", "external_id",
        "person_name", "person_role",
        "status", "detection_count",
        "first_seen", "last_seen", "confidence_avg", "override_reason",
    ])
    for r in rows:
        writer.writerow([
            str(r.record_id), str(r.session_id), str(r.person_id), r.external_id,
            r.person_name, r.person_role,
            r.status, r.detection_count,
            r.first_seen, r.last_seen, r.confidence_avg, r.override_reason,
        ])
    output.seek(0)
    ts = now_epoch()
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="attendance_{cid}_{ts}.csv"'},
    )


# ── Held actions ──────────────────────────────────────────────────────────────

@router.post("/held/{session_id}/force-sync", dependencies=[_UP])
async def force_sync(session_id: str, body: HeldActionRequest, request: Request, db: DBSession) -> dict:
    """Approve a held session and re-publish to ERP via Kafka."""
    s = await _get_session_or_404(db, session_id)
    if s.sync_status not in (SyncStatus.HELD, SyncStatus.FAILED):
        raise HTTPException(status_code=409, detail=f"Session is {s.sync_status}, not HELD or FAILED")

    s.sync_status = SyncStatus.SYNCED
    s.synced_at   = now_epoch()
    await db.flush()

    # Publish override event
    try:
        producer = request.app.state.kafka_producer
        actor_id = request.state.user_id
        producer.produce(
            "admin.overrides",
            key=session_id.encode(),
            value=json.dumps({
                "override_id": uuid.uuid4().hex,
                "session_id":  session_id,
                "client_id":   str(s.client_id),
                "action":      "FORCE_SYNC",
                "actor_id":    str(actor_id),
                "reason":      body.reason,
            }).encode(),
        )
        producer.poll(0)
    except Exception:
        pass

    audit(request, "FORCE_SYNC", "session", session_id, {"reason": body.reason})
    return {"session_id": session_id, "sync_status": "SYNCED"}


@router.post("/held/{session_id}/discard", dependencies=[_UP])
async def discard_session(session_id: str, body: HeldActionRequest, request: Request, db: DBSession) -> dict:
    """Mark a held session as DISCARDED — it will never be sent to ERP."""
    s = await _get_session_or_404(db, session_id)
    s.sync_status = SyncStatus.DISCARDED
    s.held_reason = f"admin_discard:{request.state.user_id}:{body.reason or ''}"
    await db.flush()

    try:
        producer = request.app.state.kafka_producer
        producer.produce(
            "admin.overrides",
            key=session_id.encode(),
            value=json.dumps({
                "override_id": uuid.uuid4().hex,
                "session_id":  session_id,
                "client_id":   str(s.client_id),
                "action":      "DISCARD",
                "actor_id":    str(request.state.user_id),
                "reason":      body.reason,
            }).encode(),
        )
        producer.poll(0)
    except Exception:
        pass

    audit(request, "DISCARD_SESSION", "session", session_id, {"reason": body.reason})
    return {"session_id": session_id, "sync_status": "DISCARDED"}


# ── Evidence image proxy ──────────────────────────────────────────────────────

@router.get("/records/{record_id}/evidence")
async def serve_record_evidence(
    record_id: str, request: Request, db: DBSession,
    index: int = Query(0, ge=0, description="Index into evidence_refs array"),
):
    """
    Proxy an evidence image (face crop captured at recognition time)
    for an attendance record. Use ?index=N to select which image (default 0).
    """
    import asyncio
    import pathlib
    from fastapi.responses import FileResponse, Response

    r = await _get_record_or_404(db, record_id)
    refs = r.evidence_refs
    if not refs:
        raise HTTPException(status_code=404, detail="No evidence image for this record")
    if index >= len(refs):
        raise HTTPException(status_code=404, detail=f"Index {index} out of range ({len(refs)} images)")

    ref = refs[index]

    if ref.startswith("local:"):
        raw_key = ref[6:]
        base_dir = getattr(request.app.state.settings, "enrollment_fallback_dir", "/enrollment-images")
        fpath = pathlib.Path(base_dir) / raw_key
        if not fpath.is_file():
            raise HTTPException(status_code=404, detail="Evidence image not found on disk")
        return FileResponse(fpath, media_type="image/jpeg")

    minio = request.app.state.minio
    bucket = getattr(request.app.state.settings, "minio_evidence_bucket", "face-evidence")
    # Try evidence bucket first, then enrollment bucket (older records may use it)
    for bkt in (bucket, "face-enrollment"):
        try:
            data = await asyncio.to_thread(lambda b=bkt: minio.get_object(b, ref).read())
            return Response(content=data, media_type="image/jpeg",
                            headers={"Cache-Control": "max-age=3600"})
        except Exception:
            continue
    raise HTTPException(status_code=404, detail="Evidence image not found in storage")


# ── Record override ───────────────────────────────────────────────────────────

@router.post("/records/{record_id}/override", dependencies=[_UP])
async def override_record(record_id: str, body: OverrideRequest, request: Request, db: DBSession) -> dict:
    """Manually override a student's attendance status."""
    body.validate_status()
    await rate_limit(
        request.app.state.redis,
        _OVERRIDE_RL_KEY.format(client_id=resolve_client_id(request)),
        _OVERRIDE_MAX,
        _OVERRIDE_WIN_S,
    )
    r = await _get_record_or_404(db, record_id)
    old_status  = r.status
    r.status    = body.status
    r.override_by     = uuid.UUID(request.state.user_id)
    r.override_reason = body.reason
    await db.flush()

    audit(request, "OVERRIDE_RECORD", "attendance_record", record_id,
          {"from": old_status, "to": body.status, "reason": body.reason})
    return _record_dict(r)


# ── Unknown detections ────────────────────────────────────────────────────────

class AssignUnknownRequest(BaseModel):
    person_id: str


def _unknown_dict(u: UnknownDetection) -> dict:
    return {
        "detection_id":      str(u.detection_id),
        "session_id":        str(u.session_id) if u.session_id else None,
        "camera_id":         str(u.camera_id) if u.camera_id else None,
        "image_ref":         u.image_ref,
        "confidence":        u.confidence,
        "liveness_score":    u.liveness_score,
        "detected_at":       u.detected_at,
        "status":            u.status,
        "assigned_person_id": str(u.assigned_person_id) if u.assigned_person_id else None,
        "assigned_at":       u.assigned_at,
    }


@router.get("/unknown", dependencies=[_RD])
async def list_unknown_detections(
    request: Request,
    db: DBSession,
    session_id: str | None = Query(None),
    status: str = Query("UNASSIGNED"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> list[dict]:
    """List unrecognized face detections, optionally filtered by session."""
    client_id = resolve_client_id(request)
    q = (
        select(UnknownDetection)
        .where(
            UnknownDetection.client_id == uuid.UUID(client_id),
            UnknownDetection.status == status,
        )
        .order_by(UnknownDetection.detected_at.desc())
        .limit(limit)
        .offset(offset)
    )
    if session_id:
        q = q.where(UnknownDetection.session_id == uuid.UUID(session_id))
    result = await db.execute(q)
    return [_unknown_dict(u) for u in result.scalars().all()]


@router.get("/unknown/{detection_id}/image")
async def serve_unknown_image(detection_id: str, request: Request, db: DBSession):
    """Proxy the face-crop image for an unknown detection (no auth — UUID is capability)."""
    import asyncio, pathlib
    from fastapi.responses import FileResponse, Response

    result = await db.execute(
        select(UnknownDetection).where(
            UnknownDetection.detection_id == uuid.UUID(detection_id),
        )
    )
    det = result.scalar_one_or_none()
    if det is None or not det.image_ref:
        raise HTTPException(status_code=404, detail="Not found")

    ref = det.image_ref
    if ref.startswith("local:"):
        raw_key = ref[6:]
        base_dir = getattr(request.app.state.settings, "enrollment_fallback_dir", "/enrollment-images")
        fpath = pathlib.Path(base_dir) / raw_key
        if not fpath.is_file():
            raise HTTPException(status_code=404, detail="Image not found on disk")
        return FileResponse(fpath, media_type="image/jpeg")

    minio = request.app.state.minio
    try:
        data = await asyncio.to_thread(
            lambda: minio.get_object("face-evidence", ref).read()
        )
        return Response(content=data, media_type="image/jpeg",
                        headers={"Cache-Control": "max-age=3600"})
    except Exception:
        raise HTTPException(status_code=404, detail="Image not found in storage")


@router.post("/unknown/{detection_id}/assign", dependencies=[_UP])
async def assign_unknown_detection(
    detection_id: str,
    body: AssignUnknownRequest,
    request: Request,
    db: DBSession,
) -> dict:
    """
    Assign an unrecognized detection to a known person and mark their
    attendance as Present for the associated session.
    """
    client_id = resolve_client_id(request)
    result = await db.execute(
        select(UnknownDetection).where(
            UnknownDetection.detection_id == uuid.UUID(detection_id),
            UnknownDetection.client_id == uuid.UUID(client_id),
        )
    )
    det = result.scalar_one_or_none()
    if det is None:
        raise HTTPException(status_code=404, detail="Detection not found")
    if det.status != "UNASSIGNED":
        raise HTTPException(status_code=409, detail="Already assigned or dismissed")

    person_id = uuid.UUID(body.person_id)
    now = now_epoch()
    det.assigned_person_id = person_id
    det.assigned_by        = uuid.UUID(request.state.user_id)
    det.assigned_at        = now
    det.status             = "ASSIGNED"

    # Mark/upsert attendance as Present for this session
    if det.session_id:
        await db.execute(
            text("""
                INSERT INTO attendance_records
                    (client_id, session_id, person_id, status, detection_count,
                     first_seen, last_seen, cycles_present, total_cycles)
                VALUES
                    ((:cid)::uuid, (:sid)::uuid, (:pid)::uuid,
                     'P', 1,
                     :now, :now, 1, 1)
                ON CONFLICT (session_id, person_id)
                DO UPDATE SET
                    status         = 'P',
                    override_by    = (:uid)::uuid,
                    override_reason = 'assigned from unknown detection'
            """),
            {
                "cid": client_id,
                "sid": str(det.session_id),
                "pid": str(person_id),
                "uid": request.state.user_id,
                "now": now,
            },
        )

    await db.commit()

    # ── Re-enroll the face chip so FAISS learns this camera-angle appearance ──
    # The face chip in image_ref is an aligned 112×112 crop from the live camera.
    # Adding it as a new embedding improves future auto-recognition for this person.
    if det.image_ref:
        asyncio.create_task(
            _enroll_unknown_chip(
                request.app,
                client_id,
                str(person_id),
                det.image_ref,
            )
        )

    return _unknown_dict(det)


async def _enroll_unknown_chip(app, client_id: str, person_id: str, image_ref: str) -> None:
    """
    Background task: embed the unknown face chip and APPEND it as a new
    face_embeddings row WITHOUT deactivating existing embeddings.

    enroll_person() does a clean-slate re-enroll (deactivates old rows).
    Here we only add a new row so the person keeps all prior templates and
    gains the live-camera angle for better future recognition.
    """
    import logging as _log, uuid as _uuid, time as _time
    _logger = _log.getLogger(__name__)

    try:
        face_repo = getattr(app.state, "face_repo", None)
        pipeline  = getattr(app.state, "ai_pipeline", None)
        if face_repo is None or pipeline is None:
            _logger.warning("_enroll_unknown_chip: face_repo=%s pipeline=%s — skipping", face_repo, pipeline)
            return

        # ── 1. Download the chip bytes ────────────────────────────────────────
        image_bytes: bytes | None = None
        if image_ref.startswith("local:"):
            import pathlib as _pl
            local_base = getattr(face_repo, "_local_dir", "/enrollment-images")
            rel  = image_ref[len("local:"):]
            path = _pl.Path(local_base) / rel
            if path.exists():
                image_bytes = path.read_bytes()
        else:
            minio = getattr(face_repo, "_minio", None)
            if minio is not None:
                try:
                    resp       = await asyncio.to_thread(lambda: minio.get_object("face-evidence", image_ref))
                    image_bytes = await asyncio.to_thread(resp.read)
                    await asyncio.to_thread(resp.close)
                    await asyncio.to_thread(resp.release_conn)
                except Exception as exc:
                    _logger.debug("_enroll_unknown_chip: MinIO read failed: %s", exc)

        if not image_bytes:
            _logger.debug("_enroll_unknown_chip: could not download chip for %s", image_ref)
            return

        # ── 2. Decode image → numpy ───────────────────────────────────────────
        import cv2 as _cv2, numpy as _np
        arr  = _np.frombuffer(image_bytes, dtype=_np.uint8)
        chip = _cv2.imdecode(arr, _cv2.IMREAD_COLOR)
        if chip is None:
            _logger.warning("_enroll_unknown_chip: image decode failed for %s", image_ref)
            return

        # ── 3. Embed ─────────────────────────────────────────────────────────
        # As of the unknown-face quality upgrade, chips stored in face-evidence
        # are context-padded frame crops (~200–400 px) rather than the legacy
        # 112×112 aligned ArcFace chips.  We try detection + alignment first;
        # if that fails (e.g. legacy chip from before the upgrade, or a crop
        # too tight for SCRFD), we fall back to the direct-embed path.
        emb = None
        aligned: _np.ndarray | None = None
        try:
            h_, w_ = chip.shape[:2]
            if min(h_, w_) >= 112 and max(h_, w_) > 150:
                frame_for_det = chip
                # Upscale small padded crops so SCRFD has enough pixels
                if max(h_, w_) < 240:
                    scale = 240.0 / float(max(h_, w_))
                    frame_for_det = _cv2.resize(
                        chip, (int(w_ * scale), int(h_ * scale)),
                        interpolation=_cv2.INTER_LINEAR,
                    )
                result = await asyncio.to_thread(pipeline.process_frame, frame_for_det)
                faces = getattr(result, "faces_with_embeddings", None) or []
                if faces:
                    # Largest face wins (handles bystanders in padded crop)
                    def _area(f):
                        b = f.face.bbox
                        return float((b[2] - b[0]) * (b[3] - b[1]))
                    best = max(faces, key=_area)
                    emb = best.embedding
                    aligned = best.face_chip
        except Exception as exc:
            _logger.debug("_enroll_unknown_chip: process_frame failed: %s", exc)

        if emb is None:
            # Legacy path: treat the image as a pre-aligned 112×112 chip
            legacy = chip if chip.shape[:2] == (112, 112) else _cv2.resize(chip, (112, 112))
            emb = await asyncio.to_thread(pipeline.get_embedding, legacy)
            aligned = legacy
        if emb is None or not hasattr(emb, '__len__') or len(emb) == 0:
            _logger.warning("_enroll_unknown_chip: embedding failed for %s", image_ref)
            return
        emb_str = "[" + ",".join(f"{x:.8f}" for x in emb.tolist()) + "]"

        # ── 4. INSERT new embedding row — existing rows stay active ──────────
        import sqlalchemy as _sa
        eid = str(_uuid.uuid4())
        now = int(_time.time())

        async with face_repo._session() as db:
            await db.execute(
                _sa.text("""
                    INSERT INTO face_embeddings
                        (embedding_id, client_id, person_id, embedding,
                         version, source, confidence_avg, image_refs,
                         is_active, quality_score, created_at)
                    VALUES
                        ((:eid)::uuid, (:cid)::uuid, (:pid)::uuid, (:emb)::vector,
                         1, 'LIVE_ASSIGNMENT', 1.0, :refs,
                         true, 1.0, :now)
                    ON CONFLICT DO NOTHING
                """),
                {
                    "eid": eid, "cid": client_id, "pid": person_id,
                    "emb": emb_str,
                    "refs": [image_ref],
                    "now":  now,
                },
            )
            await db.commit()

        # ── 5. Rebuild FAISS so the new embedding is searchable immediately ──
        await face_repo.rebuild_faiss_index(client_id)

        _logger.info(
            "assign_unknown: appended live embedding → person=%s  embedding=%s  ref=%s",
            person_id, eid, image_ref,
        )

    except Exception as exc:
        import logging as _log2
        _log2.getLogger(__name__).warning(
            "assign_unknown: re-enrollment failed (non-fatal): %s", exc
        )


@router.post("/unknown/{detection_id}/dismiss", dependencies=[_UP])
async def dismiss_unknown_detection(
    detection_id: str,
    request: Request,
    db: DBSession,
) -> dict:
    """Mark an unknown detection as dismissed (not worth assigning)."""
    client_id = resolve_client_id(request)
    result = await db.execute(
        select(UnknownDetection).where(
            UnknownDetection.detection_id == uuid.UUID(detection_id),
            UnknownDetection.client_id == uuid.UUID(client_id),
        )
    )
    det = result.scalar_one_or_none()
    if det is None:
        raise HTTPException(status_code=404, detail="Detection not found")
    det.status      = "DISMISSED"
    det.assigned_by = uuid.UUID(request.state.user_id)
    det.assigned_at = now_epoch()
    await db.commit()
    return _unknown_dict(det)
