"""
Timetables API — CRUD for timetables and their entries.

Timetables define a weekly schedule of sessions.  Each entry has a day_of_week,
start_time, end_time, course info, faculty, and roster (dataset_ids or explicit
roster_ids).  Cameras can be linked to a timetable via timetable_id.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, field_validator
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.api._shared import audit, now_epoch, resolve_client_id
from app.deps import DBSession
from app.middleware.auth import require_permission
from app.models.timetables import Timetable, TimetableEntry

router = APIRouter(prefix="/api/timetables", tags=["timetables"])

_DAYS = list(range(7))  # 0=Mon … 6=Sun
_TIME_RE = __import__("re").compile(r"^\d{1,2}:\d{2}$")

_RD = require_permission("cameras:read")
_WR = require_permission("cameras:update")
_CR = require_permission("cameras:create")
_DL = require_permission("cameras:delete")


# ── Pydantic models ────────────────────────────────────────────────────────────

class EntryIn(BaseModel):
    day_of_week:  int
    start_time:   str                   # "HH:MM"
    end_time:     str                   # "HH:MM"
    course_id:    str | None = None
    course_name:  str | None = None
    faculty_id:   str | None = None
    dataset_ids:  list[str] | None = None
    roster_ids:   list[str] | None = None

    @field_validator("day_of_week")
    @classmethod
    def _dow(cls, v: int) -> int:
        if v not in _DAYS:
            raise ValueError("day_of_week must be 0–6 (Mon–Sun)")
        return v

    @field_validator("start_time", "end_time")
    @classmethod
    def _time(cls, v: str) -> str:
        if not _TIME_RE.match(v):
            raise ValueError("time must be HH:MM")
        return v


class TimetableCreate(BaseModel):
    name:        str
    description: str | None = None
    timezone:    str = "UTC"
    entries:     list[EntryIn] = []
    client_id:   str | None = None   # SUPER_ADMIN only


class TimetableUpdate(BaseModel):
    name:        str | None = None
    description: str | None = None
    timezone:    str | None = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tt_dict(tt: Timetable, extra_entries: list | None = None) -> dict:
    entries = extra_entries if extra_entries is not None else (tt.entries or [])
    return {
        "timetable_id": str(tt.timetable_id),
        "client_id":    str(tt.client_id),
        "name":         tt.name,
        "description":  tt.description,
        "timezone":     tt.timezone,
        "created_at":   tt.created_at,
        "updated_at":   tt.updated_at,
        "entries":      [_entry_dict(e) for e in entries],
    }


def _entry_dict(e: TimetableEntry) -> dict:
    return {
        "entry_id":    str(e.entry_id),
        "day_of_week": e.day_of_week,
        "start_time":  e.start_time,
        "end_time":    e.end_time,
        "course_id":   e.course_id,
        "course_name": e.course_name,
        "faculty_id":  str(e.faculty_id) if e.faculty_id else None,
        "dataset_ids": e.dataset_ids,
        "roster_ids":  e.roster_ids,
        "created_at":  e.created_at,
    }


async def _get_tt(db: DBSession, timetable_id: str, client_id: str) -> Timetable:
    result = await db.execute(
        select(Timetable)
        .where(
            Timetable.timetable_id == uuid.UUID(timetable_id),
            Timetable.client_id == uuid.UUID(client_id),
        )
        .options(selectinload(Timetable.entries))
    )
    tt = result.scalar_one_or_none()
    if tt is None:
        raise HTTPException(status_code=404, detail="Timetable not found")
    return tt


# ── Default-seed data ─────────────────────────────────────────────────────────

_DEFAULT_SCHEDULE = [
    {"day_of_week": 0, "start_time": "10:00", "end_time": "19:00", "course_name": "Mathematics"},
    {"day_of_week": 1, "start_time": "10:00", "end_time": "19:00", "course_name": "Computer Science"},
    {"day_of_week": 2, "start_time": "10:00", "end_time": "19:00", "course_name": "Physics"},
    {"day_of_week": 3, "start_time": "10:00", "end_time": "19:00", "course_name": "Chemistry"},
    {"day_of_week": 4, "start_time": "10:00", "end_time": "19:00", "course_name": "English Literature"},
    {"day_of_week": 5, "start_time": "10:00", "end_time": "19:00", "course_name": "History"},
    {"day_of_week": 6, "start_time": "10:00", "end_time": "19:00", "course_name": "Arts & Culture"},
]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("", dependencies=[_RD])
async def list_timetables(request: Request, db: DBSession) -> list[dict]:
    client_id = resolve_client_id(request)
    result = await db.execute(
        select(Timetable)
        .where(Timetable.client_id == uuid.UUID(client_id))
        .options(selectinload(Timetable.entries))
        .order_by(Timetable.name)
    )
    return [_tt_dict(tt) for tt in result.scalars().all()]


@router.post("", dependencies=[_CR])
async def create_timetable(
    body: TimetableCreate,
    request: Request,
    db: DBSession,
) -> dict:
    client_id = resolve_client_id(request, body.client_id)
    now = now_epoch()

    tt = Timetable(
        client_id   = uuid.UUID(client_id),
        name        = body.name,
        description = body.description,
        timezone    = body.timezone,
        created_at  = now,
        updated_at  = now,
    )
    db.add(tt)
    await db.flush()   # get timetable_id

    entries = []
    for e in body.entries:
        entry = TimetableEntry(
            timetable_id = tt.timetable_id,
            client_id    = uuid.UUID(client_id),
            day_of_week  = e.day_of_week,
            start_time   = e.start_time,
            end_time     = e.end_time,
            course_id    = e.course_id,
            course_name  = e.course_name,
            faculty_id   = uuid.UUID(e.faculty_id) if e.faculty_id else None,
            dataset_ids  = e.dataset_ids,
            roster_ids   = e.roster_ids,
            created_at   = now,
        )
        db.add(entry)
        entries.append(entry)

    await db.flush()   # write entries, auto-commit on context-manager exit
    audit(request, "timetable.create", "timetable", str(tt.timetable_id))
    return _tt_dict(tt, extra_entries=entries)


@router.get("/{timetable_id}", dependencies=[_RD])
async def get_timetable(timetable_id: str, request: Request, db: DBSession) -> dict:
    client_id = resolve_client_id(request)
    tt = await _get_tt(db, timetable_id, client_id)
    return _tt_dict(tt)


@router.patch("/{timetable_id}", dependencies=[_WR])
async def update_timetable(
    timetable_id: str,
    body: TimetableUpdate,
    request: Request,
    db: DBSession,
) -> dict:
    client_id = resolve_client_id(request)
    tt = await _get_tt(db, timetable_id, client_id)
    if body.name is not None:
        tt.name = body.name
    if body.description is not None:
        tt.description = body.description
    if body.timezone is not None:
        tt.timezone = body.timezone
    tt.updated_at = now_epoch()
    await db.flush()
    audit(request, "timetable.update", "timetable", timetable_id)
    return _tt_dict(tt)


@router.delete("/{timetable_id}", dependencies=[_DL])
async def delete_timetable(timetable_id: str, request: Request, db: DBSession) -> dict:
    client_id = resolve_client_id(request)
    tt = await _get_tt(db, timetable_id, client_id)
    await db.delete(tt)
    await db.flush()
    audit(request, "timetable.delete", "timetable", timetable_id)
    return {"deleted": timetable_id}


@router.post("/seed-default", dependencies=[_CR])
async def seed_default_timetable(request: Request, db: DBSession) -> dict:
    """
    Create the sample 'Default (Mon–Sun 10:00–19:00)' timetable for the current
    client if one does not already exist.  Safe to call multiple times — a 200
    response means either it was just created or it already existed.
    """
    client_id = resolve_client_id(request)

    # Return early if ANY timetable already exists for this client
    existing = await db.execute(
        select(Timetable).where(Timetable.client_id == uuid.UUID(client_id)).limit(1)
    )
    if existing.scalar_one_or_none() is not None:
        return {"created": False, "message": "Timetable already exists"}

    now = now_epoch()
    tt = Timetable(
        client_id   = uuid.UUID(client_id),
        name        = "Default (Mon–Sun 10:00–19:00)",
        description = "Sample timetable covering all 7 days 10 AM–7 PM",
        timezone    = "UTC",
        created_at  = now,
        updated_at  = now,
    )
    db.add(tt)
    await db.flush()

    for slot in _DEFAULT_SCHEDULE:
        db.add(TimetableEntry(
            timetable_id = tt.timetable_id,
            client_id    = uuid.UUID(client_id),
            day_of_week  = slot["day_of_week"],
            start_time   = slot["start_time"],
            end_time     = slot["end_time"],
            course_name  = slot["course_name"],
            created_at   = now,
        ))

    await db.flush()
    audit(request, "timetable.seed_default", "timetable", str(tt.timetable_id))
    return {"created": True, "timetable_id": str(tt.timetable_id)}


# ── Entry sub-resource ────────────────────────────────────────────────────────

@router.post("/{timetable_id}/entries", dependencies=[_WR])
async def add_entry(
    timetable_id: str,
    body: EntryIn,
    request: Request,
    db: DBSession,
) -> dict:
    client_id = resolve_client_id(request)
    tt = await _get_tt(db, timetable_id, client_id)
    entry = TimetableEntry(
        timetable_id = tt.timetable_id,
        client_id    = uuid.UUID(client_id),
        day_of_week  = body.day_of_week,
        start_time   = body.start_time,
        end_time     = body.end_time,
        course_id    = body.course_id,
        course_name  = body.course_name,
        faculty_id   = uuid.UUID(body.faculty_id) if body.faculty_id else None,
        dataset_ids  = body.dataset_ids,
        roster_ids   = body.roster_ids,
        created_at   = now_epoch(),
    )
    db.add(entry)
    await db.flush()
    return _entry_dict(entry)


@router.put("/{timetable_id}/entries/{entry_id}", dependencies=[_WR])
async def update_entry(
    timetable_id: str,
    entry_id: str,
    body: EntryIn,
    request: Request,
    db: DBSession,
) -> dict:
    client_id = resolve_client_id(request)
    result = await db.execute(
        select(TimetableEntry).where(
            TimetableEntry.entry_id    == uuid.UUID(entry_id),
            TimetableEntry.timetable_id == uuid.UUID(timetable_id),
            TimetableEntry.client_id   == uuid.UUID(client_id),
        )
    )
    entry = result.scalar_one_or_none()
    if entry is None:
        raise HTTPException(status_code=404, detail="Entry not found")

    entry.day_of_week = body.day_of_week
    entry.start_time  = body.start_time
    entry.end_time    = body.end_time
    entry.course_id   = body.course_id
    entry.course_name = body.course_name
    entry.faculty_id  = uuid.UUID(body.faculty_id) if body.faculty_id else None
    entry.dataset_ids = body.dataset_ids
    entry.roster_ids  = body.roster_ids
    await db.flush()
    return _entry_dict(entry)


@router.delete("/{timetable_id}/entries/{entry_id}", dependencies=[_WR])
async def delete_entry(
    timetable_id: str,
    entry_id: str,
    request: Request,
    db: DBSession,
) -> dict:
    client_id = resolve_client_id(request)
    result = await db.execute(
        select(TimetableEntry).where(
            TimetableEntry.entry_id    == uuid.UUID(entry_id),
            TimetableEntry.timetable_id == uuid.UUID(timetable_id),
            TimetableEntry.client_id   == uuid.UUID(client_id),
        )
    )
    entry = result.scalar_one_or_none()
    if entry is None:
        raise HTTPException(status_code=404, detail="Entry not found")
    await db.delete(entry)
    await db.flush()
    return {"deleted": entry_id}
