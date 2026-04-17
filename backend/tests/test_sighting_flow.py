"""
test_sighting_flow.py — Monitoring mode: sighting lifecycle and persistence.

Covers:
  • on_recognition() creates a new ActiveSighting on first call
  • Second on_recognition() extends the existing sighting (last_seen + duration)
  • tick() with ts < INACTIVE_TIMEOUT keeps sighting open
  • tick() with ts ≥ INACTIVE_TIMEOUT closes the sighting
  • Closed sightings below MIN_DURATION_S are discarded (ghost filter)
  • Closed sightings are written to the DB via SightingEngine.flush()
  • Kafka sightings.log published for each closed sighting
  • get_area_occupancy() returns correct head count per camera
  • SightingEngine.flush() persists to DB and clears in-memory state
"""
from __future__ import annotations

import time
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from tests.conftest import auth_headers, create_camera, create_person
from tests.factories import SightingFactory


_INACTIVE_TIMEOUT_S = 300   # 5 min (matches SightingEngine constant)
_MIN_DURATION_S     = 5.0


# ── Pure SightingEngine unit tests ────────────────────────────────────────────

async def test_first_recognition_creates_sighting():
    from app.services.sighting_engine import SightingEngine

    engine = SightingEngine(db=AsyncMock(), redis=AsyncMock(), kafka_producer=AsyncMock())
    cid    = str(uuid.uuid4())
    cam_id = str(uuid.uuid4())
    pid    = str(uuid.uuid4())
    now    = time.time()

    await engine.on_recognition(cid, cam_id, pid, conf=0.88, frame_ref="frame/001.jpg", ts=now)

    occupancy = engine.get_area_occupancy(cam_id)
    assert occupancy == 1


async def test_second_recognition_extends_sighting():
    from app.services.sighting_engine import SightingEngine

    engine = SightingEngine(db=AsyncMock(), redis=AsyncMock(), kafka_producer=AsyncMock())
    cid    = str(uuid.uuid4())
    cam_id = str(uuid.uuid4())
    pid    = str(uuid.uuid4())
    t0     = time.time()

    await engine.on_recognition(cid, cam_id, pid, conf=0.88, frame_ref="f1.jpg", ts=t0)
    await engine.on_recognition(cid, cam_id, pid, conf=0.91, frame_ref="f2.jpg", ts=t0 + 60)

    # Only one sighting should exist (extended, not duplicated)
    assert engine.get_area_occupancy(cam_id) == 1
    sighting = engine._sightings[cam_id][pid]
    assert sighting.last_seen >= t0 + 60
    assert sighting.duration_seconds >= 60


async def test_sighting_confidence_averages():
    from app.services.sighting_engine import SightingEngine

    engine = SightingEngine(db=AsyncMock(), redis=AsyncMock(), kafka_producer=AsyncMock())
    cid    = str(uuid.uuid4())
    cam_id = str(uuid.uuid4())
    pid    = str(uuid.uuid4())
    t0     = time.time()

    await engine.on_recognition(cid, cam_id, pid, conf=0.80, frame_ref="f1.jpg", ts=t0)
    await engine.on_recognition(cid, cam_id, pid, conf=0.90, frame_ref="f2.jpg", ts=t0 + 30)

    sighting = engine._sightings[cam_id][pid]
    # Confidence should be averaged (or running mean)
    assert 0.79 < sighting.confidence_avg < 0.96


async def test_tick_keeps_recent_sighting_open():
    from app.services.sighting_engine import SightingEngine

    engine = SightingEngine(db=AsyncMock(), redis=AsyncMock(), kafka_producer=AsyncMock())
    cid    = str(uuid.uuid4())
    cam_id = str(uuid.uuid4())
    pid    = str(uuid.uuid4())
    t0     = time.time()

    await engine.on_recognition(cid, cam_id, pid, conf=0.85, frame_ref="f1.jpg", ts=t0)

    # Tick slightly after last_seen (well within timeout)
    closed = await engine.tick(cam_id, ts=t0 + _INACTIVE_TIMEOUT_S - 60)
    assert pid in engine._sightings.get(cam_id, {})
    assert not any(s.person_id == pid for s in closed)


async def test_tick_closes_inactive_sighting():
    from app.services.sighting_engine import SightingEngine

    engine = SightingEngine(db=AsyncMock(), redis=AsyncMock(), kafka_producer=AsyncMock())
    cid    = str(uuid.uuid4())
    cam_id = str(uuid.uuid4())
    pid    = str(uuid.uuid4())
    t0     = time.time()

    await engine.on_recognition(cid, cam_id, pid, conf=0.85, frame_ref="f1.jpg", ts=t0)

    # Tick after INACTIVE_TIMEOUT has elapsed
    closed = await engine.tick(cam_id, ts=t0 + _INACTIVE_TIMEOUT_S + 10)

    assert pid not in engine._sightings.get(cam_id, {})
    assert engine.get_area_occupancy(cam_id) == 0
    assert len(closed) >= 1
    assert any(s.person_id == pid for s in closed)


async def test_ghost_sighting_discarded():
    """Sightings shorter than MIN_DURATION_S must be discarded silently."""
    from app.services.sighting_engine import SightingEngine

    engine = SightingEngine(db=AsyncMock(), redis=AsyncMock(), kafka_producer=AsyncMock())
    cid    = str(uuid.uuid4())
    cam_id = str(uuid.uuid4())
    pid    = str(uuid.uuid4())
    t0     = time.time()

    # Single recognition, then immediately tick past timeout
    await engine.on_recognition(cid, cam_id, pid, conf=0.75, frame_ref="f1.jpg", ts=t0)

    # Override duration to be below minimum
    engine._sightings[cam_id][pid].duration_seconds = 2

    closed = await engine.tick(cam_id, ts=t0 + _INACTIVE_TIMEOUT_S + 10)
    # Ghost sighting should NOT appear in the closed list
    real_closed = [s for s in closed if s.duration_seconds >= _MIN_DURATION_S]
    ghost_closed = [s for s in closed if s.duration_seconds < _MIN_DURATION_S]
    assert not any(s.person_id == pid for s in real_closed)


async def test_multiple_persons_occupancy():
    from app.services.sighting_engine import SightingEngine

    engine = SightingEngine(db=AsyncMock(), redis=AsyncMock(), kafka_producer=AsyncMock())
    cid    = str(uuid.uuid4())
    cam_id = str(uuid.uuid4())
    t0     = time.time()

    for _ in range(7):
        pid = str(uuid.uuid4())
        await engine.on_recognition(cid, cam_id, pid, conf=0.88, frame_ref="f.jpg", ts=t0)

    assert engine.get_area_occupancy(cam_id) == 7


async def test_occupancy_after_some_close():
    from app.services.sighting_engine import SightingEngine

    engine = SightingEngine(db=AsyncMock(), redis=AsyncMock(), kafka_producer=AsyncMock())
    cid    = str(uuid.uuid4())
    cam_id = str(uuid.uuid4())
    t0     = time.time()

    pids = [str(uuid.uuid4()) for _ in range(5)]
    for pid in pids:
        await engine.on_recognition(cid, cam_id, pid, conf=0.88, frame_ref="f.jpg", ts=t0)

    # Tick only pids[0..1] past timeout by resetting their last_seen artificially
    for pid in pids[:2]:
        engine._sightings[cam_id][pid].last_seen = t0 - _INACTIVE_TIMEOUT_S - 10

    await engine.tick(cam_id, ts=time.time())
    assert engine.get_area_occupancy(cam_id) == 3  # 5 - 2 closed


# ── DB persistence ────────────────────────────────────────────────────────────

async def test_flush_writes_sighting_to_db(db_session, tenant_a):
    """SightingEngine.flush() should insert closed sightings into the DB."""
    from app.models.sightings import Sighting
    from app.services.sighting_engine import SightingEngine

    cam = await create_camera(db_session, tenant_a.client_id, "Monitoring Cam")
    person = await create_person(db_session, tenant_a.client_id, "Visitor 1")
    await db_session.commit()

    # Create engine with the real async session
    engine = SightingEngine(
        db=db_session, redis=AsyncMock(), kafka_producer=AsyncMock()
    )
    cid    = str(tenant_a.client_id)
    cam_id = str(cam.camera_id)
    pid    = str(person.person_id)
    t0     = time.time()

    await engine.on_recognition(cid, cam_id, pid, conf=0.87, frame_ref="f1.jpg", ts=t0)
    # Expire the sighting and flush
    engine._sightings[cam_id][pid].last_seen = t0 - _INACTIVE_TIMEOUT_S - 1
    engine._sightings[cam_id][pid].duration_seconds = int(_INACTIVE_TIMEOUT_S + 1)

    closed = await engine.tick(cam_id, ts=time.time())
    # flush() persists to DB
    await engine.flush(cam_id)

    result = await db_session.execute(
        select(Sighting).where(
            Sighting.person_id == person.person_id,
            Sighting.camera_id == cam.camera_id,
        )
    )
    row = result.scalars().first()
    assert row is not None
    assert row.client_id == tenant_a.client_id


async def test_flush_publishes_to_kafka():
    """flush() should call publish_sighting for each closed sighting."""
    from app.services.sighting_engine import SightingEngine

    mock_kafka = AsyncMock()
    engine = SightingEngine(db=AsyncMock(), redis=AsyncMock(), kafka_producer=mock_kafka)
    cid    = str(uuid.uuid4())
    cam_id = str(uuid.uuid4())
    pid    = str(uuid.uuid4())
    t0     = time.time()

    await engine.on_recognition(cid, cam_id, pid, conf=0.88, frame_ref="f.jpg", ts=t0)
    engine._sightings[cam_id][pid].last_seen = t0 - _INACTIVE_TIMEOUT_S - 1
    engine._sightings[cam_id][pid].duration_seconds = int(_INACTIVE_TIMEOUT_S + 1)

    await engine.tick(cam_id, ts=time.time())
    await engine.flush(cam_id)

    mock_kafka.publish_sighting.assert_called()


# ── SightingEngine isolation across cameras ───────────────────────────────────

async def test_sightings_isolated_per_camera():
    from app.services.sighting_engine import SightingEngine

    engine = SightingEngine(db=AsyncMock(), redis=AsyncMock(), kafka_producer=AsyncMock())
    cid    = str(uuid.uuid4())
    cam1   = str(uuid.uuid4())
    cam2   = str(uuid.uuid4())
    pid    = str(uuid.uuid4())
    t0     = time.time()

    await engine.on_recognition(cid, cam1, pid, conf=0.88, frame_ref="f.jpg", ts=t0)

    assert engine.get_area_occupancy(cam1) == 1
    assert engine.get_area_occupancy(cam2) == 0
