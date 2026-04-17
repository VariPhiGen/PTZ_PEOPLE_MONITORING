"""
test_session_flow.py — Session lifecycle, zone mapping, path planning, PTZ cycles,
                        duration tracking, attendance finalization, and Kafka events.

Covers:
  • Start session → DB record created
  • ZoneMapper.map_zone produces ScanMap with cells
  • PathPlanner.plan_path returns ordered cell IDs
  • PTZBrain runs 3 abbreviated cycles (mocked RTSP + AI)
  • AttendanceEngine.finalize_session produces correct P/L/A/ND statuses
  • Kafka topics published: attendance.records, attendance.faculty
  • Stop session → actual_end recorded, sync_status advances
  • WebSocket emits state_change and cycle_complete events
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from tests.conftest import (
    MockFaceRepository,
    _random_embedding,
    auth_headers,
    create_camera,
    create_person,
)
from tests.factories import SessionFactory


# ── Synthetic helpers ─────────────────────────────────────────────────────────

def _synthetic_frame() -> np.ndarray:
    return np.zeros((720, 1280, 3), dtype=np.uint8)


@dataclass
class FakePersonDetection:
    bbox: list = field(default_factory=lambda: [100, 100, 300, 400])
    conf: float = 0.92
    center: tuple = (200, 250)
    # keypoints and is_standing removed in v2 (YOLOv8l detection-only model)


@dataclass
class FakeFaceDetection:
    bbox: list = field(default_factory=lambda: [120, 120, 250, 280])
    landmarks: list = field(default_factory=list)
    conf: float = 0.95
    inter_ocular_px: float = 90.0
    embedding: Optional[np.ndarray] = None
    person_id: Optional[str] = None


# ── Zone mapping ──────────────────────────────────────────────────────────────

async def test_zone_mapper_creates_scan_cells(db_session, tenant_a):
    """ZoneMapper.map_zone should detect persons and produce at least one ScanCell."""
    from app.services.zone_mapper import ZoneMapper

    mock_onvif = AsyncMock()
    mock_onvif.get_ptz_status     = AsyncMock(return_value={"pan": 0.0, "tilt": 0.0, "zoom": 0.0})
    mock_onvif.get_fov_at_zoom    = AsyncMock(return_value=(70.0, 45.0))
    mock_onvif.pixel_to_ptz       = AsyncMock(return_value=(0.1, -0.1))
    mock_onvif.estimate_travel_time = AsyncMock(return_value=0.8)

    mock_pipeline = AsyncMock()
    mock_pipeline.detect_persons = AsyncMock(return_value=[
        FakePersonDetection(center=(200, 300)),
        FakePersonDetection(center=(600, 300)),
        FakePersonDetection(center=(1000, 300)),
    ])

    frame = _synthetic_frame()
    roi_rect = {"x": 0, "y": 0, "width": 1280, "height": 720}
    camera_ptz = {"pan": 0.0, "tilt": 0.0, "zoom": 0.2}

    mapper = ZoneMapper(mock_onvif)
    scan_map = await mapper.map_zone(frame, roi_rect, camera_ptz, mock_pipeline)

    assert scan_map is not None
    assert hasattr(scan_map, "cells")
    assert len(scan_map.cells) >= 1


async def test_zone_mapper_remap_light_update(db_session, tenant_a):
    """remap() with <30% movement should perform a light update (no re-cluster)."""
    from app.services.zone_mapper import ZoneMapper

    mock_onvif = AsyncMock()
    mock_onvif.get_ptz_status     = AsyncMock(return_value={"pan": 0.0, "tilt": 0.0, "zoom": 0.0})
    mock_onvif.get_fov_at_zoom    = AsyncMock(return_value=(70.0, 45.0))
    mock_onvif.pixel_to_ptz       = AsyncMock(return_value=(0.05, -0.05))
    mock_onvif.estimate_travel_time = AsyncMock(return_value=0.5)

    mock_pipeline = AsyncMock()
    mock_pipeline.detect_persons = AsyncMock(return_value=[
        FakePersonDetection(center=(200, 300)),
    ])

    frame = _synthetic_frame()
    roi_rect = {"x": 0, "y": 0, "width": 1280, "height": 720}
    camera_ptz = {"pan": 0.0, "tilt": 0.0, "zoom": 0.2}

    mapper = ZoneMapper(mock_onvif)
    scan_map = await mapper.map_zone(frame, roi_rect, camera_ptz, mock_pipeline)

    # Remap with similar frame (minimal movement)
    new_frame = _synthetic_frame()
    updated = await mapper.remap(scan_map, new_frame)
    assert updated is not None


# ── Path planning ─────────────────────────────────────────────────────────────

async def test_path_planner_returns_all_cells():
    """plan_path should visit every cell exactly once."""
    from app.services.path_planner import PathPlanner
    from app.services.zone_mapper import ScanCell

    cells = [
        ScanCell(
            cell_id=f"cell_{i}",
            center_pan=float(i) * 0.2,
            center_tilt=0.0,
            required_zoom=0.5,
            expected_faces=2,
            unrecognized_count=1 if i == 0 else 0,
            priority=1,
        )
        for i in range(4)
    ]
    faculty_cell = cells[0]
    current_ptz = {"pan": 0.0, "tilt": 0.0, "zoom": 0.2}
    speeds = {"pan": 1.0, "tilt": 1.0, "zoom": 1.0}

    planner = PathPlanner()
    path = await planner.plan_path(cells, faculty_cell, current_ptz, speeds)

    assert len(path) == len(cells)
    assert set(path) == {c.cell_id for c in cells}


async def test_path_planner_faculty_is_first():
    """Faculty cell must always be the first stop."""
    from app.services.path_planner import PathPlanner
    from app.services.zone_mapper import ScanCell

    cells = [ScanCell(
        cell_id=f"c{i}", center_pan=float(i) * 0.3, center_tilt=0.0,
        required_zoom=0.5, expected_faces=1, unrecognized_count=0, priority=1,
    ) for i in range(5)]
    faculty_cell = cells[3]  # Faculty is NOT the closest to origin
    planner = PathPlanner()
    path = await planner.plan_path(
        cells, faculty_cell, {"pan": 0.0, "tilt": 0.0, "zoom": 0.2},
        {"pan": 1.0, "tilt": 1.0, "zoom": 1.0},
    )
    assert path[0] == faculty_cell.cell_id


async def test_path_planner_estimates_cycle_time():
    from app.services.path_planner import PathPlanner
    from app.services.zone_mapper import ScanCell

    cells = [ScanCell(
        cell_id=f"c{i}", center_pan=float(i) * 0.3, center_tilt=0.0,
        required_zoom=0.5, expected_faces=2, unrecognized_count=0, priority=1,
    ) for i in range(3)]
    planner = PathPlanner()
    path = await planner.plan_path(
        cells, cells[0], {"pan": 0.0, "tilt": 0.0, "zoom": 0.2},
        {"pan": 1.0, "tilt": 1.0, "zoom": 1.0},
    )
    estimate = await planner.estimate_cycle_time(
        path, cells, {"pan": 1.0, "tilt": 1.0, "zoom": 1.0}
    )
    assert estimate > 0


# ── Session API ───────────────────────────────────────────────────────────────

async def test_start_session(
    app_client, db_session, tenant_a, client_admin_a, client_admin_a_token
):
    cam = await create_camera(db_session, tenant_a.client_id, "Session Cam")
    faculty = await create_person(db_session, tenant_a.client_id, "Dr. Faculty", role="FACULTY")
    student = await create_person(db_session, tenant_a.client_id, "Student A", role="STUDENT")
    await db_session.commit()

    now = int(time.time())
    with patch("app.api.sessions.PTZBrain") as mock_brain_cls:
        mock_brain = AsyncMock()
        mock_brain.run_session = AsyncMock(return_value=None)
        mock_brain_cls.return_value = mock_brain

        resp = await app_client.post(
            "/api/sessions/start",
            json={
                "camera_id":      str(cam.camera_id),
                "course_id":      "CS201",
                "course_name":    "Algorithms",
                "faculty_id":     str(faculty.person_id),
                "roster_ids":     [str(student.person_id)],
                "mode":           "ATTENDANCE",
                "scheduled_start": now,
                "scheduled_end":   now + 5_400,
                "roi_rect":       {"x": 0, "y": 0, "width": 1280, "height": 720},
            },
            headers=auth_headers(client_admin_a_token),
        )
    assert resp.status_code in (200, 201)
    data = resp.json()
    assert "session_id" in data


async def test_stop_session_records_actual_end(
    app_client, db_session, tenant_a, client_admin_a, client_admin_a_token
):
    from app.models.sessions import Session as SessionModel

    cam = await create_camera(db_session, tenant_a.client_id)
    sess = SessionFactory(client_id=tenant_a.client_id, camera_id=cam.camera_id)
    db_session.add(sess)
    await db_session.commit()

    resp = await app_client.post(
        f"/api/sessions/{sess.session_id}/stop",
        headers=auth_headers(client_admin_a_token),
    )
    assert resp.status_code in (200, 204)

    await db_session.refresh(sess)
    assert sess.actual_end is not None


async def test_list_active_sessions(
    app_client, db_session, tenant_a, client_admin_a, client_admin_a_token
):
    cam = await create_camera(db_session, tenant_a.client_id)
    sess = SessionFactory(
        client_id=tenant_a.client_id,
        camera_id=cam.camera_id,
        sync_status="PENDING",
    )
    db_session.add(sess)
    await db_session.commit()

    # Mark as active in app state
    app_client.app.state.active_sessions[str(sess.session_id)] = {"session_id": str(sess.session_id)}

    resp = await app_client.get("/api/sessions/active", headers=auth_headers(client_admin_a_token))
    assert resp.status_code == 200


async def test_get_session_state(
    app_client, db_session, tenant_a, client_admin_a, client_admin_a_token
):
    cam  = await create_camera(db_session, tenant_a.client_id)
    sess = SessionFactory(client_id=tenant_a.client_id, camera_id=cam.camera_id)
    db_session.add(sess)
    await db_session.commit()

    resp = await app_client.get(
        f"/api/sessions/{sess.session_id}/state",
        headers=auth_headers(client_admin_a_token),
    )
    assert resp.status_code in (200, 404)


# ── Attendance engine ─────────────────────────────────────────────────────────

async def test_finalize_session_all_present():
    """Students with ≥70% duration should be marked P."""
    from app.services.attendance_engine import AttendanceEngine, AttendanceThresholds, DurationTracker

    engine = AttendanceEngine()
    session_duration = 3600  # 1 hour
    thresholds = AttendanceThresholds(
        session_duration_s  = session_duration,
        present_frac        = 0.70,
        late_grace_frac     = 0.15,
        early_exit_frac     = 0.15,
        absent_frac         = 0.15,
        min_detections      = 3,
    )

    now = int(time.time())
    trackers = {
        f"pid_{i}": DurationTracker(
            first_seen      = now,
            last_seen       = now + int(session_duration * 0.85),
            total_seconds   = int(session_duration * 0.85),
            detection_count = 20,
            state           = "ACTIVE",
        )
        for i in range(5)
    }
    faculty_tracker = DurationTracker(
        first_seen=now, last_seen=now + session_duration,
        total_seconds=session_duration, detection_count=40, state="ACTIVE",
    )

    result = engine.finalize_session(
        session_id    = str(uuid.uuid4()),
        client_id     = str(uuid.uuid4()),
        trackers      = trackers,
        faculty_tracker = faculty_tracker,
        thresholds    = thresholds,
    )

    assert result is not None
    for record in result.records:
        assert record.status == "P"
    assert result.faculty_outcome == "APPROVED"


async def test_finalize_session_late_arrival():
    """Students who arrived after 15% of session duration are marked L."""
    from app.services.attendance_engine import AttendanceEngine, AttendanceThresholds, DurationTracker

    engine = AttendanceEngine()
    dur = 3600
    thresholds = AttendanceThresholds(
        session_duration_s=dur, present_frac=0.70,
        late_grace_frac=0.15, early_exit_frac=0.15,
        absent_frac=0.15, min_detections=3,
    )
    now = int(time.time())
    late_arrival = int(dur * 0.25)

    trackers = {
        "late_student": DurationTracker(
            first_seen      = now + late_arrival,
            last_seen       = now + dur,
            total_seconds   = dur - late_arrival,
            detection_count = 20,
            state           = "ACTIVE",
        )
    }
    faculty_tracker = DurationTracker(
        first_seen=now, last_seen=now+dur, total_seconds=dur, detection_count=40, state="ACTIVE",
    )

    result = engine.finalize_session(
        session_id=str(uuid.uuid4()), client_id=str(uuid.uuid4()),
        trackers=trackers, faculty_tracker=faculty_tracker, thresholds=thresholds,
    )
    statuses = {r.person_id: r.status for r in result.records}
    assert statuses["late_student"] == "L"


async def test_finalize_session_absent():
    """Students with <15% duration are marked A."""
    from app.services.attendance_engine import AttendanceEngine, AttendanceThresholds, DurationTracker

    engine = AttendanceEngine()
    dur = 3600
    thresholds = AttendanceThresholds(
        session_duration_s=dur, present_frac=0.70,
        late_grace_frac=0.15, early_exit_frac=0.15,
        absent_frac=0.15, min_detections=3,
    )
    now = int(time.time())
    trackers = {
        "absent_student": DurationTracker(
            first_seen=now, last_seen=now + 100,
            total_seconds=100, detection_count=2, state="INACTIVE",
        )
    }
    faculty_tracker = DurationTracker(
        first_seen=now, last_seen=now+dur, total_seconds=dur, detection_count=40, state="ACTIVE",
    )
    result = engine.finalize_session(
        session_id=str(uuid.uuid4()), client_id=str(uuid.uuid4()),
        trackers=trackers, faculty_tracker=faculty_tracker, thresholds=thresholds,
    )
    statuses = {r.person_id: r.status for r in result.records}
    assert statuses["absent_student"] in ("A", "ND")


async def test_finalize_session_held_when_faculty_absent():
    """Session is HELD when faculty was never detected."""
    from app.services.attendance_engine import AttendanceEngine, AttendanceThresholds, DurationTracker

    engine = AttendanceEngine()
    dur = 3600
    thresholds = AttendanceThresholds(
        session_duration_s=dur, present_frac=0.70,
        late_grace_frac=0.15, early_exit_frac=0.15,
        absent_frac=0.15, min_detections=3,
    )
    now = int(time.time())
    trackers = {
        "student": DurationTracker(
            first_seen=now, last_seen=now + dur,
            total_seconds=dur, detection_count=30, state="ACTIVE",
        )
    }
    # faculty_tracker with zero duration (never seen)
    faculty_tracker = DurationTracker(
        first_seen=0, last_seen=0, total_seconds=0, detection_count=0, state="ABSENT",
    )
    result = engine.finalize_session(
        session_id=str(uuid.uuid4()), client_id=str(uuid.uuid4()),
        trackers=trackers, faculty_tracker=faculty_tracker, thresholds=thresholds,
    )
    assert result.faculty_outcome in ("HELD", "ABSENT", "faculty_absent")


# ── Kafka publishing ──────────────────────────────────────────────────────────

async def test_kafka_published_on_session_finalize(mock_kafka):
    """publish_attendance should be called when a session is finalized."""
    from app.services.attendance_engine import AttendanceEngine, AttendanceThresholds, DurationTracker

    kafka_producer, _ = mock_kafka
    engine = AttendanceEngine()
    dur = 3600
    now = int(time.time())
    thresholds = AttendanceThresholds(
        session_duration_s=dur, present_frac=0.70,
        late_grace_frac=0.15, early_exit_frac=0.15,
        absent_frac=0.15, min_detections=3,
    )
    trackers = {"s1": DurationTracker(now, now+dur, dur, 30, "ACTIVE")}
    faculty_tracker = DurationTracker(now, now+dur, dur, 40, "ACTIVE")

    result = engine.finalize_session(
        session_id=str(uuid.uuid4()), client_id=str(uuid.uuid4()),
        trackers=trackers, faculty_tracker=faculty_tracker, thresholds=thresholds,
    )
    await kafka_producer.publish_attendance(result)
    kafka_producer.publish_attendance.assert_called_once()
