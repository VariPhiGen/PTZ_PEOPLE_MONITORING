"""
factory_boy-style data factories for ACAS tests.

Usage:
    from tests.factories import PersonFactory, CameraFactory
    person = PersonFactory(client_id=some_uuid)   # returns an unsaved model instance
    # then:  db_session.add(person); await db_session.flush()

All factories accept kwargs that override defaults.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

# ── Base factory ──────────────────────────────────────────────────────────────
_seq: dict[str, int] = {}


def _next(key: str) -> int:
    _seq[key] = _seq.get(key, 0) + 1
    return _seq[key]


def _uid() -> uuid.UUID:
    return uuid.uuid4()


def _now() -> int:
    return int(time.time())


# ── Client factory ────────────────────────────────────────────────────────────
def ClientFactory(**kwargs):
    from app.models.clients import Client
    n = _next("client")
    defaults = dict(
        client_id     = _uid(),
        name          = f"Acme Corp {n}",
        slug          = f"acme-corp-{n}",
        status        = "ACTIVE",
        max_cameras   = 50,
        max_persons   = 10_000,
        settings      = {},
        created_at    = _now(),
        updated_at    = _now(),
    )
    defaults.update(kwargs)
    return Client(**defaults)


# ── User factory ──────────────────────────────────────────────────────────────
def UserFactory(**kwargs):
    from app.models.users import User
    from app.utils.security import hash_password
    n = _next("user")
    defaults = dict(
        user_id       = _uid(),
        email         = f"user{n}@acas.test",
        password_hash = hash_password("Test@pass1234"),
        name          = f"Test User {n}",
        role          = "CLIENT_ADMIN",
        client_id     = None,
        status        = "ACTIVE",
        mfa_enabled   = False,
        mfa_secret    = None,
        created_at    = _now(),
        updated_at    = _now(),
    )
    defaults.update(kwargs)
    return User(**defaults)


# ── Camera factory ────────────────────────────────────────────────────────────
def CameraFactory(**kwargs):
    from app.models.cameras import Camera
    n = _next("camera")
    defaults = dict(
        camera_id                  = _uid(),
        client_id                  = _uid(),  # override in tests
        name                       = f"Camera {n}",
        room_name                  = f"Room {n:03d}",
        building                   = "Main Building",
        floor                      = "1",
        rtsp_url                   = f"rtsp://192.168.1.{n % 254 + 1}:554/stream",
        onvif_host                 = f"192.168.1.{n % 254 + 1}",
        onvif_port                 = 80,
        onvif_username             = "admin_onvif",
        onvif_password_encrypted   = "enc:Test@2024",
        status                     = "ONLINE",
        mode                       = "ATTENDANCE",
        fov_h                      = 70.0,
        fov_v                      = 45.0,
        pan_speed                  = 1.0,
        tilt_speed                 = 1.0,
        zoom_speed                 = 1.0,
        restricted_zone            = False,
        alert_on_unknown           = False,
        created_at                 = _now(),
        updated_at                 = _now(),
    )
    defaults.update(kwargs)
    return Camera(**defaults)


# ── Person factory ────────────────────────────────────────────────────────────
def PersonFactory(**kwargs):
    from app.models.persons import Person
    n = _next("person")
    defaults = dict(
        person_id   = _uid(),
        client_id   = _uid(),  # override in tests
        external_id = f"EXT-{n:05d}",
        name        = f"Student {n}",
        role        = "STUDENT",
        department  = "Engineering",
        email       = f"student{n}@university.edu",
        status      = "ACTIVE",
        consent_at  = _now(),
        created_at  = _now(),
        updated_at  = _now(),
    )
    defaults.update(kwargs)
    return Person(**defaults)


# ── FaceEmbedding factory ─────────────────────────────────────────────────────
def FaceEmbeddingFactory(**kwargs):
    from app.models.face_embeddings import FaceEmbedding
    import numpy as np
    n = _next("embedding")
    rng = np.random.default_rng(seed=n)
    emb = rng.random(512).astype(np.float32)
    emb /= np.linalg.norm(emb)
    defaults = dict(
        embedding_id   = _uid(),
        client_id      = _uid(),  # override
        person_id      = _uid(),  # override
        embedding      = emb.tolist(),
        version        = 1,
        source         = "ENROLLMENT",
        confidence_avg = 0.92,
        image_refs     = [],
        is_active      = True,
        quality_score  = 88.5,
        created_at     = _now(),
    )
    defaults.update(kwargs)
    return FaceEmbedding(**defaults)


# ── Session factory ───────────────────────────────────────────────────────────
def SessionFactory(**kwargs):
    from app.models.sessions import Session as SessionModel
    now = _now()
    defaults = dict(
        session_id      = _uid(),
        client_id       = _uid(),  # override
        camera_id       = _uid(),  # override
        course_id       = "CS101",
        course_name     = "Introduction to Computer Science",
        scheduled_start = now,
        scheduled_end   = now + 5_400,
        actual_start    = now,
        sync_status     = "PENDING",
        cycle_count     = 0,
        recognition_rate = 0.0,
        created_at      = now,
    )
    defaults.update(kwargs)
    return SessionModel(**defaults)


# ── AttendanceRecord factory ──────────────────────────────────────────────────
def AttendanceRecordFactory(**kwargs):
    from app.models.attendance_records import AttendanceRecord
    import datetime
    now = _now()
    defaults = dict(
        record_id       = _uid(),
        client_id       = _uid(),  # override
        session_id      = _uid(),  # override
        person_id       = _uid(),  # override
        total_duration  = datetime.timedelta(minutes=45),
        detection_count = 30,
        first_seen      = now,
        last_seen       = now + 2_700,
        status          = "P",
        confidence_avg  = 0.88,
        liveness_avg    = 0.91,
        evidence_refs   = [],
        created_at      = now,
    )
    defaults.update(kwargs)
    return AttendanceRecord(**defaults)


# ── Sighting factory ──────────────────────────────────────────────────────────
def SightingFactory(**kwargs):
    from app.models.sightings import Sighting
    now = _now()
    defaults = dict(
        sighting_id      = _uid(),
        client_id        = _uid(),  # override
        person_id        = _uid(),  # override
        camera_id        = _uid(),  # override
        zone             = "main_hall",
        first_seen       = now,
        last_seen        = now + 180,
        duration_seconds = 180,
        confidence_avg   = 0.87,
        frame_refs       = [],
        created_at       = now,
    )
    defaults.update(kwargs)
    return Sighting(**defaults)


# ── ClientNodeAssignment factory ──────────────────────────────────────────────
def ClientNodeAssignmentFactory(**kwargs):
    from app.models.client_node_assignments import ClientNodeAssignment
    defaults = dict(
        assignment_id       = _uid(),
        client_id           = _uid(),  # override
        node_id             = "test-node-aaaaaaaa",
        max_cameras_on_node = 10,
        assigned_at         = _now(),
    )
    defaults.update(kwargs)
    return ClientNodeAssignment(**defaults)


# ── Batch helpers ─────────────────────────────────────────────────────────────
def many(factory_fn, count: int, **shared_kwargs):
    """Create `count` model instances sharing the same kwargs."""
    return [factory_fn(**shared_kwargs) for _ in range(count)]
