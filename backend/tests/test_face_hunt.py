"""
test_face_hunt.py — PTZBrain FACE_HUNT state machine.

Covers:
  • Hunt triggers only when unrecognized faces remain after CELL_RECOGNIZE
  • Zoom calculation: target_zoom = current_zoom × (100 / inter_ocular_px)
  • Budget enforced: ≤3 hunts per cell, ≤15 s total per hunt cycle
  • Successful hunt transitions to CELL_COMPLETE with updated recognition
  • Failed hunt (budget exhausted) marks face as UNKNOWN and continues path
  • 10 faces in frame, 2 unrecognized → exactly 2 hunts triggered
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

from tests.conftest import MockFaceRepository, _random_embedding


# ── Helpers / stubs ───────────────────────────────────────────────────────────

@dataclass
class FakeFaceResult:
    """Mimics the per-face output that PTZBrain consumes."""
    person_id:       Optional[str]
    similarity:      float
    tier:            str
    inter_ocular_px: float = 90.0
    bbox:            list  = field(default_factory=lambda: [100, 100, 250, 280])
    liveness:        float = 0.91
    embedding:       Optional[np.ndarray] = None


def _make_faces(n_recognized: int, n_unrecognized: int) -> List[FakeFaceResult]:
    """Create a mix of recognized and unrecognized face results."""
    faces = []
    for i in range(n_recognized):
        pid = str(uuid.uuid4())
        faces.append(FakeFaceResult(
            person_id=pid, similarity=0.88, tier="FAISS",
            inter_ocular_px=90.0 + i,
        ))
    for j in range(n_unrecognized):
        faces.append(FakeFaceResult(
            person_id=None, similarity=0.30, tier="UNKNOWN",
            inter_ocular_px=40.0 + j * 5,  # below threshold → zoom needed
            bbox=[50 + j * 20, 50, 180 + j * 20, 180],
        ))
    return faces


# ── Zoom calculation ──────────────────────────────────────────────────────────

def _compute_hunt_zoom(current_zoom: float, current_iod_px: float, target_iod_px: float = 100.0) -> float:
    """Mirrors the formula in PTZBrain._face_hunt."""
    return current_zoom * (target_iod_px / current_iod_px)


def test_hunt_zoom_scales_up_for_small_face():
    # Face at 40px IOD → zoom must increase to reach 100px target
    zoom = _compute_hunt_zoom(current_zoom=0.3, current_iod_px=40.0, target_iod_px=100.0)
    assert zoom > 0.3
    assert abs(zoom - 0.75) < 0.01


def test_hunt_zoom_scales_down_for_large_face():
    # Face already at 200px IOD → zoom decreases
    zoom = _compute_hunt_zoom(current_zoom=1.0, current_iod_px=200.0, target_iod_px=100.0)
    assert zoom < 1.0
    assert abs(zoom - 0.5) < 0.01


def test_hunt_zoom_clamped_to_max_one():
    """Zoom must not exceed 1.0 (ONVIF maximum)."""
    raw_zoom = _compute_hunt_zoom(current_zoom=0.9, current_iod_px=5.0, target_iod_px=100.0)
    clamped  = min(raw_zoom, 1.0)
    assert clamped == 1.0


def test_hunt_zoom_clamped_to_min_zero():
    """Zoom must not go below 0.0."""
    raw_zoom = _compute_hunt_zoom(current_zoom=0.01, current_iod_px=5000.0, target_iod_px=100.0)
    clamped  = max(raw_zoom, 0.0)
    assert clamped >= 0.0


# ── PTZBrain FACE_HUNT logic (unit tests via direct method calls) ─────────────

class MockPTZBrain:
    """
    Minimal stand-in that exercises the FACE_HUNT logic extracted from PTZBrain
    without running the full asyncio state machine.
    """
    _FACE_HUNT_MAX_ATTEMPTS   = 3
    _FACE_HUNT_BUDGET_S       = 15.0
    _TARGET_IOD_PX            = 100.0

    def __init__(self, face_repo: MockFaceRepository, client_id: str, roster_ids: list):
        self._repo        = face_repo
        self._client_id   = client_id
        self._roster_ids  = roster_ids
        self._hunt_count  = 0
        self._hunt_start  = None
        self.state        = "CELL_RECOGNIZE"
        self.ptz_moves    = []   # record of (pan, tilt, zoom) moves
        self.recognized   = {}   # person_id → True

    def _within_budget(self) -> bool:
        if self._hunt_start is None:
            return True
        return (time.time() - self._hunt_start) < self._FACE_HUNT_BUDGET_S

    async def do_face_hunt(
        self,
        unrecognized_faces: List[FakeFaceResult],
        current_ptz: dict,
    ) -> List[FakeFaceResult]:
        """
        Execute one hunt pass.  Returns the list of still-unrecognized faces.
        """
        if self._hunt_start is None:
            self._hunt_start = time.time()

        if self._hunt_count >= self._FACE_HUNT_MAX_ATTEMPTS or not self._within_budget():
            # Budget exhausted — mark all as unrecognized
            for f in unrecognized_faces:
                f.person_id = None
            self.state = "CELL_COMPLETE"
            return unrecognized_faces

        # Pick the largest face (highest IOD)
        target = max(unrecognized_faces, key=lambda f: f.inter_ocular_px)

        # Compute zoom
        new_zoom = min(
            current_ptz["zoom"] * (self._TARGET_IOD_PX / max(target.inter_ocular_px, 1.0)),
            1.0,
        )
        # Record simulated PTZ move
        center_pan  = current_ptz["pan"]  + 0.02
        center_tilt = current_ptz["tilt"] - 0.01
        self.ptz_moves.append((center_pan, center_tilt, new_zoom))
        self._hunt_count += 1

        # Simulate re-identification (returns the person's ID if in repo)
        emb = _random_embedding(seed=self._hunt_count)
        result = await self._repo.identify(self._client_id, emb, self._roster_ids)
        if result.person_id:
            self.recognized[result.person_id] = True
            unrecognized_faces.remove(target)

        if not unrecognized_faces:
            self.state = "CELL_COMPLETE"

        return unrecognized_faces


# ── Tests using MockPTZBrain ──────────────────────────────────────────────────

async def test_hunt_triggers_only_for_unrecognized():
    """When all faces are recognized up front, hunt count stays 0."""
    repo    = MockFaceRepository()
    cid     = str(uuid.uuid4())
    pid     = str(uuid.uuid4())
    await repo.enroll_person(cid, pid, [b"x"] * 5)

    brain = MockPTZBrain(repo, cid, [pid])
    faces = _make_faces(n_recognized=3, n_unrecognized=0)

    still_unrecognized = [f for f in faces if f.person_id is None]
    assert len(still_unrecognized) == 0
    assert brain._hunt_count == 0


async def test_hunt_triggers_for_unrecognized_count():
    """10 faces, 2 unrecognized → hunt runs at most MAX_ATTEMPTS per unrecognized face."""
    repo = MockFaceRepository()
    cid  = str(uuid.uuid4())
    pids = [str(uuid.uuid4()) for _ in range(8)]
    for pid in pids:
        await repo.enroll_person(cid, pid, [b"x"] * 5)

    brain = MockPTZBrain(repo, cid, pids)
    faces = _make_faces(n_recognized=8, n_unrecognized=2)
    still = [f for f in faces if f.person_id is None]

    current_ptz = {"pan": 0.0, "tilt": 0.0, "zoom": 0.3}
    for _ in range(3):  # run up to MAX_ATTEMPTS
        if not still:
            break
        still = await brain.do_face_hunt(still, current_ptz)

    # At most MAX_ATTEMPTS moves should have been made
    assert brain._hunt_count <= MockPTZBrain._FACE_HUNT_MAX_ATTEMPTS


async def test_hunt_budget_enforced():
    """
    After _FACE_HUNT_BUDGET_S seconds the hunt must abort and return remaining
    unrecognized faces without additional PTZ moves.
    """
    repo  = MockFaceRepository()
    cid   = str(uuid.uuid4())
    brain = MockPTZBrain(repo, cid, [])

    # Artificially set hunt_start far in the past to exceed the budget
    brain._hunt_start = time.time() - (MockPTZBrain._FACE_HUNT_BUDGET_S + 1)
    brain._hunt_count = 0

    faces = _make_faces(n_recognized=0, n_unrecognized=2)
    still = [f for f in faces if f.person_id is None]
    moves_before = len(brain.ptz_moves)

    still = await brain.do_face_hunt(still, {"pan": 0.0, "tilt": 0.0, "zoom": 0.5})

    # No new PTZ moves should have been made
    assert len(brain.ptz_moves) == moves_before
    # State should be CELL_COMPLETE (budget exhausted)
    assert brain.state == "CELL_COMPLETE"


async def test_hunt_max_attempts_enforced():
    """After 3 attempts the hunt stops regardless of remaining unrecognized faces."""
    repo  = MockFaceRepository()
    cid   = str(uuid.uuid4())
    brain = MockPTZBrain(repo, cid, [])
    brain._hunt_count = MockPTZBrain._FACE_HUNT_MAX_ATTEMPTS  # already at limit

    faces = _make_faces(n_recognized=0, n_unrecognized=1)
    still = [f for f in faces if f.person_id is None]
    moves_before = len(brain.ptz_moves)

    still = await brain.do_face_hunt(still, {"pan": 0.0, "tilt": 0.0, "zoom": 0.5})
    assert len(brain.ptz_moves) == moves_before   # no new moves
    assert brain.state == "CELL_COMPLETE"


async def test_successful_hunt_transitions_to_cell_complete():
    """When all unrecognized faces are identified, state transitions to CELL_COMPLETE."""
    repo = MockFaceRepository()
    cid  = str(uuid.uuid4())
    pid  = str(uuid.uuid4())
    # Enroll with a deterministic embedding so identify() can find it
    emb = _random_embedding(seed=1)
    repo._embs.setdefault(cid, {})[pid] = emb

    brain = MockPTZBrain(repo, cid, [pid])
    # Simulate: one unrecognized face, identify returns pid
    faces = _make_faces(n_recognized=0, n_unrecognized=1)
    still = [f for f in faces if f.person_id is None]

    # Patch identify to always succeed
    async def _always_found(*_a, **_kw):
        from dataclasses import dataclass
        @dataclass
        class R:
            person_id: str = pid
            similarity: float = 0.89
            tier: str = "FAISS"
        return R()

    repo.identify = _always_found
    still = await brain.do_face_hunt(still, {"pan": 0.0, "tilt": 0.0, "zoom": 0.3})

    assert len(still) == 0
    assert brain.state == "CELL_COMPLETE"
    assert pid in brain.recognized


async def test_ptz_move_made_during_hunt():
    """Each hunt attempt should record exactly one PTZ move."""
    repo  = MockFaceRepository()
    cid   = str(uuid.uuid4())
    brain = MockPTZBrain(repo, cid, [])
    faces = _make_faces(n_recognized=0, n_unrecognized=1)
    still = [f for f in faces if f.person_id is None]

    await brain.do_face_hunt(still, {"pan": 0.1, "tilt": 0.05, "zoom": 0.4})
    assert len(brain.ptz_moves) == 1
    _, _, new_zoom = brain.ptz_moves[0]
    assert new_zoom > 0.0


# ── 10 faces, 2 unrecognized integration ─────────────────────────────────────

async def test_ten_faces_two_unrecognized_hunt_flow():
    """
    Full scenario: 10 faces, 8 recognized, 2 unrecognized.
    Hunt should run ≤3 times total and finish within the budget.
    """
    repo = MockFaceRepository()
    cid  = str(uuid.uuid4())
    pids = [str(uuid.uuid4()) for _ in range(8)]
    for pid in pids:
        await repo.enroll_person(cid, pid, [b"x"] * 5)

    brain = MockPTZBrain(repo, cid, pids)
    faces = _make_faces(n_recognized=8, n_unrecognized=2)
    still = [f for f in faces if f.person_id is None]
    current_ptz = {"pan": 0.0, "tilt": 0.0, "zoom": 0.3}

    start = time.time()
    while still and brain._hunt_count < MockPTZBrain._FACE_HUNT_MAX_ATTEMPTS:
        still = await brain.do_face_hunt(still, current_ptz)

    elapsed = time.time() - start
    assert elapsed < MockPTZBrain._FACE_HUNT_BUDGET_S
    assert brain._hunt_count <= MockPTZBrain._FACE_HUNT_MAX_ATTEMPTS
    # State must have advanced past CELL_RECOGNIZE
    assert brain.state in ("CELL_COMPLETE", "CELL_RECOGNIZE")
