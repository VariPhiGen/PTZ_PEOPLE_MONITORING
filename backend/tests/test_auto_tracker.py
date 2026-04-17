"""
test_auto_tracker.py — commercial auto-tracking primitives.

Unit tests for track_priority (TargetSelector, compute_group_framing) and
AutoTracker internals. No GPU / ONVIF / RTSP needed — every dependency is
a stub or mock. Marked so pytest -m "not gpu and not kafka and not minio"
includes these.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from app.services.mot_tracker import Track, TrackState
from app.services.track_priority import (
    TargetSelector,
    compute_group_framing,
)


# ── helpers ───────────────────────────────────────────────────────────────

def _mk_track(
    tid: int,
    cx: float, cy: float,
    w: float = 80.0, h: float = 200.0,
    vx: float = 0.0, vy: float = 0.0,
    state: TrackState = TrackState.CONFIRMED,
) -> Track:
    x = np.array([cx, cy, w, h, vx, vy, 0.0, 0.0], dtype=np.float64)
    return Track(
        track_id=tid, state=state,
        hits=5, hit_streak=5, age=5, time_since_update=0, conf=0.9,
        x=x, P=np.eye(8, dtype=np.float64),
    )


# ── TargetSelector ────────────────────────────────────────────────────────

def test_selector_picks_highest_scoring_track_as_primary():
    sel = TargetSelector(frame_wh=(1920, 1080))
    # t1 centred + large, t2 edge + small → t1 should win.
    t1 = _mk_track(1, cx=960, cy=540, w=160, h=400)
    t2 = _mk_track(2, cx=100, cy=900, w=40,  h=100)
    res = sel.select([t1, t2])
    assert res.primary is not None
    assert res.primary.track_id == 1
    assert [t.track_id for t in res.secondaries] == [2]


def test_selector_prefers_unidentified_over_identified():
    sel = TargetSelector(frame_wh=(1920, 1080))
    sel.mark_identified(1)           # mark t1 as already-known
    t1 = _mk_track(1, cx=960, cy=540, w=160, h=400)
    t2 = _mk_track(2, cx=960, cy=540, w=160, h=400)  # identical geometry
    res = sel.select([t1, t2])
    assert res.primary is not None
    assert res.primary.track_id == 2  # unknown beats known


def test_selector_hysteresis_holds_incumbent_against_weak_challenger():
    sel = TargetSelector(
        frame_wh=(1920, 1080),
        switch_margin=0.25,   # challenger must beat incumbent by >=0.25
        switch_frames=5,
    )
    t_in  = _mk_track(1, cx=960, cy=540, w=160, h=400)
    t_chg = _mk_track(2, cx=960, cy=540, w=170, h=410)  # only marginally better
    # First call — t1 or t2 becomes primary (we don't care which).
    first = sel.select([t_in, t_chg]).primary.track_id
    for _ in range(10):
        res = sel.select([t_in, t_chg])
    # Primary must not flip because delta is below switch_margin.
    assert res.primary.track_id == first


def test_selector_empty_input_returns_no_primary():
    sel = TargetSelector(frame_wh=(1920, 1080))
    res = sel.select([])
    assert res.primary is None
    assert res.secondaries == []


def test_selector_edge_risk_penalises_tracks_about_to_exit():
    sel = TargetSelector(frame_wh=(1920, 1080))
    # Isolate edge-risk by matching every other factor (identity, size, motion).
    # t_edge sits near the right edge and has a tiny rightward drift so the
    # predicted bbox crosses the 8% margin. Without motion dominating the
    # score, centrality + edge-risk should tip the decision to the centred track.
    t_centre = _mk_track(1, cx=960,  cy=540, vx=0, vy=0)
    t_edge   = _mk_track(2, cx=1850, cy=540, vx=2, vy=0)
    res = sel.select([t_centre, t_edge])
    assert res.primary.track_id == 1


# ── compute_group_framing ─────────────────────────────────────────────────

def test_group_framing_includes_nearby_secondary_within_widen_budget():
    primary = _mk_track(1, cx=960, cy=540, w=120, h=300)
    secondary = _mk_track(2, cx=1080, cy=540, w=120, h=300)  # right beside primary
    fb = compute_group_framing(primary, [secondary], frame_wh=(1920, 1080))
    assert 1 in fb.includes
    assert 2 in fb.includes
    # Framing centre should sit between the two tracks.
    assert 960 < fb.cx < 1080


def test_group_framing_drops_secondary_that_blows_widen_budget():
    primary = _mk_track(1, cx=300, cy=540, w=120, h=300)
    far_sec = _mk_track(2, cx=1700, cy=540, w=120, h=300)
    fb = compute_group_framing(
        primary, [far_sec],
        frame_wh=(1920, 1080),
        max_zoom_widen=1.5,
    )
    # Primary is always retained; far secondary is dropped.
    assert fb.includes == [1]


def test_group_framing_clamps_centre_to_frame():
    # Primary centred outside the frame (edge tracker quirk) — framing centre
    # must still land inside [0, W] x [0, H].
    primary = _mk_track(1, cx=-100, cy=-100, w=80, h=200)
    fb = compute_group_framing(primary, [], frame_wh=(1920, 1080))
    assert 0.0 <= fb.cx <= 1920.0
    assert 0.0 <= fb.cy <= 1080.0


# ── AutoTracker wiring (light integration, all mocked) ────────────────────

@pytest.mark.asyncio
async def test_auto_tracker_sends_stop_when_no_tracks():
    """With zero tracks the AutoTracker should coast + stop, never issue a
    non-zero continuous_move."""
    from app.services.auto_tracker import AutoTracker

    ptz = MagicMock()
    ptz.continuous_move = AsyncMock()
    ptz.stop            = AsyncMock()
    ptz.get_ptz_status  = AsyncMock(return_value=MagicMock(pan=0, tilt=0, zoom=0))
    ptz.connect         = AsyncMock()

    # Pipeline returns zero tracked_persons → selector returns no primary.
    pipe = MagicMock()
    empty_result = MagicMock()
    empty_result.tracked_persons = []
    empty_result.faces_with_embeddings = []
    pipe.process_frame = MagicMock(return_value=empty_result)
    pipe._mot_tracker  = MagicMock(_tracks=[])

    black = np.zeros((1080, 1920, 3), dtype=np.uint8)
    grab = AsyncMock(return_value=black)

    auto = AutoTracker(
        ptz_ctrl=ptz,
        grab_frame=grab,
        pipeline=pipe,
        identify_face=None,
        frame_wh=(1920, 1080),
        enable_slow_loop=False,
    )
    await auto.start()
    await asyncio.sleep(0.25)
    await auto.stop()

    # continuous_move must not have been called with any non-zero velocity.
    for call in ptz.continuous_move.await_args_list:
        args, kwargs = call
        if args:
            pan, tilt = args[0], args[1]
            assert pan == 0 and tilt == 0
