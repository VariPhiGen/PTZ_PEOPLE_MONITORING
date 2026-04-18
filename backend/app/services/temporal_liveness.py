"""
Temporal / multi-frame liveness gate.

MiniFASNet gives a reasonable single-frame spoof score, but commercial systems
(Axis, Bosch, BioID) fuse it with a short temporal window because two failure
modes slip through single-frame anti-spoofing:

  1. High-quality printed photos held at the right distance pass a static CNN
     about 5–8 % of the time.  They are static by construction — no jitter.
  2. Phone-screen replays pass intermittently as the screen refreshes; the
     refresh ripple is regular (60 Hz) with a characteristic variance.

This module keeps a tiny ring buffer per MOT track and extracts two cheap cues:

  smoothed    — EMA of MiniFASNet scores (denoises frame-level noise).
  motion_score — std of inter-frame pixel differences in the aligned chip,
                 rescaled so 0 = static (likely photo) and 1 = natural face.

A live face exhibits micro-motion (blinks, micro-expressions, breathing-induced
sway) that produces a small-but-non-zero motion score; a photo or screen replay
either stays flat (photo) or has suspiciously uniform noise (screen).  We fuse
the smoothed CNN score with the motion score into a single composite
probability used downstream as the authoritative liveness decision.

Compute cost per face: ~0.2 ms on CPU (two grayscale conversions + one std).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


_WINDOW          = 10      # frames of history to retain per track
_EMA_ALPHA       = 0.3     # EMA weight on the NEW score
_MOTION_LOW_STD  = 0.5     # sub-pixel diff std below this → static (photo)
_MOTION_HIGH_STD = 6.0     # above this → natural face motion
_MOTION_SAT_STD  = 40.0    # beyond saturation → motion blur, cap contribution
_W_FRAME         = 0.6     # composite weight on smoothed CNN score
_W_MOTION        = 0.4     # composite weight on motion score


@dataclass
class TemporalLivenessScore:
    """Per-call result of the temporal tracker."""
    single_frame: float      # raw MiniFASNet score this frame
    smoothed:     float      # EMA-smoothed score across the window
    motion:       float      # [0, 1]; 0 = static, 1 = natural micro-motion
    composite:    float      # fused probability of liveness
    n_frames:     int        # number of frames seen for this track


@dataclass
class _TrackState:
    scores:    list[float]           = field(default_factory=list)
    motions:   list[float]           = field(default_factory=list)
    smoothed:  Optional[float]       = None
    prev_gray: Optional[np.ndarray]  = None
    last_ts:   float                 = 0.0


class TemporalLivenessTracker:
    """
    Stateful per-track fusion of MiniFASNet scores and inter-frame motion.

    Thread-safety: not thread-safe.  One instance per AIPipeline.
    """

    def __init__(
        self,
        *,
        window: int = _WINDOW,
        ema_alpha: float = _EMA_ALPHA,
    ) -> None:
        self._window    = int(window)
        self._ema_alpha = float(ema_alpha)
        self._states: dict[int, _TrackState] = {}

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def forget(self, track_id: int) -> None:
        self._states.pop(int(track_id), None)

    def gc(self, max_age_s: float = 10.0) -> None:
        """Drop state for tracks that have not been updated in `max_age_s`."""
        now = time.monotonic()
        dead = [tid for tid, s in self._states.items() if now - s.last_ts > max_age_s]
        for tid in dead:
            self._states.pop(tid, None)

    # ── Update ───────────────────────────────────────────────────────────────

    def update(
        self,
        track_id: int,
        single_frame_score: float,
        face_chip: np.ndarray,
    ) -> TemporalLivenessScore:
        """
        Add a new observation for ``track_id`` and return the fused score.

        Args:
            track_id:           MOT track id this face belongs to.
            single_frame_score: MiniFASNet probability of liveness, [0, 1].
            face_chip:          Aligned 112×112 BGR ArcFace chip for this frame.
        """
        tid = int(track_id)
        state = self._states.get(tid)
        if state is None:
            state = _TrackState()
            self._states[tid] = state

        state.last_ts = time.monotonic()

        gray = (
            cv2.cvtColor(face_chip, cv2.COLOR_BGR2GRAY)
            if face_chip.ndim == 3 else face_chip
        )

        # ── Motion cue: std of inter-frame pixel difference ──────────────────
        motion_score = 0.5   # neutral prior while we have no previous chip
        if state.prev_gray is not None and state.prev_gray.shape == gray.shape:
            diff = cv2.absdiff(gray, state.prev_gray).astype(np.float32)
            d_std = float(diff.std())
            if d_std <= _MOTION_LOW_STD:
                motion_score = 0.0                       # static — probably a photo
            elif d_std >= _MOTION_SAT_STD:
                motion_score = 0.3                       # motion blur / camera shake
            elif d_std >= _MOTION_HIGH_STD:
                motion_score = 1.0                       # natural face motion
            else:
                motion_score = (d_std - _MOTION_LOW_STD) \
                             / (_MOTION_HIGH_STD - _MOTION_LOW_STD)
        state.prev_gray = gray

        # ── Scalar history ring buffers ──────────────────────────────────────
        state.scores.append(float(single_frame_score))
        state.motions.append(float(motion_score))
        if len(state.scores)  > self._window: state.scores.pop(0)
        if len(state.motions) > self._window: state.motions.pop(0)

        # ── Smoothed CNN score (EMA, seeded on first frame) ─────────────────
        if state.smoothed is None:
            state.smoothed = float(single_frame_score)
        else:
            state.smoothed = (
                self._ema_alpha * float(single_frame_score)
                + (1.0 - self._ema_alpha) * state.smoothed
            )

        # Average motion over the window gives a stabler cue than last-frame alone
        avg_motion = float(np.mean(state.motions))

        composite = _W_FRAME * state.smoothed + _W_MOTION * avg_motion
        composite = float(np.clip(composite, 0.0, 1.0))

        return TemporalLivenessScore(
            single_frame=float(single_frame_score),
            smoothed=float(state.smoothed),
            motion=float(avg_motion),
            composite=composite,
            n_frames=len(state.scores),
        )
