"""
Target selection policy for commercial PTZ auto-tracking.

Inputs are MOT `Track`s (from `mot_tracker.BoTSORTTracker`). The policy
scores every confirmed track on a composite of: identity status, face
visibility, motion energy, subject size, centrality and edge risk, then
picks a `primary` (the one the camera actively follows) plus up to
`max_secondaries` others to keep inside the framing when feasible.

Hysteresis is enforced so the camera doesn't oscillate between two
similarly-scored targets: a challenger must beat the incumbent's score by
`switch_margin` for `switch_frames` consecutive calls before the primary
changes.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from app.services.mot_tracker import Track


# ── Tunables ──────────────────────────────────────────────────────────────

# Score weights — kept as module-level floats so they are easy to audit and
# tune from `CLAUDE.md` / runbook without hunting dataclass defaults.
_W_UNKNOWN       = 0.35   # boost for tracks without an identity yet
_W_FACE_VISIBLE  = 0.20   # boost when a face has been seen on this track
_W_MOTION        = 0.15   # boost for moving subjects (Kalman velocity norm)
_W_SIZE          = 0.15   # boost for larger subjects (closer to camera)
_W_CENTRALITY    = 0.10   # boost when subject is near frame centre
_W_EDGE_RISK     = 0.05   # negative weight — subjects near the edge are
                          # about to leave the FOV, deprioritise them
_W_AGE_DECAY     = 0.10   # gentle decay once a track has been primary for
                          # a while, so the camera eventually rotates

_MOTION_SAT_PX   = 40.0   # above this per-frame speed, motion score saturates
_SIZE_SAT_FRAC   = 0.35   # body height fraction at which size score saturates


@dataclass
class TrackScore:
    track_id:   int
    score:      float
    breakdown:  dict[str, float] = field(default_factory=dict)


@dataclass
class SelectionResult:
    primary:     Track | None
    secondaries: list[Track]
    scores:      list[TrackScore]


@dataclass
class _PrimaryHistory:
    track_id:         int
    held_since:       float              # monotonic ts when this track first became primary
    challenger_id:    int | None = None  # track that is currently contesting
    challenger_since: float = 0.0


class TargetSelector:
    """
    Stateful selector that applies hysteresis to the primary choice.

    Thread-safety: not thread-safe. One instance per camera / AutoTracker.
    """

    def __init__(
        self,
        frame_wh: tuple[int, int],
        *,
        max_secondaries: int = 3,
        switch_margin: float = 0.08,
        switch_frames: int = 6,
        identified_tracks: set[int] | None = None,
        face_seen_tracks: set[int] | None = None,
    ) -> None:
        """
        Args:
          frame_wh:          (width, height) of the input frame in pixels.
          max_secondaries:   how many non-primary targets to return for framing.
          switch_margin:     challenger must outscore incumbent by this delta.
          switch_frames:     challenger must hold that lead for N frames.
          identified_tracks: track IDs already bound to an identity (optional).
          face_seen_tracks:  track IDs a face has been observed on (optional).
        """
        self._frame_w, self._frame_h = frame_wh
        self._max_secondaries = max(0, int(max_secondaries))
        self._switch_margin   = float(switch_margin)
        self._switch_frames   = max(1, int(switch_frames))

        self._identified = set(identified_tracks or ())
        self._face_seen  = set(face_seen_tracks or ())
        self._history: _PrimaryHistory | None = None

    # ── Identity / face-observation updates ───────────────────────────────

    def set_frame_size(self, width: int, height: int) -> None:
        self._frame_w = int(width)
        self._frame_h = int(height)

    def mark_identified(self, track_id: int) -> None:
        self._identified.add(int(track_id))

    def mark_face_seen(self, track_id: int) -> None:
        self._face_seen.add(int(track_id))

    def forget(self, track_id: int) -> None:
        """Drop a track from internal state when the MOT tracker deletes it."""
        self._identified.discard(int(track_id))
        self._face_seen.discard(int(track_id))
        if self._history and self._history.track_id == track_id:
            self._history = None

    # ── Scoring ───────────────────────────────────────────────────────────

    def _score(self, track: Track, now: float) -> TrackScore:
        cx, cy = track.center
        w = max(1.0, float(track.x[2]))
        h = max(1.0, float(track.x[3]))
        vx, vy = float(track.x[4]), float(track.x[5])

        # 1. Unknown-first bias — commercial systems prioritise the unseen.
        unknown = 0.0 if track.track_id in self._identified else 1.0

        # 2. Face visible on this track at some point this session.
        face_vis = 1.0 if track.track_id in self._face_seen else 0.0

        # 3. Motion energy — normalised pixel-per-frame speed, saturates so a
        #    single running person doesn't monopolise the cam forever.
        speed = float(np.hypot(vx, vy))
        motion = min(1.0, speed / _MOTION_SAT_PX)

        # 4. Size — larger body height fraction = closer to camera = more
        #    interesting to commercial auto-track.
        size = min(1.0, (h / max(1.0, self._frame_h)) / _SIZE_SAT_FRAC)

        # 5. Centrality — ±1 at centre, 0 at frame edge. Helps tie-break.
        dx = (cx - self._frame_w / 2.0) / max(1.0, self._frame_w / 2.0)
        dy = (cy - self._frame_h / 2.0) / max(1.0, self._frame_h / 2.0)
        centrality = max(0.0, 1.0 - float(np.hypot(dx, dy)))

        # 6. Edge risk — predicted next-frame bbox would cross the frame edge.
        #    Penalises targets about to leave the FOV.
        pred_cx, pred_cy = cx + vx, cy + vy
        edge_risk = 0.0
        margin_x = 0.08 * self._frame_w
        margin_y = 0.08 * self._frame_h
        if (pred_cx - w / 2.0) < margin_x or (pred_cx + w / 2.0) > (self._frame_w - margin_x):
            edge_risk += 0.5
        if (pred_cy - h / 2.0) < margin_y or (pred_cy + h / 2.0) > (self._frame_h - margin_y):
            edge_risk += 0.5

        # 7. Incumbency decay — once the same track has been primary for a
        #    while, dampen its score so another waiting target gets a turn.
        age_pen = 0.0
        if self._history and self._history.track_id == track.track_id:
            held_s = max(0.0, now - self._history.held_since)
            # Saturate after 12 s — matches Axis auto-track default dwell.
            age_pen = min(1.0, held_s / 12.0)

        score = (
            _W_UNKNOWN      * unknown
            + _W_FACE_VISIBLE * face_vis
            + _W_MOTION       * motion
            + _W_SIZE         * size
            + _W_CENTRALITY   * centrality
            - _W_EDGE_RISK    * edge_risk
            - _W_AGE_DECAY    * age_pen
        )

        return TrackScore(
            track_id=track.track_id,
            score=float(score),
            breakdown={
                "unknown":    unknown,
                "face":       face_vis,
                "motion":     motion,
                "size":       size,
                "centrality": centrality,
                "edge_risk":  edge_risk,
                "age_pen":    age_pen,
            },
        )

    # ── Selection with hysteresis ─────────────────────────────────────────

    def select(self, tracks: list[Track]) -> SelectionResult:
        now = time.monotonic()
        if not tracks:
            self._history = None
            return SelectionResult(primary=None, secondaries=[], scores=[])

        scored = [(self._score(t, now), t) for t in tracks]
        scored.sort(key=lambda it: it[0].score, reverse=True)
        scores = [s for s, _ in scored]
        ranked_tracks = [t for _, t in scored]

        top_score, top_track = scored[0]

        incumbent: Track | None = None
        if self._history is not None:
            incumbent = next(
                (t for t in tracks if t.track_id == self._history.track_id),
                None,
            )

        # Case A: no incumbent — top is primary, start history.
        if incumbent is None:
            self._history = _PrimaryHistory(
                track_id=top_track.track_id,
                held_since=now,
            )
            primary = top_track
        # Case B: top is the incumbent — reset challenger state.
        elif top_track.track_id == incumbent.track_id:
            self._history.challenger_id = None
            self._history.challenger_since = 0.0
            primary = incumbent
        # Case C: a challenger outranks incumbent — apply hysteresis.
        else:
            incumbent_score = next(
                (s.score for s in scores if s.track_id == incumbent.track_id),
                -1e9,
            )
            delta = top_score.score - incumbent_score
            if delta < self._switch_margin:
                # not enough lead — keep incumbent.
                self._history.challenger_id = None
                self._history.challenger_since = 0.0
                primary = incumbent
            else:
                if self._history.challenger_id != top_track.track_id:
                    self._history.challenger_id = top_track.track_id
                    self._history.challenger_since = now
                held_frames_equiv = (now - self._history.challenger_since) / 0.033  # ~30 fps
                if held_frames_equiv >= self._switch_frames:
                    self._history = _PrimaryHistory(
                        track_id=top_track.track_id,
                        held_since=now,
                    )
                    primary = top_track
                else:
                    primary = incumbent

        # Build secondaries — everyone else in score order, primary excluded.
        secondaries = [t for t in ranked_tracks if t.track_id != primary.track_id][
            : self._max_secondaries
        ]

        return SelectionResult(
            primary=primary,
            secondaries=secondaries,
            scores=scores,
        )


# ── Multi-target framing ──────────────────────────────────────────────────

@dataclass
class FramingBox:
    """Union bbox of primary + secondaries with a safety margin."""
    cx: float                 # target centre x in pixels
    cy: float                 # target centre y in pixels
    width: float              # desired framing width in pixels
    height: float             # desired framing height in pixels
    includes: list[int]       # track_ids actually inside the box


def compute_group_framing(
    primary: Track,
    secondaries: list[Track],
    frame_wh: tuple[int, int],
    *,
    margin: float = 0.15,
    max_zoom_widen: float = 2.5,
) -> FramingBox:
    """
    Compute a framing rectangle that contains the primary plus any
    secondaries whose inclusion doesn't require widening beyond
    ``max_zoom_widen`` × primary bbox.

    The returned box is in pixel coordinates of the current frame and is
    meant to be fed to ``ONVIFController.pixel_to_ptz()`` (centre) plus a
    zoom calculation based on its width/height relative to frame size.

    Secondaries outside the widen budget are dropped — primary is never
    dropped. This mirrors how Axis handles group auto-track: the camera
    keeps the main subject framed and opportunistically widens if others
    fit cheaply.
    """
    fw, fh = frame_wh
    p_bbox = primary.bbox.astype(np.float64)
    p_w = max(1.0, float(p_bbox[2] - p_bbox[0]))
    p_h = max(1.0, float(p_bbox[3] - p_bbox[1]))
    p_area = p_w * p_h

    x1, y1, x2, y2 = float(p_bbox[0]), float(p_bbox[1]), float(p_bbox[2]), float(p_bbox[3])
    includes: list[int] = [primary.track_id]

    for sec in secondaries:
        s = sec.bbox.astype(np.float64)
        nx1 = min(x1, float(s[0]))
        ny1 = min(y1, float(s[1]))
        nx2 = max(x2, float(s[2]))
        ny2 = max(y2, float(s[3]))
        new_area = max(1.0, (nx2 - nx1) * (ny2 - ny1))
        if new_area <= p_area * max_zoom_widen:
            x1, y1, x2, y2 = nx1, ny1, nx2, ny2
            includes.append(sec.track_id)

    # Apply symmetric margin so subjects aren't kissing the frame edge.
    w = (x2 - x1) * (1.0 + 2.0 * margin)
    h = (y2 - y1) * (1.0 + 2.0 * margin)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    # Clamp to frame so we never request a framing centre outside the image.
    cx = max(0.0, min(float(fw), cx))
    cy = max(0.0, min(float(fh), cy))

    return FramingBox(
        cx=cx, cy=cy,
        width=float(w), height=float(h),
        includes=includes,
    )
