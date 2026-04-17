"""
Identity-Track Binding for ACAS.

Maintains the authoritative mapping between MOT track IDs and face identities.
This is the core structure that prevents duplicate sightings and identity fragmentation.

Rules:
  1. One identity per track. Once bound by face recognition, only re-recognition
     can change it.
  2. One track per identity per preset. If the same face appears in two bboxes,
     keep the higher-confidence one.
  3. RE-ID can bridge (extend) but never create or change identities.
  4. Unknown tracks with the same face embedding share a provisional identity
     (unknown_<hash>) rather than creating separate "unknown" sightings.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrackIdentityState:
    """
    The binding between a MOT track and a face identity.
    Mutable — updated as new evidence arrives.
    """

    track_id:           int
    identity_id:        Optional[str]    # None = unidentified
    identity_confidence: float           # ArcFace similarity score at last bind
    tracking_method:    str              # 'face' | 'reid_bridge' | 'face_reconfirmed'

    # Face tracking
    last_face_seen_at:  float            # monotonic timestamp
    last_face_quality:  float            # composite quality score when last seen
    face_seen_count:    int              # total frames with face visible

    # RE-ID bridging
    reid_bridge_start:       Optional[float] = None
    reid_bridge_confidence:  Optional[float] = None
    reid_bridged_cycles:     int             = 0

    # Sighting continuity
    first_seen_at:      float   = field(default_factory=time.monotonic)
    last_seen_at:       float   = field(default_factory=time.monotonic)
    preset_id:          str     = ""

    # Future: activity state
    activity_label:     Optional[str]   = None
    activity_confidence: Optional[float] = None


class IdentityTrackManager:
    """
    Manages the mapping between MOT tracks and face identities.

    Not thread-safe — call from a single async processing loop.
    """

    def __init__(self) -> None:
        # track_id → TrackIdentityState
        self._tracks:     dict[int, TrackIdentityState] = {}
        # identity_id → set of track_ids (within current preset)
        self._id_to_tracks: dict[str, set[int]] = {}

    # ── Binding ────────────────────────────────────────────────────────────────

    def bind_identity(
        self,
        track_id:    int,
        identity_id: str,
        confidence:  float,
        method:      str = "face",
        preset_id:   str = "",
    ) -> None:
        """
        Bind an identity to a track.

        Rules:
          - Face bindings always win over RE-ID bindings.
          - Higher confidence face binding wins over lower confidence face binding.
          - RE-ID binding only applies if track has no face binding yet.
        """
        now = time.monotonic()
        state = self._tracks.get(track_id)

        if state is None:
            # New track binding
            state = TrackIdentityState(
                track_id=track_id,
                identity_id=identity_id,
                identity_confidence=confidence,
                tracking_method=method,
                last_face_seen_at=now if method == "face" else 0.0,
                last_face_quality=0.0,
                face_seen_count=1 if method == "face" else 0,
                first_seen_at=now,
                last_seen_at=now,
                preset_id=preset_id,
            )
            self._tracks[track_id] = state
        else:
            # Update existing binding — apply precedence rules
            if method == "face":
                if state.tracking_method != "face" or confidence > state.identity_confidence:
                    # Face binding always wins; higher confidence wins among face binds
                    prev_id = state.identity_id
                    state.identity_id         = identity_id
                    state.identity_confidence = confidence
                    state.tracking_method     = (
                        "face_reconfirmed" if state.tracking_method in ("face", "reid_bridge")
                        else "face"
                    )
                    state.last_face_seen_at   = now
                    state.face_seen_count    += 1
                    state.reid_bridge_start   = None  # end any RE-ID bridge

                    # Update identity→track index
                    if prev_id and prev_id in self._id_to_tracks:
                        self._id_to_tracks[prev_id].discard(track_id)
                    self._id_to_tracks.setdefault(identity_id, set()).add(track_id)
                else:
                    state.last_face_seen_at  = now
                    state.face_seen_count   += 1
                return

            elif method == "reid_bridge":
                if state.identity_id is None:
                    # Track not yet identified — accept RE-ID bridge
                    state.identity_id         = identity_id
                    state.identity_confidence = confidence
                    state.tracking_method     = "reid_bridge"
                    if state.reid_bridge_start is None:
                        state.reid_bridge_start = now
                    state.reid_bridge_confidence = confidence
                    state.reid_bridged_cycles   += 1
                    self._id_to_tracks.setdefault(identity_id, set()).add(track_id)
                # If already identified by face, RE-ID cannot change it
                return

            state.last_seen_at = now
            state.preset_id    = preset_id or state.preset_id

        # Update identity→track reverse index
        self._id_to_tracks.setdefault(identity_id, set()).add(track_id)

    def touch_track(self, track_id: int, preset_id: str = "") -> None:
        """Update last_seen_at for an existing track (called every frame)."""
        state = self._tracks.get(track_id)
        if state:
            state.last_seen_at = time.monotonic()
            if preset_id:
                state.preset_id = preset_id

    def update_face_quality(self, track_id: int, quality: float) -> None:
        """Record latest face quality score for a track."""
        state = self._tracks.get(track_id)
        if state:
            state.last_face_quality = quality
            state.last_face_seen_at = time.monotonic()

    # ── Lookups ────────────────────────────────────────────────────────────────

    def get(self, track_id: int) -> Optional[TrackIdentityState]:
        return self._tracks.get(track_id)

    def get_track_for_identity(
        self, identity_id: str, preset_id: str
    ) -> Optional[int]:
        """
        Return the active track_id for this identity in this preset, if any.
        Used to prevent creating duplicate tracks for the same face.
        """
        tracks = self._id_to_tracks.get(identity_id, set())
        for tid in tracks:
            state = self._tracks.get(tid)
            if state and state.preset_id == preset_id:
                return tid
        return None

    def all_states(self) -> list[TrackIdentityState]:
        return list(self._tracks.values())

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def on_track_lost(self, track_id: int) -> Optional[TrackIdentityState]:
        """
        Called when MOT tracker deletes a track.
        Returns the final state for the sighting engine to close the sighting.
        """
        state = self._tracks.pop(track_id, None)
        if state and state.identity_id:
            self._id_to_tracks.get(state.identity_id, set()).discard(track_id)
        return state

    def on_preset_change(self, old_preset: str, new_preset: str) -> list[TrackIdentityState]:
        """
        PTZ moved to a new preset.
        Closes all tracks from the old preset and returns their final states.
        """
        lost = [
            s for s in self._tracks.values()
            if s.preset_id == old_preset
        ]
        for s in lost:
            self._tracks.pop(s.track_id, None)
            if s.identity_id:
                self._id_to_tracks.get(s.identity_id, set()).discard(s.track_id)
        return lost

    def prune_stale(self, max_age_s: float = 10.0) -> list[TrackIdentityState]:
        """Remove tracks not updated within max_age_s seconds."""
        now    = time.monotonic()
        stale  = [s for s in self._tracks.values() if now - s.last_seen_at > max_age_s]
        for s in stale:
            self._tracks.pop(s.track_id, None)
            if s.identity_id:
                self._id_to_tracks.get(s.identity_id, set()).discard(s.track_id)
        return stale
