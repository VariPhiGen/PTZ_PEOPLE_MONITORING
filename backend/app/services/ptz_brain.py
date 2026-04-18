"""
PTZBrain — autonomous PTZ scan-and-recognize state machine for ACAS.

One PTZBrain instance manages one camera.  Multiple instances run as parallel
asyncio Tasks, sharing the GPU through a module-level inference semaphore.

State machine
─────────────
  OVERVIEW_SCAN ──► PATH_PLAN ──► CELL_TRANSIT ──► CELL_RECOGNIZE
                                                        │        │
                                                   FACE_HUNT  CELL_COMPLETE
                                                        │        │
                                                   CELL_COMPLETE │
                                                        └────────┘
                                                            │
                                                     CYCLE_COMPLETE ──► OVERVIEW_SCAN

Interrupt levels (evaluated before every state transition):
  P0  EMERGENCY        decoder failure or camera offline
  P1  MOTION_INTERRUPT new person detected at frame edge during recognition
  P2  FACULTY_HUNT     faculty not seen for > 5 minutes

Modes:
  ATTENDANCE  — roster-scoped FAISS search, duration tracking, gatekeeper events
  MONITORING  — institution-wide pgvector search, sighting engine, 5-min overview

Multiple cameras share one AIPipeline via GPUManager which provides
bounded concurrency (default 3), VRAM monitoring, and graceful degradation.

Kafka events published (fire-and-forget; no-op if producer is None):
  acas.detection_events  — per-face detection (person_id, conf, liveness, bbox)
  acas.attendance        — attendance status change (PRESENT / ABSENT)
  acas.sightings         — new / updated sighting (MONITORING mode)
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import cv2
import numpy as np
from sqlalchemy import text

from app.services.ai_pipeline import AIPipeline, FaceWithEmbedding, FrameResult, PersonDetection
from app.services.auto_tracker import AutoTracker
from app.services.face_repository import FaceRepository, IdentifyResult
from app.services.onvif_controller import ONVIFController, PTZPosition
from app.services.path_planner import PathPlanner, PlanResult
from app.services.rtsp_decoder import RTSPDecoder
from app.services.zone_mapper import ScanMap, ZoneMapper, _zoom_for_fov

logger = logging.getLogger(__name__)

# ── Tunable constants ──────────────────────────────────────────────────────────

_OVERVIEW_ZOOM          = 0.0     # wide-angle for zone mapping
_OVERVIEW_SETTLE_S      = 1.5     # settle after zooming out before mapping
_CELL_SETTLE_S          = 0.70    # settle after absolute_move before grabbing frame
_FRAME_GRAB_TIMEOUT_S   = 4.0     # max wait for a fresh frame
_RECOGNIZE_FRAMES       = 2       # frames grabbed per cell for recognition
_RECOGNIZE_FRAME_GAP_S  = 0.10    # gap between recognition frames
_FACE_HUNT_TARGET_IOD   = 130.0   # target IOD (px) when zooming in on a face
_FACE_HUNT_MAX_ATTEMPTS = 2       # max zoom-in attempts per unrecognized face
_FACE_HUNT_BUDGET_S     = 12.0    # hard time budget per cell for face hunt
_FACULTY_HUNT_TIMEOUT_S = 300.0   # 5 min without faculty → P2 interrupt
_MONITORING_OVERVIEW_S  = 300.0   # 5 min between overviews in MONITORING mode
_EDGE_FRACTION          = 0.10    # person within this fraction of edge → P1
_ABSENCE_GRACE_S        = 30.0    # gap within which person still counts as present
_EMERGENCY_RETRY_LIMIT  = 5       # P0 reconnect attempts before giving up
_CAMERA_MOVE_SPEED      = 0.15    # ONVIF normalised move speed — deliberately slow for sharpness
_PRECISION_MOVE_SPEED   = 0.05    # speed for zoom-in / face-hunt precision moves (very slow — commercial smoothness)
_FACE_HUNT_P_GAIN       = 0.60    # proportional gain on tracking correction: each iteration
                                   # commands 60% of the way to the target. Lets the control
                                   # loop converge smoothly rather than slamming corner-to-corner
                                   # in one command. Acts as a critically-damped first-order
                                   # filter on PTZ motion (commercial-grade smoothness).
_TRANSIT_BLUR_THRESHOLD = 80.0    # Laplacian variance below which transit frames are skipped

_PERSON_TRACK_BUDGET_S  = 8.0    # max seconds to follow one faceless person body
_PERSON_TRACK_POLL_S    = 0.10   # control-loop interval — 10 Hz (was 5 Hz / 200 ms)
_PERSON_TRACK_BODY_FRAC = 0.45   # target: person body fills this fraction of frame height
_PERSON_MIN_BODY_FRAC   = 0.08   # ignore persons whose body height < 8% of frame (too small / background)
_FACE_HUNT_FRAMES_NEEDED = 2     # face must appear in N consecutive frames before triggering FACE_HUNT
_UNKNOWN_DEDUP_THRESHOLD = 0.45  # cosine similarity above this → same unknown person → skip saving duplicate
_UNKNOWN_SESSION_CAP     = 500   # max embeddings kept in session dedup cache

# Commercial-grade quality gates for unknown-face capture.  The admin UI fills
# up with useless thumbnails when every blurry / angled / cropped face that
# failed recognition is persisted; these gates eliminate those at source.
_UNKNOWN_MIN_FACE_CONF   = 0.70   # SCRFD detection confidence floor
_UNKNOWN_MIN_IOD_PX      = 28.0   # inter-ocular distance — rejects tiny / cropped faces
_UNKNOWN_MAX_YAW_DEG     = 25.0   # absolute yaw — rejects profile / angled shots
_UNKNOWN_MAX_PITCH_DEG   = 20.0   # absolute pitch
_UNKNOWN_MIN_SHARPNESS   = 60.0   # Laplacian variance on the aligned chip — rejects motion blur
_UNKNOWN_MIN_LIVENESS    = 0.50   # fused temporal liveness floor (was 0.35)

# Persistent cross-session dedup memory (Redis).  Prevents the same unknown
# person being saved over and over across PTZ cycles and service restarts.
_UNKNOWN_REDIS_TTL_S     = 86400  # 24 h rolling window
_UNKNOWN_REDIS_CAP       = 1000   # max embeddings retained per client

# ── Person-tracking controller (continuous_move, P-only) ───────────────────
# Derivative term removed: at a variable poll period (50–200 ms depending on
# GPU load) d(err)/dt is numerically noisy and causes direction reversals.
# Pure-P with a moderate gain is stable, predictable, and sufficient at 10 Hz.
_TRACK_KP        = 0.26   # P-gain (was 0.18 — snappier without D noise)
_TRACK_MAX_SPEED = 0.18   # velocity ceiling at wide zoom (scaled by FOV)
_TRACK_DEAD_ZONE = 0.04   # pixel dead zone: 4% of half-frame (was 7%)

# Commercial-grade safety guards — added 2026-04-17 to eliminate ceiling-jump
# symptoms observed during body tracking.
_TRACK_MIN_SPEED_FACTOR = 0.30   # velocity ceiling never drops below 30% of wide-zoom value
_TILT_CEILING_MARGIN    = 0.15   # within this fraction of tilt range → watchdog arms
_TILT_CEILING_FRAMES    = 4      # consecutive upward-drift frames near ceiling → hard-stop
_REVERSAL_QUIET_S       = 0.05   # stop+settle when velocity sign flips
_SOFT_LIMIT_MIN         = 0.05   # wide-zoom soft-limit buffer
_SOFT_LIMIT_MAX         = 0.12   # narrow-zoom (high-risk) soft-limit buffer
_FACE_HUNT_ZOOM_STABLE_FRAMES = 3  # face must be this-many frames stable before zoom nudge

# FACE_HUNT closed-loop absolute-move dead zone (ONVIF normalised units).
# Face within this angular distance of frame centre → no correction issued.
# 0.005 ≈ 0.5% of the ONVIF ±1 range, roughly 0.1–0.3° at typical FOV.
_FACE_TRACK_DEAD_ZONE_ONVIF = 0.012   # was 0.005 — larger dead-zone prevents micro-corrections
                                       # from sub-pixel detection jitter. 1.2% of PTZ range ≈
                                       # ±0.4° at 40° FOV — below visual jitter threshold.

# Fixed settle time after each absolute_move in the face-tracking control loop.
# 0.15 s keeps the loop at ~5–6 Hz even for large corrections.  The camera
# accepts a new absolute_move while still moving, so under-settling is safe —
# it just means the next correction redirects an in-flight move.
_FH_SETTLE_S = 0.12   # was 0.08 — slower command cadence (≈ 8 Hz) gives the
                       # camera time to execute each move smoothly, instead of
                       # stacking conflicting commands every 80 ms.

_MIN_DWELL_S              = 2.0   # raised: allow at least 2s before adaptive exit
_ADAPTIVE_DWELL_STREAK    = 3     # raised: 3 consecutive all-recognised frames for early exit
_MAX_PRESET_SKIP_STREAK   = 1     # reduced: only skip one empty cycle before forced revisit
_DWELL_EXTEND_MAX_FACTOR  = 2.5   # max dwell = preset dwell_s × this factor when persons unresolved

_PRE_ZOOM_MIN_IOD         = 60.0  # faces with IOD below this (px) get a pre-recognition zoom
_PRE_ZOOM_BUDGET_S        = 12.0  # total time budget for the pre-zoom pass per preset (raised)

# FACE_HUNT embedding averaging: collect this many frames at stable centre,
# average their embeddings before FAISS identification — reduces per-frame noise.
_HUNT_EMBED_AVG_N         = 5     # embeddings averaged per recognition attempt
# Cross-attempt embedding bank: keep the last N high-quality embeddings across
# consecutive failed attempts on the same face — a larger bank means more
# evidence when FAISS runs, reducing false-negatives on marginal identities.
_HUNT_EMB_BANK_MAX        = 25    # max embeddings accumulated across all attempts
# Adaptive quality / liveness gates — relaxed after repeated failed attempts so
# we never permanently give up on a face that is borderline in quality.
_HUNT_QUALITY_RETRY_FLOOR = 0.55  # quality floor on attempt ≥ 2 (base 0.75 → relax to 0.55)
_HUNT_LIVENESS_RETRY_FLOOR= 0.25  # liveness floor on attempt ≥ 3 (raised from 0.20)

# Lucas-Kanade optical flow — bridges SCRFD calls so PTZ corrections run every
# frame (~25 Hz) rather than only when SCRFD fires (~5 Hz at 25fps).
# LK tracks feature points inside the face ROI; SCRFD re-seeds them periodically.
_LK_WIN_SIZE       = (15, 15)
_LK_MAX_LEVEL      = 3
_LK_CRITERIA       = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
_LK_MIN_POINTS     = 4           # min tracked points before LK is considered valid
_LK_SCRFD_INTERVAL = 5          # run SCRFD every N frames; LK bridges the others


# ── Enums ──────────────────────────────────────────────────────────────────────

class BrainState(str, Enum):
    IDLE             = "IDLE"
    # ── Preset-based scanning (active mode) ───────────────────────────────────
    PRESET_TRANSIT   = "PRESET_TRANSIT"    # smooth move to preset, capture in-flight
    PRESET_RECOGNIZE = "PRESET_RECOGNIZE"  # dwell at preset, full recognition pipeline
    FACE_HUNT        = "FACE_HUNT"         # zoom-in on unrecognized face
    PRESET_COMPLETE  = "PRESET_COMPLETE"   # advance to next preset
    CYCLE_COMPLETE   = "CYCLE_COMPLETE"    # all presets visited; reload + start over
    STOPPED          = "STOPPED"
    ERROR            = "ERROR"


class OperatingMode(str, Enum):
    ATTENDANCE  = "ATTENDANCE"
    MONITORING  = "MONITORING"
    AUTO_TRACK  = "AUTO_TRACK"   # Axis/Hikvision-style multi-person auto-tracking


class InterruptFlag(str, Enum):
    P0_EMERGENCY        = "P0_EMERGENCY"
    P1_MOTION_INTERRUPT = "P1_MOTION_INTERRUPT"
    P2_FACULTY_HUNT     = "P2_FACULTY_HUNT"


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class DurationTracker:
    """Accumulates per-person presence duration across recognition events."""
    person_id:        str
    first_seen:       float          # epoch
    last_seen:        float          # epoch of most recent detection
    total_seconds:    float = 0.0    # accumulated presence duration
    detection_count:  int   = 0
    state:            str   = "PRESENT"  # PRESENT | ABSENT | LEFT
    similarity_sum:   float = 0.0
    similarity_count: int   = 0
    cycles_seen:      int   = 0              # how many cycles this person was detected in
    # Evidence chips: sampled every _EVIDENCE_SAMPLE_N detections, capped at _EVIDENCE_MAX
    _pending_chips:   list  = field(default_factory=list, repr=False)  # not yet uploaded
    _uploaded_count:  int   = field(default=0, repr=False)             # how many already in DB
    _window_start:    float = field(default=0.0, repr=False)
    absence_grace_s:  float = field(default=_ABSENCE_GRACE_S, repr=False)

    @property
    def confidence_avg(self) -> float | None:
        return (self.similarity_sum / self.similarity_count) if self.similarity_count > 0 else None

    def on_detected(self, ts: float, similarity: float | None = None) -> None:
        """Call each time the person is detected (ts = epoch)."""
        self.detection_count += 1
        if similarity is not None:
            self.similarity_sum   += similarity
            self.similarity_count += 1
        if self._window_start == 0.0:
            self._window_start = ts
        elif ts - self.last_seen > self.absence_grace_s:
            # Gap too large — close the old presence window, start new one
            self.total_seconds += self.last_seen - self._window_start
            self._window_start = ts
        self.last_seen = ts
        self.state = "PRESENT"

    def mark_absent(self) -> None:
        """Call when a full cycle passes without detecting this person."""
        if self._window_start > 0.0:
            self.total_seconds += self.last_seen - self._window_start
            self._window_start = 0.0
        self.state = "ABSENT"

    def finalize(self) -> None:
        """Call when the session ends to close any open window."""
        if self._window_start > 0.0:
            self.total_seconds += self.last_seen - self._window_start
            self._window_start = 0.0
        self.state = "LEFT"


_EVIDENCE_SAMPLE_N = 5    # capture a chip every N recognitions
_EVIDENCE_MAX      = 5    # max chips stored per record


@dataclass
class ScanCellSnapshot:
    """Per-cell stats accumulated during a session, used by SelfLearner."""
    cell_id:            str
    center_pan:         float
    center_tilt:        float
    zoom_used:          float   = 0.0
    total_dwell_s:      float   = 0.0
    recognition_count:  int     = 0
    unrecognized_count: int     = 0
    hunt_count:         int     = 0
    hunt_success_count: int     = 0
    expected_faces:     int     = 0


@dataclass
class SessionState:
    """Snapshot of the brain's live state — sent over WebSocket to dashboard."""
    session_id:      str
    camera_id:       str
    client_id:       str
    mode:            str
    brain_state:     str
    cycle_count:     int
    path_index:      int
    total_cells:     int
    faculty_cell_id: str | None
    recognition_rate: float          # 0.0–1.0 for the last cycle
    present_count:   int
    absent_count:    int
    unknown_count:   int             # unmatched faces this cycle
    durations:       dict[str, float]  # person_id → total_seconds
    interrupt_flags: list[str]
    fps_actual:      float
    last_frame_time: float
    cycle_time_s:    float
    updated_at:      float = field(default_factory=time.time)
    # Fields consumed by SelfLearner
    started_at:      float = 0.0
    elapsed_s:       float = 0.0
    scan_cells:      list[ScanCellSnapshot] = field(default_factory=list)
    person_count_peak: int = 0
    faculty_detected: bool = False
    measured_pan_speed:  float | None = None
    measured_tilt_speed: float | None = None


@dataclass
class _SessionConfig:
    """Immutable config set when run_session() is called."""
    session_id:  str
    client_id:   str
    camera_id:   str
    roster_ids:  list[str]
    roi_rect:    dict | None
    faculty_id:  str | None
    mode:        OperatingMode
    on_complete: Callable[[str, Exception | None], Any] | None
    dataset_id:  str | None = None
    dataset_ids: list[str] | None = None   # all datasets for multi-dataset recognition
    # Per-client recognition thresholds (loaded from client settings at session start)
    face_tier1_threshold: float = 0.78
    face_tier2_threshold: float = 0.85
    camera_type: str = "PTZ"   # PTZ | PTZ_ZOOM | BULLET_ZOOM | BULLET
    force_start: bool = False   # True = manually started; skip timetable auto-stop


# ── Simple IoU-based person tracker ──────────────────────────────────────────

def _compute_iou(a: tuple, b: tuple) -> float:
    """Compute IoU between two bboxes (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


@dataclass
class _TrackedPerson:
    """One tracked person bounding box with a persistent ID across frames."""
    track_id:  int
    bbox:      tuple[float, float, float, float]  # x1, y1, x2, y2
    last_seen: float                               # epoch timestamp
    person_id: str | None = None                  # set once recognised


@dataclass
class _EmbeddingCacheEntry:
    """Per-person embedding stored for cross-preset re-identification."""
    person_id: str
    embedding: "np.ndarray"   # [512] float32 L2-normalised ArcFace embedding
    last_seen: float           # epoch of most recent observation this cycle


class SimplePersonTracker:
    """
    Lightweight IoU-based multi-person tracker with cross-preset embedding cache.

    Within a preset:
      update() matches PersonDetections to existing tracks via greedy best-IoU.
      Tracks not seen for _MAX_AGE_S are evicted.

    Across presets (camera has panned away):
      IoU matching fails because the person's bbox is in a completely different
      position (or off-screen).  Instead, cache the 512-D ArcFace embedding the
      first time a person is identified.  At each subsequent preset, call
      find_by_embedding() before running FAISS — if cosine similarity exceeds
      _EMBED_SIM_THRESHOLD the person is matched instantly at zero GPU cost.

    Lifecycle:
      reset() is called at CYCLE_COMPLETE.  It clears both tracks AND the
      embedding cache so that a new cycle starts fresh (e.g. new arrivals are
      properly re-checked for attendance).
    """

    _IOU_THRESHOLD    = 0.25   # minimum IoU to associate detection → track
    _MAX_AGE_S        = 8.0    # evict track if not updated within this window
    _EMBED_SIM_THRESH = 0.62   # cosine similarity to accept cross-preset re-ID
                               # (ArcFace L2-normed: ~0.4 = same, ~0.7 = very confident)

    def __init__(self) -> None:
        self._tracks:  list[_TrackedPerson] = []
        self._next_id: int = 1
        self._embed_cache: list[_EmbeddingCacheEntry] = []

    # ── Public ────────────────────────────────────────────────────────────────

    def update(
        self,
        persons: list["PersonDetection"],
        ts: float,
    ) -> list[_TrackedPerson]:
        """
        Match *persons* to existing tracks and return the full updated track list.

        Persons whose bbox IoU with an existing track exceeds _IOU_THRESHOLD are
        associated to that track; the rest become new tracks.
        """
        # Evict stale tracks first
        self._tracks = [t for t in self._tracks if ts - t.last_seen < self._MAX_AGE_S]

        new_bboxes: list[tuple[float, float, float, float]] = [
            (float(p.bbox[0]), float(p.bbox[1]), float(p.bbox[2]), float(p.bbox[3]))
            for p in persons
        ]
        matched_track_idxs: set[int] = set()
        matched_det_idxs:   set[int] = set()

        # Greedy best-IoU matching
        for ti, track in enumerate(self._tracks):
            best_iou, best_di = 0.0, -1
            for di, bbox in enumerate(new_bboxes):
                if di in matched_det_idxs:
                    continue
                iou = _compute_iou(track.bbox, bbox)
                if iou > best_iou:
                    best_iou, best_di = iou, di
            if best_iou >= self._IOU_THRESHOLD and best_di >= 0:
                self._tracks[ti].bbox      = new_bboxes[best_di]
                self._tracks[ti].last_seen = ts
                matched_track_idxs.add(ti)
                matched_det_idxs.add(best_di)

        # Create new tracks for unmatched detections
        for di, bbox in enumerate(new_bboxes):
            if di not in matched_det_idxs:
                self._tracks.append(_TrackedPerson(
                    track_id=self._next_id,
                    bbox=bbox,
                    last_seen=ts,
                ))
                self._next_id += 1

        return list(self._tracks)

    def get_track_for_bbox(
        self,
        bbox: tuple[float, float, float, float],
    ) -> _TrackedPerson | None:
        """Return the track whose bbox best overlaps *bbox*, or None if below threshold."""
        best_iou, best_track = 0.0, None
        for track in self._tracks:
            iou = _compute_iou(track.bbox, bbox)
            if iou > best_iou:
                best_iou, best_track = iou, track
        return best_track if best_iou >= self._IOU_THRESHOLD else None

    def set_recognised(self, track_id: int, person_id: str) -> None:
        """Mark a track as recognised; future calls will expose person_id."""
        for t in self._tracks:
            if t.track_id == track_id:
                t.person_id = person_id
                return

    # ── Cross-preset embedding cache ──────────────────────────────────────────

    def cache_embedding(self, person_id: str, embedding: "np.ndarray", ts: float) -> None:
        """
        Store or update the embedding for a recognised person.

        Called immediately after a successful FAISS identification so that
        subsequent presets can re-identify the same person without GPU cost.
        The embedding is the ArcFace 512-D L2-normalised vector from the frame
        where identification succeeded (highest quality frame wins, if called
        multiple times the latest replaces the previous).
        """
        for entry in self._embed_cache:
            if entry.person_id == person_id:
                entry.embedding = embedding
                entry.last_seen = ts
                return
        self._embed_cache.append(_EmbeddingCacheEntry(
            person_id=person_id,
            embedding=embedding,
            last_seen=ts,
        ))

    def find_by_embedding(self, embedding: "np.ndarray") -> str | None:
        """
        Return a person_id if *embedding* matches a cached embedding above
        _EMBED_SIM_THRESH, else None.

        Uses cosine similarity (dot product of L2-normalised vectors).
        Called before FAISS so that cross-preset re-identification costs
        only a few dot products instead of a GPU round-trip.
        """
        if not self._embed_cache:
            return None
        best_sim, best_pid = 0.0, None
        for entry in self._embed_cache:
            sim = float(np.dot(embedding, entry.embedding))
            if sim > best_sim:
                best_sim, best_pid = sim, entry.person_id
        return best_pid if best_sim >= self._EMBED_SIM_THRESH else None

    def reset(self) -> None:
        """Clear all tracks and embedding cache (call at the start of each new cycle)."""
        self._tracks.clear()
        self._next_id = 1
        self._embed_cache.clear()

    def __len__(self) -> int:
        return len(self._tracks)


# ── GPU inference guard ───────────────────────────────────────────────────────
# When GPUManager is available, its semaphore gates all GPU work (inference AND
# zone mapping).  Fallback: module-level Semaphore(1) for tests without GPUManager.

_GPU_SEM_FALLBACK: asyncio.Semaphore | None = None


def _get_fallback_sem() -> asyncio.Semaphore:
    global _GPU_SEM_FALLBACK
    if _GPU_SEM_FALLBACK is None:
        _GPU_SEM_FALLBACK = asyncio.Semaphore(1)
    return _GPU_SEM_FALLBACK


# ── PTZBrain ──────────────────────────────────────────────────────────────────

class PTZBrain:
    """
    Autonomous PTZ scan-and-recognize controller for one camera.

    Parameters
    ----------
    camera          Connected ONVIFController.
    pipeline        Loaded AIPipeline (shared across cameras).
    face_repo       FaceRepository (shared across cameras).
    zone_mapper     ZoneMapper (owns its own ONVIFController reference).
    path_planner    PathPlanner.
    decoder         Optional RTSPDecoder; created from camera RTSP URL if None.
    kafka_producer  Optional confluent_kafka.Producer for event publishing.
    """

    def __init__(
        self,
        camera:        ONVIFController,
        pipeline:      AIPipeline,
        face_repo:     FaceRepository,
        zone_mapper:   ZoneMapper,
        path_planner:  PathPlanner,
        decoder:       RTSPDecoder | None = None,
        kafka_producer: Any = None,
        redis:         Any = None,
        gpu_manager:   Any = None,
        learned_params: dict | None = None,
        preset_loader: Any = None,   # async callable → list[dict] (presets from DB)
        session_factory: Any = None,  # async_sessionmaker — for direct DB attendance writes
        # Runtime-configurable constants (override module defaults via Settings)
        faculty_hunt_timeout_s: float = _FACULTY_HUNT_TIMEOUT_S,
        monitoring_overview_s:  float = _MONITORING_OVERVIEW_S,
        recognize_frames:       int   = _RECOGNIZE_FRAMES,
        face_hunt_budget_s:     float = _FACE_HUNT_BUDGET_S,
        absence_grace_s:        float = _ABSENCE_GRACE_S,
    ) -> None:
        self._camera    = camera
        self._pipeline  = pipeline
        self._repo      = face_repo
        self._mapper    = zone_mapper
        self._planner   = path_planner
        self._decoder   = decoder
        self._kafka     = kafka_producer
        self._redis     = redis
        self._gpu_mgr   = gpu_manager
        self._preset_loader = preset_loader  # async () → list[dict]
        self._session_factory = session_factory  # async_sessionmaker for attendance DB writes

        # Store runtime-configurable constants (constructor overrides module defaults)
        self._faculty_hunt_timeout = faculty_hunt_timeout_s
        self._monitoring_overview  = monitoring_overview_s
        self._recognize_frames     = recognize_frames
        self._face_hunt_budget     = face_hunt_budget_s
        self._absence_grace        = absence_grace_s

        # Learned params from self-learning pipeline (overrides constructor defaults)
        lp = learned_params or {}
        self._learned_params: dict = dict(lp)   # kept live for calibration saves
        cam_speeds = lp.get("camera_speeds", {})
        self._move_speed = cam_speeds.get("move_speed", _CAMERA_MOVE_SPEED)
        self._cell_settle = cam_speeds.get("settle_s", _CELL_SETTLE_S)
        self._dwell_per_cell = lp.get("dwell_per_cell", {})
        self._scan_cell_meters: float | None = lp.get("scan_cell_meters")
        self._roi_world: list[dict] | None = lp.get("roi_rect_world")
        self._camera_distance_m: float = float(lp.get("camera_distance_m", 7.0))

        # Apply any pre-measured FOV K calibration stored in learned_params
        # so pixel_to_ptz is accurate from the very first frame.
        if lp.get("K_pan_wide"):
            self._camera.apply_calibration(lp)

        # ── Zoom-induced pan drift learning ──────────────────────────────────
        # Many PTZ cameras have a mechanically-coupled zoom motor that tugs the
        # pan axis during zoom changes.  We learn a per-camera drift factor
        # (ONVIF pan units per unit of zoom change) via an EMA over observed
        # corrections and use it to pre-compensate future moves.
        # sign convention: positive = drifts rightward when zooming in.
        self._zoom_pan_drift: float = float(lp.get("zoom_pan_drift", 0.0))
        self._zoom_pan_drift_samples: int = 0  # number of observations so far

        # Session
        self._cfg: _SessionConfig | None = None

        # State machine
        self._state       = BrainState.IDLE
        self._stop_event  = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._last_error: Exception | None = None

        # Scan state — reset each run_session()
        self._scan_map:    ScanMap | None   = None
        self._plan:        PlanResult | None = None
        self._path_index:  int              = 0
        self._cycle_count: int              = 0
        self._cycle_start: float            = 0.0
        self._last_cycle_s: float           = 0.0

        # Preset-based scanning state
        self._presets:       list[dict] = []   # [{name, pan, tilt, zoom, dwell_s}, ...]
        self._preset_index:  int        = 0    # index into self._presets for current scan
        self._preset_ptz:    PTZPosition | None = None  # current preset's target PTZ

        # Cross-frame person tracker — persists through an entire preset cycle so
        # the same physical person is not re-identified at every preset stop.
        self._person_tracker = SimplePersonTracker()

        # Per-preset face count tracking (priority ordering + occupancy-aware skip)
        # _preset_cur_faces:  accumulates during the current cycle
        # _preset_prev_faces: snapshot of the just-completed cycle (read by skip logic)
        # _preset_skip_streak: consecutive cycles a preset was skipped (reset on visit)
        # _preset_visited:    set of preset names visited at least once this session
        self._preset_cur_faces:   dict[str, int] = {}
        self._preset_prev_faces:  dict[str, int] = {}
        self._preset_skip_streak: dict[str, int] = {}
        self._preset_visited:     set[str]       = set()

        # Per-cycle recognition state
        self._cycle_seen:    set[str] = set()   # person_ids seen this cycle
        self._unknown_count: int      = 0       # unmatched faces this cycle
        # Unknown face crops to persist at end of cycle (capped to avoid RAM growth)
        self._unknown_chips: list[tuple] = []   # list of (chip_ndarray, confidence, liveness, embedding)
        # Session-level embedding cache for unknown-face deduplication.
        # A new unknown face is only saved if its cosine similarity to every
        # previously saved unknown is below _UNKNOWN_DEDUP_THRESHOLD.
        self._unknown_seen_embeddings: list[np.ndarray] = []
        # Lazy-loaded flag: on first dedup check we seed the cache from Redis
        # so duplicates persist across cycles / process restarts (24 h window).
        self._unknown_seen_loaded: bool = False

        # Faces awaiting FACE_HUNT (set during CELL_RECOGNIZE)
        self._unrec_for_hunt: list[FaceWithEmbedding] = []
        # Temporal filter: count consecutive frames each face cluster appeared unrecognised
        self._unrec_frame_count: dict[int, int] = {}  # cluster_hash → frame count
        self._last_frame: np.ndarray | None      = None
        self._last_frame_ptz: PTZPosition | None = None

        # Duration tracking
        self._durations: dict[str, DurationTracker] = {}
        self._recognition_rate: float = 0.0

        # Per-cell stats for SelfLearner
        self._cell_dwell_start: float = 0.0
        self._cell_stats: dict[str, ScanCellSnapshot] = {}
        self._person_count_peak: int = 0
        self._session_start_time: float = 0.0
        self._faculty_detected: bool = False

        # Faculty tracking
        self._faculty_last_seen: float = 0.0
        self._faculty_hunt_active: bool = False

        # Motion interrupt
        self._motion_interrupt: bool = False
        self._active_interrupts: list[str] = []

        # Monitoring mode: track last overview time
        self._last_overview_time: float = 0.0

        # Emergency retry counter
        self._emergency_retries: int = 0

        # Frame buffer (maxsize=2 keeps the freshest frame without stalling)
        self._frame_q: asyncio.Queue[tuple[int, float, np.ndarray]] = asyncio.Queue(maxsize=2)
        self._frame_reader_task: asyncio.Task | None = None

        # Annotated debug frame — updated after every pipeline call
        # Consumers (annotated-stream endpoint) read this directly.
        self._annotated_frame_jpg: bytes | None = None
        self._overlay_state:  str = "IDLE"      # brain state label
        self._overlay_preset: str = ""           # current preset name

        # PTZ position cache — avoids a network round-trip on every call.
        # The camera's position is physically stable on the 300 ms timescale
        # (move commands are issued far less frequently), so caching is safe.
        self._ptz_cache: PTZPosition | None = None
        self._ptz_cache_ts: float = 0.0
        self._PTZ_CACHE_TTL: float = 0.30   # seconds
        self._overlay_action: str = ""           # e.g. "ZOOMING", "PAN/TILT"

    # ── Public API ─────────────────────────────────────────────────────────────

    async def run_session(
        self,
        session_id:  str,
        client_id:   str,
        camera_id:   str,
        roster_ids:  list[str],
        roi_rect:    dict | None,
        faculty_id:  str | None,
        mode:        str,
        on_complete: Callable[[str, Exception | None], Any] | None = None,
        dataset_id:  str | None = None,
        dataset_ids: list[str] | None = None,
        face_tier1_threshold: float = 0.40,
        face_tier2_threshold: float = 0.45,
        camera_type: str = "PTZ",
        force_start: bool = False,
    ) -> asyncio.Task:
        """
        Start the PTZ scan session as an asyncio Task.

        Returns the Task so the caller can await it or cancel it.
        Multiple cameras: call run_session() on each brain; they run concurrently
        and share the GPU via the module-level semaphore.
        """
        if self._task and not self._task.done():
            raise RuntimeError(f"Session {self._cfg.session_id} is already running")

        self._cfg = _SessionConfig(
            session_id=session_id,
            client_id=client_id,
            camera_id=camera_id,
            roster_ids=list(roster_ids),
            roi_rect=roi_rect,
            faculty_id=faculty_id,
            mode=OperatingMode(mode),
            on_complete=on_complete,
            dataset_id=dataset_id,
            dataset_ids=dataset_ids,
            face_tier1_threshold=face_tier1_threshold,
            face_tier2_threshold=face_tier2_threshold,
            camera_type=camera_type,
            force_start=force_start,
        )
        self._stop_event.clear()
        self._session_start_time = time.time()
        self._cell_stats.clear()
        self._person_count_peak = 0
        self._faculty_detected = False
        self._task = asyncio.create_task(
            self._run(),
            name=f"ptz_brain:{camera_id}:{session_id}",
        )
        logger.info(
            "PTZBrain session started  camera=%s  session=%s  mode=%s  camera_type=%s  force_start=%s",
            camera_id, session_id, mode, camera_type, force_start,
        )
        return self._task

    async def stop(self) -> None:
        """Gracefully stop the running session."""
        self._stop_event.set()
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        logger.info("PTZBrain stopped  camera=%s", self._cfg and self._cfg.camera_id)

    def get_session_state(self) -> SessionState:
        """Return a snapshot of the current session state for WebSocket push."""
        durations = {
            pid: t.total_seconds
            for pid, t in self._durations.items()
        }
        present = sum(1 for t in self._durations.values() if t.state == "PRESENT")
        absent  = sum(1 for t in self._durations.values() if t.state == "ABSENT")

        # Preset mode: report preset count and index
        total_cells = len(self._presets) if self._presets else (
            len(self._scan_map.cells) if self._scan_map else 0
        )
        path_idx = self._preset_index if self._presets else self._path_index
        fac_id = self._scan_map.faculty_cell_id if self._scan_map else None

        h = self._decoder.health if self._decoder else None

        now = time.time()
        return SessionState(
            session_id      = self._cfg.session_id if self._cfg else "",
            camera_id       = self._cfg.camera_id if self._cfg else "",
            client_id       = self._cfg.client_id if self._cfg else "",
            mode            = self._cfg.mode.value if self._cfg else "",
            brain_state     = self._state.value,
            cycle_count     = self._cycle_count,
            path_index      = path_idx,
            total_cells     = total_cells,
            faculty_cell_id = fac_id,
            recognition_rate= self._recognition_rate,
            present_count   = present,
            absent_count    = absent,
            unknown_count   = self._unknown_count,
            durations       = durations,
            interrupt_flags = list(self._active_interrupts),
            fps_actual      = h.fps_actual if h else 0.0,
            last_frame_time = h.last_frame_time if h else 0.0,
            cycle_time_s    = self._last_cycle_s,
            updated_at      = now,
            started_at      = self._session_start_time,
            elapsed_s       = now - self._session_start_time if self._session_start_time else 0.0,
            scan_cells      = list(self._cell_stats.values()),
            person_count_peak = self._person_count_peak,
            faculty_detected  = self._faculty_detected,
            measured_pan_speed  = None,
            measured_tilt_speed = None,
        )

    # ── Redis state publishing ────────────────────────────────────────────────

    async def _publish_state(self) -> None:
        """Push current state to Redis so the WebSocket relay can broadcast it."""
        if self._redis is None or self._cfg is None:
            return
        try:
            state = self.get_session_state()
            payload = json.dumps({
                "session_id":      state.session_id,
                "camera_id":       state.camera_id,
                "client_id":       state.client_id,
                "mode":            state.mode,
                "brain_state":     state.brain_state,
                "cycle_count":     state.cycle_count,
                "path_index":      state.path_index,
                "total_cells":     state.total_cells,
                "faculty_cell_id": state.faculty_cell_id,
                "recognition_rate": state.recognition_rate,
                "present_count":   state.present_count,
                "absent_count":    state.absent_count,
                "unknown_count":   state.unknown_count,
                "interrupt_flags": state.interrupt_flags,
                "fps_actual":      state.fps_actual,
                "cycle_time_s":    state.cycle_time_s,
                "updated_at":      state.updated_at,
                "scan_cells":      self._scan_cells_snapshot(),
                "current_ptz":     self._current_ptz_snapshot(),
            })
            key     = f"acas:session:{self._cfg.session_id}:state"
            channel = key
            pipe = self._redis.pipeline()
            pipe.setex(key, 300, payload)
            pipe.publish(channel, payload)
            await pipe.execute()
        except Exception as exc:
            logger.warning("_publish_state failed: %s", exc, exc_info=True)

    def _scan_cells_snapshot(self) -> list[dict]:
        """Return preset positions as scan-cell equivalents for the dashboard."""
        if self._presets:
            return [
                {
                    "cell_id":     p.get("name", str(i)),
                    "center_pan":  float(p["pan"]),
                    "center_tilt": float(p["tilt"]),
                    "zoom":        float(p["zoom"]),
                    "faces":       0,
                    "unrec":       0,
                }
                for i, p in enumerate(self._presets)
            ]
        # Legacy zone-map fallback
        if not self._scan_map:
            return []
        return [
            {
                "cell_id":     c.cell_id,
                "center_pan":  c.center_pan,
                "center_tilt": c.center_tilt,
                "zoom":        c.required_zoom,
                "faces":       c.expected_faces,
                "unrec":       c.unrecognized_count,
            }
            for c in self._scan_map.cells
        ]

    def _current_ptz_snapshot(self) -> dict | None:
        if self._last_frame_ptz is None:
            return None
        return {
            "pan":  self._last_frame_ptz.pan,
            "tilt": self._last_frame_ptz.tilt,
            "zoom": self._last_frame_ptz.zoom,
        }

    # ── Main loop ──────────────────────────────────────────────────────────────

    async def _run(self) -> None:
        cfg = self._cfg
        assert cfg is not None

        # Register with GPUManager for per-session tracking + CUDA streams
        if self._gpu_mgr is not None:
            self._gpu_mgr.register_session(cfg.session_id)

        await self._start_decoder(cfg)

        handlers = {
            BrainState.PRESET_TRANSIT:   self._state_preset_transit,
            BrainState.PRESET_RECOGNIZE: self._state_preset_recognize,
            BrainState.FACE_HUNT:        self._state_face_hunt,
            BrainState.PRESET_COMPLETE:  self._state_preset_complete,
            BrainState.CYCLE_COMPLETE:   self._state_cycle_complete,
        }

        # Load presets from DB for first cycle
        if self._preset_loader:
            try:
                self._presets = await self._preset_loader()
            except Exception as exc:
                logger.warning("Failed to load presets: %s", exc)
                self._presets = []

        if not self._presets and cfg.camera_type in ("BULLET", "BULLET_ZOOM"):
            self._presets = [{"name": "fixed", "pan": 0.0, "tilt": 0.0, "zoom": 0.0, "dwell_s": 20.0}]
            logger.info(
                "PTZBrain: bullet camera %s — using synthetic fixed preset (20 s/cycle, no PTZ movement)",
                cfg.camera_id,
            )
        elif not self._presets:
            logger.warning(
                "PTZBrain: no scan presets defined for camera %s — "
                "add presets in the dashboard Presets tab then restart the session.",
                cfg.camera_id,
            )

        self._preset_index = 0
        self._state = BrainState.PRESET_TRANSIT if self._presets else BrainState.IDLE
        self._cycle_start = time.time()
        self._last_overview_time = time.time()

        # ── Auto-calibrate FOV if no K values stored yet ─────────────────────
        # Runs once per session start (≈30 s), measures pixel displacement for
        # known ONVIF moves, and persists K values to the camera's learned_params.
        # On subsequent session starts the stored values are loaded immediately
        # (via apply_calibration in __init__), skipping the live measurement.
        if cfg.camera_type in ("BULLET", "BULLET_ZOOM"):
            logger.info(
                "PTZBrain: skipping FOV calibration for fixed camera %s (type=%s)",
                cfg.camera_id, cfg.camera_type,
            )
        elif not self._learned_params.get("K_pan_wide"):
            logger.info(
                "PTZBrain: no FOV calibration found for camera %s — "
                "running auto-calibration now (~30 s)", cfg.camera_id,
            )
            async def _grab_for_calib():
                fd = await self._grab_frame()
                return fd[2] if fd is not None else None

            try:
                calib = await self._camera.auto_calibrate_fov(_grab_for_calib)
                if calib:
                    await self._save_calibration(calib, cfg.camera_id)
            except Exception as exc:
                logger.warning("PTZBrain: FOV auto-calibration failed: %s", exc)
        else:
            logger.info(
                "PTZBrain: loaded FOV calibration for camera %s  "
                "K_pan_wide=%.5f K_pan_narrow=%s",
                cfg.camera_id,
                self._learned_params["K_pan_wide"],
                self._learned_params.get("K_pan_narrow"),
            )

        try:
            # ── AUTO_TRACK mode bypasses the scan-and-recognize state machine.
            # Delegates to AutoTracker (two-loop commercial design: fast PTZ
            # loop + slow recognition loop) and returns when stop_event is set.
            if cfg.mode == OperatingMode.AUTO_TRACK:
                await self._run_auto_track(cfg)
                self._state = BrainState.STOPPED
                return

            while self._state not in (BrainState.STOPPED, BrainState.ERROR):
                if self._stop_event.is_set():
                    self._state = BrainState.STOPPED
                    break

                self._active_interrupts = self._check_interrupts()
                if InterruptFlag.P0_EMERGENCY.value in self._active_interrupts:
                    recovered = await self._handle_p0_emergency()
                    if not recovered:
                        self._state = BrainState.ERROR
                        self._last_error = RuntimeError("Camera offline; all reconnects exhausted")
                        break

                # ── Manual override check ─────────────────────────────────────
                # If the operator has switched the camera to MANUAL mode via the
                # dashboard, pause all autonomous moves until AI mode is restored.
                if self._redis is not None:
                    try:
                        _mode = await self._redis.get(f"camera:{cfg.camera_id}:ai_mode")
                        if isinstance(_mode, bytes):
                            _mode = _mode.decode()
                        if _mode == "MANUAL":
                            await asyncio.sleep(0.1)  # yield without burning CPU
                            continue
                    except Exception:
                        pass  # Redis unavailable — proceed normally

                if self._state == BrainState.IDLE:
                    # No presets yet — poll every 2s for new ones (was 10s)
                    await asyncio.sleep(2.0)
                    if self._preset_loader:
                        try:
                            self._presets = await self._preset_loader()
                        except Exception:
                            pass
                    if self._presets:
                        self._preset_index = 0
                        self._state = BrainState.PRESET_TRANSIT
                    continue

                handler = handlers.get(self._state)
                if handler is None:
                    logger.error("PTZBrain: no handler for state %s", self._state.value)
                    self._state = BrainState.ERROR
                    continue
                try:
                    self._state = await handler()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.error(
                        "PTZBrain error in %s: %s", self._state.value, exc, exc_info=True
                    )
                    self._last_error = exc
                    self._state = BrainState.ERROR

                # Publish state to Redis after every transition
                await self._publish_state()

        except asyncio.CancelledError:
            logger.info("PTZBrain task cancelled  camera=%s", cfg.camera_id)
        finally:
            self._state = BrainState.STOPPED
            await self._publish_state()
            await self._teardown()
            error = self._last_error if self._last_error else None
            if cfg.on_complete:
                try:
                    result = cfg.on_complete(cfg.session_id, error)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:
                    logger.exception("on_complete callback raised")

    # ── AUTO_TRACK mode ────────────────────────────────────────────────────────

    async def _run_auto_track(self, cfg: _SessionConfig) -> None:
        """
        Commercial-grade continuous auto-tracking driver.

        Wires the existing decoder/frame-queue and AIPipeline into the
        AutoTracker engine, which runs its own fast (≈25 fps) PTZ control
        loop and slow (≈4 Hz) recognition loop. Blocks until stop_event.
        """
        # Frame grabber adapter — AutoTracker expects "() -> np.ndarray | None".
        async def _grab() -> np.ndarray | None:
            fd = await self._grab_frame(timeout=1.0)
            return fd[2] if fd is not None else None

        # Identification adapter — reuses the per-session routing in _identify().
        async def _identify(fwe: FaceWithEmbedding):
            ident = await self._identify(fwe, cfg)
            if ident and ident.person_id:
                return (ident.person_id, float(ident.similarity))
            return None

        # Best-effort initial frame size from the decoder health snapshot.
        frame_wh: tuple[int, int] = (1920, 1080)
        try:
            first = await self._grab_frame(timeout=4.0)
            if first is not None:
                _, _, f = first
                fh, fw = f.shape[:2]
                frame_wh = (fw, fh)
        except Exception:
            pass

        auto = AutoTracker(
            ptz_ctrl=self._camera,
            grab_frame=_grab,
            pipeline=self._pipeline,
            identify_face=_identify,
            frame_wh=frame_wh,
            max_secondaries=3,
            enable_slow_loop=True,
        )
        await auto.start()
        try:
            # Periodically publish state + metrics while waiting for stop.
            while not self._stop_event.is_set():
                await asyncio.sleep(1.0)
                try:
                    await self._publish_state()
                except Exception:
                    pass
                logger.debug("AutoTracker metrics: %s", auto.metrics())
        finally:
            await auto.stop()

    # ── State: OVERVIEW_SCAN ───────────────────────────────────────────────────

    async def _state_overview_scan(self) -> BrainState:
        """
        Zoom out to wide angle, grab frame(s) at the zone centre and (re)build
        the ScanMap.  First cycle: ``map_zone()``.  Subsequent: ``remap()``
        unless a motion interrupt or monitoring interval forces a full remap.

        When the zone spans more than the camera's FOV at overview zoom, we
        grab frames at multiple pan/tilt positions so person detection covers
        the entire zone.
        """
        cfg = self._cfg
        assert cfg is not None

        # ── Compute overview position(s) ──────────────────────────────────
        ov_positions = self._overview_positions()

        first_cycle = self._cycle_count == 0
        force_remap = self._motion_interrupt
        self._motion_interrupt = False

        # Grab a frame at each overview position, merge person detections
        combined_frame = None
        combined_ptz: PTZPosition | None = None

        for ov_pan, ov_tilt in ov_positions:
            try:
                await self._camera.absolute_move(
                    ov_pan, ov_tilt, _OVERVIEW_ZOOM, speed=self._move_speed,
                )
                await asyncio.sleep(_OVERVIEW_SETTLE_S)
            except Exception as exc:
                logger.warning("overview move failed: %s", exc)

            frame_data = await self._grab_frame()
            if frame_data is None:
                continue

            _, ts, frame = frame_data
            ptz = await self._safe_get_ptz()
            if combined_frame is None:
                combined_frame = frame
                combined_ptz = ptz
                self._last_overview_time = ts

        if combined_frame is None:
            logger.warning("OVERVIEW_SCAN: no frame; retrying")
            await asyncio.sleep(1.0)
            return BrainState.OVERVIEW_SCAN

        self._last_frame = combined_frame
        self._last_frame_ptz = combined_ptz

        frame = combined_frame
        ptz = combined_ptz

        def _map_sync():
            if first_cycle or force_remap or self._scan_map is None:
                return self._mapper.map_zone(
                    frame, cfg.roi_rect, ptz, self._pipeline,
                    self._scan_cell_meters, self._roi_world,
                    self._camera_distance_m,
                )
            return self._mapper.remap(
                self._scan_map, frame, ptz, self._pipeline,
                cfg.roi_rect, self._scan_cell_meters, self._roi_world,
                self._camera_distance_m,
            )

        if self._gpu_mgr is not None:
            async with self._gpu_mgr.gpu_slot():
                self._scan_map = await asyncio.to_thread(_map_sync)
        else:
            sem = _get_fallback_sem()
            async with sem:
                self._scan_map = await asyncio.to_thread(_map_sync)

        logger.info(
            "OVERVIEW_SCAN: %d cells  faculty=%s  cycle=%d",
            len(self._scan_map.cells),
            self._scan_map.faculty_cell_id,
            self._cycle_count,
        )
        return BrainState.PATH_PLAN

    def _overview_positions(self) -> list[tuple[float, float]]:
        """Return (pan, tilt) positions for the overview scan.

        If the zone fits within one FOV at overview zoom, a single position at
        the zone centre is returned.  Otherwise, the zone is tiled into a grid
        of overlapping positions.
        """
        if not self._roi_world or len(self._roi_world) < 3:
            return [(0.0, 0.0)]

        pans  = [p["pan"]  for p in self._roi_world]
        tilts = [p["tilt"] for p in self._roi_world]
        bb_pan_min, bb_pan_max = min(pans), max(pans)
        bb_tilt_min, bb_tilt_max = min(tilts), max(tilts)
        zone_dpan  = bb_pan_max  - bb_pan_min
        zone_dtilt = bb_tilt_max - bb_tilt_min

        fov_h, fov_v = self._camera.get_fov_at_zoom(_OVERVIEW_ZOOM)
        limits = self._camera.limits
        pan_range  = limits.pan_max  - limits.pan_min
        tilt_range = limits.tilt_max - limits.tilt_min
        ps = self._camera._pan_scale
        ts = self._camera._tilt_scale
        # Divide by scale: converts physical FOV degrees → ONVIF units correctly
        fov_dpan  = fov_h / 360.0 * pan_range  / ps
        fov_dtilt = fov_v / 180.0 * tilt_range / ts

        # Single position if zone fits within 80% of the overview FOV
        if zone_dpan <= fov_dpan * 0.8 and zone_dtilt <= fov_dtilt * 0.8:
            return [((bb_pan_min + bb_pan_max) / 2, (bb_tilt_min + bb_tilt_max) / 2)]

        # Tile with 30% overlap
        step_pan  = fov_dpan  * 0.7
        step_tilt = fov_dtilt * 0.7
        cols = max(1, math.ceil(zone_dpan  / step_pan))
        rows = max(1, math.ceil(zone_dtilt / step_tilt))

        positions: list[tuple[float, float]] = []
        for r in range(rows):
            for c in range(cols):
                p = bb_pan_min  + step_pan  * (c + 0.5) if cols > 1 else (bb_pan_min + bb_pan_max) / 2
                t = bb_tilt_min + step_tilt * (r + 0.5) if rows > 1 else (bb_tilt_min + bb_tilt_max) / 2
                positions.append((p, t))

        logger.debug("overview tiling: %d×%d = %d positions", cols, rows, len(positions))
        return positions

    # ── State: PATH_PLAN ──────────────────────────────────────────────────────

    async def _state_path_plan(self) -> BrainState:
        cfg = self._cfg
        assert cfg is not None and self._scan_map is not None

        if not self._scan_map.cells:
            logger.info("PATH_PLAN: no cells — sleeping 5s before re-scan")
            await asyncio.sleep(5.0)
            return BrainState.OVERVIEW_SCAN

        # unrecognized_count is already maintained live by _update_cell_unrecognized
        # called from CELL_RECOGNIZE and FACE_HUNT; no reset needed here.

        ptz = await self._safe_get_ptz()
        faculty_cell = next(
            (c for c in self._scan_map.cells
             if c.cell_id == self._scan_map.faculty_cell_id),
            None,
        )

        # P2: when faculty hunt is active, force the faculty cell to highest priority
        if self._faculty_hunt_active and faculty_cell is not None:
            faculty_cell.priority += 100.0

        def _plan_sync():
            return self._planner.plan_path(
                self._scan_map.cells,
                faculty_cell,
                ptz,
                self._camera.limits,
            )

        self._plan = await asyncio.to_thread(_plan_sync)
        self._path_index = 0
        self._cycle_start = time.time()
        self._cycle_seen.clear()
        self._unknown_count = 0

        logger.info(
            "PATH_PLAN: %d cells ordered %s  est=%.1fs",
            len(self._plan.ordered_cell_ids),
            self._plan.cell_order,
            self._plan.estimated_total_s,
        )
        return BrainState.CELL_TRANSIT

    # ── State: PRESET_TRANSIT ─────────────────────────────────────────────────

    async def _state_preset_transit(self) -> BrainState:
        """
        Smoothly move to the next preset position.  Runs the full AI pipeline
        on frames captured during the move — the camera is moving slowly enough
        that faces are often recognisable in-transit.
        """
        cfg = self._cfg
        assert cfg is not None

        if not self._presets:
            return BrainState.CYCLE_COMPLETE

        if self._preset_index >= len(self._presets):
            return BrainState.CYCLE_COMPLETE

        preset = self._presets[self._preset_index]
        p_pan  = float(preset["pan"])
        p_tilt = float(preset["tilt"])
        p_zoom = float(preset["zoom"])
        preset_name = preset.get("name", str(self._preset_index))

        # ── Occupancy-aware skip ──────────────────────────────────────────────
        # If this preset was empty last cycle AND we haven't maxed out the skip
        # streak, skip it this cycle to save time.  Force a visit every
        # _MAX_PRESET_SKIP_STREAK+1 cycles so newly-arrived persons aren't missed.
        if (preset_name in self._preset_visited
                and self._preset_prev_faces.get(preset_name, -1) == 0):
            streak = self._preset_skip_streak.get(preset_name, 0)
            if streak < _MAX_PRESET_SKIP_STREAK:
                self._preset_skip_streak[preset_name] = streak + 1
                logger.info(
                    "Occupancy skip: preset '%s' was empty last cycle (streak %d/%d)",
                    preset_name, streak + 1, _MAX_PRESET_SKIP_STREAK,
                )
                self._preset_index += 1
                return (BrainState.PRESET_TRANSIT
                        if self._preset_index < len(self._presets)
                        else BrainState.CYCLE_COMPLETE)
        # Visiting this preset — reset its skip streak
        self._preset_skip_streak[preset_name] = 0
        self._preset_visited.add(preset_name)

        logger.info(
            "PRESET_TRANSIT → preset %d/%d '%s'  pan=%.3f tilt=%.3f zoom=%.3f",
            self._preset_index + 1, len(self._presets), preset.get("name", ""),
            p_pan, p_tilt, p_zoom,
        )

        self._overlay_state  = "PRESET_TRANSIT"
        self._overlay_preset = preset_name
        self._overlay_action = "MOVING"

        ptz_before = await self._safe_get_ptz()
        target_pos = PTZPosition(p_pan, p_tilt, p_zoom)
        self._preset_ptz = target_pos

        # Start smooth move (fire-and-forget — ONVIF camera handles deceleration)
        # Bullet cameras are fixed — skip the physical move entirely.
        # PTZ and PTZ_ZOOM both pan/tilt/zoom to each preset position.
        if cfg.camera_type not in ("BULLET", "BULLET_ZOOM"):
            try:
                await self._camera.smooth_absolute_move(
                    p_pan, p_tilt, p_zoom,
                    nominal_speed=self._move_speed,
                    current_pos=ptz_before,
                )
            except Exception as exc:
                logger.warning("PRESET_TRANSIT: move failed (%s); skipping preset", exc)
                return BrainState.PRESET_COMPLETE
        else:
            await asyncio.sleep(self._cell_settle)  # brief pause for bullet cameras

        # Estimate travel time and capture frames during the move
        travel_s = self._camera.estimate_travel_time(ptz_before, target_pos)
        t_end    = time.time() + max(travel_s, 0.3)
        location = preset.get("name", "transit")

        while time.time() < t_end and not self._stop_event.is_set():
            # Non-blocking frame grab — skip if queue is empty
            try:
                fd = self._frame_q.get_nowait()
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.01)  # yield; frame arrives within one decode interval
                continue

            _, _, frame = fd
            self._last_frame = frame
            self._last_frame_ptz = await self._safe_get_ptz()

            # Skip frames that are blurry due to camera movement (Laplacian variance check)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if blur_score < _TRANSIT_BLUR_THRESHOLD:
                await asyncio.sleep(0.05)
                continue

            result = await self._run_pipeline(frame, cfg.roi_rect)
            ts_now = time.time()

            if result.persons:
                self._person_count_peak = max(self._person_count_peak, len(result.persons))
                # Update cross-frame tracker so FACE_HUNT can skip already-recognised persons
                self._person_tracker.update(result.persons, ts_now)

            for fwe in self._deduplicate_faces(result.faces_with_embeddings):
                # Find the track this face belongs to (by bbox containment/proximity)
                face_bbox = (float(fwe.face.bbox[0]), float(fwe.face.bbox[1]),
                             float(fwe.face.bbox[2]), float(fwe.face.bbox[3]))
                track = self._person_tracker.get_track_for_bbox(face_bbox)
                if track and track.person_id:
                    # IoU match + already recognised in a prior frame — just update duration
                    self._cycle_seen.add(track.person_id)
                    self._update_duration(track.person_id, ts_now)
                    continue

                # Cross-preset embedding check — O(N) dot products, zero GPU cost
                cached_pid = self._person_tracker.find_by_embedding(fwe.embedding)
                if cached_pid:
                    self._cycle_seen.add(cached_pid)
                    self._update_duration(cached_pid, ts_now)
                    if track:
                        self._person_tracker.set_recognised(track.track_id, cached_pid)
                    continue

                id_result = await self._identify(fwe, cfg)
                if id_result.person_id:
                    self._cycle_seen.add(id_result.person_id)
                    self._update_duration(id_result.person_id, ts_now, similarity=id_result.similarity, chip=fwe.face_chip)
                    await self._publish_detection(fwe, id_result, cfg, location)
                    self._person_tracker.cache_embedding(id_result.person_id, fwe.embedding, ts_now)
                    if track:
                        self._person_tracker.set_recognised(track.track_id, id_result.person_id)
                    if cfg.faculty_id and id_result.person_id == cfg.faculty_id:
                        self._faculty_last_seen = ts_now
                        self._faculty_detected = True
            await asyncio.sleep(0.08)  # fast transit loop — more frames = better coverage

        # Settle at preset before recognition dwell
        await asyncio.sleep(self._cell_settle)
        self._cell_dwell_start = time.time()
        return BrainState.PRESET_RECOGNIZE

    # ── State: PRESET_RECOGNIZE ───────────────────────────────────────────────

    async def _state_preset_recognize(self) -> BrainState:
        """
        Dwell at the current preset for ``dwell_s`` seconds running the full AI
        pipeline continuously.

        - Cross-frame tracker is updated each frame; persons already recognised
          in this cycle (tracker.person_id set) are skipped to avoid redundant
          GPU work.
        - Persons with no visible face trigger body-tracking (P-controller) to
          find the face before the dwell window expires.
        - Unrecognised faces at the end of the dwell period trigger FACE_HUNT.
        """
        cfg = self._cfg
        assert cfg is not None

        if not self._presets or self._preset_index >= len(self._presets):
            return BrainState.PRESET_COMPLETE

        preset   = self._presets[self._preset_index]
        dwell_s  = float(preset.get("dwell_s", 3.0))
        location = preset.get("name", "")

        all_fwes:        list[FaceWithEmbedding] = []
        last_persons:    list[PersonDetection]   = []
        t_end            = time.time() + dwell_s
        dwell_start      = time.time()
        _adaptive_streak = 0
        _dwell_recognized: dict[str, str] = {}   # face_idx → person_name for overlay

        self._overlay_state  = "PRESET_RECOGNIZE"
        self._overlay_preset = location
        self._overlay_action = ""

        while time.time() < t_end and not self._stop_event.is_set():
            fd = await self._grab_frame()
            if fd is None:
                break
            _, ts_frame, frame = fd
            self._last_frame     = frame
            self._last_frame_ptz = await self._safe_get_ptz()

            result = await self._run_pipeline(frame, cfg.roi_rect)
            ts_now = time.time()

            self._annotate_and_store(frame, result, _dwell_recognized)

            logger.debug(
                "PRESET_RECOGNIZE frame  preset='%s'  shape=%s  "
                "persons=%d  faces=%d  liveness=%s",
                location,
                frame.shape if frame is not None else "None",
                len(result.persons),
                len(result.faces_with_embeddings),
                [round(f.liveness, 2) for f in result.faces_with_embeddings],
            )

            if result.persons:
                self._person_count_peak = max(self._person_count_peak, len(result.persons))
                last_persons = result.persons
                self._person_tracker.update(result.persons, ts_now)

            # Quick-recognise: check IoU-track person_id, then embedding cache
            for fwe in result.faces_with_embeddings:
                face_bbox = (float(fwe.face.bbox[0]), float(fwe.face.bbox[1]),
                             float(fwe.face.bbox[2]), float(fwe.face.bbox[3]))
                track = self._person_tracker.get_track_for_bbox(face_bbox)
                if track and track.person_id:
                    self._cycle_seen.add(track.person_id)
                    self._update_duration(track.person_id, ts_now)
                    continue
                # Cross-preset: check embedding cache before deferring to post-dwell FAISS
                cached_pid = self._person_tracker.find_by_embedding(fwe.embedding)
                if cached_pid:
                    self._cycle_seen.add(cached_pid)
                    self._update_duration(cached_pid, ts_now)
                    if track:
                        self._person_tracker.set_recognised(track.track_id, cached_pid)
                    continue
                all_fwes.append(fwe)

            # ── Adaptive dwell early-exit ─────────────────────────────────────
            # If every currently-visible person is already identified, there is
            # nothing left to learn at this preset — exit the dwell loop early.
            # Guard: must have elapsed at least _MIN_DWELL_S to avoid triggering
            # on the first frame before slow-walkers have entered the frame.
            t_now = time.time()
            elapsed = t_now - dwell_start

            if result.persons and elapsed >= _MIN_DWELL_S:
                def _resolved(p: "PersonDetection") -> bool:
                    t = self._person_tracker.get_track_for_bbox((
                        float(p.bbox[0]), float(p.bbox[1]),
                        float(p.bbox[2]), float(p.bbox[3]),
                    ))
                    return t is not None and t.person_id is not None

                all_resolved = all(_resolved(p) for p in result.persons)
                if all_resolved:
                    _adaptive_streak += 1
                    if _adaptive_streak >= _ADAPTIVE_DWELL_STREAK:
                        logger.debug(
                            "Adaptive dwell: all %d person(s) recognised — "
                            "early exit at preset '%s'",
                            len(result.persons), location,
                        )
                        break
                else:
                    _adaptive_streak = 0

                    # ── Adaptive dwell EXTENSION ──────────────────────────────
                    # When the initial dwell window is about to close but at least
                    # one visible person is still unresolved, extend t_end by up
                    # to _DWELL_EXTEND_MAX_FACTOR × dwell_s total.  This prevents
                    # missing people who were briefly occluded or looking away.
                    max_t_end = dwell_start + dwell_s * _DWELL_EXTEND_MAX_FACTOR
                    if t_now >= t_end - 0.25 and t_now < max_t_end:
                        n_unresolved = sum(
                            1 for p in result.persons if not _resolved(p)
                        )
                        extend_s = min(2.0, max_t_end - t_now)
                        if extend_s > 0.1:
                            t_end = min(max_t_end, t_now + extend_s)
                            logger.debug(
                                "PRESET_RECOGNIZE: extending dwell +%.1fs "
                                "(%d unresolved) at preset '%s'",
                                extend_s, n_unresolved, location,
                            )

            await asyncio.sleep(0.10)   # tighter loop — faster coverage within dwell

        # Body-tracking pass for persons whose face was never visible during dwell
        persons_no_face = self._persons_without_face(last_persons, all_fwes)
        if persons_no_face:
            # Filter out persons whose track is already recognised
            untracked_no_face: list[PersonDetection] = []
            for p in persons_no_face:
                p_bbox = (float(p.bbox[0]), float(p.bbox[1]),
                          float(p.bbox[2]), float(p.bbox[3]))
                track = self._person_tracker.get_track_for_bbox(p_bbox)
                if track and track.person_id:
                    continue  # already known — skip body-tracking cost
                # Skip persons too small to be real (background clutter, far-away objects)
                _frame_h = self._last_frame.shape[0] if self._last_frame is not None else 1080
                body_h_frac = (p.bbox[3] - p.bbox[1]) / _frame_h
                if body_h_frac < _PERSON_MIN_BODY_FRAC:
                    logger.debug("PRESET_RECOGNIZE: skipping tiny person (h=%.2f%%) at '%s'",
                                 body_h_frac * 100, location)
                    continue
                untracked_no_face.append(p)
            if untracked_no_face:
                logger.info(
                    "PRESET_RECOGNIZE: %d person(s) without face → body tracking  preset='%s'",
                    len(untracked_no_face), location,
                )
                tracked = await self._track_person_for_face(untracked_no_face, cfg, location)
                all_fwes.extend(tracked)

        if not all_fwes:
            logger.debug("PRESET_RECOGNIZE: no new faces at preset '%s'", location)
            return BrainState.PRESET_COMPLETE

        # ── Pre-recognition zoom pass ─────────────────────────────────────────
        # Upgrade any face whose IOD is too small for reliable ArcFace embedding
        # BEFORE running FAISS — improves first-attempt accuracy and produces
        # higher-quality embeddings for cross-preset cache re-use.
        if self._presets and self._preset_index < len(self._presets):
            _p = self._presets[self._preset_index]
            _restore = PTZPosition(float(_p["pan"]), float(_p["tilt"]), float(_p["zoom"]))
            all_fwes = await self._pre_zoom_faces(all_fwes, _restore, cfg, location)

        unique_fwes   = self._deduplicate_faces(all_fwes)
        unrecognized: list[FaceWithEmbedding] = []
        ts_now = time.time()
        _dwell_recognized = {}   # reset — will be populated as FAISS runs

        for fwe in unique_fwes:
            face_bbox = (float(fwe.face.bbox[0]), float(fwe.face.bbox[1]),
                         float(fwe.face.bbox[2]), float(fwe.face.bbox[3]))
            track = self._person_tracker.get_track_for_bbox(face_bbox)
            if track and track.person_id:
                # IoU match + tracker already has this person — no GPU call needed
                self._cycle_seen.add(track.person_id)
                self._update_duration(track.person_id, ts_now)
                continue

            # Cross-preset embedding check — O(N) dot products, zero GPU cost.
            # Handles the case where the same person appears at a new preset where
            # IoU matching fails because the camera has panned to a different position.
            cached_pid = self._person_tracker.find_by_embedding(fwe.embedding)
            if cached_pid:
                self._cycle_seen.add(cached_pid)
                self._update_duration(cached_pid, ts_now)
                if track:
                    self._person_tracker.set_recognised(track.track_id, cached_pid)
                continue

            id_result = await self._identify(fwe, cfg)
            if id_result.person_id:
                self._cycle_seen.add(id_result.person_id)
                self._update_duration(id_result.person_id, ts_now, similarity=id_result.similarity, chip=fwe.face_chip)
                await self._publish_detection(fwe, id_result, cfg, location)
                # Cache embedding so later presets can re-identify without GPU
                self._person_tracker.cache_embedding(id_result.person_id, fwe.embedding, ts_now)
                if track:
                    self._person_tracker.set_recognised(track.track_id, id_result.person_id)
                if cfg.faculty_id and id_result.person_id == cfg.faculty_id:
                    self._faculty_last_seen = ts_now
                    self._faculty_detected = True
                # Update overlay so the recognised name appears immediately
                face_idx = str(len(_dwell_recognized))
                _dwell_recognized[face_idx] = id_result.person_id[:20]
                if self._last_frame is not None:
                    from app.services.ai_pipeline import FrameResult as _FR
                    _dummy = _FR(persons=[], faces_with_embeddings=unique_fwes, tracked_persons=[], time_ms=0.0)
                    self._annotate_and_store(self._last_frame, _dummy, _dwell_recognized)
            else:
                self._unknown_count += 1
                unrecognized.append(fwe)
                # Collect a sample chip for unknown_detections (cap at 3 per cycle).
                # Quality + dedup gates live in _unknown_chip_gate(); passing chips
                # get a context-padded frame crop (hair + shoulders visible) rather
                # than the tight 112×112 ArcFace chip.
                if len(self._unknown_chips) < 3 and fwe.face_chip is not None:
                    if await self._unknown_chip_gate(fwe):
                        crop = self._padded_face_crop(fwe)
                        if crop is None:
                            crop = fwe.face_chip
                        self._unknown_chips.append(
                            (crop, fwe.face.conf, fwe.liveness, fwe.embedding)
                        )
                        if fwe.embedding is not None:
                            self._unknown_seen_embeddings.append(fwe.embedding)
                            if len(self._unknown_seen_embeddings) > _UNKNOWN_SESSION_CAP:
                                self._unknown_seen_embeddings = \
                                    self._unknown_seen_embeddings[-_UNKNOWN_SESSION_CAP:]

        # Update per-preset face counter (total unique faces: recognised + unknown)
        self._preset_cur_faces[location] = (
            self._preset_cur_faces.get(location, 0) + len(unique_fwes)
        )

        # ── Temporal consistency filter ───────────────────────────────────────
        # A face must appear unrecognised in ≥_FACE_HUNT_FRAMES_NEEDED consecutive
        # dwell frames before triggering FACE_HUNT.  Single-frame false positives
        # (wall posters, boards, reflections) are silently dropped.
        confirmed: list[FaceWithEmbedding] = []
        seen_hashes: set[int] = set()
        for fwe in unrecognized:
            # Use rounded bbox centre as a stable cluster key across frames
            cx = round((fwe.face.bbox[0] + fwe.face.bbox[2]) / 2.0 / 20) * 20
            cy = round((fwe.face.bbox[1] + fwe.face.bbox[3]) / 2.0 / 20) * 20
            key = hash((cx, cy))
            seen_hashes.add(key)
            count = self._unrec_frame_count.get(key, 0) + 1
            self._unrec_frame_count[key] = count
            if count >= _FACE_HUNT_FRAMES_NEEDED:
                confirmed.append(fwe)
        # Expire clusters not seen this dwell
        self._unrec_frame_count = {k: v for k, v in self._unrec_frame_count.items()
                                   if k in seen_hashes}

        self._unrec_for_hunt = confirmed
        if confirmed:
            logger.debug(
                "PRESET_RECOGNIZE: %d confirmed unrecognised → FACE_HUNT  preset='%s'",
                len(confirmed), location,
            )
            return BrainState.FACE_HUNT

        if unrecognized:
            logger.debug(
                "PRESET_RECOGNIZE: %d unrecognised filtered by temporal check (need %d frames)  preset='%s'",
                len(unrecognized), _FACE_HUNT_FRAMES_NEEDED, location,
            )

        return BrainState.PRESET_COMPLETE

    # ── State: PRESET_COMPLETE ────────────────────────────────────────────────

    async def _state_preset_complete(self) -> BrainState:
        """Advance to next preset or finish the cycle."""
        dwell = time.time() - self._cell_dwell_start if self._cell_dwell_start else 0.0
        if self._presets and self._preset_index < len(self._presets):
            name = self._presets[self._preset_index].get("name", str(self._preset_index))
            snap = self._cell_stats.get(name)
            if snap is None:
                p = self._presets[self._preset_index]
                snap = ScanCellSnapshot(
                    cell_id=name,
                    center_pan=float(p["pan"]),
                    center_tilt=float(p["tilt"]),
                    zoom_used=float(p["zoom"]),
                )
                self._cell_stats[name] = snap
            snap.total_dwell_s     += dwell
            # Accumulate total face sightings (recognised + unknown) for priority sort
            snap.recognition_count += self._preset_cur_faces.get(name, 0)

        self._preset_index += 1
        if self._preset_index < len(self._presets):
            return BrainState.PRESET_TRANSIT
        return BrainState.CYCLE_COMPLETE

    # ── State: CELL_TRANSIT ───────────────────────────────────────────────────

    async def _state_cell_transit(self) -> BrainState:
        assert self._plan is not None and self._scan_map is not None

        if self._path_index >= len(self._plan.cell_order):
            return BrainState.CYCLE_COMPLETE

        cell_idx = self._plan.cell_order[self._path_index]
        cell = self._scan_map.cells[cell_idx]

        ptz_before = await self._safe_get_ptz()
        zoom_before = ptz_before.zoom if ptz_before else cell.required_zoom
        zoom_delta  = cell.required_zoom - zoom_before          # signed: +ve = zooming in
        zoom_mag    = abs(zoom_delta)

        # ── Pre-compensate for learned zoom-induced pan drift ─────────────
        # After enough observations the drift factor converges.  We pre-aim
        # the camera so that when zoom drags pan, it lands on target.
        # compensation_pan = target_pan - drift_factor * zoom_delta
        # (e.g. if drift_factor=-0.02 and zoom_delta=+0.4, add +0.008 to pan)
        drift_comp_pan = -self._zoom_pan_drift * zoom_delta if zoom_mag > 0.05 else 0.0
        adjusted_pan   = cell.center_pan + drift_comp_pan

        target = PTZPosition(
            pan=cell.center_pan,
            tilt=cell.center_tilt,
            zoom=cell.required_zoom,
        )

        # Move camera using S-curve velocity profiling.
        # smooth_absolute_move() selects speed based on move distance (micro /
        # short / medium / large tiers) and returns an adaptive settle time.
        # zoom_speed runs the zoom motor at 90% — independent axis, high inertia.
        try:
            adaptive_settle = await self._camera.smooth_absolute_move(
                adjusted_pan, cell.center_tilt, cell.required_zoom,
                nominal_speed=self._move_speed,
                current_pos=ptz_before,         # avoids an extra GetStatus call
            )
        except Exception as exc:
            logger.warning("CELL_TRANSIT: move failed (%s); skipping cell", exc)
            self._path_index += 1
            return BrainState.CELL_TRANSIT

        # Wait for mechanical movement + adaptive settle.
        # travel_s: time for the axes to physically reach the target.
        # adaptive_settle: extra time for vibrations / zoom ringing to damp out.
        travel_s = self._camera.estimate_travel_time(ptz_before, target)
        await asyncio.sleep(travel_s + adaptive_settle)

        # ── Position verification, drift learning & correction ────────────
        # Read back actual PTZ.  If pan/tilt drifted more than the threshold:
        #   1. Update the learned drift factor (EMA) from this observation
        #   2. Issue a pan/tilt-ONLY correction — zoom kept at actual.zoom so
        #      we don't re-trigger zoom-induced drift in a correction loop.
        # Threshold: 0.01 ONVIF ≈ 0.9° on CPPlus (was 0.03, too loose for
        # zoom-induced drift that's typically 0.01–0.02 units).
        try:
            actual = await self._safe_get_ptz()
            if actual is not None:
                raw_err_pan  = actual.pan  - cell.center_pan   # signed residual
                raw_err_tilt = actual.tilt - cell.center_tilt
                _THRESH = 0.01
                if abs(raw_err_pan) > _THRESH or abs(raw_err_tilt) > _THRESH:
                    # Learn from the pan error that *wasn't* pre-compensated away.
                    # new_sample = (residual + what we already compensated) / zoom_delta
                    # gives the true per-unit drift of this camera.
                    if zoom_mag > 0.08:
                        new_sample = (raw_err_pan + drift_comp_pan) / zoom_delta
                        # EMA: weight old knowledge heavily until we have many samples
                        alpha = min(0.4, 1.0 / (self._zoom_pan_drift_samples + 1))
                        self._zoom_pan_drift = (
                            (1 - alpha) * self._zoom_pan_drift + alpha * new_sample
                        )
                        self._zoom_pan_drift_samples += 1
                        logger.debug(
                            "CELL_TRANSIT: zoom drift sample %.4f → learned=%.4f (n=%d)",
                            new_sample, self._zoom_pan_drift,
                            self._zoom_pan_drift_samples,
                        )
                    logger.debug(
                        "CELL_TRANSIT: residual pan=%.4f tilt=%.4f — correcting pan/tilt only",
                        raw_err_pan, raw_err_tilt,
                    )
                    # Correction is pan/tilt-only; zoom stays at actual.zoom
                    # to prevent re-triggering zoom-induced drift.
                    await self._camera.absolute_move(
                        cell.center_pan, cell.center_tilt, actual.zoom,
                        speed=max(0.15, self._move_speed * 0.4),
                    )
                    await asyncio.sleep(self._cell_settle + 0.15)
        except Exception:
            pass  # non-fatal; proceed with scan regardless

        self._cell_dwell_start = time.time()
        logger.debug(
            "CELL_TRANSIT → cell %d/%d  pan=%.3f tilt=%.3f zoom=%.3f  travel=%.2fs",
            self._path_index + 1, len(self._plan.cell_order),
            cell.center_pan, cell.center_tilt, cell.required_zoom, travel_s,
        )
        return BrainState.CELL_RECOGNIZE

    # ── State: CELL_RECOGNIZE ─────────────────────────────────────────────────

    async def _state_cell_recognize(self) -> BrainState:
        cfg = self._cfg
        assert cfg is not None and self._scan_map is not None and self._plan is not None

        cell_idx = self._plan.cell_order[self._path_index]
        cell = self._scan_map.cells[cell_idx]

        # Collect frames for temporal averaging (count from config)
        all_fwes: list[FaceWithEmbedding] = []
        last_persons: list[PersonDetection] = []
        for fidx in range(self._recognize_frames):
            fd = await self._grab_frame()
            if fd is None:
                continue
            _, ts, frame = fd
            self._last_frame = frame
            # Refresh PTZ each frame so pixel→PTZ conversions stay accurate
            self._last_frame_ptz = await self._safe_get_ptz()

            result = await self._run_pipeline(frame, cfg.roi_rect)

            # Track person peak across the session
            if result.persons:
                self._person_count_peak = max(self._person_count_peak, len(result.persons))
                last_persons = result.persons  # keep most-recent list for body tracking

            # P1 edge-motion check (use the fresh person list)
            if result.persons and self._detect_motion_at_edge(frame, result.persons):
                self._motion_interrupt = True

            all_fwes.extend(result.faces_with_embeddings)
            if fidx < self._recognize_frames - 1:
                await asyncio.sleep(_RECOGNIZE_FRAME_GAP_S)

        # Person body tracking: follow persons whose face isn't visible yet.
        # Runs before the early-return so a person with a turned/hidden face
        # is not silently skipped — the camera chases them and waits for the
        # face to come into view, then feeds the detected face into the normal
        # dedup → identify pipeline below.
        persons_no_face = self._persons_without_face(last_persons, all_fwes)
        if persons_no_face:
            logger.info(
                "CELL_RECOGNIZE: %d person(s) with no visible face → body tracking",
                len(persons_no_face),
            )
            tracked_fwes = await self._track_person_for_face(
                persons_no_face, cfg, cell.cell_id,
            )
            all_fwes.extend(tracked_fwes)

        if not all_fwes:
            logger.debug("CELL_RECOGNIZE: no faces in cell %d", self._path_index)
            return BrainState.CELL_COMPLETE

        # Deduplicate faces by cosine similarity (threshold 0.6 = same person)
        unique_fwes = self._deduplicate_faces(all_fwes)

        # Identify each face
        unrecognized: list[FaceWithEmbedding] = []
        recognized_count = 0
        ts_now = time.time()

        for fwe in unique_fwes:
            id_result = await self._identify(fwe, cfg)
            if id_result.person_id:
                recognized_count += 1
                self._cycle_seen.add(id_result.person_id)
                self._update_duration(id_result.person_id, ts_now, similarity=id_result.similarity, chip=fwe.face_chip)
                self._update_cell_unrecognized(cell, recognized=True)
                await self._publish_detection(fwe, id_result, cfg, cell.cell_id)
                if cfg.faculty_id and id_result.person_id == cfg.faculty_id:
                    self._faculty_last_seen = ts_now
                    self._faculty_detected = True
            else:
                unrecognized.append(fwe)

        # Update per-cell recognition stats
        snap = self._cell_stats.get(cell.cell_id)
        if snap is None:
            snap = ScanCellSnapshot(
                cell_id=cell.cell_id, center_pan=cell.center_pan,
                center_tilt=cell.center_tilt, zoom_used=cell.required_zoom,
                expected_faces=cell.expected_faces,
            )
            self._cell_stats[cell.cell_id] = snap
        snap.recognition_count += recognized_count
        snap.unrecognized_count += len(unrecognized)

        self._unrec_for_hunt = unrecognized

        if unrecognized:
            logger.debug(
                "CELL_RECOGNIZE: %d unrecognized → FACE_HUNT",
                len(unrecognized),
            )
            return BrainState.FACE_HUNT

        return BrainState.CELL_COMPLETE

    # ── Pre-recognition zoom pass ─────────────────────────────────────────────

    async def _pre_zoom_faces(
        self,
        fwes: list[FaceWithEmbedding],
        restore_ptz: PTZPosition,
        cfg: "_SessionConfig",
        location: str,
    ) -> list[FaceWithEmbedding]:
        """
        For every detected face, zoom in + pan/tilt to centre it, capture a
        dedicated high-resolution frame, then hand the zoomed crop to FAISS.

        This runs for ALL faces unconditionally — not just small ones — because
        even a face that appears large at preset zoom gets a sharper, better-
        centred crop after the camera physically aligns to it.  This gives
        ArcFace the maximum possible pixel count and removes off-axis distortion
        before the first identification attempt.

        Processing order: largest IOD first (best chance within time budget).
        If the time budget (_PRE_ZOOM_BUDGET_S) is exhausted the remaining
        faces keep their original crops so the pipeline still runs on them.

        After all zooms the camera is restored to *restore_ptz* so the rest of
        the preset logic (FACE_HUNT, advance to next preset) behaves normally.
        """
        # Filter out faces with no valid IOD (landmark detection failed)
        valid = [f for f in fwes if f.face.inter_ocular_px > 1.0]
        invalid = [f for f in fwes if f.face.inter_ocular_px <= 1.0]

        if not valid:
            return fwes

        self._overlay_action = "PAN/TILT ZOOM"

        logger.info(
            "PRE_ZOOM: zooming to %d face(s) at preset '%s' for max-pixel recognition",
            len(valid), location,
        )

        budget_end = time.time() + _PRE_ZOOM_BUDGET_S
        results: list[FaceWithEmbedding] = list(invalid)

        # Capture the reference frame and PTZ ONCE before any camera moves.
        # All face bboxes in `fwes` originate from this frame at this PTZ.
        # After zooming into the first face, self._last_frame and the live
        # camera position no longer match the original bboxes — so we must
        # pin the reference here and reuse it for every face target calculation.
        _ref_frame = self._last_frame
        _ref_ptz   = self._last_frame_ptz  # PTZ at frame-capture time (not live)
        if _ref_ptz is None:
            # Fallback: read live PTZ once (best effort before any moves)
            _ref_ptz = await self._safe_get_ptz()

        # Largest face first — gets the zoomed frame with most pixels first
        for fwe in sorted(valid, key=lambda f: f.face.inter_ocular_px, reverse=True):
            if time.time() >= budget_end or self._stop_event.is_set():
                results.append(fwe)   # keep original crop if out of budget
                continue

            frame_ref = _ref_frame
            if frame_ref is None:
                results.append(fwe)
                continue

            frame_w, frame_h = frame_ref.shape[1], frame_ref.shape[0]
            ptz_now = _ref_ptz  # always the PTZ that matches the original frame
            if ptz_now is None:
                results.append(fwe)
                continue

            iod = fwe.face.inter_ocular_px

            # ── Zoom: scale so this face reaches _FACE_HUNT_TARGET_IOD ────────
            target_zoom = self._face_hunt_zoom(frame_w, ptz_now.zoom, iod)

            # ── Pan + Tilt: centre camera exactly on the face ─────────────────
            face_cx = (fwe.face.bbox[0] + fwe.face.bbox[2]) / 2.0
            face_cy = (fwe.face.bbox[1] + fwe.face.bbox[3]) / 2.0
            target_pan, target_tilt = self._camera.pixel_to_ptz(
                face_cx, face_cy, frame_w, frame_h, ptz_now
            )

            logger.debug(
                "PRE_ZOOM: face centre (%.0f,%.0f) frame %dx%d  "
                "ref_ptz=(%.3f,%.3f,%.3f) → target=(%.3f,%.3f,%.3f)",
                face_cx, face_cy, frame_w, frame_h,
                ptz_now.pan, ptz_now.tilt, ptz_now.zoom,
                target_pan, target_tilt, target_zoom,
            )

            try:
                if cfg.camera_type in ("PTZ_ZOOM", "BULLET_ZOOM", "BULLET"):
                    # Zoom only — keep current pan/tilt
                    _cur = await self._safe_get_ptz()
                    _pan = _cur.pan if _cur else ptz_now.pan
                    _tilt = _cur.tilt if _cur else ptz_now.tilt
                    await self._camera.absolute_move(
                        _pan, _tilt, target_zoom,
                        speed=_PRECISION_MOVE_SPEED,
                        zoom_speed=_PRECISION_MOVE_SPEED,
                    )
                else:
                    await self._camera.absolute_move(
                        target_pan, target_tilt, target_zoom,
                        speed=_PRECISION_MOVE_SPEED,
                        zoom_speed=_PRECISION_MOVE_SPEED,
                    )
                # Invalidate cache: zoom changed, stale K constant would cause
                # wrong pixel_to_ptz results on subsequent face centering.
                self._invalidate_ptz_cache()
                await asyncio.sleep(self._cell_settle)
            except Exception as exc:
                logger.warning("PRE_ZOOM: PTZ move failed (%s) — using original crop", exc)
                results.append(fwe)
                continue

            fd = await self._grab_frame()
            if fd is None:
                results.append(fwe)
                continue

            _, _, zoomed_frame = fd
            self._last_frame     = zoomed_frame
            self._last_frame_ptz = await self._safe_get_ptz()

            result = await self._run_pipeline(zoomed_frame, cfg.roi_rect)
            if not result.faces_with_embeddings:
                logger.debug(
                    "PRE_ZOOM: no face in zoomed frame at preset '%s' — keeping original",
                    location,
                )
                results.append(fwe)
                continue

            # Pick face closest to frame centre — that's the one we centred on
            cx = zoomed_frame.shape[1] / 2.0
            cy = zoomed_frame.shape[0] / 2.0
            best = min(
                result.faces_with_embeddings,
                key=lambda f2: math.hypot(
                    (f2.face.bbox[0] + f2.face.bbox[2]) / 2 - cx,
                    (f2.face.bbox[1] + f2.face.bbox[3]) / 2 - cy,
                ),
            )
            logger.debug(
                "PRE_ZOOM: face IOD %.0f → %.0fpx  pan=%.3f tilt=%.3f zoom=%.3f  preset='%s'",
                iod, best.face.inter_ocular_px,
                target_pan, target_tilt, target_zoom, location,
            )
            results.append(best)

        # ── Restore camera to preset position ────────────────────────────────
        self._overlay_action = "RETURNING"
        try:
            if cfg.camera_type in ("PTZ_ZOOM", "BULLET_ZOOM", "BULLET"):
                _cur = await self._safe_get_ptz()
                _pan = _cur.pan if _cur else restore_ptz.pan
                _tilt = _cur.tilt if _cur else restore_ptz.tilt
                await self._camera.absolute_move(
                    _pan, _tilt, restore_ptz.zoom,
                    speed=self._move_speed,
                    zoom_speed=_PRECISION_MOVE_SPEED,
                )
            else:
                await self._camera.absolute_move(
                    restore_ptz.pan, restore_ptz.tilt, restore_ptz.zoom,
                    speed=self._move_speed,
                    zoom_speed=_PRECISION_MOVE_SPEED,
                )
            await asyncio.sleep(self._cell_settle)
        except Exception as exc:
            logger.warning("PRE_ZOOM: restore to preset '%s' failed (%s)", location, exc)
        self._overlay_action = ""

        return results

    # ── State: FACE_HUNT ──────────────────────────────────────────────────────

    async def _state_face_hunt(self) -> BrainState:
        """
        Zoom in on each unrecognized face and retry identification.

        Budget per cell: _FACE_HUNT_MAX_ATTEMPTS per face, _FACE_HUNT_BUDGET_S total.
        """
        cfg = self._cfg
        assert cfg is not None

        # Preset mode: use current preset position; legacy mode: use cell
        if self._presets and self._preset_index < len(self._presets):
            _p = self._presets[self._preset_index]
            _restore_ptz = PTZPosition(float(_p["pan"]), float(_p["tilt"]), float(_p["zoom"]))
            _location_id = _p.get("name", str(self._preset_index))
        elif self._plan is not None and self._scan_map is not None:
            cell_idx = self._plan.cell_order[self._path_index]
            cell = self._scan_map.cells[cell_idx]
            _restore_ptz = PTZPosition(cell.center_pan, cell.center_tilt, cell.required_zoom)
            _location_id = cell.cell_id
        else:
            self._unrec_for_hunt.clear()
            return BrainState.PRESET_COMPLETE

        self._overlay_state  = "FACE_HUNT"
        self._overlay_preset = _location_id
        self._overlay_action = "ZOOMING"

        budget_end = time.time() + self._face_hunt_budget
        still_unrec: list[FaceWithEmbedding] = []
        hunt_total = 0
        hunt_ok    = 0

        # Grab a fresh frame at the current (preset/restored) position so that
        # face bbox coordinates and IOD values match ptz_now.  _pre_zoom_faces()
        # stores a ZOOMED frame in self._last_frame before restoring the camera;
        # using those stale coords with the preset PTZ causes pixel_to_ptz to
        # compute a wrong angular offset and _face_hunt_zoom to compute the wrong
        # zoom level (face IOD from zoomed frame >> target IOD → tries to zoom out).
        fresh_fd = await self._grab_frame()
        if fresh_fd is not None:
            _, _, _fresh_frame = fresh_fd
            self._last_frame     = _fresh_frame
            self._last_frame_ptz = await self._safe_get_ptz()
            _fresh_result = await self._run_pipeline(_fresh_frame, cfg.roi_rect)
            if _fresh_result.faces_with_embeddings:
                targets = sorted(
                    _fresh_result.faces_with_embeddings,
                    key=lambda f: f.face.inter_ocular_px,
                    reverse=True,
                )
                logger.info(
                    "FACE_HUNT: refreshed %d face target(s) from current preset frame",
                    len(targets),
                )
            else:
                # No faces in fresh frame — fall back to stale coords from _unrec_for_hunt
                logger.debug("FACE_HUNT: no faces in fresh frame, using stale targets")
                targets = sorted(
                    self._unrec_for_hunt,
                    key=lambda fwe: fwe.face.inter_ocular_px,
                    reverse=True,
                )
        else:
            # Frame grab failed — fall back to stale coords
            targets = sorted(
                self._unrec_for_hunt,
                key=lambda fwe: fwe.face.inter_ocular_px,
                reverse=True,
            )

        for fwe in targets:
            if time.time() >= budget_end:
                still_unrec.append(fwe)
                continue

            iod = fwe.face.inter_ocular_px
            if iod < 1.0:
                still_unrec.append(fwe)
                continue

            hunt_total += 1

            # Target PTZ: zoom + center on the face using reference PTZ that
            # matches the frame the face bbox was detected in.  Using a live
            # _safe_get_ptz() here would be stale after the pipeline run above.
            frame_w = (self._last_frame.shape[1] if self._last_frame is not None else 1920)
            h = self._last_frame.shape[0] if self._last_frame is not None else 1080
            ptz_now = self._last_frame_ptz or await self._safe_get_ptz()
            if ptz_now is None:
                still_unrec.append(fwe)
                continue
            target_zoom = self._face_hunt_zoom(frame_w, ptz_now.zoom, iod)
            face_cx = (fwe.face.bbox[0] + fwe.face.bbox[2]) / 2.0
            face_cy = (fwe.face.bbox[1] + fwe.face.bbox[3]) / 2.0
            target_pan, target_tilt = self._camera.pixel_to_ptz(
                face_cx, face_cy, frame_w, h, ptz_now
            )

            logger.debug(
                "FACE_HUNT: face centre (%.0f,%.0f) frame %dx%d  "
                "ref_ptz=(%.3f,%.3f,%.3f) → target=(%.3f,%.3f,%.3f)",
                face_cx, face_cy, frame_w, h,
                ptz_now.pan, ptz_now.tilt, ptz_now.zoom,
                target_pan, target_tilt, target_zoom,
            )

            recognized = False
            ts = time.time()

            # ── Phase 1: initial absolute move — zoom in + rough centre ───────
            # Zoom to 80% of target so the face is large but we have headroom
            # for the P-controller to fine-tune without hitting the zoom ceiling.
            try:
                if cfg.camera_type in ("PTZ_ZOOM", "BULLET_ZOOM"):
                    # Zoom only — keep current pan/tilt
                    _cur = await self._safe_get_ptz()
                    _pan = _cur.pan if _cur else ptz_now.pan
                    _tilt = _cur.tilt if _cur else ptz_now.tilt
                    await self._camera.absolute_move(
                        _pan, _tilt, target_zoom,
                        speed=_PRECISION_MOVE_SPEED,
                        zoom_speed=_PRECISION_MOVE_SPEED,
                    )
                elif cfg.camera_type == "BULLET":
                    # No movement at all — just wait
                    await asyncio.sleep(self._cell_settle)
                else:
                    await self._camera.absolute_move(
                        target_pan, target_tilt, target_zoom,
                        speed=_PRECISION_MOVE_SPEED,
                        zoom_speed=_PRECISION_MOVE_SPEED,
                    )
                await asyncio.sleep(self._cell_settle)
            except Exception as exc:
                logger.warning("FACE_HUNT: initial move failed (%s) — skipping face", exc)
                still_unrec.append(fwe)
                continue

            # ── Phase 2: P-controller — keep face centroid at frame centre ────
            # Mirrors _track_person_for_face but tracks FACE centroid and also
            # nudges zoom incrementally when IOD is still too small.
            # Loop runs until: face recognised, budget exhausted, or face lost.
            _stable_frames = 0   # consecutive frames where face is centred
            _STABLE_NEEDED  = 2  # frames needed before taking the recognition shot
            # Latency compensation: track consecutive face positions to predict where
            # the face will be when the absolute_move command reaches the camera.
            # Reset on zoom change (FOV shift invalidates the pixel-velocity estimate).
            _fh_prev_face_cx: float | None = None
            _fh_prev_face_cy: float | None = None
            _fh_prev_ts:      float | None = None
            _v_cx:            float        = 0.0
            _v_cy:            float        = 0.0
            # Background recognition task — runs full pipeline + FAISS while the PTZ
            # correction loop keeps running.  None = no task pending.
            _recog_task: asyncio.Task | None = None
            # Cross-attempt embedding bank: accumulates (embedding, iod_weight) tuples
            # across all failed recognition attempts for this face target so that FAISS
            # searches a richer averaged embedding on each retry.  Cleared on success.
            _emb_bank: list[tuple[np.ndarray, float]] = []
            _recog_attempt_n: int = 0   # incremented each time a recognition task launches
            # Lucas-Kanade optical flow state — allows face-centroid updates every
            # frame without running SCRFD each time (~1 ms vs ~15 ms per call).
            _lk_prev_gray: np.ndarray | None = None
            _lk_prev_pts:  np.ndarray | None = None   # tracked feature points
            _lk_face_cx:   float | None      = None   # last known face centroid X
            _lk_face_cy:   float | None      = None   # last known face centroid Y
            _scrfd_timer:  int               = 0      # countdown; 0 → run SCRFD
            _lk_miss:      int               = 0      # consecutive SCRFD misses
            cur_iod:       float             = 0.0    # persists across LK frames

            try:
                while time.time() < budget_end and not self._stop_event.is_set():
                    t_loop = time.monotonic()  # wall-clock start of this iteration

                    fd = await self._grab_frame(timeout=_PERSON_TRACK_POLL_S + 0.5)
                    if fd is None:
                        break

                    _, ts, frame = fd
                    self._last_frame     = frame
                    # fresh=True: the 300 ms PTZ cache is long enough that a
                    # rapidly-moving camera drifts several degrees from the
                    # cached position, and pixel_to_ptz computed against that
                    # stale reference sends the next absolute_move to a point
                    # behind the actual target. Launch the ONVIF fetch NOW so
                    # it runs concurrently with SCRFD/LK inference — total
                    # latency becomes max(ONVIF_RTT, inference) instead of the
                    # sum, shaving 20-30 ms off every FACE_HUNT iteration.
                    _ptz_task = asyncio.create_task(self._safe_get_ptz(fresh=True))
                    fw, fh = frame.shape[1], frame.shape[0]
                    frame_cx, frame_cy = fw / 2.0, fh / 2.0

                    # ── Fast face tracking: LK flow (CPU ~1 ms) → SCRFD fallback ────
                    # On frames where _scrfd_timer > 0 we try Lucas-Kanade optical
                    # flow first — it updates the face centroid in ~1 ms without GPU
                    # work, giving up to 10 Hz PTZ corrections at 10 fps RTSP.
                    # SCRFD runs every _LK_SCRFD_INTERVAL frames (or immediately when
                    # LK degrades) to re-seed the feature points.
                    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    face_cx: float
                    face_cy: float
                    _lk_ok = False

                    if (
                        _scrfd_timer > 0
                        and _lk_prev_gray is not None
                        and _lk_prev_pts is not None
                        and _lk_face_cx is not None
                        and _lk_face_cy is not None
                    ):
                        _pts_next, _status, _ = cv2.calcOpticalFlowPyrLK(
                            _lk_prev_gray, curr_gray, _lk_prev_pts, None,
                            winSize=_LK_WIN_SIZE,
                            maxLevel=_LK_MAX_LEVEL,
                            criteria=_LK_CRITERIA,
                        )
                        _good = (_status.ravel() == 1) if _status is not None else []
                        if _good.sum() >= _LK_MIN_POINTS:  # type: ignore[union-attr]
                            _dx = float(np.median(
                                _pts_next[_good, 0, 0] - _lk_prev_pts[_good, 0, 0]  # type: ignore[index]
                            ))
                            _dy = float(np.median(
                                _pts_next[_good, 0, 1] - _lk_prev_pts[_good, 0, 1]  # type: ignore[index]
                            ))
                            _lk_face_cx += _dx
                            _lk_face_cy += _dy
                            _lk_prev_pts  = _pts_next[_good].reshape(-1, 1, 2)  # type: ignore[index]
                            _lk_prev_gray = curr_gray
                            _scrfd_timer -= 1
                            face_cx = _lk_face_cx
                            face_cy = _lk_face_cy
                            _lk_ok  = True

                    if not _lk_ok:
                        # LK failed or timer expired — run SCRFD to re-anchor
                        raw_faces = await self._run_face_only(frame)
                        _scrfd_timer = _LK_SCRFD_INTERVAL

                        if not raw_faces:
                            _lk_miss += 1
                            if _lk_face_cx is not None and _lk_miss <= 3:
                                # Coasting: use last known centroid for a few frames
                                face_cx = _lk_face_cx
                                face_cy = _lk_face_cy  # type: ignore[assignment]
                            else:
                                _ptz_task.cancel()
                                await self._camera.stop()
                                _stable_frames = 0
                                logger.debug(
                                    "FACE_HUNT: face lost  lk_miss=%d", _lk_miss
                                )
                                break
                        else:
                            _lk_miss = 0
                            # Select face closest to frame centre
                            _cf = min(
                                raw_faces,
                                key=lambda f: math.hypot(
                                    (f.bbox[0] + f.bbox[2]) / 2 - frame_cx,
                                    (f.bbox[1] + f.bbox[3]) / 2 - frame_cy,
                                ),
                            )
                            face_cx = (_cf.bbox[0] + _cf.bbox[2]) / 2.0
                            face_cy = (_cf.bbox[1] + _cf.bbox[3]) / 2.0
                            cur_iod = _cf.inter_ocular_px
                            # Re-seed LK from face bounding-box ROI
                            _lk_x1 = max(0,  int(_cf.bbox[0]))
                            _lk_y1 = max(0,  int(_cf.bbox[1]))
                            _lk_x2 = min(fw, int(_cf.bbox[2]))
                            _lk_y2 = min(fh, int(_cf.bbox[3]))
                            if _lk_x2 > _lk_x1 and _lk_y2 > _lk_y1:
                                _roi_g = curr_gray[_lk_y1:_lk_y2, _lk_x1:_lk_x2]
                                _pts = cv2.goodFeaturesToTrack(
                                    _roi_g, maxCorners=30,
                                    qualityLevel=0.3, minDistance=5, blockSize=7,
                                )
                                if _pts is not None and len(_pts) >= _LK_MIN_POINTS:
                                    _pts[:, 0, 0] += _lk_x1
                                    _pts[:, 0, 1] += _lk_y1
                                    _lk_prev_pts = _pts
                                else:
                                    _lk_prev_pts = None
                            else:
                                _lk_prev_pts = None
                            _lk_face_cx   = face_cx
                            _lk_face_cy   = face_cy
                            _lk_prev_gray = curr_gray
                            # SCRFD re-anchor produces a position JUMP vs. the
                            # last (LK-drifted) centroid. Feeding that jump into
                            # the velocity estimator yields a huge spurious
                            # _v_cy which, multiplied by the 200 ms prediction
                            # horizon, aims the camera at the ceiling. Reset
                            # the velocity history so the next iteration starts
                            # fresh from this re-anchored position.
                            _fh_prev_face_cx = None
                            _fh_prev_face_cy = None
                            _fh_prev_ts = None
                            _v_cx = _v_cy = 0.0

                    # Normalised centroid error (for logging only — not used for control)
                    err_x =  (face_cx - frame_cx) / (fw / 2.0)
                    err_y = -(face_cy - frame_cy) / (fh / 2.0)

                    # ── Phase 4: Latency compensation ──────────────────────────────────
                    # Estimate face velocity from consecutive frame positions, then
                    # predict where the face will be once absolute_move completes.
                    # JUMP REJECTION: if the centroid displaced more than
                    # _VEL_JUMP_PX in one frame (≈ 10 Hz), the sample is almost
                    # certainly a detection error or an LK drift correction, not
                    # real motion. Treat it as zero-velocity — anything else
                    # will slam the PTZ toward the jump direction (the "camera
                    # goes to the ceiling" failure mode).
                    _VEL_JUMP_PX = 90.0
                    _VEL_EMA_ALPHA = 0.5
                    if _fh_prev_face_cx is not None and _fh_prev_ts is not None:
                        _dt = ts - _fh_prev_ts
                        _disp_x = face_cx - _fh_prev_face_cx
                        _disp_y = face_cy - _fh_prev_face_cy
                        if _dt > 0.005 and abs(_disp_x) < _VEL_JUMP_PX and abs(_disp_y) < _VEL_JUMP_PX:
                            _v_cx_raw = _disp_x / _dt
                            _v_cy_raw = _disp_y / _dt
                        else:
                            _v_cx_raw = _v_cy_raw = 0.0
                    else:
                        _v_cx_raw = _v_cy_raw = 0.0
                    # EMA smoothing damps single-frame SCRFD bbox jitter that
                    # the jump filter lets through.
                    _v_cx = _VEL_EMA_ALPHA * _v_cx_raw + (1.0 - _VEL_EMA_ALPHA) * _v_cx
                    _v_cy = _VEL_EMA_ALPHA * _v_cy_raw + (1.0 - _VEL_EMA_ALPHA) * _v_cy
                    _fh_prev_face_cx = face_cx
                    _fh_prev_face_cy = face_cy
                    _fh_prev_ts      = ts

                    # Project forward by inference + ONVIF RTT + settle. Cap
                    # prediction at 50 px: larger horizons amplify any residual
                    # velocity noise into tilt overshoot (the "ceiling-drift"
                    # failure). 50 px covers a walking subject (≤ 300 px/s) over
                    # a 170 ms horizon — enough to close the lag gap without
                    # risking runaway prediction on a noisy velocity sample.
                    _inference_s = time.monotonic() - t_loop
                    _CMD_RTT_S   = 0.12
                    _latency_s   = max(0.08, _inference_s + _CMD_RTT_S + _FH_SETTLE_S)
                    _MAX_PRED_PX = 50.0
                    face_cx_pred = face_cx + max(-_MAX_PRED_PX,
                                                 min(_MAX_PRED_PX, _v_cx * _latency_s))
                    face_cy_pred = face_cy + max(-_MAX_PRED_PX,
                                                 min(_MAX_PRED_PX, _v_cy * _latency_s))

                    # Await the PTZ read launched in parallel with face inference.
                    # By this point, SCRFD / LK has already run, so the ONVIF
                    # round-trip has been hidden behind GPU work.
                    try:
                        ptz_now = await _ptz_task
                    except Exception:
                        ptz_now = await self._safe_get_ptz(fresh=False)
                    self._last_frame_ptz = ptz_now

                    # Incremental zoom nudge — only when face is significantly below target
                    # AND pan/tilt has been stable for at least _FACE_HUNT_ZOOM_STABLE_FRAMES
                    # iterations.  A zoom change shifts the FOV which invalidates the K
                    # constants used by pixel_to_ptz; issuing it while the P-controller
                    # is still chasing the face causes a large tilt overshoot on the
                    # next correction.  Gating behind stability guarantees zoom and
                    # pan/tilt never fight each other.
                    if (
                        cur_iod > 1.0
                        and cur_iod < _FACE_HUNT_TARGET_IOD * 0.70
                        and _stable_frames >= _FACE_HUNT_ZOOM_STABLE_FRAMES
                        and abs(err_x) < _TRACK_DEAD_ZONE
                        and abs(err_y) < _TRACK_DEAD_ZONE
                    ):
                        new_zoom = self._face_hunt_zoom(fw, ptz_now.zoom, cur_iod)
                        if new_zoom > ptz_now.zoom + 0.03:  # only nudge when change is meaningful
                            try:
                                await self._camera.absolute_move(
                                    ptz_now.pan, ptz_now.tilt, new_zoom,
                                    speed=_PRECISION_MOVE_SPEED,
                                    zoom_speed=_PRECISION_MOVE_SPEED,
                                )
                            except Exception:
                                pass
                            # Invalidate PTZ cache so the next iteration reads the new
                            # zoom from the camera — stale zoom causes pixel_to_ptz to
                            # use the wrong K constant and overcorrect pan/tilt.
                            self._invalidate_ptz_cache()
                            _stable_frames = 0
                            # Zoom changed FOV — reset velocity estimator AND LK state so
                            # stale pixel-velocity / keypoints from before the zoom don't
                            # mislead prediction or flow tracking.
                            _fh_prev_face_cx = _fh_prev_face_cy = _fh_prev_ts = None
                            _v_cx = _v_cy = 0.0
                            _lk_prev_gray = _lk_prev_pts = None
                            _lk_face_cx = _lk_face_cy = None
                            _scrfd_timer = 0
                            await asyncio.sleep(self._cell_settle)
                            continue

                    # ── Phase 2: Closed-loop absolute_move ────────────────────────────
                    # Convert the latency-compensated face pixel to an exact PTZ angle
                    # target.  Fixed _FH_SETTLE_S (0.15 s) keeps the loop at ≥ 5 Hz —
                    # the camera accepts a new command before it fully settles, so
                    # each correction redirects the in-flight move toward the new target.
                    _skip_pantilt = cfg.camera_type in ("PTZ_ZOOM", "BULLET_ZOOM", "BULLET")
                    if not _skip_pantilt:
                        tgt_pan, tgt_tilt = self._camera.pixel_to_ptz(
                            face_cx_pred, face_cy_pred, fw, fh, ptz_now,
                        )
                        # Proportional control: command only _FACE_HUNT_P_GAIN of
                        # the way to target per iteration. Produces smooth
                        # convergence instead of snapping to target on every
                        # frame (which overshoots under latency and looks
                        # jittery). At P=0.6 and 8 Hz loop, the camera reaches
                        # target within ~3 iterations (< 400 ms) — imperceptible
                        # lag but visibly cinematic motion.
                        tgt_pan  = ptz_now.pan  + _FACE_HUNT_P_GAIN * (tgt_pan  - ptz_now.pan)
                        tgt_tilt = ptz_now.tilt + _FACE_HUNT_P_GAIN * (tgt_tilt - ptz_now.tilt)
                        _dist = math.hypot(tgt_pan - ptz_now.pan, tgt_tilt - ptz_now.tilt)

                        if _dist < _FACE_TRACK_DEAD_ZONE_ONVIF:
                            # Face already centred — count stable frames, no move needed
                            _stable_frames += 1
                        else:
                            try:
                                await self._camera.absolute_move(
                                    tgt_pan, tgt_tilt, ptz_now.zoom,
                                    speed=_PRECISION_MOVE_SPEED,
                                )
                                # Invalidate cache so the next ptz_now read reflects
                                # the new commanded position, not the stale cached one.
                                self._invalidate_ptz_cache()
                                # During the settle window the recognition task (if running)
                                # can complete on the GPU.  Check it immediately after settle
                                # so we don't wait a full extra iteration to apply the result.
                                await asyncio.sleep(_FH_SETTLE_S)
                            except Exception as exc:
                                logger.warning(
                                    "FACE_HUNT: absolute_move failed (%s)", exc,
                                )
                                break
                            _stable_frames = 0
                            # Check if recognition finished during the settle sleep —
                            # if yes, we still need a stable frame to confirm before
                            # accepting the result, so we just let _stable_frames
                            # rebuild naturally.  The task result will be picked up on
                            # the next stable iteration via the block below.
                            # (do NOT continue here — fall through to the recognition
                            #  block so .done() is checked this very iteration)
                            # continue   ← removed: let fall-through check .done()
                    else:
                        # Zoom-only or fixed-mount — no pan/tilt correction possible
                        _stable_frames += 1

                    # ── Background recognition (non-blocking) ─────────────────────────
                    # When stable, launch the full pipeline + FAISS as a background task.
                    # The PTZ correction loop (above) keeps running while recognition runs
                    # on the GPU in parallel — corrections during the 0.15 s settle window
                    # allow recognition's _run_pipeline (~50 ms) to complete without
                    # blocking a single absolute_move.
                    if _stable_frames >= _STABLE_NEEDED and _recog_task is None:
                        _recog_attempt_n += 1
                        _recog_task = asyncio.ensure_future(
                            self._face_hunt_recognize(
                                frame, ts, frame_cx, frame_cy,
                                cur_iod, err_x, err_y,
                                budget_end, cfg, _location_id,
                                emb_bank=_emb_bank,
                                attempt_n=_recog_attempt_n,
                            )
                        )
                        logger.debug(
                            "FACE_HUNT: recognition task launched  "
                            "attempt=%d  stable=%d  iod=%.0fpx  bank=%d  err=(%.3f,%.3f)",
                            _recog_attempt_n, _stable_frames, cur_iod,
                            len(_emb_bank), err_x, err_y,
                        )

                    # Non-blocking result check — done() returns True immediately
                    # when the task finished; we don't await here so the PTZ loop
                    # keeps running.
                    if _recog_task is not None and _recog_task.done():
                        try:
                            _recog = _recog_task.result()
                        except asyncio.CancelledError:
                            _recog = None
                        except Exception as _rex:
                            logger.warning("FACE_HUNT: recognition task raised: %s", _rex)
                            _recog = None
                        _recog_task = None  # allow retry next stable window

                        if _recog is not None:
                            # Recognition succeeded — clear the bank and exit
                            _emb_bank.clear()
                            if _recog["is_faculty"]:
                                self._faculty_last_seen = ts
                                self._faculty_detected  = True
                            recognized = True
                            hunt_ok   += 1
                            self._annotate_and_store(
                                _recog["frame"], _recog["full_result"],
                                {"0": _recog["person_id"][:20]},
                            )
                            logger.info(
                                "FACE_HUNT: recognised %s  iod=%.0fpx  "
                                "avg_emb_n=%d  bank_used=%d  attempt=%d  "
                                "err_x=%.3f err_y=%.3f",
                                _recog["person_id"], _recog["cur_iod"],
                                _recog["emb_list_len"], _recog["bank_used"],
                                _recog_attempt_n,
                                _recog["err_x"], _recog["err_y"],
                            )
                            break
                        else:
                            # Recognition failed — cool down stable count.
                            # _emb_bank retains embeddings for the next attempt.
                            _stable_frames = 0

            finally:
                # Cancel any in-flight recognition task before stopping the camera.
                # Without this, the task would keep calling _run_pipeline / _identify
                # after Phase 2 exits, potentially racing with the restore move.
                if _recog_task is not None and not _recog_task.done():
                    _recog_task.cancel()
                    try:
                        await _recog_task
                    except (asyncio.CancelledError, Exception):
                        pass
                    _recog_task = None
                try:
                    await self._camera.stop()
                except Exception:
                    pass

            if not recognized:
                still_unrec.append(fwe)

        # Return camera to preset/cell position after hunt
        try:
            if cfg.camera_type in ("PTZ_ZOOM", "BULLET_ZOOM", "BULLET"):
                _cur = await self._safe_get_ptz()
                _pan = _cur.pan if _cur else _restore_ptz.pan
                _tilt = _cur.tilt if _cur else _restore_ptz.tilt
                await self._camera.absolute_move(
                    _pan, _tilt, _restore_ptz.zoom,
                    speed=self._move_speed,
                    zoom_speed=_PRECISION_MOVE_SPEED,
                )
            else:
                await self._camera.absolute_move(
                    _restore_ptz.pan, _restore_ptz.tilt, _restore_ptz.zoom,
                    speed=self._move_speed,
                    zoom_speed=_PRECISION_MOVE_SPEED,
                )
        except Exception:
            pass

        logger.debug(
            "FACE_HUNT: %d recognized, %d still unknown",
            hunt_ok, len(still_unrec),
        )
        # Correct the unknown count: the targets were previously counted as unknown
        # in CELL_RECOGNIZE; subtract those and add back only the still-unknown ones.
        self._unknown_count = max(0, self._unknown_count - len(targets) + len(still_unrec))

        # Update location hunt stats
        snap = self._cell_stats.get(_location_id)
        if snap is not None:
            snap.hunt_count += hunt_total
            snap.hunt_success_count += hunt_ok
            snap.recognition_count += hunt_ok

        self._unrec_for_hunt.clear()
        # In preset mode, return to PRESET_COMPLETE; legacy mode returns to CELL_COMPLETE
        if self._presets:
            return BrainState.PRESET_COMPLETE
        return BrainState.CELL_COMPLETE

    # ── State: CELL_COMPLETE ──────────────────────────────────────────────────

    async def _state_cell_complete(self) -> BrainState:
        assert self._plan is not None

        # Record per-cell stats for the SelfLearner
        if self._scan_map and self._path_index < len(self._plan.cell_order):
            cell_idx = self._plan.cell_order[self._path_index]
            cell = self._scan_map.cells[cell_idx]
            dwell = time.time() - self._cell_dwell_start if self._cell_dwell_start else 0.0
            snap = self._cell_stats.get(cell.cell_id)
            if snap is None:
                snap = ScanCellSnapshot(
                    cell_id=cell.cell_id,
                    center_pan=cell.center_pan,
                    center_tilt=cell.center_tilt,
                    zoom_used=cell.required_zoom,
                    expected_faces=cell.expected_faces,
                )
                self._cell_stats[cell.cell_id] = snap
            snap.total_dwell_s += dwell

        self._path_index += 1

        # B6: if a motion interrupt occurred, trigger an immediate overview scan
        # rather than continuing the stale plan
        if self._motion_interrupt:
            logger.info("CELL_COMPLETE: motion interrupt → immediate OVERVIEW_SCAN")
            return BrainState.OVERVIEW_SCAN

        if self._path_index < len(self._plan.cell_order):
            # Zoom pre-stage: start the zoom motor toward the next cell's zoom
            # level now, during the negligible CELL_COMPLETE compute window.
            # The motor will be partially (or fully) at target by the time
            # CELL_TRANSIT issues its absolute_move, reducing effective zoom
            # travel time and settle overhead.
            await self._pre_stage_next_zoom()

            # Periodic replan to reflect recognition progress
            if self._path_index % 3 == 0 and self._scan_map:
                ptz = await self._safe_get_ptz()
                remaining_cells = [
                    self._scan_map.cells[i]
                    for i in self._plan.cell_order[self._path_index:]
                ]
                if len(remaining_cells) > 2:
                    def _replan():
                        return self._planner.plan_path(
                            remaining_cells, None, ptz, self._camera.limits
                        )
                    new_plan = await asyncio.to_thread(_replan)
                    new_order = [
                        self._scan_map.cells.index(
                            next(c for c in self._scan_map.cells
                                 if c.cell_id == cid)
                        )
                        for cid in new_plan.ordered_cell_ids
                        if any(c.cell_id == cid for c in self._scan_map.cells)
                    ]
                    if new_order:
                        self._plan.cell_order[self._path_index:] = new_order

            return BrainState.CELL_TRANSIT

        return BrainState.CYCLE_COMPLETE

    def _reorder_presets_by_priority(self) -> None:
        """
        Sort ``self._presets`` by descending total historical face count so that
        positions that historically detect the most people are visited first.

        Uses cumulative ``ScanCellSnapshot.recognition_count`` which is updated
        every cycle.  Has no effect on the first cycle (no history yet).
        """
        if not self._cell_stats:
            return  # no history yet — keep original order

        def _total_faces(p: dict) -> int:
            snap = self._cell_stats.get(p.get("name", ""))
            return snap.recognition_count if snap else 0

        self._presets.sort(key=_total_faces, reverse=True)
        logger.debug(
            "Priority sort order: %s",
            [p.get("name", i) for i, p in enumerate(self._presets)],
        )

    # ── State: CYCLE_COMPLETE ─────────────────────────────────────────────────

    async def _state_cycle_complete(self) -> BrainState:
        """
        All presets visited.  Publish cycle summary, reload presets from DB
        (so any additions/removals take effect on the next pass), reset the
        cross-frame tracker and recognition state, then restart the cycle.
        """
        cfg = self._cfg
        assert cfg is not None

        self._cycle_count += 1
        cycle_s = time.time() - self._cycle_start
        self._last_cycle_s = cycle_s
        now = time.time()

        # Recognition rate: recognised / unique persons ever tracked this cycle
        recognized_count  = len(self._cycle_seen)
        # In preset mode we don't have expected_faces from a zone map;
        # use max(recognised, unknown) as a proxy for total persons seen.
        total_seen = recognized_count + self._unknown_count
        self._recognition_rate = (
            recognized_count / total_seen if total_seen > 0 else 0.0
        )

        logger.info(
            "CYCLE_COMPLETE  cycle=%d  presets=%d  recognised=%d  unknown=%d  "
            "rate=%.0f%%  time=%.1fs",
            self._cycle_count,
            len(self._presets),
            recognized_count,
            self._unknown_count,
            self._recognition_rate * 100,
            cycle_s,
        )

        # Increment cycles_seen for every person detected this cycle
        for pid in self._cycle_seen:
            if pid in self._durations:
                self._durations[pid].cycles_seen += 1

        # Mark persons NOT seen this cycle as absent
        for pid, tracker in self._durations.items():
            if pid not in self._cycle_seen and tracker.state == "PRESENT":
                tracker.mark_absent()

        # P2: faculty hunt check
        if (cfg.faculty_id
                and self._faculty_last_seen > 0
                and now - self._faculty_last_seen > self._faculty_hunt_timeout):
            self._faculty_hunt_active = True
            logger.info("P2 FACULTY_HUNT: faculty not seen for >5 min")
        else:
            self._faculty_hunt_active = False

        # Publish attendance summary (ATTENDANCE mode)
        if cfg.mode == OperatingMode.ATTENDANCE:
            await self._publish_attendance_summary(cfg)
            await self._save_unknown_detections(cfg)

        # Reset per-cycle state
        self._cycle_seen.clear()
        self._unknown_count = 0
        self._unknown_chips.clear()
        self._motion_interrupt = False
        self._person_tracker.reset()   # fresh tracker for next cycle
        self._unrec_frame_count.clear()  # reset temporal filter counters
        self._cycle_start = time.time()

        # Reload presets from DB so changes take effect immediately
        if self._preset_loader:
            try:
                fresh = await self._preset_loader()
                if fresh:
                    self._presets = fresh
                    logger.debug("CYCLE_COMPLETE: reloaded %d presets", len(fresh))
            except Exception as exc:
                logger.warning("CYCLE_COMPLETE: preset reload failed (%s)", exc)

        # Bullet cameras don't use scan presets — inject a single fixed-position preset
        if not self._presets and cfg.camera_type in ("BULLET", "BULLET_ZOOM"):
            self._presets = [{"name": "fixed", "pan": 0.0, "tilt": 0.0, "zoom": 0.0, "dwell_s": 20.0}]
            logger.debug("CYCLE_COMPLETE: bullet camera — using synthetic fixed preset (20 s/cycle)")
        elif not self._presets:
            logger.warning("CYCLE_COMPLETE: no presets — entering IDLE")
            return BrainState.IDLE

        # Snapshot current-cycle face counts → previous cycle (for occupancy skip)
        new_prev: dict[str, int] = {name: 0 for name in self._preset_visited}
        new_prev.update(self._preset_cur_faces)
        self._preset_prev_faces = new_prev
        self._preset_cur_faces.clear()

        # Priority ordering: sort presets by total historical face count (desc)
        # so high-traffic positions are visited first in subsequent cycles.
        self._reorder_presets_by_priority()

        # Timetable check — stop session if outside scheduled hours.
        # Skipped for force-started sessions (manually launched from UI/API).
        if not cfg.force_start and not await self._is_within_timetable(cfg):
            logger.info(
                "CYCLE_COMPLETE: camera %s outside timetable hours — stopping session",
                cfg.camera_id,
            )
            return BrainState.STOPPED

        self._preset_index = 0
        return BrainState.PRESET_TRANSIT

    async def _is_within_timetable(self, cfg: _SessionConfig) -> bool:
        """
        Returns True if the camera should be running now (no timetable, or
        current time falls within a scheduled entry for today).  Returns False
        to signal CYCLE_COMPLETE should stop the session.

        Fails open: returns True on any DB/parse error so a transient outage
        does not kill a legitimate session.
        """
        if self._session_factory is None:
            return True
        try:
            from datetime import datetime as _dt

            async with self._session_factory() as db:
                # Set RLS context so tenant-scoped tables are visible
                await db.execute(
                    text("SELECT set_config('app.current_client_id', :cid, TRUE)"),
                    {"cid": cfg.client_id},
                )
                row = await db.execute(
                    text(
                        "SELECT timetable_id FROM cameras "
                        "WHERE camera_id = (:cid)::uuid"
                    ),
                    {"cid": cfg.camera_id},
                )
                cam_row = row.fetchone()
                if cam_row is None or cam_row[0] is None:
                    return True  # no timetable → always run

                timetable_id = str(cam_row[0])

                # Fetch timezone from the timetable row so times are compared
                # in the correct local timezone (not always UTC).
                tz_row = await db.execute(
                    text("SELECT timezone FROM timetables WHERE timetable_id = (:tid)::uuid"),
                    {"tid": timetable_id},
                )
                tz_rec = tz_row.fetchone()
                tz_name = (tz_rec[0] if tz_rec and tz_rec[0] else None) or "UTC"
                try:
                    from zoneinfo import ZoneInfo as _ZI
                    _tz = _ZI(tz_name)
                except Exception:
                    _tz = None

                from datetime import timezone as _timezone
                now = _dt.now(_timezone.utc).astimezone(_tz) if _tz else _dt.now()
                dow = now.weekday()          # 0=Mon … 6=Sun
                current_hhmm = now.strftime("%H:%M")

                entries = await db.execute(
                    text(
                        "SELECT start_time, end_time FROM timetable_entries "
                        "WHERE timetable_id = (:tid)::uuid AND day_of_week = :dow"
                    ),
                    {"tid": timetable_id, "dow": dow},
                )
                for st, et in entries.fetchall():
                    if st <= current_hhmm <= et:
                        return True
                return False   # no entry covers current time
        except Exception as exc:
            logger.warning("_is_within_timetable: check failed (%s) — failing open", exc)
            return True

    # ── Frame management ──────────────────────────────────────────────────────

    async def _start_decoder(self, cfg: _SessionConfig) -> None:
        """Start decoder (creating it from RTSP URL if not injected) and frame reader."""
        if self._decoder is None:
            rtsp_url = await self._camera.get_rtsp_url()
            self._decoder = RTSPDecoder(rtsp_url, fps=25, reconnect_delay=0.5)

        # Drain any stale frames from a prior session
        while not self._frame_q.empty():
            try:
                self._frame_q.get_nowait()
            except asyncio.QueueEmpty:
                break

        self._frame_reader_task = asyncio.create_task(
            self._frame_reader_loop(),
            name=f"frame_reader:{cfg.camera_id}",
        )

    async def _frame_reader_loop(self) -> None:
        """Continuously read from the decoder and keep only the latest two frames."""
        assert self._decoder is not None
        try:
            async for fid, ts, frame in self._decoder.frames():
                if self._stop_event.is_set():
                    break
                # Always update _last_frame so the MJPEG stream relay has a
                # fresh frame regardless of which brain state is active.
                self._last_frame = frame
                if self._frame_q.full():
                    try:
                        self._frame_q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                try:
                    self._frame_q.put_nowait((fid, ts, frame))
                except asyncio.QueueFull:
                    pass
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("frame_reader_loop crashed: %s", exc, exc_info=True)

    async def _grab_frame(
        self, timeout: float = _FRAME_GRAB_TIMEOUT_S
    ) -> tuple[int, float, np.ndarray] | None:
        """
        Return the LATEST available frame (drains any older queued frames).

        The decoder runs at 25 fps but the tracking loop at 5–10 Hz; without
        draining, each grab returns a frame that is one or two producer cycles
        old (40–80 ms). Draining to the newest in-queue item removes that lag
        so every control iteration reacts to real-time scene state.
        """
        try:
            fd = await asyncio.wait_for(self._frame_q.get(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("_grab_frame: timeout after %.1fs", timeout)
            return None
        # Drain any newer frames that arrived while we were blocked.
        while True:
            try:
                fd = self._frame_q.get_nowait()
            except asyncio.QueueEmpty:
                break
        return fd

    # ── GPU inference ──────────────────────────────────────────────────────────

    @staticmethod
    def _roi_to_rect(roi: dict | list | None) -> dict | None:
        """Convert polygon ROI to {x,y,w,h} bounding rect for the AI pipeline."""
        if roi is None:
            return None
        if isinstance(roi, dict):
            return roi
        # Polygon: list of {x,y} dicts or [x,y] lists (normalised 0-1)
        xs, ys = [], []
        for p in roi:
            if isinstance(p, dict):
                xs.append(float(p["x"]))
                ys.append(float(p["y"]))
            elif isinstance(p, (list, tuple)):
                xs.append(float(p[0]))
                ys.append(float(p[1]))
        if not xs:
            return None
        return {"x": min(xs), "y": min(ys), "w": max(xs) - min(xs), "h": max(ys) - min(ys)}

    async def _run_pipeline(
        self,
        frame: np.ndarray,
        roi_rect: dict | list | None = None,
        *,
        quality_min: float | None = None,
    ) -> FrameResult:
        """
        Run process_frame through GPUManager (bounded concurrency + VRAM guard)
        or fall back to a module-level semaphore if GPUManager is unavailable.

        quality_min: when set, overrides the recognition quality gate threshold
        in the pipeline (used for retry passes to avoid permanently rejecting
        borderline faces).
        """
        rect = self._roi_to_rect(roi_rect)
        if self._gpu_mgr is not None and self._cfg is not None:
            # GPUManager path: pass quality_min as extra kwarg if supported
            return await self._gpu_mgr.infer(
                self._cfg.session_id, frame, rect,
                quality_min=quality_min,
            )
        sem = _get_fallback_sem()
        async with sem:
            return await asyncio.to_thread(
                self._pipeline.process_frame, frame, rect,
                quality_min=quality_min,
            )

    async def _run_face_only(self, frame: np.ndarray) -> list:
        """
        SCRFD-only face detection for the high-rate PTZ control loop.

        Skips YOLO person detection, MOT, AdaFace embedding, and liveness —
        only SCRFD (~15 ms on GPU) so the face-tracking loop can issue
        absolute_move corrections at ≥ 5 Hz without waiting for the full
        50 ms inference pipeline.

        Returns a ``list[FaceDetection]`` sorted by confidence (same as
        ``AIPipeline.detect_faces``).  Person-bbox filtering is deliberately
        omitted so faces are detected even when no person was detected by YOLO
        (e.g. very close-up or partial body in frame after zoom-in).
        """
        sem = _get_fallback_sem()
        async with sem:
            return await asyncio.to_thread(
                self._pipeline.detect_faces, frame, None
            )

    # ── Debug frame annotation ────────────────────────────────────────────────

    def _annotate_and_store(
        self,
        frame: np.ndarray,
        result: "FrameResult",
        recognized: dict[str, str] | None = None,
    ) -> None:
        """
        Draw person/face bboxes + labels onto *frame* and store as JPEG in
        ``self._annotated_frame_jpg``.

        Args:
            frame:       BGR numpy array (not mutated — copy is made).
            result:      FrameResult from ``_run_pipeline``.
            recognized:  Optional mapping face_index → person_name for the
                         faces that have been identified so far this dwell.
        """
        if frame is None:
            return
        try:
            img = frame.copy()
            h, w = img.shape[:2]
            recognized = recognized or {}

            # ── Person boxes (green) ─────────────────────────────────────
            for p in result.persons:
                x1, y1, x2, y2 = [int(v) for v in p.bbox]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(img, f"{p.confidence:.2f}", (x1, max(y1 - 5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)

            # ── Face boxes ───────────────────────────────────────────────
            for idx, fwe in enumerate(result.faces_with_embeddings):
                fx1, fy1, fx2, fy2 = [int(v) for v in fwe.face.bbox]
                live  = fwe.liveness >= 0.40
                name  = recognized.get(str(idx), None)
                if name:
                    colour = (0, 255, 120)          # bright green  — recognised
                    label  = name
                elif not live:
                    colour = (0, 0, 220)             # red           — spoof
                    label  = f"SPOOF {fwe.liveness:.2f}"
                else:
                    colour = (255, 160, 0)           # orange        — unrecognised live face
                    label  = "Unrecognized"
                cv2.rectangle(img, (fx1, fy1), (fx2, fy2), colour, 2)
                # Label above box, clamped so it never goes off-screen
                ty = max(fy1 - 6, 12)
                cv2.putText(img, label, (fx1, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
                # IOD below box
                iod = getattr(fwe.face, "inter_ocular_px", 0)
                cv2.putText(img, f"IOD {iod:.0f}px", (fx1, fy2 + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 0), 1)

            # ── Top status bar ───────────────────────────────────────────
            bar_h = 32
            cv2.rectangle(img, (0, 0), (w, bar_h), (20, 20, 20), -1)

            state_colours = {
                "PRESET_TRANSIT":   (255, 200, 0),
                "PRESET_RECOGNIZE": (0, 220, 255),
                "FACE_HUNT":        (0, 80, 255),
                "PRESET_COMPLETE":  (0, 200, 100),
                "CYCLE_COMPLETE":   (200, 0, 200),
                "IDLE":             (120, 120, 120),
            }
            sc = state_colours.get(self._overlay_state, (200, 200, 200))
            cv2.putText(img, self._overlay_state, (6, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, sc, 1)

            if self._overlay_preset:
                cv2.putText(img, f"| {self._overlay_preset}", (200, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

            if self._overlay_action:
                cv2.putText(img, self._overlay_action, (w - 220, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)

            # ── Bottom PTZ bar ───────────────────────────────────────────
            ptz = self._last_frame_ptz
            if ptz is not None:
                ptz_txt = f"P {ptz.pan:.3f}  T {ptz.tilt:.3f}  Z {ptz.zoom:.3f}"
                cv2.rectangle(img, (0, h - 22), (w, h), (20, 20, 20), -1)
                cv2.putText(img, ptz_txt, (6, h - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 220, 255), 1)

            # ── Encode to JPEG ───────────────────────────────────────────
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 55])
            if ok:
                self._annotated_frame_jpg = buf.tobytes()
        except Exception:
            pass  # never crash the brain loop over a display glitch

    # ── Identification ────────────────────────────────────────────────────────

    async def _identify(
        self,
        fwe: FaceWithEmbedding,
        cfg: _SessionConfig,
        *,
        liveness_min: float = 0.35,
    ) -> IdentifyResult:
        """
        Route identification through Tier-1 (FAISS) or Tier-2 (pgvector)
        depending on mode.  Liveness gate: discard faces with score < liveness_min.

        Default liveness_min=0.35 (not 0.45) because MiniFASNet was trained on
        selfie-style close-ups; CCTV frames of real faces routinely score 0.35–0.55.
        Printed-photo spoofs are typically below 0.20.

        Pass liveness_min < 0.35 for retry attempts where we accept borderline
        liveness rather than permanently failing to identify a person.
        """
        if fwe.liveness < liveness_min:
            logger.info(
                "IDENTIFY liveness_gate BLOCKED liveness=%.3f (threshold=%.2f)",
                fwe.liveness, liveness_min,
            )
            return IdentifyResult(person_id=None, similarity=0.0, tier=0)

        logger.info("IDENTIFY liveness_gate PASSED liveness=%.3f", fwe.liveness)
        roster = cfg.roster_ids if cfg.mode == OperatingMode.ATTENDANCE else []
        thresholds = {
            "tier1": cfg.face_tier1_threshold,
            "tier2": cfg.face_tier2_threshold,
        }

        # Multi-dataset: search each assigned dataset and return the best match
        search_ids: list[str | None] = cfg.dataset_ids if cfg.dataset_ids else [cfg.dataset_id]
        if len(search_ids) <= 1:
            return await self._repo.identify(
                cfg.client_id, fwe.embedding, roster,
                thresholds=thresholds,
                dataset_id=search_ids[0] if search_ids else None,
            )

        best = IdentifyResult(person_id=None, similarity=0.0, tier=0)
        for did in search_ids:
            result = await self._repo.identify(
                cfg.client_id, fwe.embedding, roster,
                thresholds=thresholds,
                dataset_id=did,
            )
            if result.person_id and result.similarity > best.similarity:
                best = result
        return best

    # ── Duration tracking ─────────────────────────────────────────────────────

    def _update_duration(
        self, person_id: str, ts: float,
        similarity: float | None = None,
        chip: object = None,
    ) -> None:
        if person_id not in self._durations:
            self._durations[person_id] = DurationTracker(
                person_id=person_id,
                first_seen=ts,
                last_seen=ts,
                absence_grace_s=self._absence_grace,
            )
        tracker = self._durations[person_id]
        tracker.on_detected(ts, similarity=similarity)
        # Sample a chip every _EVIDENCE_SAMPLE_N detections, capped at _EVIDENCE_MAX total
        if chip is not None:
            total_stored = tracker._uploaded_count + len(tracker._pending_chips)
            if (total_stored < _EVIDENCE_MAX and
                    tracker.detection_count % _EVIDENCE_SAMPLE_N == 1):
                tracker._pending_chips.append(chip)

    # ── Interrupt management ──────────────────────────────────────────────────

    def _check_interrupts(self) -> list[str]:
        flags: list[str] = []

        # P0: decoder offline
        if self._decoder and not self._decoder.health.running:
            flags.append(InterruptFlag.P0_EMERGENCY.value)

        # P2: faculty missing
        if self._faculty_hunt_active:
            flags.append(InterruptFlag.P2_FACULTY_HUNT.value)

        # P1: motion (set by _detect_motion_at_edge during CELL_RECOGNIZE)
        if self._motion_interrupt:
            flags.append(InterruptFlag.P1_MOTION_INTERRUPT.value)

        return flags

    async def _handle_p0_emergency(self) -> bool:
        """
        Attempt to restart the decoder.  Returns True if recovered.
        """
        self._emergency_retries += 1
        if self._emergency_retries > _EMERGENCY_RETRY_LIMIT:
            logger.error("P0 EMERGENCY: exceeded retry limit (%d)", _EMERGENCY_RETRY_LIMIT)
            return False

        logger.warning(
            "P0 EMERGENCY: decoder offline — restart attempt %d/%d",
            self._emergency_retries, _EMERGENCY_RETRY_LIMIT,
        )

        if self._decoder:
            try:
                await self._decoder.stop()
            except Exception:
                pass

        await asyncio.sleep(3.0 * self._emergency_retries)

        if self._decoder:
            # Reset internal stop_event and restart frame reader
            self._decoder._stop_event.clear()
            if self._frame_reader_task and not self._frame_reader_task.done():
                self._frame_reader_task.cancel()
            self._frame_reader_task = asyncio.create_task(
                self._frame_reader_loop(),
                name=f"frame_reader_recovery:{self._cfg and self._cfg.camera_id}",
            )

        # Wait for first frame to confirm recovery
        fd = await self._grab_frame(timeout=10.0)
        if fd is not None:
            self._emergency_retries = 0
            logger.info("P0 EMERGENCY: decoder recovered")
            return True

        return False

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _deduplicate_faces(
        fwes: list[FaceWithEmbedding],
        cos_threshold: float = 0.60,
    ) -> list[FaceWithEmbedding]:
        """
        Merge faces across frames by cosine similarity (full 512-D).
        For each cluster: pick the face with the best IOD for metadata (bbox,
        chip, liveness) and replace its embedding with the L2-normalised mean
        of the cluster — temporal averaging gives a more robust signal.
        """
        if not fwes:
            return []
        used = [False] * len(fwes)
        result: list[FaceWithEmbedding] = []
        embs = np.stack([f.embedding for f in fwes])
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        embs_normed = embs / norms
        sim_matrix = embs_normed @ embs_normed.T

        for i in range(len(fwes)):
            if used[i]:
                continue
            cluster_indices = [i]
            for j in range(i + 1, len(fwes)):
                if not used[j] and sim_matrix[i, j] >= cos_threshold:
                    cluster_indices.append(j)
                    used[j] = True
            used[i] = True
            best_idx = max(cluster_indices, key=lambda k: fwes[k].face.inter_ocular_px)
            best_fwe = fwes[best_idx]

            # Temporal embedding averaging: mean of all cluster embeddings, re-normalised
            if len(cluster_indices) > 1:
                avg_emb = embs_normed[cluster_indices].mean(axis=0)
                avg_norm = np.linalg.norm(avg_emb)
                if avg_norm > 1e-9:
                    avg_emb /= avg_norm
                best_fwe = FaceWithEmbedding(
                    face=best_fwe.face,
                    embedding=avg_emb,
                    liveness=best_fwe.liveness,
                    face_chip=best_fwe.face_chip,
                )

            result.append(best_fwe)
        return result

    @staticmethod
    def _persons_without_face(
        persons: list[PersonDetection],
        fwes: list[FaceWithEmbedding],
    ) -> list[PersonDetection]:
        """
        Return persons whose upper-body region contains no detected face centroid.

        A face is associated with a person when its centroid (midpoint of the
        face bbox) falls inside the person's bbox from the top down to 65% of
        the body height — the region where the head is expected to appear.
        Persons whose head region overlaps no detected face are candidates for
        body-tracking so the camera can chase them until a face becomes visible.
        """
        if not persons:
            return []
        unmatched: list[PersonDetection] = []
        for person in persons:
            px1, py1, px2, py2 = (float(v) for v in person.bbox)
            head_y_max = py1 + (py2 - py1) * 0.65
            associated = any(
                px1 <= (fwe.face.bbox[0] + fwe.face.bbox[2]) / 2.0 <= px2
                and py1 <= (fwe.face.bbox[1] + fwe.face.bbox[3]) / 2.0 <= head_y_max
                for fwe in fwes
            )
            if not associated:
                unmatched.append(person)
        return unmatched

    async def _safe_get_ptz(self, *, fresh: bool = False) -> PTZPosition:
        """
        Get current PTZ position with a short-lived cache to avoid redundant
        network round-trips.  The camera's physical position is stable on the
        300 ms timescale; most callers only need an approximate value.

        Pass fresh=True to bypass the cache (e.g. immediately after a move).
        Any PTZ command issued via _camera also invalidates the cache.
        """
        now = time.monotonic()
        if (
            not fresh
            and self._ptz_cache is not None
            and now - self._ptz_cache_ts < self._PTZ_CACHE_TTL
        ):
            return self._ptz_cache
        try:
            pos = await self._camera.get_ptz_status()
        except Exception:
            pos = PTZPosition(0.0, 0.0, 0.0)
        self._ptz_cache    = pos
        self._ptz_cache_ts = now
        return pos

    def _invalidate_ptz_cache(self) -> None:
        """Call immediately after issuing any PTZ move command."""
        self._ptz_cache = None
        self._ptz_cache_ts = 0.0

    async def _save_calibration(self, calib: dict, camera_id: str) -> None:
        """
        Persist FOV calibration results to the camera's learned_params column in DB.
        Merges new K values into any existing learned_params so other fields
        (camera_speeds, zoom_pan_drift, etc.) are not overwritten.
        """
        if not self._session_factory:
            logger.debug("_save_calibration: no session_factory, skipping DB write")
            return
        try:
            async with self._session_factory() as db:
                await db.execute(
                    text("""
                        UPDATE cameras
                        SET learned_params = COALESCE(learned_params, '{}'::jsonb)
                                         || (:patch)::jsonb
                        WHERE camera_id = (:cid)::uuid
                    """),
                    {"cid": camera_id, "patch": __import__("json").dumps(calib)},
                )
                await db.commit()
            self._learned_params.update(calib)
            logger.info(
                "PTZBrain: calibration saved to camera %s: %s", camera_id, calib
            )
        except Exception as exc:
            logger.warning("PTZBrain: failed to save calibration: %s", exc)

    def _face_hunt_zoom(
        self,
        frame_w: int,
        current_zoom: float,
        iod_px: float,
    ) -> float:
        """
        Compute zoom level so the face appears at _FACE_HUNT_TARGET_IOD pixels.
        Uses the inverse log-linear FoV model (same as zone_mapper._required_zoom).
        """
        fov_h_cur, _ = self._camera.get_fov_at_zoom(current_zoom)
        fov_target = fov_h_cur * iod_px / _FACE_HUNT_TARGET_IOD
        return _zoom_for_fov(fov_target, self._camera)

    def _body_track_zoom(
        self,
        person: PersonDetection,
        frame_h: int,
        current_zoom: float,
    ) -> float:
        """
        Compute the zoom level that scales the person's body height to fill
        _PERSON_TRACK_BODY_FRAC of the frame, ensuring the face region is
        large enough for reliable SCRFD detection and ArcFace embedding.
        """
        body_h = float(person.bbox[3] - person.bbox[1])
        body_frac = body_h / max(frame_h, 1)
        if body_frac < 1e-3:
            return current_zoom
        zoom_scale = _PERSON_TRACK_BODY_FRAC / body_frac
        fov_h_cur, _ = self._camera.get_fov_at_zoom(current_zoom)
        fov_target = fov_h_cur / max(zoom_scale, 0.1)
        return _zoom_for_fov(fov_target, self._camera)

    async def _face_hunt_recognize(
        self,
        frame: np.ndarray,
        ts: float,
        frame_cx: float,
        frame_cy: float,
        cur_iod: float,
        err_x: float,
        err_y: float,
        budget_end: float,
        cfg: "_SessionConfig",
        location_id: str,
        *,
        emb_bank: "list[tuple[np.ndarray, float]]",
        attempt_n: int = 1,
    ) -> dict | None:
        """
        Background recognition coroutine for FACE_HUNT Phase 2.

        Commercial-level recognition strategy
        ──────────────────────────────────────
        1. Collect ``_HUNT_EMBED_AVG_N`` embeddings from consecutive frames.
        2. IOD-weight each embedding (larger face crop → higher confidence).
        3. Accumulate all (embedding, weight) pairs into ``emb_bank`` so that
           embeddings from previous failed attempts contribute to the FAISS
           search — the bank grows with every retry until the person is
           identified.
        4. After ``_HUNT_EMB_BANK_MAX`` embeddings, prune the lowest-weight
           half so the bank stays bounded while keeping the best crops.
        5. Adaptive quality / liveness gate: after attempt 2+ the quality
           floor drops to ``_HUNT_QUALITY_RETRY_FLOOR`` (0.38); after attempt
           3+ the liveness floor drops to ``_HUNT_LIVENESS_RETRY_FLOOR``
           (0.20) — we accept borderline faces rather than permanently failing.
        6. Compute IOD-weighted L2-mean from the entire bank before FAISS.

        Returns a result dict on successful identification, or ``None``.
        All publish / state-update side-effects are handled here.
        """
        # Adaptive quality gate — relax on repeated attempts
        _quality_min = (
            _HUNT_QUALITY_RETRY_FLOOR if attempt_n >= 2 else None
        )
        _liveness_min: float = (
            _HUNT_LIVENESS_RETRY_FLOOR if attempt_n >= 3 else 0.35
        )

        # ── Full pipeline on the stable frame ────────────────────────────────
        _full_result = await self._run_pipeline(
            frame, cfg.roi_rect, quality_min=_quality_min
        )
        self._annotate_and_store(frame, _full_result, {})

        if not _full_result.faces_with_embeddings:
            # Quality gate rejected the face — embeddings not gathered this
            # iteration; bank is preserved for next attempt.
            return None

        closest = min(
            _full_result.faces_with_embeddings,
            key=lambda f2: math.hypot(
                (f2.face.bbox[0] + f2.face.bbox[2]) / 2 - frame_cx,
                (f2.face.bbox[1] + f2.face.bbox[3]) / 2 - frame_cy,
            ),
        )

        # ── IOD-weighted embedding collection (_HUNT_EMBED_AVG_N frames) ─────
        # Gather consecutive stable-frame embeddings.  Each embedding is stored
        # with its IOD as weight so higher-quality crops dominate the average.
        new_pairs: list[tuple[np.ndarray, float]] = [
            (closest.embedding, max(1.0, closest.face.inter_ocular_px))
        ]
        best_chip = closest.face_chip
        best_iod  = closest.face.inter_ocular_px

        for _ in range(_HUNT_EMBED_AVG_N - 1):
            if time.time() >= budget_end:
                break
            _efd = await self._grab_frame(timeout=0.5)
            if _efd is None:
                break
            _, _, _eframe = _efd
            _eres = await self._run_pipeline(
                _eframe, cfg.roi_rect, quality_min=_quality_min
            )
            if not _eres.faces_with_embeddings:
                continue   # don't abort — keep collecting remaining slots
            _ec = min(
                _eres.faces_with_embeddings,
                key=lambda f2: math.hypot(
                    (f2.face.bbox[0] + f2.face.bbox[2]) / 2 - frame_cx,
                    (f2.face.bbox[1] + f2.face.bbox[3]) / 2 - frame_cy,
                ),
            )
            new_pairs.append((_ec.embedding, max(1.0, _ec.face.inter_ocular_px)))
            if _ec.face.inter_ocular_px > best_iod:
                best_chip = _ec.face_chip
                best_iod  = _ec.face.inter_ocular_px

        # ── Merge new pairs into the cross-attempt bank ───────────────────────
        emb_bank.extend(new_pairs)

        # Prune bank when it exceeds the cap: keep the half with highest IOD
        if len(emb_bank) > _HUNT_EMB_BANK_MAX:
            emb_bank.sort(key=lambda p: p[1], reverse=True)
            del emb_bank[_HUNT_EMB_BANK_MAX:]

        # ── IOD-weighted L2-mean over the full bank ───────────────────────────
        bank_embs    = np.stack([p[0] for p in emb_bank], axis=0)   # [N, 512]
        bank_weights = np.array([p[1] for p in emb_bank], dtype=np.float32)
        bank_weights /= bank_weights.sum()                           # normalise
        avg_emb = np.average(bank_embs, axis=0, weights=bank_weights).astype(np.float32)
        _norm = np.linalg.norm(avg_emb)
        if _norm > 1e-8:
            avg_emb /= _norm

        avg_fwe = FaceWithEmbedding(
            face=closest.face,
            embedding=avg_emb,
            liveness=closest.liveness,
            face_chip=best_chip,
        )

        # ── FAISS identification with adaptive liveness gate ─────────────────
        id_result = await self._identify(avg_fwe, cfg, liveness_min=_liveness_min)
        if not id_result.person_id:
            # FAISS miss — bank is preserved; next stable window will retry
            # with more accumulated embeddings.
            return None

        # ── Publish all side-effects (safe: asyncio is single-threaded) ──────
        self._cycle_seen.add(id_result.person_id)
        self._update_duration(
            id_result.person_id, ts,
            similarity=id_result.similarity,
            chip=avg_fwe.face_chip,
        )
        self._update_cell_unrecognized(location_id, recognized=True)
        await self._publish_detection(avg_fwe, id_result, cfg, location_id)

        return {
            "person_id":    id_result.person_id,
            "similarity":   id_result.similarity,
            "avg_fwe":      avg_fwe,
            "full_result":  _full_result,
            "frame":        frame,
            "cur_iod":      cur_iod,
            "emb_list_len": len(new_pairs),
            "bank_used":    len(emb_bank),
            "err_x":        err_x,
            "err_y":        err_y,
            "is_faculty":   bool(
                cfg.faculty_id and id_result.person_id == cfg.faculty_id
            ),
        }

    def _zoom_scaled_speed_cap(self, current_zoom: float) -> float:
        """
        Commercial-grade velocity ceiling that scales down with zoom-in.

        At narrow FOV a given ONVIF velocity corresponds to a much larger
        angular rate in pixel space — the same 0.18 velocity that is safe at
        zoom=0 will overshoot by the full tilt range in one poll cycle at
        zoom=1.  Scale the ceiling by the ratio of the current horizontal FOV
        to the wide-angle FOV, floored at 30 % so the camera remains
        responsive at telephoto.
        """
        try:
            fov_h_cur, _ = self._camera.get_fov_at_zoom(current_zoom)
            fov_h_wide, _ = self._camera.get_fov_at_zoom(0.0)
            ratio = fov_h_cur / max(fov_h_wide, 1e-6)
        except Exception:
            ratio = 1.0
        ratio = max(_TRACK_MIN_SPEED_FACTOR, min(1.0, ratio))
        return _TRACK_MAX_SPEED * ratio

    def _zoom_scaled_soft_limit(self, current_zoom: float) -> float:
        """
        Soft-limit buffer around the mechanical pan/tilt boundaries, scaled
        with zoom level.  At narrow FOV command-to-motion latency is longer
        (the camera must decelerate from a higher angular rate), so a wider
        buffer is needed to avoid overshoot into the ceiling/floor.
        """
        try:
            fov_h_cur, _ = self._camera.get_fov_at_zoom(current_zoom)
            fov_h_wide, _ = self._camera.get_fov_at_zoom(0.0)
            ratio = fov_h_cur / max(fov_h_wide, 1e-6)
        except Exception:
            ratio = 1.0
        ratio = max(0.0, min(1.0, ratio))
        return _SOFT_LIMIT_MAX - (_SOFT_LIMIT_MAX - _SOFT_LIMIT_MIN) * ratio

    async def _track_person_for_face(
        self,
        persons_no_face: list[PersonDetection],
        cfg: _SessionConfig,
        cell_id: str,
    ) -> list[FaceWithEmbedding]:
        """
        Smoothly track each faceless person body using a continuous-velocity
        proportional controller, keeping the estimated face position centred
        in the frame until a face becomes visible or the per-person budget
        expires.

        Phases per person
        -----------------
        1. **Initial absolute move** — zoom + rough centre on the estimated
           head position so the person fills _PERSON_TRACK_BODY_FRAC of the
           frame.  Zoom is fixed for the rest of the tracking pass to keep
           the FOV stable (zoom changes disturb the P-controller).

        2. **Continuous tracking loop** — every _PERSON_TRACK_POLL_S seconds:
             a. Grab the latest frame and run the full AI pipeline.
             b. Select the tracking target centroid:
                  • Face detected   → face bbox centre  (face is in view!)
                  • No face yet     → body bbox at 25% from top  (head estimate)
             c. Compute normalised pixel error from frame centre:
                  err_x =  (target_cx − frame_cx) / (frame_w / 2)   [−1..1]
                  err_y = −(target_cy − frame_cy) / (frame_h / 2)   [−1..1, Y inverted]
             d. Apply proportional law with dead zone and velocity ceiling:
                  pan_vel  = clamp(_TRACK_KP × err_x, ±_TRACK_MAX_SPEED)
                  tilt_vel = clamp(_TRACK_KP × err_y, ±_TRACK_MAX_SPEED)
                  |err| < _TRACK_DEAD_ZONE  →  velocity = 0  (prevents micro-jitter)
             e. Issue continuous_move(pan_vel, tilt_vel) — the camera moves
                smoothly at this velocity until the next update.
             f. If a face is visible → stop() and collect; if target lost →
                stop() and abandon this person.

        3. **Cleanup** — stop() is always called in a finally block; the camera
           is restored to its pre-tracking position after all persons are
           processed so subsequent FACE_HUNT / CELL_TRANSIT start cleanly.

        Returns
        -------
        All FaceWithEmbedding objects collected during tracking.  These are
        merged into ``all_fwes`` by the caller and go through the normal
        dedup → identify pipeline in CELL_RECOGNIZE.
        """
        found_fwes: list[FaceWithEmbedding] = []
        if not persons_no_face or self._last_frame is None:
            return found_fwes

        # Remember current position so we can restore it afterwards
        ptz_before = await self._safe_get_ptz()

        # Largest body area first — most likely to yield a recognisable face
        targets = sorted(
            persons_no_face,
            key=lambda p: (p.bbox[2] - p.bbox[0]) * (p.bbox[3] - p.bbox[1]),
            reverse=True,
        )

        for person in targets:
            if self._stop_event.is_set():
                break

            ref_frame = self._last_frame
            frame_h, frame_w = ref_frame.shape[:2]
            ptz_now = await self._safe_get_ptz()

            # ── Phase 1: initial absolute move ────────────────────────────────
            # Quickly centre on the estimated head position so Phase 2's
            # P-controller starts from a good position.
            #
            # IMPORTANT: use self._move_speed (≈0.5), NOT _PRECISION_MOVE_SPEED
            # (0.08).  At 8% speed a 20° pan takes ~2.5 s — the camera is still
            # moving when Phase 2 starts, and the P-controller piles on velocity
            # in the same direction → overshoot to ceiling.  At 50% speed the
            # move completes within the cell_settle window.
            #
            # IMPORTANT: do NOT change zoom here.  _body_track_zoom can produce
            # extreme zoom values for distant subjects, which combined with the
            # slow original speed took many seconds to complete and caused the
            # "random zoom" symptom.  Zoom is left at the current level; the
            # Phase 2 P-controller converges on the correct frame size naturally.
            body_cx     = (person.bbox[0] + person.bbox[2]) / 2.0
            face_est_cy = person.bbox[1] + (person.bbox[3] - person.bbox[1]) * 0.25

            target_pan, target_tilt = self._camera.pixel_to_ptz(
                body_cx, face_est_cy, frame_w, frame_h, ptz_now,
            )
            target_zoom = ptz_now.zoom   # hold current zoom — no Phase 1 zoom change

            # BULLET (fixed mount) skips pan/tilt — zoom-only or no-op
            _is_bullet = cfg.camera_type in ("BULLET",)
            try:
                if _is_bullet:
                    # Fixed-mount camera: no pan/tilt available, skip Phase 1 move
                    pass
                else:
                    # PTZ and PTZ_ZOOM both support pan/tilt — move to face estimate
                    await self._camera.absolute_move(
                        target_pan, target_tilt, target_zoom,
                        speed=self._move_speed,
                    )
                self._invalidate_ptz_cache()
                await asyncio.sleep(self._cell_settle)
            except Exception as exc:
                logger.warning("PERSON_TRACK: initial move failed (%s); skipping", exc)
                continue

            # ── Phase 2: smooth proportional tracking loop ────────────────────
            budget_end = time.time() + _PERSON_TRACK_BUDGET_S
            face_found = False

            # Pin the target's MOT track_id so we follow the same person across
            # iterations even if multiple people are visible. Stable across brief
            # YOLO misses because result.tracked_persons uses the Kalman filter.
            target_track_id: int | None = None

            # PTZ position estimate — refreshed from camera every iteration
            # (see note below) so the soft-limit guard compares against ground
            # truth, not against an integrator that drifts positive over time.
            _ptz_phase2 = await self._safe_get_ptz()
            _lim        = self._camera.limits

            # Ceiling-watchdog state — counts consecutive iterations where tilt
            # is within the upper margin AND tilt velocity is still upward.
            # If it reaches _TILT_CEILING_FRAMES we hard-stop + abort the
            # target.  This is an independent safety layer on top of the
            # soft-limit guard (which can be defeated by latency/drift).
            _tilt_up_streak: int = 0
            _tilt_range = _lim.tilt_max - _lim.tilt_min
            _tilt_ceiling_band = _tilt_range * _TILT_CEILING_MARGIN

            # Direction-reversal state — remember the previous velocity signs
            # so we can insert a brief stop() + settle when the commanded
            # direction flips (some PTZ mechanics produce a visible jolt
            # otherwise, especially at high zoom).
            _prev_pan_sign: int  = 0
            _prev_tilt_sign: int = 0

            try:
                while time.time() < budget_end and not self._stop_event.is_set():
                    t_loop = time.monotonic()  # wall-clock start of this iteration

                    fd = await self._grab_frame(timeout=_PERSON_TRACK_POLL_S + 0.5)
                    if fd is None:
                        break

                    _, _ts, frame = fd
                    self._last_frame  = frame
                    self._last_frame_ptz = _ptz_phase2  # use cached — see note above
                    fh, fw = frame.shape[:2]

                    result = await self._run_pipeline(frame, cfg.roi_rect)

                    # Face visible — optionally zoom in for recognition quality, then collect
                    if result.faces_with_embeddings:
                        await self._camera.stop()
                        # If the best face is too small, zoom in once before collecting.
                        # Mirrors the FACE_HUNT zoom nudge so body tracking also
                        # produces recognition-quality crops, not just detection hits.
                        _best_iod = max(
                            f.face.inter_ocular_px for f in result.faces_with_embeddings
                        )
                        if _best_iod > 1.0 and _best_iod < _FACE_HUNT_TARGET_IOD * 0.70:
                            _cur_ptz = await self._safe_get_ptz()
                            if _cur_ptz:
                                _z2 = self._face_hunt_zoom(fw, _cur_ptz.zoom, _best_iod)
                                if _z2 > _cur_ptz.zoom + 0.03:
                                    try:
                                        await self._camera.absolute_move(
                                            _cur_ptz.pan, _cur_ptz.tilt, _z2,
                                            speed=_PRECISION_MOVE_SPEED,
                                            zoom_speed=_PRECISION_MOVE_SPEED,
                                        )
                                        await asyncio.sleep(self._cell_settle)
                                        _fd2 = await self._grab_frame(
                                            timeout=_PERSON_TRACK_POLL_S + 0.5
                                        )
                                        if _fd2 is not None:
                                            _, _, _f2 = _fd2
                                            _r2 = await self._run_pipeline(
                                                _f2, cfg.roi_rect
                                            )
                                            if _r2.faces_with_embeddings:
                                                result = _r2
                                    except Exception:
                                        pass
                        found_fwes.extend(result.faces_with_embeddings)
                        face_found = True
                        logger.info(
                            "PERSON_TRACK: face visible — iod_max=%.0fpx  cell=%s",
                            max(f.face.inter_ocular_px for f in result.faces_with_embeddings),
                            cell_id,
                        )
                        break

                    # ── Find tracking target ──────────────────────────────────
                    # Prefer Kalman-stable tracked_persons — they persist through
                    # brief YOLO misses (motion blur, partial occlusion) for up to
                    # track_buffer frames, preventing premature abort.
                    # Fall back to raw persons only when tracked_persons is empty
                    # (first frame before MOT confirms the track).
                    target_cx: float | None = None
                    target_cy: float | None = None

                    if result.tracked_persons:
                        if target_track_id is not None:
                            # Stay locked to the pinned track
                            pinned = next(
                                (tp for tp in result.tracked_persons
                                 if tp.track_id == target_track_id),
                                None,
                            )
                            if pinned is None:
                                # Track ID gone (ID switch or track expired) — re-pin
                                # to the closest remaining track
                                pinned = min(
                                    result.tracked_persons,
                                    key=lambda p: math.hypot(
                                        (p.bbox[0] + p.bbox[2]) / 2.0 - fw / 2.0,
                                        (p.bbox[1] + p.bbox[3]) / 2.0 - fh / 2.0,
                                    ),
                                )
                                target_track_id = pinned.track_id
                        else:
                            # First iteration — pin the closest tracked person
                            pinned = min(
                                result.tracked_persons,
                                key=lambda p: math.hypot(
                                    (p.bbox[0] + p.bbox[2]) / 2.0 - fw / 2.0,
                                    (p.bbox[1] + p.bbox[3]) / 2.0 - fh / 2.0,
                                ),
                            )
                            target_track_id = pinned.track_id

                        target_cx = (pinned.bbox[0] + pinned.bbox[2]) / 2.0
                        target_cy = (
                            pinned.bbox[1]
                            + (pinned.bbox[3] - pinned.bbox[1]) * 0.25
                        )

                    elif result.persons:
                        # MOT not yet confirmed — fall back to raw YOLO (first few frames)
                        closest_p = min(
                            result.persons,
                            key=lambda p: math.hypot(
                                (p.bbox[0] + p.bbox[2]) / 2.0 - fw / 2.0,
                                (p.bbox[1] + p.bbox[3]) / 2.0 - fh / 2.0,
                            ),
                        )
                        target_cx = (closest_p.bbox[0] + closest_p.bbox[2]) / 2.0
                        target_cy = (
                            closest_p.bbox[1]
                            + (closest_p.bbox[3] - closest_p.bbox[1]) * 0.25
                        )

                    if target_cx is None:
                        # Both tracked_persons and persons are empty — person truly gone
                        await self._camera.stop()
                        logger.debug("PERSON_TRACK: target lost  cell=%s", cell_id)
                        break

                    # ── Pure-P continuous_move controller ──────────────────────
                    # Normalised error: positive → target is right/up of centre.
                    # D-term removed: at variable poll periods (50–200 ms) the
                    # derivative is numerically noisy and causes direction reversals.
                    err_x =  (target_cx - fw / 2.0) / (fw / 2.0)
                    err_y = -(target_cy - fh / 2.0) / (fh / 2.0)  # Y inverted

                    # Zoom-scaled velocity ceiling — see _zoom_scaled_speed_cap.
                    # At narrow FOV the same ONVIF velocity produces a larger
                    # angular rate, so cap accordingly to avoid overshoot.
                    _max_spd = self._zoom_scaled_speed_cap(_ptz_phase2.zoom)

                    pan_vel  = max(-_max_spd, min(_max_spd, _TRACK_KP * err_x))
                    tilt_vel = max(-_max_spd, min(_max_spd, _TRACK_KP * err_y))

                    # Dead zone — stop micro-corrections when already centred
                    if abs(err_x) < _TRACK_DEAD_ZONE:
                        pan_vel = 0.0
                    if abs(err_y) < _TRACK_DEAD_ZONE:
                        tilt_vel = 0.0

                    # Tilt-ceiling circuit-breaker — watchdog independent of the
                    # soft-limit guard.  If tilt has been in the upper margin
                    # with an upward-commanded velocity for _TILT_CEILING_FRAMES
                    # consecutive iterations, hard-stop and abandon this target.
                    # Prevents estimator-drift or latency-induced overshoot from
                    # silently driving the camera into the tilt_max rail.
                    if (
                        _ptz_phase2.tilt >= _lim.tilt_max - _tilt_ceiling_band
                        and tilt_vel > 0
                    ):
                        _tilt_up_streak += 1
                    else:
                        _tilt_up_streak = 0
                    if _tilt_up_streak >= _TILT_CEILING_FRAMES:
                        try:
                            await self._camera.stop()
                        except Exception:
                            pass
                        logger.warning(
                            "PERSON_TRACK: tilt-ceiling breaker tripped "
                            "(tilt=%.3f max=%.3f) — aborting target cell=%s",
                            _ptz_phase2.tilt, _lim.tilt_max, cell_id,
                        )
                        break

                    # Zoom-aware soft-limit boundary guard — wider buffer at
                    # narrow FOV where motion is faster in pixel space.
                    if pan_vel != 0.0 or tilt_vel != 0.0:
                        _soft = self._zoom_scaled_soft_limit(_ptz_phase2.zoom)
                        if pan_vel  > 0 and _ptz_phase2.pan  >= _lim.pan_max  - _soft:
                            pan_vel  = 0.0
                        if pan_vel  < 0 and _ptz_phase2.pan  <= _lim.pan_min  + _soft:
                            pan_vel  = 0.0
                        if tilt_vel > 0 and _ptz_phase2.tilt >= _lim.tilt_max - _soft:
                            tilt_vel = 0.0
                        if tilt_vel < 0 and _ptz_phase2.tilt <= _lim.tilt_min + _soft:
                            tilt_vel = 0.0

                    # Direction-reversal quiet period — the commanded pan/tilt
                    # direction just flipped.  Many PTZ mechanics jolt on a
                    # sign-flip mid-move; issue a brief stop() and settle so
                    # the next continuous_move starts from rest.
                    _pan_sign  = 1 if pan_vel  > 0 else (-1 if pan_vel  < 0 else 0)
                    _tilt_sign = 1 if tilt_vel > 0 else (-1 if tilt_vel < 0 else 0)
                    _reversed = (
                        (_prev_pan_sign  != 0 and _pan_sign  != 0 and _prev_pan_sign  != _pan_sign)
                        or
                        (_prev_tilt_sign != 0 and _tilt_sign != 0 and _prev_tilt_sign != _tilt_sign)
                    )
                    if _reversed:
                        try:
                            await self._camera.stop()
                        except Exception:
                            pass
                        await asyncio.sleep(_REVERSAL_QUIET_S)
                    _prev_pan_sign, _prev_tilt_sign = _pan_sign, _tilt_sign

                    # BULLET cameras have no pan/tilt — skip
                    _skip_pt = _is_bullet
                    if (pan_vel != 0.0 or tilt_vel != 0.0) and not _skip_pt:
                        elapsed   = time.monotonic() - t_loop
                        move_dur  = max(0.05, _PERSON_TRACK_POLL_S - elapsed)
                        try:
                            await self._camera.continuous_move(
                                pan_vel, tilt_vel, 0.0,
                                timeout=move_dur,
                            )
                        except Exception as exc:
                            logger.warning(
                                "PERSON_TRACK: continuous_move failed (%s)", exc,
                            )
                            break
                    else:
                        try:
                            await self._camera.stop()
                        except Exception:
                            pass

                    # Per-iteration ground-truth PTZ refresh — the boundary
                    # guard MUST compare against actual camera state, not an
                    # integrator that accumulates drift.  A single missed
                    # refresh while tilt is drifting up is what historically
                    # let the camera overshoot into the ceiling.  One extra
                    # ONVIF round-trip per iteration (≈10 Hz) is cheap.
                    _actual = await self._safe_get_ptz(fresh=True)
                    if _actual is not None:
                        _ptz_phase2 = _actual

                    # Adaptive sleep — fill remaining poll window so total loop
                    # time ≈ _PERSON_TRACK_POLL_S regardless of inference time.
                    elapsed = time.monotonic() - t_loop
                    sleep_remaining = _PERSON_TRACK_POLL_S - elapsed
                    if sleep_remaining > 0.005:
                        await asyncio.sleep(sleep_remaining)

            finally:
                # Always stop the camera before moving on — even if an exception
                # or budget expiry interrupted the loop.
                try:
                    await self._camera.stop()
                except Exception:
                    pass

            if not face_found:
                logger.debug(
                    "PERSON_TRACK: budget exhausted without face  cell=%s", cell_id,
                )

        # ── Phase 3: restore camera ───────────────────────────────────────────
        # BULLET cameras have no pan/tilt; all others restore to pre-tracking
        # pan/tilt/zoom so the next cell starts from a known position.
        try:
            if cfg.camera_type == "BULLET":
                # Fixed-mount: no pan/tilt to restore
                pass
            else:
                # PTZ, PTZ_ZOOM, BULLET_ZOOM: restore full position
                await self._camera.absolute_move(
                    ptz_before.pan, ptz_before.tilt, ptz_before.zoom,
                    speed=self._move_speed,
                    zoom_speed=_PRECISION_MOVE_SPEED,
                )
        except Exception:
            pass

        return found_fwes

    async def _pre_stage_next_zoom(self) -> None:
        """
        Issue a zoom-only pre-position toward the next cell's required zoom.

        The zoom motor starts moving during the negligible compute time of
        CELL_COMPLETE, so that when CELL_TRANSIT begins the motor is already
        partway (or fully) at the target — reducing effective zoom travel time
        and total settle time for zoom-heavy zone transitions.

        Pan/tilt are NOT changed: the camera stays framed on the current cell.
        Only triggered when the zoom delta exceeds 0.06 ONVIF units (~6%
        of full zoom range) to avoid pointless micro-adjustments.
        """
        if self._scan_map is None or self._plan is None:
            return
        if self._path_index >= len(self._plan.cell_order):
            return

        next_cell = self._scan_map.cells[self._plan.cell_order[self._path_index]]
        try:
            cur = await self._safe_get_ptz()
            zoom_delta = abs(next_cell.required_zoom - cur.zoom)
            if zoom_delta > 0.06:
                # Pan/tilt stay at current position; zoom motor races ahead.
                await self._camera.absolute_move(
                    cur.pan, cur.tilt, next_cell.required_zoom,
                    speed=self._move_speed,
                    zoom_speed=min(self._move_speed * 1.1, 1.0),  # zoom at max speed
                )
                logger.debug(
                    "zoom pre-stage: %.3f → %.3f (Δ=%.3f)",
                    cur.zoom, next_cell.required_zoom, zoom_delta,
                )
        except Exception as exc:
            logger.debug("zoom pre-stage skipped: %s", exc)

    def _detect_motion_at_edge(
        self,
        frame: np.ndarray,
        persons: list,
        edge_fraction: float = _EDGE_FRACTION,
    ) -> bool:
        """Return True if any detected person's centroid is near any frame edge."""
        h, w = frame.shape[:2]
        edge_px = edge_fraction * min(w, h)
        for p in persons:
            cx, cy = p.center
            if cx < edge_px or cx > w - edge_px or cy < edge_px or cy > h - edge_px:
                return True
        return False

    def _update_cell_unrecognized(self, cell, recognized: bool) -> None:
        """Decrement unrecognized_count when a face in the cell is matched (legacy zone mode only)."""
        if recognized and hasattr(cell, "unrecognized_count") and cell.unrecognized_count > 0:
            cell.unrecognized_count -= 1

    async def _teardown(self) -> None:
        """Clean up decoder and finalize all duration trackers."""
        if self._frame_reader_task and not self._frame_reader_task.done():
            self._frame_reader_task.cancel()
            try:
                await self._frame_reader_task
            except (asyncio.CancelledError, Exception):
                pass

        if self._decoder:
            try:
                await self._decoder.stop()
            except Exception:
                pass

        now = time.time()
        for tracker in self._durations.values():
            tracker.finalize()

        # Deregister from GPUManager to free CUDA stream + session slot
        if self._gpu_mgr is not None and self._cfg is not None:
            self._gpu_mgr.deregister_session(self._cfg.session_id)

        logger.info(
            "PTZBrain teardown  camera=%s  tracked_persons=%d",
            self._cfg and self._cfg.camera_id, len(self._durations),
        )

    # ── Kafka event publishing ────────────────────────────────────────────────

    async def _publish_detection(
        self,
        fwe: FaceWithEmbedding,
        id_result: IdentifyResult,
        cfg: _SessionConfig,
        cell_id: str,
    ) -> None:
        """
        Publish a detection event to Kafka (fire-and-forget).
        If Kafka is unavailable, log at INFO level.
        """
        x1, y1, x2, y2 = (float(v) for v in fwe.face.bbox)
        payload = {
            "event_id":   str(uuid.uuid4()),
            "session_id": cfg.session_id,
            "camera_id":  cfg.camera_id,
            "client_id":  cfg.client_id,
            "cell_id":    cell_id,
            "person_id":  id_result.person_id,
            "similarity": round(id_result.similarity, 4),
            "liveness":   round(fwe.liveness, 4),
            "iod_px":     round(fwe.face.inter_ocular_px, 1),
            "bbox":       [x1, y1, x2, y2],
            "tier":       id_result.tier,
            "ts":         time.time(),
        }
        topic = (
            "acas.attendance"
            if cfg.mode == OperatingMode.ATTENDANCE
            else "acas.sightings"
        )
        self._kafka_produce(topic, cfg.client_id, payload)

        # Template update: live detection above moderate confidence.
        # Thresholds are intentionally lower than enrollment (0.80/0.85) so the
        # template self-calibrates toward PTZ camera conditions over time.
        if id_result.person_id and fwe.liveness >= 0.55 and id_result.similarity >= 0.52:
            try:
                await self._repo.update_template(
                    cfg.client_id, id_result.person_id,
                    fwe.embedding, fwe.liveness,
                )
            except Exception as exc:
                logger.debug("update_template skipped: %s", exc)

    async def _publish_attendance_summary(self, cfg: _SessionConfig) -> None:
        """
        Upsert per-person attendance records into the DB at the end of each cycle,
        and publish to Kafka for downstream consumers.
        """
        now = time.time()
        for pid, tracker in self._durations.items():
            payload = {
                "event_id":        str(uuid.uuid4()),
                "session_id":      cfg.session_id,
                "client_id":       cfg.client_id,
                "person_id":       pid,
                "status":          tracker.state,
                "total_seconds":   round(tracker.total_seconds, 1),
                "detection_count": tracker.detection_count,
                "first_seen":      tracker.first_seen,
                "last_seen":       tracker.last_seen,
                "ts":              now,
            }
            self._kafka_produce("acas.attendance", cfg.client_id, payload)

            # Write directly to attendance_records DB (upsert per session+person)
            if self._session_factory is not None:
                try:
                    import io as _io, uuid as _uuid, pathlib as _pl

                    # Upload any pending evidence chips (sampled across this session)
                    new_refs: list[str] = []
                    minio  = getattr(self._repo, "_minio", None)
                    bucket = "face-evidence"
                    base   = getattr(self._repo, "_local_dir", "/enrollment-images")

                    for chip in tracker._pending_chips:
                        try:
                            ok, buf = cv2.imencode(".jpg", chip, [cv2.IMWRITE_JPEG_QUALITY, 92])
                            if not ok:
                                continue
                            data = buf.tobytes()
                            key  = f"{cfg.client_id}/{pid}/{_uuid.uuid4()}.jpg"
                            ref: str | None = None
                            if minio is not None:
                                try:
                                    await asyncio.to_thread(
                                        lambda k=key, d=data: minio.put_object(
                                            bucket, k, _io.BytesIO(d),
                                            length=len(d), content_type="image/jpeg",
                                        )
                                    )
                                    ref = key
                                except Exception as _me:
                                    logger.debug("MinIO evidence upload failed: %s", _me)
                            if ref is None:
                                ldir = _pl.Path(base) / cfg.client_id / pid
                                ldir.mkdir(parents=True, exist_ok=True)
                                lpath = ldir / f"{_uuid.uuid4()}.jpg"
                                lpath.write_bytes(data)
                                ref = f"local:{cfg.client_id}/{pid}/{lpath.name}"
                            new_refs.append(ref)
                        except Exception as _ce:
                            logger.debug("Evidence chip upload error: %s", _ce)

                    tracker._pending_chips.clear()
                    tracker._uploaded_count += len(new_refs)

                    # Cycle-based status: present if seen in at least 1 cycle
                    db_status = "P" if tracker.cycles_seen >= 1 else "ND"

                    async with self._session_factory() as db:
                        await db.execute(
                            text("""
                                INSERT INTO attendance_records
                                    (client_id, session_id, person_id,
                                     status, detection_count,
                                     first_seen, last_seen, total_duration,
                                     confidence_avg, evidence_refs,
                                     cycles_present, total_cycles)
                                VALUES
                                    ((:cid)::uuid, (:sid)::uuid, (:pid)::uuid,
                                     :status, :det_count,
                                     :first_seen, :last_seen,
                                     make_interval(secs => :total_secs),
                                     :conf_avg,
                                     :new_refs,
                                     :cycles_present, :total_cycles)
                                ON CONFLICT (session_id, person_id)
                                DO UPDATE SET
                                    status          = EXCLUDED.status,
                                    detection_count = EXCLUDED.detection_count,
                                    first_seen      = LEAST(attendance_records.first_seen, EXCLUDED.first_seen),
                                    last_seen       = GREATEST(attendance_records.last_seen, EXCLUDED.last_seen),
                                    total_duration  = make_interval(secs => :total_secs),
                                    confidence_avg  = EXCLUDED.confidence_avg,
                                    evidence_refs   = (
                                        SELECT array_agg(r) FROM (
                                            SELECT unnest(
                                                COALESCE(attendance_records.evidence_refs, '{}') ||
                                                COALESCE(EXCLUDED.evidence_refs, '{}')
                                            ) AS r
                                            LIMIT 5
                                        ) sub
                                    ),
                                    cycles_present  = EXCLUDED.cycles_present,
                                    total_cycles    = EXCLUDED.total_cycles
                            """),
                            {
                                "cid":           cfg.client_id,
                                "sid":           cfg.session_id,
                                "pid":           pid,
                                "status":        db_status,
                                "det_count":     tracker.detection_count,
                                "first_seen":    int(tracker.first_seen),
                                "last_seen":     int(tracker.last_seen),
                                "total_secs":    tracker.total_seconds,
                                "conf_avg":      tracker.confidence_avg,
                                "new_refs":      new_refs,
                                "cycles_present": tracker.cycles_seen,
                                "total_cycles":   self._cycle_count,
                            },
                        )
                        await db.commit()
                except Exception as exc:
                    logger.warning("attendance_records upsert failed for person=%s: %s", pid, exc)

        # Update session-level cycle_count and recognition_rate so analytics
        # endpoints can aggregate correctly.  This is a best-effort write.
        if self._session_factory is not None:
            try:
                async with self._session_factory() as db:
                    await db.execute(
                        text("""
                            UPDATE sessions SET
                                cycle_count      = :cycles,
                                recognition_rate = :rate
                            WHERE session_id = (:sid)::uuid
                        """),
                        {
                            "cycles": self._cycle_count,
                            "rate":   self._recognition_rate,
                            "sid":    cfg.session_id,
                        },
                    )
                    await db.commit()
            except Exception as exc:
                logger.debug("session cycle_count/recognition_rate update failed: %s", exc)

    async def _save_unknown_detections(self, cfg: _SessionConfig) -> None:
        """Persist this cycle's unknown face crops to the unknown_detections table."""
        if not self._unknown_chips or self._session_factory is None:
            return
        import io as _io, uuid as _uuid

        minio  = getattr(self._repo, "_minio", None)
        bucket = "face-evidence"
        base   = getattr(self._repo, "_local_dir", "/enrollment-images")
        now    = int(time.time())

        rows = []
        saved_embs: list[np.ndarray] = []
        for chip, conf, live, emb in self._unknown_chips:
            try:
                ok, buf = cv2.imencode(".jpg", chip, [cv2.IMWRITE_JPEG_QUALITY, 88])
                if not ok:
                    continue
                data = buf.tobytes()
                key  = f"{cfg.client_id}/unknown/{_uuid.uuid4()}.jpg"
                ref: str | None = None
                if minio is not None:
                    try:
                        await asyncio.to_thread(
                            lambda k=key, d=data: minio.put_object(
                                bucket, k, _io.BytesIO(d),
                                length=len(d), content_type="image/jpeg",
                            )
                        )
                        ref = key
                    except Exception:
                        pass
                if ref is None:
                    import pathlib as _pl
                    ldir = _pl.Path(base) / cfg.client_id / "unknown"
                    ldir.mkdir(parents=True, exist_ok=True)
                    lpath = ldir / f"{_uuid.uuid4()}.jpg"
                    lpath.write_bytes(data)
                    ref = f"local:{cfg.client_id}/unknown/{lpath.name}"
                rows.append({
                    "det_id":    str(_uuid.uuid4()),
                    "cid":       cfg.client_id,
                    "sid":       cfg.session_id,
                    "cam_id":    cfg.camera_id,
                    "image_ref": ref,
                    "conf":      float(conf) if conf is not None else None,
                    "live":      float(live) if live is not None else None,
                    "det_at":    now,
                })
                if emb is not None:
                    saved_embs.append(emb)
            except Exception as exc:
                logger.debug("Unknown chip save error: %s", exc)

        if not rows:
            return
        try:
            async with self._session_factory() as db:
                await db.execute(
                    text("""
                        INSERT INTO unknown_detections
                            (detection_id, client_id, session_id, camera_id,
                             image_ref, confidence, liveness_score, detected_at)
                        SELECT
                            (v.det_id)::uuid, (v.cid)::uuid, (v.sid)::uuid, (v.cam_id)::uuid,
                            v.image_ref, v.conf::float, v.live::float, v.det_at::bigint
                        FROM jsonb_to_recordset(CAST(:rows AS jsonb)) AS v(
                            det_id text, cid text, sid text, cam_id text,
                            image_ref text, conf text, live text, det_at text
                        )
                    """),
                    {"rows": json.dumps(rows)},
                )
                await db.commit()
        except Exception as exc:
            logger.warning("unknown_detections insert failed: %s", exc)
            return

        # Mirror the saved embeddings into Redis for cross-session dedup.
        await self._persist_unknown_embeddings(cfg.client_id, saved_embs)

    # ── Unknown-face quality gate + helpers ──────────────────────────────────

    async def _unknown_chip_gate(self, fwe: FaceWithEmbedding) -> bool:
        """
        True iff this unrecognised face is worth persisting.

        Gates (all must pass):
          • liveness (fused temporal) ≥ _UNKNOWN_MIN_LIVENESS
          • SCRFD confidence           ≥ _UNKNOWN_MIN_FACE_CONF
          • inter-ocular distance      ≥ _UNKNOWN_MIN_IOD_PX
          • |yaw| ≤ _UNKNOWN_MAX_YAW_DEG and |pitch| ≤ _UNKNOWN_MAX_PITCH_DEG
            (3D pose if available; falls back to 0° when the pose head is
            disabled, which keeps the gate permissive for that case)
          • Laplacian sharpness on the aligned chip ≥ _UNKNOWN_MIN_SHARPNESS
          • chip is not overexposed / featureless
          • embedding is not within _UNKNOWN_DEDUP_THRESHOLD of any already-saved
            unknown for this client in the last 24 h
        """
        if fwe.liveness < _UNKNOWN_MIN_LIVENESS:
            return False
        if fwe.face.conf < _UNKNOWN_MIN_FACE_CONF:
            return False
        if float(fwe.face.inter_ocular_px) < _UNKNOWN_MIN_IOD_PX:
            return False
        if abs(fwe.yaw) > _UNKNOWN_MAX_YAW_DEG or abs(fwe.pitch) > _UNKNOWN_MAX_PITCH_DEG:
            return False

        try:
            gray = cv2.cvtColor(fwe.face_chip, cv2.COLOR_BGR2GRAY)
        except Exception:
            return False
        if float(cv2.Laplacian(gray, cv2.CV_64F).var()) < _UNKNOWN_MIN_SHARPNESS:
            return False
        if float(gray.mean()) > 200.0 or float(gray.std()) < 15.0:
            return False

        emb = fwe.embedding
        if emb is not None:
            if not self._unknown_seen_loaded:
                await self._load_persistent_unknowns()
                self._unknown_seen_loaded = True
            if self._unknown_seen_embeddings:
                gallery = np.stack(self._unknown_seen_embeddings)
                if float((gallery @ emb).max()) >= _UNKNOWN_DEDUP_THRESHOLD:
                    return False
        return True

    def _padded_face_crop(self, fwe: FaceWithEmbedding) -> np.ndarray | None:
        """
        Crop the last full frame around ``fwe.face.bbox`` with generous padding
        so the stored image shows face + hair + shoulders.  The 112×112 aligned
        ArcFace chip is too tight for human review.
        """
        frame = self._last_frame
        if frame is None or fwe.face.bbox is None:
            return None
        h, w = frame.shape[:2]
        try:
            x1, y1, x2, y2 = fwe.face.bbox.astype(float)
        except Exception:
            return None
        bw = x2 - x1
        bh = y2 - y1
        if bw <= 2.0 or bh <= 2.0:
            return None
        pad_x   = bw * 0.55
        pad_top = bh * 0.45
        pad_bot = bh * 0.65
        cx1 = int(max(0, x1 - pad_x))
        cy1 = int(max(0, y1 - pad_top))
        cx2 = int(min(w, x2 + pad_x))
        cy2 = int(min(h, y2 + pad_bot))
        if cx2 - cx1 < 32 or cy2 - cy1 < 32:
            return None
        return frame[cy1:cy2, cx1:cx2].copy()

    async def _load_persistent_unknowns(self) -> None:
        """Seed the dedup cache from the Redis 24 h rolling window."""
        if self._redis is None or self._cfg is None:
            return
        import base64 as _b64
        key = f"acas:unknown:seen:{self._cfg.client_id}"
        try:
            members = await self._redis.lrange(key, 0, -1)
        except Exception:
            return
        for m in members or []:
            try:
                data = _b64.b64decode(m)
                emb = np.frombuffer(data, dtype=np.float16).astype(np.float32)
                if emb.size != 512:
                    continue
                n = float(np.linalg.norm(emb))
                if n < 1e-6:
                    continue
                self._unknown_seen_embeddings.append(emb / n)
            except Exception:
                continue
        if len(self._unknown_seen_embeddings) > _UNKNOWN_SESSION_CAP:
            self._unknown_seen_embeddings = \
                self._unknown_seen_embeddings[-_UNKNOWN_SESSION_CAP:]

    async def _persist_unknown_embeddings(
        self,
        client_id: str,
        embs: list[np.ndarray],
    ) -> None:
        """Push newly-saved unknown embeddings to Redis (fp16, base64)."""
        if self._redis is None or not embs:
            return
        import base64 as _b64
        key = f"acas:unknown:seen:{client_id}"
        try:
            pipe = self._redis.pipeline()
            for e in embs:
                data = _b64.b64encode(
                    e.astype(np.float16).tobytes()
                ).decode("ascii")
                pipe.rpush(key, data)
            pipe.ltrim(key, -_UNKNOWN_REDIS_CAP, -1)
            pipe.expire(key, _UNKNOWN_REDIS_TTL_S)
            await pipe.execute()
        except Exception as exc:
            logger.debug("persist unknown embeddings failed: %s", exc)

    def _kafka_produce(self, topic: str, key: str, payload: dict) -> None:
        if self._kafka is None:
            logger.debug("Kafka(→%s): %s", topic, payload.get("event_id", "?"))
            return
        try:
            self._kafka._produce(topic, key, payload)
        except Exception as exc:
            logger.warning("Kafka produce failed (%s): %s", topic, exc)
