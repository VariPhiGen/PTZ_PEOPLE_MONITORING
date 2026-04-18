"""
BoT-SORT: Multi-Object Tracker with Camera Motion Compensation.

Implements a simplified BoT-SORT tracker suitable for PTZ camera footage:
  - Kalman filter with 8D state (cx, cy, w, h, vx, vy, vw, vh)
  - Camera Motion Compensation (GMC) via sparse optical flow — critical for PTZ
  - Two-stage ByteTrack association (high-conf first, then low-conf)
  - Track lifecycle: TENTATIVE (N frames) → CONFIRMED → LOST → DELETED

References:
  BoT-SORT: Robust Associations Multi-Pedestrian Tracking (2022)
  ByteTrack: Multi-Object Tracking by Associating Every Detection Box (2022)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

# ── Kalman filter matrices ─────────────────────────────────────────────────────

def _build_kalman_matrices() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Constant-velocity Kalman filter for bbox tracking.

    State:       [cx, cy, w, h, vx, vy, vw, vh]
    Measurement: [cx, cy, w, h]
    """
    dt = 1.0  # one frame time step

    # State transition matrix (constant velocity)
    F = np.eye(8, dtype=np.float64)
    F[0, 4] = dt
    F[1, 5] = dt
    F[2, 6] = dt
    F[3, 7] = dt

    # Measurement matrix (observe position only)
    H = np.zeros((4, 8), dtype=np.float64)
    H[0, 0] = H[1, 1] = H[2, 2] = H[3, 3] = 1.0

    # Process noise covariance
    Q = np.eye(8, dtype=np.float64)
    Q[0, 0] = Q[1, 1] = 1.0    # position uncertainty
    Q[2, 2] = Q[3, 3] = 10.0   # size uncertainty
    Q[4, 4] = Q[5, 5] = 0.01   # velocity uncertainty
    Q[6, 6] = Q[7, 7] = 0.1

    # Measurement noise covariance
    R = np.eye(4, dtype=np.float64)
    R[0, 0] = R[1, 1] = 1.0    # position measurement noise
    R[2, 2] = R[3, 3] = 10.0   # size measurement noise

    return F, H, Q, R


_KF_F, _KF_H, _KF_Q, _KF_R = _build_kalman_matrices()


class TrackState(Enum):
    TENTATIVE = auto()   # Just created — wait for min_hits before confirming
    CONFIRMED = auto()   # Stable track — reported to downstream
    LOST      = auto()   # No detection — kept for max_age frames then deleted
    DELETED   = auto()   # Marked for removal


_FEATURE_BANK_CAPACITY: int = 10   # StrongSORT-style ring buffer size


@dataclass
class Track:
    """Single tracked object with Kalman filter state."""

    track_id:    int
    state:       TrackState
    hits:        int         # consecutive frames with detection
    hit_streak:  int         # streak since last miss
    age:         int         # frames since track was created
    time_since_update: int   # frames since last matched detection
    conf:        float       # detection confidence at last update
    reid_emb:    Optional[np.ndarray] = None   # most recent raw RE-ID embedding

    # StrongSORT-style appearance memory.
    #   smooth_emb: EMA of observed embeddings (primary cost vector).  L2-normalised.
    #   feature_bank: short ring buffer of recent raw embeddings, used as a
    #                 robust fallback when smooth_emb drifts (occlusion, fast motion).
    smooth_emb:    Optional[np.ndarray] = None
    feature_bank:  list[np.ndarray] = field(default_factory=list)

    # Kalman state [cx, cy, w, h, vx, vy, vw, vh]
    x:  np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float64))
    P:  np.ndarray = field(default_factory=lambda: np.eye(8, dtype=np.float64) * 10.0)

    @property
    def bbox(self) -> np.ndarray:
        """Return [x1, y1, x2, y2] float32."""
        cx, cy, w, h = self.x[0], self.x[1], self.x[2], self.x[3]
        return np.array([
            cx - w / 2, cy - h / 2,
            cx + w / 2, cy + h / 2,
        ], dtype=np.float32)

    @property
    def center(self) -> tuple[float, float]:
        return float(self.x[0]), float(self.x[1])

    def predict(self) -> None:
        """Kalman predict step (call once per frame for every active track)."""
        self.x = _KF_F @ self.x
        self.P = _KF_F @ self.P @ _KF_F.T + _KF_Q
        # Clamp size to positive
        self.x[2] = max(self.x[2], 1.0)
        self.x[3] = max(self.x[3], 1.0)
        self.age += 1
        self.time_since_update += 1
        # NOTE: hit_streak is NOT reset here.  It is reset to 0 by the tracker
        # after determining which tracks were unmatched this frame.  Resetting
        # here would cap hit_streak at 1 regardless of consecutive hits, making
        # min_hits > 1 unreachable and preventing CONFIRMED promotion.

    # StrongSORT EMA smoothing factor.  α=0.9 → slow drift, resilient to
    # single-frame contamination (partial occlusion, motion blur).
    _EMA_ALPHA: float = 0.9

    def update(self, bbox: np.ndarray, conf: float,
               reid_emb: Optional[np.ndarray] = None) -> None:
        """Kalman update step with matched detection."""
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w  = float(bbox[2] - bbox[0])
        h  = float(bbox[3] - bbox[1])
        z  = np.array([cx, cy, w, h], dtype=np.float64)

        # Innovation
        y = z - _KF_H @ self.x
        # Innovation covariance
        S = _KF_H @ self.P @ _KF_H.T + _KF_R
        # Kalman gain
        K = self.P @ _KF_H.T @ np.linalg.inv(S)
        # State update
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ _KF_H) @ self.P

        self.conf = conf
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0

        if reid_emb is not None:
            self.reid_emb = reid_emb
            # StrongSORT EMA: blend new observation into smoothed feature,
            # re-normalise so cosine distance stays well-defined.
            if self.smooth_emb is None:
                self.smooth_emb = reid_emb.astype(np.float32, copy=True)
            else:
                blended = self._EMA_ALPHA * self.smooth_emb \
                        + (1.0 - self._EMA_ALPHA) * reid_emb
                n = float(np.linalg.norm(blended)) + 1e-8
                self.smooth_emb = (blended / n).astype(np.float32)
            # Ring-buffer the raw embedding — bounded O(_FEATURE_BANK_CAPACITY)
            self.feature_bank.append(reid_emb.astype(np.float32, copy=True))
            if len(self.feature_bank) > _FEATURE_BANK_CAPACITY:
                self.feature_bank.pop(0)


# ── Global Motion Compensation ─────────────────────────────────────────────────

class GlobalMotionCompensation:
    """
    Estimates camera motion between consecutive frames using sparse optical flow.

    Critical for PTZ cameras: when the camera pans/tilts, all bounding boxes
    shift simultaneously.  Without GMC, Kalman predictions go stale immediately
    and the tracker loses every track on each PTZ move.

    Method: sparse Lucas-Kanade optical flow on FAST corners.
    Falls back to identity transform if insufficient features are found.
    """

    def __init__(
        self,
        max_corners: int = 200,
        quality_level: float = 0.01,
        min_distance: int = 10,
        lk_win_size: tuple[int, int] = (21, 21),
        lk_max_level: int = 3,
        ransac_reproj: float = 5.0,
    ) -> None:
        self._max_corners   = max_corners
        self._quality_level = quality_level
        self._min_distance  = min_distance
        self._lk_params = dict(
            winSize=lk_win_size,
            maxLevel=lk_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        self._ransac_reproj = ransac_reproj
        self._prev_gray: Optional[np.ndarray] = None

        # GFTTDetector for robust corner detection
        self._gftt = cv2.GFTTDetector.create(
            maxCorners=self._max_corners,
            qualityLevel=self._quality_level,
            minDistance=self._min_distance,
            blockSize=3,
        )

    def reset(self) -> None:
        """Call on large discontinuities (scene cuts, camera restarts)."""
        self._prev_gray = None

    def estimate(
        self,
        frame: np.ndarray,
        person_bboxes: Optional[list[np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Estimate 2×3 affine transform from previous frame to current.
        Returns identity matrix if estimation fails (first frame or low-feature scene).

        The transform can be applied to predicted track positions:
            new_center = M[:, :2] @ old_center + M[:, 2]

        Args:
            frame:         Current BGR or grayscale frame.
            person_bboxes: Optional list of [x1,y1,x2,y2] detection bboxes.
                           When provided, features inside these regions are
                           masked out so moving people don't contaminate the
                           camera-motion estimate.  RANSAC handles random
                           outliers but a mask is more reliable when the frame
                           is crowded with independently-moving people.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

        identity = np.eye(2, 3, dtype=np.float64)

        if self._prev_gray is None:
            self._prev_gray = gray
            return identity

        # Build person mask for the PREVIOUS frame — exclude corners detected
        # inside person bboxes so only static background features are used.
        if person_bboxes:
            mask = np.ones(self._prev_gray.shape[:2], dtype=np.uint8) * 255
            h, w = mask.shape
            for bbox in person_bboxes:
                x1 = max(0, int(bbox[0]))
                y1 = max(0, int(bbox[1]))
                x2 = min(w, int(bbox[2]))
                y2 = min(h, int(bbox[3]))
                if x2 > x1 and y2 > y1:
                    mask[y1:y2, x1:x2] = 0
        else:
            mask = None

        # Detect corners in previous frame (excluding person regions)
        kps = self._gftt.detect(self._prev_gray, mask)
        if not kps or len(kps) < 4:
            self._prev_gray = gray
            return identity

        prev_pts = np.array([kp.pt for kp in kps], dtype=np.float32).reshape(-1, 1, 2)

        # Track forward with Lucas-Kanade
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, prev_pts, None, **self._lk_params
        )

        if next_pts is None or status is None:
            self._prev_gray = gray
            return identity

        # Keep only successfully tracked points
        mask = status.ravel() == 1
        if mask.sum() < 4:
            self._prev_gray = gray
            return identity

        prev_good = prev_pts[mask].reshape(-1, 2)
        next_good = next_pts[mask].reshape(-1, 2)

        # Estimate partial affine transform with RANSAC
        M, inliers = cv2.estimateAffine2D(
            prev_good, next_good,
            method=cv2.RANSAC,
            ransacReprojThreshold=self._ransac_reproj,
            confidence=0.99,
        )

        self._prev_gray = gray

        if M is None:
            return identity

        return M.astype(np.float64)

    def compensate_tracks(
        self, tracks: list[Track], M: np.ndarray
    ) -> None:
        """
        Warp predicted track center positions by the estimated camera motion.
        Modifies tracks in-place.
        """
        # Skip if identity transform (no camera motion)
        if np.allclose(M, np.eye(2, 3)):
            return

        R = M[:2, :2]
        t = M[:, 2]

        for track in tracks:
            center = np.array([track.x[0], track.x[1]], dtype=np.float64)
            new_center = R @ center + t
            track.x[0] = new_center[0]
            track.x[1] = new_center[1]
            # Also compensate velocity by the rotation component
            vel = np.array([track.x[4], track.x[5]], dtype=np.float64)
            new_vel = R @ vel
            track.x[4] = new_vel[0]
            track.x[5] = new_vel[1]


# ── IoU computation ────────────────────────────────────────────────────────────

def _iou_batch(bboxes_a: np.ndarray, bboxes_b: np.ndarray) -> np.ndarray:
    """
    Compute IoU between all pairs of bboxes.

    Args:
        bboxes_a: [M, 4] xyxy
        bboxes_b: [N, 4] xyxy
    Returns:
        [M, N] IoU matrix
    """
    if len(bboxes_a) == 0 or len(bboxes_b) == 0:
        return np.zeros((len(bboxes_a), len(bboxes_b)), dtype=np.float32)

    # Broadcast intersection
    x1 = np.maximum(bboxes_a[:, None, 0], bboxes_b[None, :, 0])
    y1 = np.maximum(bboxes_a[:, None, 1], bboxes_b[None, :, 1])
    x2 = np.minimum(bboxes_a[:, None, 2], bboxes_b[None, :, 2])
    y2 = np.minimum(bboxes_a[:, None, 3], bboxes_b[None, :, 3])

    inter_w = np.maximum(0, x2 - x1)
    inter_h = np.maximum(0, y2 - y1)
    inter = inter_w * inter_h

    area_a = (bboxes_a[:, 2] - bboxes_a[:, 0]) * (bboxes_a[:, 3] - bboxes_a[:, 1])
    area_b = (bboxes_b[:, 2] - bboxes_b[:, 0]) * (bboxes_b[:, 3] - bboxes_b[:, 1])

    union = area_a[:, None] + area_b[None, :] - inter + 1e-6
    return (inter / union).astype(np.float32)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Cosine distance matrix between embeddings.

    Args:
        a: [M, D] L2-normalised embeddings
        b: [N, D] L2-normalised embeddings
    Returns:
        [M, N] distance matrix (0 = identical, 2 = opposite)
    """
    return 1.0 - a @ b.T


def _fuse_cost(
    iou_cost: np.ndarray,
    reid_cost: Optional[np.ndarray],
    reid_weight: float,
) -> np.ndarray:
    """Combine IoU cost and appearance cost."""
    iou_weight = 1.0 - reid_weight
    if reid_cost is None or reid_weight == 0.0:
        return iou_cost
    return iou_weight * iou_cost + reid_weight * reid_cost


def _hungarian_match(
    cost: np.ndarray,
    threshold: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """
    Run Hungarian algorithm and return matched/unmatched indices.

    Returns:
        matches:          list of (row_idx, col_idx) matched pairs
        unmatched_rows:   row indices with no match
        unmatched_cols:   col indices with no match
    """
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))

    row_idx, col_idx = linear_sum_assignment(cost)

    matches, unmatched_rows, unmatched_cols = [], [], []

    for r, c in zip(row_idx, col_idx):
        if cost[r, c] > threshold:
            unmatched_rows.append(r)
            unmatched_cols.append(c)
        else:
            matches.append((r, c))

    matched_rows = {r for r, _ in matches}
    matched_cols = {c for _, c in matches}
    unmatched_rows.extend(i for i in range(cost.shape[0]) if i not in matched_rows)
    unmatched_cols.extend(j for j in range(cost.shape[1]) if j not in matched_cols)

    return matches, unmatched_rows, unmatched_cols


# ── Main tracker ───────────────────────────────────────────────────────────────

@dataclass
class BoTSORTConfig:
    """Configuration for BoTSORTTracker."""
    track_high_thresh:  float = 0.5    # Min confidence for high-priority detections
    track_low_thresh:   float = 0.1    # Min confidence for low-priority detections
    match_thresh:       float = 0.8    # Max IoU cost to accept a match (1 - IoU)
    track_buffer:       int   = 30     # Frames to keep LOST track before deletion
    min_hits:           int   = 3      # Frames in TENTATIVE before CONFIRMED
    appearance_weight:  float = 0.2    # Weight for RE-ID cost in association (0=disabled)
    gmc_enabled:        bool  = True   # Enable camera motion compensation


class BoTSORTTracker:
    """
    Multi-Object Tracker with camera motion compensation.

    Usage:
        tracker = BoTSORTTracker(BoTSORTConfig())
        tracks = tracker.update(detections, frame)
        # tracks is list[Track] — confirmed tracks only
    """

    def __init__(self, config: BoTSORTConfig | None = None) -> None:
        self.cfg       = config or BoTSORTConfig()
        self._tracks:  list[Track] = []
        self._next_id: int = 1
        self._gmc      = GlobalMotionCompensation() if self.cfg.gmc_enabled else None

    def reset(self) -> None:
        """Reset all tracks (e.g., when switching camera or preset)."""
        self._tracks.clear()
        self._next_id = 1
        if self._gmc:
            self._gmc.reset()

    def update(
        self,
        detections:      list[tuple[np.ndarray, float]],  # (bbox_xyxy, conf)
        frame:           np.ndarray,
        reid_embeddings: Optional[np.ndarray] = None,      # [N, D] per-detection
    ) -> list[Track]:
        """
        Process one frame and return active (CONFIRMED) tracks.

        Args:
            detections:      List of (bbox_xyxy, confidence) from person detector.
            frame:           Current BGR frame for GMC.
            reid_embeddings: Optional [N, D] L2-normalised appearance embeddings,
                             one per detection (same order as detections).
        Returns:
            List of CONFIRMED Track objects with current bboxes.
        """
        # ── Step 1: Camera motion compensation ────────────────────────────────
        M_identity = np.eye(2, 3, dtype=np.float64)

        # Pass detection bboxes so GMC masks out moving people and only uses
        # static background features for the camera-motion estimate.
        det_bboxes = [d[0] for d in detections] if detections else None

        if self._gmc is not None and len(self._tracks) > 0:
            try:
                M = self._gmc.estimate(frame, det_bboxes)
            except Exception:
                M = M_identity
        else:
            M = M_identity
            if self._gmc is not None:
                try:
                    self._gmc.estimate(frame, det_bboxes)   # update prev_gray even if no tracks
                except Exception:
                    pass

        # ── Step 2: Kalman predict + compensate for camera motion ──────────────
        for track in self._tracks:
            track.predict()

        if self._gmc is not None and not np.allclose(M, M_identity):
            self._gmc.compensate_tracks(self._tracks, M)

        # ── Step 3: Split detections into high/low confidence ─────────────────
        dets_high, dets_low = [], []
        embs_high, embs_low = [], []

        for i, (bbox, conf) in enumerate(detections):
            emb = reid_embeddings[i] if reid_embeddings is not None else None
            if conf >= self.cfg.track_high_thresh:
                dets_high.append((bbox, conf))
                embs_high.append(emb)
            elif conf >= self.cfg.track_low_thresh:
                dets_low.append((bbox, conf))
                embs_low.append(emb)

        # Active tracks (not yet fully lost)
        active_tracks = [t for t in self._tracks if t.state != TrackState.DELETED]
        confirmed     = [t for t in active_tracks if t.state == TrackState.CONFIRMED]
        unconfirmed   = [t for t in active_tracks if t.state == TrackState.TENTATIVE]

        # ── Stage 1: Match high-conf detections to confirmed tracks ───────────
        unmatched_confirmed, unmatched_high = self._associate(
            confirmed, dets_high, embs_high,
            iou_thresh=self.cfg.match_thresh,
        )

        # ── Stage 2: Match low-conf detections to remaining confirmed tracks ──
        remaining_confirmed = [confirmed[i] for i in unmatched_confirmed]
        still_unmatched_confirmed, _ = self._associate(
            remaining_confirmed, dets_low, embs_low,
            iou_thresh=self.cfg.match_thresh,
            reid_weight=0.0,          # IoU only for low-conf
        )
        # These confirmed tracks got no match from low-conf either
        lost_tracks = [remaining_confirmed[i] for i in still_unmatched_confirmed]

        # ── Stage 3: Match remaining high-conf to tentative tracks ────────────
        remaining_high = [dets_high[j] for j in unmatched_high]
        remaining_embs = [embs_high[j] for j in unmatched_high]
        _, unmatched_remaining_high = self._associate(
            unconfirmed, remaining_high, remaining_embs,
            iou_thresh=self.cfg.match_thresh,
        )

        # ── Step 4: Reset hit_streak for all tracks that were NOT updated ────────
        # A track's time_since_update is 0 if it was updated this frame (in
        # _associate), > 0 if it was only predicted.  Tracks with no match this
        # frame should have their hit_streak zeroed so consecutive-hit counting
        # works correctly for the TENTATIVE→CONFIRMED promotion gate.
        for track in self._tracks:
            if track.time_since_update > 0:
                track.hit_streak = 0

        # ── Step 5: Mark confirmed tracks lost; delete stale tentative tracks ──
        for track in lost_tracks:
            if track.state == TrackState.CONFIRMED:
                track.state = TrackState.LOST

        for track in active_tracks:
            if track.state == TrackState.LOST:
                if track.time_since_update > self.cfg.track_buffer:
                    track.state = TrackState.DELETED
            elif track.state == TrackState.TENTATIVE:
                # Tentative tracks that miss more than 1 frame have broken their
                # consecutive-hit streak and can never reach min_hits anyway —
                # delete them to prevent unbounded track accumulation.
                if track.time_since_update > 1:
                    track.state = TrackState.DELETED

        # ── Step 6: Initialize new tracks from unmatched high-conf detections ──
        for j in unmatched_remaining_high:
            bbox, conf = remaining_high[j]
            emb = remaining_embs[j]
            track = self._new_track(bbox, conf, emb)
            self._tracks.append(track)

        # ── Step 7: Promote tentative → confirmed ─────────────────────────────
        for track in self._tracks:
            if track.state == TrackState.TENTATIVE and track.hit_streak >= self.cfg.min_hits:
                track.state = TrackState.CONFIRMED

        # ── Step 8: Remove deleted tracks ─────────────────────────────────────
        self._tracks = [t for t in self._tracks if t.state != TrackState.DELETED]

        # Return CONFIRMED tracks + TENTATIVE tracks that matched this frame.
        # CONFIRMED tracks are returned even on missed frames (for track_buffer
        # frames) giving the downstream stable track IDs during brief occlusions.
        return [
            t for t in self._tracks
            if t.state == TrackState.CONFIRMED
            or (t.state == TrackState.TENTATIVE and t.hit_streak >= 1)
        ]

    def _associate(
        self,
        tracks:      list[Track],
        detections:  list[tuple[np.ndarray, float]],
        embeddings:  list[Optional[np.ndarray]],
        iou_thresh:  float,
        reid_weight: Optional[float] = None,
    ) -> tuple[list[int], list[int]]:
        """
        Associate detections to tracks using IoU (+ optional appearance) cost.

        Updates matched tracks in-place.
        Returns (unmatched_track_indices, unmatched_detection_indices).
        """
        if not tracks or not detections:
            return list(range(len(tracks))), list(range(len(detections)))

        w = reid_weight if reid_weight is not None else self.cfg.appearance_weight

        # Build cost matrices
        track_bboxes = np.array([t.bbox for t in tracks])
        det_bboxes   = np.array([d[0] for d in detections])

        iou_mat = _iou_batch(track_bboxes, det_bboxes)
        iou_cost = 1.0 - iou_mat  # lower is better

        # Appearance cost (only if embeddings available and weight > 0).
        # StrongSORT: compare each detection to the track's EMA-smoothed
        # embedding AND to every raw embedding in the track's feature bank;
        # keep the minimum (i.e. best) cosine distance.  This recovers tracks
        # through brief appearance drifts (pose change, partial occlusion)
        # where the EMA alone would lag.
        reid_cost = None
        if w > 0.0:
            det_embs = embeddings
            track_primary = [
                t.smooth_emb if t.smooth_emb is not None else t.reid_emb
                for t in tracks
            ]
            if (all(e is not None for e in track_primary)
                    and all(e is not None for e in det_embs)):
                T_emb = np.stack(track_primary)        # [M, D]
                D_emb = np.stack(det_embs)             # [N, D]
                reid_cost = _cosine_distance(T_emb, D_emb) / 2.0

                # Fallback: for each track take the min cost across its bank
                for i, tr in enumerate(tracks):
                    bank = tr.feature_bank
                    if not bank:
                        continue
                    bank_arr  = np.stack(bank)                 # [B, D]
                    bank_cost = _cosine_distance(bank_arr, D_emb) / 2.0  # [B, N]
                    best_row  = bank_cost.min(axis=0)          # [N]
                    reid_cost[i] = np.minimum(reid_cost[i], best_row)

        cost = _fuse_cost(iou_cost, reid_cost, w)

        matches, unmatched_t, unmatched_d = _hungarian_match(cost, iou_thresh)

        for t_idx, d_idx in matches:
            bbox, conf = detections[d_idx]
            emb = embeddings[d_idx]
            tracks[t_idx].update(bbox, conf, emb)
            if tracks[t_idx].state == TrackState.LOST:
                tracks[t_idx].state = TrackState.CONFIRMED

        return unmatched_t, unmatched_d

    def _new_track(
        self,
        bbox: np.ndarray,
        conf: float,
        emb: Optional[np.ndarray],
    ) -> Track:
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w  = float(bbox[2] - bbox[0])
        h  = float(bbox[3] - bbox[1])

        x = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        P = np.eye(8, dtype=np.float64)
        P[0, 0] = P[1, 1] = 10.0   # high position uncertainty at birth
        P[2, 2] = P[3, 3] = 100.0  # high size uncertainty at birth
        P[4, 4] = P[5, 5] = 1000.0 # high velocity uncertainty at birth

        track = Track(
            track_id=self._next_id,
            state=TrackState.TENTATIVE,
            hits=1,
            hit_streak=1,
            age=1,
            time_since_update=0,
            conf=conf,
            reid_emb=emb,
            # Seed the EMA with the first observation so the first cost
            # computation has something to compare against.
            smooth_emb=emb.astype(np.float32, copy=True) if emb is not None else None,
            feature_bank=[emb.astype(np.float32, copy=True)] if emb is not None else [],
            x=x,
            P=P,
        )
        self._next_id += 1
        return track
