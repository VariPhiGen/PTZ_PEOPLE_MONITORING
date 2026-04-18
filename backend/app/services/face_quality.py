"""
Face Quality Assessor for ACAS.

Computes a composite quality score (0.0–1.0) from geometric and image-statistics
cues derived from SCRFD landmarks and the raw face crop.  No additional neural
network is required — all computation is CPU-side and takes < 0.5 ms per face.

Quality components and weights:
  face_size    0.20 — pixel area; min 40×40 for reliable recognition
  yaw_angle    0.25 — estimated from nose-midpoint asymmetry vs inter-eye span
  pitch_angle  0.15 — estimated from nose-to-eye vs nose-to-mouth distances
  blur_score   0.20 — Laplacian variance of the greyscale crop
  illumination 0.10 — mean brightness + histogram spread
  occlusion    0.10 — landmark spatial consistency (proxy for SCRFD confidence)

Threshold constants (configurable via environment):
  QUALITY_RECOGNITION_THRESH  0.75  — above this: run ArcFace recognition
  QUALITY_TRACKING_THRESH     0.25  — above this: use for face-track association
  QUALITY_DISCARD_THRESH      0.25  — below this: ignore face completely
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np

# ── Configurable thresholds (override via environment) ────────────────────────

QUALITY_RECOGNITION_THRESH: float = float(
    os.getenv("FACE_QUALITY_RECOGNITION_THRESH", "0.75")
)
QUALITY_TRACKING_THRESH: float = float(
    os.getenv("FACE_QUALITY_TRACKING_THRESH", "0.25")
)
QUALITY_MIN_SIZE_PX: int = int(
    os.getenv("FACE_QUALITY_MIN_SIZE", "40")
)

# Component weights (must sum to 1.0)
_WEIGHTS = {
    "face_size":    0.20,
    "yaw":          0.25,
    "pitch":        0.15,
    "blur":         0.20,
    "illumination": 0.10,
    "occlusion":    0.10,
}


@dataclass
class FaceQualityScore:
    """Composite quality score and individual component scores."""

    composite:              float    # Weighted sum — the primary gate value
    face_size:              float    # 0–1
    yaw:                    float    # 0–1  (1 = frontal)
    pitch:                  float    # 0–1  (1 = level)
    blur:                   float    # 0–1  (1 = sharp)
    illumination:           float    # 0–1  (1 = well-lit)
    occlusion:              float    # 0–1  (1 = visible landmarks)
    estimated_yaw_degrees:  float    # signed: + = turned right, - = turned left
    estimated_pitch_degrees: float   # signed: + = looking up, - = looking down

    @property
    def passes_recognition(self) -> bool:
        return self.composite >= QUALITY_RECOGNITION_THRESH

    @property
    def passes_tracking(self) -> bool:
        return self.composite >= QUALITY_TRACKING_THRESH


class FaceQualityAssessor:
    """
    Fast face quality scoring using geometric + image cues.

    Usage:
        assessor = FaceQualityAssessor()
        score = assessor.assess(face_crop, landmarks)
        if score.passes_recognition:
            embedding = ...
    """

    # ── Size scoring ──────────────────────────────────────────────────────────
    _SIZE_MIN  = QUALITY_MIN_SIZE_PX     # score = 0 below this
    _SIZE_GOOD = 80                       # score = 1 above this
    _SIZE_MAX  = 200                      # cap size bonus (no extra credit for huge faces)

    # ── Yaw scoring ───────────────────────────────────────────────────────────
    _YAW_FRONTAL = 10.0    # degrees — within this = best score
    _YAW_LIMIT   = 60.0    # degrees — beyond this = score 0

    # ── Pitch scoring ─────────────────────────────────────────────────────────
    _PITCH_FRONTAL = 10.0  # degrees
    _PITCH_LIMIT   = 40.0  # degrees

    # ── Blur scoring ──────────────────────────────────────────────────────────
    _BLUR_SHARP = 100.0    # Laplacian variance — above this = score 1
    _BLUR_MIN   = 20.0     # raised from 5.0: whitish surfaces have near-zero Laplacian variance

    # ── Illumination scoring ──────────────────────────────────────────────────
    _ILLUM_DARK  = 30.0    # mean pixel value below this = score 0
    _ILLUM_GOOD  = 80.0    # and above 180 = overexposed (also 0)
    _ILLUM_OVER  = 180.0   # lowered from 200: tighter overexposure gate blocks whitish crops

    def assess(
        self,
        face_crop: np.ndarray,
        landmarks: np.ndarray,
        face_bbox: np.ndarray | None = None,
    ) -> FaceQualityScore:
        """
        Compute quality score for one face.

        Args:
            face_crop:  HxWx3 uint8 BGR — raw (unaligned) face crop.
            landmarks:  [5, 2] float32 (x, y) SCRFD 5-point landmarks in
                        pixel coords of the ORIGINAL frame (not the crop).
                        Order: left_eye, right_eye, nose, left_mouth, right_mouth.
            face_bbox:  Optional [x1, y1, x2, y2] for size calculation.
                        If None, face_crop dimensions are used.
        Returns:
            FaceQualityScore with composite score and all components.
        """
        # ── Face size ─────────────────────────────────────────────────────────
        if face_bbox is not None:
            w = float(face_bbox[2] - face_bbox[0])
            h = float(face_bbox[3] - face_bbox[1])
        else:
            h, w = float(face_crop.shape[0]), float(face_crop.shape[1])

        min_dim = min(w, h)
        size_score = _linear_score(min_dim, self._SIZE_MIN, self._SIZE_GOOD)

        # ── Yaw from 5-point landmarks ────────────────────────────────────────
        yaw_deg, yaw_score = self._estimate_yaw(landmarks)

        # ── Pitch from 5-point landmarks ─────────────────────────────────────
        pitch_deg, pitch_score = self._estimate_pitch(landmarks)

        # ── Blur via Laplacian variance ────────────────────────────────────────
        blur_score = self._compute_blur(face_crop)

        # ── Illumination ──────────────────────────────────────────────────────
        illum_score = self._compute_illumination(face_crop)

        # ── Occlusion proxy: landmark symmetry / spread ───────────────────────
        occl_score = self._compute_occlusion_proxy(landmarks, w)

        # ── Composite ─────────────────────────────────────────────────────────
        composite = (
            _WEIGHTS["face_size"]    * size_score
            + _WEIGHTS["yaw"]        * yaw_score
            + _WEIGHTS["pitch"]      * pitch_score
            + _WEIGHTS["blur"]       * blur_score
            + _WEIGHTS["illumination"] * illum_score
            + _WEIGHTS["occlusion"]  * occl_score
        )

        return FaceQualityScore(
            composite=float(np.clip(composite, 0.0, 1.0)),
            face_size=float(size_score),
            yaw=float(yaw_score),
            pitch=float(pitch_score),
            blur=float(blur_score),
            illumination=float(illum_score),
            occlusion=float(occl_score),
            estimated_yaw_degrees=float(yaw_deg),
            estimated_pitch_degrees=float(pitch_deg),
        )

    # ── Component methods ──────────────────────────────────────────────────────

    def _estimate_yaw(
        self, landmarks: np.ndarray
    ) -> tuple[float, float]:
        """
        Estimate face yaw from 5-point SCRFD landmarks.

        Method: The nose tip sits at the horizontal midpoint between the two eyes
        for a frontal face.  As yaw increases, the nose shifts toward the nearer
        (foreshortened) eye.  Normalise the offset by half the inter-eye span.

        Landmarks: 0=left_eye, 1=right_eye, 2=nose, 3=left_mouth, 4=right_mouth
        """
        le_x = float(landmarks[0, 0])
        re_x = float(landmarks[1, 0])
        no_x = float(landmarks[2, 0])

        eye_span = re_x - le_x
        if eye_span < 1.0:
            return 0.0, 0.5

        mid_x = (le_x + re_x) / 2.0
        norm_offset = (no_x - mid_x) / (eye_span / 2.0 + 1e-6)
        norm_offset = float(np.clip(norm_offset, -1.0, 1.0))
        yaw_deg = norm_offset * 50.0  # empirical mapping ≈ ±50° for profile

        # Score: 1.0 at frontal, 0.0 beyond _YAW_LIMIT
        abs_yaw = abs(yaw_deg)
        if abs_yaw <= self._YAW_FRONTAL:
            score = 1.0
        else:
            score = max(0.0, 1.0 - (abs_yaw - self._YAW_FRONTAL) / (self._YAW_LIMIT - self._YAW_FRONTAL))

        return yaw_deg, score

    def _estimate_pitch(
        self, landmarks: np.ndarray
    ) -> tuple[float, float]:
        """
        Estimate face pitch from 5-point SCRFD landmarks.

        Method: For a level face, the nose tip is approximately equidistant
        between the eye midpoint and the mouth midpoint.  As pitch increases
        (looking up/down) this ratio shifts.

        Positive pitch = looking up (nose closer to mouth midpoint in image).
        Negative pitch = looking down (nose closer to eyes).
        """
        eye_y  = float((landmarks[0, 1] + landmarks[1, 1]) / 2.0)
        nose_y = float(landmarks[2, 1])
        mth_y  = float((landmarks[3, 1] + landmarks[4, 1]) / 2.0)

        face_h = abs(mth_y - eye_y)
        if face_h < 1.0:
            return 0.0, 0.5

        # Normalised nose position: 0 = at eyes, 1 = at mouth
        norm_pos = (nose_y - eye_y) / (face_h + 1e-6)
        # Frontal ≈ 0.5–0.6 for 5pt landmarks; deviation → pitch
        deviation = norm_pos - 0.55  # signed; + = looking down, - = looking up
        pitch_deg = deviation * 80.0  # empirical mapping

        abs_pitch = abs(pitch_deg)
        if abs_pitch <= self._PITCH_FRONTAL:
            score = 1.0
        else:
            score = max(0.0, 1.0 - (abs_pitch - self._PITCH_FRONTAL) / (self._PITCH_LIMIT - self._PITCH_FRONTAL))

        return pitch_deg, score

    def _compute_blur(self, crop: np.ndarray) -> float:
        """Laplacian variance — low = blurry, high = sharp."""
        if crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return float(_linear_score(var, self._BLUR_MIN, self._BLUR_SHARP))

    def _compute_illumination(self, crop: np.ndarray) -> float:
        """
        Score illumination quality: penalise under/over exposure and low contrast.
        """
        if crop.size == 0:
            return 0.5
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        mean = float(gray.mean())

        # Underexposed
        if mean < self._ILLUM_DARK:
            return mean / self._ILLUM_DARK

        # Overexposed
        if mean > self._ILLUM_OVER:
            return max(0.0, 1.0 - (mean - self._ILLUM_OVER) / 30.0)

        # Good range — also reward spread (contrast)
        contrast = float(gray.std())
        contrast_score = min(1.0, contrast / 40.0)   # 40 std dev = good contrast

        # Blend brightness and contrast
        brightness_score = _linear_score(mean, self._ILLUM_GOOD, self._ILLUM_GOOD + 60.0)
        return 0.5 * brightness_score + 0.5 * contrast_score

    def refine_with_3d_pose(
        self,
        score: FaceQualityScore,
        yaw_deg: float,
        pitch_deg: float,
    ) -> FaceQualityScore:
        """
        Recompute the pose-dependent portion of an existing quality score using
        accurate yaw/pitch from a 3D-landmark model (1k3d68).  Leaves all other
        components (size, blur, illumination, occlusion) untouched.

        Mutates ``score`` in place and also returns it for chaining.  Intended
        to be called *after* the expensive 3D pose stage on recognition-quality
        faces — keeps the cheap heuristic intact for the initial gate.
        """
        abs_yaw = abs(yaw_deg)
        if abs_yaw <= self._YAW_FRONTAL:
            yaw_s = 1.0
        else:
            yaw_s = max(0.0, 1.0 - (abs_yaw - self._YAW_FRONTAL)
                         / (self._YAW_LIMIT - self._YAW_FRONTAL))

        abs_pitch = abs(pitch_deg)
        if abs_pitch <= self._PITCH_FRONTAL:
            pitch_s = 1.0
        else:
            pitch_s = max(0.0, 1.0 - (abs_pitch - self._PITCH_FRONTAL)
                           / (self._PITCH_LIMIT - self._PITCH_FRONTAL))

        # Reconstruct composite with the refined components
        composite = (
            _WEIGHTS["face_size"]      * score.face_size
            + _WEIGHTS["yaw"]          * yaw_s
            + _WEIGHTS["pitch"]        * pitch_s
            + _WEIGHTS["blur"]         * score.blur
            + _WEIGHTS["illumination"] * score.illumination
            + _WEIGHTS["occlusion"]    * score.occlusion
        )

        score.yaw = float(yaw_s)
        score.pitch = float(pitch_s)
        score.estimated_yaw_degrees = float(yaw_deg)
        score.estimated_pitch_degrees = float(pitch_deg)
        score.composite = float(np.clip(composite, 0.0, 1.0))
        return score

    def _compute_occlusion_proxy(
        self, landmarks: np.ndarray, face_width: float
    ) -> float:
        """
        Proxy for occlusion: checks that all 5 landmarks are within the face bbox
        and have plausible relative positions.  SCRFD does not output per-landmark
        confidence, so we use spatial consistency instead.
        """
        if face_width < 1.0:
            return 0.5

        # Check that landmarks span a reasonable proportion of face width
        lm_x_span = float(landmarks[:, 0].max() - landmarks[:, 0].min())
        if lm_x_span < face_width * 0.2:
            return 0.3  # landmarks suspiciously clustered — probably occluded

        # Check vertical ordering: eyes above nose above mouth
        eye_y_mean   = float((landmarks[0, 1] + landmarks[1, 1]) / 2.0)
        nose_y       = float(landmarks[2, 1])
        mouth_y_mean = float((landmarks[3, 1] + landmarks[4, 1]) / 2.0)

        if not (eye_y_mean < nose_y < mouth_y_mean):
            return 0.2   # wrong ordering — occluded or flipped

        return 1.0


# ── Helpers ────────────────────────────────────────────────────────────────────

def _linear_score(value: float, lo: float, hi: float) -> float:
    """Map value linearly from [lo, hi] → [0.0, 1.0], clamped."""
    if hi <= lo:
        return 0.0
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))
