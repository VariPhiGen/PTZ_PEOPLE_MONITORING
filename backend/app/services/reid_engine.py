"""
RE-ID Engine: OSNet x1_0 person appearance embeddings + gallery management.

Provides 512-D appearance embeddings from person crops using OSNet x1_0 (ONNX).
Used to bridge identity continuity when face is temporarily unavailable.

Gallery lifecycle:
  1. Face recognised → register anchor (identity → person crop embedding)
  2. Face lost but person bbox persists → match crop against gallery
  3. Face reappears → confirm or correct the bridged identity
  4. Anchors expire after TTL or on preset change

Config (environment variables):
  REID_MODEL_PATH         models/osnet_x1_0.onnx
  REID_MATCH_THRESHOLD    0.45   (cosine similarity; OSNet-specific)
  REID_ANCHOR_TTL_SECONDS 30
  REID_ANCHOR_GALLERY_SIZE 5
  REID_PRESET_SCOPED      true
  REID_MAX_BRIDGE_SECONDS 60
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────

REID_MATCH_THRESHOLD:    float = float(os.getenv("REID_MATCH_THRESHOLD",    "0.45"))
REID_ANCHOR_TTL:         float = float(os.getenv("REID_ANCHOR_TTL_SECONDS", "30"))
REID_GALLERY_SIZE:       int   = int(  os.getenv("REID_ANCHOR_GALLERY_SIZE", "5"))
REID_PRESET_SCOPED:      bool  = os.getenv("REID_PRESET_SCOPED", "true").lower() == "true"
REID_MAX_BRIDGE_SECONDS: float = float(os.getenv("REID_MAX_BRIDGE_SECONDS", "60"))

# ImageNet normalisation constants for OSNet
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_INPUT_H = 256   # OSNet x1_0 standard input height
_INPUT_W = 128   # OSNet x1_0 standard input width


@dataclass
class REIDAnchor:
    """Appearance anchor for a recognised identity."""
    identity_id:      str
    track_id:         int
    preset_id:        str
    registered_at:    float             # monotonic timestamp
    embeddings:       list[np.ndarray]  # gallery of up to REID_GALLERY_SIZE embeddings


@dataclass
class REIDMatch:
    identity_id: str
    confidence:  float   # cosine similarity [0, 1]


@dataclass
class BridgedCycle:
    cycle_id:       int
    matched_id:     str
    confidence:     float
    bridged_at:     float


class REIDEngine:
    """
    Person RE-ID using OSNet x1_0 ONNX model.

    Falls back to a dummy zero-embedding extractor if the model file is absent
    so the rest of the pipeline can run without OSNet installed.
    """

    def __init__(self, model_path: str = "models/osnet_x1_0.onnx") -> None:
        self._session = None
        self._input_name: str = "input"
        self._model_path = model_path
        self._loaded = False

    def load(self) -> None:
        """Load OSNet ONNX session.  Called once at startup."""
        if self._loaded:
            return
        import os
        if not os.path.exists(self._model_path):
            logger.warning(
                "OSNet model not found at %s — RE-ID disabled (person tracking will "
                "not use appearance cues; run scripts/download_models.py to install).",
                self._model_path,
            )
            self._loaded = True
            return

        try:
            import onnxruntime as ort

            # Mirror the provider selection from ai_pipeline._make_session
            from app.services.ai_pipeline import _HAS_CUDA, _HAS_TRT

            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.log_severity_level = 3

            if _HAS_TRT:
                providers = [
                    ("TensorrtExecutionProvider", {"device_id": 0}),
                    ("CUDAExecutionProvider",     {"device_id": 0}),
                    ("CPUExecutionProvider",      {}),
                ]
            elif _HAS_CUDA:
                providers = [
                    ("CUDAExecutionProvider", {"device_id": 0}),
                    ("CPUExecutionProvider",  {}),
                ]
            else:
                providers = [("CPUExecutionProvider", {})]

            self._session = ort.InferenceSession(
                self._model_path, sess_options=opts, providers=providers
            )
            self._input_name = self._session.get_inputs()[0].name
            active = self._session.get_providers()[0]
            logger.info("OSNet x1_0 loaded → %s", active)
        except Exception as exc:
            logger.error("Failed to load OSNet: %s — RE-ID disabled", exc)
            self._session = None

        self._loaded = True

    def extract(self, person_crop: np.ndarray) -> np.ndarray:
        """
        Extract a 512-D L2-normalised appearance embedding from a person crop.

        Args:
            person_crop: HxWx3 uint8 BGR.
        Returns:
            [512] float32 L2-normalised, or zero vector if model unavailable.
        """
        if self._session is None:
            return np.zeros(512, dtype=np.float32)

        blob = self._preprocess(person_crop[None])  # [1, 3, 256, 128]
        emb = self._session.run(None, {self._input_name: blob})[0][0]  # [512]
        norm = np.linalg.norm(emb) + 1e-8
        return (emb / norm).astype(np.float32)

    def batch_extract(self, crops: list[np.ndarray]) -> np.ndarray:
        """
        Batch extract embeddings.

        Args:
            crops: List of HxWx3 uint8 BGR person crops.
        Returns:
            [N, 512] float32 L2-normalised, or zero array if model unavailable.
        """
        if not crops:
            return np.zeros((0, 512), dtype=np.float32)

        if self._session is None:
            return np.zeros((len(crops), 512), dtype=np.float32)

        blob = self._preprocess(np.stack([self._resize(c) for c in crops]))  # [N,3,256,128]
        emb = self._session.run(None, {self._input_name: blob})[0]   # [N, 512]
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
        return (emb / norms).astype(np.float32)

    def _resize(self, crop: np.ndarray) -> np.ndarray:
        import cv2
        return cv2.resize(crop, (_INPUT_W, _INPUT_H), interpolation=cv2.INTER_LINEAR)

    def _preprocess(self, crops_hwc: np.ndarray) -> np.ndarray:
        """
        Preprocess N×H×W×3 BGR uint8 → N×3×H×W float32 ImageNet-normalised.
        """
        import cv2
        if crops_hwc.ndim == 3:
            crops_hwc = crops_hwc[None]

        N = crops_hwc.shape[0]
        out = np.empty((N, _INPUT_H, _INPUT_W, 3), dtype=np.float32)
        for i in range(N):
            resized = cv2.resize(crops_hwc[i], (_INPUT_W, _INPUT_H),
                                 interpolation=cv2.INTER_LINEAR)
            out[i] = resized[:, :, ::-1].astype(np.float32) / 255.0  # BGR→RGB, normalise

        out = (out - _MEAN) / _STD      # ImageNet normalisation [N, H, W, 3]
        return np.ascontiguousarray(out.transpose(0, 3, 1, 2))  # NCHW


class REIDGallery:
    """
    Short-lived appearance gallery for identity bridging.

    Thread-safety: Not thread-safe; call from a single async task (the camera
    processing loop).
    """

    def __init__(
        self,
        reid_engine:       REIDEngine,
        match_threshold:   float = REID_MATCH_THRESHOLD,
        anchor_ttl:        float = REID_ANCHOR_TTL,
        gallery_size:      int   = REID_GALLERY_SIZE,
        preset_scoped:     bool  = REID_PRESET_SCOPED,
        max_bridge_secs:   float = REID_MAX_BRIDGE_SECONDS,
    ) -> None:
        self._engine          = reid_engine
        self.match_threshold  = match_threshold
        self.anchor_ttl       = anchor_ttl
        self.gallery_size     = gallery_size
        self.preset_scoped    = preset_scoped
        self.max_bridge_secs  = max_bridge_secs

        # identity_id → REIDAnchor
        self._anchors: dict[str, REIDAnchor] = {}
        # track_id → list of BridgedCycles
        self._bridges: dict[int, list[BridgedCycle]] = {}

    def register_anchor(
        self,
        identity_id:  str,
        track_id:     int,
        person_crop:  np.ndarray,
        preset_id:    str,
    ) -> None:
        """
        Called when face recognition succeeds — store appearance anchor.
        Maintains a rolling gallery of up to gallery_size embeddings per identity.
        """
        emb = self._engine.extract(person_crop)
        if np.allclose(emb, 0):
            return  # RE-ID engine not available

        now = time.monotonic()
        if identity_id in self._anchors:
            anchor = self._anchors[identity_id]
            anchor.registered_at = now
            anchor.track_id      = track_id
            anchor.preset_id     = preset_id
            # Rolling gallery: add new, drop oldest if over size
            anchor.embeddings.append(emb)
            if len(anchor.embeddings) > self.gallery_size:
                anchor.embeddings.pop(0)
        else:
            self._anchors[identity_id] = REIDAnchor(
                identity_id=identity_id,
                track_id=track_id,
                preset_id=preset_id,
                registered_at=now,
                embeddings=[emb],
            )

    def match(
        self,
        person_crop: np.ndarray,
        preset_id:   str,
    ) -> Optional[REIDMatch]:
        """
        Match a faceless person crop against the gallery.

        Returns best match above threshold, or None.
        Expired anchors are skipped (not deleted to avoid O(n) pruning per frame).
        """
        query_emb = self._engine.extract(person_crop)
        if np.allclose(query_emb, 0):
            return None  # RE-ID engine not available

        now = time.monotonic()
        best_id:    Optional[str] = None
        best_score: float         = 0.0

        for identity_id, anchor in self._anchors.items():
            if now - anchor.registered_at > self.anchor_ttl:
                continue
            if self.preset_scoped and anchor.preset_id != preset_id:
                continue

            # Cosine similarity: dot product of L2-normalised vectors
            scores = [float(np.dot(query_emb, e)) for e in anchor.embeddings]
            score  = max(scores)

            if score > self.match_threshold and score > best_score:
                best_score = score
                best_id    = identity_id

        if best_id is None:
            return None

        return REIDMatch(identity_id=best_id, confidence=best_score)

    def record_bridge(
        self,
        track_id: int,
        cycle_id: int,
        matched_identity: str,
        confidence: float,
    ) -> None:
        """Record a cycle where identity was inferred via RE-ID."""
        bridge = BridgedCycle(
            cycle_id=cycle_id,
            matched_id=matched_identity,
            confidence=confidence,
            bridged_at=time.monotonic(),
        )
        self._bridges.setdefault(track_id, []).append(bridge)

    def resolve_bridges(
        self, track_id: int, confirmed_identity: str
    ) -> list[int]:
        """
        When face reappears and confirms identity, return the cycle IDs that
        were bridged and should now be attributed to confirmed_identity.
        """
        cycles = self._bridges.get(track_id, [])
        resolved = [b.cycle_id for b in cycles if b.matched_id == confirmed_identity]
        # Clean up resolved bridges
        self._bridges[track_id] = [b for b in cycles if b.matched_id != confirmed_identity]
        return resolved

    def on_preset_change(self, new_preset_id: str) -> None:
        """
        Called when the PTZ camera moves to a new preset.
        Clears all anchors from the previous preset (scope per preset).
        """
        if self.preset_scoped:
            self._anchors = {
                k: v for k, v in self._anchors.items()
                if v.preset_id == new_preset_id
            }
        self._bridges.clear()

    def expire_stale(self) -> None:
        """Remove anchors and bridges past their TTL.  Call periodically."""
        now = time.monotonic()
        self._anchors = {
            k: v for k, v in self._anchors.items()
            if now - v.registered_at <= self.anchor_ttl
        }
        # Expire bridge records older than max bridge time
        def _fresh(b: BridgedCycle) -> bool:
            return now - b.bridged_at <= self.max_bridge_secs

        self._bridges = {
            k: [b for b in v if _fresh(b)]
            for k, v in self._bridges.items()
            if any(_fresh(b) for b in v)
        }
