"""
3D head-pose estimator using InsightFace buffalo_l 1k3d68.onnx.

The model predicts 68 3D facial landmarks for each input face; head pose
(yaw, pitch, roll) is then recovered by fitting a 3D-to-3D affine transform
from a canonical 68-point mean face (`_meanshape_68.npy`) to the prediction
and decomposing the rotation matrix.  Math matches the upstream InsightFace
`Landmark` reference implementation (`insightface/utils/transform.py`).

Input chips may be the same aligned ArcFace 112×112 crops used by the embedder
— the model re-scales them to 192×192 internally.  Because SCRFD's similarity
warp already zeroes in-plane roll, the recovered roll component is near zero;
the yaw and pitch components remain intact because that alignment is purely
2-D.

Performance (RTX A5000, FP16): < 4 ms per face, batched.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

_INPUT_SIZE = (192, 192)
_MEANSHAPE_PATH = Path(__file__).with_name("_meanshape_68.npy")


class PoseEstimator3D:
    """Batched 68-point 3D landmark + head-pose inference."""

    def __init__(self, model_path: str | Path) -> None:
        self._model_path = Path(model_path)
        self._session: ort.InferenceSession | None = None
        self._input_name: str = "data"
        self._mean_lmk: np.ndarray | None = None

    # ── Loading ──────────────────────────────────────────────────────────────

    def load(
        self,
        *,
        device_id: int = 0,
        cache_dir: str | None = None,
        has_trt: bool = False,
        has_cuda: bool = False,
    ) -> None:
        if not self._model_path.exists():
            logger.warning(
                "1k3d68.onnx not found at %s — 3D pose disabled",
                self._model_path,
            )
            return
        if not _MEANSHAPE_PATH.exists():
            logger.warning(
                "canonical mean-shape missing at %s — 3D pose disabled",
                _MEANSHAPE_PATH,
            )
            return

        try:
            self._mean_lmk = np.load(_MEANSHAPE_PATH).astype(np.float32)
        except Exception as exc:
            logger.error("could not load meanshape_68: %s", exc)
            return

        cuda_opts: dict[str, Any] = {
            "device_id": device_id,
            "cudnn_conv_algo_search": "HEURISTIC",
            "do_copy_in_default_stream": True,
            "arena_extend_strategy": "kNextPowerOfTwo",
        }
        providers: list
        if has_trt and cache_dir:
            providers = [
                ("TensorrtExecutionProvider", {
                    "device_id": device_id,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": cache_dir,
                    "trt_fp16_enable": True,
                }),
                ("CUDAExecutionProvider", cuda_opts),
                ("CPUExecutionProvider", {}),
            ]
        elif has_cuda:
            providers = [
                ("CUDAExecutionProvider", cuda_opts),
                ("CPUExecutionProvider", {}),
            ]
        else:
            providers = [("CPUExecutionProvider", {})]

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.log_severity_level = 3

        try:
            self._session = ort.InferenceSession(
                str(self._model_path), sess_options=sess_opts, providers=providers,
            )
            self._input_name = self._session.get_inputs()[0].name
            logger.info(
                "PoseEstimator3D loaded (%s, provider=%s)",
                self._model_path.name, self._session.get_providers()[0],
            )
        except Exception as exc:
            logger.error("failed to load 1k3d68.onnx: %s", exc)
            self._session = None

    @property
    def ready(self) -> bool:
        return self._session is not None and self._mean_lmk is not None

    # ── Inference ────────────────────────────────────────────────────────────

    def estimate_batch(
        self,
        chips: np.ndarray | list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Args:
            chips: [N, H, W, 3] uint8 BGR or list of such arrays.  Any size;
                   each chip is resized to 192×192.  The ArcFace-aligned
                   112×112 chip is the intended input.
        Returns:
            (yaws, pitches, rolls) all shape [N] float32 in degrees.
            Positive yaw  = face turned to the camera's right.
            Positive pitch = face tilted down (chin toward chest).
            Positive roll  = head tilted clockwise from the camera's view.
            Returns zeros if the model isn't loaded.
        """
        if not self.ready:
            n = len(chips) if chips is not None else 0
            z = np.zeros((n,), dtype=np.float32)
            return z, z.copy(), z.copy()

        if isinstance(chips, list):
            if not chips:
                z = np.zeros((0,), dtype=np.float32)
                return z, z.copy(), z.copy()
            chips = np.stack(chips, axis=0)

        if chips.ndim != 4 or chips.shape[-1] != 3:
            raise ValueError(f"expected [N, H, W, 3] chips, got shape {chips.shape}")

        n = chips.shape[0]
        if n == 0:
            z = np.zeros((0,), dtype=np.float32)
            return z, z.copy(), z.copy()

        resized = np.empty((n, _INPUT_SIZE[1], _INPUT_SIZE[0], 3), dtype=np.uint8)
        for i in range(n):
            ci = chips[i]
            if ci.shape[0] != _INPUT_SIZE[1] or ci.shape[1] != _INPUT_SIZE[0]:
                resized[i] = cv2.resize(ci, _INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
            else:
                resized[i] = ci

        # 1k3d68 uses input_mean=0, input_std=1 (mxnet arcface-style prefix graph).
        blob = cv2.dnn.blobFromImages(
            resized, scalefactor=1.0, size=_INPUT_SIZE,
            mean=(0.0, 0.0, 0.0), swapRB=True, crop=False,
        )  # [N, 3, 192, 192] float32

        try:
            out = self._session.run(None, {self._input_name: blob})[0]  # [N, 3309]
        except Exception as exc:
            logger.error("1k3d68 inference failed: %s", exc)
            z = np.zeros((n,), dtype=np.float32)
            return z, z.copy(), z.copy()

        out = np.asarray(out, dtype=np.float32)
        # → [N, 1103, 3], take the last 68 as the 68-pt 3D landmarks
        preds = out.reshape(n, -1, 3)[:, -68:, :].copy()
        preds[:, :, 0:2] += 1.0
        preds[:, :, 0:2] *= _INPUT_SIZE[0] // 2
        preds[:, :, 2]   *= _INPUT_SIZE[0] // 2

        yaws    = np.zeros((n,), dtype=np.float32)
        pitches = np.zeros((n,), dtype=np.float32)
        rolls   = np.zeros((n,), dtype=np.float32)
        for i in range(n):
            try:
                rx, ry, rz = _pose_from_landmarks(self._mean_lmk, preds[i])
                pitches[i] = rx
                yaws[i]    = ry
                rolls[i]   = rz
            except Exception:
                pass
        return yaws, pitches, rolls


# ── Pose math (ported from insightface/utils/transform.py) ────────────────────

def _pose_from_landmarks(mean_lmk: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    """
    Fit P s.t. pred ≈ P · [mean | 1]^T  (least squares), decompose into (s, R, t),
    and return Euler angles (pitch, yaw, roll) in degrees.
    """
    X_homo = np.hstack([mean_lmk, np.ones((mean_lmk.shape[0], 1), dtype=np.float32)])
    P = np.linalg.lstsq(X_homo, pred, rcond=None)[0].T   # [3, 4]

    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    n1 = np.linalg.norm(R1) + 1e-9
    n2 = np.linalg.norm(R2) + 1e-9
    r1 = R1 / n1
    r2 = R2 / n2
    r3 = np.cross(r1, r2)
    R  = np.concatenate([r1, r2, r3], axis=0)

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    if sy >= 1e-6:
        x = math.atan2( R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2( R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0.0

    deg = 180.0 / math.pi
    return float(x * deg), float(y * deg), float(z * deg)
