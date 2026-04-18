"""
Age/gender estimator using InsightFace buffalo_l genderage.onnx.

Input:  aligned ArcFace 112×112 chip (same chip used by the embedder).
Output: (age_years: float, gender: int)   gender 0=female, 1=male.

The model is the standard InsightFace GenderAge head: MobileFaceNet backbone
with a 3-wide FC output — `[female_logit, male_logit, age/100]`.  Preprocessing
follows the upstream reference (`cv2.dnn.blobFromImage(chip, 1.0, (96,96),
mean=0, swapRB=True)`) so no explicit normalisation is needed.

This service is optional; if `buffalo_l/genderage.onnx` is missing the
estimator loads as a no-op and `predict_batch` returns empty arrays.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

_INPUT_SIZE = (96, 96)


class DemographicsEstimator:
    """Batched age/gender inference on aligned ArcFace 112×112 chips."""

    def __init__(self, model_path: str | Path) -> None:
        self._model_path = Path(model_path)
        self._session: ort.InferenceSession | None = None
        self._input_name: str = "data"

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
                "genderage.onnx not found at %s — demographics disabled",
                self._model_path,
            )
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
                "DemographicsEstimator loaded (%s, provider=%s)",
                self._model_path.name, self._session.get_providers()[0],
            )
        except Exception as exc:
            logger.error("Failed to load genderage.onnx: %s", exc)
            self._session = None

    @property
    def ready(self) -> bool:
        return self._session is not None

    # ── Inference ────────────────────────────────────────────────────────────

    def predict_batch(
        self,
        chips: np.ndarray | list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            chips: [N, H, W, 3] uint8 BGR or list of such arrays.  Any size;
                   each chip is resized to 96×96.  Passing the ArcFace 112×112
                   aligned chip directly is the intended use.
        Returns:
            (ages, genders)
              ages:    [N] float32, years (capped at [0, 100])
              genders: [N] int8      (0=female, 1=male)
        """
        if self._session is None:
            n = len(chips) if chips is not None else 0
            return np.zeros((n,), dtype=np.float32), np.full((n,), -1, dtype=np.int8)

        if isinstance(chips, list):
            if not chips:
                return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int8)
            chips = np.stack(chips, axis=0)

        if chips.ndim != 4 or chips.shape[-1] != 3:
            raise ValueError(f"expected [N, H, W, 3] chips, got shape {chips.shape}")

        n = chips.shape[0]
        if n == 0:
            return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int8)

        # Resize each chip to 96×96 in a tight batch.  cv2.dnn.blobFromImages
        # handles BGR→RGB swap and layout conversion in one pass to match the
        # reference InsightFace preprocessing exactly.
        resized = np.empty((n, _INPUT_SIZE[1], _INPUT_SIZE[0], 3), dtype=np.uint8)
        for i in range(n):
            ci = chips[i]
            if ci.shape[0] != _INPUT_SIZE[1] or ci.shape[1] != _INPUT_SIZE[0]:
                resized[i] = cv2.resize(ci, _INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
            else:
                resized[i] = ci

        blob = cv2.dnn.blobFromImages(
            resized, scalefactor=1.0, size=_INPUT_SIZE,
            mean=(0.0, 0.0, 0.0), swapRB=True, crop=False,
        )  # [N, 3, 96, 96] float32

        try:
            out = self._session.run(None, {self._input_name: blob})[0]  # [N, 3]
        except Exception as exc:
            logger.error("genderage inference failed: %s", exc)
            return np.zeros((n,), dtype=np.float32), np.full((n,), -1, dtype=np.int8)

        out = np.asarray(out, dtype=np.float32)
        if out.ndim == 1:
            out = out[None]
        # Cols 0-1: gender logits (female, male) — argmax → int
        genders = out[:, :2].argmax(axis=1).astype(np.int8)
        # Col 2: age / 100  → years
        ages = np.clip(out[:, 2] * 100.0, 0.0, 100.0).astype(np.float32)
        return ages, genders
