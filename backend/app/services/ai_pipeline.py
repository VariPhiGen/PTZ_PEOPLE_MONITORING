"""
AIPipeline — full-stack inference pipeline for ACAS (v2).

Stages:
  1. Person detection   — YOLOv8l (bbox only)  ONNX → TRT FP16
  2. Face detection     — SCRFD-10GF            ONNX → TRT FP16  (buffalo_l)
  3. Face-to-person spatial association
  4. BoT-SORT MOT tracking with camera motion compensation
  5. Face quality gate (composite landmark-based score)
  6. Face alignment     — similarity-warp to 112×112 ArcFace canonical crop
  7. Face embedding     — AdaFace IR-101         ONNX → TRT FP16  (512-D L2)
  8. Liveness scoring   — MiniFASNetV2           ONNX → TRT FP16
  9. Identity-track binding + RE-ID bridge
  10. Best-shot gallery update

All ONNX→TRT conversions are handled transparently by ONNX Runtime's
TensorrtExecutionProvider; compiled engines are cached on disk so that
subsequent process startups skip the slow build.

Typical first-run build:  ~2–5 min per model on RTX A5000
Subsequent runs:          engines load in < 2 s each

Performance targets (RTX A5000, 1080p input):
  YOLO:      < 15 ms  (detection-only, no pose)
  SCRFD:     < 15 ms
  AdaFace:   <  3 ms / face  (batched)
  Total:     < 50 ms for 15 faces

Usage:
    import asyncio
    pipe = AIPipeline("/models", device_id=0)
    await asyncio.to_thread(pipe.load)          # once at startup
    result = await asyncio.to_thread(pipe.process_frame, frame, roi_rect)
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from app.services.demographics import DemographicsEstimator
from app.services.face_quality import FaceQualityAssessor, FaceQualityScore, QUALITY_RECOGNITION_THRESH
from app.services.identity_state import IdentityTrackManager, TrackIdentityState
from app.services.mot_tracker import BoTSORTConfig, BoTSORTTracker, Track
from app.services.pose_estimator import PoseEstimator3D
from app.services.reid_engine import REIDEngine, REIDGallery
from app.services.temporal_liveness import TemporalLivenessTracker

logger = logging.getLogger(__name__)

# ── ArcFace 5-point canonical template (112×112, from InsightFace) ─────────────
_ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],   # left  eye   (person-left = image-right)
        [73.5318, 51.5014],   # right eye
        [56.0252, 71.7366],   # nose tip
        [41.5493, 92.3655],   # left  mouth corner
        [70.7299, 92.2041],   # right mouth corner
    ],
    dtype=np.float32,
)

# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class PersonDetection:
    """Single person detected by YOLOv8l (bbox only, no pose)."""
    bbox:   np.ndarray             # [x1, y1, x2, y2] float32 absolute pixels
    conf:   float
    center: tuple[float, float]    # (cx_px, cy_px)


@dataclass
class FaceDetection:
    """Single face detected by SCRFD (buffalo_l/det_10g.onnx)."""
    bbox: np.ndarray        # [x1, y1, x2, y2] float32 absolute pixels
    landmarks: np.ndarray   # [5, 2] float32 (x, y) for 5-point face keypoints
    conf: float
    inter_ocular_px: float  # Euclidean distance between eye landmarks


@dataclass
class FaceWithEmbedding:
    """Face detection enriched with embedding and liveness score."""
    face: FaceDetection
    embedding: np.ndarray   # [512] float32 L2-normalised
    liveness: float         # fused temporal liveness (preferred for decisions)
    face_chip: np.ndarray   # [112, 112, 3] uint8 BGR — aligned ArcFace crop
    age: float = -1.0       # estimated age in years (-1 = unknown)
    gender: int = -1        # 0=female, 1=male, -1=unknown
    yaw: float = 0.0        # degrees, signed — from 1k3d68 when available
    pitch: float = 0.0      # degrees, signed
    roll: float = 0.0       # degrees, signed
    liveness_single: float = 1.0   # raw single-frame MiniFASNet score
    liveness_motion: float = 0.5   # micro-motion cue, 0=static/photo, 1=natural


@dataclass
class TrackedPerson:
    """
    A person in the current frame with stable MOT track_id, face identity
    (if recognised), and quality information.
    """
    track_id:           int
    bbox:               np.ndarray             # [x1, y1, x2, y2] float32
    conf:               float
    center:             tuple[float, float]

    # Identity (from face recognition or RE-ID bridge)
    identity_id:        Any = None             # person UUID or provisional str
    identity_confidence: float = 0.0
    tracking_method:    str  = "motion"        # 'face' | 'reid_bridge' | 'face_reconfirmed' | 'motion'

    # Face quality (None if no face detected this frame)
    face_quality:       Any = None             # FaceQualityScore or None
    best_shot_url:      Any = None             # MinIO URL or None

    # Linked FaceWithEmbedding (if face was detected and passed quality gate)
    fwe:                Any = None             # FaceWithEmbedding or None

    # Demographics (from genderage.onnx; -1 = unknown)
    age:                float = -1.0
    gender:             int   = -1             # 0=female, 1=male, -1=unknown


@dataclass
class FrameResult:
    """Full inference result for one video frame."""
    persons:               list[PersonDetection]
    faces_with_embeddings: list[FaceWithEmbedding]
    tracked_persons:       list[TrackedPerson]
    time_ms:               float
    latency_breakdown:     dict[str, float] = field(default_factory=dict)


# ── Helper — numpy NMS ─────────────────────────────────────────────────────────

def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    """Greedy NMS over xyxy boxes.  Returns indices of kept boxes."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return keep


# ── Helper — YOLO letterbox ────────────────────────────────────────────────────

def _letterbox(
    img: np.ndarray,
    new_shape: tuple[int, int] = (640, 640),
    pad_value: int = 114,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """
    Scale image to new_shape maintaining aspect ratio.
    Returns (padded_image, scale_factor, (pad_top, pad_left)).
    """
    h, w = img.shape[:2]
    scale = min(new_shape[0] / h, new_shape[1] / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    pad_h = (new_shape[0] - nh) // 2
    pad_w = (new_shape[1] - nw) // 2
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = pad_h, new_shape[0] - nh - pad_h
    left, right  = pad_w, new_shape[1] - nw - pad_w
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(pad_value,) * 3,
    )
    return padded, scale, (pad_h, pad_w)



# ── Runtime EP detection via direct library scan (no ORT API calls) ───────────

def _find_lib(*names: str) -> bool:
    """
    Return True if any of the given shared-library names can be dlopen'd.
    Respects LD_LIBRARY_PATH, system ldcache, and common NVIDIA pip paths.
    """
    import ctypes
    # Collect extra search dirs from LD_LIBRARY_PATH and common nvidia pip locations
    extra_dirs: list[str] = []
    for env_path in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        if env_path:
            extra_dirs.append(env_path)
    # pip-installed nvidia packages
    import sys as _sys
    sp = getattr(_sys, "prefix", "/usr/local")
    for pkg in ("cudnn", "cublas", "cuda_runtime", "cufft", "curand"):
        extra_dirs.append(f"{sp}/lib/python{_sys.version_info.major}.{_sys.version_info.minor}"
                          f"/dist-packages/nvidia/{pkg}/lib")

    for name in names:
        # bare name — relies on LD_LIBRARY_PATH / ldconfig
        try:
            ctypes.CDLL(name)
            return True
        except OSError:
            pass
        # absolute-path search in extra dirs
        from pathlib import Path as _P
        for d in extra_dirs:
            p = _P(d) / name
            if p.exists():
                try:
                    ctypes.CDLL(str(p))
                    return True
                except OSError:
                    pass
    return False


def _check_nvidia_driver() -> bool:
    """
    Verify the NVIDIA kernel driver is accessible via libcuda.so.1.

    cuDNN/cuBLAS libs can be present (bundled with pip-installed nvidia-*
    packages) even when the host driver is missing or the container was
    started without --runtime=nvidia.  In that case ORT's CUDAExecutionProvider
    will call cuInit() and receive CUDA_ERROR_NO_DEVICE, then segfault rather
    than raising a Python exception — killing the worker silently.

    This check catches that condition at import time so we can fall back to
    CPU cleanly instead of crashing.
    """
    try:
        import ctypes
        driver = ctypes.CDLL("libcuda.so.1")
        driver.cuInit.restype = ctypes.c_int
        return driver.cuInit(0) == 0  # 0 == CUDA_SUCCESS
    except (OSError, AttributeError):
        return False


# TRT requires libnvinfer; CUDA requires cuDNN (any version) AND the driver
_HAS_CUDA = (
    _find_lib("libcudnn.so.9", "libcudnn_ops.so.9", "libcudnn_adv.so.9",
              "libcudnn.so.8", "libcudnn.so")
    and _check_nvidia_driver()
)
_HAS_TRT  = _HAS_CUDA and _find_lib("libnvinfer.so.10", "libnvinfer.so.8", "libnvinfer.so")

logger.info(
    "ORT GPU detection — TensorRT: %s  CUDA: %s",
    "YES ✓" if _HAS_TRT  else "NO  (TensorRT not installed — using CUDA EP)",
    "YES ✓" if _HAS_CUDA else "NO  ← models will run on CPU",
)
if not _HAS_CUDA:
    logger.warning(
        "NVIDIA driver not accessible (libcuda.so.1 / cuInit failed). "
        "Models will fall back to CPU — inference will be slow. "
        "Ensure the container is started with --runtime=nvidia / nvidia-container-toolkit."
    )


# ── ORT session factory ────────────────────────────────────────────────────────

def _make_session(
    model_path: str,
    device_id: int,
    cache_dir: str,
    fp16: bool = True,
    max_workspace_gb: int = 4,
) -> ort.InferenceSession:
    """
    Build an ORT InferenceSession on the best available provider:
      1. TensorRT EP  (fastest, FP16 — requires TensorRT 10 + cuDNN 9)
      2. CUDA EP      (fast GPU, no TRT needed — requires cuDNN 9)
      3. CPU EP       (fallback only; logs a WARNING so the user knows)

    Provider selection is determined at import time (_HAS_TRT / _HAS_CUDA)
    so no unnecessary provider-library probing (and its noisy errors) happens
    at session-creation time.
    """
    os.makedirs(cache_dir, exist_ok=True)

    cuda_opts: dict[str, Any] = {
        "device_id": device_id,
        "cudnn_conv_algo_search": "HEURISTIC",
        "do_copy_in_default_stream": True,
        "arena_extend_strategy": "kNextPowerOfTwo",
    }

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.log_severity_level = 3   # suppress ORT INFO spam

    # Build provider list from what's actually available
    if _HAS_TRT:
        trt_opts: dict[str, Any] = {
            "device_id": device_id,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": cache_dir,
            "trt_timing_cache_enable": True,
            "trt_timing_cache_path": cache_dir,
            "trt_max_workspace_size": max_workspace_gb * (1 << 30),
            "trt_builder_optimization_level": 3,
            "trt_fp16_enable": fp16,
            # Keep LayerNorm/BatchNorm in FP32 to preserve embedding accuracy
            "trt_layer_norm_fp32_fallback": True,
            "trt_detailed_build_log": False,
        }
        providers: list = [
            ("TensorrtExecutionProvider", trt_opts),
            ("CUDAExecutionProvider", cuda_opts),
            ("CPUExecutionProvider", {}),
        ]
    elif _HAS_CUDA:
        providers = [
            ("CUDAExecutionProvider", cuda_opts),
            ("CPUExecutionProvider", {}),
        ]
    else:
        logger.warning("No GPU provider available — %s will run on CPU", Path(model_path).name)
        providers = [("CPUExecutionProvider", {})]

    sess = ort.InferenceSession(model_path, sess_options=sess_opts, providers=providers)
    active = sess.get_providers()[0]

    if active == "CPUExecutionProvider" and _HAS_CUDA:
        # ORT silently fell back to CPU even though CUDA is available — surface this
        logger.error(
            "CUDA EP failed for %s — running on CPU. "
            "Check that CUDA/cuDNN libraries match onnxruntime-gpu version.",
            Path(model_path).name,
        )
    else:
        logger.info("%-40s → %s  (fp16=%s)", Path(model_path).name, active, fp16)

    return sess


# ── Main pipeline ──────────────────────────────────────────────────────────────

class AIPipeline:
    """
    Orchestrates person detection, face detection, embedding, and liveness.
    Thread-safe for concurrent inference (ORT sessions are re-entrant).
    """

    _YOLO_SIZE        = (640, 640)
    _YOLO_CONF        = 0.50   # raised from 0.30 — reduces person false positives
    _YOLO_IOU         = 0.45
    _YOLO_MIN_HEIGHT  = 60     # pixels — ignore person bboxes shorter than this
    _FACE_DET_SIZE    = (640, 640)
    _FACE_CONF        = 0.60   # raised from 0.45: filters walls/bright-surface false detections
    _EMBED_SIZE       = 112
    _LIVENESS_SIZE    = 80
    _LIVENESS_SCALE   = 2.7   # face bbox expansion for MiniFASNet context
    _LIVE_CLASS_IDX   = 1     # MiniFASNetV2: class 1 = live

    # ── Runtime quality gate (CPU-only, runs before ArcFace) ──────────────────
    # More lenient than enrollment thresholds — we just need enough quality
    # for the model to produce a meaningful embedding.
    _RQ_MIN_IOD        = 20.0   # pixels — minimum inter-ocular distance (SR handles small faces)
    _RQ_MIN_CONF       = 0.60   # raised from 0.40: keep consistent with _FACE_CONF
    _RQ_MAX_YAW        = 50.0   # degrees — allow naturally angled faces (was 40, too strict)
    _RQ_MIN_SHARPNESS  = 20.0   # raised from 5.0: whitish/featureless surfaces have near-zero Laplacian
    _RQ_MIN_ASPECT     = 0.5    # face bbox width/height — reject very wide or very tall blobs
    _RQ_MAX_ASPECT     = 1.8    # bullet wide-angle distortion produces blobs outside this range

    def __init__(self, model_dir: str, device_id: int = 0) -> None:
        self._model_dir  = Path(model_dir)
        self._device_id  = device_id
        self._cache_dir  = str(self._model_dir / "engines")

        self._yolo_sess:     ort.InferenceSession | None = None
        self._face_detector                              = None  # InsightFace SCRFD
        self._embed_sess:    ort.InferenceSession | None = None
        self._liveness_sess: ort.InferenceSession | None = None
        self._demographics:  DemographicsEstimator | None = None
        self._pose_estimator: PoseEstimator3D | None = None
        self._temporal_liveness: TemporalLivenessTracker | None = None

        # Input/output name cache (filled on load)
        self._yolo_input:     str = "images"
        self._embed_input:    str = "input"
        self._liveness_input: str = "input"

        # v2: new pipeline components (lazy-initialised so existing code still works)
        self._mot_tracker:      BoTSORTTracker      | None = None
        self._face_quality:     FaceQualityAssessor | None = None
        self._reid_engine:      REIDEngine          | None = None
        self._reid_gallery:     REIDGallery         | None = None
        self._identity_manager: IdentityTrackManager| None = None
        self._best_shot         = None   # BestShotGallery (optional MinIO)

        self._loaded = False

    # ── Startup ────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """
        Load and warm up all models.  Call once at process startup, ideally in
        a thread (TRT engine builds can take several minutes on first run).
        """
        if self._loaded:
            return
        self._load_yolo()
        self._load_face_detector()
        self._load_embedder()
        self._load_liveness()
        self._load_sr_model()      # optional — skipped if model file absent
        self._load_demographics()  # optional — skipped if genderage.onnx absent
        self._load_pose_estimator()  # optional — skipped if 1k3d68.onnx absent
        self._load_v2_components()
        self._loaded = True

        # ── Startup GPU summary ────────────────────────────────────────────────
        def _ep(sess: Any) -> str:
            if sess is None:
                return "MISSING"
            p = sess.get_providers()[0] if hasattr(sess, "get_providers") else "unknown"
            if p == "CUDAExecutionProvider":
                return "CUDA GPU ✓"
            if p == "TensorrtExecutionProvider":
                return "TensorRT GPU ✓"
            return f"CPU ← WARNING ({p})"

        det_sess = getattr(self._face_detector, "session", None) if self._face_detector else None

        reid_status = "LOADED ✓" if (self._reid_engine and self._reid_engine._session) else "disabled (no model)"
        demog_status = "LOADED ✓" if (self._demographics and self._demographics.ready) else "disabled (no model)"
        pose_status  = "LOADED ✓" if (self._pose_estimator and self._pose_estimator.ready) else "disabled (no model)"

        logger.info(
            "\n"
            "  ╔══ AI Pipeline GPU Status (device %d) ══════════════════╗\n"
            "  ║  YOLO person detect    : %-34s ║\n"
            "  ║  SCRFD face detector  : %-34s ║\n"
            "  ║  ArcFace embedder     : %-34s ║\n"
            "  ║  MiniFASNet liveness  : %-34s ║\n"
            "  ║  OSNet RE-ID          : %-34s ║\n"
            "  ║  GenderAge head       : %-34s ║\n"
            "  ║  1k3d68 3D pose       : %-34s ║\n"
            "  ╚════════════════════════════════════════════════════════╝",
            self._device_id,
            _ep(self._yolo_sess),
            _ep(det_sess) if det_sess else ("CUDA GPU ✓" if self._face_detector else "MISSING"),
            _ep(self._embed_sess),
            _ep(self._liveness_sess),
            reid_status,
            demog_status,
            pose_status,
        )

    def _load_yolo(self) -> None:
        # Model priority: env-configured name → nano → small → large.
        # A smaller model (nano/small) runs in ~5–8 ms vs ~15 ms for large,
        # which matters because YOLO is now only used for person context /
        # RE-ID embedding storage, NOT for PTZ movement decisions.
        _default = os.environ.get("YOLO_MODEL", "yolov8n.onnx")
        _candidates = [_default, "yolov8n.onnx", "yolov8s.onnx", "yolov8l.onnx"]
        seen: set[str] = set()
        p = None
        for name in _candidates:
            if name in seen:
                continue
            seen.add(name)
            candidate = self._model_dir / name
            if candidate.exists():
                p = candidate
                break
        if p is None:
            logger.warning("No YOLO model found in %s — person detection disabled", self._model_dir)
            return
        if p.name != _default:
            logger.info("YOLO: %s not found, using %s instead", _default, p.name)
        self._yolo_sess = _make_session(str(p), self._device_id, self._cache_dir)
        self._yolo_input = self._yolo_sess.get_inputs()[0].name

    def _load_v2_components(self) -> None:
        """Initialise v2 pipeline components: MOT, quality, RE-ID, identity binding."""
        self._mot_tracker      = BoTSORTTracker(BoTSORTConfig())
        self._face_quality     = FaceQualityAssessor()
        self._reid_engine      = REIDEngine(str(self._model_dir / "osnet_x1_0.onnx"))
        self._reid_engine.load()
        self._reid_gallery     = REIDGallery(self._reid_engine)
        self._identity_manager = IdentityTrackManager()
        self._temporal_liveness = TemporalLivenessTracker()

    def _load_face_detector(self) -> None:
        """
        Load SCRFD face detector from buffalo_l pack via InsightFace's native
        loader.  InsightFace appends 'models/' to the root path, so we pass the
        parent of the buffalo_l directory as root.
        """
        buffalo_dir = self._model_dir / "buffalo_l"
        det_model   = buffalo_dir / "det_10g.onnx"
        if not det_model.exists():
            logger.warning("buffalo_l/det_10g.onnx not found — face detection disabled")
            return
        try:
            from insightface.model_zoo.scrfd import SCRFD
            # InsightFace 0.7.x creates the ORT session in __init__ with no
            # providers (defaults to CPU).  We inject a CUDA session instead.
            cuda_opts: dict[str, Any] = {
                "device_id": self._device_id,
                "cudnn_conv_algo_search": "HEURISTIC",
                "do_copy_in_default_stream": True,
            }
            _providers = (
                [("TensorrtExecutionProvider", {
                    "device_id": self._device_id,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": self._cache_dir,
                    "trt_fp16_enable": True,
                }),
                 ("CUDAExecutionProvider", cuda_opts),
                 ("CPUExecutionProvider", {})]
                if _HAS_TRT else
                [("CUDAExecutionProvider", cuda_opts), ("CPUExecutionProvider", {})]
                if _HAS_CUDA else
                [("CPUExecutionProvider", {})]
            )
            _sess_opts = ort.SessionOptions()
            _sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            _sess_opts.log_severity_level = 3
            scrfd_sess = ort.InferenceSession(
                str(det_model), sess_options=_sess_opts, providers=_providers
            )
            det = SCRFD(model_file=str(det_model), session=scrfd_sess)
            det.prepare(
                ctx_id=self._device_id,
                det_thresh=self._FACE_CONF,
                input_size=self._FACE_DET_SIZE,
            )
            self._face_detector = det
            _active = scrfd_sess.get_providers()[0]
            logger.info("SCRFD face detector loaded (det_10g, thresh=%.2f, provider=%s)",
                        self._FACE_CONF, _active)
        except Exception as exc:
            logger.error("Failed to load SCRFD: %s", exc)

    def _load_embedder(self) -> None:
        p = self._model_dir / "adaface_ir101_webface12m.onnx"
        if not p.exists():
            logger.warning("adaface_ir101_webface12m.onnx not found — embedding disabled")
            return
        self._embed_sess = _make_session(str(p), self._device_id, self._cache_dir, fp16=True)
        self._embed_input = self._embed_sess.get_inputs()[0].name
        logger.info(
            "AdaFace IR-101 loaded  (FP16=True, layer_norm FP32 fallback=True — "
            "verify LFW accuracy with scripts/optimize_models.py if this is a new build)"
        )

    def _load_liveness(self) -> None:
        p = self._model_dir / "minifasnet_v2.onnx"
        if not p.exists():
            logger.warning("minifasnet_v2.onnx not found — liveness scoring disabled")
            return
        self._liveness_sess = _make_session(
            str(p), self._device_id, self._cache_dir, fp16=True
        )
        self._liveness_input = self._liveness_sess.get_inputs()[0].name

    def _load_demographics(self) -> None:
        """Optional age/gender head from buffalo_l/genderage.onnx."""
        p = self._model_dir / "buffalo_l" / "genderage.onnx"
        self._demographics = DemographicsEstimator(p)
        self._demographics.load(
            device_id=self._device_id,
            cache_dir=self._cache_dir,
            has_trt=_HAS_TRT,
            has_cuda=_HAS_CUDA,
        )

    def _load_pose_estimator(self) -> None:
        """Optional 3D pose head from buffalo_l/1k3d68.onnx."""
        p = self._model_dir / "buffalo_l" / "1k3d68.onnx"
        self._pose_estimator = PoseEstimator3D(p)
        self._pose_estimator.load(
            device_id=self._device_id,
            cache_dir=self._cache_dir,
            has_trt=_HAS_TRT,
            has_cuda=_HAS_CUDA,
        )

    # ── Person detection (YOLOv8l) ────────────────────────────────────────────

    def detect_persons(
        self,
        frame: np.ndarray,
        roi_rect: dict | None = None,
    ) -> list[PersonDetection]:
        """
        Run YOLOv8l on `frame`.

        YOLOv8l COCO detect output: [1, 84, 8400]
          - 84 = 4 bbox coords (cx,cy,w,h) + 80 COCO class scores
          - Person = COCO class 0 → column 4 after transposing

        Args:
            frame:    HxWx3 uint8 BGR image.
            roi_rect: Optional {"x": int, "y": int, "w": int, "h": int} mask.
                      Detections whose centre lies outside the ROI are dropped.
        Returns:
            List of PersonDetection sorted by confidence descending.
        """
        if self._yolo_sess is None:
            return []

        h, w = frame.shape[:2]
        img_lb, scale, (pad_h, pad_w) = _letterbox(frame, self._YOLO_SIZE)

        # BGR → RGB → float32 [0,1] NCHW
        inp = img_lb[:, :, ::-1].astype(np.float32) / 255.0
        inp = np.ascontiguousarray(inp.transpose(2, 0, 1)[None])   # [1,3,640,640]

        raw_out = self._yolo_sess.run(None, {self._yolo_input: inp})[0]  # [1, 84, 8400]
        raw = raw_out[0].T   # [8400, 84]

        # cols 0-3: cx,cy,w,h  |  col 4: person (COCO class 0) score
        confs = raw[:, 4]

        cx, cy, bw, bh = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]

        # xyxy in letterboxed space
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2
        boxes_lb = np.stack([x1, y1, x2, y2], axis=1)

        # Confidence filter
        mask = confs >= self._YOLO_CONF
        if not mask.any():
            return []
        boxes_lb = boxes_lb[mask]
        confs    = confs[mask]

        # NMS
        keep = _nms(boxes_lb, confs, self._YOLO_IOU)
        boxes_lb = boxes_lb[keep]
        confs    = confs[keep]

        # Scale from letterbox space → original pixel space
        def _unpad_x(x):
            return np.clip((x - pad_w) / scale, 0, w)
        def _unpad_y(y):
            return np.clip((y - pad_h) / scale, 0, h)

        boxes = np.stack([
            _unpad_x(boxes_lb[:, 0]),
            _unpad_y(boxes_lb[:, 1]),
            _unpad_x(boxes_lb[:, 2]),
            _unpad_y(boxes_lb[:, 3]),
        ], axis=1).astype(np.float32)

        results: list[PersonDetection] = []
        for i in range(len(boxes)):
            bb = boxes[i]
            cx_i = float((bb[0] + bb[2]) / 2)
            cy_i = float((bb[1] + bb[3]) / 2)

            # Minimum person height — reject tiny blobs that are unlikely to be real people
            bbox_h = float(bb[3] - bb[1])
            if bbox_h < self._YOLO_MIN_HEIGHT:
                continue

            # ROI filter
            if roi_rect is not None:
                rx, ry = roi_rect.get("x", 0), roi_rect.get("y", 0)
                rw, rh = roi_rect.get("w", w), roi_rect.get("h", h)
                if not (rx <= cx_i <= rx + rw and ry <= cy_i <= ry + rh):
                    continue

            results.append(PersonDetection(
                bbox=bb,
                conf=float(confs[i]),
                center=(cx_i, cy_i),
            ))

        results.sort(key=lambda p: p.conf, reverse=True)
        return results

    # ── Face detection (SCRFD / buffalo_l) ────────────────────────────────────

    def detect_faces(
        self,
        frame: np.ndarray,
        person_bboxes: list[np.ndarray] | None = None,
    ) -> list[FaceDetection]:
        """
        Detect faces in the frame using SCRFD.

        Args:
            frame:        HxWx3 uint8 BGR.
            person_bboxes: If provided, only return faces whose centre falls
                           within the upper 60 % of at least one person bbox.
        Returns:
            List of FaceDetection sorted by confidence descending.
        """
        if self._face_detector is None:
            return []

        # SCRFD.detect() signature: (img, input_size=None, max_num=0, metric='default')
        # det_thresh is already set via prepare() — do NOT pass thresh here.
        try:
            bboxes, kps = self._face_detector.detect(
                frame,
                input_size=self._FACE_DET_SIZE,
            )
        except Exception as exc:
            logger.error("SCRFD detect() failed: %s", exc)
            return []

        if bboxes is None or len(bboxes) == 0:
            return []

        # bboxes: [N, 5]  cols: x1,y1,x2,y2,score
        # kps:    [N, 10] cols: (x0,y0, x1,y1, x2,y2, x3,y3, x4,y4) — row-major 5pt
        results: list[FaceDetection] = []
        for i in range(len(bboxes)):
            bb   = bboxes[i, :4].astype(np.float32)
            conf = float(bboxes[i, 4])
            lm   = kps[i].reshape(5, 2).astype(np.float32) if kps is not None else None

            # Face centre
            fcx = float((bb[0] + bb[2]) / 2)
            fcy = float((bb[1] + bb[3]) / 2)

            # Person association filter
            if person_bboxes is not None:
                inside = False
                for pb in person_bboxes:
                    px1, py1, px2, py2 = pb[:4]
                    upper_y2 = py1 + (py2 - py1) * 0.70   # upper 70% of body (was 60%)
                    htol = (px2 - px1) * 0.15              # 15% horizontal tolerance
                    if (px1 - htol) <= fcx <= (px2 + htol) and py1 <= fcy <= upper_y2:
                        inside = True
                        break
                if not inside:
                    continue

            # inter-ocular distance: landmark[0]=left-eye, landmark[1]=right-eye
            iod = 0.0
            if lm is not None:
                iod = float(np.linalg.norm(lm[0] - lm[1]))

            results.append(FaceDetection(
                bbox=bb,
                landmarks=lm if lm is not None else np.zeros((5, 2), dtype=np.float32),
                conf=conf,
                inter_ocular_px=iod,
            ))

        results.sort(key=lambda f: f.conf, reverse=True)
        return results

    # ── Face alignment ────────────────────────────────────────────────────────

    # CLAHE instance (shared, not thread-safe but only called from GPU thread)
    _clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Super-resolution ONNX session (Real-ESRGAN x2, optional)
    _sr_sess: "ort.InferenceSession | None" = None

    # IOD threshold below which super-resolution upsampling is applied
    _SR_IOD_THRESHOLD = 60.0

    @staticmethod
    def estimate_yaw(landmarks: np.ndarray) -> float:
        """
        Estimate face yaw angle (degrees) from 5-point SCRFD landmarks.

        Landmarks layout:
          0 = left  eye, 1 = right eye, 2 = nose tip,
          3 = left  mouth corner, 4 = right mouth corner

        Method:
          For a frontal face the nose sits at the horizontal midpoint between
          the two eyes.  As yaw increases, the nose shifts toward the nearer
          eye.  The normalised nose-offset maps to ≈ ±45° for a profile face.

        Returns:
            Signed yaw in degrees.  Positive = face turned right (our right),
            negative = turned left.  0° = frontal.
        """
        le_x = float(landmarks[0, 0])   # left  eye x
        re_x = float(landmarks[1, 0])   # right eye x
        no_x = float(landmarks[2, 0])   # nose  tip x

        eye_span = re_x - le_x
        if eye_span < 1.0:
            return 0.0

        mid_x = (le_x + re_x) / 2.0
        # Normalised offset: 0 = frontal, ±1 = extreme profile
        norm_offset = (no_x - mid_x) / (eye_span / 2.0 + 1e-6)
        # Clamp and map to degrees (empirical linear mapping; real yaw is asin-shaped)
        norm_offset = max(-1.0, min(1.0, norm_offset))
        return float(norm_offset * 50.0)

    def _load_sr_model(self) -> None:
        """
        Optionally load a Real-ESRGAN x2 ONNX model for face super-resolution.

        Place the model at `{model_dir}/realesrgan_x2.onnx` to enable it.
        The model input must be [1, 3, H, W] RGB float32 in [0, 1].
        """
        p = self._model_dir / "realesrgan_x2.onnx"
        if not p.exists():
            return
        try:
            self._sr_sess = _make_session(str(p), self._device_id, self._cache_dir)
            logger.info("AIPipeline: Real-ESRGAN x2 loaded from %s", p)
        except Exception as exc:
            logger.warning("AIPipeline: could not load SR model: %s", exc)

    def _super_resolve(self, crop: np.ndarray) -> np.ndarray:
        """
        Upscale a small face crop by 2×.

        If the Real-ESRGAN ONNX session is loaded, uses the neural SR model.
        Otherwise falls back to LANCZOS4 + mild unsharp mask — still measurably
        better than INTER_CUBIC for faces with IOD < 60px.

        Args:
            crop: HxWx3 uint8 BGR face crop (any size).
        Returns:
            2× upscaled uint8 BGR crop.
        """
        H, W = crop.shape[:2]
        target_H, target_W = H * 2, W * 2

        if self._sr_sess is not None:
            try:
                # Preprocess: BGR→RGB, uint8→float32 [0,1], HWC→CHW, add batch
                rgb = crop[:, :, ::-1].astype(np.float32) / 255.0
                inp = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])  # [1,3,H,W]
                out = self._sr_sess.run(None, {self._sr_sess.get_inputs()[0].name: inp})[0]
                # out: [1,3,2H,2W] float32 [0,1]
                out_np = np.clip(out[0].transpose(1, 2, 0)[:, :, ::-1] * 255.0, 0, 255).astype(np.uint8)
                return cv2.resize(out_np, (target_W, target_H), interpolation=cv2.INTER_LINEAR)
            except Exception:
                pass  # fall through to classical SR

        # Classical SR: LANCZOS4 + unsharp mask
        up = cv2.resize(crop, (target_W, target_H), interpolation=cv2.INTER_LANCZOS4)
        # Unsharp mask: sharpened = original + strength * (original - blurred)
        blurred = cv2.GaussianBlur(up, (0, 0), sigmaX=1.5)
        return cv2.addWeighted(up, 1.5, blurred, -0.5, 0)

    def align_face(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        iod_px: float = 0.0,
    ) -> np.ndarray:
        """
        Warp the face to ArcFace canonical 112×112 crop using a partial affine
        transform estimated from the 5-point landmarks.

        Pipeline:
          1. If IOD < _SR_IOD_THRESHOLD: super-resolve the bounding region 2×
             (LANCZOS4 + unsharp mask, or Real-ESRGAN if model is loaded).
             This recovers texture lost in small/distant PTZ faces.
          2. Apply partial-affine similarity warp to 112×112 canonical grid.
          3. CLAHE per-channel in LAB space — normalises surveillance lighting.

        Yaw filtering is the caller's responsibility (see `estimate_yaw`).

        Args:
            frame:     HxWx3 uint8 BGR.
            landmarks: [5, 2] float32 (x, y) in pixel coordinates.
            iod_px:    Inter-ocular distance in pixels (from FaceDetection).
        Returns:
            [112, 112, 3] uint8 BGR face chip.
        """
        src_frame = frame
        src_lm    = landmarks.copy()

        # ── Super-resolve small faces ────────────────────────────────────────
        if iod_px > 0.0 and iod_px < self._SR_IOD_THRESHOLD:
            # Crop a generous padded region around the detected face, SR it,
            # then scale landmarks to match the upsampled crop's coordinate space.
            x1 = max(0, int(src_lm[:, 0].min()) - 20)
            y1 = max(0, int(src_lm[:, 1].min()) - 20)
            x2 = min(frame.shape[1], int(src_lm[:, 0].max()) + 20)
            y2 = min(frame.shape[0], int(src_lm[:, 1].max()) + 20)

            if x2 > x1 and y2 > y1:
                crop_2x = self._super_resolve(frame[y1:y2, x1:x2])
                # Place only the up-scaled crop on a minimal canvas sized to
                # the crop region — avoids allocating a full 2× frame (~16 MB).
                ch, cw = crop_2x.shape[:2]
                src_frame = np.zeros((y1 * 2 + ch, x1 * 2 + cw, 3), dtype=np.uint8)
                src_frame[y1 * 2 : y1 * 2 + ch,
                          x1 * 2 : x1 * 2 + cw] = crop_2x
                src_lm = src_lm * 2.0  # scale landmarks to 2× space

        # ── Affine warp to 112×112 ───────────────────────────────────────────
        M, _ = cv2.estimateAffinePartial2D(
            src_lm,
            _ARCFACE_DST,
            method=cv2.LMEDS,
        )
        if M is None:
            # Fallback: identity crop centred on landmarks mean
            cx = float(src_lm[:, 0].mean())
            cy = float(src_lm[:, 1].mean())
            half = 56
            x1f = max(0, int(cx - half))
            y1f = max(0, int(cy - half))
            chip = src_frame[y1f:y1f + 112, x1f:x1f + 112]
            chip = cv2.resize(chip, (112, 112), interpolation=cv2.INTER_CUBIC)
        else:
            chip = cv2.warpAffine(
                src_frame, M, (self._EMBED_SIZE, self._EMBED_SIZE),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )

        # ── CLAHE per-channel ────────────────────────────────────────────────
        # Normalises lighting/contrast differences between enrollment photos
        # (studio/phone) and PTZ camera (surveillance) conditions.
        lab = cv2.cvtColor(chip, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # ── Embedding (AdaFace IR-101) ─────────────────────────────────────────────

    def get_embedding(self, face_chip: np.ndarray) -> np.ndarray:
        """
        Extract a single 512-D L2-normalised embedding from a 112×112 face chip.
        For multiple faces use the batched `_embed_batch` path via process_frame.

        Args:
            face_chip: [112, 112, 3] uint8 BGR.
        Returns:
            [512] float32 L2-normalised embedding.
        """
        return self._embed_batch(face_chip[None])[0]

    def _embed_batch(self, chips: np.ndarray) -> np.ndarray:
        """
        Batch embedding forward pass.

        Args:
            chips: [N, 112, 112, 3] uint8 BGR.
        Returns:
            [N, 512] float32 L2-normalised.
        """
        if self._embed_sess is None or len(chips) == 0:
            return np.zeros((len(chips), 512), dtype=np.float32)

        # BGR → RGB, [0,255] → [-1,1], NHWC → NCHW in one fused contiguous pass
        inp = np.ascontiguousarray(
            chips[:, :, :, ::-1].transpose(0, 3, 1, 2).astype(np.float32)
        )
        inp -= 127.5
        inp /= 128.0

        emb = self._embed_sess.run(None, {self._embed_input: inp})[0]  # [N, 512]
        # L2 normalise
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
        return (emb / norms).astype(np.float32)

    # ── Liveness (MiniFASNetV2) ────────────────────────────────────────────────

    def check_liveness(self, frame: np.ndarray, face_bbox: np.ndarray) -> float:
        """
        Score a single face for liveness.

        Args:
            frame:     HxWx3 uint8 BGR — original full resolution frame.
            face_bbox: [x1, y1, x2, y2] face bbox in pixel coords.
        Returns:
            Float in [0, 1]; higher = more likely live.
        """
        if self._liveness_sess is None:
            return 1.0  # neutral default when model not loaded

        crop = self._crop_liveness(frame, face_bbox)
        if crop is None:
            return 1.0

        # BGR → float32, normalise to [-1, 1], NHWC → NCHW
        inp = crop.astype(np.float32)
        inp = (inp / 255.0 - 0.5) / 0.5
        inp = np.ascontiguousarray(inp.transpose(2, 0, 1)[None])   # [1, 3, 80, 80]

        logits = self._liveness_sess.run(None, {self._liveness_input: inp})[0]  # [1, 3]
        probs  = _softmax(logits[0])
        # If model outputs near-uniform probabilities the weights are not loaded
        # correctly — treat as live (disabled) rather than blocking all faces.
        if float(probs.max() - probs.min()) < 0.05:
            return 1.0
        return float(probs[self._LIVE_CLASS_IDX])

    def _crop_liveness(
        self, frame: np.ndarray, bbox: np.ndarray
    ) -> np.ndarray | None:
        """
        Expand face bbox by _LIVENESS_SCALE to give MiniFASNet enough context,
        pad at frame boundaries, resize to _LIVENESS_SIZE × _LIVENESS_SIZE.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox.astype(float)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw = (x2 - x1) * self._LIVENESS_SCALE / 2
        bh = (y2 - y1) * self._LIVENESS_SCALE / 2

        nx1 = max(0, int(cx - bw))
        ny1 = max(0, int(cy - bh))
        nx2 = min(w, int(cx + bw))
        ny2 = min(h, int(cy + bh))

        if nx2 <= nx1 or ny2 <= ny1:
            return None

        crop = frame[ny1:ny2, nx1:nx2]
        return cv2.resize(crop, (self._LIVENESS_SIZE, self._LIVENESS_SIZE))

    # ── Runtime face quality pre-screen ──────────────────────────────────────

    def _runtime_quality_check(
        self,
        frame: np.ndarray,
        face: "FaceDetection",
    ) -> tuple[bool, str]:
        """
        Fast CPU-only quality pre-screen applied to each detected face BEFORE
        the expensive GPU stages (alignment, ArcFace embedding, liveness).

        All checks are ordered cheapest-first so we exit early.

        Checks
        ------
        1. Detection confidence  — very cheap, just a float compare
        2. Bounding-box aspect ratio — rejects wide-angle distorted/irregular blobs
        3. Inter-ocular distance — computed by SCRFD, free
        4. Yaw angle             — 5 arithmetic ops on 5 landmarks
        5. Face crop sharpness   — one Laplacian on the small raw face crop

        Returns
        -------
        (True, "OK")                  — face is good enough for ArcFace
        (False, "<reason>:<value>")   — face rejected; reason explains why
        """
        # 1. Confidence
        if face.conf < self._RQ_MIN_CONF:
            return False, f"low_conf:{face.conf:.2f}"

        # 2. Bounding-box aspect ratio — rejects irregular blobs from wide-angle distortion
        bw = float(face.bbox[2] - face.bbox[0])
        bh = float(face.bbox[3] - face.bbox[1])
        if bh > 0:
            aspect = bw / bh
            if aspect < self._RQ_MIN_ASPECT or aspect > self._RQ_MAX_ASPECT:
                return False, f"bad_aspect:{aspect:.2f}"

        # 3. Inter-ocular distance — too far or too occluded
        iod = face.inter_ocular_px
        if iod < self._RQ_MIN_IOD:
            return False, f"too_small:{iod:.1f}px"

        # 4. Yaw — profile or heavily angled face
        yaw = self.estimate_yaw(face.landmarks)
        if abs(yaw) > self._RQ_MAX_YAW:
            return False, f"yaw:{yaw:.0f}deg"

        # 5. Sharpness on the raw (unaligned) face crop
        #    Aligned chip sharpness would be more accurate, but that costs the
        #    warp we are trying to avoid.  Raw crop variance correlates well.
        x1 = max(0, int(face.bbox[0]))
        y1 = max(0, int(face.bbox[1]))
        x2 = min(frame.shape[1], int(face.bbox[2]))
        y2 = min(frame.shape[0], int(face.bbox[3]))
        if x2 > x1 and y2 > y1:
            crop = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if sharpness < self._RQ_MIN_SHARPNESS:
                return False, f"blurry:{sharpness:.1f}"

        return True, "OK"

    # ── Full-frame processing ─────────────────────────────────────────────────

    def process_frame(
        self,
        frame: np.ndarray,
        roi_rect: dict | None = None,
        *,
        skip_liveness: bool = False,
        skip_faces: bool = False,
        quality_min: float | None = None,
    ) -> FrameResult:
        """
        Run the complete inference pipeline on one frame.

        Pipeline:
          1. Detect persons (YOLO) with optional ROI mask
          2. Detect faces (SCRFD) filtered to upper bodies of detected persons
          3. For each face: align → batch-embed (AdaFace) → score liveness
          4. Return structured result with per-stage latencies

        Degradation flags (set by GPUManager under VRAM pressure):
          skip_liveness: skip MiniFASNet — saves ~30ms per face
          skip_faces:    skip SCRFD+embed+liveness — YOLO-only mode
        """
        t_total = time.perf_counter()
        breakdown: dict[str, float] = {}

        # ── Stage 1: Person detection ─────────────────────────────────────────
        t0 = time.perf_counter()
        persons = self.detect_persons(frame, roi_rect)
        breakdown["yolo_ms"] = (time.perf_counter() - t0) * 1000

        if skip_faces:
            # MOT update with person detections only (no face data)
            tracked = self._update_mot_no_faces(persons, frame)
            total_ms = (time.perf_counter() - t_total) * 1000
            breakdown["total_ms"] = total_ms
            return FrameResult(
                persons=persons,
                faces_with_embeddings=[],
                tracked_persons=tracked,
                time_ms=total_ms,
                latency_breakdown=breakdown,
            )

        # ── Stage 2: Face detection ───────────────────────────────────────────
        t0 = time.perf_counter()
        person_bboxes = [p.bbox for p in persons] if persons else None
        faces = self.detect_faces(frame, person_bboxes)
        breakdown["scrfd_ms"] = (time.perf_counter() - t0) * 1000

        # ── Stage 2.5: v2 Face Quality Gate ───────────────────────────────────
        # Replace the old binary quality check with composite scoring.
        # - Faces passing RECOGNITION threshold → full ArcFace pipeline
        # - Faces passing TRACKING threshold only → track association but no recognition
        # - Faces below TRACKING threshold → discard
        t0 = time.perf_counter()
        faces_for_recognition: list[tuple[FaceDetection, FaceQualityScore]] = []
        faces_for_tracking:    list[tuple[FaceDetection, FaceQualityScore]] = []

        if self._face_quality is not None:
            # quality_min overrides the default recognition threshold —
            # used for retry passes where we want to accept lower-quality faces
            # rather than miss a recognition opportunity entirely.
            _recog_floor = quality_min if quality_min is not None else QUALITY_RECOGNITION_THRESH
            for face in faces:
                crop = self._crop_raw_face(frame, face.bbox)
                quality = self._face_quality.assess(crop, face.landmarks, face.bbox)
                if quality.composite >= _recog_floor:
                    faces_for_recognition.append((face, quality))
                elif quality.passes_tracking:
                    faces_for_tracking.append((face, quality))
                else:
                    logger.debug(
                        "process_frame: face discarded quality=%.2f iod=%.1fpx",
                        quality.composite, face.inter_ocular_px,
                    )
            # Also apply legacy hard-gate as secondary filter
            recog_passed = []
            for face, quality in faces_for_recognition:
                passes, reason = self._runtime_quality_check(frame, face)
                if passes:
                    recog_passed.append((face, quality))
                else:
                    logger.debug(
                        "process_frame: face downgraded to tracking-only [%s]", reason
                    )
                    faces_for_tracking.append((face, quality))
            faces_for_recognition = recog_passed
        else:
            # Fallback: use legacy quality check for all faces
            for face in faces:
                passes, reason = self._runtime_quality_check(frame, face)
                if passes:
                    # Synthesize a dummy quality score
                    dummy_q = _DummyQuality(face)
                    faces_for_recognition.append((face, dummy_q))
                else:
                    logger.debug(
                        "process_frame: face skipped [%s] iod=%.1fpx conf=%.2f",
                        reason, face.inter_ocular_px, face.conf,
                    )

        breakdown["quality_ms"] = (time.perf_counter() - t0) * 1000
        breakdown["quality_rejected"] = float(
            len(faces) - len(faces_for_recognition) - len(faces_for_tracking)
        )

        # ── Stage 3: Alignment + Embedding + Liveness (recognition-quality faces) ─
        t0 = time.perf_counter()
        chips:       list[np.ndarray]    = []
        valid_faces: list[FaceDetection] = []
        valid_quals: list[Any]           = []
        for face, quality in faces_for_recognition:
            chip = self.align_face(frame, face.landmarks, iod_px=face.inter_ocular_px)
            chips.append(chip)
            valid_faces.append(face)
            valid_quals.append(quality)
        breakdown["align_ms"] = (time.perf_counter() - t0) * 1000

        fwes: list[FaceWithEmbedding] = []
        if chips:
            # ── Stage 4: Batch embedding ──────────────────────────────────────
            t0 = time.perf_counter()
            chips_arr  = np.stack(chips, axis=0)   # [N, 112, 112, 3]
            embeddings = self._embed_batch(chips_arr)
            breakdown["adaface_ms"] = (time.perf_counter() - t0) * 1000

            # ── Stage 4b: Demographics (age/gender) ───────────────────────────
            t0 = time.perf_counter()
            if self._demographics is not None and self._demographics.ready:
                ages, genders = self._demographics.predict_batch(chips_arr)
            else:
                n = len(chips)
                ages    = np.full((n,), -1.0, dtype=np.float32)
                genders = np.full((n,), -1,   dtype=np.int8)
            breakdown["demog_ms"] = (time.perf_counter() - t0) * 1000

            # ── Stage 4c: 3D pose ────────────────────────────────────────────
            t0 = time.perf_counter()
            if self._pose_estimator is not None and self._pose_estimator.ready:
                yaws, pitches, rolls = self._pose_estimator.estimate_batch(chips_arr)
                # Refine quality scores with accurate 3D pose (replaces 5-pt heuristic)
                if self._face_quality is not None:
                    for i, q in enumerate(valid_quals):
                        if isinstance(q, FaceQualityScore):
                            self._face_quality.refine_with_3d_pose(
                                q, float(yaws[i]), float(pitches[i]),
                            )
            else:
                n = len(chips)
                yaws    = np.zeros((n,), dtype=np.float32)
                pitches = np.zeros((n,), dtype=np.float32)
                rolls   = np.zeros((n,), dtype=np.float32)
            breakdown["pose_ms"] = (time.perf_counter() - t0) * 1000

            # ── Stage 5: Liveness (single-frame) ─────────────────────────────
            t0 = time.perf_counter()
            for i, (face, chip, emb) in enumerate(zip(valid_faces, chips, embeddings)):
                liveness = 1.0 if skip_liveness else self.check_liveness(frame, face.bbox)
                fwes.append(FaceWithEmbedding(
                    face=face,
                    embedding=emb,
                    liveness=liveness,           # will be overwritten by temporal fusion
                    liveness_single=liveness,
                    face_chip=chip,
                    age=float(ages[i]),
                    gender=int(genders[i]),
                    yaw=float(yaws[i]),
                    pitch=float(pitches[i]),
                    roll=float(rolls[i]),
                ))
            breakdown["liveness_ms"] = (time.perf_counter() - t0) * 1000
        else:
            breakdown["adaface_ms"]  = 0.0
            breakdown["demog_ms"]    = 0.0
            breakdown["pose_ms"]     = 0.0
            breakdown["liveness_ms"] = 0.0

        # ── Stage 6: MOT update + identity binding ────────────────────────────
        t0 = time.perf_counter()
        tracked = self._update_mot_with_faces(
            persons, fwes, faces_for_recognition, faces_for_tracking,
            valid_quals, frame, preset_id=roi_rect.get("_preset_id", "") if roi_rect else "",
        )
        breakdown["mot_ms"] = (time.perf_counter() - t0) * 1000

        total_ms = (time.perf_counter() - t_total) * 1000
        breakdown["total_ms"] = total_ms

        n_faces_total = len(faces)
        n_rejected    = int(breakdown["quality_rejected"])
        logger.debug(
            "process_frame: %d persons, %d faces "
            "(%d recog, %d track-only, %d rejected) | "
            "YOLO=%.1f  SCRFD=%.1f  quality=%.1f  embed=%.1f(×%d)  "
            "live=%.1f  mot=%.1f  total=%.1fms",
            len(persons), n_faces_total,
            len(faces_for_recognition), len(faces_for_tracking), n_rejected,
            breakdown["yolo_ms"], breakdown["scrfd_ms"],
            breakdown["quality_ms"],
            breakdown.get("adaface_ms", 0), len(chips),
            breakdown.get("liveness_ms", 0), breakdown["mot_ms"], total_ms,
        )

        return FrameResult(
            persons=persons,
            faces_with_embeddings=fwes,
            tracked_persons=tracked,
            time_ms=total_ms,
            latency_breakdown=breakdown,
        )

    # ── v2 MOT integration helpers ─────────────────────────────────────────────

    def _crop_raw_face(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Crop the raw (unaligned) face region from a frame."""
        h, w = frame.shape[:2]
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(w, int(bbox[2]))
        y2 = min(h, int(bbox[3]))
        if x2 <= x1 or y2 <= y1:
            return np.zeros((40, 40, 3), dtype=np.uint8)
        return frame[y1:y2, x1:x2]

    def _crop_person(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Crop person bbox from frame, with small padding."""
        h, w = frame.shape[:2]
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(w, int(bbox[2]))
        y2 = min(h, int(bbox[3]))
        if x2 <= x1 or y2 <= y1:
            return np.zeros((64, 32, 3), dtype=np.uint8)
        return frame[y1:y2, x1:x2]

    def _find_face_for_track(
        self,
        track_bbox: np.ndarray,
        faces_with_quality: list[tuple[FaceDetection, Any]],
        tolerance_x: float = 0.20,
        upper_body_frac: float = 0.70,
    ) -> tuple[int, FaceDetection, Any] | None:
        """
        Find the best-matching face for a track bbox.
        Returns (index, face, quality) or None.
        """
        best_idx:   int | None    = None
        best_iou:   float         = 0.0
        best_face:  FaceDetection | None = None
        best_qual:  Any           = None

        px1, py1, px2, py2 = float(track_bbox[0]), float(track_bbox[1]), \
                              float(track_bbox[2]), float(track_bbox[3])
        upper_y2 = py1 + (py2 - py1) * upper_body_frac
        htol = (px2 - px1) * tolerance_x

        for idx, (face, quality) in enumerate(faces_with_quality):
            fcx = float((face.bbox[0] + face.bbox[2]) / 2)
            fcy = float((face.bbox[1] + face.bbox[3]) / 2)

            if not ((px1 - htol) <= fcx <= (px2 + htol) and py1 <= fcy <= upper_y2):
                continue

            # Use face-to-track IoU as a tiebreaker
            fx1, fy1, fx2, fy2 = face.bbox
            ix1 = max(px1, float(fx1))
            iy1 = max(py1, float(fy1))
            ix2 = min(px2, float(fx2))
            iy2 = min(py2, float(fy2))
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area_f = (fx2 - fx1) * (fy2 - fy1) + 1e-6
            iou = inter / area_f

            if iou > best_iou:
                best_iou   = iou
                best_idx   = idx
                best_face  = face
                best_qual  = quality

        if best_idx is None:
            return None
        return best_idx, best_face, best_qual

    def _update_mot_no_faces(
        self, persons: list[PersonDetection], frame: np.ndarray
    ) -> list[TrackedPerson]:
        """Update MOT with person detections only (skip_faces=True path)."""
        if self._mot_tracker is None:
            return []
        dets = [(p.bbox, p.conf) for p in persons]
        tracks = self._mot_tracker.update(dets, frame)

        result = []
        for track in tracks:
            state = None
            if self._identity_manager:
                state = self._identity_manager.get(track.track_id)
                self._identity_manager.touch_track(track.track_id)
            result.append(TrackedPerson(
                track_id=track.track_id,
                bbox=track.bbox,
                conf=track.conf,
                center=track.center,
                identity_id=state.identity_id if state else None,
                identity_confidence=state.identity_confidence if state else 0.0,
                tracking_method=state.tracking_method if state else "motion",
            ))
        return result

    def _update_mot_with_faces(
        self,
        persons:              list[PersonDetection],
        fwes:                 list[FaceWithEmbedding],
        faces_for_recognition: list[tuple[FaceDetection, Any]],
        faces_for_tracking:    list[tuple[FaceDetection, Any]],
        valid_quals:          list[Any],
        frame:                np.ndarray,
        preset_id:            str = "",
    ) -> list[TrackedPerson]:
        """
        Full v2 MOT + identity update.

        1. Optional RE-ID embedding extraction for person crops
        2. MOT update
        3. Per-track face association
        4. Identity binding from FaceWithEmbedding (caller must do FAISS lookup externally,
           so here we just store the FWE on TrackedPerson for ptz_brain to use)
        5. RE-ID gallery update for tracks without faces
        """
        if self._mot_tracker is None:
            # Fall back to simple PersonDetection wrapping
            return [
                TrackedPerson(
                    track_id=i,
                    bbox=p.bbox,
                    conf=p.conf,
                    center=p.center,
                    fwe=fwes[0] if fwes else None,
                )
                for i, p in enumerate(persons)
            ]

        # Optionally extract RE-ID embeddings for MOT cost
        reid_embs: np.ndarray | None = None
        if (self._reid_engine is not None
                and self._reid_engine._session is not None
                and len(persons) > 0):
            try:
                crops = [self._crop_person(frame, p.bbox) for p in persons]
                reid_embs = self._reid_engine.batch_extract(crops)
            except Exception:
                reid_embs = None

        # MOT update
        dets   = [(p.bbox, p.conf) for p in persons]
        tracks = self._mot_tracker.update(dets, frame, reid_embs)

        # All face detections (recognition + tracking quality)
        all_faces_with_quality: list[tuple[FaceDetection, Any]] = (
            faces_for_recognition + faces_for_tracking
        )

        # Build FWE lookup by face (for recognition-quality faces only)
        fwe_by_face: dict[int, FaceWithEmbedding] = {}
        for fwe in fwes:
            fid = id(fwe.face)
            fwe_by_face[fid] = fwe

        result: list[TrackedPerson] = []

        for track in tracks:
            # Touch / create identity state
            if self._identity_manager is not None:
                self._identity_manager.touch_track(track.track_id, preset_id)

            # Find best matching face for this track
            face_match = self._find_face_for_track(track.bbox, all_faces_with_quality)

            fwe:    FaceWithEmbedding | None = None
            quality: Any                     = None

            if face_match is not None:
                _, matched_face, quality = face_match
                fwe = fwe_by_face.get(id(matched_face))

                # Update quality in identity manager
                if self._identity_manager is not None and quality is not None:
                    self._identity_manager.update_face_quality(
                        track.track_id, quality.composite
                    )
            else:
                # No face — attempt RE-ID bridge
                if (self._reid_gallery is not None
                        and self._identity_manager is not None):
                    state = self._identity_manager.get(track.track_id)
                    if state and state.identity_id:
                        # Already have identity — confirm via RE-ID
                        person_crop = self._crop_person(frame, track.bbox)
                        match = self._reid_gallery.match(person_crop, preset_id)
                        if match:
                            self._identity_manager.bind_identity(
                                track.track_id,
                                match.identity_id,
                                match.confidence,
                                method="reid_bridge",
                                preset_id=preset_id,
                            )
                    elif state is None or not state.identity_id:
                        # Try to identify via RE-ID against all anchors
                        person_crop = self._crop_person(frame, track.bbox)
                        match = self._reid_gallery.match(person_crop, preset_id)
                        if match:
                            self._identity_manager.bind_identity(
                                track.track_id,
                                match.identity_id,
                                match.confidence,
                                method="reid_bridge",
                                preset_id=preset_id,
                            )

            # Get current identity state
            state: TrackIdentityState | None = None
            if self._identity_manager is not None:
                state = self._identity_manager.get(track.track_id)

            # Temporal liveness fusion (per track), only when we have a face
            # observation this frame — the tracker keeps per-track history so
            # a real face hit by a single bad MiniFASNet score still passes.
            if fwe is not None and self._temporal_liveness is not None:
                tscore = self._temporal_liveness.update(
                    track.track_id, fwe.liveness_single, fwe.face_chip,
                )
                fwe.liveness        = tscore.composite
                fwe.liveness_motion = tscore.motion

            result.append(TrackedPerson(
                track_id=track.track_id,
                bbox=track.bbox,
                conf=track.conf,
                center=track.center,
                identity_id=state.identity_id if state else None,
                identity_confidence=state.identity_confidence if state else 0.0,
                tracking_method=state.tracking_method if state else "motion",
                face_quality=quality,
                fwe=fwe,
                age=fwe.age if fwe is not None else -1.0,
                gender=fwe.gender if fwe is not None else -1,
            ))

        # Periodic GC of stale temporal-liveness state
        if self._temporal_liveness is not None:
            self._temporal_liveness.gc()

        return result

    def register_identity_for_track(
        self,
        track_id:    int,
        identity_id: str,
        confidence:  float,
        method:      str = "face",
        preset_id:   str = "",
        person_crop: np.ndarray | None = None,
    ) -> None:
        """
        Called by ptz_brain (or the sighting engine) after FAISS lookup succeeds.
        Binds the resolved identity to the MOT track and registers a RE-ID anchor.
        """
        if self._identity_manager is not None:
            self._identity_manager.bind_identity(
                track_id, identity_id, confidence, method, preset_id
            )
        if (person_crop is not None
                and self._reid_gallery is not None):
            self._reid_gallery.register_anchor(identity_id, track_id, person_crop, preset_id)

    def on_preset_change(self, old_preset: str, new_preset: str) -> None:
        """Notify pipeline of PTZ preset change (clears RE-ID gallery, resets MOT)."""
        if self._reid_gallery is not None:
            self._reid_gallery.on_preset_change(new_preset)
        if self._identity_manager is not None:
            self._identity_manager.on_preset_change(old_preset, new_preset)
        # Don't reset MOT on preset change — GMC handles the camera motion naturally


# ── Utility ────────────────────────────────────────────────────────────────────

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


class _DummyQuality:
    """
    Minimal quality-score proxy used when FaceQualityAssessor is not available.
    Passes the old hard-gate criteria as a composite=0.6 score so downstream
    code treats these faces as recognition-quality.
    """
    def __init__(self, face: "FaceDetection") -> None:
        self.composite              = 0.6
        self.face_size              = 0.6
        self.yaw                    = 0.6
        self.pitch                  = 0.6
        self.blur                   = 0.6
        self.illumination           = 0.6
        self.occlusion              = 0.6
        self.estimated_yaw_degrees  = 0.0
        self.estimated_pitch_degrees = 0.0
        self.passes_recognition     = True
        self.passes_tracking        = True
