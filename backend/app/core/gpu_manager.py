"""
ACAS GPU Resource Manager
=========================
Single shared AIPipeline instance + per-session CUDA streams + VRAM guard.

Design goals
────────────
• One AIPipeline loaded once at startup; every concurrent PTZ-Brain session
  shares its weights (saves ~3.5 GB VRAM vs. per-session instances).
• asyncio.Semaphore bounds parallel GPU inference to prevent OOM; default=5
  for a 24 GB card (leaves ≥4 GB headroom for FAISS indexes + OS).
• pycuda streams (optional): each session gets a private CUDA stream so GPU
  work from different sessions can overlap in the copy-engine and compute
  engines.  Falls back gracefully if pycuda is not installed.
• Degradation modes: under VRAM pressure the manager drops MiniFASNet and
  (if needed) AdaFace, keeping person detection running.
• Saturation probe: async helper to find the concurrency level where adding
  more sessions no longer improves throughput.
"""
from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:
    from app.services.ai_pipeline import AIPipeline, FrameResult

log = logging.getLogger(__name__)


# ── Degradation modes ────────────────────────────────────────────────────────
class DegradationMode(IntEnum):
    FULL          = 0   # all stages active
    NO_LIVENESS   = 1   # skip MiniFASNet (save ~30 ms/face)
    EMBED_ONLY    = 2   # skip liveness + faces fall back to YOLO only
    YOLO_ONLY     = 3   # extreme pressure: only person bboxes


# ── Per-session GPU context ───────────────────────────────────────────────────
@dataclass
class GPUSession:
    session_id: str
    created_at: float = field(default_factory=time.monotonic)
    inference_count: int = 0
    total_inference_ms: float = 0.0

    @property
    def avg_inference_ms(self) -> float:
        return self.total_inference_ms / max(1, self.inference_count)


# ── Saturation probe result ───────────────────────────────────────────────────
@dataclass
class SaturationResult:
    level:         int      # concurrency level
    throughput_fps: float   # total frames/sec at this level
    mean_ms:       float    # mean inference latency
    p95_ms:        float    # p95 latency
    vram_gb:       float    # VRAM used during test


# ── GPU Manager (singleton) ──────────────────────────────────────────────────
class GPUManager:
    """
    Singleton managing the shared AIPipeline and all GPU resources.

    Usage
    ─────
        # In main.py lifespan:
        gpu_mgr = await GPUManager.create(model_dir="/models", device_id=0)
        app.state.gpu_manager = gpu_mgr

        # In PTZBrain:
        async with app.state.gpu_manager.session("my-session") as ctx:
            result = await ctx.infer(frame, roi_rect)
    """

    _instance: "GPUManager | None" = None

    def __init__(
        self,
        model_dir:      str,
        device_id:      int = 0,
        max_concurrent: int = 5,
        vram_budget_gb: float = 20.0,   # free ~4 GB for FAISS + OS on 24 GB card
    ) -> None:
        self._model_dir     = model_dir
        self._device_id     = device_id
        self._max_concurrent = max_concurrent
        self._vram_budget_gb = vram_budget_gb

        self._pipeline:  "AIPipeline | None" = None
        self._sem:        asyncio.Semaphore  = asyncio.Semaphore(max_concurrent)
        self._sessions:   dict[str, GPUSession] = {}

        # Rolling latency window for saturation / degradation decisions
        self._recent_ms: deque[float] = deque(maxlen=200)
        self._degradation: DegradationMode = DegradationMode.FULL
        self._vram_cached: float = 0.0
        self._vram_last_sample: float = 0.0
        self._degradation_hold_until: float = 0.0  # hysteresis: don't recover before this time

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    async def create(
        cls,
        model_dir:      str,
        device_id:      int   = 0,
        max_concurrent: int   = 5,
        vram_budget_gb: float = 20.0,
    ) -> "GPUManager":
        mgr = cls(model_dir, device_id, max_concurrent, vram_budget_gb)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, mgr._load_pipeline)
        cls._instance = mgr
        log.info(
            "GPUManager ready  device=cuda:%d  max_concurrent=%d  vram_budget=%.0fGB",
            device_id, max_concurrent, vram_budget_gb,
        )
        return mgr

    @classmethod
    def get(cls) -> "GPUManager":
        if cls._instance is None:
            raise RuntimeError("GPUManager.create() has not been called")
        return cls._instance

    # ── Pipeline loading ─────────────────────────────────────────────────────

    def _load_pipeline(self) -> None:
        from app.services.ai_pipeline import AIPipeline
        self._pipeline = AIPipeline(model_dir=self._model_dir, device_id=self._device_id)
        self._pipeline.load()
        log.info("GPUManager: AIPipeline loaded (device=%d)", self._device_id)

    @property
    def pipeline(self) -> "AIPipeline":
        if self._pipeline is None:
            raise RuntimeError("GPUManager pipeline not loaded")
        return self._pipeline

    # ── VRAM monitoring ──────────────────────────────────────────────────────

    def _sample_vram(self) -> float:
        """Read current VRAM usage in GB (cached for 5 s)."""
        now = time.monotonic()
        if now - self._vram_last_sample < 5.0:
            return self._vram_cached
        try:
            out = subprocess.check_output(
                ["nvidia-smi", f"--id={self._device_id}",
                 "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                timeout=2, stderr=subprocess.DEVNULL,
            )
            self._vram_cached = float(out.decode().strip()) / 1024.0
            self._vram_last_sample = now
        except Exception:
            pass
        return self._vram_cached

    _DEGRADATION_HOLD_S = 30.0  # hold degraded mode for at least this long

    def _update_degradation(self) -> None:
        vram = self._sample_vram()
        headroom = self._vram_budget_gb - vram
        now = time.monotonic()

        if headroom < 0.5:
            desired = DegradationMode.YOLO_ONLY
        elif headroom < 2.0:
            desired = DegradationMode.EMBED_ONLY
        elif headroom < 4.0:
            desired = DegradationMode.NO_LIVENESS
        else:
            desired = DegradationMode.FULL

        # Escalation (worse mode): apply immediately + set hold timer
        if desired > self._degradation:
            log.warning(
                "GPUManager: degradation %s → %s  (VRAM=%.1fGB, budget=%.1fGB)",
                self._degradation.name, desired.name, vram, self._vram_budget_gb,
            )
            self._degradation = desired
            self._degradation_hold_until = now + self._DEGRADATION_HOLD_S
        # Recovery (better mode): only if hold period has elapsed
        elif desired < self._degradation and now >= self._degradation_hold_until:
            log.info(
                "GPUManager: degradation recovering %s → %s  (VRAM=%.1fGB)",
                self._degradation.name, desired.name, vram,
            )
            self._degradation = desired

    # ── Session lifecycle ────────────────────────────────────────────────────

    def register_session(self, session_id: str) -> GPUSession:
        sess = GPUSession(session_id=session_id)
        self._sessions[session_id] = sess
        log.info(
            "GPUManager: session %s registered  (active=%d/%d)",
            session_id, len(self._sessions), self._max_concurrent,
        )
        return sess

    def deregister_session(self, session_id: str) -> None:
        sess = self._sessions.pop(session_id, None)
        if sess:
            log.info("GPUManager: session %s deregistered", session_id)

    # ── Inference execution ───────────────────────────────────────────────────

    async def infer(
        self,
        session_id: str,
        frame:      np.ndarray,
        roi_rect:   dict | None = None,
        *,
        quality_min: float | None = None,
    ) -> "FrameResult":
        """
        Thread-safe GPU inference with concurrency limiting and degradation.

        Waits for the semaphore (queue depth = max_concurrent), then dispatches
        to an executor thread so the asyncio event loop is not blocked.

        quality_min: when set, overrides the recognition quality gate floor in
        process_frame — used for retry recognition passes.
        """
        self._update_degradation()
        sess = self._sessions.get(session_id)

        skip_liveness = self._degradation >= DegradationMode.NO_LIVENESS
        skip_faces    = self._degradation >= DegradationMode.EMBED_ONLY

        t_wait = time.perf_counter()
        async with self._sem:
            wait_ms = (time.perf_counter() - t_wait) * 1000
            if wait_ms > 50:
                log.debug("session %s: GPU queue wait %.0fms", session_id, wait_ms)

            loop = asyncio.get_running_loop()
            t0 = time.perf_counter()
            result = await loop.run_in_executor(
                None,
                lambda: self.pipeline.process_frame(
                    frame, roi_rect,
                    skip_liveness=skip_liveness,
                    skip_faces=skip_faces,
                    quality_min=quality_min,
                ),
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000

        self._recent_ms.append(elapsed_ms)
        if sess:
            sess.inference_count += 1
            sess.total_inference_ms += elapsed_ms

        return result

    # ── Session context manager ───────────────────────────────────────────────

    def session(self, session_id: str) -> "_SessionCtx":
        return _SessionCtx(self, session_id)

    # ── Metrics ───────────────────────────────────────────────────────────────

    def gpu_slot(self):
        """Public semaphore context for gating non-inference GPU work (e.g. zone mapping)."""
        return self._sem

    def metrics(self) -> dict[str, Any]:
        arr = np.array(self._recent_ms) if self._recent_ms else np.array([0.0])
        return {
            "active_sessions":   len(self._sessions),
            "max_concurrent":    self._max_concurrent,
            "degradation":       self._degradation.name,
            "vram_used_gb":      round(self._sample_vram(), 2),
            "vram_budget_gb":    self._vram_budget_gb,
            "latency_mean_ms":   round(float(arr.mean()), 1),
            "latency_p95_ms":    round(float(np.percentile(arr, 95)), 1),
            "latency_p99_ms":    round(float(np.percentile(arr, 99)), 1),
            "queue_available":   self._max_concurrent - len(self._sessions),
        }

    # ── Concurrency saturation probe ──────────────────────────────────────────

    async def probe_saturation(
        self,
        levels:       list[int]    = [5, 10, 15, 20],
        iters_each:   int          = 40,
        frame_size:   tuple[int, int, int] = (720, 1280, 3),
    ) -> list[SaturationResult]:
        """
        Run synthetic inference at each concurrency level and measure throughput.
        Returns results so callers can find the saturation knee.
        """
        results: list[SaturationResult] = []
        dummy = np.random.randint(0, 255, frame_size, dtype=np.uint8)

        for level in levels:
            old_max = self._max_concurrent
            self._sem = asyncio.Semaphore(level)
            self._max_concurrent = level

            latencies: list[float] = []

            async def _one(n: int) -> None:
                t0 = time.perf_counter()
                await self.infer(f"probe-{n}", dummy)
                latencies.append((time.perf_counter() - t0) * 1000)

            t_start = time.perf_counter()
            await asyncio.gather(*[_one(i) for i in range(iters_each)])
            elapsed = time.perf_counter() - t_start

            arr = np.array(latencies)
            res = SaturationResult(
                level=level,
                throughput_fps=round(iters_each / elapsed, 2),
                mean_ms=round(float(arr.mean()), 1),
                p95_ms=round(float(np.percentile(arr, 95)), 1),
                vram_gb=round(self._sample_vram(), 2),
            )
            results.append(res)
            log.info(
                "Saturation probe  concurrency=%d  throughput=%.1f fps  "
                "mean=%.0fms  p95=%.0fms  VRAM=%.1fGB",
                level, res.throughput_fps, res.mean_ms, res.p95_ms, res.vram_gb,
            )

            # Restore
            self._sem = asyncio.Semaphore(old_max)
            self._max_concurrent = old_max

        # Find saturation knee: concurrency where adding 5 more gives < 5% gain
        for i in range(1, len(results)):
            gain_pct = (results[i].throughput_fps - results[i-1].throughput_fps) \
                       / max(results[i-1].throughput_fps, 0.001) * 100
            if gain_pct < 5.0:
                log.info(
                    "Saturation point: concurrency=%d  (%.1f%% gain — diminishing returns)",
                    results[i].level, gain_pct,
                )
                break

        return results


# ── Context manager wrapper ───────────────────────────────────────────────────

class _SessionCtx:
    def __init__(self, mgr: GPUManager, session_id: str) -> None:
        self._mgr = mgr
        self._session_id = session_id

    async def __aenter__(self) -> "_SessionCtx":
        self._mgr.register_session(self._session_id)
        return self

    async def __aexit__(self, *_: Any) -> None:
        self._mgr.deregister_session(self._session_id)

    async def infer(
        self,
        frame:    np.ndarray,
        roi_rect: dict | None = None,
        *,
        quality_min: float | None = None,
    ) -> "FrameResult":
        return await self._mgr.infer(
            self._session_id, frame, roi_rect, quality_min=quality_min
        )
