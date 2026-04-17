"""
Commercial PTZ auto-tracker — Axis/Hikvision-style continuous multi-person
tracking engine.

Architecture
────────────
Two concurrently-running loops per camera:

  fast_loop  (≈25-30 fps)
    grab frame → AIPipeline(skip_faces=True) → BoT-SORT tracks →
    TargetSelector.select() → predictive aim → ONVIF continuous_move

  slow_loop  (3-5 fps, async)
    pop most-recent unrecognised track crop → AIPipeline.process_frame()
    (full recognition) → FaceRepository.identify() → TargetSelector
    learns "this track is identified"

The two loops share nothing mutable except the `TargetSelector` (which is
thread-safe for our single-producer single-consumer usage — reads from
fast loop, writes from slow loop are `set.add()` on disjoint sets) and a
bounded frame/crop queue.

Why two loops
─────────────
Face recognition (SCRFD + quality gate + ArcFace + liveness) is heavy —
on CPU it runs 200-400 ms per frame. Blocking the PTZ control loop on
that produces visible stutter. Detect+track is 30-60 ms on CPU so the
fast loop can keep the camera on target even when GPU is absent.

Recognition still happens; it just doesn't gate PTZ.

Public API
──────────
  AutoTracker(ptz_ctrl, decoder, pipeline, face_repo)
  await auto.start()                      # spawns both loops
  await auto.stop()                       # graceful shutdown
  auto.metrics() -> dict                  # per-loop p50/p99 + drop counters
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.services.ai_pipeline import AIPipeline, FrameResult
from app.services.mot_tracker import Track, TrackState
from app.services.onvif_controller import ONVIFController, PTZPosition
from app.services.track_priority import (
    FramingBox,
    TargetSelector,
    compute_group_framing,
)

logger = logging.getLogger(__name__)


# ── Tunable constants ─────────────────────────────────────────────────────

_FAST_LOOP_HZ          = 25.0   # target rate of the PTZ control loop
_SLOW_LOOP_HZ          = 4.0    # target rate of recognition passes
_RECOGNITION_QUEUE_MAX = 4      # bounded queue — drop oldest if slow loop lags

# Proportional control — kept low for commercial smoothness. The loop
# converges geometrically and a moderate gain avoids overshoot at wide
# zoom where a small ONVIF velocity still means a big pixel jump.
_KP_PAN  = 0.45
_KP_TILT = 0.45
_KP_ZOOM = 0.30

_DEAD_ZONE_FRAC = 0.03     # ignore errors smaller than 3% of frame half-dim
_MAX_VELOCITY   = 0.35     # cap normalised ONVIF velocity
_MIN_VELOCITY   = 0.03     # cam won't move below this — issue stop instead
_VEL_DEDUP_EPS  = 0.02     # drop continuous_move cmds that change little
_CMD_INTERVAL_S = 0.08     # min seconds between two continuous_move cmds

# Zoom policy — widen if primary body fraction < LOW, tighten if > HIGH.
_ZOOM_TARGET_BODY_FRAC = 0.40
_ZOOM_LOW              = 0.25
_ZOOM_HIGH             = 0.60

# Latency estimate used for predictive aim when no RTT has been measured.
_DEFAULT_CMD_LATENCY_S = 0.18

# Reliability
_WATCHDOG_INTERVAL_S   = 5.0
_MAX_FRAME_MISS_S      = 3.0    # no fresh frame → decoder restart
_MAX_ONVIF_FAIL_STREAK = 5


@dataclass
class _RecogJob:
    frame:     np.ndarray
    tracks:    list[Track]
    ts:        float


@dataclass
class _Metrics:
    fast_frames:  int = 0
    fast_dropped: int = 0
    slow_frames:  int = 0
    slow_dropped: int = 0
    onvif_cmds:   int = 0
    onvif_fails:  int = 0
    last_fast_ms: float = 0.0
    last_slow_ms: float = 0.0
    fast_p99_ms:  float = 0.0
    slow_p99_ms:  float = 0.0
    _fast_hist:   deque = field(default_factory=lambda: deque(maxlen=200))
    _slow_hist:   deque = field(default_factory=lambda: deque(maxlen=100))

    def record_fast(self, ms: float) -> None:
        self.last_fast_ms = ms
        self._fast_hist.append(ms)
        if self._fast_hist:
            self.fast_p99_ms = float(np.percentile(self._fast_hist, 99))

    def record_slow(self, ms: float) -> None:
        self.last_slow_ms = ms
        self._slow_hist.append(ms)
        if self._slow_hist:
            self.slow_p99_ms = float(np.percentile(self._slow_hist, 99))

    def snapshot(self) -> dict[str, Any]:
        return {
            "fast_frames":  self.fast_frames,
            "fast_dropped": self.fast_dropped,
            "slow_frames":  self.slow_frames,
            "slow_dropped": self.slow_dropped,
            "onvif_cmds":   self.onvif_cmds,
            "onvif_fails":  self.onvif_fails,
            "last_fast_ms": round(self.last_fast_ms, 1),
            "fast_p99_ms":  round(self.fast_p99_ms, 1),
            "last_slow_ms": round(self.last_slow_ms, 1),
            "slow_p99_ms":  round(self.slow_p99_ms, 1),
        }


class AutoTracker:
    """
    Commercial PTZ auto-tracker. One instance per camera.

    Dependencies are passed in to keep the engine testable — no global
    state, no direct DB reads. The FastAPI layer (ptz_brain) is responsible
    for instantiating it with the right controllers.
    """

    def __init__(
        self,
        ptz_ctrl:    ONVIFController,
        grab_frame:  "callable",       # async () -> np.ndarray | None
        pipeline:    AIPipeline,
        *,
        identify_face: "callable | None" = None,  # async (fwe) -> (pid, conf) | None
        frame_wh:    tuple[int, int] = (1920, 1080),
        max_secondaries: int = 3,
        enable_slow_loop: bool = True,
    ) -> None:
        self._ptz       = ptz_ctrl
        self._grab      = grab_frame
        self._pipe      = pipeline
        self._identify  = identify_face
        self._selector  = TargetSelector(
            frame_wh=frame_wh,
            max_secondaries=max_secondaries,
        )
        self._frame_wh  = frame_wh
        self._enable_slow = bool(enable_slow_loop)

        self._fast_task: asyncio.Task | None = None
        self._slow_task: asyncio.Task | None = None
        self._watchdog_task: asyncio.Task | None = None
        self._running = False

        self._recog_q: asyncio.Queue[_RecogJob] = asyncio.Queue(
            maxsize=_RECOGNITION_QUEUE_MAX,
        )
        self._last_cmd_ts = 0.0
        self._last_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._last_frame_ts = 0.0
        self._onvif_fail_streak = 0
        self._cmd_latency_s = _DEFAULT_CMD_LATENCY_S
        self._m = _Metrics()

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._fast_task     = asyncio.create_task(self._fast_loop(), name="auto_fast")
        self._watchdog_task = asyncio.create_task(self._watchdog(), name="auto_watchdog")
        if self._enable_slow:
            self._slow_task = asyncio.create_task(self._slow_loop(), name="auto_slow")
        logger.info("AutoTracker started (slow_loop=%s)", self._enable_slow)

    async def stop(self) -> None:
        self._running = False
        for task in (self._fast_task, self._slow_task, self._watchdog_task):
            if task and not task.done():
                task.cancel()
        for task in (self._fast_task, self._slow_task, self._watchdog_task):
            if task:
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        try:
            await self._ptz.stop()
        except Exception:
            pass
        logger.info("AutoTracker stopped — %s", self._m.snapshot())

    # ── Public helpers ────────────────────────────────────────────────────

    def metrics(self) -> dict[str, Any]:
        return self._m.snapshot()

    def mark_identified(self, track_id: int, person_id: Any) -> None:
        """Called by the slow loop when a track is successfully identified."""
        self._selector.mark_identified(track_id)
        logger.debug("track %d bound to identity %s", track_id, person_id)

    # ── Fast loop ─────────────────────────────────────────────────────────

    async def _fast_loop(self) -> None:
        period = 1.0 / _FAST_LOOP_HZ
        while self._running:
            cycle_start = time.monotonic()
            try:
                frame = await self._grab()
                if frame is None:
                    await asyncio.sleep(0.05)
                    continue
                self._last_frame_ts = time.monotonic()

                # Keep selector frame size in sync with whatever the decoder returns.
                fh, fw = frame.shape[:2]
                if (fw, fh) != self._frame_wh:
                    self._frame_wh = (fw, fh)
                    self._selector.set_frame_size(fw, fh)

                # Track-only pass — no face detection, no embeddings.
                result = await asyncio.to_thread(
                    self._pipe.process_frame, frame, None, skip_faces=True,
                )
                tracks = self._extract_confirmed_tracks(self._pipe, result)
                sel = self._selector.select(tracks)

                if sel.primary is None:
                    await self._coast_stop()
                else:
                    framing = compute_group_framing(
                        sel.primary, sel.secondaries, self._frame_wh,
                    )
                    await self._drive_ptz(sel.primary, framing)

                    # Forward the frame + tracks to the slow loop for recognition
                    # if any high-priority target is still unidentified.
                    if self._enable_slow and self._identify is not None:
                        unknown = [
                            t for t in [sel.primary, *sel.secondaries]
                            if t.track_id not in self._selector._identified
                        ]
                        if unknown:
                            self._enqueue_recognition(frame, unknown)

                self._m.fast_frames += 1
                self._m.record_fast((time.monotonic() - cycle_start) * 1000.0)

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._m.fast_dropped += 1
                logger.warning("fast_loop iter failed: %s", exc, exc_info=False)

            elapsed = time.monotonic() - cycle_start
            sleep_for = max(0.0, period - elapsed)
            await asyncio.sleep(sleep_for)

    # ── Slow loop ─────────────────────────────────────────────────────────

    async def _slow_loop(self) -> None:
        period = 1.0 / _SLOW_LOOP_HZ
        while self._running:
            cycle_start = time.monotonic()
            try:
                job: _RecogJob = await asyncio.wait_for(
                    self._recog_q.get(), timeout=period * 4,
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                raise

            try:
                result: FrameResult = await asyncio.to_thread(
                    self._pipe.process_frame, job.frame, None,
                )

                # Anything returning a face is eligible for identify().
                for fwe in result.faces_with_embeddings:
                    if self._identify is None:
                        break
                    ident = await self._identify(fwe)
                    if not ident:
                        continue
                    person_id, _conf = ident
                    # Best-effort: associate face bbox to the track it came from
                    # by IoU against the tracks we stored in the job.
                    tid = self._face_to_track(fwe.face.bbox, job.tracks)
                    if tid is not None:
                        self.mark_identified(tid, person_id)
                        self._selector.mark_face_seen(tid)

                self._m.slow_frames += 1
                self._m.record_slow((time.monotonic() - cycle_start) * 1000.0)

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._m.slow_dropped += 1
                logger.warning("slow_loop recognition failed: %s", exc, exc_info=False)

    # ── Watchdog ──────────────────────────────────────────────────────────

    async def _watchdog(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(_WATCHDOG_INTERVAL_S)
                now = time.monotonic()
                if self._last_frame_ts > 0 and (now - self._last_frame_ts) > _MAX_FRAME_MISS_S:
                    logger.error(
                        "AutoTracker watchdog: no frames for %.1fs — stopping PTZ",
                        now - self._last_frame_ts,
                    )
                    try:
                        await self._ptz.stop()
                    except Exception:
                        pass
                if self._onvif_fail_streak >= _MAX_ONVIF_FAIL_STREAK:
                    logger.error(
                        "AutoTracker watchdog: %d consecutive ONVIF failures — "
                        "attempting reconnect",
                        self._onvif_fail_streak,
                    )
                    try:
                        await self._ptz.connect()
                        self._onvif_fail_streak = 0
                    except Exception as exc:
                        logger.warning("ONVIF reconnect failed: %s", exc)
            except asyncio.CancelledError:
                raise

    # ── Internals ─────────────────────────────────────────────────────────

    @staticmethod
    def _extract_confirmed_tracks(
        pipeline: AIPipeline, result: FrameResult,
    ) -> list[Track]:
        """
        `FrameResult.tracked_persons` lacks Kalman state (velocity, P), so
        cross-reference its track_ids against the pipeline's raw MOT tracker
        list to get the underlying `Track` objects needed for predictive aim.
        """
        mot = getattr(pipeline, "_mot_tracker", None)
        if mot is None:
            return []
        valid_ids = {tp.track_id for tp in result.tracked_persons}
        return [
            t for t in mot._tracks
            if t.track_id in valid_ids and t.state == TrackState.CONFIRMED
        ]

    @staticmethod
    def _face_to_track(face_bbox: np.ndarray, tracks: list[Track]) -> int | None:
        """Best-IoU association of a face bbox back to its parent track."""
        fx1, fy1, fx2, fy2 = [float(v) for v in face_bbox]
        fcx, fcy = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0
        best_id, best_d = None, 1e18
        for t in tracks:
            tcx, tcy = t.center
            d = (tcx - fcx) ** 2 + (tcy - fcy) ** 2
            if d < best_d:
                best_d, best_id = d, t.track_id
        return best_id

    def _enqueue_recognition(self, frame: np.ndarray, tracks: list[Track]) -> None:
        try:
            self._recog_q.put_nowait(
                _RecogJob(frame=frame.copy(), tracks=list(tracks), ts=time.monotonic()),
            )
        except asyncio.QueueFull:
            # Drop the oldest — always process the freshest frame.
            try:
                _ = self._recog_q.get_nowait()
                self._m.slow_dropped += 1
                self._recog_q.put_nowait(
                    _RecogJob(frame=frame.copy(), tracks=list(tracks), ts=time.monotonic()),
                )
            except Exception:
                pass

    # ── PTZ control ───────────────────────────────────────────────────────

    async def _drive_ptz(self, primary: Track, framing: FramingBox) -> None:
        """
        Translate a framing target (pixels) into an ONVIF continuous_move
        velocity command with predictive lead and command de-duplication.
        """
        fw, fh = self._frame_wh
        # Predict where `primary` will be by the time the camera has moved.
        vx, vy = float(primary.x[4]), float(primary.x[5])
        lead = self._cmd_latency_s
        target_cx = framing.cx + vx * lead * (1.0 / 0.033)   # vx is px/frame; ~30 fps
        target_cy = framing.cy + vy * lead * (1.0 / 0.033)

        # Pixel error, normalised to [-1, 1] half-frame.
        err_x = (target_cx - fw / 2.0) / (fw / 2.0)
        err_y = (target_cy - fh / 2.0) / (fh / 2.0)

        # Dead zone — skip tiny corrections that would just jitter the cam.
        if abs(err_x) < _DEAD_ZONE_FRAC and abs(err_y) < _DEAD_ZONE_FRAC:
            # Also handle zoom if needed.
            zoom_vel = self._zoom_velocity(primary)
            if abs(zoom_vel) < _MIN_VELOCITY:
                await self._coast_stop()
                return
            await self._send_continuous_move(0.0, 0.0, zoom_vel)
            return

        # Proportional with velocity saturation. Tilt is inverted because
        # ONVIF positive tilt = up but positive pixel-y = down.
        pan_vel  = max(-_MAX_VELOCITY, min(_MAX_VELOCITY, _KP_PAN  *  err_x))
        tilt_vel = max(-_MAX_VELOCITY, min(_MAX_VELOCITY, _KP_TILT * -err_y))

        # Scale velocity by zoom — at narrow FOV the same pixel error is a
        # much smaller angular error, so velocity should drop accordingly.
        try:
            zoom_now = float((await self._safe_get_ptz()).zoom)
        except Exception:
            zoom_now = 0.0
        zoom_scale = max(0.2, 1.0 - 0.6 * zoom_now)
        pan_vel  *= zoom_scale
        tilt_vel *= zoom_scale

        zoom_vel = self._zoom_velocity(primary)

        await self._send_continuous_move(pan_vel, tilt_vel, zoom_vel)

    def _zoom_velocity(self, primary: Track) -> float:
        """Return a zoom velocity (-MAX..+MAX) driving body-frac toward target."""
        _, fh = self._frame_wh
        h = max(1.0, float(primary.x[3]))
        body_frac = h / float(fh)
        if body_frac < _ZOOM_LOW:
            # Too small — zoom in.
            err = (_ZOOM_TARGET_BODY_FRAC - body_frac) / _ZOOM_TARGET_BODY_FRAC
            return max(_MIN_VELOCITY, min(_MAX_VELOCITY, _KP_ZOOM * err))
        if body_frac > _ZOOM_HIGH:
            # Too large — zoom out.
            err = (body_frac - _ZOOM_TARGET_BODY_FRAC) / _ZOOM_TARGET_BODY_FRAC
            return -max(_MIN_VELOCITY, min(_MAX_VELOCITY, _KP_ZOOM * err))
        return 0.0

    async def _send_continuous_move(self, pan: float, tilt: float, zoom: float) -> None:
        """
        Send a continuous_move with dedup + RTT tracking.

        Dedup: skip if all three velocities are within _VEL_DEDUP_EPS of the
        previous command AND less than _CMD_INTERVAL_S has passed. Commercial
        pan/tilt firmwares queue or merge rapid identical velocities in ways
        that produce visible jitter — the dedup avoids that entirely.
        """
        now = time.monotonic()
        dpan  = pan  - self._last_vel[0]
        dtilt = tilt - self._last_vel[1]
        dzoom = zoom - self._last_vel[2]
        close = (abs(dpan) < _VEL_DEDUP_EPS
                 and abs(dtilt) < _VEL_DEDUP_EPS
                 and abs(dzoom) < _VEL_DEDUP_EPS)
        if close and (now - self._last_cmd_ts) < _CMD_INTERVAL_S:
            return

        # timeout=0.5s means firmware auto-stops if we stop feeding it —
        # critical safety net if this process dies mid-pan.
        t_send = time.monotonic()
        try:
            await self._ptz.continuous_move(pan, tilt, zoom, timeout=0.5)
            self._onvif_fail_streak = 0
            self._m.onvif_cmds += 1
        except Exception as exc:
            self._onvif_fail_streak += 1
            self._m.onvif_fails += 1
            logger.debug("continuous_move failed: %s", exc)
            return

        rtt = time.monotonic() - t_send
        # EMA RTT for predictive aim.
        self._cmd_latency_s = 0.8 * self._cmd_latency_s + 0.2 * rtt
        self._last_vel = (pan, tilt, zoom)
        self._last_cmd_ts = now

    async def _coast_stop(self) -> None:
        """Stop only if we weren't already stopped — avoids spamming Stop."""
        if self._last_vel == (0.0, 0.0, 0.0):
            return
        try:
            await self._ptz.stop()
            self._last_vel = (0.0, 0.0, 0.0)
        except Exception as exc:
            logger.debug("stop() failed: %s", exc)

    async def _safe_get_ptz(self) -> PTZPosition:
        try:
            return await self._ptz.get_ptz_status()
        except Exception:
            return PTZPosition()
