"""
RTSPDecoder — hardware-accelerated RTSP frame source using FFmpeg + NVDEC.

The decoder spawns an ffmpeg subprocess with NVDEC (cuvid) hardware decode and
pipes raw BGR24 frames into a numpy array.  An async generator interface exposes
frames as (frame_id, timestamp_s, np.ndarray).

Auto-reconnect is handled internally: on any pipe error or frame gap > timeout,
the subprocess is killed and restarted transparently.

Usage:
    decoder = RTSPDecoder("rtsp://cam/stream", fps=25)
    async for frame_id, ts, frame in decoder.frames():
        process(frame)  # frame is HxWx3 uint8 BGR

    # Health check (safe to call from any async context):
    health = decoder.health
    print(health.fps_actual, health.frames_decoded)
"""
from __future__ import annotations

import asyncio
import ctypes
import os
import logging
import shutil
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

import numpy as np

logger = logging.getLogger(__name__)


def _cuda_available() -> bool:
    """Return True only if libcuda.so.1 is present AND cuInit() succeeds."""
    try:
        lib = ctypes.CDLL("libcuda.so.1")
        lib.cuInit.restype = ctypes.c_int
        return lib.cuInit(0) == 0
    except (OSError, AttributeError):
        return False


_HAS_CUDA: bool = _cuda_available()
if not _HAS_CUDA:
    logger.info(
        "RTSPDecoder: CUDA driver not available — hardware (NVDEC) decode disabled; "
        "all streams will use software (CPU) decode."
    )

# ── Health data ───────────────────────────────────────────────────────────────


@dataclass
class DecoderHealth:
    frames_decoded: int = 0
    fps_actual: float = 0.0
    last_frame_time: float = 0.0   # epoch seconds; 0 means no frame yet
    reconnect_count: int = 0
    running: bool = False


# ── Decoder ───────────────────────────────────────────────────────────────────


class RTSPDecoder:
    """
    Hardware-accelerated RTSP decoder via FFmpeg + NVDEC.

    Parameters
    ----------
    rtsp_url:
        Full RTSP URL, e.g. ``rtsp://admin:pass@192.168.1.10:554/stream1``.
    fps:
        Target decode rate.  Frames are throttled (by `-vf fps=…`) to this
        value before being piped out, keeping CPU/GPU load predictable.
    width / height:
        If set, output frames are scaled to this resolution.  Leave as None
        to use the native stream resolution.
    reconnect_delay:
        Seconds to wait between reconnect attempts.
    frame_timeout:
        Seconds without a frame before a reconnect is forced.
    gpu_device:
        CUDA device index used with cuvid.  0 for the first GPU.
    use_hw_decode:
        Set to False to fall back to CPU decode (useful for testing).
    tcp_transport:
        Use RTSP over TCP instead of UDP.  More reliable on lossy networks.
    """

    _FFMPEG_CMD_TEMPLATE = (
        "ffmpeg "
        "-loglevel error "
        "{transport} "
        "{hw_input} "
        "-i {url} "
        "{scale} "
        "-vf fps={fps} "
        "-f rawvideo "
        "-pix_fmt bgr24 "
        "pipe:1"
    )

    # Map ffprobe codec_name → NVIDIA NVDEC decoder name
    _CUVID_MAP: dict[str, str] = {
        "h264":       "h264_cuvid",
        "hevc":       "hevc_cuvid",
        "h265":       "hevc_cuvid",   # alias some cameras report
        "vp9":        "vp9_cuvid",
        "mpeg2video": "mpeg2_cuvid",
        "mpeg4":      "mpeg4_cuvid",
    }

    # Maximum decode resolution.  Streams wider than this are downscaled by
    # the ffmpeg -vf filter before being piped out.  YOLOv8n was trained at
    # 640 px; anything above 1920×1080 wastes GPU memory without improving
    # detection accuracy.  Override via env RTSP_MAX_DECODE_WIDTH.
    _MAX_DECODE_WIDTH:  int = int(os.getenv("RTSP_MAX_DECODE_WIDTH",  "1920"))
    _MAX_DECODE_HEIGHT: int = int(os.getenv("RTSP_MAX_DECODE_HEIGHT", "1080"))

    def __init__(
        self,
        rtsp_url: str,
        fps: float = 25.0,
        *,
        width: int | None = None,
        height: int | None = None,
        reconnect_delay: float = 2.0,
        frame_timeout: float = 10.0,
        gpu_device: int = 0,
        use_hw_decode: bool = True,
        tcp_transport: bool = True,
    ) -> None:
        self._url = rtsp_url
        self._fps = fps
        self._width = width
        self._height = height
        self._reconnect_delay = reconnect_delay
        self._frame_timeout = frame_timeout
        self._codec_name: str = ""   # filled by _preflight_probe; used in _build_cmd
        self._gpu_device = gpu_device
        # Honour caller's use_hw_decode=False; also force CPU when no CUDA driver.
        self._use_hw_decode = use_hw_decode and _HAS_CUDA
        self._tcp_transport = tcp_transport

        self._health = DecoderHealth()
        self._proc: asyncio.subprocess.Process | None = None
        self._frame_size: int = 0
        self._frame_shape: tuple[int, int, int] = (0, 0, 3)
        self._stop_event = asyncio.Event()

        # FPS tracking
        self._fps_bucket_start: float = 0.0
        self._fps_bucket_count: int = 0

        # Background stderr drainer — prevents the OS pipe buffer from filling
        # and blocking ffmpeg stdout when the camera emits warnings/errors.
        self._stderr_drain_task: asyncio.Task | None = None

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def health(self) -> DecoderHealth:
        return DecoderHealth(
            frames_decoded=self._health.frames_decoded,
            fps_actual=self._health.fps_actual,
            last_frame_time=self._health.last_frame_time,
            reconnect_count=self._health.reconnect_count,
            running=self._health.running,
        )

    async def stop(self) -> None:
        """Signal the async generator to stop and clean up the ffmpeg process."""
        self._stop_event.set()
        await self._kill_proc()

    async def frames(self) -> AsyncIterator[tuple[int, float, np.ndarray]]:
        """
        Async generator that yields ``(frame_id, timestamp, frame)`` tuples.

        ``frame_id`` is a monotonically increasing integer.
        ``timestamp`` is a float epoch timestamp (seconds).
        ``frame`` is a ``numpy.ndarray`` of shape ``(H, W, 3)`` in BGR24.

        The generator runs until ``stop()`` is called or the containing task
        is cancelled.
        """
        self._stop_event.clear()
        self._health.running = True
        frame_id = 0

        # Pre-flight: detect stream codec before the first ffmpeg subprocess so
        # _build_cmd picks the right NVDEC decoder (h264_cuvid vs hevc_cuvid)
        # immediately.  This is a separate short-lived ffprobe connection — most
        # cameras allow concurrent RTSP clients, so it completes without stalling
        # the main ffmpeg connection.  On single-client cameras, ffprobe fails
        # silently and the first ffmpeg start falls back to h264_cuvid; after one
        # reconnect the probe runs concurrently and the correct decoder is used.
        if self._use_hw_decode and not self._codec_name:
            try:
                info = await asyncio.wait_for(
                    self._ffprobe_stream_info(), timeout=8.0
                )
                if info:
                    w, h, codec = info
                    self._codec_name = codec.lower()
                    if not (self._width and self._height):
                        w, h = self._cap_resolution(w, h)
                        self._width, self._height = w, h
                        self._frame_shape = (h, w, 3)
                        self._frame_size  = w * h * 3
            except asyncio.TimeoutError:
                logger.debug("RTSPDecoder: preflight probe timed out — using default decoder")
            except Exception as exc:
                logger.debug("RTSPDecoder: preflight probe failed: %s", exc)

        try:
            while not self._stop_event.is_set():
                try:
                    await self._start_proc()
                    async for raw in self._read_frames():
                        if self._stop_event.is_set():
                            return
                        ts = time.time()
                        self._record_frame(ts)
                        yield frame_id, ts, raw
                        frame_id += 1
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    if self._stop_event.is_set():
                        break
                    logger.warning(
                        "RTSPDecoder: stream error (%s); reconnecting in %.1fs",
                        exc,
                        self._reconnect_delay,
                    )
                    self._health.reconnect_count += 1
                    await self._kill_proc()
                    await asyncio.sleep(self._reconnect_delay)
        finally:
            # Flush any partial FPS bucket before marking stopped
            if self._fps_bucket_count > 0 and self._fps_bucket_start > 0:
                elapsed = time.time() - self._fps_bucket_start
                if elapsed > 0:
                    self._health.fps_actual = self._fps_bucket_count / elapsed
            self._health.running = False
            await self._kill_proc()

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _build_cmd(self) -> list[str]:
        # Select the right NVDEC decoder based on the stream codec (detected by
        # _preflight_probe).  Default to h264_cuvid when codec is unknown —
        # _check_hw_failure will disable hw decode if it errors.
        cuvid_decoder = (
            self._CUVID_MAP.get(self._codec_name.lower(), "h264_cuvid")
            if self._codec_name else "h264_cuvid"
        )
        if self._codec_name and self._codec_name.lower() not in self._CUVID_MAP:
            logger.warning(
                "RTSPDecoder: unknown codec %r — falling back to h264_cuvid",
                self._codec_name,
            )

        cmd = ["ffmpeg", "-loglevel", "error"]
        if self._tcp_transport:
            cmd += ["-rtsp_transport", "tcp"]
        if self._use_hw_decode and shutil.which("ffmpeg"):
            # hwaccel_output_format=cuda keeps frames in GPU memory; hwdownload
            # + format=nv12 copies them to CPU so software filters work.
            cmd += [
                "-hwaccel", "cuda",
                "-hwaccel_device", str(self._gpu_device),
                "-hwaccel_output_format", "cuda",
                "-c:v", cuvid_decoder,
            ]
        cmd += ["-i", self._url]
        if self._use_hw_decode and shutil.which("ffmpeg"):
            if self._width and self._height:
                vf = (
                    f"hwdownload,format=nv12,"
                    f"scale={self._width}:{self._height},"
                    f"fps={self._fps}"
                )
            else:
                vf = f"hwdownload,format=nv12,fps={self._fps}"
        else:
            if self._width and self._height:
                vf = f"scale={self._width}:{self._height},fps={self._fps}"
            else:
                vf = f"fps={self._fps}"
        cmd += ["-vf", vf, "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
        return cmd

    async def _start_proc(self) -> None:
        await self._kill_proc()
        cmd = self._build_cmd()
        logger.info("RTSPDecoder: starting ffmpeg: %s", " ".join(cmd))
        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=1 << 25,  # 32 MB pipe buffer
        )
        # Probe the first frame to determine resolution; also checks stderr for
        # NVDEC init failures so we can disable hardware decode immediately.
        await self._probe_resolution()
        await self._check_hw_failure()
        # Start background stderr drainer.  Without this, ffmpeg blocks as soon
        # as the OS pipe buffer (typically 64 KB) fills with warnings/errors —
        # which stalls stdout and causes readexactly() to time out, producing
        # visible stream freezes.
        self._start_stderr_drainer()

    async def _probe_resolution(self) -> None:
        """
        Discover stream width × height (and codec name) via ffprobe.

        If width/height were supplied by the caller they are used directly and
        ffprobe is skipped.  Otherwise a single ffprobe call is made AFTER the
        ffmpeg subprocess has started (so ffprobe connects concurrently while
        ffmpeg holds the main stream slot — this is safe for cameras that allow
        two RTSP connections, and causes one reconnect for cameras that don't).

        The codec name detected here is stored so that the NEXT reconnect's
        ``_build_cmd`` can pick the right NVDEC decoder (h264_cuvid vs
        hevc_cuvid).
        """
        if self._width and self._height:
            self._frame_shape = (self._height, self._width, 3)
            self._frame_size = self._width * self._height * 3
        else:
            # If resolution was already probed (e.g. after a reconnect that
            # successfully used ffprobe), reuse it without another ffprobe call.
            if self._frame_size > 0:
                return
            size = await self._ffprobe_resolution()
            if size:
                w, h = size
                self._frame_shape = (h, w, 3)
                self._frame_size = w * h * 3
            else:
                # ffprobe unavailable or timed-out — will retry on next reconnect
                self._frame_size = 0

    async def _check_hw_failure(self) -> None:
        """
        Read a brief window of ffmpeg stderr to detect NVDEC init errors.
        If libnvcuvid cannot be loaded, disable hw decode immediately so the
        reconnect loop doesn't retry it dozens of times.
        """
        if not self._use_hw_decode or self._proc is None:
            return
        assert self._proc.stderr is not None
        _NVDEC_ERRORS = (
            b"Cannot load libnvcuvid",
            b"Failed loading nvcuvid",
            b"No capable devices found",
            b"CUDA_ERROR_NO_DEVICE",
            b"Failed to create CUDA context",
            b"cuda: CUDA_ERROR",
            b"No CUDA capable device",
            b"CUDA driver version is insufficient",
            b"cannot load nvcuda.dll",
        )
        try:
            # A short non-blocking read window; ffmpeg prints init errors quickly
            chunk = await asyncio.wait_for(
                self._proc.stderr.read(4096), timeout=2.0
            )
            if any(e in chunk for e in _NVDEC_ERRORS):
                logger.warning(
                    "RTSPDecoder: NVDEC unavailable (%s) — switching to CPU decode",
                    chunk.decode(errors="replace").strip()[:120],
                )
                self._use_hw_decode = False
                # Kill the failing process; frames() will restart with CPU cmd
                await self._kill_proc()
                raise RuntimeError("NVDEC init failed — restarting with CPU decode")
        except asyncio.TimeoutError:
            pass  # no error output in 2s → hw decode is likely starting fine

    def _cap_resolution(self, w: int, h: int) -> tuple[int, int]:
        """
        Scale ``(w, h)`` down to fit within ``_MAX_DECODE_WIDTH × _MAX_DECODE_HEIGHT``
        while preserving aspect ratio, rounding to the nearest even pixel.
        """
        max_w, max_h = self._MAX_DECODE_WIDTH, self._MAX_DECODE_HEIGHT
        if w <= max_w and h <= max_h:
            return w, h
        scale = min(max_w / w, max_h / h)
        new_w = int(w * scale) & ~1   # round down to even
        new_h = int(h * scale) & ~1
        logger.info(
            "RTSPDecoder: capping %dx%d → %dx%d (max %dx%d)",
            w, h, new_w, new_h, max_w, max_h,
        )
        return new_w, new_h

    async def _preflight_probe(self) -> None:
        """
        Run ffprobe once to discover stream width, height, and codec name.

        Results are stored so ``_build_cmd`` can pick the correct NVDEC
        decoder (h264_cuvid vs hevc_cuvid) before the first subprocess is
        spawned.  Also pre-sets ``_frame_shape`` / ``_frame_size`` so the
        inline ``_probe_resolution`` step is a no-op on the first start.
        """
        info = await self._ffprobe_stream_info()
        if info is None:
            return
        w, h, codec = info
        if not self._codec_name:
            self._codec_name = codec.lower()
        if not (self._width and self._height):
            self._width  = w
            self._height = h
        self._frame_shape = (self._height, self._width, 3)
        self._frame_size  = self._width * self._height * 3
        logger.info(
            "RTSPDecoder preflight: %dx%d  codec=%s  decoder=%s",
            self._width, self._height, self._codec_name,
            self._CUVID_MAP.get(self._codec_name, "h264_cuvid"),
        )

    async def _ffprobe_stream_info(self) -> tuple[int, int, str] | None:
        """
        Return ``(width, height, codec_name)`` for the first video stream.

        Parses only the first non-empty output line so multi-stream cameras or
        ffprobe version differences don't cause parsing failures.
        """
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            return None
        cmd = [
            ffprobe, "-v", "error",
            "-rtsp_transport", "tcp" if self._tcp_transport else "udp",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,codec_name",
            "-of", "csv=p=0",
            self._url,
        ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=12.0)
            # Take the first non-empty line only (some cameras emit multiple streams).
            # ffprobe outputs fields in its internal order (codec_name, width, height)
            # which may differ from the order in -show_entries, so we auto-detect
            # by type: integer fields are dimensions, string field is codec name.
            for line in out.decode().splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                dims:  list[int] = []
                codec: str       = ""
                for p in parts:
                    try:
                        dims.append(int(p))
                    except ValueError:
                        if p:
                            codec = p
                if len(dims) >= 2:
                    logger.info(
                        "RTSPDecoder: stream probe → %dx%d  codec=%s",
                        dims[0], dims[1], codec or "unknown",
                    )
                    return dims[0], dims[1], codec or "h264"
        except Exception as exc:
            logger.debug("ffprobe stream info failed: %s", exc)
        return None

    async def _ffprobe_resolution(self) -> tuple[int, int] | None:
        """
        Use ffprobe to get stream width × height without consuming frames.

        If preflight already set _frame_size, skip the ffprobe call to avoid
        a redundant network round-trip on each reconnect.
        """
        # If preflight already probed resolution, reuse it
        if self._frame_size > 0 and self._width and self._height:
            return self._width, self._height
        info = await self._ffprobe_stream_info()
        if info is None:
            return None
        w, h, codec = info
        if not self._codec_name and codec:
            self._codec_name = codec.lower()
        # Apply resolution cap so callers don't set frame_size to a huge value
        w, h = self._cap_resolution(w, h)
        return w, h

    async def _read_frames(self) -> AsyncIterator[np.ndarray]:
        """Read raw BGR frames from the ffmpeg stdout pipe."""
        assert self._proc is not None and self._proc.stdout is not None

        if self._frame_size == 0:
            raise RuntimeError(
                "Cannot read frames: frame resolution unknown. "
                "Provide width/height to RTSPDecoder or ensure ffprobe is available."
            )

        while True:
            # Check for frame timeout
            if (
                self._health.last_frame_time > 0
                and time.time() - self._health.last_frame_time > self._frame_timeout
            ):
                raise TimeoutError(
                    f"No frame received for {self._frame_timeout:.0f}s"
                )

            try:
                raw = await asyncio.wait_for(
                    self._proc.stdout.readexactly(self._frame_size),
                    timeout=self._frame_timeout,
                )
            except asyncio.IncompleteReadError:
                # Pipe closed — ffmpeg exited; stop this generator cleanly
                return
            except asyncio.TimeoutError:
                raise TimeoutError(f"Frame read timed out after {self._frame_timeout}s")

            frame = np.frombuffer(raw, dtype=np.uint8).reshape(self._frame_shape)
            yield frame

    def _start_stderr_drainer(self) -> None:
        """
        Launch a fire-and-forget task that continuously reads and discards
        stderr from the ffmpeg subprocess.

        This prevents the 64 KB OS pipe buffer from filling when ffmpeg emits
        warnings (connection retries, codec errors, etc.), which would otherwise
        block ffmpeg from writing to stdout and cause frame-read timeouts.
        """
        if self._proc is None or self._proc.stderr is None:
            return
        if self._stderr_drain_task and not self._stderr_drain_task.done():
            self._stderr_drain_task.cancel()

        async def _drain() -> None:
            assert self._proc is not None and self._proc.stderr is not None
            try:
                while True:
                    chunk = await self._proc.stderr.read(4096)
                    if not chunk:
                        break
                    # Log at DEBUG so operators can still see ffmpeg messages
                    # without them filling the OS pipe buffer.
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "RTSPDecoder stderr: %s",
                            chunk.decode(errors="replace").strip()[:200],
                        )
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

        self._stderr_drain_task = asyncio.ensure_future(_drain())

    async def _kill_proc(self) -> None:
        # Cancel the stderr drainer first so it doesn't try to read from a
        # process we're about to kill.
        if self._stderr_drain_task and not self._stderr_drain_task.done():
            self._stderr_drain_task.cancel()
            self._stderr_drain_task = None
        if self._proc is None:
            return
        try:
            if self._proc.returncode is None:
                self._proc.kill()
                await asyncio.wait_for(self._proc.wait(), timeout=3.0)
        except (ProcessLookupError, asyncio.TimeoutError):
            pass
        finally:
            self._proc = None

    def _record_frame(self, ts: float) -> None:
        """Update health counters with a newly decoded frame."""
        self._health.frames_decoded += 1
        self._health.last_frame_time = ts

        # Rolling FPS over 1-second buckets
        if self._fps_bucket_start == 0.0:
            self._fps_bucket_start = ts
            self._fps_bucket_count = 1
        else:
            self._fps_bucket_count += 1
            elapsed = ts - self._fps_bucket_start
            if elapsed >= 1.0:
                self._health.fps_actual = self._fps_bucket_count / elapsed
                self._fps_bucket_start = ts
                self._fps_bucket_count = 0
