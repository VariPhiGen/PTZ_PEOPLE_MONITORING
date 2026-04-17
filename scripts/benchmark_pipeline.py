#!/usr/bin/env python3
"""
ACAS — End-to-End AI Pipeline Benchmark
========================================
Measures real-world throughput and per-stage latency across the full ACAS
inference stack (YOLO → SCRFD → AdaFace → MiniFASNet).

Modes:
  full      Run on a real video file; processes every frame at the
            camera's native rate, measuring actual pipeline throughput.

  stress    Inject synthetic face chips at N faces/frame with no real
            decoding overhead — isolates pure AI inference throughput.
            Target: 15 faces × 10 FPS → 150 recognitions/sec sustained.

Usage:
    # Standard end-to-end (real video):
    python scripts/benchmark_pipeline.py --video recording.mp4

    # Stress test (synthetic, no video needed):
    python scripts/benchmark_pipeline.py --mode stress --faces 15 --fps 10

    # Both modes, explicit model dir:
    python scripts/benchmark_pipeline.py \\
        --mode full \\
        --video /path/to/test.mp4 \\
        --model-dir /models \\
        --device 0 \\
        --duration 60

Options:
    --mode      full | stress (default: stress if no video, else full)
    --video     Path to test video (full mode)
    --model-dir /models        (default: /models)
    --device    0              CUDA device (default: 0)
    --duration  60             Test duration in seconds (default: 60)
    --faces     15             Faces/frame for stress mode (default: 15)
    --fps       10             Target FPS for stress mode (default: 10)
    --warmup    5              Warmup seconds before recording stats (default: 5)
    --no-gpu-stats             Skip nvidia-smi GPU sampling
    --json      PATH           Write full per-frame stats to JSON
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Deque

import numpy as np

# ── Path setup — allow running from project root ───────────────────────────────
_HERE = Path(__file__).resolve().parent
_BACKEND = _HERE.parent / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

# ── Colours ───────────────────────────────────────────────────────────────────
RED   = "\033[0;31m"; GREEN = "\033[0;32m"; YELLOW = "\033[1;33m"
BLUE  = "\033[0;34m"; CYAN  = "\033[0;36m"; BOLD   = "\033[1m"; NC = "\033[0m"
DIM   = "\033[2m"

OK   = f"{GREEN}✓{NC}"
FAIL = f"{RED}✗{NC}"
WARN = f"{YELLOW}⚠{NC}"

TARGET_FACES_PER_SEC  = 150.0    # 15 faces × 10 FPS
TARGET_FRAME_MS       = 100.0    # 1000 / 10 FPS
TARGET_YOLO_MS        = 18.0
TARGET_RETINA_MS      = 12.0
TARGET_ADAFACE_MS_PF  = 3.0      # per face
TARGET_LIVENESS_MS_PF = 10.0     # per face

# ── Per-frame stats dataclass ──────────────────────────────────────────────────
@dataclass
class FrameStats:
    frame_id:     int
    ts:           float           # wall clock at frame start
    total_ms:     float
    yolo_ms:      float
    retina_ms:    float
    align_ms:     float
    embed_ms:     float
    liveness_ms:  float
    n_persons:    int
    n_faces:      int
    gpu_pct:      float = 0.0
    vram_used_mb: float = 0.0


@dataclass
class PipelineStats:
    """Rolling statistics over a sliding window of frames."""
    window_s:     float = 10.0
    _frames:      Deque[FrameStats] = field(default_factory=collections.deque)
    _start_ts:    float = field(default_factory=time.perf_counter)

    def add(self, fs: FrameStats):
        self._frames.append(fs)
        cutoff = time.perf_counter() - self.window_s
        while self._frames and self._frames[0].ts < cutoff:
            self._frames.popleft()

    @property
    def window_frames(self) -> list[FrameStats]:
        return list(self._frames)

    def fps(self) -> float:
        if len(self._frames) < 2:
            return 0.0
        span = self._frames[-1].ts - self._frames[0].ts
        return (len(self._frames) - 1) / span if span > 0 else 0.0

    def faces_per_sec(self) -> float:
        if len(self._frames) < 2:
            return 0.0
        span = self._frames[-1].ts - self._frames[0].ts
        total_faces = sum(f.n_faces for f in self._frames)
        return total_faces / span if span > 0 else 0.0

    def pct(self, attr: str) -> tuple[float, float, float, float]:
        """Return (mean, p50, p95, p99) in ms for a latency attribute."""
        vals = [getattr(f, attr) for f in self._frames if getattr(f, attr) > 0]
        if not vals:
            return (0, 0, 0, 0)
        arr = np.array(vals)
        return (
            float(arr.mean()),
            float(np.percentile(arr, 50)),
            float(np.percentile(arr, 95)),
            float(np.percentile(arr, 99)),
        )


# ── GPU stats sampler (background thread) ─────────────────────────────────────
class GpuSampler:
    def __init__(self, device: int = 0, interval: float = 1.0):
        self.device   = device
        self.interval = interval
        self.gpu_pct:  float = 0.0
        self.vram_mb:  float = 0.0
        self._stop    = threading.Event()
        self._t       = threading.Thread(target=self._run, daemon=True)

    def start(self): self._t.start()
    def stop(self):  self._stop.set(); self._t.join(timeout=2)

    def _run(self):
        while not self._stop.wait(self.interval):
            try:
                out = subprocess.check_output(
                    ["nvidia-smi",
                     f"--id={self.device}",
                     "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    stderr=subprocess.DEVNULL,
                    timeout=3,
                ).decode()
                parts = out.strip().split(",")
                self.gpu_pct = float(parts[0].strip())
                self.vram_mb = float(parts[1].strip())
            except Exception:
                pass


# ── AIPipeline import helper ────────────────────────────────────────────────────
def load_pipeline(model_dir: str, device: int):
    try:
        from app.services.ai_pipeline import AIPipeline
        pipe = AIPipeline(model_dir=model_dir, device_id=device)
        pipe.load()
        return pipe
    except ImportError as e:
        print(f"\n{FAIL} Cannot import AIPipeline from {_BACKEND}: {e}")
        print(   "  Run this script from inside the acas-backend container or with the")
        print(   "  backend directory on PYTHONPATH.")
        sys.exit(1)
    except Exception as e:
        print(f"\n{FAIL} AIPipeline.load() failed: {e}")
        sys.exit(1)


# ── Synthetic frame generator ──────────────────────────────────────────────────
def make_synthetic_frame(height: int = 1080, width: int = 1920) -> np.ndarray:
    """Return a uint8 BGR frame with noise (sufficient to keep YOLO busy)."""
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)


def make_synthetic_face_chips(n: int) -> list[np.ndarray]:
    """Return N 112×112 uint8 BGR face chips."""
    return [np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8) for _ in range(n)]


# ── Stress mode: inject synthetic faces into pipeline stages ────────────────────
def run_stress(pipeline, n_faces: int, target_fps: float, duration_s: float,
               warmup_s: float, gpu: GpuSampler, args) -> tuple[PipelineStats, list[FrameStats]]:
    """
    Call each pipeline stage directly with pre-generated fake inputs.
    Bypasses video decoding and YOLO/SCRFD to isolate pure embedding throughput.
    Tests: face alignment + AdaFace + liveness at the target load.
    """
    import cv2
    stats  = PipelineStats(window_s=10.0)
    all_fs: list[FrameStats] = []
    frame_id = 0
    interval = 1.0 / target_fps
    rng = np.random.default_rng(0)

    # Pre-generate synthetic person-crop frames and face chips
    print(f"  Pre-generating {n_faces} synthetic face chips…")
    frame_h, frame_w = 720, 1280
    # We'll call detect_faces on a real-ish frame with bboxes injected artificially
    # by calling get_embedding and check_liveness directly (the only parts we want to stress)

    end_ts   = time.perf_counter() + warmup_s + duration_s
    warmup_end = time.perf_counter() + warmup_s
    recording = False
    frame_period = interval

    print(f"  Warming up ({warmup_s:.0f}s)…", end="", flush=True)

    while time.perf_counter() < end_ts:
        t_frame = time.perf_counter()

        if not recording and t_frame >= warmup_end:
            recording = True
            print(f"\r  Recording…{' ' * 20}", flush=True)

        # ── Stage: Face alignment (simulated — just reshape) ────────────────
        t0 = time.perf_counter()
        face_chips = [
            rng.integers(0, 255, (112, 112, 3), dtype=np.uint8)
            for _ in range(n_faces)
        ]
        align_ms = (time.perf_counter() - t0) * 1000

        # ── Stage: AdaFace embedding ─────────────────────────────────────────
        t0 = time.perf_counter()
        embeddings = []
        for chip in face_chips:
            emb = pipeline.get_embedding(chip)
            embeddings.append(emb)
        embed_ms = (time.perf_counter() - t0) * 1000

        # ── Stage: Liveness ──────────────────────────────────────────────────
        t0 = time.perf_counter()
        _dummy_frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        _bbox = np.array([100, 100, 212, 212], dtype=np.float32)  # 112px face
        for _ in face_chips:
            pipeline.check_liveness(_dummy_frame, _bbox)
        liveness_ms = (time.perf_counter() - t0) * 1000

        total_ms = align_ms + embed_ms + liveness_ms

        fs = FrameStats(
            frame_id=frame_id,
            ts=t_frame,
            total_ms=total_ms,
            yolo_ms=0.0,
            retina_ms=0.0,
            align_ms=align_ms,
            embed_ms=embed_ms,
            liveness_ms=liveness_ms,
            n_persons=n_faces,
            n_faces=n_faces,
            gpu_pct=gpu.gpu_pct,
            vram_used_mb=gpu.vram_mb,
        )

        if recording:
            stats.add(fs)
            all_fs.append(fs)

        frame_id += 1

        # Pace to target FPS
        elapsed = time.perf_counter() - t_frame
        sleep_t = frame_period - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)

    return stats, all_fs


# ── Full mode: real video ──────────────────────────────────────────────────────
def run_full(pipeline, video_path: Path, duration_s: float, warmup_s: float,
             gpu: GpuSampler, args) -> tuple[PipelineStats, list[FrameStats]]:
    try:
        import cv2
    except ImportError:
        print(f"{FAIL} opencv-python required for video mode: pip install opencv-python-headless")
        sys.exit(1)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"{FAIL} Cannot open video: {video_path}")
        sys.exit(1)

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Video: {video_path.name}  {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}×"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}  {native_fps:.1f} fps  {total_frames} frames")

    stats   = PipelineStats(window_s=10.0)
    all_fs: list[FrameStats] = []
    frame_id = 0
    t_start  = time.perf_counter()
    warmup_end = t_start + warmup_s
    end_ts   = t_start + warmup_s + duration_s
    recording = False

    print(f"  Warming up ({warmup_s:.0f}s)…", end="", flush=True)

    while time.perf_counter() < end_ts:
        ret, frame = cap.read()
        if not ret:
            # Loop video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break

        t_now = time.perf_counter()
        if not recording and t_now >= warmup_end:
            recording = True
            print(f"\r  Recording {duration_s:.0f}s from {video_path.name}…{' ' * 10}", flush=True)

        t0 = time.perf_counter()
        # Run full pipeline
        result = pipeline.process_frame(frame, roi_rect=None)
        bd = result.latency_breakdown

        fs = FrameStats(
            frame_id=frame_id,
            ts=t0,
            total_ms=result.time_ms,
            yolo_ms=bd.get("yolo_ms", 0.0),
            retina_ms=bd.get("retina_ms", 0.0),
            align_ms=bd.get("align_ms", 0.0),
            embed_ms=bd.get("embed_ms", 0.0),
            liveness_ms=bd.get("liveness_ms", 0.0),
            n_persons=len(result.persons),
            n_faces=len(result.faces_with_embeddings),
            gpu_pct=gpu.gpu_pct,
            vram_used_mb=gpu.vram_mb,
        )

        if recording:
            stats.add(fs)
            all_fs.append(fs)

        frame_id += 1

    cap.release()
    return stats, all_fs


# ── Report printer ─────────────────────────────────────────────────────────────
def print_report(stats: PipelineStats, all_fs: list[FrameStats],
                 mode: str, args, duration_s: float):
    fps         = stats.fps()
    faces_per_s = stats.faces_per_sec()
    total_faces = sum(f.n_faces for f in all_fs)
    avg_faces   = total_faces / len(all_fs) if all_fs else 0
    mean_gpu    = np.mean([f.gpu_pct     for f in all_fs]) if all_fs else 0
    mean_vram   = np.mean([f.vram_used_mb for f in all_fs]) if all_fs else 0

    # Per-stage stats
    stages = [
        ("YOLO detection",       "yolo_ms",     TARGET_YOLO_MS,        f"< {TARGET_YOLO_MS:.0f} ms"),
        ("SCRFD RetinaFace",     "retina_ms",   TARGET_RETINA_MS,      f"< {TARGET_RETINA_MS:.0f} ms"),
        ("Face alignment",       "align_ms",    None,                  "—"),
        ("AdaFace (total)",      "embed_ms",    None,                  f"< {TARGET_ADAFACE_MS_PF:.0f} ms/face"),
        ("MiniFASNet (total)",   "liveness_ms", None,                  f"< {TARGET_LIVENESS_MS_PF:.0f} ms/face"),
        ("Full frame",           "total_ms",    TARGET_FRAME_MS,       f"< {TARGET_FRAME_MS:.0f} ms"),
    ]

    fps_pass         = fps >= args.fps if mode == "stress" else True
    faces_pass       = faces_per_s >= TARGET_FACES_PER_SEC
    overall_pass     = fps_pass and faces_pass

    print(f"\n{BOLD}{CYAN}╔══════════════════════════════════════════════════════════════════════╗{NC}")
    print(f"{BOLD}{CYAN}║              ACAS AI Pipeline Benchmark — Results                   ║{NC}")
    print(f"{BOLD}{CYAN}╠══════════════════════════════════════════════════════════════════════╣{NC}")
    print(f"{BOLD}  Mode: {mode.upper()}{'  │  Video: ' + Path(args.video).name if args.video else ''}  │  Duration: {duration_s:.0f}s  │  Device: cuda:{args.device}{NC}")
    print(f"{BOLD}{CYAN}╠══════════════════════════════════════════════════════════════════════╣{NC}")

    sym_fps   = OK if fps_pass   else FAIL
    sym_faces = OK if faces_pass else FAIL
    print(f"  {BOLD}Throughput:{NC}")
    print(f"    {sym_fps}  FPS:               {fps:>7.2f}   "
          f"{'(target: ' + str(args.fps) + ')' if mode == 'stress' else ''}")
    print(f"    {sym_faces}  Faces/sec:         {faces_per_s:>7.2f}   (target: {TARGET_FACES_PER_SEC:.0f})")
    print(f"         Avg faces/frame:   {avg_faces:>7.1f}   Frames recorded: {len(all_fs)}")
    print(f"         Total faces:       {total_faces:>7d}")
    if mean_gpu > 0:
        print(f"         GPU utilisation:   {mean_gpu:>6.1f}%   VRAM: {mean_vram:.0f} MB used")

    print(f"\n  {BOLD}{'Stage':<26}  {'Mean':>8}  {'p50':>8}  {'p95':>8}  {'p99':>8}  Target{NC}")
    print(  "  " + "─" * 76)
    for name, attr, tgt_ms, tgt_lbl in stages:
        mean, p50, p95, p99 = stats.pct(attr)
        if mean == 0 and attr not in ("yolo_ms", "total_ms"):
            # Skipped stage (stress mode)
            print(f"  {DIM}  {name:<26}  {'—':>8}  {'—':>8}  {'—':>8}  {'—':>8}  {tgt_lbl}{NC}")
            continue
        passed = (mean <= tgt_ms) if tgt_ms is not None else None
        sym = (OK if passed else FAIL) if passed is not None else " "
        # Per-face addendum for embed and liveness
        extra = ""
        if attr == "embed_ms" and avg_faces > 0:
            pf = mean / avg_faces
            extra = f"  ({pf:.2f} ms/face — {'OK' if pf <= TARGET_ADAFACE_MS_PF else 'SLOW'})"
        elif attr == "liveness_ms" and avg_faces > 0:
            pf = mean / avg_faces
            extra = f"  ({pf:.2f} ms/face — {'OK' if pf <= TARGET_LIVENESS_MS_PF else 'SLOW'})"
        print(f"  {sym}  {name:<26}  {mean:>6.1f}ms  {p50:>6.1f}ms  {p95:>6.1f}ms  {p99:>6.1f}ms  {tgt_lbl}{extra}")

    # Sustained throughput check (trailing 10s window)
    trailing = [f for f in all_fs if f.ts >= all_fs[-1].ts - 10.0] if all_fs else []
    if len(trailing) >= 2:
        span = trailing[-1].ts - trailing[0].ts
        trail_fps   = len(trailing) / span if span > 0 else 0
        trail_faces = sum(f.n_faces for f in trailing) / span if span > 0 else 0
        print(f"\n  {BOLD}Sustained (last 10s):{NC}")
        sym_sf = OK if trail_faces >= TARGET_FACES_PER_SEC else FAIL
        print(f"    {sym_sf}  {trail_fps:.2f} fps  │  {trail_faces:.1f} recognitions/sec  "
              f"(target: {TARGET_FACES_PER_SEC:.0f}/sec)")

    # Overall verdict
    print(f"\n{BOLD}{CYAN}╠══════════════════════════════════════════════════════════════════════╣{NC}")
    if overall_pass:
        print(f"  {OK}  {BOLD}{GREEN}PASS — pipeline meets all throughput targets{NC}")
    else:
        print(f"  {FAIL}  {BOLD}{RED}FAIL — pipeline below target (see above){NC}")
        if not faces_pass:
            print(f"  → {faces_per_s:.1f} rec/sec < {TARGET_FACES_PER_SEC:.0f} target")
            print( "  → Check: TRT engine cache built? (scripts/optimize_models.py)")
            print( "           GPU utilisation < 100%?  (nvidia-smi)")
            print( "           Multiple workers?  (reduce faces/worker, add a node)")
    print(f"{BOLD}{CYAN}╚══════════════════════════════════════════════════════════════════════╝{NC}\n")

    # Histogram of face counts per frame
    face_counts = collections.Counter(f.n_faces for f in all_fs)
    if face_counts and max(face_counts.keys()) > 0:
        print(f"  {DIM}Face count histogram (frames):{NC}")
        for cnt in sorted(face_counts.keys()):
            bar = "█" * min(40, face_counts[cnt])
            print(f"  {DIM}  {cnt:>3} faces: {bar} {face_counts[cnt]}{NC}")
        print()

    return overall_pass


# ── JSON export ────────────────────────────────────────────────────────────────
def export_json(all_fs: list[FrameStats], path: str):
    with open(path, "w") as f:
        json.dump([asdict(fs) for fs in all_fs], f, indent=2)
    print(f"  {OK} Per-frame stats written to {path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ACAS pipeline benchmark")
    p.add_argument("--mode",      choices=["full", "stress"], default="",
                   help="full (real video) or stress (synthetic). Defaults to full if --video given.")
    p.add_argument("--video",     default="",   help="Video file path (full mode)")
    p.add_argument("--model-dir", default="/models")
    p.add_argument("--device",    type=int,   default=0)
    p.add_argument("--duration",  type=float, default=60.0, help="Benchmark duration (s)")
    p.add_argument("--warmup",    type=float, default=5.0,  help="Warmup duration (s)")
    p.add_argument("--faces",     type=int,   default=15,   help="Faces/frame (stress mode)")
    p.add_argument("--fps",       type=float, default=10.0, help="Target FPS (stress mode)")
    p.add_argument("--no-gpu-stats", action="store_true")
    p.add_argument("--json",      default="",  help="Export per-frame stats to JSON")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Resolve mode
    if not args.mode:
        args.mode = "full" if args.video else "stress"
    if args.mode == "full" and not args.video:
        print(f"{FAIL} --video is required for --mode full")
        return 1

    print(f"\n{BOLD}{CYAN}╔══════════════════════════════════════════════════════════════════════╗{NC}")
    print(f"{BOLD}{CYAN}║              ACAS AI Pipeline Benchmark                              ║{NC}")
    print(f"{BOLD}{CYAN}╚══════════════════════════════════════════════════════════════════════╝{NC}")
    print(f"  Mode:      {args.mode.upper()}")
    print(f"  Model dir: {args.model_dir}")
    print(f"  Device:    cuda:{args.device}")
    print(f"  Duration:  {args.duration:.0f}s + {args.warmup:.0f}s warmup")
    if args.mode == "stress":
        print(f"  Load:      {args.faces} faces/frame @ {args.fps} fps = {args.faces * args.fps:.0f} recognitions/sec target")
    print()

    # ── GPU sampler ───────────────────────────────────────────────────────────
    gpu = GpuSampler(device=args.device)
    if not args.no_gpu_stats:
        gpu.start()

    # ── Load pipeline ─────────────────────────────────────────────────────────
    print(f"  Loading AIPipeline (device={args.device})…", flush=True)
    t0 = time.perf_counter()
    pipeline = load_pipeline(args.model_dir, args.device)
    print(f"  {OK} Pipeline loaded in {time.perf_counter() - t0:.1f}s")

    # Check which models actually loaded
    loaded = []
    if pipeline._yolo_sess:     loaded.append("YOLOv8x-pose")
    if pipeline._face_detector: loaded.append("SCRFD")
    if pipeline._embed_sess:    loaded.append("AdaFace")
    if pipeline._liveness_sess: loaded.append("MiniFASNet")
    print(f"  Active models: {', '.join(loaded) if loaded else 'NONE'}\n")

    if not loaded:
        print(f"{FAIL} No models loaded.  Run scripts/download_models.py first.")
        return 1

    # ── Run benchmark ─────────────────────────────────────────────────────────
    t_bench = time.perf_counter()
    try:
        if args.mode == "stress":
            stats, all_fs = run_stress(
                pipeline, args.faces, args.fps, args.duration,
                args.warmup, gpu, args,
            )
        else:
            stats, all_fs = run_full(
                pipeline, Path(args.video), args.duration,
                args.warmup, gpu, args,
            )
    except KeyboardInterrupt:
        print("\n  Interrupted — printing partial results…")
        all_fs = list(stats._frames) if hasattr(stats, '_frames') else []

    elapsed = time.perf_counter() - t_bench
    print(f"\n  Benchmark complete in {elapsed:.1f}s  ({len(all_fs)} frames recorded)")

    if not args.no_gpu_stats:
        gpu.stop()

    if not all_fs:
        print(f"{FAIL} No frames recorded.  Check model loading and video file.")
        return 1

    # ── Print report ──────────────────────────────────────────────────────────
    passed = print_report(stats, all_fs, args.mode, args, args.duration)

    # ── Export JSON ───────────────────────────────────────────────────────────
    if args.json:
        export_json(all_fs, args.json)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
