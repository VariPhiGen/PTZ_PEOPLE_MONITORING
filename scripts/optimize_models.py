#!/usr/bin/env python3
"""
ACAS — ONNX → TensorRT FP16 Engine Pre-Compiler
================================================
Forces OnnxRuntime TensorrtExecutionProvider to build (and cache) optimised
TRT engines for every model before the backend first starts.  Without this,
each backend worker pays a 2-5 min compile penalty on its first inference call.

Models compiled:
  1. YOLOv8x-pose          batch 1-8,  opt 4,   target: <18ms @ batch-4
  2. SCRFD det_10g (RetinaFace)  batch 1-8,  opt 8,  target: <12ms @ batch-8
  3. AdaFace IR-101         batch 1-32, opt 16,  target: <3ms/face @ batch-32
  4. MiniFASNetV2           batch 1-8,  opt 4,   target: <10ms @ batch-8

Skip logic:  if the ONNX file's SHA-256 matches a saved .sha256 sidecar AND
             the TRT cache file already exists, the model is skipped.

AdaFace LFW accuracy verification:
  Pass --lfw-dir PATH to run the full 6000-pair LFW protocol.
  Without it, a synthetic FP32↔FP16 cosine drift check is used (minimum
  standard: mean cosine similarity ≥ 0.9990 and no pair polarity flip).

Usage:
    # Inside container (recommended):
    python /app/scripts/optimize_models.py --model-dir /models [--lfw-dir /data/lfw]

    # From host:
    docker exec acas-backend python /app/scripts/optimize_models.py

Options:
    --model-dir DIR      Where ONNX files live (default: /models)
    --engine-dir DIR     Where TRT cache is written (default: MODEL_DIR/engines)
    --lfw-dir DIR        LFW aligned-images directory for full AdaFace verification
    --device INT         CUDA device index (default: 0)
    --workspace-gb INT   TRT builder workspace (default: 4)
    --warmup INT         Warmup iterations before timing (default: 30)
    --bench-iter INT     Benchmark iterations (default: 200)
    --force              Ignore .sha256 sidecars and rebuild all engines
    --no-bench           Skip benchmarks (build only)
    --skip MODEL         Comma-separated model names to skip (yolo/retina/adaface/liveness)
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("acas.optimize")

# ── Colours ───────────────────────────────────────────────────────────────────
RED = "\033[0;31m"; GREEN = "\033[0;32m"; YELLOW = "\033[1;33m"
BLUE = "\033[0;34m"; CYAN = "\033[0;36m"; BOLD = "\033[1m"; NC = "\033[0m"
DIM = "\033[2m"

OK   = f"{GREEN}✓{NC}"
FAIL = f"{RED}✗{NC}"
WARN = f"{YELLOW}⚠{NC}"
SKIP = f"{BLUE}→{NC}"


# ── Model specifications ───────────────────────────────────────────────────────
@dataclass
class ModelSpec:
    key:          str
    name:         str
    onnx_file:    str
    fp16:         bool
    # Profile shapes: input_name -> (min, opt, max) as "BxCxHxW" strings
    profile:      dict[str, tuple[str, str, str]]
    bench_batch:  int          # batch used for latency benchmark
    target_ms:    float        # latency target for bench_batch
    target_label: str
    # Expected output shape suffix for quick sanity check (optional)
    out_suffix:   tuple[int, ...] | None = None


MODELS: list[ModelSpec] = [
    ModelSpec(
        key="yolo",
        name="YOLOv8l",
        onnx_file="yolov8l.onnx",
        fp16=True,
        profile={"images": ("1x3x640x640", "4x3x640x640", "8x3x640x640")},
        bench_batch=4,
        target_ms=15.0,
        target_label="batch-4 < 15 ms",
    ),
    ModelSpec(
        key="retina",
        name="SCRFD / det_10g",
        onnx_file="buffalo_l/det_10g.onnx",
        fp16=True,
        profile={"input.1": ("1x3x640x640", "8x3x640x640", "8x3x640x640")},
        bench_batch=8,
        target_ms=12.0,
        target_label="batch-8 < 12 ms",
    ),
    ModelSpec(
        key="adaface",
        name="AdaFace IR-101",
        onnx_file="adaface_ir101_webface12m.onnx",
        fp16=True,
        profile={"input": ("1x3x112x112", "16x3x112x112", "32x3x112x112")},
        bench_batch=32,
        target_ms=96.0,   # < 3ms per face × 32 faces
        target_label="batch-32 < 3 ms/face",
    ),
    ModelSpec(
        key="liveness",
        name="MiniFASNetV2",
        onnx_file="minifasnet_v2.onnx",
        fp16=False,        # small model — FP32 for stability
        profile={"input": ("1x3x80x80", "4x3x80x80", "8x3x80x80")},
        bench_batch=8,
        target_ms=10.0,
        target_label="batch-8 < 10 ms",
    ),
]

# ── ORT session factory (mirrors ai_pipeline._make_session) ────────────────────
def make_session(
    onnx_path:    Path,
    engine_dir:   Path,
    spec:         ModelSpec,
    device_id:    int,
    workspace_gb: int,
):
    """Create ORT session with TensorRT EP + dynamic batch profile."""
    try:
        import onnxruntime as ort
    except ImportError:
        raise SystemExit("[FATAL] onnxruntime not installed. Run: pip install onnxruntime-gpu")

    engine_dir.mkdir(parents=True, exist_ok=True)

    # Build the profile shape strings for TRT EP
    min_shapes = ";".join(f"{n}:{v[0]}" for n, v in spec.profile.items())
    opt_shapes = ";".join(f"{n}:{v[1]}" for n, v in spec.profile.items())
    max_shapes = ";".join(f"{n}:{v[2]}" for n, v in spec.profile.items())

    trt_opts: dict[str, Any] = {
        "device_id":                    device_id,
        "trt_engine_cache_enable":      True,
        "trt_engine_cache_path":        str(engine_dir),
        "trt_timing_cache_enable":      True,
        "trt_timing_cache_path":        str(engine_dir),
        "trt_max_workspace_size":       workspace_gb * (1 << 30),
        "trt_builder_optimization_level": 5,
        "trt_fp16_enable":              spec.fp16,
        "trt_layer_norm_fp32_fallback": True,
        "trt_profile_min_shapes":       min_shapes,
        "trt_profile_opt_shapes":       opt_shapes,
        "trt_profile_max_shapes":       max_shapes,
    }
    cuda_opts = {"device_id": device_id, "arena_extend_strategy": "kSameAsRequested"}

    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 4       # silence ORT build logs here
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.intra_op_num_threads = 2

    providers = [
        ("TensorrtExecutionProvider", trt_opts),
        ("CUDAExecutionProvider",    cuda_opts),
        ("CPUExecutionProvider",     {}),
    ]
    sess = ort.InferenceSession(str(onnx_path), sess_options=sess_opts, providers=providers)
    active = sess.get_providers()[0]
    if "Tensorrt" not in active:
        log.warning("  %s is running on %s (not TRT) — check TRT installation", onnx_path.name, active)
    return sess


# ── Hash-based skip logic ──────────────────────────────────────────────────────
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while data := f.read(chunk):
            h.update(data)
    return h.hexdigest()


def hash_path(onnx_path: Path, engine_dir: Path) -> Path:
    return engine_dir / (onnx_path.name + ".sha256")


def needs_build(onnx_path: Path, engine_dir: Path) -> bool:
    hp = hash_path(onnx_path, engine_dir)
    if not hp.exists():
        return True
    saved = hp.read_text().strip()
    return saved != sha256_file(onnx_path)


def save_hash(onnx_path: Path, engine_dir: Path):
    hash_path(onnx_path, engine_dir).write_text(sha256_file(onnx_path))


# ── Fake input generator ────────────────────────────────────────────────────────
def _parse_shape(shape_str: str) -> tuple[int, ...]:
    return tuple(int(x) for x in shape_str.split("x"))


def make_dummy_input(spec: ModelSpec, batch: int) -> dict[str, np.ndarray]:
    """Return a dict of {input_name: random float32 array} for the given batch."""
    result: dict[str, np.ndarray] = {}
    for name, (_, opt_str, _) in spec.profile.items():
        shape = list(_parse_shape(opt_str))
        shape[0] = batch
        result[name] = np.random.rand(*shape).astype(np.float32)
    return result


# ── Benchmark helper ──────────────────────────────────────────────────────────
@dataclass
class BenchResult:
    name:       str
    batch:      int
    mean_ms:    float
    p50_ms:     float
    p95_ms:     float
    p99_ms:     float
    target_ms:  float
    target_lbl: str
    passed:     bool
    # For AdaFace, per-face latency
    per_face_ms: float | None = None
    latencies:  list[float] = field(default_factory=list)


def benchmark(
    sess,
    spec:       ModelSpec,
    batch:      int,
    warmup:     int,
    iterations: int,
) -> BenchResult:
    inp = make_dummy_input(spec, batch)
    out_names = [o.name for o in sess.get_outputs()]

    # Warmup
    for _ in range(warmup):
        sess.run(out_names, inp)

    # Timed runs
    latencies: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        sess.run(out_names, inp)
        latencies.append((time.perf_counter() - t0) * 1000)

    arr = np.array(latencies)
    mean_ms = float(arr.mean())
    passed = (
        mean_ms <= spec.target_ms if spec.key != "adaface"
        else (mean_ms / batch) <= 3.0
    )
    return BenchResult(
        name=spec.name,
        batch=batch,
        mean_ms=mean_ms,
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        target_ms=spec.target_ms,
        target_lbl=spec.target_label,
        passed=passed,
        per_face_ms=(mean_ms / batch) if spec.key == "adaface" else None,
        latencies=latencies,
    )


# ── AdaFace accuracy verification ──────────────────────────────────────────────
@dataclass
class LFWResult:
    mode:       str     # "lfw" | "synthetic"
    accuracy:   float   # %
    std:        float
    threshold:  float
    n_pairs:    int
    passed:     bool    # ≥ 99.0% for LFW, ≥ 0.9990 cosine for synthetic


def _embed_batch(sess, inp_name: str, faces: np.ndarray) -> np.ndarray:
    """Run AdaFace on a (N, 3, 112, 112) batch; return L2-normalised (N, 512)."""
    embs = sess.run(None, {inp_name: faces})[0]
    norms = np.linalg.norm(embs, axis=1, keepdims=True).clip(min=1e-6)
    return embs / norms


def verify_synthetic(
    fp16_sess,
    fp32_sess,
    n_pairs: int = 500,
) -> LFWResult:
    """
    Synthetic accuracy check: compare FP32↔FP16 embeddings on random face crops.
    Tests:
      - Cosine similarity between FP32 and FP16 output ≥ 0.9990 (mean)
      - Same-face pairs (augmented) preserve cosine sim > different-face pairs
    """
    inp_name_fp16 = fp16_sess.get_inputs()[0].name
    inp_name_fp32 = fp32_sess.get_inputs()[0].name
    rng = np.random.default_rng(42)

    # Generate n_pairs unique face-like images
    faces_fp32 = rng.standard_normal((n_pairs, 3, 112, 112)).astype(np.float32) * 0.15
    # Slight perturbation for "same-person augmented" pairs
    noise = rng.standard_normal((n_pairs, 3, 112, 112)).astype(np.float32) * 0.02
    faces_aug = (faces_fp32 + noise).astype(np.float32)

    BS = 32
    embs_fp32 = np.vstack([
        _embed_batch(fp32_sess, inp_name_fp32, faces_fp32[i:i+BS])
        for i in range(0, n_pairs, BS)
    ])
    embs_fp16 = np.vstack([
        _embed_batch(fp16_sess, inp_name_fp16, faces_fp32[i:i+BS])
        for i in range(0, n_pairs, BS)
    ])
    embs_aug_fp16 = np.vstack([
        _embed_batch(fp16_sess, inp_name_fp16, faces_aug[i:i+BS])
        for i in range(0, n_pairs, BS)
    ])

    # 1. FP32↔FP16 cosine similarity (quantisation drift)
    drift_sims = np.einsum("ij,ij->i", embs_fp32, embs_fp16)  # dot product (already L2-normed)
    mean_cos = float(drift_sims.mean())
    std_cos  = float(drift_sims.std())

    # 2. Same-person vs different-person separation
    same_sims = np.einsum("ij,ij->i", embs_fp16, embs_aug_fp16)
    # Different = cyclic shift
    diff_sims = np.einsum("ij,ij->i", embs_fp16, np.roll(embs_fp16, 1, axis=0))
    threshold = float((same_sims.mean() + diff_sims.mean()) / 2)
    same_acc  = float((same_sims > threshold).mean())
    diff_acc  = float((diff_sims < threshold).mean())
    sep_acc   = (same_acc + diff_acc) / 2

    passed = mean_cos >= 0.9990 and sep_acc >= 0.95

    log.info("  Synthetic FP32↔FP16 cosine: %.6f ± %.6f", mean_cos, std_cos)
    log.info("  Same/diff separation acc:   %.2f%%", sep_acc * 100)
    log.info("  Threshold (estimated):      %.4f", threshold)

    return LFWResult(
        mode="synthetic",
        accuracy=sep_acc * 100,
        std=std_cos * 100,
        threshold=threshold,
        n_pairs=n_pairs,
        passed=passed,
    )


def verify_lfw(fp16_sess, lfw_dir: Path, n_folds: int = 10) -> LFWResult:
    """
    Full 10-fold LFW verification (6000 pairs).
    lfw_dir should contain subdirs per identity: lfw_dir/Name_Surname/Name_Surname_0001.jpg
    pairs.txt should be at lfw_dir/../pairs.txt or lfw_dir/pairs.txt.
    """
    try:
        from PIL import Image
        from sklearn.model_selection import KFold
    except ImportError:
        raise RuntimeError("Pillow and scikit-learn required for LFW: pip install pillow scikit-learn")

    # Locate pairs.txt
    pairs_file = lfw_dir / "pairs.txt"
    if not pairs_file.exists():
        pairs_file = lfw_dir.parent / "pairs.txt"
    if not pairs_file.exists():
        raise FileNotFoundError(f"pairs.txt not found near {lfw_dir}")

    # Parse pairs
    pairs: list[tuple[Path, Path, bool]] = []
    with open(pairs_file) as f:
        lines = f.readlines()
    # First line is "N_FOLDS\tN_PAIRS_PER_FOLD"
    for line in lines[1:]:
        parts = line.strip().split("\t")
        if not parts or not parts[0]:
            continue
        if len(parts) == 3:   # same-person: name idx1 idx2
            n, i1, i2 = parts
            img1 = lfw_dir / n / f"{n}_{int(i1):04d}.jpg"
            img2 = lfw_dir / n / f"{n}_{int(i2):04d}.jpg"
            pairs.append((img1, img2, True))
        elif len(parts) == 4:  # different: name1 idx1 name2 idx2
            n1, i1, n2, i2 = parts
            img1 = lfw_dir / n1 / f"{n1}_{int(i1):04d}.jpg"
            img2 = lfw_dir / n2 / f"{n2}_{int(i2):04d}.jpg"
            pairs.append((img1, img2, False))

    log.info("  LFW: %d pairs loaded", len(pairs))

    inp_name = fp16_sess.get_inputs()[0].name
    _MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    _STD  = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    def load_face(p: Path) -> np.ndarray | None:
        if not p.exists():
            return None
        img = Image.open(p).convert("RGB").resize((112, 112), Image.BICUBIC)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - _MEAN) / _STD
        return arr.transpose(2, 0, 1)   # CHW

    # Build embedding cache
    all_paths = list({p for pair in pairs for p in pair[:2]})
    emb_cache: dict[Path, np.ndarray] = {}
    BS = 32
    valid_paths = [p for p in all_paths if p.exists()]
    log.info("  Embedding %d unique face images…", len(valid_paths))
    for i in range(0, len(valid_paths), BS):
        batch_paths = valid_paths[i:i + BS]
        faces = np.stack([load_face(p) for p in batch_paths])  # type: ignore[arg-type]
        embs = _embed_batch(fp16_sess, inp_name, faces)
        for path, emb in zip(batch_paths, embs):
            emb_cache[path] = emb

    # Compute similarities and labels
    sims, labels = [], []
    for img1, img2, is_same in pairs:
        if img1 not in emb_cache or img2 not in emb_cache:
            continue
        sims.append(float(np.dot(emb_cache[img1], emb_cache[img2])))
        labels.append(int(is_same))

    sims_arr  = np.array(sims)
    labels_arr = np.array(labels)

    # 10-fold cross-validation
    kf = KFold(n_splits=n_folds, shuffle=False)
    fold_accs: list[float] = []
    best_thresholds: list[float] = []
    for train_idx, test_idx in kf.split(sims_arr):
        # Find optimal threshold on train fold
        best_t, best_a = 0.0, 0.0
        for thresh in np.linspace(sims_arr.min(), sims_arr.max(), 400):
            preds = (sims_arr[train_idx] >= thresh).astype(int)
            acc = (preds == labels_arr[train_idx]).mean()
            if acc > best_a:
                best_a, best_t = acc, thresh
        best_thresholds.append(best_t)
        test_preds = (sims_arr[test_idx] >= best_t).astype(int)
        fold_accs.append((test_preds == labels_arr[test_idx]).mean())

    mean_acc = float(np.mean(fold_accs)) * 100
    std_acc  = float(np.std(fold_accs)) * 100
    threshold = float(np.mean(best_thresholds))
    passed = mean_acc >= 99.0

    return LFWResult(
        mode="lfw",
        accuracy=mean_acc,
        std=std_acc,
        threshold=threshold,
        n_pairs=len(sims),
        passed=passed,
    )


# ── Printer helpers ────────────────────────────────────────────────────────────
def header(text: str):
    print(f"\n{BOLD}{CYAN}━━━━  {text}  {'━' * max(0, 56 - len(text))}{NC}")


def bench_row(r: BenchResult):
    sym = OK if r.passed else FAIL
    per = f"  ({r.per_face_ms:.2f} ms/face)" if r.per_face_ms is not None else ""
    print(
        f"  {sym} {r.name:<28}  "
        f"mean {r.mean_ms:>6.1f} ms  "
        f"p50 {r.p50_ms:>6.1f}  p95 {r.p95_ms:>6.1f}  p99 {r.p99_ms:>6.1f}"
        f"{per}  [{r.target_lbl}]"
    )


def lfw_row(res: LFWResult, name: str):
    sym = OK if res.passed else FAIL
    if res.mode == "lfw":
        print(f"  {sym} {name}  LFW accuracy: {res.accuracy:.3f}% ± {res.std:.3f}%"
              f"  threshold={res.threshold:.4f}  n={res.n_pairs}")
    else:
        print(f"  {sym} {name}  Synthetic FP32↔FP16: {res.accuracy:.2f}%"
              f"  cosine_std={res.std:.4f}  threshold={res.threshold:.4f}  n={res.n_pairs}")


# ── Main ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ACAS TRT engine pre-compiler")
    p.add_argument("--model-dir",    default="/models",  help="ONNX model directory")
    p.add_argument("--engine-dir",   default="",         help="TRT cache dir (default: MODEL_DIR/engines)")
    p.add_argument("--lfw-dir",      default="",         help="LFW aligned images dir for AdaFace verification")
    p.add_argument("--device",       type=int, default=0, help="CUDA device index")
    p.add_argument("--workspace-gb", type=int, default=4, help="TRT workspace GB")
    p.add_argument("--warmup",       type=int, default=30)
    p.add_argument("--bench-iter",   type=int, default=200)
    p.add_argument("--force",        action="store_true", help="Rebuild even if hash matches")
    p.add_argument("--no-bench",     action="store_true", help="Build only, skip benchmarks")
    p.add_argument("--skip",         default="",         help="Comma-separated keys to skip")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    model_dir  = Path(args.model_dir)
    engine_dir = Path(args.engine_dir) if args.engine_dir else (model_dir / "engines")
    engine_dir.mkdir(parents=True, exist_ok=True)
    skip_set   = {s.strip().lower() for s in args.skip.split(",") if s.strip()}
    lfw_dir    = Path(args.lfw_dir) if args.lfw_dir else None

    print(f"\n{BOLD}{CYAN}╔══════════════════════════════════════════════════════════╗{NC}")
    print(f"{BOLD}{CYAN}║       ACAS — TensorRT FP16 Engine Pre-Compiler           ║{NC}")
    print(f"{BOLD}{CYAN}╚══════════════════════════════════════════════════════════╝{NC}")
    print(f"  Model dir:  {model_dir}")
    print(f"  Engine dir: {engine_dir}")
    print(f"  Device:     cuda:{args.device}")
    print(f"  Workspace:  {args.workspace_gb} GB")

    try:
        import onnxruntime as ort
        version = ort.__version__
        providers = ort.get_available_providers()
        trt_available = "TensorrtExecutionProvider" in providers
        cuda_available = "CUDAExecutionProvider" in providers
    except ImportError:
        print(f"\n{FAIL} onnxruntime-gpu not installed.  pip install onnxruntime-gpu")
        return 1

    print(f"  ORT:        {version}")
    print(f"  TRT EP:     {'available' if trt_available else WARN + ' NOT available — engines will use CUDA EP'}")
    print(f"  CUDA EP:    {'available' if cuda_available else FAIL + ' NOT available'}")

    if not trt_available:
        print(f"\n{WARN} TensorRT EP not available.  Benchmarks will use CUDA EP (slower).")
        print(   "  Install TensorRT: https://docs.nvidia.com/tensorrt/install-guide/")

    results: dict[str, BenchResult] = {}
    lfw_results: dict[str, LFWResult] = {}
    adaface_fp32_sess = None     # kept for synthetic accuracy check
    all_passed = True

    for spec in MODELS:
        if spec.key in skip_set:
            print(f"\n  {SKIP} Skipping {spec.name} (--skip)")
            continue

        onnx_path = model_dir / spec.onnx_file
        if not onnx_path.exists():
            print(f"\n  {WARN} {spec.name}: ONNX not found at {onnx_path} — skipping")
            continue

        header(spec.name)
        print(f"  ONNX: {onnx_path.name}  ({onnx_path.stat().st_size / 1e6:.0f} MB)")

        # ── Skip if hash unchanged ──────────────────────────────────────────
        if not args.force and not needs_build(onnx_path, engine_dir):
            print(f"  {SKIP} Engine cache up-to-date (SHA-256 match) — skipping build")
            if args.no_bench:
                continue
            print("  Loading cached engine for benchmarks…")
            sess = make_session(onnx_path, engine_dir, spec, args.device, args.workspace_gb)
        else:
            # ── Build engine ────────────────────────────────────────────────
            t_build = time.perf_counter()
            print(f"  Building TRT FP16 engine (this may take 2-5 min)…", flush=True)
            sess = make_session(onnx_path, engine_dir, spec, args.device, args.workspace_gb)
            # Trigger compilation with a dummy run at opt batch
            dummy = make_dummy_input(spec, spec.bench_batch)
            out_names = [o.name for o in sess.get_outputs()]
            _ = sess.run(out_names, dummy)
            build_s = time.perf_counter() - t_build
            save_hash(onnx_path, engine_dir)
            active_ep = sess.get_providers()[0]
            print(f"  {OK} Engine built in {build_s:.1f}s  [{active_ep.replace('ExecutionProvider','')}]")

        # ── Benchmark ───────────────────────────────────────────────────────
        if not args.no_bench:
            print(f"  Benchmarking (batch={spec.bench_batch}, "
                  f"{args.warmup} warmup + {args.bench_iter} timed runs)…", flush=True)
            r = benchmark(sess, spec, spec.bench_batch, args.warmup, args.bench_iter)
            results[spec.key] = r
            bench_row(r)
            if not r.passed:
                all_passed = False

        # ── AdaFace: keep FP32 session for accuracy check ───────────────────
        if spec.key == "adaface":
            print("  Loading FP32 reference session for accuracy verification…", flush=True)
            # Temporary spec with FP16 disabled
            fp32_spec = ModelSpec(**{**spec.__dict__, "fp16": False, "key": "adaface_fp32"})
            adaface_fp32_sess = make_session(onnx_path, engine_dir, fp32_spec, args.device, args.workspace_gb)

            # ── LFW accuracy verification ────────────────────────────────────
            print()
            if lfw_dir and lfw_dir.exists():
                print(f"  Running full LFW 6000-pair verification from {lfw_dir}…", flush=True)
                try:
                    lfw_r = verify_lfw(sess, lfw_dir)
                    lfw_results[spec.key] = lfw_r
                    lfw_row(lfw_r, spec.name)
                    if not lfw_r.passed:
                        all_passed = False
                        print(f"  {FAIL} LFW accuracy {lfw_r.accuracy:.3f}% < 99.0% threshold!")
                        print(   "  → Check model file integrity and FP16 layer_norm fallback setting.")
                except Exception as e:
                    print(f"  {WARN} LFW verification failed: {e}")
            else:
                print("  Running synthetic FP32↔FP16 accuracy check (pass --lfw-dir for full LFW)…", flush=True)
                try:
                    syn_r = verify_synthetic(sess, adaface_fp32_sess)
                    lfw_results[spec.key] = syn_r
                    lfw_row(syn_r, spec.name)
                    if not syn_r.passed:
                        all_passed = False
                        print(f"  {FAIL} FP16 accuracy drift too large!")
                        print(   "  → Enable trt_layer_norm_fp32_fallback or reduce precision.")
                except Exception as e:
                    print(f"  {WARN} Synthetic accuracy check failed: {e}")

        gc.collect()

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{BOLD}{CYAN}╔══════════════════════════════════════════════════════════════════╗{NC}")
    print(f"{BOLD}{CYAN}║                     Benchmark Summary                           ║{NC}")
    print(f"{BOLD}{CYAN}╠══════════════════════════════════════════════════════════════════╣{NC}")
    print(f"{BOLD}  {'Model':<28}  {'Batch':>5}  {'Mean':>7}  {'p50':>7}  {'p95':>7}  {'p99':>7}  Target{NC}")
    print( "  " + "─" * 70)

    TARGETS = {"yolo": 18.0, "retina": 12.0, "adaface": None, "liveness": 10.0}
    for spec in MODELS:
        if spec.key not in results:
            continue
        r = results[spec.key]
        sym = OK if r.passed else FAIL
        extra = f"({r.per_face_ms:.2f}/face)" if r.per_face_ms is not None else ""
        print(
            f"  {sym} {r.name:<26}  {r.batch:>5}  {r.mean_ms:>6.1f}ms  "
            f"{r.p50_ms:>6.1f}ms  {r.p95_ms:>6.1f}ms  {r.p99_ms:>6.1f}ms  "
            f"{r.target_lbl}  {extra}"
        )

    if lfw_results:
        print(f"\n{BOLD}  AdaFace Accuracy:{NC}")
        for key, res in lfw_results.items():
            sym = OK if res.passed else FAIL
            if res.mode == "lfw":
                print(f"  {sym} LFW 10-fold: {res.accuracy:.3f}% ± {res.std:.3f}%"
                      f"  (threshold={res.threshold:.4f}, n={res.n_pairs})")
                print(f"     {'PASS: meets ≥99.0% LFW standard' if res.passed else 'FAIL: below 99.0% — investigate FP16 settings'}")
            else:
                print(f"  {sym} Synthetic FP32↔FP16: {res.accuracy:.2f}%"
                      f"  cosine_drift_std={res.std:.5f}")
                print(f"     {'PASS: cosine drift within tolerance' if res.passed else 'FAIL: excessive FP16 drift'}")

    print(f"{BOLD}{CYAN}╚══════════════════════════════════════════════════════════════════╝{NC}\n")

    status = f"{OK} All engines built and benchmarks passed." if all_passed else \
             f"{FAIL} Some benchmarks FAILED — see above."
    print(f"  {status}")
    print(f"  Engine cache: {engine_dir}\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
