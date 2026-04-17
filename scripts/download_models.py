#!/usr/bin/env python3
"""
Download and prepare all AI model files for ACAS v2.

Models fetched:
  1. yolov8l.onnx            — YOLOv8l detection-only (person bbox, no pose keypoints)
  2. buffalo_l/det_10g.onnx  — InsightFace SCRFD face detector
  3. adaface_ir101_webface12m.onnx — AdaFace IR-101 face embeddings (HuggingFace)
  4. minifasnet_v2.onnx       — MiniFASNetV2 liveness anti-spoofing
  5. osnet_x1_0.onnx          — OSNet x1_0 person RE-ID (~8MB, from Torchreid)

Usage:
    # From inside the container:
    python /app/scripts/download_models.py --model-dir /models

    # From the host (with the ACAS project root as CWD):
    docker exec acas-backend python /app/scripts/download_models.py --model-dir /models

Flags:
    --model-dir     Where to save models (default: /models)
    --skip-yolo     Skip YOLOv8l download/export
    --skip-face     Skip buffalo_l face detector
    --skip-adaface  Skip AdaFace IR-101
    --skip-liveness Skip MiniFASNetV2
    --skip-reid     Skip OSNet x1_0 RE-ID model
    --force         Re-download even if file already exists
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


# ── Known checksums (SHA-256, first 16 hex chars) for integrity check ─────────
# Update these after downloading reference copies; leave empty to skip check.
_CHECKSUMS: dict[str, str] = {
    "yolov8l.onnx":                          "",   # varies by ultralytics export
    "buffalo_l/det_10g.onnx":                "5d97648813fa",
    "adaface_ir101_webface12m.onnx":         "",   # varies by export
    "minifasnet_v2.onnx":                    "",
    "osnet_x1_0.onnx":                       "",
}

# ── Download sources ───────────────────────────────────────────────────────────
_ADAFACE_HF_URL = (
    "https://huggingface.co/mk-minchul/adaface/resolve/main/"
    "adaface_ir101_webface12m.onnx"
)
_MINIFASNET_URL = (
    "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/"
    "releases/download/v1.3.4/2.7_80x80_MiniFASNetV2.pth"
)
# Backup: community ONNX-converted MiniFASNetV2
_MINIFASNET_ONNX_URL = (
    "https://huggingface.co/Dezmound/MiniFASNetV2/resolve/main/"
    "2.7_80x80_MiniFASNetV2.onnx"
)


def _sha256_prefix(path: Path, length: int = 14) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:length]


def _check(path: Path, key: str) -> bool:
    """Return True if file exists and checksum matches (skipped if checksum empty)."""
    if not path.exists():
        return False
    expected = _CHECKSUMS.get(key, "")
    if not expected:
        return True
    actual = _sha256_prefix(path, len(expected))
    if actual != expected:
        print(f"  [WARN] Checksum mismatch for {path.name}: expected {expected}, got {actual}")
        return False
    return True


def _download(url: str, dest: Path, desc: str) -> bool:
    """Download url → dest with progress. Returns True on success."""
    print(f"  Downloading {desc}\n    from: {url}\n      to: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    try:
        def _progress(count, block, total):
            if total > 0:
                pct = min(100, count * block * 100 // total)
                bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
                print(f"\r  [{bar}] {pct:3d}%", end="", flush=True)

        urllib.request.urlretrieve(url, str(tmp), reporthook=_progress)
        print()  # newline after progress bar
        tmp.rename(dest)
        mb = dest.stat().st_size / 1e6
        print(f"  Saved {mb:.1f} MB → {dest}")
        return True
    except Exception as exc:
        print(f"  [FAIL] {exc}")
        if tmp.exists():
            tmp.unlink()
        return False


# ── 1. YOLOv8l (detection-only, no pose) ──────────────────────────────────────

def download_yolo(model_dir: Path, force: bool = False) -> bool:
    dest = model_dir / "yolov8l.onnx"
    if not force and dest.exists():
        print(f"  [SKIP] {dest.name} already exists")
        return True

    print("\n─── YOLOv8l detection-only ────────────────────────────────────")
    print("  Downloading yolov8l.pt from Ultralytics hub and exporting to ONNX …")
    print("  (Detection-only, bbox only, ~130MB)")

    # Step 1: Download + export via ultralytics
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8l.pt")   # auto-downloads from ultralytics hub
    except ImportError:
        print("  [ERROR] ultralytics not installed — run: pip install ultralytics")
        return False
    except Exception as exc:
        print(f"  [FAIL] Could not load model: {exc}")
        return False

    # Step 2: Export to ONNX (static shapes, opset 17, detection task only)
    print("  Exporting to ONNX (opset 17, dynamic=False, imgsz=640) …")
    try:
        export_path = model.export(
            format="onnx",
            imgsz=640,
            opset=17,
            dynamic=False,
            half=False,     # FP32 ONNX; TRT FP16 conversion happens at runtime
            simplify=True,
        )
        exported = Path(export_path)
        if exported.exists():
            shutil.copy2(exported, dest)
            print(f"  [OK] Saved to {dest}  ({dest.stat().st_size / 1e6:.1f} MB)")
            return True
    except Exception as exc:
        print(f"  [FAIL] ONNX export failed: {exc}")

    return False


# ── 2. buffalo_l (InsightFace SCRFD face detector) ────────────────────────────

def download_buffalo_l(model_dir: Path, force: bool = False) -> bool:
    dest_dir = model_dir / "buffalo_l"
    dest = dest_dir / "det_10g.onnx"
    if not force and _check(dest, "buffalo_l/det_10g.onnx"):
        print(f"  [SKIP] buffalo_l/det_10g.onnx already exists")
        return True

    print("\n─── InsightFace buffalo_l ─────────────────────────────────────")
    try:
        import insightface
        from insightface.app import FaceAnalysis

        # InsightFace auto-downloads buffalo_l to ~/.insightface/models/buffalo_l/
        print("  Triggering InsightFace auto-download for 'buffalo_l' …")
        app = FaceAnalysis(
            name="buffalo_l",
            root=str(model_dir.parent / ".insightface"),  # use predictable path
            providers=["CPUExecutionProvider"],
        )
        app.prepare(ctx_id=-1)  # CPU; just to trigger download

        # Locate the downloaded files
        cache_dir = Path.home() / ".insightface" / "models" / "buffalo_l"
        if not cache_dir.exists():
            # Try the root we set
            cache_dir = model_dir.parent / ".insightface" / "models" / "buffalo_l"

        if cache_dir.exists():
            dest_dir.mkdir(parents=True, exist_ok=True)
            for src_file in cache_dir.glob("*.onnx"):
                shutil.copy2(src_file, dest_dir / src_file.name)
                print(f"  Copied {src_file.name} → {dest_dir / src_file.name}")
            print(f"  [OK] buffalo_l models in {dest_dir}")
            return True
        else:
            print(f"  [FAIL] buffalo_l not found after auto-download in {cache_dir}")
    except ImportError:
        print("  [ERROR] insightface not installed — run: pip install insightface")
    except Exception as exc:
        print(f"  [FAIL] {exc}")

    # Fallback: direct ZIP download from CDN
    zip_url = "https://huggingface.co/deepinsight/insightface/resolve/main/models/buffalo_l.zip"
    print(f"  Trying direct download from HuggingFace …")
    zip_path = model_dir / "buffalo_l.zip"
    if _download(zip_url, zip_path, "buffalo_l.zip"):
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(model_dir)
        zip_path.unlink()
        print(f"  [OK] Extracted to {model_dir / 'buffalo_l'}")
        return True

    return False


# ── 3. AdaFace IR-101 ─────────────────────────────────────────────────────────

def download_adaface(model_dir: Path, force: bool = False) -> bool:
    dest = model_dir / "adaface_ir101_webface12m.onnx"
    if not force and dest.exists() and dest.stat().st_size > 1e6:
        print(f"  [SKIP] {dest.name} already exists")
        return True

    print("\n─── AdaFace IR-101 (WebFace12M) ──────────────────────────────")

    # Attempt 1: direct ONNX from HuggingFace
    if _download(_ADAFACE_HF_URL, dest, "adaface_ir101_webface12m.onnx"):
        return True

    # Attempt 2: download .ckpt from HuggingFace and export to ONNX
    print("  Direct ONNX download failed — trying .ckpt + ONNX export …")
    ckpt_url = (
        "https://huggingface.co/mk-minchul/adaface/resolve/main/"
        "adaface_ir101_webface12m.ckpt"
    )
    ckpt_path = model_dir / "adaface_ir101_webface12m.ckpt"
    if not _download(ckpt_url, ckpt_path, "adaface_ir101_webface12m.ckpt"):
        _manual_instructions_adaface(dest)
        return False

    return _export_adaface_to_onnx(ckpt_path, dest)


def _export_adaface_to_onnx(ckpt_path: Path, dest: Path) -> bool:
    """Convert AdaFace .ckpt checkpoint to ONNX via torch.onnx.export."""
    print(f"  Exporting {ckpt_path.name} → ONNX …")
    try:
        import torch
        import torch.nn as nn

        # Try to import AdaFace net definition from the repo if cloned nearby
        adaface_repo = Path("/tmp/adaface_repo")
        if not adaface_repo.exists():
            print("  Cloning AdaFace repository for IR-101 architecture …")
            subprocess.check_call([
                "git", "clone", "--depth", "1",
                "https://github.com/mk-minchul/AdaFace.git",
                str(adaface_repo),
            ])
        sys.path.insert(0, str(adaface_repo))

        from net import build_model
        model = build_model("ir_101")
        statedict = torch.load(str(ckpt_path), map_location="cpu")
        if "state_dict" in statedict:
            statedict = statedict["state_dict"]
        statedict = {k.replace("model.", ""): v for k, v in statedict.items()}
        model.load_state_dict(statedict, strict=False)
        model.eval()

        dummy = torch.zeros(1, 3, 112, 112)
        torch.onnx.export(
            model,
            dummy,
            str(dest),
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
        mb = dest.stat().st_size / 1e6
        print(f"  [OK] Exported AdaFace ONNX  {mb:.1f} MB → {dest}")
        return True
    except Exception as exc:
        print(f"  [FAIL] ONNX export failed: {exc}")
        _manual_instructions_adaface(dest)
        return False


def _manual_instructions_adaface(dest: Path) -> None:
    print(f"""
  ─── Manual AdaFace Download Instructions ──────────────────────
  1. Visit https://github.com/mk-minchul/AdaFace
  2. Download adaface_ir101_webface12m.ckpt from the GDrive link
     in the README (Models section → WebFace12M → IR-101)
  3. Export to ONNX:
       python scripts/download_models.py --export-adaface /path/to/adaface_ir101_webface12m.ckpt
  4. Place the resulting ONNX file at:
       {dest}
  ───────────────────────────────────────────────────────────────
""")


# ── 4. MiniFASNetV2 ───────────────────────────────────────────────────────────

def download_minifasnet(model_dir: Path, force: bool = False) -> bool:
    dest = model_dir / "minifasnet_v2.onnx"
    if not force and dest.exists() and dest.stat().st_size > 1e4:
        print(f"  [SKIP] {dest.name} already exists")
        return True

    print("\n─── MiniFASNetV2 (anti-spoofing) ─────────────────────────────")

    # Attempt 1: community ONNX from HuggingFace
    if _download(_MINIFASNET_ONNX_URL, dest, "minifasnet_v2.onnx (HuggingFace)"):
        print(f"  [OK] Saved {dest.name}")
        return True

    # Attempt 2: download .pth and export
    print("  Direct ONNX failed — downloading .pth and exporting …")
    pth_path = model_dir / "2.7_80x80_MiniFASNetV2.pth"
    if not _download(_MINIFASNET_URL, pth_path, "2.7_80x80_MiniFASNetV2.pth"):
        _manual_instructions_minifasnet(dest)
        return False

    return _export_minifasnet_to_onnx(pth_path, dest)


def _export_minifasnet_to_onnx(pth_path: Path, dest: Path) -> bool:
    print(f"  Exporting {pth_path.name} → ONNX …")
    try:
        import torch

        # Clone the Silent-Face-Anti-Spoofing repo for model definition
        repo = Path("/tmp/silent_face_repo")
        if not repo.exists():
            print("  Cloning minivision-ai/Silent-Face-Anti-Spoofing …")
            subprocess.check_call([
                "git", "clone", "--depth", "1",
                "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing.git",
                str(repo),
            ])
        sys.path.insert(0, str(repo / "src"))

        from model_lib.MiniFASNet import MiniFASNetV2

        # Build model: depth_multiplier inferred from filename
        model = MiniFASNetV2(embedding_size=128, conv6_kernel=(5, 5))
        weights = torch.load(str(pth_path), map_location="cpu")
        model.load_state_dict(weights, strict=False)
        model.eval()

        dummy = torch.zeros(1, 3, 80, 80)
        torch.onnx.export(
            model,
            dummy,
            str(dest),
            opset_version=12,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
        mb = dest.stat().st_size / 1e6
        print(f"  [OK] Exported MiniFASNetV2  {mb:.2f} MB → {dest}")
        return True
    except Exception as exc:
        print(f"  [FAIL] Export failed: {exc}")
        _manual_instructions_minifasnet(dest)
        return False


def _manual_instructions_minifasnet(dest: Path) -> None:
    print(f"""
  ─── Manual MiniFASNetV2 Download Instructions ──────────────────
  1. Clone https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
  2. Download 2.7_80x80_MiniFASNetV2.pth from the repo's release page
  3. Export:
       from model_lib.MiniFASNet import MiniFASNetV2
       import torch
       m = MiniFASNetV2(embedding_size=128, conv6_kernel=(5,5))
       m.load_state_dict(torch.load('2.7_80x80_MiniFASNetV2.pth'), strict=False)
       m.eval()
       torch.onnx.export(m, torch.zeros(1,3,80,80), 'minifasnet_v2.onnx', opset_version=12)
  4. Copy to {dest}
  ─────────────────────────────────────────────────────────────────
""")


# ── 5. OSNet x1_0 (person RE-ID) ─────────────────────────────────────────────

# Primary: ONNX exported from torchreid via community HuggingFace repo
_OSNET_HF_URL = (
    "https://huggingface.co/spaces/keras-io/deep-person-reid/resolve/main/"
    "osnet_x1_0_imagenet.pth"
)

def download_osnet(model_dir: Path, force: bool = False) -> bool:
    dest = model_dir / "osnet_x1_0.onnx"
    if not force and dest.exists() and dest.stat().st_size > 1e6:
        print(f"  [SKIP] {dest.name} already exists")
        return True

    print("\n─── OSNet x1_0 (person RE-ID) ────────────────────────────────")

    # Attempt 1: Export via torchreid if installed
    ok = _export_osnet_via_torchreid(dest)
    if ok:
        return True

    # Attempt 2: Export via torch.hub (KaiyangZhou/torchreid)
    ok = _export_osnet_via_hub(dest)
    if ok:
        return True

    _manual_instructions_osnet(dest)
    return False


def _export_osnet_via_torchreid(dest: Path) -> bool:
    """Try to export OSNet using the torchreid package if installed."""
    try:
        import torch
        import torchreid

        print("  Found torchreid — exporting OSNet x1_0 to ONNX …")
        model = torchreid.models.build_model(
            name="osnet_x1_0",
            num_classes=1000,
            pretrained=True,
        )
        model.eval()

        dummy = torch.zeros(1, 3, 256, 128)
        torch.onnx.export(
            model,
            dummy,
            str(dest),
            opset_version=12,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
        mb = dest.stat().st_size / 1e6
        print(f"  [OK] Exported OSNet x1_0  {mb:.1f} MB → {dest}")
        return True
    except ImportError:
        return False
    except Exception as exc:
        print(f"  [INFO] torchreid export failed: {exc}")
        return False


def _export_osnet_via_hub(dest: Path) -> bool:
    """Try torch.hub to load and export OSNet x1_0."""
    try:
        import torch

        print("  Trying torch.hub (KaiyangZhou/deep-person-reid) …")
        model = torch.hub.load(
            "KaiyangZhou/deep-person-reid",
            "osnet_x1_0",
            pretrained=True,
        )
        model.eval()

        dummy = torch.zeros(1, 3, 256, 128)
        torch.onnx.export(
            model,
            dummy,
            str(dest),
            opset_version=12,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
        mb = dest.stat().st_size / 1e6
        print(f"  [OK] Exported OSNet x1_0 via hub  {mb:.1f} MB → {dest}")
        return True
    except Exception as exc:
        print(f"  [INFO] torch.hub export failed: {exc}")
        return False


def _manual_instructions_osnet(dest: Path) -> None:
    print(f"""
  ─── Manual OSNet x1_0 Download Instructions ──────────────────────
  Option A (torchreid):
    pip install torchreid
    python scripts/download_models.py --model-dir {dest.parent}

  Option B (torch.hub):
    pip install torch torchvision
    python scripts/download_models.py --model-dir {dest.parent}

  Option C (manual export):
    pip install torchreid
    python -c "
    import torch, torchreid
    m = torchreid.models.build_model('osnet_x1_0', num_classes=1000, pretrained=True)
    m.eval()
    torch.onnx.export(m, torch.zeros(1,3,256,128), '{dest}',
                      opset_version=12,
                      input_names=['input'], output_names=['output'])
    "

  RE-ID is optional — the system works without it, but identity continuity
  during face occlusion will be limited to MOT bounding-box tracking only.
  ─────────────────────────────────────────────────────────────────────
""")


# ── Optional: export AdaFace from CLI ─────────────────────────────────────────

def export_adaface_cli(ckpt_path: str, model_dir: Path) -> None:
    dest = model_dir / "adaface_ir101_webface12m.onnx"
    ok = _export_adaface_to_onnx(Path(ckpt_path), dest)
    sys.exit(0 if ok else 1)


# ── Print final status ────────────────────────────────────────────────────────

def _status(model_dir: Path) -> None:
    print("\n─── Model Status ─────────────────────────────────────────────")
    # Required models (pipeline won't start without these)
    required = {
        "yolov8l.onnx":                     "YOLOv8l       (person detection)",
        "buffalo_l/det_10g.onnx":           "SCRFD-10GF    (face detection)",
        "adaface_ir101_webface12m.onnx":    "AdaFace IR-101 (face embedding)",
        "minifasnet_v2.onnx":               "MiniFASNetV2   (liveness)",
    }
    # Optional models (graceful degradation if missing)
    optional = {
        "osnet_x1_0.onnx":                  "OSNet x1_0    (person RE-ID, optional)",
        "realesrgan_x2.onnx":               "Real-ESRGAN x2 (face SR, optional)",
    }
    all_required_ok = True
    for rel, desc in required.items():
        p = model_dir / rel
        if p.exists():
            mb = p.stat().st_size / 1e6
            print(f"  ✓  {desc:45s} {mb:7.1f} MB")
        else:
            print(f"  ✗  {desc:45s} NOT FOUND")
            all_required_ok = False

    for rel, desc in optional.items():
        p = model_dir / rel
        if p.exists():
            mb = p.stat().st_size / 1e6
            print(f"  ✓  {desc:45s} {mb:7.1f} MB")
        else:
            print(f"  -  {desc:45s} not installed (optional)")

    print()
    if all_required_ok:
        print("  All required models ready.  Run scripts/optimize_models.py to pre-build TRT engines.")
    else:
        print("  Some required models are missing.  Re-run with --force or follow manual instructions above.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir",       default="/models",  help="Model output directory")
    ap.add_argument("--skip-yolo",       action="store_true", help="Skip YOLOv8l")
    ap.add_argument("--skip-face",       action="store_true", help="Skip SCRFD buffalo_l")
    ap.add_argument("--skip-adaface",    action="store_true", help="Skip AdaFace IR-101")
    ap.add_argument("--skip-liveness",   action="store_true", help="Skip MiniFASNetV2")
    ap.add_argument("--skip-reid",       action="store_true", help="Skip OSNet x1_0 RE-ID model")
    ap.add_argument("--force",           action="store_true", help="Re-download even if file exists")
    ap.add_argument("--export-adaface",  metavar="CKPT",
                    help="Export AdaFace .ckpt → ONNX instead of downloading")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model directory: {model_dir.resolve()}")

    if args.export_adaface:
        export_adaface_cli(args.export_adaface, model_dir)
        return

    results: dict[str, bool] = {}

    if not args.skip_yolo:
        results["yolov8l"]      = download_yolo(model_dir, args.force)
    if not args.skip_face:
        results["buffalo_l"]    = download_buffalo_l(model_dir, args.force)
    if not args.skip_adaface:
        results["adaface"]      = download_adaface(model_dir, args.force)
    if not args.skip_liveness:
        results["minifasnet"]   = download_minifasnet(model_dir, args.force)
    if not args.skip_reid:
        results["osnet_x1_0"]   = download_osnet(model_dir, args.force)

    _status(model_dir)

    failed = [k for k, v in results.items() if not v]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
