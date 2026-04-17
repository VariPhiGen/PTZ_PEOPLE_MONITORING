"""
Pipeline Diagnostic Test
========================
Grabs a live frame from the RTSP stream, runs the full AI pipeline
(person detect → face detect → liveness → embedding), and saves an
annotated image so you can visually confirm each stage works.

Usage (inside backend container):
    python /app/scripts/test_pipeline.py [--rtsp <url>] [--out /tmp/out.jpg]

Or run from host via docker exec:
    docker exec acas-backend python /scripts/test_pipeline.py
"""
import argparse
import asyncio
import sys
import time
import os

import cv2
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_RTSP = "rtsp://172.16.16.206:554/video/live?channel=1&subtype=0&unicast=true&proto=Onvif"
DEFAULT_OUT  = "/tmp/pipeline_test.jpg"
MODEL_DIR    = os.environ.get("MODEL_DIR", "/models")
GPU_DEVICE   = int(os.environ.get("GPU_DEVICE_ID", "0"))

# ── Colours ───────────────────────────────────────────────────────────────────
C_PERSON = (0, 200, 0)     # green  — person box
C_FACE   = (255, 120, 0)   # blue   — face box
C_LIVE   = (0, 255, 255)   # yellow — liveness label
C_DEAD   = (0, 0, 255)     # red    — spoof label


def grab_frame_ffmpeg(rtsp_url: str, timeout: int = 10) -> np.ndarray | None:
    """Pull one frame via subprocess ffmpeg → BGR numpy array."""
    import subprocess, shlex
    cmd = (
        f"ffmpeg -loglevel error -rtsp_transport tcp "
        f"-i \"{rtsp_url}\" "
        f"-frames:v 1 -f image2 -vcodec mjpeg pipe:1"
    )
    try:
        result = subprocess.run(
            shlex.split(cmd), capture_output=True, timeout=timeout
        )
        if result.returncode != 0 or not result.stdout:
            print(f"[ERROR] ffmpeg failed: {result.stderr.decode()[:200]}")
            return None
        arr = np.frombuffer(result.stdout, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame
    except subprocess.TimeoutExpired:
        print("[ERROR] ffmpeg timed out grabbing frame")
        return None


def draw_results(frame, persons, faces_with_embeddings):
    """Draw bounding boxes and labels on a copy of frame."""
    out = frame.copy()
    h, w = out.shape[:2]

    for p in persons:
        x1, y1, x2, y2 = [int(v) for v in p.bbox]
        cv2.rectangle(out, (x1, y1), (x2, y2), C_PERSON, 2)
        cv2.putText(out, f"PERSON {p.confidence:.2f}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_PERSON, 1)

    for fwe in faces_with_embeddings:
        fx1, fy1, fx2, fy2 = [int(v) for v in fwe.face.bbox]
        live = fwe.liveness >= 0.40
        colour = C_LIVE if live else C_DEAD
        label  = f"LIVE {fwe.liveness:.2f}" if live else f"SPOOF {fwe.liveness:.2f}"
        cv2.rectangle(out, (fx1, fy1), (fx2, fy2), colour, 2)
        cv2.putText(out, label, (fx1, fy1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
        iod = getattr(fwe.face, "inter_ocular_px", 0)
        cv2.putText(out, f"IOD={iod:.0f}px", (fx1, fy2 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1)

    # Summary banner
    summary = (
        f"Persons: {len(persons)}  "
        f"Faces: {len(faces_with_embeddings)}  "
        f"Live: {sum(1 for f in faces_with_embeddings if f.liveness >= 0.40)}"
    )
    cv2.rectangle(out, (0, 0), (w, 28), (0, 0, 0), -1)
    cv2.putText(out, summary, (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtsp",  default=DEFAULT_RTSP)
    parser.add_argument("--out",   default=DEFAULT_OUT)
    parser.add_argument("--frames", type=int, default=3,
                        help="Number of frames to test (default 3)")
    args = parser.parse_args()

    # ── Step 1: Load AI pipeline ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 1 — Loading AI pipeline")
    print(f"{'='*60}")
    sys.path.insert(0, "/app")
    try:
        from app.services.ai_pipeline import AIPipeline
        t0 = time.time()
        pipeline = AIPipeline(model_dir=MODEL_DIR, device_id=GPU_DEVICE)
        print(f"  AIPipeline instance created, calling load() ...")
        pipeline.load()
        print(f"  Pipeline loaded in {time.time()-t0:.1f}s")
        print(f"  Models dir: {MODEL_DIR}")
    except Exception as exc:
        print(f"  [FAIL] Could not load pipeline: {exc}")
        sys.exit(1)

    # ── Step 2: Grab frame from RTSP ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 2 — Grabbing frame from RTSP")
    print(f"{'='*60}")
    print(f"  URL: {args.rtsp}")
    frame = grab_frame_ffmpeg(args.rtsp)
    if frame is None:
        print("  [FAIL] Could not grab frame — is the camera online?")
        sys.exit(1)
    h, w = frame.shape[:2]
    print(f"  Frame shape: {w}x{h}  dtype={frame.dtype}")
    # Save raw frame
    raw_path = args.out.replace(".jpg", "_raw.jpg")
    cv2.imwrite(raw_path, frame)
    print(f"  Raw frame saved → {raw_path}")

    # ── Step 3: Run pipeline ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"STEP 3 — Running AI pipeline ({args.frames} frames)")
    print(f"{'='*60}")

    all_results = []
    for i in range(args.frames):
        t0 = time.time()
        result = pipeline.process_frame(frame, roi_rect=None)
        elapsed = time.time() - t0

        persons = result.persons
        fwes    = result.faces_with_embeddings
        print(f"\n  Frame {i+1}/{args.frames}  ({elapsed*1000:.0f}ms)")
        print(f"    Persons detected : {len(persons)}")
        if persons:
            for j, p in enumerate(persons):
                print(f"      [{j}] conf={p.confidence:.3f}  bbox={[int(v) for v in p.bbox]}")
        print(f"    Faces detected   : {len(fwes)}")
        if fwes:
            for j, f in enumerate(fwes):
                iod  = getattr(f.face, "inter_ocular_px", "?")
                live = f.liveness
                emb_norm = float(np.linalg.norm(f.embedding)) if f.embedding is not None else 0
                live_str = f"LIVE ({live:.3f})" if live >= 0.40 else f"SPOOF ({live:.3f})"
                print(f"      [{j}] {live_str}  IOD={iod:.1f}px  "
                      f"emb_norm={emb_norm:.4f}  bbox={[int(v) for v in f.face.bbox]}")
        else:
            print("    (No faces — room may be empty or person too far)")
        all_results.append(result)

    # ── Step 4: Save annotated image ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 4 — Saving annotated image")
    print(f"{'='*60}")
    # Use result with most detections
    best = max(all_results, key=lambda r: len(r.persons) + len(r.faces_with_embeddings))
    annotated = draw_results(frame, best.persons, best.faces_with_embeddings)
    cv2.imwrite(args.out, annotated)
    print(f"  Annotated image saved → {args.out}")

    # ── Step 5: FAISS check ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 5 — FAISS / face repo check (via running backend)")
    print(f"{'='*60}")
    try:
        import urllib.request, json
        req = urllib.request.urlopen("http://localhost:8000/api/persons/?limit=5", timeout=3)
        data = json.loads(req.read())
        total = data.get("total", len(data.get("items", data if isinstance(data, list) else [])))
        print(f"  Enrolled persons in DB: {total}")
    except Exception as exc:
        print(f"  Could not reach backend API: {exc}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    total_persons = sum(len(r.persons) for r in all_results)
    total_faces   = sum(len(r.faces_with_embeddings) for r in all_results)
    total_live    = sum(
        sum(1 for f in r.faces_with_embeddings if f.liveness >= 0.40)
        for r in all_results
    )
    print(f"  Pipeline:       OK")
    print(f"  RTSP stream:    OK  ({w}x{h})")
    print(f"  Persons seen:   {total_persons} across {args.frames} frames")
    print(f"  Faces seen:     {total_faces} across {args.frames} frames")
    print(f"  Live faces:     {total_live}")
    if total_persons == 0:
        print("\n  NOTE: No persons detected — room appears empty.")
        print("        Try standing in front of the camera and re-running,")
        print("        or check the RTSP URL is correct.")
    elif total_faces == 0:
        print("\n  NOTE: Persons detected but no faces.")
        print("        Try facing the camera directly (not from behind/side).")
    elif total_live == 0:
        print("\n  NOTE: Faces detected but all flagged as SPOOF.")
        print("        Liveness threshold is 0.40 — try better lighting.")
    else:
        print(f"\n  All pipeline stages working correctly.")
    print(f"\n  Output images:")
    print(f"    Raw frame  : {raw_path}")
    print(f"    Annotated  : {args.out}")


if __name__ == "__main__":
    main()
