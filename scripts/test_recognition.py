"""
Face Recognition Flow Test — Image injection + Real PTZ camera moves
=====================================================================

Tests the complete ACAS recognition pipeline end-to-end using:
  - A LOCAL IMAGE (instead of RTSP) as the simulated camera view
  - REAL ONVIF PTZ moves on the physical camera (zoom in/out, pan/tilt)
  - The live AIPipeline for person + face detection + liveness + embedding
  - The live FAISS index for identification

This validates the entire intelligence stack without needing someone to
stand in front of the camera. The camera physically moves (so ONVIF
control is verified) but recognition runs on your test image.

Usage (inside backend container):
    python /app/scripts/test_recognition.py \
        --image /tmp/srk_faces/srk1.jpg \
        [--expected-id <person_uuid>] \
        [--out /tmp/recognition_test.jpg]

Or from host:
    docker exec acas-backend bash -c \
        "cd /app && python scripts/test_recognition.py --image /tmp/test.jpg"

Flow:
  1. Load AI pipeline + FAISS index
  2. Verify ONVIF PTZ camera connection
  3. Load test image (simulates what camera would see)
  4. Run person + face detection on image
  5. For each face found:
     a. Compute target PTZ (as if face were at that pixel in the real frame)
     b. Move camera to that position (real ONVIF move)
     c. Wait for camera to settle
     d. Run ArcFace embedding on face crop
     e. Query FAISS for identity
     f. Annotate result on image
  6. Restore camera to original position
  7. Save annotated image + print report
"""
import argparse
import asyncio
import sys
import time
import os

import cv2
import numpy as np

# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_OUT      = "/tmp/recognition_test.jpg"
MODEL_DIR        = os.environ.get("MODEL_DIR", "/models")
GPU_DEVICE       = int(os.environ.get("GPU_DEVICE_ID", "0"))

# Camera from DB (auto-detected from running backend)
DEFAULT_CAM_ID   = "6f440e96-7e27-4080-8b71-bd54f4722a79"
DEFAULT_RTSP     = "rtsp://172.16.16.206:554/video/live?channel=1&subtype=0&unicast=true&proto=Onvif"
DEFAULT_ONVIF    = "172.16.16.206"
DEFAULT_ONVIF_P  = 80
DEFAULT_ONVIF_U  = "admin"
DEFAULT_ONVIF_PW = "Cp@Plus#123"

# Recognition threshold
FAISS_THRESHOLD  = 0.40


def draw_result(img: np.ndarray, persons, faces_with_embeddings, id_results: dict) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]

    for p in persons:
        x1, y1, x2, y2 = [int(v) for v in p.bbox]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(out, f"PERSON {p.conf:.2f}", (x1, max(y1-5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,0), 1)

    for idx, fwe in enumerate(faces_with_embeddings):
        fx1, fy1, fx2, fy2 = [int(v) for v in fwe.face.bbox]
        id_res = id_results.get(idx)
        if id_res and id_res.person_id:
            colour = (0, 255, 100)
            label  = f"ID: {id_res.person_id[:16]}  sim={id_res.similarity:.3f}"
        elif fwe.liveness < 0.40:
            colour = (0, 0, 220)
            label  = f"SPOOF {fwe.liveness:.2f}"
        else:
            colour = (255, 120, 0)
            label  = f"Unknown  live={fwe.liveness:.2f}"
        cv2.rectangle(out, (fx1, fy1), (fx2, fy2), colour, 2)
        cv2.putText(out, label, (fx1, max(fy1-6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
        iod = getattr(fwe.face, "inter_ocular_px", 0)
        cv2.putText(out, f"IOD {iod:.0f}px", (fx1, fy2+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 0), 1)

    # Summary bar
    n_id = sum(1 for r in id_results.values() if r and r.person_id)
    summary = f"Persons:{len(persons)}  Faces:{len(faces_with_embeddings)}  Recognised:{n_id}"
    cv2.rectangle(out, (0, 0), (w, 28), (20,20,20), -1)
    cv2.putText(out, summary, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    return out


async def run_test(args):
    sys.path.insert(0, "/app")

    # ── Step 1: Load pipeline ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 1 — Loading AI pipeline")
    print(f"{'='*60}")
    from app.services.ai_pipeline import AIPipeline
    t0 = time.time()
    pipeline = AIPipeline(model_dir=MODEL_DIR, device_id=GPU_DEVICE)
    pipeline.load()
    print(f"  Pipeline loaded in {time.time()-t0:.1f}s")

    # ── Step 2: ONVIF connection ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 2 — Connecting to PTZ camera via ONVIF")
    print(f"{'='*60}")
    from app.services.onvif_controller import ONVIFController, CameraOfflineError, PTZPosition
    onvif = ONVIFController(
        host=args.onvif_host, port=args.onvif_port,
        username=args.onvif_user, password=args.onvif_pass,
    )
    try:
        await onvif.connect()
        ptz_home = await onvif.get_ptz_status()
        print(f"  Connected!  Current PTZ: pan={ptz_home.pan:.3f}  tilt={ptz_home.tilt:.3f}  zoom={ptz_home.zoom:.3f}")
        print(f"  Limits: pan=[{onvif.limits.pan_min:.2f},{onvif.limits.pan_max:.2f}]  "
              f"tilt=[{onvif.limits.tilt_min:.2f},{onvif.limits.tilt_max:.2f}]  "
              f"zoom=[{onvif.limits.zoom_min:.2f},{onvif.limits.zoom_max:.2f}]")
    except CameraOfflineError:
        print("  [WARN] Camera offline — PTZ moves will be skipped, recognition only")
        onvif = None
        ptz_home = PTZPosition(0.0, 0.0, 0.0)

    # ── Step 3: Load test image ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 3 — Loading test image")
    print(f"{'='*60}")
    frame = cv2.imread(args.image)
    if frame is None:
        print(f"  [ERROR] Cannot read image: {args.image}")
        sys.exit(1)
    h, w = frame.shape[:2]
    print(f"  Image: {args.image}  size={w}x{h}")
    raw_path = args.out.replace(".jpg", "_raw.jpg")
    cv2.imwrite(raw_path, frame)

    # ── Step 4: Detect persons + faces ───────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 4 — Running person + face detection")
    print(f"{'='*60}")
    t0 = time.time()
    result = pipeline.process_frame(frame, roi_rect=None)
    print(f"  Inference: {(time.time()-t0)*1000:.0f}ms")
    print(f"  Persons:   {len(result.persons)}")
    for i, p in enumerate(result.persons):
        print(f"    [{i}] conf={p.conf:.3f}  bbox={[int(v) for v in p.bbox]}")
    print(f"  Faces:     {len(result.faces_with_embeddings)}")
    for i, fwe in enumerate(result.faces_with_embeddings):
        iod = getattr(fwe.face, "inter_ocular_px", 0)
        print(f"    [{i}] IOD={iod:.1f}px  liveness={fwe.liveness:.3f}  bbox={[int(v) for v in fwe.face.bbox]}")

    if not result.faces_with_embeddings:
        print("\n  No faces detected. Check that the image contains a clear front-facing face.")
        cv2.imwrite(args.out, frame)
        return

    # ── Step 5: PTZ + Recognition per face ───────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 5 — PTZ moves + face recognition")
    print(f"{'='*60}")

    # Build in-process FAISS index from DB embeddings
    faiss_index   = None
    index_ids     = []    # list of person_id strings, parallel to FAISS vectors
    index_names   = {}    # person_id → name
    client_id     = None
    try:
        import asyncpg
        import faiss as _faiss
        db_url = os.environ.get("DATABASE_URL", "").replace("+asyncpg", "")
        if not db_url:
            raise RuntimeError("DATABASE_URL not set")
        conn = await asyncpg.connect(db_url, timeout=5)
        # Pick first client (test env has one client)
        rows = await conn.fetch("SELECT client_id FROM clients LIMIT 1")
        if not rows:
            raise RuntimeError("No clients in DB")
        client_id = str(rows[0]["client_id"])

        # Fetch all embeddings for this client
        emb_rows = await conn.fetch(
            "SELECT fe.person_id::text, fe.embedding, p.name "
            "FROM face_embeddings fe "
            "JOIN persons p ON p.person_id = fe.person_id "
            "WHERE p.client_id = $1::uuid",
            client_id,
        )
        await conn.close()

        if emb_rows:
            dim = 512
            vecs = []
            for r in emb_rows:
                raw = r["embedding"]
                if isinstance(raw, (bytes, bytearray)):
                    vec = np.frombuffer(raw, dtype=np.float32)
                elif isinstance(raw, list):
                    vec = np.array(raw, dtype=np.float32)
                else:
                    # pgvector returns string like "[0.1,0.2,...]"
                    import json as _json
                    vec = np.array(_json.loads(str(raw).replace("{","[").replace("}","]")), dtype=np.float32)
                if vec.shape[0] != dim:
                    continue
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                vecs.append(vec)
                index_ids.append(r["person_id"])
                index_names[r["person_id"]] = r["name"]

            if vecs:
                mat = np.stack(vecs).astype(np.float32)
                faiss_index = _faiss.IndexFlatIP(dim)
                faiss_index.add(mat)
                print(f"  FAISS index: {faiss_index.ntotal} embeddings for client {client_id[:8]}...")
                # Show enrolled persons
                unique_persons = {}
                for pid, name in index_names.items():
                    unique_persons[pid] = name
                for pid, name in unique_persons.items():
                    cnt = sum(1 for x in index_ids if x == pid)
                    print(f"    {name[:30]}  ({cnt} embedding(s))  id={pid[:16]}...")
        else:
            print("  [WARN] No embeddings in DB — enroll a person first")
    except Exception as exc:
        print(f"  [WARN] Could not build FAISS index: {exc}")
        print("         Recognition will show embedding norm only (no name lookup)")

    id_results: dict = {}

    for idx, fwe in enumerate(result.faces_with_embeddings):
        bbox = fwe.face.bbox
        face_cx = (bbox[0] + bbox[2]) / 2.0
        face_cy = (bbox[1] + bbox[3]) / 2.0
        iod = getattr(fwe.face, "inter_ocular_px", 0)

        print(f"\n  Face [{idx}]  IOD={iod:.0f}px  liveness={fwe.liveness:.3f}")

        # ── PTZ: zoom + pan/tilt to face ──────────────────────────────────────
        if onvif is not None:
            # Zoom level based on IOD (larger zoom for small faces)
            # Target: face fills ~20% of frame height → zoom = clip(base * scale, 0.1, 1.0)
            target_zoom = min(max(ptz_home.zoom + (80.0 / max(iod, 10.0)) * 0.1, 0.1), 0.9)
            target_pan, target_tilt = onvif.pixel_to_ptz(face_cx, face_cy, w, h, ptz_home)

            print(f"    Moving PTZ → pan={target_pan:.3f}  tilt={target_tilt:.3f}  zoom={target_zoom:.3f}")
            try:
                await onvif.absolute_move(target_pan, target_tilt, target_zoom, speed=0.5)
                await asyncio.sleep(1.5)   # settle time
                ptz_after = await onvif.get_ptz_status()
                print(f"    PTZ settled: pan={ptz_after.pan:.3f}  tilt={ptz_after.tilt:.3f}  zoom={ptz_after.zoom:.3f}")

                # Capture real frame from RTSP to show annotated result on zoomed view
                if args.rtsp:
                    import subprocess, shlex
                    cmd = (
                        f'ffmpeg -loglevel error -rtsp_transport tcp '
                        f'-i "{args.rtsp}" -frames:v 1 -f image2 -vcodec mjpeg pipe:1'
                    )
                    res = subprocess.run(shlex.split(cmd), capture_output=True, timeout=8)
                    if res.returncode == 0 and res.stdout:
                        arr = np.frombuffer(res.stdout, dtype=np.uint8)
                        zoomed_frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if zoomed_frame is not None:
                            # Run pipeline on zoomed frame
                            zoomed_result = pipeline.process_frame(zoomed_frame, roi_rect=None)
                            print(f"    Zoomed frame: {zoomed_frame.shape[1]}x{zoomed_frame.shape[0]}  "
                                  f"faces={len(zoomed_result.faces_with_embeddings)}")
                            if zoomed_result.faces_with_embeddings:
                                # Use the zoomed embedding for recognition
                                fwe = min(zoomed_result.faces_with_embeddings,
                                          key=lambda f: abs((f.face.bbox[0]+f.face.bbox[2])/2 - zoomed_frame.shape[1]/2))
                                zoom_iod = getattr(fwe.face, "inter_ocular_px", 0)
                                print(f"    Best face in zoomed frame: IOD={zoom_iod:.0f}px  live={fwe.liveness:.3f}")
                                # Save zoomed annotated frame
                                zoom_out = args.out.replace(".jpg", f"_face{idx}_zoom.jpg")
                                cv2.imwrite(zoom_out, zoomed_frame)
                                print(f"    Zoomed frame saved → {zoom_out}")
            except Exception as exc:
                print(f"    [WARN] PTZ move failed: {exc}")

        # ── Recognition via FAISS ─────────────────────────────────────────────
        if fwe.liveness < 0.40 and not args.skip_liveness:
            print(f"    Liveness FAILED ({fwe.liveness:.3f}) — skipping recognition (use --skip-liveness for photos)")
            id_results[idx] = None
            continue
        elif fwe.liveness < 0.40:
            print(f"    Liveness FAILED ({fwe.liveness:.3f}) — continuing anyway (--skip-liveness set)")

        if faiss_index is not None and fwe.embedding is not None:
            try:
                qvec = np.array(fwe.embedding, dtype=np.float32)
                norm = np.linalg.norm(qvec)
                if norm > 0:
                    qvec = qvec / norm
                D, I = faiss_index.search(qvec.reshape(1, -1), k=1)
                sim = float(D[0][0])
                best_idx = int(I[0][0])
                person_id = index_ids[best_idx] if best_idx >= 0 else None
                name = index_names.get(person_id, "?") if person_id else "?"

                class _IDResult:
                    pass
                id_res = _IDResult()
                id_res.similarity = sim
                if sim >= FAISS_THRESHOLD and person_id:
                    id_res.person_id = person_id
                    id_res.tier = "faiss"
                    print(f"    RECOGNISED: {name}  similarity={sim:.4f}  person_id={person_id[:16]}...")
                    if args.expected_id and person_id == args.expected_id:
                        print(f"    ✓ Matched expected person!")
                    elif args.expected_id:
                        print(f"    ✗ Expected {args.expected_id[:16]}... but got {person_id[:16]}...")
                else:
                    id_res.person_id = None
                    print(f"    NOT RECOGNISED (best sim={sim:.4f} < threshold={FAISS_THRESHOLD}, closest: {name})")
                id_results[idx] = id_res
            except Exception as exc:
                print(f"    [WARN] FAISS search failed: {exc}")
                id_results[idx] = None
        else:
            # No FAISS — just show embedding norm as sanity check
            emb_norm = float(np.linalg.norm(fwe.embedding)) if fwe.embedding is not None else 0
            print(f"    Embedding norm={emb_norm:.4f} (no FAISS — enroll a person first)")
            id_results[idx] = None

    # ── Step 6: Restore camera ────────────────────────────────────────────────
    if onvif is not None:
        print(f"\n{'='*60}")
        print("STEP 6 — Restoring camera to home position")
        try:
            await onvif.absolute_move(ptz_home.pan, ptz_home.tilt, ptz_home.zoom, speed=0.5)
            await asyncio.sleep(1.0)
            print(f"  Camera restored to pan={ptz_home.pan:.3f}  tilt={ptz_home.tilt:.3f}  zoom={ptz_home.zoom:.3f}")
        except Exception as exc:
            print(f"  [WARN] Restore failed: {exc}")

    # ── Step 7: Save annotated image ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 7 — Saving annotated result image")
    annotated = draw_result(frame, result.persons, result.faces_with_embeddings, id_results)
    cv2.imwrite(args.out, annotated)
    print(f"  Saved → {args.out}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    n_faces = len(result.faces_with_embeddings)
    n_live  = sum(1 for f in result.faces_with_embeddings if f.liveness >= 0.40)
    n_id    = sum(1 for r in id_results.values() if r and r.person_id)

    print(f"  Persons detected : {len(result.persons)}")
    print(f"  Faces detected   : {n_faces}")
    print(f"  Live faces       : {n_live}")
    print(f"  Recognised       : {n_id}")

    if n_id == n_live and n_live > 0:
        print(f"\n  ALL LIVE FACES RECOGNISED — pipeline working correctly.")
    elif n_live > 0 and n_id == 0:
        print(f"\n  FACES DETECTED but NOT RECOGNISED.")
        print("  → Enroll the person first using: python scripts/enroll_test_face.py")
        print(f"  → Or check the FAISS threshold ({FAISS_THRESHOLD}) in this script.")
    elif n_faces == 0:
        print(f"\n  NO FACES DETECTED in test image.")
        print("  → Use a clear front-facing photo with good lighting.")

    print(f"\n  Output files:")
    print(f"    Raw image       : {raw_path}")
    print(f"    Annotated result: {args.out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",       required=True, help="Path to test face image")
    parser.add_argument("--out",         default=DEFAULT_OUT)
    parser.add_argument("--expected-id", default="", help="Expected person_id (for pass/fail check)")
    parser.add_argument("--onvif-host",  default=DEFAULT_ONVIF)
    parser.add_argument("--onvif-port",  type=int, default=DEFAULT_ONVIF_P)
    parser.add_argument("--onvif-user",  default=DEFAULT_ONVIF_U)
    parser.add_argument("--onvif-pass",  default=DEFAULT_ONVIF_PW)
    parser.add_argument("--rtsp",          default=DEFAULT_RTSP,
                        help="RTSP URL to grab zoomed frame from camera after PTZ move")
    parser.add_argument("--skip-liveness", action="store_true",
                        help="Skip liveness check (use when testing with static photos)")
    args = parser.parse_args()
    asyncio.run(run_test(args))


if __name__ == "__main__":
    main()
