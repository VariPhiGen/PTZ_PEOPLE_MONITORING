# ACAS — Attendance & Camera Automation System

GPU-accelerated, multi-tenant face recognition and automated PTZ camera management. Recognises enrolled persons in real time from security cameras, logs attendance, and drives PTZ cameras autonomously to scan and zoom in on every person in the room.

---

## Quick Start

```bash
cp .env.example .env        # fill in passwords, NODE_NAME
docker compose build backend
docker compose up -d

# DB migrations
docker exec -w /app acas-backend alembic upgrade head
docker exec -w /app acas-backend alembic_ts upgrade head

# Create admin account
docker exec -w /app acas-backend python scripts/create_superadmin.py \
  --email admin@acas.local --password "Admin@2024!" --name "Platform Admin"

# Download AI models
docker exec acas-backend python /tmp/download_models.py --model-dir /models
```

**Dashboard** → http://localhost:3020
**Backend API / Swagger** → http://localhost:18000/docs

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Code Structure](#2-code-structure)
3. [AI Inference Pipeline](#3-ai-inference-pipeline)
4. [Face Recognition Pipeline](#4-face-recognition-pipeline)
5. [PTZ Brain — State Machine](#5-ptz-brain--state-machine)
6. [Enrollment Pipeline](#6-enrollment-pipeline)
7. [Face Search Pipeline](#7-face-search-pipeline)
8. [Attendance Engine](#8-attendance-engine)
9. [Multi-Tenancy & Security](#9-multi-tenancy--security)
10. [Infrastructure & Ports](#10-infrastructure--ports)
11. [Key Environment Variables](#11-key-environment-variables)
12. [API Reference](#12-api-reference)
13. [Testing](#13-testing)
14. [Operations](#14-operations)

---

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     IP CAMERAS  (ONVIF / RTSP)                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │ RTSP H.264 stream
                           ▼
              ┌────────────────────────┐
              │    rtsp_decoder.py     │  GPU h264_cuvid decode
              │    async frame queue   │  10 fps, 5-frame buffer
              └───────────┬────────────┘
                          │ BGR numpy frames
                          ▼
  ┌───────────────────────────────────────────────────────────┐
  │                    ai_pipeline.py                          │
  │                                                            │
  │  Stage 1  YOLOv8l  →  person bboxes + 17-pt pose    │
  │  Stage 2  SCRFD-10G     →  face bbox + 5-pt landmarks     │
  │  Stage 2.5 Quality Gate →  IOD / conf / yaw / sharpness   │
  │  Stage 3  ArcFace IR101 →  512-D L2 face embedding        │
  │  Stage 4  MiniFASNet v2 →  liveness score 0.0 – 1.0       │
  └───────────────────────┬───────────────────────────────────┘
                          │ FrameResult
                          ▼
              ┌───────────────────────┐        ┌──────────────────────┐
              │     ptz_brain.py      │───────►│  face_repository.py  │
              │  Autonomous PTZ       │        │  FAISS Tier-1        │
              │  state machine        │        │  pgvector Tier-2     │
              └────────┬──────────────┘        └──────────────────────┘
                       │
           ┌───────────┼──────────────┐
           ▼           ▼              ▼
    onvif_          attend-       sighting_
    controller      engine        engine
    (PTZ moves)     (PRESENT/     (dwell
                    LATE/ABS)      tracking)
           │           │              │
           └───────────┴──────────────┴──► Kafka → dashboard / ERP
```

### Component Responsibilities

| Component | File | Role |
|-----------|------|------|
| Frame decoder | `rtsp_decoder.py` | GPU RTSP → async frame buffer |
| Inference engine | `ai_pipeline.py` | All ONNX GPU inference, quality gate |
| Identity store | `face_repository.py` | FAISS + pgvector two-tier search |
| PTZ controller | `ptz_brain.py` | Autonomous scanning state machine |
| ONVIF driver | `onvif_controller.py` | PTZ hardware commands + FOV math |
| Attendance | `attendance_engine.py` | PRESENT / LATE / ABSENT records |
| Dwell tracking | `sighting_engine.py` | Enter / exit timestamps per person |
| Cross-camera | `cross_camera.py` | Multi-camera re-identification |
| Embedding sync | `face_sync.py` | Kafka-based cross-node FAISS sync |
| Node manager | `node_manager.py` | Control Plane heartbeat + camera assignment |
| GPU concurrency | `gpu_manager.py` | Semaphore, VRAM budget, graceful degradation |

---

## 2. Code Structure

```
ACAS/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI app, lifespan startup sequence
│   │   ├── config.py                # Pydantic settings (env vars)
│   │   ├── deps.py                  # FastAPI DI: DB session, auth
│   │   ├── api/                     # HTTP route handlers
│   │   │   ├── auth.py              # Login, refresh, JWT
│   │   │   ├── cameras.py           # Camera CRUD, MJPEG stream, PTZ presets
│   │   │   ├── enrollment.py        # Face enrollment, quality gate, bulk import
│   │   │   ├── search.py            # Image search, person journey, area query
│   │   │   ├── attendance.py        # Attendance records, reports
│   │   │   ├── datasets.py          # Dataset CRUD, person listing
│   │   │   ├── sessions.py          # Active PTZ session management
│   │   │   ├── analytics.py         # Trend reports, accuracy, uptime
│   │   │   ├── node.py              # Node-local camera start / stop / stream
│   │   │   ├── admin.py             # Super-admin: clients, nodes, users
│   │   │   └── monitoring.py        # Live session state, GPU metrics
│   │   ├── services/                # Business logic and AI  ← start here
│   │   │   ├── ai_pipeline.py       ★ Full GPU inference pipeline
│   │   │   ├── face_repository.py   ★ FAISS + pgvector identity store
│   │   │   ├── ptz_brain.py         ★ PTZ autonomous scanning brain
│   │   │   ├── onvif_controller.py  ★ ONVIF PTZ execution + FOV math
│   │   │   ├── rtsp_decoder.py      ★ GPU RTSP decoder
│   │   │   ├── attendance_engine.py ★ Sightings → attendance records
│   │   │   ├── sighting_engine.py     Dwell-time sighting tracker
│   │   │   ├── face_search.py         Multi-modal search + journey analytics
│   │   │   ├── face_sync.py           Kafka embedding sync across nodes
│   │   │   ├── node_manager.py        Control Plane heartbeat + registration
│   │   │   ├── cross_camera.py        Cross-camera person re-identification
│   │   │   ├── zone_mapper.py         PTZ ↔ pixel coordinate mapping
│   │   │   ├── path_planner.py        Greedy TSP scan path optimisation
│   │   │   └── self_learner.py        Nightly threshold auto-tuning
│   │   ├── core/
│   │   │   ├── gpu_manager.py       GPU semaphore, VRAM budget, degradation
│   │   │   └── metrics.py           Prometheus metrics exporters
│   │   ├── middleware/
│   │   │   ├── auth.py              JWT validation, RLS client_id injection
│   │   │   └── perf.py              Request timing middleware
│   │   └── models/                  SQLAlchemy ORM table definitions
│   ├── alembic/                     PostgreSQL migrations
│   ├── alembic_ts/                  TimescaleDB migrations
│   └── tests/
│       ├── conftest.py              Fixtures, MockAIPipeline
│       ├── test_attendance_engine.py
│       ├── test_face_repository.py
│       └── test_api.py
│
├── dashboard/                       Next.js 14 frontend
│   └── src/
│       ├── app/(dashboard)/         Route pages
│       │   ├── cameras/             Camera management + live stream
│       │   ├── enrollment/          Face enrollment wizard
│       │   ├── search/              Face image + text search
│       │   ├── attendance/          Attendance records + reports
│       │   ├── analytics/           Charts and trend dashboards
│       │   └── datasets/            Face dataset management
│       ├── components/              Shared UI components (shadcn/ui)
│       ├── lib/
│       │   ├── api.ts               Axios client, JWT attach, refresh
│       │   └── auth.ts              JWT cookie helpers, getPayload()
│       └── types/index.ts           TypeScript interfaces
│
├── control-plane/                   Cloudflare Worker (optional multi-node registry)
├── scripts/                         Ops utilities: start, backup, model download
├── models/                          ONNX model files (downloaded at runtime)
├── grafana/                         Prometheus datasource + dashboard JSON
├── prometheus/                      Scrape config
└── docker-compose*.yml
```

★ = Core intelligence — start here when extending ACAS.

---

## 3. AI Inference Pipeline

Every camera frame runs through a single call: `pipeline.process_frame(frame, roi_rect)`.
The pipeline has **4 inference stages** plus a **CPU quality gate** between stages 2 and 3.

### Stage 1 — Person Detection (YOLOv8l)

| Property | Value |
|----------|-------|
| Model | `yolov8l.onnx` → TensorRT FP16 |
| Input size | 640 × 640 (letterboxed BGR) |
| Output | Person bboxes + 17 COCO keypoints |
| Confidence threshold | ≥ 0.30 |
| NMS IOU | 0.45 |
| Trigger | Every frame |

What happens:
- Frame resized with letterbox padding to 640×640
- Returns `[N, 56]` tensor: 4 bbox coords + 1 conf + 17 × 3 keypoints (x, y, visibility)
- `is_standing` flag derived from shoulder-to-hip keypoint ratio
- Persons outside `roi_rect` (if configured on the camera) are discarded
- **If no persons → pipeline returns immediately.** Face stages are never called.

---

### Stage 2 — Face Detection (SCRFD-10G)

| Property | Value |
|----------|-------|
| Model | `buffalo_l/det_10g.onnx` → TensorRT FP16 |
| Input size | 640 × 640 |
| Output | Face bboxes + 5-point landmarks |
| Detection threshold | ≥ 0.50 (set in SCRFD.prepare) |
| Trigger | Only when Stage 1 detected ≥ 1 person |

What happens:
- SCRFD returns face bboxes + landmarks: left eye, right eye, nose tip, left/right mouth corners
- Each face gets `inter_ocular_px` (IOD) = Euclidean distance between eye landmarks
- **Person association filter**: face centre must fall within the upper 60% of a person bbox — faces not associated with any person body are discarded
- Results sorted by confidence descending

---

### Stage 2.5 — Runtime Quality Gate (CPU, no GPU cost)

**Trigger:** after SCRFD, before ArcFace + MiniFASNet
**Purpose:** reject low-quality faces early — saves 50–200 ms GPU per rejected face in crowd scenes

Four checks run in order; first failure discards the face:

| Check | Threshold | What it rejects |
|-------|-----------|----------------|
| Face confidence | ≥ 0.45 | Borderline detections |
| Inter-ocular distance | ≥ 20 px | Distant / tiny faces |
| Yaw angle | ≤ ±50° | Strong profile views |
| Laplacian sharpness | ≥ 4.0 | Motion blur / out-of-focus |

**Yaw estimation** (from SCRFD 5-point landmarks, no extra model needed):
```
yaw ≈ (nose_x − mid_eye_x) / (eye_span / 2) × 50°
```

Faces that pass all four checks proceed to alignment. The count of rejected faces is reported in `breakdown["quality_rejected"]`.

---

### Stage 3 — Face Alignment + Embedding (AdaFace IR-101)

| Property | Value |
|----------|-------|
| Model | `adaface_ir101_webface12m.onnx` → TensorRT FP16 |
| Input size | 112 × 112 BGR canonical chip |
| Output | 512-D float32 L2-normalised embedding |
| Batching | All faces in one frame sent as a single ONNX batch |
| Trigger | Every face that passed Stage 2.5 |

**Alignment steps** (`align_face` in `ai_pipeline.py`):

1. **Super-Resolution** — when IOD < 60 px (small / distant CCTV face):
   - Face bounding region is 2× upsampled before warping
   - Default: LANCZOS4 bicubic + unsharp mask sharpening
   - Optional: Real-ESRGAN ONNX (`models/realesrgan_x2.onnx`) for neural SR

2. **Affine warp** — `cv2.estimateAffinePartial2D` maps the 5 SCRFD landmarks to the ArcFace canonical 112×112 grid. Handles rotation, scale, and translation in one transform.

3. **CLAHE** — Contrast Limited Adaptive Histogram Equalisation, applied per channel in LAB colour space (`clipLimit=2.0, tileGridSize=(8,8)`). Normalises the lighting difference between studio enrollment photos and surveillance camera frames.

---

### Stage 4 — Liveness Detection (MiniFASNet v2)

| Property | Value |
|----------|-------|
| Model | `minifasnet_v2.onnx` → CUDA FP32 |
| Input size | 80 × 80 BGR |
| Output | `liveness` score: 0.0 (spoof) → 1.0 (live) |
| Trigger | Every face with a valid embedding |
| Degradation | Dropped first when VRAM budget is exceeded |

Liveness is checked at two points in the system:
- **Live scanning** (`ptz_brain.py`): low liveness score blocks the FAISS lookup
- **Enrollment** (`face_repository.py`): requires `liveness ≥ 0.85` before updating a stored template

---

### Pipeline Output

```python
@dataclass
class FrameResult:
    persons:               list[PersonDetection]
    faces_with_embeddings: list[FaceWithEmbedding]
    breakdown: dict  # keys: yolo_ms, scrfd_ms, quality_ms, embed_ms, live_ms,
                     #        quality_rejected (count of Stage 2.5 rejects)
```

### Timing Budget (RTX A5000, 1080p, 15 faces)

| Stage | Time |
|-------|------|
| YOLO person + pose | < 20 ms |
| SCRFD face detect | < 15 ms |
| Quality gate | < 1 ms (CPU) |
| ArcFace embed (batch 15) | < 45 ms |
| MiniFASNet liveness (batch 15) | < 20 ms |
| **Total** | **< 100 ms** |

---

## 4. Face Recognition Pipeline

A face with an embedding is identified through a **two-tier search** in `face_repository.py`:

```
FaceWithEmbedding  (512-D L2 embedding)
        │
        ▼
┌──────────────────────────────────────────────────┐
│  Tier-1: FAISS HNSW  (in-memory, sub-2ms)        │
│  Index type: IndexHNSWFlat, cosine similarity     │
│  Build params: M=32, efConstruction=200, ef=64   │
│  Scope: current session roster (fast path)        │
│  Threshold: similarity ≥ 0.40                     │
│  Active when: dataset has ≥ 20 persons            │
└──────────────┬───────────────────────────────────┘
               │ hit ≥ 0.40          miss
               ▼                       ▼
        RECOGNISED           ┌─────────────────────────────┐
        return person_id     │  Tier-2: pgvector HNSW      │
                             │  cosine search in Postgres   │
                             │  Scope: all client datasets  │
                             │  Threshold: similarity ≥ 0.45│
                             └──────────┬──────────────────┘
                                        │ hit ≥ 0.45    miss
                                        ▼                  ▼
                                 RECOGNISED            UNKNOWN
                                 return person_id
```

### Recognition in the PTZ Loop

During `PRESET_RECOGNIZE`, the brain runs a **3-level fast-path** before hitting FAISS:

| Level | Method | Cost |
|-------|--------|------|
| 1. IoU tracking | Same bbox as last frame → same identity | 0 ms |
| 2. Embedding cache | Seen this person this cycle → reuse cached `person_id` | 0 ms |
| 3. Cross-preset re-ID | Camera moved, IoU failed; cosine ≥ 0.62 on cached embed | ~0.1 ms |
| 4. FAISS Tier-1 | Full index search | ~2 ms |
| 5. pgvector Tier-2 | DB query fallback | ~15 ms |

Only persons reaching level 4 or 5 cause a real GPU/DB operation.

### FACE_HUNT — High-Quality Recognition

When a person cannot be identified during the dwell loop, the brain enters `FACE_HUNT`:

1. **Pre-zoom** — smooth PTZ move to centre the face. Target: `IOD = 130 px` in frame (derived from FOV calibration constants `K_pan`, `K_tilt`). Budget: 12 s.
2. **Frame flush** — discard stale RTSP buffer frames accumulated during the move.
3. **3-frame embedding average** — collect 3 embeddings at the stable position, compute L2-normalised mean: `avg = normalise(mean([e1, e2, e3]))`. Reduces noise by ~0.03–0.07 cosine units.
4. **FAISS lookup** on `avg`.
5. **Angle micro-sweep** (if still unknown) — pan ±0.012 ONVIF units (~1–2°) and retry once per direction. Overcomes cases where the face is borderline front-on but SCRFD landmarks have sub-pixel error.
6. Total budget: **22 s** per face, **3 attempts** max.

### Recognition Thresholds Summary

| Context | Threshold | Notes |
|---------|-----------|-------|
| FAISS Tier-1 (roster) | ≥ 0.40 | Fast in-session path |
| pgvector Tier-2 (all datasets) | ≥ 0.45 | Broader institution search |
| Cross-preset re-ID (cache) | ≥ 0.62 | High-confidence, no DB call |
| FACE_HUNT (averaged embed) | ≥ 0.40 | After 3-frame averaging |
| Face Search API (user upload) | ≥ 0.40 | Top-10 hits returned |
| Enrollment template update | liveness ≥ 0.85 | Anti-spoof guard |

---

## 5. PTZ Brain — State Machine

`ptz_brain.py` — one instance per active camera session. Drives the camera autonomously with no human input.

### State Diagram

```
IDLE
 │  (presets loaded)
 ▼
PRESET_TRANSIT ──► moves camera to next preset via ONVIF absolute_move()
 │                  speed=0.15, waits for position confirm
 │ (arrived)
 ▼
PRESET_RECOGNIZE ──► dwell loop
 │   • Grab frames at ~10 fps
 │   • Run ai_pipeline.process_frame()
 │   • Quick-recognise: IoU track → embed cache → FAISS
 │   • Adaptive exit: all persons resolved for 3 consecutive frames
 │   • Max dwell: configured dwell_s × 2.5 (extension for unresolved persons)
 │   • Min dwell: 2.0 s
 │
 ├── unknowns remain ──► FACE_HUNT
 │                         • Pre-zoom to IOD=130px
 │                         • Flush stale frames
 │                         • 3-frame embed average → FAISS
 │                         • Angle micro-sweep ±1–2° if still unknown
 │                         • Budget: 22 s / face
 │                         └──► back to PRESET_RECOGNIZE
 │
 └── all resolved ──► PRESET_COMPLETE
                         • Log results, advance preset index
                         │
                         └── last preset ──► CYCLE_COMPLETE
                                               • Reorder presets by face count
                                               • Reset occupancy counters
                                               • Write sightings + attendance
                                               └──► PRESET_TRANSIT (next cycle)
```

### Key Constants

| Constant | Value | Effect |
|----------|-------|--------|
| `_CAMERA_MOVE_SPEED` | 0.15 | Preset transit speed (slow for sharp frames) |
| `_PRECISION_MOVE_SPEED` | 0.08 | Face-hunt zoom speed |
| `_TRACK_MAX_SPEED` | 0.10 | P-controller max velocity |
| `_TRACK_DEAD_ZONE` | 0.05 | Normalised pixel dead zone (stops micro-jitter) |
| `_MIN_DWELL_S` | 2.0 s | Minimum dwell before adaptive exit |
| `_ADAPTIVE_DWELL_STREAK` | 3 | Consecutive all-resolved frames to exit early |
| `_DWELL_EXTEND_MAX_FACTOR` | 2.5× | Max dwell extension for unresolved persons |
| `_FACE_HUNT_BUDGET_S` | 22.0 s | Max time in FACE_HUNT per face |
| `_PRE_ZOOM_BUDGET_S` | 12.0 s | Budget for pre-zoom move + settle |
| `_HUNT_EMBED_AVG_N` | 3 | Embeddings averaged in FACE_HUNT |
| `_MAX_PRESET_SKIP_STREAK` | 1 | Max consecutive occupancy-aware skips |
| P-controller Kp | 0.25 | Face centering gain |

### Four Coverage Optimisations

| # | Name | How |
|---|------|-----|
| 1 | **Adaptive dwell** | Exit `PRESET_RECOGNIZE` early once every visible person is resolved for 3 frames. Saves up to `dwell_s × 1.5` per preset. |
| 2 | **Priority ordering** | At cycle end, reorder presets by historical face count descending — busy areas visited first. |
| 3 | **Occupancy-aware skip** | If a preset was empty last cycle, skip it for up to 1 cycle. Saves travel time on sparse rooms. |
| 4 | **Cross-preset re-ID** | ArcFace embeddings cached per person per cycle. On camera movement, cosine ≥ 0.62 re-identifies without FAISS. |

### FOV Calibration

`onvif_controller.py` `auto_calibrate_fov()` determines the physical relationship between ONVIF PTZ units and pixels:

```
K_pan = |d_pan| × (frame_width / 2) / |dx_pixels|
```

Where `d_pan` = ONVIF pan delta commanded, `dx_pixels` = resulting pixel shift measured by optical flow.

Stale frames (from the RTSP buffer before the move settled) are discarded with a **sign check**: if `d_pan × dx > 0` the scene shifted the wrong direction — frame rejected. Valid measurements are averaged across 3 test moves and stored in `cameras.learned_params`.

---

## 6. Enrollment Pipeline

```
POST /api/enrollment/upload  (multipart, field: "file")
        │
        ▼
  Stage 1 — Person detection (YOLO)
  Stage 2 — Face detection (SCRFD)
        │
        ▼
  _gate_image() quality gate:
  ┌─────────────────────────────────────┐
  │ IOD          ≥ 20 px                │
  │ Sharpness    ≥ 10.0 (Laplacian)     │
  │ Confidence   ≥ 0.50                 │
  │ Yaw          ≤ ±45°                 │
  └──────────────┬──────────────────────┘
                 │ pass
                 ▼
  align_face() → 112×112 chip
  (+ SR upsampling when IOD < 60 px)
                 │
                 ▼
  ArcFace embedding (512-D)
  MiniFASNet liveness check
                 │
                 ▼
  Template management (face_repository.py):
  • Max 5 templates stored per person
  • 3 augmented variants per image (compressed, darkened, brightened)
  • Feature whitening activated when ≥ 10 samples exist (ε = 1e-5)
  • EMA update with drift_limit=0.15 (prevents gradual embedding drift)
  • Template update requires liveness ≥ 0.85
                 │
                 ▼
  pgvector INSERT face_embeddings
  MinIO upload original image
  FAISS index rebuild for dataset
  Kafka publish → enrollment.events
```

**Bulk import** (`POST /api/enrollment/bulk-import`): ZIP archive of images. Each image runs through the same gate independently; failures are reported per-file without blocking the rest.

**Re-enroll** (`PUT /api/enrollment/{id}/re-enroll`): replaces all existing templates with new images. Old embeddings are deactivated, new ones built from scratch.

**Delete** (`DELETE /api/enrollment/{id}`): sets `face_embeddings.is_active = false` and `persons.status = INACTIVE`. Person disappears from datasets and FAISS immediately.

---

## 7. Face Search Pipeline

User uploads a face photo and gets back ranked identity matches from the enrolled database.

```
POST /api/search/face  (multipart, field: "file")
        │
        ▼
  Decode JPEG/PNG → BGR numpy
        │
        ▼
  SCRFD face detection
  Validation: exactly 1 face required
        │
        ▼
  Search quality gate (looser than enrollment):
  ┌──────────────────────────────────────┐
  │ Confidence   ≥ 0.50                  │
  │ IOD          ≥ 30 px                 │
  │ Sharpness    ≥ 30.0 (Laplacian)      │
  └──────────────┬───────────────────────┘
                 │ pass
                 ▼
  align_face() with iod_px for SR if needed
  ArcFace 512-D embedding
        │
        ▼
  face_repository.search_institution()
  → FAISS Tier-1 (all client datasets)
  → pgvector Tier-2 fallback
  → top-10 hits ranked by similarity
        │
        ▼
  Enrich with person metadata + MinIO thumbnail URLs
        │
        ▼
  { total, items: [
      { person_id, name, role, department,
        similarity, tier, thumbnail_url, last_seen }
  ]}
```

**Text search** (`GET /api/search/person?q=`): pg_trgm similarity on `name`, `department`, `external_id` — minimum score 0.10, top-10 results.

**Journey** (`GET /api/search/{id}/journey`): merges `attendance_records` + `sightings` into a chronological timeline with transit times and a heatmap (area × hour-of-day → seconds present).

**Area query** (`GET /api/search/area`): all persons whose sightings overlap a camera during a time window.

---

## 8. Attendance Engine

`attendance_engine.py` converts dwell-time sightings into formal attendance records with PRESENT / LATE / ABSENT status.

```
sighting  (person_id, camera_id, first_seen, last_seen, duration_s)
        │
        ▼
  Session lookup → find active PTZ session for this camera
  Roster check   → is person on session's expected roster?
        │ yes
        ▼
  Status:
  • first_seen ≤ scheduled_start + grace_s   →  PRESENT
  • first_seen ≤ session end                 →  LATE
  • session ended, person never sighted      →  ABSENT  (batch)
  • admin Kafka override                     →  EE (excused early)
        │
        ▼
  INSERT attendance_records  (deduplicated by session + person)
  Kafka → attendance.records
  Kafka → attendance.faculty / attendance.held  (role-specific)
```

**Admin overrides**: `AdminOverrideConsumer` subscribes to the `admin.overrides` Kafka topic. Admins can retroactively change any attendance record status via the dashboard.

---

## 9. Multi-Tenancy & Security

| Layer | Implementation |
|-------|---------------|
| **Row-Level Security** | PostgreSQL RLS on all tenant tables. Every request sets `app.current_client_id` via `SET LOCAL` before any query. |
| **JWT** | RS256 (production) or HS256 (dev). Claims include `client_id`, `role`, `permissions[]`. |
| **Roles** | `SUPER_ADMIN` (platform-wide), `CLIENT_ADMIN` (tenant), `VIEWER` (read-only). |
| **Permissions** | Fine-grained per route: `face_embeddings:read/create/delete`, `persons:read`, etc. |
| **Rate limiting** | Face search: 20 req/min per user (Redis INCR counter). |
| **Audit trail** | Every mutation published to `audit.events` Kafka topic. |

---

## 10. Infrastructure & Ports

| Service | Host Port | Notes |
|---------|-----------|-------|
| Backend API | 18000 | FastAPI + uvicorn |
| Dashboard | 3020 | Next.js 14 standalone |
| PostgreSQL (pgvector) | 15432 | Main app DB |
| TimescaleDB | 5433 | `detection_log` hypertable |
| Redis | 16379 | Session state, PTZ cache |
| MinIO API | 19000 | Face image storage |
| MinIO Console | 19001 | Web UI |
| Kafka | 9092 | Events (KRaft, no ZooKeeper) |
| Schema Registry | 18081 | Avro schemas |

### Kafka Topics

| Topic | Producer | Consumer |
|-------|----------|---------|
| `attendance.records` | attendance_engine | Dashboard, ERP |
| `repo.sync.embeddings` | face_sync | All nodes — FAISS rebuild |
| `admin.overrides` | Dashboard | AdminOverrideConsumer |
| `enrollment.events` | enrollment API | face_sync |
| `sightings.log` | sighting_engine | Analytics |
| `audit.events` | All mutations | Audit log |
| `system.alerts` | node_manager, gpu_manager | Ops |

### Docker Compose Files

| File | Use |
|------|-----|
| `docker-compose.yml` | Local dev — `network_mode: host` for LAN camera access |
| `docker-compose.prod.yml` | Production — bridge network, gunicorn, Cloudflare Tunnel |
| `docker-compose.test.yml` | CI — postgres + redis + backend, CPU FAISS |

---

## 11. Key Environment Variables

```bash
# Databases
DATABASE_URL=postgresql+asyncpg://acas:acas@localhost:15432/acas
TIMESCALE_URL=postgresql+asyncpg://acas:acas@localhost:5433/acas_ts

# Cache / Queue / Storage
REDIS_URL=redis://localhost:16379/0
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
MINIO_ENDPOINT=localhost:19000
MINIO_ROOT_USER=acasminio
MINIO_ROOT_PASSWORD=acasminiochange

# Auth
JWT_SECRET_KEY=          # RS256 PEM private key  or  HS256 secret string
JWT_ALGORITHM=RS256       # RS256 (production) or HS256 (dev/test)

# GPU / AI
GPU_DEVICE_ID=0           # CUDA device index
GPU_MAX_CONCURRENT=3      # Parallel inference slots (3–5 typical)
GPU_VRAM_BUDGET_GB=20     # VRAM limit — liveness dropped first when exceeded
MODEL_DIR=/models         # Directory containing .onnx files

# Node identity
NODE_ID=                  # UUID — auto-generated and persisted on first boot
NODE_NAME=acas-node-1
NODE_LOCATION=Building A
NODE_API_ENDPOINT=http://localhost:18000

# Control Plane (optional — omit for standalone single-node deployment)
CONTROL_PLANE_URL=        # Cloudflare Worker URL
NODE_AUTH_TOKEN=          # Bearer token for Control Plane auth
```

---

## 12. API Reference

### Sessions (camera AI)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/node/cameras/{id}/start` | Start AI monitoring session |
| `POST` | `/api/node/cameras/{id}/stop` | Stop session |
| `GET` | `/api/sessions/active` | List running sessions + state |
| `GET` | `/api/cameras/{id}/stream` | Live MJPEG stream |
| `GET` | `/api/cameras/{id}/annotated-stream` | MJPEG with AI overlays |
| `POST` | `/api/cameras/{id}/calibrate-fov` | Auto-calibrate FOV constants |

### Enrollment

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/enrollment/upload` | Upload + quality-gate a face image |
| `POST` | `/api/enrollment/enroll` | Enroll person into FAISS index |
| `POST` | `/api/enrollment/bulk-import` | ZIP of images, bulk enroll |
| `PUT` | `/api/enrollment/{id}/re-enroll` | Replace all face templates |
| `DELETE` | `/api/enrollment/{id}` | Soft-delete (INACTIVE + deactivate embeddings) |

### Search

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/search/face` | Image upload → top-10 identity matches |
| `POST` | `/api/search/face/base64` | Base64 image → matches |
| `GET` | `/api/search/person?q=` | Text search (pg_trgm) |
| `GET` | `/api/search/{id}/journey` | Chronological location timeline |
| `GET` | `/api/search/{id}/cross-camera` | Camera transition trail |
| `GET` | `/api/search/{id}/heatmap` | Presence heatmap (area × hour) |
| `GET` | `/api/search/area` | Who was at a camera during a time window |

### Attendance

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/attendance` | Records (filterable by date, person, status) |
| `GET` | `/api/attendance/report` | Aggregated report |
| `PUT` | `/api/attendance/{id}/override` | Manual status override |

### Cameras & PTZ

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/cameras` | List cameras |
| `POST` | `/api/cameras` | Add camera |
| `GET` | `/api/cameras/{id}/ptz-status` | Live pan / tilt / zoom |
| `GET` | `/api/cameras/{id}/presets` | List scan presets |
| `POST` | `/api/cameras/{id}/presets` | Add scan preset |

---

## 13. Testing

```bash
cd backend

# Install test dependencies (no GPU required)
pip install -r requirements-test.txt faiss-cpu

# Run all non-GPU tests
export TEST_DATABASE_URL="postgresql+asyncpg://acas:acas@localhost:15432/acas_test"
pytest tests/ -m "not gpu and not kafka and not minio" -v --timeout=60

# Run a single test file
pytest tests/test_attendance_engine.py -v

# Run full stack tests via Docker
docker compose -f docker-compose.test.yml up -d --build

# Test markers
# gpu    — requires CUDA GPU
# kafka  — requires running Kafka
# minio  — requires MinIO
```

Coverage threshold: **60%**.

---

## 14. Operations

See `RUNBOOK.md` for full recovery procedures. Quick reference:

```bash
# DB backup and restore
bash scripts/backup.sh
bash scripts/restore.sh <backup.tar.gz>

# Run migrations
docker exec -w /app acas-backend alembic upgrade head        # postgres
docker exec -w /app acas-backend alembic_ts upgrade head     # timescaledb

# Model swap (e.g. new ArcFace checkpoint)
docker cp new_model.onnx acas-backend:/models/adaface_ir101_webface12m.onnx
docker compose restart backend    # TRT engine rebuilt on first inference

# Enable neural super-resolution for small faces
docker cp realesrgan_x2.onnx acas-backend:/models/
docker compose restart backend

# Prometheus metrics
curl http://localhost:18000/api/metrics

# Start with Grafana + Prometheus
docker compose -f docker-compose.prod.yml --profile monitoring up -d

# Create Kafka topics (first-time setup)
for topic in admin.overrides repo.sync.embeddings attendance.records \
  attendance.held attendance.faculty sightings.log sightings.alerts \
  sightings.occupancy erp.sync.requests erp.sync.status \
  enrollment.events notifications.outbound system.alerts audit.events; do
  docker exec acas-kafka /opt/kafka/bin/kafka-topics.sh \
    --bootstrap-server kafka:19092 --create --if-not-exists \
    --topic "$topic" --partitions 2 --replication-factor 1
done
```

---

## AI Models

| Model | File | Architecture | Task | Input size |
|-------|------|-------------|------|-----------|
| YOLOv8l | `yolov8l.onnx` | YOLOv8x | Person detection + 17-pt pose | 640 × 640 |
| SCRFD-10G | `buffalo_l/det_10g.onnx` | SCRFD | Face detection + 5-pt landmarks | 640 × 640 |
| AdaFace IR-101 | `adaface_ir101_webface12m.onnx` | IR-101 ResNet | 512-D face embedding | 112 × 112 |
| MiniFASNet v2 | `minifasnet_v2.onnx` | MobileNet-based | Liveness anti-spoofing | 80 × 80 |
| Real-ESRGAN ×2 | `realesrgan_x2.onnx` *(optional)* | ESRGAN | Super-resolution for small faces | variable |

All models convert to **TensorRT FP16** on first inference (compiled `.engine` files cached in `/models`).

```bash
docker exec acas-backend python /tmp/download_models.py --model-dir /models
```

---

## See Also

- `README-LOCAL-DEV.md` — Detailed local development setup, hot reload, camera simulation
- `README-CLOUDFLARE-NODE.md` — Multi-node setup with Cloudflare Workers control plane
- `RUNBOOK.md` — Camera recovery, GPU OOM, Kafka restart, DB backup/restore, accuracy debugging
- `CLAUDE.md` — Guidance for AI coding assistants working in this repository
