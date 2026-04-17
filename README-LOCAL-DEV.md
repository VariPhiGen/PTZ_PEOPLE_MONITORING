# VGI ACFR — Local Development & Testing Guide

Deploy the complete VGI ACFR stack — backend, all infrastructure services, GPU AI pipeline, and dashboard — on a single local machine for development and testing.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Clone and Configure](#2-clone-and-configure)
3. [Generate JWT Keys](#3-generate-jwt-keys)
4. [Start Infrastructure Services](#4-start-infrastructure-services)
5. [Create Kafka Topics](#5-create-kafka-topics)
6. [Run Database Migrations](#6-run-database-migrations)
7. [Create the Super-Admin Account](#7-create-the-super-admin-account)
8. [Download AI Models](#8-download-ai-models)
9. [Start the Dashboard](#9-start-the-dashboard)
10. [First-Time Setup in the UI](#10-first-time-setup-in-the-ui)
11. [Register the Local GPU Node](#11-register-the-local-gpu-node)
12. [Verify End-to-End](#12-verify-end-to-end)
13. [Running the Test Suite](#13-running-the-test-suite)
14. [Service URLs Quick Reference](#14-service-urls-quick-reference)
15. [Troubleshooting](#15-troubleshooting)

---

## 1. Prerequisites

### Hardware
- NVIDIA GPU with at least 8 GB VRAM (RTX A5000 / RTX 4090 recommended; all 4 AI models together use ~2.5 GB VRAM)
- 32 GB RAM, 100 GB free disk space

### Software

| Tool | Version | Install |
|---|---|---|
| Ubuntu | 22.04 LTS | — |
| NVIDIA Driver | ≥ 560 | `ubuntu-drivers autoinstall` |
| CUDA Toolkit | 12.4+ | [developer.nvidia.com](https://developer.nvidia.com/cuda-downloads) |
| Docker Engine | ≥ 27 | [docs.docker.com](https://docs.docker.com/engine/install/ubuntu/) |
| NVIDIA Container Toolkit | latest | see below |
| Docker Compose plugin | ≥ 2.27 | included with Docker Engine |
| Node.js | 20 LTS | `nvm install 20` |
| Python | 3.11 | `apt install python3.11` |

#### Install NVIDIA Container Toolkit

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify GPU is accessible inside Docker:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

---

## 2. Clone and Configure

```bash
git clone <your-acas-repo-url> acas
cd acas
```

Copy the example environment file and edit it:

```bash
cp .env.example .env
nano .env
```

Key settings to change right away:

```bash
# Strong passwords — change all of these
POSTGRES_PASSWORD=choose-a-strong-password
TIMESCALE_PASSWORD=choose-a-strong-password
MINIO_ROOT_PASSWORD=choose-a-strong-password

# Give this node a friendly name
NODE_NAME=local-dev-node
NODE_API_ENDPOINT=http://localhost:18000
```

Leave ports at their defaults unless something on your machine is already using them. The default mapping is:

| Service | Host Port |
|---|---|
| Backend API | `18000` |
| PostgreSQL | `15432` |
| TimescaleDB | `5433` |
| Redis | `16379` |
| MinIO API | `19000` |
| MinIO Console | `19001` |
| Kafka | `9092` |
| Schema Registry | `18081` |
| Dashboard | `3020` |

> Run `ss -tlnp | awk '{print $4}' | grep -oP ':\K[0-9]+' | sort -n` to see all occupied ports before choosing values.

---

## 3. Generate JWT Keys

VGI ACFR uses RS256 (asymmetric) by default for production-grade security. For local dev you can use HS256 (simpler):

### Option A — HS256 (quick local dev)

```bash
# In .env set:
JWT_SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
JWT_ALGORITHM=HS256
```

### Option B — RS256 (matches production)

```bash
openssl genrsa -out jwt_private.pem 4096
openssl rsa -in jwt_private.pem -pubout -out jwt_public.pem

# Collapse private key to a single line for .env
JWT_PRIVATE=$(awk 'NF {printf "%s\\n", $0}' jwt_private.pem)
echo "JWT_SECRET_KEY=$JWT_PRIVATE" >> .env
# Also set:
JWT_ALGORITHM=RS256
```

---

## 4. Start Infrastructure Services

Build the backend image and bring up all services:

```bash
# Build the GPU backend image (takes ~5 min on first run)
docker compose build backend

# Start everything
docker compose up -d

# Watch startup progress
docker compose logs -f --tail=50
```

Wait until all health checks are green:

```bash
docker compose ps
```

Expected output (all should show `healthy` or `running`):

```
NAME                   STATUS         PORTS
acas-backend           running        (host networking — port 18000)
acas-postgres          healthy        0.0.0.0:15432->5432/tcp
acas-timescaledb       healthy        0.0.0.0:5433->5432/tcp
acas-redis             running        0.0.0.0:16379->6379/tcp
acas-minio             running        0.0.0.0:19000->9000/tcp
acas-kafka             healthy        0.0.0.0:9092->9092/tcp
acas-schema-registry   running        0.0.0.0:18081->8081/tcp
```

> The backend uses `network_mode: host` so it can reach LAN IP cameras directly. Its port is bound to the host interface, not forwarded via Docker NAT.

Confirm the backend is responding:

```bash
curl http://localhost:18000/api/health
```

Expected (before migrations, or without a Control Plane configured):

```json
{"status":"degraded","node_id":"...","checks":{"postgres":"ok","redis":"ok","minio":"ok","kafka":"ok","cp_registered":"no"}}
```

`status: degraded` with `cp_registered: no` is **normal** for a standalone local node — it means no Cloudflare Control Plane is configured. All core checks (`postgres`, `redis`, `minio`, `kafka`) should be `ok`.

---

## 5. Create Kafka Topics

Kafka topics are not auto-created. Run this once after the broker is healthy:

```bash
for topic in admin.overrides repo.sync.embeddings \
             attendance.records attendance.held attendance.faculty \
             sightings.log sightings.alerts sightings.occupancy \
             erp.sync.requests erp.sync.status \
             enrollment.events notifications.outbound \
             system.alerts audit.events; do
  docker exec acas-kafka /opt/kafka/bin/kafka-topics.sh \
    --bootstrap-server kafka:19092 --create --if-not-exists \
    --topic "$topic" --partitions 2 --replication-factor 1 2>&1 | grep -v WARNING
done
echo "Topics ready."
```

Topics persist in the `acas_kafka_data` volume across restarts — you only need to run this once.

---

## 6. Run Database Migrations

The Alembic migration files live in `backend/alembic/`. Copy them into the running container, then run:

```bash
docker cp backend/alembic      acas-backend:/app/alembic
docker cp backend/alembic.ini  acas-backend:/app/alembic.ini
docker exec -w /app acas-backend alembic upgrade head
```

This creates all tables, installs `pgvector` and `pg_trgm` extensions, enables Row-Level Security, builds the HNSW index, and creates the `detection_log` hypertable in TimescaleDB.

Verify the schema:

```bash
docker exec acas-postgres psql -U acas -d acas -c "\dt" | head -20
```

> **Tip:** Add these lines to the `backend` service volumes in `docker-compose.yml` to avoid re-copying on every recreate:
> ```yaml
> - ./backend/alembic:/app/alembic:ro
> - ./backend/alembic.ini:/app/alembic.ini:ro
> ```

---

## 7. Create the Super-Admin Account

```bash
docker cp backend/scripts/create_superadmin.py acas-backend:/app/create_superadmin.py

docker exec -w /app acas-backend python create_superadmin.py \
  --email admin@acas.local \
  --password "Admin@2024!" \
  --name "Platform Admin"
```

Confirm login works:

```bash
curl -s -X POST http://localhost:18000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@acas.local","password":"Admin@2024!"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin).get('token_type','FAILED'))"
# Expected: bearer
```

> If login returns "Invalid credentials" after a re-deploy, reset the hash directly:
> ```bash
> HASH=$(docker exec acas-backend python3 -c \
>   "from app.utils.security import hash_password; print(hash_password('Admin@2024!'))")
> docker exec acas-postgres psql -U acas -d acas \
>   -c "UPDATE users SET password_hash='${HASH}' WHERE email='admin@acas.local';"
> ```

---

## 8. Download AI Models

The backend AI pipeline requires four ONNX models (~440 MB total on disk). Run the download script **inside the running container** (it uses the same Python environment and writes to the mounted `/models` volume):

```bash
# Copy the download script into the container
docker cp scripts/download_models.py acas-backend:/tmp/download_models.py

# Run the download — this takes 1–3 minutes depending on internet speed
docker exec acas-backend python /tmp/download_models.py --model-dir /models
```

### What gets downloaded

| Model | File | Size | Purpose |
|---|---|---|---|
| YOLOv8l | `yolov8l.onnx` | 266 MB | Person detection + pose keypoints |
| SCRFD-10GF | `buffalo_l/det_10g.onnx` | 17 MB | Face detection |
| ArcFace W600K | `adaface_ir101_webface12m.onnx` | 167 MB | Face embedding (512-D) |
| MiniFASNetV2 | `minifasnet_v2.onnx` | 1.9 MB | Liveness anti-spoofing |

> **AdaFace note:** The original AdaFace IR-101 requires HuggingFace authentication. The download script automatically falls back to InsightFace's `w600k_r50.onnx` (ArcFace trained on WebFace600K), which is a production-quality 512-D face embedding model with equivalent accuracy for this use case.

> **MiniFASNet note:** The official `.pth` release URL has moved. The script downloads from the GitHub repository's raw path and exports to ONNX automatically.

### Verify models loaded on GPU

After the backend restarts (Uvicorn `--reload` picks up the new files), confirm all four models are running on the GPU:

```bash
docker exec acas-backend python3 -c "
import sys; sys.path.insert(0, '/app')
from app.services.ai_pipeline import AIPipeline
p = AIPipeline('/models', device_id=0)
p.load()
" 2>&1 | grep -E "GPU Status|CUDA GPU|MISSING"
```

Expected output:

```
  ╔══ AI Pipeline GPU Status (device 0) ══════════════════╗
  ║  YOLO person+pose    : CUDA GPU ✓                      ║
  ║  SCRFD face detector : CUDA GPU ✓                      ║
  ║  ArcFace embedder    : CUDA GPU ✓                      ║
  ║  MiniFASNet liveness : CUDA GPU ✓                      ║
  ╚════════════════════════════════════════════════════════╝
```

All models run on **CUDAExecutionProvider**. TensorRT EP is not required — TRT would give an additional ~1.5× speedup if installed, but CUDA EP delivers full GPU acceleration.

### GPU memory after model load

Expected VRAM usage with all 4 models loaded:

```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
# Example: 2500 MiB, 24564 MiB  (RTX A5000)
```

> If the models directory is empty or absent when the backend starts, the backend runs in **AI-disabled mode** — all non-inference API endpoints (cameras, clients, users, attendance records) still work normally, but quality checks and face recognition are skipped.

---

## 9. Start the Dashboard

The dashboard is a Next.js 14 app. On machines running many Docker containers, `npm run dev` will crash with an `ENOSPC` inotify error. Use the production build — it has no watchers:

```bash
cd dashboard

# Install dependencies (first time only)
npm install

# Create local environment file
cat > .env.local << 'EOF'
NEXT_PUBLIC_API_URL=http://localhost:18000
EOF

# Build the production bundle
NEXT_PUBLIC_API_URL=http://localhost:18000 npm run build

# Start on a free port
PORT=3020 npm start
```

The dashboard is now available at **http://localhost:3020**.

> To keep it running after terminal close: `PORT=3020 nohup npm start > /tmp/dashboard.log 2>&1 &`

---

## 10. First-Time Setup in the UI

Open **http://localhost:3020** and log in with the super-admin credentials from step 7.

### Create a Client (Tenant)

1. Go to **Admin → Clients → + Create Client**.
2. Fill in company name, slug, and contact details.
3. Set limits: cameras (50), persons (10,000) for local dev.
4. On the "GPU Nodes" step — skip for now, assign after the node registers.
5. Create an **Initial Admin** user for the client.
6. Click **Create**.

Once a client is created, you can edit its limits at any time: **Clients → [client] → Overview → Edit Limits**.

> **Client status meanings:**
> - **Active** — normal operation, all logins allowed
> - **Suspended** — temporarily disabled; can be re-activated
> - **Archive (Deactivate)** — permanently closes the account; data is kept but all logins are blocked. A confirmation dialog appears before archiving.

### Verify MinIO Buckets

The backend auto-creates two buckets on startup. Confirm via the MinIO Console at **http://localhost:19001** (credentials from your `.env`):
- `face-enrollment`
- `face-evidence`

---

## 11. Register the Local GPU Node

The backend itself is the GPU node. It auto-generates a `NODE_ID` on first start and writes it to Redis. To make the ID persistent across restarts:

```bash
# Get the current node_id from the health endpoint
NODE_ID=$(curl -s http://localhost:18000/api/health | python3 -c "import sys,json; print(json.load(sys.stdin)['node_id'])")
echo "NODE_ID=$NODE_ID" >> .env
docker compose restart backend
```

### Assign the Node to a Client

In the dashboard: **Admin → Clients → [your client] → Nodes tab → Add Node**.

Or via API:

```bash
TOKEN=$(curl -s -X POST http://localhost:18000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@acas.local","password":"Admin@2024!"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

NODE_ID=$(curl -s http://localhost:18000/api/health | python3 -c "import sys,json; print(json.load(sys.stdin)['node_id'])")

CLIENT_ID=$(curl -s http://localhost:18000/api/admin/clients \
  -H "Authorization: Bearer $TOKEN" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['items'][0]['client_id'])")

curl -X POST http://localhost:18000/api/admin/clients/${CLIENT_ID}/nodes \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"node_id\":\"$NODE_ID\",\"max_cameras_on_node\":10}"
```

### Add a Test Camera

Log in as the **Client Admin** created in step 10:

1. **Cameras → + Add Camera**.
2. Fill in:
   - **RTSP URL** — e.g. `rtsp://172.16.16.206:554/video/live?channel=1`
   - **ONVIF Host / Port / Credentials** — if the camera supports ONVIF PTZ
   - **Mode** — `ATTENDANCE`, `MONITORING`, or `BOTH`
   - **Node** — select the node assigned to this client
3. Click **Test Connection** — this verifies RTSP reachability and ONVIF access.
4. Save.

### Draw a Zone (Required Before Starting AI)

Before the AI session can start, you must define the monitoring zone on the camera:

1. Open the camera detail → **Zone & PTZ** tab.
2. The live MJPEG stream appears as the background.
3. **Click** on the canvas to place polygon vertices defining the area of interest.
4. **Click near the first point** (green snap indicator appears) or **double-click** anywhere to close the polygon.
5. Press **Esc** to cancel and start over.
6. Click **Save Zone**.

> The "Start AI" button is disabled until a zone is saved. The zone is stored as a polygon (`[{x, y}, ...]` in normalized 0–1 coordinates) in the camera record.

### PTZ Manual Control

In the **Zone & PTZ** tab, use the directional pad and zoom slider while watching the live stream in real time. To prevent the AI from interfering during manual adjustment:

1. Click **Stop AI** — pauses the PTZ brain for this camera.
2. Adjust pan/tilt/zoom manually.
3. Redraw the zone if needed.
4. Click **Restart AI** — resumes autonomous scanning.

### Enroll Faces

Log in as the **Client Admin** and navigate to **Enrollment → + New Enrollment**:

1. Fill in name, external ID, role, department (inline person creation — no need to create a person record separately).
2. Drag-and-drop face photos (JPEG/PNG, min 200×200 px).
3. Click **Analyze** — the backend runs the AI quality gate:
   - Face detected? (must have exactly 1 face)
   - Inter-ocular distance ≥ 40 px
   - Sharpness score ≥ 30
   - Liveness score passes
4. Images that pass show a green badge; failed images show the failure reason.
5. Once **at least 1 image passes**, the **Enroll** button enables.
6. Click **Enroll** — ArcFace embeddings are computed and stored in pgvector + FAISS.

> For best recognition accuracy, provide 3–8 images with varied angles (±15° yaw, slight tilt), different lighting, and no obstructions.

> If AI models are not yet loaded, quality check is gracefully skipped (all images pass with a `pipeline_not_loaded_quality_check_skipped` note) so enrollment can proceed without blocking the UI workflow.

---

## 12. Verify End-to-End

### Check AI Pipeline Status

```bash
curl -s http://localhost:18000/api/health | python3 -m json.tool
```

### Check GPU Memory

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
```

With all 4 models loaded you should see ~2.5 GB used (or more if other processes share the GPU).

### Test Quality Check

Upload a face photo and confirm the AI pipeline responds with real scores (not the skip note):

```bash
TOKEN=$(curl -s -X POST http://localhost:18000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@acas.local","password":"Admin@2024!"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

curl -s -X POST http://localhost:18000/api/enrollment/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "images=@/path/to/face.jpg" \
  | python3 -m json.tool
```

A real face photo should return `"passed": true` with `iod_px`, `sharpness`, and `confidence` populated.

### Start a Test Session

1. Go to **Cameras**, open your camera (ensure a zone is saved).
2. Click **Start AI** — the PTZ brain task launches.
3. Navigate to **Live Monitor** to see the scan map, recognition events, and cycle stats in real time via WebSocket.

---

## 13. Running the Test Suite

The test suite requires only PostgreSQL + Redis — no GPU needed for logic tests.

```bash
# Start only the services tests need
docker compose up -d postgres redis

# Install test dependencies
cd backend
pip3 install -r requirements-test.txt faiss-cpu

# Set test environment variables
export TEST_DATABASE_URL="postgresql+asyncpg://acas:acas@localhost:15432/acas_test"

# Create the test database
docker exec acas-postgres createdb -U acas acas_test

# Run tests (skips GPU/Kafka/MinIO-dependent tests automatically)
pytest tests/ -m "not gpu and not kafka and not minio" -v --timeout=60

# Run with coverage report
pytest tests/ -m "not gpu and not kafka and not minio" \
  --cov=app --cov-report=html --cov-report=term-missing
```

Open `backend/htmlcov/index.html` to browse coverage.

---

## 14. Service URLs Quick Reference

| Service | URL | Credentials |
|---|---|---|
| **Dashboard** | http://localhost:3020 | admin@acas.local / your password |
| **Backend API** | http://localhost:18000 | — |
| **API Docs (Swagger)** | http://localhost:18000/docs | — |
| **Prometheus Metrics** | http://localhost:18000/metrics | — |
| **MinIO Console** | http://localhost:19001 | from `.env` |
| PostgreSQL | localhost:15432 | acas / `POSTGRES_PASSWORD` |
| TimescaleDB | localhost:5433 | acas / `TIMESCALE_PASSWORD` |
| Redis | localhost:16379 | no auth |
| Kafka | localhost:9092 | — |
| Schema Registry | http://localhost:18081 | — |

---

## 15. Troubleshooting

### Backend fails to start — "NVIDIA runtime not found"

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
# If that fails:
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### AI pipeline shows `pipeline_not_loaded_quality_check_skipped`

This means the `ai_pipeline` state attribute is `None`. Check in order:

1. **Models downloaded?** `docker exec acas-backend ls /models/` — must show `.onnx` files.
2. **onnxruntime-gpu installed?** `docker exec acas-backend python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"` — must include `CUDAExecutionProvider`.
3. **LD_LIBRARY_PATH set?** The `docker-compose.yml` must include:
   ```yaml
   LD_LIBRARY_PATH: /usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cuda_runtime/lib:/usr/local/cuda/lib64
   ```
4. **Restart the backend** after downloading models: `docker compose restart backend`.

If `onnxruntime-gpu` got overwritten with the CPU-only version (can happen if you ran pip inside the container), reinstall it:

```bash
docker exec acas-backend pip uninstall -y onnxruntime onnxruntime-gpu
docker exec acas-backend pip install onnxruntime-gpu==1.20.1
docker compose restart backend
```

### All models running on CPUExecutionProvider

The cuDNN libraries are present but not on the dynamic linker path. Verify `LD_LIBRARY_PATH` is set in `docker-compose.yml` (see above). After updating `docker-compose.yml`, re-create the container:

```bash
docker compose up -d --force-recreate backend
```

### "Failed to save zone" when drawing polygon

The backend expects `roi_rect` as either a `dict` (legacy rectangle `{x, y, w, h}`) or a `list` (polygon `[{x, y}, ...]`). Both formats are accepted. If you see a 422 error, confirm the backend is running the latest code (`docker compose logs backend --tail=5`).

### Enroll button stays disabled after quality check passes

The Enroll button requires at least 1 image with `status === "pass"`. If images pass the quality check but the button remains grey, the `temp_key` is missing from the image item. This was a bug where the upload response returned `b64:<data>` keys but the frontend wasn't storing them. Confirm the dashboard is running the latest build: `npm run build && PORT=3020 npm start`.

### Start AI button is disabled

The **Start AI** button is intentionally disabled until the camera has a zone (`roi_rect` is not null). Draw and save a polygon zone first. The button tooltip shows "Set ROI zone first" when disabled.

### "CUDA out of memory" during inference

```bash
# In .env, add:
MAX_GPU_CONCURRENT=3   # default is 5
docker compose restart backend
```

### Kafka health check keeps failing

Kafka takes up to 45 seconds to start in KRaft mode. If it still fails after 2 minutes:

```bash
docker compose logs kafka --tail=30
# To reset (loses all messages in the volume):
docker compose down
docker volume rm acas_acas_kafka_data
docker compose up -d kafka
```

### Database migration error — "extension vector does not exist"

```bash
docker compose exec postgres psql -U acas -d acas \
  -c "CREATE EXTENSION IF NOT EXISTS vector; CREATE EXTENSION IF NOT EXISTS pg_trgm;"
docker compose exec backend alembic upgrade head
```

### Dashboard not loading / "Next.js 14 placeholder" screen

The dashboard needs to be built with a current version of the code. Any placeholder message means an old build is being served:

```bash
cd dashboard
NEXT_PUBLIC_API_URL=http://localhost:18000 npm run build
pkill -f "next-server" 2>/dev/null
PORT=3020 nohup npm start > /tmp/dashboard.log 2>&1 &
```

### No RTSP camera available for testing

Run a local RTSP server with FFmpeg and MediaMTX:

```bash
# Start RTSP server
docker run --rm -p 8554:8554 bluenviron/mediamtx:latest &

# Push a test video (loop)
ffmpeg -re -stream_loop -1 -i /path/to/video.mp4 \
  -c copy -f rtsp rtsp://localhost:8554/live

# Camera RTSP URL: rtsp://localhost:8554/live
```

### Wiping everything and starting fresh

```bash
docker compose down -v   # removes containers AND named volumes
docker rmi acas-backend:gpu
docker compose build backend
docker compose up -d
```

---

## URL Formation — Localhost / Cloudflare / Cloud

Every GPU feature (live stream, snapshot, PTZ, zone drawing, face data) routes through **the node's own `node_api_endpoint`**. This is stored in the `nodes` table and returned with every camera response. The dashboard picks it up automatically.

| Mode | `NODE_API_ENDPOINT` in `.env` | `NEXT_PUBLIC_API_URL` in `dashboard/.env.local` |
|---|---|---|
| **Localhost** | `http://localhost:18000` | `http://localhost:18000` |
| **Cloudflare Tunnel** | `https://node-1.acas.example.com` | `https://node-1.acas.example.com` |
| **Cloud VM (public IP)** | `http://203.0.113.5:18000` | `http://203.0.113.5:18000` |

The RTSP camera only needs to be reachable from the **GPU node machine** on the LAN — it never needs a public IP, regardless of how the dashboard is accessed.

### Feature URL reference

| Feature | URL pattern | Auth |
|---|---|---|
| Live MJPEG stream | `{node_api_endpoint}/api/cameras/{id}/stream?token=JWT` | JWT query param (`<img>` tag) |
| JPEG snapshot | `{node_api_endpoint}/api/cameras/{id}/snapshot` | Authorization header |
| PTZ move | `{node_api_endpoint}/api/cameras/{id}/ptz-move` | Authorization header |
| PTZ status | `{node_api_endpoint}/api/cameras/{id}/ptz-status` | Authorization header |
| Zone save (ROI) | `{node_api_endpoint}/api/cameras/{id}/roi` | Authorization header |
| AI start / stop | `{node_api_endpoint}/api/node/cameras/{id}/start\|stop` | Authorization header |
| AI mode pause | `{node_api_endpoint}/api/cameras/{id}/ai-mode` | Authorization header |
| Face enroll | `{node_api_endpoint}/api/enrollment/enroll` | Authorization header |
| Face search | `{node_api_endpoint}/api/search/face` | Authorization header |
