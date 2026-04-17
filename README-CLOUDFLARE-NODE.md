# VGI ACFR — Deployment Guide

Deploy VGI ACFR on a **local machine**, expose it via **Cloudflare Tunnel**, or host it on any **cloud provider** (AWS, GCP, Azure, DigitalOcean, etc.). All three paths share the same backend stack; only the network exposure and Control Plane differ.

---

## Choose Your Deployment Mode

| Mode | Best for | Control Plane | Dashboard |
|---|---|---|---|
| **A — Local** | Dev / testing on one machine | None (direct API) | `npm start` on localhost |
| **B — Cloudflare Tunnel** | Remote GPU behind NAT, no public IP | Cloudflare Worker | Cloudflare Pages |
| **C — Cloud (public IP)** | AWS/GCP/Azure VM, dedicated server | None or Cloudflare Worker | Cloudflare Pages or self-hosted |

> All modes use the same `docker-compose.yml` and backend image. Steps marked **[All]** apply regardless of mode. Steps marked **[B]** or **[C]** are mode-specific.

---

## Architecture

```
Mode A — Local
──────────────
  Browser → http://localhost:3020 (Dashboard)
          → http://localhost:18000 (Backend API, direct)

Mode B — Cloudflare Tunnel
──────────────────────────
  Browser → https://acas-dashboard.pages.dev (Cloudflare Pages)
          → Cloudflare Worker (Control Plane)
          → Cloudflare Tunnel (outbound from GPU node)
          → http://localhost:18000 (Backend, inside the node)

Mode C — Cloud VM
─────────────────
  Browser → https://dashboard.example.com (self-hosted or Cloudflare Pages)
          → https://api.example.com:18000 (Backend, public IP + firewall rule)
          OR
          → Cloudflare Worker → Cloudflare Tunnel → Backend (same as Mode B)
```

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Clone and Configure](#2-clone-and-configure)
3. [Create Kafka Topics](#3-create-kafka-topics)
4. [Run Database Migrations](#4-run-database-migrations)
5. [Create the Super-Admin Account](#5-create-the-super-admin-account)
6. [Deploy the Dashboard](#6-deploy-the-dashboard)
7. [Mode B — Cloudflare Tunnel Setup](#7-mode-b--cloudflare-tunnel-setup)
8. [Mode B — Deploy Control Plane (Cloudflare Worker)](#8-mode-b--deploy-control-plane-cloudflare-worker)
9. [Mode C — Cloud VM (Public IP)](#9-mode-c--cloud-vm-public-ip)
10. [Download AI Models](#10-download-ai-models)
11. [Add the Node to a Client](#11-add-the-node-to-a-client)
12. [Camera Setup — Zones & PTZ](#12-camera-setup--zones--ptz)
13. [Face Enrollment](#13-face-enrollment)
14. [Automated Bootstrap (Alternative)](#14-automated-bootstrap-alternative)
15. [Adding More Nodes](#15-adding-more-nodes)
16. [Updating Deployed Nodes](#16-updating-deployed-nodes)
17. [Service URLs Quick Reference](#17-service-urls-quick-reference)
18. [Troubleshooting](#18-troubleshooting)
19. [Security Checklist](#19-security-checklist)

---

## 1. Prerequisites

### Hardware

| Requirement | Minimum | Recommended |
|---|---|---|
| OS | Ubuntu 22.04 LTS | Ubuntu 22.04 LTS |
| GPU | NVIDIA 8 GB VRAM | RTX A5000 / RTX 4090 / A100 |
| NVIDIA Driver | ≥ 560 | latest stable |
| CUDA | 12.4+ | 12.4+ |
| RAM | 32 GB | 64 GB |
| Disk | 100 GB SSD | 500 GB NVMe |

> **Mode A only:** no public IP or open ports required.
> **Mode B:** only outbound internet needed — no inbound firewall ports.
> **Mode C:** outbound internet + one inbound port (18000 or 443 if behind nginx).

### Software

| Tool | Version | Install |
|---|---|---|
| Docker Engine | ≥ 27 | `curl -fsSL https://get.docker.com \| bash` |
| Docker Compose plugin | ≥ 2.27 | included with Docker Engine |
| NVIDIA Container Toolkit | latest | see below |
| Node.js | 20 LTS | `nvm install 20` |
| `wrangler` CLI | latest | `npm install -g wrangler` (Modes B/C with CF) |

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

# Verify
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

---

## 2. Clone and Configure

**[All modes]**

```bash
git clone <your-acas-repo-url> /opt/acas
cd /opt/acas
cp .env.example .env
nano .env
```

### Required .env settings

```bash
# ── Passwords — change every one of these ─────────────────────────────────────
POSTGRES_PASSWORD=<strong-random-password>
TIMESCALE_PASSWORD=<strong-random-password>
MINIO_ROOT_PASSWORD=<strong-random-password>
MINIO_ROOT_USER=acasminio

# ── JWT — HS256 is simplest for a single node ──────────────────────────────────
# Generate: python3 -c "import secrets; print(secrets.token_hex(32))"
JWT_SECRET_KEY=<64-char-hex-or-longer-random-string>
JWT_ALGORITHM=HS256

# ── Node identity ──────────────────────────────────────────────────────────────
# Generate once and keep stable: python3 -c "import uuid; print(uuid.uuid4())"
NODE_ID=<uuid>
NODE_NAME=acas-node-1
NODE_LOCATION=Local Machine

# ── Node API endpoint (set to whatever URL the dashboard browser will reach) ───
NODE_API_ENDPOINT=http://localhost:18000       # Mode A
# NODE_API_ENDPOINT=https://node-1.acas.example.com  # Mode B / C

# ── Mode B / C only — set after tunnel / IP is known ──────────────────────────
# CONTROL_PLANE_URL=https://acas-cp.<subdomain>.workers.dev
```

> Run `ss -tlnp | awk '{print $4}' | grep -oP ':\K[0-9]+' | sort -n` to see all occupied ports.

### Build the backend image

```bash
# Takes ~5 minutes on the first run; subsequent builds are cached.
docker compose build backend
```

### Start all infrastructure services

```bash
docker compose up -d

# Wait for services to be healthy (usually 60–90 s for Kafka)
watch docker compose ps
```

Expected output — all services should show `running` or `healthy`:

```
NAME                   STATUS
acas-backend           running   (network_mode: host — port 18000)
acas-postgres          healthy
acas-timescaledb       healthy
acas-redis             running
acas-minio             running
acas-kafka             healthy
acas-schema-registry   running
```

Check the backend:

```bash
curl http://localhost:18000/api/health
```

Expected response:

```json
{"status":"degraded","node_id":"...","checks":{"postgres":"ok","redis":"ok","minio":"ok","kafka":"ok","cp_registered":"no"}}
```

`cp_registered: no` is **expected in Mode A** — no Control Plane is configured. `status: ok` will appear after migrations and super-admin creation.

---

## 3. Create Kafka Topics

**[All modes]**

The backend consumers log warnings until the required topics exist. Create all 13 topics:

```bash
docker exec acas-kafka /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server kafka:19092 --create --if-not-exists \
  --topic admin.overrides --partitions 1 --replication-factor 1

docker exec acas-kafka /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server kafka:19092 --create --if-not-exists \
  --topic repo.sync.embeddings --partitions 3 --replication-factor 1

for topic in attendance.records attendance.held attendance.faculty \
             sightings.log sightings.alerts sightings.occupancy \
             erp.sync.requests erp.sync.status \
             enrollment.events notifications.outbound \
             system.alerts audit.events; do
  docker exec acas-kafka /opt/kafka/bin/kafka-topics.sh \
    --bootstrap-server kafka:19092 --create --if-not-exists \
    --topic "$topic" --partitions 2 --replication-factor 1
done

echo "All topics created."
docker exec acas-kafka /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server kafka:19092 --list
```

Topics persist in the `acas_kafka_data` volume — only needs to run once.

---

## 4. Run Database Migrations

**[All modes]**

```bash
# Copy migration files into the running container
docker cp backend/alembic      acas-backend:/app/alembic
docker cp backend/alembic.ini  acas-backend:/app/alembic.ini

# Apply all migrations
docker exec -w /app acas-backend alembic upgrade head
```

Expected output:

```
INFO  [alembic.runtime.migration] Running upgrade  -> 0001, initial schema ...
INFO  [alembic.runtime.migration] Running upgrade 0001 -> 0002, Performance tuning ...
```

Verify:

```bash
docker exec acas-postgres psql -U acas -d acas -c "\dt" | head -20
```

> To avoid re-copying after container recreation, add volume mounts to `docker-compose.yml`:
> ```yaml
> volumes:
>   - ./backend/alembic:/app/alembic:ro
>   - ./backend/alembic.ini:/app/alembic.ini:ro
> ```

---

## 5. Create the Super-Admin Account

**[All modes]**

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
  | python3 -m json.tool | grep '"role"'
# Should show: "role": "SUPER_ADMIN"
```

---

## 6. Deploy the Dashboard

**[All modes]** — the method differs by mode.

### Mode A — Local: production build (recommended)

Running `npm run dev` on a machine with many Docker containers exhausts the Linux `inotify` file-watcher limit. Use the production build — it serves static files without watchers:

```bash
cd /opt/acas/dashboard

# Install dependencies (first run only)
npm install

# Create the local environment file
cat > .env.local << 'EOF'
NEXT_PUBLIC_API_URL=http://localhost:18000
EOF

# Build the production bundle
NEXT_PUBLIC_API_URL=http://localhost:18000 npm run build

# Start on a free port and keep running
PORT=3020 nohup npm start > /tmp/dashboard.log 2>&1 &
```

Dashboard is at **http://localhost:3020**.

> Check the log if it doesn't load: `tail -20 /tmp/dashboard.log`

### Mode B — Cloudflare Pages

See [Section 8](#8-mode-b--deploy-control-plane-cloudflare-worker) — the dashboard is deployed to Cloudflare Pages alongside the Control Plane Worker.

### Mode C — Cloud VM (self-hosted Next.js)

Same as Mode A, but run on the cloud VM. Use PM2 for process management:

```bash
npm install -g pm2
cd /opt/acas/dashboard
NEXT_PUBLIC_API_URL=https://api.acas.example.com npm run build
pm2 start "PORT=3020 npm start" --name acas-dashboard
pm2 save && pm2 startup
```

---

## 7. Mode B — Cloudflare Tunnel Setup

**[Mode B only]** — skip for Mode A or C.

### 7a. Install cloudflared

```bash
curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg \
  | sudo gpg --dearmor -o /usr/share/keyrings/cloudflare-main.gpg

echo "deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] \
  https://pkg.cloudflare.com/cloudflared $(lsb_release -cs) main" \
  | sudo tee /etc/apt/sources.list.d/cloudflared.list

sudo apt-get update && sudo apt-get install -y cloudflared
cloudflared --version
```

### 7b. Authenticate and create a tunnel

```bash
cloudflared tunnel login
# Opens browser OAuth — select your Cloudflare-managed domain.

cloudflared tunnel create acas-node-1
# Note the Tunnel ID (UUID) in the output
```

### 7c. Write the tunnel config file

```bash
TUNNEL_ID=<paste-your-tunnel-id-here>

sudo mkdir -p /etc/cloudflared
sudo cp ~/.cloudflared/${TUNNEL_ID}.json /etc/cloudflared/acas-node-1.json
sudo chmod 600 /etc/cloudflared/acas-node-1.json

cat | sudo tee /etc/cloudflared/config.yml << EOF
tunnel: ${TUNNEL_ID}
credentials-file: /etc/cloudflared/acas-node-1.json

ingress:
  - hostname: node-1.acas.example.com   # ← your subdomain
    service: http://localhost:18000      # backend API port
  - service: http_status:404
EOF
```

### 7d. Create the DNS record and start as a service

```bash
cloudflared tunnel route dns acas-node-1 node-1.acas.example.com

sudo cloudflared service install
sudo systemctl enable cloudflared
sudo systemctl start  cloudflared
sudo systemctl status cloudflared
```

### 7e. Update .env with the tunnel URL

```bash
sed -i 's|^NODE_API_ENDPOINT=.*|NODE_API_ENDPOINT=https://node-1.acas.example.com|' /opt/acas/.env
docker compose restart backend
```

Verify the tunnel reaches the backend:

```bash
curl https://node-1.acas.example.com/api/health
```

> **Live streaming via Cloudflare Tunnel:** The MJPEG stream endpoint sends `X-Accel-Buffering: no` and `Cache-Control: no-cache` headers, and emits a keepalive boundary every 25 seconds to prevent Cloudflare's 100-second idle timeout from closing the connection.

---

## 8. Mode B — Deploy Control Plane (Cloudflare Worker)

**[Mode B only]** — skip for Mode A or self-hosted Mode C.

### 8a. One-time Cloudflare setup

```bash
wrangler login
wrangler whoami   # note your Account ID

# Create the D1 database
wrangler d1 create acas-db
# Output: database_id = "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"

# Create KV namespaces
wrangler kv:namespace create "ACAS_CONFIG"
wrangler kv:namespace create "ACAS_ROUTING"
```

Edit `control-plane/wrangler.toml` with the IDs returned above:

```toml
[[d1_databases]]
binding       = "DB"
database_name = "acas-db"
database_id   = "<your-d1-id>"

[[kv_namespaces]]
binding = "ACAS_CONFIG"
id      = "<your-kv-config-id>"

[[kv_namespaces]]
binding = "ACAS_ROUTING"
id      = "<your-kv-routing-id>"
```

### 8b. Set secrets

```bash
# Must match JWT_SECRET_KEY in every node's .env exactly
wrangler secret put JWT_SECRET

# Shared token nodes use for heartbeats
wrangler secret put NODE_AUTH_TOKEN
```

### 8c. Apply D1 migrations and deploy

```bash
cd control-plane
npm install
wrangler d1 execute acas-db --file=./schema.sql
wrangler deploy
# Output: https://acas-control-plane.<subdomain>.workers.dev
```

Update the node's `.env`:

```bash
sed -i 's|^CONTROL_PLANE_URL=.*|CONTROL_PLANE_URL=https://acas-control-plane.<subdomain>.workers.dev|' /opt/acas/.env
docker compose restart backend
```

### 8d. Deploy the dashboard to Cloudflare Pages

```bash
cd /opt/acas/dashboard
npm install

echo 'NEXT_PUBLIC_API_URL=https://acas-control-plane.<subdomain>.workers.dev' > .env.production

npm run pages:build
npm run pages:deploy
# Dashboard URL: https://acas-dashboard.pages.dev
```

To use a custom domain: Cloudflare Dashboard → **Pages → acas-dashboard → Custom domains → Add**.

---

## 9. Mode C — Cloud VM (Public IP)

**[Mode C only]** — skip for Modes A and B.

### 9a. Open firewall ports

Allow inbound TCP on your cloud provider's security group:

| Port | Service |
|---|---|
| 18000 | ACAS Backend API |
| 443 | (if using nginx/TLS reverse proxy) |

**Do not expose** 15432 (Postgres), 16379 (Redis), 19000 (MinIO), or 9092 (Kafka) publicly.

### 9b. Point the dashboard at the public API

```bash
cat > /opt/acas/dashboard/.env.local << EOF
NEXT_PUBLIC_API_URL=http://<your-vm-public-ip>:18000
# or with TLS:
# NEXT_PUBLIC_API_URL=https://api.acas.example.com
EOF
```

### 9c. (Optional) TLS with nginx + Certbot

```bash
sudo apt-get install -y nginx certbot python3-certbot-nginx

# /etc/nginx/sites-available/acas
server {
    server_name api.acas.example.com;
    location / {
        proxy_pass http://127.0.0.1:18000;
        proxy_set_header Host $host;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_buffering off;   # required for MJPEG live stream
    }
}

sudo certbot --nginx -d api.acas.example.com
sudo systemctl reload nginx
```

---

## 10. Download AI Models

**[All modes]** — required for live face recognition. Skip for UI-only testing.

Run the download script **inside the running backend container** — it has all required dependencies and writes directly to the mounted `/models` volume:

```bash
docker cp /opt/acas/scripts/download_models.py acas-backend:/tmp/download_models.py
docker exec acas-backend python /tmp/download_models.py --model-dir /models
```

### Models downloaded

| Model | File | Size | Purpose |
|---|---|---|---|
| YOLOv8l | `yolov8l.onnx` | 266 MB | Person detection + 17-point pose |
| SCRFD-10GF | `buffalo_l/det_10g.onnx` | 17 MB | Face detection |
| ArcFace W600K | `adaface_ir101_webface12m.onnx` | 167 MB | Face embedding (512-D) |
| MiniFASNetV2 | `minifasnet_v2.onnx` | 1.9 MB | Liveness anti-spoofing |

> **AdaFace note:** The original AdaFace IR-101 requires HuggingFace auth. The script falls back to InsightFace's `w600k_r50.onnx` (ArcFace/WebFace600K) — equivalent 512-D embeddings, production-quality accuracy.

Restart the backend after downloading:

```bash
docker compose restart backend
```

Verify all 4 models loaded on GPU:

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

All models run on **CUDAExecutionProvider**. TensorRT EP provides an additional ~1.5× speedup if installed, but is not required — CUDA EP delivers full GPU acceleration out of the box.

> If the `models/` directory is empty or absent, the backend starts in **AI-disabled mode** — all non-inference endpoints (cameras, clients, users, attendance records) still function normally.

### GPU memory requirement

| Configuration | VRAM used |
|---|---|
| All 4 models loaded | ~2.5 GB |
| 5 concurrent sessions | ~8 GB |
| 10 concurrent sessions | ~16 GB |
| Maximum (RTX A5000 24 GB) | 20 concurrent sessions |

---

## 11. Add the Node to a Client

**[All modes]**

### Step 1 — Log into the dashboard

- **Mode A:** http://localhost:3020
- **Mode B:** https://acas-dashboard.pages.dev
- **Mode C:** http://\<vm-ip\>:3020 or your custom domain

Credentials: `admin@acas.local` / `Admin@2024!`

### Step 2 — Create a Client (Tenant)

1. **Admin → Clients → + Create Client**.
2. Fill in company name, slug, and contact info.
3. Set limits (cameras: 50, persons: 10,000 for a standard node).
4. On the "GPU Nodes" step — you can skip or assign now.
5. Add an **Initial Admin** user for this client.
6. Click **Create**.

> You can edit limits after creation: **Clients → [client] → Overview → Edit Limits**.
>
> **Client status:**
> - **Suspended** — temporarily disables logins; re-activatable
> - **Archive (Deactivate)** — permanently closes the account (data preserved, all logins blocked). Requires confirmation.

### Step 3 — Assign the Node to the Client

Via the UI: **Admin → Clients → [your client] → Nodes → Add Node**.

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
  -d "{\"node_id\":\"${NODE_ID}\",\"max_cameras_on_node\":10}"
```

### Step 4 — Add a Camera

Log in as the **Client Admin** created above:

1. **Cameras → + Add Camera**.
2. Fill in RTSP URL, ONVIF credentials, mode, and select the node.
3. Click **Test Connection** — validates RTSP reachability and ONVIF access.
4. Click **Save**.

Navigate to **GPU Cluster** (super-admin sidebar) — the node should appear as `ONLINE` with live GPU/CPU/RAM sparklines.

---

## 12. Camera Setup — Zones & PTZ

### Draw a Monitoring Zone (Required Before AI Start)

Before the AI scanning session can start, a polygon zone must be defined:

1. Open the camera → **Zone & PTZ** tab.
2. The live MJPEG stream appears as the background layer.
3. **Click** on the canvas to place polygon vertices.
4. **Click near the first vertex** (green snap indicator) or **double-click** to close the polygon.
5. Press **Esc** to cancel.
6. Click **Save Zone**.

The **Start AI** button is disabled until a zone exists. The polygon is stored as `[{x, y}, ...]` in normalized 0–1 coordinates in the camera record.

### PTZ Manual Control with Live Stream

1. In the **Zone & PTZ** tab, click **Stop AI** — pauses the PTZ brain.
2. Use the directional pad and zoom slider; the live stream updates in real time.
3. Redraw the zone if needed.
4. Click **Restart AI** — resumes autonomous scanning.

---

## 13. Face Enrollment

Log in as a **Client Admin** → **Enrollment → + New Enrollment**:

1. Fill in person details (name, external ID, role, department) — a person record is auto-created inline.
2. Drag-and-drop face photos (JPEG/PNG, min 200×200 px).
3. Click **Analyze** — the AI quality gate checks:
   - Exactly 1 face detected
   - Inter-ocular distance ≥ 40 px
   - Sharpness score ≥ 30
   - Liveness score passes
4. Green badge = passing; red badge = failure reason shown.
5. **At least 1 passing image** enables the **Enroll** button.
6. Click **Enroll** — ArcFace embeddings are computed and stored in pgvector + FAISS.

> For best recognition accuracy, provide 3–8 images with varied angles (±15° yaw), different lighting, no obstructions.
>
> If AI models are not loaded, quality check is gracefully skipped — all images pass with a `pipeline_not_loaded_quality_check_skipped` note, allowing enrollment to proceed.

---

## 14. Automated Bootstrap (Alternative)

For machines that already have Docker and CUDA installed:

```bash
curl -sSL https://raw.githubusercontent.com/<your-org>/acas/main/scripts/add-node-to-cluster.sh \
  | bash -s -- \
    --control-plane "https://acas-cp.<subdomain>.workers.dev" \
    --node-name     "prod-node-1" \
    --hostname      "node-1.acas.example.com" \
    --connectivity  cloudflare_tunnel
```

For a brand-new Ubuntu 22.04 machine (installs NVIDIA drivers + Docker too):

```bash
curl -sSL https://raw.githubusercontent.com/<your-org>/acas/main/scripts/bootstrap-node.sh \
  | bash -s -- \
    --control-plane "https://acas-cp.<subdomain>.workers.dev" \
    --node-name     "prod-node-1" \
    --location      "Data Center A" \
    --connectivity  cloudflare_tunnel
```

The bootstrap script handles the mid-installation driver reboot and resumes automatically.

---

## 15. Adding More Nodes

Each additional GPU machine follows the same steps (2 → 11):

- Each node gets a **unique `NODE_ID`** — generate a fresh UUID each time.
- Each node gets its **own Cloudflare Tunnel** (Mode B) with a distinct subdomain.
- All nodes share the same `JWT_SECRET_KEY` and `NODE_AUTH_TOKEN`.
- The Control Plane load-balances cameras automatically, or use **GPU Cluster → Camera Assignment** in the dashboard.

### Deploy to all nodes at once

```bash
# inventory.txt: name  host  user  ssh-key  remote-dir
# node-1  node-1.acas.example.com  ubuntu  ~/.ssh/id_rsa  /opt/acas
# node-2  node-2.acas.example.com  ubuntu  ~/.ssh/id_rsa  /opt/acas

scripts/deploy-all.sh --inventory inventory.txt --parallel
```

---

## 16. Updating Deployed Nodes

```bash
scripts/deploy-all.sh \
  --inventory inventory.txt \
  --node node-1 \
  --branch main \
  --rollback-on-fail
```

The script SSHs in, pulls the latest code, rebuilds the backend image, performs a rolling restart, and verifies health — rolling back if the health check fails.

---

## 17. Service URLs Quick Reference

### Mode A — Local

| Service | URL | Notes |
|---|---|---|
| **Dashboard** | http://localhost:3020 | port may differ — use whichever was free |
| **Backend API** | http://localhost:18000 | |
| **API Docs** | http://localhost:18000/docs | Swagger UI |
| **Prometheus Metrics** | http://localhost:18000/metrics | |
| **MinIO Console** | http://localhost:19001 | user/pass from `.env` |
| PostgreSQL | localhost:15432 | acas / `POSTGRES_PASSWORD` |
| TimescaleDB | localhost:5433 | acas / `TIMESCALE_PASSWORD` |
| Redis | localhost:16379 | no auth |
| Kafka | localhost:9092 | |
| Schema Registry | http://localhost:18081 | |

### Mode B — Cloudflare

| Service | URL |
|---|---|
| **Dashboard** | https://acas-dashboard.pages.dev |
| **Control Plane** | https://acas-cp.\<subdomain\>.workers.dev |
| **Backend (per node)** | https://node-1.acas.example.com |

### Mode C — Cloud VM

| Service | URL |
|---|---|
| **Dashboard** | http://\<vm-ip\>:3020 or https://dashboard.example.com |
| **Backend API** | http://\<vm-ip\>:18000 or https://api.example.com |

---

## 18. Troubleshooting

### `api/health` returns `"status":"degraded"`

Expected in **Mode A** — `cp_registered: no` simply means no Control Plane is configured. As long as `postgres`, `redis`, `minio`, and `kafka` all show `ok`, the backend is fully functional.

### AI pipeline shows `pipeline_not_loaded_quality_check_skipped`

Models are missing or ORT can't reach the GPU. Check in order:

1. **Models downloaded?**
   ```bash
   docker exec acas-backend ls /models/
   # Must show: yolov8l.onnx  adaface_ir101_webface12m.onnx  minifasnet_v2.onnx  buffalo_l/
   ```

2. **onnxruntime-gpu installed?**
   ```bash
   docker exec acas-backend python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
   # Must include CUDAExecutionProvider
   ```
   If not, reinstall: `docker exec acas-backend pip install onnxruntime-gpu==1.20.1`

3. **LD_LIBRARY_PATH set?** `docker-compose.yml` must include in the backend's `environment`:
   ```yaml
   LD_LIBRARY_PATH: /usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cuda_runtime/lib:/usr/local/cuda/lib64
   ```

4. After all fixes: `docker compose up -d --force-recreate backend`

### All models running on CPU instead of GPU

The cuDNN libraries are installed but not found at runtime. Check `LD_LIBRARY_PATH` is set (see above) and force-recreate the container:

```bash
docker compose up -d --force-recreate backend
```

### "Failed to save zone"

The backend expects the zone as a polygon list `[{x, y}, ...]` (normalized 0–1). If you see a 422 error, confirm the backend is running the latest code. Trigger a reload:

```bash
docker compose logs backend --tail=5
# If stale: docker compose restart backend
```

### Enroll button stays disabled after images pass quality check

The Enroll button requires at least 1 image with `status === "pass"`. This was a bug where the final Review step still checked `< 5` passing images. Confirm the dashboard was rebuilt after the fix: the check is now `< 1`.

```bash
cd /opt/acas/dashboard
npm run build && PORT=3020 nohup npm start > /tmp/dashboard.log 2>&1 &
```

### Start AI button is disabled on the camera card

Intentional — the AI session cannot start without a defined monitoring zone. Draw and save a polygon zone via the **Zone & PTZ** tab first.

### `npm run build` fails with TypeScript errors

```bash
# Duplicate icon imports in cluster page
sed -i 's/, TrendingDown, BarChart3,/, BarChart3,/' \
  dashboard/src/app/\(dashboard\)/cluster/page.tsx

# Invalid type cast
sed -i 's/(h\["connectivity"\] as "PUBLIC_IP"/(h["connectivity"] as unknown) as "PUBLIC_IP"/' \
  dashboard/src/app/\(dashboard\)/cluster/page.tsx
```

Then re-run `npm run build`.

### Dashboard `npm run dev` crashes with "ENOSPC: System limit for number of file watchers"

Use the production build — it has no file watchers:

```bash
NEXT_PUBLIC_API_URL=http://localhost:18000 npm run build
PORT=3020 npm start
```

### Migrations fail — "alembic: command not found"

Copy the Alembic files into the container first:

```bash
docker cp backend/alembic      acas-backend:/app/alembic
docker cp backend/alembic.ini  acas-backend:/app/alembic.ini
docker exec -w /app acas-backend alembic upgrade head
```

### Login returns "Invalid credentials" after re-deploy

Reset the password hash directly:

```bash
HASH=$(docker exec acas-backend python3 -c \
  "from app.utils.security import hash_password; print(hash_password('Admin@2024!'))")

docker exec acas-postgres psql -U acas -d acas \
  -c "UPDATE users SET password_hash='${HASH}' WHERE email='admin@acas.local';"
```

### Kafka consumer warnings — "Subscribed topic not available"

Topics haven't been created yet. Run the topic creation block from [Section 3](#3-create-kafka-topics).

### Node shows OFFLINE in the dashboard (Mode B)

Heartbeats fail every 30 s. The node is marked OFFLINE after 3 missed heartbeats (~90 s):

```bash
docker compose logs backend | grep -i "heartbeat\|register\|control.plane" | tail -20
# 401 → NODE_AUTH_TOKEN mismatch
# 503 → Control Plane Worker unreachable
```

Fix the `NODE_AUTH_TOKEN` in `.env` to match `wrangler secret put NODE_AUTH_TOKEN`, then restart.

### Tunnel is up but health check times out

```bash
sudo systemctl status cloudflared
sudo journalctl -u cloudflared -n 50
curl http://localhost:18000/api/health     # test local access
cat /etc/cloudflared/config.yml           # verify hostname matches DNS
cloudflared tunnel info acas-node-1
```

### GPU OOM during inference

```bash
watch -n 2 nvidia-smi
# Reduce concurrency in .env:
MAX_GPU_CONCURRENT=3    # default is 5
docker compose restart backend
```

### Face sync stuck or taking too long

```bash
docker compose logs backend | grep -i "face.sync\|resync\|faiss" | tail -40
```

Trigger a manual full resync:

```bash
TOKEN=$(curl -s -X POST http://localhost:18000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@acas.local","password":"Admin@2024!"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

curl -X POST http://localhost:18000/api/node/config/reload \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"full_resync": true}'
```

### Backup and restore

```bash
# Full backup to MinIO
scripts/backup.sh --type full --node-dir /opt/acas

# Restore
scripts/restore.sh --help
```

### Decommission a node (Mode B/C)

1. **GPU Cluster → [node] → Drain Node** — migrates all cameras gracefully.
2. Wait for the Migration Log to show all cameras moved.
3. **GPU Cluster → [node] → Remove Node**.
4. On the machine:

```bash
sudo systemctl stop  cloudflared
sudo systemctl disable cloudflared
docker compose down -v
cloudflared tunnel delete acas-node-1   # Mode B only
```

---

## 19. Security Checklist

Before going to production:

- [ ] All `.env` passwords are unique, ≥ 20 characters, mixed case + symbols
- [ ] `JWT_SECRET_KEY` uses RS256 with a 4096-bit RSA key (not the HS256 dev placeholder)
- [ ] `NODE_AUTH_TOKEN` is at least 32 random characters
- [ ] `NODE_ID` is a stable UUID persisted in `.env` (not regenerated on each restart)
- [ ] **(Mode B)** Cloudflare Tunnel credential file has permissions `600` owned by `root`
- [ ] **(Mode B)** No inbound firewall ports are open on the node machine
- [ ] **(Mode C)** Postgres, Redis, MinIO, Kafka ports are **not** exposed to the public internet
- [ ] **(Mode C)** Backend is behind TLS (nginx + Certbot or cloud load balancer)
- [ ] `LD_LIBRARY_PATH` is set in `docker-compose.yml` (required for CUDA model inference)
- [ ] `onnxruntime-gpu==1.20.1` is in `backend/requirements.txt` (prevents CPU fallback after image rebuild)
- [ ] Prometheus alerts in `prometheus/alerts.yml` are routed to your on-call channel
- [ ] Regular backups scheduled: `0 2 * * * /opt/acas/scripts/backup.sh --type full`
- [ ] Docker log rotation enabled: `max-size: "100m"` in the logging config of `docker-compose.yml`
