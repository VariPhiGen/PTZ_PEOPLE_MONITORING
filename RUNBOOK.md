# ACAS Operations Runbook

Last updated: 2026-03-21  
System: Attendance & Camera Automation System (GPU node + Cloudflare control plane)

---

## Table of Contents

1. [Camera Offline](#1-camera-offline)
2. [GPU OOM / Saturation](#2-gpu-oom--saturation)
3. [Kafka Restart / Consumer Lag](#3-kafka-restart--consumer-lag)
4. [Database Backup & Restore](#4-database-backup--restore)
5. [Model Swap / Hot-Reload](#5-model-swap--hot-reload)
6. [Enrollment Troubleshooting](#6-enrollment-troubleshooting)
7. [Low Recognition Accuracy](#7-low-recognition-accuracy)
8. [FAISS Rebuild](#8-faiss-rebuild)
9. [MinIO Storage Full](#9-minio-storage-full)
10. [DB Slow Queries](#10-db-slow-queries)
11. [Pipeline Slow](#11-pipeline-slow)
12. [Node Registration Failure](#12-node-registration-failure)

---

## 1. Camera Offline

**Alert:** `CameraOffline` — `acas_camera_status == 0` for > 5 minutes.

### Symptoms
- Live dashboard shows camera card as "OFFLINE" (red badge).
- PTZ scanning halted; no attendance records written for that room.
- ONVIF heartbeat errors in backend log: `[ONVIFController] CameraOfflineError`.

### Diagnosis

```bash
# 1. Check RTSP reachability from the GPU node
docker exec acas-backend \
  ffprobe -v quiet -print_format json -show_streams \
  "rtsp://<user>:<pass>@<ip>:554/video/live"

# 2. Ping the camera
ping -c 4 <camera_ip>

# 3. Check ONVIF device service
curl -s "http://<camera_ip>/onvif/device_service" | head -5

# 4. Read backend logs for this camera
docker logs acas-backend --since 15m 2>&1 | grep <camera_id>
```

### Resolution

| Cause | Fix |
|---|---|
| Camera power cycled | Wait for boot (~90 s); ACAS will auto-reconnect |
| Network switch port down | Cycle switch port; verify VLAN config |
| RTSP URL changed | Update camera record via `PUT /api/cameras/<id>` |
| ONVIF credentials expired | Re-encrypt new password: `PUT /api/cameras/<id>` with new `onvif_password` |
| IP changed (DHCP) | Assign static IP on camera; update `onvif_host` + RTSP URL in ACAS |
| Camera firmware update | Hard-reboot camera; confirm ONVIF API compatibility |

```bash
# Force ACAS to re-test connection
curl -X POST http://localhost:8000/api/cameras/<camera_id>/test \
  -H "Authorization: Bearer $TOKEN"

# Restart decoder for a single camera without full node restart
curl -X POST http://localhost:8000/api/node/cameras/<camera_id>/stop \
  -H "Authorization: Bearer $TOKEN"
curl -X POST http://localhost:8000/api/node/cameras/<camera_id>/start \
  -H "Authorization: Bearer $TOKEN"
```

---

## 2. GPU OOM / Saturation

**Alerts:** `GPUSaturation` (util > 95% for 2 min), `GPUDegradationActive` (mode > 1).

### Symptoms
- Grafana "GPU Utilisation" gauge red.
- `acas_gpu_degradation_mode` gauge > 0 in Prometheus.
- Recognition accuracy drops; liveness checks skipped.
- Backend log: `GPUManager: degradation FULL → NO_LIVENESS`.

### Diagnosis

```bash
# Live GPU state
nvidia-smi -l 2

# Current sessions and degradation mode
curl http://localhost:8000/api/node/info | jq '.gpu_manager'

# Probe saturation (non-destructive, runs synthetic inference)
docker exec acas-backend python - <<'EOF'
import asyncio
from app.core.gpu_manager import GPUManager
mgr = GPUManager.get()
results = asyncio.run(mgr.probe_saturation([5, 10, 15, 20], iters_each=20))
for r in results:
    print(r)
EOF
```

### Resolution

**Immediate relief (seconds)**
```bash
# Stop the busiest session
curl -X POST http://localhost:8000/api/sessions/<session_id>/stop \
  -H "Authorization: Bearer $TOKEN"

# Lower max_concurrent via env and restart backend
# In .env.prod:  GPU_MAX_CONCURRENT=3
docker compose -f docker-compose.prod.yml restart backend
```

**Structural fix**
1. Reduce active sessions on this node. Migrate cameras to another node via the Cluster page or control-plane API.
2. Increase `GPU_VRAM_BUDGET_GB` only if you have measured free VRAM (leave ≥ 3 GB for OS + FAISS).
3. Consider enabling MIG (Multi-Instance GPU) partitioning if you have an A100/H100.
4. Run `scripts/benchmark_pipeline.py --mode stress` to find the real saturation point before choosing `max_concurrent`.

**OOM killed (CUDA device-side)**
```bash
# Identify killed process
docker logs acas-backend 2>&1 | grep -i "killed\|OOM\|CUDA error"

# Clear GPU context and restart cleanly
docker compose -f docker-compose.prod.yml restart backend
# Wait for FAISS reload (~60 s for 50K vectors)
```

---

## 3. Kafka Restart / Consumer Lag

**Alerts:** `KafkaConsumerLagHigh` (lag > 1000), `KafkaProduceErrors` (error rate > 5%).

### Symptoms
- ERP sync delayed; attendance records stuck in `PENDING` or `HELD`.
- `acas_kafka_consumer_lag` rising in Grafana Kafka dashboard.
- Backend log: `KafkaProducer: delivery failed`.

### Diagnosis

```bash
# List consumer groups and lag
docker exec acas-kafka /opt/kafka/bin/kafka-consumer-groups.sh \
  --bootstrap-server kafka:9092 \
  --describe --all-groups

# Topic status
docker exec acas-kafka /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server kafka:9092 \
  --describe

# Broker health
docker exec acas-kafka /opt/kafka/bin/kafka-metadata-quorum.sh \
  --bootstrap-server kafka:9092 \
  --command describe

# Check broker logs
docker logs acas-kafka --tail 100 2>&1 | grep -E "ERROR|WARN|leader"
```

### Resolution

**Consumer lag (producer healthy)**
```bash
# Option A: restart the override consumer worker
docker compose -f docker-compose.prod.yml restart backend
# The AdminOverrideConsumer restarts automatically during lifespan

# Option B: manually reset consumer group offset (use only during maintenance)
docker exec acas-kafka /opt/kafka/bin/kafka-consumer-groups.sh \
  --bootstrap-server kafka:9092 \
  --group acas-override-worker \
  --reset-offsets --to-latest \
  --topic admin.overrides --execute
```

**Broker restart**
```bash
# Graceful restart (KRaft — no Zookeeper)
docker compose -f docker-compose.prod.yml restart kafka

# Wait for broker to become leader (~15 s)
docker exec acas-kafka /opt/kafka/bin/kafka-metadata-quorum.sh \
  --bootstrap-server kafka:9092 --command describe | grep "LeaderId"

# Verify topic list
docker exec acas-kafka /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server kafka:9092 --list

# Restart schema registry after broker
docker compose -f docker-compose.prod.yml restart schema-registry
```

**Schema Registry incompatibility**
```bash
# List subjects
curl http://localhost:8081/subjects

# Delete a subject (only on dev/staging)
curl -X DELETE http://localhost:8081/subjects/<subject-name>

# Check compatibility
curl http://localhost:8081/config
```

---

## 4. Database Backup & Restore

### Automated daily backup
The `postgres-backup` sidecar runs `pg_dumpall` daily at 02:00 UTC, storing compressed dumps to the `acas_pg_backups` Docker volume (mapped to `/var/backups/acas` on the host if configured).

```bash
# Confirm latest backup exists
docker exec acas-postgres-backup ls -lh /backups/ | tail -5

# Force a backup now
docker exec acas-postgres-backup /scripts/postgres/backup.sh
```

### Manual backup
```bash
# Full cluster dump (roles + all databases)
docker exec acas-postgres pg_dumpall \
  -U postgres \
  | gzip -9 > acas_backup_$(date +%Y%m%d_%H%M%S).sql.gz

# Single database dump
docker exec acas-postgres pg_dump \
  -U postgres acas \
  -Fc -Z9 \
  > acas_$(date +%Y%m%d).dump
```

### Restore
```bash
# Stop backend first (prevent writes during restore)
docker compose -f docker-compose.prod.yml stop backend

# Restore full dump
gunzip -c acas_backup_YYYYMMDD_HHMMSS.sql.gz \
  | docker exec -i acas-postgres psql -U postgres

# Restore single database
docker exec -i acas-postgres pg_restore \
  -U postgres -d acas -c < acas_YYYYMMDD.dump

# Run migrations to ensure schema is current
docker compose -f docker-compose.prod.yml run --rm backend \
  alembic upgrade head

# Restart backend
docker compose -f docker-compose.prod.yml start backend
```

### TimescaleDB considerations
```sql
-- After restore, verify hypertable is intact
SELECT * FROM timescaledb_information.hypertables;

-- Reapply compression policy if missing
SELECT add_compression_policy('detection_log', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_retention_policy('detection_log', INTERVAL '90 days', if_not_exists => TRUE);
```

---

## 5. Model Swap / Hot-Reload

Use this procedure when deploying updated ONNX weights (e.g., a retrained AdaFace model) without full service downtime.

### Prepare new engine
```bash
# 1. Copy new ONNX to /models (while old engine still in use)
cp /path/to/adaface_ir101_v2.onnx /models/adaface_ir101_webface12m.onnx

# 2. Pre-compile TensorRT engine in a throwaway container
docker run --rm --gpus all \
  -v /models:/models \
  acas-backend:gpu \
  python scripts/optimize_models.py \
    --model-dir /models \
    --engine-dir /models/engines \
    --force              # force rebuild even if hash matches

# 3. Verify accuracy on LFW (optional, ~5 min)
docker run --rm --gpus all \
  -v /models:/models \
  -v /path/to/lfw:/lfw \
  acas-backend:gpu \
  python scripts/optimize_models.py \
    --model-dir /models \
    --engine-dir /models/engines \
    --lfw-dir /lfw \
    --skip yolo,retina,fasnet   # only benchmark adaface
```

### Rolling reload
```bash
# 4. Gracefully restart backend — Docker Compose will pull new engine on load
docker compose -f docker-compose.prod.yml restart backend

# 5. Monitor recognition rate in Grafana — should not drop below 88%
# Alert: RecognitionAccuracyLow fires if it drops below 90% for 1 hr

# 6. Roll back if needed
cp /models/engines/adaface_ir101_webface12m.onnx.engine.bak \
   /models/engines/adaface_ir101_webface12m.onnx.engine
docker compose -f docker-compose.prod.yml restart backend
```

### Model version tracking
The `Model Versions` table in the Settings → AI & PTZ dashboard tab stores `{model_name, version, sha256, deployed_at}`.  After a swap, run:
```bash
curl -X PUT http://localhost:8000/api/kafka/config \
  -H "Authorization: Bearer $SUPERADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model_versions": {"adaface": "v2.0.0-webface12m"}}'
```

---

## 6. Enrollment Troubleshooting

### Symptom: all images rejected at quality gate

Backend log: `quality_gate: FAIL  reason=no_face` or `reason=blur`.

**Check image requirements:**
- Exactly 1 face detected by RetinaFace.
- Inter-ocular distance ≥ 80 px (at minimum camera resolution).
- Sharpness (Laplacian variance) ≥ 50.
- No face occlusion (mask, sunglasses).
- Minimum image size: 200 × 200 px.

```bash
# Run quality gate manually on a test image
docker exec acas-backend python - <<'EOF'
from app.services.ai_pipeline import AIPipeline
import cv2, sys
pipeline = AIPipeline("/models", device_id=0)
pipeline.load()

img = cv2.imread("/path/to/test.jpg")
faces = pipeline.detect_faces(img, [])
for f in faces:
    print(f"IOD={f.inter_ocular_px:.0f}  conf={f.conf:.2f}")
    chip = pipeline.align_face(img, f.landmarks)
    lap = cv2.Laplacian(chip, cv2.CV_64F).var()
    print(f"  Laplacian sharpness={lap:.1f} (need >=50)")
EOF
```

### Symptom: enrollment succeeds but person never recognised

1. **Embedding stored but FAISS not rebuilt:**
```bash
# Trigger manual FAISS rebuild for client
curl -X POST http://localhost:8000/api/node/config/reload \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"action": "rebuild_faiss", "client_id": "<uuid>"}'
```

2. **Wrong client_id on enrollment:**
Check that the JWT `client_id` matches the person's `client_id` in the DB.

3. **Embedding drift — enrolled image quality too different from real pose:**
Re-enroll with 8–12 images covering +/- 15° yaw and ±10° pitch. Use the Guidelines Panel.

4. **Template stuck at INACTIVE:**
```sql
-- In PostgreSQL
SELECT is_active, confidence_avg, quality_score
FROM face_embeddings
WHERE person_id = '<uuid>'
ORDER BY created_at DESC LIMIT 5;

-- Reactivate
UPDATE face_embeddings SET is_active = true
WHERE person_id = '<uuid>' AND version = (
  SELECT MAX(version) FROM face_embeddings WHERE person_id = '<uuid>'
);
```

### Bulk import CSV failures
```bash
# View last bulk import errors
docker logs acas-backend --since 10m 2>&1 \
  | grep "bulk_import\|csv"

# CSV column order: external_id,name,role,department,email,phone
# Role values: STUDENT | FACULTY | ADMIN
# Images must be in a ZIP file matching external_id as filename stem
```

---

## 7. Low Recognition Accuracy

**Alert:** `RecognitionAccuracyLow` — rate below 90% for 1 hour.

### Diagnosis checklist

```bash
# Per-client accuracy breakdown
curl http://localhost:8000/api/analytics/recognition-accuracy \
  -H "Authorization: Bearer $TOKEN" | jq '.by_client[]'

# Check liveness fail rate (is this accuracy or liveness?)
# Grafana → AI Pipeline → "Liveness Fail %" panel

# Check FAISS index freshness
curl http://localhost:8000/api/node/info | jq '.faiss_indexes'

# Sample recognition events from the last hour
docker exec acas-redis redis-cli \
  LRANGE "acas:debug:recognition:$(date -u +%Y%m%d%H)" 0 20
```

### Common causes and fixes

| Cause | Evidence | Fix |
|---|---|---|
| Lighting changed in room | Accuracy drops at specific hour | Add supplemental lighting; re-enroll in new conditions |
| Camera repositioned | All confidence scores drop | Re-draw ROI; re-run session to rebuild seat heatmap |
| FAISS index stale after enrollment | New person never matched | Trigger FAISS rebuild (see §8) |
| AdaFace FP16 accuracy regression | New model deployed recently | Roll back model (see §5) |
| ef_search too low | HNSW recall below expected | Increase via `SET hnsw.ef_search = 150` |
| Many unknown visitors | Monitoring mode camera | Expected — only enrolled persons are recognised |

```bash
# Increase ef_search temporarily for a session
docker exec acas-postgres psql -U postgres acas \
  -c "ALTER DATABASE acas SET hnsw.ef_search = 150;"
docker compose -f docker-compose.prod.yml restart backend
```

---

## 8. FAISS Rebuild

**Alert:** `FAISSIndexStale` — index size < 1 for a client.

### Manual rebuild
```bash
# Via API (preferred — runs async in backend)
curl -X POST http://localhost:8000/api/node/config/reload \
  -H "Authorization: Bearer $SUPERADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"action": "rebuild_faiss", "client_id": null}'
  # null = rebuild all clients on this node

# Monitor rebuild progress
docker logs acas-backend --follow 2>&1 | grep "FAISS\|faiss"
```

### Via face sync service (cross-node)
```bash
docker exec acas-backend python - <<'EOF'
import asyncio
from app.main import app
sync = app.state.face_sync
asyncio.run(sync.full_resync(client_id=None))   # all clients
EOF
```

### Expected rebuild time
| Vectors | Time |
|---|---|
| 10 K | ~3 s |
| 50 K | ~25 s |
| 200 K | ~90 s |
| 1 M | ~7 min |

For indexes > 200 K, schedule the rebuild during a low-traffic window.

---

## 9. MinIO Storage Full

**Alert:** `MinIOStorageHigh` — cluster storage > 80%.

### Diagnosis
```bash
# Disk usage by bucket
docker exec acas-minio mc du /data --depth 2

# Check lifecycle rules are applied
docker exec acas-minio-init mc ilm ls acas/face-evidence
docker exec acas-minio-init mc ilm ls acas/snapshots
```

### Immediate actions
```bash
# Force expire objects past lifecycle date (run lifecycle scan now)
docker exec acas-minio mc ilm tier ls acas/face-evidence

# Delete old exports manually (safe — regeneratable)
docker exec acas-minio-init mc rm --recursive --force \
  --older-than 30d acas/exports/

# Verify MinIO has erasure coding headroom
docker exec acas-minio mc admin info acas
```

### Structural fix
1. Reduce `face-evidence` retention from 30 d → 14 d in MinIO lifecycle config.
2. Add a second MinIO node (scale-out erasure set from 4 → 8 disks).
3. Archive old buckets to cold storage (Cloudflare R2, AWS S3 Glacier) via `mc mirror`.

```bash
# Update lifecycle rule: shorten face-evidence to 14 days
docker exec acas-minio-init mc ilm add \
  --expiry-days 14 acas/face-evidence
```

---

## 10. DB Slow Queries

**Alert:** `DBPoolExhausted` — pool checked-out ≥ 28.

### Identify slow queries
```sql
-- Long-running queries (> 5 s)
SELECT pid, now() - query_start AS duration, query
FROM pg_stat_activity
WHERE state != 'idle' AND query_start < now() - INTERVAL '5 seconds'
ORDER BY duration DESC;

-- Kill a blocking query
SELECT pg_cancel_backend(<pid>);

-- Top queries by total time (requires pg_stat_statements)
SELECT query, calls, mean_exec_time, total_exec_time
FROM pg_stat_statements
ORDER BY total_exec_time DESC LIMIT 20;

-- Check for sequential scans on hot tables
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
  SELECT * FROM attendance_records
  WHERE session_id = '<uuid>' AND status = 'P';
```

### Connection leak check
```bash
# Count connections by state
docker exec acas-postgres psql -U postgres acas -c \
  "SELECT state, count(*) FROM pg_stat_activity GROUP BY state;"

# Backend DB pool stats
curl http://localhost:9090/api/v1/query?query=acas_db_pool_checkedout
```

### Missing index check
```sql
-- Tables with highest sequential scan ratios
SELECT relname, seq_scan, idx_scan,
       round(seq_scan * 100.0 / (seq_scan + idx_scan + 1), 1) AS seq_pct
FROM pg_stat_user_tables
WHERE seq_scan + idx_scan > 100
ORDER BY seq_pct DESC LIMIT 10;
```

If a table is missing an index, run migration `0002` or add an index manually:
```bash
docker exec acas-backend alembic upgrade 0002
```

---

## 11. Pipeline Slow

**Alert:** `PipelineLatencyHigh` — p95 > 200 ms for 1 min.

### Quick diagnosis
```bash
# Real-time per-stage latency from Prometheus
curl -s "http://localhost:9090/api/v1/query?query=\
histogram_quantile(0.95,rate(acas_pipeline_latency_seconds_bucket[1m]))" \
  | jq '.data.result[] | {stage: .metric.stage, p95_ms: (.value[1] | tonumber * 1000 | round)}'

# Run offline benchmark
docker run --rm --gpus all \
  -v /models:/models \
  acas-backend:gpu \
  python scripts/benchmark_pipeline.py \
    --mode stress --faces 15 --fps 10 --duration 30
```

### Causes and fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| YOLO slow (>25 ms) | Batch too large or TRT engine stale | Re-run `optimize_models.py --force --skip retina,adaface,fasnet` |
| AdaFace slow (>5 ms/face) | FP32 fallback active | Check TRT engine log for FP32 layers; re-optimize with `--workspace-gb 4` |
| Total slow, stages fine | GPU→CPU copy overhead | Verify `CUDAExecutionProvider` is first in ORT EP list |
| Latency spikes every ~60 s | TRT engine JIT recompile | Pre-warm the engine: run `optimize_models.py` before starting sessions |
| High variance | Thermal throttling | Check `nvidia-smi -q -d TEMPERATURE`; clean GPU fans |

---

## 12. Node Registration Failure

### Symptoms
- Backend log on startup: `NodeManager: registration failed`.
- Control plane shows node as `OFFLINE` immediately after deployment.
- No cameras assigned to node.

### Diagnosis
```bash
# Test control plane reachability
docker exec acas-backend curl -s \
  "$CONTROL_PLANE_URL/api/health" | jq '.status'

# Test node API key (returned at registration)
docker exec acas-backend curl -s \
  "$CONTROL_PLANE_URL/api/nodes/$NODE_ID" \
  -H "Authorization: Bearer $NODE_API_KEY"

# Check .env.prod for required vars
grep -E "CONTROL_PLANE_URL|NODE_ID|NODE_NAME|NODE_API_KEY" /opt/acas/.env.prod
```

### Re-registration
```bash
# Force re-registration (clears stored NODE_API_KEY so a new one is issued)
docker exec acas-backend python - <<'EOF'
import asyncio
from app.main import app
mgr = app.state.node_manager
asyncio.run(mgr.register())
print("NODE_API_KEY:", mgr._api_key)
EOF

# Save the new key to .env.prod
# NODE_API_KEY=<new-key>
```

### Heartbeat loop recovery
The heartbeat loop auto-restarts on failure (the `NodeManager` catches exceptions internally). If the control plane was unreachable for > 3 cycles, cameras may have been reassigned. After recovery:
```bash
curl -X POST http://localhost:8000/api/node/config/reload \
  -H "Authorization: Bearer $SUPERADMIN_TOKEN" \
  -d '{"action": "reload_cameras"}'
```

---

## Quick Reference: Service Commands

```bash
# Start everything
bash scripts/start.sh

# Start with monitoring stack
bash scripts/start.sh --monitoring

# Individual service restart
docker compose -f docker-compose.prod.yml restart <service>
# services: backend | postgres | timescaledb | redis | minio | kafka | schema-registry | cloudflared

# Backend logs (last 200 lines, follow)
docker compose -f docker-compose.prod.yml logs -f --tail 200 backend

# Connect to PostgreSQL
docker exec -it acas-postgres psql -U postgres acas

# Connect to Redis
docker exec -it acas-redis redis-cli

# Kafka topic list
docker exec acas-kafka /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server kafka:9092 --list

# MinIO console (if mapped locally — only during maintenance)
# Access via Cloudflare Tunnel: https://minio-console.your-domain.com
```

---

## Escalation Matrix

| Severity | Response time | Who |
|---|---|---|
| CRITICAL alert (GPU, camera, Kafka) | 15 min | On-call engineer |
| WARNING alert | 4 hr (business hours) | Platform team |
| Accuracy drift | Next business day | ML team |
| Enrollment issue | Same day | Support + ML team |
| Data privacy / consent | Immediately | DPO + Legal |
