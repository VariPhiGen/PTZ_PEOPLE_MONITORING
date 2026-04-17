#!/usr/bin/env bash
# ACAS Production Startup Script
# Usage:  bash scripts/start.sh [--skip-pull] [--skip-models] [--monitoring]
set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.prod.yml"
ENV_FILE="${PROJECT_ROOT}/.env.prod"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; NC='\033[0m'

log()  { echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} $*"; }
ok()   { echo -e "${GREEN}[$(date +%H:%M:%S)] ✓${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] ⚠${NC} $*"; }
die()  { echo -e "${RED}[$(date +%H:%M:%S)] ✗${NC} $*" >&2; exit 1; }
step() { echo -e "\n${BOLD}${CYAN}══════════════════════════════════════════════${NC}"; \
         echo -e "${BOLD}${CYAN}  $*${NC}"; \
         echo -e "${BOLD}${CYAN}══════════════════════════════════════════════${NC}"; }

# ── Argument parsing ──────────────────────────────────────────────────────────
SKIP_PULL=0; SKIP_MODELS=0; WITH_MONITORING=0
for arg in "$@"; do
  case "$arg" in
    --skip-pull)    SKIP_PULL=1    ;;
    --skip-models)  SKIP_MODELS=1  ;;
    --monitoring)   WITH_MONITORING=1 ;;
    --help)
      echo "Usage: $0 [--skip-pull] [--skip-models] [--monitoring]"
      echo "  --skip-pull     Skip docker pull (use local images)"
      echo "  --skip-models   Skip model download check"
      echo "  --monitoring    Start Prometheus+Grafana profile"
      exit 0 ;;
    *) die "Unknown argument: $arg" ;;
  esac
done

# Build compose command
if [[ $WITH_MONITORING -eq 1 ]]; then
  COMPOSE_CMD="docker compose -f ${COMPOSE_FILE} --env-file ${ENV_FILE} --profile monitoring"
else
  COMPOSE_CMD="docker compose -f ${COMPOSE_FILE} --env-file ${ENV_FILE}"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 1. PRE-FLIGHT: .env.prod + required secrets
# ─────────────────────────────────────────────────────────────────────────────
step "1/10  Pre-flight checks"

[[ -f "$ENV_FILE" ]] || die ".env.prod not found. Copy .env.prod.example and fill in secrets."

# Load env for use in this script
set -a; source "$ENV_FILE"; set +a

REQUIRED_VARS=(
  POSTGRES_PASSWORD TIMESCALE_PASSWORD
  MINIO_ROOT_USER MINIO_ROOT_PASSWORD
  JWT_SECRET_KEY KAFKA_CLUSTER_ID
)
missing=()
for var in "${REQUIRED_VARS[@]}"; do
  [[ -n "${!var:-}" ]] || missing+=("$var")
done
if [[ ${#missing[@]} -gt 0 ]]; then
  die "Missing required variables in .env.prod: ${missing[*]}"
fi

# Warn about default/weak values
[[ "${JWT_SECRET_KEY:-}" != "change-me-use-rsa-pem-for-rs256" ]] || \
  warn "JWT_SECRET_KEY appears to be the default placeholder. Change it before production."

ok ".env.prod loaded (${#REQUIRED_VARS[@]} required vars present)"

# ─────────────────────────────────────────────────────────────────────────────
# 2. GPU + NVIDIA runtime
# ─────────────────────────────────────────────────────────────────────────────
step "2/10  Checking GPU + NVIDIA runtime"

command -v nvidia-smi &>/dev/null  || die "nvidia-smi not found. Install NVIDIA drivers first."
nvidia-smi --query-gpu=name,driver_version,memory.total \
           --format=csv,noheader 2>/dev/null | while IFS=',' read -r name drv mem; do
  ok "GPU: ${name// /} | Driver ${drv// /} | VRAM ${mem// /}"
done

# Check nvidia container runtime
if docker info 2>/dev/null | grep -q "nvidia"; then
  ok "NVIDIA container runtime detected"
elif command -v nvidia-container-runtime &>/dev/null; then
  ok "nvidia-container-runtime binary found"
else
  warn "nvidia-container-runtime not found in docker info. GPU containers may fail."
  warn "Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
fi

# Quick CUDA sanity check using a throwaway container
log "Testing CUDA availability in container…"
if docker run --rm --runtime=nvidia \
     -e NVIDIA_VISIBLE_DEVICES=all \
     nvidia/cuda:12.4.1-base-ubuntu22.04 \
     nvidia-smi -L &>/dev/null; then
  ok "CUDA container test passed"
else
  warn "CUDA container test failed — GPU may be unavailable inside Docker"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 3. Pull images
# ─────────────────────────────────────────────────────────────────────────────
step "3/10  Pulling images"

if [[ $SKIP_PULL -eq 1 ]]; then
  warn "Skipping image pull (--skip-pull)"
else
  log "Building acas-backend:gpu…"
  docker build -t acas-backend:gpu \
    -f "${PROJECT_ROOT}/backend/Dockerfile.gpu" \
    "${PROJECT_ROOT}/backend" | \
    grep -E "^(Step|Successfully|=>)" || true
  ok "acas-backend:gpu built"

  log "Pulling service images…"
  $COMPOSE_CMD pull --quiet \
    postgres timescaledb redis minio kafka schema-registry cloudflared 2>&1 | \
    grep -v "^$" | head -40 || true
  ok "Images pulled"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 4. AI Model files
# ─────────────────────────────────────────────────────────────────────────────
step "4/10  Checking AI model files"

MODELS_DIR="${PROJECT_ROOT}/models"
mkdir -p "$MODELS_DIR"

REQUIRED_MODELS=(
  "yolov8l.onnx"
  "buffalo_l/det_10g.onnx"
  "adaface_ir101_webface12m.onnx"
  "minifasnet_v2.onnx"
)

missing_models=()
for m in "${REQUIRED_MODELS[@]}"; do
  [[ -f "${MODELS_DIR}/${m}" ]] || missing_models+=("$m")
done

if [[ ${#missing_models[@]} -eq 0 ]]; then
  ok "All model files present"
elif [[ $SKIP_MODELS -eq 1 ]]; then
  warn "Missing models: ${missing_models[*]} (skipped with --skip-models)"
else
  warn "Missing: ${missing_models[*]}"
  log "Running model download script…"
  if command -v python3 &>/dev/null; then
    python3 "${PROJECT_ROOT}/scripts/download_models.py" --output-dir "$MODELS_DIR" || \
      warn "download_models.py exited with errors — check manually"
  else
    warn "python3 not found on host. Run in container:"
    warn "  docker run --rm -v ${MODELS_DIR}:/models acas-backend:gpu python scripts/download_models.py"
  fi
fi

# Check for TensorRT engines (optional — generated on first startup if absent)
trt_count=$(find "$MODELS_DIR" -name "*.engine" 2>/dev/null | wc -l)
if [[ $trt_count -gt 0 ]]; then
  ok "TensorRT engines: ${trt_count} found"
else
  warn "No .engine files found — TensorRT engines will be compiled on first backend start (~5-15 min)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 5. Start infrastructure services
# ─────────────────────────────────────────────────────────────────────────────
step "5/10  Starting infrastructure services"

log "Starting postgres, timescaledb, redis, minio, kafka, schema-registry…"
$COMPOSE_CMD up -d \
  postgres timescaledb redis minio kafka schema-registry

ok "Infrastructure containers started"

# ─────────────────────────────────────────────────────────────────────────────
# 6. Wait for health
# ─────────────────────────────────────────────────────────────────────────────
step "6/10  Waiting for services to become healthy"

wait_healthy() {
  local service="$1" timeout="${2:-120}" elapsed=0
  log "Waiting for ${service}…"
  while true; do
    local status
    status=$(docker inspect --format='{{.State.Health.Status}}' "acas-${service}" 2>/dev/null || echo "missing")
    case "$status" in
      healthy)  ok "${service} is healthy"; return 0 ;;
      missing)  warn "${service} container not found yet"; ;;
      unhealthy) die "${service} is unhealthy. Check logs: docker logs acas-${service}" ;;
    esac
    if [[ $elapsed -ge $timeout ]]; then
      die "${service} did not become healthy within ${timeout}s"
    fi
    sleep 5; elapsed=$((elapsed + 5))
    echo -n "."
  done
}

wait_healthy postgres      120
wait_healthy timescaledb   120
wait_healthy redis          60
wait_healthy minio          90
wait_healthy kafka         120
wait_healthy schema-registry 60

# ─────────────────────────────────────────────────────────────────────────────
# 7. Database migrations (Alembic)
# ─────────────────────────────────────────────────────────────────────────────
step "7/10  Running database migrations"

log "Running Alembic upgrade head on primary DB…"
docker run --rm \
  --network acas-prod-net \
  -e DATABASE_URL="postgresql+asyncpg://acas:${POSTGRES_PASSWORD}@postgres:5432/acas" \
  -e TIMESCALE_URL="postgresql+asyncpg://acas:${TIMESCALE_PASSWORD}@timescaledb:5432/acas_ts" \
  -e REDIS_URL="redis://redis:6379/0" \
  -e MINIO_ENDPOINT="minio:9000" \
  -e MINIO_ACCESS_KEY="${MINIO_ROOT_USER}" \
  -e MINIO_SECRET_KEY="${MINIO_ROOT_PASSWORD}" \
  -e KAFKA_BOOTSTRAP_SERVERS="kafka:19092" \
  -e JWT_SECRET_KEY="${JWT_SECRET_KEY}" \
  -e JWT_ALGORITHM="${JWT_ALGORITHM:-RS256}" \
  -e NODE_NAME="${NODE_NAME:-acas-node-1}" \
  -v "${PROJECT_ROOT}/backend:/app:ro" \
  acas-backend:gpu \
  alembic -c /app/alembic.ini upgrade head

ok "Alembic migrations complete"

# Apply TimescaleDB compression/retention policies (idempotent)
log "Applying TimescaleDB compression and retention policies…"
docker exec acas-timescaledb \
  psql -U acas -d acas_ts -c \
  "SELECT add_compression_policy('detection_log', INTERVAL '7 days', if_not_exists => TRUE);
   SELECT add_retention_policy('detection_log', INTERVAL '90 days', if_not_exists => TRUE);" \
  2>/dev/null && ok "TimescaleDB policies set" || warn "TimescaleDB policies already set (OK)"

# ─────────────────────────────────────────────────────────────────────────────
# 8. MinIO buckets + lifecycle
# ─────────────────────────────────────────────────────────────────────────────
step "8/10  Initialising MinIO buckets and lifecycle"

log "Running minio-init container…"
$COMPOSE_CMD up --no-deps minio-init
ok "MinIO buckets and lifecycle policies configured"

# ─────────────────────────────────────────────────────────────────────────────
# 9. Kafka topics
# ─────────────────────────────────────────────────────────────────────────────
step "9/10  Creating Kafka topics"

log "Running kafka-init container…"
$COMPOSE_CMD up --no-deps kafka-init
ok "Kafka topics created"

# Verify topic count
TOPIC_COUNT=$(docker run --rm --network acas-prod-net \
  apache/kafka:3.9.2 \
  /opt/kafka/bin/kafka-topics.sh --bootstrap-server kafka:19092 --list 2>/dev/null | \
  grep -c "^" || echo 0)
ok "Kafka: ${TOPIC_COUNT} topics visible"

# ─────────────────────────────────────────────────────────────────────────────
# 10. Start application + tunnel
# ─────────────────────────────────────────────────────────────────────────────
step "10/10  Starting backend, tunnel, and optional services"

log "Starting postgres-backup sidecar…"
$COMPOSE_CMD up -d postgres-backup

log "Starting backend (gunicorn 4 workers)…"
$COMPOSE_CMD up -d backend

log "Waiting for backend health check…"
BACKEND_TIMEOUT=120; elapsed=0
while true; do
  status=$(docker inspect --format='{{.State.Health.Status}}' acas-backend 2>/dev/null || echo "missing")
  [[ "$status" == "healthy" ]] && break
  [[ "$status" == "unhealthy" ]] && die "Backend is unhealthy. Logs:\n$(docker logs --tail 50 acas-backend 2>&1)"
  [[ $elapsed -ge $BACKEND_TIMEOUT ]] && die "Backend did not become healthy within ${BACKEND_TIMEOUT}s"
  sleep 5; elapsed=$((elapsed + 5)); echo -n "."
done
ok "Backend healthy"

log "Starting Cloudflare tunnel…"
$COMPOSE_CMD up -d cloudflared
sleep 3
if docker ps --filter "name=acas-cloudflared" --filter "status=running" -q | grep -q .; then
  ok "cloudflared running"
else
  warn "cloudflared may not be running — check CLOUDFLARE_TUNNEL_TOKEN in .env.prod"
fi

if [[ $WITH_MONITORING -eq 1 ]]; then
  log "Starting monitoring stack (Prometheus + Grafana + exporters)…"
  $COMPOSE_CMD up -d prometheus grafana node-exporter redis-exporter
  ok "Monitoring stack started"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Final status table
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║               ACAS — Deployment Status                      ║${NC}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

SERVICES=(
  "acas-backend:Backend (4 workers)"
  "acas-postgres:PostgreSQL 16"
  "acas-timescaledb:TimescaleDB"
  "acas-redis:Redis 7"
  "acas-minio:MinIO (EC 4-drive)"
  "acas-kafka:Kafka KRaft"
  "acas-schema-registry:Schema Registry"
  "acas-cloudflared:Cloudflare Tunnel"
  "acas-pg-backup:PG Backup"
)
if [[ $WITH_MONITORING -eq 1 ]]; then
  SERVICES+=(
    "acas-prometheus:Prometheus"
    "acas-grafana:Grafana"
    "acas-node-exporter:Node Exporter"
    "acas-redis-exporter:Redis Exporter"
  )
fi

printf "  %-30s %-12s %s\n" "Service" "Status" "Container"
printf "  %-30s %-12s %s\n" "───────────────────────────────" "──────────" "───────────────"
all_ok=1
for entry in "${SERVICES[@]}"; do
  container="${entry%%:*}"
  label="${entry##*:}"
  status=$(docker inspect --format='{{.State.Status}}' "$container" 2>/dev/null || echo "missing")
  health=$(docker inspect --format='{{if .State.Health}}{{.State.Health.Status}}{{else}}-{{end}}' "$container" 2>/dev/null || echo "-")
  if [[ "$status" == "running" ]]; then
    if [[ "$health" == "healthy" || "$health" == "-" ]]; then
      symbol="${GREEN}●${NC}"; badge="${GREEN}running${NC}"
    else
      symbol="${YELLOW}●${NC}"; badge="${YELLOW}${health}${NC}"; all_ok=0
    fi
  else
    symbol="${RED}●${NC}"; badge="${RED}${status}${NC}"; all_ok=0
  fi
  printf "  %b %-28s %b\n" "$symbol" "$label" "$badge"
done

echo ""
if [[ $all_ok -eq 1 ]]; then
  echo -e "${GREEN}${BOLD}  All services healthy.${NC}"
else
  echo -e "${YELLOW}${BOLD}  Some services need attention.${NC}"
  echo -e "${DIM}  Inspect logs: docker logs <container-name>${NC}"
fi

echo ""
echo -e "${DIM}  API (internal):    http://acas-backend:8000/api/health${NC}"
echo -e "${DIM}  External access:   via Cloudflare Tunnel (acas-cloudflared)${NC}"
echo -e "${DIM}  Stop all:          docker compose -f docker-compose.prod.yml down${NC}"
echo -e "${DIM}  Follow logs:       docker compose -f docker-compose.prod.yml logs -f backend${NC}"
if [[ $WITH_MONITORING -eq 1 ]]; then
  echo -e "${DIM}  Prometheus:        http://localhost:9090  (tunnel or SSH port-forward)${NC}"
  echo -e "${DIM}  Grafana:           http://localhost:3000  (tunnel or SSH port-forward)${NC}"
fi
echo ""
