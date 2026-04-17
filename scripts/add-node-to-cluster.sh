#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ACAS  —  Add Node to Cluster                                              ║
# ║  For machines that already have Docker + CUDA + nvidia-container-toolkit   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Usage:
#   bash scripts/add-node-to-cluster.sh \
#     --control-plane https://acas-cp.your-workers.dev \
#     --node-name     acas-node-3 \
#     --location      "Chennai Edge" \
#     --connectivity  cloudflare_tunnel \
#     --tunnel-token  <CF_TUNNEL_TOKEN>
#
# Assumes:
#   • Docker CE installed and running
#   • NVIDIA driver ≥ 525 installed (nvidia-smi works)
#   • NVIDIA Container Toolkit installed (docker info shows nvidia runtime)
#   • Either: running from ACAS project root, OR --source-url set
#
# Options:
#   --control-plane URL      (required) Control Plane Workers URL
#   --node-name     NAME     (required) Human-readable node name
#   --location      LOC      Physical/logical location (default: hostname)
#   --connectivity  MODE     cloudflare_tunnel | public_ip (default: cloudflare_tunnel)
#   --tunnel-token  TOKEN    CF tunnel token (for cloudflare_tunnel mode)
#   --hostname      FQDN     Public FQDN (for public_ip mode, TLS must already be configured)
#   --gpu-device    ID       CUDA device index (default: 0)
#   --project-dir   DIR      ACAS project directory (default: current dir or /opt/acas)
#   --source-url    URL      Fallback URL to download ACAS files from
#   --skip-models            Skip model download check
#   --force-new-secrets      Regenerate .env.prod secrets (destructive for running instances)
#   --dry-run                Show what would be done without executing

set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; NC='\033[0m'

log()  { echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} $*"; }
ok()   { echo -e "${GREEN}[$(date +%H:%M:%S)] ✓${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] ⚠${NC} $*"; }
die()  { echo -e "${RED}[$(date +%H:%M:%S)] ✗${NC} $*" >&2; exit 1; }
step() {
  echo -e "\n${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo -e "${BOLD}${CYAN}  $*${NC}"
  echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# ── Argument parsing ──────────────────────────────────────────────────────────
CONTROL_PLANE_URL=""; NODE_NAME=""; LOCATION=""
CONNECTIVITY="cloudflare_tunnel"; TUNNEL_TOKEN=""; PUBLIC_HOSTNAME=""
GPU_DEVICE="0"; PROJECT_DIR=""; SOURCE_URL=""
SKIP_MODELS=0; FORCE_SECRETS=0; DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --control-plane)     CONTROL_PLANE_URL="$2"; shift 2 ;;
    --node-name)         NODE_NAME="$2";         shift 2 ;;
    --location)          LOCATION="$2";          shift 2 ;;
    --connectivity)      CONNECTIVITY="$2";      shift 2 ;;
    --tunnel-token)      TUNNEL_TOKEN="$2";      shift 2 ;;
    --hostname)          PUBLIC_HOSTNAME="$2";   shift 2 ;;
    --gpu-device)        GPU_DEVICE="$2";        shift 2 ;;
    --project-dir)       PROJECT_DIR="$2";       shift 2 ;;
    --source-url)        SOURCE_URL="$2";        shift 2 ;;
    --skip-models)       SKIP_MODELS=1;          shift   ;;
    --force-new-secrets) FORCE_SECRETS=1;        shift   ;;
    --dry-run)           DRY_RUN=1;              shift   ;;
    --help|-h)
      sed -n '/^# Usage/,/^#$/p' "$0" | sed 's/^# \?//'
      exit 0 ;;
    *) die "Unknown argument: $1. Run with --help." ;;
  esac
done

# Resolve project directory
if [[ -z "$PROJECT_DIR" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [[ -f "${SCRIPT_DIR}/../docker-compose.prod.yml" ]]; then
    PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
  elif [[ -f "/opt/acas/docker-compose.prod.yml" ]]; then
    PROJECT_DIR="/opt/acas"
  else
    PROJECT_DIR="$(pwd)"
  fi
fi
SOURCE_URL="${SOURCE_URL:-${CONTROL_PLANE_URL}/install}"
LOCATION="${LOCATION:-$(hostname -f 2>/dev/null || hostname)}"

ACAS_CONFIG_DIR="/etc/acas"
ACAS_KEY_DIR="/etc/acas/keys"
ACAS_NODE_ID_FILE="/etc/acas/node.id"

# ─────────────────────────────────────────────────────────────────────────────
banner() {
  echo -e "${BOLD}${CYAN}"
  echo "  ╔─────────────────────────────────────────╗"
  echo "  │     ACAS — Add Node to Cluster          │"
  echo "  ╚─────────────────────────────────────────╝"
  echo -e "${NC}"
  echo -e "  Project dir: ${DIM}${PROJECT_DIR}${NC}"
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1  Verify prerequisites
# ─────────────────────────────────────────────────────────────────────────────
verify_prerequisites() {
  step "1/7  Verifying prerequisites"

  # Required arguments
  [[ -n "$CONTROL_PLANE_URL" ]] || die "--control-plane is required"
  [[ -n "$NODE_NAME"         ]] || die "--node-name is required"
  [[ "$CONNECTIVITY" =~ ^(cloudflare_tunnel|public_ip)$ ]] || \
    die "--connectivity must be cloudflare_tunnel or public_ip"

  # Root check (needed for /etc/acas writes)
  [[ $EUID -eq 0 ]] || die "Run as root: sudo bash $0 ..."

  # Docker
  command -v docker &>/dev/null || die "Docker not found. Install Docker CE first."
  systemctl is-active docker &>/dev/null  || systemctl start docker
  docker info &>/dev/null                 || die "Docker is not running."
  ok "Docker: $(docker --version | cut -d' ' -f3 | tr -d ',')"

  # NVIDIA driver
  command -v nvidia-smi &>/dev/null || die "nvidia-smi not found. Install NVIDIA driver ≥525 first."
  local gpu_info
  gpu_info=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)
  ok "GPU: ${gpu_info}"

  # NVIDIA container runtime
  if ! docker info 2>/dev/null | grep -q "nvidia"; then
    warn "NVIDIA container runtime not found in docker info."
    log "Attempting to configure automatically…"
    if command -v nvidia-ctk &>/dev/null; then
      nvidia-ctk runtime configure --runtime=docker && systemctl restart docker
      ok "NVIDIA Container Toolkit configured"
    else
      die "nvidia-ctk not found. Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/"
    fi
  else
    ok "NVIDIA container runtime: present"
  fi

  # GPU container smoke test
  log "Running GPU container smoke test…"
  if docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
       nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L &>/dev/null; then
    ok "GPU accessible inside containers"
  else
    warn "GPU smoke test failed — check container runtime configuration"
  fi

  # CUDA version
  if command -v nvcc &>/dev/null; then
    local cuda_ver
    cuda_ver=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $5}' | tr -d ',')
    local major minor
    major="${cuda_ver%%.*}"; minor="${cuda_ver#*.}"; minor="${minor%%.*}"
    if [[ $major -gt 12 ]] || { [[ $major -eq 12 ]] && [[ $minor -ge 4 ]]; }; then
      ok "CUDA: ${cuda_ver} (≥12.4 required)"
    else
      warn "CUDA ${cuda_ver} detected — ACAS requires CUDA 12.4+. TensorRT may not compile."
    fi
  else
    warn "nvcc not found — CUDA toolkit not installed. TensorRT engine compilation unavailable."
  fi

  # Disk space
  local free_gb
  free_gb=$(df -BG "$PROJECT_DIR" 2>/dev/null | awk 'NR==2 {gsub("G",""); print $4}' || echo "?")
  if [[ "$free_gb" != "?" ]] && [[ $free_gb -lt 40 ]]; then
    warn "Free disk in ${PROJECT_DIR}: ${free_gb}GB (recommend ≥40GB)"
  else
    ok "Disk: ${free_gb}GB free in ${PROJECT_DIR}"
  fi

  ok "All prerequisites satisfied"
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2  Node identity
# ─────────────────────────────────────────────────────────────────────────────
setup_node_identity() {
  step "2/7  Node identity"

  mkdir -p "$ACAS_CONFIG_DIR" "$ACAS_KEY_DIR"
  chmod 700 "$ACAS_CONFIG_DIR" "$ACAS_KEY_DIR"

  if [[ -f "$ACAS_NODE_ID_FILE" ]]; then
    NODE_ID=$(cat "$ACAS_NODE_ID_FILE")
    ok "Reusing existing node ID: ${NODE_ID}"
  else
    NODE_ID=$(python3 -c "import uuid; print(uuid.uuid4())" 2>/dev/null \
              || cat /proc/sys/kernel/random/uuid)
    [[ $DRY_RUN -eq 0 ]] && {
      echo "$NODE_ID" > "$ACAS_NODE_ID_FILE"
      chmod 600 "$ACAS_NODE_ID_FILE"
    }
    ok "Generated node ID: ${NODE_ID}"
  fi
  export NODE_ID

  # JWT RSA key pair
  local priv="${ACAS_KEY_DIR}/jwt_private.pem"
  local pub="${ACAS_KEY_DIR}/jwt_public.pem"
  if [[ -f "$priv" && -f "$pub" ]]; then
    ok "JWT RSA key pair already exists"
  elif [[ $DRY_RUN -eq 0 ]]; then
    log "Generating RSA-4096 JWT key pair…"
    openssl genrsa -out "$priv" 4096 2>/dev/null
    openssl rsa -in "$priv" -pubout -out "$pub" 2>/dev/null
    chmod 600 "$priv"; chmod 644 "$pub"
    ok "JWT key pair generated"
  else
    warn "[dry-run] Would generate JWT key pair"
  fi
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3  Connectivity configuration
# ─────────────────────────────────────────────────────────────────────────────
setup_connectivity() {
  step "3/7  Connectivity (${CONNECTIVITY})"

  CLOUDFLARE_TUNNEL_TOKEN=""
  API_ENDPOINT=""

  if [[ "$CONNECTIVITY" == "cloudflare_tunnel" ]]; then
    if [[ -z "$TUNNEL_TOKEN" ]]; then
      echo -e "${YELLOW}"
      echo "  Paste your Cloudflare tunnel token:"
      echo "  (Get from https://one.dash.cloudflare.com → Access → Tunnels)"
      echo -n "  Token: "
      echo -e "${NC}"
      [[ $DRY_RUN -eq 0 ]] && read -r -s TUNNEL_TOKEN && echo ""
    fi
    [[ -n "$TUNNEL_TOKEN" ]] || die "Tunnel token required for cloudflare_tunnel mode"

    # Install cloudflared if not present (needed for the Docker image)
    if ! docker image inspect cloudflare/cloudflared:latest &>/dev/null; then
      log "Pulling cloudflared Docker image…"
      [[ $DRY_RUN -eq 0 ]] && docker pull --quiet cloudflare/cloudflared:latest
    fi

    CLOUDFLARE_TUNNEL_TOKEN="$TUNNEL_TOKEN"
    API_ENDPOINT="(routed via Cloudflare Tunnel)"
    ok "Cloudflare tunnel token set"

  else  # public_ip
    [[ -n "$PUBLIC_HOSTNAME" ]] || die "--hostname required for public_ip connectivity"
    API_ENDPOINT="https://${PUBLIC_HOSTNAME}"
    # TLS certs must already exist (managed externally or via nginx + certbot)
    if [[ ! -f "/etc/letsencrypt/live/${PUBLIC_HOSTNAME}/fullchain.pem" ]]; then
      warn "TLS cert not found at /etc/letsencrypt/live/${PUBLIC_HOSTNAME}/"
      warn "Ensure certbot has issued a certificate, or use cloudflare_tunnel mode."
    else
      ok "TLS cert found for ${PUBLIC_HOSTNAME}"
    fi
    ok "Public IP endpoint: ${API_ENDPOINT}"
  fi

  export CLOUDFLARE_TUNNEL_TOKEN API_ENDPOINT
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4  Environment file
# ─────────────────────────────────────────────────────────────────────────────
write_env_file() {
  step "4/7  Environment configuration"

  local env_file="${PROJECT_DIR}/.env.prod"

  if [[ -f "$env_file" && $FORCE_SECRETS -eq 0 ]]; then
    ok ".env.prod already exists — loading existing secrets"
    set -a; source "$env_file"; set +a
    # Patch the fields that may have changed for this node
    sed -i "s|^NODE_ID=.*|NODE_ID=${NODE_ID}|"              "$env_file"
    sed -i "s|^NODE_NAME=.*|NODE_NAME=${NODE_NAME}|"         "$env_file"
    sed -i "s|^CLOUDFLARE_TUNNEL_TOKEN=.*|CLOUDFLARE_TUNNEL_TOKEN=${CLOUDFLARE_TUNNEL_TOKEN}|" "$env_file"
    sed -i "s|^CONTROL_PLANE_URL=.*|CONTROL_PLANE_URL=${CONTROL_PLANE_URL}|" "$env_file"
    sed -i "s|^GPU_DEVICE_ID=.*|GPU_DEVICE_ID=${GPU_DEVICE}|" "$env_file"
    ok ".env.prod updated with new node identity fields"
    return
  fi

  log "Writing fresh .env.prod…"
  local pg_pass ts_pass minio_pass kafka_id
  pg_pass="$(openssl rand -base64 42 | tr -dc 'a-zA-Z0-9' | head -c 40)"
  ts_pass="$(openssl rand -base64 42 | tr -dc 'a-zA-Z0-9' | head -c 40)"
  minio_pass="$(openssl rand -base64 42 | tr -dc 'a-zA-Z0-9' | head -c 40)"
  kafka_id="$(openssl rand -base64 16 | tr '+/' '-_' | tr -dc 'a-zA-Z0-9_-' | head -c 22)"

  [[ $DRY_RUN -eq 0 ]] && cat > "$env_file" <<EOF
# ACAS node: ${NODE_NAME}
# Node ID:   ${NODE_ID}
# Generated: $(date -u +"%Y-%m-%dT%H:%M:%SZ")

POSTGRES_PASSWORD=${pg_pass}
TIMESCALE_PASSWORD=${ts_pass}

MINIO_ROOT_USER=acasminio
MINIO_ROOT_PASSWORD=${minio_pass}

JWT_SECRET_KEY=${ACAS_KEY_DIR}/jwt_private.pem
JWT_ALGORITHM=RS256

KAFKA_CLUSTER_ID=${kafka_id}

CLOUDFLARE_TUNNEL_TOKEN=${CLOUDFLARE_TUNNEL_TOKEN:-}

NODE_ID=${NODE_ID}
NODE_NAME=${NODE_NAME}

GPU_DEVICE_ID=${GPU_DEVICE}
NVIDIA_VISIBLE_DEVICES=all

CONTROL_PLANE_URL=${CONTROL_PLANE_URL}

BACKUP_RETAIN_DAYS=14

GRAFANA_USER=admin
GRAFANA_PASSWORD=$(openssl rand -base64 18 | tr -dc 'a-zA-Z0-9' | head -c 24)
GRAFANA_ROOT_URL=http://localhost:3000
EOF
  [[ $DRY_RUN -eq 0 ]] && chmod 600 "$env_file"
  ok ".env.prod written (600 permissions)"
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5  Model files
# ─────────────────────────────────────────────────────────────────────────────
check_models() {
  step "5/7  AI model files"
  [[ $SKIP_MODELS -eq 1 ]] && { warn "Skipping (--skip-models)"; return; }

  local models_dir="${PROJECT_DIR}/models"
  mkdir -p "$models_dir"

  local required=("yolov8l.onnx" "retinaface_resnet50.onnx" "adaface_ir101_webface12m.onnx" "minifasnetv2.onnx")
  local missing=()
  for m in "${required[@]}"; do
    [[ -f "${models_dir}/${m}" ]] || missing+=("$m")
  done

  if [[ ${#missing[@]} -eq 0 ]]; then
    ok "All model files present"
    return
  fi

  warn "Missing model files: ${missing[*]}"
  log "Attempting download via acas-backend:gpu container…"

  if [[ $DRY_RUN -eq 0 ]]; then
    if docker image inspect acas-backend:gpu &>/dev/null; then
      docker run --rm \
        -v "${models_dir}:/models" \
        acas-backend:gpu \
        python /app/scripts/download_models.py --output-dir /models 2>&1 | \
        grep -v "^$" || true
      ok "Model download attempted — verify files in ${models_dir}"
    else
      log "Building acas-backend:gpu to enable model download…"
      if [[ -f "${PROJECT_DIR}/backend/Dockerfile.gpu" ]]; then
        docker build -t acas-backend:gpu \
          -f "${PROJECT_DIR}/backend/Dockerfile.gpu" \
          "${PROJECT_DIR}/backend" 2>&1 | grep -E "Step|=>" | head -40 || true
        docker run --rm \
          -v "${models_dir}:/models" \
          acas-backend:gpu \
          python /app/scripts/download_models.py --output-dir /models 2>&1 || \
          warn "Download had errors — check ${models_dir}"
      else
        warn "Dockerfile.gpu not found. Manually copy model files to ${models_dir}"
        warn "Required: ${required[*]}"
      fi
    fi
  else
    warn "[dry-run] Would download missing models"
  fi

  local trt_count
  trt_count=$(find "$models_dir" -name "*.engine" 2>/dev/null | wc -l || echo 0)
  if [[ $trt_count -eq 0 ]]; then
    warn "No .engine files — TensorRT engines compile on first backend start (10-15 min)"
  else
    ok "TensorRT engines: ${trt_count} found"
  fi
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6  Start services
# ─────────────────────────────────────────────────────────────────────────────
start_services() {
  step "6/7  Starting services"
  [[ $DRY_RUN -eq 1 ]] && { warn "[dry-run] Would run start.sh"; return; }

  local start_script="${PROJECT_DIR}/scripts/start.sh"
  if [[ ! -f "$start_script" ]]; then
    warn "start.sh not found at ${start_script}"
    log "Falling back to direct docker compose up…"
    docker compose \
      -f "${PROJECT_DIR}/docker-compose.prod.yml" \
      --env-file "${PROJECT_DIR}/.env.prod" \
      up -d
  else
    chmod +x "$start_script"
    bash "$start_script" --skip-pull 2>&1 | tee /tmp/acas-start.log
    ok "start.sh completed"
  fi
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7  Verify registration with control plane
# ─────────────────────────────────────────────────────────────────────────────
verify_registration() {
  step "7/7  Verifying control plane registration"
  [[ $DRY_RUN -eq 1 ]] && { warn "[dry-run] Would verify registration"; return; }

  local max_wait=240 elapsed=0 interval=8
  log "Polling ${CONTROL_PLANE_URL}/api/nodes/${NODE_ID} (max ${max_wait}s)…"

  while [[ $elapsed -lt $max_wait ]]; do
    local code
    code=$(curl -s -o /dev/null -w "%{http_code}" \
      "${CONTROL_PLANE_URL}/api/nodes/${NODE_ID}" \
      --connect-timeout 5 --max-time 8 2>/dev/null || echo "000")

    case "$code" in
      200)
        echo ""
        ok "Node registered with control plane ✓"
        # Fetch and show node details
        local node_json
        node_json=$(curl -s "${CONTROL_PLANE_URL}/api/nodes/${NODE_ID}" 2>/dev/null || echo "{}")
        local cp_status
        cp_status=$(echo "$node_json" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','?'))" 2>/dev/null || echo "?")
        ok "Control plane status: ${cp_status}"
        return 0 ;;
      401|403) warn "Auth error (${code}) — verify control plane API key configuration"; break ;;
      404) : ;; # not yet registered — keep polling
      000) warn "Control plane unreachable (${code}) — check CONTROL_PLANE_URL" ;;
    esac
    sleep $interval; elapsed=$((elapsed + interval)); printf "."
  done

  echo ""
  warn "Registration not confirmed after ${max_wait}s."
  warn "The backend registers itself on startup — it may still be initialising."
  warn "Check: curl ${CONTROL_PLANE_URL}/api/nodes/${NODE_ID}"
  warn "Or:    docker logs acas-backend --tail 50"
}

# ─────────────────────────────────────────────────────────────────────────────
# Print final status
# ─────────────────────────────────────────────────────────────────────────────
print_status() {
  echo ""
  echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
  echo -e "${BOLD}${CYAN}║         ACAS Node — Service Status                       ║${NC}"
  echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"
  printf "\n  %-30s %-12s\n" "Service" "Status"
  printf   "  %-30s %-12s\n" "──────────────────────────────" "────────────"

  local CONTAINERS=(
    "acas-backend:Backend"
    "acas-postgres:PostgreSQL"
    "acas-timescaledb:TimescaleDB"
    "acas-redis:Redis"
    "acas-minio:MinIO"
    "acas-kafka:Kafka"
    "acas-schema-registry:Schema Registry"
    "acas-cloudflared:Cloudflare Tunnel"
  )
  local all_ok=1
  for entry in "${CONTAINERS[@]}"; do
    local cname="${entry%%:*}" label="${entry##*:}"
    local st health
    st=$(docker inspect --format='{{.State.Status}}' "$cname" 2>/dev/null || echo "missing")
    health=$(docker inspect --format='{{if .State.Health}}{{.State.Health.Status}}{{else}}-{{end}}' "$cname" 2>/dev/null || echo "-")
    if [[ "$st" == "running" && ( "$health" == "healthy" || "$health" == "-" ) ]]; then
      printf "  ${GREEN}●${NC} %-28s ${GREEN}%-12s${NC}\n" "$label" "running"
    else
      printf "  ${RED}●${NC} %-28s ${RED}%-12s${NC}\n" "$label" "${st}/${health}"
      all_ok=0
    fi
  done

  echo ""
  echo -e "  ${DIM}Node ID:       ${NODE_ID}${NC}"
  echo -e "  ${DIM}Name:          ${NODE_NAME}${NC}"
  echo -e "  ${DIM}Location:      ${LOCATION}${NC}"
  echo -e "  ${DIM}Project dir:   ${PROJECT_DIR}${NC}"
  echo -e "  ${DIM}Control plane: ${CONTROL_PLANE_URL}${NC}"
  echo ""

  if [[ $all_ok -eq 1 ]]; then
    echo -e "${GREEN}${BOLD}  ✓ Node successfully added to cluster!${NC}"
    echo ""
    echo -e "${DIM}  Next steps:${NC}"
    echo -e "${DIM}    1. Assign cameras via Control Plane or ACAS dashboard (/cluster)${NC}"
    echo -e "${DIM}    2. Create superadmin (if first node):${NC}"
    echo -e "${DIM}       docker exec acas-backend python backend/scripts/create_superadmin.py${NC}"
    echo -e "${DIM}    3. Monitor: docker compose -f ${PROJECT_DIR}/docker-compose.prod.yml logs -f${NC}"
  else
    echo -e "${YELLOW}${BOLD}  ⚠ Some services need attention.${NC}"
    echo -e "${DIM}  docker compose -f ${PROJECT_DIR}/docker-compose.prod.yml logs${NC}"
  fi
  echo ""
}

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
main() {
  banner
  verify_prerequisites
  setup_node_identity
  setup_connectivity
  write_env_file
  check_models
  start_services
  verify_registration
  print_status
}

main "$@"
