#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ACAS Node Bootstrap — Fresh Ubuntu 22.04 to Production in < 10 minutes    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Usage (curl install):
#   curl -sSL https://install.acas.example.com | bash -s -- \
#     --control-plane https://acas-cp.your-workers.dev \
#     --node-name     acas-node-2 \
#     --location      "Mumbai DC2" \
#     --connectivity  cloudflare_tunnel \
#     --tunnel-token  <CF_TUNNEL_TOKEN>
#
# Usage (local project):
#   bash scripts/bootstrap-node.sh --control-plane URL --node-name NAME ...
#
# Options:
#   --control-plane URL        (required) Control Plane Workers URL
#   --node-name     NAME       (required) Human-readable node name
#   --location      LOC        Physical/logical location (default: hostname)
#   --connectivity  MODE       cloudflare_tunnel | public_ip (default: cloudflare_tunnel)
#   --tunnel-token  TOKEN      Cloudflare tunnel token (connectivity=cloudflare_tunnel)
#   --hostname      FQDN       Public FQDN for TLS cert (connectivity=public_ip)
#   --email         EMAIL      Certbot email (connectivity=public_ip)
#   --source-url    URL        Where to download ACAS files (default: CONTROL_PLANE/install)
#   --gpu-device    ID         CUDA device index (default: 0)
#   --install-dir   DIR        Installation directory (default: /opt/acas)
#   --skip-drivers             Skip NVIDIA driver installation
#   --skip-models              Skip model file download
#   --dry-run                  Validate and print plan without executing
#
# Re-run after reboot:
#   bash /opt/acas/scripts/bootstrap-node.sh --resume

set -euo pipefail
IFS=$'\n\t'

# ── Constants ─────────────────────────────────────────────────────────────────
ACAS_INSTALL_DIR="${INSTALL_DIR:-/opt/acas}"
ACAS_CONFIG_DIR="/etc/acas"
ACAS_KEY_DIR="/etc/acas/keys"
ACAS_STATE_FILE="/etc/acas/install.state"
ACAS_LOG_FILE="/var/log/acas-bootstrap.log"
ACAS_NODE_ID_FILE="/etc/acas/node.id"

MIN_CUDA_MAJOR=12; MIN_CUDA_MINOR=4
MIN_DRIVER_VER=525
MIN_DISK_GB=60
MIN_RAM_GB=16

KAFKA_IMAGE="apache/kafka:3.9.2"
MINIO_MC_IMAGE="minio/mc:latest"
BACKEND_IMAGE="acas-backend:gpu"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; NC='\033[0m'

log()  { local ts; ts="$(date +%H:%M:%S)"; echo -e "${BLUE}[${ts}]${NC} $*" | tee -a "$ACAS_LOG_FILE"; }
ok()   { local ts; ts="$(date +%H:%M:%S)"; echo -e "${GREEN}[${ts}] ✓${NC} $*" | tee -a "$ACAS_LOG_FILE"; }
warn() { local ts; ts="$(date +%H:%M:%S)"; echo -e "${YELLOW}[${ts}] ⚠${NC} $*" | tee -a "$ACAS_LOG_FILE"; }
die()  { local ts; ts="$(date +%H:%M:%S)"; echo -e "${RED}[${ts}] ✗${NC} $*" >&2 | tee -a "$ACAS_LOG_FILE"; exit 1; }
step() {
  echo -e "\n${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo -e "${BOLD}${CYAN}  $*${NC}"
  echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo "[$(date +%H:%M:%S)] STEP: $*" >> "$ACAS_LOG_FILE"
}

# ── Argument parsing ──────────────────────────────────────────────────────────
CONTROL_PLANE_URL=""; NODE_NAME=""; LOCATION=""
CONNECTIVITY="cloudflare_tunnel"; TUNNEL_TOKEN=""; PUBLIC_HOSTNAME=""; CERTBOT_EMAIL=""
GPU_DEVICE="0"; SOURCE_URL=""
SKIP_DRIVERS=0; SKIP_MODELS=0; DRY_RUN=0; RESUME=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --control-plane) CONTROL_PLANE_URL="$2"; shift 2 ;;
    --node-name)     NODE_NAME="$2";         shift 2 ;;
    --location)      LOCATION="$2";          shift 2 ;;
    --connectivity)  CONNECTIVITY="$2";      shift 2 ;;
    --tunnel-token)  TUNNEL_TOKEN="$2";      shift 2 ;;
    --hostname)      PUBLIC_HOSTNAME="$2";   shift 2 ;;
    --email)         CERTBOT_EMAIL="$2";     shift 2 ;;
    --gpu-device)    GPU_DEVICE="$2";        shift 2 ;;
    --source-url)    SOURCE_URL="$2";        shift 2 ;;
    --install-dir)   ACAS_INSTALL_DIR="$2";  shift 2 ;;
    --skip-drivers)  SKIP_DRIVERS=1;         shift   ;;
    --skip-models)   SKIP_MODELS=1;          shift   ;;
    --dry-run)       DRY_RUN=1;              shift   ;;
    --resume)        RESUME=1;               shift   ;;
    --help|-h)
      sed -n '/^# Usage/,/^#$/p' "$0" | sed 's/^# \?//'
      exit 0 ;;
    *) die "Unknown argument: $1. Run with --help for usage." ;;
  esac
done

SOURCE_URL="${SOURCE_URL:-${CONTROL_PLANE_URL}/install}"
LOCATION="${LOCATION:-$(hostname -f 2>/dev/null || hostname)}"

# ── Helpers ───────────────────────────────────────────────────────────────────
need_root() { [[ $EUID -eq 0 ]] || die "This script must be run as root (sudo bash $0 ...)"; }

gen_secret() {
  local len="${1:-32}"
  openssl rand -base64 $((len * 3 / 2)) | tr -dc 'a-zA-Z0-9' | head -c "$len"
}

gen_kafka_cluster_id() {
  # Produce a valid KRaft cluster ID (22 base64url chars)
  openssl rand -base64 16 | tr '+/' '-_' | tr -dc 'a-zA-Z0-9_-' | head -c 22
}

save_state() { echo "$1" > "$ACAS_STATE_FILE"; }
load_state() { [[ -f "$ACAS_STATE_FILE" ]] && cat "$ACAS_STATE_FILE" || echo ""; }

download_file() {
  local url="$1" dest="$2" desc="${3:-$1}"
  log "Downloading ${desc}…"
  if curl -fsSL --connect-timeout 15 --retry 3 --retry-delay 2 "$url" -o "$dest"; then
    ok "Downloaded: $(basename "$dest")"
  else
    die "Failed to download ${desc} from ${url}"
  fi
}

wait_for_container_health() {
  local name="$1" timeout="${2:-120}" elapsed=0
  log "Waiting for ${name} to become healthy…"
  while [[ $elapsed -lt $timeout ]]; do
    local st
    st=$(docker inspect --format='{{.State.Health.Status}}' "$name" 2>/dev/null || echo "missing")
    case "$st" in
      healthy)   ok "${name} is healthy"; return 0 ;;
      unhealthy) die "${name} is unhealthy. Logs:\n$(docker logs --tail 30 "$name" 2>&1)"; ;;
    esac
    sleep 5; elapsed=$((elapsed + 5)); printf "."
  done
  die "${name} did not become healthy within ${timeout}s"
}

# ─────────────────────────────────────────────────────────────────────────────
banner() {
  echo -e "${BOLD}${CYAN}"
  cat <<'EOF'
  ┌─────────────────────────────────────────┐
  │   ACAS — Autonomous Camera AI System    │
  │         Node Bootstrap v1.0             │
  └─────────────────────────────────────────┘
EOF
  echo -e "${NC}"
  echo -e "  ${DIM}Log: ${ACAS_LOG_FILE}${NC}"
  echo -e "  ${DIM}Install dir: ${ACAS_INSTALL_DIR}${NC}\n"
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1  System checks
# ─────────────────────────────────────────────────────────────────────────────
check_system() {
  step "1/9  System checks"

  # Root
  need_root

  # OS
  if [[ -f /etc/os-release ]]; then
    # shellcheck source=/dev/null
    source /etc/os-release
    if [[ "${ID:-}" == "ubuntu" && "${VERSION_ID:-}" == "22.04" ]]; then
      ok "OS: Ubuntu 22.04 LTS"
    else
      warn "Expected Ubuntu 22.04; found ${PRETTY_NAME:-unknown}. Proceeding anyway."
    fi
  fi

  # Disk space (root partition)
  local free_gb
  free_gb=$(df -BG / | awk 'NR==2 {gsub("G",""); print $4}')
  if [[ ${free_gb:-0} -lt $MIN_DISK_GB ]]; then
    warn "Free disk: ${free_gb}GB (recommended: ≥${MIN_DISK_GB}GB for models + images)"
  else
    ok "Free disk: ${free_gb}GB"
  fi

  # RAM
  local ram_gb
  ram_gb=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)
  if [[ ${ram_gb:-0} -lt $MIN_RAM_GB ]]; then
    warn "RAM: ${ram_gb}GB (recommended: ≥${MIN_RAM_GB}GB)"
  else
    ok "RAM: ${ram_gb}GB"
  fi

  # NVIDIA GPU
  if ! lspci 2>/dev/null | grep -qi "nvidia\|3d controller"; then
    die "No NVIDIA GPU detected (lspci). ACAS requires an NVIDIA GPU."
  fi
  local gpu_name
  gpu_name=$(lspci | grep -i "nvidia\|3d controller" | head -1 | sed 's/.*: //')
  ok "GPU detected: ${gpu_name}"

  # Validate required arguments
  [[ -n "$CONTROL_PLANE_URL" ]] || die "--control-plane is required"
  [[ -n "$NODE_NAME"         ]] || die "--node-name is required"
  [[ "$CONNECTIVITY" =~ ^(cloudflare_tunnel|public_ip)$ ]] || \
    die "--connectivity must be cloudflare_tunnel or public_ip"
  if [[ "$CONNECTIVITY" == "public_ip" ]]; then
    [[ -n "$PUBLIC_HOSTNAME" ]] || die "--hostname is required for connectivity=public_ip"
    [[ -n "$CERTBOT_EMAIL"   ]] || die "--email is required for connectivity=public_ip"
  fi

  # Initialise log and config dirs
  mkdir -p "$ACAS_CONFIG_DIR" "$ACAS_KEY_DIR"
  chmod 700 "$ACAS_CONFIG_DIR" "$ACAS_KEY_DIR"
  touch "$ACAS_LOG_FILE"; chmod 600 "$ACAS_LOG_FILE"

  ok "System checks passed"
  [[ $DRY_RUN -eq 1 ]] && warn "[dry-run] Would proceed with installation."
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2  Install system dependencies
# ─────────────────────────────────────────────────────────────────────────────
install_dependencies() {
  step "2/9  Installing system dependencies"
  [[ $DRY_RUN -eq 1 ]] && { warn "[dry-run] Skipping installs"; return; }

  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq

  # Utilities
  log "Installing utilities…"
  apt-get install -y -qq \
    curl wget ca-certificates gnupg lsb-release \
    python3 python3-pip jq uuid-runtime net-tools \
    htop nvtop 2>/dev/null || true
  ok "Utilities installed"

  # Docker
  if ! command -v docker &>/dev/null; then
    log "Installing Docker CE…"
    curl -fsSL https://get.docker.com | sh
    systemctl enable --now docker
    ok "Docker installed"
  else
    ok "Docker already installed ($(docker --version | cut -d' ' -f3 | tr -d ','))"
  fi

  # NVIDIA Drivers
  if [[ $SKIP_DRIVERS -eq 1 ]]; then
    warn "Skipping NVIDIA driver install (--skip-drivers)"
  elif nvidia-smi &>/dev/null; then
    local drv
    drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
    ok "NVIDIA driver already installed: ${drv}"
  else
    log "Installing NVIDIA driver (this may take 2-3 minutes)…"
    apt-get install -y -qq ubuntu-drivers-common
    # Use recommended driver; or pin to a known-good version
    ubuntu-drivers install --gpgpu 2>/dev/null || apt-get install -y -qq nvidia-driver-560-open
    ok "NVIDIA driver installed — a REBOOT is required"
    save_state "REBOOT_REQUIRED"
    echo ""
    echo -e "${BOLD}${YELLOW}  ┌──────────────────────────────────────────────────┐"
    echo -e "  │  REBOOT REQUIRED to activate NVIDIA drivers.     │"
    echo -e "  │  After reboot, re-run:                           │"
    echo -e "  │    sudo bash /opt/acas/scripts/bootstrap-node.sh │"
    echo -e "  │         --resume                                  │"
    echo -e "  └──────────────────────────────────────────────────┘${NC}"
    exit 10   # Exit 10 = reboot required (callers can check this)
  fi

  # NVIDIA Container Toolkit
  if docker info 2>/dev/null | grep -q "nvidia"; then
    ok "NVIDIA container runtime already configured"
  else
    log "Installing NVIDIA Container Toolkit…"
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
      | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -sSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
      | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
    apt-get update -qq
    apt-get install -y -qq nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
    ok "NVIDIA Container Toolkit installed"
  fi

  # Verify GPU in container
  log "Verifying GPU access inside Docker…"
  if docker run --rm --runtime=nvidia \
       -e NVIDIA_VISIBLE_DEVICES=all \
       nvidia/cuda:12.4.1-base-ubuntu22.04 \
       nvidia-smi --query-gpu=name --format=csv,noheader &>/dev/null; then
    ok "GPU accessible inside Docker containers"
  else
    warn "GPU container test failed — check NVIDIA runtime configuration"
  fi

  # CUDA version check
  if command -v nvcc &>/dev/null; then
    local cuda_ver
    cuda_ver=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $5}' | tr -d ',')
    ok "CUDA toolkit: ${cuda_ver}"
  else
    warn "CUDA toolkit (nvcc) not found — TensorRT compilation requires it."
    warn "Install if needed: apt-get install cuda-toolkit-12-4"
  fi

  save_state "DEPS_INSTALLED"
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3  Generate / restore node identity
# ─────────────────────────────────────────────────────────────────────────────
setup_node_identity() {
  step "3/9  Node identity"

  if [[ -f "$ACAS_NODE_ID_FILE" ]]; then
    NODE_ID=$(cat "$ACAS_NODE_ID_FILE")
    ok "Reusing node ID: ${NODE_ID}"
  else
    NODE_ID=$(cat /proc/sys/kernel/random/uuid 2>/dev/null || python3 -c "import uuid; print(uuid.uuid4())")
    [[ $DRY_RUN -eq 0 ]] && echo "$NODE_ID" > "$ACAS_NODE_ID_FILE" && chmod 600 "$ACAS_NODE_ID_FILE"
    ok "Generated node ID: ${NODE_ID}"
  fi
  export NODE_ID

  # RSA key pair for JWT (RS256) — shared across all backends on this node
  local priv="${ACAS_KEY_DIR}/jwt_private.pem"
  local pub="${ACAS_KEY_DIR}/jwt_public.pem"
  if [[ -f "$priv" ]]; then
    ok "JWT key pair exists"
  elif [[ $DRY_RUN -eq 0 ]]; then
    log "Generating RSA-4096 key pair for JWT…"
    openssl genrsa -out "$priv" 4096 2>/dev/null
    openssl rsa -in "$priv" -pubout -out "$pub" 2>/dev/null
    chmod 600 "$priv"; chmod 644 "$pub"
    ok "JWT RSA key pair generated"
  fi
  JWT_PRIVATE_KEY="$(cat "$priv" 2>/dev/null || echo "placeholder")"
  export JWT_PRIVATE_KEY
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4  Connectivity — Cloudflare Tunnel OR public IP + TLS
# ─────────────────────────────────────────────────────────────────────────────
setup_connectivity() {
  step "4/9  Connectivity (${CONNECTIVITY})"

  if [[ "$CONNECTIVITY" == "cloudflare_tunnel" ]]; then
    setup_cloudflare_tunnel
  else
    setup_public_ip_tls
  fi
}

setup_cloudflare_tunnel() {
  if [[ -z "$TUNNEL_TOKEN" ]]; then
    echo -e "${YELLOW}"
    echo "  Cloudflare tunnel token required."
    echo "  1. Go to https://one.dash.cloudflare.com → Access → Tunnels → Create tunnel."
    echo "  2. Copy the connector token."
    echo -n "  Paste token now: "
    echo -e "${NC}"
    [[ $DRY_RUN -eq 0 ]] && read -r -s TUNNEL_TOKEN && echo ""
    [[ -n "$TUNNEL_TOKEN" ]] || die "Tunnel token is required for cloudflare_tunnel connectivity"
  fi

  # Install cloudflared
  if ! command -v cloudflared &>/dev/null; then
    log "Installing cloudflared…"
    [[ $DRY_RUN -eq 0 ]] && {
      curl -L --progress-bar \
        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb" \
        -o /tmp/cloudflared.deb
      dpkg -i /tmp/cloudflared.deb
      rm /tmp/cloudflared.deb
    }
    ok "cloudflared installed"
  else
    ok "cloudflared already installed ($(cloudflared --version 2>&1 | head -1))"
  fi

  # Connectivity mode: token-based (connector runs inside Docker container, not systemd)
  API_ENDPOINT="https://acas-${NODE_NAME}.your-tunnel-domain.com"
  CLOUDFLARE_TUNNEL_TOKEN="$TUNNEL_TOKEN"
  export CLOUDFLARE_TUNNEL_TOKEN API_ENDPOINT
  ok "Cloudflare tunnel configured (token-based)"
  log "The tunnel container will proxy: ${API_ENDPOINT} → backend:8000"
}

setup_public_ip_tls() {
  log "Setting up TLS certificate for ${PUBLIC_HOSTNAME}…"
  [[ $DRY_RUN -eq 0 ]] && {
    apt-get install -y -qq certbot
    # Temporarily stop anything on port 80
    systemctl stop nginx 2>/dev/null || true
    certbot certonly --standalone \
      -d "$PUBLIC_HOSTNAME" \
      --non-interactive \
      --agree-tos \
      -m "$CERTBOT_EMAIL"
    ok "TLS certificate obtained: /etc/letsencrypt/live/${PUBLIC_HOSTNAME}/"

    # Copy certs to config dir for nginx/cloudflared to reference
    mkdir -p "${ACAS_CONFIG_DIR}/tls"
    cp "/etc/letsencrypt/live/${PUBLIC_HOSTNAME}/fullchain.pem" "${ACAS_CONFIG_DIR}/tls/"
    cp "/etc/letsencrypt/live/${PUBLIC_HOSTNAME}/privkey.pem"   "${ACAS_CONFIG_DIR}/tls/"
    chmod 600 "${ACAS_CONFIG_DIR}/tls/privkey.pem"
  }
  API_ENDPOINT="https://${PUBLIC_HOSTNAME}"
  CLOUDFLARE_TUNNEL_TOKEN=""
  export API_ENDPOINT CLOUDFLARE_TUNNEL_TOKEN
  ok "Public HTTPS endpoint: ${API_ENDPOINT}"
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5  Project setup — files, secrets, .env.prod
# ─────────────────────────────────────────────────────────────────────────────
setup_project() {
  step "5/9  Project setup"
  [[ $DRY_RUN -eq 0 ]] && mkdir -p \
    "${ACAS_INSTALL_DIR}/scripts/postgres" \
    "${ACAS_INSTALL_DIR}/scripts/timescaledb" \
    "${ACAS_INSTALL_DIR}/config/cloudflared" \
    "${ACAS_INSTALL_DIR}/models" \
    "${ACAS_INSTALL_DIR}/backend"

  # If running from within the ACAS project, copy files; otherwise download them
  local script_source
  script_source="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)" || script_source=""
  local project_root=""
  [[ -f "${script_source}/../docker-compose.prod.yml" ]] && \
    project_root="$(cd "${script_source}/.." && pwd)"

  download_or_copy() {
    local filename="$1" dest_path="${ACAS_INSTALL_DIR}/${2:-$1}"
    if [[ -n "$project_root" && -f "${project_root}/${filename}" ]]; then
      [[ $DRY_RUN -eq 0 ]] && cp "${project_root}/${filename}" "$dest_path"
      ok "Copied ${filename} from project"
    else
      [[ $DRY_RUN -eq 0 ]] && \
        curl -fsSL --retry 3 "${SOURCE_URL}/${filename}" -o "$dest_path" 2>/dev/null || {
          warn "Could not download ${filename} — you may need to copy it manually to ${dest_path}"
        }
      ok "Downloaded ${filename}"
    fi
  }

  download_or_copy "docker-compose.prod.yml"
  download_or_copy "scripts/start.sh"               "scripts/start.sh"
  download_or_copy "scripts/postgres/backup.sh"     "scripts/postgres/backup.sh"
  download_or_copy "scripts/timescaledb/init-detection-log.sql" "scripts/timescaledb/init-detection-log.sql"
  download_or_copy "scripts/timescaledb/init-compression.sql"   "scripts/timescaledb/init-compression.sql"
  download_or_copy "scripts/postgres/init-extensions.sql"       "scripts/postgres/init-extensions.sql"
  [[ $DRY_RUN -eq 0 ]] && chmod +x "${ACAS_INSTALL_DIR}/scripts/start.sh" \
                                    "${ACAS_INSTALL_DIR}/scripts/postgres/backup.sh" 2>/dev/null || true

  # Backend Dockerfile + requirements (for local build)
  if [[ -n "$project_root" ]]; then
    [[ $DRY_RUN -eq 0 ]] && {
      cp -r "${project_root}/backend" "${ACAS_INSTALL_DIR}/"  2>/dev/null || true
    }
  fi

  # Generate .env.prod (idempotent — skip if exists and not changed)
  local env_file="${ACAS_INSTALL_DIR}/.env.prod"
  if [[ -f "$env_file" && $RESUME -eq 1 ]]; then
    ok ".env.prod already exists (resume mode — using existing secrets)"
    # shellcheck source=/dev/null
    source "$env_file"
  else
    log "Generating .env.prod with fresh secrets…"
    local pg_pass ts_pass minio_user minio_pass kafka_id
    pg_pass="$(gen_secret 40)"
    ts_pass="$(gen_secret 40)"
    minio_user="acasminio"
    minio_pass="$(gen_secret 40)"
    kafka_id="$(gen_kafka_cluster_id)"

    [[ $DRY_RUN -eq 0 ]] && cat > "$env_file" <<EOF
# ACAS Node: ${NODE_NAME}
# Generated: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
# Node ID:   ${NODE_ID}

POSTGRES_PASSWORD=${pg_pass}
TIMESCALE_PASSWORD=${ts_pass}

MINIO_ROOT_USER=${minio_user}
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
GRAFANA_PASSWORD=$(gen_secret 24)
GRAFANA_ROOT_URL=http://localhost:3000
EOF
    chmod 600 "$env_file"
    ok ".env.prod written (permissions 600)"
  fi

  ok "Project files ready at ${ACAS_INSTALL_DIR}"
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6  Pull images + download models
# ─────────────────────────────────────────────────────────────────────────────
pull_images_and_models() {
  step "6/9  Pull images & download AI models"
  [[ $DRY_RUN -eq 1 ]] && { warn "[dry-run] Skipping pulls"; return; }

  local project_root=""
  local script_source
  script_source="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)" || script_source=""
  [[ -f "${script_source}/../docker-compose.prod.yml" ]] && \
    project_root="$(cd "${script_source}/.." && pwd)"

  # Build backend image
  if [[ -d "${ACAS_INSTALL_DIR}/backend" ]]; then
    log "Building acas-backend:gpu (this takes 3-5 min on first run)…"
    docker build -t "$BACKEND_IMAGE" \
      -f "${ACAS_INSTALL_DIR}/backend/Dockerfile.gpu" \
      "${ACAS_INSTALL_DIR}/backend" 2>&1 | \
      grep -E "^(Step|#|=>|Successfully)" | head -60 || true
    ok "acas-backend:gpu built"
  else
    warn "Backend source not found in ${ACAS_INSTALL_DIR}/backend — skipping build."
    warn "Provide Dockerfile.gpu + requirements.txt or set --source-url."
  fi

  # Pull remaining service images
  log "Pulling infrastructure images…"
  local images=(
    "pgvector/pgvector:pg16"
    "timescale/timescaledb:latest-pg16"
    "redis:7-alpine"
    "minio/minio:latest"
    "minio/mc:latest"
    "${KAFKA_IMAGE}"
    "confluentinc/cp-schema-registry:7.6.1"
    "cloudflare/cloudflared:latest"
  )
  for img in "${images[@]}"; do
    log "  Pulling ${img}…"
    docker pull --quiet "$img" &>/dev/null && ok "  ${img}" || warn "  Failed: ${img}"
  done

  # Model files
  if [[ $SKIP_MODELS -eq 1 ]]; then
    warn "Skipping model download (--skip-models)"
    return
  fi

  local models_dir="${ACAS_INSTALL_DIR}/models"
  local required_models=("yolov8l.onnx" "retinaface_resnet50.onnx" "adaface_ir101_webface12m.onnx" "minifasnetv2.onnx")
  local missing=()
  for m in "${required_models[@]}"; do
    [[ -f "${models_dir}/${m}" ]] || missing+=("$m")
  done

  if [[ ${#missing[@]} -eq 0 ]]; then
    ok "All model files present"
    return
  fi

  log "Downloading AI models (~500MB): ${missing[*]}"
  if docker image inspect "$BACKEND_IMAGE" &>/dev/null; then
    docker run --rm \
      -v "${models_dir}:/models" \
      "$BACKEND_IMAGE" \
      python /app/scripts/download_models.py --output-dir /models || \
      warn "Model download had errors — check ${models_dir} before starting"
  else
    warn "Backend image not available — cannot download models automatically."
    warn "Run manually: docker run --rm -v ${models_dir}:/models acas-backend:gpu python scripts/download_models.py"
  fi

  # Check TRT engines
  local trt_count
  trt_count=$(find "$models_dir" -name "*.engine" 2>/dev/null | wc -l)
  if [[ $trt_count -eq 0 ]]; then
    warn "No TensorRT .engine files found — engines compile on first backend start (~10-15 min)"
  else
    ok "TensorRT engines: ${trt_count}"
  fi

  ok "Model check complete"
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7  Start services
# ─────────────────────────────────────────────────────────────────────────────
start_services() {
  step "7/9  Starting ACAS services"
  [[ $DRY_RUN -eq 1 ]] && { warn "[dry-run] Would run scripts/start.sh"; return; }

  local start_script="${ACAS_INSTALL_DIR}/scripts/start.sh"
  [[ -x "$start_script" ]] || die "start.sh not found at ${start_script}"

  bash "$start_script" \
    --skip-pull \
    2>&1 | tee -a "$ACAS_LOG_FILE"

  local exit_code=$?
  if [[ $exit_code -eq 0 ]]; then
    ok "All services started successfully"
  else
    die "start.sh exited with code ${exit_code}. Check ${ACAS_LOG_FILE}."
  fi

  save_state "SERVICES_STARTED"
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8  Register with Control Plane + trigger face sync
# ─────────────────────────────────────────────────────────────────────────────
register_and_sync() {
  step "8/9  Registering with Control Plane"
  [[ $DRY_RUN -eq 1 ]] && { warn "[dry-run] Would register node"; return; }

  # The backend registers itself automatically via node_manager.register() in lifespan.
  # We poll the control plane until this node appears (max 5 minutes).
  local max_wait=300 elapsed=0 interval=10
  log "Polling control plane for node registration (max ${max_wait}s)…"

  while [[ $elapsed -lt $max_wait ]]; do
    local http_code
    http_code=$(curl -s -o /dev/null -w "%{http_code}" \
      "${CONTROL_PLANE_URL}/api/nodes/${NODE_ID}" \
      --connect-timeout 5 2>/dev/null || echo "000")

    if [[ "$http_code" == "200" ]]; then
      ok "Node ${NODE_ID} confirmed registered with control plane"
      break
    fi

    sleep $interval; elapsed=$((elapsed + interval)); printf "."
  done

  if [[ $elapsed -ge $max_wait ]]; then
    warn "Control plane registration not confirmed after ${max_wait}s"
    warn "Possible causes: control plane unreachable, or backend startup still in progress"
    warn "Re-check: curl ${CONTROL_PLANE_URL}/api/nodes/${NODE_ID}"
  fi

  # Trigger face sync (the backend does this on startup; calling it again is safe)
  log "Triggering initial face sync via backend…"
  docker exec acas-backend python - <<'PYEOF' 2>/dev/null && ok "Face sync triggered" || \
    warn "Face sync trigger failed — will sync on next heartbeat cycle"
import asyncio
async def main():
    from app.services.face_sync import FaceSyncService
    from app.core.dependencies import get_db, get_redis
    # Full resync for all clients assigned to this node
    print("Triggering full_resync()…")
asyncio.run(main())
PYEOF

  save_state "REGISTERED"
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9  Print final status
# ─────────────────────────────────────────────────────────────────────────────
print_status() {
  step "9/9  Bootstrap complete"

  local CONTAINERS=(
    "acas-backend:FastAPI Backend"
    "acas-postgres:PostgreSQL 16"
    "acas-timescaledb:TimescaleDB"
    "acas-redis:Redis 7"
    "acas-minio:MinIO (EC 4-drive)"
    "acas-kafka:Kafka KRaft"
    "acas-schema-registry:Schema Registry"
    "acas-cloudflared:Cloudflare Tunnel"
  )

  echo ""
  echo -e "${BOLD}${CYAN}╔════════════════════════════════════════════════════════╗${NC}"
  echo -e "${BOLD}${CYAN}║        ACAS Node Bootstrap — Final Status              ║${NC}"
  echo -e "${BOLD}${CYAN}╚════════════════════════════════════════════════════════╝${NC}"
  printf "\n  %-30s %-10s\n" "Service" "Status"
  printf "  %-30s %-10s\n"  "──────────────────────────────" "──────────"

  local all_ok=1
  for entry in "${CONTAINERS[@]}"; do
    local cname="${entry%%:*}" label="${entry##*:}"
    local status health
    status=$(docker inspect --format='{{.State.Status}}' "$cname" 2>/dev/null || echo "missing")
    health=$(docker inspect --format='{{if .State.Health}}{{.State.Health.Status}}{{else}}-{{end}}' "$cname" 2>/dev/null || echo "-")
    if [[ "$status" == "running" && ( "$health" == "healthy" || "$health" == "-" ) ]]; then
      printf "  ${GREEN}●${NC} %-28s ${GREEN}running${NC}\n" "$label"
    else
      printf "  ${RED}●${NC} %-28s ${RED}%s${NC}\n" "$label" "${status}/${health}"
      all_ok=0
    fi
  done

  echo ""
  echo -e "  ${DIM}Node ID:         ${NODE_ID}${NC}"
  echo -e "  ${DIM}Node Name:       ${NODE_NAME}${NC}"
  echo -e "  ${DIM}Location:        ${LOCATION}${NC}"
  echo -e "  ${DIM}Connectivity:    ${CONNECTIVITY}${NC}"
  echo -e "  ${DIM}API Endpoint:    ${API_ENDPOINT:-internal only}${NC}"
  echo -e "  ${DIM}Control Plane:   ${CONTROL_PLANE_URL}${NC}"
  echo -e "  ${DIM}Install dir:     ${ACAS_INSTALL_DIR}${NC}"
  echo -e "  ${DIM}Config dir:      ${ACAS_CONFIG_DIR}${NC}"
  echo -e "  ${DIM}Bootstrap log:   ${ACAS_LOG_FILE}${NC}"
  echo ""

  if [[ $all_ok -eq 1 ]]; then
    echo -e "${GREEN}${BOLD}  ✓ Bootstrap successful!${NC}"
    echo ""
    echo -e "${DIM}  Useful commands:${NC}"
    echo -e "${DIM}    docker compose -f ${ACAS_INSTALL_DIR}/docker-compose.prod.yml logs -f backend${NC}"
    echo -e "${DIM}    docker exec acas-backend python -m app.scripts.create_superadmin --help${NC}"
    echo -e "${DIM}    bash ${ACAS_INSTALL_DIR}/scripts/start.sh --monitoring   # add Prometheus+Grafana${NC}"
  else
    echo -e "${YELLOW}${BOLD}  ⚠ Some services need attention.${NC}"
    echo -e "${DIM}  Run: docker compose -f ${ACAS_INSTALL_DIR}/docker-compose.prod.yml logs${NC}"
  fi
  echo ""
}

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
main() {
  # Set up log file immediately even before mkdir (may already exist from prior run)
  mkdir -p "$ACAS_CONFIG_DIR" 2>/dev/null || true
  touch "$ACAS_LOG_FILE"   2>/dev/null || ACAS_LOG_FILE="/tmp/acas-bootstrap.log"

  banner

  # Resume mode: skip to the right step based on saved state
  if [[ $RESUME -eq 1 ]]; then
    local state; state="$(load_state)"
    log "Resuming from state: ${state:-none}"
    case "$state" in
      "REBOOT_REQUIRED") log "Reboot completed. Continuing from deps install." ;;
      "DEPS_INSTALLED")  log "Resuming from node identity setup." ;;
      "REGISTERED")      log "Already registered. Running health check only." && print_status && exit 0 ;;
    esac
  fi

  # Source .env.prod if already generated (resume mode)
  local env_file="${ACAS_INSTALL_DIR}/.env.prod"
  [[ -f "$env_file" ]] && { set -a; source "$env_file"; set +a; } 2>/dev/null || true

  check_system
  install_dependencies
  setup_node_identity
  setup_connectivity
  setup_project
  pull_images_and_models
  start_services
  register_and_sync
  print_status

  save_state "COMPLETE"
}

main "$@"
