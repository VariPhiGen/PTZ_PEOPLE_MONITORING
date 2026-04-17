#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ACAS Multi-Node Deploy                                                     ║
# ║  Orchestrates a rolling deployment across every registered GPU node.        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Usage:
#   bash scripts/deploy-all.sh [OPTIONS]
#
# Options:
#   --nodes FILE        Path to node inventory file (default: /etc/acas/nodes.conf)
#   --parallel          Deploy to all nodes simultaneously (default: sequential)
#   --skip-pull         Use locally cached Docker images
#   --skip-build        Skip rebuilding acas-backend:gpu
#   --branch NAME       Git branch to deploy (default: current branch)
#   --rollback-on-fail  Auto-rollback a node to previous commit on health failure
#   --health-timeout N  Seconds to wait for backend health (default: 120)
#   --dry-run           Print plan without executing remote commands
#   --ssh-opts OPTS     Extra SSH options (e.g. '-o StrictHostKeyChecking=no')
#   --help
#
# Node inventory file format (whitespace-delimited, # = comment):
#   # node_name   host_or_ip    ssh_user   ssh_key_path             deploy_dir
#   node-1        10.1.1.10     acas       /etc/acas/keys/id_ed25519  /opt/acas
#   node-2        10.1.1.11     acas       /etc/acas/keys/id_ed25519  /opt/acas
#
# Exit codes:
#   0  All nodes deployed successfully
#   1  One or more nodes failed (rolled back if --rollback-on-fail)
#   2  Pre-flight error (bad arguments, missing inventory)

set -euo pipefail
IFS=$'\n\t'

# ── Defaults ──────────────────────────────────────────────────────────────────
NODES_FILE="/etc/acas/nodes.conf"
PARALLEL=0
SKIP_PULL=0
SKIP_BUILD=0
BRANCH=""
ROLLBACK_ON_FAIL=0
HEALTH_TIMEOUT=120
DRY_RUN=0
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; NC='\033[0m'

log()  { echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} $*"; }
ok()   { echo -e "${GREEN}[$(date +%H:%M:%S)] ✓${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] ⚠${NC} $*"; }
die()  { echo -e "${RED}[$(date +%H:%M:%S)] ✗${NC} $*" >&2; exit 2; }
info() { echo -e "${DIM}[$(date +%H:%M:%S)]   $*${NC}"; }
step() {
  echo -e "\n${BOLD}${CYAN}══════════════════════════════════════════════════════════════${NC}"
  echo -e "${BOLD}${CYAN}  $*${NC}"
  echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════════════════${NC}"
}

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --nodes)            NODES_FILE="$2";         shift 2 ;;
    --parallel)         PARALLEL=1;              shift   ;;
    --skip-pull)        SKIP_PULL=1;             shift   ;;
    --skip-build)       SKIP_BUILD=1;            shift   ;;
    --branch)           BRANCH="$2";             shift 2 ;;
    --rollback-on-fail) ROLLBACK_ON_FAIL=1;      shift   ;;
    --health-timeout)   HEALTH_TIMEOUT="$2";     shift 2 ;;
    --dry-run)          DRY_RUN=1;               shift   ;;
    --ssh-opts)         SSH_OPTS="$2";           shift 2 ;;
    --help)
      sed -n '/^# Usage:/,/^[^#]/p' "$0" | head -30
      exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

# ── Load inventory ─────────────────────────────────────────────────────────────
[[ -f "$NODES_FILE" ]] || die "Node inventory not found: ${NODES_FILE}"

declare -a NAMES HOSTS USERS KEYS DIRS
while IFS= read -r line; do
  # Skip blank lines and comments
  [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
  read -r name host user key dir <<< "$line"
  NAMES+=("$name")
  HOSTS+=("$host")
  USERS+=("$user")
  KEYS+=("$key")
  DIRS+=("${dir:-/opt/acas}")
done < "$NODES_FILE"

[[ ${#NAMES[@]} -gt 0 ]] || die "No nodes found in ${NODES_FILE}"
log "Loaded ${#NAMES[@]} node(s) from ${NODES_FILE}"

# ── State tracking ────────────────────────────────────────────────────────────
declare -A NODE_STATUS NODE_PREV_HASH NODE_NEW_HASH

# ── SSH helper ────────────────────────────────────────────────────────────────
ssh_run() {
  local user="$1" host="$2" key="$3"
  shift 3
  # shellcheck disable=SC2086
  ssh $SSH_OPTS -i "$key" "${user}@${host}" "$@"
}

ssh_run_q() {
  local user="$1" host="$2" key="$3"
  shift 3
  # shellcheck disable=SC2086
  ssh $SSH_OPTS -i "$key" "${user}@${host}" "$@" 2>/dev/null
}

dry() {
  if [[ $DRY_RUN -eq 1 ]]; then
    info "[DRY-RUN] $*"
    return 0
  fi
  "$@"
}

# ── Per-node deploy function ──────────────────────────────────────────────────
deploy_node() {
  local idx="$1"
  local name="${NAMES[$idx]}" host="${HOSTS[$idx]}" user="${USERS[$idx]}"
  local key="${KEYS[$idx]}"   dir="${DIRS[$idx]}"
  local prefix="${BOLD}[${name}]${NC}"

  echo -e "\n${prefix} ${CYAN}Deploying to ${user}@${host}:${dir}${NC}"

  # ── 1. SSH connectivity check ──────────────────────────────────────────────
  if [[ $DRY_RUN -eq 0 ]]; then
    if ! ssh_run_q "$user" "$host" "$key" "echo ok" > /dev/null 2>&1; then
      echo -e "${prefix} ${RED}✗ SSH connection failed — skipping${NC}"
      NODE_STATUS[$name]="SKIP_SSH"
      return 1
    fi
  fi
  echo -e "${prefix} ${GREEN}✓${NC} SSH ok"

  # ── 2. Record current git hash (for rollback) ──────────────────────────────
  local prev_hash=""
  if [[ $DRY_RUN -eq 0 ]]; then
    prev_hash=$(ssh_run "$user" "$host" "$key" \
      "cd ${dir} && git rev-parse --short HEAD 2>/dev/null || echo unknown")
  fi
  NODE_PREV_HASH[$name]="${prev_hash:-unknown}"
  echo -e "${prefix} ${DIM}Current commit: ${prev_hash:-unknown}${NC}"

  # ── 3. Git pull ────────────────────────────────────────────────────────────
  local pull_cmd="cd ${dir} && git fetch --quiet"
  if [[ -n "$BRANCH" ]]; then
    pull_cmd+=" && git checkout ${BRANCH} --quiet"
  fi
  pull_cmd+=" && git pull --ff-only --quiet"

  echo -e "${prefix} Pulling latest code…"
  if ! dry ssh_run "$user" "$host" "$key" "$pull_cmd"; then
    echo -e "${prefix} ${RED}✗ git pull failed${NC}"
    NODE_STATUS[$name]="FAIL_PULL"
    return 1
  fi

  local new_hash=""
  if [[ $DRY_RUN -eq 0 ]]; then
    new_hash=$(ssh_run_q "$user" "$host" "$key" \
      "cd ${dir} && git rev-parse --short HEAD 2>/dev/null || echo unknown")
  fi
  NODE_NEW_HASH[$name]="${new_hash:-unknown}"
  echo -e "${prefix} ${GREEN}✓${NC} Pulled (${prev_hash:-?} → ${new_hash:-?})"

  # ── 4. Build backend image (optional) ──────────────────────────────────────
  if [[ $SKIP_BUILD -eq 0 ]]; then
    echo -e "${prefix} Building acas-backend:gpu…"
    local build_cmd="cd ${dir} && docker build -t acas-backend:gpu -f backend/Dockerfile.gpu backend/ --quiet"
    if ! dry ssh_run "$user" "$host" "$key" "$build_cmd"; then
      echo -e "${prefix} ${YELLOW}⚠ Build failed — continuing with existing image${NC}"
    else
      echo -e "${prefix} ${GREEN}✓${NC} Image built"
    fi
  fi

  # ── 5. Pull service images ──────────────────────────────────────────────────
  if [[ $SKIP_PULL -eq 0 ]]; then
    echo -e "${prefix} Pulling service images…"
    local pull_images_cmd="cd ${dir} && docker compose -f docker-compose.prod.yml pull --quiet 2>/dev/null || true"
    dry ssh_run "$user" "$host" "$key" "$pull_images_cmd"
    echo -e "${prefix} ${GREEN}✓${NC} Images pulled"
  fi

  # ── 6. Rolling restart (backend only — DB/Redis/Kafka stay up) ─────────────
  echo -e "${prefix} Restarting backend container…"
  local restart_cmd="cd ${dir} && docker compose -f docker-compose.prod.yml up -d --no-deps --quiet backend"
  if ! dry ssh_run "$user" "$host" "$key" "$restart_cmd"; then
    echo -e "${prefix} ${RED}✗ Container restart failed${NC}"
    NODE_STATUS[$name]="FAIL_RESTART"
    [[ $ROLLBACK_ON_FAIL -eq 1 ]] && _rollback "$idx"
    return 1
  fi
  echo -e "${prefix} ${GREEN}✓${NC} Backend container restarted"

  # ── 7. Health check ────────────────────────────────────────────────────────
  echo -e "${prefix} Waiting for health check (timeout ${HEALTH_TIMEOUT}s)…"
  if [[ $DRY_RUN -eq 1 ]]; then
    echo -e "${prefix} ${DIM}[DRY-RUN] Skipping health check${NC}"
    NODE_STATUS[$name]="DRY_OK"
    return 0
  fi

  local elapsed=0
  while true; do
    local hstatus
    hstatus=$(ssh_run_q "$user" "$host" "$key" \
      "docker inspect --format='{{.State.Health.Status}}' acas-backend 2>/dev/null || echo missing")
    case "$hstatus" in
      healthy)
        echo -e "${prefix} ${GREEN}✓ Backend healthy${NC}"
        NODE_STATUS[$name]="OK"
        return 0 ;;
      unhealthy)
        echo -e "${prefix} ${RED}✗ Backend unhealthy${NC}"
        NODE_STATUS[$name]="FAIL_HEALTH"
        [[ $ROLLBACK_ON_FAIL -eq 1 ]] && _rollback "$idx"
        return 1 ;;
    esac
    if [[ $elapsed -ge $HEALTH_TIMEOUT ]]; then
      echo -e "${prefix} ${RED}✗ Health check timed out after ${HEALTH_TIMEOUT}s${NC}"
      NODE_STATUS[$name]="FAIL_TIMEOUT"
      [[ $ROLLBACK_ON_FAIL -eq 1 ]] && _rollback "$idx"
      return 1
    fi
    sleep 5; elapsed=$((elapsed + 5)); printf "."
  done
}

# ── Rollback helper ───────────────────────────────────────────────────────────
_rollback() {
  local idx="$1"
  local name="${NAMES[$idx]}" host="${HOSTS[$idx]}" user="${USERS[$idx]}"
  local key="${KEYS[$idx]}"   dir="${DIRS[$idx]}"
  local prev="${NODE_PREV_HASH[$name]:-}"
  local prefix="${BOLD}[${name}]${NC}"

  if [[ -z "$prev" || "$prev" == "unknown" ]]; then
    echo -e "${prefix} ${YELLOW}⚠ No previous hash to rollback to${NC}"
    return
  fi

  warn "${prefix} Rolling back to ${prev}…"
  ssh_run "$user" "$host" "$key" \
    "cd ${dir} && git checkout ${prev} --quiet && \
     docker compose -f docker-compose.prod.yml up -d --no-deps --quiet backend" || true
  echo -e "${prefix} ${YELLOW}Rollback triggered — verify manually${NC}"
  NODE_STATUS[$name]="ROLLED_BACK"
}

# ── Main ──────────────────────────────────────────────────────────────────────

step "ACAS Multi-Node Deploy"
log "Mode:    $([ $PARALLEL -eq 1 ] && echo PARALLEL || echo SEQUENTIAL)"
log "Nodes:   ${#NAMES[@]}"
log "Branch:  ${BRANCH:-current}"
log "Dry-run: $([ $DRY_RUN -eq 1 ] && echo YES || echo NO)"
echo ""

declare -a PIDS=()

if [[ $PARALLEL -eq 1 ]]; then
  # Launch all nodes in background
  for i in "${!NAMES[@]}"; do
    deploy_node "$i" &
    PIDS+=($!)
  done
  # Wait for all and collect exit codes
  FAIL=0
  for pid in "${PIDS[@]}"; do
    wait "$pid" || FAIL=1
  done
else
  # Sequential: stop on first failure only if rollback is active
  FAIL=0
  for i in "${!NAMES[@]}"; do
    deploy_node "$i" || FAIL=1
  done
fi

# ── Final status report ────────────────────────────────────────────────────────
step "Deployment Summary"
echo ""
printf "  %-20s %-14s %-10s %-10s\n" "Node" "Status" "Before" "After"
printf "  %-20s %-14s %-10s %-10s\n" "────────────────────" "──────────────" "──────────" "──────────"
all_ok=1
for name in "${NAMES[@]}"; do
  status="${NODE_STATUS[$name]:-UNKNOWN}"
  prev="${NODE_PREV_HASH[$name]:-?}"
  new="${NODE_NEW_HASH[$name]:-?}"
  case "$status" in
    OK|DRY_OK)
      sym="${GREEN}●${NC}"; badge="${GREEN}${status}${NC}" ;;
    ROLLED_BACK)
      sym="${YELLOW}●${NC}"; badge="${YELLOW}ROLLED BACK${NC}"; all_ok=0 ;;
    SKIP_SSH)
      sym="${YELLOW}●${NC}"; badge="${YELLOW}SSH FAILED${NC}"; all_ok=0 ;;
    *)
      sym="${RED}●${NC}"; badge="${RED}${status}${NC}"; all_ok=0 ;;
  esac
  printf "  %b %-18s %b  %-10s %-10s\n" "$sym" "$name" "$badge" "$prev" "$new"
done
echo ""

if [[ $all_ok -eq 1 ]]; then
  echo -e "${GREEN}${BOLD}  All nodes deployed successfully.${NC}"
  exit 0
else
  echo -e "${YELLOW}${BOLD}  Some nodes failed — check logs above.${NC}"
  echo -e "${DIM}  Rollback available: --rollback-on-fail${NC}"
  exit 1
fi
