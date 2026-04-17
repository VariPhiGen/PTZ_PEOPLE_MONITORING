#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ACAS Restore Script                                                        ║
# ║  Restore PostgreSQL from MinIO backup. Supports standard and               ║
# ║  point-in-time recovery (PITR) when WAL archiving is configured.           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Usage:
#   bash scripts/restore.sh [OPTIONS]
#
# Options:
#   --backup-date YYYY-MM-DD    Restore the backup from this date (latest for the day)
#   --backup-id   ID            Exact backup ID (prefix) to restore
#   --list                      List available backups and exit
#   --target-time DATETIME      Point-in-time target: "2026-03-21 14:30:00 UTC"
#                               Requires WAL archiving to be configured.
#   --db-only                   Restore databases only (skip config restore)
#   --config-only               Restore configuration only
#   --skip-migrations           Skip Alembic migrate after restore
#   --no-confirm                Skip interactive confirmation prompts
#   --env FILE                  .env.prod path (default: /opt/acas/.env.prod)
#   --dry-run                   Print plan without executing
#   --help
#
# PITR notes:
#   Point-in-time recovery requires:
#     1. archive_mode = on in postgresql.conf
#     2. archive_command configured to save WAL segments to a shared location
#     3. The WAL segments to be available in POSTGRES_WAL_ARCHIVE_DIR
#   If these are not configured, only the latest dump-based restore is available.

set -euo pipefail
IFS=$'\n\t'

# ── Defaults ──────────────────────────────────────────────────────────────────
BACKUP_DATE=""
BACKUP_ID=""
DO_LIST=0
TARGET_TIME=""
DB_ONLY=0
CONFIG_ONLY=0
SKIP_MIGRATIONS=0
NO_CONFIRM=0
ENV_FILE="/opt/acas/.env.prod"
DRY_RUN=0

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; NC='\033[0m'

log()  { echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} $*"; }
ok()   { echo -e "${GREEN}[$(date +%H:%M:%S)] ✓${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] ⚠${NC} $*"; }
die()  { echo -e "${RED}[$(date +%H:%M:%S)] ✗${NC} $*" >&2; exit 1; }
dry()  { [[ $DRY_RUN -eq 1 ]] && { echo -e "${DIM}  [DRY-RUN] $*${NC}"; return 0; }; "$@"; }

confirm() {
  [[ $NO_CONFIRM -eq 1 ]] && return 0
  local prompt="${1:-Continue?}"
  read -r -p "$(echo -e "${YELLOW}${prompt} [y/N]: ${NC}")" ans
  [[ "$ans" =~ ^[Yy]$ ]] || { log "Aborted."; exit 0; }
}

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --backup-date)     BACKUP_DATE="$2";     shift 2 ;;
    --backup-id)       BACKUP_ID="$2";       shift 2 ;;
    --list)            DO_LIST=1;            shift   ;;
    --target-time)     TARGET_TIME="$2";     shift 2 ;;
    --db-only)         DB_ONLY=1;            shift   ;;
    --config-only)     CONFIG_ONLY=1;        shift   ;;
    --skip-migrations) SKIP_MIGRATIONS=1;    shift   ;;
    --no-confirm)      NO_CONFIRM=1;         shift   ;;
    --env)             ENV_FILE="$2";        shift 2 ;;
    --dry-run)         DRY_RUN=1;            shift   ;;
    --help)
      sed -n '/^# Usage:/,/^[^#]/p' "$0" | head -30
      exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

# ── Load env ──────────────────────────────────────────────────────────────────
[[ -f "$ENV_FILE" ]] || die ".env.prod not found at ${ENV_FILE}"
set -a; source "$ENV_FILE"; set +a

POSTGRES_PASSWORD="${POSTGRES_PASSWORD:?}"
TIMESCALE_PASSWORD="${TIMESCALE_PASSWORD:?}"
MINIO_ROOT_USER="${MINIO_ROOT_USER:?}"
MINIO_ROOT_PASSWORD="${MINIO_ROOT_PASSWORD:?}"

BUCKET="acas-backups"
MINIO_ALIAS="acasmc"
WORK_DIR="/tmp/acas-restore-$$"

# ── Container helpers ──────────────────────────────────────────────────────────
dc_exec() { docker exec "$1" "${@:2}"; }
require_container() {
  docker inspect --format='{{.State.Status}}' "$1" 2>/dev/null | grep -q "running" || \
    die "Container $1 is not running"
}

# ── Configure mc ─────────────────────────────────────────────────────────────
setup_mc() {
  require_container "acas-minio"
  dc_exec "acas-minio" \
    mc alias set "$MINIO_ALIAS" \
      "http://localhost:9000" \
      "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD" \
      --api S3v4 > /dev/null 2>&1
}

# ─────────────────────────────────────────────────────────────────────────────
# LIST available backups
# ─────────────────────────────────────────────────────────────────────────────
list_backups() {
  setup_mc
  log "Available backups in s3://${BUCKET}:"
  echo ""
  printf "  %-40s %-16s %-10s\n" "Backup ID" "Date" "Type"
  printf "  %-40s %-16s %-10s\n" "───────────────────────────────────────" "───────────────" "──────────"

  dc_exec "acas-minio" mc find "${MINIO_ALIAS}/${BUCKET}" --name "*_manifest.json" --print 2>/dev/null | \
    while read -r obj; do
      local bdate btype bid
      bdate=$(echo "$obj" | grep -oP '\d{4}/\d{2}/\d{2}' | head -1 | tr '/' '-')
      bid=$(basename "$obj" | sed 's/_manifest.json//')
      btype=$(dc_exec "acas-minio" mc cat "$obj" 2>/dev/null | \
        python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('backup_type','?'))" 2>/dev/null || echo "?")
      printf "  %-40s %-16s %-10s\n" "$bid" "$bdate" "$btype"
    done

  echo ""
  echo -e "${DIM}  Restore: bash scripts/restore.sh --backup-date YYYY-MM-DD${NC}"
  echo -e "${DIM}  Restore: bash scripts/restore.sh --backup-id <ID>${NC}"
}

if [[ $DO_LIST -eq 1 ]]; then
  setup_mc
  list_backups
  exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# Resolve which backup to restore
# ─────────────────────────────────────────────────────────────────────────────
resolve_backup() {
  setup_mc

  if [[ -n "$BACKUP_ID" ]]; then
    log "Using explicit backup ID: ${BACKUP_ID}"
    return
  fi

  if [[ -n "$BACKUP_DATE" ]]; then
    local date_path="${BACKUP_DATE//-//}"   # YYYY-MM-DD → YYYY/MM/DD
    log "Searching for backup on ${BACKUP_DATE}…"
    local latest
    latest=$(dc_exec "acas-minio" mc find "${MINIO_ALIAS}/${BUCKET}/${date_path}" \
      --name "*_acas.dump" --print 2>/dev/null | sort | tail -1)

    if [[ -z "$latest" ]]; then
      # Interactive fallback: list and ask
      warn "No dump found for ${BACKUP_DATE}. Available dates:"
      list_backups
      die "Specify a valid --backup-date or --backup-id"
    fi

    # Extract ID from path:  .../YYYY/MM/DD/PREFIX_acas.dump
    BACKUP_ID=$(basename "$latest" "_acas.dump")
    log "Resolved backup ID: ${BACKUP_ID}"
    return
  fi

  # Nothing specified — list and prompt
  list_backups
  echo ""
  read -r -p "$(echo -e "${CYAN}Enter backup ID to restore: ${NC}")" BACKUP_ID
  [[ -n "$BACKUP_ID" ]] || die "No backup ID provided"
}

resolve_backup

# ─────────────────────────────────────────────────────────────────────────────
# Determine object paths from BACKUP_ID
# ─────────────────────────────────────────────────────────────────────────────
# Backup ID format: hostname_YYYY-MM-DD_HH-MM-SS[_tag]
# Extract date part for bucket path
BDATE_RAW=$(echo "$BACKUP_ID" | grep -oP '\d{4}-\d{2}-\d{2}' | head -1)
BDATE_PATH="${BDATE_RAW//-//}"

find_object() {
  local suffix="$1"
  dc_exec "acas-minio" mc find \
    "${MINIO_ALIAS}/${BUCKET}/${BDATE_PATH}" \
    --name "${BACKUP_ID}*${suffix}" \
    --print 2>/dev/null | head -1
}

OBJ_ACAS=$(find_object "_acas.dump")
OBJ_ACAS_TS=$(find_object "_acas_ts.dump")
OBJ_GLOBALS=$(find_object "_globals.sql")
OBJ_CONFIG=$(find_object "_config.tar.gz")

[[ -n "$OBJ_ACAS" ]] || die "Backup not found: ${BACKUP_ID} (looking in ${BDATE_PATH})"

# ─────────────────────────────────────────────────────────────────────────────
# CONFIRMATION
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${RED}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${RED}║  WARNING: Database Restore                                   ║${NC}"
echo -e "${BOLD}${RED}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  Backup ID  : ${CYAN}${BACKUP_ID}${NC}"
echo -e "  acas dump  : ${OBJ_ACAS:-not found}"
echo -e "  ts dump    : ${OBJ_ACAS_TS:-not found}"
[[ -n "$TARGET_TIME" ]] && echo -e "  PITR target: ${YELLOW}${TARGET_TIME}${NC}"
echo ""
echo -e "  ${RED}This will DESTROY the current database and replace it with the backup.${NC}"
echo -e "  ${RED}The backend will be stopped during restore.${NC}"
echo ""
[[ $DRY_RUN -eq 0 ]] && confirm "Proceed with restore?"

# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOAD backup files
# ─────────────────────────────────────────────────────────────────────────────
mkdir -p "$WORK_DIR"
trap 'rm -rf "$WORK_DIR"' EXIT

download_from_minio() {
  local obj="$1" dest="$2" label="$3"
  [[ -z "$obj" ]] && { warn "Skipping ${label} (not found in backup)"; return; }
  log "Downloading ${label}…"
  # Copy from MinIO to minio container temp, then docker cp to host
  local fname; fname=$(basename "$obj")
  dry dc_exec "acas-minio" mc cp "$obj" "/tmp/${fname}" --quiet
  dry docker cp "acas-minio:/tmp/${fname}" "$dest"
  dry dc_exec "acas-minio" rm -f "/tmp/${fname}"
  if [[ $DRY_RUN -eq 0 ]] && [[ -f "$dest" ]]; then
    ok "  ${label}: $(du -sh "$dest" | cut -f1)"
  fi
}

if [[ $CONFIG_ONLY -eq 0 ]]; then
  download_from_minio "$OBJ_ACAS"    "${WORK_DIR}/acas.dump"    "acas.dump"
  download_from_minio "$OBJ_ACAS_TS" "${WORK_DIR}/acas_ts.dump" "acas_ts.dump"
  download_from_minio "$OBJ_GLOBALS" "${WORK_DIR}/globals.sql"  "globals.sql"
fi
if [[ $DB_ONLY -eq 0 ]] && [[ -n "$OBJ_CONFIG" ]]; then
  download_from_minio "$OBJ_CONFIG"  "${WORK_DIR}/config.tar.gz" "config.tar.gz"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STOP BACKEND
# ─────────────────────────────────────────────────────────────────────────────
stop_backend() {
  log "Stopping backend containers…"
  local acas_root; acas_root="$(cd "$(dirname "${ENV_FILE}")/.." && pwd || echo "/opt/acas")"
  dry docker compose \
    -f "${acas_root}/docker-compose.prod.yml" \
    --env-file "${ENV_FILE}" \
    stop backend 2>/dev/null || true
  ok "Backend stopped"
}

if [[ $CONFIG_ONLY -eq 0 ]]; then
  stop_backend
fi

# ─────────────────────────────────────────────────────────────────────────────
# RESTORE PostgreSQL (standard dump-based)
# ─────────────────────────────────────────────────────────────────────────────
restore_postgres() {
  require_container "acas-postgres"
  local dump="${WORK_DIR}/acas.dump"
  [[ -f "$dump" ]] || die "Dump file not found: ${dump}"

  log "Restoring PostgreSQL database: acas…"

  # Pre-restore safety snapshot (quick — in case we need to undo)
  log "  Creating pre-restore savepoint dump…"
  dry dc_exec "acas-postgres" \
    pg_dump -U acas -d acas --format=custom --compress=5 --no-password \
    -f "/tmp/pre_restore_savepoint.dump" 2>/dev/null || warn "  Savepoint failed (continuing)"

  # Drop and recreate the database (allows full pg_restore)
  log "  Dropping existing acas database…"
  dry dc_exec "acas-postgres" \
    psql -U postgres -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname='acas' AND pid <> pg_backend_pid();" \
    > /dev/null 2>&1 || true
  dry dc_exec "acas-postgres" \
    psql -U postgres -c "DROP DATABASE IF EXISTS acas;" > /dev/null 2>&1
  dry dc_exec "acas-postgres" \
    psql -U postgres -c "CREATE DATABASE acas OWNER acas;" > /dev/null 2>&1

  # Restore global roles first
  if [[ -f "${WORK_DIR}/globals.sql" ]]; then
    log "  Restoring global roles…"
    docker cp "${WORK_DIR}/globals.sql" "acas-postgres:/tmp/globals.sql"
    dry dc_exec "acas-postgres" \
      psql -U postgres -f "/tmp/globals.sql" > /dev/null 2>&1 || warn "  Role restore had warnings (OK if roles exist)"
    dc_exec "acas-postgres" rm -f "/tmp/globals.sql"
  fi

  # Restore main DB
  log "  Restoring from dump (this may take several minutes)…"
  docker cp "$dump" "acas-postgres:/tmp/acas.dump"
  if ! dry dc_exec "acas-postgres" \
    pg_restore -U acas -d acas --no-password \
      --jobs=4 \
      --exit-on-error \
      /tmp/acas.dump 2>&1 | grep -v "^pg_restore: warning" ; then
    warn "pg_restore completed with warnings — check output above"
  fi
  dc_exec "acas-postgres" rm -f "/tmp/acas.dump"
  ok "acas database restored"
}

restore_timescaledb() {
  require_container "acas-timescaledb"
  local dump="${WORK_DIR}/acas_ts.dump"
  [[ -f "$dump" ]] || { warn "TimescaleDB dump not found — skipping"; return; }

  log "Restoring TimescaleDB database: acas_ts…"
  dry dc_exec "acas-timescaledb" \
    psql -U postgres -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname='acas_ts' AND pid <> pg_backend_pid();" \
    > /dev/null 2>&1 || true
  dry dc_exec "acas-timescaledb" \
    psql -U postgres -c "DROP DATABASE IF EXISTS acas_ts;" > /dev/null 2>&1
  dry dc_exec "acas-timescaledb" \
    psql -U postgres -c "CREATE DATABASE acas_ts OWNER acas;" > /dev/null 2>&1

  docker cp "$dump" "acas-timescaledb:/tmp/acas_ts.dump"
  dry dc_exec "acas-timescaledb" \
    pg_restore -U acas -d acas_ts --no-password \
      --jobs=4 \
      /tmp/acas_ts.dump 2>&1 | grep -v "^pg_restore: warning" || warn "Restore had warnings"
  dc_exec "acas-timescaledb" rm -f "/tmp/acas_ts.dump"
  ok "acas_ts database restored"
}

# ─────────────────────────────────────────────────────────────────────────────
# POINT-IN-TIME RECOVERY (PITR)
# ─────────────────────────────────────────────────────────────────────────────
# Requires:
#   - archive_mode = on  in postgresql.conf
#   - WAL segments available in POSTGRES_WAL_ARCHIVE_DIR (env var)
#   - PostgreSQL >= 12 (uses recovery_target_time in postgresql.auto.conf)
# ─────────────────────────────────────────────────────────────────────────────
configure_pitr() {
  local target_time="$1"
  local wal_archive_dir="${POSTGRES_WAL_ARCHIVE_DIR:-}"

  if [[ -z "$wal_archive_dir" ]]; then
    die "PITR requires POSTGRES_WAL_ARCHIVE_DIR to be set in .env.prod.
  Set archive_mode=on and archive_command in postgresql.conf, then re-run."
  fi

  log "Configuring point-in-time recovery to: ${target_time}"
  log "WAL archive directory: ${wal_archive_dir}"

  # Verify WAL archive is accessible from postgres container
  dry dc_exec "acas-postgres" test -d "$wal_archive_dir" || \
    die "WAL archive directory not accessible inside postgres container: ${wal_archive_dir}"

  # Write recovery configuration to postgresql.auto.conf (PostgreSQL 12+)
  # This tells PostgreSQL to replay WAL until the target time on next startup
  local recovery_conf
  read -r -d '' recovery_conf << EOCONF || true
# ACAS PITR recovery settings (auto-generated by restore.sh)
restore_command = 'cp ${wal_archive_dir}/%f %p'
recovery_target_time = '${target_time}'
recovery_target_action = 'promote'
EOCONF

  log "Writing recovery configuration…"
  dry dc_exec "acas-postgres" \
    sh -c "cat >> /var/lib/postgresql/data/postgresql.auto.conf << 'EOF'
${recovery_conf}
EOF"

  # Create recovery signal file (PostgreSQL 12+ uses standby.signal or recovery.signal)
  dry dc_exec "acas-postgres" \
    touch /var/lib/postgresql/data/recovery.signal

  ok "PITR configured — restart postgres to begin WAL replay"
  echo ""
  warn "PostgreSQL will replay WAL segments until ${target_time}."
  warn "After recovery completes, run: bash scripts/restore.sh --skip-migrations (if schema is current)"
  warn "or re-run Alembic migrations: docker exec acas-backend alembic upgrade head"
}

# ─────────────────────────────────────────────────────────────────────────────
# RESTORE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
restore_config() {
  local archive="${WORK_DIR}/config.tar.gz"
  [[ -f "$archive" ]] || { warn "Config archive not found — skipping"; return; }

  log "Restoring configuration files…"
  local restore_staging="${WORK_DIR}/config-restore"
  mkdir -p "$restore_staging"
  dry tar -xzf "$archive" -C "$restore_staging"

  # Show what would be restored without auto-applying secrets
  echo "  Configuration files found in backup:"
  find "$restore_staging" -type f | while read -r f; do
    echo "    ${f#${restore_staging}/}"
  done

  echo ""
  warn "Configuration files extracted to: ${restore_staging}"
  warn "Review and manually apply any settings changes."
  warn "Do NOT blindly overwrite .env.prod — it may contain different secrets."
  ok "Config archive extracted"
}

# ─────────────────────────────────────────────────────────────────────────────
# ALEMBIC MIGRATIONS
# ─────────────────────────────────────────────────────────────────────────────
run_migrations() {
  log "Running Alembic migrations (upgrade head)…"
  local acas_root; acas_root="$(cd "$(dirname "${ENV_FILE}")/.." && pwd || echo "/opt/acas")"

  dry docker run --rm \
    --network acas-prod-net \
    -e DATABASE_URL="postgresql+asyncpg://acas:${POSTGRES_PASSWORD}@postgres:5432/acas" \
    -e TIMESCALE_URL="postgresql+asyncpg://acas:${TIMESCALE_PASSWORD}@timescaledb:5432/acas_ts" \
    -e REDIS_URL="${REDIS_URL:-redis://redis:6379/0}" \
    -e MINIO_ENDPOINT="${MINIO_ENDPOINT:-minio:9000}" \
    -e MINIO_ACCESS_KEY="${MINIO_ROOT_USER}" \
    -e MINIO_SECRET_KEY="${MINIO_ROOT_PASSWORD}" \
    -e KAFKA_BOOTSTRAP_SERVERS="${KAFKA_BOOTSTRAP_SERVERS:-kafka:19092}" \
    -e JWT_SECRET_KEY="${JWT_SECRET_KEY}" \
    -v "${acas_root}/backend:/app:ro" \
    acas-backend:gpu \
    alembic -c /app/alembic.ini upgrade head
  ok "Migrations complete"
}

# ─────────────────────────────────────────────────────────────────────────────
# RESTART BACKEND
# ─────────────────────────────────────────────────────────────────────────────
restart_backend() {
  log "Starting backend…"
  local acas_root; acas_root="$(cd "$(dirname "${ENV_FILE}")/.." && pwd || echo "/opt/acas")"
  dry docker compose \
    -f "${acas_root}/docker-compose.prod.yml" \
    --env-file "${ENV_FILE}" \
    up -d --no-deps backend

  log "Waiting for backend health check…"
  local elapsed=0
  while [[ $DRY_RUN -eq 0 ]]; do
    local hst
    hst=$(docker inspect --format='{{.State.Health.Status}}' acas-backend 2>/dev/null || echo missing)
    [[ "$hst" == "healthy" ]] && { ok "Backend healthy"; return 0; }
    [[ "$hst" == "unhealthy" ]] && die "Backend unhealthy after restore — check logs: docker logs acas-backend"
    [[ $elapsed -ge 120 ]] && die "Backend health timeout"
    sleep 5; elapsed=$((elapsed + 5)); printf "."
  done
  [[ $DRY_RUN -eq 1 ]] && echo -e "${DIM}  [DRY-RUN] Skipped health check${NC}"
}

# ─────────────────────────────────────────────────────────────────────────────
# EXECUTE
# ─────────────────────────────────────────────────────────────────────────────
if [[ -n "$TARGET_TIME" ]] && [[ $CONFIG_ONLY -eq 0 ]]; then
  # PITR path: restore base dump first, then configure WAL replay
  restore_postgres
  restore_timescaledb
  configure_pitr "$TARGET_TIME"
  log "Restarting postgres to begin WAL replay…"
  docker restart acas-postgres 2>/dev/null || true
  log "Monitor postgres logs: docker logs -f acas-postgres"
  log "Once recovery completes, restart the backend: docker compose ... up -d backend"
  exit 0
fi

# Standard restore path
if [[ $CONFIG_ONLY -eq 0 ]]; then
  restore_postgres
  restore_timescaledb
  [[ $SKIP_MIGRATIONS -eq 0 ]] && run_migrations
fi

if [[ $DB_ONLY -eq 0 ]]; then
  restore_config
fi

restart_backend

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${GREEN}║  Restore Complete                                            ║${NC}"
echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  Backup ID : ${CYAN}${BACKUP_ID}${NC}"
[[ -n "$TARGET_TIME" ]] && echo -e "  PITR time : ${TARGET_TIME}"
echo -e "  Status    : ${GREEN}Backend healthy${NC}"
echo ""
echo -e "${DIM}  Verify:  curl http://localhost:8000/api/health${NC}"
echo -e "${DIM}  Face DB will rebuild automatically on first recognition (full_resync).${NC}"
echo ""
