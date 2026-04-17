#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ACAS Backup Script                                                         ║
# ║  pg_dump → MinIO, FAISS metadata, configuration export.                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Usage:
#   bash scripts/backup.sh [OPTIONS]
#
# Options:
#   --type full|db-only|config-only   What to back up (default: full)
#   --env FILE                        .env.prod path (default: /opt/acas/.env.prod)
#   --retention-days N                Delete backups older than N days (default: 30)
#   --tag LABEL                       Extra tag appended to archive names
#   --notify                          Publish completion to Kafka notifications.outbound
#   --dry-run                         Print plan without running backups
#   --help
#
# Designed for daily cron:
#   0 2 * * * /opt/acas/scripts/backup.sh >> /var/log/acas-backup.log 2>&1
#
# What is backed up:
#   1. PostgreSQL (acas)     — pg_dump custom-format, compressed
#   2. PostgreSQL (acas_ts)  — TimescaleDB with hypertables
#   3. Global roles          — pg_dumpall --globals-only
#   4. Learned params JSON   — exported from cameras table
#   5. Configuration files   — .env.prod (secrets redacted unless --include-secrets)
#
# MinIO layout:
#   acas-backups/
#     YYYY/MM/DD/
#       {hostname}_{timestamp}_acas.dump
#       {hostname}_{timestamp}_acas_ts.dump
#       {hostname}_{timestamp}_globals.sql
#       {hostname}_{timestamp}_learned_params.json
#       {hostname}_{timestamp}_config.tar.gz
#       manifest.json

set -euo pipefail
IFS=$'\n\t'

# ── Defaults ──────────────────────────────────────────────────────────────────
BACKUP_TYPE="full"
ENV_FILE="/opt/acas/.env.prod"
RETENTION_DAYS=30
TAG=""
DO_NOTIFY=0
DRY_RUN=0

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; NC='\033[0m'

log()  { echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} $*"; }
ok()   { echo -e "${GREEN}[$(date +%H:%M:%S)] ✓${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] ⚠${NC} $*"; }
die()  { echo -e "${RED}[$(date +%H:%M:%S)] ✗${NC} $*" >&2; exit 1; }
dry()  { [[ $DRY_RUN -eq 1 ]] && { echo -e "${DIM}  [DRY-RUN] $*${NC}"; return 0; }; "$@"; }

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --type)            BACKUP_TYPE="$2";    shift 2 ;;
    --env)             ENV_FILE="$2";       shift 2 ;;
    --retention-days)  RETENTION_DAYS="$2"; shift 2 ;;
    --tag)             TAG="_${2}";         shift 2 ;;
    --notify)          DO_NOTIFY=1;         shift   ;;
    --dry-run)         DRY_RUN=1;           shift   ;;
    --help)
      sed -n '/^# Usage:/,/^[^#]/p' "$0" | head -25
      exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

# ── Load env ──────────────────────────────────────────────────────────────────
[[ -f "$ENV_FILE" ]] || die ".env.prod not found at ${ENV_FILE}. Use --env to specify path."
set -a; source "$ENV_FILE"; set +a

POSTGRES_PASSWORD="${POSTGRES_PASSWORD:?POSTGRES_PASSWORD not set in .env.prod}"
TIMESCALE_PASSWORD="${TIMESCALE_PASSWORD:?TIMESCALE_PASSWORD not set in .env.prod}"
MINIO_ROOT_USER="${MINIO_ROOT_USER:?MINIO_ROOT_USER not set}"
MINIO_ROOT_PASSWORD="${MINIO_ROOT_PASSWORD:?MINIO_ROOT_PASSWORD not set}"

# ── Setup ─────────────────────────────────────────────────────────────────────
HOSTNAME_TAG="${HOSTNAME:-$(hostname -s)}"
TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
DATE_PATH="$(date +%Y/%m/%d)"
BACKUP_PREFIX="${HOSTNAME_TAG}_${TIMESTAMP}${TAG}"
WORK_DIR="/tmp/acas-backup-${TIMESTAMP}"
MANIFEST_FILE="${WORK_DIR}/manifest.json"

BUCKET="acas-backups"
MINIO_ALIAS="acasmc"
DEST_PREFIX="${BUCKET}/${DATE_PATH}/${HOSTNAME_TAG}_${TIMESTAMP}${TAG}"

# ── Helper: run docker exec against a container ───────────────────────────────
dc_exec() {
  local container="$1"; shift
  docker exec "$container" "$@"
}

# ── Helper: check container is running ────────────────────────────────────────
require_container() {
  local name="$1"
  docker inspect --format='{{.State.Status}}' "$name" 2>/dev/null | grep -q "running" || \
    die "Container ${name} is not running. Start services before backup."
}

# ─────────────────────────────────────────────────────────────────────────────
# PRE-FLIGHT
# ─────────────────────────────────────────────────────────────────────────────
log "ACAS Backup  type=${BACKUP_TYPE}  timestamp=${TIMESTAMP}"

require_container "acas-postgres"
require_container "acas-minio"

mkdir -p "$WORK_DIR"
trap 'rm -rf "$WORK_DIR"' EXIT

# Configure mc alias (using the minio container so no host mc binary needed)
log "Configuring MinIO client alias…"
dry dc_exec "acas-minio" \
  mc alias set "$MINIO_ALIAS" \
    "http://localhost:9000" \
    "$MINIO_ROOT_USER" \
    "$MINIO_ROOT_PASSWORD" \
    --api S3v4 > /dev/null

# Create backup bucket if needed
dry dc_exec "acas-minio" mc mb --ignore-existing "${MINIO_ALIAS}/${BUCKET}"
ok "MinIO client ready"

# Start manifest
declare -A MANIFEST_FILES=()
START_EPOCH=$(date +%s)

# ─────────────────────────────────────────────────────────────────────────────
# 1. PostgreSQL backup
# ─────────────────────────────────────────────────────────────────────────────
backup_postgres() {
  log "Dumping PostgreSQL database: acas"

  # Main DB
  local dump_path="${WORK_DIR}/${BACKUP_PREFIX}_acas.dump"
  dry dc_exec "acas-postgres" \
    pg_dump \
      -U acas \
      -d acas \
      --format=custom \
      --compress=9 \
      --no-password \
      -f "/tmp/${BACKUP_PREFIX}_acas.dump"

  if [[ $DRY_RUN -eq 0 ]]; then
    docker cp "acas-postgres:/tmp/${BACKUP_PREFIX}_acas.dump" "$dump_path"
    dc_exec "acas-postgres" rm -f "/tmp/${BACKUP_PREFIX}_acas.dump"
    MANIFEST_FILES[acas_dump]="${BACKUP_PREFIX}_acas.dump"
    ok "acas.dump: $(du -sh "$dump_path" | cut -f1)"
  fi

  # TimescaleDB
  log "Dumping TimescaleDB database: acas_ts"
  require_container "acas-timescaledb"
  local ts_path="${WORK_DIR}/${BACKUP_PREFIX}_acas_ts.dump"
  dry dc_exec "acas-timescaledb" \
    pg_dump \
      -U acas \
      -d acas_ts \
      --format=custom \
      --compress=9 \
      --no-password \
      -f "/tmp/${BACKUP_PREFIX}_acas_ts.dump"

  if [[ $DRY_RUN -eq 0 ]]; then
    docker cp "acas-timescaledb:/tmp/${BACKUP_PREFIX}_acas_ts.dump" "$ts_path"
    dc_exec "acas-timescaledb" rm -f "/tmp/${BACKUP_PREFIX}_acas_ts.dump"
    MANIFEST_FILES[acas_ts_dump]="${BACKUP_PREFIX}_acas_ts.dump"
    ok "acas_ts.dump: $(du -sh "$ts_path" | cut -f1)"
  fi

  # Global roles (for full environment portability)
  log "Exporting global PostgreSQL roles…"
  local globals_path="${WORK_DIR}/${BACKUP_PREFIX}_globals.sql"
  dry dc_exec "acas-postgres" \
    pg_dumpall -U acas --globals-only --no-password \
    > "$globals_path" 2>/dev/null || \
    dry dc_exec "acas-postgres" \
    sh -c "pg_dumpall -U acas --globals-only" > "$globals_path"

  if [[ $DRY_RUN -eq 0 ]]; then
    MANIFEST_FILES[globals_sql]="${BACKUP_PREFIX}_globals.sql"
    ok "globals.sql exported"
  fi
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. FAISS index metadata / learned params
# ─────────────────────────────────────────────────────────────────────────────
backup_faiss_meta() {
  log "Exporting FAISS metadata (learned_params from cameras table)…"
  local params_path="${WORK_DIR}/${BACKUP_PREFIX}_learned_params.json"

  # The FAISS indexes themselves are rebuilt from pgvector on startup.
  # We backup learned_params (seat_heatmap, zoom_per_region, etc.) and
  # the face_embeddings rows (already in the pg_dump, but also exported
  # here as JSON for quick inspection / partial restores).
  dry dc_exec "acas-postgres" \
    psql -U acas -d acas \
    -t -A \
    -c "SELECT json_agg(row_to_json(t)) FROM (
          SELECT camera_id, name, room_name, building, floor,
                 learned_params, updated_at
          FROM cameras
          WHERE learned_params IS NOT NULL
        ) t;" > "$params_path" 2>/dev/null

  if [[ $DRY_RUN -eq 0 ]]; then
    MANIFEST_FILES[learned_params]="${BACKUP_PREFIX}_learned_params.json"
    ok "learned_params.json exported"
  fi
}

# ─────────────────────────────────────────────────────────────────────────────
# 3. Configuration export
# ─────────────────────────────────────────────────────────────────────────────
backup_config() {
  log "Exporting configuration files…"
  local config_dir="${WORK_DIR}/config"
  local config_archive="${WORK_DIR}/${BACKUP_PREFIX}_config.tar.gz"
  mkdir -p "$config_dir"

  local acas_root
  acas_root="$(cd "$(dirname "${ENV_FILE}")/.." 2>/dev/null && pwd || echo "/opt/acas")"

  # Copy files that define runtime behaviour (keys excluded — backup separately)
  for f in \
      "${acas_root}/docker-compose.prod.yml" \
      "${acas_root}/prometheus/prometheus.yml" \
      "${acas_root}/prometheus/alerts.yml" \
      "/etc/acas/node.id" \
      "/etc/nginx/nginx.conf" ; do
    [[ -f "$f" ]] && cp "$f" "$config_dir/" 2>/dev/null || true
  done

  # .env.prod with secrets REDACTED — store separately if full secrets backup needed
  sed 's/\(PASSWORD\|SECRET\|TOKEN\|KEY\)=.*/\1=<REDACTED>/' "$ENV_FILE" \
    > "${config_dir}/.env.prod.redacted" 2>/dev/null || true

  # Archive
  dry tar -czf "$config_archive" -C "$(dirname "$config_dir")" "$(basename "$config_dir")"
  if [[ $DRY_RUN -eq 0 ]]; then
    MANIFEST_FILES[config]="${BACKUP_PREFIX}_config.tar.gz"
    ok "config.tar.gz: $(du -sh "$config_archive" | cut -f1)"
  fi
}

# ─────────────────────────────────────────────────────────────────────────────
# 4. Upload to MinIO
# ─────────────────────────────────────────────────────────────────────────────
upload_to_minio() {
  log "Uploading backup files to MinIO…"
  local total_bytes=0

  for key in "${!MANIFEST_FILES[@]}"; do
    local fname="${MANIFEST_FILES[$key]}"
    local local_path="${WORK_DIR}/${fname}"
    [[ -f "$local_path" ]] || { warn "Missing file: ${fname}"; continue; }

    local fsize
    fsize=$(stat -c%s "$local_path" 2>/dev/null || stat -f%z "$local_path" 2>/dev/null || echo 0)
    total_bytes=$((total_bytes + fsize))

    log "  Uploading ${fname} ($(numfmt --to=iec "$fsize" 2>/dev/null || echo "${fsize}B"))…"

    # Copy file into minio container then upload
    dry docker cp "$local_path" "acas-minio:/tmp/${fname}"
    dry dc_exec "acas-minio" \
      mc cp "/tmp/${fname}" "${MINIO_ALIAS}/${DEST_PREFIX}_${fname}" \
      --quiet
    dry dc_exec "acas-minio" rm -f "/tmp/${fname}"
  done

  ok "All files uploaded ($(numfmt --to=iec "$total_bytes" 2>/dev/null || echo "${total_bytes}B") total)"
}

# ─────────────────────────────────────────────────────────────────────────────
# 5. Write and upload manifest
# ─────────────────────────────────────────────────────────────────────────────
write_manifest() {
  local end_epoch; end_epoch=$(date +%s)
  local duration=$((end_epoch - START_EPOCH))

  # Build files list JSON
  local files_json=""
  for key in "${!MANIFEST_FILES[@]}"; do
    local fname="${MANIFEST_FILES[$key]}"
    local local_path="${WORK_DIR}/${fname}"
    local fsize=0
    [[ -f "$local_path" ]] && fsize=$(stat -c%s "$local_path" 2>/dev/null || echo 0)
    files_json+="    {\"type\": \"${key}\", \"filename\": \"${fname}\", \"size_bytes\": ${fsize}},\n"
  done
  # Remove trailing comma+newline
  files_json="${files_json%,*}"

  cat > "$MANIFEST_FILE" <<EOF
{
  "backup_id":      "${BACKUP_PREFIX}",
  "timestamp":      "${TIMESTAMP}",
  "hostname":       "${HOSTNAME_TAG}",
  "backup_type":    "${BACKUP_TYPE}",
  "duration_s":     ${duration},
  "minio_prefix":   "${DEST_PREFIX}",
  "retention_days": ${RETENTION_DAYS},
  "files": [
$(echo -e "$files_json")
  ]
}
EOF

  if [[ $DRY_RUN -eq 0 ]]; then
    docker cp "$MANIFEST_FILE" "acas-minio:/tmp/manifest.json"
    dc_exec "acas-minio" \
      mc cp "/tmp/manifest.json" "${MINIO_ALIAS}/${BUCKET}/${DATE_PATH}/${HOSTNAME_TAG}_${TIMESTAMP}_manifest.json" \
      --quiet
    dc_exec "acas-minio" rm -f "/tmp/manifest.json"
    ok "Manifest uploaded"
  fi
}

# ─────────────────────────────────────────────────────────────────────────────
# 6. Retention cleanup (delete backups older than RETENTION_DAYS)
# ─────────────────────────────────────────────────────────────────────────────
cleanup_old_backups() {
  log "Cleaning up backups older than ${RETENTION_DAYS} days…"
  local cutoff_date; cutoff_date=$(date -d "${RETENTION_DAYS} days ago" +%Y/%m/%d 2>/dev/null || \
                                   date -v "-${RETENTION_DAYS}d" +%Y/%m/%d 2>/dev/null || echo "")

  if [[ -z "$cutoff_date" ]]; then
    warn "Could not compute cutoff date — skipping retention cleanup"
    return
  fi

  # List objects older than cutoff and remove them
  # mc find returns paths; filter by date prefix
  dry dc_exec "acas-minio" \
    mc find "${MINIO_ALIAS}/${BUCKET}" \
      --older-than "${RETENTION_DAYS}d" \
      --print | while read -r obj; do
    log "  Removing old backup: ${obj}"
    dc_exec "acas-minio" mc rm --quiet "$obj" 2>/dev/null || true
  done
  ok "Retention cleanup done"
}

# ─────────────────────────────────────────────────────────────────────────────
# 7. Kafka notification (optional)
# ─────────────────────────────────────────────────────────────────────────────
send_notification() {
  [[ $DO_NOTIFY -eq 0 ]] && return
  log "Publishing backup completion notification to Kafka…"

  local kafka_url="${KAFKA_BOOTSTRAP_SERVERS:-kafka:19092}"
  local payload
  payload=$(printf '{"type":"backup_complete","hostname":"%s","timestamp":"%s","backup_type":"%s"}' \
    "$HOSTNAME_TAG" "$TIMESTAMP" "$BACKUP_TYPE")

  dry docker run --rm --network acas-prod-net apache/kafka:3.9.2 \
    sh -c "echo '${payload}' | /opt/kafka/bin/kafka-console-producer.sh \
      --bootstrap-server ${kafka_url} \
      --topic notifications.outbound" 2>/dev/null || warn "Kafka notification skipped"
}

# ─────────────────────────────────────────────────────────────────────────────
# EXECUTE
# ─────────────────────────────────────────────────────────────────────────────
case "$BACKUP_TYPE" in
  full)
    backup_postgres
    backup_faiss_meta
    backup_config
    ;;
  db-only)
    backup_postgres
    ;;
  config-only)
    backup_config
    ;;
  *) die "Unknown backup type: ${BACKUP_TYPE}. Use full|db-only|config-only" ;;
esac

upload_to_minio
write_manifest
cleanup_old_backups
send_notification

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
END_EPOCH=$(date +%s)
DURATION=$((END_EPOCH - START_EPOCH))

echo ""
echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${GREEN}║  Backup Complete                                             ║${NC}"
echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo -e "  Timestamp : ${TIMESTAMP}"
echo -e "  Type      : ${BACKUP_TYPE}"
echo -e "  Duration  : ${DURATION}s"
echo -e "  Location  : s3://${DEST_PREFIX}*"
echo -e "  Retention : ${RETENTION_DAYS} days"
echo ""
echo -e "${DIM}  To restore:  bash scripts/restore.sh --backup-date $(date +%Y-%m-%d)${NC}"
echo ""

exit 0
