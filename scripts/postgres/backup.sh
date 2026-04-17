#!/bin/sh
# Daily PostgreSQL backup — runs inside the postgres-backup container via cron.
# PGHOST, PGUSER, PGDATABASE, PGPASSWORD, BACKUP_RETAIN are injected by docker-compose.
set -e

BACKUP_DIR="${BACKUP_DIR:-/backups}"
RETAIN="${BACKUP_RETAIN:-14}"   # days
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
FILENAME="${BACKUP_DIR}/acas_full_${TIMESTAMP}.sql.gz"

mkdir -p "$BACKUP_DIR"

echo "[$(date)] Starting backup → ${FILENAME}"

# Full cluster dump (all databases + roles) compressed
pg_dumpall \
  --host="$PGHOST" \
  --username="$PGUSER" \
  --no-password \
  --clean \
  --if-exists \
  | gzip -9 > "$FILENAME"

SIZE=$(du -sh "$FILENAME" | cut -f1)
echo "[$(date)] Backup complete: ${SIZE}"

# Prune backups older than RETAIN days
echo "[$(date)] Pruning backups older than ${RETAIN} days…"
find "$BACKUP_DIR" -maxdepth 1 -name "acas_full_*.sql.gz" \
     -mtime "+${RETAIN}" -type f -delete -print | \
     sed 's/^/  Deleted: /'

REMAINING=$(find "$BACKUP_DIR" -name "acas_full_*.sql.gz" | wc -l)
echo "[$(date)] Backup retention: ${REMAINING} files kept"
