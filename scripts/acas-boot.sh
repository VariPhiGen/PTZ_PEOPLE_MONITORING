#!/usr/bin/env bash
# ACAS Boot Script — called by systemd on startup.
# Brings up the Docker Compose stack without re-running migrations or re-pulling images.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec docker compose \
  -f "${PROJECT_ROOT}/docker-compose.yml" \
  -f "${PROJECT_ROOT}/docker-compose.override.yml" \
  --env-file "${PROJECT_ROOT}/.env" \
  up -d --remove-orphans
