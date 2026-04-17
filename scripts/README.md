# scripts/

| Path | Purpose |
|------|---------|
| `postgres/init-extensions.sql` | Runs once on first DB init: enables `vector` and `pg_trgm`. |
| `timescaledb/init-detection-log.sql` | Runs once on first DB init: creates `detection_log` hypertable. |
