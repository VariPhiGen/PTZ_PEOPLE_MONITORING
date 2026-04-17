-- TimescaleDB hypertable for detection logs (runs on first DB init only)
CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS detection_log (
    time TIMESTAMPTZ NOT NULL DEFAULT now(),
    node_id TEXT,
    camera_id TEXT,
    track_id TEXT,
    label TEXT,
    score DOUBLE PRECISION,
    payload JSONB
);

SELECT public.create_hypertable('detection_log', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS detection_log_camera_time_idx
  ON detection_log (camera_id, time DESC);
