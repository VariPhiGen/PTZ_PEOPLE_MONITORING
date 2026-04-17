-- TimescaleDB compression and retention policies for detection_log.
-- Runs at first DB init (docker-entrypoint-initdb.d), after the hypertable is created.

-- Enable compression on the hypertable, segmenting by camera_id for efficient
-- per-camera queries on compressed chunks.
ALTER TABLE detection_log SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'camera_id',
    timescaledb.compress_orderby   = 'time DESC'
);

-- Compress chunks older than 7 days automatically (background job).
SELECT add_compression_policy(
    'detection_log',
    compress_after   => INTERVAL '7 days',
    if_not_exists    => TRUE
);

-- Drop chunks older than 90 days automatically (background job).
SELECT add_retention_policy(
    'detection_log',
    drop_after       => INTERVAL '90 days',
    if_not_exists    => TRUE
);
