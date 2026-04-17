"""
ACAS Prometheus Metrics
========================
All application metrics are defined here as module-level singletons so every
part of the codebase can import and record without circular dependencies.

Exposed at GET /metrics via the prometheus_client ASGI app mounted in main.py.

Metric inventory
────────────────
pipeline_latency_seconds    Histogram  per-stage inference latency
faces_recognized_total      Counter    recognition outcomes
face_hunt_total             Counter    face-hunt attempts and outcomes
sessions_active             Gauge      live PTZ-Brain sessions
attendance_records_total    Counter    attendance writes by status
kafka_produce_total         Counter    Kafka produce calls by topic+cluster+status
camera_status               Gauge      1=ONLINE 0=OFFLINE per camera
gpu_util_percent            Gauge      nvidia-smi GPU utilisation
gpu_vram_used_bytes         Gauge      VRAM in use
faiss_index_size            Gauge      vectors in FAISS per client
db_pool_checkedout          Gauge      SQLAlchemy pool connections in use
http_request_duration_ms    Histogram  API endpoint latency (added by PerfMiddleware)
ws_events_buffered_total    Counter    WebSocket events pushed / dropped
"""
from __future__ import annotations

import subprocess
import time
from typing import Any

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    multiprocess,
)

# Use the default (non-multiprocess) registry for single-process deployments.
# If PROMETHEUS_MULTIPROC_DIR is set uvicorn/gunicorn workers share metrics
# via the multi-process collector automatically.
REGISTRY = CollectorRegistry(auto_describe=False)

try:
    from prometheus_client import REGISTRY as _DEFAULT
    REGISTRY = _DEFAULT   # fall back to default for standard deployments
except ImportError:
    pass


# ─── Inference pipeline ────────────────────────────────────────────────────────

PIPELINE_LATENCY = Histogram(
    "acas_pipeline_latency_seconds",
    "Per-stage AI inference latency",
    labelnames=["stage"],   # yolo | retina | align | adaface | liveness | total
    buckets=[0.005, 0.010, 0.018, 0.025, 0.050, 0.100, 0.200, 0.500, 1.0],
)

FACES_RECOGNIZED = Counter(
    "acas_faces_recognized_total",
    "Face recognition outcomes",
    labelnames=["client_id", "outcome"],   # KNOWN | UNKNOWN | LIVENESS_FAIL | LOW_QUALITY
)

FACE_HUNT = Counter(
    "acas_face_hunt_total",
    "PTZ face-hunt attempts",
    labelnames=["client_id", "outcome"],   # SUCCESS | FAIL | TIMEOUT | BUDGET_EXCEEDED
)

FACE_HUNT_DURATION = Histogram(
    "acas_face_hunt_duration_seconds",
    "Duration of a face-hunt manoeuvre",
    buckets=[1, 2, 5, 8, 10, 15, 20, 30],
)


# ─── Sessions ─────────────────────────────────────────────────────────────────

SESSIONS_ACTIVE = Gauge(
    "acas_sessions_active",
    "Number of currently running PTZ-Brain sessions",
    labelnames=["client_id", "mode"],   # ATTENDANCE | MONITORING | BOTH
)

SESSION_CYCLE_DURATION = Histogram(
    "acas_session_cycle_duration_seconds",
    "Time to complete one full scan cycle",
    labelnames=["client_id", "camera_id"],
    buckets=[5, 10, 15, 20, 30, 45, 60, 90, 120],
)

SESSION_RECOGNITION_RATE = Gauge(
    "acas_session_recognition_rate",
    "Fraction of detected persons identified in last cycle (0–1)",
    labelnames=["session_id"],
)


# ─── Attendance ───────────────────────────────────────────────────────────────

ATTENDANCE_RECORDS = Counter(
    "acas_attendance_records_total",
    "Attendance records written by status",
    labelnames=["client_id", "status"],   # P | L | EE | A | ND | EX
)

ATTENDANCE_SYNC_LAG = Histogram(
    "acas_attendance_sync_lag_seconds",
    "Seconds between session end and ERP sync",
    buckets=[1, 5, 10, 30, 60, 300, 600, 1800],
)


# ─── Kafka ─────────────────────────────────────────────────────────────────────

KAFKA_PRODUCE = Counter(
    "acas_kafka_produce_total",
    "Kafka produce calls",
    labelnames=["topic", "cluster", "status"],   # cluster: local|central; status: ok|error
)

KAFKA_PRODUCE_LATENCY = Histogram(
    "acas_kafka_produce_latency_seconds",
    "Kafka produce round-trip latency",
    labelnames=["topic"],
    buckets=[0.001, 0.005, 0.010, 0.025, 0.050, 0.100, 0.250],
)

KAFKA_CONSUMER_LAG = Gauge(
    "acas_kafka_consumer_lag",
    "Consumer group lag (messages behind)",
    labelnames=["topic", "group"],
)


# ─── Cameras ──────────────────────────────────────────────────────────────────

CAMERA_STATUS = Gauge(
    "acas_camera_status",
    "Camera reachability: 1=ONLINE 0=OFFLINE 0.5=DEGRADED",
    labelnames=["camera_id", "client_id", "room_name"],
)

CAMERA_FPS = Gauge(
    "acas_camera_fps_actual",
    "Measured RTSP decode frame-rate",
    labelnames=["camera_id"],
)

ONVIF_ERRORS = Counter(
    "acas_onvif_errors_total",
    "ONVIF command failures",
    labelnames=["camera_id", "command"],
)


# ─── GPU / hardware ───────────────────────────────────────────────────────────

GPU_UTIL = Gauge(
    "acas_gpu_util_percent",
    "GPU compute utilisation percent",
    labelnames=["device_id"],
)

GPU_VRAM_USED = Gauge(
    "acas_gpu_vram_used_bytes",
    "GPU VRAM currently in use (bytes)",
    labelnames=["device_id"],
)

GPU_VRAM_TOTAL = Gauge(
    "acas_gpu_vram_total_bytes",
    "Total GPU VRAM (bytes)",
    labelnames=["device_id"],
)

GPU_DEGRADATION_MODE = Gauge(
    "acas_gpu_degradation_mode",
    "Active GPU degradation level: 0=FULL 1=NO_LIVENESS 2=EMBED_ONLY 3=YOLO_ONLY",
    labelnames=["device_id"],
)


# ─── FAISS / face repository ──────────────────────────────────────────────────

FAISS_INDEX_SIZE = Gauge(
    "acas_faiss_index_size",
    "Number of vectors in the FAISS index for a client",
    labelnames=["client_id"],
)

FAISS_SEARCH_LATENCY = Histogram(
    "acas_faiss_search_latency_seconds",
    "FAISS kNN search latency",
    labelnames=["client_id", "tier"],   # tier: gpu | pgvector
    buckets=[0.0005, 0.001, 0.002, 0.005, 0.010, 0.025, 0.050],
)

ENROLLMENT_QUALITY_FAIL = Counter(
    "acas_enrollment_quality_fail_total",
    "Images rejected at enrollment quality gate",
    labelnames=["client_id", "reason"],   # no_face | blur | occlusion | small_iod | multi_face
)


# ─── Database pool ────────────────────────────────────────────────────────────

DB_POOL_CHECKED_OUT = Gauge(
    "acas_db_pool_checkedout",
    "SQLAlchemy DB connections currently checked out",
)

DB_POOL_OVERFLOW = Gauge(
    "acas_db_pool_overflow",
    "SQLAlchemy DB overflow connections currently open",
)


# ─── HTTP API ─────────────────────────────────────────────────────────────────

HTTP_REQUEST_DURATION = Histogram(
    "acas_http_request_duration_ms",
    "API endpoint response time in milliseconds",
    labelnames=["method", "path_template", "status"],
    buckets=[5, 10, 25, 50, 100, 200, 500, 1000, 5000],
)

HTTP_CACHE_HIT = Counter(
    "acas_http_cache_hit_total",
    "Analytics responses served from Redis cache",
    labelnames=["path_template"],
)


# ─── WebSocket ────────────────────────────────────────────────────────────────

WS_EVENTS_PUSHED = Counter(
    "acas_ws_events_pushed_total",
    "Events successfully delivered via WebSocket",
    labelnames=["session_id"],
)

WS_EVENTS_DROPPED = Counter(
    "acas_ws_events_dropped_total",
    "Events dropped due to buffer overflow (back-pressure)",
    labelnames=["session_id"],
)


# ─── Recording helpers ────────────────────────────────────────────────────────

def record_pipeline_stage(stage: str, elapsed_s: float) -> None:
    PIPELINE_LATENCY.labels(stage=stage).observe(elapsed_s)


def record_recognition(client_id: str, outcome: str) -> None:
    """outcome: KNOWN | UNKNOWN | LIVENESS_FAIL | LOW_QUALITY"""
    FACES_RECOGNIZED.labels(client_id=client_id, outcome=outcome).inc()


def record_kafka_produce(topic: str, cluster: str, success: bool) -> None:
    KAFKA_PRODUCE.labels(
        topic=topic,
        cluster=cluster,
        status="ok" if success else "error",
    ).inc()


def set_camera_status(camera_id: str, client_id: str, room_name: str, status: str) -> None:
    """status: ONLINE | OFFLINE | DEGRADED"""
    value_map = {"ONLINE": 1.0, "DEGRADED": 0.5, "OFFLINE": 0.0}
    CAMERA_STATUS.labels(camera_id=camera_id, client_id=client_id, room_name=room_name).set(
        value_map.get(status, 0.0)
    )


# ─── GPU poller (run as a background asyncio task) ───────────────────────────

async def gpu_metrics_loop(device_id: int = 0, interval_s: float = 10.0) -> None:
    """
    Periodically samples nvidia-smi and updates GPU metrics.
    Run as asyncio.create_task() in the lifespan.
    """
    import asyncio as _asyncio
    dev = str(device_id)
    while True:
        try:
            out = await _asyncio.to_thread(
                lambda: subprocess.check_output(
                    [
                        "nvidia-smi",
                        f"--id={device_id}",
                        "--query-gpu=utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    timeout=5,
                    stderr=subprocess.DEVNULL,
                ).decode().strip()
            )
            util_s, used_s, total_s = [x.strip() for x in out.split(",")]
            GPU_UTIL.labels(device_id=dev).set(float(util_s))
            GPU_VRAM_USED.labels(device_id=dev).set(float(used_s) * 1024 * 1024)
            GPU_VRAM_TOTAL.labels(device_id=dev).set(float(total_s) * 1024 * 1024)
        except Exception:
            pass
        await _asyncio.sleep(interval_s)


# ─── DB pool scraper ─────────────────────────────────────────────────────────

def scrape_db_pool(engine: Any) -> None:
    """Call from a periodic task; reads SQLAlchemy pool stats."""
    try:
        pool = engine.sync_engine.pool
        DB_POOL_CHECKED_OUT.set(pool.checkedout())
        DB_POOL_OVERFLOW.set(pool.overflow())
    except Exception:
        pass


# ─── ASGI /metrics endpoint ──────────────────────────────────────────────────

async def metrics_endpoint(scope: Any, receive: Any, send: Any) -> None:
    """Bare ASGI app serving Prometheus text format at /metrics."""
    if scope["type"] != "http":
        return
    body = generate_latest()
    await send({
        "type": "http.response.start",
        "status": 200,
        "headers": [
            (b"content-type", CONTENT_TYPE_LATEST.encode()),
            (b"content-length", str(len(body)).encode()),
        ],
    })
    await send({"type": "http.response.body", "body": body})
