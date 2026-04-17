"""
Kafka Config API — topic management, health checks, and test produce/consume.

All routes require system:admin (SUPER_ADMIN).

GET  /api/kafka/config        — read current bootstrap + topic config from app.state
PUT  /api/kafka/config        — update runtime config (persisted to Redis)
GET  /api/kafka/topics        — list topics with metadata (partition count, offsets)
GET  /api/kafka/health        — broker liveness + consumer group lag
POST /api/kafka/test          — produce a test message and verify it is delivered
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.api._shared import audit, now_epoch
from app.middleware.auth import require_permission

router = APIRouter(prefix="/api/kafka", tags=["kafka"])

_SA = require_permission("system:admin")

_CONFIG_REDIS_KEY = "acas:kafka:config"


# ── Schemas ───────────────────────────────────────────────────────────────────

class KafkaConfigUpdate(BaseModel):
    acks:                  str | None = None   # "all" | "1" | "0"
    compression_type:      str | None = None   # "snappy" | "gzip" | "lz4" | "none"
    retries:               int | None = None
    retry_backoff_ms:      int | None = None
    message_max_bytes:     int | None = None
    request_timeout_ms:    int | None = None
    schema_registry_url:   str | None = None


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _get_config_from_redis(redis: Any) -> dict:
    raw = await redis.get(_CONFIG_REDIS_KEY)
    return json.loads(raw) if raw else {}


def _topic_metadata(producer: Any, topic_name: str) -> dict:
    """Use librdkafka metadata API to get partition count and leader info."""
    try:
        meta   = producer.list_topics(topic=topic_name, timeout=5)
        topics = meta.topics
        if topic_name not in topics:
            return {"topic": topic_name, "error": "not_found"}
        t = topics[topic_name]
        return {
            "topic":           topic_name,
            "partition_count": len(t.partitions),
            "error":           str(t.error) if t.error else None,
            "partitions": [
                {
                    "id":     pid,
                    "leader": p.leader,
                    "replicas": list(p.replicas),
                    "isrs":   list(p.isrs),
                }
                for pid, p in t.partitions.items()
            ],
        }
    except Exception as exc:
        return {"topic": topic_name, "error": str(exc)}


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/config", dependencies=[_SA])
async def get_config(request: Request) -> dict:
    """Return current Kafka configuration (bootstrap servers + overrides)."""
    settings = request.app.state.settings
    redis    = request.app.state.redis
    overrides = await _get_config_from_redis(redis)
    return {
        "bootstrap_servers":  settings.kafka_bootstrap_servers,
        "schema_registry_url": getattr(settings, "schema_registry_url", None),
        "overrides":          overrides,
        "producer_config": {
            "acks":               "all",
            "enable.idempotence": True,
        },
    }


@router.put("/config", dependencies=[_SA])
async def update_config(body: KafkaConfigUpdate, request: Request) -> dict:
    """
    Persist Kafka configuration overrides to Redis.
    The new config takes effect on next producer/consumer initialisation.
    """
    redis    = request.app.state.redis
    current  = await _get_config_from_redis(redis)
    updates  = body.model_dump(exclude_none=True)
    current.update(updates)
    await redis.set(_CONFIG_REDIS_KEY, json.dumps(current))
    audit(request, "UPDATE_KAFKA_CONFIG", "system", "kafka", updates)
    return {"status": "saved", "config": current}


@router.get("/topics", dependencies=[_SA])
async def list_topics(request: Request) -> dict:
    """List all topics known to the broker with partition metadata."""
    producer = request.app.state.kafka_producer
    try:
        meta   = await asyncio.to_thread(producer.list_topics, timeout=10)
        topics = meta.topics
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Kafka broker unreachable: {exc}")

    # Exclude internal __consumer_offsets topic
    items = [
        _topic_metadata(producer, name)
        for name in sorted(topics.keys())
        if not name.startswith("__")
    ]
    return {"total": len(items), "topics": items}


@router.get("/health", dependencies=[_SA])
async def kafka_health(request: Request) -> dict:
    """
    Broker liveness and consumer group lag for all ACAS groups.
    """
    from confluent_kafka.admin import AdminClient

    settings = request.app.state.settings
    try:
        admin = AdminClient({"bootstrap.servers": settings.kafka_bootstrap_servers})
        t0    = time.monotonic()
        meta  = await asyncio.to_thread(admin.list_topics, timeout=5)
        latency_ms = round((time.monotonic() - t0) * 1000, 2)
        broker_count = len(meta.brokers)
        topic_count  = len([t for t in meta.topics if not t.startswith("__")])
        return {
            "status":       "ok",
            "broker_count": broker_count,
            "topic_count":  topic_count,
            "latency_ms":   latency_ms,
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


@router.post("/test", dependencies=[_SA])
async def kafka_test(request: Request) -> dict:
    """
    Produce one test message to 'system.alerts' and confirm delivery
    via the librdkafka delivery callback within 10 s.
    """
    producer = request.app.state.kafka_producer
    msg_id   = uuid.uuid4().hex
    delivered: dict[str, Any] = {}
    error_info: dict[str, Any] = {}

    def _cb(err: Any, msg: Any) -> None:
        if err:
            error_info["err"] = str(err)
        else:
            delivered["topic"]     = msg.topic()
            delivered["partition"] = msg.partition()
            delivered["offset"]    = msg.offset()

    try:
        producer.produce(
            "system.alerts",
            key=b"test",
            value=json.dumps({
                "alert_id":  msg_id,
                "severity":  "TEST",
                "component": "api",
                "message":   "Kafka test message",
                "ts":        time.time(),
            }).encode(),
            on_delivery=_cb,
        )
        remaining = await asyncio.to_thread(producer.flush, timeout=10)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Produce failed: {exc}")

    if error_info:
        return {"success": False, "msg_id": msg_id, "error": error_info.get("err")}
    if not delivered:
        return {"success": False, "msg_id": msg_id, "error": "timeout — no delivery confirmation"}

    return {"success": True, "msg_id": msg_id, "delivery": delivered}
