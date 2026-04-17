"""
KafkaProducer — Avro-serialised multi-topic event bus for ACAS.

Two logical clusters
────────────────────
  local   — intra-datacenter (Kafka broker at bootstrap_servers)
  central — optional second cluster for central aggregation; falls back to
            local if central_bootstrap_servers is not configured.

13 Topics
─────────
  attendance.records     Approved full attendance payloads → ERP / dashboard
  attendance.held        Held sessions awaiting manual review
  attendance.faculty     Faculty presence summaries
  sightings.log          Closed sighting events (MONITORING mode)
  sightings.alerts       Restricted-zone / unknown-person alerts
  sightings.occupancy    Camera-level headcount snapshots
  erp.sync.requests      Outbound ERP write payloads
  erp.sync.status        Acknowledgements / errors from ERP webhook worker
  enrollment.events      New enrollment and template-update notifications
  notifications.outbound Email / push notifications (password reset, alerts)
  system.alerts          P0 camera failures, DB errors, model degradation
  audit.events           Admin actions, overrides, config changes
  admin.overrides        Inbound admin commands (force-sync, discard, re-hold)

All messages are Avro-serialised using the Confluent Schema Registry.
If the Schema Registry is unavailable or a schema fetch fails, the message
is serialised as plain JSON (fallback mode) so the application never blocks.

Usage
─────
    producer = KafkaProducer(
        bootstrap_servers     = "kafka:19092",
        schema_registry_url   = "http://schema-registry:8081",
    )
    producer.publish_attendance(session_result)
    producer.publish_sighting(closed_sighting)
    producer.publish_alert("P0_CAMERA_OFFLINE", {...})
    producer.flush()        # call at shutdown
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any

from confluent_kafka import Producer as _CKProducer, KafkaException

logger = logging.getLogger(__name__)

# ── Try to import Avro support; degrade gracefully if unavailable ─────────────
try:
    from confluent_kafka.schema_registry import SchemaRegistryClient
    from confluent_kafka.schema_registry.avro import AvroSerializer, AvroDeserializer
    from confluent_kafka.serialization import (
        MessageField,
        SerializationContext,
        StringSerializer,
    )
    _AVRO_AVAILABLE = True
except ImportError:
    _AVRO_AVAILABLE = False
    logger.warning("confluent_kafka Avro support unavailable — using JSON fallback")


# ── Avro schemas (inline; can be replaced by registry lookups) ────────────────

_SCHEMAS: dict[str, str] = {
    "attendance.records": json.dumps({
        "type": "record",
        "name": "AttendanceRecord",
        "namespace": "acas.attendance",
        "fields": [
            {"name": "result_id",        "type": "string"},
            {"name": "session_id",       "type": "string"},
            {"name": "client_id",        "type": "string"},
            {"name": "sync_status",      "type": "string"},
            {"name": "total_roster",     "type": "int"},
            {"name": "present_count",    "type": "int"},
            {"name": "late_count",       "type": "int"},
            {"name": "early_exit_count", "type": "int"},
            {"name": "absent_count",     "type": "int"},
            {"name": "nd_count",         "type": "int"},
            {"name": "recognition_rate", "type": "double"},
            {"name": "records_json",     "type": "string"},  # serialised list
            {"name": "ts",               "type": "double"},
        ],
    }),
    "attendance.held": json.dumps({
        "type": "record",
        "name": "AttendanceHeld",
        "namespace": "acas.attendance",
        "fields": [
            {"name": "result_id",    "type": "string"},
            {"name": "session_id",   "type": "string"},
            {"name": "client_id",    "type": "string"},
            {"name": "held_reason",  "type": "string"},
            {"name": "faculty_status","type": "string"},
            {"name": "records_json", "type": "string"},
            {"name": "ts",           "type": "double"},
        ],
    }),
    "attendance.faculty": json.dumps({
        "type": "record",
        "name": "FacultyEvent",
        "namespace": "acas.attendance",
        "fields": [
            {"name": "result_id",     "type": "string"},
            {"name": "session_id",    "type": "string"},
            {"name": "client_id",     "type": "string"},
            {"name": "person_id",     "type": ["null", "string"], "default": None},
            {"name": "status",        "type": "string"},
            {"name": "pct_present",   "type": "double"},
            {"name": "total_seconds", "type": "double"},
            {"name": "held_reason",   "type": ["null", "string"], "default": None},
            {"name": "ts",            "type": "double"},
        ],
    }),
    "sightings.log": json.dumps({
        "type": "record",
        "name": "SightingEvent",
        "namespace": "acas.sightings",
        "fields": [
            {"name": "sighting_id",   "type": "string"},
            {"name": "client_id",     "type": "string"},
            {"name": "camera_id",     "type": "string"},
            {"name": "person_id",     "type": "string"},
            {"name": "zone",          "type": ["null", "string"], "default": None},
            {"name": "first_seen",    "type": "double"},
            {"name": "last_seen",     "type": "double"},
            {"name": "duration_s",    "type": "int"},
            {"name": "confidence_avg","type": "double"},
            {"name": "frame_refs",    "type": {"type": "array", "items": "string"},
             "default": []},
            {"name": "ts",            "type": "double"},
        ],
    }),
    "sightings.alerts": json.dumps({
        "type": "record",
        "name": "SightingAlert",
        "namespace": "acas.sightings",
        "fields": [
            {"name": "alert_id",   "type": "string"},
            {"name": "client_id",  "type": "string"},
            {"name": "camera_id",  "type": "string"},
            {"name": "person_id",  "type": ["null", "string"], "default": None},
            {"name": "alert_type", "type": "string"},
            {"name": "detail",     "type": "string"},
            {"name": "ts",         "type": "double"},
        ],
    }),
    "sightings.occupancy": json.dumps({
        "type": "record",
        "name": "OccupancySnapshot",
        "namespace": "acas.sightings",
        "fields": [
            {"name": "snapshot_id", "type": "string"},
            {"name": "client_id",   "type": "string"},
            {"name": "camera_id",   "type": "string"},
            {"name": "count",       "type": "int"},
            {"name": "ts",          "type": "double"},
        ],
    }),
    "erp.sync.requests": json.dumps({
        "type": "record",
        "name": "ErpSyncRequest",
        "namespace": "acas.erp",
        "fields": [
            {"name": "request_id",  "type": "string"},
            {"name": "session_id",  "type": "string"},
            {"name": "client_id",   "type": "string"},
            {"name": "payload_json","type": "string"},
            {"name": "ts",          "type": "double"},
        ],
    }),
    "erp.sync.status": json.dumps({
        "type": "record",
        "name": "ErpSyncStatus",
        "namespace": "acas.erp",
        "fields": [
            {"name": "request_id", "type": "string"},
            {"name": "status",     "type": "string"},  # ACK / ERROR / RETRY
            {"name": "detail",     "type": ["null", "string"], "default": None},
            {"name": "ts",         "type": "double"},
        ],
    }),
    "enrollment.events": json.dumps({
        "type": "record",
        "name": "EnrollmentEvent",
        "namespace": "acas.enrollment",
        "fields": [
            {"name": "event_id",   "type": "string"},
            {"name": "client_id",  "type": "string"},
            {"name": "person_id",  "type": "string"},
            {"name": "event_type", "type": "string"},  # ENROLLED / UPDATED / DELETED
            {"name": "quality",    "type": "double"},
            {"name": "ts",         "type": "double"},
        ],
    }),
    "notifications.outbound": json.dumps({
        "type": "record",
        "name": "Notification",
        "namespace": "acas.notifications",
        "fields": [
            {"name": "notification_id", "type": "string"},
            {"name": "recipient_id",    "type": "string"},
            {"name": "channel",         "type": "string"},  # email / push / sms
            {"name": "template",        "type": "string"},
            {"name": "payload_json",    "type": "string"},
            {"name": "ts",              "type": "double"},
        ],
    }),
    "system.alerts": json.dumps({
        "type": "record",
        "name": "SystemAlert",
        "namespace": "acas.system",
        "fields": [
            {"name": "alert_id",  "type": "string"},
            {"name": "severity",  "type": "string"},  # P0 / P1 / P2
            {"name": "component", "type": "string"},
            {"name": "message",   "type": "string"},
            {"name": "detail",    "type": ["null", "string"], "default": None},
            {"name": "ts",        "type": "double"},
        ],
    }),
    "audit.events": json.dumps({
        "type": "record",
        "name": "AuditEvent",
        "namespace": "acas.audit",
        "fields": [
            {"name": "audit_id",    "type": "string"},
            {"name": "client_id",   "type": ["null", "string"], "default": None},
            {"name": "actor_id",    "type": "string"},
            {"name": "actor_type",  "type": "string"},
            {"name": "action",      "type": "string"},
            {"name": "target_type", "type": "string"},
            {"name": "target_id",   "type": "string"},
            {"name": "detail_json", "type": ["null", "string"], "default": None},
            {"name": "ts",          "type": "double"},
        ],
    }),
    "admin.overrides": json.dumps({
        "type": "record",
        "name": "AdminOverride",
        "namespace": "acas.admin",
        "fields": [
            {"name": "override_id", "type": "string"},
            {"name": "session_id",  "type": "string"},
            {"name": "client_id",   "type": "string"},
            {"name": "action",      "type": "string"},  # FORCE_SYNC / DISCARD / RE_HOLD
            {"name": "actor_id",    "type": "string"},
            {"name": "reason",      "type": ["null", "string"], "default": None},
            {"name": "ts",          "type": "double"},
        ],
    }),
    # Cross-node face embedding synchronisation
    "repo.sync.embeddings": json.dumps({
        "type": "record",
        "name": "EmbeddingSync",
        "namespace": "acas.repo",
        "fields": [
            {"name": "event_id",        "type": "string"},
            {"name": "event_type",      "type": "string"},    # ENROLL | UPDATE | DELETE
            {"name": "client_id",       "type": "string"},
            {"name": "person_id",       "type": "string"},
            {"name": "source_node",     "type": "string"},    # originating node_id
            {"name": "embedding_json",  "type": "string"},    # JSON-encoded float[]
            {"name": "version",         "type": "int",        "default": 1},
            {"name": "confidence_avg",  "type": "double",     "default": 0.0},
            {"name": "quality_score",   "type": "double",     "default": 0.0},
            {"name": "metadata_json",   "type": ["null", "string"], "default": None},
            {"name": "ts",              "type": "double"},
        ],
    }),
}

# Topics routed to the CENTRAL cluster (written to both local AND central).
# repo.sync.embeddings must reach all nodes — use central cluster for fan-out.
_CENTRAL_TOPICS = {
    "attendance.records",
    "attendance.held",
    "attendance.faculty",
    "erp.sync.requests",
    "repo.sync.embeddings",
}

# Topics that are HIGH-FREQUENCY and must stay on the local cluster only
# (detection-level telemetry — too noisy for central).
_LOCAL_ONLY_TOPICS = {
    "sightings.log",
    "sightings.occupancy",
}


# ── Delivery callbacks ────────────────────────────────────────────────────────

def _on_delivery(err: Any, msg: Any) -> None:
    if err:
        logger.error(
            "Kafka delivery error  topic=%s  err=%s",
            msg.topic() if msg else "?", err,
        )
    else:
        logger.debug(
            "Kafka delivered  topic=%s  partition=%d  offset=%d",
            msg.topic(), msg.partition(), msg.offset(),
        )


# ── Main producer class ───────────────────────────────────────────────────────

class KafkaProducer:
    """
    Avro-serialised Kafka producer for all ACAS event topics.

    Parameters
    ----------
    bootstrap_servers       Local Kafka broker list ("kafka:19092").
    schema_registry_url     Confluent Schema Registry URL.
    central_bootstrap_servers
                            Optional second cluster for cross-datacenter events.
    producer_config         Extra librdkafka config (compression, acks, etc.).
    """

    def __init__(
        self,
        bootstrap_servers:          str,
        schema_registry_url:        str | None = None,
        central_bootstrap_servers:  str | None = None,
        producer_config:            dict | None = None,
    ) -> None:
        # Local cluster: optimised for throughput on high-frequency topics
        # (sightings, occupancy, detection logs).  lz4 has lower CPU cost than
        # snappy and similar compression ratio for JSON/Avro payloads.
        base_cfg: dict[str, Any] = {
            "bootstrap.servers":             bootstrap_servers,
            "acks":                          "1",       # leader-only ack — latency wins over durability for real-time telemetry
            "retries":                       3,
            "retry.backoff.ms":              100,
            "compression.type":              "lz4",     # lower CPU than snappy, ~same ratio
            "batch.size":                    65536,     # 64 KB — fill larger batches before send
            "linger.ms":                     5,         # wait up to 5 ms to accumulate a batch
            "queue.buffering.max.messages":  200000,
            "queue.buffering.max.kbytes":    524288,    # 512 MB in-memory buffer
            "delivery.report.only.error":    False,
        }
        if producer_config:
            base_cfg.update(producer_config)

        self._local: _CKProducer = _CKProducer(base_cfg)

        if central_bootstrap_servers:
            # Central cluster: business events (attendance, ERP, enrollment).
            # Use safer settings — all-replicas ack, slightly longer linger for
            # better batching across the WAN link.
            central_cfg = dict(base_cfg)
            central_cfg["bootstrap.servers"] = central_bootstrap_servers
            central_cfg["acks"]              = "all"
            central_cfg["retries"]           = 5
            central_cfg["linger.ms"]         = 10
            self._central: _CKProducer | None = _CKProducer(central_cfg)
        else:
            self._central = None

        # Schema Registry + Avro serialisers (one per topic)
        self._schema_registry: SchemaRegistryClient | None = None
        self._avro_ser: dict[str, AvroSerializer] = {}
        self._key_ser = StringSerializer("utf_8") if _AVRO_AVAILABLE else None

        if _AVRO_AVAILABLE and schema_registry_url:
            try:
                self._schema_registry = SchemaRegistryClient(
                    {"url": schema_registry_url}
                )
                self._build_serialisers()
                logger.info(
                    "KafkaProducer: Avro serialisers ready  SR=%s", schema_registry_url
                )
            except Exception as exc:
                logger.warning(
                    "KafkaProducer: Schema Registry unreachable (%s) — JSON fallback",
                    exc,
                )

        logger.info(
            "KafkaProducer ready  local=%s  central=%s  avro=%s",
            bootstrap_servers,
            central_bootstrap_servers or "none",
            bool(self._avro_ser),
        )

    # ── Schema serialiser factory ─────────────────────────────────────────────

    def _build_serialisers(self) -> None:
        from confluent_kafka.schema_registry import Schema

        for topic, schema_str in _SCHEMAS.items():
            try:
                schema = Schema(schema_str, schema_type="AVRO")
                self._avro_ser[topic] = AvroSerializer(
                    self._schema_registry,
                    schema,
                )
            except Exception as exc:
                logger.warning("Failed to build Avro serialiser for %s: %s", topic, exc)

    # ── High-level publish helpers ────────────────────────────────────────────

    def publish_attendance(self, result: Any) -> None:
        """
        Route a SessionResult to the correct topics:
          APPROVED → attendance.records + attendance.faculty + erp.sync.requests
          HELD     → attendance.held   + attendance.faculty + sightings.alerts
        """
        from app.services.attendance_engine import SessionResult  # local import

        if not isinstance(result, SessionResult):
            logger.error("publish_attendance: expected SessionResult, got %s", type(result))
            return

        now = time.time()
        records_json = json.dumps([
            {
                "record_id":       r.record_id,
                "person_id":       r.person_id,
                "status":          r.status,
                "total_seconds":   r.total_seconds,
                "detection_count": r.detection_count,
                "first_seen":      r.first_seen,
                "last_seen":       r.last_seen,
                "flags":           r.flags,
            }
            for r in result.records
        ])

        fac = result.faculty

        # Always publish faculty event
        self._produce(
            "attendance.faculty",
            key=result.session_id,
            payload={
                "result_id":     result.result_id,
                "session_id":    result.session_id,
                "client_id":     result.client_id,
                "person_id":     fac.person_id,
                "status":        fac.status,
                "pct_present":   fac.pct_present,
                "total_seconds": fac.total_seconds,
                "held_reason":   fac.held_reason,
                "ts":            now,
            },
        )

        if result.sync_status == "APPROVED":
            # attendance.records
            self._produce(
                "attendance.records",
                key=result.session_id,
                payload={
                    "result_id":        result.result_id,
                    "session_id":       result.session_id,
                    "client_id":        result.client_id,
                    "sync_status":      result.sync_status,
                    "total_roster":     result.total_roster,
                    "present_count":    result.present_count,
                    "late_count":       result.late_count,
                    "early_exit_count": result.early_exit_count,
                    "absent_count":     result.absent_count,
                    "nd_count":         result.nd_count,
                    "recognition_rate": result.recognition_rate,
                    "records_json":     records_json,
                    "ts":               now,
                },
            )
            # ERP sync request
            self._produce(
                "erp.sync.requests",
                key=result.session_id,
                payload={
                    "request_id":   str(uuid.uuid4()),
                    "session_id":   result.session_id,
                    "client_id":    result.client_id,
                    "payload_json": records_json,
                    "ts":           now,
                },
            )

        else:  # HELD
            self._produce(
                "attendance.held",
                key=result.session_id,
                payload={
                    "result_id":     result.result_id,
                    "session_id":    result.session_id,
                    "client_id":     result.client_id,
                    "held_reason":   result.held_reason or "",
                    "faculty_status": fac.status,
                    "records_json":  records_json,
                    "ts":            now,
                },
            )
            # Alert for held session
            self._produce(
                "sightings.alerts",
                key=result.client_id,
                payload={
                    "alert_id":   str(uuid.uuid4()),
                    "client_id":  result.client_id,
                    "camera_id":  "",
                    "person_id":  fac.person_id,
                    "alert_type": "SESSION_HELD",
                    "detail":     result.held_reason or "",
                    "ts":         now,
                },
            )

    def publish_sighting(self, sighting: Any) -> None:
        """Publish a ClosedSighting to sightings.log."""
        from app.services.sighting_engine import ClosedSighting  # local import

        if not isinstance(sighting, ClosedSighting):
            logger.error("publish_sighting: expected ClosedSighting, got %s", type(sighting))
            return

        self._produce(
            "sightings.log",
            key=sighting.camera_id,
            payload={
                "sighting_id":    sighting.sighting_id,
                "client_id":      sighting.client_id,
                "camera_id":      sighting.camera_id,
                "person_id":      sighting.person_id,
                "zone":           sighting.zone,
                "first_seen":     sighting.first_seen,
                "last_seen":      sighting.last_seen,
                "duration_s":     sighting.duration_s,
                "confidence_avg": sighting.confidence_avg,
                "frame_refs":     sighting.frame_refs,
                "ts":             time.time(),
            },
        )

    def publish_occupancy(
        self,
        client_id: str,
        camera_id: str,
        count:     int,
    ) -> None:
        self._produce(
            "sightings.occupancy",
            key=camera_id,
            payload={
                "snapshot_id": str(uuid.uuid4()),
                "client_id":   client_id,
                "camera_id":   camera_id,
                "count":       count,
                "ts":          time.time(),
            },
        )

    def publish_alert(
        self,
        severity:  str,
        component: str,
        message:   str,
        detail:    str | None = None,
    ) -> None:
        self._produce(
            "system.alerts",
            key=component,
            payload={
                "alert_id":  str(uuid.uuid4()),
                "severity":  severity,
                "component": component,
                "message":   message,
                "detail":    detail,
                "ts":        time.time(),
            },
        )

    def publish_notification(
        self,
        recipient_id: str,
        channel:      str,
        template:     str,
        payload:      dict,
    ) -> None:
        self._produce(
            "notifications.outbound",
            key=recipient_id,
            payload={
                "notification_id": str(uuid.uuid4()),
                "recipient_id":    recipient_id,
                "channel":         channel,
                "template":        template,
                "payload_json":    json.dumps(payload),
                "ts":              time.time(),
            },
        )

    def publish_enrollment_event(
        self,
        client_id:  str,
        person_id:  str,
        event_type: str,
        quality:    float = 0.0,
    ) -> None:
        self._produce(
            "enrollment.events",
            key=client_id,
            payload={
                "event_id":   str(uuid.uuid4()),
                "client_id":  client_id,
                "person_id":  person_id,
                "event_type": event_type,
                "quality":    quality,
                "ts":         time.time(),
            },
        )

    def publish_audit(
        self,
        actor_id:    str,
        actor_type:  str,
        action:      str,
        target_type: str,
        target_id:   str,
        client_id:   str | None = None,
        detail:      dict | None = None,
    ) -> None:
        self._produce(
            "audit.events",
            key=actor_id,
            payload={
                "audit_id":    str(uuid.uuid4()),
                "client_id":   client_id,
                "actor_id":    actor_id,
                "actor_type":  actor_type,
                "action":      action,
                "target_type": target_type,
                "target_id":   target_id,
                "detail_json": json.dumps(detail) if detail else None,
                "ts":          time.time(),
            },
        )

    def broadcast_config_reload(
        self,
        config:      dict[str, Any],
        target_node: str = "*",
        actor_id:    str = "system",
        client_id:   str = "__all__",
    ) -> None:
        """
        Publish a RELOAD_CONFIG override so all nodes (or a specific node)
        reload their in-memory settings and rebuild derived state.

        Convention: session_id = "__broadcast__" (no real session),
        reason = JSON {"target_node": "...", "config": {...}}.

        Called from PUT /api/settings/* and PUT /api/kafka/config routes.
        """
        self._produce(
            "admin.overrides",
            key=f"config_broadcast/{target_node}",
            payload={
                "override_id": str(uuid.uuid4()),
                "session_id":  "__broadcast__",
                "client_id":   client_id,
                "action":      "RELOAD_CONFIG",
                "actor_id":    actor_id,
                "reason":      json.dumps({
                    "target_node": target_node,
                    "config":      config,
                }),
                "ts":          time.time(),
            },
        )
        logger.info(
            "KafkaProducer: broadcast RELOAD_CONFIG  target=%s  keys=%s",
            target_node, list(config.keys()),
        )

    def flush(self, timeout_s: float = 10.0) -> None:
        """Flush all in-flight messages.  Call at application shutdown."""
        remaining = self._local.flush(timeout=int(timeout_s))
        if self._central:
            remaining += self._central.flush(timeout=int(timeout_s))
        if remaining:
            logger.warning("KafkaProducer.flush: %d message(s) not delivered", remaining)

    # ── Low-level produce ─────────────────────────────────────────────────────

    def _produce(self, topic: str, key: str, payload: dict) -> None:
        """
        Serialise and produce one message.

        Tries Avro serialisation first; falls back to JSON bytes if the
        serialiser for this topic is not registered or fails.
        Routes to both producers if the topic is in _CENTRAL_TOPICS.
        """
        try:
            value_bytes = self._serialise(topic, payload)
            key_bytes   = key.encode("utf-8")
        except Exception as exc:
            logger.error("Serialisation error for topic=%s: %s", topic, exc)
            return

        for prod in self._producers_for(topic):
            try:
                prod.produce(
                    topic,
                    key   = key_bytes,
                    value = value_bytes,
                    on_delivery = _on_delivery,
                )
                prod.poll(0)   # trigger delivery callbacks without blocking
            except KafkaException as exc:
                logger.error("Kafka produce failed  topic=%s  err=%s", topic, exc)
            except BufferError:
                # Local queue full — flush then retry once
                logger.warning("Kafka buffer full for topic=%s; flushing", topic)
                prod.flush(timeout=2)
                try:
                    prod.produce(
                        topic,
                        key   = key_bytes,
                        value = value_bytes,
                        on_delivery = _on_delivery,
                    )
                except Exception as exc2:
                    logger.error("Kafka retry produce failed: %s", exc2)

    def _serialise(self, topic: str, payload: dict) -> bytes:
        ser = self._avro_ser.get(topic)
        if ser is not None and _AVRO_AVAILABLE:
            ctx = SerializationContext(topic, MessageField.VALUE)
            return ser(payload, ctx)
        # JSON fallback
        return json.dumps(payload).encode("utf-8")

    def _producers_for(self, topic: str) -> list[_CKProducer]:
        if topic in _LOCAL_ONLY_TOPICS:
            return [self._local]
        if topic in _CENTRAL_TOPICS and self._central is not None:
            return [self._local, self._central]
        return [self._local]

    # ── Face embedding sync ────────────────────────────────────────────────────

    def publish_embedding_sync(
        self,
        client_id:   str,
        person_id:   str,
        embedding:   list[float],
        metadata:    dict | None = None,
        event_type:  str = "ENROLL",    # ENROLL | UPDATE | DELETE
        source_node: str = "",
    ) -> None:
        """
        Publish a face embedding to `repo.sync.embeddings` so other nodes can
        update their local FAISS indexes without a full DB scan.

        On DELETE, `embedding` may be an empty list — other nodes will soft-delete
        from pgvector and rebuild FAISS.
        """
        self._produce(
            "repo.sync.embeddings",
            key=f"{client_id}/{person_id}",
            payload={
                "event_id":       str(uuid.uuid4()),
                "event_type":     event_type,
                "client_id":      client_id,
                "person_id":      person_id,
                "source_node":    source_node,
                "embedding_json": json.dumps(embedding),
                "version":        int(metadata.get("version", 1)) if metadata else 1,
                "confidence_avg": float(metadata.get("confidence_avg", 0.0)) if metadata else 0.0,
                "quality_score":  float(metadata.get("quality_score", 0.0)) if metadata else 0.0,
                "metadata_json":  json.dumps(metadata) if metadata else None,
                "ts":             time.time(),
            },
        )


# ── KafkaConsumer worker for admin.overrides ──────────────────────────────────

class AdminOverrideConsumer:
    """
    Asyncio-compatible consumer for the admin.overrides topic.

    Inbound commands:
      FORCE_SYNC   — immediately sync a held session to ERP
      DISCARD      — mark the session as DISCARDED, no ERP sync
      RE_HOLD      — re-hold an already-synced session for correction

    Usage
    ─────
        consumer = AdminOverrideConsumer(
            bootstrap_servers  = "kafka:19092",
            group_id           = "acas-override-worker",
            kafka_producer     = producer,
            db                 = db_session,
        )
        await consumer.start()          # runs until stop() is called
        await consumer.stop()
    """

    _POLL_TIMEOUT_S = 0.25

    def __init__(
        self,
        bootstrap_servers:    str,
        group_id:             str,
        kafka_producer:       KafkaProducer,
        db:                   Any = None,    # AsyncSession | None
        schema_registry_url:  str | None = None,
        node_id:              str = "unknown",
        config_reload_cb: "Any | None" = None,  # Callable[[dict], Awaitable[None]]
    ) -> None:
        from confluent_kafka import Consumer as _CKConsumer, OFFSET_BEGINNING  # noqa: F401

        self._kafka             = kafka_producer
        self._db                = db
        self._node_id           = node_id
        self._config_reload_cb  = config_reload_cb   # BUG 7 FIX
        self._running           = False
        self._task: "asyncio.Task[None] | None" = None

        cfg = {
            "bootstrap.servers":   bootstrap_servers,
            "group.id":            group_id,
            "auto.offset.reset":   "latest",
            "enable.auto.commit":  False,
        }
        self._consumer = _CKConsumer(cfg)
        self._consumer.subscribe(["admin.overrides"])

        # Avro deserialiser
        self._avro_de = None
        if _AVRO_AVAILABLE and schema_registry_url:
            try:
                sr = SchemaRegistryClient({"url": schema_registry_url})
                self._avro_de = AvroDeserializer(sr)
            except Exception as exc:
                logger.warning("AdminOverrideConsumer: Avro deserialiser init failed: %s", exc)

    async def start(self) -> None:
        """Start the consumer loop as a background asyncio Task."""
        self._running = True
        self._task = asyncio.get_event_loop().create_task(
            self._loop(), name="admin_override_consumer"
        )
        logger.info("AdminOverrideConsumer started")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        self._consumer.close()
        logger.info("AdminOverrideConsumer stopped")

    async def _loop(self) -> None:
        import asyncio

        while self._running:
            msg = await asyncio.to_thread(
                self._consumer.poll, self._POLL_TIMEOUT_S
            )
            if msg is None:
                continue
            if msg.error():
                from confluent_kafka import KafkaError
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    logger.error("AdminOverrideConsumer error: %s", msg.error())
                continue

            try:
                payload = self._deserialise(msg)
                await self._handle(payload)
                self._consumer.commit(message=msg, asynchronous=True)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(
                    "AdminOverrideConsumer: failed to process message: %s", exc,
                    exc_info=True,
                )

    def _deserialise(self, msg: Any) -> dict:
        if self._avro_de and _AVRO_AVAILABLE:
            try:
                ctx = SerializationContext(msg.topic(), MessageField.VALUE)
                return self._avro_de(msg.value(), ctx)
            except Exception:
                pass
        return json.loads(msg.value().decode("utf-8"))

    async def _handle(self, payload: dict) -> None:
        action     = payload.get("action", "")
        session_id = payload.get("session_id", "")
        client_id  = payload.get("client_id", "")
        actor_id   = payload.get("actor_id", "")
        reason     = payload.get("reason")
        now        = time.time()

        logger.info(
            "AdminOverride  action=%s  session=%s  actor=%s",
            action, session_id, actor_id,
        )

        if action == "FORCE_SYNC":
            await self._force_sync(session_id, client_id, actor_id, reason, now)
        elif action == "DISCARD":
            await self._discard(session_id, client_id, actor_id, reason, now)
        elif action == "RE_HOLD":
            await self._re_hold(session_id, client_id, actor_id, reason, now)
        elif action == "RELOAD_CONFIG":
            await self._reload_config(payload, actor_id)
        else:
            logger.warning("Unknown override action: %s", action)

        # Audit trail
        self._kafka.publish_audit(
            actor_id    = actor_id,
            actor_type  = "ADMIN",
            action      = f"OVERRIDE_{action}",
            target_type = "session",
            target_id   = session_id,
            client_id   = client_id,
            detail      = {"reason": reason, "override_id": payload.get("override_id")},
        )

    async def _force_sync(
        self,
        session_id: str,
        client_id:  str,
        actor_id:   str,
        reason:     str | None,
        now:        float,
    ) -> None:
        """Update session sync_status to SYNCED and re-publish to ERP."""
        if self._db:
            try:
                from sqlalchemy import text

                await self._db.execute(
                    text("""
                        UPDATE sessions
                        SET sync_status = 'SYNCED',
                            synced_at   = :now
                        WHERE session_id = (:sid)::uuid
                          AND client_id  = (:cid)::uuid
                    """),
                    {"sid": session_id, "cid": client_id, "now": int(now)},
                )
                await self._db.commit()
            except Exception as exc:
                logger.error("force_sync DB update failed: %s", exc)

        # Re-emit ERP sync request
        self._kafka._produce(
            "erp.sync.requests",
            key = session_id,
            payload = {
                "request_id":   str(uuid.uuid4()),
                "session_id":   session_id,
                "client_id":    client_id,
                "payload_json": json.dumps(
                    {"forced": True, "by": actor_id, "reason": reason}
                ),
                "ts": now,
            },
        )
        logger.info("force_sync: session=%s  by=%s", session_id, actor_id)

    async def _discard(
        self,
        session_id: str,
        client_id:  str,
        actor_id:   str,
        reason:     str | None,
        now:        float,
    ) -> None:
        """Mark session as DISCARDED — no ERP sync ever."""
        if self._db:
            try:
                from sqlalchemy import text

                await self._db.execute(
                    text("""
                        UPDATE sessions
                        SET sync_status = 'DISCARDED',
                            held_reason = :reason
                        WHERE session_id = (:sid)::uuid
                          AND client_id  = (:cid)::uuid
                    """),
                    {"sid": session_id, "cid": client_id,
                     "reason": f"admin_discard:{actor_id}:{reason}"},
                )
                await self._db.commit()
            except Exception as exc:
                logger.error("discard DB update failed: %s", exc)

        logger.info("discard: session=%s  by=%s", session_id, actor_id)

    async def _re_hold(
        self,
        session_id: str,
        client_id:  str,
        actor_id:   str,
        reason:     str | None,
        now:        float,
    ) -> None:
        """Reset a synced session back to HELD for correction."""
        if self._db:
            try:
                from sqlalchemy import text

                await self._db.execute(
                    text("""
                        UPDATE sessions
                        SET sync_status = 'HELD',
                            held_reason = :reason,
                            synced_at   = NULL
                        WHERE session_id = (:sid)::uuid
                          AND client_id  = (:cid)::uuid
                    """),
                    {"sid": session_id, "cid": client_id,
                     "reason": f"admin_re_hold:{actor_id}:{reason}"},
                )
                await self._db.commit()
            except Exception as exc:
                logger.error("re_hold DB update failed: %s", exc)

        logger.info("re_hold: session=%s  by=%s", session_id, actor_id)

    async def _reload_config(
        self,
        payload:  dict[str, Any],
        actor_id: str,
    ) -> None:
        """
        BUG 7 FIX — RELOAD_CONFIG broadcast handler.

        Published payload convention (uses existing admin.overrides schema):
            action     = "RELOAD_CONFIG"
            session_id = "__broadcast__"          (sentinel — no real session)
            client_id  = "__all__" | "<client-uuid>"  (scope)
            reason     = JSON-encoded config dict  (piggy-backs existing field)

        target_node is encoded inside the reason JSON:
            {"target_node": "*" | "<node_id>", "config": {...}}

        All nodes consume from admin.overrides; each checks whether
        target_node matches its own node_id or is the wildcard "*".
        """
        try:
            data: dict[str, Any] = json.loads(payload.get("reason") or "{}")
        except (json.JSONDecodeError, TypeError):
            data = {}

        target_node = data.get("target_node", "*")
        config      = data.get("config", {})

        # Only process if this message is addressed to us
        if target_node not in ("*", self._node_id):
            logger.debug(
                "AdminOverrideConsumer: RELOAD_CONFIG not for this node "
                "(target=%s  this=%s) — skipped",
                target_node, self._node_id,
            )
            return

        logger.info(
            "AdminOverrideConsumer: RELOAD_CONFIG  target=%s  keys=%s  by=%s",
            target_node, list(config.keys()), actor_id,
        )

        if self._config_reload_cb:
            try:
                await self._config_reload_cb(config)
            except Exception as exc:
                logger.error(
                    "AdminOverrideConsumer: config reload callback error: %s", exc
                )
