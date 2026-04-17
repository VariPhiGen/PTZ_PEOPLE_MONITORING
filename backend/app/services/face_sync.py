"""
FaceSyncService — cross-node face embedding synchronisation.

Architecture (multi-node)
──────────────────────────
  All nodes share the same PostgreSQL cluster (pgvector source of truth).
  Each node maintains its own in-memory FAISS index per client for < 0.5 ms
  roster search.  When a new enrollment or template update happens on Node A:

    1. Node A writes the embedding to pgvector (shared DB).
    2. Node A publishes to the central Kafka topic `repo.sync.embeddings`.
    3. All other nodes (B, C …) consume that event, upsert into their local
       pgvector replica, and rebuild the affected client's FAISS index.

  The Kafka message includes client_id so each node can filter to only the
  clients it actually serves (i.e. has cameras assigned for).

  If Kafka is down, the consistency_check loop catches drift every 6 hours
  and triggers a selective resync.

Fixes applied (vs original)
────────────────────────────
  BUG 1  ON CONFLICT (person_id) is wrong — no unique constraint exists.
         Replaced with version-guarded UPDATE + conditional INSERT pattern
         that is idempotent under at-least-once Kafka delivery.

  BUG 2  Events were processed for ALL clients regardless of whether this
         node serves them.  Added _served_clients set with TTL cache so
         events for foreign clients are skipped immediately.

  BUG 3  Drift tolerance used max(1, …) floor so a single-vector desync
         on a small index (< 100 vectors) was silently ignored.
         Removed the floor — any mismatch now triggers resync.

  BUG 4  (In node_manager.py) handle_migration only logged; never called
         on_camera_assigned.  Fixed in node_manager.py.

  BUG 5  After migration the new node had no FAISS for the acquired client.
         on_camera_assigned now triggers a targeted FAISS resync when the
         client is new to this node.

  BUG 6  Initial stagger was 3 600 s (1 h). Reduced to 300 s (5 min) so
         a post-startup desync heals quickly.

  BUG 7  RELOAD_CONFIG broadcast — handled via AdminOverrideConsumer in
         kafka_producer.py; a callback is registered here so the service
         can react (rebuild indexes, update thresholds, etc.).
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

if TYPE_CHECKING:
    from app.services.face_repository import FaceRepository
    from app.services.kafka_producer import KafkaProducer

logger = logging.getLogger(__name__)

_SYNC_TOPIC = "repo.sync.embeddings"


class FaceSyncService:
    """
    Consumes `repo.sync.embeddings` from the central Kafka cluster and keeps
    the local FAISS indexes in sync with pgvector.

    Only processes embedding events for clients whose cameras are assigned to
    this node — events for foreign clients are discarded immediately.

    Parameters
    ----------
    session_factory      SQLAlchemy async session factory (shared DB).
    face_repo            The local FaceRepository instance (owns FAISS indexes).
    kafka_producer       The node's KafkaProducer (for publish_enrollment).
    bootstrap_servers    Central Kafka broker(s) — "" / None to disable consumer.
    group_id             Kafka consumer group (unique per node: "face-sync-{node_id}").
    schema_registry_url  Optional Confluent Schema Registry for Avro.
    node_id              This node's ID — used to skip self-published messages.
    """

    _POLL_TIMEOUT_S              = 0.5
    _CONSISTENCY_INTERVAL_S      = 6 * 3600   # every 6 hours
    _CONSISTENCY_INITIAL_STAGGER = 300         # BUG 6 FIX: was 3600 s
    _DRIFT_TOLERANCE             = 0.01        # 1 % — no integer floor (BUG 3 FIX)
    _SERVED_CACHE_TTL_S          = 300.0       # refresh served-clients cache every 5 min

    def __init__(
        self,
        session_factory:     async_sessionmaker[AsyncSession],
        face_repo:           "FaceRepository",
        kafka_producer:      "KafkaProducer",
        bootstrap_servers:   str | None,
        group_id:            str,
        schema_registry_url: str | None = None,
        node_id:             str        = "unknown",
    ) -> None:
        self._factory    = session_factory
        self._repo       = face_repo
        self._producer   = kafka_producer
        self._node_id    = node_id

        self._running           = False
        self._consumer_task:    asyncio.Task[None] | None = None
        self._consistency_task: asyncio.Task[None] | None = None

        # BUG 2 FIX: served-clients cache so we skip foreign events fast
        self._served_clients: set[str] = set()
        self._served_lock             = asyncio.Lock()
        self._served_cache_ts: float  = 0.0           # 0 forces immediate refresh

        # BUG 7 FIX: optional callback invoked on RELOAD_CONFIG broadcast
        self._config_reload_cb: Callable[[dict[str, Any]], Awaitable[None]] | None = None

        # Build the central Kafka consumer (optional)
        self._consumer = None
        if bootstrap_servers:
            try:
                from confluent_kafka import Consumer as _CKConsumer

                self._consumer = _CKConsumer({
                    "bootstrap.servers":  bootstrap_servers,
                    "group.id":           group_id,
                    "auto.offset.reset":  "earliest",
                    "enable.auto.commit": False,
                    "session.timeout.ms": 30_000,
                    "max.poll.interval.ms": 60_000,
                })
                self._consumer.subscribe([_SYNC_TOPIC])
                logger.info(
                    "FaceSyncService: consumer subscribed  topic=%s  group=%s",
                    _SYNC_TOPIC, group_id,
                )
            except Exception as exc:
                logger.warning(
                    "FaceSyncService: consumer init failed (%s) — sync disabled", exc
                )

        # Optional Avro deserialiser
        self._avro_de: Any = None
        if schema_registry_url:
            try:
                from confluent_kafka.schema_registry import SchemaRegistryClient
                from confluent_kafka.schema_registry.avro import AvroDeserializer

                sr = SchemaRegistryClient({"url": schema_registry_url})
                self._avro_de = AvroDeserializer(sr)
            except Exception as exc:
                logger.warning(
                    "FaceSyncService: Avro deserialiser init failed (%s) — JSON fallback", exc
                )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._running = True
        # Populate served-clients cache before the consumer loop starts
        await self._refresh_served_clients()

        if self._consumer:
            self._consumer_task = asyncio.create_task(
                self._consumer_loop(), name="face_sync_consumer"
            )
        self._consistency_task = asyncio.create_task(
            self._consistency_loop(), name="face_sync_consistency"
        )
        logger.info("FaceSyncService started  node_id=%s", self._node_id)

    async def stop(self) -> None:
        self._running = False
        for task in (self._consumer_task, self._consistency_task):
            if task:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        if self._consumer:
            try:
                await asyncio.to_thread(self._consumer.close)
            except Exception:
                pass
        logger.info("FaceSyncService stopped")

    # ── Config reload callback (BUG 7 FIX) ───────────────────────────────────

    def set_config_reload_callback(
        self,
        callback: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        """
        Register a callback invoked when a RELOAD_CONFIG admin override is
        broadcast to this node.  The callback receives the config dict.
        """
        self._config_reload_cb = callback

    async def handle_config_reload(self, config: dict[str, Any]) -> None:
        """
        Called by AdminOverrideConsumer when a RELOAD_CONFIG event arrives.
        Propagates to the registered callback (e.g., update in-memory
        thresholds, rebuild FAISS with new parameters).
        """
        logger.info(
            "FaceSyncService: RELOAD_CONFIG received  keys=%s  node=%s",
            list(config.keys()), self._node_id,
        )
        if self._config_reload_cb:
            try:
                await self._config_reload_cb(config)
            except Exception as exc:
                logger.error("FaceSyncService: config reload callback error: %s", exc)

    # ── Served-clients cache (BUG 2 FIX) ─────────────────────────────────────

    async def _refresh_served_clients(self) -> None:
        """Reload the set of client_ids that have cameras on this node from DB."""
        try:
            cids = await self._get_served_clients()
            async with self._served_lock:
                self._served_clients = set(cids)
                self._served_cache_ts = time.monotonic()
            logger.debug(
                "FaceSyncService: served clients refreshed  count=%d", len(cids)
            )
        except Exception as exc:
            logger.error("FaceSyncService: served clients refresh failed: %s", exc)

    async def _serves_client(self, client_id: str) -> bool:
        """
        Return True if this node serves `client_id`.
        Refreshes the cache when it is older than _SERVED_CACHE_TTL_S.
        """
        if time.monotonic() - self._served_cache_ts > self._SERVED_CACHE_TTL_S:
            await self._refresh_served_clients()
        async with self._served_lock:
            return client_id in self._served_clients

    def notify_camera_assigned(self, camera_id: str, client_id: str) -> None:
        """
        Called by NodeManager (synchronously) when a camera is assigned.
        Eagerly adds the client to the served set so subsequent Kafka events
        are processed immediately — without waiting for the next cache refresh.
        Also triggers a FAISS resync if this is the first camera for this client
        (BUG 5 FIX: schedule async resync).
        """
        was_new = client_id not in self._served_clients
        self._served_clients.add(client_id)
        if was_new:
            logger.info(
                "FaceSyncService: new client %s via camera %s — scheduling FAISS resync",
                client_id, camera_id,
            )
            # Schedule async resync without blocking the caller
            asyncio.create_task(
                self._resync_client(client_id),
                name=f"faiss_resync_{client_id[:8]}",
            )

    def notify_camera_removed(self, camera_id: str, client_id: str) -> None:
        """
        Called by NodeManager when a camera is removed.
        Forces a cache refresh on the next event so we stop processing
        embeddings for clients we no longer serve.
        """
        # Don't eagerly remove — another camera for the same client may remain.
        # Force a cache refresh to get the accurate picture from DB.
        self._served_cache_ts = 0.0

    # ── Publishing (outbound sync) ─────────────────────────────────────────────

    def publish_enrollment(
        self,
        client_id:  str,
        person_id:  str,
        embedding:  list[float],
        metadata:   dict[str, Any],
        event_type: str = "ENROLL",   # ENROLL | UPDATE | DELETE
    ) -> None:
        """
        Publish a face embedding event to the central sync topic.
        Other nodes consume this and rebuild the affected client's FAISS index.
        client_id is always included so receivers can filter by served clients.
        """
        self._producer.publish_embedding_sync(
            client_id   = client_id,
            person_id   = person_id,
            embedding   = embedding,
            metadata    = metadata,
            event_type  = event_type,
            source_node = self._node_id,
        )

    # ── Consumer loop ─────────────────────────────────────────────────────────

    async def _consumer_loop(self) -> None:
        logger.info(
            "FaceSyncService: consumer loop started  topic=%s", _SYNC_TOPIC
        )
        while self._running:
            try:
                msg = await asyncio.to_thread(
                    self._consumer.poll, self._POLL_TIMEOUT_S
                )
                if msg is None:
                    continue
                if msg.error():
                    from confluent_kafka import KafkaError
                    if msg.error().code() != KafkaError._PARTITION_EOF:
                        logger.error(
                            "FaceSyncService: consumer error: %s", msg.error()
                        )
                    continue

                payload = self._deserialise(msg)
                await self._handle_sync_event(payload)
                await asyncio.to_thread(
                    self._consumer.commit, message=msg, asynchronous=False
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(
                    "FaceSyncService: processing error: %s", exc, exc_info=True
                )

    async def _handle_sync_event(self, payload: dict[str, Any]) -> None:
        # Skip events we published ourselves
        source_node = payload.get("source_node", "")
        if source_node == self._node_id:
            return

        event_type = payload.get("event_type", "ENROLL")
        client_id  = payload.get("client_id", "")
        person_id  = payload.get("person_id", "")

        if not client_id:
            logger.warning(
                "FaceSyncService: sync event missing client_id — discarding"
            )
            return

        # BUG 2 FIX: only process clients this node actually serves
        if not await self._serves_client(client_id):
            logger.debug(
                "FaceSyncService: skip event  event=%s  client=%s  (not served)",
                event_type, client_id,
            )
            return

        logger.debug(
            "FaceSyncService: event=%s  client=%s  person=%s  from=%s",
            event_type, client_id, person_id, source_node,
        )

        if event_type in ("ENROLL", "UPDATE"):
            await self._handle_enroll(payload, client_id, person_id)
        elif event_type == "DELETE":
            await self._handle_delete(client_id, person_id)
        else:
            logger.warning("FaceSyncService: unknown event_type=%s", event_type)

    # ── Enroll / update handler ────────────────────────────────────────────────

    async def _handle_enroll(
        self,
        payload:   dict[str, Any],
        client_id: str,
        person_id: str,
    ) -> None:
        """
        Upsert the embedding into local pgvector then rebuild FAISS.

        BUG 1 FIX: The original code used ON CONFLICT (person_id) which fails
        because face_embeddings has no unique constraint on person_id — only a
        UUID PK on embedding_id.

        Correct pattern (version-guarded, idempotent):
          Step 1: Deactivate embeddings whose version < incoming version so the
                  new one becomes the sole active template.
          Step 2: INSERT the new embedding only if this (person_id, version) pair
                  doesn't already exist — handles duplicate Kafka delivery and
                  out-of-order events gracefully.
        """
        raw_emb = payload.get("embedding_json", "[]")
        try:
            emb_list = (
                json.loads(raw_emb) if isinstance(raw_emb, str) else raw_emb
            )
        except json.JSONDecodeError:
            logger.error(
                "FaceSyncService: invalid embedding_json for person %s", person_id
            )
            return

        if not emb_list:
            logger.warning(
                "FaceSyncService: empty embedding for person %s — skipping", person_id
            )
            return

        vec_str  = "[" + ",".join(str(float(x)) for x in emb_list) + "]"
        version  = int(payload.get("version", 1))
        conf     = float(payload.get("confidence_avg", 0.0))
        qual     = float(payload.get("quality_score", 0.0))

        async with self._factory() as session:
            try:
                # Step 1: deactivate older versions only
                # (don't touch same or newer versions already present)
                await session.execute(
                    text("""
                        UPDATE face_embeddings
                        SET    is_active = false
                        WHERE  client_id = (:cid)::uuid
                          AND  person_id  = (:pid)::uuid
                          AND  is_active  = true
                          AND  version    < :ver
                    """),
                    {"cid": client_id, "pid": person_id, "ver": version},
                )

                # Step 2: insert only if this version is not already present
                # (idempotent under duplicate Kafka delivery)
                await session.execute(
                    text("""
                        INSERT INTO face_embeddings
                            (embedding_id, client_id, person_id, embedding,
                             version, source, confidence_avg, is_active,
                             quality_score, created_at)
                        SELECT
                            gen_random_uuid(),
                            (:cid)::uuid, (:pid)::uuid, (:emb)::vector,
                            :ver, 'AUTO_UPDATE', :conf, true, :qual,
                            now()
                        WHERE NOT EXISTS (
                            SELECT 1
                            FROM   face_embeddings
                            WHERE  client_id = (:cid)::uuid
                              AND  person_id  = (:pid)::uuid
                              AND  version    = :ver
                        )
                    """),
                    {
                        "cid":  client_id,
                        "pid":  person_id,
                        "emb":  vec_str,
                        "ver":  version,
                        "conf": conf,
                        "qual": qual,
                    },
                )
                await session.commit()
            except Exception as exc:
                await session.rollback()
                logger.error(
                    "FaceSyncService: DB upsert failed  person=%s  error=%s",
                    person_id, exc,
                )
                return

        try:
            await self._repo.rebuild_faiss_index(client_id)
            logger.debug(
                "FaceSyncService: FAISS rebuilt  client=%s  person=%s  version=%d",
                client_id, person_id, version,
            )
        except Exception as exc:
            logger.error(
                "FaceSyncService: FAISS rebuild failed  client=%s  error=%s",
                client_id, exc,
            )

    # ── Delete handler ────────────────────────────────────────────────────────

    async def _handle_delete(self, client_id: str, person_id: str) -> None:
        """Soft-delete all embeddings for this person and rebuild FAISS."""
        async with self._factory() as session:
            try:
                await session.execute(
                    text("""
                        UPDATE face_embeddings
                        SET    is_active = false
                        WHERE  client_id = (:cid)::uuid
                          AND  person_id  = (:pid)::uuid
                    """),
                    {"cid": client_id, "pid": person_id},
                )
                await session.commit()
            except Exception as exc:
                await session.rollback()
                logger.error(
                    "FaceSyncService: soft-delete failed  person=%s  error=%s",
                    person_id, exc,
                )
                return

        try:
            await self._repo.rebuild_faiss_index(client_id)
        except Exception as exc:
            logger.error(
                "FaceSyncService: FAISS rebuild after delete failed  client=%s  error=%s",
                client_id, exc,
            )

    # ── Full resync ────────────────────────────────────────────────────────────

    async def full_resync(self, client_id: str | None = None) -> dict[str, int]:
        """
        Rebuild FAISS indexes from the shared pgvector DB.

        client_id=None  → resync all clients whose cameras are on this node.
        client_id=<id>  → resync only that client (even if not in served cache).

        Returns {client_id: count_of_embeddings_loaded}.
        """
        t0 = time.monotonic()
        logger.info(
            "FaceSyncService: full_resync starting  client_id=%s",
            client_id or "all",
        )

        if client_id:
            client_ids = [client_id]
        else:
            client_ids = await self._get_served_clients()
            if not client_ids:
                logger.info(
                    "FaceSyncService: no clients served by this node — skipping resync"
                )
                return {}

        results: dict[str, int] = {}
        for cid in client_ids:
            try:
                count = await self._resync_client(cid)
                results[cid] = count
            except Exception as exc:
                logger.error(
                    "FaceSyncService: resync failed  client=%s  error=%s", cid, exc
                )
                results[cid] = -1

        elapsed = time.monotonic() - t0
        total   = sum(v for v in results.values() if v >= 0)
        logger.info(
            "FaceSyncService: full_resync done  clients=%d  embeddings=%d  elapsed=%.1fs",
            len(client_ids), total, elapsed,
        )
        # Refresh served-clients cache after resync (cameras may have changed)
        await self._refresh_served_clients()
        return results

    async def _resync_client(self, client_id: str) -> int:
        """Rebuild this client's FAISS index from pgvector. Returns embedding count."""
        await self._repo.rebuild_faiss_index(client_id)
        count = self._repo.get_faiss_count(client_id)
        logger.info(
            "FaceSyncService: client %s resynced  embeddings=%d", client_id, count
        )
        return count

    async def _get_served_clients(self) -> list[str]:
        """
        Return distinct client_ids for cameras assigned to this node.
        Includes ONLINE and DEGRADED cameras; excludes ARCHIVED / unassigned.
        """
        async with self._factory() as session:
            rows = await session.execute(
                text("""
                    SELECT DISTINCT client_id::text
                    FROM cameras
                    WHERE node_id = :nid
                      AND status  != 'ARCHIVED'
                """),
                {"nid": self._node_id},
            )
            return [str(r[0]) for r in rows.fetchall()]

    # ── Consistency check loop ─────────────────────────────────────────────────

    async def _consistency_loop(self) -> None:
        """
        BUG 6 FIX: Initial stagger reduced from 3 600 s → 300 s.
        After startup a Kafka outage during enrollment could leave nodes
        desynced; healing within 5 min is acceptable for production.
        """
        await asyncio.sleep(self._CONSISTENCY_INITIAL_STAGGER)
        while self._running:
            try:
                await self.consistency_check()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("FaceSyncService: consistency check error: %s", exc)
            await asyncio.sleep(self._CONSISTENCY_INTERVAL_S)

    async def consistency_check(self) -> dict[str, dict[str, Any]]:
        """
        Compare in-memory FAISS count vs pgvector count for each served client.

        BUG 3 FIX: tolerance = int(primary_count * _DRIFT_TOLERANCE)
        No longer uses max(1, …) floor, so even a single-vector mismatch
        on a small index triggers a resync.

        Returns a report dict keyed by client_id.
        """
        logger.info("FaceSyncService: running consistency check")
        report: dict[str, dict[str, Any]] = {}
        client_ids = await self._get_served_clients()

        for cid in client_ids:
            try:
                async with self._factory() as session:
                    row = await session.execute(
                        text("""
                            SELECT COUNT(*)
                            FROM face_embeddings
                            WHERE client_id = (:cid)::uuid
                              AND is_active  = true
                        """),
                        {"cid": cid},
                    )
                    primary_count = row.scalar() or 0

                faiss_count = self._repo.get_faiss_count(cid)
                # BUG 3 FIX: no max(1,...) floor — any mismatch triggers resync
                tolerance   = int(primary_count * self._DRIFT_TOLERANCE)
                in_sync     = abs(primary_count - faiss_count) <= tolerance

                report[cid] = {
                    "primary_count": primary_count,
                    "faiss_count":   faiss_count,
                    "in_sync":       in_sync,
                    "tolerance":     tolerance,
                }

                if not in_sync:
                    logger.warning(
                        "FaceSyncService: DRIFT  client=%s  primary=%d  faiss=%d  "
                        "delta=%d  tolerance=%d — triggering resync",
                        cid, primary_count, faiss_count,
                        abs(primary_count - faiss_count), tolerance,
                    )
                    await self._resync_client(cid)
                    report[cid]["resync_triggered"] = True

            except Exception as exc:
                logger.error(
                    "FaceSyncService: consistency error for client %s: %s", cid, exc
                )
                report[cid] = {"error": str(exc)}

        out_of_sync = sum(1 for v in report.values() if not v.get("in_sync", True))
        logger.info(
            "FaceSyncService: consistency done  clients=%d  out_of_sync=%d",
            len(client_ids), out_of_sync,
        )
        return report

    # ── Deserialisation ────────────────────────────────────────────────────────

    def _deserialise(self, msg: Any) -> dict[str, Any]:
        if self._avro_de:
            try:
                from confluent_kafka.schema_registry.avro import AvroDeserializer  # noqa: F401
                from confluent_kafka.serialization import (
                    MessageField,
                    SerializationContext,
                )
                ctx = SerializationContext(msg.topic(), MessageField.VALUE)
                return self._avro_de(msg.value(), ctx)  # type: ignore[no-any-return]
            except Exception:
                pass
        try:
            return json.loads(msg.value().decode("utf-8"))  # type: ignore[no-any-return]
        except Exception:
            return {}
