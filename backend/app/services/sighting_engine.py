"""
SightingEngine — real-time tracking of person sightings across cameras.

State
─────
  _sightings: Dict[camera_id → Dict[person_id → ActiveSighting]]

  An ActiveSighting is created on the first recognition of a person on a
  camera, extended on every subsequent recognition within _INACTIVE_TIMEOUT_S,
  and closed (written to DB + published to Kafka) when:
    • tick() detects inactivity for > _INACTIVE_TIMEOUT_S (default 5 min)
    • flush() is called explicitly (session end)

Usage
─────
    engine = SightingEngine(db, kafka_producer)
    # Called from PTZBrain / CELL_RECOGNIZE for each identified face:
    await engine.on_recognition(client_id, camera_id, person_id, conf, frame_ref, ts)
    # Called from PTZBrain CYCLE_COMPLETE (or a periodic background task):
    await engine.tick(camera_id, ts=time.time())
    # Called at session teardown:
    await engine.flush(camera_id)
    # Dashboard occupancy query:
    count = engine.get_area_occupancy(camera_id)
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# ── Tunable constants ──────────────────────────────────────────────────────────

_INACTIVE_TIMEOUT_S  = 300.0   # 5 minutes — close sighting after this gap
_MAX_FRAME_REFS      = 20      # keep at most this many frame_refs per sighting
_MIN_DURATION_S      = 5.0     # discard ghost sightings shorter than this


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class ActiveSighting:
    """In-memory state for one ongoing person–camera sighting."""
    sighting_id:     str
    client_id:       str
    camera_id:       str
    person_id:       str
    zone:            str | None
    first_seen:      float          # epoch
    last_seen:       float          # epoch
    duration_s:      float          # accumulated seconds (last_seen - first_seen)
    conf_sum:        float          # running sum for avg
    conf_count:      int
    frame_refs:      list[str]      # capped at _MAX_FRAME_REFS


@dataclass
class ClosedSighting:
    """Finalised sighting ready for DB write and Kafka publish."""
    sighting_id:     str
    client_id:       str
    camera_id:       str
    person_id:       str
    zone:            str | None
    first_seen:      float
    last_seen:       float
    duration_s:      int
    confidence_avg:  float
    frame_refs:      list[str]
    created_at:      float


# ── Engine ────────────────────────────────────────────────────────────────────

class SightingEngine:
    """
    Tracks and closes person sightings per camera.

    Parameters
    ----------
    db              AsyncSession for direct DB writes (optional; skip if None).
    kafka_producer  KafkaProducer instance for Kafka publishes (optional).
    inactive_timeout_s  Override _INACTIVE_TIMEOUT_S.
    """

    def __init__(
        self,
        db:              AsyncSession | None = None,
        kafka_producer:  Any = None,           # KafkaProducer | None
        inactive_timeout_s: float = _INACTIVE_TIMEOUT_S,
        min_duration_s:     float = _MIN_DURATION_S,
    ) -> None:
        self._db            = db
        self._kafka         = kafka_producer
        self._timeout       = inactive_timeout_s
        self._min_duration  = min_duration_s

        # camera_id → person_id → ActiveSighting
        self._sightings: dict[str, dict[str, ActiveSighting]] = {}

    # ── Core events ───────────────────────────────────────────────────────────

    async def on_recognition(
        self,
        client_id:  str,
        camera_id:  str,
        person_id:  str,
        conf:       float,
        frame_ref:  str | None,
        ts:         float | None = None,
        zone:       str | None = None,
    ) -> None:
        """
        Record one face-recognition event for a person on a camera.

        Creates a new ActiveSighting if none exists for this (camera, person)
        pair, or extends the existing one.
        """
        ts = ts or time.time()

        cam_sightings = self._sightings.setdefault(camera_id, {})

        if person_id in cam_sightings:
            s = cam_sightings[person_id]
            gap = ts - s.last_seen

            if gap > self._timeout:
                # Person disappeared and reappeared — close old, open new
                closed = self._close_sighting(s, ts=s.last_seen)
                await self._persist(closed)
                del cam_sightings[person_id]
                cam_sightings[person_id] = self._new_sighting(
                    client_id, camera_id, person_id, conf, frame_ref, ts, zone,
                )
            else:
                # Extend existing sighting
                s.last_seen  = ts
                s.duration_s = ts - s.first_seen
                s.conf_sum   += conf
                s.conf_count += 1
                if frame_ref and len(s.frame_refs) < _MAX_FRAME_REFS:
                    s.frame_refs.append(frame_ref)
                if zone:
                    s.zone = zone
        else:
            cam_sightings[person_id] = self._new_sighting(
                client_id, camera_id, person_id, conf, frame_ref, ts, zone,
            )

    async def tick(
        self,
        camera_id: str,
        ts:        float | None = None,
    ) -> list[ClosedSighting]:
        """
        Close all sightings on `camera_id` that have been inactive for
        > _INACTIVE_TIMEOUT_S.  Call this from PTZBrain CYCLE_COMPLETE or
        a periodic background task.

        Returns the list of closed sightings (for testing / logging).
        """
        ts = ts or time.time()
        cam_sightings = self._sightings.get(camera_id, {})
        if not cam_sightings:
            return []

        to_close = [
            pid for pid, s in cam_sightings.items()
            if ts - s.last_seen > self._timeout
        ]

        closed: list[ClosedSighting] = []
        for pid in to_close:
            s = cam_sightings.pop(pid)
            c = self._close_sighting(s, ts=s.last_seen)
            await self._persist(c)
            closed.append(c)

        if closed:
            logger.info(
                "tick: closed %d sighting(s) on camera=%s", len(closed), camera_id
            )

        return closed

    async def flush(
        self,
        camera_id: str,
        ts:        float | None = None,
    ) -> list[ClosedSighting]:
        """
        Force-close ALL active sightings on a camera (call at session end).
        """
        ts = ts or time.time()
        cam_sightings = self._sightings.pop(camera_id, {})
        closed: list[ClosedSighting] = []
        for s in cam_sightings.values():
            c = self._close_sighting(s, ts=ts)
            await self._persist(c)
            closed.append(c)

        logger.info(
            "flush: closed %d sighting(s) on camera=%s", len(closed), camera_id
        )
        return closed

    async def flush_all(self, ts: float | None = None) -> list[ClosedSighting]:
        """Force-close all sightings across every camera."""
        ts = ts or time.time()
        cameras = list(self._sightings.keys())
        all_closed: list[ClosedSighting] = []
        for cam_id in cameras:
            all_closed.extend(await self.flush(cam_id, ts))
        return all_closed

    # ── Occupancy query ───────────────────────────────────────────────────────

    def get_area_occupancy(self, camera_id: str) -> int:
        """
        Return the number of persons currently active on a camera
        (i.e. have an open sighting not yet timed out).
        """
        now = time.time()
        cam = self._sightings.get(camera_id, {})
        return sum(
            1 for s in cam.values()
            if now - s.last_seen <= self._timeout
        )

    def get_all_occupancy(self) -> dict[str, int]:
        """Return {camera_id: occupancy_count} for all cameras."""
        return {cam_id: self.get_area_occupancy(cam_id) for cam_id in self._sightings}

    def get_active_persons(self, camera_id: str) -> list[str]:
        """Return person_ids currently active on a camera."""
        now = time.time()
        cam = self._sightings.get(camera_id, {})
        return [
            pid for pid, s in cam.items()
            if now - s.last_seen <= self._timeout
        ]

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _new_sighting(
        client_id: str,
        camera_id: str,
        person_id: str,
        conf:      float,
        frame_ref: str | None,
        ts:        float,
        zone:      str | None,
    ) -> ActiveSighting:
        return ActiveSighting(
            sighting_id = str(uuid.uuid4()),
            client_id   = client_id,
            camera_id   = camera_id,
            person_id   = person_id,
            zone        = zone,
            first_seen  = ts,
            last_seen   = ts,
            duration_s  = 0.0,
            conf_sum    = conf,
            conf_count  = 1,
            frame_refs  = [frame_ref] if frame_ref else [],
        )

    @staticmethod
    def _close_sighting(s: ActiveSighting, ts: float) -> ClosedSighting:
        conf_avg = s.conf_sum / max(s.conf_count, 1)
        return ClosedSighting(
            sighting_id   = s.sighting_id,
            client_id     = s.client_id,
            camera_id     = s.camera_id,
            person_id     = s.person_id,
            zone          = s.zone,
            first_seen    = s.first_seen,
            last_seen     = ts,
            duration_s    = max(0, int(ts - s.first_seen)),
            confidence_avg= round(conf_avg, 4),
            frame_refs    = s.frame_refs,
            created_at    = time.time(),
        )

    async def _persist(self, c: ClosedSighting) -> None:
        """Write a closed sighting to DB and publish to Kafka."""
        # Discard ghost sightings (noise / false positives)
        if c.duration_s < self._min_duration:
            logger.info(
                "sighting filtered (duration %ds < min %ds)  person=%s  camera=%s",
                c.duration_s, self._min_duration, c.person_id, c.camera_id,
            )
            return

        if self._db is not None:
            await self._write_db(c)

        if self._kafka is not None:
            try:
                self._kafka.publish_sighting(c)
            except Exception as exc:
                logger.warning("Kafka publish_sighting failed: %s", exc)

    async def _write_db(self, c: ClosedSighting) -> None:
        """Insert or update a sighting row via raw SQL."""
        refs_pg = (
            "{" + ",".join(f'"{r}"' for r in c.frame_refs) + "}"
            if c.frame_refs else None
        )
        try:
            await self._db.execute(
                text("""
                    INSERT INTO sightings
                        (sighting_id, client_id, person_id, camera_id,
                         zone, first_seen, last_seen, duration_seconds,
                         confidence_avg, frame_refs, created_at)
                    VALUES
                        ((:sid)::uuid, (:cid)::uuid, (:pid)::uuid, (:cam)::uuid,
                         :zone, :first_seen, :last_seen, :dur,
                         :conf, :refs::text[], :now)
                    ON CONFLICT (sighting_id) DO UPDATE
                        SET last_seen       = EXCLUDED.last_seen,
                            duration_seconds= EXCLUDED.duration_seconds,
                            confidence_avg  = EXCLUDED.confidence_avg,
                            frame_refs      = EXCLUDED.frame_refs
                """),
                {
                    "sid":        c.sighting_id,
                    "cid":        c.client_id,
                    "pid":        c.person_id,
                    "cam":        c.camera_id,
                    "zone":       c.zone,
                    "first_seen": int(c.first_seen),
                    "last_seen":  int(c.last_seen),
                    "dur":        c.duration_s,
                    "conf":       c.confidence_avg,
                    "refs":       refs_pg,
                    "now":        int(c.created_at),
                },
            )
            await self._db.commit()
        except Exception as exc:
            logger.error("DB write failed for sighting %s: %s", c.sighting_id, exc)
            try:
                await self._db.rollback()
            except Exception:
                pass
