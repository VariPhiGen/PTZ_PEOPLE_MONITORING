"""
FaceSearchEngine — multi-modal search and journey analytics for ACAS.

Methods
───────
  search_by_text(client_id, query, limit=10)
      pg_trgm similarity search on persons.name within a client's tenant.
      Returns ranked PersonSearchHit list with last_seen and thumbnail URL.

  search_by_face(client_id, image_bytes)
      Decode JPEG/PNG → detect one face → quality gate → AdaFace embedding
      → FAISS institution search.  Returns top-10 FaceSearchHit with person
      metadata enriched from the persons table.

  build_journey(client_id, person_id, date_from, date_to)
      Merge attendance_records + sightings into a chronological event list.
      Computes transit times between locations, a location/duration summary,
      attendance stats, and a heatmap (area × hour-of-day → seconds).

  search_area(client_id, camera_id, time_from, time_to, date=None)
      Who was physically present at a camera during a time window.
      Queries sightings; returns list of AreaOccupant sorted by first_seen.

  get_cross_camera_trail(client_id, person_id, date)
      Ordered list of camera transitions for a person on a given calendar day,
      including transit times between consecutive locations.

All DB queries run as parameterised async SQL and carry mandatory
WHERE client_id = :cid clauses that work with RLS.
"""
from __future__ import annotations

import asyncio
import datetime
import logging
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from minio import Minio
from sqlalchemy import text

logger = logging.getLogger(__name__)

# ── Search quality thresholds (looser than enrollment — search mode) ───────────

_SEARCH_MIN_IOD_PX    = 30.0    # px  — looser than enrollment (80 px)
_SEARCH_MIN_SHARPNESS = 30.0    # Laplacian variance on 112×112 chip
_SEARCH_MIN_FACE_CONF = 0.50
_TRGM_MIN_SCORE       = 0.10    # minimum pg_trgm similarity to surface a result
_TOP_K_FACE_SEARCH    = 10
_THUMBNAIL_EXPIRY_S   = 86_400  # 1-day presigned URL TTL
_MINIO_ENROLL_BUCKET  = "face-enrollment"


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class PersonSearchHit:
    """One result from a text or face search."""
    person_id:    str
    name:         str
    role:         str
    department:   str | None
    thumbnail_url: str | None   # MinIO presigned URL, or None
    last_seen:    float | None  # epoch of most recent sighting
    match_score:  float         # trgm similarity (text) or cosine sim (face)


@dataclass
class FaceSearchHit(PersonSearchHit):
    """Face search extends PersonSearchHit with embedding match metadata."""
    similarity:   float = 0.0
    tier:         int   = 0     # 1 = FAISS, 2 = pgvector
    embedding_id: str | None = None


@dataclass
class JourneyEvent:
    """One atomic presence event — either an attendance record or a sighting."""
    event_type:          str           # "ATTENDANCE" | "SIGHTING"
    source_id:           str           # record_id or sighting_id
    camera_id:           str | None
    camera_name:         str | None
    room_name:           str | None
    building:            str | None
    floor:               str | None
    zone:                str | None    # MONITORING zone label, None for attendance
    first_seen:          float         # epoch
    last_seen:           float         # epoch
    duration_s:          float
    transit_from_prev_s: float | None  # seconds since previous event's last_seen
    status:              str | None    # P/L/EE/A/ND for attendance, None for sightings
    confidence_avg:      float | None


@dataclass
class JourneySummary:
    total_events:      int
    total_duration_s:  float
    locations_visited: list[str]          # unique, in order of first appearance
    cameras_visited:   int
    attendance_stats:  dict[str, int]     # {"P":n, "L":n, "EE":n, "A":n, "ND":n}
    first_appearance:  float | None       # epoch
    last_appearance:   float | None       # epoch


@dataclass
class PersonJourney:
    person_id:   str
    name:        str
    role:        str
    department:  str | None
    date_from:   float                              # epoch query bound
    date_to:     float                              # epoch query bound
    events:      list[JourneyEvent]
    summary:     JourneySummary
    heatmap:     dict[str, dict[int, float]]        # area_label → {hour: seconds}


@dataclass
class AreaOccupant:
    person_id:      str
    name:           str
    role:           str
    department:     str | None
    first_seen:     float
    last_seen:      float
    duration_s:     int
    confidence_avg: float | None


@dataclass
class CameraTransition:
    """One step in a person's cross-camera trail."""
    from_camera_id:   str | None
    from_camera_name: str | None
    from_location:    str | None   # "{room_name}, {building}" or camera name
    to_camera_id:     str
    to_camera_name:   str
    to_location:      str | None
    left_at:          float | None  # epoch last seen at prev camera
    arrived_at:       float         # epoch first seen at this camera
    transit_time_s:   float | None  # arrived_at - left_at
    duration_s:       int           # how long at this camera
    confidence_avg:   float | None


# ── Engine ────────────────────────────────────────────────────────────────────

class FaceSearchEngine:
    """
    Multi-modal search and journey analytics engine.

    Parameters
    ----------
    face_repo       FaceRepository — for FAISS/pgvector face search.
    session_factory async_sessionmaker — creates per-request DB sessions.
    pipeline        AIPipeline — for face detection and embedding.
    minio           Minio client (optional) — for thumbnail presigned URLs.
    """

    def __init__(
        self,
        face_repo:       Any,           # FaceRepository
        session_factory: Any,           # async_sessionmaker
        pipeline:        Any,           # AIPipeline
        minio:           Minio | None = None,
    ) -> None:
        self._repo            = face_repo
        self._session_factory = session_factory
        self._pipeline        = pipeline
        self._minio           = minio

    # ── Text search ───────────────────────────────────────────────────────────

    async def search_by_text(
        self,
        client_id: str,
        query:     str,
        limit:     int = 10,
    ) -> list[PersonSearchHit]:
        """
        Full-text search over persons.name using PostgreSQL pg_trgm similarity.

        Also searches external_id and department to improve recall.
        Returns persons sorted by similarity score descending.
        """
        if not query or not query.strip():
            return []

        query_str = query.strip()
        async with self._session_factory() as db:
            rows = await db.execute(
                text("""
                    SELECT
                        p.person_id::text,
                        p.name,
                        p.role,
                        p.department,
                        GREATEST(
                            similarity(p.name,        :q),
                            similarity(COALESCE(p.department, ''), :q),
                            similarity(COALESCE(p.external_id, ''), :q)
                        ) AS match_score,
                        MAX(s.last_seen) AS last_seen
                    FROM persons p
                    LEFT JOIN sightings s
                           ON s.person_id = p.person_id
                          AND s.client_id = (:cid)::uuid
                    WHERE p.client_id   = (:cid)::uuid
                      AND p.status      = 'ACTIVE'
                      AND (
                          similarity(p.name,        :q) > :min_score
                       OR similarity(COALESCE(p.department, ''), :q) > :min_score
                       OR similarity(COALESCE(p.external_id, ''), :q) > :min_score
                      )
                    GROUP BY p.person_id, p.name, p.role, p.department
                    ORDER BY match_score DESC
                    LIMIT :lim
                """),
                {
                    "cid":       client_id,
                    "q":         query_str,
                    "min_score": _TRGM_MIN_SCORE,
                    "lim":       limit,
                },
            )
            persons = rows.fetchall()

        hits: list[PersonSearchHit] = []
        for row in persons:
            pid = str(row.person_id)
            thumbnail = await asyncio.to_thread(
                self._get_thumbnail_url, client_id, pid
            )
            hits.append(PersonSearchHit(
                person_id     = pid,
                name          = row.name,
                role          = row.role,
                department    = row.department,
                thumbnail_url = thumbnail,
                last_seen     = float(row.last_seen) if row.last_seen else None,
                match_score   = round(float(row.match_score), 4),
            ))

        return hits

    # ── Face search ───────────────────────────────────────────────────────────

    async def search_by_face(
        self,
        client_id:   str,
        image_bytes: bytes,
    ) -> list[FaceSearchHit]:
        """
        Detect one face in image_bytes, compute AdaFace embedding, and search
        the client's FAISS institution index.

        Raises ValueError for bad input (no face, multiple faces, quality fail).
        Returns up to 10 FaceSearchHit enriched with person metadata.
        """
        frame = _decode_image(image_bytes)

        # ── Detect faces ──────────────────────────────────────────────────────
        faces = await asyncio.to_thread(self._pipeline.detect_faces, frame, None)

        if len(faces) == 0:
            raise ValueError("no_face_detected")
        if len(faces) > 1:
            raise ValueError(
                f"multiple_faces_detected:{len(faces)} — submit a single-face image"
            )

        face = faces[0]

        # ── Quality gate ──────────────────────────────────────────────────────
        if face.conf < _SEARCH_MIN_FACE_CONF:
            raise ValueError(
                f"low_face_confidence:{face.conf:.2f} < {_SEARCH_MIN_FACE_CONF}"
            )
        if face.inter_ocular_px < _SEARCH_MIN_IOD_PX:
            raise ValueError(
                f"face_too_small:iod={face.inter_ocular_px:.1f}px "
                f"< {_SEARCH_MIN_IOD_PX}px — use a higher-resolution image"
            )

        chip = await asyncio.to_thread(
            self._pipeline.align_face, frame, face.landmarks, face.inter_ocular_px
        )
        sharpness = float(cv2.Laplacian(
            cv2.cvtColor(chip, cv2.COLOR_BGR2GRAY), cv2.CV_64F
        ).var())
        if sharpness < _SEARCH_MIN_SHARPNESS:
            raise ValueError(
                f"blurry_face:sharpness={sharpness:.1f} < {_SEARCH_MIN_SHARPNESS}"
            )

        # ── Embedding ─────────────────────────────────────────────────────────
        embedding = await asyncio.to_thread(self._pipeline.get_embedding, chip)

        # ── FAISS institution search (Tier-2 for full institution sweep) ──────
        hits = await self._repo.search_institution(
            client_id, embedding, top_k=_TOP_K_FACE_SEARCH
        )
        if not hits:
            raise ValueError("Person not found — no matching identity enrolled in this system")

        # ── Enrich with person metadata ───────────────────────────────────────
        person_ids = [h.person_id for h in hits]
        async with self._session_factory() as db:
            metadata   = await self._fetch_persons_bulk(db, client_id, person_ids)
            last_seens = await self._fetch_last_seen_bulk(db, client_id, person_ids)

        results: list[FaceSearchHit] = []
        for hit in hits:
            pid  = hit.person_id
            meta = metadata.get(pid)
            if meta is None:
                continue
            thumbnail = await asyncio.to_thread(
                self._get_thumbnail_url, client_id, pid
            )
            results.append(FaceSearchHit(
                person_id     = pid,
                name          = meta["name"],
                role          = meta["role"],
                department    = meta.get("department"),
                thumbnail_url = thumbnail,
                last_seen     = last_seens.get(pid),
                match_score   = round(hit.similarity, 4),
                similarity    = round(hit.similarity, 4),
                tier          = 2,
                embedding_id  = hit.embedding_id,
            ))

        return results

    # ── Journey builder ───────────────────────────────────────────────────────

    async def build_journey(
        self,
        client_id:  str,
        person_id:  str,
        date_from:  float,   # epoch seconds
        date_to:    float,   # epoch seconds
    ) -> PersonJourney:
        """
        Merge attendance_records and sightings into a unified chronological
        timeline for a person within [date_from, date_to].

        Computes per-event transit times, a JourneySummary, and a heatmap
        of presence (area × hour-of-day → seconds spent).
        """
        async with self._session_factory() as db:
            # ── Fetch person metadata ──────────────────────────────────────────
            person_row = await db.execute(
                text("""
                    SELECT person_id::text, name, role, department
                    FROM   persons
                    WHERE  person_id = (:pid)::uuid
                      AND  client_id = (:cid)::uuid
                """),
                {"pid": person_id, "cid": client_id},
            )
            person = person_row.fetchone()
            if person is None:
                raise ValueError(f"person_not_found:{person_id}")

            # ── Attendance records ─────────────────────────────────────────────
            att_rows = await db.execute(
                text("""
                    SELECT
                        ar.record_id::text,
                        ar.first_seen,
                        ar.last_seen,
                        ar.status,
                        ar.confidence_avg,
                        s.camera_id::text,
                        c.name        AS camera_name,
                        c.room_name,
                        c.building,
                        c.floor
                    FROM attendance_records ar
                    JOIN sessions s
                      ON s.session_id = ar.session_id
                    LEFT JOIN cameras c
                      ON c.camera_id = s.camera_id
                    WHERE ar.client_id = (:cid)::uuid
                      AND ar.person_id = (:pid)::uuid
                      AND ar.first_seen IS NOT NULL
                      AND ar.first_seen >= :ts_from
                      AND ar.first_seen <  :ts_to
                    ORDER BY ar.first_seen
                """),
                {
                    "cid":     client_id,
                    "pid":     person_id,
                    "ts_from": int(date_from),
                    "ts_to":   int(date_to),
                },
            )
            att_records = att_rows.fetchall()

            # ── Sightings ──────────────────────────────────────────────────────
            sight_rows = await db.execute(
                text("""
                    SELECT
                        s.sighting_id::text,
                        s.first_seen,
                        s.last_seen,
                        s.duration_seconds,
                        s.confidence_avg,
                        s.zone,
                        s.camera_id::text,
                        c.name    AS camera_name,
                        c.room_name,
                        c.building,
                        c.floor
                    FROM sightings s
                    LEFT JOIN cameras c
                           ON c.camera_id = s.camera_id
                    WHERE s.client_id = (:cid)::uuid
                      AND s.person_id = (:pid)::uuid
                      AND s.first_seen >= :ts_from
                      AND s.first_seen <  :ts_to
                    ORDER BY s.first_seen
                """),
                {
                    "cid":     client_id,
                    "pid":     person_id,
                    "ts_from": int(date_from),
                    "ts_to":   int(date_to),
                },
            )
            sight_records = sight_rows.fetchall()

        # ── Merge into JourneyEvent list ───────────────────────────────────────
        def _same_day_cap(first: float, last: float) -> float:
            """Cap last at 23:59:59 UTC of the same calendar day as first."""
            dt = datetime.datetime.fromtimestamp(first, tz=datetime.timezone.utc)
            day_end = dt.replace(hour=23, minute=59, second=59, microsecond=0).timestamp()
            return min(last, day_end)

        raw_events: list[dict] = []

        for r in att_records:
            fs = float(r.first_seen)
            ls = _same_day_cap(fs, float(r.last_seen)) if r.last_seen else fs
            raw_events.append({
                "event_type":   "ATTENDANCE",
                "source_id":    r.record_id,
                "camera_id":    r.camera_id,
                "camera_name":  r.camera_name,
                "room_name":    r.room_name,
                "building":     r.building,
                "floor":        r.floor,
                "zone":         None,
                "first_seen":   fs,
                "last_seen":    ls,
                "status":       r.status,
                "confidence_avg": float(r.confidence_avg) if r.confidence_avg is not None else None,
            })

        for r in sight_records:
            fs = float(r.first_seen)
            ls = _same_day_cap(fs, float(r.last_seen)) if r.last_seen else fs
            raw_events.append({
                "event_type":   "SIGHTING",
                "source_id":    r.sighting_id,
                "camera_id":    r.camera_id,
                "camera_name":  r.camera_name,
                "room_name":    r.room_name,
                "building":     r.building,
                "floor":        r.floor,
                "zone":         r.zone,
                "first_seen":   fs,
                "last_seen":    ls,
                "status":       None,
                "confidence_avg": float(r.confidence_avg) if r.confidence_avg is not None else None,
            })

        raw_events.sort(key=lambda e: e["first_seen"])

        # ── Compute transit times and assemble JourneyEvent objects ───────────
        events: list[JourneyEvent] = []
        prev_last_seen: float | None = None

        for ev in raw_events:
            dur = max(0.0, ev["last_seen"] - ev["first_seen"])
            transit = (
                float(ev["first_seen"] - prev_last_seen)
                if prev_last_seen is not None else None
            )
            events.append(JourneyEvent(
                event_type          = ev["event_type"],
                source_id           = ev["source_id"],
                camera_id           = ev["camera_id"],
                camera_name         = ev["camera_name"],
                room_name           = ev["room_name"],
                building            = ev["building"],
                floor               = ev["floor"],
                zone                = ev["zone"],
                first_seen          = ev["first_seen"],
                last_seen           = ev["last_seen"],
                duration_s          = dur,
                transit_from_prev_s = transit,
                status              = ev["status"],
                confidence_avg      = ev["confidence_avg"],
            ))
            prev_last_seen = ev["last_seen"]

        # ── Summary ────────────────────────────────────────────────────────────
        summary = _build_summary(events)

        # ── Heatmap (area × hour-of-day → seconds) ────────────────────────────
        heatmap = _build_heatmap(events)

        return PersonJourney(
            person_id  = person_id,
            name       = person.name,
            role       = person.role,
            department = person.department,
            date_from  = date_from,
            date_to    = date_to,
            events     = events,
            summary    = summary,
            heatmap    = heatmap,
        )

    # ── Area search ───────────────────────────────────────────────────────────

    async def search_area(
        self,
        client_id: str,
        camera_id: str,
        time_from: float,           # epoch seconds within the day
        time_to:   float,           # epoch seconds within the day
        date:      float | None = None,  # epoch seconds for the calendar date
                                         # (optional; used to anchor time_from/to)
    ) -> list[AreaOccupant]:
        """
        Return all persons whose sighting overlaps the [time_from, time_to] window
        on a specific camera.

        If `date` is provided, time_from/to are treated as seconds-since-midnight
        offsets relative to that date (in UTC); otherwise they are absolute epochs.
        """
        if date is not None:
            day_start = _day_start_epoch(date)
            time_from = day_start + time_from
            time_to   = day_start + time_to

        async with self._session_factory() as db:
            rows = await db.execute(
                text("""
                    SELECT
                        p.person_id::text,
                        p.name,
                        p.role,
                        p.department,
                        s.first_seen,
                        s.last_seen,
                        s.duration_seconds,
                        s.confidence_avg
                    FROM sightings s
                    JOIN persons p
                      ON p.person_id = s.person_id
                     AND p.client_id = (:cid)::uuid
                    WHERE s.client_id  = (:cid)::uuid
                      AND s.camera_id  = (:cam)::uuid
                      AND s.last_seen  >= :ts_from
                      AND s.first_seen <= :ts_to
                    ORDER BY s.first_seen
                """),
                {
                    "cid":     client_id,
                    "cam":     camera_id,
                    "ts_from": int(time_from),
                    "ts_to":   int(time_to),
                },
            )
            return [
                AreaOccupant(
                    person_id      = str(r.person_id),
                    name           = r.name,
                    role           = r.role,
                    department     = r.department,
                    first_seen     = float(r.first_seen),
                    last_seen      = float(r.last_seen),
                    duration_s     = int(r.duration_seconds or 0),
                    confidence_avg = float(r.confidence_avg) if r.confidence_avg is not None else None,
                )
                for r in rows.fetchall()
            ]

    # ── Cross-camera trail ────────────────────────────────────────────────────

    async def get_cross_camera_trail(
        self,
        client_id: str,
        person_id: str,
        date:      float,   # any epoch within the desired calendar day (UTC)
    ) -> list[CameraTransition]:
        """
        Return a chronologically ordered list of camera transitions for a
        person on a given calendar day.

        Each CameraTransition carries the previous and next camera names plus
        the transit time between them (gap between last_seen at A and first_seen
        at B).  The first entry has from_camera_id=None and transit_time_s=None.
        """
        day_start = _day_start_epoch(date)
        day_end   = day_start + 86_400

        async with self._session_factory() as db:
            rows = await db.execute(
                text("""
                    WITH ordered AS (
                        SELECT
                            s.sighting_id::text,
                            s.camera_id::text,
                            c.name         AS camera_name,
                            c.room_name,
                            c.building,
                            c.floor,
                            s.first_seen,
                            s.last_seen,
                            s.duration_seconds,
                            s.confidence_avg,
                            LAG(s.last_seen)       OVER w AS prev_last_seen,
                            LAG(s.camera_id::text) OVER w AS prev_camera_id,
                            LAG(c.name)            OVER w AS prev_camera_name,
                            LAG(c.room_name)       OVER w AS prev_room_name,
                            LAG(c.building)        OVER w AS prev_building
                        FROM sightings s
                        LEFT JOIN cameras c
                               ON c.camera_id = s.camera_id
                        WHERE s.client_id = (:cid)::uuid
                          AND s.person_id = (:pid)::uuid
                          AND s.first_seen >= :day_start
                          AND s.first_seen <  :day_end
                        WINDOW w AS (ORDER BY s.first_seen)
                    )
                    SELECT * FROM ordered ORDER BY first_seen
                """),
                {
                    "cid":       client_id,
                    "pid":       person_id,
                    "day_start": int(day_start),
                    "day_end":   int(day_end),
                },
            )
            raw_rows = rows.fetchall()

        transitions: list[CameraTransition] = []
        for r in raw_rows:
            transit: float | None = None
            if r.prev_last_seen is not None:
                transit = float(r.first_seen) - float(r.prev_last_seen)

            from_loc = _format_location(r.prev_room_name, r.prev_building) \
                if r.prev_camera_id else None
            to_loc   = _format_location(r.room_name, r.building)

            transitions.append(CameraTransition(
                from_camera_id   = r.prev_camera_id,
                from_camera_name = r.prev_camera_name,
                from_location    = from_loc,
                to_camera_id     = str(r.camera_id),
                to_camera_name   = r.camera_name or str(r.camera_id),
                to_location      = to_loc,
                left_at          = float(r.prev_last_seen) if r.prev_last_seen else None,
                arrived_at       = float(r.first_seen),
                transit_time_s   = transit,
                duration_s       = int(r.duration_seconds or 0),
                confidence_avg   = float(r.confidence_avg) if r.confidence_avg is not None else None,
            ))

        return transitions

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _fetch_persons_bulk(
        self,
        db:         Any,
        client_id:  str,
        person_ids: list[str],
    ) -> dict[str, dict]:
        """Fetch name/role/department for a list of person_ids in one query."""
        if not person_ids:
            return {}
        placeholders = ", ".join(f"(:p{i})::uuid" for i in range(len(person_ids)))
        params: dict = {"cid": client_id}
        for i, pid in enumerate(person_ids):
            params[f"p{i}"] = pid

        rows = await db.execute(
            text(f"""
                SELECT person_id::text, name, role, department
                FROM   persons
                WHERE  client_id  = (:cid)::uuid
                  AND  person_id IN ({placeholders})
            """),
            params,
        )
        return {
            str(r.person_id): {
                "name":       r.name,
                "role":       r.role,
                "department": r.department,
            }
            for r in rows.fetchall()
        }

    async def _fetch_last_seen_bulk(
        self,
        db:         Any,
        client_id:  str,
        person_ids: list[str],
    ) -> dict[str, float | None]:
        """Return the most recent sighting.last_seen per person_id."""
        if not person_ids:
            return {}
        placeholders = ", ".join(f"(:p{i})::uuid" for i in range(len(person_ids)))
        params: dict = {"cid": client_id}
        for i, pid in enumerate(person_ids):
            params[f"p{i}"] = pid

        rows = await db.execute(
            text(f"""
                SELECT person_id::text, MAX(last_seen) AS last_seen
                FROM   sightings
                WHERE  client_id  = (:cid)::uuid
                  AND  person_id IN ({placeholders})
                GROUP BY person_id
            """),
            params,
        )
        return {
            str(r.person_id): float(r.last_seen)
            for r in rows.fetchall()
        }

    def _get_thumbnail_url(
        self,
        client_id: str,
        person_id: str,
    ) -> str | None:
        """
        Return a presigned GET URL for the first enrollment image of a person.
        Runs synchronously (call with asyncio.to_thread).
        Returns None if MinIO is unavailable or no images exist.
        """
        if self._minio is None:
            return None
        try:
            prefix  = f"{client_id}/{person_id}/"
            objects = list(self._minio.list_objects(
                _MINIO_ENROLL_BUCKET, prefix=prefix, recursive=False
            ))
            if not objects:
                return None
            first_key = objects[0].object_name
            url = self._minio.presigned_get_object(
                _MINIO_ENROLL_BUCKET,
                first_key,
                expires=datetime.timedelta(seconds=_THUMBNAIL_EXPIRY_S),
            )
            return url
        except Exception as exc:
            logger.debug("thumbnail lookup failed for %s/%s: %s", client_id, person_id, exc)
            return None


# ── Module-level helpers ──────────────────────────────────────────────────────

def _decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode JPEG/PNG bytes to a BGR uint8 numpy array."""
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise ValueError("cannot_decode_image — must be JPEG or PNG")
    return frame


def _day_start_epoch(ts: float) -> float:
    """Return the UTC midnight epoch for the calendar day containing ts."""
    dt = datetime.datetime.utcfromtimestamp(ts).replace(
        hour=0, minute=0, second=0, microsecond=0,
        tzinfo=datetime.timezone.utc,
    )
    return dt.timestamp()


def _format_location(
    room_name: str | None,
    building:  str | None,
) -> str | None:
    parts = [p for p in (room_name, building) if p]
    return ", ".join(parts) if parts else None


def _build_summary(events: list[JourneyEvent]) -> JourneySummary:
    """Derive a JourneySummary from a sorted list of JourneyEvents."""
    if not events:
        return JourneySummary(
            total_events      = 0,
            total_duration_s  = 0.0,
            locations_visited = [],
            cameras_visited   = 0,
            attendance_stats  = {s: 0 for s in ("P", "L", "EE", "A", "ND")},
            first_appearance  = None,
            last_appearance   = None,
        )

    total_dur = sum(e.duration_s for e in events)
    att_stats: dict[str, int] = {s: 0 for s in ("P", "L", "EE", "A", "ND")}

    seen_locations: list[str] = []
    seen_cameras:   set[str]  = set()

    for e in events:
        if e.status and e.status in att_stats:
            att_stats[e.status] += 1

        loc = _format_location(e.room_name, e.building) or e.camera_name
        if loc and loc not in seen_locations:
            seen_locations.append(loc)

        if e.camera_id:
            seen_cameras.add(e.camera_id)

    return JourneySummary(
        total_events      = len(events),
        total_duration_s  = total_dur,
        locations_visited = seen_locations,
        cameras_visited   = len(seen_cameras),
        attendance_stats  = att_stats,
        first_appearance  = events[0].first_seen,
        last_appearance   = events[-1].last_seen,
    )


def _build_heatmap(
    events: list[JourneyEvent],
) -> dict[str, dict[int, float]]:
    """
    Build a heatmap: area_label → hour-of-day (0–23) → seconds_present.

    For each event, determine which UTC hours it spans and distribute
    the presence seconds proportionally across those hours.
    """
    heatmap: dict[str, dict[int, float]] = {}

    for e in events:
        area = (
            _format_location(e.room_name, e.building)
            or e.camera_name
            or e.camera_id
            or "unknown"
        )
        if area not in heatmap:
            heatmap[area] = {h: 0.0 for h in range(24)}

        if e.duration_s <= 0:
            continue

        # Walk hour by hour across the event's time span
        ts = e.first_seen
        remaining = e.duration_s
        while remaining > 0 and ts < e.last_seen:
            hour        = int(datetime.datetime.utcfromtimestamp(ts).hour)
            next_hour_ts = _hour_boundary(ts)
            chunk = min(remaining, next_hour_ts - ts)
            heatmap[area][hour] += chunk
            ts       += chunk
            remaining -= chunk

    return heatmap


def _hour_boundary(ts: float) -> float:
    """Return the epoch of the next UTC hour boundary after ts."""
    dt = datetime.datetime.utcfromtimestamp(ts).replace(
        minute=0, second=0, microsecond=0,
        tzinfo=datetime.timezone.utc,
    )
    return (dt + datetime.timedelta(hours=1)).timestamp()
