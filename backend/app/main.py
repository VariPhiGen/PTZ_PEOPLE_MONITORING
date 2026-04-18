"""
ACAS Backend — FastAPI application entry point.

Lifespan startup sequence
──────────────────────────
  1. Core infrastructure
       PostgreSQL engine + session factory
       Redis client
       MinIO buckets
  2. Kafka producers (local + central cluster)
       Raw confluent Producer for backward-compat health check kept on app.state
       Typed KafkaProducer wrapper for all business events
       AdminOverrideConsumer background task
  3. Node identity
       Generate / load node_id (persisted in .env after first boot)
       Instantiate NodeManager
  4. Control Plane registration
       NodeManager.start() → async register + heartbeat loop
  5. AI services (lazy — only if MODEL_DIR has content)
       AIPipeline, FaceRepository
  6. Face sync
       FaceSyncService.start() → subscribe to repo.sync.embeddings consumer
       full_resync() → rebuild FAISS from pgvector for every served client
  7. Load cameras assigned to this node
       Query DB for cameras WHERE node_id = this node
       Fire NodeManager.on_camera_assigned() for each
  8. Ready — yield to serve requests

Shutdown
─────────
  • Stop face sync consumer
  • Flush Kafka producers
  • Cancel heartbeat loop, close HTTP client
  • Close Redis + DB connections
"""
from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
    stream=sys.stderr,
)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from minio import Minio
from redis import asyncio as redis_async
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from app.config import Settings, get_settings
from app.core.metrics import gpu_metrics_loop, metrics_endpoint, scrape_db_pool
from app.middleware.auth import AuthMiddleware
from app.middleware.perf import PerfMiddleware

logger = logging.getLogger(__name__)

_MINIO_BUCKETS = ("face-enrollment", "face-evidence")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_minio_buckets(client: Minio, buckets: tuple[str, ...]) -> None:
    for name in buckets:
        if not client.bucket_exists(name):
            client.make_bucket(name)


async def _load_assigned_cameras(
    app: FastAPI,
    settings: "Settings | None" = None,
    node_manager=None,
    session_factory=None,
) -> None:
    """
    Query the DB for cameras assigned to this node and fire on_camera_assigned
    for each so that ONVIF / PTZBrain sessions can be started automatically.
    """
    settings         = settings         or getattr(app.state, "settings", None)
    node_manager     = node_manager     or getattr(app.state, "node_manager", None)
    session_factory  = session_factory  or getattr(app.state, "session_factory", None)
    if settings is None or node_manager is None or session_factory is None:
        logger.warning("lifespan: _load_assigned_cameras skipped — state not ready")
        return

    async with session_factory() as session:
        rows = await session.execute(
            text("""
                SELECT
                    camera_id::text, client_id::text, rtsp_url, onvif_host,
                    onvif_port, onvif_username, onvif_password_encrypted,
                    roi_rect, faculty_zone, mode, faculty_id::text,
                    name, room_name,
                    learned_params, camera_distance_m, scan_cell_meters, camera_type
                FROM cameras
                WHERE node_id = :nid
                  AND status != 'OFFLINE'
            """),
            {"nid": settings.node_id},
        )
        cameras = rows.fetchall()

    if not cameras:
        logger.info("lifespan: no cameras assigned to this node — nothing to start")
        return

    logger.info("lifespan: loading %d assigned cameras", len(cameras))

    from app.services.node_manager import CameraConfig

    for row in cameras:
        # Merge camera_distance_m and scan_cell_meters into learned_params so
        # PTZBrain picks them up from the single learned_params dict it already reads.
        raw_lp: dict = dict(row[13]) if row[13] else {}
        if row[14] is not None:
            raw_lp["camera_distance_m"] = float(row[14])
        if row[15] is not None:
            raw_lp["scan_cell_meters"] = float(row[15])

        cfg = CameraConfig(
            camera_id                = str(row[0]),
            client_id                = str(row[1]),
            rtsp_url                 = row[2] or "",
            onvif_host               = row[3] or "",
            onvif_port               = int(row[4] or 80),
            onvif_username           = row[5] or "",
            onvif_password_encrypted = row[6] or "",
            roi_rect                 = row[7],
            faculty_zone             = row[8],
            mode                     = row[9] or "MONITORING",
            faculty_id               = str(row[10]) if row[10] else None,
            name                     = row[11] or "",
            room_name                = row[12] or "",
            learned_params           = raw_lp or None,
            camera_type              = row[16] or "PTZ",
        )
        try:
            await node_manager.on_camera_assigned(cfg.camera_id, cfg)
        except Exception as exc:
            logger.error(
                "lifespan: failed to start camera %s: %s", cfg.camera_id, exc
            )


# ── Camera assignment callbacks ───────────────────────────────────────────────

async def _on_camera_assigned(camera_id: str, config: Any) -> None:
    """
    Auto-start a PTZBrain session when a camera is assigned to this node.
    Falls back to logging-only if required services aren't available.
    """
    import app as _app_module
    app_ref = getattr(_app_module, "_app_ref", None)
    if app_ref is None:
        logger.info("Camera assigned (pre-ready)  camera_id=%s", camera_id)
        return

    pipeline    = getattr(app_ref.state, "pipeline", None)
    face_repo   = getattr(app_ref.state, "face_repo", None)
    gpu_manager = getattr(app_ref.state, "gpu_manager", None)
    redis       = getattr(app_ref.state, "redis", None)
    kafka       = getattr(app_ref.state, "kafka_producer", None)
    learner     = getattr(app_ref.state, "self_learner", None)
    brains      = getattr(app_ref.state, "ptz_brains", {})
    sessions    = getattr(app_ref.state, "active_sessions", {})

    if not pipeline or not face_repo:
        logger.info(
            "Camera assigned but AI disabled  camera_id=%s  mode=%s",
            camera_id, getattr(config, "mode", "?"),
        )
        return

    cfg = config
    cam_type   = getattr(cfg, "camera_type", "PTZ") or "PTZ"
    onvif_host = getattr(cfg, "onvif_host", "") or ""
    rtsp_url   = getattr(cfg, "rtsp_url", "") or ""

    # Pure BULLET cameras have no PTZ hardware — only RTSP is required.
    # BULLET_ZOOM still needs ONVIF for zoom control.
    _is_fixed = cam_type == "BULLET"

    if not _is_fixed and not onvif_host:
        logger.warning("Camera assigned but missing ONVIF host  camera_id=%s", camera_id)
        return
    if not rtsp_url:
        logger.warning("Camera assigned but missing RTSP URL  camera_id=%s", camera_id)
        return

    try:
        from app.services.onvif_controller import FixedCameraController
        from app.services.isapi_controller import make_camera_controller
        from app.services.rtsp_decoder import RTSPDecoder
        from app.services.zone_mapper import ZoneMapper
        from app.services.path_planner import PathPlanner
        from app.services.ptz_brain import PTZBrain

        settings = app_ref.state.settings
        gpu_device = int(settings.gpu_device_id)

        if _is_fixed:
            onvif = FixedCameraController(rtsp_url=rtsp_url)
        else:
            _lp = getattr(cfg, "learned_params", None) or {}
            _protocol = _lp.get("ptz_protocol") or "ONVIF"
            onvif = make_camera_controller(
                host           = onvif_host,
                port           = getattr(cfg, "onvif_port", 80) or 80,
                username       = getattr(cfg, "onvif_username", "") or "",
                password       = getattr(cfg, "onvif_password_encrypted", "") or "",
                protocol       = _protocol,
                learned_params = _lp,
            )
            await onvif.connect()

        decoder      = RTSPDecoder(rtsp_url=rtsp_url, fps=25, gpu_device=gpu_device)
        zone_mapper  = ZoneMapper(onvif_controller=onvif)
        path_planner = PathPlanner(limits=onvif.limits)

        import uuid as _uuid
        from sqlalchemy import text as _text

        session_id   = str(_uuid.uuid4())
        client_id    = getattr(cfg, "client_id", "")
        mode         = getattr(cfg, "mode", "MONITORING") or "MONITORING"
        faculty_id   = getattr(cfg, "faculty_id", None)
        roi_rect     = getattr(cfg, "roi_rect", None)
        _now         = int(__import__("time").time())

        # Insert a DB row so /api/sessions/active returns this session
        from datetime import datetime, timezone
        try:
            from zoneinfo import ZoneInfo as _ZI
        except ImportError:
            _ZI = None  # type: ignore[assignment,misc]
        try:
            session_factory = getattr(app_ref.state, "session_factory", None)
            if session_factory:
                async with session_factory() as _sess:
                    # Enrolled = distinct persons with at least one active face embedding
                    _ecnt_row = await _sess.execute(
                        _text("""
                            SELECT COUNT(DISTINCT fe.person_id)
                            FROM face_embeddings fe
                            JOIN persons p ON p.person_id = fe.person_id
                            WHERE p.client_id = (:cid)::uuid
                              AND p.status = 'ACTIVE'
                              AND fe.is_active = true
                        """),
                        {"cid": client_id},
                    )
                    _enrolled_count = int((_ecnt_row.scalar() or 0))

                    # Look up the camera's timetable + current course in the
                    # timetable's declared timezone (not container UTC).
                    _course_id:   str | None = None
                    _course_name: str | None = None
                    _tt_faculty:  str | None = None
                    _sched_start: int  | None = None
                    _sched_end:   int  | None = None
                    try:
                        _tt_row = await _sess.execute(
                            _text("""
                                SELECT c.timetable_id::text, t.timezone
                                FROM cameras c
                                LEFT JOIN timetables t ON t.timetable_id = c.timetable_id
                                WHERE c.camera_id = (:camid)::uuid
                            """),
                            {"camid": camera_id},
                        )
                        _tt_fetch = _tt_row.first()
                        _tt_id = _tt_fetch[0] if _tt_fetch else None
                        _tz_name = (_tt_fetch[1] if _tt_fetch else None) or "UTC"
                        if _tt_id:
                            _tz = timezone.utc
                            if _ZI is not None:
                                try:
                                    _tz = _ZI(_tz_name)
                                except Exception:
                                    logger.warning(
                                        "Auto-start: unknown timezone %r on timetable %s; using UTC",
                                        _tz_name, _tt_id,
                                    )
                            _nl = datetime.now(_tz)
                            _dow = _nl.weekday()
                            _hhmm = _nl.strftime("%H:%M")
                            _ent_rows = await _sess.execute(
                                _text("""
                                    SELECT course_id::text, course_name,
                                           faculty_id::text, start_time, end_time
                                    FROM timetable_entries
                                    WHERE timetable_id = (:tid)::uuid
                                      AND day_of_week = :dow
                                    ORDER BY start_time
                                """),
                                {"tid": _tt_id, "dow": _dow},
                            )
                            _ent_list = _ent_rows.fetchall()
                            for _cid_s, _cn, _fid_s, _st, _et in _ent_list:
                                if _st <= _hhmm < _et:
                                    _course_id   = _cid_s
                                    _course_name = _cn
                                    _tt_faculty  = _fid_s
                                    try:
                                        _today = _nl.date()
                                        _shm, _ssm = (int(x) for x in _st.split(":"))
                                        _ehm, _esm = (int(x) for x in _et.split(":"))
                                        _sched_start = int(datetime(
                                            _today.year, _today.month, _today.day,
                                            _shm, _ssm, tzinfo=_tz,
                                        ).timestamp())
                                        _sched_end = int(datetime(
                                            _today.year, _today.month, _today.day,
                                            _ehm, _esm, tzinfo=_tz,
                                        ).timestamp())
                                    except Exception:
                                        _sched_start = _sched_end = None
                                    break
                            if _course_name is None:
                                logger.info(
                                    "Auto-start: no timetable entry matched "
                                    "tid=%s dow=%s hhmm=%s tz=%s (%d rows)",
                                    _tt_id, _dow, _hhmm, _tz_name, len(_ent_list),
                                )
                    except Exception as _tt_exc:
                        logger.warning(
                            "Auto-start: timetable lookup failed for camera %s: %s",
                            camera_id, _tt_exc, exc_info=True,
                        )

                    await _sess.execute(
                        _text("""
                            INSERT INTO sessions
                                (session_id, client_id, camera_id,
                                 course_id, course_name, faculty_id,
                                 scheduled_start, scheduled_end,
                                 actual_start, sync_status, cycle_count,
                                 enrolled_count, created_at)
                            VALUES
                                ((:sid)::uuid, (:cid)::uuid, (:camid)::uuid,
                                 :course_id, :course_name,
                                 (:faculty_id)::uuid,
                                 :sched_start, :sched_end,
                                 :now, 'PENDING', 0, :enrolled, :now)
                            ON CONFLICT (session_id) DO NOTHING
                        """),
                        {
                            "sid":         session_id,
                            "cid":         client_id,
                            "camid":       camera_id,
                            "course_id":   _course_id,
                            "course_name": _course_name,
                            "faculty_id":  _tt_faculty or faculty_id,
                            "sched_start": _sched_start,
                            "sched_end":   _sched_end,
                            "now":         _now,
                            "enrolled":    _enrolled_count,
                        },
                    )
                    await _sess.commit()
        except Exception as _exc:
            logger.warning("Auto-start: failed to insert session row: %s", _exc)

        # Build a preset loader that queries the DB for this camera's scan presets
        _cam_id_for_loader = camera_id
        _client_id_for_loader = client_id
        _sf_for_loader = getattr(app_ref.state, "session_factory", None)

        async def _preset_loader() -> list[dict]:
            """Load CameraScanPreset rows ordered by order_idx."""
            if _sf_for_loader is None:
                return []
            from sqlalchemy import text as _sql
            async with _sf_for_loader() as _db:
                # Set RLS context so the query respects row-level security
                await _db.execute(
                    _sql("SELECT set_config('app.client_id', :cid, true)"),
                    {"cid": _client_id_for_loader},
                )
                rows = await _db.execute(
                    _sql("""
                        SELECT name, pan, tilt, zoom, dwell_s
                        FROM camera_scan_presets
                        WHERE camera_id = (:camid)::uuid
                        ORDER BY order_idx ASC
                    """),
                    {"camid": _cam_id_for_loader},
                )
                return [
                    {
                        "name":    r.name,
                        "pan":     float(r.pan),
                        "tilt":    float(r.tilt),
                        "zoom":    float(r.zoom),
                        "dwell_s": float(r.dwell_s),
                    }
                    for r in rows.mappings()
                ]

        brain = PTZBrain(
            camera=onvif, pipeline=pipeline, face_repo=face_repo,
            zone_mapper=zone_mapper, path_planner=path_planner,
            decoder=decoder, kafka_producer=kafka, redis=redis,
            gpu_manager=gpu_manager,
            learned_params         = getattr(cfg, "learned_params", None),
            preset_loader          = _preset_loader,
            session_factory        = session_factory,
            faculty_hunt_timeout_s = settings.ptz_faculty_hunt_timeout_s,
            monitoring_overview_s  = settings.ptz_monitoring_overview_s,
            recognize_frames       = settings.ptz_recognize_frames,
            face_hunt_budget_s     = settings.ptz_face_hunt_budget_s,
            absence_grace_s        = settings.ptz_absence_grace_s,
        )

        async def _on_complete(sid: str, error: Exception | None) -> None:
            sessions.pop(sid, None)
            brains.pop(sid, None)
            # Mark session ended in DB
            try:
                _sf = getattr(app_ref.state, "session_factory", None)
                if _sf:
                    async with _sf() as _sess:
                        await _sess.execute(
                            _text("UPDATE sessions SET actual_end = :ts WHERE session_id = (:sid)::uuid"),
                            {"ts": int(__import__("time").time()), "sid": sid},
                        )
                        await _sess.commit()
            except Exception:
                pass
            if learner and error is None:
                try:
                    state = brain.get_session_state()
                    await learner.write_session_summary(
                        brain_state=state, camera_id=camera_id,
                        client_id=client_id, faculty_id=faculty_id,
                    )
                except Exception:
                    pass

        task = await brain.run_session(
            session_id=session_id, client_id=client_id,
            camera_id=camera_id, roster_ids=[],
            roi_rect=roi_rect, faculty_id=faculty_id,
            mode=mode, on_complete=_on_complete,
            face_tier1_threshold=settings.faiss_tier1_threshold,
            face_tier2_threshold=settings.faiss_tier2_threshold,
            camera_type=getattr(cfg, "camera_type", "PTZ"),
            force_start=True,
        )
        sessions[session_id] = task
        brains[session_id]   = brain

        if redis:
            try:
                pipe = redis.pipeline()
                pipe.setex(f"acas:session:{session_id}:client_id", 86_400, client_id)
                pipe.setex(f"acas:session:{session_id}:camera_id", 86_400, camera_id)
                await pipe.execute()
            except Exception:
                pass

        logger.info(
            "Camera auto-started  camera_id=%s  session_id=%s  mode=%s",
            camera_id, session_id, mode,
        )
    except Exception as exc:
        logger.error("Auto-start failed for camera %s: %s", camera_id, exc)


async def _on_camera_removed(camera_id: str) -> None:
    """Stop any running PTZBrain task for this camera."""
    from app import _app_ref   # noqa: PLC0415 — circular avoided via late import

    brains: dict = getattr(_app_ref.state, "ptz_brains", {})
    sessions: dict = getattr(_app_ref.state, "active_sessions", {})
    redis = getattr(_app_ref.state, "redis", None)

    # ptz_brains is keyed by session_id; find the session for this camera via Redis
    target_sid: str | None = None
    if redis:
        for sid in list(brains.keys()):
            try:
                raw = await redis.get(f"acas:session:{sid}:camera_id")
                stored = raw.decode() if isinstance(raw, bytes) else str(raw) if raw else ""
                if stored == camera_id:
                    target_sid = sid
                    break
            except Exception:
                pass

    if target_sid is None:
        return

    brain = brains.pop(target_sid, None)
    task  = sessions.pop(target_sid, None)
    if brain is not None:
        try:
            await brain.stop()
        except Exception:
            pass
    elif task and not task.done():
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
    logger.info("PTZBrain stopped for removed camera  camera_id=%s  session_id=%s", camera_id, target_sid)


async def _seed_default_timetables(session_factory) -> None:
    """
    For every client that has no timetables yet, create a default timetable
    covering Monday–Sunday 10:00–19:00.
    """
    import uuid as _uuid

    _COURSES = [
        "Mathematics", "Computer Science", "Physics",
        "Chemistry", "English Literature", "History", "Arts & Culture",
    ]
    _DEFAULT_ENTRIES = [
        {"day_of_week": d, "start_time": "10:00", "end_time": "19:00",
         "course_name": _COURSES[d]}
        for d in range(7)
    ]

    async with session_factory() as db:
        # Get all client IDs from the clients table
        result = await db.execute(text("SELECT client_id FROM clients"))
        client_ids = [str(r[0]) for r in result.fetchall()]

    for client_id in client_ids:
        try:
            async with session_factory() as db:
                existing = await db.execute(
                    text("SELECT 1 FROM timetables WHERE client_id = (:cid)::uuid LIMIT 1"),
                    {"cid": client_id},
                )
                if existing.fetchone() is not None:
                    continue  # client already has at least one timetable

                now = int(__import__("time").time())
                tt_id = str(_uuid.uuid4())

                await db.execute(
                    text("""
                        INSERT INTO timetables
                            (timetable_id, client_id, name, description, timezone,
                             created_at, updated_at)
                        VALUES
                            ((:tid)::uuid, (:cid)::uuid,
                             'Default (Mon–Sun 10:00–19:00)',
                             'Sample timetable covering all 7 days 10 AM–7 PM',
                             'UTC', :now, :now)
                    """),
                    {"tid": tt_id, "cid": client_id, "now": now},
                )

                for entry in _DEFAULT_ENTRIES:
                    await db.execute(
                        text("""
                            INSERT INTO timetable_entries
                                (entry_id, timetable_id, client_id,
                                 day_of_week, start_time, end_time,
                                 course_name, created_at)
                            VALUES
                                (gen_random_uuid(), (:tid)::uuid, (:cid)::uuid,
                                 :dow, :st, :et, :cn, :now)
                        """),
                        {
                            "tid": tt_id,
                            "cid": client_id,
                            "dow": entry["day_of_week"],
                            "st":  entry["start_time"],
                            "et":  entry["end_time"],
                            "cn":  entry["course_name"],
                            "now": now,
                        },
                    )
                await db.commit()
                logger.info(
                    "lifespan: created default timetable for client_id=%s", client_id
                )
        except Exception as exc:
            logger.warning(
                "lifespan: failed to seed timetable for client %s: %s", client_id, exc
            )


# ── Timetable scheduler ───────────────────────────────────────────────────────

async def _timetable_scheduler_loop(app: FastAPI) -> None:
    """
    Background loop that ensures sessions are running for cameras whose
    timetable window is currently active.

    Every 60 seconds:
      1. Find cameras on this node that have a timetable assigned.
      2. For each camera inside its timetable window, start a session if none
         is already running.
      3. Cameras outside their window are stopped naturally at CYCLE_COMPLETE by
         PTZBrain._is_within_timetable() — no extra action needed here.
    """
    from datetime import datetime, timezone

    try:
        from zoneinfo import ZoneInfo as _ZI
    except ImportError:
        _ZI = None  # type: ignore[assignment,misc]

    logger.info("timetable-scheduler: started")
    while True:
        await asyncio.sleep(60)
        try:
            session_factory = getattr(app.state, "session_factory", None)
            pipeline        = getattr(app.state, "pipeline", None)
            if session_factory is None or pipeline is None:
                continue

            settings  = app.state.settings
            sessions: dict = getattr(app.state, "active_sessions", {})
            brains:   dict = getattr(app.state, "ptz_brains", {})

            # Build set of camera_ids that already have a live (not-done) session
            running: set[str] = set()
            for sid, task in list(sessions.items()):
                if not task.done():
                    brain = brains.get(sid)
                    if brain:
                        try:
                            running.add(brain.get_session_state().camera_id)
                        except Exception:
                            pass

            # Query cameras on this node that have a timetable assigned
            async with session_factory() as db:
                rows = await db.execute(
                    text("""
                        SELECT
                            c.camera_id::text, c.client_id::text, c.rtsp_url,
                            c.onvif_host, c.onvif_port, c.onvif_username,
                            c.onvif_password_encrypted, c.roi_rect, c.faculty_zone,
                            c.mode, c.faculty_id::text, c.name, c.room_name,
                            c.learned_params, c.camera_distance_m, c.scan_cell_meters,
                            c.camera_type, c.timetable_id::text, t.timezone
                        FROM cameras c
                        JOIN timetables t ON t.timetable_id = c.timetable_id
                        WHERE c.node_id = :nid
                          AND c.status != 'OFFLINE'
                    """),
                    {"nid": settings.node_id},
                )
                cameras = rows.fetchall()

            now_utc = datetime.now(timezone.utc)

            for row in cameras:
                camera_id    = str(row[0])
                timetable_id = str(row[17])
                tz_name      = row[18] or "UTC"

                if camera_id in running:
                    continue  # already has a live session — nothing to do

                # Resolve the camera's local timezone
                tz = timezone.utc
                if _ZI is not None:
                    try:
                        tz = _ZI(tz_name)
                    except Exception:
                        pass

                now_local = now_utc.astimezone(tz)
                dow       = now_local.weekday()          # 0=Mon … 6=Sun
                hhmm      = now_local.strftime("%H:%M")

                # Check timetable entries for today
                async with session_factory() as db:
                    entry_rows = await db.execute(
                        text("""
                            SELECT start_time, end_time
                            FROM timetable_entries
                            WHERE timetable_id = (:tid)::uuid
                              AND day_of_week = :dow
                        """),
                        {"tid": timetable_id, "dow": dow},
                    )
                    in_window = any(
                        st <= hhmm <= et
                        for st, et in entry_rows.fetchall()
                    )

                if not in_window:
                    continue  # outside scheduled hours — leave stopped

                # Build CameraConfig and fire the auto-start callback
                from app.services.node_manager import CameraConfig
                raw_lp: dict = dict(row[13]) if row[13] else {}
                if row[14] is not None:
                    raw_lp["camera_distance_m"] = float(row[14])
                if row[15] is not None:
                    raw_lp["scan_cell_meters"] = float(row[15])

                cfg = CameraConfig(
                    camera_id                = camera_id,
                    client_id                = str(row[1]),
                    rtsp_url                 = row[2] or "",
                    onvif_host               = row[3] or "",
                    onvif_port               = int(row[4] or 80),
                    onvif_username           = row[5] or "",
                    onvif_password_encrypted = row[6] or "",
                    roi_rect                 = row[7],
                    faculty_zone             = row[8],
                    mode                     = row[9] or "MONITORING",
                    faculty_id               = str(row[10]) if row[10] else None,
                    name                     = row[11] or "",
                    room_name                = row[12] or "",
                    learned_params           = raw_lp or None,
                    camera_type              = row[16] or "PTZ",
                )

                logger.info(
                    "timetable-scheduler: starting session  camera=%s  local_time=%s",
                    camera_id, hhmm,
                )
                try:
                    await _on_camera_assigned(camera_id, cfg)
                except Exception as exc:
                    logger.error(
                        "timetable-scheduler: auto-start failed  camera=%s  err=%s",
                        camera_id, exc,
                    )

        except asyncio.CancelledError:
            logger.info("timetable-scheduler: cancelled")
            break
        except Exception as exc:
            logger.warning("timetable-scheduler: tick failed: %s", exc)

    logger.info("timetable-scheduler: stopped")


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Store a module-level ref so callbacks can access app.state
    import app as _app_module
    _app_module._app_ref = app  # type: ignore[attr-defined]

    settings = get_settings()

    # ── 1. Core infrastructure ─────────────────────────────────────────────────
    logger.info("lifespan: initialising core infrastructure")

    engine: AsyncEngine = create_async_engine(
        settings.database_url,
        pool_pre_ping=True,
        pool_size=20,           # 20 persistent connections (matches PG max_connections budget)
        max_overflow=10,        # allow 10 burst connections; total cap = 30
        pool_timeout=30,
        pool_recycle=1800,      # recycle connections every 30 min to avoid stale idle sockets
        # Tell pgvector to use our preferred ef_search for every connection
        connect_args={"server_settings": {
            "hnsw.ef_search": "100",
            "idle_in_transaction_session_timeout": "30000",  # 30s — kills stuck txns
        }},
    )
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    redis_client: redis_async.Redis = redis_async.from_url(
        settings.redis_url, decode_responses=True
    )

    host, _, port_str = settings.minio_endpoint.partition(":")
    minio_client = Minio(
        f"{host}:{port_str or '9000'}",
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=False,
    )
    await asyncio.to_thread(_ensure_minio_buckets, minio_client, _MINIO_BUCKETS)

    # ── 2. Kafka producers ─────────────────────────────────────────────────────
    logger.info("lifespan: initialising Kafka producers")

    from confluent_kafka import Producer as _RawProducer

    # Raw producer kept for legacy health check
    _raw_producer = _RawProducer({
        "bootstrap.servers": settings.kafka_bootstrap_servers,
        "client.id":         f"acas-backend-{settings.node_id}",
        "acks":              "all",
        "enable.idempotence": True,
    })

    from app.services.kafka_producer import KafkaProducer, AdminOverrideConsumer

    kafka_producer = KafkaProducer(
        bootstrap_servers          = settings.kafka_bootstrap_servers,
        schema_registry_url        = settings.kafka_schema_registry_url,
        central_bootstrap_servers  = settings.central_kafka_bootstrap_servers,
    )

    # AdminOverride consumer — note: config_reload_cb is wired AFTER face_sync starts
    override_consumer = AdminOverrideConsumer(
        bootstrap_servers   = settings.kafka_bootstrap_servers,
        group_id            = f"acas-override-{settings.node_id}",
        kafka_producer      = kafka_producer,
        db                  = session_factory,
        schema_registry_url = settings.kafka_schema_registry_url,
        node_id             = settings.node_id,
    )
    await override_consumer.start()

    # ── 3. Node identity ───────────────────────────────────────────────────────
    logger.info("lifespan: configuring node identity  node_id=%s", settings.node_id)

    from app.services.node_manager import NodeManager

    # Determine connectivity type from the endpoint URL
    _endpoint = settings.node_api_endpoint or "http://localhost:18000"
    if "trycloudflare.com" in _endpoint or "cloudflare" in _endpoint or \
       (not _endpoint.startswith("http://localhost") and not _endpoint.startswith("http://127")):
        _connectivity = "CLOUDFLARE_TUNNEL" if "cloudflare" in _endpoint else "PUBLIC_IP"
    else:
        _connectivity = "LOCAL"

    node_manager = NodeManager(
        node_id           = settings.node_id,
        node_name         = settings.node_name,
        api_endpoint      = _endpoint,
        control_plane_url = settings.control_plane_url,
        node_auth_token   = settings.node_auth_token,
        gpu_model         = settings.gpu_model,
        location          = settings.node_location,
        max_cameras       = settings.max_cameras_per_node,
        connectivity      = _connectivity,
        session_factory   = session_factory,
    )
    node_manager.set_callbacks(
        on_assigned = _on_camera_assigned,
        on_removed  = _on_camera_removed,
    )

    # ── 4. Control Plane registration ─────────────────────────────────────────
    logger.info("lifespan: starting NodeManager (CP registration + heartbeat)")
    await node_manager.start()

    # ── 5. AI services via GPUManager ────────────────────────────────────────
    face_repo   = None
    ai_pipeline = None
    gpu_manager = None

    # ── OSNet auto-export (RE-ID) ─────────────────────────────────────────────
    # If osnet_x1_0.onnx is missing from the model volume, export it from
    # the cached torchreid weights at startup.  This is a one-time ~5 s op;
    # subsequent starts are a no-op (file already exists).
    try:
        import os as _os
        _osnet_path = _os.path.join(settings.model_dir, "osnet_x1_0.onnx")
        if not _os.path.exists(_osnet_path):
            logger.info("lifespan: osnet_x1_0.onnx not found — exporting from torchreid …")
            import sys as _sys, unittest.mock as _mock, pathlib as _pl
            _sys.modules.setdefault("tensorboard", _mock.MagicMock())
            _sys.modules.setdefault("torch.utils.tensorboard", _mock.MagicMock())
            _hub = _pl.Path("/root/.cache/torch/hub/KaiyangZhou_deep-person-reid_master")
            if _hub.exists():
                _sys.path.insert(0, str(_hub))
            from torchreid.models import osnet as _osnet_mod
            import torch as _torch
            _m = _osnet_mod.osnet_x1_0(num_classes=1000, pretrained=True, loss="softmax")
            _m.eval()
            _dummy = _torch.zeros(1, 3, 256, 128)
            _torch.onnx.export(
                _m, _dummy, _osnet_path, opset_version=12,
                input_names=["input"], output_names=["output"],
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                do_constant_folding=True,
            )
            _mb = _pl.Path(_osnet_path).stat().st_size / 1e6
            logger.info("lifespan: OSNet exported  %.1f MB → %s", _mb, _osnet_path)
        else:
            logger.debug("lifespan: osnet_x1_0.onnx already present — skipping export")
    except Exception as _exc:
        logger.warning("lifespan: OSNet auto-export failed (%s) — RE-ID will be disabled", _exc)

    try:
        import os
        if os.path.isdir(settings.model_dir) and os.listdir(settings.model_dir):
            from app.core.gpu_manager import GPUManager
            from app.services.face_repository import FaceRepository

            logger.info(
                "lifespan: loading AI via GPUManager  model_dir=%s  device=%s  "
                "max_concurrent=%d  vram_budget=%.0fGB",
                settings.model_dir, settings.gpu_device_id,
                settings.gpu_max_concurrent, settings.gpu_vram_budget_gb,
            )
            gpu_manager = await GPUManager.create(
                model_dir      = settings.model_dir,
                device_id      = int(settings.gpu_device_id),
                max_concurrent = settings.gpu_max_concurrent,
                vram_budget_gb = settings.gpu_vram_budget_gb,
            )
            ai_pipeline = gpu_manager.pipeline

            face_repo = FaceRepository(
                db                   = session_factory,
                redis                = redis_client,
                minio                = minio_client,
                gpu_device           = int(settings.gpu_device_id),
                pipeline             = ai_pipeline,
                tier1_threshold      = settings.faiss_tier1_threshold,
                tier2_threshold      = settings.faiss_tier2_threshold,
                minio_bucket         = settings.minio_enrollment_bucket,
                local_enrollment_dir = settings.enrollment_fallback_dir,
                redis_key_prefix     = settings.redis_key_prefix,
            )
            logger.info("lifespan: GPUManager + FaceRepository ready")
        else:
            logger.warning(
                "lifespan: MODEL_DIR=%s empty or missing — AI pipeline disabled",
                settings.model_dir,
            )
    except Exception as exc:
        logger.warning("lifespan: AI pipeline load failed (%s) — continuing without AI", exc)

    # ── Face search engine (works even without AI for text/journey queries) ──
    from app.services.face_search import FaceSearchEngine
    face_search_engine = FaceSearchEngine(
        face_repo       = face_repo,
        session_factory = session_factory,
        pipeline        = ai_pipeline,
        minio           = minio_client,
    )

    # ── 6. Face sync ───────────────────────────────────────────────────────────
    face_sync = None
    if face_repo is not None:
        logger.info("lifespan: starting FaceSyncService")

        from app.services.face_sync import FaceSyncService

        face_sync = FaceSyncService(
            session_factory     = session_factory,
            face_repo           = face_repo,
            kafka_producer      = kafka_producer,
            bootstrap_servers   = (
                settings.central_kafka_bootstrap_servers
                or settings.kafka_bootstrap_servers
            ),
            group_id            = f"face-sync-{settings.node_id}",
            schema_registry_url = settings.kafka_schema_registry_url,
            node_id             = settings.node_id,
        )

        # Wire config-reload: AdminOverrideConsumer RELOAD_CONFIG → face_sync
        override_consumer._config_reload_cb = face_sync.handle_config_reload

        # Wire face_sync into NodeManager for served-client cache invalidation
        # and FAISS resync on camera migration (BUG 5 FIX)
        node_manager.set_face_sync(face_sync)

        await face_sync.start()

        # Full resync: rebuild FAISS from pgvector for all served clients.
        # Runs in background so the server can accept health checks during startup.
        async def _do_resync() -> None:
            try:
                await face_sync.full_resync()
            except Exception as exc:
                logger.error("lifespan: full_resync failed: %s", exc)

        asyncio.create_task(_do_resync(), name="face_full_resync")

    # ── 7. Self-learning service ─────────────────────────────────────────────
    self_learner = None
    _self_learner_task = None
    try:
        from app.services.self_learner import SelfLearner

        self_learner = SelfLearner(
            db=session_factory,
            redis=redis_client,
            kafka_producer=kafka_producer,
        )
        _self_learner_task = asyncio.create_task(
            self_learner.start_scheduler(), name="self-learner-scheduler"
        )
        logger.info("lifespan: SelfLearner + nightly scheduler started")
    except Exception as exc:
        logger.warning("lifespan: SelfLearner init failed (%s) — continuing without", exc)

    # ── Populate app.state BEFORE loading cameras ──────────────────────────────
    # Must be done here so that _on_camera_assigned (fired by _load_assigned_cameras)
    # can read pipeline / face_repo from app.state when it starts PTZBrain.
    app.state.settings        = settings
    app.state.engine          = engine
    app.state.session_factory = session_factory
    app.state.redis           = redis_client
    app.state.minio           = minio_client
    app.state.kafka_producer  = kafka_producer    # typed wrapper
    app.state._raw_producer   = _raw_producer     # for legacy health check
    app.state.node_manager    = node_manager
    app.state.face_repo           = face_repo           # None if AI disabled
    app.state.ai_pipeline         = ai_pipeline         # None if AI disabled
    app.state.pipeline            = ai_pipeline         # alias used by node.py start
    app.state.gpu_manager         = gpu_manager         # None if AI disabled
    app.state.face_sync           = face_sync           # None if AI disabled
    app.state.self_learner        = self_learner        # None if init failed
    app.state.face_search_engine  = face_search_engine  # always available

    # Zone mapper & path planner are per-camera, created at session start.
    # We just flag that the services are available.
    app.state.zone_mapper   = "available" if ai_pipeline else None
    app.state.path_planner  = "available" if ai_pipeline else None

    app.state.ptz_brains: dict = {}           # session_id → PTZBrain
    app.state.active_sessions: dict = {}      # session_id → asyncio.Task

    # ── 8a. Seed default timetables ────────────────────────────────────────────
    logger.info("lifespan: seeding default timetables")
    try:
        await _seed_default_timetables(session_factory)
    except Exception as exc:
        logger.warning("lifespan: default timetable seed failed: %s", exc)

    # ── 8b. Load cameras assigned to this node ─────────────────────────────────
    logger.info("lifespan: loading cameras for node_id=%s", settings.node_id)
    try:
        await _load_assigned_cameras(
            app,
            settings=settings,
            node_manager=node_manager,
            session_factory=session_factory,
        )
    except Exception as exc:
        logger.error("lifespan: load_assigned_cameras failed: %s", exc)

    # ── 8c. Timetable scheduler ────────────────────────────────────────────────
    _timetable_task = asyncio.create_task(
        _timetable_scheduler_loop(app),
        name="timetable-scheduler",
    )

    # ── Background metrics tasks ──────────────────────────────────────────────
    _gpu_task = asyncio.create_task(
        gpu_metrics_loop(device_id=settings.gpu_device_id, interval_s=10.0),
        name="gpu-metrics",
    )

    async def _db_pool_scrape_loop() -> None:
        while True:
            scrape_db_pool(engine)
            await asyncio.sleep(15.0)

    _db_pool_task = asyncio.create_task(_db_pool_scrape_loop(), name="db-pool-metrics")

    logger.info("lifespan: startup complete  node_id=%s", settings.node_id)

    try:
        yield
    finally:
        logger.info("lifespan: shutting down")

        # Face sync consumer
        if face_sync:
            await face_sync.stop()

        # Admin override consumer
        await override_consumer.stop()

        # Stop all running PTZBrain instances gracefully
        for brain in list(app.state.ptz_brains.values()):
            try:
                await brain.stop()
            except Exception:
                pass
        # Cancel any remaining tasks
        for task in list(app.state.active_sessions.values()):
            if not task.done():
                task.cancel()
        if app.state.active_sessions:
            await asyncio.gather(
                *app.state.active_sessions.values(),
                return_exceptions=True,
            )

        # Self-learner scheduler
        if _self_learner_task and not _self_learner_task.done():
            _self_learner_task.cancel()

        # Timetable scheduler
        if not _timetable_task.done():
            _timetable_task.cancel()

        # Background metric tasks
        _gpu_task.cancel()
        _db_pool_task.cancel()

        # NodeManager (heartbeat loop + HTTP client)
        await node_manager.stop()

        # Kafka flush
        kafka_producer.flush(timeout_s=10.0)
        _raw_producer.flush(timeout=10)

        # Infrastructure
        await redis_client.aclose()
        await engine.dispose()

        logger.info("lifespan: shutdown complete")


# ── Application ────────────────────────────────────────────────────────────────

# Module-level ref used by callbacks that need app.state but aren't injected
_app_ref: FastAPI | None = None

app = FastAPI(title="ACAS Backend", version="0.1.0", lifespan=lifespan)

# ── Middleware (order matters: outermost added last) ──────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# AuthMiddleware must be inside CORS so it sees the request after CORS preflight
app.add_middleware(AuthMiddleware)
app.add_middleware(PerfMiddleware)

# ── Prometheus /metrics ───────────────────────────────────────────────────────
from starlette.routing import Route as _Route, Router as _Router  # noqa: E402
app.mount("/metrics", app=_Router(routes=[_Route("/", metrics_endpoint)]))

# ── Routers ───────────────────────────────────────────────────────────────────
from app.api.auth       import router as auth_router        # noqa: E402
from app.api.admin      import router as admin_router       # noqa: E402
from app.api.users      import router as users_router       # noqa: E402
from app.api.cameras    import router as cameras_router     # noqa: E402
from app.api.enrollment import router as enrollment_router  # noqa: E402
from app.api.attendance import router as attendance_router  # noqa: E402
from app.api.sessions   import router as sessions_router    # noqa: E402
from app.api.search     import router as search_router      # noqa: E402
from app.api.analytics  import router as analytics_router   # noqa: E402
from app.api.kafka_config import router as kafka_router     # noqa: E402
from app.api.node       import router as node_router        # noqa: E402
from app.api.datasets   import router as datasets_router    # noqa: E402
from app.api.settings    import router as settings_router    # noqa: E402
from app.api.monitoring          import router as monitoring_router          # noqa: E402
from app.api.timetables          import router as timetables_router          # noqa: E402
from app.api.public_enrollment   import router as public_enrollment_router   # noqa: E402
from app.api.enrollment_tokens   import router as enrollment_tokens_router   # noqa: E402

app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(users_router)
app.include_router(cameras_router)
app.include_router(enrollment_router)
app.include_router(attendance_router)
app.include_router(sessions_router)
app.include_router(search_router)
app.include_router(analytics_router)
app.include_router(kafka_router)
app.include_router(node_router)
app.include_router(datasets_router)
app.include_router(settings_router)
app.include_router(monitoring_router)
app.include_router(timetables_router)
app.include_router(public_enrollment_router)   # no-auth self-enrollment
app.include_router(enrollment_tokens_router)   # admin token management


# ── Node migrate endpoint (called by Control Plane on camera reassignment) ────
from fastapi import Request   # noqa: E402

@app.post("/api/node/cameras/migrate", tags=["node"])
async def camera_migrate(request: Request) -> dict[str, Any]:
    """
    Receive camera migration commands from the Control Plane when a node goes
    offline and its cameras are reassigned here.
    """
    body = await request.json()
    await request.app.state.node_manager.handle_migration(
        cameras   = body.get("cameras", []),
        from_node = body.get("from_node", ""),
        client_id = body.get("client_id", ""),
    )
    return {"received": len(body.get("cameras", []))}


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/api/health", tags=["ops"])
async def health(request: Request) -> dict[str, Any]:
    settings: Settings = request.app.state.settings
    checks: dict[str, str] = {}

    try:
        async with request.app.state.engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        checks["postgres"] = "ok"
    except Exception as exc:
        logger.exception("postgres health check failed")
        checks["postgres"] = f"error: {exc}"

    try:
        await request.app.state.redis.ping()
        checks["redis"] = "ok"
    except Exception as exc:
        logger.exception("redis health check failed")
        checks["redis"] = f"error: {exc}"

    try:
        await asyncio.to_thread(
            lambda: list(request.app.state.minio.list_buckets())
        )
        checks["minio"] = "ok"
    except Exception as exc:
        logger.exception("minio health check failed")
        checks["minio"] = f"error: {exc}"

    try:
        await asyncio.to_thread(
            lambda: request.app.state._raw_producer.list_topics(timeout=5)
        )
        checks["kafka"] = "ok"
    except Exception as exc:
        logger.exception("kafka health check failed")
        checks["kafka"] = f"error: {exc}"

    nm: Any = getattr(request.app.state, "node_manager", None)
    if nm is not None:
        checks["cp_registered"] = "yes" if nm.registered else "no"

    # AI / pipeline status
    gpu_mgr = getattr(request.app.state, "gpu_manager", None)
    checks["ai_pipeline"] = "loaded" if getattr(request.app.state, "pipeline", None) else "disabled"
    checks["face_repo"]   = "loaded" if getattr(request.app.state, "face_repo", None) else "disabled"
    checks["gpu_manager"] = gpu_mgr._degradation.name if gpu_mgr else "disabled"
    checks["face_sync"]   = "running" if getattr(request.app.state, "face_sync", None) else "disabled"
    checks["self_learner"] = "running" if getattr(request.app.state, "self_learner", None) else "disabled"
    checks["active_sessions"] = len(getattr(request.app.state, "active_sessions", {}))

    all_ok = all(v in ("ok", "yes", "loaded", "running", "FULL") for v in checks.values() if isinstance(v, str))
    return {
        "status":    "healthy" if all_ok else "degraded",
        "node_id":   settings.node_id,
        "node_name": settings.node_name,
        "checks":    checks,
    }
