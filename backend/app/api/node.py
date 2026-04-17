"""
Node API — per-node management, camera stream lifecycle, and config reload.

POST /api/node/config/reload       — hot-reload settings from env / Redis
GET  /api/node/info                — this node's identity, capabilities, load
POST /api/node/cameras/{id}/start  — start a PTZBrain task on this node
POST /api/node/cameras/{id}/stop   — stop a running PTZBrain task on this node

Access:
  GET /api/node/info          — cameras:read (dashboard overview)
  All other routes             — system:admin (SUPER_ADMIN / CLIENT_ADMIN)
"""
from __future__ import annotations

import asyncio
import json
import platform
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import select, text

from app.api._shared import audit, now_epoch, resolve_client_id
from app.deps import CurrentUser, DBSession
from app.middleware.auth import require_permission
from app.models.cameras import Camera
from app.models.sessions import Session, SyncStatus

router = APIRouter(prefix="/api/node", tags=["node"])

_INFO  = require_permission("cameras:read")
_ADMIN = require_permission("system:admin")


# ── Schemas ───────────────────────────────────────────────────────────────────

class StartCameraRequest(BaseModel):
    session_id:  str | None = None
    client_id:   str | None = None
    mode:        str        = "MONITORING"  # ATTENDANCE | MONITORING
    roster_ids:  list[str]  = []
    faculty_id:  str | None = None
    roi_rect:    dict | list | None = None


class StopCameraRequest(BaseModel):
    session_id: str | None = None
    reason:     str | None = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_active_sessions(request: Request) -> dict[str, asyncio.Task]:
    if not hasattr(request.app.state, "active_sessions"):
        request.app.state.active_sessions = {}
    return request.app.state.active_sessions


def _get_ptz_brains(request: Request) -> dict[str, Any]:
    if not hasattr(request.app.state, "ptz_brains"):
        request.app.state.ptz_brains = {}
    return request.app.state.ptz_brains


async def _get_camera_or_404(db: DBSession, camera_id: str) -> Camera:
    cam = await db.get(Camera, uuid.UUID(camera_id))
    if cam is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    return cam


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/config/reload", dependencies=[_ADMIN])
async def config_reload(request: Request) -> dict:
    """
    Hot-reload application settings.  Reads from environment variables and
    any overrides stored in Redis.  Does NOT restart the process.
    """
    from app.config import Settings

    settings = Settings()   # re-parse env
    request.app.state.settings = settings

    audit(request, "CONFIG_RELOAD", "system", "node")
    return {
        "status":    "reloaded",
        "node_id":   settings.node_id,
        "node_name": settings.node_name,
        "ts":        now_epoch(),
    }


@router.get("/info", dependencies=[_INFO])
async def node_info(request: Request, db: DBSession) -> dict:
    """
    Return this node's identity, software versions, resource utilisation,
    and the list of camera tasks currently running on it.
    """
    import sys

    settings        = request.app.state.settings
    active_sessions = _get_active_sessions(request)
    brains          = _get_ptz_brains(request)

    task_summaries = []
    for sid, task in active_sessions.items():
        brain = brains.get(sid)
        summary: dict[str, Any] = {
            "session_id": sid,
            "running":    not task.done(),
            "cancelled":  task.cancelled(),
        }
        if brain is not None:
            try:
                state = brain.get_session_state()
                summary["brain_state"]  = state.brain_state
                summary["camera_id"]    = state.camera_id
                summary["cycle_count"]  = state.cycle_count
                summary["fps_actual"]   = state.fps_actual
            except Exception:
                pass
        task_summaries.append(summary)

    gpu_info: dict[str, Any] = {}
    try:
        import subprocess
        out = await asyncio.to_thread(
            subprocess.check_output,
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            timeout=3,
        )
        for i, line in enumerate(out.decode().strip().splitlines()):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 4:
                gpu_info[f"gpu{i}"] = {
                    "name":          parts[0],
                    "memory_used_mb": int(parts[1]),
                    "memory_total_mb": int(parts[2]),
                    "utilization_pct": int(parts[3]),
                }
    except Exception:
        gpu_info["error"] = "nvidia-smi unavailable"

    # GPUManager metrics (concurrency, latency, degradation)
    gpu_mgr = getattr(request.app.state, "gpu_manager", None)
    gpu_mgr_info = gpu_mgr.metrics() if gpu_mgr else None

    return {
        "node_id":           settings.node_id,
        "node_name":         settings.node_name,
        "hostname":          platform.node(),
        "python_version":    sys.version,
        "active_sessions":   len(active_sessions),
        "max_cameras":       settings.max_cameras_per_node,
        "tasks":             task_summaries,
        "gpu":               gpu_info,
        "gpu_manager":       gpu_mgr_info,
        "ts":                now_epoch(),
    }


_SESSION_PERM = require_permission("sessions:create")

@router.post("/cameras/{camera_id}/start", dependencies=[_SESSION_PERM])
async def start_camera(
    camera_id: str,
    body:      StartCameraRequest,
    request:   Request,
    db:        DBSession,
) -> dict:
    """
    Start a PTZBrain task on this node for the given camera and session.

    Requires:
    - Zones (ROI polygon) defined on the camera
    - ONVIF credentials and RTSP URL configured
    - All AI services initialised in app.state
    """
    cam      = await _get_camera_or_404(db, camera_id)
    sessions = _get_active_sessions(request)
    brains   = _get_ptz_brains(request)

    # Eagerly read all camera attributes before any db.commit()
    cam_client_id    = str(cam.client_id)
    cam_onvif_host   = cam.onvif_host or ""
    cam_onvif_port   = cam.onvif_port or 80
    cam_onvif_user   = cam.onvif_username or ""
    cam_onvif_pass   = cam.onvif_password_encrypted or ""
    cam_rtsp_url     = cam.rtsp_url or ""
    cam_roi          = cam.roi_rect
    cam_mode         = cam.mode or "MONITORING"
    cam_camera_type  = getattr(cam, "camera_type", None) or "PTZ"
    cam_dataset_id   = cam.dataset_id
    cam_dataset_ids: list[str] = (
        cam.dataset_ids
        or ([str(cam.dataset_id)] if cam.dataset_id else [])
    )

    # Merge camera_distance_m / scan_cell_meters columns into learned_params
    # so PTZBrain reads them from a single dict (avoids touching its constructor).
    cam_learned: dict = dict(cam.learned_params) if cam.learned_params else {}
    if cam.camera_distance_m is not None:
        cam_learned.setdefault("camera_distance_m", cam.camera_distance_m)
    if cam.scan_cell_meters is not None:
        cam_learned.setdefault("scan_cell_meters", cam.scan_cell_meters)

    effective_client_id: str = body.client_id or cam_client_id
    # Use camera's configured mode if the request doesn't specify one
    effective_mode: str = body.mode if body.mode != "MONITORING" else cam_mode

    # ROI is optional — preset-based scanning does not require zones.
    # Pass it through only if explicitly set on the camera or in the request body.
    effective_roi = body.roi_rect or (cam_roi if cam_roi else None)

    # Pure BULLET: no ONVIF PTZ at all. BULLET_ZOOM: still needs ONVIF for zoom.
    _is_fixed_cam = cam_camera_type == "BULLET"

    if not _is_fixed_cam and not cam_onvif_host:
        raise HTTPException(
            status_code=422,
            detail="Cannot start monitoring — camera has no ONVIF host configured.",
        )
    if not cam_rtsp_url:
        raise HTTPException(
            status_code=422,
            detail="Cannot start monitoring — camera has no RTSP URL configured.",
        )

    # ── Auto-populate roster for ATTENDANCE mode ──────────────────────────
    effective_roster = list(body.roster_ids)
    if not effective_roster and effective_mode == "ATTENDANCE" and cam_dataset_ids:
        try:
            import logging as _log
            seen: set[str] = set()
            for _did in cam_dataset_ids:
                rows = await db.execute(
                    text("SELECT person_id FROM persons WHERE dataset_id = (:did)::uuid AND status = 'ACTIVE'"),
                    {"did": str(_did)},
                )
                for r in rows.fetchall():
                    seen.add(str(r[0]))
            effective_roster = list(seen)
            _log.getLogger(__name__).info(
                "Auto-populated roster from %d dataset(s): %d persons",
                len(cam_dataset_ids), len(effective_roster),
            )
        except Exception as exc:
            import logging as _log
            _log.getLogger(__name__).warning("Failed to load roster from datasets: %s", exc)

    # ── Check for already-running session ──────────────────────────────────
    for sid, task in list(sessions.items()):
        if not task.done():
            brain_check = brains.get(sid)
            if brain_check is not None:
                try:
                    st = brain_check.get_session_state()
                    if st.camera_id == camera_id:
                        raise HTTPException(
                            status_code=409,
                            detail="A session is already running for this camera",
                        )
                except HTTPException:
                    raise
                except Exception:
                    pass
            try:
                cid_bytes = await request.app.state.redis.get(
                    f"acas:session:{sid}:camera_id"
                )
                if cid_bytes and (cid_bytes.decode() if isinstance(cid_bytes, bytes) else str(cid_bytes)) == camera_id:
                    raise HTTPException(
                        status_code=409,
                        detail="A session is already running for this camera",
                    )
            except HTTPException:
                raise
            except Exception:
                pass

    # ── Look up current timetable entry for course attribution ─────────────
    timetable_course_id:   str | None = None
    timetable_course_name: str | None = None
    timetable_faculty_id:  str | None = None
    timetable_start:       str | None = None
    timetable_end:         str | None = None
    try:
        cam_timetable_id = getattr(cam, "timetable_id", None)
        if cam_timetable_id:
            from datetime import datetime as _dt
            _now = _dt.now()
            _dow = _now.weekday()
            _hhmm = _now.strftime("%H:%M")
            tt_rows = await db.execute(
                text("""
                    SELECT course_id, course_name, faculty_id, start_time, end_time
                    FROM timetable_entries
                    WHERE timetable_id = (:tid)::uuid
                      AND day_of_week = :dow
                    ORDER BY start_time
                """),
                {"tid": str(cam_timetable_id), "dow": _dow},
            )
            for _cid, _cn, _fid, _st, _et in tt_rows.fetchall():
                if _st <= _hhmm < _et:
                    timetable_course_id   = str(_cid) if _cid else None
                    timetable_course_name = _cn
                    timetable_faculty_id  = str(_fid) if _fid else None
                    timetable_start       = _st
                    timetable_end         = _et
                    break
    except Exception:
        pass  # non-fatal; session proceeds without course info

    # ── Auto-create session row ────────────────────────────────────────────
    if body.session_id:
        effective_session_id = body.session_id
    else:
        effective_session_id = str(uuid.uuid4())
        try:
            _now_ts = now_epoch()
            await db.execute(
                text("""
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
                         :now, 'PENDING', 0,
                         :enrolled, :now)
                    ON CONFLICT (session_id) DO NOTHING
                """),
                {
                    "sid":         effective_session_id,
                    "cid":         effective_client_id,
                    "camid":       camera_id,
                    "course_id":   timetable_course_id,
                    "course_name": timetable_course_name,
                    "faculty_id":  timetable_faculty_id,
                    "sched_start": _now_ts,
                    "sched_end":   _now_ts,
                    "now":         _now_ts,
                    "enrolled":    len(effective_roster),
                },
            )
            await db.commit()
        except Exception:
            await db.rollback()

    # ── AI services ────────────────────────────────────────────────────────
    pipeline     = getattr(request.app.state, "pipeline",     None)
    face_repo    = getattr(request.app.state, "face_repo",    None)
    zone_mapper  = getattr(request.app.state, "zone_mapper",  None)
    path_planner = getattr(request.app.state, "path_planner", None)
    gpu_manager  = getattr(request.app.state, "gpu_manager",  None)

    if any(s is None for s in (pipeline, face_repo, zone_mapper, path_planner)):
        raise HTTPException(
            status_code=503,
            detail="AI services not fully initialised on this node",
        )

    from app.services.onvif_controller import FixedCameraController, CameraOfflineError
    from app.services.isapi_controller import make_camera_controller
    from app.services.rtsp_decoder import RTSPDecoder
    from app.services.zone_mapper import ZoneMapper
    from app.services.path_planner import PathPlanner
    from app.services.ptz_brain import PTZBrain

    settings = request.app.state.settings
    gpu_device = int(settings.gpu_device_id)

    # ── Capacity guard — reject if max concurrent sessions reached ─────
    active_count = sum(1 for t in sessions.values() if not t.done())
    if active_count >= settings.max_cameras_per_node:
        raise HTTPException(
            status_code=429,
            detail=f"Node at capacity ({active_count}/{settings.max_cameras_per_node} cameras). "
                   "Stop a session before starting a new one.",
        )

    # ── Build and connect camera controller ───────────────────────────────
    # Protocol is stored in learned_params["ptz_protocol"]:
    #   "ONVIF"     — default; SOAP-based (higher latency)
    #   "ISAPI"     — Hikvision direct HTTP (lowest latency)
    #   "CGI_DAHUA" — Dahua/Amcrest CGI HTTP (lowest latency)
    if _is_fixed_cam:
        # BULLET / BULLET_ZOOM: no PTZ hardware — use a no-op stub controller.
        # PTZBrain already handles these types (skips pan/tilt moves).
        onvif = FixedCameraController(rtsp_url=cam_rtsp_url)
    else:
        _protocol = cam_learned.get("ptz_protocol") or "ONVIF"
        onvif = make_camera_controller(
            host           = cam_onvif_host,
            port           = cam_onvif_port,
            username       = cam_onvif_user,
            password       = cam_onvif_pass,
            protocol       = _protocol,
            learned_params = cam_learned,
        )
        try:
            await onvif.connect()
        except CameraOfflineError:
            raise HTTPException(
                status_code=503,
                detail=f"Camera offline — {_protocol} connection failed",
            )
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"{_protocol} connection error: {exc}",
            )

    decoder       = RTSPDecoder(rtsp_url=cam_rtsp_url, fps=25, gpu_device=gpu_device)
    zone_mapper_i = ZoneMapper(onvif_controller=onvif)
    path_planner_i = PathPlanner(limits=onvif.limits)

    redis_client = request.app.state.redis

    # ── Preset loader — called at startup and after each cycle ────────────
    async def _preset_loader() -> list[dict]:
        """Load scan presets from DB for this camera with RLS context."""
        try:
            async with request.app.state.session_factory() as _sess:
                async with _sess.begin():
                    await _sess.execute(
                        text("SELECT set_config('app.current_client_id', :cid, TRUE)"),
                        {"cid": effective_client_id},
                    )
                    rows = await _sess.execute(
                        text("""
                            SELECT name, pan, tilt, zoom, dwell_s
                            FROM camera_scan_presets
                            WHERE camera_id = (:camid)::uuid
                            ORDER BY order_idx ASC
                        """),
                        {"camid": camera_id},
                    )
                    return [
                        {
                            "name":   r.name,
                            "pan":    float(r.pan),
                            "tilt":   float(r.tilt),
                            "zoom":   float(r.zoom),
                            "dwell_s": float(r.dwell_s),
                        }
                        for r in rows.mappings()
                    ]
        except Exception as exc:
            import logging as _log
            _log.getLogger(__name__).warning("preset_loader failed: %s", exc)
            return []

    brain = PTZBrain(
        camera           = onvif,
        pipeline         = pipeline,
        face_repo        = face_repo,
        zone_mapper      = zone_mapper_i,
        path_planner     = path_planner_i,
        decoder          = decoder,
        kafka_producer   = request.app.state.kafka_producer,
        redis            = redis_client,
        gpu_manager      = gpu_manager,
        learned_params   = cam_learned,
        preset_loader    = _preset_loader,
        session_factory  = getattr(request.app.state, "session_factory", None),
    )

    # ── on_complete — matches PTZBrain's Callable[[str, Exception|None], Any]
    async def _on_complete(session_id: str, error: Exception | None) -> None:
        finished_brain = brains.pop(session_id, None)
        sessions.pop(session_id, None)

        # Persist session end time
        try:
            async with request.app.state.session_factory() as sess:
                async with sess.begin():
                    await sess.execute(
                        text("""
                            UPDATE sessions SET actual_end = :ts
                            WHERE session_id = (:sid)::uuid
                        """),
                        {"ts": now_epoch(), "sid": session_id},
                    )
        except Exception:
            pass

        # Write session summary for self-learning pipeline
        learner = getattr(request.app.state, "self_learner", None)
        if learner and finished_brain and error is None:
            try:
                state = finished_brain.get_session_state()
                await learner.write_session_summary(
                    brain_state=state,
                    camera_id=camera_id,
                    client_id=effective_client_id,
                    faculty_id=body.faculty_id,
                )
            except Exception as _exc:
                import logging as _log
                _log.getLogger(__name__).warning(
                    "on_complete: write_session_summary failed: %s", _exc
                )

    # ── Load per-client face recognition thresholds ─────────────────────────
    from app.config import get_settings as _get_settings
    _s = _get_settings()
    _face_t1 = _s.faiss_tier1_threshold
    _face_t2 = _s.faiss_tier2_threshold
    try:
        from sqlalchemy import text as _t
        _cli_row = (await db.execute(
            _t("SELECT settings FROM clients WHERE client_id = (:cid)::uuid LIMIT 1"),
            {"cid": effective_client_id},
        )).fetchone()
        if _cli_row and _cli_row.settings:
            _ai_cfg = (_cli_row.settings or {}).get("ai_ptz", {})
            _face_t1 = float(_ai_cfg.get("face_tier1_threshold", _face_t1))
            _face_t2 = float(_ai_cfg.get("face_tier2_threshold", _face_t2))
    except Exception:
        pass  # fall back to module defaults

    # ── Launch ─────────────────────────────────────────────────────────────
    # run_session() creates and returns the internal asyncio.Task — don't double-wrap
    task = await brain.run_session(
        session_id           = effective_session_id,
        client_id            = effective_client_id,
        camera_id            = camera_id,
        roster_ids           = effective_roster,
        roi_rect             = effective_roi,
        faculty_id           = body.faculty_id,
        mode                 = effective_mode,
        on_complete          = _on_complete,
        dataset_id           = str(cam_dataset_id) if cam_dataset_id else None,
        dataset_ids          = cam_dataset_ids or None,
        face_tier1_threshold = _face_t1,
        face_tier2_threshold = _face_t2,
        camera_type          = cam_camera_type,
        force_start          = True,   # manual start → never auto-stop on timetable
    )
    sessions[effective_session_id] = task
    brains[effective_session_id]   = brain

    try:
        pipe = redis_client.pipeline()
        pipe.setex(f"acas:session:{effective_session_id}:client_id", 86_400, effective_client_id)
        pipe.setex(f"acas:session:{effective_session_id}:camera_id", 86_400, camera_id)
        await pipe.execute()
    except Exception:
        pass

    audit(request, "START_CAMERA_TASK", "session", effective_session_id,
          {"camera_id": camera_id, "mode": effective_mode, "roster_count": len(effective_roster)})
    return {
        "session_id": effective_session_id,
        "camera_id":  camera_id,
        "status":     "started",
        "ts":         now_epoch(),
    }


_SESSION_STOP = require_permission("sessions:update")

@router.post("/cameras/{camera_id}/stop", dependencies=[_SESSION_STOP])
async def stop_camera(
    camera_id: str,
    body:      StopCameraRequest,
    request:   Request,
    db:        DBSession,
) -> dict:
    """Stop a running PTZBrain task on this node."""
    sessions = _get_active_sessions(request)
    brains   = _get_ptz_brains(request)

    import logging as _log
    target_session_id = body.session_id if body.session_id else None

    if target_session_id is None:
        # Primary: match via PTZBrain state (authoritative, no Redis dependency)
        for sid, brain_candidate in list(brains.items()):
            try:
                state = brain_candidate.get_session_state()
                if state.camera_id == camera_id:
                    target_session_id = sid
                    break
            except Exception:
                pass

    if target_session_id is None:
        # Fallback: Redis camera_id key (set at session start)
        try:
            redis = request.app.state.redis
            for sid in list(sessions.keys()):
                try:
                    raw = await redis.get(f"acas:session:{sid}:camera_id")
                    stored_cam = raw.decode() if isinstance(raw, bytes) else str(raw) if raw else ""
                    if stored_cam == camera_id:
                        target_session_id = sid
                        break
                except Exception as exc:
                    _log.getLogger(__name__).debug("stop: Redis lookup failed for %s: %s", sid, exc)
        except Exception:
            pass

    if target_session_id is None:
        # No in-memory session found — still try to close any DB session for this
        # camera (handles backend restarts / Redis key expiry that leave orphans).
        try:
            await db.execute(
                text("""UPDATE sessions SET actual_end = :ts
                        WHERE camera_id = (:cid)::uuid
                          AND actual_end IS NULL"""),
                {"ts": now_epoch(), "cid": camera_id},
            )
            await db.commit()
        except Exception:
            await db.rollback()
        return {"session_id": None, "camera_id": camera_id, "status": "not_running", "ts": now_epoch()}

    brain = brains.get(target_session_id)
    task  = sessions.get(target_session_id)

    if task is None or task.done():
        sessions.pop(target_session_id, None)
        brains.pop(target_session_id, None)
        # Ensure the DB session row is closed even if on_complete didn't fire
        try:
            await db.execute(
                text("UPDATE sessions SET actual_end = :ts WHERE session_id = (:sid)::uuid AND actual_end IS NULL"),
                {"ts": now_epoch(), "sid": target_session_id},
            )
            await db.commit()
        except Exception:
            await db.rollback()
        return {"session_id": target_session_id, "camera_id": camera_id, "status": "not_running", "ts": now_epoch()}

    if brain is not None:
        try:
            await brain.stop()
        except Exception:
            pass
    else:
        task.cancel()

    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
    except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
        pass

    sessions.pop(target_session_id, None)
    brains.pop(target_session_id,   None)

    try:
        await db.execute(
            text("UPDATE sessions SET actual_end = :ts WHERE session_id = (:sid)::uuid"),
            {"ts": now_epoch(), "sid": target_session_id},
        )
        await db.commit()
    except Exception:
        await db.rollback()

    # Clean up Redis state keys
    try:
        redis = request.app.state.redis
        await redis.delete(
            f"acas:session:{target_session_id}:client_id",
            f"acas:session:{target_session_id}:camera_id",
            f"acas:session:{target_session_id}:state",
        )
    except Exception:
        pass

    audit(request, "STOP_CAMERA_TASK", "session", target_session_id,
          {"camera_id": camera_id, "reason": body.reason})
    return {
        "session_id": target_session_id,
        "camera_id":  camera_id,
        "status":     "stopped",
        "ts":         now_epoch(),
    }


_STATUS_PERM = require_permission("sessions:read")

@router.get("/cameras/{camera_id}/status", dependencies=[_STATUS_PERM])
async def camera_monitoring_status(
    camera_id: str,
    request:   Request,
) -> dict:
    """Return real-time monitoring status for a camera on this node."""
    sessions = _get_active_sessions(request)
    brains   = _get_ptz_brains(request)
    redis    = request.app.state.redis

    target_sid: str | None = None
    for sid in list(sessions.keys()):
        try:
            raw = await redis.get(f"acas:session:{sid}:camera_id")
            stored = raw.decode() if isinstance(raw, bytes) else str(raw) if raw else ""
            if stored == camera_id:
                target_sid = sid
                break
        except Exception:
            pass

    if target_sid is None:
        return {"monitoring": False, "camera_id": camera_id, "ts": now_epoch()}

    brain = brains.get(target_sid)
    task  = sessions.get(target_sid)

    if task is None or task.done():
        return {"monitoring": False, "camera_id": camera_id, "ts": now_epoch()}

    result: dict[str, Any] = {
        "monitoring":  True,
        "camera_id":   camera_id,
        "session_id":  target_sid,
        "ts":          now_epoch(),
    }

    if brain is not None:
        try:
            state = brain.get_session_state()
            result.update({
                "brain_state":      state.brain_state,
                "cycle_count":      state.cycle_count,
                "total_cells":      state.total_cells,
                "path_index":       state.path_index,
                "fps_actual":       round(state.fps_actual, 1),
                "cycle_time_s":     round(state.cycle_time_s, 1),
                "recognition_rate": round(state.recognition_rate, 2),
                "present_count":    state.present_count,
                "absent_count":     state.absent_count,
                "unknown_count":    state.unknown_count,
                "scan_cells":       brain._scan_cells_snapshot(),
                "current_ptz":      brain._current_ptz_snapshot(),
            })
        except Exception:
            pass
    else:
        try:
            cached = await redis.get(f"acas:session:{target_sid}:state")
            if cached:
                import json as _json
                result.update(_json.loads(cached))
        except Exception:
            pass

    return result


# ── /api/nodes — list all nodes visible to the caller ────────────────────────

@router.get("s", dependencies=[_INFO])   # prefix="/api/node"  → /api/nodes
async def list_nodes(request: Request, db: DBSession, user: CurrentUser) -> list[dict]:
    """
    Return all nodes visible to the caller.
    SUPER_ADMIN: every registered node.
    CLIENT_ADMIN/VIEWER: only nodes assigned to their client via client_node_assignments.
    """
    stale_threshold = now_epoch() - 90

    if user["role"] == "SUPER_ADMIN":
        rows = await db.execute(
            text("""
                SELECT
                    n.node_id, n.node_name, n.location, n.connectivity,
                    n.api_endpoint, n.gpu_model, n.max_cameras, n.active_cameras,
                    n.status, n.last_heartbeat, n.health_json
                FROM nodes n
                ORDER BY n.last_heartbeat DESC
            """),
        )
    else:
        rows = await db.execute(
            text("""
                SELECT
                    n.node_id, n.node_name, n.location, n.connectivity,
                    n.api_endpoint, n.gpu_model, n.max_cameras, n.active_cameras,
                    n.status, n.last_heartbeat, n.health_json
                FROM nodes n
                JOIN client_node_assignments cna ON cna.node_id = n.node_id
                WHERE cna.client_id = (:cid)::uuid
                ORDER BY n.last_heartbeat DESC
            """),
            {"cid": str(user["client_id"])},
        )

    result = []
    for r in rows.fetchall():
        effective_status = r.status if r.last_heartbeat >= stale_threshold else "OFFLINE"
        health = r.health_json if isinstance(r.health_json, dict) else {}
        result.append({
            "node_id":        r.node_id,
            "node_name":      r.node_name,
            "location":       r.location,
            "connectivity":   r.connectivity,
            "api_endpoint":   r.api_endpoint,
            "gpu_model":      r.gpu_model,
            "max_cameras":    r.max_cameras,
            "active_cameras": r.active_cameras,
            "status":         effective_status,
            "last_heartbeat": r.last_heartbeat,
            "health_json":    health,
        })
    return result
