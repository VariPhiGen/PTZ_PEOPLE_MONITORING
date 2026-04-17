"""
Cameras API — CRUD, ONVIF control, PTZ, snapshots.

All write operations require cameras:create/update/delete.
Read operations require cameras:read.
PTZ/snapshot endpoints require cameras:update (operator-level access).
"""
from __future__ import annotations

import asyncio
import base64
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator
from sqlalchemy import func, select, text

import re

from app.api._shared import audit, now_epoch, resolve_client_id
from app.deps import CurrentUser, DBSession
from app.middleware.auth import require_permission
from app.models.cameras import Camera, CameraMode, CameraStatus, CameraScanPreset, CameraType
from app.models.client_node_assignments import ClientNodeAssignment
from app.models.face_datasets import FaceDataset

router = APIRouter(prefix="/api/cameras", tags=["cameras"])

_TIME_RE = re.compile(r"^\d{1,2}:\d{2}$")
_VALID_DAYS = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
_DAY_ABBREV = {
    "mon": "monday", "tue": "tuesday", "wed": "wednesday",
    "thu": "thursday", "fri": "friday", "sat": "saturday", "sun": "sunday",
}


def _validate_monitoring_hours(hours: dict | None) -> None:
    """Validate monitoring_hours in either accepted format:

    Frontend format (preferred):
        {"start": "HH:MM", "end": "HH:MM", "days": ["MON", "TUE", ...]}

    Legacy day-keyed format:
        {"monday": [{"start": "HH:MM", "end": "HH:MM"}], "tuesday": True, ...}
    """
    if hours is None:
        return

    # ── Frontend {start, end, days} format ────────────────────────────────────
    if "start" in hours or "end" in hours or "days" in hours:
        for key in ("start", "end", "days"):
            if key not in hours:
                raise HTTPException(
                    status_code=422,
                    detail=f"monitoring_hours with 'start'/'end'/'days' format requires all three keys",
                )
        if not _TIME_RE.match(str(hours["start"])):
            raise HTTPException(
                status_code=422,
                detail=f"monitoring_hours.start must be HH:MM, got {hours['start']!r}",
            )
        if not _TIME_RE.match(str(hours["end"])):
            raise HTTPException(
                status_code=422,
                detail=f"monitoring_hours.end must be HH:MM, got {hours['end']!r}",
            )
        if not isinstance(hours["days"], list):
            raise HTTPException(status_code=422, detail="monitoring_hours.days must be a list")
        for d in hours["days"]:
            normalised = str(d).lower()
            if normalised not in _VALID_DAYS and normalised not in _DAY_ABBREV:
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid day {d!r} in monitoring_hours.days. Use full names or Mon/Tue/… abbreviations.",
                )
        return

    # ── Legacy day-keyed format ────────────────────────────────────────────────
    for day, value in hours.items():
        if day.lower() not in _VALID_DAYS:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid day in monitoring_hours: {day!r}. Valid: {sorted(_VALID_DAYS)}",
            )
        if isinstance(value, bool) or value is None:
            continue
        if not isinstance(value, list):
            raise HTTPException(
                status_code=422,
                detail=f"monitoring_hours[{day!r}] must be a list of {{start, end}} slots or a boolean",
            )
        for slot in value:
            if not isinstance(slot, dict) or "start" not in slot or "end" not in slot:
                raise HTTPException(
                    status_code=422,
                    detail=f"Each slot in monitoring_hours[{day!r}] must have 'start' and 'end' keys",
                )
            if not _TIME_RE.match(str(slot["start"])) or not _TIME_RE.match(str(slot["end"])):
                raise HTTPException(
                    status_code=422,
                    detail=f"monitoring_hours time values must be HH:MM format, got start={slot['start']!r} end={slot['end']!r}",
                )


def _validate_roi(roi: dict | list | None, field_name: str = "roi_rect") -> None:
    """Validate ROI polygon or rect dict. Coordinates must be normalised (0–1)."""
    if roi is None:
        return
    if isinstance(roi, list):
        if len(roi) < 3:
            raise HTTPException(
                status_code=422,
                detail=f"{field_name} polygon must have at least 3 vertices",
            )
        for i, pt in enumerate(roi):
            try:
                x, y = float(pt.get("x", -1)), float(pt.get("y", -1))
            except (TypeError, ValueError):
                raise HTTPException(status_code=422, detail=f"{field_name}[{i}] must have numeric x, y")
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise HTTPException(
                    status_code=422,
                    detail=f"{field_name}[{i}] coordinates must be normalised (0–1), got ({x}, {y})",
                )
    elif isinstance(roi, dict):
        for k in ("x", "y", "w", "h"):
            if k not in roi:
                raise HTTPException(status_code=422, detail=f"{field_name} dict must contain keys x, y, w, h")
        x, y, w, h = float(roi["x"]), float(roi["y"]), float(roi["w"]), float(roi["h"])
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise HTTPException(status_code=422, detail=f"{field_name} x, y must be normalised (0–1)")
        if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            raise HTTPException(status_code=422, detail=f"{field_name} w, h must be > 0 and ≤ 1")

_RD  = require_permission("cameras:read")
_WR  = require_permission("cameras:create")
_UP  = require_permission("cameras:update")
_DEL = require_permission("cameras:delete")


# ── Schemas ───────────────────────────────────────────────────────────────────

class CameraCreate(BaseModel):
    name:               str
    room_name:          str | None = None
    building:           str | None = None
    floor:              str | None = None
    rtsp_url:           str | None = None
    onvif_host:         str | None = None
    onvif_port:         int        = 80
    onvif_username:     str | None = None
    onvif_password:     str | None = None   # stored as onvif_password_encrypted
    mode:               str        = CameraMode.MONITORING
    roi_rect:           dict | list | None = None   # rect {x,y,w,h} or polygon [{x,y},…]
    faculty_zone:       dict | list | None = None
    fov_h:              float | None = None
    fov_v:              float | None = None
    pan_speed:          float | None = None
    tilt_speed:         float | None = None
    zoom_speed:         float | None = None
    monitoring_hours:   dict | None = None
    restricted_zone:    bool = False
    alert_on_unknown:   bool = False
    dataset_id:         str | None = None         # primary dataset (for backward compat)
    dataset_ids:        list[str] | None = None   # all datasets to recognise from
    timetable_id:       str | None = None         # optional linked timetable
    node_id:            str | None = None
    client_id:          str | None = None   # SUPER_ADMIN only
    # Zone / scan geometry — avoid hardcoded 7 m default
    camera_distance_m:  float = 7.0    # metres from camera to monitored area
    scan_cell_meters:   float | None = None  # scan cell width in metres; None = auto
    camera_type:        str        = CameraType.PTZ
    ptz_protocol:       str | None = None   # "ONVIF" | "ISAPI" | "CGI_DAHUA"; stored in learned_params

    @field_validator("mode")
    @classmethod
    def _mode(cls, v: str) -> str:
        if v not in (CameraMode.ATTENDANCE, CameraMode.MONITORING, CameraMode.BOTH):
            raise ValueError(f"mode must be one of {list(CameraMode)}")
        return v

    @field_validator("camera_type")
    @classmethod
    def _cam_type(cls, v: str) -> str:
        valid = {t.value for t in CameraType}
        if v not in valid:
            raise ValueError(f"camera_type must be one of {list(valid)}")
        return v


class CameraUpdate(BaseModel):
    name:               str | None = None
    room_name:          str | None = None
    building:           str | None = None
    floor:              str | None = None
    rtsp_url:           str | None = None
    onvif_host:         str | None = None
    onvif_port:         int | None = None
    onvif_username:     str | None = None
    onvif_password:     str | None = None
    mode:               str | None = None
    roi_rect:           dict | list | None = None
    faculty_zone:       dict | list | None = None
    fov_h:              float | None = None
    fov_v:              float | None = None
    pan_speed:          float | None = None
    tilt_speed:         float | None = None
    zoom_speed:         float | None = None
    monitoring_hours:   dict | None = None
    restricted_zone:    bool | None = None
    alert_on_unknown:   bool | None = None
    dataset_id:         str | None = None
    dataset_ids:        list[str] | None = None
    timetable_id:       str | None = None
    node_id:            str | None = None
    camera_distance_m:  float | None = None
    scan_cell_meters:   float | None = None
    camera_type:        str | None = None
    ptz_protocol:       str | None = None   # "ONVIF" | "ISAPI" | "CGI_DAHUA"; stored in learned_params


class ROIRequest(BaseModel):
    # Accepts legacy rect dict {x,y,w,h} OR new polygon list [{x,y}, ...]
    roi_rect:     dict | list | None = None
    faculty_zone: dict | list | None = None
    # PTZ position at the moment the zone was drawn (kept for backwards-compat).
    zone_ref_ptz: dict | None = None  # {pan, tilt, zoom}
    # Preferred: polygon vertices converted to absolute PTZ world-space [{pan, tilt}, ...].
    # Camera-position-independent — display works correctly regardless of who moved the
    # camera (AI, dashboard manual control, or physical remote).
    roi_rect_world:     list | None = None
    faculty_zone_world: list | None = None
    scan_cell_meters:   float | None = None  # cell scan size in meters
    # Per-camera FOV calibration — telephoto-end FOV in degrees (zoom=1).
    # When set, this overrides the global default (3.0° H / 1.7° V) used in
    # the log-linear FOV model.  Eliminates zoom-axis drift for cameras whose
    # actual max-zoom FOV differs from the default assumption.
    fov_h_narrow: float | None = None
    fov_v_narrow: float | None = None
    # Pan/tilt scale — corrects for cameras whose physical range ≠ 180°.
    # pan_scale = physical_pan_range_deg / 180.  CPPlus 350° → 1.944.
    pan_scale:    float | None = None
    # Tilt scale — corrects for cameras whose vertical physical range ≠ 90°.
    # tilt_scale = physical_tilt_range_deg / 90.  Defaults to pan_scale when not set.
    tilt_scale:   float | None = None


class PTZMoveRequest(BaseModel):
    pan:   float = 0.0     # -1.0 to 1.0
    tilt:  float = 0.0     # -1.0 to 1.0
    zoom:  float = 0.0     # 0.0 to 1.0
    # Accept both "mode" and "type" (dashboard uses "type")
    mode:  str | None = None
    type:  str | None = None   # alias for mode
    speed: float = 0.5
    free:  bool  = False   # when True, bypass zone-boundary clamping (used during calibration)

    @property
    def move_mode(self) -> str:
        """Resolve whichever field the caller sent."""
        v = self.mode or self.type or "absolute"
        if v not in ("absolute", "relative", "continuous", "home"):
            raise ValueError(f"mode must be absolute | relative | continuous | home, got '{v}'")
        return v


# ── Helpers ───────────────────────────────────────────────────────────────────

def _camera_dict(c: Camera, include_credentials: bool = False, node_api_endpoint: str | None = None) -> dict:
    d: dict[str, Any] = {
        "camera_id":          str(c.camera_id),
        "client_id":          str(c.client_id),
        "name":               c.name,
        "room_name":          c.room_name,
        "building":           c.building,
        "floor":              c.floor,
        "rtsp_url":           c.rtsp_url,
        "onvif_host":         c.onvif_host,
        "onvif_port":         c.onvif_port,
        "onvif_username":     c.onvif_username,
        "status":             c.status,
        "mode":               c.mode,
        "roi_rect":           c.roi_rect,
        "faculty_zone":       c.faculty_zone,
        "fov_h":              c.fov_h,
        "fov_v":              c.fov_v,
        "pan_speed":          c.pan_speed,
        "tilt_speed":         c.tilt_speed,
        "zoom_speed":         c.zoom_speed,
        "monitoring_hours":   c.monitoring_hours,
        "restricted_zone":    c.restricted_zone,
        "alert_on_unknown":   c.alert_on_unknown,
        "dataset_id":         c.dataset_id,
        "dataset_ids":        c.dataset_ids or ([c.dataset_id] if c.dataset_id else []),
        "timetable_id":       str(c.timetable_id) if c.timetable_id else None,
        "camera_type":        c.camera_type,
        "camera_distance_m":  c.camera_distance_m,
        "scan_cell_meters":   c.scan_cell_meters,
        "learned_params":     c.learned_params,
        "node_id":            c.node_id,
        "node_api_endpoint":  node_api_endpoint,   # the API base URL for this camera's node
        "created_at":         c.created_at,
        "updated_at":      c.updated_at,
    }
    return d


async def _get_camera_or_404(db: DBSession, camera_id: str) -> Camera:
    result = await db.execute(
        select(Camera).where(Camera.camera_id == uuid.UUID(camera_id))
    )
    cam = result.scalar_one_or_none()
    if cam is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    return cam


def _encrypt_onvif_password(password: str, settings: Any) -> str:
    """Encrypt plaintext ONVIF password with Fernet if key is configured."""
    key = getattr(settings, "onvif_encryption_key", None)
    if not key or not password:
        return password
    from cryptography.fernet import Fernet
    return Fernet(key.encode()).encrypt(password.encode()).decode()


def _onvif_password(cam: Camera, settings: Any = None) -> str:
    """Decrypt ONVIF password. Falls back to raw value if no key configured."""
    raw = cam.onvif_password_encrypted or ""
    if not raw:
        return raw
    key = getattr(settings, "onvif_encryption_key", None) if settings else None
    if not key:
        return raw
    try:
        from cryptography.fernet import Fernet, InvalidToken
        return Fernet(key.encode()).decrypt(raw.encode()).decode()
    except (InvalidToken, Exception):
        return raw  # not encrypted or wrong key — return as-is


async def _build_onvif(cam: Camera, settings: Any = None, request: Request | None = None) -> Any:
    """
    Return a connected camera controller for *cam*.

    Controllers are cached in ``request.app.state._onvif_cache`` keyed by
    ``camera_id``.  A cached entry is reused as long as it is ≤ 5 minutes old
    AND the camera credentials/host have not changed.  This avoids 4 SOAP
    round-trips (GetDeviceInformation, GetCapabilities, GetProfiles,
    GetStreamUri) on every API call.
    """
    import time as _time
    from app.services.onvif_controller import CameraOfflineError
    from app.services.isapi_controller import make_camera_controller
    if not cam.onvif_host:
        raise HTTPException(status_code=400, detail="Camera has no ONVIF host configured")

    # ── Cache key: camera_id + host:port + username (password changes invalidate) ──
    cache_key = f"{cam.camera_id}:{cam.onvif_host}:{cam.onvif_port}:{cam.onvif_username}"
    _TTL = 300  # seconds — reuse the connection for up to 5 minutes

    if request is not None:
        cache: dict = getattr(request.app.state, "_onvif_cache", None)
        if cache is None:
            request.app.state._onvif_cache = {}
            cache = request.app.state._onvif_cache
        entry = cache.get(cache_key)
        if entry is not None:
            ctrl, ts, stored_pw = entry
            if _time.monotonic() - ts < _TTL and stored_pw == _onvif_password(cam, settings):
                if getattr(ctrl, "is_connected", False):
                    return ctrl
            # Stale or credential-changed — evict
            cache.pop(cache_key, None)

    # ── Build fresh controller ─────────────────────────────────────────────────
    lp: dict = dict(cam.learned_params or {})
    if cam.fov_h and "fov_h_wide" not in lp:
        lp["fov_h_wide"] = cam.fov_h
    if cam.fov_v and "fov_v_wide" not in lp:
        lp["fov_v_wide"] = cam.fov_v
    protocol = lp.get("ptz_protocol") or "ONVIF"
    pw = _onvif_password(cam, settings)
    ctrl = make_camera_controller(
        host=cam.onvif_host,
        port=cam.onvif_port or 80,
        username=cam.onvif_username or "",
        password=pw,
        protocol=protocol,
        learned_params=lp,
    )
    try:
        await ctrl.connect()
    except CameraOfflineError as exc:
        raise HTTPException(status_code=503, detail=f"Camera unreachable: {exc}")

    if request is not None:
        request.app.state._onvif_cache[cache_key] = (ctrl, _time.monotonic(), pw)
    return ctrl


# ── Test connection (pre-save) ────────────────────────────────────────────────

class TestConnectionBody(BaseModel):
    rtsp_url:       str | None = None
    onvif_host:     str | None = None
    onvif_port:     int        = 80
    onvif_username: str | None = None
    onvif_password: str | None = None


@router.post("/test-connection", dependencies=[_RD])
async def test_connection(body: TestConnectionBody) -> dict:
    """
    Probe ONVIF + RTSP before the camera record is saved.
    Returns a result dict with onvif_ok, rtsp_ok, details.
    """
    import subprocess, shlex, socket

    result: dict = {
        "onvif_ok":   False,
        "rtsp_ok":    False,
        "onvif_info": None,
        "rtsp_info":  None,
        "error":      None,
    }

    # ── ONVIF probe ──────────────────────────────────────────────────────────
    if body.onvif_host:
        try:
            from app.services.onvif_controller import ONVIFController, CameraOfflineError
            ctrl = ONVIFController(
                host     = body.onvif_host,
                port     = body.onvif_port or 80,
                username = body.onvif_username or "",
                password = body.onvif_password or "",
            )
            await ctrl.connect()   # async connect with retry
            info: dict = {}
            try:
                device_info = await asyncio.to_thread(
                    lambda: ctrl._cam.devicemgmt.GetDeviceInformation()
                )
                info["manufacturer"] = getattr(device_info, "Manufacturer", "")
                info["model"]        = getattr(device_info, "Model", "")
                info["firmware"]     = getattr(device_info, "FirmwareVersion", "")
            except Exception:
                pass
            try:
                info["rtsp_url"] = await ctrl.get_rtsp_url()
            except Exception:
                pass
            try:
                status = await ctrl.get_ptz_status()
                info["ptz_status"] = status
            except Exception:
                pass
            result["onvif_ok"]   = True
            result["onvif_info"] = info
        except Exception as exc:
            result["error"] = f"ONVIF: {exc}"

    # ── RTSP probe via ffprobe ────────────────────────────────────────────────
    rtsp_url = body.rtsp_url
    if not rtsp_url and result.get("onvif_info", {}) and result["onvif_info"].get("rtsp_url"):
        rtsp_url = result["onvif_info"]["rtsp_url"]

    if rtsp_url:
        try:
            cmd = [
                "ffprobe", "-v", "quiet",
                "-rtsp_transport", "tcp",
                "-user_agent",     "LibVLC",
                "-analyzeduration", "2000000",
                "-probesize",       "1000000",
                "-print_format",    "json",
                "-show_streams",
                rtsp_url,
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
            except asyncio.TimeoutError:
                proc.kill()
                result["error"] = (result.get("error") or "") + " | RTSP: timed out"
                stdout, stderr = b"{}", b""

            if proc.returncode == 0:
                import json
                probe = json.loads(stdout or b"{}")
                streams = probe.get("streams", [])
                video = next((s for s in streams if s.get("codec_type") == "video"), None)
                result["rtsp_ok"] = True
                result["rtsp_info"] = {
                    "codec":  video.get("codec_name")   if video else None,
                    "width":  video.get("width")         if video else None,
                    "height": video.get("height")        if video else None,
                    "fps":    video.get("r_frame_rate")  if video else None,
                    "streams": len(streams),
                }
            else:
                err_msg = stderr.decode(errors="ignore").strip().split("\n")[-1]
                result["error"] = (result.get("error") or "") + f" | RTSP: {err_msg}"
        except Exception as exc:
            result["error"] = (result.get("error") or "") + f" | RTSP: {exc}"

    if not result["onvif_ok"] and not result["rtsp_ok"]:
        raise HTTPException(status_code=422, detail=result.get("error") or "Both ONVIF and RTSP probes failed")

    return result


# ── CRUD ──────────────────────────────────────────────────────────────────────

@router.post("", status_code=201, dependencies=[_WR])
async def create_camera(body: CameraCreate, request: Request, db: DBSession) -> dict:
    client_id = resolve_client_id(request, body.client_id)

    # Node assignment check
    if body.node_id:
        result = await db.execute(
            select(ClientNodeAssignment).where(
                ClientNodeAssignment.client_id == uuid.UUID(client_id),
                ClientNodeAssignment.node_id   == body.node_id,
            )
        )
        if result.scalar_one_or_none() is None:
            raise HTTPException(status_code=400, detail="node_id not assigned to this client")

    # Camera count quota
    from app.models.clients import Client
    client = await db.get(Client, uuid.UUID(client_id))
    if client is None:
        raise HTTPException(status_code=404, detail="Client not found")
    cam_count = await db.scalar(
        select(func.count()).select_from(Camera).where(Camera.client_id == uuid.UUID(client_id))
    ) or 0
    if cam_count >= client.max_cameras:
        raise HTTPException(status_code=409, detail=f"Camera quota reached ({client.max_cameras})")

    # Camera name uniqueness per client
    dup_name = await db.scalar(
        select(func.count()).select_from(Camera).where(
            Camera.client_id == uuid.UUID(client_id),
            Camera.name == body.name,
        )
    )
    if dup_name:
        raise HTTPException(status_code=409, detail=f"Camera name '{body.name}' already exists for this client")

    # Validate ROI and monitoring hours
    _validate_roi(body.roi_rect, "roi_rect")
    _validate_roi(body.faculty_zone, "faculty_zone")
    _validate_monitoring_hours(body.monitoring_hours)

    # Encrypt ONVIF password if key is configured
    settings = getattr(request.app.state, "settings", None)
    encrypted_pw = _encrypt_onvif_password(body.onvif_password or "", settings) if body.onvif_password else None

    # Resolve dataset_ids / dataset_id:
    # - If caller passed dataset_ids, use that list; primary = first element.
    # - If caller passed only dataset_id (legacy), wrap it in a list.
    # - If neither, auto-assign the client's default dataset.
    resolved_ids: list[str] = []
    if body.dataset_ids:
        resolved_ids = [d for d in body.dataset_ids if d]
    elif body.dataset_id:
        resolved_ids = [body.dataset_id]

    if not resolved_ids:
        default_ds = await db.scalar(
            select(FaceDataset.dataset_id).where(
                FaceDataset.client_id == uuid.UUID(client_id),
                FaceDataset.is_default.is_(True),
                FaceDataset.status == "ACTIVE",
            ).limit(1)
        )
        if default_ds:
            resolved_ids = [str(default_ds)]

    dataset_id  = resolved_ids[0] if resolved_ids else None
    dataset_ids = resolved_ids or None

    ts = now_epoch()
    cam = Camera(
        client_id              = uuid.UUID(client_id),
        name                   = body.name,
        room_name              = body.room_name,
        building               = body.building,
        floor                  = body.floor,
        rtsp_url               = body.rtsp_url,
        onvif_host             = body.onvif_host,
        onvif_port             = body.onvif_port,
        onvif_username         = body.onvif_username,
        onvif_password_encrypted = encrypted_pw,
        mode                   = body.mode,
        roi_rect               = body.roi_rect,
        faculty_zone           = body.faculty_zone,
        fov_h                  = body.fov_h,
        fov_v                  = body.fov_v,
        pan_speed              = body.pan_speed,
        tilt_speed             = body.tilt_speed,
        zoom_speed             = body.zoom_speed,
        monitoring_hours       = body.monitoring_hours,
        restricted_zone        = body.restricted_zone,
        alert_on_unknown       = body.alert_on_unknown,
        dataset_id             = dataset_id,
        dataset_ids            = dataset_ids,
        timetable_id           = uuid.UUID(body.timetable_id) if body.timetable_id else None,
        node_id                = body.node_id,
        camera_type            = body.camera_type,
        camera_distance_m      = body.camera_distance_m,
        scan_cell_meters       = body.scan_cell_meters,
        learned_params         = {"ptz_protocol": body.ptz_protocol} if body.ptz_protocol else None,
        created_at             = ts,
        updated_at             = ts,
    )
    db.add(cam)
    await db.flush()
    await db.refresh(cam)
    audit(request, "CREATE_CAMERA", "camera", str(cam.camera_id))
    return _camera_dict(cam)


@router.get("", dependencies=[_RD])
async def list_cameras(
    request: Request,
    db:      DBSession,
    q:       str | None = Query(None),
    status:  str | None = Query(None),
    mode:    str | None = Query(None),
    node_id: str | None = Query(None),
    limit:   int        = Query(50, ge=1, le=200),
    offset:  int        = Query(0, ge=0),
) -> dict:
    client_id = resolve_client_id(request, require=False)
    stmt = select(Camera)
    if client_id:
        stmt = stmt.where(Camera.client_id == uuid.UUID(client_id))
    if q:
        like = f"%{q}%"
        stmt = stmt.where(
            Camera.name.ilike(like)
            | Camera.room_name.ilike(like)
            | Camera.building.ilike(like)
            | Camera.onvif_host.ilike(like)
        )
    if status:
        stmt = stmt.where(Camera.status == status)
    if mode:
        stmt = stmt.where(Camera.mode == mode)
    if node_id:
        stmt = stmt.where(Camera.node_id == node_id)

    total  = await db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    result = await db.execute(stmt.order_by(Camera.created_at.desc()).offset(offset).limit(limit))
    cams   = result.scalars().all()

    # Enrich with node API endpoints in one query
    node_ids = list({c.node_id for c in cams if c.node_id})
    node_endpoints: dict[str, str] = {}
    if node_ids:
        rows = await db.execute(
            text("SELECT node_id, api_endpoint FROM nodes WHERE node_id = ANY(:ids)"),
            {"ids": node_ids},
        )
        node_endpoints = {r.node_id: r.api_endpoint for r in rows}

    return {
        "total": total, "offset": offset, "limit": limit,
        "items": [
            _camera_dict(c, node_api_endpoint=node_endpoints.get(c.node_id or ""))
            for c in cams
        ],
    }


@router.get("/available-nodes", dependencies=[_RD])
async def available_nodes(request: Request, db: DBSession) -> list:
    """
    Nodes assigned to the current user's client, in full GpuNode shape.
    CLIENT_ADMIN/VIEWER: their client's nodes.
    SUPER_ADMIN: all nodes (so the cluster page works when logged in as super admin).
    """
    from sqlalchemy import text as _text
    import time as _time
    stale = int(_time.time()) - 90
    cid = resolve_client_id(request, require=False)

    if cid:
        rows = await db.execute(_text("""
            SELECT n.node_id, n.node_name, n.location, n.connectivity,
                   n.api_endpoint, n.gpu_model, n.max_cameras, n.active_cameras,
                   n.status, n.last_heartbeat, n.health_json
            FROM nodes n
            JOIN client_node_assignments cna ON cna.node_id = n.node_id
            WHERE cna.client_id = (:cid)::uuid
            ORDER BY n.last_heartbeat DESC
        """), {"cid": cid})
    else:
        rows = await db.execute(_text("""
            SELECT node_id, node_name, location, connectivity,
                   api_endpoint, gpu_model, max_cameras, active_cameras,
                   status, last_heartbeat, health_json
            FROM nodes ORDER BY last_heartbeat DESC
        """))

    result = []
    for r in rows.fetchall():
        health = dict(r.health_json) if isinstance(r.health_json, dict) else {}
        # Normalise NodeManager key names → dashboard expected names
        health["gpu_util"]      = health.get("gpu_util")     or health.get("gpu0_utilization", 0)
        health["cpu_util"]      = health.get("cpu_util")     or health.get("cpu_percent", 0)
        health["ram_util"]      = health.get("ram_util")     or health.get("ram_percent", 0)
        health["vram_used_gb"]  = health.get("vram_used_gb") or round(health.get("gpu0_mem_used_mb", 0) / 1024, 1)
        health["vram_total_gb"] = health.get("vram_total_gb") or round(health.get("gpu0_mem_total_mb", 24576) / 1024, 1)
        health["connectivity"]  = r.connectivity
        health.setdefault("latency_ms",     0)
        health.setdefault("uptime_pct",     100)
        health.setdefault("faces_db_size",  0)
        health.setdefault("throughput_fps", 0)
        result.append({
            "node_id":        r.node_id,
            "node_name":      r.node_name,
            "location":       r.location,
            "connectivity":   r.connectivity,
            "api_endpoint":   r.api_endpoint,
            "gpu_model":      r.gpu_model,
            "max_cameras":    r.max_cameras,
            "active_cameras": r.active_cameras,
            "status":         r.status if r.last_heartbeat >= stale else "OFFLINE",
            "last_heartbeat": r.last_heartbeat,
            "health_json":    health,
        })
    return result


@router.get("/{camera_id}", dependencies=[_RD])
async def get_camera(camera_id: str, db: DBSession) -> dict:
    cam = await _get_camera_or_404(db, camera_id)
    return _camera_dict(cam)


@router.put("/{camera_id}", dependencies=[_UP])
async def update_camera(camera_id: str, body: CameraUpdate, request: Request, db: DBSession) -> dict:
    cam = await _get_camera_or_404(db, camera_id)

    # Use exclude_unset so callers can explicitly pass null to clear a field
    updates = body.model_dump(exclude_unset=True)

    # Validate fields if provided (skip validation when value is None — clearing is always OK)
    if updates.get("roi_rect") is not None:
        _validate_roi(updates["roi_rect"], "roi_rect")
    if updates.get("faculty_zone") is not None:
        _validate_roi(updates["faculty_zone"], "faculty_zone")
    if "monitoring_hours" in updates and updates["monitoring_hours"] is not None:
        _validate_monitoring_hours(updates["monitoring_hours"])

    # Camera name uniqueness per client (only when name is being changed)
    if "name" in updates and updates["name"] != cam.name:
        dup_name = await db.scalar(
            select(func.count()).select_from(Camera).where(
                Camera.client_id == cam.client_id,
                Camera.name == updates["name"],
            )
        )
        if dup_name:
            raise HTTPException(status_code=409, detail=f"Camera name '{updates['name']}' already exists for this client")

    settings = getattr(request.app.state, "settings", None)
    for field, val in updates.items():
        if field == "onvif_password":
            cam.onvif_password_encrypted = _encrypt_onvif_password(val, settings) if val else None
        elif field == "timetable_id":
            cam.timetable_id = uuid.UUID(val) if val else None
        elif field == "camera_type":
            cam.camera_type = val or CameraType.PTZ
        elif field == "ptz_protocol":
            # Merge ptz_protocol into learned_params so it travels with calibration data
            lp = dict(cam.learned_params or {})
            if val:
                lp["ptz_protocol"] = val
            else:
                lp.pop("ptz_protocol", None)
            cam.learned_params = lp
        else:
            setattr(cam, field, val)

    # Keep dataset_id / dataset_ids in sync
    if "dataset_ids" in updates:
        ids = [d for d in (updates["dataset_ids"] or []) if d]
        cam.dataset_ids = ids or None
        cam.dataset_id  = ids[0] if ids else None
    elif "dataset_id" in updates and "dataset_ids" not in updates:
        # Legacy single-dataset update — rebuild the list
        did = updates["dataset_id"]
        if did:
            existing = list(cam.dataset_ids or [])
            if did not in existing:
                existing.insert(0, did)
            cam.dataset_ids = existing
        else:
            cam.dataset_ids = None

    cam.updated_at = now_epoch()
    await db.flush()
    await db.refresh(cam)
    audit(request, "UPDATE_CAMERA", "camera", camera_id)
    return _camera_dict(cam)


@router.delete("/{camera_id}", status_code=204, dependencies=[_DEL])
async def delete_camera(camera_id: str, request: Request, db: DBSession) -> None:
    cam = await _get_camera_or_404(db, camera_id)
    await db.delete(cam)
    audit(request, "DELETE_CAMERA", "camera", camera_id)


# ── Operational endpoints ─────────────────────────────────────────────────────

@router.post("/{camera_id}/test", dependencies=[_RD])
async def test_camera(camera_id: str, request: Request, db: DBSession) -> dict:
    """Test camera connectivity. BULLET cameras probe RTSP; PTZ cameras probe ONVIF."""
    from app.services.onvif_controller import ONVIFController
    cam = await _get_camera_or_404(db, camera_id)

    # BULLET cameras have no PTZ service — probe the RTSP port directly
    if getattr(cam, "camera_type", None) == "BULLET":
        import socket as _socket
        rtsp_url = cam.rtsp_url or ""
        # Extract host:port from rtsp_url (rtsp://[user:pass@]host[:port]/...)
        try:
            from urllib.parse import urlparse
            parsed   = urlparse(rtsp_url)
            rtsp_host = parsed.hostname or (cam.onvif_host or "")
            rtsp_port = parsed.port or 554
            reachable = False
            try:
                await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: _socket.create_connection((rtsp_host, rtsp_port), timeout=4).close()
                    ),
                    timeout=5,
                )
                reachable = True
            except Exception:
                pass
        except Exception:
            reachable = False

        cam.status = CameraStatus.ONLINE if reachable else CameraStatus.OFFLINE
        await db.flush()
        return {
            "online":      reachable,
            "rtsp_url":    rtsp_url,
            "device_info": {"type": "fixed_bullet"},
            **({"error": "rtsp_unreachable"} if not reachable else {}),
        }

    if not cam.onvif_host:
        return {"online": False, "error": "no_onvif_host"}

    settings = getattr(request.app.state, "settings", None)
    try:
        ctrl = ONVIFController(
            host     = cam.onvif_host,
            port     = cam.onvif_port or 80,
            username = cam.onvif_username or "",
            password = _onvif_password(cam, settings),
        )
        await ctrl.connect()
        ptz_status = await ctrl.get_ptz_status()
        rtsp_url   = await ctrl.get_rtsp_url()
        device_info: dict = {}
        try:
            di = await asyncio.to_thread(lambda: ctrl._cam.devicemgmt.GetDeviceInformation())
            device_info = {
                "manufacturer": getattr(di, "Manufacturer", ""),
                "model":        getattr(di, "Model", ""),
                "firmware":     getattr(di, "FirmwareVersion", ""),
            }
        except Exception:
            pass
        cam.status = CameraStatus.ONLINE
        await db.flush()
        return {
            "online":      True,
            "ptz_status":  ptz_status,
            "rtsp_url":    rtsp_url,
            "device_info": device_info,
        }
    except Exception as exc:
        cam.status = CameraStatus.OFFLINE
        await db.flush()
        return {"online": False, "error": str(exc)}


@router.post("/{camera_id}/calibrate-fov", dependencies=[_UP])
async def calibrate_fov(camera_id: str, request: Request, db: DBSession) -> dict:
    """
    Auto-calibrate the camera's FOV by measuring optical-flow pixel displacement
    for known ONVIF movements at wide and narrow zoom.

    Stores K_pan_wide, K_pan_narrow, K_tilt_wide, K_tilt_narrow (and derived
    fov_h_wide/narrow) in camera learned_params.  Takes ~30 s.

    Requires the camera to have a textured scene visible (not a blank wall).
    The camera must be controllable (ONVIF PTZ must be connected).
    """
    from app.services.onvif_controller import ONVIFController, CameraOfflineError
    from app.services.rtsp_decoder import RTSPDecoder
    import asyncio as _asyncio
    import json as _json

    cam = await _get_camera_or_404(db, camera_id)
    if not cam.onvif_host:
        raise HTTPException(status_code=400, detail="Camera has no ONVIF host configured")

    settings = getattr(request.app.state, "settings", None)
    lp: dict = cam.learned_params or {}

    # Snapshot what we need before releasing the DB connection — the calibration
    # takes ~45 s, and holding the idle-in-transaction session that long trips
    # PostgreSQL's idle-in-transaction timeout, causing the eventual UPDATE to
    # fail with "connection is closed" → cascading 500 to the UI.
    _onvif_host     = cam.onvif_host
    _onvif_port     = cam.onvif_port or 80
    _onvif_username = cam.onvif_username or ""
    _onvif_password_v = _onvif_password(cam, settings)
    _learned_params_snapshot = dict(cam.learned_params or {})
    # Commit the short read transaction so the pooled connection is released
    # back to asyncpg. A fresh session is opened at the end to persist results.
    await db.commit()

    ctrl = ONVIFController(
        host=_onvif_host,
        port=_onvif_port,
        username=_onvif_username,
        password=_onvif_password_v,
        K_pan_wide=lp.get("K_pan_wide")       or None,
        K_pan_narrow=lp.get("K_pan_narrow")   or None,
        K_tilt_wide=lp.get("K_tilt_wide")     or None,
        K_tilt_narrow=lp.get("K_tilt_narrow") or None,
        # pan_scale / tilt_scale feed the fov_h/fov_v inversion inside
        # auto_calibrate_fov. Without them the derived FoVs are scaled by the
        # wrong factor and produce unphysical values like 150–200°.
        pan_scale=lp.get("pan_scale")  or None,
        tilt_scale=lp.get("tilt_scale") or None,
    )
    try:
        await ctrl.connect()
    except CameraOfflineError as exc:
        raise HTTPException(status_code=503, detail=f"Camera offline: {exc}")

    rtsp_url = await ctrl.get_rtsp_url()
    gpu_device = int(getattr(settings, "gpu_device_id", 0)) if settings else 0
    decoder = RTSPDecoder(rtsp_url=rtsp_url, fps=15, gpu_device=gpu_device)

    # RTSPDecoder exposes an async generator (`frames()`), not a start/get_frame
    # pair. Run it in a background task that keeps the most recent frame in a
    # shared slot; _grab() returns that slot.
    import numpy as _np
    latest: dict[str, _np.ndarray | None] = {"frame": None}
    frame_evt = _asyncio.Event()

    async def _drain():
        async for _fid, _ts, frame in decoder.frames():
            latest["frame"] = frame
            frame_evt.set()

    drain_task = _asyncio.create_task(_drain())
    try:
        # Wait up to 10 s for the first frame (stream handshake + key-frame).
        try:
            await _asyncio.wait_for(frame_evt.wait(), timeout=10.0)
        except _asyncio.TimeoutError:
            raise HTTPException(
                status_code=503,
                detail="No frames received from camera RTSP stream within 10 s.",
            )

        async def _grab():
            frame_evt.clear()
            try:
                await _asyncio.wait_for(frame_evt.wait(), timeout=2.0)
            except _asyncio.TimeoutError:
                pass
            return latest["frame"]

        calib = await ctrl.auto_calibrate_fov(_grab)
    finally:
        await decoder.stop()
        drain_task.cancel()
        try:
            await drain_task
        except (_asyncio.CancelledError, Exception):
            pass

    if not calib:
        raise HTTPException(
            status_code=422,
            detail="Calibration produced no measurements. Ensure a textured scene "
                   "is visible and the camera is not at its pan/tilt limits.",
        )

    # Persist to DB — merge into existing learned_params using a FRESH session.
    # The dependency-injected `db` was committed before the long calibration,
    # so its context may no longer be valid. Opening a new short-lived session
    # avoids any idle-connection or stale-transaction pitfalls.
    merged = {**_learned_params_snapshot, **calib}
    from sqlalchemy import text as _text
    async with request.app.state.session_factory() as write_sess:
        async with write_sess.begin():
            # Re-apply RLS client_id scope on the new session.
            _client_id = getattr(request.state, "client_id", None)
            await write_sess.execute(
                _text("SELECT set_config('app.current_client_id', :val, TRUE)"),
                {"val": str(_client_id) if _client_id else ""},
            )
            await write_sess.execute(
                _text("UPDATE cameras SET learned_params = (:lp)::jsonb "
                      "WHERE camera_id = (:cid)::uuid"),
                {"lp": _json.dumps(merged), "cid": camera_id},
            )

    return {"status": "ok", "calibration": calib}


@router.post("/{camera_id}/roi", dependencies=[_UP])
async def set_roi(camera_id: str, body: ROIRequest, request: Request, db: DBSession) -> dict:
    """
    Update or clear the camera's region-of-interest and optional faculty zone.

    Pass roi_rect=null to delete the zone (also clears world-space calibration data).
    """
    _validate_roi(body.roi_rect, "roi_rect")
    _validate_roi(body.faculty_zone, "faculty_zone")
    cam = await _get_camera_or_404(db, camera_id)

    cam.roi_rect     = body.roi_rect
    cam.faculty_zone = body.faculty_zone

    # Maintain the world-space calibration in learned_params.
    # When a zone is explicitly cleared (set to null), remove the matching
    # world-space keys so stale data does not mislead PTZBrain.
    existing = dict(cam.learned_params) if cam.learned_params else {}

    if body.roi_rect is None:
        # Zone deleted — purge all zone world-space and reference data
        for key in ("zone_ref_ptz", "roi_rect_world"):
            existing.pop(key, None)
    else:
        if body.zone_ref_ptz   is not None: existing["zone_ref_ptz"]   = body.zone_ref_ptz
        if body.roi_rect_world is not None: existing["roi_rect_world"] = body.roi_rect_world

    if body.faculty_zone is None:
        existing.pop("faculty_zone_world", None)
    else:
        if body.faculty_zone_world is not None:
            existing["faculty_zone_world"] = body.faculty_zone_world

    # scan_cell_meters in the body overrides the camera column value
    if body.scan_cell_meters is not None:
        cam.scan_cell_meters = body.scan_cell_meters
        existing.pop("scan_cell_meters", None)   # keep it on the column, not learned_params

    # Per-camera FOV calibration — stored in learned_params so they survive
    # column-level updates and are available to _build_onvif() at any time.
    if body.fov_h_narrow is not None:
        existing["fov_h_narrow"] = body.fov_h_narrow
    if body.fov_v_narrow is not None:
        existing["fov_v_narrow"] = body.fov_v_narrow
    if body.pan_scale is not None:
        existing["pan_scale"] = body.pan_scale
    if body.tilt_scale is not None:
        existing["tilt_scale"] = body.tilt_scale

    cam.learned_params = existing or None
    cam.updated_at = now_epoch()
    await db.flush()
    audit(request, "SET_ROI", "camera", camera_id)
    lp = cam.learned_params or {}
    return {
        "camera_id":           camera_id,
        "roi_rect":            cam.roi_rect,
        "faculty_zone":        cam.faculty_zone,
        "camera_distance_m":   cam.camera_distance_m,
        "scan_cell_meters":    cam.scan_cell_meters,
        "zone_ref_ptz":        lp.get("zone_ref_ptz"),
        "roi_rect_world":      lp.get("roi_rect_world"),
        "faculty_zone_world":  lp.get("faculty_zone_world"),
    }


@router.delete("/{camera_id}/roi", status_code=204, dependencies=[_UP])
async def delete_roi(camera_id: str, request: Request, db: DBSession) -> None:
    """
    Fully delete the zone definition for a camera.

    Clears roi_rect, faculty_zone, and all world-space calibration data
    (roi_rect_world, faculty_zone_world, zone_ref_ptz) from learned_params.
    The camera will revert to scanning the full frame.
    """
    cam = await _get_camera_or_404(db, camera_id)
    cam.roi_rect     = None
    cam.faculty_zone = None

    # Strip zone keys from learned_params; preserve other learned data
    if cam.learned_params:
        lp = dict(cam.learned_params)
        for key in ("zone_ref_ptz", "roi_rect_world", "faculty_zone_world"):
            lp.pop(key, None)
        cam.learned_params = lp or None

    cam.updated_at = now_epoch()
    await db.flush()
    audit(request, "DELETE_ROI", "camera", camera_id)


@router.get("/{camera_id}/snapshot", dependencies=[_RD])
async def get_snapshot(camera_id: str, request: Request, db: DBSession) -> dict:
    """Capture a JPEG snapshot from the camera via ONVIF.  Returns base64-encoded JPEG."""
    cam = await _get_camera_or_404(db, camera_id)
    settings = getattr(request.app.state, "settings", None)
    ctrl = await _build_onvif(cam, settings, request)
    try:
        image_bytes = await ctrl.get_snapshot()
        return {
            "camera_id": camera_id,
            "format":    "jpeg",
            "data":      base64.b64encode(image_bytes).decode(),
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Snapshot failed: {exc}")


def _clamp_to_zone(pan: float, tilt: float, zoom: float, cam: Camera, ctrl) -> tuple[float, float, bool]:
    """
    Clamp pan/tilt so the camera centre stays within the ROI world-space bounding
    box (with a half-FOV margin so the zone edge is visible).
    Returns (clamped_pan, clamped_tilt, was_clamped).
    """
    lp = cam.learned_params or {}
    roi_world = lp.get("roi_rect_world")
    if not roi_world or not isinstance(roi_world, list) or len(roi_world) < 3:
        return pan, tilt, False

    pans  = [float(p.get("pan",  0)) for p in roi_world if isinstance(p, dict)]
    tilts = [float(p.get("tilt", 0)) for p in roi_world if isinstance(p, dict)]
    if not pans or not tilts:
        return pan, tilt, False

    fov_h, fov_v = ctrl.get_fov_at_zoom(zoom)
    pan_range  = ctrl.limits.pan_max  - ctrl.limits.pan_min
    tilt_range = ctrl.limits.tilt_max - ctrl.limits.tilt_min
    # Half-FOV margin in ONVIF units — keeps zone edge at frame edge
    margin_pan  = (fov_h / 360.0) * pan_range  / 2.0
    margin_tilt = (fov_v / 180.0) * tilt_range / 2.0

    zone_pan_min  = min(pans)  + margin_pan
    zone_pan_max  = max(pans)  - margin_pan
    zone_tilt_min = min(tilts) + margin_tilt
    zone_tilt_max = max(tilts) - margin_tilt

    # If zone is smaller than FOV, target the centre instead of clamping
    if zone_pan_min > zone_pan_max:
        zone_pan_min = zone_pan_max = (min(pans) + max(pans)) / 2.0
    if zone_tilt_min > zone_tilt_max:
        zone_tilt_min = zone_tilt_max = (min(tilts) + max(tilts)) / 2.0

    clamped_pan  = max(zone_pan_min,  min(zone_pan_max,  pan))
    clamped_tilt = max(zone_tilt_min, min(zone_tilt_max, tilt))
    was_clamped  = (clamped_pan != pan or clamped_tilt != tilt)
    return clamped_pan, clamped_tilt, was_clamped


@router.post("/{camera_id}/ptz-move", dependencies=[_UP])
async def ptz_move(camera_id: str, body: PTZMoveRequest, request: Request, db: DBSession) -> dict:
    """Execute a PTZ move command, clamping pan/tilt within the zone bounds."""
    cam = await _get_camera_or_404(db, camera_id)
    settings = getattr(request.app.state, "settings", None)
    ctrl = await _build_onvif(cam, settings, request)
    try:
        m = body.move_mode
        clamped = False
        if m == "absolute":
            if body.free:
                p, t = body.pan, body.tilt
            else:
                p, t, clamped = _clamp_to_zone(body.pan, body.tilt, body.zoom, cam, ctrl)
            await ctrl.absolute_move(p, t, body.zoom, body.speed)
        elif m == "relative":
            # Relative: resolve target first, clamp, then move absolute
            status = await ctrl.get_ptz_status()
            target_pan  = status.pan  + body.pan
            target_tilt = status.tilt + body.tilt
            target_zoom = max(0.0, min(1.0, status.zoom + body.zoom))
            if body.free:
                p, t = target_pan, target_tilt
            else:
                p, t, clamped = _clamp_to_zone(target_pan, target_tilt, target_zoom, cam, ctrl)
            await ctrl.absolute_move(p, t, target_zoom, body.speed)
        elif m == "home":
            await ctrl.goto_home(speed=body.speed)
        else:  # continuous
            await ctrl.continuous_move(body.pan, body.tilt, body.zoom, timeout=2.0)
        audit(request, "PTZ_MOVE", "camera", camera_id,
              {"mode": m, "pan": body.pan, "tilt": body.tilt, "zoom": body.zoom, "free": body.free})
        return {"ok": True, "mode": m, "clamped": clamped}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"PTZ move failed: {exc}")


@router.get("/{camera_id}/ptz-status", dependencies=[_RD])
async def ptz_status(camera_id: str, request: Request, db: DBSession) -> dict:
    """Return current PTZ position."""
    cam = await _get_camera_or_404(db, camera_id)
    settings = getattr(request.app.state, "settings", None)
    ctrl = await _build_onvif(cam, settings, request)
    try:
        status = await ctrl.get_ptz_status()
        lim = ctrl.limits
        return {
            "camera_id": camera_id,
            "pan": status.pan, "tilt": status.tilt, "zoom": status.zoom,
            "pan_min":  lim.pan_min,  "pan_max":  lim.pan_max,
            "tilt_min": lim.tilt_min, "tilt_max": lim.tilt_max,
            **ctrl.fov_params,   # fov_h_wide, fov_h_narrow, fov_v_wide, fov_v_narrow
            "pan_scale":  (cam.learned_params or {}).get("pan_scale",  1.0),
            "tilt_scale": (cam.learned_params or {}).get("tilt_scale", None),
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"PTZ status failed: {exc}")


# ── MJPEG live stream ─────────────────────────────────────────────────────────

@router.get("/{camera_id}/stream")
async def mjpeg_stream(
    camera_id: str,
    token:     str,
    request:   Request,
    db:        DBSession,
) -> StreamingResponse:
    """
    Push a live MJPEG stream from the camera's RTSP URL.
    Auth is via ?token=<JWT> query parameter (required because <img src>
    cannot send Authorization headers).
    """
    from app.utils.jwt import TokenError, decode_token

    settings = request.app.state.settings
    try:
        decode_token(token, settings)
    except TokenError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    cam = await _get_camera_or_404(db, camera_id)
    if not cam.rtsp_url:
        raise HTTPException(status_code=400, detail="Camera has no RTSP URL configured")

    rtsp_url = cam.rtsp_url

    # Determine the origin to echo back for CORS (canvas drawImage needs it)
    origin = request.headers.get("origin", "*")

    # ── Brain-frame relay ────────────────────────────────────────────────────
    # When a PTZBrain is active for this camera it already holds the RTSP
    # connection.  Always relay from the brain's _last_frame rather than
    # opening a second ffmpeg process — many cameras allow only one RTSP
    # session and would reject the second connection.
    #
    # The brain's _frame_reader_loop keeps _last_frame continuously fresh
    # (updated on every decoded frame), so this is safe even during inference.
    import cv2 as _cv
    _active_brain = None
    _brains = getattr(request.app.state, "ptz_brains", {})
    for _b in _brains.values():
        _cfg = getattr(_b, "_cfg", None)
        if _cfg and str(getattr(_cfg, "camera_id", "")) == camera_id:
            _active_brain = _b
            break

    def _get_brain_raw_frame():
        if _active_brain is None:
            return None
        frame = getattr(_active_brain, "_last_frame", None)
        if frame is not None:
            ok, jpg = _cv.imencode(".jpg", frame, [_cv.IMWRITE_JPEG_QUALITY, 80])
            return jpg.tobytes() if ok else None
        return None

    # Use brain relay whenever a brain is running — no socket probe needed.
    _brain_relay = _active_brain is not None

    if _brain_relay:
        # Serve the raw brain frame as MJPEG — no second RTSP connection needed
        async def _generate_brain():
            import cv2 as _cv, numpy as _np
            blank = _np.full((480, 854, 3), 40, dtype=_np.uint8)
            _cv.putText(blank, "No frame yet", (300, 240),
                        _cv.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2)
            _, _blank_jpg = _cv.imencode(".jpg", blank)
            blank_bytes = _blank_jpg.tobytes()

            last_ka_ts = asyncio.get_event_loop().time()
            last_frame_bytes: bytes = blank_bytes

            while True:
                if await request.is_disconnected():
                    break

                now = asyncio.get_event_loop().time()

                # Keepalive boundary every 25 s — prevents Nginx/Cloudflare from
                # closing an idle connection when the brain hasn't produced a new
                # frame (e.g. during PTZ moves, decoder reconnection).
                if now - last_ka_ts > 25:
                    yield b"--frame\r\nContent-Type: text/plain\r\n\r\nkeepalive\r\n"
                    last_ka_ts = now

                new_bytes = _get_brain_raw_frame()
                if new_bytes is not None:
                    last_frame_bytes = new_bytes
                    last_ka_ts = now  # real frame counts as keepalive

                frame_bytes = last_frame_bytes
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(frame_bytes)).encode() + b"\r\n\r\n"
                    + frame_bytes + b"\r\n"
                )
                await asyncio.sleep(0.067)  # ~15 fps

        return StreamingResponse(
            _generate_brain(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={
                "X-Accel-Buffering":            "no",
                "Cache-Control":                "no-cache, no-store, must-revalidate",
                "Pragma":                       "no-cache",
                "Connection":                   "keep-alive",
                "Access-Control-Allow-Origin":  origin,
                "Access-Control-Allow-Methods": "GET, OPTIONS",
                "Access-Control-Allow-Headers": "Authorization, Content-Type",
                "Vary":                         "Origin",
            },
        )

    async def _generate():
        cmd = [
            "ffmpeg",
            "-loglevel",        "error",
            # ── Low-latency input flags ───────────────────────────────────────
            "-fflags",          "nobuffer",       # disable input buffering
            "-probesize",       "4096",           # minimal stream probe
            "-analyzeduration", "0",              # skip stream analysis delay
            "-rtsp_transport",  "tcp",
            "-i",               rtsp_url,
            # ── Output: lightweight MJPEG ─────────────────────────────────────
            "-f",               "mjpeg",
            "-q:v",             "7",
            "-vf",              "fps=15,scale=960:540",
            "-flush_packets",   "1",              # flush every frame immediately
            "pipe:1",
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        buf           = b""
        last_frame_ts = asyncio.get_event_loop().time()

        try:
            while True:
                if await request.is_disconnected():
                    break

                # keepalive boundary every 25 s so Cloudflare / Nginx proxies
                # don't consider the connection idle and close it (CF timeout = 100s)
                now = asyncio.get_event_loop().time()
                if now - last_frame_ts > 25:
                    yield b"--frame\r\nContent-Type: text/plain\r\n\r\nkeepalive\r\n"
                    last_frame_ts = now

                try:
                    chunk = await asyncio.wait_for(proc.stdout.read(8192), timeout=2.0)
                except asyncio.TimeoutError:
                    continue
                if not chunk:
                    break

                buf += chunk
                # Extract complete JPEG frames (SOI = FFD8 … EOI = FFD9)
                while True:
                    start = buf.find(b"\xff\xd8")
                    if start == -1:
                        buf = b""
                        break
                    end = buf.find(b"\xff\xd9", start + 2)
                    if end == -1:
                        buf = buf[start:]
                        break
                    end  += 2
                    frame = buf[start:end]
                    buf   = buf[end:]
                    last_frame_ts = asyncio.get_event_loop().time()
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n"
                        + frame + b"\r\n"
                    )
        finally:
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass

    return StreamingResponse(
        _generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            # Prevent any proxy (Nginx, Cloudflare Workers, CDN) from buffering the stream
            "X-Accel-Buffering":          "no",
            "Cache-Control":              "no-cache, no-store, must-revalidate",
            "Pragma":                     "no-cache",
            "Connection":                 "keep-alive",
            # CORS — allows browser canvas drawImage() from any origin
            "Access-Control-Allow-Origin":  origin,
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "Authorization, Content-Type",
            "Vary":                         "Origin",
        },
    )


# ── Annotated debug MJPEG stream ──────────────────────────────────────────────

@router.get("/{camera_id}/annotated-stream")
async def annotated_stream(
    camera_id: str,
    token:     str,
    request:   Request,
) -> StreamingResponse:
    """
    MJPEG stream of the latest annotated frame from the active PTZBrain.
    Each frame has:
      - Green person bounding boxes
      - Face boxes (green=recognised, orange=unrecognised, red=spoof)
      - Recognised person ID / "Unrecognized" label
      - Top bar: brain state + preset name + PTZ action
      - Bottom bar: live P/T/Z values

    Falls back to a blank grey frame when monitoring is not active.
    Auth via ?token=<JWT> (same as /stream).
    """
    from app.utils.jwt import TokenError, decode_token
    import cv2 as _cv2, numpy as _np

    settings = request.app.state.settings
    try:
        decode_token(token, settings)
    except TokenError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    origin = request.headers.get("origin", "*")

    async def _generate():
        # Build a fallback blank frame (sent when no brain is active)
        blank = _np.full((480, 854, 3), 40, dtype=_np.uint8)
        _cv2.putText(blank, "Monitoring not active", (200, 240),
                     _cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2)
        _, blank_jpg = _cv2.imencode(".jpg", blank)
        blank_bytes = blank_jpg.tobytes()

        def _get_brain_frame() -> bytes:
            brains = getattr(request.app.state, "ptz_brains", {})
            for brain in brains.values():
                if hasattr(brain, "_annotated_frame_jpg") and brain._annotated_frame_jpg:
                    cfg = getattr(brain, "_cfg", None)
                    # Compare as strings — cfg.camera_id may be a UUID object
                    if cfg and str(getattr(cfg, "camera_id", "")) == camera_id:
                        return brain._annotated_frame_jpg
            # No matching brain found for this specific camera — return blank
            return blank_bytes

        last_frame_id = None
        last_ka_ts    = asyncio.get_event_loop().time()
        while True:
            if await request.is_disconnected():
                break
            frame_bytes = _get_brain_frame()
            # Only push when the brain has produced a new frame (avoids resending
            # the same JPEG repeatedly when inference is slower than poll rate)
            frame_id = id(frame_bytes)
            now = asyncio.get_event_loop().time()
            if frame_id != last_frame_id:
                last_frame_id = frame_id
                last_ka_ts    = now
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(frame_bytes)).encode() + b"\r\n\r\n"
                    + frame_bytes + b"\r\n"
                )
            elif now - last_ka_ts > 20:
                # Keepalive boundary — prevents Nginx/Cloudflare from closing
                # an idle connection while the brain is between recognition states.
                yield b"--frame\r\nContent-Type: text/plain\r\n\r\nkeepalive\r\n"
                last_ka_ts = now
            await asyncio.sleep(0.033)   # ~30 fps poll

    return StreamingResponse(
        _generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "X-Accel-Buffering":            "no",
            "Cache-Control":                "no-cache, no-store, must-revalidate",
            "Pragma":                       "no-cache",
            "Connection":                   "keep-alive",
            "Access-Control-Allow-Origin":  origin,
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "Authorization, Content-Type",
            "Vary":                         "Origin",
        },
    )


# ── AI mode (pause / resume autonomous PTZ brain) ─────────────────────────────

class AiModeRequest(BaseModel):
    mode: str   # "AI" | "MANUAL"


@router.post("/{camera_id}/ai-mode", dependencies=[_UP])
async def set_ai_mode(
    camera_id: str,
    body:      AiModeRequest,
    request:   Request,
    db:        DBSession,
) -> dict:
    """Pause or resume the AI-driven PTZ brain for this camera."""
    if body.mode not in ("AI", "MANUAL"):
        raise HTTPException(status_code=400, detail="mode must be 'AI' or 'MANUAL'")

    cam = await _get_camera_or_404(db, camera_id)

    # Persist mode in learned_params so it survives restarts
    params = dict(cam.learned_params or {})
    params["ai_mode"] = body.mode
    cam.learned_params = params
    cam.updated_at     = now_epoch()
    await db.flush()

    # Best-effort: signal PTZBrain via Redis if available
    try:
        redis = request.app.state.redis
        await redis.set(f"camera:{camera_id}:ai_mode", body.mode, ex=86400)
    except Exception:
        pass

    audit(request, f"AI_MODE_{body.mode}", "camera", camera_id)
    return {"camera_id": camera_id, "mode": body.mode}


@router.get("/{camera_id}/ai-mode", dependencies=[_RD])
async def get_ai_mode(
    camera_id: str,
    request:   Request,
    db:        DBSession,
) -> dict:
    """Return the current AI/MANUAL mode for this camera."""
    cam = await _get_camera_or_404(db, camera_id)

    # Check Redis first (most up-to-date), fall back to DB
    mode = None
    try:
        redis = request.app.state.redis
        mode  = await redis.get(f"camera:{camera_id}:ai_mode")
        if isinstance(mode, bytes):
            mode = mode.decode()
    except Exception:
        pass

    if not mode:
        mode = (cam.learned_params or {}).get("ai_mode", "AI")

    return {"camera_id": camera_id, "mode": mode}


# ── Move Speed ────────────────────────────────────────────────────────────────

class MoveSpeedBody(BaseModel):
    move_speed: float  # 0.1 – 1.0 normalised ONVIF speed
    settle_s:   float | None = None  # optional settle delay override


@router.put("/{camera_id}/move-speed", dependencies=[_RD])
async def set_move_speed(
    camera_id: str,
    body:      MoveSpeedBody,
    request:   Request,
    db:        DBSession,
) -> dict:
    """Persist camera move-speed (and optional settle delay) in learned_params."""
    if not (0.05 <= body.move_speed <= 1.0):
        raise HTTPException(status_code=400, detail="move_speed must be between 0.05 and 1.0")

    cam = await _get_camera_or_404(db, camera_id)
    params = dict(cam.learned_params or {})
    speeds = dict(params.get("camera_speeds", {}))
    speeds["move_speed"] = round(body.move_speed, 3)
    if body.settle_s is not None:
        if not (0.0 <= body.settle_s <= 5.0):
            raise HTTPException(status_code=400, detail="settle_s must be between 0 and 5")
        speeds["settle_s"] = round(body.settle_s, 2)
    params["camera_speeds"] = speeds
    cam.learned_params = params
    cam.updated_at     = now_epoch()
    await db.flush()

    # Signal running PTZBrain immediately if possible
    try:
        redis = request.app.state.redis
        import json as _json
        await redis.set(f"camera:{camera_id}:move_speed", _json.dumps(speeds), ex=86400)
    except Exception:
        pass

    audit(request, "SET_MOVE_SPEED", "camera", camera_id)
    return {"camera_id": camera_id, **speeds}


# ── Scan Presets ──────────────────────────────────────────────────────────────

class PresetCreate(BaseModel):
    name:    str
    pan:     float
    tilt:    float
    zoom:    float = 0.0
    dwell_s: float = 3.0


class PresetUpdate(BaseModel):
    name:    str | None = None
    pan:     float | None = None
    tilt:    float | None = None
    zoom:    float | None = None
    dwell_s: float | None = None
    order_idx: int | None = None


def _preset_dict(p: CameraScanPreset) -> dict:
    return {
        "preset_id": str(p.preset_id),
        "camera_id": str(p.camera_id),
        "name":      p.name,
        "order_idx": p.order_idx,
        "pan":       p.pan,
        "tilt":      p.tilt,
        "zoom":      p.zoom,
        "dwell_s":   p.dwell_s,
        "created_at": p.created_at,
    }


@router.get("/{camera_id}/presets", dependencies=[_RD])
async def list_presets(camera_id: str, request: Request, db: DBSession) -> list:
    """List all scan presets for a camera, ordered by order_idx."""
    await _get_camera_or_404(db, camera_id)
    result = await db.execute(
        select(CameraScanPreset)
        .where(CameraScanPreset.camera_id == camera_id)
        .order_by(CameraScanPreset.order_idx)
    )
    return [_preset_dict(p) for p in result.scalars().all()]


@router.post("/{camera_id}/presets", status_code=201, dependencies=[_UP])
async def create_preset(
    camera_id: str, body: PresetCreate, request: Request, db: DBSession,
) -> dict:
    """Save current PTZ position as a new named scan preset."""
    cam = await _get_camera_or_404(db, camera_id)
    client_id = str(cam.client_id)

    # Assign next order_idx
    max_idx = await db.scalar(
        select(func.max(CameraScanPreset.order_idx))
        .where(CameraScanPreset.camera_id == camera_id)
    ) or -1

    preset = CameraScanPreset(
        camera_id=camera_id,
        client_id=client_id,
        name=body.name,
        order_idx=max_idx + 1,
        pan=body.pan,
        tilt=body.tilt,
        zoom=body.zoom,
        dwell_s=body.dwell_s,
        created_at=now_epoch(),
    )
    db.add(preset)
    await db.flush()
    await db.refresh(preset)
    audit(request, "CREATE_PRESET", "camera_scan_preset", str(preset.preset_id))
    return _preset_dict(preset)


@router.put("/{camera_id}/presets/{preset_id}", dependencies=[_UP])
async def update_preset(
    camera_id: str, preset_id: str, body: PresetUpdate, request: Request, db: DBSession,
) -> dict:
    """Update a preset's name, position, dwell time, or order."""
    result = await db.execute(
        select(CameraScanPreset).where(
            CameraScanPreset.preset_id == preset_id,
            CameraScanPreset.camera_id == camera_id,
        )
    )
    preset = result.scalar_one_or_none()
    if preset is None:
        raise HTTPException(status_code=404, detail="Preset not found")

    if body.name      is not None: preset.name      = body.name
    if body.pan       is not None: preset.pan        = body.pan
    if body.tilt      is not None: preset.tilt       = body.tilt
    if body.zoom      is not None: preset.zoom       = body.zoom
    if body.dwell_s   is not None: preset.dwell_s    = body.dwell_s
    if body.order_idx is not None: preset.order_idx  = body.order_idx

    await db.flush()
    audit(request, "UPDATE_PRESET", "camera_scan_preset", preset_id)
    return _preset_dict(preset)


@router.delete("/{camera_id}/presets/{preset_id}", status_code=204, dependencies=[_DEL])
async def delete_preset(
    camera_id: str, preset_id: str, request: Request, db: DBSession,
) -> None:
    """Delete a scan preset."""
    result = await db.execute(
        select(CameraScanPreset).where(
            CameraScanPreset.preset_id == preset_id,
            CameraScanPreset.camera_id == camera_id,
        )
    )
    preset = result.scalar_one_or_none()
    if preset is None:
        raise HTTPException(status_code=404, detail="Preset not found")
    await db.delete(preset)
    audit(request, "DELETE_PRESET", "camera_scan_preset", preset_id)


@router.post("/{camera_id}/presets/reorder", dependencies=[_UP])
async def reorder_presets(
    camera_id: str, body: list[str], request: Request, db: DBSession,
) -> list:
    """Reorder presets. Body is an ordered list of preset_ids."""
    await _get_camera_or_404(db, camera_id)
    for idx, pid in enumerate(body):
        await db.execute(
            text("UPDATE camera_scan_presets SET order_idx = :idx WHERE preset_id = :pid AND camera_id = :cid"),
            {"idx": idx, "pid": pid, "cid": camera_id},
        )
    result = await db.execute(
        select(CameraScanPreset)
        .where(CameraScanPreset.camera_id == camera_id)
        .order_by(CameraScanPreset.order_idx)
    )
    return [_preset_dict(p) for p in result.scalars().all()]


@router.post("/{camera_id}/presets/{preset_id}/goto", dependencies=[_UP])
async def goto_preset(
    camera_id: str, preset_id: str, request: Request, db: DBSession,
) -> dict:
    """Move the camera to a preset position immediately."""
    result = await db.execute(
        select(CameraScanPreset).where(
            CameraScanPreset.preset_id == preset_id,
            CameraScanPreset.camera_id == camera_id,
        )
    )
    preset = result.scalar_one_or_none()
    if preset is None:
        raise HTTPException(status_code=404, detail="Preset not found")

    cam = await _get_camera_or_404(db, camera_id)
    settings = getattr(request.app.state, "settings", None)
    try:
        ctrl = await _build_onvif(cam, settings, request)
        await ctrl.absolute_move(preset.pan, preset.tilt, preset.zoom, speed=0.5)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"PTZ move failed: {exc}") from exc

    return {"ok": True, "preset_id": preset_id, "pan": preset.pan, "tilt": preset.tilt, "zoom": preset.zoom}


# ── Protocol test ─────────────────────────────────────────────────────────────

_PTZ_CAMERA_TYPES = {CameraType.PTZ, CameraType.PTZ_ZOOM}
_VALID_PROTOCOLS  = {"ONVIF", "ISAPI", "CGI_DAHUA", "CGI", "CGI_CPPLUS"}


class TestProtocolRequest(BaseModel):
    protocol: str   # "ONVIF" | "ISAPI" | "CGI_DAHUA"


@router.post("/{camera_id}/test-protocol", dependencies=[_UP])
async def test_camera_protocol(
    camera_id: str, body: TestProtocolRequest, request: Request, db: DBSession,
) -> dict:
    """
    Probe whether a PTZ camera supports the requested control protocol.
    Only valid for PTZ / PTZ_ZOOM camera types.
    Returns {ok, protocol, latency_ms, detail}.
    """
    import time

    cam = await _get_camera_or_404(db, camera_id)

    if cam.camera_type not in _PTZ_CAMERA_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Protocol test is only available for PTZ / PTZ_ZOOM cameras (this camera is {cam.camera_type})",
        )

    protocol = body.protocol.upper()
    if protocol not in _VALID_PROTOCOLS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown protocol '{body.protocol}'. Valid values: ONVIF, ISAPI, CGI_DAHUA",
        )

    if not cam.onvif_host:
        raise HTTPException(status_code=400, detail="Camera has no host address configured")

    settings = getattr(request.app.state, "settings", None)
    lp: dict = cam.learned_params or {}

    from app.services.isapi_controller import make_camera_controller
    from app.services.onvif_controller import CameraOfflineError

    # Merge column-level FOV into learned_params
    if cam.fov_h and "fov_h_wide" not in lp:
        lp["fov_h_wide"] = cam.fov_h
    if cam.fov_v and "fov_v_wide" not in lp:
        lp["fov_v_wide"] = cam.fov_v

    ctrl = make_camera_controller(
        host=cam.onvif_host,
        port=cam.onvif_port or 80,
        username=cam.onvif_username or "",
        password=_onvif_password(cam, settings),
        protocol=protocol,
        learned_params=lp,
    )

    t0 = time.monotonic()
    try:
        await ctrl.connect()
        status = await ctrl.get_ptz_status()
        latency_ms = round((time.monotonic() - t0) * 1000, 1)
        return {
            "ok":         True,
            "protocol":   protocol,
            "latency_ms": latency_ms,
            "detail":     f"Connected via {protocol}. PTZ position: pan={status.pan:.3f} tilt={status.tilt:.3f} zoom={status.zoom:.3f}",
        }
    except CameraOfflineError as exc:
        latency_ms = round((time.monotonic() - t0) * 1000, 1)
        err_str = str(exc)
        # If protocol endpoint returned 404 the camera IS reachable but doesn't support
        # this protocol.  Run brand-detection so the user gets an actionable suggestion.
        if "404" in err_str or "not supported" in err_str.lower():
            from app.services.isapi_controller import probe_camera_protocol
            probe = await probe_camera_protocol(
                host=cam.onvif_host,
                port=cam.onvif_port or 80,
                username=cam.onvif_username or "",
                password=_onvif_password(cam, settings),
            )
            suggestion = f"Recommended protocol: {probe['recommended']}"
            brand_info = f"Brand: {probe['brand'] or 'Unknown'}"
            if probe["model"]:
                brand_info += f" ({probe['model']})"
            detail = f"{err_str} | {brand_info} | {suggestion} | {probe['detail']}"
        else:
            detail = err_str
        return {
            "ok":         False,
            "protocol":   protocol,
            "latency_ms": latency_ms,
            "detail":     detail,
        }
    except Exception as exc:
        latency_ms = round((time.monotonic() - t0) * 1000, 1)
        return {
            "ok":         False,
            "protocol":   protocol,
            "latency_ms": latency_ms,
            "detail":     f"{type(exc).__name__}: {exc}",
        }


@router.post("/{camera_id}/detect-protocol", dependencies=[_UP])
async def detect_camera_protocol(
    camera_id: str, request: Request, db: DBSession,
) -> dict:
    """
    Auto-detect camera brand and best-supported PTZ protocol.
    Returns full probe report including brand, model, isapi_ok, cgi_ok, recommended.
    """
    cam = await _get_camera_or_404(db, camera_id)

    if cam.camera_type not in _PTZ_CAMERA_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Protocol detection only available for PTZ / PTZ_ZOOM cameras (this camera is {cam.camera_type})",
        )
    if not cam.onvif_host:
        raise HTTPException(status_code=400, detail="Camera has no host address configured")

    settings = getattr(request.app.state, "settings", None)
    from app.services.isapi_controller import probe_camera_protocol

    result = await probe_camera_protocol(
        host=cam.onvif_host,
        port=cam.onvif_port or 80,
        username=cam.onvif_username or "",
        password=_onvif_password(cam, settings),
    )
    return result
