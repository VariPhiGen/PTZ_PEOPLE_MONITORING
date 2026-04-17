"""
NodeManager — Control Plane integration for a GPU node.

Responsibilities
────────────────
  register()         POST to CP /api/nodes/register on startup.
  heartbeat_loop()   Every 30 s: collect GPU / CPU / RAM / disk metrics,
                     active session IDs, and POST to CP /api/nodes/{id}/heartbeat.
  on_camera_assigned(camera_id, config)
                     Called by the CP (via /api/node/cameras/migrate or on startup)
                     when a camera is assigned to this node.
  on_camera_removed(camera_id)
                     Called when the CP removes a camera from this node.

The heartbeat payload doubles as a lightweight telemetry push — the CP's
cron aggregates per-node health into D1 analytics_aggregate.
"""
from __future__ import annotations

import asyncio
import json
import logging
import platform
import subprocess
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable

if TYPE_CHECKING:
    from app.services.face_sync import FaceSyncService

import httpx

logger = logging.getLogger(__name__)

# ── Camera config (subset of the DB Camera model) ────────────────────────────

@dataclass
class CameraConfig:
    camera_id:                str
    client_id:                str
    rtsp_url:                 str
    onvif_host:               str
    onvif_port:               int
    onvif_username:           str
    onvif_password_encrypted: str
    roi_rect:                 dict[str, Any] | list | None = None   # rect or polygon
    faculty_zone:             dict[str, Any] | list | None = None
    mode:                     str = "MONITORING"   # ATTENDANCE | MONITORING | BOTH
    faculty_id:               str | None = None
    name:                     str = ""
    room_name:                str = ""
    learned_params:           dict[str, Any] | None = None  # carries camera_distance_m, scan_cell_meters, roi_rect_world
    camera_type:              str = "PTZ"


# ── NodeManager ────────────────────────────────────────────────────────────────

class NodeManager:
    """
    Manages this GPU node's registration and heartbeat with the Control Plane.

    All CP calls are optional — if CONTROL_PLANE_URL is not set the node
    operates in standalone mode (no registration, heartbeats still computed
    locally for the health endpoint).
    """

    HEARTBEAT_INTERVAL_S    = 30
    REGISTER_RETRY_DELAY_S  = 60

    def __init__(
        self,
        node_id:           str,
        node_name:         str,
        api_endpoint:      str,
        control_plane_url: str | None     = None,
        node_auth_token:   str | None     = None,
        gpu_model:         str | None     = None,
        location:          str | None     = None,
        max_cameras:       int            = 10,
        connectivity:      str            = "PUBLIC_IP",  # PUBLIC_IP | CLOUDFLARE_TUNNEL | LOCAL
        session_factory:   Any | None     = None,         # sqlalchemy async_sessionmaker for local DB writes
    ) -> None:
        self.node_id           = node_id
        self.node_name         = node_name
        self.api_endpoint      = api_endpoint
        self.control_plane_url = control_plane_url
        self.node_auth_token   = node_auth_token
        self.gpu_model         = gpu_model or _detect_gpu_model()
        self.location          = location or platform.node()
        self.max_cameras       = max_cameras
        self.connectivity      = connectivity
        self._session_factory  = session_factory

        self._registered   = False
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._http: httpx.AsyncClient | None            = None

        # Cameras currently assigned to this node (set by on_camera_assigned)
        self._cameras: dict[str, CameraConfig] = {}

        # Callbacks wired by main.py lifespan after services are ready
        self._on_assigned: Callable[[str, CameraConfig], Awaitable[None]] | None = None
        self._on_removed:  Callable[[str], Awaitable[None]] | None                = None

        # Extra analytics contributed by other services (e.g. pipeline latency)
        self._extra_analytics: dict[str, float] = {}

        # Optional FaceSyncService reference — used to notify on camera events
        self._face_sync: "FaceSyncService | None" = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(15.0))

        # Always register in the local PostgreSQL nodes table first.
        # This is the source-of-truth for the dashboard node list regardless
        # of whether a Cloudflare Control Plane is configured.
        await self._register_local_db()

        if self.control_plane_url:
            asyncio.create_task(
                self._register_with_retry(), name="node_register"
            )
        else:
            logger.info(
                "NodeManager: no CONTROL_PLANE_URL — local-DB-only mode"
            )

        self._heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(), name="node_heartbeat"
        )
        logger.info(
            "NodeManager started  node_id=%s  endpoint=%s  connectivity=%s",
            self.node_id, self.api_endpoint, self.connectivity,
        )

    async def stop(self) -> None:
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except (asyncio.CancelledError, Exception):
                pass

        if self._http:
            await self._http.aclose()
            self._http = None

        logger.info("NodeManager stopped")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def set_callbacks(
        self,
        on_assigned: Callable[[str, CameraConfig], Awaitable[None]],
        on_removed:  Callable[[str], Awaitable[None]],
    ) -> None:
        self._on_assigned = on_assigned
        self._on_removed  = on_removed

    def set_face_sync(self, face_sync: "FaceSyncService") -> None:
        """
        Wire the FaceSyncService so NodeManager can notify it when cameras are
        assigned or removed — keeping the served-clients cache accurate.
        """
        self._face_sync = face_sync

    def set_extra_analytics(self, analytics: dict[str, float]) -> None:
        """
        Allow other services to contribute analytics to the heartbeat payload.
        Called from app.state references in main.py or from background tasks.
        """
        self._extra_analytics.update(analytics)

    # ── Registration ──────────────────────────────────────────────────────────

    async def register(self) -> bool:
        """POST /api/nodes/register. Returns True on success."""
        if not self.control_plane_url or not self._http:
            return False
        try:
            resp = await self._http.post(
                f"{self.control_plane_url}/api/nodes/register",
                headers=self._auth_headers(),
                json={
                    "node_id":      self.node_id,
                    "node_name":    self.node_name,
                    "location":     self.location,
                    "connectivity": self.connectivity,
                    "api_endpoint": self.api_endpoint,
                    "gpu_model":    self.gpu_model,
                    "max_cameras":  self.max_cameras,
                },
            )
            resp.raise_for_status()
            self._registered = True
            logger.info(
                "NodeManager: registered with Control Plane  cp=%s  node=%s",
                self.control_plane_url, self.node_id,
            )
            return True
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "NodeManager: registration HTTP error %d: %s",
                exc.response.status_code, exc,
            )
        except Exception as exc:
            logger.warning("NodeManager: registration failed: %s", exc)
        return False

    async def _register_with_retry(self) -> None:
        while not self._registered:
            if await self.register():
                return
            logger.info(
                "NodeManager: CP registration failed — retrying in %ds",
                self.REGISTER_RETRY_DELAY_S,
            )
            await asyncio.sleep(self.REGISTER_RETRY_DELAY_S)

    # ── Local PostgreSQL node registry ────────────────────────────────────────

    async def _register_local_db(self) -> None:
        """Upsert this node into the local `nodes` table so the dashboard can see it."""
        if not self._session_factory:
            logger.warning("NodeManager: no session_factory — skipping local DB registration")
            return
        try:
            from sqlalchemy import text
            now = int(time.time())
            async with self._session_factory() as session:
                await session.execute(
                    text("""
                        INSERT INTO nodes
                            (node_id, node_name, location, connectivity, api_endpoint,
                             gpu_model, max_cameras, active_cameras, status,
                             last_heartbeat, health_json, registered_at, updated_at)
                        VALUES
                            (:node_id, :node_name, :location, :connectivity, :api_endpoint,
                             :gpu_model, :max_cameras, 0, 'ONLINE',
                             :now, :health, :now, :now)
                        ON CONFLICT (node_id) DO UPDATE SET
                            node_name     = EXCLUDED.node_name,
                            location      = EXCLUDED.location,
                            connectivity  = EXCLUDED.connectivity,
                            api_endpoint  = EXCLUDED.api_endpoint,
                            gpu_model     = COALESCE(EXCLUDED.gpu_model, nodes.gpu_model),
                            max_cameras   = EXCLUDED.max_cameras,
                            status        = 'ONLINE',
                            last_heartbeat = EXCLUDED.last_heartbeat,
                            updated_at    = EXCLUDED.updated_at
                    """),
                    {
                        "node_id":      self.node_id,
                        "node_name":    self.node_name,
                        "location":     self.location,
                        "connectivity": self.connectivity,
                        "api_endpoint": self.api_endpoint,
                        "gpu_model":    self.gpu_model,
                        "max_cameras":  self.max_cameras,
                        "now":          now,
                        "health":       json.dumps({}),
                    },
                )
                await session.commit()
            logger.info(
                "NodeManager: registered in local DB  node=%s  endpoint=%s",
                self.node_id, self.api_endpoint,
            )
        except Exception as exc:
            logger.warning("NodeManager: local DB registration failed: %s", exc)

    async def _update_local_db_heartbeat(self, active_cameras: int, health: dict) -> None:
        """Update heartbeat + health in the local `nodes` table."""
        if not self._session_factory:
            return
        try:
            from sqlalchemy import text
            now = int(time.time())
            async with self._session_factory() as session:
                await session.execute(
                    text("""
                        UPDATE nodes SET
                            active_cameras = :active_cameras,
                            status         = 'ONLINE',
                            last_heartbeat = :now,
                            health_json    = :health,
                            updated_at     = :now
                        WHERE node_id = :node_id
                    """),
                    {
                        "node_id":        self.node_id,
                        "active_cameras": active_cameras,
                        "now":            now,
                        "health":         json.dumps(health),
                    },
                )
                await session.commit()
        except Exception as exc:
            logger.debug("NodeManager: local DB heartbeat update failed: %s", exc)

    # ── Heartbeat loop ────────────────────────────────────────────────────────

    async def _heartbeat_loop(self) -> None:
        while True:
            try:
                await self._send_heartbeat()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("NodeManager: heartbeat error: %s", exc)
            await asyncio.sleep(self.HEARTBEAT_INTERVAL_S)

    async def _send_heartbeat(self) -> None:
        health = await asyncio.to_thread(_collect_health)
        self._last_health = health

        # Always update local DB (works even in standalone mode)
        await self._update_local_db_heartbeat(len(self._cameras), health)

        if not self.control_plane_url or not self._http:
            return

        # Re-register if we lost our slot
        if not self._registered:
            await self.register()

        payload: dict[str, Any] = {
            "active_cameras": len(self._cameras),
            "health":         health,
            "camera_ids":     list(self._cameras.keys()),
            "analytics":      dict(self._extra_analytics),
        }

        try:
            resp = await self._http.post(
                f"{self.control_plane_url}/api/nodes/{self.node_id}/heartbeat",
                headers=self._auth_headers(),
                json=payload,
            )
            resp.raise_for_status()
            logger.debug(
                "NodeManager: heartbeat sent  cameras=%d  ts=%d",
                len(self._cameras), int(time.time()),
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                # CP evicted us — re-register
                logger.warning("NodeManager: node evicted from CP, re-registering")
                self._registered = False
                await self.register()
            else:
                logger.warning("NodeManager: heartbeat HTTP error: %s", exc)
        except Exception as exc:
            logger.warning("NodeManager: heartbeat send failed: %s", exc)

    # ── Camera assignment / removal ───────────────────────────────────────────

    async def on_camera_assigned(self, camera_id: str, config: CameraConfig) -> None:
        """
        Called when a camera is assigned to this node.
        Stores the config, notifies FaceSyncService (BUG 5 FIX: ensures FAISS
        is loaded for the client before the first recognition attempt), then
        fires the registered callback which starts ONVIF + PTZBrain session.
        """
        logger.info(
            "NodeManager: camera assigned  camera_id=%s  client=%s",
            camera_id, config.client_id,
        )
        self._cameras[camera_id] = config

        # Notify face_sync so it can eagerly load FAISS for new clients
        if self._face_sync:
            self._face_sync.notify_camera_assigned(camera_id, config.client_id)

        if self._on_assigned:
            try:
                await self._on_assigned(camera_id, config)
            except Exception as exc:
                logger.error(
                    "NodeManager: on_assigned callback error for %s: %s",
                    camera_id, exc, exc_info=True,
                )

    async def on_camera_removed(self, camera_id: str) -> None:
        """
        Called when the CP removes a camera from this node.
        Fires the registered callback (which stops any running PTZBrain / ONVIF
        session) and notifies FaceSyncService to refresh its served-clients cache.
        """
        config = self._cameras.pop(camera_id, None)
        logger.info(
            "NodeManager: camera removed  camera_id=%s  client=%s",
            camera_id, config.client_id if config else "?",
        )

        if self._face_sync and config:
            self._face_sync.notify_camera_removed(camera_id, config.client_id)

        if self._on_removed:
            try:
                await self._on_removed(camera_id)
            except Exception as exc:
                logger.error(
                    "NodeManager: on_removed callback error for %s: %s",
                    camera_id, exc, exc_info=True,
                )

    async def handle_migration(
        self,
        cameras:   list[dict[str, Any]],
        from_node: str,
        client_id: str,
    ) -> None:
        """
        Called by /api/node/cameras/migrate when the CP reassigns cameras from
        an offline node to this one.

        BUG 4 FIX: The original implementation only logged the migration and
        never called on_camera_assigned, so sessions never resumed on the
        target node.  We now build a CameraConfig from each dict and call
        on_camera_assigned, which starts ONVIF + PTZBrain and notifies
        FaceSyncService to build the client's FAISS index if needed.
        """
        logger.info(
            "NodeManager: migration received  cameras=%d  from_node=%s  client=%s",
            len(cameras), from_node, client_id,
        )

        started = 0
        for cam in cameras:
            camera_id = cam.get("camera_id", "")
            if not camera_id:
                logger.warning("NodeManager: migration entry missing camera_id — skipped")
                continue

            # Skip cameras already running on this node (idempotent)
            if camera_id in self._cameras:
                logger.debug(
                    "NodeManager: migration skipped (already running)  camera=%s", camera_id
                )
                continue

            config = CameraConfig(
                camera_id                 = camera_id,
                client_id                 = cam.get("client_id", client_id),
                rtsp_url                  = cam.get("rtsp_url", ""),
                onvif_host                = cam.get("onvif_host", ""),
                onvif_port                = int(cam.get("onvif_port", 80)),
                onvif_username            = cam.get("onvif_username", ""),
                onvif_password_encrypted  = cam.get("onvif_password_encrypted", ""),
                roi_rect                  = cam.get("roi_rect"),
                faculty_zone              = cam.get("faculty_zone"),
                mode                      = cam.get("mode", "MONITORING"),
                faculty_id                = cam.get("faculty_id"),
                name                      = cam.get("name", ""),
                room_name                 = cam.get("room_name", ""),
            )

            try:
                await self.on_camera_assigned(camera_id, config)
                started += 1
                logger.info(
                    "NodeManager: migration started  camera=%s  room=%s",
                    camera_id, config.room_name,
                )
            except Exception as exc:
                logger.error(
                    "NodeManager: migration failed for camera %s: %s",
                    camera_id, exc, exc_info=True,
                )

        logger.info(
            "NodeManager: migration complete  started=%d/%d  from_node=%s",
            started, len(cameras), from_node,
        )

    # ── Health snapshot (accessible to /api/node/info) ────────────────────────

    def get_last_health(self) -> dict[str, float]:
        return getattr(self, "_last_health", {})

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _auth_headers(self) -> dict[str, str]:
        if self.node_auth_token:
            return {"Authorization": f"Bearer {self.node_auth_token}"}
        return {}

    @property
    def registered(self) -> bool:
        return self._registered

    @property
    def active_camera_count(self) -> int:
        return len(self._cameras)


# ── Health / metrics collectors ───────────────────────────────────────────────

def _collect_health() -> dict[str, float]:
    """
    Collect CPU, RAM, disk, and GPU metrics synchronously.
    Designed to be called via asyncio.to_thread().  Never raises.
    """
    result: dict[str, float] = {}

    # CPU / RAM / disk (psutil)
    try:
        import psutil  # optional dependency; graceful no-op if missing
        result["cpu_percent"]   = psutil.cpu_percent(interval=0.2)
        vm = psutil.virtual_memory()
        result["ram_used_mb"]   = vm.used  / 1_048_576
        result["ram_total_mb"]  = vm.total / 1_048_576
        result["ram_percent"]   = vm.percent
        disk = psutil.disk_usage("/")
        result["disk_used_gb"]  = disk.used  / 1_073_741_824
        result["disk_total_gb"] = disk.total / 1_073_741_824
        result["disk_percent"]  = disk.percent
    except Exception as exc:
        logger.debug("NodeManager: psutil metrics error: %s", exc)

    # GPU (nvidia-smi)
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,"
                "memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            timeout=5,
            text=True,
        )
        for i, line in enumerate(raw.strip().splitlines()):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            pfx = f"gpu{i}_"
            result[pfx + "utilization"]  = float(parts[0])
            result[pfx + "mem_util"]     = float(parts[1])
            result[pfx + "mem_used_mb"]  = float(parts[2])
            result[pfx + "mem_total_mb"] = float(parts[3])
            result[pfx + "temp_c"]       = float(parts[4])
    except FileNotFoundError:
        pass  # no GPU / nvidia-smi not installed
    except Exception as exc:
        logger.debug("NodeManager: nvidia-smi error: %s", exc)

    return result


def _detect_gpu_model() -> str | None:
    """Best-effort GPU model name from nvidia-smi. Returns None if unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            timeout=5,
            text=True,
        )
        model = out.strip().splitlines()[0].strip()
        return model or None
    except Exception:
        return None
