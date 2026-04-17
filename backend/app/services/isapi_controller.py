"""
ISAPI / CGI direct-HTTP PTZ controllers — zero-SOAP, low-latency camera control.

Protocol support
────────────────
  ISAPIController  — Hikvision ISAPI over HTTP/HTTPS
    PUT /ISAPI/PTZCtrl/channels/{ch}/continuous   → velocity move
    PUT /ISAPI/PTZCtrl/channels/{ch}/AbsoluteEx   → absolute position
    GET /ISAPI/PTZCtrl/channels/{ch}/status       → position query

  CGIController    — Dahua / Amcrest / generic CGI
    GET /cgi-bin/ptz.cgi?action=start&code=…     → directional velocity
    GET /cgi-bin/ptz.cgi?action=start&code=PositionABS → absolute position
    GET /cgi-bin/ptz.cgi?action=getStatus         → position query

Both classes expose the identical interface as ONVIFController so PTZBrain
can use them as drop-in replacements.  Protocol is chosen via the camera's
learned_params["ptz_protocol"] field ("ISAPI" | "CGI_DAHUA").

Coordinate convention (same as ONVIF throughout)
─────────────────────────────────────────────────
  pan   ∈ [-1.0, +1.0]   −1 = full-left,   +1 = full-right
  tilt  ∈ [-1.0, +1.0]   −1 = full-down,   +1 = full-up
  zoom  ∈ [ 0.0,  1.0]    0 = widest,        1 = telephoto end
  speed ∈ [ 0.0,  1.0]   fractional max-speed
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

import httpx

from app.services.onvif_controller import (
    CameraOfflineError,
    PTZKalmanFilter,
    PTZLimits,
    PTZPosition,
    PTZTimeoutError,
    _K_at_zoom,
    _optical_flow_shift,
)

logger = logging.getLogger(__name__)

# ── Shared helpers ────────────────────────────────────────────────────────────

_HTTP_TIMEOUT = 2.0        # seconds — tight timeout; direct HTTP should be fast
_CONNECT_TIMEOUT = 5.0


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _vel_to_pct(v: float) -> int:
    """Normalised velocity [-1, 1] → integer percentage [-100, 100]."""
    return int(_clamp(v, -1.0, 1.0) * 100)


def _speed_to_pct(s: float) -> int:
    """Normalised speed [0, 1] → integer percentage [0, 100]."""
    return int(_clamp(s, 0.0, 1.0) * 100)


# ── ISAPIController (Hikvision) ───────────────────────────────────────────────

class ISAPIController:
    """
    Direct Hikvision ISAPI PTZ controller.

    Replaces ONVIFController with native HTTP commands:
      • continuous_move  → PUT /ISAPI/PTZCtrl/channels/{ch}/continuous
      • absolute_move    → PUT /ISAPI/PTZCtrl/channels/{ch}/AbsoluteEx
      • get_ptz_status   → GET /ISAPI/PTZCtrl/channels/{ch}/status

    All values use ONVIF-normalised coordinates internally so PTZBrain needs
    no modification.

    ISAPI ↔ normalised mapping
    ─────────────────────────
      azimuth  0–3600  (tenths°, 0°–360°)  ↔  pan  −1 to +1
      elevation −900–900 (tenths°)          ↔  tilt −1 to +1
      absoluteZoom 0–100                    ↔  zoom  0 to  1
    """

    # Default FOV model (degrees) — override via constructor / apply_calibration
    _FOV_H_WIDE:   float = 60.0
    _FOV_H_NARROW: float = 3.0
    _FOV_V_WIDE:   float = 34.0
    _FOV_V_NARROW: float = 1.7

    def __init__(
        self,
        host: str,
        port: int = 80,
        username: str = "admin",
        password: str = "",
        *,
        channel: int = 1,
        https: bool = False,
        fov_h_wide: float | None = None,
        fov_h_narrow: float | None = None,
        fov_v_wide: float | None = None,
        fov_v_narrow: float | None = None,
        pan_scale: float | None = None,
        tilt_scale: float | None = None,
        K_pan_wide: float | None = None,
        K_pan_narrow: float | None = None,
        K_tilt_wide: float | None = None,
        K_tilt_narrow: float | None = None,
    ) -> None:
        scheme = "https" if https else "http"
        self._base = f"{scheme}://{host}:{port}"
        self._ch   = channel
        self._auth = httpx.DigestAuth(username, password)

        self._fov_h_wide   = fov_h_wide   or self._FOV_H_WIDE
        self._fov_h_narrow = fov_h_narrow or self._FOV_H_NARROW
        self._fov_v_wide   = fov_v_wide   or self._FOV_V_WIDE
        self._fov_v_narrow = fov_v_narrow or self._FOV_V_NARROW
        self._pan_scale    = pan_scale    or 1.0
        self._tilt_scale   = tilt_scale   or 1.0
        self._K_pan_wide   = K_pan_wide
        self._K_pan_narrow = K_pan_narrow
        self._K_tilt_wide  = K_tilt_wide
        self._K_tilt_narrow = K_tilt_narrow

        self._limits   = PTZLimits()
        self._kalman   = PTZKalmanFilter()
        self._client: httpx.AsyncClient | None = None
        self._connected = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Open persistent HTTP session and verify reachability."""
        self._client = httpx.AsyncClient(
            auth=self._auth,
            timeout=httpx.Timeout(_HTTP_TIMEOUT, connect=_CONNECT_TIMEOUT),
            verify=False,  # self-signed certs common on cameras
        )
        url = f"{self._base}/ISAPI/PTZCtrl/channels/{self._ch}/status"
        try:
            resp = await self._client.get(url)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            sc = exc.response.status_code
            if sc == 404:
                raise CameraOfflineError(
                    "ISAPI not supported by this camera (404) — try ONVIF or CGI_DAHUA instead"
                ) from exc
            if sc in (401, 403):
                raise CameraOfflineError(
                    f"ISAPI authentication failed ({sc}) — check username / password"
                ) from exc
            raise CameraOfflineError(f"ISAPI probe returned HTTP {sc}") from exc
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.TimeoutException) as exc:
            raise CameraOfflineError(f"Camera unreachable at {self._base}: {exc}") from exc
        except Exception as exc:
            raise CameraOfflineError(f"ISAPI connect error: {exc}") from exc
        self._connected = True
        logger.info("ISAPIController: connected  host=%s  ch=%d", self._base, self._ch)

    async def _put(self, path: str, body: str) -> httpx.Response:
        assert self._client, "not connected"
        url = f"{self._base}{path}"
        try:
            resp = await self._client.put(
                url,
                content=body.encode(),
                headers={"Content-Type": "application/xml"},
            )
            resp.raise_for_status()
            return resp
        except httpx.HTTPStatusError as exc:
            raise CameraOfflineError(f"ISAPI PUT {path} → {exc.response.status_code}") from exc
        except Exception as exc:
            raise CameraOfflineError(f"ISAPI PUT {path} failed: {exc}") from exc

    async def _get(self, path: str, **params) -> httpx.Response:
        assert self._client, "not connected"
        url = f"{self._base}{path}"
        try:
            resp = await self._client.get(url, params=params or None)
            resp.raise_for_status()
            return resp
        except httpx.HTTPStatusError as exc:
            raise CameraOfflineError(f"ISAPI GET {path} → {exc.response.status_code}") from exc
        except Exception as exc:
            raise CameraOfflineError(f"ISAPI GET {path} failed: {exc}") from exc

    # ── Coordinate conversion ─────────────────────────────────────────────────

    @staticmethod
    def _pan_to_azimuth(pan: float) -> int:
        """Normalised pan [-1, 1] → ISAPI azimuth [0, 3600] (tenths of degrees)."""
        return int(_clamp((pan + 1.0) * 1800.0, 0, 3600))

    @staticmethod
    def _azimuth_to_pan(az: float) -> float:
        return _clamp(az / 1800.0 - 1.0, -1.0, 1.0)

    @staticmethod
    def _tilt_to_elevation(tilt: float) -> int:
        """Normalised tilt [-1, 1] → ISAPI elevation [-900, 900] (tenths of degrees)."""
        return int(_clamp(tilt * 900.0, -900, 900))

    @staticmethod
    def _elevation_to_tilt(el: float) -> float:
        return _clamp(el / 900.0, -1.0, 1.0)

    @staticmethod
    def _zoom_to_isapi(zoom: float) -> int:
        """Normalised zoom [0, 1] → ISAPI absoluteZoom [0, 100]."""
        return int(_clamp(zoom * 100.0, 0, 100))

    @staticmethod
    def _isapi_to_zoom(iz: float) -> float:
        return _clamp(iz / 100.0, 0.0, 1.0)

    # ── PTZ movement ──────────────────────────────────────────────────────────

    async def continuous_move(
        self,
        pan_velocity: float,
        tilt_velocity: float,
        zoom_velocity: float = 0.0,
        timeout: float = 0.0,
    ) -> None:
        """
        Continuous velocity move.

        Velocities are normalised: -1.0 = max CCW/down/wide, +1.0 = max CW/up/tele.
        If timeout > 0, the move is stopped automatically after that many seconds.
        """
        pan_pct  = _vel_to_pct(pan_velocity)
        tilt_pct = _vel_to_pct(tilt_velocity)
        zoom_pct = _vel_to_pct(zoom_velocity)

        body = (
            f"<PTZData>"
            f"<pan>{pan_pct}</pan>"
            f"<tilt>{tilt_pct}</tilt>"
            f"<zoom>{zoom_pct}</zoom>"
            f"</PTZData>"
        )
        await self._put(f"/ISAPI/PTZCtrl/channels/{self._ch}/continuous", body)
        logger.debug(
            "ISAPI continuous_move  pan=%d  tilt=%d  zoom=%d",
            pan_pct, tilt_pct, zoom_pct,
        )

        if timeout > 0:
            await asyncio.sleep(timeout)
            await self.stop()

    async def stop(self, pan_tilt: bool = True, zoom: bool = True) -> None:
        """Stop any ongoing continuous move."""
        pan_pct  = 0 if pan_tilt else None
        zoom_pct = 0 if zoom else None
        body = (
            f"<PTZData>"
            f"<pan>{pan_pct if pan_pct is not None else 0}</pan>"
            f"<tilt>0</tilt>"
            f"<zoom>{zoom_pct if zoom_pct is not None else 0}</zoom>"
            f"</PTZData>"
        )
        try:
            await self._put(f"/ISAPI/PTZCtrl/channels/{self._ch}/continuous", body)
        except CameraOfflineError:
            pass  # best-effort stop

    async def absolute_move(
        self,
        pan: float,
        tilt: float,
        zoom: float,
        speed: float = 0.5,
        *,
        zoom_speed: float | None = None,
    ) -> None:
        """
        Move to absolute position (normalised coordinates).

        Uses ISAPI AbsoluteEx endpoint which accepts azimuth/elevation/zoom.
        Speed is provided via a separate continuous pre-move to trigger fast
        slew; AbsoluteEx itself uses the camera's configured default speed.
        """
        pan  = _clamp(pan,  self._limits.pan_min,  self._limits.pan_max)
        tilt = _clamp(tilt, self._limits.tilt_min, self._limits.tilt_max)
        zoom = _clamp(zoom, self._limits.zoom_min,  self._limits.zoom_max)

        az  = self._pan_to_azimuth(pan)
        el  = self._tilt_to_elevation(tilt)
        izm = self._zoom_to_isapi(zoom)

        body = (
            f"<PTZData>"
            f"<AbsoluteHigh>"
            f"<elevation>{el}</elevation>"
            f"<azimuth>{az}</azimuth>"
            f"<absoluteZoom>{izm}</absoluteZoom>"
            f"</AbsoluteHigh>"
            f"</PTZData>"
        )
        await self._put(f"/ISAPI/PTZCtrl/channels/{self._ch}/AbsoluteEx", body)
        self._kalman.notify_commanded(pan, tilt, zoom)
        logger.debug(
            "ISAPI absolute_move  az=%d  el=%d  zoom=%d  (pan=%.3f tilt=%.3f)",
            az, el, izm, pan, tilt,
        )

    async def smooth_absolute_move(
        self,
        pan: float,
        tilt: float,
        zoom: float,
        *,
        nominal_speed: float = 0.5,
        current_pos: PTZPosition | None = None,
        zoom_speed: float | None = None,
    ) -> float:
        """
        Absolute move with speed-tiered approach and settle-time estimate.

        Returns: recommended settle time (seconds).
        """
        pan  = _clamp(pan,  self._limits.pan_min,  self._limits.pan_max)
        tilt = _clamp(tilt, self._limits.tilt_min, self._limits.tilt_max)
        zoom = _clamp(zoom, self._limits.zoom_min,  self._limits.zoom_max)

        if current_pos is None:
            try:
                current_pos = await self.get_ptz_status()
            except Exception:
                current_pos = PTZPosition()

        pt_dist = math.hypot(pan - current_pos.pan, tilt - current_pos.tilt)
        dzoom   = abs(zoom - current_pos.zoom)

        # Speed tiers by pan/tilt Euclidean distance
        if pt_dist < 0.02:
            speed = nominal_speed * 0.25
        elif pt_dist < 0.08:
            speed = nominal_speed * 0.50
        elif pt_dist < 0.20:
            speed = nominal_speed * 0.75
        else:
            speed = nominal_speed

        await self.absolute_move(pan, tilt, zoom, speed=speed, zoom_speed=zoom_speed)

        settle = 0.20 + pt_dist * 0.80 + dzoom * 1.20
        return float(max(0.20, min(settle, 1.0)))

    async def relative_move(
        self,
        pan_delta: float,
        tilt_delta: float,
        zoom_delta: float = 0.0,
        speed: float = 0.5,
    ) -> None:
        """
        Relative move: implemented as get-position → compute-target → absolute_move.
        ISAPI does not natively support a delta move without current position.
        """
        pos = await self.get_ptz_status()
        await self.absolute_move(
            pos.pan  + pan_delta,
            pos.tilt + tilt_delta,
            pos.zoom + zoom_delta,
            speed=speed,
        )

    async def goto_home(self, speed: float = 0.5) -> None:
        """Move to home position (azimuth=1800, elevation=0, zoom=0)."""
        await self.absolute_move(0.0, 0.0, 0.0, speed=speed)

    # ── Status ────────────────────────────────────────────────────────────────

    async def get_raw_ptz_status(self) -> PTZPosition:
        """Read physical PTZ position from ISAPI status endpoint."""
        resp = await self._get(f"/ISAPI/PTZCtrl/channels/{self._ch}/status")
        xml  = resp.text
        try:
            az  = float(_xml_text(xml, "azimuth")      or "1800")
            el  = float(_xml_text(xml, "elevation")    or "0")
            izm = float(_xml_text(xml, "absoluteZoom") or "0")
        except (ValueError, TypeError):
            return PTZPosition()

        return PTZPosition(
            pan  = self._azimuth_to_pan(az),
            tilt = self._elevation_to_tilt(el),
            zoom = self._isapi_to_zoom(izm),
        )

    async def get_ptz_status(self) -> PTZPosition:
        """Return Kalman-filtered PTZ position."""
        raw = await self.get_raw_ptz_status()
        return self._kalman.update(raw.pan, raw.tilt, raw.zoom)

    # ── FOV & calibration ─────────────────────────────────────────────────────

    def get_fov_at_zoom(self, zoom: float) -> tuple[float, float]:
        """Return (fov_h_deg, fov_v_deg) at normalised zoom level."""
        zoom = _clamp(zoom, 0.0, 1.0)
        fov_h = self._fov_h_wide * math.exp(
            zoom * math.log(self._fov_h_narrow / self._fov_h_wide)
        )
        fov_v = self._fov_v_wide * math.exp(
            zoom * math.log(self._fov_v_narrow / self._fov_v_wide)
        )
        return fov_h, fov_v

    @property
    def fov_params(self) -> dict:
        return {
            "fov_h_wide":   self._fov_h_wide,
            "fov_h_narrow": self._fov_h_narrow,
            "fov_v_wide":   self._fov_v_wide,
            "fov_v_narrow": self._fov_v_narrow,
        }

    @property
    def calibration(self) -> dict:
        return {
            "K_pan_wide":    self._K_pan_wide,
            "K_pan_narrow":  self._K_pan_narrow,
            "K_tilt_wide":   self._K_tilt_wide,
            "K_tilt_narrow": self._K_tilt_narrow,
            **self.fov_params,
            "pan_scale":  self._pan_scale,
            "tilt_scale": self._tilt_scale,
        }

    def apply_calibration(self, lp: dict) -> None:
        """Load previously measured calibration values from learned_params dict."""
        if lp.get("K_pan_wide"):
            self._K_pan_wide   = float(lp["K_pan_wide"])
        if lp.get("K_pan_narrow"):
            self._K_pan_narrow = float(lp["K_pan_narrow"])
        if lp.get("K_tilt_wide"):
            self._K_tilt_wide  = float(lp["K_tilt_wide"])
        if lp.get("K_tilt_narrow"):
            self._K_tilt_narrow = float(lp["K_tilt_narrow"])
        if lp.get("fov_h_wide"):
            self._fov_h_wide   = float(lp["fov_h_wide"])
        if lp.get("fov_h_narrow"):
            self._fov_h_narrow = float(lp["fov_h_narrow"])
        if lp.get("fov_v_wide"):
            self._fov_v_wide   = float(lp["fov_v_wide"])
        if lp.get("fov_v_narrow"):
            self._fov_v_narrow = float(lp["fov_v_narrow"])
        if lp.get("pan_scale"):
            self._pan_scale  = float(lp["pan_scale"])
        if lp.get("tilt_scale"):
            self._tilt_scale = float(lp["tilt_scale"])

    # ── Pixel ↔ PTZ coordinate conversion ────────────────────────────────────

    def pixel_to_ptz(
        self,
        px: float,
        py: float,
        frame_w: int,
        frame_h: int,
        current_ptz: PTZPosition,
    ) -> tuple[float, float]:
        """
        Convert a pixel coordinate to the PTZ absolute position that would
        centre the camera on that point.

        Uses measured K constants when available, otherwise falls back to
        the FOV angle model.
        """
        cx = (px - frame_w / 2.0) / (frame_w / 2.0)    # -1 … +1 (left … right)
        cy = -(py - frame_h / 2.0) / (frame_h / 2.0)   # +1 … -1 (top … bottom)

        zoom = current_ptz.zoom

        if (self._K_pan_wide is not None and self._K_pan_narrow is not None and
                self._K_tilt_wide is not None and self._K_tilt_narrow is not None):
            K_pan  = _K_at_zoom(zoom, self._K_pan_wide,  self._K_pan_narrow)
            K_tilt = _K_at_zoom(zoom, self._K_tilt_wide, self._K_tilt_narrow)
            d_pan  = cx * K_pan
            d_tilt = cy * K_tilt
        else:
            fov_h, fov_v = self.get_fov_at_zoom(zoom)
            pan_range  = self._limits.pan_max  - self._limits.pan_min
            tilt_range = self._limits.tilt_max - self._limits.tilt_min
            d_pan  = cx * (fov_h / 360.0) * pan_range  / self._pan_scale
            d_tilt = cy * (fov_v / 180.0) * tilt_range / self._tilt_scale

        target_pan  = _clamp(current_ptz.pan  + d_pan,  self._limits.pan_min,  self._limits.pan_max)
        target_tilt = _clamp(current_ptz.tilt + d_tilt, self._limits.tilt_min, self._limits.tilt_max)
        return target_pan, target_tilt

    def ptz_to_pixel(
        self,
        pan: float,
        tilt: float,
        frame_w: int,
        frame_h: int,
        current_ptz: PTZPosition,
    ) -> tuple[float, float]:
        """Project a PTZ absolute position to pixel coordinates in the current frame."""
        zoom = current_ptz.zoom
        d_pan  = pan  - current_ptz.pan
        d_tilt = tilt - current_ptz.tilt

        if (self._K_pan_wide is not None and self._K_pan_narrow is not None and
                self._K_tilt_wide is not None and self._K_tilt_narrow is not None):
            K_pan  = _K_at_zoom(zoom, self._K_pan_wide,  self._K_pan_narrow)
            K_tilt = _K_at_zoom(zoom, self._K_tilt_wide, self._K_tilt_narrow)
            cx = d_pan  / K_pan  if K_pan  != 0 else 0.0
            cy = d_tilt / K_tilt if K_tilt != 0 else 0.0
        else:
            fov_h, fov_v = self.get_fov_at_zoom(zoom)
            pan_range  = self._limits.pan_max  - self._limits.pan_min
            tilt_range = self._limits.tilt_max - self._limits.tilt_min
            cx = d_pan  / ((fov_h / 360.0) * pan_range  / self._pan_scale)  if pan_range  else 0.0
            cy = d_tilt / ((fov_v / 180.0) * tilt_range / self._tilt_scale) if tilt_range else 0.0

        px = (cx + 1.0) * frame_w  / 2.0
        py = (-cy + 1.0) * frame_h / 2.0
        return px, py

    # ── Travel time estimate ──────────────────────────────────────────────────

    def estimate_travel_time(
        self,
        from_pos: PTZPosition,
        to_pos: PTZPosition,
    ) -> float:
        """Estimate dominant-axis travel time in seconds (same model as ONVIF)."""
        pan_range  = self._limits.pan_max  - self._limits.pan_min  or 2.0
        tilt_range = self._limits.tilt_max - self._limits.tilt_min or 2.0
        zoom_range = self._limits.zoom_max - self._limits.zoom_min or 1.0

        pan_frac  = abs(to_pos.pan  - from_pos.pan)  / pan_range
        tilt_frac = abs(to_pos.tilt - from_pos.tilt) / tilt_range
        zoom_frac = abs(to_pos.zoom - from_pos.zoom)  / zoom_range

        # Assume 2 s full-range pan/tilt at max speed, 3 s for zoom
        pan_t  = pan_frac  * 2.0 / self._limits.pan_speed
        tilt_t = tilt_frac * 2.0 / self._limits.tilt_speed
        zoom_t = zoom_frac * 3.0 / self._limits.zoom_speed
        return max(pan_t, tilt_t, zoom_t) + 0.2   # +0.2 s for command round-trip

    # ── Auto-calibration ──────────────────────────────────────────────────────

    async def auto_calibrate_fov(
        self,
        grab_frame_fn: Callable[[], Coroutine[Any, Any, Any]],
        *,
        delta_pan:  float = 0.03,
        delta_tilt: float = 0.02,
        settle_s:   float = 1.2,
    ) -> dict:
        """
        Measure K constants and FOV by optical flow (same algorithm as ONVIFController).

        Returns dict suitable for apply_calibration() and DB persistence.
        """
        import numpy as np

        origin = await self.get_ptz_status()
        results: dict = {}

        for zoom_target, label in ((0.0, "wide"), (1.0, "narrow")):
            await self.absolute_move(origin.pan, origin.tilt, zoom_target, speed=0.6)
            await asyncio.sleep(settle_s * 1.5)

            frame0 = await grab_frame_fn()
            if frame0 is None:
                logger.warning("auto_calibrate_fov: no frame at zoom=%s", label)
                continue

            H, W = frame0.shape[:2]

            # Pan displacement
            await self.absolute_move(
                origin.pan + delta_pan, origin.tilt, zoom_target, speed=0.4
            )
            await asyncio.sleep(settle_s)
            frame1 = await grab_frame_fn()
            await self.absolute_move(origin.pan, origin.tilt, zoom_target, speed=0.4)
            await asyncio.sleep(settle_s * 0.8)

            K_pan = None
            if frame1 is not None:
                dx, _ = _optical_flow_shift(frame0, frame1)
                if dx is not None and abs(dx) > 2:
                    K_pan = abs(delta_pan) * (W / 2.0) / abs(dx)

            # Tilt displacement
            frame0t = await grab_frame_fn()
            await self.absolute_move(
                origin.pan, origin.tilt + delta_tilt, zoom_target, speed=0.4
            )
            await asyncio.sleep(settle_s)
            frame1t = await grab_frame_fn()
            await self.absolute_move(origin.pan, origin.tilt, zoom_target, speed=0.4)
            await asyncio.sleep(settle_s * 0.8)

            K_tilt = None
            if frame0t is not None and frame1t is not None:
                _, dy = _optical_flow_shift(frame0t, frame1t)
                if dy is not None and abs(dy) > 2:
                    K_tilt = abs(delta_tilt) * (H / 2.0) / abs(dy)

            results[label] = {"K_pan": K_pan, "K_tilt": K_tilt, "W": W, "H": H}
            logger.info(
                "auto_calibrate_fov: zoom=%s  K_pan=%.5f  K_tilt=%.5f",
                label,
                K_pan or 0,
                K_tilt or 0,
            )

        # Restore camera
        await self.absolute_move(origin.pan, origin.tilt, origin.zoom, speed=0.6)

        # Build output dict
        out: dict = {}
        wide   = results.get("wide",   {})
        narrow = results.get("narrow", {})

        if wide.get("K_pan"):
            out["K_pan_wide"] = wide["K_pan"]
        if narrow.get("K_pan"):
            out["K_pan_narrow"] = narrow["K_pan"]
        if wide.get("K_tilt"):
            out["K_tilt_wide"] = wide["K_tilt"]
        if narrow.get("K_tilt"):
            out["K_tilt_narrow"] = narrow["K_tilt"]

        # Derive FOV from K constants (K = fov_fraction per half-frame)
        if out.get("K_pan_wide") and wide.get("W"):
            out["fov_h_wide"] = out["K_pan_wide"] * 360.0
        if out.get("K_pan_narrow") and narrow.get("W"):
            out["fov_h_narrow"] = out["K_pan_narrow"] * 360.0
        if out.get("K_tilt_wide") and wide.get("H"):
            out["fov_v_wide"] = out["K_tilt_wide"] * 180.0
        if out.get("K_tilt_narrow") and narrow.get("H"):
            out["fov_v_narrow"] = out["K_tilt_narrow"] * 180.0

        if out:
            self.apply_calibration(out)
        return out

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def limits(self) -> PTZLimits:
        return self._limits

    @property
    def is_connected(self) -> bool:
        return self._connected


# ── CGIController (Dahua / Amcrest / generic HTTP CGI) ────────────────────────

class CGIController:
    """
    Generic CGI PTZ controller for Dahua / Amcrest / clone cameras.

    All movement uses GET requests to /cgi-bin/ptz.cgi.
    Absolute position uses code=PositionABS.
    Status uses action=getStatus.

    CGI ↔ normalised mapping
    ────────────────────────
      pan  0.0–360.0°  ↔  pan  −1 to +1  (centre = 180°)
      tilt −90.0–90.0° ↔  tilt −1 to +1
      zoom 0–100%      ↔  zoom  0 to  1
    """

    _FOV_H_WIDE:   float = 60.0
    _FOV_H_NARROW: float = 5.0
    _FOV_V_WIDE:   float = 34.0
    _FOV_V_NARROW: float = 3.0

    # Dahua max speed value for arg1/arg2 in CGI commands
    _MAX_SPEED: int = 8

    def __init__(
        self,
        host: str,
        port: int = 80,
        username: str = "admin",
        password: str = "",
        *,
        channel: int = 1,
        https: bool = False,
        fov_h_wide: float | None = None,
        fov_h_narrow: float | None = None,
        fov_v_wide: float | None = None,
        fov_v_narrow: float | None = None,
        pan_scale: float | None = None,
        tilt_scale: float | None = None,
        K_pan_wide: float | None = None,
        K_pan_narrow: float | None = None,
        K_tilt_wide: float | None = None,
        K_tilt_narrow: float | None = None,
    ) -> None:
        scheme = "https" if https else "http"
        self._base = f"{scheme}://{host}:{port}"
        self._ch   = channel
        self._auth = httpx.BasicAuth(username, password)

        self._fov_h_wide   = fov_h_wide   or self._FOV_H_WIDE
        self._fov_h_narrow = fov_h_narrow or self._FOV_H_NARROW
        self._fov_v_wide   = fov_v_wide   or self._FOV_V_WIDE
        self._fov_v_narrow = fov_v_narrow or self._FOV_V_NARROW
        self._pan_scale    = pan_scale    or 1.0
        self._tilt_scale   = tilt_scale   or 1.0
        self._K_pan_wide   = K_pan_wide
        self._K_pan_narrow = K_pan_narrow
        self._K_tilt_wide  = K_tilt_wide
        self._K_tilt_narrow = K_tilt_narrow

        self._limits   = PTZLimits()
        self._kalman   = PTZKalmanFilter()
        self._client: httpx.AsyncClient | None = None
        self._connected = False
        # Track the last movement code so stop can reference it
        self._last_code: str = "Stop"

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Open persistent HTTP session and verify camera reachability."""
        self._client = httpx.AsyncClient(
            auth=self._auth,
            timeout=httpx.Timeout(_HTTP_TIMEOUT, connect=_CONNECT_TIMEOUT),
            verify=False,
        )
        url = f"{self._base}/cgi-bin/ptz.cgi"
        try:
            resp = await self._client.get(
                url, params={"action": "getStatus", "channel": self._ch}
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            sc = exc.response.status_code
            if sc == 404:
                raise CameraOfflineError(
                    "CGI not supported by this camera (404) — try ISAPI for Hikvision or ONVIF for others"
                ) from exc
            if sc in (401, 403):
                raise CameraOfflineError(
                    f"CGI authentication failed ({sc}) — check username / password"
                ) from exc
            raise CameraOfflineError(f"CGI probe returned HTTP {sc}") from exc
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.TimeoutException) as exc:
            raise CameraOfflineError(f"Camera unreachable at {self._base}: {exc}") from exc
        except Exception as exc:
            raise CameraOfflineError(f"CGI connect error: {exc}") from exc
        self._connected = True
        logger.info("CGIController: connected  host=%s  ch=%d", self._base, self._ch)

    async def _cgi(self, **params) -> httpx.Response:
        assert self._client, "not connected"
        params.setdefault("channel", self._ch)
        url = f"{self._base}/cgi-bin/ptz.cgi"
        try:
            resp = await self._client.get(url, params=params)
            resp.raise_for_status()
            return resp
        except httpx.HTTPStatusError as exc:
            raise CameraOfflineError(
                f"CGI {params} → {exc.response.status_code}"
            ) from exc
        except Exception as exc:
            raise CameraOfflineError(f"CGI request failed: {exc}") from exc

    # ── Coordinate conversion ─────────────────────────────────────────────────

    @staticmethod
    def _pan_to_deg(pan: float) -> float:
        """Normalised pan [-1, 1] → Dahua pan degrees [0, 360] (centre=180°)."""
        return _clamp((pan + 1.0) * 180.0, 0.0, 360.0)

    @staticmethod
    def _deg_to_pan(deg: float) -> float:
        return _clamp(deg / 180.0 - 1.0, -1.0, 1.0)

    @staticmethod
    def _tilt_to_deg(tilt: float) -> float:
        """Normalised tilt [-1, 1] → Dahua tilt degrees [-90, 90]."""
        return _clamp(tilt * 90.0, -90.0, 90.0)

    @staticmethod
    def _deg_to_tilt(deg: float) -> float:
        return _clamp(deg / 90.0, -1.0, 1.0)

    @staticmethod
    def _zoom_to_pct(zoom: float) -> float:
        return _clamp(zoom * 100.0, 0.0, 100.0)

    @staticmethod
    def _pct_to_zoom(pct: float) -> float:
        return _clamp(pct / 100.0, 0.0, 1.0)

    def _vel_to_speed(self, v: float) -> int:
        """Normalised velocity magnitude [0, 1] → Dahua speed [0, _MAX_SPEED]."""
        return max(1, int(abs(v) * self._MAX_SPEED))

    @staticmethod
    def _vel_to_code(pan_vel: float, tilt_vel: float) -> str:
        """Map (pan_vel, tilt_vel) signs to Dahua directional command code."""
        pan_pos  = pan_vel  >  0.01
        pan_neg  = pan_vel  < -0.01
        tilt_pos = tilt_vel >  0.01
        tilt_neg = tilt_vel < -0.01

        if pan_pos and tilt_pos:
            return "RightUp"
        if pan_pos and tilt_neg:
            return "RightDown"
        if pan_neg and tilt_pos:
            return "LeftUp"
        if pan_neg and tilt_neg:
            return "LeftDown"
        if pan_pos:
            return "Right"
        if pan_neg:
            return "Left"
        if tilt_pos:
            return "Up"
        if tilt_neg:
            return "Down"
        return "Stop"

    # ── PTZ movement ──────────────────────────────────────────────────────────

    async def continuous_move(
        self,
        pan_velocity: float,
        tilt_velocity: float,
        zoom_velocity: float = 0.0,
        timeout: float = 0.0,
    ) -> None:
        """
        Continuous directional move.

        CGI cameras don't accept velocity vectors directly; we map the signed
        velocities to a direction command (Left/Right/Up/LeftDown etc.) and
        a speed level.
        """
        code = self._vel_to_code(pan_velocity, tilt_velocity)
        pt_speed = self._vel_to_speed(max(abs(pan_velocity), abs(tilt_velocity)))

        if code != "Stop":
            self._last_code = code
            await self._cgi(action="start", code=code, arg1=pt_speed, arg2=pt_speed, arg3=0)
            logger.debug("CGI continuous  code=%s  speed=%d", code, pt_speed)

        # Handle zoom separately
        if abs(zoom_velocity) > 0.01:
            zoom_code   = "ZoomTele" if zoom_velocity > 0 else "ZoomWide"
            zoom_speed  = self._vel_to_speed(zoom_velocity)
            await self._cgi(action="start", code=zoom_code, arg1=zoom_speed, arg2=zoom_speed, arg3=0)

        if timeout > 0:
            await asyncio.sleep(timeout)
            await self.stop()

    async def stop(self, pan_tilt: bool = True, zoom: bool = True) -> None:
        """Stop all movement."""
        try:
            if pan_tilt:
                await self._cgi(action="stop", code=self._last_code, arg1=0, arg2=0, arg3=0)
            if zoom:
                for zc in ("ZoomTele", "ZoomWide"):
                    await self._cgi(action="stop", code=zc, arg1=0, arg2=0, arg3=0)
        except CameraOfflineError:
            pass

    async def absolute_move(
        self,
        pan: float,
        tilt: float,
        zoom: float,
        speed: float = 0.5,
        *,
        zoom_speed: float | None = None,
    ) -> None:
        """
        Absolute position move via CGI PositionABS command.

        arg1 = pan degrees (0–360), arg2 = tilt degrees (−90–90),
        arg3 = zoom percent (0–100).
        """
        pan  = _clamp(pan,  self._limits.pan_min,  self._limits.pan_max)
        tilt = _clamp(tilt, self._limits.tilt_min, self._limits.tilt_max)
        zoom = _clamp(zoom, self._limits.zoom_min,  self._limits.zoom_max)

        pan_deg  = self._pan_to_deg(pan)
        tilt_deg = self._tilt_to_deg(tilt)
        zoom_pct = self._zoom_to_pct(zoom)

        await self._cgi(
            action="start", code="PositionABS",
            arg1=f"{pan_deg:.2f}",
            arg2=f"{tilt_deg:.2f}",
            arg3=f"{zoom_pct:.0f}",
        )
        self._kalman.notify_commanded(pan, tilt, zoom)
        logger.debug(
            "CGI absolute_move  pan=%.2f°  tilt=%.2f°  zoom=%.0f%%",
            pan_deg, tilt_deg, zoom_pct,
        )

    async def smooth_absolute_move(
        self,
        pan: float,
        tilt: float,
        zoom: float,
        *,
        nominal_speed: float = 0.5,
        current_pos: PTZPosition | None = None,
        zoom_speed: float | None = None,
    ) -> float:
        """Absolute move with speed-tiered approach and settle-time estimate."""
        pan  = _clamp(pan,  self._limits.pan_min,  self._limits.pan_max)
        tilt = _clamp(tilt, self._limits.tilt_min, self._limits.tilt_max)
        zoom = _clamp(zoom, self._limits.zoom_min,  self._limits.zoom_max)

        if current_pos is None:
            try:
                current_pos = await self.get_ptz_status()
            except Exception:
                current_pos = PTZPosition()

        pt_dist = math.hypot(pan - current_pos.pan, tilt - current_pos.tilt)
        dzoom   = abs(zoom - current_pos.zoom)

        if pt_dist < 0.02:
            speed = nominal_speed * 0.25
        elif pt_dist < 0.08:
            speed = nominal_speed * 0.50
        elif pt_dist < 0.20:
            speed = nominal_speed * 0.75
        else:
            speed = nominal_speed

        await self.absolute_move(pan, tilt, zoom, speed=speed, zoom_speed=zoom_speed)
        settle = 0.20 + pt_dist * 0.80 + dzoom * 1.20
        return float(max(0.20, min(settle, 1.0)))

    async def relative_move(
        self,
        pan_delta: float,
        tilt_delta: float,
        zoom_delta: float = 0.0,
        speed: float = 0.5,
    ) -> None:
        """Relative move via current position + delta → absolute."""
        pos = await self.get_ptz_status()
        await self.absolute_move(
            pos.pan  + pan_delta,
            pos.tilt + tilt_delta,
            pos.zoom + zoom_delta,
            speed=speed,
        )

    async def goto_home(self, speed: float = 0.5) -> None:
        await self.absolute_move(0.0, 0.0, 0.0, speed=speed)

    # ── Status ────────────────────────────────────────────────────────────────

    async def get_raw_ptz_status(self) -> PTZPosition:
        """
        Query camera position via getStatus CGI action.

        Dahua response format (INI-style):
          pan=180.000000\r\ntilt=0.000000\r\nzoom=1.000000
        """
        resp = await self._cgi(action="getStatus")
        text = resp.text
        vals: dict[str, float] = {}
        for line in text.replace("&", "\n").splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                try:
                    vals[k.strip().lower()] = float(v.strip())
                except ValueError:
                    pass

        pan_deg  = vals.get("pan",  180.0)
        tilt_deg = vals.get("tilt",   0.0)
        zoom_pct = vals.get("zoom",   0.0)

        # Dahua sometimes returns zoom as factor (1.0=wide) rather than percent
        if zoom_pct <= 1.0 and zoom_pct >= 0.0:
            # likely already normalised (some firmware versions)
            zoom_norm = zoom_pct
        else:
            zoom_norm = self._pct_to_zoom(zoom_pct)

        return PTZPosition(
            pan  = self._deg_to_pan(pan_deg),
            tilt = self._deg_to_tilt(tilt_deg),
            zoom = zoom_norm,
        )

    async def get_ptz_status(self) -> PTZPosition:
        raw = await self.get_raw_ptz_status()
        return self._kalman.update(raw.pan, raw.tilt, raw.zoom)

    # ── FOV & calibration (identical logic to ISAPIController) ────────────────

    def get_fov_at_zoom(self, zoom: float) -> tuple[float, float]:
        zoom = _clamp(zoom, 0.0, 1.0)
        fov_h = self._fov_h_wide * math.exp(
            zoom * math.log(self._fov_h_narrow / self._fov_h_wide)
        )
        fov_v = self._fov_v_wide * math.exp(
            zoom * math.log(self._fov_v_narrow / self._fov_v_wide)
        )
        return fov_h, fov_v

    @property
    def fov_params(self) -> dict:
        return {
            "fov_h_wide":   self._fov_h_wide,
            "fov_h_narrow": self._fov_h_narrow,
            "fov_v_wide":   self._fov_v_wide,
            "fov_v_narrow": self._fov_v_narrow,
        }

    @property
    def calibration(self) -> dict:
        return {
            "K_pan_wide":   self._K_pan_wide,
            "K_pan_narrow": self._K_pan_narrow,
            "K_tilt_wide":  self._K_tilt_wide,
            "K_tilt_narrow": self._K_tilt_narrow,
            **self.fov_params,
            "pan_scale":  self._pan_scale,
            "tilt_scale": self._tilt_scale,
        }

    def apply_calibration(self, lp: dict) -> None:
        if lp.get("K_pan_wide"):
            self._K_pan_wide   = float(lp["K_pan_wide"])
        if lp.get("K_pan_narrow"):
            self._K_pan_narrow = float(lp["K_pan_narrow"])
        if lp.get("K_tilt_wide"):
            self._K_tilt_wide  = float(lp["K_tilt_wide"])
        if lp.get("K_tilt_narrow"):
            self._K_tilt_narrow = float(lp["K_tilt_narrow"])
        if lp.get("fov_h_wide"):
            self._fov_h_wide   = float(lp["fov_h_wide"])
        if lp.get("fov_h_narrow"):
            self._fov_h_narrow = float(lp["fov_h_narrow"])
        if lp.get("fov_v_wide"):
            self._fov_v_wide   = float(lp["fov_v_wide"])
        if lp.get("fov_v_narrow"):
            self._fov_v_narrow = float(lp["fov_v_narrow"])
        if lp.get("pan_scale"):
            self._pan_scale  = float(lp["pan_scale"])
        if lp.get("tilt_scale"):
            self._tilt_scale = float(lp["tilt_scale"])

    def pixel_to_ptz(
        self,
        px: float,
        py: float,
        frame_w: int,
        frame_h: int,
        current_ptz: PTZPosition,
    ) -> tuple[float, float]:
        cx = (px - frame_w / 2.0) / (frame_w / 2.0)
        cy = -(py - frame_h / 2.0) / (frame_h / 2.0)
        zoom = current_ptz.zoom

        if (self._K_pan_wide is not None and self._K_pan_narrow is not None and
                self._K_tilt_wide is not None and self._K_tilt_narrow is not None):
            K_pan  = _K_at_zoom(zoom, self._K_pan_wide,  self._K_pan_narrow)
            K_tilt = _K_at_zoom(zoom, self._K_tilt_wide, self._K_tilt_narrow)
            d_pan  = cx * K_pan
            d_tilt = cy * K_tilt
        else:
            fov_h, fov_v = self.get_fov_at_zoom(zoom)
            pan_range  = self._limits.pan_max  - self._limits.pan_min
            tilt_range = self._limits.tilt_max - self._limits.tilt_min
            d_pan  = cx * (fov_h / 360.0) * pan_range  / self._pan_scale
            d_tilt = cy * (fov_v / 180.0) * tilt_range / self._tilt_scale

        return (
            _clamp(current_ptz.pan  + d_pan,  self._limits.pan_min,  self._limits.pan_max),
            _clamp(current_ptz.tilt + d_tilt, self._limits.tilt_min, self._limits.tilt_max),
        )

    def ptz_to_pixel(
        self,
        pan: float,
        tilt: float,
        frame_w: int,
        frame_h: int,
        current_ptz: PTZPosition,
    ) -> tuple[float, float]:
        zoom  = current_ptz.zoom
        d_pan  = pan  - current_ptz.pan
        d_tilt = tilt - current_ptz.tilt

        if (self._K_pan_wide is not None and self._K_pan_narrow is not None and
                self._K_tilt_wide is not None and self._K_tilt_narrow is not None):
            K_pan  = _K_at_zoom(zoom, self._K_pan_wide,  self._K_pan_narrow)
            K_tilt = _K_at_zoom(zoom, self._K_tilt_wide, self._K_tilt_narrow)
            cx = d_pan  / K_pan  if K_pan  else 0.0
            cy = d_tilt / K_tilt if K_tilt else 0.0
        else:
            fov_h, fov_v = self.get_fov_at_zoom(zoom)
            pan_range  = self._limits.pan_max  - self._limits.pan_min
            tilt_range = self._limits.tilt_max - self._limits.tilt_min
            cx = d_pan  / ((fov_h / 360.0) * pan_range  / self._pan_scale)  if pan_range  else 0.0
            cy = d_tilt / ((fov_v / 180.0) * tilt_range / self._tilt_scale) if tilt_range else 0.0

        return (cx + 1.0) * frame_w / 2.0, (-cy + 1.0) * frame_h / 2.0

    def estimate_travel_time(self, from_pos: PTZPosition, to_pos: PTZPosition) -> float:
        pan_range  = self._limits.pan_max  - self._limits.pan_min  or 2.0
        tilt_range = self._limits.tilt_max - self._limits.tilt_min or 2.0
        zoom_range = self._limits.zoom_max - self._limits.zoom_min or 1.0
        pan_t  = abs(to_pos.pan  - from_pos.pan)  / pan_range  * 2.0 / self._limits.pan_speed
        tilt_t = abs(to_pos.tilt - from_pos.tilt) / tilt_range * 2.0 / self._limits.tilt_speed
        zoom_t = abs(to_pos.zoom - from_pos.zoom)  / zoom_range * 3.0 / self._limits.zoom_speed
        return max(pan_t, tilt_t, zoom_t) + 0.2

    async def auto_calibrate_fov(
        self,
        grab_frame_fn: Callable[[], Coroutine[Any, Any, Any]],
        *,
        delta_pan:  float = 0.03,
        delta_tilt: float = 0.02,
        settle_s:   float = 1.2,
    ) -> dict:
        """Same optical-flow calibration as ISAPIController."""
        origin = await self.get_ptz_status()
        results: dict = {}

        for zoom_target, label in ((0.0, "wide"), (1.0, "narrow")):
            await self.absolute_move(origin.pan, origin.tilt, zoom_target, speed=0.6)
            await asyncio.sleep(settle_s * 1.5)
            frame0 = await grab_frame_fn()
            if frame0 is None:
                continue
            H, W = frame0.shape[:2]

            await self.absolute_move(origin.pan + delta_pan, origin.tilt, zoom_target, speed=0.4)
            await asyncio.sleep(settle_s)
            frame1 = await grab_frame_fn()
            await self.absolute_move(origin.pan, origin.tilt, zoom_target, speed=0.4)
            await asyncio.sleep(settle_s * 0.8)

            K_pan = None
            if frame1 is not None:
                dx, _ = _optical_flow_shift(frame0, frame1)
                if dx is not None and abs(dx) > 2:
                    K_pan = abs(delta_pan) * (W / 2.0) / abs(dx)

            frame0t = await grab_frame_fn()
            await self.absolute_move(origin.pan, origin.tilt + delta_tilt, zoom_target, speed=0.4)
            await asyncio.sleep(settle_s)
            frame1t = await grab_frame_fn()
            await self.absolute_move(origin.pan, origin.tilt, zoom_target, speed=0.4)
            await asyncio.sleep(settle_s * 0.8)

            K_tilt = None
            if frame0t is not None and frame1t is not None:
                _, dy = _optical_flow_shift(frame0t, frame1t)
                if dy is not None and abs(dy) > 2:
                    K_tilt = abs(delta_tilt) * (H / 2.0) / abs(dy)

            results[label] = {"K_pan": K_pan, "K_tilt": K_tilt, "W": W, "H": H}

        await self.absolute_move(origin.pan, origin.tilt, origin.zoom, speed=0.6)

        out: dict = {}
        wide, narrow = results.get("wide", {}), results.get("narrow", {})
        if wide.get("K_pan"):    out["K_pan_wide"]    = wide["K_pan"]
        if narrow.get("K_pan"):  out["K_pan_narrow"]  = narrow["K_pan"]
        if wide.get("K_tilt"):   out["K_tilt_wide"]   = wide["K_tilt"]
        if narrow.get("K_tilt"): out["K_tilt_narrow"] = narrow["K_tilt"]
        if out.get("K_pan_wide"):    out["fov_h_wide"]   = out["K_pan_wide"]   * 360.0
        if out.get("K_pan_narrow"):  out["fov_h_narrow"] = out["K_pan_narrow"] * 360.0
        if out.get("K_tilt_wide"):   out["fov_v_wide"]   = out["K_tilt_wide"]  * 180.0
        if out.get("K_tilt_narrow"): out["fov_v_narrow"] = out["K_tilt_narrow"]* 180.0
        if out:
            self.apply_calibration(out)
        return out

    @property
    def limits(self) -> PTZLimits:
        return self._limits

    @property
    def is_connected(self) -> bool:
        return self._connected


# ── Factory function ──────────────────────────────────────────────────────────

def make_camera_controller(
    host: str,
    port: int,
    username: str,
    password: str,
    protocol: str,
    *,
    learned_params: dict | None = None,
    channel: int = 1,
    https: bool = False,
):
    """
    Return the appropriate controller for a given protocol string.

    protocol values:
      "ONVIF"     → ONVIFController  (raw SOAP, no zeep dependency)
      "ISAPI"     → ISAPIController  (Hikvision)
      "CGI_DAHUA" → CGIController    (Dahua / Amcrest / clones)

    learned_params is forwarded to apply_calibration() if provided.
    """
    lp = learned_params or {}
    common = dict(
        fov_h_wide   = lp.get("fov_h_wide")    or None,
        fov_h_narrow = lp.get("fov_h_narrow")  or None,
        fov_v_wide   = lp.get("fov_v_wide")    or None,
        fov_v_narrow = lp.get("fov_v_narrow")  or None,
        pan_scale    = lp.get("pan_scale")      or None,
        tilt_scale   = lp.get("tilt_scale")     or None,
        K_pan_wide   = lp.get("K_pan_wide")     or None,
        K_pan_narrow = lp.get("K_pan_narrow")   or None,
        K_tilt_wide  = lp.get("K_tilt_wide")    or None,
        K_tilt_narrow= lp.get("K_tilt_narrow")  or None,
    )

    if protocol == "ISAPI":
        return ISAPIController(
            host, port, username, password,
            channel=channel, https=https, **common,
        )
    if protocol in ("CGI_DAHUA", "CGI", "CGI_CPPLUS"):
        return CGIController(
            host, port, username, password,
            channel=channel, https=https, **common,
        )

    # Default: ONVIF
    from app.services.onvif_controller import ONVIFController
    return ONVIFController(
        host=host, port=port, username=username, password=password, **common
    )


# ── Brand auto-detection probe ───────────────────────────────────────────────

async def probe_camera_protocol(
    host: str,
    port: int,
    username: str,
    password: str,
    *,
    channel: int = 1,
    https: bool = False,
) -> dict:
    """
    Probe a camera and return a brand-detection + protocol-support report.

    Returns a dict:
        {
            "reachable":   bool,
            "brand":       str | None,   # "Hikvision" | "Dahua" | "Unknown" | None
            "model":       str | None,
            "isapi_ok":    bool,
            "cgi_ok":      bool,
            "recommended": str,          # "ISAPI" | "CGI_DAHUA" | "ONVIF"
            "detail":      str,
        }
    """
    import time
    scheme = "https" if https else "http"
    base   = f"{scheme}://{host}:{port}"

    result: dict = {
        "reachable":   False,
        "brand":       None,
        "model":       None,
        "isapi_ok":    False,
        "cgi_ok":      False,
        "recommended": "ONVIF",
        "detail":      "",
        "steps":       [],   # diagnostic: list of {url, status, note}
    }
    notes: list[str] = []
    steps: list[dict] = result["steps"]   # type: ignore[assignment]

    def _step(url: str, status: int | str, note: str = "") -> None:
        steps.append({"url": url, "status": status, "note": note})
        logger.debug("probe  %s  →  %s  %s", url, status, note)

    digest_auth = httpx.DigestAuth(username, password)
    basic_auth  = httpx.BasicAuth(username, password)

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(5.0, connect=3.0),
        verify=False,
    ) as client:

        # ── 1. Check basic TCP reachability via HTTP root ─────────────────────
        root_body = ""
        try:
            r = await client.get(f"{base}/", follow_redirects=True)
            result["reachable"] = True
            root_body = r.text
            _step(f"{base}/", r.status_code, r.headers.get("server", ""))
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.TimeoutException) as exc:
            result["detail"] = f"Camera at {base} is unreachable (connection refused / timeout): {exc}"
            return result
        except Exception as exc:
            result["reachable"] = True
            _step(f"{base}/", "err", str(exc))

        # ── 2. Hikvision ISAPI device info ────────────────────────────────────
        try:
            r = await client.get(
                f"{base}/ISAPI/System/deviceInfo",
                auth=digest_auth,
            )
            if r.status_code == 200:
                result["brand"] = "Hikvision"
                result["model"] = _xml_text(r.text, "model") or _xml_text(r.text, "deviceName") or "Unknown model"
                notes.append(f"Hikvision device confirmed: {result['model']}")
                # Try PTZ channels: 1, 101 (NVR ch1), 2
                for ch in (channel, 101, 2):
                    try:
                        t0 = time.monotonic()
                        rp = await client.get(
                            f"{base}/ISAPI/PTZCtrl/channels/{ch}/status",
                            auth=digest_auth,
                        )
                        if rp.status_code == 200:
                            result["isapi_ok"]    = True
                            result["recommended"] = "ISAPI"
                            result["isapi_channel"] = ch
                            notes.append(f"ISAPI PTZ channel {ch} OK ({round((time.monotonic()-t0)*1000)} ms)")
                            break
                    except Exception:
                        pass
                if not result["isapi_ok"]:
                    notes.append("Hikvision confirmed but no ISAPI PTZ channel responded — camera may be fixed/bullet or PTZ not enabled")
            elif r.status_code in (401, 403):
                notes.append("ISAPI device info: authentication failed — check credentials")
        except Exception:
            pass  # not Hikvision or network error caught above

        # ── 3. Dahua / Amcrest / CPPlus CGI device info ───────────────────────
        # CPPlus cameras are Dahua OEM — they share the magicBox.cgi device info
        # endpoint but their PTZ CGI path varies by model/firmware generation.
        if result["brand"] is None:
            try:
                r = await client.get(
                    f"{base}/cgi-bin/magicBox.cgi",
                    params={"action": "getDeviceType"},
                    auth=basic_auth,
                )
                if r.status_code == 200 and ("DeviceType" in r.text or "OK" in r.text):
                    model_str = r.text.strip().split("=", 1)[-1].strip() if "=" in r.text else r.text.strip()
                    # CPPlus branding shows up in the model string or page title
                    if any(k in model_str.upper() for k in ("CP-", "CPPLUS", "C-PLUS", "CP PLUS")):
                        result["brand"] = "CPPlus"
                    else:
                        result["brand"] = "Dahua"
                    result["model"] = model_str or "Unknown"
                    notes.append(f"{result['brand']} device confirmed: {result['model']}")
                elif r.status_code in (401, 403):
                    notes.append("Dahua/CPPlus CGI: authentication failed — check credentials")
            except Exception:
                pass

        # ── 4. CPPlus-specific PTZ probe (if brand identified as CPPlus/Dahua) ─
        if result["brand"] in ("Dahua", "CPPlus") and not result["cgi_ok"]:
            # Try multiple PTZ CGI paths — CPPlus cameras vary by firmware generation
            ptz_candidates = [
                ("/cgi-bin/ptz.cgi",             {"action": "getStatus", "channel": channel}),
                ("/cgi-bin/ptz.cgi",             {"action": "getStatus", "channel": 0}),
                ("/cgi-bin/ptz.cgi",             {"action": "getStatus", "channel": 1, "cam": 1}),
                ("/cgi-bin/ptz.cgi",             {"action": "getStatus"}),
                ("/cgi-bin/nobody/ptz.cgi",      {"action": "getStatus", "channel": channel}),
                ("/cgi-bin/operator/ptz.cgi",    {"action": "getStatus", "channel": channel}),
            ]
            for ptz_path, ptz_params in ptz_candidates:
                try:
                    t0 = time.monotonic()
                    rp = await client.get(
                        f"{base}{ptz_path}", params=ptz_params, auth=basic_auth,
                    )
                    if rp.status_code == 200:
                        result["cgi_ok"]      = True
                        result["recommended"] = "CGI_CPPLUS" if result["brand"] == "CPPlus" else "CGI_DAHUA"
                        result["cgi_path"]    = ptz_path
                        notes.append(
                            f"{result['brand']} CGI PTZ OK at {ptz_path} "
                            f"({round((time.monotonic()-t0)*1000)} ms)"
                        )
                        break
                except Exception:
                    pass
            if not result["cgi_ok"]:
                notes.append(
                    f"{result['brand']} CGI PTZ not found on any known path — "
                    "camera may need ONVIF, or PTZ may not be enabled in camera settings"
                )

        # ── 5. CPPlus brand detection via HTTP page title / server header ──────
        # If magicBox didn't work, try the root page for CPPlus branding markers
        if result["brand"] is None:
            try:
                r = await client.get(f"{base}/", follow_redirects=True)
                body = r.text.lower()
                srv  = r.headers.get("server", "").lower()
                if "cpplus" in body or "c-plus" in body or "cp-plus" in body or "cp plus" in body or "arresto" in body:
                    result["brand"] = "CPPlus"
                    notes.append("CPPlus branding detected in HTTP response")
                elif "dahua" in body or "dhop" in body or "dahua" in srv:
                    result["brand"] = "Dahua"
                    notes.append("Dahua branding detected in HTTP response")
                elif "hikvision" in body or "hikvision" in srv or "webs" in srv:
                    result["brand"] = "Hikvision"
                    notes.append("Hikvision branding detected in HTTP response")
            except Exception:
                pass

        # ── 6. Summarise ──────────────────────────────────────────────────────
        if result["brand"] is None:
            notes.append(
                "Brand unknown — camera responds but did not match Hikvision, Dahua, or CPPlus fingerprints. "
                "Use ONVIF (universal protocol)."
            )
            result["recommended"] = "ONVIF"

        if not result["isapi_ok"] and not result["cgi_ok"]:
            result["recommended"] = "ONVIF"

        result["detail"] = " | ".join(notes) if notes else "No findings"
        return result


# ── XML mini-parser (avoids lxml/ElementTree for speed) ──────────────────────

def _xml_text(xml: str, tag: str) -> str | None:
    """Extract text content of the first occurrence of <tag>…</tag>."""
    open_tag  = f"<{tag}>"
    close_tag = f"</{tag}>"
    start = xml.find(open_tag)
    if start == -1:
        return None
    start += len(open_tag)
    end = xml.find(close_tag, start)
    if end == -1:
        return None
    return xml[start:end].strip()
