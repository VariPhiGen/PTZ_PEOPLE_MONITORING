"""
ONVIFController — async PTZ interface using libonvif + raw SOAP.

  • libonvif (C extension) handles WS-Discovery only.
  • All camera operations use raw SOAP over HTTP with WS-Security PasswordDigest
    (no WSDL files, no zeep dependency):
    GetDeviceInformation, GetCapabilities, GetProfiles, GetStreamUri,
    GetStatus, AbsoluteMove, RelativeMove, ContinuousMove, Stop,
    GotoPreset, GotoHomePosition, GetConfigurations, GetSnapshotUri, SystemReboot.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import math
import os
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx
import numpy as np
import libonvif as onvif

logger = logging.getLogger(__name__)


# ── WS-Security helpers ────────────────────────────────────────────────────────

def _wsse_header(username: str, password: str) -> str:
    """
    Build a WS-Security UsernameToken header with PasswordDigest.
    digest = base64(SHA1(nonce_bytes + created_utf8 + password_utf8))
    """
    nonce_bytes = os.urandom(16)
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    nonce_b64 = base64.b64encode(nonce_bytes).decode()
    raw = nonce_bytes + created.encode("utf-8") + password.encode("utf-8")
    digest = base64.b64encode(hashlib.sha1(raw).digest()).decode()
    return (
        '<wsse:Security xmlns:wsse="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd"'
        ' xmlns:wsu="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd">'
        "<wsse:UsernameToken>"
        f"<wsse:Username>{username}</wsse:Username>"
        '<wsse:Password Type="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordDigest">'
        f"{digest}</wsse:Password>"
        '<wsse:Nonce EncodingType="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-soap-message-security-1.0#Base64Binary">'
        f"{nonce_b64}</wsse:Nonce>"
        f"<wsu:Created>{created}</wsu:Created>"
        "</wsse:UsernameToken>"
        "</wsse:Security>"
    )


def _soap_envelope(header: str, body: str) -> bytes:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"'
        ' xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl"'
        ' xmlns:tds="http://www.onvif.org/ver10/device/wsdl"'
        ' xmlns:tt="http://www.onvif.org/ver10/schema">'
        f"<s:Header>{header}</s:Header>"
        f"<s:Body>{body}</s:Body>"
        "</s:Envelope>"
    ).encode("utf-8")


async def _soap(
    url: str,
    body: str,
    username: str,
    password: str,
    timeout: float = 8.0,
    *,
    client: httpx.AsyncClient | None = None,
) -> ET.Element:
    """
    POST a SOAP 1.2 request with WS-Security PasswordDigest authentication.
    Returns the parsed <s:Body> element.
    Raises httpx.HTTPStatusError on HTTP errors or RuntimeError on bad response.

    Pass a persistent *client* to reuse the TCP connection (recommended).
    When None a fresh one-shot client is created (fallback / standalone use).
    """
    envelope = _soap_envelope(_wsse_header(username, password), body)
    headers = {"Content-Type": "application/soap+xml; charset=utf-8"}
    if client is not None:
        r = await client.post(url, content=envelope, headers=headers,
                              timeout=timeout)
        r.raise_for_status()
    else:
        async with httpx.AsyncClient(timeout=timeout) as _c:
            r = await _c.post(url, content=envelope, headers=headers)
            r.raise_for_status()
    try:
        root = ET.fromstring(r.text)
    except ET.ParseError as exc:
        raise RuntimeError(f"SOAP parse error: {exc} — {r.text[:200]}") from exc

    # Strip namespace from tag to find Body regardless of prefix
    def _find_body(el: ET.Element) -> ET.Element | None:
        tag = el.tag.split("}")[-1] if "}" in el.tag else el.tag
        if tag == "Body":
            return el
        for child in el:
            found = _find_body(child)
            if found is not None:
                return found
        return None

    body_el = _find_body(root)
    if body_el is None:
        raise RuntimeError(f"No SOAP Body in response: {r.text[:200]}")
    return body_el


def _find(el: ET.Element, local_name: str) -> ET.Element | None:
    """Find first descendant whose local-name (after namespace prefix) matches."""
    for child in el.iter():
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == local_name:
            return child
    return None


def _findall(el: ET.Element, local_name: str) -> list[ET.Element]:
    return [
        child for child in el.iter()
        if (child.tag.split("}")[-1] if "}" in child.tag else child.tag) == local_name
    ]


def _text(el: ET.Element, local_name: str, default: str = "") -> str:
    found = _find(el, local_name)
    return (found.text or default) if found is not None else default


# ── Optical-flow helpers ───────────────────────────────────────────────────────

def _optical_flow_shift(
    frame0: "np.ndarray",
    frame1: "np.ndarray",
    axis: str = "both",
) -> tuple[float | None, float | None]:
    """
    Measure (dx_px, dy_px) — robust feature displacement between two frames.

    Uses Lucas-Kanade sparse optical flow on Shi-Tomasi corners, then
    rejects outliers (moving objects, reflective surfaces, bad tracks)
    with MAD-based inlier filtering and a cross-axis consistency check.

    Parameters
    ----------
    axis : "x" | "y" | "both"
        When set, enforces that orthogonal motion is small (median of the
        other axis < 25% of this axis' magnitude). Rejects measurements
        polluted by independent scene motion (e.g. a person walking while
        we pan the camera).

    Sign convention:
      dx > 0 : features moved RIGHT  (camera panned LEFT  / scene shifted right)
      dy > 0 : features moved DOWN   (camera tilted UP    / scene shifted down)

    Returns (None, None) when too few features survive filtering.
    """
    import cv2 as _cv2
    g0 = _cv2.cvtColor(frame0, _cv2.COLOR_BGR2GRAY) if frame0.ndim == 3 else frame0
    g1 = _cv2.cvtColor(frame1, _cv2.COLOR_BGR2GRAY) if frame1.ndim == 3 else frame1

    # Denser, higher-quality feature set — filters out low-contrast false
    # corners that track unreliably on repetitive textures.
    p0 = _cv2.goodFeaturesToTrack(g0, maxCorners=400, qualityLevel=0.03, minDistance=12)
    if p0 is None or len(p0) < 12:
        return None, None

    # Bi-directional LK flow with sub-pixel refinement criteria for stability.
    p1, status, _ = _cv2.calcOpticalFlowPyrLK(
        g0, g1, p0, None,
        winSize=(21, 21), maxLevel=3,
        criteria=(_cv2.TERM_CRITERIA_EPS | _cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if p1 is None or status is None:
        return None, None
    # Forward-backward consistency: flow p0→p1→p0' must return within 1 px.
    p0b, status_b, _ = _cv2.calcOpticalFlowPyrLK(
        g1, g0, p1, None,
        winSize=(21, 21), maxLevel=3,
        criteria=(_cv2.TERM_CRITERIA_EPS | _cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if p0b is None or status_b is None:
        return None, None

    fb_err = np.linalg.norm((p0b - p0).reshape(-1, 2), axis=1)
    good = (
        status.ravel().astype(bool)
        & status_b.ravel().astype(bool)
        & (fb_err < 1.0)
    )
    if good.sum() < 10:
        return None, None

    flow = (p1[good] - p0[good]).reshape(-1, 2)
    dxs, dys = flow[:, 0], flow[:, 1]

    # MAD-based outlier rejection — drop features whose motion deviates
    # from the median by more than 3 × MAD on either axis. Robustly
    # rejects moving objects while keeping static-scene features.
    med_x, med_y = float(np.median(dxs)), float(np.median(dys))
    mad_x = float(np.median(np.abs(dxs - med_x))) or 0.5
    mad_y = float(np.median(np.abs(dys - med_y))) or 0.5
    inliers = (np.abs(dxs - med_x) < 3.0 * mad_x) & (np.abs(dys - med_y) < 3.0 * mad_y)
    if inliers.sum() < 10:
        return None, None

    dx = float(np.mean(dxs[inliers]))
    dy = float(np.mean(dys[inliers]))

    # Cross-axis consistency — a pure pan move should produce mostly
    # horizontal shift (and vice versa). Some wobble is normal on cheap
    # PTZ mechanics at high zoom, so allow up to 50% orthogonal component
    # before rejecting as scene-motion pollution.
    if axis == "x" and abs(dx) > 1.5 and abs(dy) > 0.50 * abs(dx):
        return None, None
    if axis == "y" and abs(dy) > 1.5 and abs(dx) > 0.50 * abs(dy):
        return None, None

    return dx, dy


def _K_at_zoom(zoom: float, K_wide: float, K_narrow: float) -> float:
    """
    Interpolate K constant log-linearly between wide (zoom=0) and narrow (zoom=1).

    Uses the same log-linear model as PTZ optics:  K(z) = K_wide * r^z
    where r = K_narrow / K_wide.
    """
    if K_wide <= 0 or K_narrow <= 0:
        return K_wide
    zoom = max(0.0, min(1.0, zoom))
    return K_wide * math.exp(zoom * math.log(K_narrow / K_wide))


# ── Exceptions ────────────────────────────────────────────────────────────────

class PTZTimeoutError(RuntimeError):
    pass


class CameraOfflineError(RuntimeError):
    pass


# ── Retry helper ──────────────────────────────────────────────────────────────

async def _retry(coro_fn, *args, attempts: int = 3, delay: float = 0.3, **kwargs):
    """Retry *coro_fn* up to *attempts* times with capped exponential backoff.

    Delays: attempt 1 → delay, attempt 2 → delay*2, capped at 0.5s.
    Kept short because cameras are on LAN — 300ms is already generous.
    """
    last_exc: Exception = RuntimeError("no attempts")
    for attempt in range(attempts):
        try:
            return await coro_fn(*args, **kwargs)
        except (CameraOfflineError, asyncio.TimeoutError) as exc:
            last_exc = exc
            if attempt < attempts - 1:
                await asyncio.sleep(min(delay * (attempt + 1), 0.5))
        except Exception as exc:
            last_exc = exc
            if attempt < attempts - 1:
                await asyncio.sleep(min(delay, 0.5))
    raise last_exc


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class PTZLimits:
    pan_min:    float = -1.0
    pan_max:    float =  1.0
    tilt_min:   float = -1.0
    tilt_max:   float =  1.0
    zoom_min:   float =  0.0
    zoom_max:   float =  1.0
    # Mechanical speeds (normalised 0-1), sourced from PTZ config or defaults
    pan_speed:  float =  0.5
    tilt_speed: float =  0.5
    zoom_speed: float =  0.5


@dataclass
class PTZPosition:
    pan:  float = 0.0
    tilt: float = 0.0
    zoom: float = 0.0


# ── PTZ Kalman filter ─────────────────────────────────────────────────────────

class PTZKalmanFilter:
    """
    Constant-velocity Kalman filter for smoothing noisy PTZ position feedback.

    State vector: [pan, tilt, zoom, pan_vel, tilt_vel, zoom_vel]
    Observation:  [pan, tilt, zoom]              (from GetStatus)

    The filter reduces quantisation noise (many cameras only report 2–3 decimal
    places) and ONVIF polling jitter (~50–200 ms latency variation).

    After a commanded move the filter is reset to the target position with high
    velocity uncertainty so it converges quickly to the first new measurement
    rather than fighting a stale prediction from before the move.
    """

    def __init__(
        self,
        process_noise_pos: float = 1e-4,
        process_noise_vel: float = 1e-3,
        measurement_noise: float = 5e-4,
    ) -> None:
        self._q_pos = process_noise_pos
        self._q_vel = process_noise_vel
        self._r     = measurement_noise
        self._x: np.ndarray | None = None   # state [6]
        self._P: np.ndarray | None = None   # covariance [6×6]
        self._t_last: float = 0.0

    def _transition(self, dt: float) -> np.ndarray:
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        return F

    def _process_noise_matrix(self) -> np.ndarray:
        Q = np.zeros((6, 6))
        for i in range(3):
            Q[i,     i    ] = self._q_pos
            Q[i + 3, i + 3] = self._q_vel
        return Q

    def reset(self, pan: float, tilt: float, zoom: float) -> None:
        self._x = np.array([pan, tilt, zoom, 0.0, 0.0, 0.0])
        self._P = np.diag([1e-3, 1e-3, 1e-3, 1.0, 1.0, 1.0])
        self._t_last = time.monotonic()

    def notify_commanded(self, pan: float, tilt: float, zoom: float) -> None:
        """
        Called immediately after sending an AbsoluteMove command.
        Seeds the filter at the target position with high velocity uncertainty.
        """
        if self._x is None:
            self.reset(pan, tilt, zoom)
            return
        self._x[0] = pan
        self._x[1] = tilt
        self._x[2] = zoom
        # high velocity uncertainty so the filter adapts fast to first measurement
        self._P[3, 3] = 1.0
        self._P[4, 4] = 1.0
        self._P[5, 5] = 1.0
        self._t_last = time.monotonic()

    def update(self, pan: float, tilt: float, zoom: float) -> PTZPosition:
        """
        Feed a raw GetStatus measurement; return the filtered PTZPosition.
        """
        now = time.monotonic()
        z = np.array([pan, tilt, zoom])

        if self._x is None:
            self.reset(pan, tilt, zoom)
            return PTZPosition(pan=pan, tilt=tilt, zoom=zoom)

        dt = max(0.0, now - self._t_last)
        self._t_last = now

        # Predict
        F = self._transition(dt)
        Q = self._process_noise_matrix()
        x_pred = F @ self._x
        P_pred = F @ self._P @ F.T + Q

        # Update (H = [I3 | 0] )
        H = np.zeros((3, 6))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        R = np.eye(3) * self._r
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        self._x = x_pred + K @ (z - H @ x_pred)
        self._P = (np.eye(6) - K @ H) @ P_pred

        return PTZPosition(
            pan=float(self._x[0]),
            tilt=float(self._x[1]),
            zoom=float(self._x[2]),
        )

    def predict(self, dt: float) -> PTZPosition:
        if self._x is None:
            return PTZPosition()
        F = self._transition(dt)
        x = F @ self._x
        return PTZPosition(pan=float(x[0]), tilt=float(x[1]), zoom=float(x[2]))


# ── Main controller ───────────────────────────────────────────────────────────

class ONVIFController:
    """
    Async PTZ controller using raw SOAP + WS-Security PasswordDigest.

    No WSDL, no zeep.  All ONVIF communication is direct HTTP POST with
    hand-crafted SOAP envelopes.  libonvif is used only for WS-Discovery
    (discover_cameras class method).

    Usage:
        ctrl = ONVIFController("192.168.1.64", 80, "admin", "pass")
        await ctrl.connect()
        url  = await ctrl.get_rtsp_url()
        pos  = await ctrl.get_ptz_status()
        await ctrl.absolute_move(0.0, 0.0, 0.5)
    """

    _FOV_H_WIDE:   float = 60.0
    _FOV_H_NARROW: float =  3.0
    _FOV_V_WIDE:   float = 34.0
    _FOV_V_NARROW: float =  1.7

    def __init__(
        self,
        host: str,
        port: int,
        username: str,
        password: str,
        *,
        connect_timeout: float = 10.0,
        fov_h_wide:   float | None = None,
        fov_h_narrow: float | None = None,
        fov_v_wide:   float | None = None,
        fov_v_narrow: float | None = None,
        pan_scale:    float | None = None,
        tilt_scale:   float | None = None,
        K_pan_wide:   float | None = None,
        K_pan_narrow: float | None = None,
        K_tilt_wide:  float | None = None,
        K_tilt_narrow:float | None = None,
    ) -> None:
        self._host     = host
        self._port     = port
        self._username = username
        self._password = password
        self._connect_timeout = connect_timeout

        # Connection sentinel (set to True after successful connect)
        self._data: bool | None = None
        self._profile_token: str = ""
        self._ptz_url: str = ""          # PTZ service URL for raw SOAP
        self._media_url: str = ""        # Media service URL (GetProfiles, GetSnapshotUri, etc.)
        self._device_url: str = f"http://{host}:{port}/onvif/device_service"

        # Persistent HTTP client — reuses the TCP connection to the camera so
        # every SOAP call avoids a fresh TCP/socket handshake (~50-100ms each).
        # keepalive_expiry=30s matches most camera idle-connection timeouts.
        self._http: httpx.AsyncClient = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=2, keepalive_expiry=30),
            timeout=8.0,
        )
        self._limits: PTZLimits = PTZLimits()

        self._fov_h_wide   = fov_h_wide   if fov_h_wide   and fov_h_wide   > 0 else self._FOV_H_WIDE
        self._fov_h_narrow = fov_h_narrow if fov_h_narrow and fov_h_narrow > 0 else self._FOV_H_NARROW
        self._fov_v_wide   = fov_v_wide   if fov_v_wide   and fov_v_wide   > 0 else self._FOV_V_WIDE
        self._fov_v_narrow = fov_v_narrow if fov_v_narrow and fov_v_narrow > 0 else self._FOV_V_NARROW
        self._pan_scale  = pan_scale  if pan_scale  and pan_scale  > 0 else 1.0
        self._tilt_scale = tilt_scale if tilt_scale and tilt_scale > 0 else self._pan_scale

        self._K_pan_wide    = K_pan_wide    if K_pan_wide    and K_pan_wide    > 0 else None
        self._K_pan_narrow  = K_pan_narrow  if K_pan_narrow  and K_pan_narrow  > 0 else None
        self._K_tilt_wide   = K_tilt_wide   if K_tilt_wide   and K_tilt_wide   > 0 else None
        self._K_tilt_narrow = K_tilt_narrow if K_tilt_narrow and K_tilt_narrow > 0 else None

        self._kalman: PTZKalmanFilter = PTZKalmanFilter()

        # Tracks the single pending deferred-stop task so we can cancel it
        # before the next continuous_move fires — prevents stale stops from
        # interrupting the tracking loop mid-movement.
        self._deferred_stop_task: asyncio.Task | None = None

    # ── Connection ─────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Establish ONVIF session via raw SOAP + discover PTZ service URL."""
        await _retry(self._connect_once)

    async def _connect_once(self) -> None:
        """
        Connect using pure SOAP (no libonvif manual_fill).

        Steps:
          1. GetDeviceInformation  — verifies auth + gets camera name
          2. GetCapabilities       — discovers media + PTZ service URLs
          3. GetProfiles           — gets media profile token from media service
          4. GetStreamUri          — gets RTSP stream URI
          5. GetConfigurations     — reads pan/tilt/zoom limits
        """
        # 1. Verify connectivity + get camera name
        dev_body = "<tds:GetDeviceInformation/>"
        try:
            resp = await asyncio.wait_for(
                _soap(self._device_url, dev_body, self._username, self._password,
                      client=self._http),
                timeout=self._connect_timeout,
            )
        except asyncio.TimeoutError as exc:
            raise CameraOfflineError(f"{self._host}: connect timed out") from exc
        except Exception as exc:
            raise CameraOfflineError(f"{self._host}: {exc}") from exc

        cam_name = _text(resp, "Manufacturer", "") + " " + _text(resp, "Model", "")
        cam_name = cam_name.strip() or self._host

        # 2. Discover media service URL via GetCapabilities (All)
        media_url = self._device_url  # fallback
        try:
            cap_body = "<tds:GetCapabilities><tds:Category>All</tds:Category></tds:GetCapabilities>"
            resp = await _soap(self._device_url, cap_body, self._username, self._password,
                               client=self._http)
            # Find XAddr under Media element
            for el in resp.iter():
                tag = el.tag.split("}")[-1] if "}" in el.tag else el.tag
                if tag == "Media":
                    xaddr = _find(el, "XAddr")
                    if xaddr is not None and xaddr.text:
                        media_url = xaddr.text.strip()
                        break
        except Exception as exc:
            logger.debug("%s: GetCapabilities(All) failed (%s)", self._host, exc)

        # 3. Get media profiles → profile token (from media service)
        profile_token = ""
        try:
            prof_body = "<trt:GetProfiles xmlns:trt='http://www.onvif.org/ver10/media/wsdl'/>"
            resp = await _soap(media_url, prof_body, self._username, self._password,
                               client=self._http)
            tokens = [
                el.get("token", "")
                for el in resp.iter()
                if (el.tag.split("}")[-1] if "}" in el.tag else el.tag) == "Profiles"
            ]
            profile_token = tokens[0] if tokens else ""
        except Exception as exc:
            logger.debug("%s: GetProfiles failed (%s)", self._host, exc)

        # 4. GetStreamUri (from media service)
        stream_uri = ""
        if profile_token:
            su_body = (
                "<trt:GetStreamUri xmlns:trt='http://www.onvif.org/ver10/media/wsdl'"
                " xmlns:tt='http://www.onvif.org/ver10/schema'>"
                "<trt:StreamSetup>"
                "<tt:Stream>RTP-Unicast</tt:Stream>"
                "<tt:Transport><tt:Protocol>RTSP</tt:Protocol></tt:Transport>"
                "</trt:StreamSetup>"
                f"<trt:ProfileToken>{profile_token}</trt:ProfileToken>"
                "</trt:GetStreamUri>"
            )
            try:
                resp = await _soap(media_url, su_body, self._username, self._password,
                                   client=self._http)
                uri_el = _find(resp, "Uri")
                if uri_el is not None and uri_el.text:
                    stream_uri = uri_el.text.strip()
            except Exception as exc:
                logger.debug("%s: GetStreamUri failed (%s)", self._host, exc)

        # Mark as connected (True sentinel — is_connected checks `is not None`)
        self._data = True  # type: ignore[assignment]
        self._profile_token = profile_token
        self._media_url = media_url
        self._stream_uri_cache = stream_uri
        logger.info(
            "%s: connected  camera=%s  profile=%s  stream=%s",
            self._host, cam_name, profile_token, stream_uri,
        )

        self._ptz_url = await self._discover_ptz_url()
        await self._load_ptz_limits()

    async def _discover_ptz_url(self) -> str:
        """
        Use GetCapabilities to find the PTZ service URL.
        Falls back to the device service URL if the call fails.
        """
        body = (
            "<tds:GetCapabilities>"
            "<tds:Category>PTZ</tds:Category>"
            "</tds:GetCapabilities>"
        )
        try:
            resp = await _soap(self._device_url, body, self._username, self._password,
                               client=self._http)
            xaddr_el = _find(resp, "XAddr")
            if xaddr_el is not None and xaddr_el.text:
                url = xaddr_el.text.strip()
                logger.debug("%s: PTZ service URL = %s", self._host, url)
                return url
        except Exception as exc:
            logger.debug("%s: GetCapabilities failed (%s); using device URL", self._host, exc)
        return self._device_url

    async def _load_ptz_limits(self) -> None:
        """Read PTZ configuration spaces (pan/tilt/zoom ranges, speeds) from camera."""
        body = "<tptz:GetConfigurations/>"
        try:
            resp = await _soap(self._ptz_url, body, self._username, self._password,
                               client=self._http)
        except Exception as exc:
            logger.warning("%s: could not read PTZ config ranges (%s); using defaults", self._host, exc)
            return

        try:
            xrange = _findall(resp, "XRange")
            yrange = _findall(resp, "YRange")

            # PanTiltLimits has one XRange (pan) + one YRange (tilt)
            # ZoomLimits has one XRange (zoom)
            def _float(el: ET.Element, child: str, default: float) -> float:
                found = _find(el, child)
                try:
                    return float(found.text) if found is not None and found.text else default
                except ValueError:
                    return default

            if len(xrange) >= 2:
                self._limits.pan_min  = _float(xrange[0], "Min", self._limits.pan_min)
                self._limits.pan_max  = _float(xrange[0], "Max", self._limits.pan_max)
                self._limits.zoom_min = _float(xrange[1], "Min", self._limits.zoom_min)
                self._limits.zoom_max = _float(xrange[1], "Max", self._limits.zoom_max)
            elif len(xrange) == 1:
                self._limits.pan_min = _float(xrange[0], "Min", self._limits.pan_min)
                self._limits.pan_max = _float(xrange[0], "Max", self._limits.pan_max)

            if yrange:
                self._limits.tilt_min = _float(yrange[0], "Min", self._limits.tilt_min)
                self._limits.tilt_max = _float(yrange[0], "Max", self._limits.tilt_max)

            # DefaultPTZSpeed if present
            pt_spd = _find(resp, "PanTilt")
            if pt_spd is not None:
                try:
                    self._limits.pan_speed  = float(pt_spd.get("x", self._limits.pan_speed))
                    self._limits.tilt_speed = float(pt_spd.get("y", self._limits.tilt_speed))
                except (ValueError, TypeError):
                    pass
            z_spd = _find(resp, "Zoom")
            if z_spd is not None:
                try:
                    self._limits.zoom_speed = float(z_spd.get("x", self._limits.zoom_speed))
                except (ValueError, TypeError):
                    pass

        except Exception as exc:
            logger.warning("%s: error parsing PTZ config (%s); using defaults", self._host, exc)

    def _require_connected(self) -> None:
        if self._data is None:
            raise CameraOfflineError("Not connected. Call connect() first.")

    # ── Discovery ─────────────────────────────────────────────────────────────

    @staticmethod
    async def discover_cameras(timeout: float = 5.0) -> list[dict[str, Any]]:
        """
        Use libonvif WS-Discovery to find ONVIF cameras on the local network.
        Returns list of dicts: {host, xaddrs, camera_name}.
        """
        results: list[dict[str, Any]] = []
        found_event = asyncio.Event()
        loop = asyncio.get_event_loop()

        def _on_data(d: onvif.Data) -> None:
            try:
                entry = {
                    "host":        d.host(),
                    "xaddrs":      d.xaddrs() if callable(d.xaddrs) else "",
                    "camera_name": d.camera_name(),
                }
                results.append(entry)
            except Exception:
                pass

        session = onvif.Session()
        session.getData = _on_data

        def _discover() -> None:
            session.startDiscover()
            time.sleep(timeout)
            session.abort = True

        try:
            await asyncio.to_thread(_discover)
        except Exception as exc:
            logger.warning("WS-Discovery failed: %s", exc)

        return results

    # ── Stream URL ────────────────────────────────────────────────────────────

    async def get_rtsp_url(self) -> str:
        """Return the primary RTSP stream URI (cached from connect)."""
        self._require_connected()
        uri = getattr(self, "_stream_uri_cache", "")
        if not uri:
            raise CameraOfflineError(f"{self._host}: no stream URI")
        return uri

    # ── PTZ status ────────────────────────────────────────────────────────────

    async def get_ptz_status(self) -> PTZPosition:
        """
        Return the Kalman-filtered PTZ position (smoothed GetStatus reading).
        Use get_raw_ptz_status() for the unfiltered hardware value.
        """
        raw = await self.get_raw_ptz_status()
        return self._kalman.update(raw.pan, raw.tilt, raw.zoom)

    async def get_raw_ptz_status(self) -> PTZPosition:
        """Return unfiltered PTZ position directly from camera firmware."""
        self._require_connected()
        body = (
            f"<tptz:GetStatus>"
            f"<tptz:ProfileToken>{self._profile_token}</tptz:ProfileToken>"
            f"</tptz:GetStatus>"
        )
        resp = await _retry(_soap, self._ptz_url, body, self._username, self._password,
                     client=self._http)
        pt = _find(resp, "PanTilt")
        z  = _find(resp, "Zoom")
        try:
            pan  = float(pt.get("x", 0.0)) if pt is not None else 0.0
            tilt = float(pt.get("y", 0.0)) if pt is not None else 0.0
            zoom = float(z.get("x",  0.0)) if z  is not None else 0.0
        except (TypeError, ValueError):
            pan = tilt = zoom = 0.0
        return PTZPosition(pan=pan, tilt=tilt, zoom=zoom)

    # ── Absolute move ─────────────────────────────────────────────────────────

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
        Move to an absolute normalised position (-1..1 pan/tilt, 0..1 zoom).
        zoom_speed overrides the speed for the zoom axis independently.
        """
        # Cancel any pending deferred stop — absolute_move supersedes continuous.
        if self._deferred_stop_task and not self._deferred_stop_task.done():
            self._deferred_stop_task.cancel()
            self._deferred_stop_task = None
        self._require_connected()
        pan  = max(self._limits.pan_min,  min(self._limits.pan_max,  pan))
        tilt = max(self._limits.tilt_min, min(self._limits.tilt_max, tilt))
        zoom = max(self._limits.zoom_min, min(self._limits.zoom_max, zoom))
        speed = max(0.0, min(1.0, speed))
        z_spd = max(0.0, min(1.0, zoom_speed)) if zoom_speed is not None else speed

        body = (
            f"<tptz:AbsoluteMove>"
            f"<tptz:ProfileToken>{self._profile_token}</tptz:ProfileToken>"
            f'<tptz:Position>'
            f'<tt:PanTilt x="{pan:.6f}" y="{tilt:.6f}"/>'
            f'<tt:Zoom x="{zoom:.6f}"/>'
            f"</tptz:Position>"
            f"<tptz:Speed>"
            f'<tt:PanTilt x="{speed:.4f}" y="{speed:.4f}"/>'
            f'<tt:Zoom x="{z_spd:.4f}"/>'
            f"</tptz:Speed>"
            f"</tptz:AbsoluteMove>"
        )
        await _retry(_soap, self._ptz_url, body, self._username, self._password,
                     client=self._http)
        self._kalman.notify_commanded(pan, tilt, zoom)

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
        Velocity-adaptive absolute move with S-curve speed selection.

        Chooses pan/tilt speed based on the Euclidean move distance so that
        large zone-to-zone transits run at full speed while small corrections
        use reduced speed (less overshoot, less mechanical vibration).

        Speed tiers (pan/tilt Euclidean distance in ONVIF normalised units):
          micro  (< 0.02) : 25 % of nominal
          short  (< 0.08) : 50 %
          medium (< 0.20) : 75 %
          large  (>= 0.20): 100 %

        Returns recommended post-move settle time in seconds.
        """
        pan  = max(self._limits.pan_min,  min(self._limits.pan_max,  pan))
        tilt = max(self._limits.tilt_min, min(self._limits.tilt_max, tilt))
        zoom = max(self._limits.zoom_min, min(self._limits.zoom_max, zoom))

        if current_pos is None:
            try:
                current_pos = await self.get_ptz_status()
            except Exception:
                current_pos = PTZPosition()

        pt_dist = math.hypot(pan - current_pos.pan, tilt - current_pos.tilt)
        dzoom   = abs(zoom - current_pos.zoom)

        if   pt_dist < 0.02: pt_speed = nominal_speed * 0.25
        elif pt_dist < 0.08: pt_speed = nominal_speed * 0.50
        elif pt_dist < 0.20: pt_speed = nominal_speed * 0.75
        else:                 pt_speed = nominal_speed

        z_spd = zoom_speed if zoom_speed is not None else min(nominal_speed * 0.40, 0.25)
        await self.absolute_move(pan, tilt, zoom, speed=pt_speed, zoom_speed=z_spd)

        settle = 0.20 + pt_dist * 0.80 + dzoom * 1.20
        return float(max(0.20, min(settle, 1.0)))

    # ── Relative move ─────────────────────────────────────────────────────────

    async def relative_move(
        self,
        pan_delta: float,
        tilt_delta: float,
        zoom_delta: float = 0.0,
        speed: float = 0.5,
    ) -> None:
        self._require_connected()
        body = (
            f"<tptz:RelativeMove>"
            f"<tptz:ProfileToken>{self._profile_token}</tptz:ProfileToken>"
            f"<tptz:Translation>"
            f'<tt:PanTilt x="{pan_delta:.6f}" y="{tilt_delta:.6f}"/>'
            f'<tt:Zoom x="{zoom_delta:.6f}"/>'
            f"</tptz:Translation>"
            f"<tptz:Speed>"
            f'<tt:PanTilt x="{speed:.4f}" y="{speed:.4f}"/>'
            f'<tt:Zoom x="{speed:.4f}"/>'
            f"</tptz:Speed>"
            f"</tptz:RelativeMove>"
        )
        await _retry(_soap, self._ptz_url, body, self._username, self._password,
                     client=self._http)

    # ── Continuous move ───────────────────────────────────────────────────────

    async def continuous_move(
        self,
        pan_velocity: float,
        tilt_velocity: float,
        zoom_velocity: float = 0.0,
        timeout: float = 0.0,
    ) -> None:
        """
        Start continuous movement at given velocities (-1..1).
        Pass timeout > 0 to automatically stop after that many seconds.
        """
        self._require_connected()
        timeout_tag = f"<tptz:Timeout>PT{timeout:.3f}S</tptz:Timeout>" if timeout > 0 else ""
        body = (
            f"<tptz:ContinuousMove>"
            f"<tptz:ProfileToken>{self._profile_token}</tptz:ProfileToken>"
            f"<tptz:Velocity>"
            f'<tt:PanTilt x="{pan_velocity:.4f}" y="{tilt_velocity:.4f}"/>'
            f'<tt:Zoom x="{zoom_velocity:.4f}"/>'
            f"</tptz:Velocity>"
            f"{timeout_tag}"
            f"</tptz:ContinuousMove>"
        )
        await _retry(_soap, self._ptz_url, body, self._username, self._password,
                     client=self._http)
        if timeout > 0:
            # Cancel any previous deferred stop before scheduling a new one.
            # Without this, rapid tracking-loop calls accumulate background tasks
            # that fire stale Stop commands mid-movement, sending the camera in
            # random directions.
            if self._deferred_stop_task and not self._deferred_stop_task.done():
                self._deferred_stop_task.cancel()
            async def _deferred_stop(delay: float) -> None:
                await asyncio.sleep(delay)
                try:
                    await self.stop()
                except Exception:
                    pass
            self._deferred_stop_task = asyncio.ensure_future(_deferred_stop(timeout))

    # ── Stop ──────────────────────────────────────────────────────────────────

    async def stop(self, pan_tilt: bool = True, zoom: bool = True) -> None:
        # Cancel any pending deferred-stop task — this is a real stop, no need
        # for the background task to fire again afterwards.
        if self._deferred_stop_task and not self._deferred_stop_task.done():
            self._deferred_stop_task.cancel()
            self._deferred_stop_task = None
        self._require_connected()
        pt_str = "true" if pan_tilt else "false"
        z_str  = "true" if zoom     else "false"
        body = (
            f"<tptz:Stop>"
            f"<tptz:ProfileToken>{self._profile_token}</tptz:ProfileToken>"
            f"<tptz:PanTilt>{pt_str}</tptz:PanTilt>"
            f"<tptz:Zoom>{z_str}</tptz:Zoom>"
            f"</tptz:Stop>"
        )
        await _retry(_soap, self._ptz_url, body, self._username, self._password,
                     client=self._http)

    # ── Home ──────────────────────────────────────────────────────────────────

    async def goto_home(self, speed: float = 0.5) -> None:
        self._require_connected()
        body = (
            f"<tptz:GotoHomePosition>"
            f"<tptz:ProfileToken>{self._profile_token}</tptz:ProfileToken>"
            f"<tptz:Speed>"
            f'<tt:PanTilt x="{speed:.4f}" y="{speed:.4f}"/>'
            f'<tt:Zoom x="{speed:.4f}"/>'
            f"</tptz:Speed>"
            f"</tptz:GotoHomePosition>"
        )
        await _retry(_soap, self._ptz_url, body, self._username, self._password,
                     client=self._http)

    # ── Goto preset ───────────────────────────────────────────────────────────

    async def goto_preset(self, preset_token: str, speed: float = 0.5) -> None:
        """Move to a named/numbered PTZ preset."""
        self._require_connected()
        body = (
            f"<tptz:GotoPreset>"
            f"<tptz:ProfileToken>{self._profile_token}</tptz:ProfileToken>"
            f"<tptz:PresetToken>{preset_token}</tptz:PresetToken>"
            f"<tptz:Speed>"
            f'<tt:PanTilt x="{speed:.4f}" y="{speed:.4f}"/>'
            f'<tt:Zoom x="{speed:.4f}"/>'
            f"</tptz:Speed>"
            f"</tptz:GotoPreset>"
        )
        await _retry(_soap, self._ptz_url, body, self._username, self._password,
                     client=self._http)

    # ── FoV ───────────────────────────────────────────────────────────────────

    def get_fov_at_zoom(self, zoom: float) -> tuple[float, float]:
        """
        Return (fov_h_deg, fov_v_deg) for the given normalised zoom (0–1).
        Uses a log-linear model matching real PTZ optics.
        """
        zoom = max(0.0, min(1.0, zoom))
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

    # ── Travel time estimate ──────────────────────────────────────────────────

    def estimate_travel_time(
        self,
        from_pos: PTZPosition,
        to_pos: PTZPosition,
    ) -> float:
        """
        Estimate travel time in seconds from from_pos to to_pos based on
        mechanical speeds stored in PTZLimits. Returns the dominant axis time.
        """
        def _time(delta: float, speed_normalised: float, full_range: float) -> float:
            if speed_normalised <= 0:
                return 0.0
            deg_per_sec = speed_normalised * full_range
            return abs(delta) / deg_per_sec if deg_per_sec > 0 else 0.0

        pan_range  = self._limits.pan_max  - self._limits.pan_min
        tilt_range = self._limits.tilt_max - self._limits.tilt_min
        zoom_range = self._limits.zoom_max - self._limits.zoom_min

        return max(
            _time(to_pos.pan  - from_pos.pan,  self._limits.pan_speed,  pan_range),
            _time(to_pos.tilt - from_pos.tilt, self._limits.tilt_speed, tilt_range),
            _time(to_pos.zoom - from_pos.zoom, self._limits.zoom_speed, zoom_range),
        )

    # ── Reboot ────────────────────────────────────────────────────────────────

    async def reboot(self) -> None:
        self._require_connected()
        body = "<tds:SystemReboot/>"
        try:
            await _soap(self._device_url, body, self._username, self._password,
                        timeout=10.0, client=self._http)
        except Exception as exc:
            logger.warning("%s: SystemReboot SOAP failed: %s", self._host, exc)

    # ── Snapshot ──────────────────────────────────────────────────────────────

    async def get_snapshot(self) -> bytes:
        """Download a JPEG snapshot via ONVIF GetSnapshotUri + HTTP GET."""
        self._require_connected()
        # Get snapshot URI via SOAP (media service operation, not device service)
        if self._profile_token:
            body = (
                "<trt:GetSnapshotUri xmlns:trt='http://www.onvif.org/ver10/media/wsdl'>"
                f"<trt:ProfileToken>{self._profile_token}</trt:ProfileToken>"
                "</trt:GetSnapshotUri>"
            )
            media_url = self._media_url or self._device_url
            try:
                resp = await _soap(media_url, body, self._username, self._password,
                                   client=self._http)
                uri_el = _find(resp, "Uri")
                if uri_el is not None and uri_el.text:
                    uri = uri_el.text.strip()
                    # Reuse the persistent client for the JPEG download too
                    r = await self._http.get(
                        uri,
                        auth=(self._username, self._password),
                        timeout=10.0,
                    )
                    r.raise_for_status()
                    return r.content
            except Exception as exc:
                raise CameraOfflineError(f"{self._host}: snapshot failed: {exc}") from exc
        raise CameraOfflineError(f"{self._host}: no profile token for snapshot")

    # ── Coordinate conversions ────────────────────────────────────────────────

    def pixel_to_ptz(
        self,
        px: float,
        py: float,
        frame_w: int,
        frame_h: int,
        current_ptz: PTZPosition,
    ) -> tuple[float, float]:
        """
        Convert pixel (px, py) in a frame to an absolute PTZ pan/tilt target.

        Two modes:
          • Calibrated (preferred): uses measured K constants from auto_calibrate_fov().
          • Fallback: angle-based model using fov_h/v and pan_scale.
        """
        cx = (px - frame_w / 2.0) / (frame_w / 2.0)
        cy = -(py - frame_h / 2.0) / (frame_h / 2.0)

        if self._K_pan_wide is not None:
            K_pan_narrow  = self._K_pan_narrow or (
                self._K_pan_wide * (self._fov_h_narrow / self._fov_h_wide)
            )
            # K_tilt fallback: derive from K_pan using the V/H FOV aspect ratio.
            # Using K_pan directly (old behaviour) caused a ~1.77x overcorrection
            # (60°/34° ≈ 1.76) which sent the camera to the tilt ceiling.
            K_tilt_wide   = self._K_tilt_wide or (
                self._K_pan_wide * (self._fov_v_wide / self._fov_h_wide)
            )
            K_tilt_narrow = self._K_tilt_narrow or (
                K_tilt_wide * (self._fov_v_narrow / self._fov_v_wide)
            )
            K_pan  = _K_at_zoom(current_ptz.zoom, self._K_pan_wide, K_pan_narrow)
            K_tilt = _K_at_zoom(current_ptz.zoom, K_tilt_wide, K_tilt_narrow)
            d_pan  = cx * K_pan
            d_tilt = cy * K_tilt
        else:
            fov_h, fov_v = self.get_fov_at_zoom(current_ptz.zoom)
            pan_range  = self._limits.pan_max  - self._limits.pan_min
            tilt_range = self._limits.tilt_max - self._limits.tilt_min
            d_pan  = cx * (fov_h / 360.0) * pan_range  / self._pan_scale
            d_tilt = cy * (fov_v / 180.0) * tilt_range / self._tilt_scale

        # Safety cap: a single PTZ jump must not exceed 8% of the tilt range
        # (and 15% of pan range). 30% was too loose — a mis-signed K drove the
        # camera to the tilt limit in ~3 iterations. At 8% even a wrong-sign
        # correction stays within a narrow band while the next frame's feedback
        # corrects direction.
        tilt_range = self._limits.tilt_max - self._limits.tilt_min
        pan_range  = self._limits.pan_max  - self._limits.pan_min
        _MAX_TILT_STEP = tilt_range * 0.08
        _MAX_PAN_STEP  = pan_range  * 0.15
        d_tilt = max(-_MAX_TILT_STEP, min(_MAX_TILT_STEP, d_tilt))
        d_pan  = max(-_MAX_PAN_STEP,  min(_MAX_PAN_STEP,  d_pan))

        target_pan  = current_ptz.pan  + d_pan
        target_tilt = current_ptz.tilt + d_tilt
        return (
            max(self._limits.pan_min,  min(self._limits.pan_max,  target_pan)),
            max(self._limits.tilt_min, min(self._limits.tilt_max, target_tilt)),
        )

    def ptz_to_pixel(
        self,
        target_pan: float,
        target_tilt: float,
        frame_w: int,
        frame_h: int,
        current_ptz: PTZPosition,
    ) -> tuple[float, float]:
        """Inverse of pixel_to_ptz — project an absolute PTZ position to pixel coords."""
        d_pan  = target_pan  - current_ptz.pan
        d_tilt = target_tilt - current_ptz.tilt

        if self._K_pan_wide is not None:
            K_pan_narrow  = self._K_pan_narrow or (
                self._K_pan_wide * (self._fov_h_narrow / self._fov_h_wide)
            )
            K_tilt_wide   = self._K_tilt_wide or (
                self._K_pan_wide * (self._fov_v_wide / self._fov_h_wide)
            )
            K_tilt_narrow = self._K_tilt_narrow or (
                K_tilt_wide * (self._fov_v_narrow / self._fov_v_wide)
            )
            K_pan  = _K_at_zoom(current_ptz.zoom, self._K_pan_wide, K_pan_narrow)
            K_tilt = _K_at_zoom(current_ptz.zoom, K_tilt_wide, K_tilt_narrow)
            if K_pan == 0 or K_tilt == 0:
                return frame_w / 2.0, frame_h / 2.0
            cx = d_pan  / K_pan
            cy = d_tilt / K_tilt
        else:
            fov_h, fov_v = self.get_fov_at_zoom(current_ptz.zoom)
            pan_range  = self._limits.pan_max  - self._limits.pan_min
            tilt_range = self._limits.tilt_max - self._limits.tilt_min
            scale_pan  = (fov_h / 360.0) * pan_range  / self._pan_scale
            scale_tilt = (fov_v / 180.0) * tilt_range / self._tilt_scale
            if scale_pan == 0 or scale_tilt == 0:
                return frame_w / 2.0, frame_h / 2.0
            cx =  d_pan  / scale_pan
            cy =  d_tilt / scale_tilt

        px = (cx + 1.0) * (frame_w / 2.0)
        py = (-cy + 1.0) * (frame_h / 2.0)
        return px, py

    # ── Calibration ───────────────────────────────────────────────────────────

    def apply_calibration(self, params: dict) -> None:
        """Apply previously measured K calibration constants."""
        if params.get("K_pan_wide"):
            self._K_pan_wide    = float(params["K_pan_wide"])
        if params.get("K_pan_narrow"):
            self._K_pan_narrow  = float(params["K_pan_narrow"])
        if params.get("K_tilt_wide"):
            self._K_tilt_wide   = float(params["K_tilt_wide"])
        if params.get("K_tilt_narrow"):
            self._K_tilt_narrow = float(params["K_tilt_narrow"])
        if params.get("fov_h_wide"):
            self._fov_h_wide    = float(params["fov_h_wide"])
        if params.get("fov_h_narrow"):
            self._fov_h_narrow  = float(params["fov_h_narrow"])
        if params.get("fov_v_wide"):
            self._fov_v_wide    = float(params["fov_v_wide"])
        if params.get("fov_v_narrow"):
            self._fov_v_narrow  = float(params["fov_v_narrow"])
        logger.info(
            "%s: calibration applied — K_pan_wide=%.4f K_pan_narrow=%s "
            "K_tilt_wide=%s fov_h_wide=%.1f fov_h_narrow=%.1f",
            self._host,
            self._K_pan_wide or 0, self._K_pan_narrow,
            self._K_tilt_wide, self._fov_h_wide, self._fov_h_narrow,
        )

    @property
    def calibration(self) -> dict:
        """Return current calibration values (K + FoV) for persistence."""
        d: dict = {}
        if self._K_pan_wide    is not None: d["K_pan_wide"]    = round(self._K_pan_wide,    6)
        if self._K_pan_narrow  is not None: d["K_pan_narrow"]  = round(self._K_pan_narrow,  6)
        if self._K_tilt_wide   is not None: d["K_tilt_wide"]   = round(self._K_tilt_wide,   6)
        if self._K_tilt_narrow is not None: d["K_tilt_narrow"] = round(self._K_tilt_narrow, 6)
        d["fov_h_wide"]   = round(self._fov_h_wide,   2)
        d["fov_h_narrow"] = round(self._fov_h_narrow, 2)
        d["fov_v_wide"]   = round(self._fov_v_wide,   2)
        d["fov_v_narrow"] = round(self._fov_v_narrow, 2)
        return d

    async def auto_calibrate_fov(
        self,
        grab_frame_fn,
        *,
        delta_pan:  float = 0.03,
        delta_tilt: float = 0.02,
        settle_s:   float = 1.2,
    ) -> dict:
        """
        Auto-calibrate per-camera FOV by measuring optical-flow pixel displacement
        for a known ONVIF movement at wide (zoom=0) and narrow (zoom=1) settings.

        Returns a dict of measured K + FoV values suitable for storing in
        camera learned_params.  Also applies the results immediately.
        The camera is always restored to its original position after calibration.
        """
        self._require_connected()
        results: dict = {}

        origin = await self.get_raw_ptz_status()
        logger.info(
            "%s: starting FOV auto-calibration (origin pan=%.3f tilt=%.3f zoom=%.3f)",
            self._host, origin.pan, origin.tilt, origin.zoom,
        )

        async def _flush_and_grab(n_flush: int = 3) -> "np.ndarray | None":
            for _ in range(n_flush):
                f = await grab_frame_fn()
                if f is None:
                    return None
                await asyncio.sleep(0.08)
            return await grab_frame_fn()

        async def _measure(
            pan0: float, tilt0: float, zoom0: float,
            d_pan: float, d_tilt: float,
            label: str,
        ) -> tuple[float | None, float | None, float | None, float | None]:
            """
            Measure K_pan and K_tilt at a given zoom level.
            Moves by d_pan and d_tilt separately, measures optical flow each time.
            Returns (K_pan, K_tilt, fov_h_deg, fov_v_deg) or (None,…) on failure.
            """
            await self.absolute_move(pan0, tilt0, zoom0, speed=0.5)
            await asyncio.sleep(settle_s)

            frame_ref = await _flush_and_grab()
            if frame_ref is None:
                return None, None, None, None

            fh, fv = frame_ref.shape[1], frame_ref.shape[0]

            async def _wait_settled(
                target_pan: float, target_tilt: float,
                tol: float = 0.003, timeout: float = 3.0,
            ) -> PTZPosition | None:
                """Poll PTZ status until position is within *tol* of target, or timeout."""
                t_end = time.time() + timeout
                last: PTZPosition | None = None
                while time.time() < t_end:
                    try:
                        last = await self.get_raw_ptz_status()
                    except Exception:
                        await asyncio.sleep(0.1)
                        continue
                    if (abs(last.pan  - target_pan)  < tol
                            and abs(last.tilt - target_tilt) < tol):
                        return last
                    await asyncio.sleep(0.15)
                return last

            async def _repeat_measure(
                dp: float, dt: float, ax: str, n_reps: int = 3,
            ) -> float | None:
                """
                Run the same ±move N times and return median |pixel_shift / actual_delta|
                scaled back to the nominal d_pan for K computation.

                Using the actual PTZ delta (measured by polling) as the denominator
                neutralises cameras that silently swallow small moves or take longer
                to settle than ``settle_s`` — only reps where the camera actually
                moved contribute to the K estimate.
                """
                shifts: list[float] = []
                raw: list[tuple[float, float] | None] = []
                actual_deltas: list[float] = []
                for _ in range(n_reps):
                    # Verify and re-home if needed
                    ptz_ref = await _wait_settled(pan0, tilt0, tol=0.003, timeout=2.0)
                    ref = await _flush_and_grab()
                    if ref is None or ptz_ref is None:
                        raw.append(None)
                        continue
                    await self.absolute_move(pan0 + dp, tilt0 + dt, zoom0, speed=0.5)
                    ptz_obs = await _wait_settled(pan0 + dp, tilt0 + dt,
                                                  tol=0.003, timeout=3.0)
                    obs = await _flush_and_grab()
                    await self.absolute_move(pan0, tilt0, zoom0, speed=0.5)
                    await _wait_settled(pan0, tilt0, tol=0.003, timeout=3.0)
                    if obs is None or ptz_obs is None:
                        raw.append(None)
                        continue
                    # Confirm the camera actually moved — cheap sanity check
                    d_actual = (
                        (ptz_obs.pan - ptz_ref.pan) if ax == "x"
                        else (ptz_obs.tilt - ptz_ref.tilt)
                    )
                    d_nom = dp if ax == "x" else dt
                    if d_nom != 0 and abs(d_actual) < abs(d_nom) * 0.5:
                        logger.info(
                            "%s [%s]: %s-axis move not executed "
                            "(commanded=%.4f  actual=%.4f) — skipping rep",
                            self._host, label, ax, d_nom, d_actual,
                        )
                        raw.append(None)
                        continue
                    dx, dy = await asyncio.to_thread(
                        _optical_flow_shift, ref, obs, ax,
                    )
                    raw.append((dx, dy) if (dx is not None and dy is not None) else None)
                    val = dx if ax == "x" else dy
                    if val is None or abs(val) < 2:
                        continue
                    # Normalise pixel shift to nominal commanded delta so K is
                    # always expressed in pixels-per-(d_nom) units downstream.
                    # PRESERVE SIGN — some cameras invert tilt (positive PTZ
                    # = aim down). Without a signed K the tracker drives the
                    # camera away from the face every correction (the classic
                    # "tilts to the ceiling" failure).
                    shifts.append(val * d_nom / d_actual)
                    actual_deltas.append(abs(d_actual))
                logger.info(
                    "%s [%s]: %s-axis actual_deltas=%s",
                    self._host, label, ax, [f"{v:.4f}" for v in actual_deltas],
                )
                logger.info(
                    "%s [%s]: %s-axis repeats raw=%s  kept=%s",
                    self._host, label, ax, raw, shifts,
                )
                if not shifts:
                    return None
                # Sign consistency — all repeats must agree on direction.
                # A single flipped-sign sample usually means a feature lock
                # onto a moving object moved opposite to the pan; drop it.
                pos = [s for s in shifts if s > 0]
                neg = [s for s in shifts if s < 0]
                if pos and neg:
                    majority = pos if len(pos) >= len(neg) else neg
                    logger.info(
                        "%s [%s]: %s-axis mixed signs %s — keeping majority %s",
                        self._host, label, ax, shifts, majority,
                    )
                    shifts = majority
                    if not shifts:
                        return None
                import statistics as _st
                med = _st.median(shifts)
                if len(shifts) >= 2:
                    spread = max(shifts) - min(shifts)
                    if spread > abs(med) * 0.50:   # > 50% spread across repeats
                        # Drop the single most-deviant sample and retry the
                        # consistency check on the rest. If the remaining
                        # values still disagree, give up.
                        pruned = sorted(shifts, key=lambda v: abs(v - med))[:-1]
                        if len(pruned) >= 1:
                            med2 = _st.median(pruned)
                            spread2 = max(pruned) - min(pruned) if len(pruned) > 1 else 0.0
                            if spread2 <= abs(med2) * 0.50:
                                logger.info(
                                    "%s [%s]: %s-axis dropped outlier → using "
                                    "median=%.1f of %d pruned samples",
                                    self._host, label, ax, med2, len(pruned),
                                )
                                return med2
                        logger.warning(
                            "%s [%s]: %s-axis measurements unstable "
                            "(shifts=%s  med=%.1f  spread=%.1f)",
                            self._host, label, ax, shifts, med, spread,
                        )
                        return None
                else:
                    logger.warning(
                        "%s [%s]: %s-axis only 1 of %d repeats succeeded "
                        "(shift=%.1f) — accepting low-confidence value",
                        self._host, label, ax, n_reps, med,
                    )
                return med

            # ── Pan calibration ──────────────────────────────────────────────
            K_pan = None
            dx_med = await _repeat_measure(d_pan, 0.0, "x")
            if dx_med is not None:
                K_pan = d_pan * (fh / 2.0) / dx_med
                logger.info(
                    "%s [%s]: K_pan=%.5f (dx_med=%.1fpx  d_pan=%.4f)",
                    self._host, label, K_pan, dx_med, d_pan,
                )

            # ── Tilt calibration ─────────────────────────────────────────────
            K_tilt = None
            dy_med = await _repeat_measure(0.0, d_tilt, "y")
            if dy_med is not None:
                K_tilt = d_tilt * (fv / 2.0) / dy_med
                logger.info(
                    "%s [%s]: K_tilt=%.5f (dy_med=%.1fpx  d_tilt=%.4f)",
                    self._host, label, K_tilt, dy_med, d_tilt,
                )

            # FoV from K, inverted from the fallback pixel_to_ptz model:
            #   d_pan  = cx · (fov_h/360) · pan_range  / pan_scale
            #   d_tilt = cy · (fov_v/180) · tilt_range / tilt_scale
            # With K_pan  = d_pan  · (fw/2) / |dx| and cx=|dx|/(fw/2) at the
            # measurement, solving for fov_h gives:
            #   fov_h = 360 · K_pan  · pan_scale  / pan_range
            #   fov_v = 180 · K_tilt · tilt_scale / tilt_range
            pan_range  = max(self._limits.pan_max  - self._limits.pan_min,  1e-6)
            tilt_range = max(self._limits.tilt_max - self._limits.tilt_min, 1e-6)
            # FoV is a magnitude — take abs() so a sign-inverted K (camera
            # that reports tilt+= as "aim down") still yields a positive
            # physical angle. Sign stays on K for pixel_to_ptz direction.
            fov_h = None if K_pan  is None else 360.0 * abs(K_pan)  * self._pan_scale  / pan_range
            fov_v = None if K_tilt is None else 180.0 * abs(K_tilt) * self._tilt_scale / tilt_range

            return K_pan, K_tilt, fov_h, fov_v

        try:
            # ── Wide zoom (zoom=0) calibration ───────────────────────────────
            K_pw, K_tw, fov_hw, fov_vw = await _measure(
                origin.pan, origin.tilt, 0.0,
                delta_pan, delta_tilt, "wide",
            )
            if K_pw is not None:
                results["K_pan_wide"]  = round(K_pw, 6)
            if K_tw is not None:
                results["K_tilt_wide"] = round(K_tw, 6)
            if fov_hw is not None:
                results["fov_h_wide"]  = round(fov_hw, 2)
            if fov_vw is not None:
                results["fov_v_wide"]  = round(fov_vw, 2)

            # ── Narrow zoom (zoom=1) calibration ─────────────────────────────
            # Use smaller deltas at telephoto to stay within FOV
            K_pn, K_tn, fov_hn, fov_vn = await _measure(
                origin.pan, origin.tilt, 1.0,
                delta_pan * 0.1, delta_tilt * 0.1, "narrow",
            )
            if K_pn is not None:
                results["K_pan_narrow"]  = round(K_pn, 6)
            if K_tn is not None:
                results["K_tilt_narrow"] = round(K_tn, 6)
            if fov_hn is not None:
                results["fov_h_narrow"]  = round(fov_hn, 2)
            if fov_vn is not None:
                results["fov_v_narrow"]  = round(fov_vn, 2)

        except Exception as exc:
            logger.error("%s: auto_calibrate_fov failed: %s", self._host, exc)
        finally:
            # Always restore original position
            try:
                await self.absolute_move(origin.pan, origin.tilt, origin.zoom, speed=0.6)
            except Exception:
                pass

        if results:
            self.apply_calibration(results)
            logger.info("%s: auto-calibration complete: %s", self._host, results)
        else:
            logger.warning(
                "%s: auto-calibration produced no results — "
                "check that the camera FOV contains trackable features",
                self._host,
            )

        return results

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def limits(self) -> PTZLimits:
        return self._limits

    @property
    def is_connected(self) -> bool:
        return self._data is not None  # True sentinel set in _connect_once


# ── Fixed (non-PTZ) camera stub ───────────────────────────────────────────────

class FixedCameraController:
    """
    No-op camera controller for fixed cameras (BULLET / BULLET_ZOOM types).

    Satisfies the same interface as ONVIFController so PTZBrain and ZoneMapper
    can be initialised without an actual PTZ connection. All movement methods
    are silent no-ops; status always returns the origin position (0, 0, 0).
    """

    _limits: PTZLimits = PTZLimits()
    _pan_scale:  float = 1.0
    _tilt_scale: float = 1.0
    _FOV_H: float = 60.0
    _FOV_V: float = 45.0

    def __init__(self, rtsp_url: str = "") -> None:
        self._rtsp_url = rtsp_url
        self._fov_h_wide   = self._FOV_H
        self._fov_h_narrow = self._FOV_H
        self._fov_v_wide   = self._FOV_V
        self._fov_v_narrow = self._FOV_V

    @property
    def limits(self) -> PTZLimits:
        return self._limits

    @property
    def is_connected(self) -> bool:
        return True

    def apply_calibration(self, lp: dict) -> None:  # noqa: ARG002
        pass

    async def auto_calibrate_fov(self, grab_fn) -> None:  # noqa: ANN001
        return None

    async def get_ptz_status(self) -> PTZPosition:
        return PTZPosition(0.0, 0.0, 0.0)

    async def get_raw_ptz_status(self) -> PTZPosition:
        return PTZPosition(0.0, 0.0, 0.0)

    async def absolute_move(
        self,
        pan: float,       # noqa: ARG002
        tilt: float,      # noqa: ARG002
        zoom: float,      # noqa: ARG002
        speed: float = 0.5,
        *,
        zoom_speed: float | None = None,
    ) -> None:
        pass

    async def smooth_absolute_move(
        self,
        pan: float,       # noqa: ARG002
        tilt: float,      # noqa: ARG002
        zoom: float,      # noqa: ARG002
        *,
        nominal_speed: float = 0.5,
        current_pos: PTZPosition | None = None,
        zoom_speed: float | None = None,
    ) -> float:
        return 0.0

    async def relative_move(self, pan: float, tilt: float, zoom: float) -> None:
        pass

    async def continuous_move(
        self,
        pan: float,               # noqa: ARG002
        tilt: float,              # noqa: ARG002
        zoom: float = 0.0,        # noqa: ARG002
        timeout: float = 0.0,     # noqa: ARG002
    ) -> None:
        pass

    async def stop(self, pan_tilt: bool = True, zoom: bool = True) -> None:
        pass

    async def get_rtsp_url(self) -> str:
        return self._rtsp_url

    def get_fov_at_zoom(self, zoom: float) -> tuple[float, float]:
        return (self._FOV_H, self._FOV_V)

    def estimate_travel_time(
        self,
        from_pos: PTZPosition,  # noqa: ARG002
        to_pos: PTZPosition,    # noqa: ARG002
    ) -> float:
        return 0.0

    def pixel_to_ptz(
        self,
        px: float,
        py: float,
        frame_w: int,
        frame_h: int,
        current_ptz: PTZPosition,
    ) -> tuple[float, float]:
        return (current_ptz.pan, current_ptz.tilt)
