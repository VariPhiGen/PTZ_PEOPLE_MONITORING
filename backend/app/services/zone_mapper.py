"""
ZoneMapper — converts a live video frame into a ScanMap of PTZ scan cells.

Algorithm per map_zone() call:
  1. YOLO person detection filtered to the ROI.
  2. Each person centroid projected from pixels → PTZ (pan, tilt) coordinates.
  3. DBSCAN on (pan, tilt) with eps = fov_h_norm / faces_per_cell, min_samples=1.
  4. Per cluster: ScanCell with center PTZ, required_zoom, expected_faces,
     unrecognized_count, and priority.
  5. Faculty auto-detect: standing person nearest the top edge of the ROI.

remap() decides between:
  - Light remap: shift cell centroids (person drift < 30%).
  - Heavy remap: full re-cluster (> 30% of persons moved by > eps).

Required-zoom derivation:
  The log-linear FoV model (from ONVIFController) is:
      fov_h(z) = FOV_H_WIDE * exp(z * ln(FOV_H_NARROW / FOV_H_WIDE))
  Observing IOD_obs pixels at current zoom (fov_h_cur):
      θ_face = IOD_obs * fov_h_cur / frame_width       (angular size, degrees)
  Target IOD = 60 px at required zoom:
      fov_target = θ_face * frame_width / 60  =  fov_h_cur * IOD_obs / 60
  Invert the model:
      z_target = ln(fov_target / FOV_H_WIDE) / ln(FOV_H_NARROW / FOV_H_WIDE)
  If no face is observed, IOD is estimated anthropometrically from the person
  bounding box width (IOD_est ≈ 0.12 * bbox_width).
"""
from __future__ import annotations

import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from sklearn.cluster import DBSCAN

from app.services.onvif_controller import ONVIFController, PTZPosition

if TYPE_CHECKING:
    from app.services.ai_pipeline import AIPipeline, PersonDetection

logger = logging.getLogger(__name__)

# ── Tunable constants ──────────────────────────────────────────────────────────

_TARGET_IOD_PX           = 60.0    # minimum inter-ocular distance at scan zoom
_IOD_FROM_BBOX_FACTOR    = 0.12    # anthropometric: IOD ≈ 12% of person bbox width
_DEFAULT_FACES_PER_CELL  = 4       # DBSCAN eps denominator
_HEAVY_REMAP_THRESHOLD   = 0.30    # fraction of moved persons that triggers re-cluster
_MIN_EPS_PTZ             = 0.01    # minimum DBSCAN eps in ONVIF units (safety floor)
_PRIORITY_UNREC_WEIGHT   = 2.0     # extra priority per unrecognized face
_CELL_EMA_ALPHA          = 0.75    # light-remap centroid EMA weight (new vs old)
                                   # 0.75 → 75% new measurement, 25% old position
                                   # prevents abrupt cell-position jumps between
                                   # frames while still tracking real crowd movement


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class ScanCell:
    """One PTZ position that the camera should visit during a scan cycle."""
    cell_id:            str
    center_pan:         float           # ONVIF normalised pan
    center_tilt:        float           # ONVIF normalised tilt
    required_zoom:      float           # zoom needed for >= 60px IOD
    expected_faces:     int             # number of persons in this cluster
    unrecognized_count: int             # how many have not been matched yet
    priority:           float           # higher = scan sooner
    pixel_centroid:     tuple[float, float]   # (cx, cy) in frame pixels
    person_indices:     list[int] = field(default_factory=list)  # detection row indices


@dataclass
class ScanMap:
    """Full zone description for one camera frame."""
    cells:           list[ScanCell]
    faculty_cell_id: str | None        # cell containing the auto-detected faculty
    frame_w:         int
    frame_h:         int
    camera_ptz:      PTZPosition
    version:         int   = 0
    created_at:      float = field(default_factory=time.time)


# ── Private helpers ────────────────────────────────────────────────────────────

def _zoom_for_fov(target_fov_h: float, ctrl: ONVIFController) -> float:
    """
    Inverse of the controller's log-linear FoV model.
    Returns zoom ∈ [0, 1] that achieves `target_fov_h` (degrees).
    Uses calibrated per-instance FOV values (not class-level defaults).
    """
    wide   = ctrl._fov_h_wide    # per-instance calibrated value
    narrow = ctrl._fov_h_narrow  # per-instance calibrated value
    # Fixed-FOV cameras (BULLET): wide == narrow, no zoom possible — return 0.
    if abs(wide - narrow) < 1e-6 or wide <= 0 or narrow <= 0:
        return 0.0
    # Clamp target to achievable range
    target = max(narrow, min(wide, target_fov_h))
    z = math.log(target / wide) / math.log(narrow / wide)
    return max(0.0, min(1.0, z))


def _required_zoom(
    frame_w: int,
    current_zoom: float,
    ctrl: ONVIFController,
    observed_iod_px: float | None,
    person_bbox_w: float | None,
) -> float:
    """
    Compute the zoom needed so that the face in this cluster reaches
    _TARGET_IOD_PX.  Falls back to the anthropometric bbox estimate when
    no face detection is available.
    """
    fov_h_cur, _ = ctrl.get_fov_at_zoom(current_zoom)

    if observed_iod_px is not None and observed_iod_px > 1.0:
        iod_obs = observed_iod_px
    elif person_bbox_w is not None and person_bbox_w > 1.0:
        iod_obs = _IOD_FROM_BBOX_FACTOR * person_bbox_w
    else:
        return current_zoom   # no data → stay at current zoom

    # Angular size of the face at current zoom
    # fov_target such that: fov_target * _TARGET_IOD_PX = fov_h_cur * iod_obs
    fov_target = fov_h_cur * iod_obs / _TARGET_IOD_PX
    return _zoom_for_fov(fov_target, ctrl)


def _polygon_contains(px: float, py: float, poly: list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test (normalised coords)."""
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _world_polygon_contains(pan: float, tilt: float, poly: list[dict]) -> bool:
    """Ray-casting point-in-polygon for world-space {pan, tilt} polygons."""
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]["pan"], poly[i]["tilt"]
        xj, yj = poly[j]["pan"], poly[j]["tilt"]
        if ((yi > tilt) != (yj > tilt)) and (pan < (xj - xi) * (tilt - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _roi_to_polygon(roi: dict | list, frame_w: int, frame_h: int) -> list[tuple[float, float]]:
    """Convert any ROI format to a list of (pixel_x, pixel_y) polygon vertices."""
    if isinstance(roi, list):
        pts: list[tuple[float, float]] = []
        for p in roi:
            if isinstance(p, dict):
                pts.append((float(p["x"]), float(p["y"])))
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                pts.append((float(p[0]), float(p[1])))
        if pts and all(0 <= x <= 1.0 and 0 <= y <= 1.0 for x, y in pts):
            return [(x * frame_w, y * frame_h) for x, y in pts]
        return pts

    rx = float(roi.get("x", 0))
    ry = float(roi.get("y", 0))
    rw = float(roi.get("w", 1.0))
    rh = float(roi.get("h", 1.0))
    if all(v <= 1.0 for v in (rx, ry, rw, rh)):
        rx *= frame_w; ry *= frame_h; rw *= frame_w; rh *= frame_h
    return [(rx, ry), (rx + rw, ry), (rx + rw, ry + rh), (rx, ry + rh)]


def _roi_bounding_rect(roi: dict | list | None, frame_w: int, frame_h: int) -> dict | None:
    """Return {x, y, w, h} bounding rect (pixel coords) for the AI pipeline."""
    if roi is None:
        return None
    poly = _roi_to_polygon(roi, frame_w, frame_h)
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return {"x": min(xs), "y": min(ys), "w": max(xs) - min(xs), "h": max(ys) - min(ys)}


def _inside_roi(
    cx: float,
    cy: float,
    roi: dict | list | None,
    frame_w: int,
    frame_h: int,
) -> bool:
    if roi is None:
        return True
    poly = _roi_to_polygon(roi, frame_w, frame_h)
    return _polygon_contains(cx, cy, poly)


def _faculty_index(
    persons: list[PersonDetection],
    roi: dict | None,
    frame_w: int,
    frame_h: int,
) -> int | None:
    """
    Return the index of the faculty candidate — the standing person whose
    centroid is nearest the top edge of the ROI.

    Standing persons are preferred; if none, any person near the top is used.
    """
    if roi is None:
        roi_top = 0
    elif isinstance(roi, dict):
        roi_top = roi.get("y", 0)
    else:
        poly = _roi_to_polygon(roi, frame_w, frame_h)
        roi_top = min(p[1] for p in poly)

    # Compute distance-from-top for all ROI-filtered persons
    best_idx: int | None = None
    best_dist = float("inf")
    fallback_idx: int | None = None
    fallback_dist = float("inf")

    for i, p in enumerate(persons):
        if not _inside_roi(p.center[0], p.center[1], roi, frame_w, frame_h):
            continue
        dist = p.center[1] - roi_top   # smaller y → nearer top → faculty candidate
        if dist < best_dist:  # is_standing removed in v2 (bbox-only YOLO)
            best_dist = dist
            best_idx = i
        if dist < fallback_dist:
            fallback_dist = dist
            fallback_idx = i

    return best_idx if best_idx is not None else fallback_idx


# ── Main class ─────────────────────────────────────────────────────────────────

class ZoneMapper:
    """
    Maps persons in a camera frame to a list of PTZ ScanCells.

    Parameters:
        onvif_controller  — connected ONVIFController instance.
        faces_per_cell    — expected cluster density used to set DBSCAN eps.
    """

    def __init__(
        self,
        onvif_controller: ONVIFController,
        faces_per_cell: int = _DEFAULT_FACES_PER_CELL,
    ) -> None:
        self._ctrl = onvif_controller
        self._faces_per_cell = faces_per_cell

    # ── Patrol fallback ──────────────────────────────────────────────────────

    def _patrol_cells_from_world_roi(
        self,
        roi_world: list[dict],
        scan_cell_meters: float,
        assumed_distance_m: float = 7.0,
    ) -> list[ScanCell]:
        """Generate scan cells directly from world-space (pan/tilt) ROI polygon.

        Works entirely in PTZ coordinate space — no pixel conversion needed.
        ``scan_cell_meters`` controls the angular width of each cell at the
        assumed observation distance.  If the resulting grid is too coarse,
        the algorithm switches to a recognition-zoom-based subdivision to
        ensure dense-enough coverage.
        """
        if not roi_world or len(roi_world) < 3:
            return []

        pans = [p["pan"] for p in roi_world]
        tilts = [p["tilt"] for p in roi_world]
        bb_pan_min, bb_pan_max = min(pans), max(pans)
        bb_tilt_min, bb_tilt_max = min(tilts), max(tilts)

        limits = self._ctrl.limits
        pan_range = limits.pan_max - limits.pan_min
        tilt_range = limits.tilt_max - limits.tilt_min

        roi_dpan = bb_pan_max - bb_pan_min
        roi_dtilt = bb_tilt_max - bb_tilt_min

        # Scale factors: physical_range / ONVIF_convention_range.
        # Needed to convert between physical degrees and ONVIF units correctly.
        ps = self._ctrl._pan_scale   # physical_pan_range_deg / 180°
        ts = self._ctrl._tilt_scale  # physical_tilt_range_deg / 90° (defaults = pan_scale)

        # --- Primary method: distance-based cell sizing ---
        cell_angle_rad = 2.0 * math.atan(scan_cell_meters / (2.0 * assumed_distance_m))
        cell_angle_deg = math.degrees(cell_angle_rad)

        # Convert physical degrees → ONVIF units (divide by pan_scale/tilt_scale)
        cell_dpan  = cell_angle_deg / 360.0 * pan_range  / ps
        cell_dtilt = cell_angle_deg / 180.0 * tilt_range / ts

        cols = max(1, round(roi_dpan / cell_dpan)) if cell_dpan > 1e-9 else 1
        rows = max(1, round(roi_dtilt / cell_dtilt)) if cell_dtilt > 1e-9 else 1

        # --- Fallback: recognition-zoom-based sizing ---
        # If the grid is too coarse (< 3x3), use the FOV at a recognition-
        # quality zoom (0.35) to size cells instead.  This ensures the patrol
        # grid is dense enough for detailed scanning.
        _RECOG_ZOOM = 0.35
        if cols * rows < 9:
            fov_h_recog, fov_v_recog = self._ctrl.get_fov_at_zoom(_RECOG_ZOOM)
            recog_dpan  = fov_h_recog / 360.0 * pan_range  / ps
            recog_dtilt = fov_v_recog / 180.0 * tilt_range / ts
            # Each cell = one recognition FOV (with 20% overlap)
            step_dpan  = recog_dpan  * 0.8
            step_dtilt = recog_dtilt * 0.8
            cols = max(cols, max(2, math.ceil(roi_dpan  / step_dpan)))
            rows = max(rows, max(2, math.ceil(roi_dtilt / step_dtilt)))
            cell_angle_deg = fov_h_recog

        # Ensure minimum 2×2 grid and hard cap to prevent combinatorial explosion
        _MAX_CELLS = 200
        cols = max(cols, 2)
        rows = max(rows, 2)
        if cols * rows > _MAX_CELLS:
            scale = (_MAX_CELLS / (cols * rows)) ** 0.5
            cols = max(2, round(cols * scale))
            rows = max(2, round(rows * scale))
            logger.warning(
                "patrol grid capped to %d×%d (was too many cells — check camera_distance_m and scan_cell_meters)",
                cols, rows,
            )

        actual_dpan  = roi_dpan  / cols
        actual_dtilt = roi_dtilt / rows

        # Required zoom: FOV should cover roughly one cell's physical angular width.
        # Convert ONVIF units back to physical degrees (* pan_scale) before inverting FOV model.
        actual_cell_angle_h = actual_dpan / pan_range * 360.0 * ps
        req_zoom = _zoom_for_fov(actual_cell_angle_h, self._ctrl)
        req_zoom = max(0.10, min(req_zoom, 0.55))

        cells: list[ScanCell] = []
        idx = 0
        for r in range(rows):
            for c in range(cols):
                cpan = bb_pan_min + actual_dpan * (c + 0.5)
                ctilt = bb_tilt_min + actual_dtilt * (r + 0.5)

                if _world_polygon_contains(cpan, ctilt, roi_world):
                    cells.append(ScanCell(
                        cell_id=f"patrol-{idx}",
                        center_pan=cpan,
                        center_tilt=ctilt,
                        required_zoom=req_zoom,
                        expected_faces=0,
                        unrecognized_count=0,
                        priority=1.0,
                        pixel_centroid=(0.0, 0.0),
                        person_indices=[],
                    ))
                    idx += 1

        if not cells:
            center_pan = (bb_pan_min + bb_pan_max) / 2
            center_tilt = (bb_tilt_min + bb_tilt_max) / 2
            cells.append(ScanCell(
                cell_id="patrol-0",
                center_pan=center_pan,
                center_tilt=center_tilt,
                required_zoom=req_zoom,
                expected_faces=0,
                unrecognized_count=0,
                priority=1.0,
                pixel_centroid=(0.0, 0.0),
                person_indices=[],
            ))

        logger.info(
            "world-space patrol grid: %d×%d = %d cells  "
            "cell_angle=%.1f°  zoom=%.2f  scan_m=%.1f  dist=%.0fm",
            cols, rows, len(cells), cell_angle_deg, req_zoom,
            scan_cell_meters, assumed_distance_m,
        )
        return cells

    def _patrol_cells_from_roi(
        self,
        roi: dict | list | None,
        frame_w: int,
        frame_h: int,
        camera_ptz: PTZPosition,
        scan_cell_meters: float | None = None,
        camera_distance_m: float = 7.0,
    ) -> list[ScanCell]:
        """Generate scan cells covering the ROI polygon when no persons detected.

        When ``scan_cell_meters`` is provided, the grid density is derived from
        the physical cell size rather than a fixed 2×2/3×2 pattern.
        """
        if roi is None:
            cx, cy = frame_w / 2, frame_h / 2
            pan, tilt = self._ctrl.pixel_to_ptz(cx, cy, frame_w, frame_h, camera_ptz)
            return [ScanCell(
                cell_id="patrol-0", center_pan=pan, center_tilt=tilt,
                required_zoom=0.0, expected_faces=0, unrecognized_count=0,
                priority=1.0, pixel_centroid=(cx, cy), person_indices=[],
            )]

        poly = _roi_to_polygon(roi, frame_w, frame_h)
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        roi_w = max_x - min_x
        roi_h = max_y - min_y

        _MAX_CELLS = 200
        if scan_cell_meters and scan_cell_meters > 0:
            # Estimate pixels-per-meter from the FOV at current zoom:
            # At overview zoom the full frame covers fov_h degrees.
            fov_h, _ = self._ctrl.get_fov_at_zoom(camera_ptz.zoom)
            frame_width_m = 2.0 * camera_distance_m * math.tan(math.radians(fov_h / 2))
            px_per_meter = frame_w / frame_width_m if frame_width_m > 0 else 100.0
            cell_px = scan_cell_meters * px_per_meter
            cols = max(1, round(roi_w / cell_px))
            rows = max(1, round(roi_h / cell_px))
            if cols * rows > _MAX_CELLS:
                scale = (_MAX_CELLS / (cols * rows)) ** 0.5
                cols = max(1, round(cols * scale))
                rows = max(1, round(rows * scale))
                logger.warning("patrol grid capped to %d×%d — check camera_distance_m/scan_cell_meters", cols, rows)
        else:
            cols = 3 if roi_w > roi_h * 1.5 else 2
            rows = 2

        cells: list[ScanCell] = []
        cell_w = roi_w / cols
        cell_h = roi_h / rows
        idx = 0
        for r in range(rows):
            for c in range(cols):
                cx = min_x + cell_w * (c + 0.5)
                cy = min_y + cell_h * (r + 0.5)
                if _polygon_contains(cx, cy, poly):
                    pan, tilt = self._ctrl.pixel_to_ptz(cx, cy, frame_w, frame_h, camera_ptz)
                    cells.append(ScanCell(
                        cell_id=f"patrol-{idx}",
                        center_pan=pan, center_tilt=tilt,
                        required_zoom=0.3,
                        expected_faces=0, unrecognized_count=0,
                        priority=1.0,
                        pixel_centroid=(cx, cy),
                        person_indices=[],
                    ))
                    idx += 1
        if not cells:
            cx = (min_x + max_x) / 2
            cy = (min_y + max_y) / 2
            pan, tilt = self._ctrl.pixel_to_ptz(cx, cy, frame_w, frame_h, camera_ptz)
            cells.append(ScanCell(
                cell_id="patrol-0", center_pan=pan, center_tilt=tilt,
                required_zoom=0.0, expected_faces=0, unrecognized_count=0,
                priority=1.0, pixel_centroid=(cx, cy), person_indices=[],
            ))
        return cells

    # ── Primary API ────────────────────────────────────────────────────────────

    def map_zone(
        self,
        frame: np.ndarray,
        roi_rect: dict | None,
        camera_ptz: PTZPosition,
        ai_pipeline: AIPipeline,
        scan_cell_meters: float | None = None,
        roi_world: list[dict] | None = None,
        camera_distance_m: float = 7.0,
    ) -> ScanMap:
        """
        Detect all persons in the frame, cluster them into ScanCells, and
        identify the faculty.

        Parameters:
            frame       — BGR image (HxWx3 uint8).
            roi_rect    — {"x","y","w","h"} active monitoring region, or None.
            camera_ptz  — current camera PTZ state.
            ai_pipeline — loaded AIPipeline for person + face detection.

        Returns a ScanMap ready to hand to PathPlanner.
        """
        h, w = frame.shape[:2]

        # ── Step 1: Person detection ──────────────────────────────────────────
        # YOLO expects a {x,y,w,h} rect; convert polygon ROI to bounding rect
        bbox_roi = _roi_bounding_rect(roi_rect, w, h)
        persons: list[PersonDetection] = ai_pipeline.detect_persons(frame, bbox_roi)
        roi_persons = [
            p for p in persons
            if _inside_roi(p.center[0], p.center[1], roi_rect, w, h)
        ]

        if not roi_persons:
            # No persons detected — generate patrol cells so the camera
            # still scans the defined zone.  Prefer world-space coordinates.
            if roi_world and len(roi_world) >= 3:
                patrol_cells = self._patrol_cells_from_world_roi(
                    roi_world, scan_cell_meters or 5.0, camera_distance_m,
                )
            else:
                patrol_cells = self._patrol_cells_from_roi(
                    roi_rect, w, h, camera_ptz, scan_cell_meters, camera_distance_m,
                )
            return ScanMap(
                cells=patrol_cells, faculty_cell_id=None,
                frame_w=w, frame_h=h,
                camera_ptz=camera_ptz,
            )

        # ── Step 2: Pixel centroids → PTZ ────────────────────────────────────
        ptz_coords = np.array(
            [
                self._ctrl.pixel_to_ptz(p.center[0], p.center[1], w, h, camera_ptz)
                for p in roi_persons
            ],
            dtype=np.float64,
        )   # [N, 2]  columns: (pan, tilt)

        # ── Step 3: DBSCAN clustering in PTZ space ────────────────────────────
        fov_h, _ = self._ctrl.get_fov_at_zoom(camera_ptz.zoom)
        limits = self._ctrl.limits
        pan_range = limits.pan_max - limits.pan_min
        # Convert FoV to normalised pan units
        fov_h_norm = (fov_h / 360.0) * pan_range
        eps = max(_MIN_EPS_PTZ, fov_h_norm / self._faces_per_cell)

        labels = DBSCAN(eps=eps, min_samples=1, metric="euclidean").fit_predict(ptz_coords)
        # DBSCAN may label some points as noise (-1); assign each noise point
        # to its own cluster so it still gets scanned.
        next_label = max(labels) + 1 if len(labels) else 0
        for i in range(len(labels)):
            if labels[i] == -1:
                labels[i] = next_label
                next_label += 1
        num_clusters = int(max(labels) + 1) if len(labels) else 0

        # ── Step 4: Face detections for IOD estimates ────────────────────────
        faces = ai_pipeline.detect_faces(frame, [p.bbox for p in roi_persons])
        # Map each face to the nearest person by centroid proximity
        face_iod_by_person: dict[int, float] = {}
        for face in faces:
            face_cx = (face.bbox[0] + face.bbox[2]) / 2.0
            face_cy = (face.bbox[1] + face.bbox[3]) / 2.0
            nearest_idx, nearest_dist = 0, float("inf")
            for i, p in enumerate(roi_persons):
                d = math.hypot(face_cx - p.center[0], face_cy - p.center[1])
                if d < nearest_dist:
                    nearest_dist = d
                    nearest_idx = i
            # Accept if within 0.6 × person bbox height
            p_h = roi_persons[nearest_idx].bbox[3] - roi_persons[nearest_idx].bbox[1]
            if nearest_dist < 0.6 * p_h:
                face_iod_by_person[nearest_idx] = face.inter_ocular_px

        # ── Step 5: Build ScanCells ───────────────────────────────────────────
        cells: list[ScanCell] = []
        faculty_person_index: int | None = _faculty_index(roi_persons, roi_rect, w, h)

        for cluster_id in range(num_clusters):
            member_indices = [i for i, lbl in enumerate(labels) if lbl == cluster_id]

            # Cluster centroid in PTZ space
            cluster_pts  = ptz_coords[member_indices]
            center_pan   = float(cluster_pts[:, 0].mean())
            center_tilt  = float(cluster_pts[:, 1].mean())

            # Pixel centroid (for display)
            pxs = [roi_persons[i].center[0] for i in member_indices]
            pys = [roi_persons[i].center[1] for i in member_indices]
            pixel_cx = float(np.mean(pxs))
            pixel_cy = float(np.mean(pys))

            # IOD estimate: prefer observed face IOD, fallback to bbox
            iod_values = [face_iod_by_person[i] for i in member_indices if i in face_iod_by_person]
            obs_iod  = float(np.mean(iod_values)) if iod_values else None
            bbox_ws  = [roi_persons[i].bbox[2] - roi_persons[i].bbox[0] for i in member_indices]
            mean_bbox_w = float(np.mean(bbox_ws))

            req_zoom = _required_zoom(
                frame_w=w,
                current_zoom=camera_ptz.zoom,
                ctrl=self._ctrl,
                observed_iod_px=obs_iod,
                person_bbox_w=mean_bbox_w,
            )

            n_faces = len(member_indices)
            cell = ScanCell(
                cell_id=str(uuid.uuid4()),
                center_pan=center_pan,
                center_tilt=center_tilt,
                required_zoom=req_zoom,
                expected_faces=n_faces,
                unrecognized_count=n_faces,    # starts fully unrecognized
                priority=n_faces + _PRIORITY_UNREC_WEIGHT * n_faces,
                pixel_centroid=(pixel_cx, pixel_cy),
                person_indices=member_indices,
            )
            cells.append(cell)

        # ── Step 6: Faculty cell ──────────────────────────────────────────────
        faculty_cell_id: str | None = None
        if faculty_person_index is not None:
            cluster_lbl = labels[faculty_person_index]
            for cell in cells:
                if any(labels[idx] == cluster_lbl for idx in cell.person_indices):
                    faculty_cell_id = cell.cell_id
                    # Boost faculty cell priority
                    cell.priority = max(cell.priority, 999.0)
                    break

        return ScanMap(
            cells=cells,
            faculty_cell_id=faculty_cell_id,
            frame_w=w,
            frame_h=h,
            camera_ptz=camera_ptz,
        )

    # ── Remap ──────────────────────────────────────────────────────────────────

    def remap(
        self,
        current_map: ScanMap,
        new_frame: np.ndarray,
        camera_ptz: PTZPosition,
        ai_pipeline: AIPipeline,
        roi_rect: dict | None = None,
        scan_cell_meters: float | None = None,
        roi_world: list[dict] | None = None,
        camera_distance_m: float = 7.0,
    ) -> ScanMap:
        """
        Update an existing ScanMap from a new frame.

        Light remap — cell centroids shifted:
            Used when < 30% of detected persons moved farther than eps from
            their currently assigned cluster centroid.

        Heavy remap — full re-cluster:
            Used when >= 30% of persons have drifted significantly, indicating
            crowd movement, new arrivals, or camera movement.
        """
        h, w = new_frame.shape[:2]
        bbox_roi = _roi_bounding_rect(roi_rect, w, h)
        persons: list[PersonDetection] = ai_pipeline.detect_persons(new_frame, bbox_roi)
        roi_persons = [
            p for p in persons
            if _inside_roi(p.center[0], p.center[1], roi_rect, w, h)
        ]

        if not roi_persons:
            if roi_world and len(roi_world) >= 3:
                patrol_cells = self._patrol_cells_from_world_roi(
                    roi_world, scan_cell_meters or 5.0, camera_distance_m,
                )
            else:
                patrol_cells = self._patrol_cells_from_roi(
                    roi_rect, w, h, camera_ptz, scan_cell_meters, camera_distance_m,
                )
            return ScanMap(
                cells=patrol_cells, faculty_cell_id=None,
                frame_w=w, frame_h=h,
                camera_ptz=camera_ptz,
                version=current_map.version + 1,
            )

        # Project new detections to PTZ
        new_ptz = np.array(
            [self._ctrl.pixel_to_ptz(p.center[0], p.center[1], w, h, camera_ptz)
             for p in roi_persons],
            dtype=np.float64,
        )

        # Old cell centroids
        old_centers = np.array(
            [[c.center_pan, c.center_tilt] for c in current_map.cells],
            dtype=np.float64,
        )

        # Compute eps for motion-detection threshold
        fov_h, _ = self._ctrl.get_fov_at_zoom(camera_ptz.zoom)
        pan_range = self._ctrl.limits.pan_max - self._ctrl.limits.pan_min
        eps = max(_MIN_EPS_PTZ, (fov_h / 360.0) * pan_range / self._faces_per_cell)

        # For each new person, find nearest old cell center
        moved_count = 0
        for pt in new_ptz:
            if len(old_centers) == 0:
                moved_count += 1
                continue
            dists = np.linalg.norm(old_centers - pt, axis=1)
            if dists.min() > eps:
                moved_count += 1

        fraction_moved = moved_count / len(roi_persons)
        logger.debug(
            "remap: %.0f%% persons moved (threshold %.0f%%)",
            fraction_moved * 100, _HEAVY_REMAP_THRESHOLD * 100,
        )

        if fraction_moved >= _HEAVY_REMAP_THRESHOLD:
            logger.info("remap → heavy (re-cluster)")
            new_map = self.map_zone(
                new_frame, roi_rect, camera_ptz, ai_pipeline,
                scan_cell_meters, roi_world, camera_distance_m,
            )
            # Preserve existing unrecognized counts where cells overlap
            new_map = self._carry_recognition_state(current_map, new_map)
            new_map.version = current_map.version + 1
            return new_map

        # Light remap: update PTZ centroids with EMA smoothing.
        # Raw new centroid is blended with the current position using
        # _CELL_EMA_ALPHA so gradual crowd drift is tracked accurately
        # while random frame-to-frame jitter is suppressed.
        logger.debug("remap → light (centroid update, EMA α=%.2f)", _CELL_EMA_ALPHA)
        updated_cells: list[ScanCell] = []
        for cell in current_map.cells:
            c_center = np.array([cell.center_pan, cell.center_tilt])
            # Assign persons within eps to this cell
            assigned = [
                pt for pt in new_ptz
                if np.linalg.norm(c_center - pt) <= eps
            ]
            if assigned:
                pts = np.array(assigned)
                raw_pan  = float(pts[:, 0].mean())
                raw_tilt = float(pts[:, 1].mean())
                # EMA: blend new measurement toward current position
                new_pan  = _CELL_EMA_ALPHA * raw_pan  + (1.0 - _CELL_EMA_ALPHA) * cell.center_pan
                new_tilt = _CELL_EMA_ALPHA * raw_tilt + (1.0 - _CELL_EMA_ALPHA) * cell.center_tilt
            else:
                new_pan, new_tilt = cell.center_pan, cell.center_tilt

            from dataclasses import replace
            updated_cells.append(replace(cell, center_pan=new_pan, center_tilt=new_tilt))

        return ScanMap(
            cells=updated_cells,
            faculty_cell_id=current_map.faculty_cell_id,
            frame_w=w,
            frame_h=h,
            camera_ptz=camera_ptz,
            version=current_map.version + 1,
        )

    def _carry_recognition_state(self, old_map: ScanMap, new_map: ScanMap) -> ScanMap:
        """
        After a heavy remap, transfer unrecognized_count from old cells to
        spatially overlapping new cells, preserving recognition progress.
        """
        for new_cell in new_map.cells:
            for old_cell in old_map.cells:
                d = math.hypot(
                    new_cell.center_pan  - old_cell.center_pan,
                    new_cell.center_tilt - old_cell.center_tilt,
                )
                # Cells are "the same" if centroids are within one cell width
                fov_h, _ = self._ctrl.get_fov_at_zoom(new_map.camera_ptz.zoom)
                pan_range = self._ctrl.limits.pan_max - self._ctrl.limits.pan_min
                cell_width = (fov_h / 360.0) * pan_range / self._faces_per_cell
                if d <= cell_width:
                    recognised = old_cell.expected_faces - old_cell.unrecognized_count
                    new_unrecog = max(0, new_cell.expected_faces - recognised)
                    from dataclasses import replace
                    idx = new_map.cells.index(new_cell)
                    new_map.cells[idx] = replace(new_cell, unrecognized_count=new_unrecog)
                    break
        return new_map
