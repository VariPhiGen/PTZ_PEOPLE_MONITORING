"""
PathPlanner — optimal PTZ scan-path ordering over ScanCells.

Algorithm:
  1. Build an N×N travel-time distance matrix.
       travel(i→j) = max(|Δpan|/pan_spd, |Δtilt|/tilt_spd) + |Δzoom|/zoom_spd + settle_s
       (pan and tilt move simultaneously; zoom is additive per the spec)

  2. Nearest-Neighbour (NN) TSP heuristic with a recognition-weighted edge cost:
       effective_cost(i→j) = travel(i,j) − 0.3 × unrecognized_count[j]
     Cells with more unidentified faces are preferred at each greedy step.
     Faculty cell is forced to be the first visited.

  3. 2-opt local search on pure travel time.
     (Recognition rewards cancel under segment reversal so only distances matter.)
     Faculty cell at position 0 is pinned and never displaced.

  4. Cycle-time and per-cell dwell estimation.

replan_path() re-runs the NN + 2-opt from the current head of the tour,
retaining unrecognized-count updates that occurred mid-cycle.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

import numpy as np

from app.services.onvif_controller import PTZLimits, PTZPosition
from app.services.zone_mapper import ScanCell, ScanMap

logger = logging.getLogger(__name__)

# ── Tunable constants ──────────────────────────────────────────────────────────

_DEFAULT_SETTLE_S   = 0.35   # seconds for mechanical vibration to settle after a move
_DWELL_PER_FACE_S   = 0.40   # seconds of dwell time per expected face in a cell
_MIN_DWELL_S        = 0.50   # minimum dwell at any cell (shutter + exposure)
_2OPT_MAX_ITERS     = 8      # maximum 2-opt passes (caps runtime for large N)


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class PlanResult:
    """Output of plan_path()."""
    ordered_cell_ids:      list[str]       # cell_ids in visit order
    estimated_total_s:     float           # full cycle time (travel + dwell)
    travel_matrix:         np.ndarray      # [N, N] float64 travel times in seconds
    cell_order:            list[int]       # indices into input cells list
    travel_breakdown_s:    list[float]     # travel time for each leg
    dwell_breakdown_s:     list[float]     # dwell time at each cell


# ── Travel-time helper ─────────────────────────────────────────────────────────

def _travel(
    from_pos: PTZPosition,
    to_pos: PTZPosition,
    limits: PTZLimits,
    settle_s: float,
) -> float:
    """
    Estimate seconds to move from `from_pos` to `to_pos`.

    Pan and tilt move simultaneously → dominant axis determines time.
    Zoom moves concurrently but its time is added (per spec) because
    optical stabilisation is zoom-dependent.
        t = max(|Δpan|/pan_spd, |Δtilt|/tilt_spd) + |Δzoom|/zoom_spd + settle_s
    """
    pan_range  = max(limits.pan_max  - limits.pan_min,  1e-9)
    tilt_range = max(limits.tilt_max - limits.tilt_min, 1e-9)
    zoom_range = max(limits.zoom_max - limits.zoom_min, 1e-9)

    def _t(delta: float, speed: float, rng: float) -> float:
        if speed <= 0:
            return 0.0
        return abs(delta) / (speed * rng)

    t_pan  = _t(to_pos.pan  - from_pos.pan,  limits.pan_speed,  pan_range)
    t_tilt = _t(to_pos.tilt - from_pos.tilt, limits.tilt_speed, tilt_range)
    t_zoom = _t(to_pos.zoom - from_pos.zoom, limits.zoom_speed, zoom_range)
    return max(t_pan, t_tilt) + t_zoom + settle_s


def _dwell(cell: ScanCell) -> float:
    return max(_MIN_DWELL_S, cell.expected_faces * _DWELL_PER_FACE_S)


def _cell_ptz(cell: ScanCell) -> PTZPosition:
    return PTZPosition(
        pan=cell.center_pan,
        tilt=cell.center_tilt,
        zoom=cell.required_zoom,
    )


def _heading_cost(
    prev_heading: tuple[float, float] | None,
    from_cell: ScanCell,
    to_cell: ScanCell,
    weight_s: float = 0.08,
) -> float:
    """
    Penalty (in equivalent seconds of travel) for a direction change at `from_cell`
    when coming from the direction given by `prev_heading` and continuing to `to_cell`.

    Formula:  weight_s × (1 − cos θ) / 2
      θ = 0°   (straight ahead)  → 0.0 s penalty
      θ = 90°  (right-angle turn) → weight_s / 2
      θ = 180° (U-turn reversal)  → weight_s

    This biases the nearest-neighbour heuristic to prefer sweeping trajectories
    (row-by-row or column-by-column) over zigzag paths that incur sharp reversals,
    mechanical stress, and image jitter at zone boundaries.

    The weight is deliberately modest (0.08 s) so it does not override significant
    travel-time savings; it only acts as a tiebreaker between near-equal options.
    """
    if prev_heading is None:
        return 0.0

    mag_prev = math.hypot(prev_heading[0], prev_heading[1])
    if mag_prev < 1e-9:
        return 0.0

    new_dir = (
        to_cell.center_pan  - from_cell.center_pan,
        to_cell.center_tilt - from_cell.center_tilt,
    )
    mag_new = math.hypot(new_dir[0], new_dir[1])
    if mag_new < 1e-9:
        return 0.0

    cos_theta = (
        prev_heading[0] * new_dir[0] + prev_heading[1] * new_dir[1]
    ) / (mag_prev * mag_new)
    cos_theta = max(-1.0, min(1.0, cos_theta))

    return weight_s * (1.0 - cos_theta) / 2.0


# ── PathPlanner ────────────────────────────────────────────────────────────────

class PathPlanner:
    """
    Computes an efficient ordered scan path over a list of ScanCells.

    Parameters:
        limits    — PTZLimits from the connected ONVIFController.
        settle_s  — extra settle time added to every move (default 0.35 s).
    """

    def __init__(
        self,
        limits: PTZLimits,
        settle_s: float = _DEFAULT_SETTLE_S,
    ) -> None:
        self._limits   = limits
        self._settle_s = settle_s

    # ── Public API ─────────────────────────────────────────────────────────────

    def plan_path(
        self,
        cells: list[ScanCell],
        faculty_cell: ScanCell | None,
        current_ptz: PTZPosition,
        speeds: PTZLimits | None = None,
    ) -> PlanResult:
        """
        Compute an optimised scan path over `cells`.

        Parameters:
            cells        — list of ScanCells from ZoneMapper.
            faculty_cell — if provided, this cell is forced to position 0.
            current_ptz  — camera's current PTZ state (defines tour start).
            speeds       — optional PTZLimits override (uses self._limits by default).

        Returns a PlanResult with ordered cell_ids, estimated duration, and
        the raw travel-time matrix.
        """
        if not cells:
            return PlanResult(
                ordered_cell_ids=[], estimated_total_s=0.0,
                travel_matrix=np.zeros((0, 0)),
                cell_order=[], travel_breakdown_s=[], dwell_breakdown_s=[],
            )

        lim = speeds or self._limits
        n   = len(cells)

        # ── Step 1: N×N travel-time matrix ────────────────────────────────────
        # Row 0 is the "current_ptz" start position (virtual origin node).
        # Matrix indices 0..n-1 correspond to cells[0..n-1].
        D = self._build_distance_matrix(cells, lim)

        # Travel times from current_ptz to each cell (origin row)
        start_costs = np.array([
            _travel(current_ptz, _cell_ptz(cells[j]), lim, self._settle_s)
            for j in range(n)
        ])

        # ── Step 2: Nearest-Neighbour TSP with recognition weighting ──────────
        tour = self._nearest_neighbour(cells, faculty_cell, start_costs, D)

        # ── Step 3: 2-opt improvement on pure travel distances ────────────────
        # Pin faculty (position 0) — never let it leave first place.
        pin = 1 if faculty_cell is not None else 0
        tour = self._two_opt(tour, D, pin_prefix=pin)

        # ── Assemble result ───────────────────────────────────────────────────
        ordered_ids  = [cells[i].cell_id for i in tour]
        travel_legs  = [float(start_costs[tour[0]])]
        dwell_legs   = [_dwell(cells[tour[0]])]
        total_s      = travel_legs[0] + dwell_legs[0]

        for step in range(1, n):
            t = float(D[tour[step - 1]][tour[step]])
            d = _dwell(cells[tour[step]])
            travel_legs.append(t)
            dwell_legs.append(d)
            total_s += t + d

        return PlanResult(
            ordered_cell_ids=ordered_ids,
            estimated_total_s=total_s,
            travel_matrix=D,
            cell_order=tour,
            travel_breakdown_s=travel_legs,
            dwell_breakdown_s=dwell_legs,
        )

    def replan_path(
        self,
        scan_map: ScanMap,
        current_idx: int,
        current_ptz: PTZPosition,
        speeds: PTZLimits | None = None,
    ) -> PlanResult:
        """
        Re-plan the remaining scan path from `current_idx` onward.

        Useful when face-recognition results mid-cycle update `unrecognized_count`
        on future cells.  Cells already visited (indices < current_idx in the
        original plan) are excluded from re-planning.

        Parameters:
            scan_map    — the live ScanMap (cells carry updated unrecognized_count).
            current_idx — index in the original tour that has just been serviced.
            current_ptz — camera's current PTZ state.
        """
        remaining = scan_map.cells[current_idx:]
        if len(remaining) <= 2:
            # Not worth re-planning; return as-is.
            return PlanResult(
                ordered_cell_ids=[c.cell_id for c in remaining],
                estimated_total_s=self.estimate_cycle_time(
                    [c.cell_id for c in remaining], scan_map.cells,
                    {c.cell_id: i for i, c in enumerate(scan_map.cells)},
                    speeds or self._limits,
                    current_ptz,
                ).estimated_total_s,
                travel_matrix=np.zeros((len(remaining), len(remaining))),
                cell_order=list(range(len(remaining))),
                travel_breakdown_s=[],
                dwell_breakdown_s=[_dwell(c) for c in remaining],
            )

        # Faculty cell among remaining (if any)
        faculty_cell = None
        if scan_map.faculty_cell_id:
            for c in remaining:
                if c.cell_id == scan_map.faculty_cell_id:
                    faculty_cell = c
                    break

        return self.plan_path(remaining, faculty_cell, current_ptz, speeds)

    def estimate_cycle_time(
        self,
        path: list[str],
        all_cells: list[ScanCell],
        cell_id_to_idx: dict[str, int],
        speeds: PTZLimits,
        start_ptz: PTZPosition,
    ) -> PlanResult:
        """
        Compute the estimated total cycle time for a given ordered path.

        Parameters:
            path           — ordered list of cell_ids.
            all_cells      — all ScanCells (including unlisted ones).
            cell_id_to_idx — mapping cell_id → index in all_cells.
            speeds         — PTZLimits to use for travel-time calculation.
            start_ptz      — camera position at the start of the cycle.

        Returns a PlanResult populated with travel/dwell breakdowns.
        """
        cells_in_path = [all_cells[cell_id_to_idx[cid]] for cid in path if cid in cell_id_to_idx]
        if not cells_in_path:
            return PlanResult([], 0.0, np.zeros((0, 0)), [], [], [])

        travel_legs: list[float] = []
        dwell_legs:  list[float] = []

        prev_ptz = start_ptz
        for cell in cells_in_path:
            t = _travel(prev_ptz, _cell_ptz(cell), speeds, self._settle_s)
            d = _dwell(cell)
            travel_legs.append(t)
            dwell_legs.append(d)
            prev_ptz = _cell_ptz(cell)

        total_s = sum(travel_legs) + sum(dwell_legs)
        return PlanResult(
            ordered_cell_ids=path,
            estimated_total_s=total_s,
            travel_matrix=np.zeros((len(cells_in_path), len(cells_in_path))),
            cell_order=list(range(len(cells_in_path))),
            travel_breakdown_s=travel_legs,
            dwell_breakdown_s=dwell_legs,
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _build_distance_matrix(
        self,
        cells: list[ScanCell],
        lim: PTZLimits,
    ) -> np.ndarray:
        """Build a symmetric [N×N] matrix of cell-to-cell travel times."""
        n = len(cells)
        D = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                t = _travel(_cell_ptz(cells[i]), _cell_ptz(cells[j]), lim, self._settle_s)
                D[i][j] = t
                D[j][i] = t
        return D

    def _nearest_neighbour(
        self,
        cells: list[ScanCell],
        faculty_cell: ScanCell | None,
        start_costs: np.ndarray,
        D: np.ndarray,
    ) -> list[int]:
        """
        Greedy Nearest-Neighbour TSP with recognition weighting and heading
        continuity.

        Edge cost from current node i to candidate j:
            effective_cost(i→j) = D[i][j]
                                   − 0.3 × unrecognized_count[j]
                                   + heading_cost(prev→i→j)

        The heading cost penalises sharp direction reversals so the tour tends
        to sweep in a consistent direction (e.g. left-to-right row by row)
        rather than zigzagging.  The penalty weight is small enough that it
        never overrides a large travel-time saving; it acts as a tiebreaker.

        Faculty cell is inserted at position 0 unconditionally.
        """
        n = len(cells)
        faculty_idx = next(
            (i for i, c in enumerate(cells) if faculty_cell and c.cell_id == faculty_cell.cell_id),
            None,
        )

        unvisited = set(range(n))
        tour: list[int] = []

        # Force faculty first
        if faculty_idx is not None:
            tour.append(faculty_idx)
            unvisited.discard(faculty_idx)

        # First step from start_ptz; cost already incorporates travel from camera origin
        if not tour:
            # No faculty — pick lowest effective cost from start (no heading yet)
            best_j = min(
                unvisited,
                key=lambda j: start_costs[j] - 0.3 * cells[j].unrecognized_count,
            )
            tour.append(best_j)
            unvisited.discard(best_j)

        # Main NN loop — track incoming heading for direction-change penalty
        while unvisited:
            current = tour[-1]

            # Incoming direction at `current` node
            if len(tour) >= 2:
                prev = tour[-2]
                prev_heading: tuple[float, float] | None = (
                    cells[current].center_pan  - cells[prev].center_pan,
                    cells[current].center_tilt - cells[prev].center_tilt,
                )
            else:
                prev_heading = None

            best_j = min(
                unvisited,
                key=lambda j: (
                    D[current][j]
                    - 0.3 * cells[j].unrecognized_count
                    + _heading_cost(prev_heading, cells[current], cells[j])
                ),
            )
            tour.append(best_j)
            unvisited.discard(best_j)

        return tour

    def _two_opt(
        self,
        tour: list[int],
        D: np.ndarray,
        pin_prefix: int = 0,
    ) -> list[int]:
        """
        Iterative 2-opt local search to reduce total travel distance.

        `pin_prefix` positions at the front of the tour are locked and
        never included in a reversal (used to protect the faculty cell).

        The cost used is pure travel time (recognition rewards cancel under
        segment reversal; the weighting only affects the NN initialisation).
        """
        n = len(tour)
        if n < 4:
            return tour

        improved = True
        iteration = 0
        best_tour = list(tour)

        while improved and iteration < _2OPT_MAX_ITERS:
            improved = False
            iteration += 1
            for i in range(pin_prefix, n - 2):
                for j in range(i + 2, n):
                    # Skip the wrap-around edge that would unpin the prefix
                    if j == n - 1 and i == pin_prefix:
                        continue
                    a, b = best_tour[i], best_tour[i + 1]
                    c, d = best_tour[j], best_tour[(j + 1) % n]
                    old_cost = D[a][b] + D[c][d]
                    new_cost = D[a][c] + D[b][d]
                    if new_cost < old_cost - 1e-9:
                        # Reverse segment [i+1 .. j]
                        best_tour[i + 1 : j + 1] = best_tour[i + 1 : j + 1][::-1]
                        improved = True

        if iteration > 1:
            logger.debug("2-opt: %d iteration(s), tour length %d", iteration - 1, n)
        return best_tour
