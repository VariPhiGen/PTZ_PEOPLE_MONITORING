"""
CrossCameraAnalyzer — multi-camera person flow and anomaly analytics.

Provides four analytical capabilities:

build_transit_model(client_id, date_from, date_to)
    Build per camera-pair transit time distributions (mean, σ, p5–p95) from
    30 days of sightings.  Results are cached in Redis for 1 hour.

detect_anomalous_transit(client_id, person_id, from_cam, to_cam, transit_s)
    Flag a single transit as IMPOSSIBLE / SUSPICIOUS / LATE / NORMAL by
    comparing it against the statistical model for that camera pair.

build_flow_matrix(client_id, date, hour)
    Compute camera-to-camera person counts for a given date (+ optional hour).
    Returns a Sankey-ready {nodes, links} structure.

get_occupancy_forecast(client_id, camera_id, dow, days_back)
    Predict expected occupancy per hour for a day-of-week, derived from
    historical sightings.  Returns mean ± 1.5σ confidence bands.

scan_anomalies(client_id, date_from, date_to, camera_id)
    Batch variant: query all transits in a window and return every anomalous
    one annotated with severity, z-score, and expected range.

Notes
──────
• first_seen / last_seen on Sighting are Unix epoch integers (seconds).
• transit_s = next_sighting.first_seen − current_sighting.last_seen
• PostgreSQL PERCENTILE_CONT computes exact percentiles server-side.
• All statistical decisions use a minimum of 5 samples per camera pair.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

log = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
_TRANSIT_CACHE_TTL  = 3_600    # 1 h — transit models change slowly
_FLOW_CACHE_TTL     = 1_800    # 30 min
_FORECAST_CACHE_TTL = 1_800    # 30 min
_MAX_TRANSIT_S      = 3_600    # ignore gaps > 1 hr (separate sessions)
_MIN_SAMPLES        = 5        # minimum data points before building a model
_PHYSICAL_MIN_S     = 10.0     # transit below 10 s is always IMPOSSIBLE

# Severity thresholds (z-score)
_Z_IMPOSSIBLE   = -5.0
_Z_SUSPICIOUS   = -2.5
_Z_LATE         = +3.5

_DOW_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class TransitStats:
    from_camera_id:   str
    to_camera_id:     str
    from_camera_name: str
    to_camera_name:   str
    n:    int       # sample count
    mean: float     # mean transit seconds
    std:  float     # std dev (0 if n < 2)
    p5:   float
    p25:  float
    p50:  float     # median
    p75:  float
    p95:  float
    lo:   float     # min observed
    hi:   float     # max observed

    # ── Derived helpers ────────────────────────────────────────────────────

    def z_score(self, transit_s: float) -> float:
        """Signed z-score; 0 if std is effectively zero."""
        if self.std < 0.5:
            return 0.0
        return (transit_s - self.mean) / self.std

    @property
    def normal_range(self) -> tuple[float, float]:
        """(mean − 2σ, mean + 2σ), lower bound clamped to 0."""
        return (max(0.0, self.mean - 2 * self.std), self.mean + 2 * self.std)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["normal_range_lo"], d["normal_range_hi"] = self.normal_range
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TransitStats":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class AnomalyResult:
    person_id:          str
    person_name:        str
    from_camera_id:     str
    from_camera_name:   str
    to_camera_id:       str
    to_camera_name:     str
    transit_seconds:    float
    timestamp:          int     # epoch of the TO sighting
    severity:           str     # IMPOSSIBLE | SUSPICIOUS | LATE | NORMAL
    z_score:            float
    expected_mean:      float
    expected_std:       float
    expected_range_lo:  float
    expected_range_hi:  float
    reason:             str
    model_samples:      int


@dataclass
class FlowEdge:
    source:         str     # camera_id
    target:         str     # camera_id
    source_name:    str
    target_name:    str
    value:          int     # distinct persons who made this transition
    avg_transit_s:  float


@dataclass
class FlowMatrix:
    date:               str         # "YYYY-MM-DD"
    hour:               int | None  # None = whole day
    nodes:              list[dict[str, str]]
    links:              list[dict[str, Any]]
    total_transitions:  int


@dataclass
class HourBand:
    hour:       int
    mean:       float   # mean distinct persons
    std:        float
    lower:      float   # mean − 1.5σ, clamped ≥ 0
    upper:      float   # mean + 1.5σ
    samples:    int     # number of calendar days with observations


@dataclass
class CameraForecast:
    camera_id:      str
    camera_name:    str
    dow:            int | None
    dow_name:       str | None
    peak_hour:      int
    peak_mean:      float
    days_analysed:  int
    hourly:         list[HourBand] = field(default_factory=list)


# ── Main service ──────────────────────────────────────────────────────────────

class CrossCameraAnalyzer:
    """
    Pure-computation analytics service.  Instantiate per-request (no warmup).

    Parameters
    ──────────
    session_factory    SQLAlchemy async_sessionmaker for the main DB.
    redis              aioredis.Redis client for result caching.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        redis:           Any,
    ) -> None:
        self._factory = session_factory
        self._redis   = redis

    # ── 1. Transit model ──────────────────────────────────────────────────────

    async def build_transit_model(
        self,
        client_id:   str,
        date_from:   int,   # unix epoch (seconds)
        date_to:     int,   # unix epoch (seconds)
        min_samples: int = _MIN_SAMPLES,
    ) -> dict[str, TransitStats]:
        """
        Return a model dict keyed by "from_cam_id:to_cam_id".

        Uses PostgreSQL PERCENTILE_CONT for exact percentile computation.
        Excludes transitions > 1 hour (treat as different sessions).
        Results are cached in Redis for 1 hour.
        """
        cache_key = _cache_key(
            "transit", client_id,
            str(date_from // 86_400),   # day-granularity for key stability
            str(date_to   // 86_400),
            str(min_samples),
        )
        cached = await _redis_get(self._redis, cache_key)
        if cached is not None:
            return {k: TransitStats.from_dict(v) for k, v in cached.items()}

        model: dict[str, TransitStats] = {}

        async with self._factory() as session:
            rows = await session.execute(
                text("""
                    WITH consecutive AS (
                        SELECT
                            s.person_id,
                            s.camera_id                                         AS from_cam,
                            LEAD(s.camera_id)  OVER w                          AS to_cam,
                            (LEAD(s.first_seen) OVER w - s.last_seen)::float   AS transit_s
                        FROM sightings s
                        WHERE s.client_id = CAST(:cid AS uuid)
                          AND s.first_seen  >= :df
                          AND s.first_seen  <= :dt
                        WINDOW w AS (PARTITION BY s.person_id ORDER BY s.first_seen)
                    )
                    SELECT
                        c.from_cam::text,
                        c.to_cam::text,
                        fc.name  AS from_name,
                        tc.name  AS to_name,
                        COUNT(*)::int                                               AS n,
                        AVG(c.transit_s)                                           AS mean_s,
                        COALESCE(STDDEV_SAMP(c.transit_s), 0.0)                   AS std_s,
                        PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY c.transit_s) AS p5,
                        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY c.transit_s) AS p25,
                        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY c.transit_s) AS p50,
                        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY c.transit_s) AS p75,
                        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY c.transit_s) AS p95,
                        MIN(c.transit_s)                                           AS lo,
                        MAX(c.transit_s)                                           AS hi
                    FROM consecutive c
                    JOIN cameras fc ON fc.camera_id = c.from_cam
                    JOIN cameras tc ON tc.camera_id = c.to_cam
                    WHERE c.to_cam    IS NOT NULL
                      AND c.from_cam != c.to_cam
                      AND c.transit_s >  0
                      AND c.transit_s <= :max_t
                    GROUP BY c.from_cam, c.to_cam, fc.name, tc.name
                    HAVING COUNT(*) >= :min_n
                    ORDER BY n DESC
                """),
                {
                    "cid":   client_id,
                    "df":    date_from,
                    "dt":    date_to,
                    "max_t": _MAX_TRANSIT_S,
                    "min_n": min_samples,
                },
            )
            for r in rows.fetchall():
                key = f"{r.from_cam}:{r.to_cam}"
                model[key] = TransitStats(
                    from_camera_id   = r.from_cam,
                    to_camera_id     = r.to_cam,
                    from_camera_name = r.from_name or r.from_cam[:8],
                    to_camera_name   = r.to_name   or r.to_cam[:8],
                    n    = int(r.n),
                    mean = round(float(r.mean_s), 2),
                    std  = round(float(r.std_s),  2),
                    p5   = round(float(r.p5),     2),
                    p25  = round(float(r.p25),    2),
                    p50  = round(float(r.p50),    2),
                    p75  = round(float(r.p75),    2),
                    p95  = round(float(r.p95),    2),
                    lo   = round(float(r.lo),     2),
                    hi   = round(float(r.hi),     2),
                )

        await _redis_set(
            self._redis, cache_key,
            {k: v.to_dict() for k, v in model.items()},
            ttl=_TRANSIT_CACHE_TTL,
        )
        log.info(
            "CrossCameraAnalyzer: transit model built  client=%s  pairs=%d",
            client_id, len(model),
        )
        return model

    # ── 2. Anomaly detection ──────────────────────────────────────────────────

    async def detect_anomalous_transit(
        self,
        client_id:       str,
        person_id:       str,
        from_cam:        str,
        to_cam:          str,
        transit_seconds: float,
        model_window_days: int = 30,
    ) -> AnomalyResult:
        """
        Classify a single transit against the statistical model.

        Severity levels
        ───────────────
        IMPOSSIBLE   z ≤ −5  or  transit < 10 s  (physically impossible)
        SUSPICIOUS   z ≤ −2.5                     (unusually fast)
        LATE         z ≥ +3.5                     (unusually slow)
        NORMAL       otherwise
        """
        now  = _now_epoch()
        dfr  = now - model_window_days * 86_400
        model = await self.build_transit_model(client_id, dfr, now)
        pair_key = f"{from_cam}:{to_cam}"
        stats    = model.get(pair_key)

        # Fetch camera and person names once
        async with self._factory() as session:
            cam_rows = await session.execute(
                text("""
                    SELECT camera_id::text, name
                    FROM cameras
                    WHERE camera_id = CAST(:fc AS uuid) OR camera_id = CAST(:tc AS uuid)
                """),
                {"fc": from_cam, "tc": to_cam},
            )
            cam_names = {r.camera_id: r.name for r in cam_rows.fetchall()}

            p_row = await session.execute(
                text("SELECT name FROM persons WHERE person_id = CAST(:pid AS uuid)"),
                {"pid": person_id},
            )
            person_name = (p_row.scalar() or person_id[:8]) if p_row else person_id[:8]

        from_name = cam_names.get(from_cam, from_cam[:8])
        to_name   = cam_names.get(to_cam,   to_cam[:8])

        if stats is None:
            # No model available — only check physical minimum
            if transit_seconds < _PHYSICAL_MIN_S:
                severity = "IMPOSSIBLE"
                reason   = (
                    f"Transit {transit_seconds:.1f}s < physical minimum "
                    f"{_PHYSICAL_MIN_S:.0f}s (no model for this pair)"
                )
            else:
                severity = "NORMAL"
                reason   = "No transit model for this camera pair yet"
            return AnomalyResult(
                person_id=person_id, person_name=person_name,
                from_camera_id=from_cam, from_camera_name=from_name,
                to_camera_id=to_cam,   to_camera_name=to_name,
                transit_seconds=round(transit_seconds, 2),
                timestamp=now,
                severity=severity, z_score=0.0,
                expected_mean=0.0, expected_std=0.0,
                expected_range_lo=0.0, expected_range_hi=0.0,
                reason=reason, model_samples=0,
            )

        z   = stats.z_score(transit_seconds)
        lo, hi = stats.normal_range

        # Physical impossibility check takes precedence over z-score
        if transit_seconds < _PHYSICAL_MIN_S or z <= _Z_IMPOSSIBLE:
            severity = "IMPOSSIBLE"
            reason   = (
                f"Transit {transit_seconds:.1f}s — z={z:+.1f} "
                f"(model mean {stats.mean:.0f}s ±{stats.std:.0f}s)"
            )
        elif z <= _Z_SUSPICIOUS:
            severity = "SUSPICIOUS"
            reason   = (
                f"Unusually fast transit {transit_seconds:.1f}s — z={z:+.1f} "
                f"(expected {stats.p25:.0f}–{stats.p75:.0f}s)"
            )
        elif z >= _Z_LATE:
            severity = "LATE"
            reason   = (
                f"Unusually slow transit {transit_seconds:.1f}s — z={z:+.1f} "
                f"(p95={stats.p95:.0f}s)"
            )
        else:
            severity = "NORMAL"
            reason   = (
                f"Transit {transit_seconds:.1f}s within expected range "
                f"({lo:.0f}–{hi:.0f}s)"
            )

        return AnomalyResult(
            person_id=person_id, person_name=person_name,
            from_camera_id=from_cam, from_camera_name=from_name,
            to_camera_id=to_cam,   to_camera_name=to_name,
            transit_seconds=round(transit_seconds, 2),
            timestamp=now,
            severity=severity,
            z_score=round(z, 2),
            expected_mean=stats.mean,
            expected_std=stats.std,
            expected_range_lo=round(lo, 1),
            expected_range_hi=round(hi, 1),
            reason=reason,
            model_samples=stats.n,
        )

    # ── 3. Flow matrix ────────────────────────────────────────────────────────

    async def build_flow_matrix(
        self,
        client_id: str,
        date:      str,         # "YYYY-MM-DD"
        hour:      int | None,  # 0–23; None = whole day
    ) -> FlowMatrix:
        """
        Build a Sankey-ready flow matrix for a given date (and optional hour).

        Counts distinct persons who transitioned between each camera pair within
        the time window.  Returns nodes (with building/floor metadata) and
        directional links.
        """
        cache_key = _cache_key("flow", client_id, date, str(hour or "all"))
        cached = await _redis_get(self._redis, cache_key)
        if cached is not None:
            return FlowMatrix(**cached)

        # Convert date string to epoch window
        try:
            day_start = int(
                datetime.strptime(date, "%Y-%m-%d")
                .replace(tzinfo=timezone.utc)
                .timestamp()
            )
        except ValueError:
            day_start = _now_epoch() - (_now_epoch() % 86_400)

        if hour is not None:
            epoch_from = day_start + hour * 3_600
            epoch_to   = epoch_from + 3_600
        else:
            epoch_from = day_start
            epoch_to   = day_start + 86_400

        params: dict[str, Any] = {
            "cid": client_id,
            "df":  epoch_from,
            "dt":  epoch_to,
        }

        async with self._factory() as session:
            # Edges: distinct persons per camera-pair transition
            edge_rows = await session.execute(
                text("""
                    WITH consecutive AS (
                        SELECT
                            s.person_id,
                            s.camera_id                                       AS from_cam,
                            s.last_seen                                       AS from_last,
                            LEAD(s.camera_id)   OVER w                       AS to_cam,
                            LEAD(s.first_seen)  OVER w                       AS to_first
                        FROM sightings s
                        WHERE s.client_id  = CAST(:cid AS uuid)
                          AND s.first_seen >= :df
                          AND s.first_seen <  :dt
                        WINDOW w AS (PARTITION BY s.person_id ORDER BY s.first_seen)
                    )
                    SELECT
                        c.from_cam::text                        AS source,
                        c.to_cam::text                          AS target,
                        fc.name                                 AS source_name,
                        tc.name                                 AS target_name,
                        COUNT(DISTINCT c.person_id)::int        AS value,
                        ROUND(
                          AVG(c.to_first - c.from_last)::numeric, 1
                        )::float                               AS avg_transit_s
                    FROM consecutive c
                    JOIN cameras fc ON fc.camera_id = c.from_cam
                    JOIN cameras tc ON tc.camera_id = c.to_cam
                    WHERE c.to_cam    IS NOT NULL
                      AND c.from_cam != c.to_cam
                      AND (c.to_first - c.from_last) > 0
                      AND (c.to_first - c.from_last) <= :max_t
                    GROUP BY c.from_cam, c.to_cam, fc.name, tc.name
                    ORDER BY value DESC
                    LIMIT 500
                """),
                {**params, "max_t": _MAX_TRANSIT_S},
            )
            edges = edge_rows.fetchall()

            # Node metadata: all cameras seen in the window
            node_rows = await session.execute(
                text("""
                    SELECT DISTINCT
                        s.camera_id::text AS id,
                        c.name,
                        COALESCE(c.building, '') AS building,
                        COALESCE(c.floor,    '') AS floor
                    FROM sightings s
                    JOIN cameras c ON c.camera_id = s.camera_id
                    WHERE s.client_id  = CAST(:cid AS uuid)
                      AND s.first_seen >= :df
                      AND s.first_seen <  :dt
                """),
                params,
            )
            nodes = [
                {"id": r.id, "name": r.name, "building": r.building, "floor": r.floor}
                for r in node_rows.fetchall()
            ]

        links = [
            {
                "source":       e.source,
                "target":       e.target,
                "source_name":  e.source_name,
                "target_name":  e.target_name,
                "value":        int(e.value),
                "avg_transit_s": float(e.avg_transit_s or 0),
            }
            for e in edges
        ]

        result = FlowMatrix(
            date=date, hour=hour,
            nodes=nodes,
            links=links,
            total_transitions=sum(k["value"] for k in links),
        )
        await _redis_set(self._redis, cache_key, asdict(result), ttl=_FLOW_CACHE_TTL)
        log.info(
            "CrossCameraAnalyzer: flow matrix built  client=%s  date=%s  hour=%s  "
            "nodes=%d  edges=%d",
            client_id, date, hour, len(nodes), len(links),
        )
        return result

    # ── 4. Occupancy forecast ─────────────────────────────────────────────────

    async def get_occupancy_forecast(
        self,
        client_id: str,
        camera_id: str | None = None,
        dow:       int | None = None,   # 0=Monday … 6=Sunday; None = all days
        days_back: int        = 28,
    ) -> list[CameraForecast]:
        """
        Predict occupancy per hour using historical sightings.

        For each camera, groups observed distinct-person counts by (dow, hour,
        date) and derives mean ± 1.5σ confidence bands.  When dow is supplied
        only that day-of-week is included (e.g., all Mondays in the window).

        Returns one CameraForecast per camera.
        """
        cache_key = _cache_key(
            "forecast", client_id,
            camera_id or "all",
            str(dow if dow is not None else "any"),
            str(days_back),
        )
        cached = await _redis_get(self._redis, cache_key)
        if cached is not None:
            return [
                CameraForecast(
                    **{k: v for k, v in cam.items() if k != "hourly"},
                    hourly=[HourBand(**h) for h in cam.get("hourly", [])],
                )
                for cam in cached
            ]

        cutoff = _now_epoch() - days_back * 86_400
        params: dict[str, Any] = {"cid": client_id, "cutoff": cutoff}
        cam_filter = ""
        if camera_id:
            cam_filter = "AND s.camera_id = CAST(:cam_id AS uuid)"
            params["cam_id"] = camera_id

        # PostgreSQL ISODOW: 1=Monday … 7=Sunday → convert to 0-based
        dow_filter = ""
        if dow is not None:
            dow_filter = "AND (EXTRACT(ISODOW FROM to_timestamp(s.first_seen))::int - 1) = :dow"
            params["dow"] = dow

        async with self._factory() as session:
            rows = await session.execute(
                text(f"""
                    SELECT
                        s.camera_id::text                                           AS cam_id,
                        c.name                                                      AS cam_name,
                        (EXTRACT(ISODOW FROM to_timestamp(s.first_seen))::int - 1) AS dow,
                        EXTRACT(HOUR  FROM to_timestamp(s.first_seen))::int         AS hour,
                        DATE(to_timestamp(s.first_seen))                            AS cal_date,
                        COUNT(DISTINCT s.person_id)::int                            AS occupancy
                    FROM sightings s
                    JOIN cameras c ON c.camera_id = s.camera_id
                    WHERE s.client_id  = CAST(:cid AS uuid)
                      AND s.first_seen >= :cutoff
                      {cam_filter}
                      {dow_filter}
                    GROUP BY s.camera_id, c.name, dow, hour, cal_date
                    ORDER BY s.camera_id, dow, hour, cal_date
                """),
                params,
            )
            raw = rows.fetchall()

        # Group in Python: {cam_id → {(dow, hour) → [occupancy per day]}}
        buckets: dict[str, dict] = {}
        cam_names: dict[str, str] = {}
        for r in raw:
            cid_key = r.cam_id
            cam_names[cid_key] = r.cam_name
            bucket = buckets.setdefault(cid_key, {})
            dh_key = (int(r.dow), int(r.hour))
            bucket.setdefault(dh_key, []).append(int(r.occupancy))

        forecasts: list[CameraForecast] = []
        for cid_key, bucket in buckets.items():
            hours: list[HourBand] = []
            peak_mean = 0.0
            peak_hour = 0
            all_dates: set[str] = set()

            for h in range(24):
                # Collect all (dow, h) entries regardless of which dow
                all_obs: list[int] = []
                if dow is not None:
                    all_obs = bucket.get((dow, h), [])
                else:
                    for d in range(7):
                        all_obs.extend(bucket.get((d, h), []))

                if not all_obs:
                    hours.append(HourBand(hour=h, mean=0.0, std=0.0, lower=0.0, upper=0.0, samples=0))
                    continue

                mn  = _mean(all_obs)
                sd  = _std(all_obs)
                band = HourBand(
                    hour=h,
                    mean=round(mn, 2),
                    std=round(sd, 2),
                    lower=round(max(0.0, mn - 1.5 * sd), 2),
                    upper=round(mn + 1.5 * sd, 2),
                    samples=len(all_obs),
                )
                hours.append(band)
                if mn > peak_mean:
                    peak_mean = mn
                    peak_hour = h

            days_analysed = len({r.cal_date for r in raw if r.cam_id == cid_key})
            forecasts.append(CameraForecast(
                camera_id     = cid_key,
                camera_name   = cam_names[cid_key],
                dow           = dow,
                dow_name      = _DOW_NAMES[dow] if dow is not None else None,
                peak_hour     = peak_hour,
                peak_mean     = round(peak_mean, 2),
                days_analysed = days_analysed,
                hourly        = hours,
            ))

        await _redis_set(
            self._redis, cache_key,
            [asdict(f) for f in forecasts],
            ttl=_FORECAST_CACHE_TTL,
        )
        return forecasts

    # ── 5. Batch anomaly scan ─────────────────────────────────────────────────

    async def scan_anomalies(
        self,
        client_id:  str,
        date_from:  int,
        date_to:    int,
        camera_id:  str | None = None,
        model_window_days: int = 30,
        limit:      int = 200,
    ) -> list[AnomalyResult]:
        """
        Scan all transits in [date_from, date_to] and return anomalous ones.

        Loads the transit model once (cached), then classifies each consecutive
        sighting pair in bulk.  Only IMPOSSIBLE / SUSPICIOUS / LATE events are
        returned.
        """
        # Load transit model from a 30-day lookback window ending at date_to
        model_from = date_to - model_window_days * 86_400
        model = await self.build_transit_model(client_id, model_from, date_to)

        cam_filter = ""
        params: dict[str, Any] = {
            "cid": client_id,
            "df":  date_from,
            "dt":  date_to,
        }
        if camera_id:
            cam_filter = "AND (c.from_cam = CAST(:cam_id AS uuid) OR c.to_cam = CAST(:cam_id AS uuid))"
            params["cam_id"] = camera_id

        async with self._factory() as session:
            rows = await session.execute(
                text(f"""
                    WITH consecutive AS (
                        SELECT
                            s.person_id,
                            p.name                                           AS person_name,
                            s.camera_id                                      AS from_cam,
                            fc.name                                          AS from_name,
                            LEAD(s.camera_id) OVER w                        AS to_cam,
                            LEAD(tc.name)     OVER w                        AS to_name,
                            s.last_seen                                      AS from_last,
                            LEAD(s.first_seen) OVER w                       AS to_first,
                            (LEAD(s.first_seen) OVER w - s.last_seen)::float AS transit_s
                        FROM sightings s
                        JOIN cameras  fc  ON fc.camera_id  = s.camera_id
                        LEFT JOIN LATERAL (
                            SELECT c2.name FROM cameras c2
                            WHERE c2.camera_id = s.camera_id
                        ) tc ON true
                        JOIN persons  p   ON p.person_id   = s.person_id
                        WHERE s.client_id  = CAST(:cid AS uuid)
                          AND s.first_seen >= :df
                          AND s.first_seen <  :dt
                        WINDOW w AS (PARTITION BY s.person_id ORDER BY s.first_seen)
                    )
                    SELECT
                        c.person_id::text,
                        c.person_name,
                        c.from_cam::text,
                        c.from_name,
                        c.to_cam::text,
                        c.to_name,
                        c.transit_s,
                        c.to_first AS ts
                    FROM consecutive c
                    WHERE c.to_cam   IS NOT NULL
                      AND c.from_cam != c.to_cam
                      AND c.transit_s > 0
                      AND c.transit_s <= :max_t
                    {cam_filter}
                    ORDER BY c.to_first DESC
                    LIMIT :lim
                """),
                {**params, "max_t": _MAX_TRANSIT_S, "lim": limit * 5},
            )
            transit_rows = rows.fetchall()

        anomalies: list[AnomalyResult] = []
        for r in transit_rows:
            pair_key = f"{r.from_cam}:{r.to_cam}"
            stats    = model.get(pair_key)
            transit  = float(r.transit_s)

            if stats is None:
                if transit < _PHYSICAL_MIN_S:
                    z, severity = 0.0, "IMPOSSIBLE"
                    reason = f"Transit {transit:.1f}s < {_PHYSICAL_MIN_S:.0f}s (no model)"
                else:
                    continue
            else:
                z = stats.z_score(transit)
                lo, hi = stats.normal_range
                if transit < _PHYSICAL_MIN_S or z <= _Z_IMPOSSIBLE:
                    severity = "IMPOSSIBLE"
                    reason = (
                        f"Transit {transit:.1f}s, z={z:+.1f} "
                        f"(mean {stats.mean:.0f}s ±{stats.std:.0f}s)"
                    )
                elif z <= _Z_SUSPICIOUS:
                    severity = "SUSPICIOUS"
                    reason = f"Fast transit {transit:.1f}s, z={z:+.1f} (expected {stats.p25:.0f}–{stats.p75:.0f}s)"
                elif z >= _Z_LATE:
                    severity = "LATE"
                    reason = f"Slow transit {transit:.1f}s, z={z:+.1f} (p95={stats.p95:.0f}s)"
                else:
                    continue  # normal — skip

            lo_r, hi_r = (stats.normal_range if stats else (0.0, 0.0))
            anomalies.append(AnomalyResult(
                person_id=r.person_id, person_name=r.person_name or r.person_id[:8],
                from_camera_id=r.from_cam, from_camera_name=r.from_name or "",
                to_camera_id=r.to_cam,   to_camera_name=r.to_name   or "",
                transit_seconds=round(transit, 2),
                timestamp=int(r.ts),
                severity=severity, z_score=round(z, 2),
                expected_mean=stats.mean if stats else 0.0,
                expected_std=stats.std  if stats else 0.0,
                expected_range_lo=round(lo_r, 1),
                expected_range_hi=round(hi_r, 1),
                reason=reason,
                model_samples=stats.n if stats else 0,
            ))
            if len(anomalies) >= limit:
                break

        log.info(
            "CrossCameraAnalyzer: anomaly scan  client=%s  transits_checked=%d  "
            "anomalies=%d",
            client_id, len(transit_rows), len(anomalies),
        )
        return anomalies


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now_epoch() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def _cache_key(*parts: str) -> str:
    digest = hashlib.sha256(":".join(parts).encode()).hexdigest()[:16]
    return f"acas:cc:{parts[0]}:{digest}"


async def _redis_get(redis: Any, key: str) -> Any:
    try:
        raw = await redis.get(key)
        return json.loads(raw) if raw else None
    except Exception:
        return None


async def _redis_set(redis: Any, key: str, value: Any, ttl: int) -> None:
    try:
        await redis.setex(key, ttl, json.dumps(value, default=str))
    except Exception:
        pass


def _mean(xs: list[float | int]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: list[float | int]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    variance = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(variance)
