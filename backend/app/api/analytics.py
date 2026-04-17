"""
Analytics API — attendance trends, recognition accuracy, system health,
camera uptime, faculty/student reports, flow matrix, occupancy forecast,
cross-camera transit models, and anomaly detection.

All routes are read-only.  Required permissions:
  attendance:read       — attendance-related endpoints
  cameras:read          — camera uptime, occupancy, cross-camera analytics
  system:admin          — system health (CLIENT_ADMIN sees subset)
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from sqlalchemy import func, select, text

from app.api._shared import now_epoch, resolve_client_id
from app.deps import CurrentUser, DBSession
from app.middleware.auth import require_permission
from app.models.attendance_records import AttendanceRecord
from app.models.cameras import Camera
from app.models.persons import Person
from app.models.sessions import Session, SyncStatus
from app.models.sightings import Sighting
from app.services.cross_camera import CrossCameraAnalyzer

router = APIRouter(prefix="/api/analytics", tags=["analytics"])

_ATT  = require_permission("attendance:read")
_CAM  = require_permission("cameras:read")
_SYS  = require_permission("system:admin")


# ── Attendance trends ─────────────────────────────────────────────────────────

@router.get("/attendance-trends", dependencies=[_ATT])
async def attendance_trends(
    request:    Request,
    db:         DBSession,
    date_from:  int | None = Query(None, description="epoch"),
    date_to:    int | None = Query(None, description="epoch"),
    camera_id:  str | None = Query(None),
    granularity: str       = Query("day", pattern="^(hour|day|week)$"),
) -> dict:
    """
    Per-granularity breakdown of P/L/EE/A/ND counts.
    Groups by UTC day (or hour/week) using PostgreSQL date_trunc.
    """
    cid = resolve_client_id(request, require=False)
    trunc_map = {"hour": "hour", "day": "day", "week": "week"}
    trunc = trunc_map[granularity]

    ar_filters = "AND ar.client_id = (:cid)::uuid" if cid else ""
    s_filters  = "AND s.client_id  = (:cid)::uuid" if cid else ""
    params: dict[str, Any] = {"trunc": trunc}
    if cid:
        params["cid"] = cid
    if date_from:
        ar_filters += " AND ar.first_seen >= :df"
        s_filters  += " AND s.actual_start  >= :df"
        params["df"] = date_from
    if date_to:
        ar_filters += " AND ar.first_seen <= :dt"
        s_filters  += " AND s.actual_start  <= :dt"
        params["dt"] = date_to
    if camera_id:
        ar_filters += " AND s.camera_id = (:cam)::uuid"
        s_filters  += " AND s.camera_id = (:cam)::uuid"
        params["cam"] = camera_id

    rows = await db.execute(
        text(f"""
            WITH
            session_enrolled AS (
                -- Enrolled headcount per bucket.
                -- For auto-start sessions (same population, many sessions/day) we use
                -- MAX so 383 sessions × 5 persons → 5, not 1915.
                -- For proper course sessions each session has a distinct enrolled_count
                -- so MAX per day is still a reasonable ceiling.
                SELECT
                    date_trunc(:trunc, to_timestamp(s.actual_start)) AS bucket,
                    MAX(s.enrolled_count) AS enrolled
                FROM sessions s
                WHERE s.actual_start IS NOT NULL
                  AND s.enrolled_count > 0
                  {s_filters}
                GROUP BY bucket
            ),
            best_per_person AS (
                -- For each (bucket, person, course) keep only the best status so
                -- one student in two sessions of the same course counts once.
                SELECT DISTINCT ON (bucket, ar.person_id, s.course_name)
                    date_trunc(:trunc, to_timestamp(ar.first_seen)) AS bucket,
                    ar.person_id,
                    ar.status
                FROM attendance_records ar
                JOIN sessions s ON s.session_id = ar.session_id
                WHERE ar.first_seen IS NOT NULL
                  AND ar.status != 'ND'
                  {ar_filters}
                ORDER BY
                    bucket,
                    ar.person_id,
                    s.course_name,
                    CASE ar.status
                        WHEN 'P'  THEN 1
                        WHEN 'L'  THEN 2
                        WHEN 'EE' THEN 3
                        WHEN 'A'  THEN 4
                        ELSE 5
                    END
            )
            SELECT
                bpp.bucket,
                bpp.status,
                COUNT(*)           AS cnt,
                COALESCE(se.enrolled, 0) AS enrolled
            FROM best_per_person bpp
            LEFT JOIN session_enrolled se ON se.bucket = bpp.bucket
            GROUP BY bpp.bucket, bpp.status, se.enrolled
            ORDER BY bpp.bucket, bpp.status
        """),
        params,
    )
    # Reshape into [{bucket, P, L, EE, A, enrolled}, ...]
    buckets: dict[str, dict] = {}
    for r in rows.fetchall():
        key = str(r.bucket)
        if key not in buckets:
            buckets[key] = {"bucket": key, "P": 0, "L": 0, "EE": 0, "A": 0, "enrolled": int(r.enrolled)}
        buckets[key][r.status] = int(r.cnt)
    return {
        "granularity": granularity,
        "total_buckets": len(buckets),
        "data": list(buckets.values()),
    }


# ── Recognition accuracy ──────────────────────────────────────────────────────

@router.get("/recognition-accuracy", dependencies=[_ATT])
async def recognition_accuracy(
    request:   Request,
    db:        DBSession,
    days:      int        = Query(14, ge=1, le=90),
    date_from: int | None = Query(None),
    date_to:   int | None = Query(None),
) -> dict:
    """
    Per-day recognition accuracy + overall aggregate.
    Returns {data: [{date, accuracy_pct, session_count}], session_count, avg_rate, ...}
    """
    import time as _time
    from datetime import datetime as _dt, timezone as _tz, timedelta as _td

    cid = resolve_client_id(request, require=False)

    now_ts   = int(_time.time())
    end_ts   = date_to   or now_ts
    start_ts = date_from or (now_ts - days * 86400)

    cid_filter = "AND client_id = (:cid)::uuid" if cid else ""
    params: dict[str, Any] = {"start_ts": start_ts, "end_ts": end_ts}
    if cid:
        params["cid"] = cid

    # Daily buckets
    daily_rows = await db.execute(
        text(f"""
            SELECT
                to_char(to_timestamp(actual_start), 'YYYY-MM-DD') AS day,
                AVG(recognition_rate) * 100                        AS accuracy_pct,
                COUNT(*)                                            AS session_count
            FROM sessions
            WHERE recognition_rate IS NOT NULL
              AND actual_start >= :start_ts
              AND actual_start <= :end_ts
              {cid_filter}
            GROUP BY day
            ORDER BY day
        """),
        params,
    )
    data = [
        {
            "date":          r.day,
            "accuracy_pct":  round(float(r.accuracy_pct), 1),
            "session_count": int(r.session_count),
        }
        for r in daily_rows.fetchall()
    ]

    # Overall aggregate
    agg = await db.execute(
        text(f"""
            SELECT
                COUNT(*)                AS session_count,
                AVG(recognition_rate)   AS avg_rate,
                MIN(recognition_rate)   AS min_rate,
                MAX(recognition_rate)   AS max_rate,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY recognition_rate) AS median_rate
            FROM sessions
            WHERE recognition_rate IS NOT NULL
              AND actual_start >= :start_ts
              AND actual_start <= :end_ts
              {cid_filter}
        """),
        params,
    )
    r = agg.fetchone()
    return {
        "data":          data,
        "session_count": int(r.session_count or 0),
        "avg_rate":      round(float(r.avg_rate or 0), 4),
        "min_rate":      round(float(r.min_rate or 0), 4),
        "max_rate":      round(float(r.max_rate or 0), 4),
        "median_rate":   round(float(r.median_rate or 0), 4),
    }


# ── System health ─────────────────────────────────────────────────────────────

@router.get("/system-health", dependencies=[_SYS])
async def system_health(request: Request, db: DBSession) -> dict:
    """
    Platform health metrics: DB pool, Redis latency, active sessions,
    GPU semaphore contention.  SUPER_ADMIN sees all clients; CLIENT_ADMIN
    sees their own client's metrics.
    """
    import asyncio, time

    # DB round-trip
    t0 = time.monotonic()
    await db.execute(text("SELECT 1"))
    db_ms = round((time.monotonic() - t0) * 1000, 2)

    # Redis round-trip
    redis  = request.app.state.redis
    t0     = time.monotonic()
    await redis.ping()
    redis_ms = round((time.monotonic() - t0) * 1000, 2)

    # Kafka metadata (fire-and-forget attempt)
    kafka_ok = False
    try:
        topics = await asyncio.to_thread(
            request.app.state.kafka_producer.list_topics, timeout=3
        )
        kafka_ok = True
    except Exception:
        pass

    # Active sessions on this node
    active_sessions = len(getattr(request.app.state, "active_sessions", {}))

    cid = resolve_client_id(request)

    # Per-client camera status counts
    cam_rows = await db.execute(
        text("""
            SELECT status, COUNT(*) AS cnt
            FROM cameras
            WHERE client_id = (:cid)::uuid
            GROUP BY status
        """),
        {"cid": cid},
    )
    cam_status = {r.status: int(r.cnt) for r in cam_rows.fetchall()}

    return {
        "db_latency_ms":     db_ms,
        "redis_latency_ms":  redis_ms,
        "kafka_ok":          kafka_ok,
        "active_sessions":   active_sessions,
        "camera_status":     cam_status,
    }


# ── Camera uptime ─────────────────────────────────────────────────────────────

@router.get("/camera-uptime", dependencies=[_CAM])
async def camera_uptime(
    request:   Request,
    db:        DBSession,
    date_from: int | None = Query(None),
    date_to:   int | None = Query(None),
) -> dict:
    """
    Per-camera session activity: session count, total actual duration,
    derived uptime fraction of the requested window.
    """
    cid = resolve_client_id(request)
    params: dict[str, Any] = {"cid": cid}
    filters = "AND s.client_id = (:cid)::uuid AND s.actual_start IS NOT NULL AND s.actual_end IS NOT NULL"
    if date_from:
        filters += " AND s.actual_start >= :df"
        params["df"] = date_from
    if date_to:
        filters += " AND s.actual_end <= :dt"
        params["dt"] = date_to

    rows = await db.execute(
        text(f"""
            SELECT
                c.camera_id::text,
                c.name,
                c.status,
                c.room_name,
                c.building,
                COUNT(s.session_id) AS session_count,
                SUM(s.actual_end - s.actual_start) AS total_seconds
            FROM cameras c
            LEFT JOIN sessions s ON s.camera_id = c.camera_id {filters}
            WHERE c.client_id = (:cid)::uuid
            GROUP BY c.camera_id, c.name, c.status, c.room_name, c.building
            ORDER BY total_seconds DESC NULLS LAST
        """),
        params,
    )
    window_s = (
        ((date_to or now_epoch()) - (date_from or (now_epoch() - 86_400)))
    )
    items = []
    for r in rows.fetchall():
        total_s = float(r.total_seconds or 0)
        items.append({
            "camera_id":     r.camera_id,
            "name":          r.name,
            "status":        r.status,
            "room_name":     r.room_name,
            "building":      r.building,
            "session_count": int(r.session_count or 0),
            "total_seconds": total_s,
            "uptime_pct":    round(min(total_s / max(window_s, 1) * 100, 100), 2),
        })
    return {"total_cameras": len(items), "window_s": window_s, "items": items}


# ── Faculty report ────────────────────────────────────────────────────────────

@router.get("/faculty-report", dependencies=[_ATT])
async def faculty_report(
    request:   Request,
    db:        DBSession,
    date_from: int | None = Query(None),
    date_to:   int | None = Query(None),
    limit:     int        = Query(50, ge=1, le=200),
    offset:    int        = Query(0, ge=0),
) -> dict:
    """
    Per-faculty summary: sessions chaired, attendance rates, held/sync rates.
    """
    cid = resolve_client_id(request)
    params: dict[str, Any] = {"cid": cid}
    filters = "AND s.client_id = (:cid)::uuid AND s.faculty_id IS NOT NULL"
    if date_from:
        filters += " AND s.created_at >= :df"
        params["df"] = date_from
    if date_to:
        filters += " AND s.created_at <= :dt"
        params["dt"] = date_to

    rows = await db.execute(
        text(f"""
            SELECT
                p.person_id::text,
                p.name,
                p.department,
                COUNT(s.session_id)                               AS total_sessions,
                AVG(s.recognition_rate)                           AS avg_recognition_rate,
                SUM(CASE WHEN s.sync_status = 'SYNCED'  THEN 1 ELSE 0 END) AS synced,
                SUM(CASE WHEN s.sync_status = 'HELD'    THEN 1 ELSE 0 END) AS held,
                SUM(CASE WHEN s.faculty_status = 'PRESENT' THEN 1 ELSE 0 END) AS present_sessions
            FROM sessions s
            JOIN persons p ON p.person_id = s.faculty_id
            WHERE 1=1 {filters}
            GROUP BY p.person_id, p.name, p.department
            ORDER BY total_sessions DESC
            LIMIT :lim OFFSET :off
        """),
        {**params, "lim": limit, "off": offset},
    )
    items = [
        {
            "person_id":          r.person_id,
            "name":               r.name,
            "department":         r.department,
            "total_sessions":     int(r.total_sessions or 0),
            "avg_recognition_rate": round(float(r.avg_recognition_rate or 0), 4),
            "synced":             int(r.synced or 0),
            "held":               int(r.held or 0),
            "present_sessions":   int(r.present_sessions or 0),
        }
        for r in rows.fetchall()
    ]
    return {"total": len(items), "offset": offset, "limit": limit, "items": items}


# ── Student report ────────────────────────────────────────────────────────────

@router.get("/student-report/{person_id}", dependencies=[_ATT])
async def student_report(
    person_id: str,
    request:   Request,
    db:        DBSession,
    date_from: int | None = Query(None),
    date_to:   int | None = Query(None),
) -> dict:
    """
    Attendance summary for a single student: status distribution, session
    list, rolling 7-day present-rate trend.
    """
    from app.models.persons import Person as _Person
    _p = await db.get(_Person, uuid.UUID(person_id))
    cid = str(_p.client_id) if _p else resolve_client_id(request)
    params: dict[str, Any] = {"cid": cid, "pid": person_id}
    filters = "AND ar.client_id = (:cid)::uuid AND ar.person_id = (:pid)::uuid"
    if date_from:
        filters += " AND ar.first_seen >= :df"
        params["df"] = date_from
    if date_to:
        filters += " AND ar.first_seen <= :dt"
        params["dt"] = date_to

    # Status distribution
    dist_rows = await db.execute(
        text(f"""
            SELECT status, COUNT(*) AS cnt
            FROM attendance_records ar
            WHERE 1=1 {filters}
            GROUP BY status
        """),
        params,
    )
    distribution = {r.status: int(r.cnt) for r in dist_rows.fetchall()}

    # Rolling 7-day trend (present rate per day)
    trend_rows = await db.execute(
        text(f"""
            SELECT
                date_trunc('day', to_timestamp(ar.first_seen)) AS day,
                SUM(CASE WHEN ar.status IN ('P','L') THEN 1 ELSE 0 END) AS present,
                COUNT(*) AS total
            FROM attendance_records ar
            WHERE ar.first_seen IS NOT NULL {filters}
            GROUP BY day
            ORDER BY day DESC
            LIMIT 30
        """),
        params,
    )
    trend = [
        {
            "day":          str(r.day),
            "present":      int(r.present),
            "total":        int(r.total),
            "present_rate": round(int(r.present) / max(int(r.total), 1), 4),
        }
        for r in trend_rows.fetchall()
    ]

    total = sum(distribution.values())
    present = distribution.get("P", 0) + distribution.get("L", 0)
    return {
        "person_id":       person_id,
        "total_records":   total,
        "present_rate":    round(present / max(total, 1), 4),
        "distribution":    distribution,
        "daily_trend":     trend,
    }


# ── Flow matrix ───────────────────────────────────────────────────────────────

@router.get("/flow-matrix", dependencies=[require_permission("cameras:read")])
async def flow_matrix(
    request:   Request,
    date:      str       = Query(
        default="",
        description="ISO date YYYY-MM-DD; defaults to today (UTC)",
    ),
    hour:      int | None = Query(None, ge=0, le=23, description="Filter to a single hour (0–23)"),
    client_id: str | None = Query(None, description="SUPER_ADMIN only — override client scope"),
) -> dict:
    """
    Camera-to-camera person-flow matrix for a given date (and optional hour).

    Returns Sankey-ready `nodes` + `links`:
      nodes  — [{id, name, building, floor}]
      links  — [{source, target, source_name, target_name, value, avg_transit_s}]
    """
    cid = resolve_client_id(request, client_id)
    if not date:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    analyzer = CrossCameraAnalyzer(
        session_factory=request.app.state.session_factory,
        redis=request.app.state.redis,
    )
    matrix = await analyzer.build_flow_matrix(cid, date, hour)
    return {
        "date":              matrix.date,
        "hour":              matrix.hour,
        "total_transitions": matrix.total_transitions,
        "nodes":             matrix.nodes,
        "links":             matrix.links,
    }


# ── Occupancy forecast ────────────────────────────────────────────────────────

@router.get("/occupancy-forecast", dependencies=[_CAM])
async def occupancy_forecast(
    request:   Request,
    camera_id: str | None = Query(None, description="Filter to a single camera"),
    dow:       int | None = Query(None, ge=0, le=6, description="Day of week 0=Mon…6=Sun; omit for all days"),
    days_back: int        = Query(28,  ge=7, le=90, description="History window"),
    client_id: str | None = Query(None, description="SUPER_ADMIN only"),
) -> dict:
    """
    Day-of-week-aware occupancy forecast with mean ± 1.5σ confidence bands.

    Each camera entry contains an `hourly` array (hours 0–23) with:
      mean, std, lower, upper (confidence band), samples (calendar-day count).
    """
    cid = resolve_client_id(request, client_id)
    analyzer = CrossCameraAnalyzer(
        session_factory=request.app.state.session_factory,
        redis=request.app.state.redis,
    )
    forecasts = await analyzer.get_occupancy_forecast(
        client_id=cid,
        camera_id=camera_id,
        dow=dow,
        days_back=days_back,
    )
    return {
        "days_back":      days_back,
        "dow":            dow,
        "total_cameras":  len(forecasts),
        "forecasts": [
            {
                "camera_id":     f.camera_id,
                "camera_name":   f.camera_name,
                "dow":           f.dow,
                "dow_name":      f.dow_name,
                "peak_hour":     f.peak_hour,
                "peak_mean":     f.peak_mean,
                "days_analysed": f.days_analysed,
                "hourly":        [
                    {
                        "hour":    h.hour,
                        "mean":    h.mean,
                        "std":     h.std,
                        "lower":   h.lower,
                        "upper":   h.upper,
                        "samples": h.samples,
                    }
                    for h in f.hourly
                ],
            }
            for f in forecasts
        ],
    }


# ── Transit model ─────────────────────────────────────────────────────────────

@router.get("/transit-model", dependencies=[require_permission("cameras:read")])
async def transit_model(
    request:     Request,
    date_from:   int | None = Query(None, description="epoch; defaults to 30 days ago"),
    date_to:     int | None = Query(None, description="epoch; defaults to now"),
    min_samples: int        = Query(5, ge=1, le=50, description="Minimum sightings per camera pair"),
    client_id:   str | None = Query(None, description="SUPER_ADMIN only"),
) -> dict:
    """
    Statistical transit-time model for every observed camera pair.

    Each entry contains:
      n, mean, std, p5, p25, p50 (median), p75, p95, min (lo), max (hi) in seconds.
      normal_range_lo / normal_range_hi — mean ± 2σ expected window.

    Results are cached in Redis for 1 hour.
    Pairs with fewer than `min_samples` observations are excluded.
    """
    cid = resolve_client_id(request, client_id)
    now = now_epoch()
    dfr = date_from or (now - 30 * 86_400)
    dto = date_to   or now

    analyzer = CrossCameraAnalyzer(
        session_factory=request.app.state.session_factory,
        redis=request.app.state.redis,
    )
    model = await analyzer.build_transit_model(
        client_id=cid, date_from=dfr, date_to=dto, min_samples=min_samples,
    )
    return {
        "date_from":    dfr,
        "date_to":      dto,
        "min_samples":  min_samples,
        "total_pairs":  len(model),
        "pairs":        [s.to_dict() for s in model.values()],
    }


# ── Cross-camera anomalies ────────────────────────────────────────────────────

@router.get("/anomalies", dependencies=[require_permission("cameras:read")])
async def cross_camera_anomalies(
    request:   Request,
    date_from: int | None = Query(None, description="epoch; defaults to 24 h ago"),
    date_to:   int | None = Query(None, description="epoch; defaults to now"),
    camera_id: str | None = Query(None, description="Scope to transits involving this camera"),
    severity:  str | None = Query(
        None,
        pattern="^(IMPOSSIBLE|SUSPICIOUS|LATE)$",
        description="Filter by severity level",
    ),
    limit:     int        = Query(100, ge=1, le=500),
    client_id: str | None = Query(None, description="SUPER_ADMIN only"),
) -> dict:
    """
    Detect statistically anomalous transits between cameras.

    Severity levels
    ───────────────
    IMPOSSIBLE  — transit faster than physical minimum (< 10 s) or z ≤ −5
    SUSPICIOUS  — unusually fast transit (z ≤ −2.5)
    LATE        — unusually slow transit (z ≥ +3.5)

    Uses the 30-day transit model as a baseline; pairs with fewer than 5
    observations are only flagged on the physical minimum rule.
    """
    cid = resolve_client_id(request, client_id)
    now = now_epoch()
    dfr = date_from or (now - 86_400)
    dto = date_to   or now

    analyzer = CrossCameraAnalyzer(
        session_factory=request.app.state.session_factory,
        redis=request.app.state.redis,
    )
    anomalies = await analyzer.scan_anomalies(
        client_id=cid,
        date_from=dfr,
        date_to=dto,
        camera_id=camera_id,
        limit=limit,
    )

    if severity:
        anomalies = [a for a in anomalies if a.severity == severity]

    return {
        "date_from": dfr,
        "date_to":   dto,
        "total":     len(anomalies),
        "anomalies": [
            {
                "person_id":          a.person_id,
                "person_name":        a.person_name,
                "from_camera_id":     a.from_camera_id,
                "from_camera_name":   a.from_camera_name,
                "to_camera_id":       a.to_camera_id,
                "to_camera_name":     a.to_camera_name,
                "transit_seconds":    a.transit_seconds,
                "timestamp":          a.timestamp,
                "severity":           a.severity,
                "z_score":            a.z_score,
                "expected_mean":      a.expected_mean,
                "expected_std":       a.expected_std,
                "expected_range_lo":  a.expected_range_lo,
                "expected_range_hi":  a.expected_range_hi,
                "reason":             a.reason,
                "model_samples":      a.model_samples,
            }
            for a in anomalies
        ],
    }


# ── Course / timetable attendance breakdown ───────────────────────────────────

@router.get("/course-attendance", dependencies=[_ATT])
async def course_attendance(
    request:     Request,
    db:          DBSession,
    date_from:   int | None = Query(None, description="epoch"),
    date_to:     int | None = Query(None, description="epoch"),
    course_name: str | None = Query(None, description="Exact course name (case-insensitive)"),
    client_id:   str | None = Query(None, description="SUPER_ADMIN only"),
    limit:       int        = Query(100, ge=1, le=500),
    offset:      int        = Query(0, ge=0),
) -> dict:
    """
    Course attendance grouped by calendar date + course_name.

    Each row is one (date, course) pair with:
      - enrolled: live count of active face-profiled persons for that client
      - present:  distinct persons marked P or L across all sessions that day
      - absent:   enrolled - present
    """
    cid = resolve_client_id(request, client_id, require=False)
    params: dict[str, Any] = {}
    if cid:
        params["cid"] = cid
    s_filters = ("AND s.client_id = (:cid)::uuid AND s.course_name IS NOT NULL"
                 if cid else "AND s.course_name IS NOT NULL")
    if date_from:
        s_filters += " AND s.actual_start >= :df"
        params["df"] = date_from
    if date_to:
        s_filters += " AND s.actual_start <= :dt"
        params["dt"] = date_to
    if course_name:
        s_filters += " AND LOWER(s.course_name) = LOWER(:cn)"
        params["cn"] = course_name

    rows = await db.execute(
        text(f"""
            WITH session_base AS (
                SELECT
                    s.session_id,
                    s.client_id,
                    s.course_id,
                    s.course_name,
                    s.camera_id,
                    s.actual_start,
                    -- calendar day in UTC (text YYYY-MM-DD)
                    to_char(to_timestamp(s.actual_start) AT TIME ZONE 'UTC', 'YYYY-MM-DD') AS day_str,
                    MIN(s.actual_start) OVER (
                        PARTITION BY s.client_id, s.course_name,
                        to_char(to_timestamp(s.actual_start) AT TIME ZONE 'UTC', 'YYYY-MM-DD')
                    ) AS day_start_epoch,
                    MAX(COALESCE(s.actual_end, s.actual_start)) OVER (
                        PARTITION BY s.client_id, s.course_name,
                        to_char(to_timestamp(s.actual_start) AT TIME ZONE 'UTC', 'YYYY-MM-DD')
                    ) AS day_end_epoch
                FROM sessions s
                WHERE s.actual_start IS NOT NULL
                  {s_filters}
            ),
            enrolled_counts AS (
                SELECT DISTINCT ON (sb.client_id)
                    sb.client_id,
                    (
                        SELECT COUNT(DISTINCT fe.person_id)
                        FROM face_embeddings fe
                        JOIN persons p ON p.person_id = fe.person_id
                        WHERE p.client_id = sb.client_id
                          AND p.status = 'ACTIVE'
                          AND fe.is_active = true
                    ) AS enrolled
                FROM session_base sb
            ),
            day_course AS (
                SELECT
                    sb.client_id,
                    sb.course_id,
                    sb.course_name,
                    sb.day_str,
                    sb.day_start_epoch,
                    sb.day_end_epoch,
                    COUNT(DISTINCT sb.session_id) AS session_count,
                    COUNT(DISTINCT CASE WHEN ar.status IN ('P','L') THEN ar.person_id END) AS present,
                    MIN(c.name) AS camera_name
                FROM session_base sb
                LEFT JOIN attendance_records ar ON ar.session_id = sb.session_id
                LEFT JOIN cameras c ON c.camera_id = sb.camera_id
                GROUP BY sb.client_id, sb.course_id, sb.course_name, sb.day_str,
                         sb.day_start_epoch, sb.day_end_epoch
            )
            SELECT
                dc.course_name,
                dc.course_id,
                dc.day_str,
                dc.day_start_epoch,
                dc.day_end_epoch,
                dc.session_count,
                dc.camera_name,
                COALESCE(ec.enrolled, 0) AS enrolled,
                dc.present
            FROM day_course dc
            LEFT JOIN enrolled_counts ec ON ec.client_id = dc.client_id
            ORDER BY dc.day_str DESC, dc.course_name
            LIMIT :lim OFFSET :off
        """),
        {**params, "lim": limit, "off": offset},
    )

    # All distinct course names for filter dropdown
    cid_filter_courses = "WHERE client_id = (:cid)::uuid AND course_name IS NOT NULL" if cid else "WHERE course_name IS NOT NULL"
    course_rows = await db.execute(
        text(f"SELECT DISTINCT course_name FROM sessions {cid_filter_courses} ORDER BY course_name"),
        {"cid": cid} if cid else {},
    )

    items = []
    for r in rows.fetchall():
        enrolled = int(r.enrolled or 0)
        present  = int(r.present  or 0)
        absent   = max(0, enrolled - present)
        items.append({
            "course_name":   r.course_name,
            "course_id":     r.course_id,
            "date":          r.day_str,
            "actual_start":  r.day_start_epoch,
            "actual_end":    r.day_end_epoch,
            "session_count": int(r.session_count or 0),
            "camera_name":   r.camera_name,
            "enrolled":      enrolled,
            "present":       present,
            "absent":        absent,
            "rate":          round(present / max(enrolled, 1), 4),
        })

    return {
        "total":   len(items),
        "offset":  offset,
        "limit":   limit,
        "courses": [r.course_name for r in course_rows.fetchall()],
        "items":   items,
    }


# ── PTZ stats (per-camera scan performance) ───────────────────────────────────

@router.get("/ptz-stats", dependencies=[_CAM])
async def ptz_stats(
    request: Request,
    current_user: CurrentUser,
    db: DBSession,
    client_id: str | None = Query(None),
) -> dict:
    """Return per-camera PTZ scan statistics from active/recent sessions."""
    client_id = resolve_client_id(request, client_id or (str(current_user.get("client_id")) if current_user.get("client_id") else None))
    brains = getattr(request.app.state, "ptz_brains", {})

    cameras: list[dict] = []
    for cam_id, brain in brains.items():
        state = brain.get_state() if hasattr(brain, "get_state") else None
        if state is None:
            continue
        cam_row = await db.execute(
            select(Camera).where(Camera.camera_id == uuid.UUID(cam_id))
        )
        cam = cam_row.scalar_one_or_none()
        cameras.append({
            "camera_name":      cam.name if cam else cam_id[:8],
            "avg_cycle_ms":     round((state.cycle_time_s or 0) * 1000),
            "face_hunt_rate":   round(getattr(state, "face_hunt_rate", 0.0), 3),
            "recognition_rate": round(state.recognition_rate or 0.0, 3),
            "cycles_today":     state.cycle_count or 0,
        })
    return {"cameras": cameras}


# ── Dwell-time distribution ────────────────────────────────────────────────────

@router.get("/dwell-distribution", dependencies=[_CAM])
async def dwell_distribution(
    request: Request,
    current_user: CurrentUser,
    db: DBSession,
    days: int = Query(7, ge=1, le=90),
    client_id: str | None = Query(None),
) -> dict:
    """Histogram of sighting dwell times bucketed by minute ranges."""
    client_id = resolve_client_id(request, client_id or (str(current_user.get("client_id")) if current_user.get("client_id") else None))
    cutoff = now_epoch() - days * 86400
    rows = await db.execute(
        text("""
            SELECT
                CASE
                    WHEN (last_seen - first_seen) < 60    THEN '<1 min'
                    WHEN (last_seen - first_seen) < 300   THEN '1-5 min'
                    WHEN (last_seen - first_seen) < 900   THEN '5-15 min'
                    WHEN (last_seen - first_seen) < 1800  THEN '15-30 min'
                    ELSE '30+ min'
                END AS bucket,
                COUNT(*)::int AS count
            FROM sightings
            WHERE client_id = CAST(:cid AS uuid)
              AND first_seen >= :cutoff
            GROUP BY bucket
            ORDER BY MIN(last_seen - first_seen)
        """),
        {"cid": str(client_id), "cutoff": cutoff},
    )
    return {"items": [{"bucket": r.bucket, "count": r.count} for r in rows]}


# ── Frequent visitors ──────────────────────────────────────────────────────────

@router.get("/frequent-visitors", dependencies=[_CAM])
async def frequent_visitors(
    request: Request,
    current_user: CurrentUser,
    db: DBSession,
    limit: int = Query(8, ge=1, le=50),
    days: int  = Query(30, ge=1, le=365),
    client_id: str | None = Query(None),
) -> dict:
    """Top N persons by sighting count over the past N days."""
    client_id = resolve_client_id(request, client_id or (str(current_user.get("client_id")) if current_user.get("client_id") else None))
    cutoff = now_epoch() - days * 86400
    rows = await db.execute(
        text("""
            SELECT p.name, p.external_id, COUNT(s.sighting_id)::int AS sighting_count
            FROM sightings s
            JOIN persons p ON p.person_id = s.person_id
            WHERE s.client_id = CAST(:cid AS uuid)
              AND s.first_seen >= :cutoff
              AND s.person_id IS NOT NULL
            GROUP BY p.person_id, p.name, p.external_id
            ORDER BY sighting_count DESC
            LIMIT :lim
        """),
        {"cid": str(client_id), "cutoff": cutoff, "lim": limit},
    )
    return {"items": [
        {"name": r.name, "external_id": r.external_id or "", "sighting_count": r.sighting_count, "thumbnail": None}
        for r in rows
    ]}


# ── Unknown persons (unidentified sightings) ───────────────────────────────────

@router.get("/unknown-persons", dependencies=[_CAM])
async def unknown_persons(
    request: Request,
    current_user: CurrentUser,
    db: DBSession,
    limit: int = Query(20, ge=1, le=100),
    days: int  = Query(7, ge=1, le=90),
    client_id: str | None = Query(None),
) -> dict:
    """Recent sightings of unidentified persons (no person_id match)."""
    client_id = resolve_client_id(request, client_id or (str(current_user.get("client_id")) if current_user.get("client_id") else None))
    cutoff = now_epoch() - days * 86400
    rows = await db.execute(
        text("""
            SELECT
                s.sighting_id::text,
                to_timestamp(s.first_seen)::text AS first_seen,
                to_timestamp(s.last_seen)::text  AS last_seen,
                c.name AS camera_name,
                1      AS sighting_count
            FROM sightings s
            JOIN cameras c ON c.camera_id = s.camera_id
            WHERE s.client_id  = CAST(:cid AS uuid)
              AND s.person_id  IS NULL
              AND s.first_seen >= :cutoff
            ORDER BY s.first_seen DESC
            LIMIT :lim
        """),
        {"cid": str(client_id), "cutoff": cutoff, "lim": limit},
    )
    return {"items": [
        {
            "sighting_id":   r.sighting_id,
            "thumbnail":     None,
            "first_seen":    r.first_seen,
            "last_seen":     r.last_seen,
            "camera_name":   r.camera_name,
            "sighting_count": r.sighting_count,
        }
        for r in rows
    ]}


# ── Hourly occupancy matrix ────────────────────────────────────────────────────

@router.get("/hourly-occupancy", dependencies=[_CAM])
async def hourly_occupancy(
    request: Request,
    current_user: CurrentUser,
    db: DBSession,
    days: int = Query(30, ge=1, le=90),
    client_id: str | None = Query(None),
) -> dict:
    """Average occupancy (unique persons) per camera per hour-of-day."""
    client_id = resolve_client_id(request, client_id or (str(current_user.get("client_id")) if current_user.get("client_id") else None))
    cutoff = now_epoch() - days * 86400
    rows = await db.execute(
        text("""
            SELECT
                c.name AS cam_name,
                EXTRACT(HOUR FROM to_timestamp(s.first_seen))::int AS hour,
                COUNT(DISTINCT s.person_id)::int AS occupancy
            FROM sightings s
            JOIN cameras c ON c.camera_id = s.camera_id
            WHERE s.client_id  = CAST(:cid AS uuid)
              AND s.first_seen >= :cutoff
            GROUP BY c.name, hour
            ORDER BY c.name, hour
        """),
        {"cid": str(client_id), "cutoff": cutoff},
    )
    # Build matrix: areas (camera names) × hours (0-23)
    from collections import defaultdict
    area_hours: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        area_hours[r.cam_name][r.hour] += r.occupancy
    areas = sorted(area_hours.keys())
    matrix = [[area_hours[a].get(h, 0) for h in range(24)] for a in areas]
    return {"areas": areas, "matrix": matrix}


# ── Occupancy heatmap (monitoring page) ───────────────────────────────────────

@router.get("/occupancy-heatmap", dependencies=[_CAM])
async def occupancy_heatmap(
    request: Request,
    current_user: CurrentUser,
    db: DBSession,
    days: int = Query(7, ge=1, le=30),
    client_id: str | None = Query(None),
) -> dict:
    """Same as hourly-occupancy — heatmap by camera × hour for the monitoring page."""
    return await hourly_occupancy(request, current_user, db, days=days, client_id=client_id)
