"""
ACAS Self-Learning Service
==========================
Closes the feedback loop between PTZ-Brain runtime observations and the
camera's `learned_params` JSON column, improving recognition performance
over time without manual tuning.

Two-phase design
────────────────
Phase 1 — Per-session write (hot path, called by PTZBrain.on_complete)
  • Produce a compact SessionSummary from the brain's final state.
  • Persist to Redis as a list (LPUSH) so the nightly job can pop them.
  • Also publish a session-report email via notifications.outbound Kafka.

Phase 2 — Nightly batch (cold path, called by apscheduler / cron)
  • Load the last 30 days of SessionSummaries for each room.
  • Re-fit: seat_heatmap, optimal_clusters, zoom_per_region,
            dwell_per_cell, camera_speeds, occupancy_curve.
  • PATCH camera.learned_params in PostgreSQL.
  • Rebuild PTZ-Brain path plan for next cycle (optional hot-reload).

learned_params schema (JSONB on cameras table)
──────────────────────────────────────────────
{
  "seat_heatmap":      [[pan, tilt, weight], ...],   # Gaussian-smoothed seat positions
  "optimal_clusters":  6,                            # DBSCAN optimal k
  "zoom_per_region":   {"cell_0": 0.45, ...},        # optimal zoom per scan cell
  "dwell_per_cell":    {"cell_0": 3.2, ...},         # median dwell seconds per cell
  "camera_speeds":     {"pan": 0.12, "tilt": 0.10},  # measured mechanical speeds
  "occupancy_curve":   [0.1, 0.3, 0.7, 0.9, ...],   # expected occupancy by hour-of-day
  "last_updated":      "2026-03-21T08:00:00Z",
  "sessions_analysed": 180
}
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# Redis key where per-session summaries are buffered per camera
_SUMMARY_KEY = "acas:self_learn:summaries:{camera_id}"
# Maximum sessions kept in Redis ring buffer (30 days × max 8 sessions/day)
_MAX_REDIS_SUMMARIES = 240


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class CellStats:
    cell_id:          int
    pan:              float
    tilt:             float
    zoom_used:        float
    dwell_s:          float          # total seconds camera spent on this cell
    recognitions:     int
    unrecognized:     int
    hunt_count:       int
    hunt_successes:   int


@dataclass
class SessionSummary:
    session_id:       str
    camera_id:        str
    client_id:        str
    course_id:        str | None
    faculty_id:       str | None
    mode:             str            # ATTENDANCE | MONITORING | BOTH
    started_at:       float          # unix epoch
    duration_s:       float
    cycle_count:      int
    recognition_rate: float
    cells:            list[CellStats] = field(default_factory=list)
    # Measured speeds from actual move commands vs. time (seconds/radian)
    measured_pan_speed:   float | None = None
    measured_tilt_speed:  float | None = None
    # Observed occupancy by hour bucket (0-23)
    occupancy_hour:   int | None = None
    occupancy_count:  int | None = None
    # Faculty detected?
    faculty_present:  bool = False


@dataclass
class LearnedParams:
    """In-memory representation of camera.learned_params."""
    seat_heatmap:       list[tuple[float, float, float]]  # (pan, tilt, weight)
    optimal_clusters:   int
    zoom_per_region:    dict[str, float]
    dwell_per_cell:     dict[str, float]
    camera_speeds:      dict[str, float]
    occupancy_curve:    list[float]          # 24 floats, one per hour
    last_updated:       str
    sessions_analysed:  int


# ─── Self-Learner ─────────────────────────────────────────────────────────────

class SelfLearner:
    """
    Parameters
    ──────────
    db            async SQLAlchemy session factory
    redis         aioredis.Redis client
    kafka_producer KafkaProducer instance (for notifications.outbound)
    """

    def __init__(
        self,
        db:             Any,   # async_sessionmaker
        redis:          Any,   # aioredis.Redis
        kafka_producer: Any,   # app.services.kafka_producer.KafkaProducer
    ) -> None:
        self._db      = db
        self._redis   = redis
        self._kafka   = kafka_producer

    # ─── Phase 1: write session summary ───────────────────────────────────────

    async def write_session_summary(
        self,
        brain_state: Any,   # SessionState from PTZBrain.get_session_state()
        camera_id:   str,
        client_id:   str,
        course_id:   str | None = None,
        faculty_id:  str | None = None,
        faculty_name: str | None = None,
    ) -> SessionSummary:
        """
        Produce a SessionSummary from the final PTZBrain state and:
        1. Push to Redis ring buffer for the nightly batch.
        2. Publish a session-report email to notifications.outbound.
        """
        cells = self._extract_cell_stats(brain_state)
        summary = SessionSummary(
            session_id       = brain_state.session_id,
            camera_id        = camera_id,
            client_id        = client_id,
            course_id        = course_id,
            faculty_id       = faculty_id,
            mode             = getattr(brain_state, "mode", "ATTENDANCE"),
            started_at       = getattr(brain_state, "started_at", time.time()),
            duration_s       = getattr(brain_state, "elapsed_s", 0.0),
            cycle_count      = getattr(brain_state, "cycle_count", 0),
            recognition_rate = getattr(brain_state, "recognition_rate", 0.0),
            cells            = cells,
            measured_pan_speed  = getattr(brain_state, "measured_pan_speed", None),
            measured_tilt_speed = getattr(brain_state, "measured_tilt_speed", None),
            occupancy_hour   = datetime.now(timezone.utc).hour,
            occupancy_count  = getattr(brain_state, "person_count_peak", None),
            faculty_present  = getattr(brain_state, "faculty_detected", False),
        )

        # Push to Redis ring buffer
        key = _SUMMARY_KEY.format(camera_id=camera_id)
        pipe = self._redis.pipeline()
        pipe.lpush(key, json.dumps(asdict(summary)))
        pipe.ltrim(key, 0, _MAX_REDIS_SUMMARIES - 1)
        await pipe.execute()

        # Publish session report email
        await asyncio.to_thread(
            self._publish_session_report, summary, faculty_name
        )

        log.info(
            "SelfLearner: session %s summary written  camera=%s  cycles=%d  rate=%.1f%%",
            summary.session_id, camera_id, summary.cycle_count,
            summary.recognition_rate * 100,
        )
        return summary

    def _extract_cell_stats(self, brain_state: Any) -> list[CellStats]:
        """Convert PTZBrain ScanCellSnapshot list into CellStats for storage."""
        cells: list[CellStats] = []
        for snap in getattr(brain_state, "scan_cells", []):
            cells.append(CellStats(
                cell_id        = hash(getattr(snap, "cell_id", "0")) % 10_000,
                pan            = getattr(snap, "center_pan", 0.0),
                tilt           = getattr(snap, "center_tilt", 0.0),
                zoom_used      = getattr(snap, "zoom_used", 0.0),
                dwell_s        = getattr(snap, "total_dwell_s", 0.0),
                recognitions   = getattr(snap, "recognition_count", 0),
                unrecognized   = getattr(snap, "unrecognized_count", 0),
                hunt_count     = getattr(snap, "hunt_count", 0),
                hunt_successes = getattr(snap, "hunt_success_count", 0),
            ))
        return cells

    # ─── Session report email ─────────────────────────────────────────────────

    def _publish_session_report(
        self,
        summary:      SessionSummary,
        faculty_name: str | None,
    ) -> None:
        """Publish a pre-rendered HTML email to notifications.outbound."""
        if not self._kafka or not summary.faculty_id:
            return

        total_recognized = sum(c.recognitions for c in summary.cells)
        total_faces      = total_recognized + sum(c.unrecognized for c in summary.cells)
        hunt_rate        = (
            sum(c.hunt_successes for c in summary.cells) /
            max(sum(c.hunt_count for c in summary.cells), 1) * 100
        )

        html = _build_report_html(
            summary, faculty_name, total_recognized, total_faces, hunt_rate
        )
        text_body = (
            f"ACAS Session Report — {summary.course_id or 'Unknown Course'}\n"
            f"Attendance: {total_recognized}/{total_faces} ({summary.recognition_rate*100:.1f}%)\n"
            f"Cycles: {summary.cycle_count}  Duration: {summary.duration_s/60:.0f} min\n"
        )

        try:
            self._kafka.publish_notification(
                recipient_id=summary.faculty_id,
                channel="email",
                template="session_report",
                payload={
                    "subject":     f"[ACAS] Session Report — {summary.course_id or 'Class'}",
                    "body":        text_body,
                    "html_body":   html,
                    "session_id":  summary.session_id,
                    "camera_id":   summary.camera_id,
                    "report_type": "session_complete",
                },
            )
        except Exception as exc:
            log.warning("SelfLearner: session report publish failed: %s", exc)

    # ─── Phase 2: nightly batch ───────────────────────────────────────────────

    async def nightly_batch(self, client_id: str | None = None) -> None:
        """
        Analyse the last 30 days of SessionSummaries for every camera
        (or only cameras belonging to `client_id`).
        Update camera.learned_params in the DB.
        """
        from sqlalchemy import select, update
        from app.models.cameras import Camera

        log.info("SelfLearner: nightly batch started  client_id=%s", client_id or "ALL")
        t0 = time.monotonic()

        async with self._db() as session:
            q = select(Camera).where(Camera.status != "ARCHIVED")
            if client_id:
                q = q.where(Camera.client_id == client_id)
            cameras = (await session.execute(q)).scalars().all()

        updated = 0
        for camera in cameras:
            summaries = await self._load_summaries(str(camera.camera_id))
            if not summaries:
                continue
            params = _fit_learned_params(summaries, camera.learned_params or {})
            async with self._db() as session:
                await session.execute(
                    update(Camera)
                    .where(Camera.camera_id == camera.camera_id)
                    .values(learned_params=params)
                )
                await session.commit()
            updated += 1

        elapsed = time.monotonic() - t0
        log.info(
            "SelfLearner: nightly batch complete  cameras=%d  elapsed=%.1fs",
            updated, elapsed,
        )

    async def _load_summaries(self, camera_id: str) -> list[SessionSummary]:
        """Pull up to _MAX_REDIS_SUMMARIES SessionSummaries from Redis."""
        key = _SUMMARY_KEY.format(camera_id=camera_id)
        raw_list = await self._redis.lrange(key, 0, _MAX_REDIS_SUMMARIES - 1)
        summaries: list[SessionSummary] = []
        for raw in raw_list:
            try:
                d = json.loads(raw)
                # Re-inflate CellStats
                d["cells"] = [CellStats(**c) for c in d.get("cells", [])]
                summaries.append(SessionSummary(**d))
            except Exception as exc:
                log.debug("SelfLearner: bad summary record: %s", exc)
        return summaries

    # ─── Scheduler entry point ────────────────────────────────────────────────

    async def start_scheduler(self) -> None:
        """
        Kick off the nightly batch at 02:00 UTC every day.
        Uses a simple asyncio loop (no APScheduler dependency required).
        """
        from datetime import timedelta

        while True:
            now = datetime.now(timezone.utc)
            next_run = now.replace(hour=2, minute=0, second=0, microsecond=0)
            if now >= next_run:
                next_run += timedelta(days=1)
            sleep_s = (next_run - now).total_seconds()
            log.info("SelfLearner: next nightly batch in %.0f s (%s UTC)", sleep_s, next_run)
            await asyncio.sleep(sleep_s)
            try:
                await self.nightly_batch()
            except Exception as exc:
                log.error("SelfLearner: nightly batch error: %s", exc, exc_info=True)


# ─── Fitting logic (pure functions — easy to unit-test) ───────────────────────

def _fit_learned_params(
    summaries:    list[SessionSummary],
    existing:     dict[str, Any],
) -> dict[str, Any]:
    """
    Derive updated learned_params from a list of SessionSummaries.
    Returns a new dict ready to be stored in camera.learned_params.
    """
    # ── seat_heatmap: kernel-density-estimate of pan/tilt centroids ────────────
    seat_heatmap = _build_seat_heatmap(summaries)

    # ── optimal_clusters: mode of cycle cell counts, clamped 3–15 ─────────────
    cluster_counts = [len(s.cells) for s in summaries if s.cells]
    optimal_clusters = int(statistics.mode(cluster_counts)) if cluster_counts else (
        existing.get("optimal_clusters", 6)
    )
    optimal_clusters = max(3, min(15, optimal_clusters))

    # ── zoom_per_region: median zoom used per cell_id ─────────────────────────
    zoom_by_cell: dict[int, list[float]] = defaultdict(list)
    for s in summaries:
        for c in s.cells:
            if c.zoom_used > 0:
                zoom_by_cell[c.cell_id].append(c.zoom_used)
    zoom_per_region = {
        f"cell_{k}": round(statistics.median(v), 3)
        for k, v in zoom_by_cell.items()
    }

    # ── dwell_per_cell: weighted median dwell time (seconds) ──────────────────
    dwell_by_cell: dict[int, list[float]] = defaultdict(list)
    for s in summaries:
        for c in s.cells:
            if c.dwell_s > 0:
                dwell_by_cell[c.cell_id].append(c.dwell_s)
    dwell_per_cell = {
        f"cell_{k}": round(statistics.median(v), 2)
        for k, v in dwell_by_cell.items()
    }

    # ── camera_speeds: EMA of measured pan/tilt speeds (rad/sec) ──────────────
    alpha = 0.1
    old_speeds = existing.get("camera_speeds", {"pan": 0.12, "tilt": 0.10})
    pan_samples  = [s.measured_pan_speed  for s in summaries if s.measured_pan_speed  is not None]
    tilt_samples = [s.measured_tilt_speed for s in summaries if s.measured_tilt_speed is not None]
    camera_speeds = {
        "pan":  round(
            _ema_update(old_speeds.get("pan", 0.12), pan_samples, alpha), 4
        ),
        "tilt": round(
            _ema_update(old_speeds.get("tilt", 0.10), tilt_samples, alpha), 4
        ),
    }

    # ── occupancy_curve: mean head-count observed per hour-of-day (0–23) ──────
    old_curve = existing.get("occupancy_curve", [0.0] * 24)
    occupancy_by_hour: dict[int, list[int]] = defaultdict(list)
    for s in summaries:
        if s.occupancy_hour is not None and s.occupancy_count is not None:
            occupancy_by_hour[s.occupancy_hour].append(s.occupancy_count)
    occupancy_curve = list(old_curve) if len(old_curve) == 24 else [0.0] * 24
    for hour, counts in occupancy_by_hour.items():
        if 0 <= hour < 24:
            new_val = statistics.mean(counts)
            occupancy_curve[hour] = round(
                occupancy_curve[hour] * (1 - alpha) + new_val * alpha, 2
            )

    return {
        "seat_heatmap":      seat_heatmap,
        "optimal_clusters":  optimal_clusters,
        "zoom_per_region":   zoom_per_region,
        "dwell_per_cell":    dwell_per_cell,
        "camera_speeds":     camera_speeds,
        "occupancy_curve":   occupancy_curve,
        "last_updated":      datetime.now(timezone.utc).isoformat(),
        "sessions_analysed": len(summaries),
    }


def _build_seat_heatmap(summaries: list[SessionSummary]) -> list[list[float]]:
    """
    Gaussian-smoothed seat positions: for each cell occurrence, add a weighted
    point at (pan, tilt) with weight = recognitions / (unrecognized + 1).
    Returns [[pan, tilt, weight], ...] sorted by weight descending.
    """
    points: list[tuple[float, float, float]] = []
    for s in summaries:
        for c in s.cells:
            w = (c.recognitions + 1) / (c.unrecognized + 1)
            points.append((c.pan, c.tilt, w))

    if not points:
        return []

    pts = np.array(points)
    # Gaussian blur: merge nearby points within 0.05 rad window
    merged: list[list[float]] = []
    used = [False] * len(pts)
    for i, (pan, tilt, w) in enumerate(pts):
        if used[i]:
            continue
        dists = np.sqrt((pts[:, 0] - pan) ** 2 + (pts[:, 1] - tilt) ** 2)
        nearby = dists < 0.05
        cluster = pts[nearby]
        total_w = cluster[:, 2].sum()
        avg_pan  = float(np.average(cluster[:, 0], weights=cluster[:, 2]))
        avg_tilt = float(np.average(cluster[:, 1], weights=cluster[:, 2]))
        merged.append([round(avg_pan, 4), round(avg_tilt, 4), round(total_w, 2)])
        used[nearby] = True  # type: ignore[index]

    return sorted(merged, key=lambda x: -x[2])[:50]  # top 50 seats


def _ema_update(old: float, samples: list[float], alpha: float) -> float:
    if not samples:
        return old
    for s in samples:
        old = old * (1 - alpha) + s * alpha
    return old


# ─── HTML session report ──────────────────────────────────────────────────────

def _build_report_html(
    summary:           SessionSummary,
    faculty_name:      str | None,
    total_recognized:  int,
    total_faces:       int,
    hunt_rate:         float,
) -> str:
    dt = datetime.fromtimestamp(summary.started_at, tz=timezone.utc)
    rows = "\n".join(
        f"<tr><td>Cell {c.cell_id}</td>"
        f"<td>{c.recognitions}</td>"
        f"<td>{c.unrecognized}</td>"
        f"<td>{c.dwell_s:.1f}s</td>"
        f"<td>{c.hunt_count}</td>"
        f"<td>{'✓' if c.hunt_successes else '—'}</td></tr>"
        for c in summary.cells
    )
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  body {{font-family:Arial,sans-serif;color:#222;background:#f9f9f9;padding:20px}}
  .card {{background:#fff;border-radius:8px;padding:20px;max-width:700px;margin:0 auto;box-shadow:0 2px 8px rgba(0,0,0,.1)}}
  h2 {{color:#1a56db;margin-top:0}} .stat {{font-size:2rem;font-weight:700;color:#1a56db}}
  .grid {{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:16px 0}}
  .box {{background:#f0f4ff;border-radius:6px;padding:12px;text-align:center}}
  table {{width:100%;border-collapse:collapse;font-size:.9rem}}
  th {{background:#1a56db;color:#fff;padding:6px 10px;text-align:left}}
  td {{padding:5px 10px;border-bottom:1px solid #e5e7eb}}
  .footer {{font-size:.8rem;color:#888;margin-top:16px;text-align:center}}
</style></head><body>
<div class="card">
  <h2>📋 ACAS Session Report</h2>
  <p><strong>Faculty:</strong> {faculty_name or 'Unknown'} &nbsp;|&nbsp;
     <strong>Course:</strong> {summary.course_id or 'N/A'} &nbsp;|&nbsp;
     <strong>Date:</strong> {dt.strftime('%d %b %Y %H:%M')} UTC</p>
  <div class="grid">
    <div class="box"><div class="stat">{total_recognized}/{total_faces}</div>Attendance</div>
    <div class="box"><div class="stat">{summary.recognition_rate*100:.0f}%</div>Recognition rate</div>
    <div class="box"><div class="stat">{summary.cycle_count}</div>Scan cycles</div>
    <div class="box"><div class="stat">{summary.duration_s/60:.0f}m</div>Duration</div>
    <div class="box"><div class="stat">{hunt_rate:.0f}%</div>Hunt success</div>
    <div class="box"><div class="stat">{'✓' if summary.faculty_present else '✗'}</div>Faculty detected</div>
  </div>
  <table><tr><th>Cell</th><th>Recognised</th><th>Unknown</th><th>Dwell</th><th>Hunts</th><th>Hunt OK</th></tr>
  {rows}
  </table>
  <p class="footer">Generated by ACAS &bull; Session {summary.session_id}</p>
</div></body></html>"""
