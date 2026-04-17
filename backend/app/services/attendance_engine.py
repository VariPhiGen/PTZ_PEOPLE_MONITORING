"""
AttendanceEngine — finalises a PTZ session into per-student attendance records.

Status codes (matching AttendanceRecord.status):
  P   Present       — detected for >= present_threshold (default 70%) of session
  L   Late          — first detection after late_grace (default 15 min)
  EE  Early Exit    — last detection before ee_grace from end (default 15 min)
  A   Absent        — detected for < absent_threshold (default 15%) of session
  ND  Not Detected  — zero detections (never seen by any camera)
  EX  Excused       — externally overridden (not set by this engine)

Faculty Gatekeeper rules:
  The gatekeeper decides whether the session result is APPROVED (publish to
  ERP, write DB) or HELD (held for manual review).

  PRESENT  (faculty_pct >= 70%)  → APPROVED
  ABSENT   (faculty_pct < 15%)   → HELD  ("faculty_absent")
  LATE     (first_seen > late_grace from scheduled_start)
                                 → PARTIAL ("faculty_late:{minutes}min")
  SUBSTITUTE (faculty_id != expected_faculty_id)
                                 → HELD  ("faculty_substitute:{person_id}")
  NO_FACULTY (faculty_tracker is None, i.e. no faculty configured)
                                 → APPROVED  (unmonitored session)

The engine is pure Python — no DB or Kafka calls. Callers (e.g. PTZBrain
CYCLE_COMPLETE or an HTTP endpoint) pass the result to KafkaProducer.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.ptz_brain import DurationTracker

logger = logging.getLogger(__name__)

# ── Default thresholds (all fractions of session_duration_s) ──────────────────

_DEFAULT_PRESENT_FRAC  = 0.70   # >= 70% duration → P
_DEFAULT_LATE_FRAC     = 0.15   # first_seen > 15% of duration from start → L
_DEFAULT_EE_FRAC       = 0.15   # last_seen < 15% of duration from end → EE
_DEFAULT_ABSENT_FRAC   = 0.15   # < 15% duration → A  (else intermediate → P or EE/L)
_DEFAULT_FACULTY_PRESENT_FRAC = 0.70
_DEFAULT_FACULTY_ABSENT_FRAC  = 0.15


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class AttendanceThresholds:
    """All tunable thresholds for one session finalisation."""
    session_duration_s:     float       # scheduled_end - scheduled_start
    present_frac:           float = _DEFAULT_PRESENT_FRAC
    late_grace_frac:        float = _DEFAULT_LATE_FRAC
    early_exit_frac:        float = _DEFAULT_EE_FRAC
    absent_frac:            float = _DEFAULT_ABSENT_FRAC
    faculty_present_frac:   float = _DEFAULT_FACULTY_PRESENT_FRAC
    faculty_absent_frac:    float = _DEFAULT_FACULTY_ABSENT_FRAC
    expected_faculty_id:    str | None = None   # if set, substitute check is active

    # Derived
    @property
    def present_s(self) -> float:
        return self.session_duration_s * self.present_frac

    @property
    def absent_s(self) -> float:
        return self.session_duration_s * self.absent_frac

    @property
    def late_grace_s(self) -> float:
        return self.session_duration_s * self.late_grace_frac

    @property
    def early_exit_s(self) -> float:
        return self.session_duration_s * self.early_exit_frac


@dataclass
class StudentRecord:
    """Finalised attendance result for one student."""
    record_id:       str
    person_id:       str
    status:          str            # P / L / EE / A / ND
    total_seconds:   float
    detection_count: int
    first_seen:      float | None   # epoch
    last_seen:       float | None   # epoch
    flags:           list[str] = field(default_factory=list)   # ["late", "early_exit"]


@dataclass
class FacultyVerdict:
    """Gatekeeper result for the faculty presence."""
    status:          str        # PRESENT | ABSENT | LATE | SUBSTITUTE | NO_FACULTY
    approved:        bool
    held_reason:     str | None
    person_id:       str | None
    total_seconds:   float
    pct_present:     float      # 0.0–1.0


@dataclass
class SessionResult:
    """Complete output of AttendanceEngine.finalize_session()."""
    result_id:       str
    session_id:      str
    client_id:       str
    records:         list[StudentRecord]
    faculty:         FacultyVerdict
    sync_status:     str        # "APPROVED" | "HELD"
    held_reason:     str | None
    # Aggregates
    total_roster:    int
    present_count:   int        # P
    late_count:      int        # L
    early_exit_count: int       # EE
    absent_count:    int        # A
    nd_count:        int        # ND
    recognition_rate: float     # (P+L+EE) / total_roster


# ── Engine ────────────────────────────────────────────────────────────────────

class AttendanceEngine:
    """
    Stateless finaliser.  Call finalize_session() once per completed session.

    Parameters
    ----------
    All threshold overrides are optional; the constructor accepts keyword
    arguments that become the defaults for every finalize_session() call
    (useful for per-client policy settings).
    """

    def __init__(
        self,
        *,
        present_frac:         float = _DEFAULT_PRESENT_FRAC,
        late_grace_frac:      float = _DEFAULT_LATE_FRAC,
        early_exit_frac:      float = _DEFAULT_EE_FRAC,
        absent_frac:          float = _DEFAULT_ABSENT_FRAC,
        faculty_present_frac: float = _DEFAULT_FACULTY_PRESENT_FRAC,
        faculty_absent_frac:  float = _DEFAULT_FACULTY_ABSENT_FRAC,
    ) -> None:
        self._defaults = dict(
            present_frac         = present_frac,
            late_grace_frac      = late_grace_frac,
            early_exit_frac      = early_exit_frac,
            absent_frac          = absent_frac,
            faculty_present_frac = faculty_present_frac,
            faculty_absent_frac  = faculty_absent_frac,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def finalize_session(
        self,
        session_id:       str,
        client_id:        str,
        trackers:         dict[str, DurationTracker],
        faculty_tracker:  DurationTracker | None,
        thresholds:       AttendanceThresholds,
        scheduled_start:  float,            # epoch seconds
        scheduled_end:    float,            # epoch seconds
        roster_ids:       list[str],        # full expected roster
    ) -> SessionResult:
        """
        Finalise a session.

        Parameters
        ----------
        trackers        person_id → DurationTracker for students who were seen.
        faculty_tracker DurationTracker for the faculty, or None if unconfigured.
        thresholds      AttendanceThresholds (includes session_duration_s).
        scheduled_start / scheduled_end  epoch timestamps from the Session row.
        roster_ids      complete roster; persons absent from trackers get ND.

        Returns a SessionResult ready to hand to KafkaProducer.publish_attendance().
        """
        session_dur = thresholds.session_duration_s

        # ── Faculty verdict ────────────────────────────────────────────────────
        faculty_verdict = self._evaluate_faculty(
            faculty_tracker, thresholds, scheduled_start,
        )

        # ── Per-student records ────────────────────────────────────────────────
        records: list[StudentRecord] = []
        for person_id in roster_ids:
            tracker = trackers.get(person_id)
            rec = self._evaluate_student(
                person_id, tracker, thresholds, scheduled_start, scheduled_end,
            )
            records.append(rec)

        # ── Aggregates ────────────────────────────────────────────────────────
        counts = {s: 0 for s in ("P", "L", "EE", "A", "ND")}
        for r in records:
            counts[r.status] = counts.get(r.status, 0) + 1

        detected_count = counts["P"] + counts["L"] + counts["EE"]
        total = len(roster_ids) or 1
        rec_rate = detected_count / total

        # ── Sync status ────────────────────────────────────────────────────────
        sync_status = "APPROVED" if faculty_verdict.approved else "HELD"
        held_reason = faculty_verdict.held_reason if not faculty_verdict.approved else None

        result = SessionResult(
            result_id        = str(uuid.uuid4()),
            session_id       = session_id,
            client_id        = client_id,
            records          = records,
            faculty          = faculty_verdict,
            sync_status      = sync_status,
            held_reason      = held_reason,
            total_roster     = len(roster_ids),
            present_count    = counts["P"],
            late_count       = counts["L"],
            early_exit_count = counts["EE"],
            absent_count     = counts["A"],
            nd_count         = counts["ND"],
            recognition_rate = rec_rate,
        )

        logger.info(
            "finalize_session  session=%s  P=%d L=%d EE=%d A=%d ND=%d  "
            "faculty=%s  sync=%s",
            session_id,
            counts["P"], counts["L"], counts["EE"], counts["A"], counts["ND"],
            faculty_verdict.status, sync_status,
        )
        return result

    # ── Private: student status logic ─────────────────────────────────────────

    def _evaluate_student(
        self,
        person_id:       str,
        tracker:         DurationTracker | None,
        th:              AttendanceThresholds,
        sched_start:     float,
        sched_end:       float,
    ) -> StudentRecord:
        """Map a DurationTracker (may be None) to a P/L/EE/A/ND status."""

        if tracker is None or tracker.detection_count == 0:
            return StudentRecord(
                record_id       = str(uuid.uuid4()),
                person_id       = person_id,
                status          = "ND",
                total_seconds   = 0.0,
                detection_count = 0,
                first_seen      = None,
                last_seen       = None,
            )

        dur = tracker.total_seconds
        flags: list[str] = []

        # Absent: detected but too briefly
        if dur < th.absent_s:
            return StudentRecord(
                record_id       = str(uuid.uuid4()),
                person_id       = person_id,
                status          = "A",
                total_seconds   = dur,
                detection_count = tracker.detection_count,
                first_seen      = tracker.first_seen,
                last_seen       = tracker.last_seen,
                flags           = flags,
            )

        # Late: first detection after late_grace from scheduled start
        is_late = (
            tracker.first_seen is not None
            and tracker.first_seen > sched_start + th.late_grace_s
        )

        # Early exit: last detection before (scheduled_end - early_exit_grace)
        ee_cutoff = sched_end - th.early_exit_s
        is_ee = (
            tracker.last_seen is not None
            and tracker.last_seen < ee_cutoff
        )

        if is_late:
            flags.append("late")
        if is_ee:
            flags.append("early_exit")

        # Presence threshold
        if dur >= th.present_s:
            # Present but may also be tagged late or EE
            if is_late and not is_ee:
                status = "L"
            elif is_ee and not is_late:
                status = "EE"
            elif is_late and is_ee:
                # Both late and early exit → whichever was more severe; default EE
                status = "EE"
            else:
                status = "P"
        else:
            # Between absent_s and present_s — borderline: tag L or EE if applicable
            if is_late:
                status = "L"
            elif is_ee:
                status = "EE"
            else:
                # Detected but insufficient total time and no clear qualifier
                status = "A"

        return StudentRecord(
            record_id       = str(uuid.uuid4()),
            person_id       = person_id,
            status          = status,
            total_seconds   = dur,
            detection_count = tracker.detection_count,
            first_seen      = tracker.first_seen,
            last_seen       = tracker.last_seen,
            flags           = flags,
        )

    # ── Private: faculty gatekeeper ────────────────────────────────────────────

    def _evaluate_faculty(
        self,
        tracker:     DurationTracker | None,
        th:          AttendanceThresholds,
        sched_start: float,
    ) -> FacultyVerdict:
        """Determine whether the session can be approved."""

        # No faculty configured → approve unconditionally
        if tracker is None:
            return FacultyVerdict(
                status        = "NO_FACULTY",
                approved      = True,
                held_reason   = None,
                person_id     = None,
                total_seconds = 0.0,
                pct_present   = 0.0,
            )

        dur   = tracker.total_seconds
        pct   = dur / th.session_duration_s if th.session_duration_s > 0 else 0.0
        pid   = tracker.person_id

        # Substitute: a different person occupied the faculty role
        if th.expected_faculty_id and pid != th.expected_faculty_id:
            return FacultyVerdict(
                status        = "SUBSTITUTE",
                approved      = False,
                held_reason   = f"faculty_substitute:{pid}",
                person_id     = pid,
                total_seconds = dur,
                pct_present   = pct,
            )

        # Absent
        if pct < th.faculty_absent_frac:
            return FacultyVerdict(
                status        = "ABSENT",
                approved      = False,
                held_reason   = "faculty_absent",
                person_id     = pid,
                total_seconds = dur,
                pct_present   = pct,
            )

        # Late: first detection after late_grace from scheduled start
        is_late = (
            tracker.first_seen is not None
            and tracker.first_seen > sched_start + th.late_grace_s
        )

        if is_late:
            late_min = int((tracker.first_seen - sched_start) / 60)
            return FacultyVerdict(
                status        = "LATE",
                approved      = True,   # partial approval — caller may downgrade
                held_reason   = f"faculty_late:{late_min}min",
                person_id     = pid,
                total_seconds = dur,
                pct_present   = pct,
            )

        # Present
        if pct >= th.faculty_present_frac:
            return FacultyVerdict(
                status        = "PRESENT",
                approved      = True,
                held_reason   = None,
                person_id     = pid,
                total_seconds = dur,
                pct_present   = pct,
            )

        # Between absent and present thresholds — treat as partial late
        return FacultyVerdict(
            status        = "LATE",
            approved      = True,
            held_reason   = f"faculty_partial:{pct * 100:.0f}pct",
            person_id     = pid,
            total_seconds = dur,
            pct_present   = pct,
        )
