"""
ACAS Load Test — Locust
========================
Simulates the full production concurrency profile:

  20  DashboardUser  — REST API browsing (attendance, live, analytics, search, cameras)
  15  CameraWorker   — persistent WebSocket sessions that push recognition events
   5  EnrollmentUser — periodic bulk-enrollment and face-search operations

Run
───
    pip install locust websocket-client requests
    cd ACAS
    locust -f scripts/load_test.py \
        --host http://localhost:8000 \
        --users 40 --spawn-rate 4 \
        --run-time 10m \
        --headless \
        --html reports/load_test.html \
        --csv  reports/load_test

Results interpretation
──────────────────────
• p50 < 100 ms for list endpoints
• p50 < 50 ms  for single-item endpoints
• p95 < 300 ms across the board
• WebSocket message latency < 600 ms (2/sec throttle = 500 ms max)
• Error rate < 0.1%
"""
from __future__ import annotations

import json
import os
import random
import string
import threading
import time
import uuid
from typing import Any

import requests
from locust import HttpUser, User, between, constant_pacing, events, task
from locust.exception import StopUser

try:
    import websocket as ws_client  # websocket-client
    _WS_AVAILABLE = True
except ImportError:
    _WS_AVAILABLE = False
    print("WARNING: websocket-client not installed — CameraWorker will be skipped")

# ─── Configuration ─────────────────────────────────────────────────────────────

_ADMIN_EMAIL    = os.getenv("ACAS_TEST_EMAIL",    "admin@acas.local")
_ADMIN_PASS     = os.getenv("ACAS_TEST_PASSWORD", "AdminPass123!")
_CLIENT_EMAIL   = os.getenv("ACAS_CLIENT_EMAIL",  "client@acas.local")
_CLIENT_PASS    = os.getenv("ACAS_CLIENT_PASSWORD","ClientPass123!")

# Fake IDs used when real data isn't available
_FAKE_SESSION_ID = "00000000-0000-0000-0000-000000000001"
_FAKE_PERSON_ID  = "00000000-0000-0000-0000-000000000002"
_FAKE_CAMERA_ID  = "00000000-0000-0000-0000-000000000003"

# ─── Shared token cache (one login per role) ────────────────────────────────────

_token_cache: dict[str, str] = {}
_token_lock = threading.Lock()


def _get_token(host: str, email: str, password: str) -> str | None:
    key = f"{email}@{host}"
    with _token_lock:
        if key in _token_cache:
            return _token_cache[key]
        try:
            r = requests.post(
                f"{host}/api/auth/login",
                json={"email": email, "password": password},
                timeout=10,
            )
            if r.status_code == 200:
                token = r.json().get("access_token")
                if token:
                    _token_cache[key] = token
                    return token
        except Exception as exc:
            print(f"Login failed for {email}: {exc}")
        return None


# ─── Custom event reporting ─────────────────────────────────────────────────────

@events.test_start.add_listener
def on_test_start(environment: Any, **_: Any) -> None:
    print("\n" + "=" * 70)
    print("ACAS Load Test Starting")
    print(f"  Users: 20 DashboardUser + 15 CameraWorker + 5 EnrollmentUser")
    print(f"  Target: p50 < 100ms list | p50 < 50ms single | p95 < 300ms")
    print("=" * 70 + "\n")


@events.quitting.add_listener
def on_quitting(environment: Any, **_: Any) -> None:
    stats = environment.stats
    print("\n" + "=" * 70)
    print("ACAS Load Test Results")
    print("-" * 70)
    print(f"{'Endpoint':<42} {'p50':>7} {'p95':>7} {'p99':>7} {'Err%':>6}")
    print("-" * 70)
    for key, entry in sorted(stats.entries.items(), key=lambda x: x[1].median_response_time or 0, reverse=True):
        err_pct = (entry.num_failures / max(entry.num_requests, 1)) * 100
        print(
            f"{str(key[1])[:42]:<42}"
            f" {entry.median_response_time or 0:>6.0f}ms"
            f" {entry.get_response_time_percentile(0.95) or 0:>6.0f}ms"
            f" {entry.get_response_time_percentile(0.99) or 0:>6.0f}ms"
            f" {err_pct:>5.1f}%"
        )
    print("=" * 70 + "\n")


# ─── Dashboard User (20 instances) ─────────────────────────────────────────────

class DashboardUser(HttpUser):
    """
    Simulates a CLIENT_ADMIN dashboard user browsing attendance, live monitor,
    analytics, camera list, and person search.

    Weight = 20 (set in locust --users split via user_classes).
    """
    wait_time = between(1, 4)

    def on_start(self) -> None:
        token = _get_token(self.host, _CLIENT_EMAIL, _CLIENT_PASS)
        if not token:
            # Try admin credentials as fallback
            token = _get_token(self.host, _ADMIN_EMAIL, _ADMIN_PASS)
        if not token:
            raise StopUser()
        self.headers = {"Authorization": f"Bearer {token}"}
        # Cache session IDs for follow-up requests
        self._session_ids: list[str] = []
        self._person_ids:  list[str] = []
        self._camera_ids:  list[str] = []
        self._prime_ids()

    def _prime_ids(self) -> None:
        """Fetch a page of IDs at startup so subsequent requests are realistic."""
        try:
            r = self.client.get(
                "/api/sessions/active", headers=self.headers, name="/api/sessions/active [prime]"
            )
            if r.status_code == 200:
                for s in r.json().get("sessions", [])[:5]:
                    if sid := s.get("session_id"):
                        self._session_ids.append(sid)
        except Exception:
            pass
        if not self._session_ids:
            self._session_ids = [_FAKE_SESSION_ID]

    # ── Attendance (weight 5 — most visited page) ─────────────────────────────

    @task(5)
    def attendance_sessions_list(self) -> None:
        self.client.get("/api/attendance/sessions", headers=self.headers)

    @task(2)
    def attendance_session_detail(self) -> None:
        sid = random.choice(self._session_ids)
        with self.client.get(
            f"/api/attendance/sessions/{sid}",
            headers=self.headers,
            name="/api/attendance/sessions/[id]",
            catch_response=True,
        ) as r:
            if r.status_code in (200, 404):
                r.success()

    @task(1)
    def attendance_session_records(self) -> None:
        sid = random.choice(self._session_ids)
        with self.client.get(
            f"/api/attendance/sessions/{sid}/records",
            headers=self.headers,
            name="/api/attendance/sessions/[id]/records",
            catch_response=True,
        ) as r:
            if r.status_code in (200, 404):
                r.success()

    @task(1)
    def held_batches(self) -> None:
        self.client.get("/api/attendance/held", headers=self.headers)

    # ── Live Monitor (weight 4) ────────────────────────────────────────────────

    @task(4)
    def active_sessions(self) -> None:
        self.client.get("/api/sessions/active", headers=self.headers)

    @task(2)
    def session_state_poll(self) -> None:
        sid = random.choice(self._session_ids)
        with self.client.get(
            f"/api/sessions/{sid}/state",
            headers=self.headers,
            name="/api/sessions/[id]/state",
            catch_response=True,
        ) as r:
            if r.status_code in (200, 404):
                r.success()

    # ── Analytics (weight 3 — cached at 30s) ──────────────────────────────────

    @task(3)
    def analytics_attendance_trends(self) -> None:
        self.client.get(
            "/api/analytics/attendance-trends?days=30",
            headers=self.headers,
        )

    @task(2)
    def analytics_system_health(self) -> None:
        self.client.get("/api/analytics/system-health", headers=self.headers)

    @task(1)
    def analytics_recognition(self) -> None:
        self.client.get("/api/analytics/recognition-accuracy", headers=self.headers)

    # ── Cameras (weight 2) ────────────────────────────────────────────────────

    @task(2)
    def camera_list(self) -> None:
        self.client.get("/api/cameras", headers=self.headers)

    # ── Search (weight 3) ─────────────────────────────────────────────────────

    @task(3)
    def text_search(self) -> None:
        query = random.choice(["alice", "bob", "test", "class", "faculty"])
        self.client.get(
            f"/api/search/person?q={query}",
            headers=self.headers,
        )

    # ── Node info (weight 1) ─────────────────────────────────────────────────

    @task(1)
    def node_info(self) -> None:
        self.client.get("/api/node/info", headers=self.headers)

    # ── Pagination smoke test ─────────────────────────────────────────────────

    @task(1)
    def attendance_cursor_paginate(self) -> None:
        """Simulate cursor pagination: fetch page 1, follow next_cursor."""
        r = self.client.get(
            "/api/attendance/sessions?limit=20",
            headers=self.headers,
            name="/api/attendance/sessions [cursor-page-1]",
        )
        if r.status_code == 200:
            cur = r.json().get("next_cursor")
            if cur:
                self.client.get(
                    f"/api/attendance/sessions?limit=20&cursor={cur}",
                    headers=self.headers,
                    name="/api/attendance/sessions [cursor-page-2]",
                )


# ─── Camera Session Worker (15 instances) ─────────────────────────────────────

class CameraWorker(User):
    """
    Simulates a GPU node pushing real-time PTZ scan state via WebSocket.
    Each worker connects to the session WebSocket and publishes events,
    mirroring what PTZBrain does during a live session.

    This exercises:
    • WebSocket upgrade + auth
    • Redis pub/sub fan-out
    • WSEventBuffer throttle under load
    """
    wait_time = constant_pacing(0.5)   # 2 events/sec per session

    def on_start(self) -> None:
        if not _WS_AVAILABLE:
            raise StopUser()

        token = _get_token(self.host, _CLIENT_EMAIL, _CLIENT_PASS)
        if not token:
            token = _get_token(self.host, _ADMIN_EMAIL, _ADMIN_PASS)
        if not token:
            raise StopUser()

        self._token   = token
        self._ws: ws_client.WebSocket | None = None
        self._session_id = str(uuid.uuid4())
        self._connect()

    def _ws_url(self) -> str:
        base = self.host.replace("https://", "wss://").replace("http://", "ws://")
        return f"{base}/api/sessions/{self._session_id}/ws?token={self._token}"

    def _connect(self) -> None:
        try:
            self._ws = ws_client.create_connection(
                self._ws_url(),
                timeout=10,
                skip_utf8_validation=True,
            )
        except Exception as exc:
            self.environment.events.request.fire(
                request_type="WS",
                name="connect",
                response_time=0,
                response_length=0,
                exception=exc,
            )

    def _send_event(self, event: dict) -> None:
        if not self._ws:
            return
        t0 = time.perf_counter()
        name = f"WS:{event.get('type', 'unknown')}"
        try:
            self._ws.send(json.dumps(event))
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            self.environment.events.request.fire(
                request_type="WS",
                name=name,
                response_time=elapsed_ms,
                response_length=len(json.dumps(event)),
                exception=None,
            )
        except Exception as exc:
            self.environment.events.request.fire(
                request_type="WS",
                name=name,
                response_time=0,
                response_length=0,
                exception=exc,
            )
            self._ws = None
            time.sleep(0.5)
            self._connect()

    def on_stop(self) -> None:
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass

    # ── Tasks ─────────────────────────────────────────────────────────────────

    @task(5)
    def push_recognition_event(self) -> None:
        self._send_event({
            "type":  "recognition_event",
            "data": {
                "person_id":   str(uuid.uuid4()),
                "name":        f"Student-{random.randint(1, 200)}",
                "confidence":  round(random.uniform(0.72, 0.99), 3),
                "liveness":    round(random.uniform(0.65, 0.99), 3),
                "cell_id":     random.randint(0, 11),
                "ts":          time.time(),
            },
        })

    @task(3)
    def push_scan_map_update(self) -> None:
        cells = [
            {
                "id":          i,
                "pan":         round(random.uniform(-0.5, 0.5), 3),
                "tilt":        round(random.uniform(-0.3, 0.3), 3),
                "zoom":        round(random.uniform(0.1, 0.8), 3),
                "recognized":  random.randint(0, 5),
                "unrecognized":random.randint(0, 3),
                "state":       random.choice(["green", "yellow", "gray", "current"]),
            }
            for i in range(random.randint(6, 12))
        ]
        self._send_event({"type": "scan_map_update", "data": {"cells": cells}})

    @task(2)
    def push_state_change(self) -> None:
        self._send_event({
            "type": "state_change",
            "data": {
                "state":    random.choice(["CELL_RECOGNIZE", "CELL_TRANSIT", "FACE_HUNT", "OVERVIEW_SCAN"]),
                "cell_id":  random.randint(0, 11),
                "ts":       time.time(),
            },
        })

    @task(1)
    def push_cycle_complete(self) -> None:
        self._send_event({
            "type": "cycle_complete",
            "data": {
                "cycle":        random.randint(1, 50),
                "recognized":   random.randint(10, 40),
                "unrecognized": random.randint(0, 5),
                "duration_ms":  random.randint(8000, 25000),
            },
        })

    @task(1)
    def ping(self) -> None:
        self._send_event({"type": "ping"})


# ─── Enrollment User (5 instances) ─────────────────────────────────────────────

class EnrollmentUser(HttpUser):
    """
    Exercises the heavier write-path operations: person search, image quality
    check, and bulk enrollment status queries.  Kept at 5 users to avoid
    overwhelming the GPU pipeline during the load test.
    """
    wait_time = between(3, 8)   # slower cadence — these hit the GPU

    def on_start(self) -> None:
        token = _get_token(self.host, _CLIENT_EMAIL, _CLIENT_PASS)
        if not token:
            token = _get_token(self.host, _ADMIN_EMAIL, _ADMIN_PASS)
        if not token:
            raise StopUser()
        self.headers = {"Authorization": f"Bearer {token}"}

    @task(4)
    def enrollment_list(self) -> None:
        self.client.get("/api/enrollment/list?limit=20", headers=self.headers)

    @task(2)
    def enrollment_guidelines(self) -> None:
        self.client.get("/api/enrollment/guidelines", headers=self.headers)

    @task(2)
    def face_search_text(self) -> None:
        q = "".join(random.choices(string.ascii_lowercase, k=3))
        self.client.get(f"/api/search/person?q={q}", headers=self.headers)

    @task(1)
    def person_journey(self) -> None:
        with self.client.get(
            f"/api/search/{_FAKE_PERSON_ID}/journey",
            headers=self.headers,
            name="/api/search/[id]/journey",
            catch_response=True,
        ) as r:
            if r.status_code in (200, 404):
                r.success()

    @task(1)
    def monitoring_occupancy(self) -> None:
        self.client.get("/api/analytics/occupancy-forecast", headers=self.headers)


# ─── User class mix ─────────────────────────────────────────────────────────────
# Locust picks from this list; weights control the ratio.
# 20 Dashboard : 15 Camera : 5 Enrollment ≈ 50% : 37.5% : 12.5%

if _WS_AVAILABLE:
    user_classes = [DashboardUser, CameraWorker, EnrollmentUser]
else:
    user_classes = [DashboardUser, EnrollmentUser]
