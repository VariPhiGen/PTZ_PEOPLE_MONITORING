"""
Sessions API — start/stop camera sessions and real-time WebSocket feed.

POST /api/sessions/start           — create and start a new session
POST /api/sessions/{id}/stop       — stop a running session
GET  /api/sessions/active          — list active sessions on this node
GET  /api/sessions/{id}/state      — current state snapshot (REST poll)
WS   /api/sessions/{id}/ws         — real-time state stream (WebSocket)
                                     token via ?token=<access_token>

WebSocket protocol
──────────────────
  Client → Server (JSON):
    {"type": "ping"}
    {"type": "subscribe"}
    {"type": "unsubscribe"}

  Server → Client (JSON):
    {"type": "state",   "data": <SessionState dict>}
    {"type": "event",   "data": <detection_event dict>}
    {"type": "error",   "data": {"message": "..."}}
    {"type": "pong"}

  The server subscribes to the Redis pub/sub channel
  "acas:session:{session_id}:state" and forwards every message.
  PTZBrain.get_session_state() must publish JSON to that channel.

Active session registry (app.state.active_sessions)
────────────────────────────────────────────────────
  dict[session_id → asyncio.Task]
  Populated by start/stop so node.py can also cancel tasks.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from app.core.ws_buffer import WSEventBuffer
from pydantic import BaseModel
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api._shared import audit, now_epoch, resolve_client_id
from app.deps import CurrentUser, DBSession
from app.middleware.auth import require_permission
from app.models.cameras import Camera
from app.models.persons import Person
from app.models.sessions import Session, SyncStatus
from app.utils.jwt import TokenError, decode_token

router = APIRouter(prefix="/api/sessions", tags=["sessions"])

_RD = require_permission("sessions:read")
_WR = require_permission("sessions:create")
_UP = require_permission("sessions:update")

_WS_PING_INTERVAL_S = 30   # overridden from Settings at runtime if configured
_WS_MAX_MESSAGE_S   = 300  # overridden from Settings at runtime if configured


# ── Schemas ───────────────────────────────────────────────────────────────────

class StartSessionRequest(BaseModel):
    camera_id:       str
    course_id:       str | None = None
    course_name:     str | None = None
    faculty_id:      str | None = None
    roster_ids:      list[str]  = []
    mode:            str        = "ATTENDANCE"   # ATTENDANCE | MONITORING
    scheduled_start: int | None = None            # epoch
    scheduled_end:   int | None = None            # epoch
    client_id:       str | None = None            # SUPER_ADMIN only


# ── Helpers ───────────────────────────────────────────────────────────────────

def _session_dict(s: Session) -> dict:
    return {
        "session_id":       str(s.session_id),
        "client_id":        str(s.client_id),
        "camera_id":        str(s.camera_id) if s.camera_id else None,
        "course_id":        s.course_id,
        "course_name":      s.course_name,
        "faculty_id":       str(s.faculty_id) if s.faculty_id else None,
        "scheduled_start":  s.scheduled_start,
        "scheduled_end":    s.scheduled_end,
        "actual_start":     s.actual_start,
        "actual_end":       s.actual_end,
        "faculty_status":   s.faculty_status,
        "sync_status":      s.sync_status,
        "held_reason":      s.held_reason,
        "cycle_count":      s.cycle_count,
        "recognition_rate": s.recognition_rate,
        "created_at":       s.created_at,
    }


def _get_active_sessions(request: Request) -> dict[str, asyncio.Task]:
    if not hasattr(request.app.state, "active_sessions"):
        request.app.state.active_sessions = {}
    return request.app.state.active_sessions


async def _get_session_or_404(db: DBSession, session_id: str) -> Session:
    s = await db.get(Session, uuid.UUID(session_id))
    if s is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return s


def _verify_ws_token(token: str, settings: Any) -> dict:
    """Decode and validate a JWT for WebSocket authentication."""
    try:
        payload = decode_token(token, settings)
    except TokenError as exc:
        raise WebSocketDisconnect(code=4001, reason=f"Invalid token: {exc}")
    if payload.get("type") != "access":
        raise WebSocketDisconnect(code=4001, reason="Refresh token not allowed")
    return payload


# ── REST endpoints ────────────────────────────────────────────────────────────

@router.post("/start", status_code=201, dependencies=[_WR])
async def start_session(body: StartSessionRequest, request: Request, db: DBSession) -> dict:
    """
    Create a new session record and dispatch a PTZBrain start event via Kafka.

    The actual PTZBrain task is started by the node worker that consumes
    the 'acas.session.start' Kafka event.  For single-node deployments the
    node.py /api/node/cameras/{id}/start route can be used directly.
    """
    cid = resolve_client_id(request, body.client_id)

    # Validate camera belongs to client
    cam = await db.get(Camera, uuid.UUID(body.camera_id))
    if cam is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    if str(cam.client_id) != cid:
        raise HTTPException(status_code=403, detail="Camera does not belong to this client")

    # Validate roster_ids: all persons must exist and belong to this client
    if body.roster_ids:
        valid_rows = (await db.execute(
            select(Person.person_id).where(
                Person.person_id.in_([uuid.UUID(r) for r in body.roster_ids]),
                Person.client_id == uuid.UUID(cid),
            )
        )).scalars().all()
        found_ids = {str(r) for r in valid_rows}
        missing = [r for r in body.roster_ids if r not in found_ids]
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown or unauthorized roster IDs: {missing[:10]}",
            )

    ts  = now_epoch()
    sid = uuid.uuid4()
    s   = Session(
        session_id      = sid,
        client_id       = uuid.UUID(cid),
        camera_id       = uuid.UUID(body.camera_id),
        course_id       = body.course_id,
        course_name     = body.course_name,
        faculty_id      = uuid.UUID(body.faculty_id) if body.faculty_id else None,
        scheduled_start = body.scheduled_start or ts,
        scheduled_end   = body.scheduled_end,
        actual_start    = ts,
        sync_status     = SyncStatus.PENDING,
        created_at      = ts,
    )
    db.add(s)
    await db.flush()

    # Publish start event — node worker picks this up.
    # On failure: delete the session row so it does not hang in PENDING forever.
    try:
        producer = request.app.state.kafka_producer
        producer.produce(
            "acas.session.commands",
            key=str(sid).encode(),
            value=json.dumps({
                "command":      "START",
                "session_id":   str(sid),
                "client_id":    cid,
                "camera_id":    body.camera_id,
                "mode":         body.mode,
                "roster_ids":   body.roster_ids,
                "faculty_id":   body.faculty_id,
                "node_id":      cam.node_id,
            }).encode(),
        )
        producer.poll(0)
    except Exception as exc:
        await db.delete(s)
        await db.commit()
        raise HTTPException(
            status_code=503,
            detail=f"Message broker unavailable — session not created: {exc}",
        )

    audit(request, "START_SESSION", "session", str(sid),
          {"camera_id": body.camera_id, "mode": body.mode})
    return _session_dict(s)


@router.post("/{session_id}/stop", dependencies=[_UP])
async def stop_session(session_id: str, request: Request, db: DBSession) -> dict:
    """Stop a running session.  Finalises attendance if in ATTENDANCE mode."""
    s   = await _get_session_or_404(db, session_id)
    now = now_epoch()
    s.actual_end = now
    await db.flush()

    # Graceful stop via brain.stop() if available, else cancel task
    active = _get_active_sessions(request)
    brains = getattr(request.app.state, "ptz_brains", {})
    brain  = brains.pop(session_id, None)
    task   = active.pop(session_id, None)
    if brain is not None:
        try:
            await brain.stop()
        except Exception:
            pass
    elif task and not task.done():
        task.cancel()

    # Publish stop command
    try:
        producer = request.app.state.kafka_producer
        producer.produce(
            "acas.session.commands",
            key=session_id.encode(),
            value=json.dumps({"command": "STOP", "session_id": session_id}).encode(),
        )
        producer.poll(0)
    except Exception:
        pass

    audit(request, "STOP_SESSION", "session", session_id)
    return _session_dict(s)


@router.get("/active", dependencies=[_RD])
async def list_active(request: Request, db: DBSession) -> dict:
    """List sessions that have started but not yet ended (actual_end IS NULL)."""
    cid = resolve_client_id(request, require=False)
    q   = select(Session).where(
        Session.actual_start.isnot(None),
        Session.actual_end.is_(None),
    )
    if cid:
        q = q.where(Session.client_id == uuid.UUID(cid))
    result   = await db.execute(q.order_by(Session.actual_start.desc()))
    sessions = result.scalars().all()

    # Tag which are running locally
    active_local = set(_get_active_sessions(request).keys())
    items = []
    for s in sessions:
        d = _session_dict(s)
        d["running_locally"] = str(s.session_id) in active_local
        items.append(d)

    return {"total": len(items), "items": items}


@router.get("/{session_id}/state", dependencies=[_RD])
async def get_session_state(session_id: str, request: Request, db: DBSession) -> dict:
    """
    Return the current session state.  Checks the in-memory PTZBrain first
    (if running locally), then falls back to the DB row.
    """
    s = await _get_session_or_404(db, session_id)

    # Try local brain state
    brains = getattr(request.app.state, "ptz_brains", {})
    brain  = brains.get(session_id)
    if brain is not None:
        try:
            state = brain.get_session_state()
            return {"source": "live", "state": vars(state)}
        except Exception:
            pass

    # Try Redis for inter-node state
    try:
        redis  = request.app.state.redis
        cached = await redis.get(f"acas:session:{session_id}:state")
        if cached:
            return {"source": "redis", "state": json.loads(cached)}
    except Exception:
        pass

    return {"source": "db", "state": _session_dict(s)}


# ── WebSocket ─────────────────────────────────────────────────────────────────

@router.websocket("/{session_id}/ws")
async def session_websocket(
    websocket:  WebSocket,
    session_id: str,
    token:      str = Query(..., description="JWT access token"),
) -> None:
    """
    Real-time session state stream.

    Authentication: JWT passed as ?token=<access_token>
    Pushes state updates published by PTZBrain to the Redis pub/sub channel
    `acas:session:{session_id}:state`.
    """
    await websocket.accept()

    # Read WS timeouts from settings if available
    _settings = getattr(websocket.app.state, "settings", None)
    ping_interval = getattr(_settings, "ws_ping_interval_s", _WS_PING_INTERVAL_S)
    max_idle      = getattr(_settings, "ws_max_idle_s",       _WS_MAX_MESSAGE_S)

    # ── Authenticate ──────────────────────────────────────────────────────────
    try:
        payload  = _verify_ws_token(token, websocket.app.state.settings)
        client_id = payload.get("client_id")
        role      = payload.get("role")
    except WebSocketDisconnect as exc:
        await websocket.close(code=exc.code)
        return

    # ── Validate session ownership ────────────────────────────────────────────
    redis = websocket.app.state.redis
    try:
        session_cid_bytes = await redis.get(f"acas:session:{session_id}:client_id")
        if session_cid_bytes and role != "SUPER_ADMIN":
            sid_str = session_cid_bytes.decode() if isinstance(session_cid_bytes, bytes) else str(session_cid_bytes)
            if str(client_id) != sid_str:
                await websocket.close(code=4003)
                return
    except Exception:
        pass

    # ── Subscribe to Redis pub/sub channel ───────────────────────────────────
    channel = f"acas:session:{session_id}:state"
    pubsub  = redis.pubsub()
    await pubsub.subscribe(channel)

    # Negotiate encoding: client sends ?encoding=msgpack for binary mode
    use_msgpack = websocket.query_params.get("encoding") == "msgpack"
    buf = WSEventBuffer(websocket, max_rate=2.0, use_msgpack=use_msgpack)

    async def _forward_redis() -> None:
        """Read from Redis pub/sub, push into throttle buffer (≤2 flushes/sec)."""
        async for message in pubsub.listen():
            if message["type"] != "message":
                continue
            try:
                data = message["data"]
                if isinstance(data, bytes):
                    data = data.decode()
                await buf.push({"type": "state", "data": json.loads(data)})
            except (WebSocketDisconnect, RuntimeError):
                return
            except Exception:
                pass

    async def _receive_client() -> None:
        """Handle incoming client messages (ping / subscribe / unsubscribe)."""
        while True:
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=ping_interval)
                msg = json.loads(raw)
                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
            except asyncio.TimeoutError:
                try:
                    await websocket.send_json({"type": "pong"})
                except Exception:
                    return
            except (WebSocketDisconnect, RuntimeError):
                return
            except Exception:
                pass

    # Run all three coroutines; cancel all when any exits
    tasks = [
        asyncio.create_task(_forward_redis()),
        asyncio.create_task(_receive_client()),
        asyncio.create_task(buf.run()),   # flush loop: batches events at 2/sec
    ]
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    except Exception:
        pass
    finally:
        try:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()
        except Exception:
            pass
        try:
            await websocket.close()
        except Exception:
            pass
