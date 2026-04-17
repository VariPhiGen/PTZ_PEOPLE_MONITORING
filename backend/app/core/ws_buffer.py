"""
ACAS WebSocket Event Buffer
============================
Throttles outbound events to ≤ N flushes/sec and batches all pending events
into a single message per flush.

Supports two encoding modes:
  • JSON  — default, human-readable, works with any browser client
  • msgpack — binary, ~30% smaller payload, client must request via
              ?encoding=msgpack query parameter

Usage (inside a FastAPI WebSocket handler)
──────────────────────────────────────────
    buf = WSEventBuffer(websocket, max_rate=2.0, use_msgpack=False)
    flush_task = asyncio.create_task(buf.run())

    async for recognition in ai_events():
        await buf.push({"type": "recognition", "data": recognition})

    flush_task.cancel()
    await flush_task                  # drains remaining events
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

log = logging.getLogger(__name__)

# Maximum events we hold in the buffer before dropping (back-pressure)
_QUEUE_MAXSIZE = 500


class WSEventBuffer:
    """
    Per-connection event buffer with throttled flushing.

    Parameters
    ──────────
    websocket    FastAPI WebSocket instance
    max_rate     Maximum flush cycles per second (default 2 → one batch every 500 ms)
    use_msgpack  Encode batches with msgpack instead of JSON
    """

    def __init__(
        self,
        websocket:   Any,           # fastapi.WebSocket
        max_rate:    float = 2.0,
        use_msgpack: bool  = False,
    ) -> None:
        self._ws          = websocket
        self._period      = 1.0 / max_rate
        self._use_msgpack = use_msgpack
        self._queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)
        self._dropped     = 0
        self._sent        = 0

    # ── Push ─────────────────────────────────────────────────────────────────

    async def push(self, event: dict) -> bool:
        """
        Enqueue an event for the next flush.

        Returns False if the buffer is full (back-pressure signal).  The caller
        should slow down event production — NOT raise an error.
        """
        try:
            self._queue.put_nowait(event)
            return True
        except asyncio.QueueFull:
            self._dropped += 1
            if self._dropped % 50 == 1:
                log.warning("WSEventBuffer: %d event(s) dropped (queue full)", self._dropped)
            return False

    # ── Flush ─────────────────────────────────────────────────────────────────

    async def _flush(self) -> int:
        if self._queue.empty():
            return 0

        events: list[dict] = []
        try:
            while True:
                events.append(self._queue.get_nowait())
        except asyncio.QueueEmpty:
            pass

        if not events:
            return 0

        batch: dict[str, Any] = {
            "batch":   True,
            "count":   len(events),
            "ts":      time.time(),
            "events":  events,
        }

        try:
            if self._use_msgpack:
                import msgpack  # type: ignore[import-untyped]
                await self._ws.send_bytes(msgpack.packb(batch, use_bin_type=True))
            else:
                await self._ws.send_json(batch)
        except Exception as exc:
            log.debug("WSEventBuffer flush failed: %s", exc)
            raise

        self._sent += len(events)
        return len(events)

    # ── Flush loop ────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """
        Coroutine that flushes the buffer at `max_rate` cycles/sec.
        Run as an asyncio.Task.  Raises CancelledError on shutdown after
        draining any remaining events.
        """
        try:
            while True:
                await asyncio.sleep(self._period)
                await self._flush()
        except asyncio.CancelledError:
            await self._flush()   # drain on shutdown
        log.debug(
            "WSEventBuffer closed  sent=%d  dropped=%d",
            self._sent, self._dropped,
        )

    # ── Stats ──────────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, int]:
        return {
            "queued":  self._queue.qsize(),
            "sent":    self._sent,
            "dropped": self._dropped,
        }
