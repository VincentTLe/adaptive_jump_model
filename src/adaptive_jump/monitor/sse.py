"""Reconnectable server-sent event projection over append-only job journals."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable

from adaptive_jump.monitor.event_store import EventStore
from adaptive_jump.monitor.queue import QueueStore

_STREAM_STOPPED = frozenset({"interrupted", "canceled", "succeeded", "failed"})
Disconnected = Callable[[], Awaitable[bool]]


class StreamError(ValueError):
    """Raised when a reconnect cursor is ambiguous or invalid."""


def resume_sequence(last_event_id: str | None, query_after: int) -> int:
    """Resolve one non-negative cursor from EventSource or an explicit query."""
    if last_event_id is None or last_event_id == "":
        return query_after
    if (
        not last_event_id.isascii()
        or not last_event_id.isdecimal()
        or int(last_event_id) > 2**63 - 1
    ):
        raise StreamError("Last-Event-ID must be a non-negative integer")
    header_after = int(last_event_id)
    if query_after not in {0, header_after}:
        raise StreamError("reconnect cursors disagree")
    return header_after


async def stream_job_events(
    queue: QueueStore,
    store: EventStore,
    job_id: str,
    after_sequence: int,
    disconnected: Disconnected,
    *,
    poll_seconds: float = 1.0,
    keepalive_seconds: float = 15.0,
) -> AsyncIterator[str]:
    """Replay missed visible events, then tail until disconnect or job stop."""
    cursor = after_sequence
    last_output = time.monotonic()
    while True:
        replay = await asyncio.to_thread(store.replay, job_id, cursor)
        for runtime in replay:
            cursor = runtime.sequence
            if runtime.event.visibility != "outcome":
                yield _message("research_event", runtime.to_dict(), runtime.sequence)
                last_output = time.monotonic()
        job = await asyncio.to_thread(queue.get, job_id)
        if job.status in _STREAM_STOPPED:
            yield _message("stream_complete", {"status": job.status})
            return
        if await disconnected():
            return
        now = time.monotonic()
        if now - last_output >= keepalive_seconds:
            yield ": keepalive\n\n"
            last_output = now
        await asyncio.sleep(poll_seconds)


def _message(event: str, data: dict, sequence: int | None = None) -> str:
    lines = []
    if sequence is not None:
        lines.append(f"id: {sequence}")
    lines.extend(
        (
            f"event: {event}",
            "data: " + json.dumps(data, separators=(",", ":"), sort_keys=True),
            "",
            "",
        )
    )
    return "\n".join(lines)
