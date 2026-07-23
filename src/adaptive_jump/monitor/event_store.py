"""Append-only per-job runtime event journals for replay and SSE recovery."""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from adaptive_jump.runtime.events import (
    EventError,
    EventObserver,
    ResearchEvent,
    RuntimeEvent,
)

_JOB_ID = re.compile(r"[0-9a-f]{32}\Z")


class EventStoreError(RuntimeError):
    pass


@dataclass
class _WriterState:
    next_sequence: int
    started_monotonic: float
    elapsed_offset: float
    file_size: int


class EventStore:
    """Own one append writer while allowing validated reconnectable reads."""

    def __init__(self, runtime_root: Path) -> None:
        self.runtime_root = runtime_root.resolve()
        if (
            self.runtime_root.name != ".monitor"
            or self.runtime_root.parent.name != "artifacts"
        ):
            raise EventStoreError("event store must use artifacts/.monitor")
        self._states: dict[str, _WriterState] = {}
        self._lock = threading.RLock()

    def observer(self, job_id: str) -> EventObserver:
        """Bind protocol-neutral research events to one immutable job identity."""
        self._path(job_id)

        def observe(event: ResearchEvent) -> None:
            self.append(job_id, event)

        return observe

    def append(self, job_id: str, event: ResearchEvent) -> RuntimeEvent:
        if not isinstance(event, ResearchEvent):
            raise EventStoreError("only ResearchEvent values may be appended")
        with self._lock:
            path = self._path(job_id)
            state = self._states.get(job_id)
            if state is None:
                state = self._restore(job_id, path)
                self._states[job_id] = state
            size = path.stat().st_size if path.exists() else 0
            if size != state.file_size:
                raise EventStoreError("event journal changed outside its writer")
            now = datetime.now(UTC)
            elapsed = state.elapsed_offset + time.monotonic() - state.started_monotonic
            runtime = RuntimeEvent(job_id, state.next_sequence, now, elapsed, event)
            payload = (
                json.dumps(
                    runtime.to_dict(),
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode()
                + b"\n"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("ab") as handle:
                handle.write(payload)
                handle.flush()
            state.next_sequence += 1
            state.file_size += len(payload)
            return runtime

    def replay(self, job_id: str, after_sequence: int = 0) -> tuple[RuntimeEvent, ...]:
        if (
            isinstance(after_sequence, bool)
            or not isinstance(after_sequence, int)
            or after_sequence < 0
        ):
            raise EventStoreError("after_sequence must be a non-negative integer")
        with self._lock:
            path = self._path(job_id)
            events, size = _read_journal(path, job_id)
            state = self._states.get(job_id)
            if state is not None and state.file_size != size:
                raise EventStoreError("event journal changed outside its writer")
            return tuple(event for event in events if event.sequence > after_sequence)

    def _restore(self, job_id: str, path: Path) -> _WriterState:
        events, size = _read_journal(path, job_id)
        elapsed_offset = 0.0
        if events:
            last = events[-1]
            wall_gap = max(0.0, (datetime.now(UTC) - last.time_utc).total_seconds())
            elapsed_offset = last.elapsed_seconds + wall_gap
        return _WriterState(
            next_sequence=len(events) + 1,
            started_monotonic=time.monotonic(),
            elapsed_offset=elapsed_offset,
            file_size=size,
        )

    def _path(self, job_id: str) -> Path:
        if not isinstance(job_id, str) or _JOB_ID.fullmatch(job_id) is None:
            raise EventStoreError("job_id is not a queue-generated identifier")
        return self.runtime_root / "jobs" / job_id / "events.jsonl"


def _read_journal(path: Path, job_id: str) -> tuple[list[RuntimeEvent], int]:
    if not path.exists():
        return [], 0
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise EventStoreError(f"cannot read event journal: {job_id}") from exc
    if payload and not payload.endswith(b"\n"):
        raise EventStoreError("event journal ends with a partial record")
    events: list[RuntimeEvent] = []
    previous_elapsed = -1.0
    for sequence, line in enumerate(payload.splitlines(), start=1):
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise TypeError
            event = RuntimeEvent.from_dict(record)
        except (EventError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EventStoreError(f"invalid event journal record {sequence}") from exc
        if (
            event.job_id != job_id
            or event.sequence != sequence
            or event.elapsed_seconds < previous_elapsed
        ):
            raise EventStoreError(f"inconsistent event journal record {sequence}")
        events.append(event)
        previous_elapsed = event.elapsed_seconds
    return events, len(payload)
