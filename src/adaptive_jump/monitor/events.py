"""Versioned, protocol-neutral events emitted by research computations."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime
from typing import Any

SCHEMA_VERSION = 1
VISIBILITIES = frozenset({"operational", "decision", "outcome"})
_EVENT_NAME = re.compile(r"[a-z][a-z0-9_]*\Z")


class EventError(ValueError):
    """Raised when runtime telemetry violates its versioned contract."""


@dataclass(frozen=True)
class ResearchEvent:
    """A scientific-stage event without process or persistence metadata."""

    kind: str
    stage: str
    visibility: str = "operational"
    market: str | None = None
    model: str | None = None
    delay: int | None = None
    date: date | None = None
    completed: int | None = None
    total: int | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_name(self.kind, "kind")
        _require_name(self.stage, "stage")
        if not isinstance(self.visibility, str) or self.visibility not in VISIBILITIES:
            raise EventError("visibility is not recognized")
        for label, value in (("market", self.market), ("model", self.model)):
            if value is not None and (not isinstance(value, str) or not value):
                raise EventError(f"{label} must be non-empty text")
        if self.delay is not None and (
            isinstance(self.delay, bool)
            or not isinstance(self.delay, int)
            or self.delay < 0
        ):
            raise EventError("delay must be a non-negative integer")
        if self.date is not None and type(self.date) is not date:
            raise EventError("date must be a calendar date")
        _validate_progress(self.completed, self.total)
        if not isinstance(self.payload, Mapping):
            raise EventError("payload must be a mapping")
        try:
            encoded = json.dumps(
                dict(self.payload),
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        except (TypeError, ValueError) as exc:
            raise EventError("payload must contain finite JSON values") from exc
        object.__setattr__(self, "payload", json.loads(encoded))


@dataclass(frozen=True)
class RuntimeEvent:
    """A persistable event with monotonic job and wall-clock metadata."""

    job_id: str
    sequence: int
    time_utc: datetime
    elapsed_seconds: float
    event: ResearchEvent

    def __post_init__(self) -> None:
        if not isinstance(self.job_id, str) or not self.job_id:
            raise EventError("job_id must be non-empty text")
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 1
        ):
            raise EventError("sequence must be a positive integer")
        if not isinstance(self.time_utc, datetime) or self.time_utc.utcoffset() is None:
            raise EventError("time_utc must be timezone-aware")
        object.__setattr__(self, "time_utc", self.time_utc.astimezone(UTC))
        if (
            isinstance(self.elapsed_seconds, bool)
            or not isinstance(self.elapsed_seconds, (int, float))
            or not math.isfinite(self.elapsed_seconds)
            or self.elapsed_seconds < 0
        ):
            raise EventError("elapsed_seconds must be finite and non-negative")
        if not isinstance(self.event, ResearchEvent):
            raise EventError("event must be a ResearchEvent")

    def to_dict(self) -> dict[str, Any]:
        """Return the exact JSON-compatible schema written to the event log."""
        event = self.event
        return {
            "schema_version": SCHEMA_VERSION,
            "job_id": self.job_id,
            "sequence": self.sequence,
            "time_utc": self.time_utc.isoformat().replace("+00:00", "Z"),
            "elapsed_seconds": float(self.elapsed_seconds),
            "kind": event.kind,
            "stage": event.stage,
            "visibility": event.visibility,
            "market": event.market,
            "model": event.model,
            "delay": event.delay,
            "date": event.date.isoformat() if event.date else None,
            "completed": event.completed,
            "total": event.total,
            "payload": json.loads(json.dumps(event.payload)),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> RuntimeEvent:
        """Parse one exact-schema JSON object for replay or SSE recovery."""
        example = cls("x", 1, datetime.now(UTC), 0, ResearchEvent("x", "x"))
        expected = set(example.to_dict())
        version = record.get("schema_version")
        valid_version = isinstance(version, int) and not isinstance(version, bool)
        if set(record) != expected or not valid_version or version != SCHEMA_VERSION:
            raise EventError("runtime event schema does not match version 1")
        try:
            timestamp = datetime.fromisoformat(
                str(record["time_utc"]).replace("Z", "+00:00")
            )
            event_date = (
                date.fromisoformat(record["date"])
                if isinstance(record["date"], str)
                else record["date"]
            )
            event = ResearchEvent(
                kind=record["kind"],
                stage=record["stage"],
                visibility=record["visibility"],
                market=record["market"],
                model=record["model"],
                delay=record["delay"],
                date=event_date,
                completed=record["completed"],
                total=record["total"],
                payload=record["payload"],
            )
            return cls(
                job_id=record["job_id"],
                sequence=record["sequence"],
                time_utc=timestamp,
                elapsed_seconds=record["elapsed_seconds"],
                event=event,
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, EventError):
                raise
            raise EventError("runtime event fields are invalid") from exc


EventObserver = Callable[[ResearchEvent], None]


def null_observer(_: ResearchEvent) -> None:
    """Accept an event without side effects."""


def emit_event(observer: EventObserver | None, **values: Any) -> None:
    """Avoid constructing telemetry when monitoring is disabled."""
    if observer is not None:
        observer(ResearchEvent(**values))


def emit_artifact_verified(
    observer: EventObserver | None, receipt: Mapping[str, Any]
) -> None:
    """Publish only the safe identity fields from a verifier receipt."""
    payload = {key: receipt.get(key) for key in ("run_id", "status")}
    if any(not isinstance(value, str) or not value for value in payload.values()):
        raise EventError("verifier receipt identity is invalid")
    emit_event(
        observer,
        kind="artifact_verified",
        stage="verification",
        visibility="decision",
        payload=payload,
    )


def bind_event_context(
    observer: EventObserver | None,
    *,
    market: str | None = None,
    model: str | None = None,
    delay: int | None = None,
) -> EventObserver | None:
    """Attach trusted runner context without overwriting event-owned values."""
    if observer is None:
        return None

    def contextual_observer(event: ResearchEvent) -> None:
        updates: dict[str, Any] = {}
        for name, value in (("market", market), ("model", model), ("delay", delay)):
            existing = getattr(event, name)
            if value is not None and existing not in (None, value):
                raise EventError(f"event {name} conflicts with bound context")
            if value is not None and existing is None:
                updates[name] = value
        observer(replace(event, **updates))

    return contextual_observer


def _require_name(value: object, label: str) -> None:
    if not isinstance(value, str) or _EVENT_NAME.fullmatch(value) is None:
        raise EventError(f"{label} must be lower snake case")


def _validate_progress(completed: object, total: object) -> None:
    if (completed is None) != (total is None):
        raise EventError("completed and total must be provided together")
    if completed is None:
        return
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (completed, total)
    ):
        raise EventError("progress values must be integers")
    if completed < 0 or total < 0 or completed > total:
        raise EventError("progress values are inconsistent")
