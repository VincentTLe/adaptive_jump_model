"""Validated event transport from one trusted CLI child to its monitor parent."""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Mapping
from datetime import date
from typing import Any

from adaptive_jump.runtime.events import EventError, EventObserver, ResearchEvent

EVENT_FD_ENV = "ADAPTIVE_JUMP_EVENT_FD"
WIRE_SCHEMA_VERSION = 1
MAX_EVENT_BYTES = 65_536
_WIRE_FIELDS = {
    "schema_version",
    "kind",
    "stage",
    "visibility",
    "market",
    "model",
    "delay",
    "date",
    "completed",
    "total",
    "payload",
}


class ChildEventError(RuntimeError):
    """Raised when child event setup or inbound validation is invalid."""


def child_observer_from_environment(
    environ: Mapping[str, str] | None = None,
) -> EventObserver | None:
    """Return an observer for the inherited monitor pipe, if one was supplied."""
    value = (os.environ if environ is None else environ).get(EVENT_FD_ENV)
    if value is None:
        return None
    if not value.isascii() or not value.isdecimal() or int(value) <= 2:
        raise ChildEventError("monitor event descriptor is invalid")
    descriptor = int(value)
    try:
        os.fstat(descriptor)
    except OSError as exc:
        raise ChildEventError("monitor event descriptor is unavailable") from exc
    lock = threading.Lock()

    def observe(event: ResearchEvent) -> None:
        nonlocal descriptor
        encoded = _encode_event(event)
        with lock:
            if descriptor < 0:
                return
            view = memoryview(encoded)
            while view:
                try:
                    written = os.write(descriptor, view)
                except OSError:
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass
                    descriptor = -1
                    return
                if written < 1:
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass
                    descriptor = -1
                    return
                view = view[written:]

    return observe


class ParentEventPipe:
    """Own the parent reader while exposing only one inherited child descriptor."""

    def __init__(self, observer: EventObserver) -> None:
        self._observer = observer
        self._read_fd, self.write_fd = os.pipe()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def start(self) -> None:
        """Close the parent's writer copy and begin validated event forwarding."""
        if self._thread is not None:
            raise ChildEventError("monitor event reader is already started")
        os.close(self.write_fd)
        self.write_fd = -1
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def close_unstarted(self) -> None:
        """Release descriptors after a child launch failure."""
        if self.write_fd >= 0:
            os.close(self.write_fd)
            self.write_fd = -1
        if self._thread is None and self._read_fd >= 0:
            os.close(self._read_fd)
            self._read_fd = -1

    def check(self) -> None:
        """Fail the supervised run if its telemetry stream became invalid."""
        if self._error is not None:
            raise ChildEventError("child event stream is invalid") from self._error

    def finish(self) -> None:
        """Wait for EOF and surface any validation or persistence failure."""
        if self._thread is None:
            self.close_unstarted()
            return
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            raise ChildEventError("child event stream did not close")
        self.check()

    def _read(self) -> None:
        buffer = bytearray()
        try:
            with os.fdopen(self._read_fd, "rb", buffering=0) as source:
                self._read_fd = -1
                while chunk := source.read(8192):
                    buffer.extend(chunk)
                    if len(buffer) > MAX_EVENT_BYTES and b"\n" not in buffer:
                        raise ChildEventError("child event exceeds the size limit")
                    while (newline := buffer.find(b"\n")) >= 0:
                        line = bytes(buffer[:newline])
                        del buffer[: newline + 1]
                        if not line or len(line) > MAX_EVENT_BYTES:
                            raise ChildEventError("child event line is invalid")
                        self._observer(_decode_event(line))
                if buffer:
                    raise ChildEventError("child event stream ended mid-record")
        except BaseException as exc:
            self._error = exc


def _encode_event(event: ResearchEvent) -> bytes:
    if not isinstance(event, ResearchEvent):
        raise ChildEventError("only ResearchEvent values may use the child pipe")
    record = {
        "schema_version": WIRE_SCHEMA_VERSION,
        "kind": event.kind,
        "stage": event.stage,
        "visibility": event.visibility,
        "market": event.market,
        "model": event.model,
        "delay": event.delay,
        "date": event.date.isoformat() if event.date else None,
        "completed": event.completed,
        "total": event.total,
        "payload": event.payload,
    }
    encoded = (
        json.dumps(
            record, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        + b"\n"
    )
    if len(encoded) > MAX_EVENT_BYTES:
        raise ChildEventError("child event exceeds the size limit")
    return encoded


def _decode_event(encoded: bytes) -> ResearchEvent:
    try:
        record: Any = json.loads(encoded)
        version = record.get("schema_version") if isinstance(record, dict) else None
        if (
            not isinstance(record, dict)
            or set(record) != _WIRE_FIELDS
            or isinstance(version, bool)
            or version != WIRE_SCHEMA_VERSION
        ):
            raise ChildEventError("child event schema does not match version 1")
        event_date = date.fromisoformat(record["date"]) if record["date"] else None
        return ResearchEvent(
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
    except ChildEventError:
        raise
    except (EventError, KeyError, TypeError, UnicodeDecodeError, ValueError) as exc:
        raise ChildEventError("child event fields are invalid") from exc
