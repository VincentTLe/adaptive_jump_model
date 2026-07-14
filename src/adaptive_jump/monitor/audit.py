"""Append-only, hash-chained audit journal for monitor mutations."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from adaptive_jump.monitor.security import Principal

SCHEMA_VERSION = 1
_GENESIS_HASH = "0" * 64
_ACTION = re.compile(r"[a-z][a-z0-9_]*\Z")
_TARGET = re.compile(r"[a-zA-Z0-9._:/-]{1,200}\Z")
_RECORD_FIELDS = {
    "schema_version",
    "sequence",
    "time_utc",
    "email",
    "role",
    "action",
    "target",
    "outcome",
    "details",
    "previous_sha256",
    "record_sha256",
}


class AuditError(RuntimeError):
    """Raised when the immutable mutation history cannot be trusted."""


@dataclass(frozen=True)
class AuditRecord:
    sequence: int
    time_utc: str
    email: str
    role: str
    action: str
    target: str
    outcome: str
    details: dict[str, Any]
    previous_sha256: str
    record_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, **self.__dict__}


class AuditStore:
    """Own the sole local append writer for authenticated control actions."""

    def __init__(self, runtime_root: Path) -> None:
        self.runtime_root = runtime_root.resolve()
        if (
            self.runtime_root.name != ".monitor"
            or self.runtime_root.parent.name != "artifacts"
        ):
            raise AuditError("audit store must use artifacts/.monitor")
        self.path = self.runtime_root / "mutations.jsonl"
        self._lock = threading.RLock()
        self._records: list[AuditRecord] | None = None
        self._file_size = 0

    def append(
        self,
        principal: Principal,
        action: str,
        target: str,
        outcome: str,
        details: dict[str, Any] | None = None,
    ) -> AuditRecord:
        """Append one accepted or rejected authenticated mutation attempt."""
        if (
            not isinstance(principal, Principal)
            or not _valid_email(principal.email)
            or principal.role not in {"owner", "viewer"}
        ):
            raise AuditError("audit principal is invalid")
        if _ACTION.fullmatch(action) is None or not _valid_target(target):
            raise AuditError("audit action or target is invalid")
        if outcome not in {"accepted", "rejected"}:
            raise AuditError("audit outcome is invalid")
        safe_details = _json_mapping(details or {})
        with self._lock:
            records = self._load()
            size = self.path.stat().st_size if self.path.exists() else 0
            if size != self._file_size:
                raise AuditError("audit journal changed outside its writer")
            previous = records[-1].record_sha256 if records else _GENESIS_HASH
            body = {
                "schema_version": SCHEMA_VERSION,
                "sequence": len(records) + 1,
                "time_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "email": principal.email,
                "role": principal.role,
                "action": action,
                "target": target,
                "outcome": outcome,
                "details": safe_details,
                "previous_sha256": previous,
            }
            digest = hashlib.sha256(_canonical(body)).hexdigest()
            record = AuditRecord(
                **{
                    key: value for key, value in body.items() if key != "schema_version"
                },
                record_sha256=digest,
            )
            payload = _canonical(record.to_dict()) + b"\n"
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("ab") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            records.append(record)
            self._file_size += len(payload)
            return record

    def replay(self) -> tuple[AuditRecord, ...]:
        """Return the complete validated chain; history is never filtered or deleted."""
        with self._lock:
            records = self._load(refresh=True)
            return tuple(records)

    def _load(self, *, refresh: bool = False) -> list[AuditRecord]:
        if self._records is not None and not refresh:
            return self._records
        records, size = _read_journal(self.path)
        if self._records is not None and size != self._file_size:
            raise AuditError("audit journal changed outside its writer")
        self._records = records
        self._file_size = size
        return records


def _read_journal(path: Path) -> tuple[list[AuditRecord], int]:
    if not path.exists():
        return [], 0
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise AuditError("cannot read audit journal") from exc
    if payload and not payload.endswith(b"\n"):
        raise AuditError("audit journal ends with a partial record")
    records: list[AuditRecord] = []
    previous = _GENESIS_HASH
    for sequence, line in enumerate(payload.splitlines(), start=1):
        try:
            document = json.loads(line)
            if not isinstance(document, dict) or set(document) != _RECORD_FIELDS:
                raise TypeError
            version = document.pop("schema_version")
            if isinstance(version, bool) or version != SCHEMA_VERSION:
                raise TypeError
            record = AuditRecord(**document)
            body = record.to_dict()
            body.pop("record_sha256")
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise AuditError(f"invalid audit record {sequence}") from exc
        if (
            not _valid_record(record, sequence, previous)
            or hashlib.sha256(_canonical(body)).hexdigest() != record.record_sha256
        ):
            raise AuditError(f"inconsistent audit record {sequence}")
        records.append(record)
        previous = record.record_sha256
    return records, len(payload)


def _valid_hash(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _valid_target(value: object) -> bool:
    return (
        isinstance(value, str)
        and _TARGET.fullmatch(value) is not None
        and ".." not in value
        and "//" not in value
        and not value.startswith("/")
    )


def _valid_record(record: AuditRecord, sequence: int, previous: str) -> bool:
    try:
        timestamp = datetime.fromisoformat(record.time_utc.replace("Z", "+00:00"))
        details = _json_mapping(record.details)
    except (AttributeError, OverflowError, TypeError, ValueError, AuditError):
        return False
    return (
        not isinstance(record.sequence, bool)
        and record.sequence == sequence
        and record.time_utc.endswith("Z")
        and timestamp.utcoffset() is not None
        and timestamp.astimezone(UTC) == timestamp
        and _valid_email(record.email)
        and record.role in {"owner", "viewer"}
        and isinstance(record.action, str)
        and _ACTION.fullmatch(record.action) is not None
        and _valid_target(record.target)
        and record.outcome in {"accepted", "rejected"}
        and details == record.details
        and record.previous_sha256 == previous
        and _valid_hash(record.previous_sha256)
        and _valid_hash(record.record_sha256)
    )


def _valid_email(value: object) -> bool:
    return (
        isinstance(value, str)
        and "@" in value
        and not any(character.isspace() for character in value)
    )


def _json_mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AuditError("audit details must be a mapping")
    try:
        encoded = json.dumps(
            value, allow_nan=False, separators=(",", ":"), sort_keys=True
        )
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise AuditError("audit details must contain finite JSON values") from exc
    return decoded


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode()
