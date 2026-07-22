"""Atomic identity-bound checkpoints for trusted local research processes."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
_NAME = re.compile(r"[a-z][a-z0-9_]*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class CheckpointStoreError(RuntimeError):
    """Raised when an operational checkpoint is incomplete or inconsistent."""


def save_checkpoint(
    stem: Path,
    value: Any,
    *,
    kind: str,
    identity: Mapping[str, str],
) -> None:
    """Atomically point a stage checkpoint at one content-addressed payload."""
    normalized = _validate(stem, kind, identity)
    payload = pickle.dumps(value, protocol=5)
    digest = hashlib.sha256(payload).hexdigest()
    payload_path = stem.parent / f"{stem.name}.{digest}.pkl"
    stem.parent.mkdir(parents=True, exist_ok=True)
    if payload_path.exists():
        if hashlib.sha256(payload_path.read_bytes()).hexdigest() != digest:
            raise CheckpointStoreError(f"checkpoint payload collision: {stem}")
    else:
        _atomic_write(payload_path, payload)
    document = {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "payload_sha256": digest,
        "identity": normalized,
    }
    metadata = (
        json.dumps(
            document, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        + b"\n"
    )
    _atomic_write(stem.with_suffix(".json"), metadata)
    for obsolete in stem.parent.glob(f"{stem.name}.*.pkl"):
        if obsolete != payload_path:
            obsolete.unlink()


def load_checkpoint(
    stem: Path,
    *,
    kind: str,
    identity: Mapping[str, str],
) -> Any | None:
    """Load the active generation only when schema, kind, identity, and hash match."""
    normalized = _validate(stem, kind, identity)
    metadata_path = stem.with_suffix(".json")
    if not metadata_path.exists():
        return None
    try:
        document = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointStoreError(f"invalid checkpoint metadata: {stem}") from exc
    expected = {"schema_version", "kind", "payload_sha256", "identity"}
    if not isinstance(document, dict) or set(document) != expected:
        raise CheckpointStoreError(f"checkpoint schema mismatch: {stem}")
    digest = document["payload_sha256"]
    version = document["schema_version"]
    if type(version) is not int or version != SCHEMA_VERSION:
        raise CheckpointStoreError(f"checkpoint schema mismatch: {stem}")
    if document["kind"] != kind or document["identity"] != normalized:
        raise CheckpointStoreError(f"checkpoint identity mismatch: {stem}")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise CheckpointStoreError(f"checkpoint hash is invalid: {stem}")
    payload_path = stem.parent / f"{stem.name}.{digest}.pkl"
    if not payload_path.is_file():
        raise CheckpointStoreError(f"checkpoint payload is missing: {stem}")
    payload = payload_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise CheckpointStoreError(f"checkpoint hash mismatch: {stem}")
    try:
        return pickle.loads(payload)  # noqa: S301 - trusted local runtime state only
    except (
        AttributeError,
        EOFError,
        ImportError,
        IndexError,
        pickle.UnpicklingError,
    ) as exc:
        raise CheckpointStoreError(f"checkpoint payload is invalid: {stem}") from exc


def clear_checkpoint(stem: Path) -> None:
    """Remove every generation for one internal stage checkpoint."""
    stem.with_suffix(".json").unlink(missing_ok=True)
    for payload in stem.parent.glob(f"{stem.name}.*.pkl"):
        payload.unlink()


def _validate(stem: Path, kind: str, identity: Mapping[str, str]) -> dict[str, str]:
    if stem.suffix or not isinstance(kind, str) or _NAME.fullmatch(kind) is None:
        raise CheckpointStoreError("checkpoint stem or kind is invalid")
    if not isinstance(identity, Mapping) or not identity:
        raise CheckpointStoreError("checkpoint identity must not be empty")
    normalized = dict(sorted(identity.items()))
    if any(
        not isinstance(key, str)
        or _NAME.fullmatch(key) is None
        or not isinstance(value, str)
        or not value
        for key, value in normalized.items()
    ):
        raise CheckpointStoreError("checkpoint identity fields are invalid")
    return normalized


def _atomic_write(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}-", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
