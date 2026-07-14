import json
import threading
from pathlib import Path

import pytest

from adaptive_jump.monitor.audit import AuditError, AuditStore
from adaptive_jump.monitor.security import Principal

OWNER = Principal("owner@example.com", "owner")


def _store(tmp_path: Path) -> AuditStore:
    return AuditStore(tmp_path / "artifacts/.monitor")


def test_audit_chain_round_trip_and_restart(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = store.append(OWNER, "enqueue", "queue", "accepted", {"study": "a"})
    second = store.append(OWNER, "cancel", "jobs/abc", "rejected")

    assert first.sequence == 1 and second.sequence == 2
    assert second.previous_sha256 == first.record_sha256
    assert AuditStore(store.runtime_root).replay() == (first, second)


def test_audit_rejects_partial_tampered_and_external_writes(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.append(OWNER, "enqueue", "queue", "accepted")
    path = store.path
    original = path.read_bytes()

    path.write_bytes(original.rstrip(b"\n"))
    with pytest.raises(AuditError, match="partial"):
        AuditStore(store.runtime_root).replay()

    path.write_bytes(original)
    document = json.loads(original)
    document["action"] = "cancel"
    path.write_text(json.dumps(document) + "\n")
    with pytest.raises(AuditError, match="inconsistent"):
        AuditStore(store.runtime_root).replay()

    path.write_bytes(original + original)
    with pytest.raises(AuditError, match="outside"):
        store.append(OWNER, "enqueue", "queue", "accepted")


def test_audit_serializes_concurrent_appends(tmp_path: Path) -> None:
    store = _store(tmp_path)
    threads = [
        threading.Thread(
            target=store.append,
            args=(OWNER, "reorder", "queue", "accepted", {"index": index}),
        )
        for index in range(30)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    records = store.replay()
    assert [record.sequence for record in records] == list(range(1, 31))
    assert len({record.record_sha256 for record in records}) == 30


@pytest.mark.parametrize(
    ("action", "target", "outcome", "details"),
    [
        ("bad-action", "queue", "accepted", {}),
        ("enqueue", "../secret", "accepted", {}),
        ("enqueue", "queue", "maybe", {}),
        ("enqueue", "queue", "accepted", {"bad": float("nan")}),
    ],
)
def test_audit_rejects_invalid_record_fields(
    tmp_path: Path, action: str, target: str, outcome: str, details: dict
) -> None:
    with pytest.raises(AuditError):
        _store(tmp_path).append(OWNER, action, target, outcome, details)


def test_audit_requires_the_ignored_runtime_location(tmp_path: Path) -> None:
    with pytest.raises(AuditError, match="artifacts/.monitor"):
        AuditStore(tmp_path / "audit")
