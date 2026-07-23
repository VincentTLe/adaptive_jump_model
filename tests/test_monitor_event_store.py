import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from adaptive_jump.monitor.event_store import EventStore, EventStoreError
from adaptive_jump.runtime.events import ResearchEvent

JOB_ID = "a" * 32


def test_event_store_appends_replays_and_resumes_sequence(tmp_path: Path) -> None:
    root = tmp_path / "artifacts/.monitor"
    store = EventStore(root)
    first = store.append(JOB_ID, ResearchEvent("stage_started", "hmm"))
    second = store.append(
        JOB_ID,
        ResearchEvent("progress", "hmm", completed=2, total=10),
    )

    reopened = EventStore(root)
    third = reopened.append(JOB_ID, ResearchEvent("stage_finished", "hmm"))

    assert (first.sequence, second.sequence, third.sequence) == (1, 2, 3)
    assert first.elapsed_seconds <= second.elapsed_seconds <= third.elapsed_seconds
    assert reopened.replay(JOB_ID, after_sequence=1) == (second, third)
    journal = root / f"jobs/{JOB_ID}/events.jsonl"
    assert len(journal.read_text().splitlines()) == 3


def test_event_store_rejects_gaps_partial_records_and_external_writes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts/.monitor"
    store = EventStore(root)
    store.append(JOB_ID, ResearchEvent("stage_started", "hmm"))
    path = root / f"jobs/{JOB_ID}/events.jsonl"
    record = json.loads(path.read_text())
    record["sequence"] = 3
    path.write_text(json.dumps(record) + "\n")

    with pytest.raises(EventStoreError, match="changed outside"):
        store.append(JOB_ID, ResearchEvent("progress", "hmm"))
    with pytest.raises(EventStoreError, match="inconsistent"):
        EventStore(root).replay(JOB_ID)

    path.write_text(json.dumps(record))
    with pytest.raises(EventStoreError, match="partial record"):
        EventStore(root).replay(JOB_ID)


def test_event_store_serializes_concurrent_observers(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "artifacts/.monitor")
    observer = store.observer(JOB_ID)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(
            pool.map(
                lambda _: observer(ResearchEvent("resource_sample", "worker")),
                range(40),
            )
        )
    assert tuple(event.sequence for event in store.replay(JOB_ID)) == tuple(
        range(1, 41)
    )


@pytest.mark.parametrize(("job_id", "after"), [("../escape", 0), (JOB_ID, -1)])
def test_event_store_rejects_unsafe_replay_inputs(
    tmp_path: Path, job_id: str, after: int
) -> None:
    with pytest.raises(EventStoreError):
        EventStore(tmp_path / "artifacts/.monitor").replay(job_id, after)
