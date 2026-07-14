import asyncio
from pathlib import Path

import pytest

from adaptive_jump.monitor.event_store import EventStore
from adaptive_jump.monitor.events import ResearchEvent
from adaptive_jump.monitor.queue import QueueStore, StudyDefinition
from adaptive_jump.monitor.sse import StreamError, resume_sequence, stream_job_events


def _fixture(tmp_path: Path):
    runtime = tmp_path / "artifacts/.monitor"
    queue = QueueStore(
        runtime / "control.sqlite3",
        {"study-a": StudyDefinition("study-a", "replication")},
    )
    store = EventStore(runtime)
    job = queue.enqueue("study-a")
    queue.claim_next()
    observer = store.observer(job.job_id)
    observer(ResearchEvent("stage_started", "hmm"))
    observer(ResearchEvent("metric_ready", "metrics", visibility="outcome"))
    observer(ResearchEvent("terminal_state", "hmm", visibility="decision"))
    queue.finish(job.job_id, "succeeded", 0)
    return queue, store, job


def _collect(queue, store, job_id, after=0):
    async def run():
        return [
            item
            async for item in stream_job_events(
                queue,
                store,
                job_id,
                after,
                lambda: asyncio.sleep(0, result=False),
                poll_seconds=0,
            )
        ]

    return asyncio.run(run())


def test_sse_replays_visible_sequences_and_closes_terminal_stream(
    tmp_path: Path,
) -> None:
    queue, store, job = _fixture(tmp_path)

    messages = _collect(queue, store, job.job_id)

    assert "id: 1" in messages[0] and "stage_started" in messages[0]
    assert "id: 3" in messages[1] and "terminal_state" in messages[1]
    assert all("metric_ready" not in message for message in messages)
    assert "event: stream_complete" in messages[2]
    assert '"status":"succeeded"' in messages[2]


def test_sse_resume_advances_past_hidden_outcome_events(tmp_path: Path) -> None:
    queue, store, job = _fixture(tmp_path)

    messages = _collect(queue, store, job.job_id, after=1)

    assert len(messages) == 2
    assert "id: 3" in messages[0]
    assert "stream_complete" in messages[1]


@pytest.mark.parametrize(
    ("header", "query", "expected"),
    [(None, 4, 4), ("", 4, 4), ("5", 0, 5), ("5", 5, 5)],
)
def test_resume_sequence_accepts_one_unambiguous_cursor(
    header, query, expected
) -> None:
    assert resume_sequence(header, query) == expected


@pytest.mark.parametrize(
    ("header", "query"),
    [("-1", 0), ("abc", 0), ("5", 4), (str(2**63), 0)],
)
def test_resume_sequence_rejects_invalid_or_conflicting_cursors(header, query) -> None:
    with pytest.raises(StreamError):
        resume_sequence(header, query)
