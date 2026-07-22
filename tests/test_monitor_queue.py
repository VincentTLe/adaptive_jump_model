from pathlib import Path

import pytest

from adaptive_jump.monitor.queue import (
    QueueError,
    QueueStore,
    StudyDefinition,
    load_frozen_studies,
)

ROOT = Path(__file__).resolve().parents[1]
STUDIES = {
    "study-a": StudyDefinition("study-a", "replication"),
    "study-b": StudyDefinition("study-b", "train-window-sensitivity"),
}


def test_project_catalog_is_empty_after_grid_evaluation() -> None:
    frozen = load_frozen_studies(ROOT / "research/experiment_registry.jsonl")
    assert tuple(frozen) == ()


def test_catalog_requires_registration_and_latest_frozen_state(tmp_path: Path) -> None:
    registry = tmp_path / "registry.jsonl"
    registry.write_text(
        '{"experiment_id":"study-a","status":"FROZEN"}\n'
        '{"experiment_id":"unknown-study","status":"FROZEN"}\n'
        '{"experiment_id":"study-a","status":"EXPERIMENT_COMPLETE"}\n'
        '{"experiment_id":"study-b","status":"FROZEN"}\n'
    )
    frozen = load_frozen_studies(registry, STUDIES)
    assert tuple(frozen) == ("study-b",)
    registry.write_text(registry.read_text() + "not-json\n")
    with pytest.raises(QueueError, match="row 5"):
        load_frozen_studies(registry, STUDIES)


def test_queue_persists_reorders_and_rejects_unregistered(tmp_path: Path) -> None:
    database = tmp_path / "monitor.sqlite3"
    queue = QueueStore(database, STUDIES)
    first = queue.enqueue("study-a")
    second = queue.enqueue("study-b")
    third = queue.enqueue("study-a")
    expected = (third.job_id, first.job_id, second.job_id)
    queue.reorder(expected)
    reopened = QueueStore(database, STUDIES)
    assert tuple(job.job_id for job in reopened.queued()) == expected
    with pytest.raises(QueueError, match="registered and FROZEN"):
        reopened.enqueue("unknown-study")
    with pytest.raises(QueueError, match="every queued job"):
        reopened.reorder((first.job_id, second.job_id))
    assert tuple(job.job_id for job in reopened.queued()) == expected


def test_queue_allows_only_one_active_job_and_retains_history(tmp_path: Path) -> None:
    queue = QueueStore(tmp_path / "monitor.sqlite3", STUDIES)
    first = queue.enqueue("study-a")
    second = queue.enqueue("study-b")
    running = queue.claim_next()
    assert running is not None
    assert running.job_id == first.job_id
    assert running.status == "running"
    assert running.attempts == 1
    assert queue.claim_next() is None

    assert queue.request_cancel(first.job_id).status == "cancel_requested"
    assert queue.finish(first.job_id, "canceled").status == "canceled"
    next_job = queue.claim_next()
    assert next_job is not None and next_job.job_id == second.job_id
    assert queue.finish(second.job_id, "succeeded").status == "succeeded"
    assert [job.status for job in queue.all_jobs()] == ["canceled", "succeeded"]
    with pytest.raises(QueueError, match="cannot cancel"):
        queue.request_cancel(second.job_id)


def test_recovery_requires_explicit_resume_and_preserves_cancel_intent(
    tmp_path: Path,
) -> None:
    queue = QueueStore(tmp_path / "monitor.sqlite3", STUDIES)
    job = queue.enqueue("study-a")
    assert queue.claim_next() is not None
    recovered = queue.recover_abandoned()
    assert len(recovered) == 1
    assert recovered[0].status == "interrupted"
    assert queue.claim_next() is None
    assert queue.resume(job.job_id).status == "queued"
    rerun = queue.claim_next()
    assert rerun is not None and rerun.attempts == 2

    queue.request_cancel(job.job_id)
    canceled = queue.recover_abandoned()
    assert len(canceled) == 1
    assert canceled[0].status == "canceled"
    with pytest.raises(QueueError, match="cannot resume"):
        queue.resume(job.job_id)
