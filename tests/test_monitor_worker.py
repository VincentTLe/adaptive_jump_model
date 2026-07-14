import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import psutil
import pytest

from adaptive_jump.monitor import worker as worker_module
from adaptive_jump.monitor.event_store import EventStore
from adaptive_jump.monitor.events import ResearchEvent
from adaptive_jump.monitor.queue import QueueStore, StudyDefinition
from adaptive_jump.monitor.worker import ResearchWorker

STUDIES = {"study-a": StudyDefinition("study-a", "replication")}


def _setup(tmp_path: Path) -> tuple[QueueStore, Path]:
    config = tmp_path / "research.toml"
    config.write_text("# harmless worker fixture\n")
    database = tmp_path / "artifacts/.monitor/control.sqlite3"
    return QueueStore(database, STUDIES), config


def _wait_for(predicate: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 3
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert predicate()


def test_worker_records_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue, config = _setup(tmp_path)
    command = worker_module._canonical_command(STUDIES["study-a"], config)
    assert Path(command[0]).name == "adaptive-jump"
    assert command[1:] == ("run", "--study", "replication", "--config", str(config))
    helper = (sys.executable, "-c", "import time; print('done'); time.sleep(.08)")
    monkeypatch.setattr(worker_module, "_canonical_command", lambda *_: helper)
    store = EventStore(tmp_path / "artifacts/.monitor")
    job = queue.enqueue("study-a")
    result = ResearchWorker(queue, config, store.observer, poll_seconds=0.01).run_once()
    events = store.replay(job.job_id)
    assert result is not None and result.status == "succeeded" and result.exit_code == 0
    assert result.process_pid and result.process_created_at
    kinds = {runtime.event.kind for runtime in events}
    assert {
        "process_started",
        "resource_sample",
        "process_finished",
    } <= kinds
    log = tmp_path / f"artifacts/.monitor/jobs/{job.job_id}/process.log"
    assert log.read_text().strip() == "done"


def test_worker_escalates_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue, config = _setup(tmp_path)
    ready = tmp_path / "ready"
    code = (
        "import signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGINT, signal.SIG_IGN); "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"Path({str(ready)!r}).write_text('ready'); "
        "time.sleep(60)"
    )
    monkeypatch.setattr(
        worker_module, "_canonical_command", lambda *_: (sys.executable, "-c", code)
    )
    events: list[ResearchEvent] = []
    worker = ResearchWorker(
        queue,
        config,
        lambda _job_id: events.append,
        poll_seconds=0.01,
        grace_seconds=(0.03,) * 3,
    )
    job = queue.enqueue("study-a")
    results = []
    thread = threading.Thread(target=lambda: results.append(worker.run_once()))
    thread.start()
    _wait_for(ready.exists)
    queue.request_cancel(job.job_id)
    thread.join(timeout=3)
    assert not thread.is_alive()
    assert results[0].status == "canceled"
    signals = [e.payload["signal"] for e in events if e.kind == "cancellation_signal"]
    assert signals == ["SIGINT", "SIGTERM", "SIGKILL"]


def test_recovery_rejects_reused_pid_identity(tmp_path: Path) -> None:
    queue, config = _setup(tmp_path)
    job = queue.enqueue("study-a")
    queue.claim_next()
    process = psutil.Process()
    queue.attach_process(job.job_id, process.pid, process.create_time() + 100)
    recovered = ResearchWorker(queue, config, poll_seconds=0.01).recover()
    assert recovered is not None and recovered.status == "interrupted"
