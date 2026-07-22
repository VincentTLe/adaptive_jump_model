import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import psutil
import pytest

from adaptive_jump.monitor import worker as worker_module
from adaptive_jump.monitor.event_store import EventStore, EventStoreError
from adaptive_jump.monitor.queue import QueueStore, StudyDefinition
from adaptive_jump.monitor.worker import ResearchWorker
from adaptive_jump.runtime.child_events import (
    EVENT_FD_ENV,
    child_observer_from_environment,
)
from adaptive_jump.runtime.events import ResearchEvent

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


def test_worker_forwards_validated_child_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue, config = _setup(tmp_path)
    code = (
        "from adaptive_jump.runtime.child_events import "
        "child_observer_from_environment; "
        "from adaptive_jump.runtime.events import ResearchEvent; "
        "observer=child_observer_from_environment(); "
        "observer(ResearchEvent('terminal_state','fixed_jm',"
        "visibility='decision',market='us',payload={'state':1}))"
    )
    monkeypatch.setattr(
        worker_module, "_canonical_command", lambda *_: (sys.executable, "-c", code)
    )
    store = EventStore(tmp_path / "artifacts/.monitor")
    job = queue.enqueue("study-a")

    result = ResearchWorker(queue, config, store.observer, poll_seconds=0.01).run_once()

    forwarded = [
        item.event
        for item in store.replay(job.job_id)
        if item.event.stage == "fixed_jm"
    ]
    assert result is not None and result.status == "succeeded"
    assert forwarded == [
        ResearchEvent(
            "terminal_state",
            "fixed_jm",
            visibility="decision",
            market="us",
            payload={"state": 1},
        )
    ]


def test_child_observer_disables_broken_transport() -> None:
    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    observer = child_observer_from_environment({EVENT_FD_ENV: str(write_fd)})
    event = ResearchEvent("terminal_state", "fixed_jm", payload={"state": 1})

    assert observer is not None
    observer(event)
    observer(event)
    with pytest.raises(OSError):
        os.fstat(write_fd)


def test_worker_fails_closed_on_malformed_child_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue, config = _setup(tmp_path)
    code = (
        "import os; fd=int(os.environ['ADAPTIVE_JUMP_EVENT_FD']); os.write(fd,b'{}\\n')"
    )
    monkeypatch.setattr(
        worker_module, "_canonical_command", lambda *_: (sys.executable, "-c", code)
    )
    store = EventStore(tmp_path / "artifacts/.monitor")
    queue.enqueue("study-a")

    result = ResearchWorker(queue, config, store.observer, poll_seconds=0.01).run_once()

    assert result is not None and result.status == "interrupted"


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


def test_worker_preserves_checkpoint_when_parent_event_persistence_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue, config = _setup(tmp_path)
    ready = tmp_path / "event-failure-ready"
    checkpoint = tmp_path / "event-failure-checkpoint"
    code = f"""
import signal
import time
from pathlib import Path

def stop(_signum, _frame):
    raise SystemExit(130)

signal.signal(signal.SIGINT, stop)
Path({str(checkpoint)!r}).write_text("checkpoint")
Path({str(ready)!r}).write_text("ready")
while True:
    time.sleep(0.05)
"""
    monkeypatch.setattr(
        worker_module, "_canonical_command", lambda *_: (sys.executable, "-c", code)
    )

    def observer(event: ResearchEvent) -> None:
        if event.kind == "resource_sample" and ready.exists():
            raise EventStoreError("event store unavailable")

    job = queue.enqueue("study-a")
    worker = ResearchWorker(
        queue,
        config,
        lambda _job_id: observer,
        poll_seconds=0.01,
        grace_seconds=(0.5, 0.1, 0.1),
    )

    with pytest.raises(EventStoreError, match="event store unavailable"):
        worker.run_once()

    assert checkpoint.read_text() == "checkpoint"
    assert queue.get(job.job_id).status == "interrupted"


def test_recovery_rejects_reused_pid_identity(tmp_path: Path) -> None:
    queue, config = _setup(tmp_path)
    job = queue.enqueue("study-a")
    queue.claim_next()
    process = psutil.Process()
    queue.attach_process(job.job_id, process.pid, process.create_time() + 100)
    recovered = ResearchWorker(queue, config, poll_seconds=0.01).recover()
    assert recovered is not None and recovered.status == "interrupted"


def test_recovery_terminates_matching_orphan(tmp_path: Path) -> None:
    queue, config = _setup(tmp_path)
    ready = tmp_path / "orphan-ready"
    child = subprocess.Popen(
        (
            sys.executable,
            "-c",
            "import signal,sys,time; from pathlib import Path; "
            "signal.signal(signal.SIGINT, lambda *_: sys.exit(0)); "
            f"Path({str(ready)!r}).write_text('ready'); time.sleep(60)",
        ),
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    reaper = threading.Thread(target=child.wait, daemon=True)
    reaper.start()
    try:
        _wait_for(ready.exists)
        job = queue.enqueue("study-a")
        queue.claim_next()
        process = psutil.Process(child.pid)
        queue.attach_process(job.job_id, child.pid, process.create_time())

        recovered = ResearchWorker(queue, config, poll_seconds=0.01).recover()

        assert recovered is not None and recovered.status == "interrupted"
        reaper.join(timeout=1)
        assert not reaper.is_alive() and child.returncode == 0
        assert not psutil.pid_exists(child.pid)
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
        reaper.join(timeout=1)
