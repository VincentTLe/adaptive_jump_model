import os
import signal
import sys
import time
from pathlib import Path

import psutil
import pytest

from adaptive_jump.monitor import server
from adaptive_jump.monitor import worker as worker_module
from adaptive_jump.monitor.queue import QueueStore, StudyDefinition
from adaptive_jump.monitor.security import AccessAuthenticator, LocalAuthenticator
from adaptive_jump.monitor.server import (
    MonitorServerError,
    WorkerSupervisor,
    build_monitor_application,
    run_monitor_server,
)
from adaptive_jump.monitor.worker import ResearchWorker

ROOT = Path(__file__).resolve().parents[1]


class _Worker:
    def __init__(self):
        self.recovered = 0
        self.runs = 0
        self.shutdown_requested = False

    @property
    def shutdown_timeout(self):
        return 0.1

    def request_shutdown(self):
        self.shutdown_requested = True

    def recover(self):
        self.recovered += 1

    def run_once(self):
        self.runs += 1
        return None


def _environment() -> dict[str, str]:
    import base64

    return {
        "ADAPTIVE_JUMP_ACCESS_ISSUER": "https://research.cloudflareaccess.com",
        "ADAPTIVE_JUMP_ACCESS_AUDIENCE": "audience",
        "ADAPTIVE_JUMP_OWNER_EMAIL": "owner@example.com",
        "ADAPTIVE_JUMP_VIEWER_EMAILS": "advisor@example.com",
        "ADAPTIVE_JUMP_MONITOR_ORIGIN": "http://127.0.0.1:8765",
        "ADAPTIVE_JUMP_CSRF_SECRET": base64.urlsafe_b64encode(b"x" * 32)
        .rstrip(b"=")
        .decode(),
    }


def test_worker_supervisor_recovers_once_and_polls_one_thread() -> None:
    worker = _Worker()
    supervisor = WorkerSupervisor(worker, idle_seconds=0.01)

    supervisor.start()
    deadline = time.monotonic() + 1
    while worker.runs < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    supervisor.stop()

    assert worker.recovered == 1 and worker.runs >= 2
    assert worker.shutdown_requested
    assert not supervisor.alive and supervisor.failure is None
    with pytest.raises(MonitorServerError, match="already"):
        supervisor.start()


def test_supervisor_shutdown_preserves_checkpoint_then_resume_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "research.toml"
    config.write_text("# monitor lifecycle fixture\n")
    studies = {"study-a": StudyDefinition("study-a", "replication")}
    queue = QueueStore(tmp_path / "artifacts/.monitor/control.sqlite3", studies)
    checkpoint = tmp_path / "checkpoint"
    ready = tmp_path / "ready"
    resumed = tmp_path / "resumed"
    code = f"""
import signal
import time
from pathlib import Path

checkpoint = Path({str(checkpoint)!r})
if checkpoint.exists():
    Path({str(resumed)!r}).write_text("resumed")
    raise SystemExit(0)

def stop(_signum, _frame):
    raise SystemExit(130)

signal.signal(signal.SIGINT, stop)
checkpoint.write_text("checkpoint")
Path({str(ready)!r}).write_text("ready")
while True:
    time.sleep(0.05)
"""
    monkeypatch.setattr(
        worker_module, "_canonical_command", lambda *_: (sys.executable, "-c", code)
    )
    job = queue.enqueue("study-a")
    worker = ResearchWorker(
        queue,
        config,
        poll_seconds=0.01,
        grace_seconds=(0.5, 0.1, 0.1),
    )
    supervisor = WorkerSupervisor(worker, idle_seconds=0.01)

    supervisor.start()
    resumed_supervisor: WorkerSupervisor | None = None
    try:
        _wait_for(ready.exists)
        running = queue.get(job.job_id)
        supervisor.stop()

        interrupted = queue.get(job.job_id)
        assert interrupted.status == "interrupted"
        assert checkpoint.read_text() == "checkpoint"
        assert running.process_pid is not None
        assert not psutil.pid_exists(running.process_pid)

        queue.resume(job.job_id)
        resumed_supervisor = WorkerSupervisor(
            ResearchWorker(queue, config, poll_seconds=0.01), idle_seconds=0.01
        )
        resumed_supervisor.start()
        _wait_for(lambda: queue.get(job.job_id).status == "succeeded")
        resumed_supervisor.stop()

        final = queue.get(job.job_id)
        assert resumed.read_text() == "resumed"
        assert final.status == "succeeded" and final.attempts == 2
        assert supervisor.failure is None and resumed_supervisor.failure is None
    finally:
        worker.request_shutdown()
        if resumed_supervisor is not None:
            resumed_supervisor.worker.request_shutdown()
        active = queue.active()
        if active is not None and active.process_pid is not None:
            try:
                os.killpg(active.process_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for current in (supervisor, resumed_supervisor):
            if current is not None:
                try:
                    current.stop()
                except MonitorServerError:
                    pass


def _wait_for(predicate, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert predicate()


def test_application_uses_canonical_paths_and_frozen_only_catalog(
    tmp_path: Path,
) -> None:
    (tmp_path / "research").mkdir()
    config = tmp_path / "research.toml"
    config.write_text("# monitor fixture\n")
    (tmp_path / "research/experiment_registry.jsonl").write_text(
        '{"experiment_id":"fixed-baselines-001-v7","status":"FROZEN"}\n'
        '{"experiment_id":"fixed-baselines-001-v7","status":"EXPERIMENT_COMPLETE"}\n'
        '{"experiment_id":"jm-train-window-sensitivity-001","status":"FROZEN"}\n'
    )

    application = build_monitor_application(
        config,
        _environment(),
        access_mode="cloudflare",
        worker_idle_seconds=0.01,
    )

    assert tuple(application.services.queue.studies) == (
        "jm-train-window-sensitivity-001",
    )
    assert application.services.queue.database == (
        tmp_path / "artifacts/.monitor/control.sqlite3"
    )
    assert application.supervisor.alive is False
    assert isinstance(application.services.authenticator, AccessAuthenticator)
    assert application.local_password is None

    local = build_monitor_application(
        config,
        {},
        local_password="correct-local-password",
        worker_idle_seconds=0.01,
    )
    assert isinstance(local.services.authenticator, LocalAuthenticator)
    assert local.local_password == "correct-local-password"
    assert local.services.request_security.config.public_origin == (
        "http://127.0.0.1:8765"
    )


def test_server_binds_only_loopback_and_rejects_invalid_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    captured = {}
    access_modes = []

    def fake_build(_path, **values):
        access_modes.append(values["access_mode"])
        password = (
            "generated-local-password" if values["access_mode"] == "local" else None
        )
        return type("Application", (), {"app": object(), "local_password": password})()

    monkeypatch.setattr(server, "build_monitor_application", fake_build)
    monkeypatch.setattr(
        server.uvicorn, "run", lambda app, **values: captured.update(values)
    )

    assert run_monitor_server(ROOT / "research.toml", 8765) == 0
    output = capsys.readouterr().out
    assert "http://127.0.0.1:8765" in output
    assert "Username: owner" in output and "generated-local-password" in output
    assert access_modes == ["local"]
    assert captured["host"] == "127.0.0.1" and captured["port"] == 8765
    assert captured["proxy_headers"] is False

    monkeypatch.setenv("ADAPTIVE_JUMP_MONITOR_ACCESS", "cloudflare")
    assert run_monitor_server(ROOT / "research.toml", 8765) == 0
    assert access_modes[-1] == "cloudflare"
    assert capsys.readouterr().out == ""
    with pytest.raises(MonitorServerError, match="port"):
        run_monitor_server(ROOT / "research.toml", 0)
    with pytest.raises(MonitorServerError, match="access mode"):
        run_monitor_server(ROOT / "research.toml", access_mode="public")
    with pytest.raises(MonitorServerError, match="research.toml"):
        build_monitor_application(tmp_path / "missing.toml", _environment())
