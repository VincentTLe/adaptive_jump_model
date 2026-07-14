import time
from pathlib import Path

import pytest

from adaptive_jump.monitor import server
from adaptive_jump.monitor.server import (
    MonitorServerError,
    WorkerSupervisor,
    build_monitor_application,
    run_monitor_server,
)

ROOT = Path(__file__).resolve().parents[1]


class _Worker:
    def __init__(self):
        self.recovered = 0
        self.runs = 0

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
    assert not supervisor.alive and supervisor.failure is None
    with pytest.raises(MonitorServerError, match="already"):
        supervisor.start()


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
        config, _environment(), worker_idle_seconds=0.01
    )

    assert tuple(application.services.queue.studies) == (
        "jm-train-window-sensitivity-001",
    )
    assert application.services.queue.database == (
        tmp_path / "artifacts/.monitor/control.sqlite3"
    )
    assert application.supervisor.alive is False


def test_server_binds_only_loopback_and_rejects_invalid_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}
    fake = type("Application", (), {"app": object()})()
    monkeypatch.setattr(server, "build_monitor_application", lambda _path: fake)
    monkeypatch.setattr(
        server.uvicorn, "run", lambda app, **values: captured.update(values)
    )

    assert run_monitor_server(ROOT / "research.toml", 8765) == 0
    assert captured["host"] == "127.0.0.1" and captured["port"] == 8765
    assert captured["proxy_headers"] is False
    with pytest.raises(MonitorServerError, match="port"):
        run_monitor_server(ROOT / "research.toml", 0)
    with pytest.raises(MonitorServerError, match="research.toml"):
        build_monitor_application(tmp_path / "missing.toml", _environment())
