"""Loopback-only server assembly and one-worker lifecycle."""

from __future__ import annotations

import logging
import os
import secrets
import threading
from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from adaptive_jump.monitor.api import MonitorServices, create_app
from adaptive_jump.monitor.audit import AuditStore
from adaptive_jump.monitor.event_store import EventStore
from adaptive_jump.monitor.evidence import EvidenceStore
from adaptive_jump.monitor.http_security import HttpSecurityConfig, RequestSecurity
from adaptive_jump.monitor.queue import QueueStore, load_frozen_studies
from adaptive_jump.monitor.security import (
    AccessAuthenticator,
    AccessConfig,
    LocalAuthenticator,
)
from adaptive_jump.monitor.worker import ResearchWorker

_LOGGER = logging.getLogger(__name__)
DEFAULT_PORT = 8765


class MonitorServerError(ValueError):
    """Raised when the canonical monitor cannot be assembled safely."""


class WorkerSupervisor:
    """Run recovery and queue polling on exactly one background thread."""

    def __init__(self, worker: ResearchWorker, *, idle_seconds: float = 1.0) -> None:
        if idle_seconds <= 0:
            raise MonitorServerError("worker idle interval must be positive")
        self.worker = worker
        self.idle_seconds = float(idle_seconds)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.failure: Exception | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise MonitorServerError("worker supervisor is already started")
        self._thread = threading.Thread(
            target=self._run, name="research-worker", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop polling and wait for the worker to terminate its active child."""
        self.worker.request_shutdown()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.worker.shutdown_timeout)
            if self._thread.is_alive():
                raise MonitorServerError(
                    "worker did not stop within its shutdown window"
                )

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        try:
            self.worker.recover()
            while not self._stop.is_set():
                job = self.worker.run_once()
                if job is None:
                    self._stop.wait(self.idle_seconds)
        except Exception as exc:
            self.failure = exc
            _LOGGER.exception("monitor research worker stopped")


@dataclass(frozen=True)
class MonitorApplication:
    app: FastAPI
    services: MonitorServices
    supervisor: WorkerSupervisor
    local_password: str | None = field(default=None, repr=False)


def build_monitor_application(
    config_path: str | Path,
    environ: Mapping[str, str] | None = None,
    *,
    access_mode: str = "local",
    port: int = DEFAULT_PORT,
    local_password: str | None = None,
    worker_idle_seconds: float = 1.0,
) -> MonitorApplication:
    """Assemble trusted stores, security, worker, and API for one project root."""
    _validate_server_options(access_mode, port)
    config = Path(config_path).resolve()
    if config.name != "research.toml" or not config.is_file():
        raise MonitorServerError("monitor requires the canonical research.toml")
    root = config.parent
    runtime = root / "artifacts/.monitor"
    registry = root / "research/experiment_registry.jsonl"
    values = os.environ if environ is None else environ
    try:
        if access_mode == "local":
            local_password = local_password or secrets.token_urlsafe(18)
            authenticator = LocalAuthenticator(local_password)
            request_security = RequestSecurity(
                HttpSecurityConfig(f"http://127.0.0.1:{port}", secrets.token_bytes(32))
            )
        else:
            local_password = None
            authenticator = AccessAuthenticator(AccessConfig.from_environment(values))
            request_security = RequestSecurity(
                HttpSecurityConfig.from_environment(values)
            )
        studies = load_frozen_studies(registry)
        queue = QueueStore(runtime / "control.sqlite3", studies)
        events = EventStore(runtime)
        services = MonitorServices(
            queue=queue,
            events=events,
            evidence=EvidenceStore(root),
            audit=AuditStore(runtime),
            authenticator=authenticator,
            request_security=request_security,
        )
        worker = ResearchWorker(queue, config, events.observer)
    except (OSError, RuntimeError, ValueError) as exc:
        raise MonitorServerError(f"monitor setup failed: {exc}") from exc
    supervisor = WorkerSupervisor(worker, idle_seconds=worker_idle_seconds)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        supervisor.start()
        try:
            yield
        finally:
            supervisor.stop()

    app = create_app(services, lifespan=lifespan)

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        status = "failed" if supervisor.failure is not None else "ok"
        return {"status": status}

    return MonitorApplication(app, services, supervisor, local_password)


def run_monitor_server(
    config_path: str | Path,
    port: int = DEFAULT_PORT,
    *,
    access_mode: str | None = None,
) -> int:
    """Run one local-first monitor, with Cloudflare available explicitly."""
    access_mode = access_mode or os.environ.get("ADAPTIVE_JUMP_MONITOR_ACCESS", "local")
    _validate_server_options(access_mode, port)
    application = build_monitor_application(
        config_path, access_mode=access_mode, port=port
    )
    if application.local_password is not None:
        print(f"Monitor URL: http://127.0.0.1:{port}", flush=True)
        print("Username: owner", flush=True)
        print(f"Password: {application.local_password}", flush=True)
    uvicorn.run(
        application.app,
        host="127.0.0.1",
        port=port,
        proxy_headers=False,
        server_header=False,
    )
    return 0


def _validate_server_options(access_mode: str, port: int) -> None:
    if access_mode not in {"local", "cloudflare"}:
        raise MonitorServerError("monitor access mode must be local or cloudflare")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65_535:
        raise MonitorServerError("monitor port must be between 1 and 65535")
