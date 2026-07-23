"""Single-process supervisor for canonical frozen-study CLI runs."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import psutil

from adaptive_jump.monitor.event_store import EventStoreError
from adaptive_jump.monitor.queue import Job, QueueStore, StudyDefinition
from adaptive_jump.runtime.child_events import (
    EVENT_FD_ENV,
    ChildEventError,
    ParentEventPipe,
)
from adaptive_jump.runtime.events import EventObserver, emit_event


class WorkerError(RuntimeError):
    pass


ObserverFactory = Callable[[str], EventObserver]


class ResearchWorker:
    def __init__(
        self,
        queue: QueueStore,
        config_path: Path,
        observer_factory: ObserverFactory | None = None,
        *,
        poll_seconds: float = 1.0,
        grace_seconds: tuple[float, float, float] = (30.0, 10.0, 5.0),
    ) -> None:
        self.queue = queue
        self.config_path = config_path.resolve()
        self.root = self.config_path.parent
        self.runtime_root = self.root / "artifacts" / ".monitor"
        self.observer_factory = observer_factory
        self.observer: EventObserver | None = None
        if (
            self.config_path.name != "research.toml"
            or not self.config_path.is_file()
            or self.queue.database.resolve().parent != self.runtime_root
        ):
            raise WorkerError("worker paths must use the canonical project layout")
        self.poll_seconds = float(poll_seconds)
        self.grace_seconds = tuple(float(value) for value in grace_seconds)
        self._shutdown = threading.Event()
        self._pending_event_error: str | None = None

    @property
    def shutdown_timeout(self) -> float:
        """Maximum graceful stop time plus polling and scheduling headroom."""
        return sum(self.grace_seconds) + 5.0 + self.poll_seconds + 1.0

    def request_shutdown(self) -> None:
        """Ask the active child to stop through the graceful signal path."""
        self._shutdown.set()

    def run_once(self) -> Job | None:
        if self._shutdown.is_set():
            return None
        job = self.queue.claim_next()
        if job is None:
            return None
        child: subprocess.Popen[bytes] | None = None
        event_pipe: ParentEventPipe | None = None
        try:
            self._bind_observer(job.job_id)
            current = self.queue.get(job.job_id)
            if current.status == "cancel_requested":
                return self.queue.finish(job.job_id, "canceled")
            if self._shutdown.is_set():
                return self.queue.recover_abandoned()[0]
            definition = self.queue.studies[job.study_id]
            command = _canonical_command(definition, self.config_path)
            log_path = self.runtime_root / "jobs" / job.job_id / "process.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("ab", buffering=0) as log:
                environment = os.environ.copy()
                environment.pop(EVENT_FD_ENV, None)
                pass_fds: tuple[int, ...] = ()
                if self.observer is not None:
                    event_pipe = ParentEventPipe(self.observer)
                    environment[EVENT_FD_ENV] = str(event_pipe.write_fd)
                    pass_fds = (event_pipe.write_fd,)
                child = subprocess.Popen(
                    command,
                    cwd=self.root,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env=environment,
                    pass_fds=pass_fds,
                )
                if event_pipe is not None:
                    event_pipe.start()
                process = psutil.Process(child.pid)
                created_at = process.create_time()
                current = self.queue.attach_process(job.job_id, child.pid, created_at)
                self._emit(
                    "process_started",
                    {"pid": child.pid, "attempt": current.attempts},
                )
                process.cpu_percent(None)
                return self._monitor_child(
                    current, child, process, created_at, event_pipe
                )
        except (OSError, psutil.Error, WorkerError, ChildEventError) as exc:
            if child is not None:
                self._stop_child(child)
            if event_pipe is not None:
                try:
                    event_pipe.finish()
                except ChildEventError:
                    pass
            current = self.queue.get(job.job_id)
            status = "canceled" if current.status == "cancel_requested" else "failed"
            final = self.queue.finish(
                job.job_id,
                status,
                child.poll() if child is not None else None,
            )
            self._emit("process_launch_failed", {"error": str(exc)})
            return final
        except BaseException:
            if child is not None:
                self._stop_child(child)
            if event_pipe is not None:
                try:
                    event_pipe.finish()
                except ChildEventError:
                    pass
            self.queue.recover_abandoned()
            raise

    def recover(self) -> Job | None:
        """Terminate a matching orphan before marking its job interrupted."""
        job = self.queue.active()
        if job is None:
            return None
        self._bind_observer(job.job_id)
        process = _matching_process(job)
        if process is not None:
            self._cancel_group(process.pid, None)
        recovered = self.queue.recover_abandoned()
        if process is not None:
            self._emit(
                "orphan_terminated",
                {"pid": process.pid, "status": recovered[0].status},
            )
        return recovered[0]

    def _monitor_child(
        self,
        job: Job,
        child: subprocess.Popen[bytes],
        process: psutil.Process,
        created_at: float,
        event_pipe: ParentEventPipe | None,
    ) -> Job:
        while child.poll() is None:
            if self._shutdown.is_set():
                return self._interrupt(job.job_id, child, event_pipe, "shutdown")
            if event_pipe is not None:
                try:
                    event_pipe.check()
                except ChildEventError:
                    return self._interrupt(
                        job.job_id, child, event_pipe, "invalid_telemetry"
                    )
            current = self.queue.get(job.job_id)
            if current.status == "cancel_requested":
                code = self._cancel_group(child.pid, child)
                if event_pipe is not None:
                    try:
                        event_pipe.finish()
                    except ChildEventError:
                        pass
                final = self.queue.finish(job.job_id, "canceled", code)
                self._emit(
                    "process_finished", {"status": "canceled", "exit_code": code}
                )
                return final
            if _same_process_alive(process, created_at):
                self._sample(process)
            time.sleep(self.poll_seconds)
        code = child.wait()
        if event_pipe is not None:
            try:
                event_pipe.finish()
            except ChildEventError:
                return self._interrupt(
                    job.job_id, child, event_pipe, "invalid_telemetry"
                )
        status = "succeeded" if code == 0 else "failed"
        final = self.queue.finish(job.job_id, status, code)
        self._emit("process_finished", {"status": status, "exit_code": code})
        return final

    def _interrupt(
        self,
        job_id: str,
        child: subprocess.Popen[bytes],
        event_pipe: ParentEventPipe | None,
        reason: str,
    ) -> Job:
        code = self._cancel_group(child.pid, child)
        if event_pipe is not None:
            try:
                event_pipe.finish()
            except ChildEventError:
                pass
        recovered = self.queue.recover_abandoned()
        final = next(item for item in recovered if item.job_id == job_id)
        self._emit(
            "process_finished",
            {"status": final.status, "exit_code": code, "reason": reason},
        )
        return final

    def _cancel_group(
        self, process_group: int, child: subprocess.Popen[bytes] | None
    ) -> int | None:
        for sent_signal, grace in zip(
            (signal.SIGINT, signal.SIGTERM, signal.SIGKILL),
            self.grace_seconds,
            strict=True,
        ):
            if not _group_alive(process_group):
                break
            self._emit_control(
                "cancellation_signal",
                {"signal": sent_signal.name, "grace_seconds": grace},
            )
            try:
                os.killpg(process_group, sent_signal)
            except ProcessLookupError:
                break
            deadline = time.monotonic() + grace
            while _group_alive(process_group) and time.monotonic() < deadline:
                if child is not None:
                    child.poll()
                time.sleep(
                    min(self.poll_seconds, max(0.0, deadline - time.monotonic()))
                )
        if _group_alive(process_group):
            raise WorkerError("process group survived SIGKILL")
        return child.wait() if child is not None else None

    def _stop_child(self, child: subprocess.Popen[bytes]) -> None:
        try:
            self._cancel_group(child.pid, child)
        except (OSError, WorkerError):
            _force_kill(child)

    def _sample(self, process: psutil.Process) -> None:
        cpu_percent, rss_bytes, process_count = 0.0, 0, 0
        try:
            processes = [process, *process.children(recursive=True)]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            processes = []
        for item in processes:
            try:
                cpu_percent += item.cpu_percent(None)
                rss_bytes += item.memory_info().rss
                process_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        if process_count:
            self._emit(
                "resource_sample",
                {
                    "cpu_percent": cpu_percent,
                    "rss_bytes": rss_bytes,
                    "process_count": process_count,
                },
            )

    def _emit(self, kind: str, payload: dict[str, object]) -> None:
        if self._pending_event_error is not None:
            payload = {**payload, "prior_event_error": self._pending_event_error}
            self._pending_event_error = None
        emit_event(self.observer, kind=kind, stage="worker", payload=payload)

    def _emit_control(self, kind: str, payload: dict[str, object]) -> None:
        try:
            self._emit(kind, payload)
        except (EventStoreError, OSError) as exc:
            self._pending_event_error = str(exc)

    def _bind_observer(self, job_id: str) -> None:
        self.observer = self.observer_factory(job_id) if self.observer_factory else None


def _canonical_command(
    definition: StudyDefinition, config_path: Path
) -> tuple[str, ...]:
    executable = Path(sys.executable).with_name("adaptive-jump")
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise WorkerError("canonical adaptive-jump executable is unavailable")
    return (
        str(executable),
        "run",
        "--study",
        definition.cli_study,
        "--config",
        str(config_path),
    )


def _matching_process(job: Job) -> psutil.Process | None:
    if job.process_pid is None or job.process_created_at is None:
        return None
    try:
        process = psutil.Process(job.process_pid)
    except psutil.NoSuchProcess:
        return None
    return process if _same_process_alive(process, job.process_created_at) else None


def _same_process_alive(process: psutil.Process, created_at: float | None) -> bool:
    try:
        return (
            created_at is not None
            and abs(process.create_time() - created_at) < 0.01
            and process.is_running()
            and process.status() != psutil.STATUS_ZOMBIE
        )
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def _group_alive(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
        return True
    except ProcessLookupError:
        return False


def _force_kill(child: subprocess.Popen[bytes]) -> None:
    if child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()
