"""Single-process supervisor for canonical frozen-study CLI runs."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil

from adaptive_jump.monitor.events import EventObserver, emit_event
from adaptive_jump.monitor.queue import Job, QueueStore, StudyDefinition


class WorkerError(RuntimeError):
    pass


class ResearchWorker:
    def __init__(
        self,
        queue: QueueStore,
        config_path: Path,
        observer: EventObserver | None = None,
        *,
        poll_seconds: float = 1.0,
        grace_seconds: tuple[float, float, float] = (30.0, 10.0, 5.0),
    ) -> None:
        self.queue = queue
        self.config_path = config_path.resolve()
        self.root = self.config_path.parent
        self.runtime_root = self.root / "artifacts" / ".monitor"
        self.observer = observer
        if (
            self.config_path.name != "research.toml"
            or not self.config_path.is_file()
            or self.queue.database.resolve().parent != self.runtime_root
        ):
            raise WorkerError("worker paths must use the canonical project layout")
        self.poll_seconds = float(poll_seconds)
        self.grace_seconds = tuple(float(value) for value in grace_seconds)

    def run_once(self) -> Job | None:
        job = self.queue.claim_next()
        if job is None:
            return None
        child: subprocess.Popen[bytes] | None = None
        try:
            current = self.queue.get(job.job_id)
            if current.status == "cancel_requested":
                return self.queue.finish(job.job_id, "canceled")
            definition = self.queue.studies[job.study_id]
            command = _canonical_command(definition, self.config_path)
            log_path = self.runtime_root / "jobs" / job.job_id / "process.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("ab", buffering=0) as log:
                child = subprocess.Popen(
                    command,
                    cwd=self.root,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                process = psutil.Process(child.pid)
                created_at = process.create_time()
                current = self.queue.attach_process(job.job_id, child.pid, created_at)
                self._emit(
                    "process_started",
                    {"pid": child.pid, "attempt": current.attempts},
                )
                process.cpu_percent(None)
                return self._monitor_child(current, child, process, created_at)
        except (OSError, psutil.Error, WorkerError) as exc:
            if child is not None:
                _force_kill(child)
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
                _force_kill(child)
            self.queue.recover_abandoned()
            raise

    def recover(self) -> Job | None:
        """Monitor a matching orphan; otherwise conservatively mark it interrupted."""
        job = self.queue.active()
        if job is None:
            return None
        process = _matching_process(job)
        if process is None:
            recovered = self.queue.recover_abandoned()
            return recovered[0]
        self._emit("process_recovered", {"pid": process.pid})
        while _same_process_alive(process, job.process_created_at):
            current = self.queue.get(job.job_id)
            if current.status == "cancel_requested":
                self._cancel_group(process.pid, None)
                break
            self._sample(process)
            time.sleep(self.poll_seconds)
        recovered = self.queue.recover_abandoned()
        return recovered[0]

    def _monitor_child(
        self,
        job: Job,
        child: subprocess.Popen[bytes],
        process: psutil.Process,
        created_at: float,
    ) -> Job:
        while child.poll() is None:
            current = self.queue.get(job.job_id)
            if current.status == "cancel_requested":
                code = self._cancel_group(child.pid, child)
                final = self.queue.finish(job.job_id, "canceled", code)
                self._emit(
                    "process_finished", {"status": "canceled", "exit_code": code}
                )
                return final
            if _same_process_alive(process, created_at):
                self._sample(process)
            time.sleep(self.poll_seconds)
        code = child.wait()
        status = "succeeded" if code == 0 else "failed"
        final = self.queue.finish(job.job_id, status, code)
        self._emit("process_finished", {"status": status, "exit_code": code})
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
            self._emit(
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
        emit_event(self.observer, kind=kind, stage="worker", payload=payload)


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
