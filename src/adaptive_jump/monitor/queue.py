"""Persistent, fail-closed control state for registered research studies."""

from __future__ import annotations

import json
import math
import re
import sqlite3
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = 2
TERMINAL_STATUSES = frozenset({"canceled", "succeeded", "failed"})
_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9-]*\Z")


class QueueError(RuntimeError):
    """Raised when queue state or study authorization is invalid."""


@dataclass(frozen=True)
class StudyDefinition:
    study_id: str
    cli_study: str


REGISTERED_STUDIES = {
    "fixed-baselines-001-v7": StudyDefinition("fixed-baselines-001-v7", "replication"),
    "jm-train-window-sensitivity-001": StudyDefinition(
        "jm-train-window-sensitivity-001", "train-window-sensitivity"
    ),
}


@dataclass(frozen=True)
class Job:
    job_id: str
    study_id: str
    status: str
    queue_position: int | None
    attempts: int
    created_at: str
    updated_at: str
    started_at: str | None
    finished_at: str | None
    process_pid: int | None
    process_created_at: float | None
    exit_code: int | None


def load_frozen_studies(
    registry_path: Path,
    definitions: Mapping[str, StudyDefinition] = REGISTERED_STUDIES,
) -> dict[str, StudyDefinition]:
    """Return code-registered studies whose final registry row is FROZEN."""
    latest: dict[str, str] = {}
    try:
        lines = registry_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise QueueError(f"cannot read experiment registry: {registry_path}") from exc
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise QueueError(f"invalid experiment registry row {number}") from exc
        if not isinstance(record, dict):
            raise QueueError(f"invalid experiment registry row {number}")
        study_id = record.get("experiment_id")
        status = record.get("status")
        if not isinstance(study_id, str) or not study_id or not isinstance(status, str):
            raise QueueError(f"invalid experiment registry row {number}")
        latest[study_id] = status
    return {
        study_id: definition
        for study_id, definition in definitions.items()
        if latest.get(study_id) == "FROZEN"
    }


class QueueStore:
    """SQLite-backed queue with at most one active research job."""

    def __init__(self, database: Path, studies: Mapping[str, StudyDefinition]) -> None:
        self.database = database
        self.studies = dict(studies)
        if any(
            not isinstance(value, StudyDefinition)
            or key != value.study_id
            or any(
                not isinstance(identifier, str)
                or _IDENTIFIER.fullmatch(identifier) is None
                for identifier in (value.study_id, value.cli_study)
            )
            for key, value in self.studies.items()
        ):
            raise QueueError("study catalog is inconsistent")
        database.parent.mkdir(parents=True, exist_ok=True)
        with self._transaction() as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version == 1:
                for statement in _MIGRATE_V1:
                    connection.execute(statement)
                connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            elif version not in (0, SCHEMA_VERSION):
                raise QueueError(f"unsupported queue schema version: {version}")
            connection.executescript(_SCHEMA)
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def enqueue(self, study_id: str) -> Job:
        self._require_study(study_id)
        now = _now()
        with self._transaction() as connection:
            position = connection.execute(
                "SELECT COALESCE(MAX(queue_position), -1) + 1 FROM jobs "
                "WHERE status = 'queued'"
            ).fetchone()[0]
            job_id = uuid.uuid4().hex
            connection.execute(
                "INSERT INTO jobs (job_id, study_id, status, queue_position, "
                "attempts, created_at, updated_at) "
                "VALUES (?, ?, 'queued', ?, 0, ?, ?)",
                (job_id, study_id, position, now, now),
            )
            return self._job(connection, job_id)

    def queued(self) -> tuple[Job, ...]:
        return self._query(
            "SELECT * FROM jobs WHERE status = 'queued' "
            "ORDER BY queue_position, created_at, job_id"
        )

    def all_jobs(self) -> tuple[Job, ...]:
        return self._query("SELECT * FROM jobs ORDER BY created_at, job_id")

    def get(self, job_id: str) -> Job:
        rows = self._query("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        if not rows:
            raise QueueError(f"unknown job: {job_id}")
        return rows[0]

    def active(self) -> Job | None:
        rows = self._query(
            "SELECT * FROM jobs WHERE status IN ('running', 'cancel_requested')"
        )
        return rows[0] if rows else None

    def reorder(self, job_ids: Sequence[str]) -> tuple[Job, ...]:
        requested = tuple(job_ids)
        with self._transaction() as connection:
            current = connection.execute(
                "SELECT job_id FROM jobs WHERE status = 'queued' "
                "ORDER BY queue_position, created_at, job_id"
            ).fetchall()
            current_ids = tuple(row[0] for row in current)
            if len(requested) != len(current_ids) or set(requested) != set(current_ids):
                raise QueueError("reorder must contain every queued job exactly once")
            if not requested:
                return ()
            connection.execute(
                "UPDATE jobs SET queue_position = -queue_position - 1 "
                "WHERE status = 'queued'"
            )
            now = _now()
            for position, job_id in enumerate(requested):
                connection.execute(
                    "UPDATE jobs SET queue_position = ?, updated_at = ? "
                    "WHERE job_id = ?",
                    (position, now, job_id),
                )
            return tuple(self._job(connection, job_id) for job_id in requested)

    def claim_next(self) -> Job | None:
        with self._transaction() as connection:
            active = connection.execute(
                "SELECT 1 FROM jobs WHERE status IN ('running', 'cancel_requested')"
            ).fetchone()
            if active is not None:
                return None
            row = connection.execute(
                "SELECT job_id, study_id FROM jobs WHERE status = 'queued' "
                "ORDER BY queue_position, created_at, job_id LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            self._require_study(row["study_id"])
            now = _now()
            connection.execute(
                "UPDATE jobs SET status = 'running', queue_position = NULL, "
                "attempts = attempts + 1, updated_at = ?, started_at = ?, "
                "finished_at = NULL, process_pid = NULL, "
                "process_created_at = NULL, exit_code = NULL WHERE job_id = ?",
                (now, now, row["job_id"]),
            )
            return self._job(connection, row["job_id"])

    def attach_process(self, job_id: str, pid: int, created_at: float) -> Job:
        if pid < 1 or created_at <= 0 or not math.isfinite(created_at):
            raise QueueError("process identity is invalid")
        with self._transaction() as connection:
            job = self._job(connection, job_id)
            if job.status not in {"running", "cancel_requested"}:
                raise QueueError(f"cannot attach a process to a {job.status} job")
            if job.process_pid is not None:
                raise QueueError("job already has a process identity")
            connection.execute(
                "UPDATE jobs SET process_pid = ?, process_created_at = ?, "
                "updated_at = ? WHERE job_id = ?",
                (pid, float(created_at), _now(), job_id),
            )
            return self._job(connection, job_id)

    def request_cancel(self, job_id: str) -> Job:
        with self._transaction() as connection:
            job = self._job(connection, job_id)
            if job.status in {"cancel_requested", "canceled"}:
                return job
            if job.status in {"succeeded", "failed"}:
                raise QueueError(f"cannot cancel a {job.status} job")
            status = "cancel_requested" if job.status == "running" else "canceled"
            now = _now()
            finished_at = now if status == "canceled" else None
            connection.execute(
                "UPDATE jobs SET status = ?, queue_position = NULL, updated_at = ?, "
                "finished_at = ? WHERE job_id = ?",
                (status, now, finished_at, job_id),
            )
            return self._job(connection, job_id)

    def finish(self, job_id: str, status: str, exit_code: int | None = None) -> Job:
        if status not in TERMINAL_STATUSES:
            raise QueueError("finish status must be canceled, succeeded, or failed")
        with self._transaction() as connection:
            job = self._job(connection, job_id)
            if job.status not in {"running", "cancel_requested"}:
                raise QueueError(f"cannot finish a {job.status} job")
            now = _now()
            connection.execute(
                "UPDATE jobs SET status = ?, updated_at = ?, finished_at = ?, "
                "exit_code = ? WHERE job_id = ?",
                (status, now, now, exit_code, job_id),
            )
            return self._job(connection, job_id)

    def recover_abandoned(self) -> tuple[Job, ...]:
        """Recover active rows after the caller establishes no child is alive."""
        with self._transaction() as connection:
            rows = connection.execute(
                "SELECT job_id, status FROM jobs "
                "WHERE status IN ('running', 'cancel_requested') ORDER BY created_at"
            ).fetchall()
            now = _now()
            for row in rows:
                status = "interrupted" if row["status"] == "running" else "canceled"
                connection.execute(
                    "UPDATE jobs SET status = ?, updated_at = ?, finished_at = ? "
                    "WHERE job_id = ?",
                    (status, now, now if status == "canceled" else None, row["job_id"]),
                )
            return tuple(self._job(connection, row["job_id"]) for row in rows)

    def resume(self, job_id: str) -> Job:
        with self._transaction() as connection:
            job = self._job(connection, job_id)
            if job.status != "interrupted":
                raise QueueError(f"cannot resume a {job.status} job")
            self._require_study(job.study_id)
            position = connection.execute(
                "SELECT COALESCE(MAX(queue_position), -1) + 1 FROM jobs "
                "WHERE status = 'queued'"
            ).fetchone()[0]
            now = _now()
            connection.execute(
                "UPDATE jobs SET status = 'queued', queue_position = ?, "
                "updated_at = ?, finished_at = NULL WHERE job_id = ?",
                (position, now, job_id),
            )
            return self._job(connection, job_id)

    def _require_study(self, study_id: str) -> StudyDefinition:
        try:
            return self.studies[study_id]
        except (KeyError, TypeError) as exc:
            raise QueueError(f"study is not registered and FROZEN: {study_id}") from exc

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _query(self, sql: str, values: tuple[object, ...] = ()) -> tuple[Job, ...]:
        connection = self._connect()
        try:
            return tuple(_as_job(row) for row in connection.execute(sql, values))
        finally:
            connection.close()

    @staticmethod
    def _job(connection: sqlite3.Connection, job_id: str) -> Job:
        row = connection.execute(
            "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        if row is None:
            raise QueueError(f"unknown job: {job_id}")
        return _as_job(row)


def _as_job(row: sqlite3.Row) -> Job:
    return Job(**dict(row))


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    study_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'cancel_requested',
        'interrupted', 'canceled', 'succeeded', 'failed')),
    queue_position INTEGER,
    attempts INTEGER NOT NULL CHECK (attempts >= 0),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    process_pid INTEGER,
    process_created_at REAL,
    exit_code INTEGER,
    CHECK ((status = 'queued') = (queue_position IS NOT NULL)),
    CHECK ((process_pid IS NULL) = (process_created_at IS NULL))
);
CREATE UNIQUE INDEX IF NOT EXISTS one_queued_position
    ON jobs(queue_position) WHERE status = 'queued';
CREATE UNIQUE INDEX IF NOT EXISTS one_active_job
    ON jobs((1)) WHERE status IN ('running', 'cancel_requested');
"""
_MIGRATE_V1 = (
    "ALTER TABLE jobs ADD COLUMN process_pid INTEGER",
    "ALTER TABLE jobs ADD COLUMN process_created_at REAL",
    "ALTER TABLE jobs ADD COLUMN exit_code INTEGER",
)
