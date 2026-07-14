import asyncio
import json
from pathlib import Path
from urllib.parse import urlsplit

from adaptive_jump.monitor.api import MonitorServices, create_app
from adaptive_jump.monitor.audit import AuditStore
from adaptive_jump.monitor.event_store import EventStore
from adaptive_jump.monitor.events import ResearchEvent
from adaptive_jump.monitor.evidence import OutcomeLocked
from adaptive_jump.monitor.http_security import HttpSecurityConfig, RequestSecurity
from adaptive_jump.monitor.queue import QueueStore, StudyDefinition
from adaptive_jump.monitor.security import AuthenticationError, Principal


class _Authenticator:
    def authenticate(self, assertion):
        if assertion == "owner-token":
            return Principal("owner@example.com", "owner")
        if assertion == "viewer-token":
            return Principal("advisor@example.com", "viewer")
        raise AuthenticationError("invalid")


class _Evidence:
    def catalog(self):
        return ({"run_id": "open-run", "title": "Open", "available": True},)

    def evidence(self, run_id):
        return {"run_id": run_id, "metrics_opened": run_id == "open-run"}

    def outcome(self, run_id):
        if run_id != "open-run":
            raise OutcomeLocked(f"outcomes remain locked for {run_id}")
        return {"run_id": run_id, "metrics": [{"sharpe": 0.7}]}


def _app(tmp_path: Path):
    runtime = tmp_path / "artifacts/.monitor"
    studies = {"study-a": StudyDefinition("study-a", "replication")}
    queue = QueueStore(runtime / "control.sqlite3", studies)
    security = RequestSecurity(
        HttpSecurityConfig("https://monitor.example.com", b"x" * 32),
        clock=lambda: 1000,
        nonce_factory=lambda: "nonce",
    )
    services = MonitorServices(
        queue=queue,
        events=EventStore(runtime),
        evidence=_Evidence(),
        audit=AuditStore(runtime),
        authenticator=_Authenticator(),
        request_security=security,
    )
    return create_app(services), services


def _request(app, path, *, token=None):
    parsed = urlsplit(path)
    headers = []
    if token:
        headers.append((b"cf-access-jwt-assertion", token.encode()))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "https",
        "path": parsed.path,
        "raw_path": parsed.path.encode(),
        "query_string": parsed.query.encode(),
        "headers": headers,
        "client": ("127.0.0.1", 50000),
        "server": ("monitor.example.com", 443),
        "root_path": "",
    }
    received = False
    messages = []

    async def receive():
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": b"", "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message):
        messages.append(message)

    asyncio.run(app(scope, receive, send))
    start = next(
        message for message in messages if message["type"] == "http.response.start"
    )
    body = b"".join(
        message.get("body", b"")
        for message in messages
        if message["type"] == "http.response.body"
    )
    return start["status"], dict(start["headers"]), json.loads(body)


def test_every_read_endpoint_requires_authentication_and_security_headers(
    tmp_path: Path,
) -> None:
    app, _services = _app(tmp_path)

    status, headers, body = _request(app, "/api/jobs")

    assert status == 401 and "authenticated" in body["detail"]
    assert headers[b"cache-control"] == b"no-store"
    assert b"default-src 'self'" in headers[b"content-security-policy"]
    assert b"access-control-allow-origin" not in headers


def test_session_studies_and_jobs_are_role_aware(tmp_path: Path) -> None:
    app, services = _app(tmp_path)
    job = services.queue.enqueue("study-a")

    status, _headers, session = _request(app, "/api/session", token="viewer-token")
    assert status == 200
    assert session["role"] == "viewer" and session["csrf_token"]

    status, _headers, studies = _request(app, "/api/studies", token="owner-token")
    assert status == 200 and studies["queueable"][0]["study_id"] == "study-a"

    status, _headers, jobs = _request(app, "/api/jobs", token="owner-token")
    assert status == 200 and jobs["jobs"][0]["job_id"] == job.job_id


def test_event_api_reconnects_and_keeps_outcomes_server_locked(tmp_path: Path) -> None:
    app, services = _app(tmp_path)
    job = services.queue.enqueue("study-a")
    observer = services.events.observer(job.job_id)
    observer(ResearchEvent("stage_started", "hmm"))
    observer(ResearchEvent("metric_ready", "metrics", visibility="outcome"))

    status, _headers, body = _request(
        app,
        f"/api/jobs/{job.job_id}/events?after_sequence=0",
        token="owner-token",
    )

    assert status == 200 and body["outcomes_locked"] is True
    assert [event["kind"] for event in body["events"]] == ["stage_started"]
    status, _headers, body = _request(
        app,
        f"/api/jobs/{job.job_id}/events?after_sequence=1",
        token="owner-token",
    )
    assert status == 200 and body["events"] == []


def test_evidence_outcomes_use_backend_lock_status(tmp_path: Path) -> None:
    app, _services = _app(tmp_path)

    opened = _request(app, "/api/evidence/open-run/outcome", token="viewer-token")
    locked = _request(app, "/api/evidence/locked-run/outcome", token="viewer-token")

    assert opened[0] == 200 and opened[2]["metrics"][0]["sharpe"] == 0.7
    assert locked[0] == 423 and "locked" in locked[2]["detail"]
