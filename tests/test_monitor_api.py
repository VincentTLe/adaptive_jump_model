import asyncio
import json
import threading
from pathlib import Path
from urllib.parse import urlsplit

from adaptive_jump.monitor.api import MonitorServices, create_app
from adaptive_jump.monitor.audit import AuditStore
from adaptive_jump.monitor.event_store import EventStore
from adaptive_jump.monitor.events import ResearchEvent
from adaptive_jump.monitor.evidence import EvidenceError, OutcomeLocked
from adaptive_jump.monitor.http_security import HttpSecurityConfig, RequestSecurity
from adaptive_jump.monitor.queue import QueueStore, StudyDefinition
from adaptive_jump.monitor.security import AuthenticationError, Principal


class _Authenticator:
    credential_header = "Cf-Access-Jwt-Assertion"
    challenge = "Cloudflare-Access"

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

    def market_data(self, run_id, market):
        if market != "us":
            raise EvidenceError(f"market is unavailable: {market}")
        return {
            "run_id": run_id,
            "market": market,
            "source": {"provider": "yahoo", "source_id": "^SP500TR"},
            "rows": [{"date": "2023-12-29", "close": 10327.83}],
        }


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


def _request(
    app,
    path,
    *,
    token=None,
    method="GET",
    body=None,
    origin="https://monitor.example.com",
    csrf=None,
):
    parsed = urlsplit(path)
    headers = []
    if token:
        headers.append((b"cf-access-jwt-assertion", token.encode()))
    if origin:
        headers.append((b"origin", origin.encode()))
    if csrf:
        headers.append((b"x-csrf-token", csrf.encode()))
    payload = b"" if body is None else json.dumps(body).encode()
    if body is not None:
        headers.append((b"content-type", b"application/json"))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
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
            return {
                "type": "http.request",
                "body": payload,
                "more_body": False,
            }
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
    replay_thread_ids = []
    original_replay = services.events.replay

    def replay(*args, **kwargs):
        replay_thread_ids.append(threading.get_ident())
        return original_replay(*args, **kwargs)

    services.events.replay = replay

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
    assert replay_thread_ids and set(replay_thread_ids) != {threading.get_ident()}


def test_market_data_requires_one_verified_completed_job(tmp_path: Path) -> None:
    app, services = _app(tmp_path)
    job = services.queue.enqueue("study-a")

    unavailable = _request(
        app,
        f"/api/jobs/{job.job_id}/markets/us/ohlcv",
        token="viewer-token",
    )
    assert unavailable[0] == 409

    services.queue.claim_next()
    observer = services.events.observer(job.job_id)
    observer(
        ResearchEvent(
            "artifact_verified",
            "verification",
            visibility="decision",
            payload={"run_id": "verified-run", "status": "complete"},
        )
    )
    services.queue.finish(job.job_id, "succeeded", 0)

    response = _request(
        app,
        f"/api/jobs/{job.job_id}/markets/us/ohlcv",
        token="viewer-token",
    )
    assert response[0] == 200
    assert response[2]["job_id"] == job.job_id
    assert response[2]["run_id"] == "verified-run"
    assert response[2]["source"]["source_id"] == "^SP500TR"

    unknown_market = _request(
        app,
        f"/api/jobs/{job.job_id}/markets/xx/ohlcv",
        token="viewer-token",
    )
    assert unknown_market[0] == 409

    duplicate = services.queue.enqueue("study-a")
    services.queue.claim_next()
    duplicate_observer = services.events.observer(duplicate.job_id)
    for _ in range(2):
        duplicate_observer(
            ResearchEvent(
                "artifact_verified",
                "verification",
                visibility="decision",
                payload={"run_id": "verified-run", "status": "complete"},
            )
        )
    services.queue.finish(duplicate.job_id, "succeeded", 0)
    rejected = _request(
        app,
        f"/api/jobs/{duplicate.job_id}/markets/us/ohlcv",
        token="viewer-token",
    )
    assert rejected[0] == 409


def test_evidence_outcomes_use_backend_lock_status(tmp_path: Path) -> None:
    app, _services = _app(tmp_path)

    opened = _request(app, "/api/evidence/open-run/outcome", token="viewer-token")
    locked = _request(app, "/api/evidence/locked-run/outcome", token="viewer-token")

    assert opened[0] == 200 and opened[2]["metrics"][0]["sharpe"] == 0.7
    assert locked[0] == 423 and "locked" in locked[2]["detail"]


def test_owner_can_enqueue_reorder_cancel_and_resume_with_csrf(tmp_path: Path) -> None:
    app, services = _app(tmp_path)
    csrf = _request(app, "/api/session", token="owner-token")[2]["csrf_token"]

    first = _request(
        app,
        "/api/jobs",
        token="owner-token",
        method="POST",
        csrf=csrf,
        body={"study_id": "study-a"},
    )
    second = _request(
        app,
        "/api/jobs",
        token="owner-token",
        method="POST",
        csrf=csrf,
        body={"study_id": "study-a"},
    )
    assert first[0] == second[0] == 201
    reordered = _request(
        app,
        "/api/jobs/reorder",
        token="owner-token",
        method="POST",
        csrf=csrf,
        body={"job_ids": [second[2]["job_id"], first[2]["job_id"]]},
    )
    assert reordered[0] == 200
    assert reordered[2]["jobs"][0]["job_id"] == second[2]["job_id"]

    canceled = _request(
        app,
        f"/api/jobs/{first[2]['job_id']}/cancel",
        token="owner-token",
        method="POST",
        csrf=csrf,
    )
    assert canceled[2]["status"] == "canceled"

    claimed = services.queue.claim_next()
    assert claimed is not None
    services.queue.recover_abandoned()
    resumed = _request(
        app,
        f"/api/jobs/{claimed.job_id}/resume",
        token="owner-token",
        method="POST",
        csrf=csrf,
    )
    assert resumed[2]["status"] == "queued"
    actions = [record.action for record in services.audit.replay()]
    assert actions.count("authorize_mutation") == 5
    assert {"enqueue", "reorder", "cancel", "resume"} <= set(actions)


def test_viewer_bad_origin_and_bad_csrf_cannot_mutate(tmp_path: Path) -> None:
    app, services = _app(tmp_path)
    owner_csrf = _request(app, "/api/session", token="owner-token")[2]["csrf_token"]
    viewer_csrf = _request(app, "/api/session", token="viewer-token")[2]["csrf_token"]
    attempts = (
        {"token": "viewer-token", "csrf": viewer_csrf},
        {"token": "owner-token", "csrf": owner_csrf, "origin": "https://evil.test"},
        {"token": "owner-token", "csrf": "bad-token"},
    )
    for values in attempts:
        result = _request(
            app,
            "/api/jobs",
            method="POST",
            body={"study_id": "study-a"},
            **values,
        )
        assert result[0] == 403

    assert services.queue.all_jobs() == ()
    audit = services.audit.replay()
    assert len(audit) == 3
    assert all(record.outcome == "rejected" for record in audit)


def test_queue_conflicts_are_audited_and_no_delete_route_exists(tmp_path: Path) -> None:
    app, services = _app(tmp_path)
    csrf = _request(app, "/api/session", token="owner-token")[2]["csrf_token"]

    rejected = _request(
        app,
        "/api/jobs",
        token="owner-token",
        method="POST",
        csrf=csrf,
        body={"study_id": "unknown-study"},
    )
    deleted = _request(
        app,
        "/api/jobs/not-a-job",
        token="owner-token",
        method="DELETE",
        csrf=csrf,
    )

    assert rejected[0] == 409
    assert deleted[0] == 405
    assert services.queue.all_jobs() == ()
    records = services.audit.replay()
    assert any(
        record.action == "enqueue" and record.outcome == "rejected"
        for record in records
    )
