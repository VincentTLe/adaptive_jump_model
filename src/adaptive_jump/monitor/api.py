"""Authenticated read API for queue state, runtime events, and sealed evidence."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from adaptive_jump.monitor.audit import AuditStore
from adaptive_jump.monitor.event_store import EventStore, EventStoreError
from adaptive_jump.monitor.evidence import EvidenceError, EvidenceStore, OutcomeLocked
from adaptive_jump.monitor.http_security import (
    CSRF_HEADER,
    SECURITY_HEADERS,
    RequestSecurity,
    RequestSecurityError,
)
from adaptive_jump.monitor.queue import QueueError, QueueStore
from adaptive_jump.monitor.security import (
    AuthenticationError,
    Authenticator,
    Principal,
)
from adaptive_jump.monitor.sse import StreamError, resume_sequence, stream_job_events

API_PREFIX = "/api"
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
StudyId = Annotated[
    str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9-]*$", max_length=100)
]
JobId = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]


class EnqueueRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    study_id: StudyId


class ReorderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job_ids: Annotated[list[JobId], Field(max_length=1000)]


@dataclass(frozen=True)
class MonitorServices:
    queue: QueueStore
    events: EventStore
    evidence: EvidenceStore
    audit: AuditStore
    authenticator: Authenticator
    request_security: RequestSecurity


def create_app(services: MonitorServices, *, lifespan=None) -> FastAPI:
    """Build one dependency-injected app without reading secrets at import time."""
    app = FastAPI(
        title="Adaptive Jump Research Monitor",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.services = services
    static_root = Path(__file__).with_name("static")
    app.mount("/assets", StaticFiles(directory=static_root), name="monitor-assets")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(static_root / "index.html")

    @app.middleware("http")
    async def secure_responses(request: Request, call_next):
        if request.url.path == "/healthz":
            response = await call_next(request)
        else:
            credential = request.headers.get(
                services.authenticator.credential_header, ""
            )
            try:
                user = services.authenticator.authenticate(credential)
            except AuthenticationError:
                response = JSONResponse(
                    status_code=401,
                    content={"detail": "authenticated monitor access is required"},
                    headers={"WWW-Authenticate": services.authenticator.challenge},
                )
            else:
                request.state.principal = user
                if (
                    request.url.path.startswith(API_PREFIX)
                    and request.method not in _SAFE_METHODS
                ):
                    try:
                        services.request_security.require_origin(
                            request.headers.get("Origin")
                        )
                        services.request_security.verify_csrf(
                            request.headers.get(CSRF_HEADER), user.email
                        )
                        if user.role != "owner":
                            raise RequestSecurityError("owner role is required")
                    except RequestSecurityError:
                        services.audit.append(
                            user,
                            "authorize_mutation",
                            "api",
                            "rejected",
                            {"method": request.method, "path": request.url.path[:300]},
                        )
                        response = JSONResponse(
                            status_code=403,
                            content={"detail": "mutation authorization failed"},
                        )
                    else:
                        services.audit.append(
                            user,
                            "authorize_mutation",
                            "api",
                            "accepted",
                            {"method": request.method, "path": request.url.path[:300]},
                        )
                        response = await call_next(request)
                else:
                    response = await call_next(request)
        for name, value in SECURITY_HEADERS.items():
            response.headers[name] = value
        if request.url.path.startswith(API_PREFIX):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(OutcomeLocked)
    async def outcome_locked(_request: Request, exc: OutcomeLocked):
        return JSONResponse(status_code=423, content={"detail": str(exc)})

    @app.exception_handler(EvidenceError)
    async def invalid_evidence(_request: Request, exc: EvidenceError):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.get(f"{API_PREFIX}/session")
    async def session(request: Request) -> dict[str, Any]:
        user: Principal = request.state.principal
        return {
            "email": user.email,
            "role": user.role,
            "csrf_token": services.request_security.issue_csrf(user.email),
        }

    @app.get(f"{API_PREFIX}/studies")
    async def studies() -> dict[str, Any]:
        return {
            "queueable": [
                asdict(definition) for definition in services.queue.studies.values()
            ],
            "sealed_runs": services.evidence.catalog(),
        }

    @app.get(f"{API_PREFIX}/jobs")
    async def jobs() -> dict[str, Any]:
        return {"jobs": [asdict(job) for job in services.queue.all_jobs()]}

    def mutate(
        request: Request,
        action: str,
        target: str,
        operation: Callable[[], Any],
        details: dict[str, Any],
    ) -> Any:
        user: Principal = request.state.principal
        try:
            result = operation()
        except QueueError as exc:
            services.audit.append(
                user,
                action,
                target,
                "rejected",
                {**details, "reason": "queue_state"},
            )
            raise HTTPException(
                status_code=409, detail="queue mutation rejected"
            ) from exc
        services.audit.append(user, action, target, "accepted", details)
        return result

    @app.post(f"{API_PREFIX}/jobs", status_code=201)
    async def enqueue(request: Request, body: EnqueueRequest) -> dict[str, Any]:
        result = mutate(
            request,
            "enqueue",
            "queue",
            lambda: services.queue.enqueue(body.study_id),
            {"study_id": body.study_id},
        )
        return asdict(result)

    @app.post(f"{API_PREFIX}/jobs/reorder")
    async def reorder(request: Request, body: ReorderRequest) -> dict[str, Any]:
        result = mutate(
            request,
            "reorder",
            "queue",
            lambda: services.queue.reorder(body.job_ids),
            {"job_ids": body.job_ids},
        )
        return {"jobs": [asdict(job) for job in result]}

    @app.post(f"{API_PREFIX}/jobs/{{job_id}}/cancel")
    async def cancel(request: Request, job_id: str) -> dict[str, Any]:
        result = mutate(
            request,
            "cancel",
            "jobs",
            lambda: services.queue.request_cancel(job_id),
            {"job_id": job_id},
        )
        return asdict(result)

    @app.post(f"{API_PREFIX}/jobs/{{job_id}}/resume")
    async def resume(request: Request, job_id: str) -> dict[str, Any]:
        result = mutate(
            request,
            "resume",
            "jobs",
            lambda: services.queue.resume(job_id),
            {"job_id": job_id},
        )
        return asdict(result)

    @app.get(f"{API_PREFIX}/jobs/{{job_id}}")
    async def job(job_id: str) -> dict[str, Any]:
        try:
            return asdict(services.queue.get(job_id))
        except QueueError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc

    @app.get(f"{API_PREFIX}/jobs/{{job_id}}/events")
    async def events(
        job_id: str,
        after_sequence: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        try:
            services.queue.get(job_id)
        except QueueError as exc:
            raise HTTPException(status_code=404, detail="job events not found") from exc
        try:
            replay = await asyncio.to_thread(
                services.events.replay, job_id, after_sequence
            )
        except EventStoreError as exc:
            raise HTTPException(
                status_code=409, detail="job event journal is invalid"
            ) from exc
        visible = [
            event.to_dict() for event in replay if event.event.visibility != "outcome"
        ]
        return {"events": visible, "outcomes_locked": True}

    @app.get(f"{API_PREFIX}/jobs/{{job_id}}/stream")
    async def stream(
        request: Request,
        job_id: str,
        after_sequence: int = Query(default=0, ge=0),
    ) -> StreamingResponse:
        try:
            cursor = resume_sequence(
                request.headers.get("Last-Event-ID"), after_sequence
            )
            await asyncio.to_thread(services.queue.get, job_id)
            await asyncio.to_thread(services.events.replay, job_id, cursor)
        except StreamError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except QueueError as exc:
            raise HTTPException(status_code=404, detail="job stream not found") from exc
        except EventStoreError as exc:
            raise HTTPException(
                status_code=409, detail="job event journal is invalid"
            ) from exc
        return StreamingResponse(
            stream_job_events(
                services.queue,
                services.events,
                job_id,
                cursor,
                request.is_disconnected,
            ),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"},
        )

    @app.get(f"{API_PREFIX}/evidence")
    async def evidence_catalog() -> dict[str, Any]:
        return {"runs": services.evidence.catalog()}

    @app.get(f"{API_PREFIX}/evidence/{{run_id}}")
    async def evidence(run_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(services.evidence.evidence, run_id)

    @app.get(f"{API_PREFIX}/evidence/{{run_id}}/outcome")
    async def outcome(run_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(services.evidence.outcome, run_id)

    return app
