"""Runtime adapters that do not participate in model mathematics."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any

from adaptive_jump.monitor.events import EventObserver, emit_event


def _emit_model_event(
    observer: EventObserver | None,
    kind: str,
    stage: str,
    completed: int,
    total: int,
    *,
    visibility: str = "operational",
    event_date: date | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    emit_event(
        observer,
        kind=kind,
        stage=stage,
        visibility=visibility,
        date=event_date,
        completed=completed,
        total=total,
        payload=payload or {},
    )


def emit_fixed_jm_started(
    observer: EventObserver | None,
    fit_window: int,
    candidates: tuple[float, ...],
    completed: int,
    total: int,
) -> None:
    _emit_model_event(
        observer,
        "stage_started",
        "fixed_jm",
        completed,
        total,
        payload={"fit_window": fit_window, "candidates": list(candidates)},
    )


def emit_fixed_jm_refit(
    observer: EventObserver | None,
    event_date: date,
    completed: int,
    total: int,
) -> None:
    _emit_model_event(
        observer,
        "refit",
        "fixed_jm",
        completed,
        total,
        visibility="decision",
        event_date=event_date,
    )


def emit_fixed_jm_terminal(
    observer: EventObserver | None,
    event_date: date,
    completed: int,
    total: int,
    states: Sequence[tuple[float, int]],
) -> None:
    _emit_model_event(
        observer,
        "terminal_state",
        "fixed_jm",
        completed,
        total,
        visibility="decision",
        event_date=event_date,
        payload={
            "states": [
                {"candidate": candidate, "state": state} for candidate, state in states
            ]
        },
    )


def emit_hmm_started(
    observer: EventObserver | None,
    fit_window: int,
    restart_count: int,
    workers: int,
    completed: int,
    total: int,
) -> None:
    _emit_model_event(
        observer,
        "stage_started",
        "hmm",
        completed=completed,
        total=total,
        payload={
            "fit_window": fit_window,
            "restart_count": restart_count,
            "workers": workers,
        },
    )


def emit_hmm_terminal(
    observer: EventObserver | None,
    event_date: date,
    completed: int,
    total: int,
    state: int,
    seed: int,
    log_likelihood: float,
    variances: tuple[float, float],
    accepted_starts: int,
    failed_starts: tuple[str, ...],
) -> None:
    _emit_model_event(
        observer,
        "terminal_state",
        "hmm",
        completed,
        total,
        visibility="decision",
        event_date=event_date,
        payload={
            "state": state,
            "seed": seed,
            "log_likelihood": log_likelihood,
            "variances": list(variances),
            "accepted_starts": accepted_starts,
            "failed_starts": list(failed_starts),
        },
    )


def emit_stage_completed(
    observer: EventObserver | None, stage: str, total: int
) -> None:
    _emit_model_event(observer, "stage_completed", stage, total, total)
