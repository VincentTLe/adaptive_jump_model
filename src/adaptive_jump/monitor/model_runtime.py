"""Runtime adapters that do not participate in model mathematics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from adaptive_jump.monitor.events import EventObserver, emit_event

_JM_REFIT_COLUMNS = (
    "fit_date",
    "training_start",
    "training_end",
    "observations",
    "scaler_mean",
    "scaler_scale",
    "lambda",
    "objective",
)


class CheckpointError(ValueError):
    """Raised when a model checkpoint is not a valid causal prefix."""


@dataclass(frozen=True)
class FixedJMResume:
    """Validated state needed to continue fixed-JM terminal inference."""

    states: pd.DataFrame
    records: list[dict[str, Any]]
    first_terminal: int
    last_refit_terminal: int | None


def prepare_fixed_jm_resume(
    states: pd.DataFrame,
    refits: pd.DataFrame,
    complete: pd.DataFrame,
    all_dates: pd.DatetimeIndex,
    fit_window: int,
    candidates: tuple[float, ...],
    refit_months: tuple[int, ...],
) -> FixedJMResume:
    """Validate a checkpoint and locate the fit that must be rebuilt."""
    checkpoint = states.copy()
    if not checkpoint.index.equals(all_dates):
        raise CheckpointError("JM checkpoint dates do not match inputs")
    if tuple(checkpoint.columns) != candidates:
        raise CheckpointError("JM checkpoint candidates do not match protocol")

    first_eligible = fit_window - 1
    eligible_dates = pd.DatetimeIndex(complete.iloc[first_eligible:]["date"])
    eligible = checkpoint.reindex(eligible_dates)
    full = eligible.notna().all(axis=1)
    empty = eligible.isna().all(axis=1)
    if not (full | empty).all():
        raise CheckpointError("JM checkpoint contains a partial candidate row")
    completed = int(full.sum())
    if not full.iloc[:completed].all() or not empty.iloc[completed:].all():
        raise CheckpointError("JM checkpoint is not a contiguous causal prefix")

    completed_dates = eligible_dates[:completed]
    expected_rows = pd.Series(
        checkpoint.index.isin(completed_dates), index=checkpoint.index
    )
    if not checkpoint.notna().any(axis=1).equals(expected_rows):
        raise CheckpointError("JM checkpoint contains states outside its prefix")
    if completed and not checkpoint.loc[completed_dates].isin([0.0, 1.0]).all().all():
        raise CheckpointError("JM checkpoint states must be 0 or 1")
    checkpoint = checkpoint.astype("Float64").astype(float)
    checkpoint.index = all_dates
    checkpoint.columns = pd.Index(candidates)

    records = refits.copy()
    if completed == 0:
        if not records.empty:
            raise CheckpointError("empty JM checkpoint cannot contain refits")
        return FixedJMResume(checkpoint, [], first_eligible, None)
    if records.empty or tuple(records.columns) != _JM_REFIT_COLUMNS:
        raise CheckpointError("JM checkpoint refits are incomplete")
    try:
        records["fit_date"] = pd.to_datetime(records["fit_date"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise CheckpointError("JM checkpoint refit dates are invalid") from exc
    if not records["fit_date"].is_monotonic_increasing:
        raise CheckpointError("JM checkpoint refits are not chronological")
    fit_dates = pd.DatetimeIndex(records["fit_date"].drop_duplicates())
    if not fit_dates.equals(_expected_refit_dates(completed_dates, refit_months)):
        raise CheckpointError("JM checkpoint refit dates violate the prefix")
    for _, group in records.groupby("fit_date", sort=False):
        try:
            found = tuple(float(value) for value in group["lambda"])
        except (TypeError, ValueError) as exc:
            raise CheckpointError("JM checkpoint candidates are invalid") from exc
        if found != candidates or not group["observations"].eq(fit_window).all():
            raise CheckpointError("JM checkpoint refit group violates protocol")
    last_refit = fit_dates[-1]
    matches = complete.index[complete["date"] == last_refit].tolist()
    if len(matches) != 1:
        raise CheckpointError("JM checkpoint refit cannot be located in inputs")
    return FixedJMResume(
        checkpoint,
        records.to_dict("records"),
        first_eligible + completed,
        int(matches[0]),
    )


def _expected_refit_dates(
    completed_dates: pd.DatetimeIndex, refit_months: tuple[int, ...]
) -> pd.DatetimeIndex:
    expected: list[pd.Timestamp] = []
    last_anchor: tuple[int, int] | None = None
    for current in completed_dates:
        anchor = (current.year, current.month)
        if not expected or (current.month in refit_months and anchor != last_anchor):
            expected.append(current)
            last_anchor = anchor
    return pd.DatetimeIndex(expected)


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
