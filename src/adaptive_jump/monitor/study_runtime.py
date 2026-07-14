"""Output-neutral adapters that translate canonical progress into live events."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from datetime import date, datetime
from typing import Any

import pandas as pd

from adaptive_jump.monitor.events import (
    EventObserver,
    ResearchEvent,
    bind_event_context,
    emit_event,
)

_FEATURE_COLUMNS = ("dd_10", "sortino_20", "sortino_60", "excess_return")


def model_observer(
    observer: EventObserver | None,
    market: str,
    model: str,
    frame: pd.DataFrame,
) -> EventObserver | None:
    """Bind model identity and attach date-matched causal feature values."""
    contextual = bind_event_context(observer, market=market, model=model)
    if contextual is None:
        return None
    columns = [column for column in _FEATURE_COLUMNS if column in frame]
    snapshots: dict[date, dict[str, float]] = {}
    dates = pd.to_datetime(frame["date"], errors="raise")
    for index, timestamp in dates.items():
        values = {
            column: float(frame.at[index, column])
            for column in columns
            if _finite(frame.at[index, column])
        }
        if values:
            snapshots[timestamp.date()] = values

    def observe(event: ResearchEvent) -> None:
        features = snapshots.get(event.date) if event.date else None
        if features:
            contextual(replace(event, payload={**event.payload, "features": features}))
        else:
            contextual(event)

    return observe


def baseline_selection_recorder(
    saver: Callable[[str, int, Any], None],
    observer: EventObserver | None,
    market: str,
) -> Callable[[str, int, Any], None]:
    """Preserve checkpoint writes and emit one decision-safe CV snapshot."""
    if observer is None:
        return saver

    def record(model: str, delay: int, progress: Any) -> None:
        saver(model, delay, progress)
        contextual = bind_event_context(
            observer, market=market, model=model, delay=delay
        )
        _emit_selection_snapshot(contextual, progress)

    return record


def selection_progress_observer(
    observer: EventObserver | None, model: str, delay: int
) -> Callable[[Any], None] | None:
    """Adapt a single selection callback to the shared event contract."""
    contextual = bind_event_context(observer, model=model, delay=delay)
    if contextual is None:
        return None
    return lambda progress: _emit_selection_snapshot(contextual, progress)


def emit_boundary_rows(
    observer: EventObserver | None,
    rows: pd.DataFrame,
    market: str | None = None,
) -> None:
    """Expose gate inputs without opening any outcome metric."""
    if observer is None:
        return
    for source in rows.to_dict("records"):
        model = str(source.pop("model"))
        delay = int(source.pop("delay"))
        contextual = bind_event_context(
            observer, market=market, model=model, delay=delay
        )
        emit_event(
            contextual,
            kind="boundary_diagnostic",
            stage="selection",
            visibility="decision",
            payload={key: _json_value(value) for key, value in source.items()},
        )


def bootstrap_recorder(
    saver: Callable[[int, Any], None],
    observer: EventObserver | None,
    replications: int,
) -> Callable[[int, Any], None]:
    """Persist bootstrap state while exposing counts, never draws or outcomes."""
    if observer is None:
        return saver

    def record(block: int, progress: Any) -> None:
        saver(block, progress)
        emit_event(
            observer,
            kind="bootstrap_progress",
            stage="bootstrap",
            completed=len(progress.draws),
            total=replications,
            payload={"mean_block_length": int(block)},
        )

    return record


def _emit_selection_snapshot(observer: EventObserver, progress: Any) -> None:
    surface = progress.surface
    if surface.empty:
        return
    decision_dates = pd.to_datetime(surface["decision_date"], errors="raise")
    latest = decision_dates.iloc[-1]
    current = surface.loc[decision_dates == latest]
    choice_dates = pd.to_datetime(progress.choices["decision_date"], errors="raise")
    choice = progress.choices.loc[choice_dates == latest, "selected"]
    selected = float(choice.iloc[-1]) if not choice.empty else None
    candidates = [
        {
            "candidate": float(row["candidate"]),
            "valid_returns": int(row["valid_returns"]),
            "sharpe": _json_value(row["sharpe"]),
            "eligible": bool(row["eligible"]),
        }
        for row in current.to_dict("records")
    ]
    emit_event(
        observer,
        kind="selection_checkpoint",
        stage="selection",
        visibility="decision",
        date=latest.date(),
        payload={
            "completed_months": int(decision_dates.nunique()),
            "selected_candidate": selected,
            "cv_surface": candidates,
        },
    )


def _finite(value: object) -> bool:
    try:
        return pd.notna(value) and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _json_value(value: object) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
