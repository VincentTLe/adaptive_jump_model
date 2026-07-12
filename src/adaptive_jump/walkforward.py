"""Monthly causal model selection for preregistered walk-forward studies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from adaptive_jump.backtest import annualized_excess_sharpe, apply_signal
from adaptive_jump.config import SelectionProtocol


class WalkForwardError(ValueError):
    """Raised when a walk-forward selection contract cannot be satisfied."""


@dataclass(frozen=True)
class SelectionResult:
    """Selected signal plus complete CV evidence."""

    signal: pd.Series
    choices: pd.DataFrame
    surface: pd.DataFrame
    candidate_returns: pd.DataFrame


@dataclass(frozen=True)
class BoundaryDiagnostic:
    """Upper-grid selection frequency before metrics are opened."""

    upper_candidate: float
    selected_months: int
    total_months: int
    fraction: float
    limit: float
    passed: bool


def select_monthly_candidate(
    returns: pd.DataFrame,
    candidate_states: pd.DataFrame,
    protocol: SelectionProtocol,
    *,
    delay_trading_days: int,
    one_way_cost_bps: float,
    periods_per_year: int = 252,
    volatility_ddof: int = 1,
) -> SelectionResult:
    """Select a state path monthly using only trailing validation returns."""
    prepared, states = _align_selection_inputs(returns, candidate_states)
    dates = pd.DatetimeIndex(prepared["date"])
    candidates = tuple(sorted(float(value) for value in states.columns))
    states = states.loc[:, candidates]

    candidate_returns = pd.DataFrame(index=dates, columns=candidates, dtype=float)
    for candidate in candidates:
        risky_signal = 1.0 - states[candidate]
        path = apply_signal(
            prepared,
            risky_signal.reset_index(drop=True),
            delay_trading_days=delay_trading_days,
            one_way_cost_bps=one_way_cost_bps,
        )
        candidate_returns[candidate] = path["strategy_return"].to_numpy()

    first_complete = states.dropna(how="any").first_valid_index()
    choices: list[dict[str, object]] = []
    surface: list[dict[str, object]] = []
    selection_started = False
    if first_complete is not None:
        earliest = first_complete + pd.DateOffset(years=protocol.validation_years)
        for decision_date in _month_end_dates(dates):
            if decision_date < earliest:
                continue
            selected = _score_decision(
                decision_date,
                candidates,
                states,
                candidate_returns,
                prepared.set_index("date")["cash_return"],
                protocol,
                periods_per_year,
                volatility_ddof,
                surface,
            )
            if selected is None:
                if selection_started:
                    raise WalkForwardError(
                        f"no eligible candidate on {decision_date.date()}"
                    )
                continue
            selection_started = True
            choices.append({"decision_date": decision_date, "selected": selected})

    choice_frame = pd.DataFrame.from_records(
        choices, columns=["decision_date", "selected"]
    )
    selected_signal = _compose_selected_signal(dates, states, choice_frame)
    return SelectionResult(
        signal=selected_signal,
        choices=choice_frame,
        surface=pd.DataFrame.from_records(surface),
        candidate_returns=candidate_returns,
    )


def boundary_diagnostic(
    choices: pd.DataFrame,
    candidate_grid: tuple[float, ...],
    *,
    oos_start: date,
    fraction_limit: float,
) -> BoundaryDiagnostic:
    """Check upper-grid choices on OOS months without opening performance metrics."""
    if not candidate_grid:
        raise WalkForwardError("candidate grid must not be empty")
    required = {"decision_date", "selected"}
    if not required.issubset(choices):
        raise WalkForwardError("choices must contain decision_date and selected")
    decision_dates = pd.to_datetime(choices["decision_date"], errors="raise")
    oos = choices.loc[decision_dates >= pd.Timestamp(oos_start)]
    if oos.empty:
        raise WalkForwardError("no OOS monthly choices for boundary check")
    upper = float(max(candidate_grid))
    selected_months = int(np.isclose(oos["selected"].astype(float), upper).sum())
    total_months = len(oos)
    fraction = selected_months / total_months
    return BoundaryDiagnostic(
        upper_candidate=upper,
        selected_months=selected_months,
        total_months=total_months,
        fraction=fraction,
        limit=fraction_limit,
        passed=fraction <= fraction_limit,
    )


def _score_decision(
    decision_date: pd.Timestamp,
    candidates: tuple[float, ...],
    states: pd.DataFrame,
    candidate_returns: pd.DataFrame,
    cash_return: pd.Series,
    protocol: SelectionProtocol,
    periods_per_year: int,
    volatility_ddof: int,
    surface: list[dict[str, object]],
) -> float | None:
    start = decision_date - pd.DateOffset(years=protocol.validation_years)
    validation = (candidate_returns.index > start) & (
        candidate_returns.index <= decision_date
    )
    eligible: list[tuple[float, float]] = []
    for candidate in candidates:
        paired = pd.concat(
            [candidate_returns.loc[validation, candidate], cash_return.loc[validation]],
            axis=1,
        ).dropna()
        count = len(paired)
        score = annualized_excess_sharpe(
            paired.iloc[:, 0],
            paired.iloc[:, 1],
            periods_per_year=periods_per_year,
            volatility_ddof=volatility_ddof,
        )
        current_state = states.loc[decision_date, candidate]
        is_eligible = (
            count >= protocol.minimum_valid_returns
            and math.isfinite(score)
            and pd.notna(current_state)
        )
        surface.append(
            {
                "decision_date": decision_date,
                "candidate": candidate,
                "valid_returns": count,
                "sharpe": score,
                "eligible": is_eligible,
            }
        )
        if is_eligible:
            eligible.append((candidate, score))
    if not eligible:
        return None
    best = max(score for _, score in eligible)
    return min(
        candidate
        for candidate, score in eligible
        if best - score <= protocol.tie_tolerance
    )


def _compose_selected_signal(
    dates: pd.DatetimeIndex, states: pd.DataFrame, choices: pd.DataFrame
) -> pd.Series:
    active = pd.Series(np.nan, index=dates, dtype=float)
    if not choices.empty:
        active.loc[pd.DatetimeIndex(choices["decision_date"])] = choices[
            "selected"
        ].to_numpy()
        active = active.ffill()
    selected_state = pd.Series(np.nan, index=dates, dtype=float)
    for candidate in states.columns:
        mask = active == candidate
        selected_state.loc[mask] = states.loc[mask, candidate]
    signal = 1.0 - selected_state
    signal.name = "selected_signal"
    return signal


def _align_selection_inputs(
    returns: pd.DataFrame, candidate_states: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = ["date", "equity_simple", "cash_return"]
    missing = [column for column in required if column not in returns]
    if missing:
        raise WalkForwardError(f"missing selection columns: {missing}")
    prepared = returns[required].copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="raise")
    dates = pd.DatetimeIndex(prepared["date"])
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise WalkForwardError("selection dates must be increasing and unique")
    states = candidate_states.copy()
    states.index = pd.to_datetime(states.index, errors="raise")
    if states.index.has_duplicates or not states.index.is_monotonic_increasing:
        raise WalkForwardError("candidate state dates must be increasing and unique")
    if not states.columns.is_unique:
        raise WalkForwardError("candidate values must be unique")
    try:
        states.columns = [float(value) for value in states.columns]
    except (TypeError, ValueError) as exc:
        raise WalkForwardError("candidate values must be numeric") from exc
    states = states.reindex(dates)
    if not states.stack().isin([0.0, 1.0]).all():
        raise WalkForwardError("candidate states must be 0, 1, or missing")
    return prepared, states


def _month_end_dates(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    values = pd.Series(dates, index=dates)
    return pd.DatetimeIndex(values.groupby(dates.to_period("M")).max())
