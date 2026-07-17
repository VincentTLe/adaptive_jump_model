"""Monthly causal model selection for preregistered walk-forward studies."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from functools import partial

import numpy as np
import pandas as pd

from adaptive_jump.backtest import (
    annualized_excess_sharpe,
    apply_signal,
    buy_and_hold,
    performance_metrics,
)
from adaptive_jump.config import (
    LEGACY_COMPARISON_SAMPLE,
    PAPER_COMPARISON_SAMPLE,
    ResearchConfig,
    SelectionProtocol,
)
from adaptive_jump.models import (
    FixedJMResult,
    HMMResult,
    fixed_jm_states,
    hmm_states,
    smoothed_hmm_states,
)


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
class SelectionProgress:
    """Causal monthly choices and CV rows completed so far."""

    choices: pd.DataFrame
    surface: pd.DataFrame


SelectionLoader = Callable[[str, int], SelectionProgress | None]
SelectionSaver = Callable[[str, int, SelectionProgress], None]


@dataclass(frozen=True)
class BoundaryDiagnostic:
    """Upper-grid selection frequency before metrics are opened."""

    upper_candidate: float
    selected_months: int
    total_months: int
    fraction: float
    limit: float
    passed: bool


@dataclass(frozen=True)
class BaselineStudy:
    """Sealed model outputs and selection evidence for one market."""

    oos_start: date
    jm: FixedJMResult
    hmm: HMMResult
    hmm_candidates: pd.DataFrame
    selections: dict[str, dict[int, SelectionResult]]
    boundaries: pd.DataFrame


def build_baseline_study(
    frame: pd.DataFrame,
    config: ResearchConfig,
    *,
    oos_start: date,
    precomputed_jm: FixedJMResult | None = None,
    precomputed_hmm: HMMResult | None = None,
    selection_initial: SelectionLoader | None = None,
    selection_progress: SelectionSaver | None = None,
) -> BaselineStudy:
    """Build all baseline choices and boundary checks without OOS metrics."""
    jm = precomputed_jm or fixed_jm_states(
        frame, config.model_protocol, config.jm_protocol
    )
    hmm = precomputed_hmm or hmm_states(
        frame, config.model_protocol, config.hmm_protocol
    )
    hmm_candidates = smoothed_hmm_states(hmm.states, config.hmm_protocol.smoothing_grid)
    returns = frame[["date", "equity_simple", "cash_return"]]
    selections: dict[str, dict[int, SelectionResult]] = {"fixed_jm": {}, "hmm": {}}
    boundary_rows: list[dict[str, object]] = []
    candidates = {"fixed_jm": jm.states, "hmm": hmm_candidates}
    grids = {
        "fixed_jm": config.jm_protocol.lambda_grid,
        "hmm": tuple(float(value) for value in config.hmm_protocol.smoothing_grid),
    }
    backtest = config.backtest_protocol
    metrics = config.metrics_protocol
    for delay in backtest.robustness_delays:
        for model_name, candidate_states in candidates.items():
            initial = (
                selection_initial(model_name, delay) if selection_initial else None
            )
            save = (
                partial(selection_progress, model_name, delay)
                if selection_progress
                else None
            )
            selection = select_monthly_candidate(
                returns,
                candidate_states,
                config.selection_protocol,
                delay_trading_days=delay,
                one_way_cost_bps=backtest.one_way_cost_bps,
                periods_per_year=metrics.periods_per_year,
                volatility_ddof=metrics.volatility_ddof,
                initial=initial,
                progress=save,
            )
            selections[model_name][delay] = selection
            diagnostic = boundary_diagnostic(
                selection.choices,
                grids[model_name],
                oos_start=oos_start,
                fraction_limit=config.selection_protocol.boundary_fraction_limit,
            )
            boundary_rows.append(
                {
                    "model": model_name,
                    "delay": delay,
                    **diagnostic.__dict__,
                }
            )
    return BaselineStudy(
        oos_start=oos_start,
        jm=jm,
        hmm=hmm,
        hmm_candidates=hmm_candidates,
        selections=selections,
        boundaries=pd.DataFrame.from_records(boundary_rows),
    )


def open_baseline_metrics(
    frame: pd.DataFrame, study: BaselineStudy, config: ResearchConfig
) -> pd.DataFrame:
    """Open OOS metrics only after every preregistered boundary check passes."""
    if study.boundaries.empty or not study.boundaries["passed"].all():
        raise WalkForwardError("OOS metrics are sealed until all boundary checks pass")
    metrics_protocol = config.metrics_protocol
    rows: list[dict[str, object]] = []
    for delay, paths in baseline_paths(frame, study, config).items():
        for model_name, path in paths.items():
            values = performance_metrics(
                path,
                periods_per_year=metrics_protocol.periods_per_year,
                volatility_ddof=metrics_protocol.volatility_ddof,
                expected_shortfall_quantile=(
                    metrics_protocol.expected_shortfall_quantile
                ),
                turnover_scale=metrics_protocol.turnover_scale,
            )
            rows.append({"model": model_name, "delay": delay, **values})
    return pd.DataFrame.from_records(rows)


def baseline_paths(
    frame: pd.DataFrame, study: BaselineStudy, config: ResearchConfig
) -> dict[int, dict[str, pd.DataFrame]]:
    """Return OOS accounting paths only after every boundary passes."""
    if study.boundaries.empty or not study.boundaries["passed"].all():
        raise WalkForwardError("OOS paths are sealed until all boundary checks pass")
    returns = frame[["date", "equity_simple", "cash_return"]]
    dates = pd.to_datetime(returns["date"], errors="raise")
    oos = dates >= pd.Timestamp(study.oos_start)
    unaligned: dict[int, dict[str, pd.DataFrame]] = {}
    for delay in config.backtest_protocol.robustness_delays:
        paths = {"buy_and_hold": buy_and_hold(returns)}
        for model_name in ("hmm", "fixed_jm"):
            selection = study.selections[model_name][delay]
            paths[model_name] = apply_signal(
                returns,
                selection.signal.reset_index(drop=True),
                delay_trading_days=delay,
                one_way_cost_bps=config.backtest_protocol.one_way_cost_bps,
            )
        unaligned[delay] = {
            model_name: path.loc[oos].reset_index(drop=True)
            for model_name, path in paths.items()
        }

    metric_columns = [
        "cash_return",
        "position",
        "one_way_turnover",
        "strategy_return",
    ]
    sample = config.metrics_protocol.comparison_sample
    if sample == LEGACY_COMPARISON_SAMPLE:
        output: dict[int, dict[str, pd.DataFrame]] = {}
        for delay, paths in unaligned.items():
            complete = pd.concat(
                [
                    path[metric_columns].notna().all(axis=1)
                    for path in paths.values()
                ],
                axis=1,
            ).all(axis=1)
            if not complete.any():
                raise WalkForwardError("no common OOS metric rows")
            output[delay] = {
                model_name: path.loc[complete].reset_index(drop=True)
                for model_name, path in paths.items()
            }
        return output
    if sample != PAPER_COMPARISON_SAMPLE:
        raise WalkForwardError("unknown comparison sample rule")

    complete = pd.concat(
        [
            path[metric_columns].notna().all(axis=1)
            for paths in unaligned.values()
            for path in paths.values()
        ],
        axis=1,
    ).all(axis=1)
    if not complete.any():
        raise WalkForwardError("no common OOS metric rows across delays")
    return {
        delay: {
            model_name: path.loc[complete].reset_index(drop=True)
            for model_name, path in paths.items()
        }
        for delay, paths in unaligned.items()
    }


def select_monthly_candidate(
    returns: pd.DataFrame,
    candidate_states: pd.DataFrame,
    protocol: SelectionProtocol,
    *,
    delay_trading_days: int,
    one_way_cost_bps: float,
    periods_per_year: int = 252,
    volatility_ddof: int = 1,
    initial: SelectionProgress | None = None,
    checkpoint_every: int = 12,
    progress: Callable[[SelectionProgress], None] | None = None,
) -> SelectionResult:
    """Select a state path monthly using only trailing validation returns."""
    if checkpoint_every < 1:
        raise WalkForwardError("selection checkpoint interval must be positive")
    if protocol.tie_rule not in {"lower_smoothing", "higher_smoothing"}:
        raise WalkForwardError(
            "selection tie rule must be lower_smoothing or higher_smoothing"
        )
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
    decision_dates = pd.DatetimeIndex([])
    if first_complete is not None:
        earliest = first_complete + pd.DateOffset(years=protocol.validation_years)
        decision_dates = _month_end_dates(dates)
        decision_dates = decision_dates[decision_dates >= earliest]
    completed = 0
    if initial is not None:
        choices, surface, completed, selection_started = _resume_selection(
            initial, decision_dates, candidates
        )
    if first_complete is not None:
        cash_return = prepared.set_index("date")["cash_return"]
        for decision_date in decision_dates[completed:]:
            selected = _score_decision(
                decision_date,
                candidates,
                states,
                candidate_returns,
                cash_return,
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
            else:
                selection_started = True
                choices.append({"decision_date": decision_date, "selected": selected})
            completed += 1
            if progress is not None and (
                completed % checkpoint_every == 0 or completed == len(decision_dates)
            ):
                progress(
                    SelectionProgress(
                        pd.DataFrame.from_records(
                            choices, columns=["decision_date", "selected"]
                        ),
                        pd.DataFrame.from_records(surface),
                    )
                )

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


def _resume_selection(
    initial: SelectionProgress,
    decision_dates: pd.DatetimeIndex,
    candidates: tuple[float, ...],
) -> tuple[list[dict[str, object]], list[dict[str, object]], int, bool]:
    choices, surface = initial.choices.copy(), initial.surface.copy()
    if tuple(choices.columns) != ("decision_date", "selected") or tuple(
        surface.columns
    ) != ("decision_date", "candidate", "valid_returns", "sharpe", "eligible"):
        raise WalkForwardError("selection checkpoint schema is invalid")
    if not candidates:
        if not choices.empty or not surface.empty:
            raise WalkForwardError("selection checkpoint has no candidate grid")
        return [], [], 0, False
    if len(surface) % len(candidates):
        raise WalkForwardError("selection checkpoint contains a partial CV month")
    completed = len(surface) // len(candidates)
    if completed > len(decision_dates):
        raise WalkForwardError("selection checkpoint exceeds the decision calendar")
    try:
        surface_dates = pd.to_datetime(surface["decision_date"], errors="raise")
        surface_candidates = pd.to_numeric(surface["candidate"], errors="raise")
        choice_dates = pd.DatetimeIndex(
            pd.to_datetime(choices["decision_date"], errors="raise")
        )
        selected = pd.to_numeric(choices["selected"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise WalkForwardError("selection checkpoint values are invalid") from exc
    expected_dates = decision_dates[:completed]
    expected = pd.MultiIndex.from_product([expected_dates, candidates])
    observed = pd.MultiIndex.from_arrays([surface_dates, surface_candidates])
    if not observed.equals(expected):
        raise WalkForwardError("selection checkpoint is not a causal CV prefix")
    if not selected.isin(candidates).all():
        raise WalkForwardError("selection checkpoint choice is outside the grid")
    if not choices.empty:
        first = expected_dates.get_indexer([choice_dates[0]])[0]
        if first < 0 or not choice_dates.equals(expected_dates[first:]):
            raise WalkForwardError("selection checkpoint choices are not contiguous")
    return (
        choices.to_dict("records"),
        surface.to_dict("records"),
        completed,
        not choices.empty,
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
    tied = [
        candidate
        for candidate, score in eligible
        if best - score <= protocol.tie_tolerance
    ]
    choose = min if protocol.tie_rule == "lower_smoothing" else max
    return choose(tied)


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
