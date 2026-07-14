"""Core calculations for the frozen JM training-window sensitivity."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import date
from functools import partial
from typing import Any

import numpy as np
import pandas as pd

from adaptive_jump.backtest import performance_metrics
from adaptive_jump.config import ResearchConfig
from adaptive_jump.inference import BootstrapProgress, bootstrap_sharpe_delta
from adaptive_jump.models import FixedJMResult, fixed_jm_states
from adaptive_jump.monitor import study_runtime
from adaptive_jump.monitor.events import EventObserver
from adaptive_jump.walkforward import (
    SelectionProgress,
    SelectionResult,
    boundary_diagnostic,
    select_monthly_candidate,
)
from adaptive_jump.window_spec import WindowStudySpec

COMPARISON_MODELS = ("buy_and_hold", "hmm_3000", "jm_3000", "jm_4000")
TRADE_COLUMNS = (
    "date",
    "equity_simple",
    "cash_return",
    "signal",
    "position",
    "gross_return",
    "one_way_turnover",
    "transaction_cost",
    "strategy_return",
)


class WindowStudyError(ValueError):
    """Raised when a window comparison cannot satisfy its frozen protocol."""


@dataclass(frozen=True)
class WindowMarketStudy:
    """JM-4000 model evidence for one market before metrics are opened."""

    oos_start: date
    jm: FixedJMResult
    selections: dict[int, SelectionResult]
    boundaries: pd.DataFrame


def build_window_market_study(
    frame: pd.DataFrame,
    config: ResearchConfig,
    spec: WindowStudySpec,
    *,
    oos_start: date,
    jm_initial: FixedJMResult | None = None,
    jm_progress: Callable[[FixedJMResult], None] | None = None,
    selection_initial: Callable[[int], SelectionProgress | None] | None = None,
    selection_progress: Callable[[int, SelectionProgress], None] | None = None,
    observer: EventObserver | None = None,
) -> WindowMarketStudy:
    """Fit and select JM-4000 while preserving every parent setting."""
    protocol = replace(config.model_protocol, fit_window=spec.challenger_window)
    jm = fixed_jm_states(
        frame,
        protocol,
        config.jm_protocol,
        initial=jm_initial,
        progress=jm_progress,
        observer=observer,
    )
    returns = frame[["date", "equity_simple", "cash_return"]]
    selections: dict[int, SelectionResult] = {}
    rows: list[dict[str, Any]] = []
    for delay in spec.delays:
        selection_observer = study_runtime.selection_progress_observer(
            observer, "jm_4000", delay
        )

        def record_selection(
            current: SelectionProgress,
            current_delay: int = delay,
            event_callback: Callable[[SelectionProgress], None]
            | None = selection_observer,
        ) -> None:
            if selection_progress is not None:
                selection_progress(current_delay, current)
            if event_callback is not None:
                event_callback(current)

        selection = select_monthly_candidate(
            returns,
            jm.states,
            config.selection_protocol,
            delay_trading_days=delay,
            one_way_cost_bps=config.backtest_protocol.one_way_cost_bps,
            periods_per_year=config.metrics_protocol.periods_per_year,
            volatility_ddof=config.metrics_protocol.volatility_ddof,
            checkpoint_every=1 if selection_observer else 12,
            initial=selection_initial(delay) if selection_initial else None,
            progress=(
                record_selection
                if selection_progress is not None or selection_observer is not None
                else None
            ),
        )
        selections[delay] = selection
        study_runtime.emit_selected_signal(observer, selection, "jm_4000", delay)
        diagnostic = boundary_diagnostic(
            selection.choices,
            config.jm_protocol.lambda_grid,
            oos_start=oos_start,
            fraction_limit=spec.boundary_fraction_limit,
        )
        rows.append({"model": "jm_4000", "delay": delay, **diagnostic.__dict__})
    return WindowMarketStudy(
        oos_start=oos_start,
        jm=jm,
        selections=selections,
        boundaries=pd.DataFrame.from_records(rows),
    )


def align_comparison_paths(
    paths: dict[str, pd.DataFrame], *, oos_start: date
) -> dict[str, pd.DataFrame]:
    """Restrict all four strategies to identical finite post-eligibility rows."""
    if tuple(paths) != COMPARISON_MODELS:
        raise WindowStudyError("comparison paths must use the frozen model order")
    indexed: dict[str, pd.DataFrame] = {}
    common: pd.DatetimeIndex | None = None
    for model, original in paths.items():
        if tuple(original.columns) != TRADE_COLUMNS or original.empty:
            raise WindowStudyError(f"{model}: trade schema is invalid")
        frame = original.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        if (
            frame["date"].duplicated().any()
            or not frame["date"].is_monotonic_increasing
        ):
            raise WindowStudyError(f"{model}: trade dates are invalid")
        frame = frame.set_index("date")
        indexed[model] = frame
        common = frame.index if common is None else common.intersection(frame.index)
    assert common is not None
    common = common[common >= pd.Timestamp(oos_start)]
    complete = pd.Series(True, index=common)
    numeric_columns = list(TRADE_COLUMNS[1:])
    for frame in indexed.values():
        values = frame.reindex(common)[numeric_columns]
        complete &= values.notna().all(axis=1) & np.isfinite(values).all(axis=1)
    common = common[complete]
    if common.empty:
        raise WindowStudyError("no common complete rows after JM-4000 eligibility")

    reference = indexed[COMPARISON_MODELS[0]].loc[
        common, ["equity_simple", "cash_return"]
    ]
    output = {}
    for model, frame in indexed.items():
        aligned = frame.loc[common]
        if not np.allclose(
            aligned[["equity_simple", "cash_return"]],
            reference,
            rtol=0,
            atol=1e-15,
        ):
            raise WindowStudyError(f"{model}: market returns differ on common dates")
        output[model] = aligned.reset_index()
    return output


def comparison_metrics(
    paths: dict[str, pd.DataFrame], config: ResearchConfig
) -> pd.DataFrame:
    """Calculate frozen metrics plus directly auditable position summaries."""
    if tuple(paths) != COMPARISON_MODELS:
        raise WindowStudyError("metric paths must use the frozen model order")
    rows = []
    protocol = config.metrics_protocol
    for model, path in paths.items():
        values = performance_metrics(
            path,
            periods_per_year=protocol.periods_per_year,
            volatility_ddof=protocol.volatility_ddof,
            expected_shortfall_quantile=protocol.expected_shortfall_quantile,
        )
        rows.append(
            {
                "model": model,
                **values,
                "cash_fraction": float(1.0 - path["position"].mean()),
                "switch_count": int((path["one_way_turnover"] > 0).sum()),
            }
        )
    return pd.DataFrame.from_records(rows)


def bootstrap_rows(
    paths: dict[str, pd.DataFrame],
    spec: WindowStudySpec,
    config: ResearchConfig,
    *,
    initial: Callable[[int], BootstrapProgress | None] | None = None,
    progress: Callable[[int, BootstrapProgress], None] | None = None,
) -> pd.DataFrame:
    """Run every preregistered paired stationary-bootstrap block length."""
    if tuple(paths) != COMPARISON_MODELS:
        raise WindowStudyError("bootstrap paths must use the frozen model order")
    challenger = paths["jm_4000"]
    baseline = paths["jm_3000"]
    rows = []
    for block in spec.bootstrap_blocks:
        checkpoint = initial(block) if initial else None
        save = partial(progress, block) if progress else None
        result = bootstrap_sharpe_delta(
            challenger["strategy_return"],
            baseline["strategy_return"],
            challenger["cash_return"],
            replications=spec.bootstrap_replications,
            mean_block_length=block,
            seed=spec.bootstrap_seed,
            confidence_level=spec.confidence_level,
            periods_per_year=config.metrics_protocol.periods_per_year,
            volatility_ddof=config.metrics_protocol.volatility_ddof,
            initial=checkpoint,
            progress=save,
        )
        rows.append(
            {
                "block_length": block,
                "observed_delta": result.observed,
                "lower_one_sided": result.lower_one_sided,
                "confidence_low": result.confidence_low,
                "confidence_high": result.confidence_high,
                "replications": result.replications,
            }
        )
    return pd.DataFrame.from_records(rows)


def window_claim(
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    *,
    market_ids: tuple[str, ...],
    primary_delay: int,
    primary_block: int,
) -> dict[str, Any]:
    """Evaluate the frozen exploratory direction and uncertainty rules."""
    primary = metrics.loc[metrics["delay"] == primary_delay]
    uncertainty = bootstrap.loc[bootstrap["block_length"] == primary_block]
    rows = []
    for market in market_ids:
        values = primary.loc[primary["market"] == market].set_index("model")
        if set(values.index) != set(COMPARISON_MODELS):
            raise WindowStudyError(f"{market}: incomplete primary metrics")
        bounds = uncertainty.loc[uncertainty["market"] == market]
        if len(bounds) != 1:
            raise WindowStudyError(f"{market}: missing primary bootstrap row")
        baseline = float(values.loc["jm_3000", "sharpe"])
        challenger = float(values.loc["jm_4000", "sharpe"])
        delta = challenger - baseline
        observed = float(bounds.iloc[0]["observed_delta"])
        if not math.isclose(delta, observed, rel_tol=0, abs_tol=1e-12):
            raise WindowStudyError(f"{market}: metric and bootstrap deltas disagree")
        lower = float(bounds.iloc[0]["lower_one_sided"])
        rows.append(
            {
                "market": market,
                "sharpe_jm_3000": baseline,
                "sharpe_jm_4000": challenger,
                "delta_sharpe": delta,
                "positive": delta > 0,
                "lower_one_sided_95pct": lower,
                "uncertainty_positive": lower > 0,
            }
        )
    positives = sum(row["positive"] for row in rows)
    outcome = (
        "consistent improvement"
        if positives == len(rows)
        else ("mixed" if positives else "not supported")
    )
    uncertainty_supported = len(rows) == len(market_ids) and all(
        row["uncertainty_positive"] for row in rows
    )
    return {
        "claim_class": "EXPLORATORY",
        "primary_delay": primary_delay,
        "primary_block_length": primary_block,
        "markets": rows,
        "positive_markets": positives,
        "directional_outcome": outcome,
        "uncertainty_supported": uncertainty_supported,
        "conclusion": (
            f"{outcome}; paired uncertainty support "
            + ("present" if uncertainty_supported else "absent")
        ),
    }
