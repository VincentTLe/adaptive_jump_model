"""Causal selection, accounting, and dated evidence for the study."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from adaptive_jump.artifacts import TRADE_COLUMNS
from adaptive_jump.backtest import apply_signal, performance_metrics
from adaptive_jump.confidence_model import StateEvidence
from adaptive_jump.confidence_spec import BETAS, ConfidenceSpec, ConfidenceStudyError
from adaptive_jump.config import ResearchConfig
from adaptive_jump.walkforward import SelectionResult, select_monthly_candidate


def _select_beta_paths(
    frame: pd.DataFrame,
    evidence: StateEvidence,
    config: ResearchConfig,
    spec: ConfidenceSpec,
) -> dict[float, SelectionResult]:
    returns = frame[["date", "equity_simple", "cash_return"]]
    return {
        beta: select_monthly_candidate(
            returns,
            evidence.states[beta],
            config.selection_protocol,
            delay_trading_days=config.backtest_protocol.primary_delay,
            one_way_cost_bps=config.backtest_protocol.one_way_cost_bps,
            periods_per_year=config.metrics_protocol.periods_per_year,
            volatility_ddof=config.metrics_protocol.volatility_ddof,
        )
        for beta in spec.betas
    }


def _assert_beta_zero_selection(
    parent: Path,
    market: str,
    selected: SelectionResult,
) -> None:
    root = parent / market / "fixed_jm-delay-1"
    choices = pd.read_csv(root / "choices.csv")
    expected_dates = pd.to_datetime(choices["decision_date"], errors="raise").to_numpy()
    observed_dates = pd.to_datetime(
        selected.choices["decision_date"], errors="raise"
    ).to_numpy()
    if not np.array_equal(expected_dates, observed_dates) or not np.array_equal(
        choices["selected"].to_numpy(dtype=float),
        selected.choices["selected"].to_numpy(dtype=float),
    ):
        raise ConfidenceStudyError(f"{market}: beta0 monthly choices differ from v7")

    stored_signal = pd.read_csv(root / "selected-signal.csv")
    expected_signal = stored_signal["selected_signal"].to_numpy(dtype=float)
    observed_signal = selected.signal.to_numpy(dtype=float)
    if not np.array_equal(expected_signal, observed_signal, equal_nan=True):
        raise ConfidenceStudyError(f"{market}: beta0 selected signal differs from v7")


def _full_path(
    frame: pd.DataFrame,
    selected: SelectionResult,
    config: ResearchConfig,
) -> pd.DataFrame:
    return apply_signal(
        frame[["date", "equity_simple", "cash_return"]],
        selected.signal.reset_index(drop=True),
        delay_trading_days=config.backtest_protocol.primary_delay,
        one_way_cost_bps=config.backtest_protocol.one_way_cost_bps,
    )


def _align_parent_sample(
    parent: Path,
    market: str,
    full_path: pd.DataFrame,
    *,
    beta_zero: bool,
) -> pd.DataFrame:
    stored = pd.read_csv(parent / market / "trades/fixed_jm-delay-1.csv")
    stored["date"] = pd.to_datetime(stored["date"], errors="raise")
    path = full_path.copy()
    path["date"] = pd.to_datetime(path["date"], errors="raise")
    indexed = path.set_index("date")
    dates = pd.DatetimeIndex(stored["date"])
    if not dates.isin(indexed.index).all():
        raise ConfidenceStudyError(f"{market}: v7 OOS dates are not available")
    aligned = indexed.loc[dates].reset_index()
    if tuple(aligned.columns) != TRADE_COLUMNS or aligned.isna().any().any():
        raise ConfidenceStudyError(f"{market}: aligned trade path is incomplete")
    if beta_zero:
        exact_columns = ["signal", "position", "one_way_turnover", "transaction_cost"]
        if not np.array_equal(
            aligned[exact_columns].to_numpy(),
            stored[exact_columns].to_numpy(),
        ):
            raise ConfidenceStudyError(f"{market}: beta0 accounting states differ")
        numeric = list(TRADE_COLUMNS[1:])
        if not np.allclose(
            aligned[numeric].to_numpy(),
            stored[numeric].to_numpy(),
            rtol=0,
            atol=1e-15,
        ):
            raise ConfidenceStudyError(f"{market}: beta0 trade path differs from v7")
    return aligned


def _active_lambda(dates: pd.DatetimeIndex, choices: pd.DataFrame) -> pd.Series:
    active = pd.Series(np.nan, index=dates, dtype=float)
    if not choices.empty:
        decision_dates = pd.DatetimeIndex(
            pd.to_datetime(choices["decision_date"], errors="raise")
        )
        active.loc[decision_dates] = choices["selected"].to_numpy(dtype=float)
        active = active.ffill()
    return active


def _selected_timeline(
    frame: pd.DataFrame,
    evidence: StateEvidence,
    selection: SelectionResult,
    full_path: pd.DataFrame,
    beta: float,
    config: ResearchConfig,
    market: str,
) -> pd.DataFrame:
    dates = pd.DatetimeIndex(pd.to_datetime(frame["date"], errors="raise"), name="date")
    active = _active_lambda(dates, selection.choices)
    state = 1.0 - selection.signal.reindex(dates)
    offset = config.backtest_protocol.return_offset
    records: list[dict[str, Any]] = []
    for row in range(len(dates) - offset):
        penalty = active.iloc[row]
        current = state.iloc[row]
        if pd.isna(penalty) or pd.isna(current):
            continue
        column = float(penalty)
        previous = state.iloc[row - 1] if row else math.nan
        previous_lambda = active.iloc[row - 1] if row else math.nan
        loss_0 = float(evidence.loss0.loc[dates[row], column])
        loss_1 = float(evidence.loss1.loc[dates[row], column])
        cost_01 = float(evidence.c01[beta].loc[dates[row], column])
        cost_10 = float(evidence.c10[beta].loc[dates[row], column])
        state_changed = pd.notna(previous) and int(previous) != int(current)
        selection_changed = (
            pd.notna(previous_lambda) and float(previous_lambda) != column
        )
        transition_cost = 0.0
        arrival_advantage = 0.0
        if state_changed:
            if int(previous) == 0 and int(current) == 1:
                transition_cost = cost_01
                arrival_advantage = max(loss_0 - loss_1, 0.0)
            else:
                transition_cost = cost_10
                arrival_advantage = max(loss_1 - loss_0, 0.0)
        execution = full_path.iloc[row + offset]
        records.append(
            {
                "market": market,
                "beta": beta,
                "beta_label": _beta_label(beta),
                "signal_date": dates[row],
                "lambda0": column,
                "q_train": float(evidence.q_train.loc[dates[row], column]),
                "loss_state_0": loss_0,
                "loss_state_1": loss_1,
                "c_0_to_1": cost_01,
                "c_1_to_0": cost_10,
                "previous_emitted_state": previous,
                "state": int(current),
                "state_changed": bool(state_changed),
                "lambda_changed": bool(selection_changed),
                "arrival_loss_advantage": arrival_advantage,
                "emitted_transition_penalty": transition_cost,
                "signal": float(selection.signal.loc[dates[row]]),
                "execution_date": pd.Timestamp(execution["date"]),
                "position": float(execution["position"]),
                "one_way_turnover": float(execution["one_way_turnover"]),
                "transaction_cost": float(execution["transaction_cost"]),
                "strategy_return": float(execution["strategy_return"]),
            }
        )
    return pd.DataFrame.from_records(records)


def _beta_label(beta: float) -> str:
    if beta == 0.0:
        return "0"
    if beta == BETAS[1]:
        return "log2"
    if beta == BETAS[2]:
        return "log4"
    raise ConfidenceStudyError(f"unexpected beta: {beta}")


def _metric_row(
    market: str,
    beta: float,
    path: pd.DataFrame,
    config: ResearchConfig,
) -> dict[str, Any]:
    protocol = config.metrics_protocol
    values = performance_metrics(
        path,
        periods_per_year=protocol.periods_per_year,
        volatility_ddof=protocol.volatility_ddof,
        expected_shortfall_quantile=protocol.expected_shortfall_quantile,
    )
    return {
        "market": market,
        "beta": beta,
        "beta_label": _beta_label(beta),
        **values,
        "cash_fraction": float(1.0 - path["position"].mean()),
        "switch_count": int((path["one_way_turnover"] > 0).sum()),
    }


def _add_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    baseline = metrics.loc[metrics["beta"] == 0.0]
    if len(baseline) != 1:
        raise ConfidenceStudyError("market summary has no unique beta0 baseline")
    base = baseline.iloc[0]
    output = metrics.copy()
    for metric in (
        "sharpe",
        "maximum_drawdown",
        "turnover",
        "cash_fraction",
        "switch_count",
    ):
        output[f"delta_{metric}"] = output[metric] - base[metric]
    reduced = (
        (output["delta_sharpe"] >= 0)
        & (output["delta_maximum_drawdown"] >= 0)
        & (output["delta_turnover"] <= 0)
        & (output["delta_switch_count"] <= 0)
        & (
            (output["delta_sharpe"] != 0)
            | (output["delta_maximum_drawdown"] != 0)
            | (output["delta_turnover"] != 0)
            | (output["delta_switch_count"] != 0)
        )
    )
    output["reduced_tradeoff"] = reduced & (output["beta"] != 0.0)
    return output
