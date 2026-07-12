"""Causal signal timing and transaction-cost accounting."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


class BacktestError(ValueError):
    """Raised when a signal or return frame violates the accounting contract."""


def apply_signal(
    returns: pd.DataFrame,
    signal: pd.Series,
    *,
    delay_trading_days: int = 1,
    one_way_cost_bps: float = 10,
    charge_initial_allocation: bool = False,
) -> pd.DataFrame:
    """Apply an end-of-day binary signal to return t + delay + 1."""
    required = ["date", "equity_simple", "cash_return"]
    missing = [column for column in required if column not in returns]
    if missing:
        raise BacktestError(f"missing return columns: {missing}")
    if len(signal) != len(returns):
        raise BacktestError("signal and return lengths must match")
    if not isinstance(delay_trading_days, int) or delay_trading_days < 0:
        raise BacktestError("delay_trading_days must be a non-negative integer")
    if not math.isfinite(one_way_cost_bps) or one_way_cost_bps < 0:
        raise BacktestError("one_way_cost_bps must be finite and non-negative")

    signal_values = pd.Series(signal, index=returns.index, dtype=float)
    valid_signal = signal_values.dropna()
    if not valid_signal.isin([0.0, 1.0]).all():
        raise BacktestError("signal values must be 0, 1, or missing")

    result = returns[required].copy()
    result["signal"] = signal_values
    result["position"] = signal_values.shift(delay_trading_days + 1)
    result["gross_return"] = (
        result["position"] * result["equity_simple"]
        + (1.0 - result["position"]) * result["cash_return"]
    )

    previous_valid = result["position"].ffill().shift(1)
    turnover = (result["position"] - previous_valid).abs()
    first_valid = result["position"].first_valid_index()
    if first_valid is not None:
        turnover.loc[first_valid] = (
            abs(result.loc[first_valid, "position"])
            if charge_initial_allocation
            else 0.0
        )
    result["one_way_turnover"] = turnover
    result["transaction_cost"] = turnover * (one_way_cost_bps / 10_000.0)
    result["strategy_return"] = result["gross_return"] - result["transaction_cost"]

    finite_columns = ["equity_simple", "cash_return", "strategy_return"]
    for column in finite_columns:
        finite = result[column].dropna().map(math.isfinite)
        if not finite.all():
            raise BacktestError(f"{column} must be finite when present")
    return result.replace([np.inf, -np.inf], np.nan)
