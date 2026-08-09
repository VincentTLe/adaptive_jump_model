"""Causal signal timing and transaction-cost accounting."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from adaptive_jump.config import (
    FLAT_IN_CASH_DRAWDOWN_BASIS,
    TOTAL_WEALTH_DRAWDOWN_BASIS,
)


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

    # positional conversion: every caller passes a signal aligned by row order,
    # and label-based reindexing silently misaligns shifted or datetime indexes
    signal_values = pd.Series(
        np.asarray(signal, dtype=float), index=returns.index, dtype=float
    )
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


def buy_and_hold(returns: pd.DataFrame) -> pd.DataFrame:
    """Build the cost-free fully invested benchmark on every return date."""
    required = ["date", "equity_simple", "cash_return"]
    missing = [column for column in required if column not in returns]
    if missing:
        raise BacktestError(f"missing return columns: {missing}")
    result = returns[required].copy()
    result["signal"] = 1.0
    result["position"] = 1.0
    result["gross_return"] = result["equity_simple"]
    result["one_way_turnover"] = 0.0
    result["transaction_cost"] = 0.0
    result["strategy_return"] = result["equity_simple"]
    return result


def annualized_excess_sharpe(
    strategy_return: pd.Series,
    cash_return: pd.Series,
    *,
    periods_per_year: int = 252,
    volatility_ddof: int = 1,
) -> float:
    """Use mean strategy excess over volatility of strategy returns."""
    paired = pd.concat(
        [
            pd.Series(strategy_return, dtype=float).rename("strategy"),
            pd.Series(cash_return, dtype=float).rename("cash"),
        ],
        axis=1,
    ).dropna()
    if len(paired) <= volatility_ddof:
        return math.nan
    if not np.isfinite(paired.to_numpy(dtype=float)).all():
        return math.nan
    volatility = paired["strategy"].std(ddof=volatility_ddof)
    if not math.isfinite(volatility) or volatility <= 0:
        return math.nan
    return float(
        math.sqrt(periods_per_year)
        * (paired["strategy"] - paired["cash"]).mean()
        / volatility
    )


def performance_metrics(
    result: pd.DataFrame,
    *,
    periods_per_year: int = 252,
    volatility_ddof: int = 1,
    expected_shortfall_quantile: float = 0.05,
    turnover_scale: float = 0.5,
    drawdown_basis: str = TOTAL_WEALTH_DRAWDOWN_BASIS,
) -> dict[str, float | int | str]:
    """Calculate metrics, reporting paper turnover unless explicitly overridden."""
    required = [
        "date",
        "cash_return",
        "position",
        "one_way_turnover",
        "strategy_return",
    ]
    if drawdown_basis == FLAT_IN_CASH_DRAWDOWN_BASIS:
        # The drawdown is measured on the risky leg alone, so the equity return
        # is needed and the cash leg is deliberately not.
        required.append("equity_simple")
    elif drawdown_basis != TOTAL_WEALTH_DRAWDOWN_BASIS:
        raise BacktestError(f"unknown drawdown basis: {drawdown_basis}")
    missing = [column for column in required if column not in result]
    if missing:
        raise BacktestError(f"missing metric columns: {missing}")
    if periods_per_year <= 0 or volatility_ddof < 0:
        raise BacktestError("metric period and ddof settings are invalid")
    if not 0 < expected_shortfall_quantile < 1:
        raise BacktestError("expected shortfall quantile must be between 0 and 1")
    if not math.isfinite(turnover_scale) or turnover_scale <= 0:
        raise BacktestError("turnover scale must be finite and positive")

    frame = result.loc[:, required].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame["date"].duplicated().any() or not frame["date"].is_monotonic_increasing:
        raise BacktestError("metric dates must be increasing and unique")
    frame = frame.dropna(subset=["strategy_return", "cash_return"])
    if len(frame) <= volatility_ddof:
        raise BacktestError("not enough valid returns for performance metrics")
    numeric = frame[
        ["strategy_return", "cash_return", "position", "one_way_turnover"]
    ].to_numpy(dtype=float)
    values = numeric[:, 0]
    if not np.isfinite(numeric).all() or (values <= -1).any():
        raise BacktestError("metric inputs must be finite and returns greater than -1")
    if not frame["position"].isin([0.0, 1.0]).all():
        raise BacktestError("metric positions must be binary")
    if (frame["one_way_turnover"] < 0).any():
        raise BacktestError("metric turnover must be non-negative")

    observations = len(frame)
    wealth = np.cumprod(1.0 + values)
    cagr = float(wealth[-1] ** (periods_per_year / observations) - 1.0)

    # Under the flat-in-cash basis, return and drawdown are read off different
    # paths. The old comment here claimed the paper pins this basis; that was
    # RETRACTED 2026-07-29 (see the constant's note in config.py) — the paper is
    # silent, the basis was chosen by fitting Table 4, and the current contracts
    # use total_wealth. The branch survives only so sealed v9.1-v9.3 runs replay
    # to their recorded numbers.
    if drawdown_basis == FLAT_IN_CASH_DRAWDOWN_BASIS:
        risky = (frame["position"] * frame["equity_simple"]).to_numpy(dtype=float)
        if not np.isfinite(risky).all() or (risky <= -1).any():
            raise BacktestError("equity returns must be finite and above -1")
        drawdown_wealth = np.cumprod(1.0 + risky)
    else:
        drawdown_wealth = wealth
    peaks = np.maximum.accumulate(np.r_[1.0, drawdown_wealth])
    drawdowns = np.r_[1.0, drawdown_wealth] / peaks - 1.0
    maximum_drawdown = float(drawdowns.min())
    volatility = float(
        frame["strategy_return"].std(ddof=volatility_ddof) * math.sqrt(periods_per_year)
    )
    excess = frame["strategy_return"] - frame["cash_return"]
    sharpe = annualized_excess_sharpe(
        frame["strategy_return"],
        frame["cash_return"],
        periods_per_year=periods_per_year,
        volatility_ddof=volatility_ddof,
    )
    annual_excess = float(excess.mean() * periods_per_year)
    calmar = annual_excess / abs(maximum_drawdown) if maximum_drawdown < 0 else math.nan
    threshold = frame["strategy_return"].quantile(expected_shortfall_quantile)
    expected_shortfall = float(
        frame.loc[frame["strategy_return"] <= threshold, "strategy_return"].mean()
    )
    return {
        "start": frame["date"].iloc[0].date().isoformat(),
        "end": frame["date"].iloc[-1].date().isoformat(),
        "observations": observations,
        "cagr": cagr,
        "volatility": volatility,
        "sharpe": sharpe,
        "maximum_drawdown": maximum_drawdown,
        "calmar": float(calmar),
        "expected_shortfall_5pct": expected_shortfall,
        "turnover": float(
            turnover_scale * frame["one_way_turnover"].mean() * periods_per_year
        ),
        "leverage": float(frame["position"].mean()),
    }
