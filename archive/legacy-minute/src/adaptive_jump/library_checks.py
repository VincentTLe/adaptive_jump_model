"""Optional third-party backtest sanity checks."""

from __future__ import annotations

import warnings
from importlib import import_module

import numpy as np
import pandas as pd

from adaptive_jump.backtesting import backtest_metrics


def quantstats_metric_check(
    strategy_returns: np.ndarray | pd.Series,
    periods_per_year: int = 252 * 390,
) -> dict[str, object]:
    """Compute an independent metrics check using quantstats."""
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    r = _as_series(strategy_returns, "strategy_returns").dropna()
    if len(r) == 0:
        raise ValueError("strategy_returns must contain at least one finite row")
    qs = _optional_import("quantstats")
    if qs is None:
        return {
            "quantstats_status": "missing",
            "quantstats_periods_per_year": periods_per_year,
            "quantstats_sharpe": np.nan,
            "quantstats_max_drawdown": np.nan,
            "quantstats_expected_shortfall": np.nan,
        }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "quantstats_status": "ok",
            "quantstats_periods_per_year": periods_per_year,
            "quantstats_sharpe": float(qs.stats.sharpe(r, periods=periods_per_year)),
            "quantstats_max_drawdown": float(abs(qs.stats.max_drawdown(r))),
            "quantstats_expected_shortfall": float(qs.stats.expected_shortfall(r)),
        }


def vectorbt_signal_check(
    price: np.ndarray | pd.Series,
    positions: np.ndarray | pd.Series,
    transaction_cost: float = 0.0,
    periods_per_year: int = 252 * 390,
) -> dict[str, object]:
    """Run an event-style vectorbt sanity check for 0/1 positions."""
    if transaction_cost < 0.0 or not np.isfinite(transaction_cost):
        raise ValueError("transaction_cost must be finite and nonnegative")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    close = _as_series(price, "price")
    position = _as_series(positions, "positions", index=close.index).fillna(0.0)
    if len(close) == 0:
        raise ValueError("price must contain at least one row")
    if not (close > 0.0).all():
        raise ValueError("price must be positive")
    if not position.isin([0.0, 1.0]).all():
        raise ValueError("positions must be 0/1")
    vbt = _optional_import("vectorbt")
    if vbt is None:
        return {
            "vectorbt_status": "missing",
            "vectorbt_periods_per_year": periods_per_year,
            "vectorbt_total_return": np.nan,
            "vectorbt_sharpe": np.nan,
            "vectorbt_max_drawdown": np.nan,
            "vectorbt_trades": np.nan,
        }

    prior = position.shift(fill_value=0.0)
    entries = (position == 1.0) & (prior == 0.0)
    exits = (position == 0.0) & (prior == 1.0)
    portfolio = vbt.Portfolio.from_signals(
        close,
        entries=entries,
        exits=exits,
        fees=transaction_cost,
        init_cash=1.0,
        freq="1min",
    )
    return {
        "vectorbt_status": "ok",
        "vectorbt_periods_per_year": periods_per_year,
        "vectorbt_total_return": float(portfolio.total_return()),
        "vectorbt_sharpe": float(portfolio.sharpe_ratio(freq="1min", year_freq=f"{periods_per_year}min")),
        "vectorbt_max_drawdown": float(abs(portfolio.max_drawdown())),
        "vectorbt_trades": int(portfolio.trades.count()),
    }


def library_check_table(symbol: str, frames: dict[str, pd.DataFrame], transaction_cost: float) -> pd.DataFrame:
    rows = []
    for key, frame in frames.items():
        model = frame.attrs.get("model", key)
        policy = frame.attrs.get("backtest_policy", "legacy")
        invested_state = frame.attrs.get("invested_state", np.nan)
        price = (1.0 + frame["return"].fillna(0.0)).cumprod()
        local_metrics = backtest_metrics(frame["net_return"], frame["position"])
        row = {
            "symbol": symbol,
            "model": model,
            "backtest_policy": policy,
            "invested_state": invested_state,
            "local_total_return": float(frame["equity"].iloc[-1] - 1.0),
            "local_sharpe": local_metrics["sharpe"],
            "periods_per_year": 252 * 390,
        }
        qs_metrics = quantstats_metric_check(frame["net_return"], periods_per_year=252 * 390)
        vbt_metrics = vectorbt_signal_check(
            price,
            frame["position"],
            0.0 if model == "Buy and Hold" else transaction_cost,
            periods_per_year=252 * 390,
        )
        row.update(qs_metrics)
        row.update(vbt_metrics)
        row["quantstats_sharpe_abs_diff"] = abs(row["local_sharpe"] - row["quantstats_sharpe"])
        row["vectorbt_total_return_abs_diff"] = abs(row["local_total_return"] - row["vectorbt_total_return"])
        row["vectorbt_sharpe_abs_diff"] = abs(row["local_sharpe"] - row["vectorbt_sharpe"])
        rows.append(row)
    return pd.DataFrame(rows)


def _optional_import(package: str):
    try:
        return import_module(package)
    except ModuleNotFoundError as exc:
        if exc.name == package:
            return None
        raise


def _as_series(values, name: str, index: pd.Index | None = None) -> pd.Series:
    if isinstance(values, pd.Series):
        result = values.astype(float)
        if index is not None:
            result = result.reindex(index)
    else:
        result = pd.Series(values, index=index, name=name, dtype=float)
    if not np.isfinite(result.dropna().to_numpy()).all():
        raise ValueError(f"{name} must be finite apart from NaN rows")
    return result
