"""Backtest and audit utilities for regime-derived 0/1 signals."""

from __future__ import annotations

import numpy as np
import pandas as pd


def positions_from_states(
    states: np.ndarray | pd.Series,
    delay_bars: int = 1,
    invested_state: int = 0,
) -> np.ndarray | pd.Series:
    """Convert states to 0/1 positions with a close-to-next-bar lag."""
    if delay_bars < 1:
        raise ValueError("delay_bars must be at least 1 for leakage-safe backtests")
    is_series = isinstance(states, pd.Series)
    index = states.index if is_series else None
    state_array = _validate_states(states)
    signal = (state_array == invested_state).astype(float)
    position = np.zeros(len(signal), dtype=float)
    if delay_bars < len(signal):
        position[delay_bars:] = signal[:-delay_bars]
    if is_series:
        return pd.Series(position, index=index, name="position")
    return position


def backtest_regime_01(
    returns: np.ndarray | pd.Series,
    states: np.ndarray | pd.Series,
    delay_bars: int = 1,
    transaction_cost: float = 0.0,
    periods_per_year: int = 252 * 390,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Vectorized 0/1 regime backtest with delay and one-way costs."""
    if transaction_cost < 0.0 or not np.isfinite(transaction_cost):
        raise ValueError("transaction_cost must be finite and nonnegative")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    r = _as_series(returns, "return")
    state_series = _as_series(states, "state", index=r.index)
    position = positions_from_states(state_series, delay_bars=delay_bars)
    signal_state = state_series.shift(delay_bars)
    turnover = position.diff().abs().fillna(position.abs())
    gross = position * r
    costs = transaction_cost * turnover
    net = gross - costs
    equity = (1.0 + net).cumprod()
    result = pd.DataFrame(
        {
            "return": r,
            "state": state_series.astype(int),
            "signal_state": signal_state,
            "position": position,
            "gross_return": gross,
            "turnover": turnover,
            "cost": costs,
            "net_return": net,
            "equity": equity,
        },
        index=r.index,
    )
    return result, backtest_metrics(result["net_return"], result["position"], periods_per_year=periods_per_year)


def backtest_metrics(
    strategy_returns: np.ndarray | pd.Series,
    positions: np.ndarray | pd.Series,
    periods_per_year: int = 252 * 390,
) -> dict[str, float]:
    """Compute standard vectorized backtest metrics."""
    r = _as_series(strategy_returns, "strategy_returns").dropna()
    p = _as_series(positions, "positions").reindex(r.index).fillna(0.0)
    if len(r) == 0:
        raise ValueError("strategy_returns must contain at least one finite row")
    equity = (1.0 + r).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    if total_return <= -1.0:
        annualized_return = -1.0
    else:
        annualized_return = float((1.0 + total_return) ** (periods_per_year / len(r)) - 1.0)
    volatility = float(r.std(ddof=1) * np.sqrt(periods_per_year)) if len(r) > 1 else 0.0
    sharpe = float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year)) if len(r) > 1 and r.std(ddof=1) > 0 else np.nan
    drawdown = equity / equity.cummax() - 1.0
    max_drawdown = float(-drawdown.min())
    calmar = float(annualized_return / max_drawdown) if max_drawdown > 0.0 else np.nan
    cutoff = r.quantile(0.05)
    expected_shortfall = float(r[r <= cutoff].mean())
    turnover = float(p.diff().abs().fillna(p.abs()).sum())
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "expected_shortfall_5pct": expected_shortfall,
        "turnover": turnover,
        "n_trades": int((p.diff().abs().fillna(p.abs()) > 0).sum()),
        "exposure": float(p.mean()),
    }


def trade_events_from_backtest_frame(
    frame: pd.DataFrame,
    symbol: str,
    model: str,
    delay_bars: int,
    transaction_cost: float,
) -> pd.DataFrame:
    """Extract one-way trade events from a vectorized 0/1 backtest frame."""
    if delay_bars < 0:
        raise ValueError("delay_bars must be nonnegative")
    if transaction_cost < 0.0 or not np.isfinite(transaction_cost):
        raise ValueError("transaction_cost must be finite and nonnegative")
    required = {"return", "state", "position", "turnover", "cost", "net_return", "equity"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"backtest frame is missing columns: {missing}")
    if len(frame) == 0:
        return pd.DataFrame(columns=_trade_event_columns())

    position = frame["position"].astype(float)
    previous_position = position.shift(fill_value=0.0)
    computed_turnover = (position - previous_position).abs()
    if not np.allclose(computed_turnover.to_numpy(), frame["turnover"].astype(float).to_numpy()):
        raise ValueError("backtest frame turnover does not match position changes")

    event_mask = computed_turnover > 0.0
    if not bool(event_mask.any()):
        return pd.DataFrame(columns=_trade_event_columns())

    index = frame.index
    event_locs = np.flatnonzero(event_mask.to_numpy())
    rows = []
    for event_id, loc in enumerate(event_locs, start=1):
        timestamp = index[loc]
        previous = float(previous_position.iloc[loc])
        current = float(position.iloc[loc])
        source_loc = loc - delay_bars
        source_timestamp = index[source_loc] if source_loc >= 0 else pd.NaT
        if "signal_state" in frame.columns:
            source_state = frame["signal_state"].iloc[loc]
        else:
            source_state = frame["state"].iloc[source_loc] if source_loc >= 0 else np.nan
        side = "buy" if current > previous else "sell"
        rows.append(
            {
                "symbol": symbol,
                "model": model,
                "event_id": event_id,
                "timestamp": timestamp,
                "bar_index": int(loc),
                "side": side,
                "previous_position": previous,
                "position": current,
                "trade_size": float(computed_turnover.iloc[loc]),
                "state_at_timestamp": int(frame["state"].iloc[loc]),
                "source_state": source_state,
                "source_state_timestamp": source_timestamp,
                "position_timing": f"delayed_by_{delay_bars}_bars" if delay_bars else "not_delayed",
                "delay_bars": int(delay_bars),
                "transaction_cost": float(transaction_cost),
                "bar_return": float(frame["return"].iloc[loc]),
                "gross_return": float(frame["gross_return"].iloc[loc]) if "gross_return" in frame.columns else np.nan,
                "cost": float(frame["cost"].iloc[loc]),
                "net_return": float(frame["net_return"].iloc[loc]),
                "equity_after_event": float(frame["equity"].iloc[loc]),
            }
        )
    return pd.DataFrame(rows, columns=_trade_event_columns())


def round_trips_from_backtest_frame(
    frame: pd.DataFrame,
    symbol: str,
    model: str,
    delay_bars: int,
    transaction_cost: float,
) -> pd.DataFrame:
    """Pair buy/sell events into auditable 0/1 round trips."""
    events = trade_events_from_backtest_frame(frame, symbol, model, delay_bars, transaction_cost)
    if events.empty:
        return pd.DataFrame(columns=_round_trip_columns())

    trips = []
    open_event = None
    trip_id = 1
    for event in events.to_dict("records"):
        if event["side"] == "buy":
            open_event = event
        elif event["side"] == "sell" and open_event is not None:
            trips.append(_round_trip_row(frame, symbol, model, trip_id, open_event, event, "closed"))
            trip_id += 1
            open_event = None

    if open_event is not None:
        trips.append(_round_trip_row(frame, symbol, model, trip_id, open_event, None, "open"))

    return pd.DataFrame(trips, columns=_round_trip_columns())


def make_backtest_outputs(
    symbol: str,
    returns: pd.Series,
    paths: dict[str, np.ndarray],
    delay_bars: int,
    transaction_cost: float,
    cost_grid: list[float],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    frames = {"Buy and Hold": _buy_hold_frame(returns)}
    trade_event_frames = [
        trade_events_from_backtest_frame(frames["Buy and Hold"], symbol, "Buy and Hold", 0, 0.0)
    ]
    round_trip_frames = [
        round_trips_from_backtest_frame(frames["Buy and Hold"], symbol, "Buy and Hold", 0, 0.0)
    ]
    buy_hold = backtest_metrics(returns, pd.Series(np.ones(len(returns)), index=returns.index))
    buy_hold.update({"symbol": symbol, "model": "Buy and Hold", "delay_bars": 0, "transaction_cost": 0.0, "is_primary_cost": True})
    rows.append(buy_hold)
    for name, states in paths.items():
        state_series = pd.Series(states, index=returns.index)
        frame, metrics = backtest_regime_01(returns, state_series, delay_bars=delay_bars, transaction_cost=transaction_cost)
        frames[name] = frame
        trade_event_frames.append(trade_events_from_backtest_frame(frame, symbol, name, delay_bars, transaction_cost))
        round_trip_frames.append(round_trips_from_backtest_frame(frame, symbol, name, delay_bars, transaction_cost))
        metrics.update({"symbol": symbol, "model": name, "delay_bars": delay_bars, "transaction_cost": transaction_cost, "is_primary_cost": True})
        rows.append(metrics)
        for cost in cost_grid:
            if cost == transaction_cost:
                continue
            _, sensitivity_metrics = backtest_regime_01(
                returns,
                state_series,
                delay_bars=delay_bars,
                transaction_cost=cost,
            )
            sensitivity_metrics.update(
                {
                    "symbol": symbol,
                    "model": name,
                    "delay_bars": delay_bars,
                    "transaction_cost": cost,
                    "is_primary_cost": False,
                }
            )
            rows.append(sensitivity_metrics)
    return (
        frames,
        pd.DataFrame(rows),
        pd.concat(trade_event_frames, ignore_index=True),
        pd.concat(round_trip_frames, ignore_index=True),
    )


def backtest_frame_table(symbol: str, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for model, frame in frames.items():
        out = frame.copy()
        out.insert(0, "timestamp", out.index)
        out.insert(1, "symbol", symbol)
        out.insert(2, "model", model)
        rows.append(out.reset_index(drop=True))
    return pd.concat(rows, ignore_index=True)


def _trade_event_columns() -> list[str]:
    return [
        "symbol",
        "model",
        "event_id",
        "timestamp",
        "bar_index",
        "side",
        "previous_position",
        "position",
        "trade_size",
        "state_at_timestamp",
        "source_state",
        "source_state_timestamp",
        "position_timing",
        "delay_bars",
        "transaction_cost",
        "bar_return",
        "gross_return",
        "cost",
        "net_return",
        "equity_after_event",
    ]


def _round_trip_columns() -> list[str]:
    return [
        "symbol",
        "model",
        "trip_id",
        "status",
        "entry_timestamp",
        "exit_timestamp",
        "mark_to_market_timestamp",
        "entry_bar_index",
        "exit_bar_index",
        "holding_bars",
        "gross_return",
        "net_return",
        "total_cost",
        "entry_equity",
        "exit_equity",
        "delay_bars",
        "entry_source_state",
        "entry_source_state_timestamp",
    ]


def _round_trip_row(
    frame: pd.DataFrame,
    symbol: str,
    model: str,
    trip_id: int,
    entry_event: dict[str, object],
    exit_event: dict[str, object] | None,
    status: str,
) -> dict[str, object]:
    entry_loc = int(entry_event["bar_index"])
    exit_loc = int(exit_event["bar_index"]) if exit_event is not None else len(frame) - 1
    window = frame.iloc[entry_loc : exit_loc + 1]
    return {
        "symbol": symbol,
        "model": model,
        "trip_id": int(trip_id),
        "status": status,
        "entry_timestamp": entry_event["timestamp"],
        "exit_timestamp": exit_event["timestamp"] if exit_event is not None else pd.NaT,
        "mark_to_market_timestamp": frame.index[exit_loc],
        "entry_bar_index": entry_loc,
        "exit_bar_index": exit_loc if exit_event is not None else np.nan,
        "holding_bars": int((window["position"] > 0.0).sum()),
        "gross_return": float((1.0 + window["gross_return"]).prod() - 1.0),
        "net_return": float((1.0 + window["net_return"]).prod() - 1.0),
        "total_cost": float(window["cost"].sum()),
        "entry_equity": float(frame["equity"].iloc[entry_loc]),
        "exit_equity": float(frame["equity"].iloc[exit_loc]),
        "delay_bars": int(entry_event["delay_bars"]),
        "entry_source_state": entry_event["source_state"],
        "entry_source_state_timestamp": entry_event["source_state_timestamp"],
    }


def _buy_hold_frame(returns: pd.Series) -> pd.DataFrame:
    r = returns.astype(float)
    position = pd.Series(np.ones(len(r)), index=r.index, name="position")
    turnover = position.diff().abs().fillna(position.abs())
    equity = (1.0 + r).cumprod()
    return pd.DataFrame(
        {
            "return": r,
            "state": 0,
            "signal_state": 0,
            "position": position,
            "gross_return": r,
            "turnover": turnover,
            "cost": 0.0,
            "net_return": r,
            "equity": equity,
        },
        index=r.index,
    )


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


def _validate_states(states) -> np.ndarray:
    values = np.asarray(states, dtype=int)
    if values.ndim != 1:
        raise ValueError("states must be a 1-D array")
    if len(values) == 0:
        raise ValueError("states must be non-empty")
    if not np.isfinite(np.asarray(states, dtype=float)).all():
        raise ValueError("states must be finite")
    if (values < 0).any():
        raise ValueError("states must be nonnegative")
    return values
