"""Accounting and per-market serialization for the endpoint-grid audit."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from adaptive_jump.backtest import apply_signal, buy_and_hold, performance_metrics
from adaptive_jump.config import ResearchConfig
from adaptive_jump.endpoint_grid_types import EndpointGridError
from adaptive_jump.walkforward import SelectionResult

PATHS = ("buy_and_hold", "J0", "J1", "K0", "K1")
SELECTION_PATHS = PATHS[1:]


def accounting_paths(
    returns: pd.DataFrame,
    selections: dict[str, dict[int, SelectionResult]],
    oos_start: date,
    config: ResearchConfig,
) -> dict[int, dict[str, pd.DataFrame]]:
    """Construct the five matched accounting paths for every delay."""
    unaligned: dict[int, dict[str, pd.DataFrame]] = {}
    oos = pd.to_datetime(returns["date"]) >= pd.Timestamp(oos_start)
    for delay in config.backtest_protocol.robustness_delays:
        paths = {"buy_and_hold": buy_and_hold(returns)}
        paths.update(
            {
                path: apply_signal(
                    returns,
                    selections[path][delay].signal.reset_index(drop=True),
                    delay_trading_days=delay,
                    one_way_cost_bps=config.backtest_protocol.one_way_cost_bps,
                )
                for path in SELECTION_PATHS
            }
        )
        unaligned[delay] = {
            name: path.loc[oos].reset_index(drop=True) for name, path in paths.items()
        }
    required = ["cash_return", "position", "one_way_turnover", "strategy_return"]
    complete = pd.concat(
        [
            path[required].notna().all(axis=1)
            for paths in unaligned.values()
            for path in paths.values()
        ],
        axis=1,
    ).all(axis=1)
    if not complete.any():
        raise EndpointGridError("no common OOS rows across all paths and delays")
    output = {
        delay: {
            name: path.loc[complete].reset_index(drop=True)
            for name, path in paths.items()
        }
        for delay, paths in unaligned.items()
    }
    dates = [
        pd.DatetimeIndex(path["date"])
        for paths in output.values()
        for path in paths.values()
    ]
    if any(not value.equals(dates[0]) for value in dates[1:]):
        raise EndpointGridError("accounting paths do not share exact dates")
    return output


def path_metrics(
    paths: dict[int, dict[str, pd.DataFrame]], config: ResearchConfig
) -> pd.DataFrame:
    """Measure every already-matched path using the frozen metric protocol."""
    rows = []
    protocol = config.metrics_protocol
    for delay, by_path in paths.items():
        if tuple(by_path) != PATHS:
            raise EndpointGridError("materialized path set changed")
        for path, frame in by_path.items():
            values = performance_metrics(
                frame,
                periods_per_year=protocol.periods_per_year,
                volatility_ddof=protocol.volatility_ddof,
                expected_shortfall_quantile=protocol.expected_shortfall_quantile,
                turnover_scale=protocol.turnover_scale,
            )
            rows.append(
                {
                    "delay": delay,
                    "path": path,
                    **values,
                    "cash_fraction": float(1.0 - frame["position"].mean()),
                    "switch_count": int((frame["one_way_turnover"] > 0).sum()),
                }
            )
    return pd.DataFrame.from_records(rows)


def write_market(target: Path, result: Any) -> None:
    """Write the exact per-market allowlisted evidence."""
    target.mkdir(parents=True)
    result.endpoint_jm.to_csv(target / "endpoint-jm-states.csv")
    result.endpoint_refits.to_csv(target / "endpoint-jm-refits.csv", index=False)
    for path, by_delay in result.selections.items():
        for delay, selection in by_delay.items():
            directory = target / f"{path}-delay-{delay}"
            directory.mkdir()
            selection.choices.to_csv(directory / "choices.csv", index=False)
            selection.surface.to_csv(directory / "cv-surface.csv", index=False)
            selection.signal.to_csv(directory / "selected-signal.csv", header=True)
    trades = target / "trades"
    trades.mkdir()
    for delay, paths in result.paths.items():
        for path, frame in paths.items():
            frame.to_csv(trades / f"{path}-delay-{delay}.csv", index=False)
