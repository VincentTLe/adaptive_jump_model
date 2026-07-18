"""Binding and concrete path-change evidence produced by the audit runner."""

from __future__ import annotations

from typing import Any

import pandas as pd

from adaptive_jump.endpoint_grid_types import (
    METRIC_CHANGE_TOLERANCE,
    PRIMARY_DELAY,
    REPORTED_METRICS,
    EndpointGridError,
)
from adaptive_jump.walkforward import SelectionResult

PAIRS = (("fixed_jm", "J0", "J1"), ("hmm", "K0", "K1"))


def classify_path_changes(
    selections: dict[str, dict[int, SelectionResult]],
    paths: dict[int, dict[str, pd.DataFrame]],
    metrics: pd.DataFrame,
    market: str,
    return_offset: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Classify endpoint binding and retain one concise causal date trace."""
    summaries: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    indexed_metrics = metrics.set_index(["delay", "path"])
    for delay, by_path in paths.items():
        for model, baseline, endpoint in PAIRS:
            base_path, end_path = by_path[baseline], by_path[endpoint]
            dates = pd.DatetimeIndex(base_path["date"], name="date")
            if not dates.equals(pd.DatetimeIndex(end_path["date"], name="date")):
                raise EndpointGridError("change evidence paths have different dates")
            base_choices = _choices(selections[baseline][delay], dates, return_offset)
            end_choices = _choices(selections[endpoint][delay], dates, return_offset)
            if not base_choices.index.equals(end_choices.index):
                raise EndpointGridError("change evidence choices have different dates")
            choice_change = base_choices["selected"].ne(end_choices["selected"])
            base_signal, end_signal = base_path["signal"], end_path["signal"]
            comparable = base_signal.notna() & end_signal.notna()
            signal_change = comparable & base_signal.ne(end_signal)
            state_change = comparable & (1.0 - base_signal).ne(1.0 - end_signal)
            position_change = _numeric_change(
                base_path["position"], end_path["position"]
            )
            trade_change = _numeric_change(
                base_path["one_way_turnover"], end_path["one_way_turnover"]
            )
            base_metric = indexed_metrics.loc[(delay, baseline)]
            end_metric = indexed_metrics.loc[(delay, endpoint)]
            changed_fields = [
                field
                for field in REPORTED_METRICS
                if abs(float(end_metric[field]) - float(base_metric[field]))
                > METRIC_CHANGE_TOLERANCE
            ]
            binding = bool(
                choice_change.any()
                or signal_change.any()
                or state_change.any()
                or position_change.any()
                or trade_change.any()
                or changed_fields
            )
            summaries.append(
                {
                    "market": market,
                    "delay": delay,
                    "model": model,
                    "baseline_path": baseline,
                    "endpoint_path": endpoint,
                    "comparable_choice_months": len(choice_change),
                    "changed_choice_months": int(choice_change.sum()),
                    "changed_signal_days": int(signal_change.sum()),
                    "comparable_state_days": int(comparable.sum()),
                    "changed_state_days": int(state_change.sum()),
                    "changed_position_days": int(position_change.sum()),
                    "baseline_trade_days": int(
                        (base_path["one_way_turnover"] > 0).sum()
                    ),
                    "endpoint_trade_days": int(
                        (end_path["one_way_turnover"] > 0).sum()
                    ),
                    "changed_trade_turnover_days": int(trade_change.sum()),
                    "annual_turnover_delta": float(
                        end_metric["turnover"] - base_metric["turnover"]
                    ),
                    "changed_metric_fields": (
                        "|".join(changed_fields) if changed_fields else "none"
                    ),
                    "metric_changed": bool(changed_fields),
                    "binding": binding,
                }
            )
            if delay == PRIMARY_DELAY:
                choice_date, signal_date, position_date, trade_date = _causal_trace(
                    selections[baseline][delay],
                    selections[endpoint][delay],
                    choice_change,
                    dates,
                    position_change,
                    trade_change,
                    return_offset,
                )
                traces.append(
                    {
                        "market": market,
                        "model": model,
                        "baseline_path": baseline,
                        "endpoint_path": endpoint,
                        "delay": delay,
                        "choice_change_date": choice_date,
                        "signal_change_date": signal_date,
                        "t_plus_2_position_date": position_date,
                        "trade_turnover_change_date": trade_date,
                        "signal_to_position_offset": return_offset,
                        "causal_chain_found": not pd.isna(choice_date),
                        "binding": binding,
                    }
                )
    return pd.DataFrame.from_records(summaries), pd.DataFrame.from_records(traces)


def _choices(
    selection: SelectionResult, dates: pd.DatetimeIndex, offset: int
) -> pd.DataFrame:
    frame = selection.choices.copy()
    frame["decision_date"] = pd.to_datetime(frame["decision_date"], errors="raise")
    frame = frame.set_index("decision_date")
    signal_dates = pd.DatetimeIndex(selection.signal.index)
    position_start = signal_dates.get_loc(dates[0])
    signal_start = signal_dates[max(0, position_start - offset)]
    before = frame.loc[frame.index <= signal_start].tail(1)
    after = frame.loc[(frame.index > signal_start) & (frame.index <= dates[-1])]
    return pd.concat([before, after])


def _numeric_change(left: pd.Series, right: pd.Series) -> pd.Series:
    comparable = left.notna() & right.notna()
    return comparable & ((left - right).abs() > METRIC_CHANGE_TOLERANCE)


def _causal_trace(
    base_selection: SelectionResult,
    end_selection: SelectionResult,
    choice_change: pd.Series,
    dates: pd.DatetimeIndex,
    position_change: pd.Series,
    trade_change: pd.Series,
    offset: int,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    base_signal = base_selection.signal
    end_signal = end_selection.signal
    signal_dates = pd.DatetimeIndex(base_signal.index)
    if not signal_dates.equals(pd.DatetimeIndex(end_signal.index)):
        raise EndpointGridError("change evidence full signals have different dates")
    for position, changed in enumerate(position_change):
        if not changed or not trade_change.iloc[position]:
            continue
        full_position = signal_dates.get_loc(dates[position])
        signal_index = full_position - offset
        if signal_index < 0:
            continue
        left, right = base_signal.iloc[signal_index], end_signal.iloc[signal_index]
        if pd.isna(left) or pd.isna(right) or left == right:
            continue
        signal_date = pd.Timestamp(signal_dates[signal_index])
        active = choice_change.loc[choice_change.index <= signal_date]
        if active.empty or not bool(active.iloc[-1]):
            continue
        position_date = pd.Timestamp(dates[position])
        return pd.Timestamp(active.index[-1]), signal_date, position_date, position_date
    return pd.NaT, pd.NaT, pd.NaT, pd.NaT
