"""Independent path-change reconstruction for endpoint-grid artifacts."""

from __future__ import annotations

from typing import Any

import pandas as pd

from adaptive_jump.artifacts import ArtifactError
from adaptive_jump.walkforward import SelectionResult

REPLAY_CHANGE_TOLERANCE = 1e-12
REPLAY_PRIMARY_DELAY = 1
REPLAY_REPORTED_METRICS = (
    "sharpe",
    "maximum_drawdown",
    "turnover",
    "cash_fraction",
    "switch_count",
)
REPLAY_PAIRS = (("fixed_jm", "J0", "J1"), ("hmm", "K0", "K1"))


def replay_path_changes(
    selections: dict[str, dict[int, SelectionResult]],
    paths: dict[int, dict[str, pd.DataFrame]],
    metrics: pd.DataFrame,
    market: str,
    return_offset: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild binding counts and a concrete primary-delay causal trace."""
    summaries: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    metric_rows = metrics.set_index(["delay", "path"])
    for delay, delay_paths in paths.items():
        for model, baseline, endpoint in REPLAY_PAIRS:
            baseline_path = delay_paths[baseline]
            endpoint_path = delay_paths[endpoint]
            dates = pd.DatetimeIndex(baseline_path["date"], name="date")
            if not dates.equals(pd.DatetimeIndex(endpoint_path["date"], name="date")):
                raise ArtifactError("replay change-evidence dates differ")
            baseline_choices = _replay_choices(
                selections[baseline][delay], dates, return_offset
            )
            endpoint_choices = _replay_choices(
                selections[endpoint][delay], dates, return_offset
            )
            if not baseline_choices.index.equals(endpoint_choices.index):
                raise ArtifactError("replay change-evidence choice dates differ")
            choice_changed = baseline_choices["selected"].ne(
                endpoint_choices["selected"]
            )
            baseline_signal = baseline_path["signal"]
            endpoint_signal = endpoint_path["signal"]
            comparable_state = baseline_signal.notna() & endpoint_signal.notna()
            signal_changed = comparable_state & baseline_signal.ne(endpoint_signal)
            state_changed = comparable_state & (1.0 - baseline_signal).ne(
                1.0 - endpoint_signal
            )
            position_changed = _replay_numeric_change(
                baseline_path["position"], endpoint_path["position"]
            )
            trade_changed = _replay_numeric_change(
                baseline_path["one_way_turnover"],
                endpoint_path["one_way_turnover"],
            )
            baseline_metric = metric_rows.loc[(delay, baseline)]
            endpoint_metric = metric_rows.loc[(delay, endpoint)]
            changed_metrics = [
                field
                for field in REPLAY_REPORTED_METRICS
                if abs(float(endpoint_metric[field]) - float(baseline_metric[field]))
                > REPLAY_CHANGE_TOLERANCE
            ]
            binding = bool(
                choice_changed.any()
                or signal_changed.any()
                or state_changed.any()
                or position_changed.any()
                or trade_changed.any()
                or changed_metrics
            )
            summaries.append(
                {
                    "market": market,
                    "delay": delay,
                    "model": model,
                    "baseline_path": baseline,
                    "endpoint_path": endpoint,
                    "comparable_choice_months": len(choice_changed),
                    "changed_choice_months": int(choice_changed.sum()),
                    "changed_signal_days": int(signal_changed.sum()),
                    "comparable_state_days": int(comparable_state.sum()),
                    "changed_state_days": int(state_changed.sum()),
                    "changed_position_days": int(position_changed.sum()),
                    "baseline_trade_days": int(
                        (baseline_path["one_way_turnover"] > 0).sum()
                    ),
                    "endpoint_trade_days": int(
                        (endpoint_path["one_way_turnover"] > 0).sum()
                    ),
                    "changed_trade_turnover_days": int(trade_changed.sum()),
                    "annual_turnover_delta": float(
                        endpoint_metric["turnover"] - baseline_metric["turnover"]
                    ),
                    "changed_metric_fields": (
                        "|".join(changed_metrics) if changed_metrics else "none"
                    ),
                    "metric_changed": bool(changed_metrics),
                    "binding": binding,
                }
            )
            if delay == REPLAY_PRIMARY_DELAY:
                choice_date, signal_date, position_date, trade_date = _replay_trace(
                    selections[baseline][delay],
                    selections[endpoint][delay],
                    choice_changed,
                    dates,
                    position_changed,
                    trade_changed,
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


def _replay_choices(
    selection: SelectionResult, dates: pd.DatetimeIndex, offset: int
) -> pd.DataFrame:
    choices = selection.choices.copy()
    choices["decision_date"] = pd.to_datetime(choices["decision_date"], errors="raise")
    choices = choices.set_index("decision_date")
    signal_dates = pd.DatetimeIndex(selection.signal.index)
    position_start = signal_dates.get_loc(dates[0])
    signal_start = signal_dates[max(0, position_start - offset)]
    before = choices.loc[choices.index <= signal_start].tail(1)
    after = choices.loc[(choices.index > signal_start) & (choices.index <= dates[-1])]
    return pd.concat([before, after])


def _replay_numeric_change(left: pd.Series, right: pd.Series) -> pd.Series:
    comparable = left.notna() & right.notna()
    return comparable & ((left - right).abs() > REPLAY_CHANGE_TOLERANCE)


def _replay_trace(
    baseline_selection: SelectionResult,
    endpoint_selection: SelectionResult,
    choice_changed: pd.Series,
    dates: pd.DatetimeIndex,
    position_changed: pd.Series,
    trade_changed: pd.Series,
    offset: int,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    baseline_signal = baseline_selection.signal
    endpoint_signal = endpoint_selection.signal
    signal_dates = pd.DatetimeIndex(baseline_signal.index)
    if not signal_dates.equals(pd.DatetimeIndex(endpoint_signal.index)):
        raise ArtifactError("replay full-signal dates differ")
    for position, changed in enumerate(position_changed):
        if not changed or not trade_changed.iloc[position]:
            continue
        full_position = signal_dates.get_loc(dates[position])
        signal_index = full_position - offset
        if signal_index < 0:
            continue
        left = baseline_signal.iloc[signal_index]
        right = endpoint_signal.iloc[signal_index]
        if pd.isna(left) or pd.isna(right) or left == right:
            continue
        signal_date = pd.Timestamp(signal_dates[signal_index])
        active = choice_changed.loc[choice_changed.index <= signal_date]
        if active.empty or not bool(active.iloc[-1]):
            continue
        position_date = pd.Timestamp(dates[position])
        return (
            pd.Timestamp(active.index[-1]),
            signal_date,
            position_date,
            position_date,
        )
    return pd.NaT, pd.NaT, pd.NaT, pd.NaT
