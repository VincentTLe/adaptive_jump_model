from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from test_endpoint_grid_audit import _small_fixture

import adaptive_jump.endpoint_grid_audit as audit
import adaptive_jump.endpoint_grid_replay as replay
from adaptive_jump.endpoint_grid_replay_evidence import replay_path_changes
from adaptive_jump.walkforward import SelectionResult


def _d_metrics(j1_mdd: float = -0.19) -> pd.DataFrame:
    rows = []
    for market in audit.MARKETS:
        rows.extend(
            (
                {
                    "market": market,
                    "delay": 1,
                    "path": "buy_and_hold",
                    "sharpe": 0.5,
                    "maximum_drawdown": -0.2,
                },
                {
                    "market": market,
                    "delay": 1,
                    "path": "J1",
                    "sharpe": 0.8,
                    "maximum_drawdown": j1_mdd,
                },
                {
                    "market": market,
                    "delay": 1,
                    "path": "K1",
                    "sharpe": 0.6,
                    "maximum_drawdown": -0.18,
                },
            )
        )
    return pd.DataFrame.from_records(rows)


def test_d_rescue_decision_is_independent_descriptive_and_deadbanded() -> None:
    produced = audit.d_rescue_decision(_d_metrics())
    reconstructed = replay.replay_d_rescue_decision(_d_metrics())
    assert produced == reconstructed
    assert produced["all_markets_passed"] is True
    assert produced["performance_claim_allowed"] is False
    assert "conclusion" not in produced
    assert (
        audit.d_rescue_decision(_d_metrics(-0.1999999995))["all_markets_passed"]
        is False
    )
    assert (
        audit.d_rescue_decision(_d_metrics(-0.199999998))["all_markets_passed"] is True
    )


@pytest.mark.parametrize(
    ("relative", "column"),
    (
        ("jm-states.csv", "0.0"),
        ("jm-refits.csv", "objective"),
        ("fixed_jm-delay-1/choices.csv", "selected"),
        ("fixed_jm-delay-1/cv-surface.csv", "sharpe"),
        ("fixed_jm-delay-1/candidate-returns.csv", "0.0"),
        ("fixed_jm-delay-1/selected-signal.csv", "selected_signal"),
        ("boundaries.csv", "fraction"),
    ),
)
def test_selection_behavior_control_covers_every_witness_component(
    tmp_path: Path,
    relative: str,
    column: str,
) -> None:
    (
        config,
        source,
        endpoints,
        jm_grid,
        hmm_grid,
        fit,
        witness,
        _prepared,
        _result,
    ) = _small_fixture(tmp_path)
    path = witness / relative
    frame = pd.read_csv(path, float_precision="round_trip")
    row = frame[column].first_valid_index()
    frame.loc[row, column] = float(frame.loc[row, column]) + 1e-13
    frame.to_csv(path, index=False)

    with pytest.raises(audit.EndpointGridError, match="selection-behavior witness"):
        audit.prepare_market(
            source,
            config,
            endpoints,
            jm_grid,
            hmm_grid,
            witness,
            "7" * 40,
            current_jm=fit,
        )


def _selection(
    dates: pd.DatetimeIndex, selected: float, first_signal: float
) -> SelectionResult:
    signal = pd.Series(0.0, index=dates, name="selected_signal")
    signal.iloc[0] = first_signal
    choices = pd.DataFrame({"decision_date": [dates[0]], "selected": [selected]})
    return SelectionResult(signal, choices, pd.DataFrame(), pd.DataFrame(index=dates))


def _path(
    dates: pd.DatetimeIndex,
    position: list[float],
    turnover: list[float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": dates,
            "signal": 0.0,
            "position": position,
            "one_way_turnover": turnover,
        }
    )


def test_warmup_trace_links_pre_sample_choice_and_signal_to_first_trade() -> None:
    full_dates = pd.bdate_range("2020-01-02", periods=6, name="date")
    matched = full_dates[2:]
    base = _selection(full_dates, 0.0, 0.0)
    endpoint = _selection(full_dates, 1.0, 1.0)
    selections = {
        "J0": {1: base},
        "J1": {1: endpoint},
        "K0": {1: base},
        "K1": {1: base},
    }
    steady = _path(matched, [0.0] * 4, [0.0] * 4)
    changed = _path(matched, [1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0])
    paths = {
        1: {
            "buy_and_hold": steady,
            "J0": steady,
            "J1": changed,
            "K0": steady,
            "K1": steady,
        }
    }
    rows = []
    for path in audit.PATHS:
        rows.append(
            {
                "delay": 1,
                "path": path,
                "sharpe": 0.0,
                "maximum_drawdown": 0.0,
                "turnover": 0.0,
                "cash_fraction": 0.0,
                "switch_count": 0,
            }
        )
    metrics = pd.DataFrame.from_records(rows)

    produced, traces = audit.classify_path_changes(selections, paths, metrics, "us", 2)
    reconstructed, replay_traces = replay_path_changes(
        selections, paths, metrics, "us", 2
    )

    pd.testing.assert_frame_equal(produced, reconstructed)
    pd.testing.assert_frame_equal(traces, replay_traces)
    trace = traces.set_index("model").loc["fixed_jm"]
    assert trace["causal_chain_found"]
    assert trace["choice_change_date"] == full_dates[0]
    assert trace["signal_change_date"] == full_dates[0]
    assert trace["signal_change_date"] < matched[0]
    assert trace["t_plus_2_position_date"] == matched[0]
    assert trace["trade_turnover_change_date"] == matched[0]


@pytest.mark.parametrize(("delta", "changed"), ((5e-13, False), (2e-12, True)))
def test_path_metric_tolerance_is_strictly_greater_than_one_e_minus_twelve(
    tmp_path: Path, delta: float, changed: bool
) -> None:
    config, *_, result = _small_fixture(tmp_path)
    metrics = result.metrics.copy()
    baseline = metrics.set_index("path").loc["K0", "sharpe"]
    metrics.loc[metrics["path"] == "K1", "sharpe"] = float(baseline) + delta
    produced, _ = audit.classify_path_changes(
        result.selections,
        result.paths,
        metrics,
        "us",
        config.backtest_protocol.return_offset,
    )
    hmm = produced.set_index("model").loc["hmm"]
    assert bool(hmm["metric_changed"]) is changed
