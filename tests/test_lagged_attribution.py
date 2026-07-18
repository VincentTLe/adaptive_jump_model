from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from adaptive_jump import lagged_attribution as attribution
from adaptive_jump.config import load_config
from adaptive_jump.walkforward import SelectionResult

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "research/lagged-selection-attribution-001.toml"
REGISTRY = ROOT / "research/experiment_registry.jsonl"


def test_frozen_attribution_spec_matches_latest_registry() -> None:
    config = load_config(ROOT / "research.toml")
    spec = attribution.load_attribution_spec(SPEC, config)
    rows = [
        json.loads(line)
        for line in REGISTRY.read_text(encoding="utf-8").splitlines()
        if json.loads(line).get("experiment_id") == spec.experiment_id
    ]

    assert rows[-1]["status"] == "EXPERIMENT_COMPLETE"
    assert rows[-1]["frozen_spec_hash"] == spec.sha256
    assert spec.cells == attribution.CELLS
    assert spec.lambdas == attribution.LAMBDAS
    assert spec.cutoff.isoformat() == "2023-12-31"


def test_two_by_two_signal_axes_are_not_reversed() -> None:
    dates = pd.bdate_range("2020-01-02", periods=4)
    frame = pd.DataFrame({"date": dates})
    states = {
        "fixed": pd.DataFrame({0.0: [0.0] * 4, 5.0: [1.0] * 4}, index=dates),
        "lagged_log4": pd.DataFrame({0.0: [1.0] * 4, 5.0: [0.0] * 4}, index=dates),
    }
    choices = pd.DataFrame(
        [
            {
                "decision_date": dates[0],
                "selected": 0.0,
                "market": "us",
                "model": "fixed",
            },
            {
                "decision_date": dates[0],
                "selected": 5.0,
                "market": "us",
                "model": "lagged_log4",
            },
        ]
    )
    inputs = SimpleNamespace(choices=choices)

    selected = attribution._cell_selections(frame, states, inputs, "us")

    assert selected["FF"].signal.tolist() == [1.0] * 4
    assert selected["FL"].signal.tolist() == [0.0] * 4
    assert selected["LF"].signal.tolist() == [0.0] * 4
    assert selected["LL"].signal.tolist() == [1.0] * 4


def _synthetic_summary() -> pd.DataFrame:
    rows = []
    values = {"FF": 1.0, "FL": 1.2, "LF": 1.3, "LL": 1.8}
    for market_index, market in enumerate(attribution.MARKETS):
        for cell, value in values.items():
            row = {"market": market, "cell": cell}
            for metric_index, metric in enumerate(attribution.METRICS):
                row[metric] = value + market_index + metric_index
            rows.append(row)
    return pd.DataFrame(rows)


def test_shapley_and_interaction_identities_hold_for_every_metric() -> None:
    result = attribution._attribution_rows(_synthetic_summary(), 1e-12)

    assert len(result) == 20
    assert np.allclose(
        result["path_shapley"] + result["choice_shapley"],
        result["total"],
        rtol=0,
        atol=1e-12,
    )
    assert np.allclose(
        result["path_at_fixed_choices"]
        + result["choice_at_fixed_path"]
        + result["interaction"],
        result["total"],
        rtol=0,
        atol=1e-12,
    )
    assert (result["market"] == "equal_market_mean").sum() == 5


def test_diagnostic_decision_cannot_select_a_winner_or_pass_performance() -> None:
    result = attribution._attribution_rows(_synthetic_summary(), 1e-12)
    decision = attribution._decision(result)

    assert decision["result"] == "diagnostic_complete"
    assert decision["supported_or_not_supported"] is None
    assert decision["cell_winner_selected"] is False
    assert decision["causal_claim_allowed"] is False
    assert decision["performance_claim_allowed"] is False


def test_cross_cell_signal_reaches_position_at_t_plus_2_with_10bps_cost() -> None:
    config = load_config(ROOT / "research.toml")
    dates = pd.bdate_range("2020-01-02", periods=6, name="date")
    frame = pd.DataFrame(
        {
            "date": dates,
            "equity_simple": [0.0] * 6,
            "cash_return": [0.0] * 6,
        }
    )
    signal = pd.Series([1.0, 1.0, 0.0, 0.0, 0.0, 0.0], index=dates)
    selection = SelectionResult(
        signal=signal,
        choices=pd.DataFrame({"decision_date": [dates[0]], "selected": [0.0]}),
        surface=pd.DataFrame(),
        candidate_returns=pd.DataFrame(index=dates),
    )

    path = attribution._full_path(frame, selection, config)

    assert path.loc[4, "position"] == signal.iloc[2]
    assert path.loc[4, "one_way_turnover"] == 1.0
    assert path.loc[4, "transaction_cost"] == pytest.approx(0.001)
    assert attribution.TURNOVER_SCALE == 0.5


def test_change_trace_skips_initial_nan_trade_and_finds_real_trade() -> None:
    dates = pd.bdate_range("2020-01-02", periods=6, name="date")
    frame = pd.DataFrame({"date": dates})

    def selection(signal: list[float], selected: float) -> SelectionResult:
        return SelectionResult(
            signal=pd.Series(signal, index=dates),
            choices=pd.DataFrame({"decision_date": [dates[0]], "selected": [selected]}),
            surface=pd.DataFrame(),
            candidate_returns=pd.DataFrame(index=dates),
        )

    selections = {
        "FF": selection([0, 1, 1, 1, 1, 1], 0.0),
        "FL": selection([0, 0, 1, 1, 1, 1], 5.0),
        "LF": selection([0, 1, 1, 1, 1, 1], 0.0),
        "LL": selection([0, 1, 1, 1, 1, 1], 5.0),
    }

    def path(positions: list[float]) -> pd.DataFrame:
        position = pd.Series(positions, dtype=float)
        turnover = position.diff().abs()
        return pd.DataFrame(
            {
                "date": dates,
                "position": position,
                "one_way_turnover": turnover,
                "transaction_cost": 0.001 * turnover,
            }
        )

    ff = path([np.nan, np.nan, 0, 1, 1, 1])
    fl = path([np.nan, np.nan, 0, 0, 1, 1])
    full = {"FF": ff, "FL": fl, "LF": ff.copy(), "LL": ff.copy()}
    aligned = {
        cell: value.iloc[2:].reset_index(drop=True) for cell, value in full.items()
    }

    traces = attribution._change_traces(
        "us", frame, selections, full, aligned, offset=2
    )
    trade = traces.loc[
        (traces["cell"] == "FL") & (traces["event_type"] == "trade")
    ].iloc[0]

    assert trade["execution_date"] == dates[3]
    assert trade["ff_trade"] == 1.0
    assert trade["cell_trade"] == 0.0
    assert np.isfinite([trade["ff_trade"], trade["cell_trade"]]).all()


def test_artifact_allowlist_contains_exact_four_cell_trade_coverage() -> None:
    config = load_config(ROOT / "research.toml")
    spec = attribution.load_attribution_spec(SPEC, config)
    files = attribution._expected_files(spec)

    assert len(files) == 27
    assert {
        f"{market}/trades/{cell}.csv"
        for market in attribution.MARKETS
        for cell in attribution.CELLS
    }.issubset(files)


@pytest.mark.parametrize("identical_axis", ["choices", "states"])
def test_identical_axis_has_zero_effect_and_interaction(identical_axis: str) -> None:
    if identical_axis == "choices":
        values = {"FF": 1.0, "FL": 1.0, "LF": 2.0, "LL": 2.0}
        zero_fields = ("choice_at_fixed_path", "choice_shapley", "interaction")
    else:
        values = {"FF": 1.0, "FL": 2.0, "LF": 1.0, "LL": 2.0}
        zero_fields = ("path_at_fixed_choices", "path_shapley", "interaction")
    rows = []
    for market in attribution.MARKETS:
        for cell, value in values.items():
            rows.append(
                {
                    "market": market,
                    "cell": cell,
                    **{metric: value for metric in attribution.METRICS},
                }
            )

    result = attribution._attribution_rows(pd.DataFrame(rows), 1e-12)

    assert np.allclose(result[list(zero_fields)], 0.0, rtol=0, atol=1e-12)


def test_choice_schedule_rejects_candidate_outside_frozen_grid() -> None:
    choices = pd.DataFrame(
        [
            {
                "decision_date": pd.Timestamp("2020-01-31"),
                "selected": 999.0,
                "market": "us",
                "model": "fixed",
            }
        ]
    )
    with pytest.raises(attribution.AttributionError, match="invalid frozen"):
        attribution._schedule(SimpleNamespace(choices=choices), "us", "fixed")
