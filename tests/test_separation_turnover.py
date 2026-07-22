"""Tests for the frozen separation-turnover exploratory mechanics."""

from pathlib import Path

import pandas as pd
import pytest

import adaptive_jump.separation_turnover as st


def _write_market(tmp_path: Path, market: str) -> Path:
    target = tmp_path / market / "dd_scaled_3x"
    target.mkdir(parents=True)
    refits = pd.DataFrame(
        {
            "fit_date": ["2020-01-15", "2020-01-15", "2020-07-15"],
            "training_start": ["2010-01-04"] * 3,
            "training_end": ["2020-01-15", "2020-01-15", "2020-07-15"],
            "observations": [3000] * 3,
            "scaler_mean": ["[0.001]"] * 3,
            "scaler_scale": ["[0.002]"] * 3,
            "lambda": [5.0, 15.0, 5.0],
            "objective": [1.0, 1.0, 1.0],
            "centers": ["[[-0.5], [1.5]]", "[[-0.25], [0.25]]", "[[0.0], [0.0]]"],
            "active_state_count": [2, 2, 1],
            "collapsed_to_one_state": [False, False, True],
        }
    )
    refits.to_csv(target / "refits.csv", index=False)
    choices = pd.DataFrame(
        {
            "decision_date": ["2020-01-31", "2020-02-28", "2020-07-31"],
            "selected": [5.0, 15.0, 5.0],
        }
    )
    choices.to_csv(target / "choices.csv", index=False)
    dates = ["2020-01-30", "2020-02-03", "2020-02-04", "2020-03-02", "2020-08-03"]
    trades = pd.DataFrame(
        {
            "date": dates,
            "equity_simple": [0.01] * 5,
            "cash_return": [0.0] * 5,
            "signal": [1, 1, 0, 0, 1],
            "position": [1, 1, 0, 0, 1],
            "gross_return": [0.01, 0.01, 0.0, 0.0, 0.01],
            "one_way_turnover": [0.0, 0.0, 1.0, 0.0, 1.0],
            "transaction_cost": [0.0, 0.0, 0.001, 0.0, 0.001],
            "strategy_return": [0.01, 0.01, -0.001, 0.0, 0.009],
        }
    )
    trades.to_csv(target / "trades.csv", index=False)
    return tmp_path


def test_parse_centers_requires_two_single_feature_states() -> None:
    assert st.parse_centers("[[-0.5], [1.5]]") == (-0.5, 1.5)
    with pytest.raises(st.SeparationTurnoverError):
        st.parse_centers("[[-0.5, 0.1], [1.5, 0.2]]")
    with pytest.raises(st.SeparationTurnoverError):
        st.parse_centers("[[-0.5]]")


def test_separation_uses_active_refit_and_collapsed_rule(tmp_path: Path) -> None:
    run_dir = _write_market(tmp_path, "us")
    table = st.separation_table(run_dir, "us", "dd_scaled_3x")

    assert list(table["selected_lambda"]) == [5.0, 15.0, 5.0]
    assert table.loc[0, "separation"] == pytest.approx(2.0)
    assert table.loc[1, "separation"] == pytest.approx(0.5)
    assert bool(table.loc[2, "collapsed"]) is True
    assert table.loc[2, "separation"] == 0.0


def test_windows_partition_and_count_boundary_switches(tmp_path: Path) -> None:
    run_dir = _write_market(tmp_path, "us")
    table = st.separation_table(run_dir, "us", "dd_scaled_3x")

    assert list(table["days_next"]) == [2, 1, 1]
    assert list(table["switches_next"]) == [1, 0, 1]
    assert list(table["turnover_next"]) == [1.0, 0.0, 1.0]
    assert pd.isna(table.loc[0, "switches_prev"])
    assert table.loc[1, "switches_prev"] == 1


def test_decide_applies_frozen_rule() -> None:
    negative = {"rho": -0.3, "p_one_sided": 0.01, "n": 100}
    weak = {"rho": -0.1, "p_one_sided": 0.30, "n": 100}
    positive = {"rho": 0.2, "p_one_sided": 0.99, "n": 100}

    assert st.decide({"us": negative, "de": negative, "jp": weak}) == "supported"
    assert st.decide({"us": positive, "de": positive, "jp": negative}) == (
        "not_supported"
    )
    assert st.decide({"us": weak, "de": weak, "jp": weak}) == "inconclusive"
    assert st.decide({"us": negative, "de": weak, "jp": positive}) == "inconclusive"
