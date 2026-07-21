"""Synthetic smoke and lifecycle regressions for the balanced runner."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from adaptive_jump.artifacts import read_json
from adaptive_jump.balanced_model import BalancedStudyError, load_balanced_spec
from adaptive_jump.balanced_performance import _namespaced_inventory_entries
from adaptive_jump.balanced_runner import (
    _dated_audit,
    _finalize_verified_run,
    _mechanical_checks,
)
from adaptive_jump.balanced_smoke import (
    _actual_formula_checks,
    candidate_parity,
    independent_balanced_terminal_penalty,
    run_us_smoke,
)
from adaptive_jump.config import load_config
from adaptive_jump.lagged_model import LockedStateEvidence
from adaptive_jump.models import FEATURE_COLUMNS
from adaptive_jump.separation_analysis import MarketInputs
from adaptive_jump.tv_jump import loss_matrix

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def spec():
    config = load_config(ROOT / "research.toml")
    return load_balanced_spec(
        ROOT / "research/balanced-lagged-mechanism-001.toml", config
    )


def test_namespaced_inventory_entries_records_every_hashed_file():
    inventories = {
        "fixed": {"a.csv": "1" * 64, "shared.csv": "2" * 64},
        "oracle": {"shared.csv": "3" * 64},
    }
    assert _namespaced_inventory_entries(inventories) == {
        "fixed/a.csv": "1" * 64,
        "fixed/shared.csv": "2" * 64,
        "oracle/shared.csv": "3" * 64,
    }


def test_candidate_parity_is_strict_unless_prefix_mode_is_explicit():
    dates = pd.date_range("2020-01-01", periods=3, name="date")
    expected = pd.DataFrame({5.0: [0.0, 1.0, 1.0]}, index=dates)
    assert candidate_parity(expected.copy(), expected, "exact") == 3

    prefix = expected.iloc[:2]
    with pytest.raises(BalancedStudyError, match="dates changed"):
        candidate_parity(prefix, expected, "implicit-prefix")
    assert candidate_parity(prefix, expected, "prefix", mode="prefix") == 2

    extra = expected.copy()
    extra[15.0] = 0.0
    with pytest.raises(BalancedStudyError, match="columns changed"):
        candidate_parity(expected, extra, "extra-column")


def test_independent_terminal_formula_has_exact_pair_sum_and_bounds(spec):
    lambda0 = 4.0
    penalty = independent_balanced_terminal_penalty(
        np.array([3.0, 1.0]), lambda0, spec.decision_beta, 2.0
    )
    directed = penalty[~np.eye(2, dtype=bool)]
    lower = lambda0 * math.exp(-spec.decision_beta)
    upper = lambda0 * (2.0 - math.exp(-spec.decision_beta))

    assert penalty[0, 1] + penalty[1, 0] == pytest.approx(2.0 * lambda0)
    assert (directed >= lower).all()
    assert (directed <= upper).all()
    missing = independent_balanced_terminal_penalty(
        np.array([math.nan, 1.0]), lambda0, spec.decision_beta, 2.0
    )
    assert np.isfinite(missing).all()
    assert missing[0, 1] + missing[1, 0] == pytest.approx(2.0 * lambda0)


def _valid_smoke():
    return {
        "parent_lagged_exact": True,
        "beta_zero_exact": True,
        "short_long_prefix_exact": True,
        "future_mutation_prefix_invariant": True,
        "prefix_invariant": True,
        "future_mutation_effect_present": True,
        "actual_formula_exact": True,
        "actual_bounds_exact": True,
        "formula_through_second_refit": True,
        "pair_balance_exact": True,
        "balanced_discounts_present": True,
        "balanced_surcharges_present": True,
        "refit_convention_numeric": True,
        "performance_files_accessed": False,
        "return_columns_accessed": False,
        "post_2023_accessed": False,
    }


def _valid_replays():
    return {
        market: {
            "parent_lagged_exact": True,
            "beta_zero_exact": True,
            "candidate_coverage_exact": True,
            "pair_balance_exact": True,
            "return_columns_accessed": False,
        }
        for market in ("us", "de", "jp")
    }


def test_mechanical_gate_requires_actual_formula_and_full_candidate_coverage():
    prerequisites = {"passed": True}
    assert _mechanical_checks(prerequisites, _valid_smoke(), _valid_replays())["passed"]

    smoke = _valid_smoke()
    smoke["actual_formula_exact"] = False
    assert not _mechanical_checks(prerequisites, smoke, _valid_replays())["passed"]

    replays = _valid_replays()
    replays["jp"]["candidate_coverage_exact"] = False
    assert not _mechanical_checks(prerequisites, _valid_smoke(), replays)["passed"]


def test_lifecycle_verifies_before_exposing_complete(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "study.lock.toml").write_text("frozen = true\n")
    metadata_path = run_dir / "run.json"
    observed = []

    def fake_verify(path):
        assert path == run_dir
        assert (run_dir / "inventory.json").is_file()
        status = read_json(metadata_path)["status"]
        observed.append(status)
        return {"status": "verified", "lifecycle": status}

    monkeypatch.setattr("adaptive_jump.balanced_runner._verify", fake_verify)
    _finalize_verified_run(run_dir, metadata_path, {"status": "running"})

    final = read_json(metadata_path)
    assert observed == ["verifying", "complete"]
    assert final["status"] == "complete"
    assert "verification_started_at_utc" in final
    assert "finished_at_utc" in final


def test_dated_audit_selects_earliest_signal_before_lambda_tie_break():
    events = pd.DataFrame.from_records(
        [
            {
                "market": "us",
                "rule": "balanced",
                "signal_date": pd.Timestamp("2020-02-01"),
                "lambda0": 5.0,
            },
            {
                "market": "us",
                "rule": "balanced",
                "signal_date": pd.Timestamp("2020-01-01"),
                "lambda0": 150.0,
            },
            {
                "market": "us",
                "rule": "balanced",
                "signal_date": pd.Timestamp("2020-01-01"),
                "lambda0": 35.0,
            },
        ]
    )

    dated = _dated_audit(events)

    assert len(dated) == 1
    assert dated.loc[0, "signal_date"] == pd.Timestamp("2020-01-01")
    assert dated.loc[0, "lambda0"] == 35.0


def test_final_verifier_failure_marks_artifact_invalid(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "study.lock.toml").write_text("frozen = true\n")
    metadata_path = run_dir / "run.json"

    def fail_on_complete(path):
        status = read_json(path / "run.json")["status"]
        if status == "complete":
            raise ValueError("synthetic final replay failure")
        return {"status": "verified", "lifecycle": status}

    monkeypatch.setattr("adaptive_jump.balanced_runner._verify", fail_on_complete)
    with pytest.raises(ValueError, match="synthetic final replay failure"):
        _finalize_verified_run(run_dir, metadata_path, {"status": "running"})

    failed = read_json(metadata_path)
    assert failed["status"] == "invalid_verification"
    assert "finished_at_utc" not in failed
    assert failed["verification_error"] == (
        "final complete-status verification failed (ValueError)"
    )


def test_actual_formula_rebuild_uses_current_fit_at_second_refit(spec):
    mini = replace(
        spec,
        markets=("us",),
        lambdas=(0.0, 5.0, 15.0),
        event_lambdas=(5.0, 15.0),
        fit_window=3,
    )
    dates = pd.date_range("2020-01-01", periods=6, name="date")
    features = pd.DataFrame(
        {
            "dd_10": [0.0, 0.5, 2.0, 1.0, 3.0, 4.0],
            "sortino_20": 0.0,
            "sortino_60": 0.0,
        },
        index=dates,
    )
    refits = pd.DataFrame.from_records(
        {
            "fit_date": fit_date,
            "lambda0": lambda0,
            "q_train": 1.0,
            "scaler_mean": [0.0, 0.0, 0.0],
            "scaler_scale": [1.0, 1.0, 1.0],
            "centers": [[0.0, 0.0, 0.0], [second_center, 0.0, 0.0]],
        }
        for fit_date, second_center in ((dates[2], 2.0), (dates[4], 4.0))
        for lambda0 in mini.lambdas
    )
    states = pd.DataFrame(math.nan, index=dates, columns=mini.lambdas)
    states.loc[dates[2] : dates[4], :] = 0.0
    c01 = states.copy()
    c10 = states.copy()
    for current in dates[2:5]:
        for lambda0 in mini.lambdas:
            fit = refits.loc[
                (refits["lambda0"] == lambda0) & (refits["fit_date"] <= current)
            ].iloc[-1]
            terminal = int(dates.get_loc(current))
            raw = features.iloc[terminal - 2 : terminal + 1].to_numpy(dtype=float)
            losses = loss_matrix(
                (raw - np.asarray(fit["scaler_mean"]))
                / np.asarray(fit["scaler_scale"]),
                np.asarray(fit["centers"]),
            )
            penalty = independent_balanced_terminal_penalty(
                losses[-2], lambda0, mini.decision_beta, 1.0
            )
            c01.loc[current, lambda0] = penalty[0, 1]
            c10.loc[current, lambda0] = penalty[1, 0]
    evidence = LockedStateEvidence(
        states={mini.decision_beta: states},
        loss0=pd.DataFrame(index=dates),
        loss1=pd.DataFrame(index=dates),
        q_train=pd.DataFrame(index=dates),
        c01={mini.decision_beta: c01},
        c10={mini.decision_beta: c10},
        refits=refits,
    )
    inputs = MarketInputs(
        market="us",
        features=features.loc[:, FEATURE_COLUMNS],
        model_dates=dates,
        candidates={},
        refits=refits,
    )

    result = _actual_formula_checks(inputs, evidence, mini)

    assert result["second_refit_date"] == dates[4]
    assert result["terminal_dates_checked"] == 3
    assert result["lambda_values_checked"] == 3
    assert result["directed_cells_checked"] == 18
    assert result["maximum_formula_abs_error"] == pytest.approx(0.0)
    assert result["maximum_second_refit_formula_abs_error"] == pytest.approx(0.0)
    assert result["bounds_exact"] is True
    assert result["minimum_stale_fit_distance"] > mini.numerical_tolerance
    assert result["maximum_stale_fit_distance"] >= result["minimum_stale_fit_distance"]
    assert result["stale_fit_lambdas_checked"] == 2
    assert result["stale_fit_lambdas_informative"] == 2
    assert result["stale_fit_lambdas_distinct"] == 2

    evidence.c01[mini.decision_beta].loc[dates[4], 0.0] = 1.0
    tampered = _actual_formula_checks(inputs, evidence, mini)
    assert tampered["maximum_formula_abs_error"] == pytest.approx(1.0)
    assert tampered["maximum_pair_sum_abs_error"] == pytest.approx(1.0)
    assert tampered["bounds_exact"] is False


def test_us_smoke_uses_full_generated_horizon_and_strict_stale_gate(spec, monkeypatch):
    config = load_config(ROOT / "research.toml")
    mini = replace(
        spec,
        markets=("us",),
        lambdas=(0.0, 5.0),
        event_lambdas=(5.0,),
        fit_window=3,
    )
    dates = pd.date_range("2020-01-01", periods=30, name="date")
    features = pd.DataFrame(0.0, index=dates, columns=FEATURE_COLUMNS)
    fixed = pd.DataFrame(math.nan, index=dates, columns=mini.lambdas)
    fixed.iloc[mini.fit_window - 1 :] = 0.0
    second_refit = dates[26]
    inputs = MarketInputs(
        market="us",
        features=features,
        model_dates=dates,
        candidates={beta: fixed.copy() for beta in mini.betas},
        refits=pd.DataFrame({"fit_date": [dates[2], second_refit]}),
    )

    def fake_generate(
        inputs_arg,
        fixed_arg,
        config_arg,
        spec_arg,
        *,
        terminal_limit=None,
        features=None,
    ):
        assert inputs_arg is inputs
        assert fixed_arg is fixed
        assert config_arg is config
        assert spec_arg is mini
        assert terminal_limit is not None
        final_terminal = mini.fit_window - 1 + terminal_limit
        states = fixed.copy()
        states.iloc[final_terminal:] = math.nan
        losses = pd.DataFrame(math.nan, index=dates, columns=mini.lambdas)
        losses.iloc[mini.fit_window - 1 : final_terminal] = 0.0
        if features is not None:
            losses.iloc[mini.fit_window + 20 - 1 : final_terminal] = 1.0
        return {
            rule: LockedStateEvidence(
                states={beta: states.copy() for beta in mini.betas},
                loss0=losses.copy(),
                loss1=losses.copy(),
                q_train=pd.DataFrame(index=dates),
                c01={beta: states.copy() for beta in mini.betas},
                c10={beta: states.copy() for beta in mini.betas},
                refits=inputs.refits.copy(),
            )
            for rule in mini.rules
        }

    actual = {
        "first_terminal_date": dates[2],
        "second_refit_date": second_refit,
        "terminal_dates_checked": 25,
        "lambda_values_checked": 2,
        "directed_cells_checked": 100,
        "maximum_formula_abs_error": 0.0,
        "maximum_second_refit_formula_abs_error": 0.0,
        "maximum_pair_sum_abs_error": 0.0,
        "bounds_exact": True,
        "minimum_stale_fit_distance": 1.0,
        "maximum_stale_fit_distance": 2.0,
        "stale_fit_lambdas_checked": 1,
        "stale_fit_lambdas_informative": 1,
        "stale_fit_lambdas_distinct": 1,
    }
    sources = SimpleNamespace(
        fixed_markets={"us": Path("fixed")},
        parent_markets={"us": Path("parent")},
    )
    monkeypatch.setattr(
        "adaptive_jump.balanced_smoke.verify_source_inputs",
        lambda *args: sources,
    )
    monkeypatch.setattr(
        "adaptive_jump.balanced_smoke.load_market_inputs",
        lambda *args: (inputs, fixed),
    )
    monkeypatch.setattr("adaptive_jump.balanced_smoke.generate_evidence", fake_generate)
    monkeypatch.setattr(
        "adaptive_jump.balanced_smoke._mutated_fixed_states",
        lambda *args, **kwargs: fixed,
    )
    monkeypatch.setattr(
        "adaptive_jump.balanced_smoke.balanced_penalty_checks",
        lambda *args: (1, 1, 0.0),
    )
    monkeypatch.setattr(
        "adaptive_jump.balanced_smoke._actual_formula_checks",
        lambda *args: actual,
    )
    monkeypatch.setattr(
        "adaptive_jump.balanced_smoke.mechanical_prerequisites",
        lambda *args: {"passed": True},
    )

    result = run_us_smoke(config, mini)

    assert result["generated_terminal_dates"] == 25
    assert result["generated_terminal_end_date"] == second_refit.date().isoformat()
    assert result["parent_lagged_state_cells_checked"] == 25 * 2 * 2
    assert result["beta_zero_state_cells_checked"] == 25 * 2 * 2
    assert result["short_long_prefix_state_cells_checked"] == 20 * 2 * 2 * 2
    assert result["future_mutation_prefix_state_cells_checked"] == 20 * 2 * 2 * 2
    assert result["actual_formula_lambda_values_checked"] == 2
    assert result["refit_convention_lambdas_checked"] == 1
    assert result["refit_convention_informative_lambdas"] == 1
    assert result["refit_convention_distinct_lambdas"] == 1

    actual["minimum_stale_fit_distance"] = 0.0
    with pytest.raises(BalancedStudyError, match="US balanced smoke failed"):
        run_us_smoke(config, mini)
    actual["minimum_stale_fit_distance"] = 1.0
    actual["stale_fit_lambdas_distinct"] = 0
    with pytest.raises(BalancedStudyError, match="US balanced smoke failed"):
        run_us_smoke(config, mini)
    actual["stale_fit_lambdas_distinct"] = 1
    actual["stale_fit_lambdas_informative"] = 0
    with pytest.raises(BalancedStudyError, match="US balanced smoke failed"):
        run_us_smoke(config, mini)
    actual["stale_fit_lambdas_informative"] = 1
