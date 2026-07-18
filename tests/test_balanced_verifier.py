from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from adaptive_jump.artifacts import read_json, write_inventory, write_json
from adaptive_jump.balanced_decision_replay import classify, dated_audit, summarize
from adaptive_jump.balanced_event_replay import (
    EVENT_COLUMNS,
    MATCHED_COLUMNS,
    matched_response,
)
from adaptive_jump.balanced_mechanics import independent_balanced_penalty
from adaptive_jump.balanced_model import BalancedStudyError
from adaptive_jump.balanced_replay import independent_lagged_penalty
from adaptive_jump.balanced_sources import SourcePaths
from adaptive_jump.balanced_verifier import (
    MarketReplay,
    _mechanical_checks,
    _smoke_coverage_exact,
    expected_files,
    verify_balanced_run,
)
from adaptive_jump.lagged_model import LockedStateEvidence


def _spec() -> SimpleNamespace:
    return SimpleNamespace(
        experiment_id="balanced-lagged-mechanism-001",
        sha256="1" * 64,
        fixed_inventory_sha256="2" * 64,
        parent_inventory_sha256="3" * 64,
        parent_spec_sha256="4" * 64,
        data_manifest_sha256="5" * 64,
        data_cutoff=pd.Timestamp("2023-12-31").date(),
        markets=("us", "de", "jp"),
        rules=("lagged", "balanced"),
        betas=(0.0, math.log(4.0)),
        decision_beta=math.log(4.0),
        lambdas=(0.0, 5.0),
        event_lambdas=(5.0,),
        numerical_tolerance=1e-12,
        horizon=2,
        matched_entry_search=2,
        matched_followup=2,
        matched_anchor_censor=4,
    )


def _event(
    market: str,
    rule: str,
    signal_date: pd.Timestamp,
    *,
    whipsaw: bool,
    confirmed: bool,
) -> dict[str, Any]:
    persistent = not whipsaw
    return {
        "market": market,
        "rule": rule,
        "beta": math.log(4.0),
        "beta_label": "log4",
        "lambda0": 5.0,
        "signal_date": signal_date,
        "evidence_date": signal_date - pd.Timedelta(days=1),
        "fit_date": pd.Timestamp("2020-01-01"),
        "source_state": 0,
        "destination_state": 1,
        "loss_source": 2.0,
        "loss_destination": 0.0,
        "q_train": 1.0,
        "normalized_gap": 2.0,
        "transition_penalty": 2.0,
        "expected_transition_penalty": 2.0,
        "reverse_penalty": 8.0 if rule == "balanced" else 5.0,
        "pair_sum_abs_error": 0.0 if rule == "balanced" else 3.0,
        "terminal_predecessor": 0,
        "terminal_state_margin": 1.0,
        "terminal_predecessor_margin": 1.0,
        "ablated_state": 0,
        "ablated_state_margin": 1.0,
        "discount_attributable": True,
        "horizon_candidate_dates": 2,
        "persistent_20": persistent,
        "whipsaw_20": whipsaw,
        "first_reversal_h": 1.0 if whipsaw else math.nan,
        "fixed_confirmation_h": 2.0 if confirmed else math.nan,
        "confirmed_early": confirmed,
        "unconfirmed_persistent_20": persistent and not confirmed,
    }


def _own_audit(admitted: int) -> dict[str, Any]:
    minimum = 1.0 if admitted else None
    return {
        "candidate_divergences": admitted,
        "horizon_censored": 0,
        "state_reconstructions": admitted,
        "terminal_tie_exclusions": 0,
        "terminal_transition_matches": admitted,
        "discounted_terminal_transitions": admitted,
        "ablation_attributable": admitted,
        "admitted_events": admitted,
        "max_penalty_abs_error": 0.0,
        "max_pair_sum_abs_error": 0.0,
        "minimum_terminal_state_margin": minimum,
        "minimum_terminal_predecessor_margin": minimum,
        "minimum_ablated_state_margin": minimum,
        "nonfinite_terminal_state_margin_count": 0,
        "nonfinite_terminal_predecessor_margin_count": 0,
        "nonfinite_ablated_state_margin_count": 0,
    }


def _market_replay(market: str, spec: SimpleNamespace) -> MarketReplay:
    dates = pd.date_range("2020-01-01", periods=4, name="date")
    base = pd.DataFrame(
        {0.0: [0.0, 0.0, 0.0, 0.0], 5.0: [0.0, 0.0, 1.0, 1.0]},
        index=dates,
    )
    changed = base.copy()
    changed[5.0] = [0.0, 1.0, 1.0, 1.0]
    states = {
        "lagged": {0.0: base.copy(), spec.decision_beta: base.copy()},
        "balanced": {0.0: base.copy(), spec.decision_beta: changed},
    }
    blank = pd.DataFrame(index=dates, columns=spec.lambdas, dtype=float)
    evidence = {
        rule: LockedStateEvidence(
            states=states[rule],
            loss0=blank.copy(),
            loss1=blank.copy(),
            q_train=blank.copy(),
            c01={beta: blank.copy() for beta in spec.betas},
            c10={beta: blank.copy() for beta in spec.betas},
            refits=pd.DataFrame(),
        )
        for rule in spec.rules
    }
    behavior = pd.DataFrame(
        [
            {
                "market": market,
                "rule": "lagged",
                "beta": spec.decision_beta,
                "beta_label": "log4",
                "lambda0": 5.0,
                "start": dates[0],
                "end": dates[-1],
                "observations": 4,
                "switch_count": 2,
                "state_0_count": 2,
                "state_1_count": 2,
                "state_differences_vs_fixed": 0,
                "state_differences_vs_lagged": 0,
            },
            {
                "market": market,
                "rule": "balanced",
                "beta": spec.decision_beta,
                "beta_label": "log4",
                "lambda0": 5.0,
                "start": dates[0],
                "end": dates[-1],
                "observations": 4,
                "switch_count": 1,
                "state_0_count": 1,
                "state_1_count": 3,
                "state_differences_vs_fixed": 1,
                "state_differences_vs_lagged": 1,
            },
        ]
    )
    events = pd.DataFrame.from_records(
        [
            _event(market, "lagged", dates[1], whipsaw=True, confirmed=False),
            _event(market, "lagged", dates[2], whipsaw=False, confirmed=True),
            _event(market, "balanced", dates[2], whipsaw=False, confirmed=True),
        ],
        columns=EVENT_COLUMNS,
    )
    anchors = pd.DataFrame.from_records(
        [
            {
                "market": market,
                "beta": spec.decision_beta,
                "beta_label": "log4",
                "lambda0": 5.0,
                "signal_date": dates[1],
                "source_state": 0,
                "destination_state": 1,
                "lagged_whipsaw_20": True,
                "lagged_confirmed_early": False,
                "lagged_unconfirmed_persistent_20": False,
                "fixed_confirmation_h": math.nan,
                "first_destination_h": math.nan,
                "matched_follow_end_h": math.nan,
                "matched_fixed_confirmation_h": math.nan,
                "matched_category": "suppressed_no_entry",
                "matched_whipsaw_20": False,
                "matched_persistent_20": False,
                "matched_unconfirmed_persistent_20": False,
                "matched_retained_confirmed_early": False,
            },
            {
                "market": market,
                "beta": spec.decision_beta,
                "beta_label": "log4",
                "lambda0": 5.0,
                "signal_date": dates[2],
                "source_state": 0,
                "destination_state": 1,
                "lagged_whipsaw_20": False,
                "lagged_confirmed_early": True,
                "lagged_unconfirmed_persistent_20": False,
                "fixed_confirmation_h": 2.0,
                "first_destination_h": 20.0,
                "matched_follow_end_h": 40.0,
                "matched_fixed_confirmation_h": 2.0,
                "matched_category": "enters_persistent_confirmed",
                "matched_whipsaw_20": False,
                "matched_persistent_20": True,
                "matched_unconfirmed_persistent_20": False,
                "matched_retained_confirmed_early": False,
            },
        ],
        columns=MATCHED_COLUMNS,
    )
    penalties = pd.DataFrame(
        [
            {
                "market": market,
                "rule": rule,
                "beta": spec.decision_beta,
                "beta_label": "log4",
                "lambda0": 5.0,
                "observations": 4,
                "minimum_cost_ratio": 0.4,
                "maximum_cost_ratio": 1.6,
                "median_pair_mean_ratio": 1.0 if rule == "balanced" else 0.7,
                "maximum_pair_sum_abs_error": 0.0 if rule == "balanced" else 3.0,
                "discount_cells": 4,
                "surcharge_cells": 4 if rule == "balanced" else 0,
            }
            for rule in spec.rules
        ]
    )
    terminal_rows = len(dates)
    checks = {
        "parent_lagged_exact": True,
        "beta_zero_exact": True,
        "candidate_coverage_exact": True,
        "pair_balance_exact": True,
        "terminal_rows": terminal_rows,
        "parent_lagged_state_cells_checked": terminal_rows * 2 * 2,
        "beta_zero_state_cells_checked": terminal_rows * 2 * 2,
        "all_candidate_state_cells_checked": terminal_rows * 2 * 2 * 2,
        "balanced_discount_cells": 4,
        "balanced_surcharge_cells": 4,
        "maximum_pair_sum_abs_error": 0.0,
        "return_columns_accessed": False,
    }
    return MarketReplay(
        market=market,
        evidence=evidence,
        behavior=behavior,
        events=events,
        anchors=anchors,
        penalties=penalties,
        audit={
            "own_events": {
                "lagged": _own_audit(2),
                "balanced": _own_audit(1),
            },
            "matched_anchors": {
                "original_lagged_admitted_events": 2,
                "matched_anchor_censored": 0,
                "eligible_matched_anchors": 2,
            },
        },
        checks=checks,
    )


@pytest.fixture
def synthetic_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    spec = _spec()
    config = SimpleNamespace(sha256="6" * 64)
    implementation = {"implementation_sha256": "7" * 64, "git_head": "8" * 40}
    source_lock = {"schema_version": 1, "performance_files_accessed": False}
    sources = SourcePaths(
        fixed_dir=tmp_path,
        parent_dir=tmp_path,
        fixed_markets={market: tmp_path for market in spec.markets},
        parent_markets={market: tmp_path for market in spec.markets},
        source_lock=source_lock,
    )
    replays = {market: _market_replay(market, spec) for market in spec.markets}
    prerequisites = {"passed": True, "checks": {"formula": True}}
    smoke = {
        "status": "passed",
        "market": "us",
        "terminal_dates": 20,
        "generated_terminal_dates": 40,
        "generated_terminal_end_date": "2020-02-09",
        "refit_probe_date": "2020-01-02",
        "mechanical_prerequisites": prerequisites,
        **{
            key: expected
            for key, expected in {
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
            }.items()
        },
    }
    smoke.update(
        {
            "parent_lagged_state_cells_checked": 160,
            "beta_zero_state_cells_checked": 160,
            "short_long_prefix_state_cells_checked": 160,
            "future_mutation_prefix_state_cells_checked": 160,
            "future_mutation_loss_cells_changed": 8,
            "future_mutation_max_abs_loss_change": 1.0,
            "balanced_discount_cells": 8,
            "balanced_surcharge_cells": 8,
            "actual_formula_terminal_dates_checked": 40,
            "actual_formula_lambda_values_checked": 2,
            "actual_formula_directed_cells_checked": 160,
            "actual_formula_first_terminal_date": "2020-01-01",
            "actual_formula_max_abs_error": 0.0,
            "actual_second_refit_formula_max_abs_error": 0.0,
            "maximum_pair_sum_abs_error": 0.0,
            "refit_convention_min_stale_distance": 1.0,
            "refit_convention_max_stale_distance": 2.0,
            "refit_convention_lambdas_checked": 1,
            "refit_convention_distinct_lambdas": 1,
            "refit_convention_max_abs_error": 0.0,
        }
    )
    monkeypatch.setattr(
        "adaptive_jump.balanced_verifier._canonical_context",
        lambda: (config, spec),
    )
    monkeypatch.setattr("adaptive_jump.balanced_verifier.load_config", lambda _: config)
    monkeypatch.setattr(
        "adaptive_jump.balanced_verifier.load_balanced_spec",
        lambda _path, _config: spec,
    )
    monkeypatch.setattr(
        "adaptive_jump.balanced_verifier.verify_source_inputs",
        lambda _root, _config, _spec: sources,
    )
    monkeypatch.setattr(
        "adaptive_jump.balanced_verifier.implementation_lock",
        lambda _root, _spec: implementation,
    )
    monkeypatch.setattr(
        "adaptive_jump.balanced_verifier.run_independent_smoke",
        lambda _config, _spec, _sources: smoke,
    )
    monkeypatch.setattr(
        "adaptive_jump.balanced_verifier.mechanical_prerequisites",
        lambda _spec: prerequisites,
    )
    monkeypatch.setattr(
        "adaptive_jump.balanced_verifier._replay_market",
        lambda market, _sources, _config, _spec: replays[market],
    )

    def build(case: str) -> Path:
        run_id = (
            f"balanced-lagged-{spec.sha256[:12]}-"
            f"{spec.parent_inventory_sha256[:12]}-"
            f"{implementation['implementation_sha256'][:12]}"
        )
        run_dir = tmp_path / case / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "study.lock.toml").write_text("locked")
        (run_dir / "config.lock.toml").write_text("locked")
        write_json(run_dir / "source-lock.json", source_lock)
        write_json(run_dir / "implementation-lock.json", implementation)
        write_json(run_dir / "smoke.json", smoke)
        for market, replay in replays.items():
            target = run_dir / market
            target.mkdir()
            for rule in spec.rules:
                for beta, label in ((0.0, "0"), (spec.decision_beta, "log4")):
                    replay.evidence[rule].states[beta].to_csv(
                        target / f"candidate-states-{rule}-beta-{label}.csv"
                    )
            replay.behavior.to_csv(target / "path-behavior.csv", index=False)
            replay.events.to_csv(target / "discount-events.csv", index=False)
            replay.anchors.to_csv(target / "matched-lagged-anchors.csv", index=False)
            replay.penalties.to_csv(target / "penalty-summary.csv", index=False)
            write_json(target / "audit.json", replay.audit)
        behavior = pd.concat(
            [replays[market].behavior for market in spec.markets], ignore_index=True
        )
        events = pd.concat(
            [replays[market].events for market in spec.markets], ignore_index=True
        )
        anchors = pd.concat(
            [replays[market].anchors for market in spec.markets], ignore_index=True
        )
        penalties = pd.concat(
            [replays[market].penalties for market in spec.markets], ignore_index=True
        )
        summary = summarize(events, behavior, spec)
        dated = dated_audit(events)
        mechanics = _mechanical_checks(
            prerequisites,
            smoke,
            {market: replays[market].checks for market in spec.markets},
            spec,
        )
        conclusion = classify(
            summary, anchors, spec, mechanical_passed=mechanics["passed"]
        )
        for filename, frame in (
            ("path-behavior.csv", behavior),
            ("discount-events.csv", events),
            ("matched-lagged-anchors.csv", anchors),
            ("penalty-summary.csv", penalties),
            ("mechanism-summary.csv", summary),
            ("dated-audit.csv", dated),
        ):
            frame.to_csv(run_dir / filename, index=False)
        write_json(run_dir / "mechanical-checks.json", mechanics)
        write_json(run_dir / "conclusion.json", conclusion)
        write_json(
            run_dir / "run.json",
            {
                "schema_version": 1,
                "study_kind": "balanced_lagged_mechanism",
                "experiment_id": spec.experiment_id,
                "run_id": run_id,
                "status": "complete",
                "claim_class": "EXPLORATORY",
                "performance_claim_allowed": False,
                "paper_replication_claim_allowed": False,
                "new_model_claim_allowed": False,
                "performance_files_accessed": False,
                "return_columns_accessed": False,
                "post_2023_accessed": False,
                "provider_accessed": False,
                "monthly_selection_performed": False,
                "created_at_utc": "2026-01-01T00:00:00+00:00",
                "verification_started_at_utc": "2026-01-01T00:01:00+00:00",
                "finished_at_utc": "2026-01-01T00:02:00+00:00",
                "spec_sha256": spec.sha256,
                "config_sha256": config.sha256,
                "fixed_inventory_sha256": spec.fixed_inventory_sha256,
                "parent_inventory_sha256": spec.parent_inventory_sha256,
                "parent_spec_sha256": spec.parent_spec_sha256,
                "data_manifest_sha256": spec.data_manifest_sha256,
                "implementation_sha256": implementation["implementation_sha256"],
                "git_head": implementation["git_head"],
                "result": conclusion["result"],
                "decision_beta_label": "log4",
                "mechanical_prerequisites_passed": True,
                "path_rows": len(behavior),
                "event_rows": len(events),
                "matched_anchor_rows": len(anchors),
                "penalty_rows": len(penalties),
                "summary_rows": len(summary),
                "dated_audit_rows": len(dated),
            },
        )
        write_inventory(run_dir)
        assert {
            str(path.relative_to(run_dir))
            for path in run_dir.rglob("*")
            if path.is_file()
        } == expected_files(spec)
        return run_dir

    return build


def test_independent_penalty_formula_and_pair_balance() -> None:
    loss = np.array([[0.0, 2.0], [1.0, 0.0], [0.5, 0.2]])
    beta = math.log(4.0)
    balanced = independent_balanced_penalty(loss, 5.0, beta, 1.0)
    lagged = independent_lagged_penalty(loss, 5.0, beta, 1.0)
    assert np.array_equal(
        independent_balanced_penalty(loss, 5.0, 0.0, 1.0),
        np.array([[[0.0, 5.0], [5.0, 0.0]]] * 3),
    )
    assert np.allclose(balanced[:, 0, 1] + balanced[:, 1, 0], 10.0)
    assert lagged[1, 1, 0] < 5.0
    assert lagged[1, 0, 1] == 5.0


def test_independent_matched_response_detects_entry_then_reversal() -> None:
    spec = SimpleNamespace(
        decision_beta=math.log(4.0),
        event_lambdas=(5.0,),
        matched_entry_search=2,
        matched_followup=2,
        matched_anchor_censor=4,
    )
    dates = pd.date_range("2020-01-01", periods=6, name="date")
    event = pd.DataFrame(
        [{**_event("us", "lagged", dates[1], whipsaw=True, confirmed=False)}],
        columns=EVENT_COLUMNS,
    )
    states = pd.DataFrame({5.0: [0, 1, 0, 0, 0, 0]}, index=dates)
    fixed = pd.DataFrame({5.0: [0, 0, 0, 0, 0, 0]}, index=dates)
    refits = pd.DataFrame({"fit_date": [dates[0]], "lambda0": [5.0]})
    result, audit = matched_response(event, states, fixed, refits, spec)
    assert result.loc[0, "matched_category"] == "enters_then_whipsaw"
    assert bool(result.loc[0, "matched_whipsaw_20"])
    assert result.loc[0, "matched_follow_end_h"] == 2
    assert audit == {
        "original_lagged_admitted_events": 1,
        "matched_anchor_censored": 0,
        "eligible_matched_anchors": 1,
    }


def test_late_h20_response_uses_full_followup_and_late_confirmation() -> None:
    spec = SimpleNamespace(
        decision_beta=math.log(4.0),
        event_lambdas=(5.0,),
        matched_entry_search=20,
        matched_followup=20,
        matched_anchor_censor=40,
    )
    dates = pd.date_range("2020-01-01", periods=45, name="date")
    position = 2
    event = pd.DataFrame(
        [{**_event("us", "lagged", dates[position], whipsaw=False, confirmed=False)}],
        columns=EVENT_COLUMNS,
    )
    fixed = pd.DataFrame({5.0: np.zeros(len(dates), dtype=int)}, index=dates)
    fixed.iloc[position + 30 :, 0] = 1
    persistent = pd.DataFrame({5.0: np.zeros(len(dates), dtype=int)}, index=dates)
    persistent.iloc[position + 20 :, 0] = 1
    refits = pd.DataFrame({"fit_date": [dates[0]], "lambda0": [5.0]})

    retained, _ = matched_response(event, persistent, fixed, refits, spec)
    row = retained.iloc[0]
    assert row["first_destination_h"] == 20
    assert row["matched_follow_end_h"] == 40
    assert row["matched_fixed_confirmation_h"] == 30
    assert row["matched_category"] == "enters_persistent_confirmed"

    reversed_path = persistent.copy()
    reversed_path.iloc[position + 21, 0] = 0
    reversed_result, _ = matched_response(event, reversed_path, fixed, refits, spec)
    reversed_row = reversed_result.iloc[0]
    assert reversed_row["matched_category"] == "enters_then_whipsaw"
    assert bool(reversed_row["matched_whipsaw_20"])


def test_matched_censor_precedes_balanced_response_access() -> None:
    spec = SimpleNamespace(
        decision_beta=math.log(4.0),
        event_lambdas=(5.0,),
        matched_entry_search=20,
        matched_followup=20,
        matched_anchor_censor=40,
    )
    short_dates = pd.date_range("2020-01-01", periods=42, name="date")
    event = pd.DataFrame(
        [{**_event("us", "lagged", short_dates[2], whipsaw=False, confirmed=False)}],
        columns=EVENT_COLUMNS,
    )
    fixed = pd.DataFrame(
        {5.0: np.zeros(len(short_dates), dtype=int)}, index=short_dates
    )
    refits = pd.DataFrame({"fit_date": [short_dates[0]], "lambda0": [5.0]})
    result, audit = matched_response(event, pd.DataFrame(), fixed, refits, spec)
    assert result.empty
    assert audit["matched_anchor_censored"] == 1
    assert audit["eligible_matched_anchors"] == 0

    dates = pd.date_range("2020-01-01", periods=45, name="date")
    event.loc[0, "signal_date"] = dates[2]
    fixed = pd.DataFrame({5.0: np.zeros(len(dates), dtype=int)}, index=dates)
    refits = pd.DataFrame({"fit_date": [dates[0], dates[30]], "lambda0": [5.0, 5.0]})
    result, audit = matched_response(event, pd.DataFrame(), fixed, refits, spec)
    assert result.empty
    assert audit["matched_anchor_censored"] == 1
    assert audit["eligible_matched_anchors"] == 0


def test_smoke_coverage_requires_full_grid_and_every_stale_lambda() -> None:
    spec = _spec()
    smoke = {
        "terminal_dates": 20,
        "generated_terminal_dates": 40,
        "parent_lagged_state_cells_checked": 160,
        "beta_zero_state_cells_checked": 160,
        "short_long_prefix_state_cells_checked": 160,
        "future_mutation_prefix_state_cells_checked": 160,
        "actual_formula_terminal_dates_checked": 40,
        "actual_formula_lambda_values_checked": 2,
        "actual_formula_directed_cells_checked": 160,
        "refit_convention_lambdas_checked": 1,
        "refit_convention_distinct_lambdas": 1,
        "refit_convention_min_stale_distance": 1.0,
        "refit_convention_max_stale_distance": 2.0,
    }
    assert _smoke_coverage_exact(smoke, spec)

    smoke["parent_lagged_state_cells_checked"] -= 1
    assert not _smoke_coverage_exact(smoke, spec)
    smoke["parent_lagged_state_cells_checked"] += 1
    smoke["actual_formula_lambda_values_checked"] -= 1
    assert not _smoke_coverage_exact(smoke, spec)
    smoke["actual_formula_lambda_values_checked"] += 1
    smoke["refit_convention_distinct_lambdas"] = 0
    assert not _smoke_coverage_exact(smoke, spec)


def test_valid_synthetic_artifact_verifies(synthetic_run) -> None:
    result = verify_balanced_run(synthetic_run("valid"))
    assert result["status"] == "verified"
    assert result["lifecycle"] == "complete"
    assert result["markets_reconstructed"] == 3


def test_valid_verifying_lifecycle_verifies(synthetic_run) -> None:
    run_dir = synthetic_run("verifying")
    path = run_dir / "run.json"
    document = read_json(path)
    document["status"] = "verifying"
    document.pop("finished_at_utc")
    write_json(path, document)
    result = verify_balanced_run(run_dir)
    assert result["status"] == "verified"
    assert result["lifecycle"] == "verifying"


@pytest.mark.parametrize(
    "tamper",
    [
        "extra",
        "smoke_coverage",
        "candidate",
        "event",
        "matched",
        "late_persistence",
        "late_confirmation",
        "censor",
        "summary",
        "conclusion",
        "metadata",
        "metadata_type",
    ],
)
def test_verifier_rejects_artifact_tampering(synthetic_run, tamper: str) -> None:
    run_dir = synthetic_run(tamper)
    if tamper == "extra":
        (run_dir / "unexpected.txt").write_text("not allowlisted")
    elif tamper == "smoke_coverage":
        path = run_dir / "smoke.json"
        document = read_json(path)
        document["actual_formula_lambda_values_checked"] -= 1
        write_json(path, document)
        write_inventory(run_dir)
    elif tamper == "candidate":
        path = run_dir / "us/candidate-states-balanced-beta-log4.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "5.0"] = 1.0 - frame.loc[0, "5.0"]
        frame.to_csv(path, index=False)
        write_inventory(run_dir)
    elif tamper == "event":
        path = run_dir / "us/discount-events.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "transition_penalty"] += 0.25
        frame.to_csv(path, index=False)
        write_inventory(run_dir)
    elif tamper == "matched":
        path = run_dir / "us/matched-lagged-anchors.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "matched_category"] = "enters_then_whipsaw"
        frame.to_csv(path, index=False)
        write_inventory(run_dir)
    elif tamper == "late_persistence":
        path = run_dir / "us/matched-lagged-anchors.csv"
        frame = pd.read_csv(path)
        row = frame.index[frame["first_destination_h"] == 20.0][0]
        frame.loc[row, "matched_category"] = "enters_then_whipsaw"
        frame.loc[row, "matched_whipsaw_20"] = True
        frame.loc[row, "matched_persistent_20"] = False
        frame.to_csv(path, index=False)
        write_inventory(run_dir)
    elif tamper == "late_confirmation":
        path = run_dir / "us/matched-lagged-anchors.csv"
        frame = pd.read_csv(path)
        row = frame.index[frame["first_destination_h"] == 20.0][0]
        frame.loc[row, "matched_fixed_confirmation_h"] += 1.0
        frame.to_csv(path, index=False)
        write_inventory(run_dir)
    elif tamper == "censor":
        path = run_dir / "us/audit.json"
        document = read_json(path)
        document["matched_anchors"]["matched_anchor_censored"] = 1
        document["matched_anchors"]["eligible_matched_anchors"] = 1
        write_json(path, document)
        write_inventory(run_dir)
    elif tamper == "summary":
        path = run_dir / "mechanism-summary.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "switch_count"] += 1
        frame.to_csv(path, index=False)
        write_inventory(run_dir)
    elif tamper == "conclusion":
        path = run_dir / "conclusion.json"
        document = read_json(path)
        document["result"] = (
            "supported" if document["result"] == "not_supported" else "not_supported"
        )
        write_json(path, document)
        write_inventory(run_dir)
    elif tamper == "metadata_type":
        path = run_dir / "run.json"
        document = read_json(path)
        document["performance_files_accessed"] = 0
        write_json(path, document)
    else:
        path = run_dir / "run.json"
        document = read_json(path)
        document["event_rows"] += 1
        write_json(path, document)
    with pytest.raises(BalancedStudyError):
        verify_balanced_run(run_dir)
