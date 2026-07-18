from __future__ import annotations

import inspect
import math
from types import SimpleNamespace

import pandas as pd
import pytest

from adaptive_jump.artifacts import write_json
from adaptive_jump.lagged_analysis import MechanismAnalysis
from adaptive_jump.lagged_model import LockedStateEvidence
from adaptive_jump.lagged_study import LaggedStudyError, summarize_mechanism
from adaptive_jump.lagged_verifier import (
    _assert_json_exact,
    _combined_mechanical_prerequisite,
    _dated_audit,
    _mechanical_checks,
    _verify_market_artifacts,
    _verify_metadata,
    _verify_root_artifacts,
    verify_lagged_run,
)

BETAS = (0.0, math.log(2.0), math.log(4.0))
LAMBDAS = (0.0, 5.0)


def test_exact_lock_comparison_rejects_resealed_hash_change() -> None:
    expected = {"implementation_sha256": "a" * 64, "files": {"model.py": "b" * 64}}
    _assert_json_exact(expected.copy(), expected, label="implementation lock")

    changed = {"implementation_sha256": "c" * 64, "files": expected["files"]}
    with pytest.raises(LaggedStudyError):
        _assert_json_exact(changed, expected, label="implementation lock")


def test_verifier_recomputes_and_exactly_compares_locked_us_smoke() -> None:
    source = inspect.getsource(verify_lagged_run)
    assert "smoke = run_locked_smoke(" in source
    assert '_assert_json_exact(stored_smoke, smoke, label="US smoke")' in source


def _market_fixture(tmp_path):
    market_dir = tmp_path / "us"
    market_dir.mkdir()
    dates = pd.bdate_range("2020-01-01", periods=3, name="date")
    states = {
        beta: pd.DataFrame(
            [[0.0, 0.0], [0.0, float(beta > 0)], [1.0, 1.0]],
            index=dates,
            columns=LAMBDAS,
        )
        for beta in BETAS
    }
    refits = pd.DataFrame(
        [
            {
                "market": "us",
                "fit_date": dates[0],
                "training_start": dates[0],
                "training_end": dates[0],
                "lambda0": lambda0,
                "q_train": 1.5 + lambda0,
                "scaler_mean": "[0, 0, 0]",
                "scaler_scale": "[1, 1, 1]",
                "centers": "[[0, 0, 0], [1, 1, 1]]",
            }
            for lambda0 in LAMBDAS
        ]
    )
    empty = pd.DataFrame(index=dates, columns=LAMBDAS, dtype=float)
    evidence = LockedStateEvidence(
        states=states,
        loss0=empty.copy(),
        loss1=empty.copy(),
        q_train=empty.copy(),
        c01={beta: empty.copy() for beta in BETAS},
        c10={beta: empty.copy() for beta in BETAS},
        refits=refits,
    )
    behavior = pd.DataFrame(
        [
            {
                "market": "us",
                "rule": "lagged",
                "beta": math.log(2.0),
                "beta_label": "log2",
                "lambda0": 5.0,
                "start": dates[0],
                "end": dates[-1],
                "observations": 3,
                "switch_count": 1,
                "state_differences_vs_fixed": 1,
            }
        ]
    )
    events = pd.DataFrame(
        [
            {
                "market": "us",
                "rule": "lagged",
                "beta_label": "log2",
                "signal_date": dates[1],
                "evidence_date": dates[0],
                "fit_date": dates[0],
                "value": 1.25,
            }
        ]
    )
    analysis = MechanismAnalysis(
        market="us",
        behavior=behavior,
        events=events,
        audit={"lagged": {"admitted_events": 1}},
    )
    spec = SimpleNamespace(betas=BETAS, lambdas=LAMBDAS)

    for beta, frame in states.items():
        label = "0" if beta == 0 else "log2" if beta == math.log(2.0) else "log4"
        frame.to_csv(market_dir / f"candidate-states-beta-{label}.csv")
    refits.to_csv(market_dir / "refits-and-scales.csv", index=False)
    behavior.to_csv(market_dir / "path-behavior.csv", index=False)
    events.to_csv(market_dir / "discount-events.csv", index=False)
    write_json(market_dir / "audit.json", analysis.audit)
    return market_dir, evidence, analysis, spec


def _mutate_market(market_dir, target):
    if target == "candidate":
        path = market_dir / "candidate-states-beta-log2.csv"
        frame = pd.read_csv(path)
        frame.loc[1, "5.0"] = 0.0
        frame.to_csv(path, index=False)
    elif target == "refit":
        path = market_dir / "refits-and-scales.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "q_train"] += 1.0
        frame.to_csv(path, index=False)
    elif target == "behavior":
        path = market_dir / "path-behavior.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "switch_count"] += 1
        frame.to_csv(path, index=False)
    elif target == "event":
        path = market_dir / "discount-events.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "value"] += 1.0
        frame.to_csv(path, index=False)
    else:
        write_json(market_dir / "audit.json", {"lagged": {"admitted_events": 2}})


@pytest.mark.parametrize(
    "target",
    ["candidate", "refit", "behavior", "event", "audit"],
)
def test_market_verifier_rejects_resealed_scientific_tampering(
    tmp_path, target
) -> None:
    market_dir, evidence, analysis, spec = _market_fixture(tmp_path)
    _verify_market_artifacts(market_dir, evidence, analysis, spec)

    _mutate_market(market_dir, target)

    with pytest.raises(LaggedStudyError):
        _verify_market_artifacts(market_dir, evidence, analysis, spec)


def _root_fixture(tmp_path):
    beta = math.log(2.0)
    spec = SimpleNamespace(
        markets=("us",),
        event_betas=(beta,),
        event_lambdas=(5.0,),
        rules=("arrival", "lagged"),
    )
    behavior = pd.DataFrame(
        [
            {
                "market": "us",
                "rule": rule,
                "beta_label": "log2",
                "lambda0": 5.0,
                "switch_count": 3 if rule == "arrival" else 2,
                "state_differences_vs_fixed": int(rule == "lagged"),
            }
            for rule in spec.rules
        ]
    )
    events = pd.DataFrame(
        [
            {
                "market": "us",
                "rule": rule,
                "beta_label": "log2",
                "signal_date": pd.Timestamp("2020-01-03"),
                "whipsaw_20": rule == "arrival",
                "persistent_20": rule == "lagged",
                "confirmed_early": rule == "lagged",
            }
            for rule in spec.rules
        ]
    )
    behavior.to_csv(tmp_path / "path-behavior.csv", index=False)
    events.to_csv(tmp_path / "discount-events.csv", index=False)
    summarize_mechanism(events, behavior, spec).to_csv(
        tmp_path / "mechanism-summary.csv", index=False
    )
    _dated_audit(events).to_csv(tmp_path / "dated-audit.csv", index=False)
    return behavior, events, spec


def _mutate_root(run_dir, target):
    names = {
        "behavior": ("path-behavior.csv", "switch_count"),
        "event": ("discount-events.csv", "whipsaw_20"),
        "summary": ("mechanism-summary.csv", "event_count"),
        "dated": ("dated-audit.csv", "signal_date"),
    }
    filename, column = names[target]
    path = run_dir / filename
    frame = pd.read_csv(path)
    if target == "event":
        frame.loc[0, column] = not bool(frame.loc[0, column])
    elif target == "dated":
        frame.loc[0, column] = "2020-01-06"
    else:
        frame.loc[0, column] += 1
    frame.to_csv(path, index=False)


@pytest.mark.parametrize("target", ["behavior", "event", "summary", "dated"])
def test_root_verifier_rejects_resealed_aggregate_tampering(tmp_path, target) -> None:
    behavior, events, spec = _root_fixture(tmp_path)
    _verify_root_artifacts(tmp_path, behavior, events, spec)

    _mutate_root(tmp_path, target)

    with pytest.raises(LaggedStudyError):
        _verify_root_artifacts(tmp_path, behavior, events, spec)


def _metadata_fixture(tmp_path):
    spec = SimpleNamespace(
        sha256="a" * 64,
        arrival_inventory_sha256="b" * 64,
        experiment_id="lagged-evidence-mechanism-001",
        fixed_inventory_sha256="c" * 64,
        data_manifest_sha256="d" * 64,
    )
    implementation = {
        "implementation_sha256": "e" * 64,
        "git_head": "f" * 40,
    }
    conclusion = {"result": "supported", "selected_beta_label": "log2"}
    run_id = "lagged-evidence-aaaaaaaaaaaa-bbbbbbbbbbbb-eeeeeeeeeeee"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    metadata = {
        "schema_version": 1,
        "study_kind": "lagged_evidence_mechanism",
        "experiment_id": spec.experiment_id,
        "run_id": run_id,
        "status": "complete",
        "claim_class": "EXPLORATORY",
        "performance_files_accessed": False,
        "return_columns_accessed": False,
        "post_2023_accessed": False,
        "mechanical_prerequisites_passed": True,
        "spec_sha256": spec.sha256,
        "config_sha256": "1" * 64,
        "fixed_inventory_sha256": spec.fixed_inventory_sha256,
        "arrival_inventory_sha256": spec.arrival_inventory_sha256,
        "data_manifest_sha256": spec.data_manifest_sha256,
        "implementation_sha256": implementation["implementation_sha256"],
        "git_head": implementation["git_head"],
        "result": conclusion["result"],
        "selected_beta_label": conclusion["selected_beta_label"],
        "events": 7,
    }
    return metadata, run_dir, spec, implementation, conclusion


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("selected_beta_label", "log4"),
        ("result", "not_supported"),
        ("events", 8),
        ("run_id", "forged"),
        ("return_columns_accessed", True),
        ("return_columns_accessed", 0),
        ("mechanical_prerequisites_passed", False),
    ],
)
def test_metadata_verifier_rejects_uninventoried_tampering(
    tmp_path, key, value
) -> None:
    metadata, run_dir, spec, implementation, conclusion = _metadata_fixture(tmp_path)
    _verify_metadata(
        metadata,
        run_dir=run_dir,
        spec=spec,
        config_sha256="1" * 64,
        implementation=implementation,
        conclusion=conclusion,
        event_count=7,
        mechanical_prerequisites_passed=True,
    )

    metadata[key] = value

    with pytest.raises(LaggedStudyError):
        _verify_metadata(
            metadata,
            run_dir=run_dir,
            spec=spec,
            config_sha256="1" * 64,
            implementation=implementation,
            conclusion=conclusion,
            event_count=7,
            mechanical_prerequisites_passed=True,
        )


def test_combined_prerequisite_recomputes_mechanics_and_real_smoke() -> None:
    mechanics = {
        "passed": True,
        "checks": {"formula": True, "toy_paths": True},
        "by_rule": {},
        "toy_paths": {},
    }
    smoke = {
        "status": "passed",
        "mechanical_prerequisites": mechanics,
        "beta_zero_exact": True,
        "prefix_invariant": True,
        "future_mutation_effect_present": True,
        "sealed_arrival_exact": True,
        "refit_convention_numeric": True,
        "lagged_discounts_present": True,
        "refit_convention": "current-fit parameters applied to previous-row loss",
        "performance_files_accessed": False,
        "return_columns_accessed": False,
        "post_2023_accessed": False,
    }

    assert _combined_mechanical_prerequisite(mechanics, smoke)

    failed_smoke = dict(smoke, prefix_invariant=False)
    assert not _combined_mechanical_prerequisite(mechanics, failed_smoke)

    inconsistent = dict(mechanics, passed=False)
    with pytest.raises(LaggedStudyError):
        _combined_mechanical_prerequisite(inconsistent, smoke)

    replay = {
        "us": {
            "sealed_arrival_exact": True,
            "beta_zero_exact": True,
            "sealed_arrival_state_cells_checked": 18,
            "beta_zero_state_cells_checked": 12,
            "return_columns_accessed": False,
        }
    }
    expected = _mechanical_checks(mechanics, smoke, replay)
    assert expected["passed"]
    assert expected["market_replays"] == replay

    tampered = {"us": dict(replay["us"], beta_zero_exact=False)}
    expected = _mechanical_checks(mechanics, smoke, tampered)
    assert not expected["passed"]
