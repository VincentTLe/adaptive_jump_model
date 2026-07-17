from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adaptive_jump.separation_study import (
    SeparationStudyError,
    arrival_ablation_state,
    classify_decision,
    fit_logistic,
    load_separation_spec,
    prediction_scores,
    reliability_from_geometry,
)
from adaptive_jump.tv_jump import dp_tv, lam_to_penalty_seq

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "research/adaptive-separation-001.toml"
REGISTRY = ROOT / "research/experiment_registry.jsonl"


def test_reliability_formula_has_expected_toy_value_and_bounds() -> None:
    centers = np.array([[0.0, 0.0], [4.0, 0.0]])
    features = np.array([[-1.0, 0.0], [1.0, 0.0], [3.0, 0.0], [5.0, 0.0]])

    result = reliability_from_geometry(features, centers)

    assert result.valid
    assert result.center_distance == pytest.approx(4.0)
    assert (result.preferred_count_0, result.preferred_count_1) == (2, 2)
    assert result.median_radius_0 == pytest.approx(1.0)
    assert result.median_radius_1 == pytest.approx(1.0)
    assert result.reliability_train == pytest.approx(2.0 / 3.0)
    assert 0.0 <= result.reliability_train <= 1.0

    compact = reliability_from_geometry(centers.copy(), centers)
    assert compact.valid
    assert compact.reliability_train == pytest.approx(1.0)


def test_reliability_is_label_symmetric() -> None:
    centers = np.array([[0.0, 0.0], [4.0, 0.0]])
    features = np.array([[-2.0, 0.0], [0.5, 0.0], [3.0, 0.0], [4.5, 0.0]])

    original = reliability_from_geometry(features, centers)
    swapped = reliability_from_geometry(features, centers[::-1])

    assert original.valid and swapped.valid
    assert swapped.center_distance == pytest.approx(original.center_distance)
    assert swapped.preferred_count_0 == original.preferred_count_1
    assert swapped.preferred_count_1 == original.preferred_count_0
    assert swapped.median_radius_0 == pytest.approx(original.median_radius_1)
    assert swapped.median_radius_1 == pytest.approx(original.median_radius_0)
    assert swapped.reliability_train == pytest.approx(original.reliability_train)


def test_reliability_is_invariant_to_common_scale_rotation_and_translation() -> None:
    centers = np.array([[0.0, 0.0], [4.0, 1.0]])
    features = np.array([[-1.0, 0.0], [0.5, 1.0], [3.0, 0.0], [4.5, 1.5]], dtype=float)
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    shift = np.array([11.0, -7.0])

    original = reliability_from_geometry(features, centers)
    transformed = reliability_from_geometry(
        3.5 * features @ rotation + shift,
        3.5 * centers @ rotation + shift,
    )

    assert original.valid and transformed.valid
    assert transformed.reliability_train == pytest.approx(
        original.reliability_train, abs=1e-14
    )
    assert transformed.center_distance == pytest.approx(3.5 * original.center_distance)


@pytest.mark.parametrize(
    ("features", "centers"),
    [
        (np.array([[0.0], [1.0]]), np.array([[0.0], [np.nan]])),
        (np.array([[0.0], [0.1]]), np.array([[0.0], [10.0]])),
        (np.array([[0.0], [1.0]]), np.array([[0.5], [0.5]])),
    ],
)
def test_reliability_marks_invalid_geometry_without_imputation(
    features: np.ndarray, centers: np.ndarray
) -> None:
    result = reliability_from_geometry(features, centers)

    assert not result.valid
    assert np.isnan(result.reliability_train)


def test_arrival_ablation_changes_only_the_terminal_transition_cost() -> None:
    loss = np.array([[6.003, 2.243], [3.882, 7.846], [7.693, 5.798]])
    penalty = np.array(
        [
            [[0.0, 3.0], [3.0, 0.0]],
            [[0.0, 1.128], [0.405, 0.0]],
            [[0.0, 0.626], [1.943, 0.0]],
        ]
    )
    original = penalty.copy()
    adaptive_values = dp_tv(loss, penalty, return_value_mx=True)
    adaptive_path = adaptive_values.argmin(axis=1)
    adaptive_predecessor = int(
        np.argmin(adaptive_values[-2] + penalty[-1, :, adaptive_path[-1]])
    )
    all_fixed = lam_to_penalty_seq(np.full(len(loss), 3.0), 2)

    result = arrival_ablation_state(loss, penalty, lambda0=3.0)

    assert np.array_equal(adaptive_path, [1, 0, 1])
    assert adaptive_predecessor == 0
    assert result == 0
    # Replacing the entire history would emit state 1, so this toy detects
    # accidental full-path ablation instead of the frozen arrival-only ablation.
    assert dp_tv(loss, all_fixed, return_value_mx=True)[-1].argmin() == 1
    assert np.array_equal(penalty, original)


def test_logistic_fit_recovers_known_grouped_event_rates() -> None:
    features = np.array([[-1.0]] * 4 + [[1.0]] * 4)
    outcome = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0])

    fit = fit_logistic(features, outcome, np.ones(len(outcome)))
    probabilities = fit.predict_proba(np.array([[-1.0], [1.0]]))

    assert fit.converged
    assert fit.coef == pytest.approx([0.0, np.log(3.0)], abs=1e-6)
    assert probabilities == pytest.approx([0.25, 0.75], abs=1e-7)


def test_prediction_scores_use_mean_brier_and_binary_log_loss() -> None:
    brier, log_loss = prediction_scores(np.array([0.0, 1.0]), np.array([0.25, 0.75]))

    assert brier == pytest.approx(0.0625)
    assert log_loss == pytest.approx(-np.log(0.75))


def _folds(
    coefficients: list[float],
    baseline: list[float],
    challenger: list[float],
    *,
    valid: bool = True,
    events: int = 10,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "held_out_market": ["us", "de", "jp"],
            "fold_valid": [valid] * 3,
            "admitted_events": [events] * 3,
            "reliability_coefficient": coefficients,
            "baseline_brier": baseline,
            "challenger_brier": challenger,
        }
    )


def test_decision_rule_distinguishes_supported_falsified_and_inconclusive() -> None:
    supported = _folds(
        [-0.4, -0.2, -0.1],
        [0.20, 0.20, 0.20],
        [0.18, 0.19, 0.21],
    )
    falsified = _folds(
        [0.0, 0.2, -0.1],
        [0.20, 0.20, 0.20],
        [0.21, 0.20, 0.20],
    )
    only_one_market_better = _folds(
        [-0.4, -0.2, -0.1],
        [0.20, 0.20, 0.20],
        [0.17, 0.21, 0.21],
    )

    assert classify_decision(supported, tol=1e-12) == "supported"
    assert classify_decision(falsified, tol=1e-12) == "falsified"
    assert classify_decision(only_one_market_better, tol=1e-12) == "inconclusive"
    assert (
        classify_decision(
            _folds(
                [-0.4, -0.2, -0.1],
                [0.20, 0.20, 0.20],
                [0.18, 0.19, 0.21],
                valid=False,
            ),
            tol=1e-12,
        )
        == "inconclusive"
    )
    assert (
        classify_decision(
            _folds(
                [-0.4, -0.2, -0.1],
                [0.20, 0.20, 0.20],
                [0.18, 0.19, 0.21],
                events=0,
            ),
            tol=1e-12,
        )
        == "inconclusive"
    )


def test_decision_rule_treats_score_changes_within_tolerance_as_equal() -> None:
    tolerance = 1e-12
    folds = _folds(
        [-0.4, -0.2, -0.1],
        [0.20, 0.20, 0.20],
        [0.20 - 0.5 * tolerance] * 3,
    )

    assert classify_decision(folds, tol=tolerance) == "inconclusive"


def test_separation_spec_is_bound_to_registry_cutoff_and_source_allowlist() -> None:
    spec = load_separation_spec(SPEC)
    records = [
        json.loads(line)
        for line in REGISTRY.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["experiment_id"] == spec.experiment_id
    ]

    assert records[-1]["frozen_spec_hash"] == spec.sha256
    assert records[-1]["status"] in {"FROZEN", "EXPERIMENT_COMPLETE"}
    assert spec.data_cutoff.isoformat() == "2023-12-31"
    assert {
        market: value.isoformat() for market, value in spec.evaluation_starts.items()
    } == {
        "us": "2007-12-04",
        "de": "2008-01-03",
        "jp": "2009-05-07",
    }
    assert spec.adaptive_allowed_files == (
        "candidate-states-beta-0.csv",
        "candidate-states-beta-log2.csv",
        "candidate-states-beta-log4.csv",
        "refits-and-scales.csv",
    )
    assert spec.fixed_allowed_files == ("features.csv",)
    assert spec.performance_files_forbidden == (
        "summary.csv",
        "selected-timeline.csv",
        "choices.csv",
        "conclusion.json",
    )


def test_separation_spec_rejects_post_2023_or_extension_access(tmp_path: Path) -> None:
    text = SPEC.read_text(encoding="utf-8")
    for old, new in (
        ("post_2023_access = false", "post_2023_access = true"),
        ("extension_access = false", "extension_access = true"),
        ('data_cutoff = "2023-12-31"', 'data_cutoff = "2024-01-02"'),
        ('us = "2007-12-04"', 'us = "2000-01-01"'),
    ):
        changed = tmp_path / f"{old.split()[0]}.toml"
        changed.write_text(text.replace(old, new), encoding="utf-8")
        with pytest.raises(SeparationStudyError):
            load_separation_spec(changed)
