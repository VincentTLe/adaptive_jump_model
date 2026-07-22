"""Focused mathematical and causal tests for the return-aware Jump Model."""

from itertools import product

import numpy as np
import pandas as pd
import pytest
from jumpmodels.jump import dp, jump_penalty_to_mx

from adaptive_jump.simple_jm_return import (
    ReturnAwareError,
    ReturnAwareJumpModel,
    align_matured_targets,
    dp_return_aware,
    feature_loss_matrix,
    return_aware_loss_matrix,
    standardize_matured_targets,
)


def _path_value(loss: np.ndarray, jump_penalty: float, path) -> float:
    path = np.asarray(path, dtype=int)
    emissions = loss[np.arange(len(path)), path].sum()
    switches = jump_penalty * np.count_nonzero(np.diff(path))
    return float(emissions + switches)


def _brute_force_value(loss: np.ndarray, jump_penalty: float) -> float:
    return min(
        _path_value(loss, jump_penalty, path)
        for path in product(range(loss.shape[1]), repeat=len(loss))
    )


def _separated_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(17)
    negative = rng.normal(-1.0, 0.08, size=(12, 2))
    positive = rng.normal(1.0, 0.08, size=(12, 2))
    X = np.vstack([negative, positive])
    target = np.r_[np.linspace(-0.03, -0.01, 12), np.linspace(0.01, 0.03, 12)]
    mask = np.ones(len(X), dtype=bool)
    mask[-2:] = False
    target[-2:] = np.nan
    return X, target, mask


def test_target_alignment_uses_full_calendar_and_masks_final_two_rows() -> None:
    dates = pd.bdate_range("2020-01-02", periods=10)
    returns = np.arange(10, dtype=float) / 100.0
    feature_dates = dates[2:8]

    aligned = align_matured_targets(
        dates, returns, feature_dates, cutoff=dates[7], offset=2
    )

    assert aligned.feature_dates.equals(feature_dates)
    assert np.array_equal(aligned.matured_mask, [True, True, True, True, False, False])
    assert np.array_equal(aligned.values[:4], returns[4:8])
    assert np.isnan(aligned.values[-2:]).all()
    assert aligned.target_dates[:4].equals(dates[4:8])
    assert aligned.target_dates[-2:].isna().all()


def test_target_alignment_is_invariant_to_post_cutoff_rows_and_values() -> None:
    dates = pd.bdate_range("2020-01-02", periods=10)
    returns = np.linspace(-0.02, 0.02, len(dates))
    feature_dates = dates[1:8]
    cutoff = dates[7]
    prefix = align_matured_targets(dates[:8], returns[:8], feature_dates, cutoff=cutoff)
    changed = returns.copy()
    changed[8:] = [1e6, -1e6]
    full = align_matured_targets(dates, changed, feature_dates, cutoff=cutoff)

    assert prefix.feature_dates.equals(full.feature_dates)
    assert prefix.target_dates.equals(full.target_dates)
    assert np.array_equal(prefix.matured_mask, full.matured_mask)
    assert np.array_equal(prefix.values, full.values, equal_nan=True)


def test_target_standardizer_uses_only_matured_population_moments() -> None:
    target = np.array([1.0, 3.0, np.inf, -np.inf])
    mask = np.array([True, True, False, False])

    result = standardize_matured_targets(target, mask)

    assert result.mean == 2.0
    assert result.scale == 1.0
    assert np.array_equal(result.values, [-1.0, 1.0, 0.0, 0.0])


def test_target_standardizer_rejects_zero_matured_scale() -> None:
    with pytest.raises(ReturnAwareError, match="scale"):
        standardize_matured_targets(
            np.array([0.1, 0.1, np.nan]), np.array([True, True, False])
        )


def test_masked_target_loss_matches_frozen_formula() -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    centers = np.array([[0.0], [2.0]])
    target = np.array([-1.0, 1.0, 99.0, -99.0])
    target_means = np.array([0.5, -0.5])
    mask = np.array([True, True, False, False])

    actual = return_aware_loss_matrix(X, centers, target, target_means, mask, gamma=1)
    expected = feature_loss_matrix(X, centers)
    expected[:2] += 0.5 * (target[:2, None] - target_means[None, :]) ** 2

    assert np.array_equal(actual, expected)
    assert np.array_equal(actual[-2:], feature_loss_matrix(X, centers)[-2:])


def test_gamma_zero_loss_and_dp_are_exact_fixed_jm_math() -> None:
    rng = np.random.default_rng(23)
    X = rng.normal(size=(8, 3))
    centers = rng.normal(size=(2, 3))
    target = rng.normal(size=8)
    target_means = rng.normal(size=2)
    mask = np.array([True] * 6 + [False, False])
    fixed_loss = feature_loss_matrix(X, centers)

    nested_loss = return_aware_loss_matrix(
        X, centers, target, target_means, mask, gamma=0
    )
    fixed_path, fixed_value = dp(fixed_loss, jump_penalty_to_mx(5.0, 2))
    nested_path, nested_value = dp_return_aware(
        X,
        centers,
        target,
        target_means,
        mask,
        gamma=0,
        jump_penalty=5.0,
    )

    assert np.array_equal(nested_loss, fixed_loss)
    assert np.array_equal(nested_path, fixed_path)
    assert nested_value == fixed_value


def test_gamma_zero_estimator_requires_canonical_control_route() -> None:
    X, target, mask = _separated_data()
    model = ReturnAwareJumpModel(jump_penalty=1.0, gamma=0, n_init=2)

    with pytest.raises(ReturnAwareError, match="canonical fixed-JM control"):
        model.fit(X, target, mask)


def test_return_aware_dp_matches_brute_force_path_objective() -> None:
    X = np.array([[-1.0], [-0.8], [0.1], [0.9], [1.1]])
    centers = np.array([[-0.9], [1.0]])
    target = np.array([-1.0, -0.8, 0.2, 0.0, 0.0])
    target_means = np.array([-0.9, 0.4])
    mask = np.array([True, True, True, False, False])
    jump_penalty = 0.7
    loss = return_aware_loss_matrix(X, centers, target, target_means, mask, gamma=1)

    path, value = dp_return_aware(
        X,
        centers,
        target,
        target_means,
        mask,
        gamma=1,
        jump_penalty=jump_penalty,
    )

    assert value == pytest.approx(_brute_force_value(loss, jump_penalty))
    assert _path_value(loss, jump_penalty, path) == pytest.approx(value)


def test_fit_labels_state_zero_by_higher_matured_target_mean() -> None:
    X, target, mask = _separated_data()

    model = ReturnAwareJumpModel(
        jump_penalty=1.0, gamma=1, n_init=4, random_state=3
    ).fit(X, target, mask)
    labels = np.asarray(model.labels_)

    assert model.target_means_[0] > model.target_means_[1]
    assert labels[:10].tolist() == [1] * 10
    assert labels[12:].tolist() == [0] * 12
    assert model.matured_target_count_ == len(X) - 2
    assert model.matured_state_counts_.sum() == len(X) - 2


def test_fit_rejects_exactly_tied_finite_target_means() -> None:
    X = np.array([[-2.1], [-2.0], [-1.9], [-2.05], [1.9], [2.0], [2.1], [2.05]])
    target = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    mask = np.ones(len(X), dtype=bool)

    model = ReturnAwareJumpModel(jump_penalty=0.1, gamma=1, n_init=4, random_state=0)

    with pytest.raises(ReturnAwareError, match="target means are exactly tied"):
        model.fit(X, target, mask)


def test_fit_m_steps_and_objective_components_are_exact() -> None:
    X, target, mask = _separated_data()
    standardized = standardize_matured_targets(target, mask)
    model = ReturnAwareJumpModel(
        jump_penalty=0.5, gamma=1, n_init=4, random_state=4
    ).fit(X, target, mask)
    labels = np.asarray(model.labels_)

    for state in (0, 1):
        rows = labels == state
        mature_rows = rows & mask
        assert np.allclose(model.centers_[state], X[rows].mean(axis=0))
        assert model.target_means_standardized_[state] == pytest.approx(
            standardized.values[mature_rows].mean()
        )
    assert model.val_ == pytest.approx(
        model.feature_value_ + model.target_value_ + model.transition_value_
    )
    for history in model.all_objective_histories_:
        assert (np.diff(history) <= 1e-10).all()


def test_online_inference_uses_features_only_and_is_prefix_invariant() -> None:
    X, target, mask = _separated_data()
    model = ReturnAwareJumpModel(
        jump_penalty=0.5, gamma=1, n_init=4, random_state=7
    ).fit(X, target, mask)
    before = np.asarray(model.predict_online(X))
    prefix = np.asarray(model.predict_online(X[:15]))

    model.target_means_standardized_[:] = [1e9, -1e9]
    model.target_means_[:] = [1e9, -1e9]
    after = np.asarray(model.predict_online(X))

    assert np.array_equal(after, before)
    assert np.array_equal(prefix, before[:15])


def test_toy_return_outcome_changes_joint_training_loss() -> None:
    X = np.array([[-0.3], [-0.2], [-0.1], [0.1], [0.2], [0.3]])
    centers = np.array([[-0.2], [0.2]])
    target = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
    mask = np.ones(len(X), dtype=bool)
    aligned_means = np.array([-1.0, 1.0])
    reversed_means = aligned_means[::-1]

    aligned_loss = return_aware_loss_matrix(
        X, centers, target, aligned_means, mask, gamma=1
    )
    reversed_loss = return_aware_loss_matrix(
        X, centers, target, reversed_means, mask, gamma=1
    )
    expected_path = np.array([0, 0, 0, 1, 1, 1])

    assert _path_value(aligned_loss, 0.0, expected_path) < _path_value(
        reversed_loss, 0.0, expected_path
    )
