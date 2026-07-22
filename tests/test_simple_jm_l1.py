"""Oracle tests for the fixed-cost raw-cityblock jump model."""

from itertools import product

import numpy as np
import pandas as pd
import pytest
from jumpmodels.jump import JumpModel

from adaptive_jump.simple_jm_l1 import (
    L1JumpModel,
    componentwise_median_centers,
    l1_loss_matrix,
    solve_l1_path,
)


def _path_objective(loss: np.ndarray, labels: np.ndarray, jump_penalty: float) -> float:
    observations = np.arange(len(labels))
    switches = int((labels[1:] != labels[:-1]).sum())
    return float(loss[observations, labels].sum() + jump_penalty * switches)


def _persistent_sample() -> tuple[pd.DataFrame, pd.Series]:
    low = np.array(
        [
            [-3.0, -2.0],
            [-2.7, -2.2],
            [-3.2, -1.8],
            [-2.9, -2.1],
            [-3.1, -1.9],
        ]
    )
    high = np.array(
        [
            [3.0, 2.0],
            [2.8, 2.3],
            [3.2, 1.9],
            [2.9, 2.1],
            [3.1, 1.8],
        ]
    )
    values = pd.DataFrame(np.vstack([low, high]), columns=["x0", "x1"])
    returns = pd.Series(
        np.r_[np.full(len(low), 0.02), np.full(len(high), -0.02)],
        index=values.index,
    )
    return values, returns


def test_l1_loss_matrix_is_unscaled_cityblock_distance() -> None:
    values = np.array([[0.0, 0.0], [2.0, -1.0]])
    centers = np.array([[0.0, 1.0], [3.0, -2.0]])

    loss = l1_loss_matrix(values, centers)

    assert np.array_equal(loss, np.array([[1.0, 5.0], [4.0, 2.0]]))


def test_componentwise_median_m_step_and_empty_state() -> None:
    values = np.array([[0.0, 10.0], [2.0, 14.0], [100.0, -4.0], [4.0, 12.0]])
    labels = np.zeros(len(values), dtype=int)

    centers = componentwise_median_centers(values, labels)

    assert np.array_equal(centers[0], np.array([3.0, 11.0]))
    assert np.isnan(centers[1]).all()
    median_loss = np.abs(values - centers[0]).sum()
    assert median_loss <= np.abs(values - np.array([0.0, 0.0])).sum()


def test_fixed_cost_dp_matches_brute_force_and_direct_objective() -> None:
    values = np.array([[0.0, 0.0], [0.3, 0.2], [2.8, 3.1], [0.2, 0.1]])
    centers = np.array([[0.0, 0.0], [3.0, 3.0]])
    jump_penalty = 1.25
    loss = l1_loss_matrix(values, centers)

    labels, value = solve_l1_path(values, centers, jump_penalty)
    candidates = []
    for path in product(range(2), repeat=len(values)):
        candidate = np.asarray(path, dtype=int)
        candidates.append(_path_objective(loss, candidate, jump_penalty))

    assert value == pytest.approx(min(candidates), rel=0, abs=1e-15)
    assert value == pytest.approx(
        _path_objective(loss, np.asarray(labels), jump_penalty), rel=0, abs=1e-15
    )


def test_initial_centers_exactly_reuse_upstream_deterministic_semantics() -> None:
    values = np.random.default_rng(17).normal(size=(40, 3))
    common = dict(n_components=2, n_init=5, random_state=11)

    expected = JumpModel(jump_penalty=7.0, **common).init_centers(values)
    actual = L1JumpModel(jump_penalty=7.0, **common).init_centers(values)

    assert np.array_equal(actual, expected)


def test_fit_uses_medians_recomputes_objective_and_sorts_by_cumret() -> None:
    values, returns = _persistent_sample()
    jump_penalty = 0.75

    model = L1JumpModel(jump_penalty=jump_penalty, n_init=5, random_state=3).fit(
        values, returns
    )

    labels = np.asarray(model.labels_, dtype=int)
    expected_centers = componentwise_median_centers(values.to_numpy(), labels)
    loss = l1_loss_matrix(values.to_numpy(), model.centers_)
    state_return_sums = np.array(
        [returns.to_numpy()[labels == state].sum() for state in range(2)]
    )

    assert np.array_equal(model.centers_, expected_centers, equal_nan=True)
    assert model.val_ == pytest.approx(
        _path_objective(loss, labels, jump_penalty), rel=0, abs=1e-15
    )
    assert state_return_sums[0] > state_return_sums[1]
    assert np.asarray(model.labels_).shape == (len(values),)


def test_fit_is_exactly_deterministic_for_the_same_seed() -> None:
    values, returns = _persistent_sample()
    settings = dict(jump_penalty=1.5, n_init=6, random_state=29)

    first = L1JumpModel(**settings).fit(values, returns)
    second = L1JumpModel(**settings).fit(values, returns)

    assert np.array_equal(first.centers_, second.centers_, equal_nan=True)
    assert np.array_equal(np.asarray(first.labels_), np.asarray(second.labels_))
    assert first.val_ == second.val_


def test_online_row_matches_offline_terminal_for_every_prefix() -> None:
    values, returns = _persistent_sample()
    model = L1JumpModel(jump_penalty=1.0, n_init=4, random_state=5).fit(values, returns)
    array = values.to_numpy()

    online = np.asarray(model.predict_online(array))

    for stop in range(1, len(array) + 1):
        offline = np.asarray(model.predict(array[:stop]))
        assert online[stop - 1] == offline[-1]


def test_online_states_are_prefix_invariant_to_added_future_values() -> None:
    values, returns = _persistent_sample()
    model = L1JumpModel(jump_penalty=0.5, n_init=4, random_state=7).fit(values, returns)
    prefix = values.to_numpy()[:7]
    future = np.array([[1000.0, -1000.0], [-1000.0, 1000.0]])

    short = np.asarray(model.predict_online(prefix))
    extended = np.asarray(model.predict_online(np.vstack([prefix, future])))

    assert np.array_equal(short, extended[: len(prefix)])


@pytest.mark.parametrize("jump_penalty", [-1.0, np.inf, np.nan])
def test_solver_rejects_invalid_jump_penalty(jump_penalty: float) -> None:
    values = np.array([[0.0], [1.0]])
    centers = np.array([[0.0], [1.0]])

    with pytest.raises(ValueError, match="finite and nonnegative"):
        solve_l1_path(values, centers, jump_penalty)


def test_model_rejects_non_two_state_or_nonfinite_fit() -> None:
    with pytest.raises(ValueError, match="exactly two"):
        L1JumpModel(n_components=3)
    model = L1JumpModel(n_init=2)
    values = np.array([[0.0], [np.inf]])

    with pytest.raises(ValueError, match="finite 2-d"):
        model.fit(values, np.array([0.01, -0.01]))
