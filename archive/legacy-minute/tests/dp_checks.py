from itertools import product

import numpy as np
import pytest

from adaptive_jump.dp import RegimePathResult, solve_regime_path


def test_lambda_zero_chooses_independent_row_argmins():
    fit_costs = np.array(
        [
            [0.0, 5.0],
            [4.0, 1.0],
            [0.0, 3.0],
        ]
    )

    result = solve_regime_path(fit_costs, 0.0)

    assert isinstance(result, RegimePathResult)
    assert result.states.tolist() == [0, 1, 0]
    assert result.fit_cost == pytest.approx(1.0)
    assert result.switch_cost == pytest.approx(0.0)
    assert result.total_cost == pytest.approx(1.0)
    assert result.n_switches == 2


def test_huge_lambda_favors_cheapest_constant_path():
    fit_costs = np.array(
        [
            [0.0, 2.0],
            [2.0, 0.0],
            [0.0, 2.0],
        ]
    )

    result = solve_regime_path(fit_costs, 1_000_000.0)

    assert result.states.tolist() == [0, 0, 0]
    assert result.fit_cost == pytest.approx(2.0)
    assert result.switch_cost == pytest.approx(0.0)
    assert result.total_cost == pytest.approx(2.0)
    assert result.n_switches == 0


def test_vector_penalty_applies_to_transition_into_row_t():
    fit_costs = np.array(
        [
            [0.0, 60.0],
            [0.0, 60.0],
            [50.0, 0.0],
        ]
    )

    cheap_switch_into_last_row = solve_regime_path(fit_costs, np.array([999.0, 999.0, 0.0]))
    expensive_switch_into_last_row = solve_regime_path(fit_costs, np.array([999.0, 0.0, 999.0]))

    assert cheap_switch_into_last_row.states.tolist() == [0, 0, 1]
    assert cheap_switch_into_last_row.total_cost == pytest.approx(0.0)
    assert cheap_switch_into_last_row.n_switches == 1

    assert expensive_switch_into_last_row.states.tolist() == [0, 0, 0]
    assert expensive_switch_into_last_row.total_cost == pytest.approx(50.0)
    assert expensive_switch_into_last_row.n_switches == 0


def test_vector_penalty_first_entry_is_ignored():
    fit_costs = np.array(
        [
            [0.0, 100.0],
            [100.0, 0.0],
        ]
    )

    result = solve_regime_path(fit_costs, np.array([1_000_000_000.0, 0.0]))

    assert result.states.tolist() == [0, 1]
    assert result.total_cost == pytest.approx(0.0)
    assert result.n_switches == 1


def test_single_row_has_no_switch_cost():
    result = solve_regime_path(np.array([[3.0, 1.0, 2.0]]), 100.0)

    assert result.states.tolist() == [1]
    assert result.fit_cost == pytest.approx(1.0)
    assert result.switch_cost == pytest.approx(0.0)
    assert result.total_cost == pytest.approx(1.0)
    assert result.n_switches == 0


def test_one_state_path_is_all_zero():
    fit_costs = np.array([[2.0], [5.0], [1.0]])

    result = solve_regime_path(fit_costs, np.array([9.0, 9.0, 9.0]))

    assert result.states.tolist() == [0, 0, 0]
    assert result.fit_cost == pytest.approx(8.0)
    assert result.switch_cost == pytest.approx(0.0)
    assert result.total_cost == pytest.approx(8.0)
    assert result.n_switches == 0


def test_ties_return_lexicographically_smallest_path():
    fit_costs = np.zeros((3, 3))

    result = solve_regime_path(fit_costs, 0.0)

    assert result.states.tolist() == [0, 0, 0]
    assert result.total_cost == pytest.approx(0.0)


def test_nontrivial_ties_match_lexicographic_brute_force_oracle():
    fit_costs = np.array(
        [
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 5.0],
            [5.0, 0.0, 0.0],
        ]
    )

    result = solve_regime_path(fit_costs, 0.0)
    oracle = _brute_force_oracle(fit_costs, 0.0)

    assert result.states.tolist() == [0, 0, 1]
    assert result.states.tolist() == oracle["states"].tolist()
    assert result.total_cost == pytest.approx(oracle["total_cost"])


@pytest.mark.parametrize(
    ("fit_costs", "switch_penalty"),
    [
        (
            np.array(
                [
                    [0.0, 2.0, 1.0],
                    [1.0, 0.0, 3.0],
                    [2.0, 1.0, 0.0],
                ]
            ),
            0.75,
        ),
        (
            np.array(
                [
                    [0.0, 4.0],
                    [2.0, 0.0],
                    [2.0, 0.0],
                    [0.0, 3.0],
                ]
            ),
            np.array([9.0, 0.5, 2.0, 0.0]),
        ),
        (np.zeros((4, 3)), 0.0),
    ],
)
def test_dp_matches_brute_force_oracle(fit_costs, switch_penalty):
    result = solve_regime_path(fit_costs, switch_penalty)
    oracle = _brute_force_oracle(fit_costs, switch_penalty)

    assert result.states.tolist() == oracle["states"].tolist()
    assert result.fit_cost == pytest.approx(oracle["fit_cost"])
    assert result.switch_cost == pytest.approx(oracle["switch_cost"])
    assert result.total_cost == pytest.approx(oracle["total_cost"])
    assert result.n_switches == oracle["n_switches"]


@pytest.mark.parametrize(
    ("fit_costs", "switch_penalty", "message"),
    [
        (np.array([1.0, 2.0]), 0.0, "2-D"),
        (np.empty((0, 2)), 0.0, "non-empty"),
        (np.empty((2, 0)), 0.0, "non-empty"),
        (np.array([[0.0, np.nan]]), 0.0, "finite"),
        (np.array([[0.0, 1.0]]), -1.0, "nonnegative"),
        (np.array([[0.0, 1.0]]), np.inf, "finite"),
        (np.zeros((3, 2)), np.array([0.0, 1.0]), "length"),
        (np.zeros((3, 2)), np.zeros((3, 1)), "scalar or 1-D"),
        (np.zeros((3, 2)), np.array([0.0, -1.0, 0.0]), "nonnegative"),
        (np.zeros((3, 2)), np.array([0.0, np.nan, 0.0]), "finite"),
    ],
)
def test_solve_regime_path_rejects_invalid_inputs(fit_costs, switch_penalty, message):
    with pytest.raises(ValueError, match=message):
        solve_regime_path(fit_costs, switch_penalty)


def _brute_force_oracle(fit_costs: np.ndarray, switch_penalty: float | np.ndarray) -> dict[str, object]:
    costs = np.asarray(fit_costs, dtype=float)
    penalties = _penalty_vector(switch_penalty, len(costs))
    n_steps, n_states = costs.shape
    best: dict[str, object] | None = None

    for path in product(range(n_states), repeat=n_steps):
        states = np.array(path, dtype=int)
        rows = np.arange(n_steps)
        fit_cost = float(costs[rows, states].sum())
        switches = states[1:] != states[:-1]
        switch_cost = float(penalties[1:][switches].sum())
        total_cost = fit_cost + switch_cost
        candidate = {
            "states": states,
            "total_cost": total_cost,
            "fit_cost": fit_cost,
            "switch_cost": switch_cost,
            "n_switches": int(switches.sum()),
        }
        if best is None or (total_cost, path) < (best["total_cost"], tuple(best["states"])):
            best = candidate

    assert best is not None
    return best


def _penalty_vector(switch_penalty: float | np.ndarray, n_steps: int) -> np.ndarray:
    penalty = np.asarray(switch_penalty, dtype=float)
    if penalty.ndim == 0:
        out = np.zeros(n_steps, dtype=float)
        out[1:] = float(penalty)
        return out
    return penalty
