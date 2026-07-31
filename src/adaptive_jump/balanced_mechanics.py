"""Synthetic mathematical oracles for the pair-balanced lagged penalty."""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np

from adaptive_jump.balanced_model import (
    BUILDERS,
    BalancedSpec,
    balanced_lagged_penalty_seq,
)
from adaptive_jump.tv_jump import dp_tv, lam_to_penalty_seq


def independent_balanced_penalty(
    loss_mx: np.ndarray, lambda0: float, beta: float, q_train: float
) -> np.ndarray:
    """Formula reconstruction that does not call the production builder."""
    loss = np.asarray(loss_mx, dtype=float)
    if loss.ndim != 2 or not np.isfinite(loss).any(axis=1).all():
        raise ValueError("independent formula requires one finite loss per row")
    clean = np.where(np.isnan(loss), np.inf, loss)
    evidence = np.zeros_like(clean)
    evidence[1:] = clean[:-1]
    left = evidence[:, :, None]
    right = evidence[:, None, :]
    with np.errstate(invalid="ignore"):
        gaps = left - right
    gaps[np.isposinf(left) & np.isposinf(right)] = 0.0
    alpha = 1.0 - math.exp(-float(beta))
    result = float(lambda0) * (1.0 - alpha * np.tanh(gaps / float(q_train)))
    indices = np.arange(loss.shape[1])
    result[:, indices, indices] = 0.0
    return result


def _path_value(loss: np.ndarray, penalty: np.ndarray, path: tuple[int, ...]) -> float:
    value = sum(float(loss[t, state]) for t, state in enumerate(path))
    value += sum(float(penalty[t, path[t - 1], path[t]]) for t in range(1, len(path)))
    return value


def _brute_force(loss: np.ndarray, penalty: np.ndarray) -> float:
    return min(
        _path_value(loss, penalty, path)
        for path in itertools.product(range(loss.shape[1]), repeat=len(loss))
    )


def _online_path(loss: np.ndarray, penalty: np.ndarray) -> list[int]:
    return dp_tv(loss, penalty, return_value_mx=True).argmin(axis=1).tolist()


def mechanical_prerequisites(spec: BalancedSpec) -> dict[str, Any]:
    """Recompute every frozen algebraic, objective, and toy prerequisite."""
    atol = spec.numerical_tolerance
    loss = np.array([[0.2, 3.1], [2.4, 0.1], [0.7, 1.6], [3.0, 0.3]], dtype=float)
    lambda0, beta, q_train = 4.0, spec.decision_beta, 1.7
    penalty = balanced_lagged_penalty_seq(loss, lambda0, beta, q_train)
    expected = independent_balanced_penalty(loss, lambda0, beta, q_train)
    off = penalty[:, ~np.eye(2, dtype=bool)]
    pair_error = float(
        np.max(np.abs(penalty[:, 0, 1] + penalty[:, 1, 0] - 2.0 * lambda0))
    )
    beta0 = balanced_lagged_penalty_seq(loss, lambda0, 0.0, q_train)
    scaled = balanced_lagged_penalty_seq(loss * 8.0, lambda0, beta, q_train * 8.0)

    hysteresis_loss = np.array(
        [[0.0, 20.0], [20.0, 0.0], [0.0, 7.0], [2.0, 1.0], [1.0, 3.0]]
    )
    hysteresis_penalty = balanced_lagged_penalty_seq(
        hysteresis_loss, lambda0, beta, q_train
    )
    hysteresis_values = dp_tv(hysteresis_loss, hysteresis_penalty, return_value_mx=True)
    observed_difference = hysteresis_values[:, 1] - hysteresis_values[:, 0]
    loss_difference = hysteresis_loss[:, 1] - hysteresis_loss[:, 0]
    recursive_difference = np.empty(len(hysteresis_loss))
    recursive_difference[0] = loss_difference[0]
    for t in range(1, len(hysteresis_loss)):
        recursive_difference[t] = loss_difference[t] + np.clip(
            recursive_difference[t - 1],
            -hysteresis_penalty[t, 1, 0],
            hysteresis_penalty[t, 0, 1],
        )
    lower = -hysteresis_penalty[1:, 1, 0]
    upper = hysteresis_penalty[1:, 0, 1]
    previous = recursive_difference[:-1]
    clipping_regimes = bool(
        (previous < lower - atol).any()
        and (previous > upper + atol).any()
        and ((previous >= lower - atol) & (previous <= upper + atol)).any()
    )
    hysteresis_width_error = float(
        np.max(
            np.abs(
                hysteresis_penalty[:, 0, 1]
                + hysteresis_penalty[:, 1, 0]
                - 2.0 * lambda0
            )
        )
    )

    missing = np.array([[0.0, np.nan, np.inf], [np.inf, 0.0, np.nan], [0.5, 1.0, 2.0]])
    missing_penalty = balanced_lagged_penalty_seq(missing, lambda0, beta, q_train)
    missing_expected = independent_balanced_penalty(missing, lambda0, beta, q_train)
    missing_off = missing_penalty[:, ~np.eye(3, dtype=bool)]
    missing_pair_error = max(
        float(
            np.max(
                np.abs(
                    missing_penalty[:, i, j] + missing_penalty[:, j, i] - 2.0 * lambda0
                )
            )
        )
        for i in range(3)
        for j in range(i + 1, 3)
    )
    swapped = balanced_lagged_penalty_seq(loss[:, ::-1], lambda0, beta, q_train)[
        :, ::-1, ::-1
    ]

    monotone_gaps = np.array([-8.0, -1.0, 0.0, 1.0, 8.0])
    monotone_costs = lambda0 * (
        1.0 - (1.0 - math.exp(-beta)) * np.tanh(monotone_gaps / q_train)
    )

    _, optimal = dp_tv(loss, penalty)
    brute = _brute_force(loss, penalty)
    alpha = 1.0 - math.exp(-beta)
    fixed = lam_to_penalty_seq(np.full(len(loss), lambda0), 2)
    objective_bound_errors: list[float] = []
    for path in itertools.product(range(2), repeat=len(loss)):
        switches = sum(path[t] != path[t - 1] for t in range(1, len(path)))
        difference = abs(
            _path_value(loss, penalty, path) - _path_value(loss, fixed, path)
        )
        objective_bound_errors.append(difference - switches * lambda0 * alpha)

    toy_paths: dict[str, dict[str, list[int]]] = {}
    for name, toy_loss in spec.toy_losses.items():
        toy_paths[name] = {
            "fixed": _online_path(
                toy_loss, lam_to_penalty_seq(np.full(len(toy_loss), 4.0), 2)
            ),
            "lagged": _online_path(
                toy_loss,
                BUILDERS["lagged"](toy_loss, 4.0, spec.decision_beta, 1.0),
            ),
            "balanced": _online_path(
                toy_loss,
                BUILDERS["balanced"](toy_loss, 4.0, spec.decision_beta, 1.0),
            ),
        }

    x = np.linspace(0.0, 1.0, 101)
    forward_difference = 1.0 - alpha * x - np.exp(-beta * x)
    checks = {
        "formula": bool(np.allclose(penalty, expected, rtol=0, atol=atol)),
        "missing_state_semantics": bool(
            np.isfinite(missing_penalty).all()
            and np.allclose(missing_penalty, missing_expected, rtol=0, atol=atol)
            and missing_pair_error <= atol
            and (missing_off >= lambda0 * math.exp(-beta) - atol).all()
            and (missing_off <= lambda0 * (2.0 - math.exp(-beta)) + atol).all()
        ),
        "beta_zero": bool(np.array_equal(beta0, fixed)),
        "pair_balance": pair_error <= atol,
        "binary_hysteresis_width": bool(
            np.allclose(observed_difference, recursive_difference, rtol=0, atol=atol)
            and hysteresis_width_error <= atol
            and clipping_regimes
        ),
        "bounds": bool(
            (off >= lambda0 * math.exp(-beta) - atol).all()
            and (off <= lambda0 * (2.0 - math.exp(-beta)) + atol).all()
        ),
        "monotonicity": bool((np.diff(monotone_costs) <= atol).all()),
        "scale_invariance": bool(np.allclose(scaled, penalty, rtol=0, atol=atol)),
        "label_swap_equivariance": bool(
            np.allclose(swapped, penalty, rtol=0, atol=atol)
        ),
        "objective_bound": max(objective_bound_errors) <= atol,
        "brute_force": abs(float(optimal) - brute) <= atol,
        "forward_discount": bool(
            forward_difference.min() >= -atol
            and abs(float(forward_difference[0])) <= atol
            and abs(float(forward_difference[-1])) <= atol
        ),
        "toy_paths": toy_paths == spec.toy_paths,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "max_formula_abs_error": float(np.max(np.abs(penalty - expected))),
        "max_pair_sum_abs_error": pair_error,
        "max_objective_bound_excess": max(objective_bound_errors),
        "toy_paths": toy_paths,
    }
