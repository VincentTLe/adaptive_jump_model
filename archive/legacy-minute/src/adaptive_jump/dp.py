"""Dynamic-programming solver for jump-model state paths."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RegimePathResult:
    """Auditable result from the switching-path dynamic program."""

    states: np.ndarray
    total_cost: float
    fit_cost: float
    switch_cost: float
    n_switches: int


def solve_regime_path(fit_costs: np.ndarray, switch_penalty: float | np.ndarray) -> RegimePathResult:
    """Solve the minimum-cost regime path with an adjacent switch penalty.

    The objective is
    ``sum_t fit_costs[t, state_t] + sum_{t=1..T-1} lambda[t] * 1[state_t != state_{t-1}]``.
    If ``switch_penalty`` is a vector, it must have length ``T`` and entry
    ``t`` applies to the transition into row ``t``. Entry zero is validated but
    not charged.
    """
    costs = _validate_fit_costs(fit_costs)
    penalties = _validate_switch_penalty(switch_penalty, costs.shape[0])
    n_steps, n_states = costs.shape

    dp = np.empty((n_steps, n_states), dtype=float)
    prev = np.full((n_steps, n_states), -1, dtype=int)
    dp[0] = costs[0]
    ranks = np.arange(n_states, dtype=int)

    for t in range(1, n_steps):
        old_ranks = ranks
        next_ranks = np.empty(n_states, dtype=int)
        rank_keys: list[tuple[int, int]] = []

        for state in range(n_states):
            best_prev = 0
            best_cost = dp[t - 1, 0] + (0.0 if state == 0 else penalties[t])
            best_rank = old_ranks[0]

            for candidate_prev in range(1, n_states):
                transition_cost = 0.0 if candidate_prev == state else penalties[t]
                candidate_cost = dp[t - 1, candidate_prev] + transition_cost
                candidate_rank = old_ranks[candidate_prev]
                if candidate_cost < best_cost or (candidate_cost == best_cost and candidate_rank < best_rank):
                    best_prev = candidate_prev
                    best_cost = candidate_cost
                    best_rank = candidate_rank

            prev[t, state] = best_prev
            dp[t, state] = costs[t, state] + best_cost
            rank_keys.append((best_rank, state))

        for rank, state in enumerate(sorted(range(n_states), key=lambda k: rank_keys[k])):
            next_ranks[state] = rank
        ranks = next_ranks

    final_state = 0
    final_cost = dp[-1, 0]
    final_rank = ranks[0]
    for state in range(1, n_states):
        if dp[-1, state] < final_cost or (dp[-1, state] == final_cost and ranks[state] < final_rank):
            final_state = state
            final_cost = dp[-1, state]
            final_rank = ranks[state]
    if not np.isfinite(final_cost):
        raise ValueError("dynamic-programming costs must remain finite")

    states = np.empty(n_steps, dtype=int)
    states[-1] = final_state
    for t in range(n_steps - 1, 0, -1):
        states[t - 1] = prev[t, states[t]]

    fit_cost, switch_cost, n_switches = _score_path(costs, penalties, states)
    total_cost = float(fit_cost + switch_cost)
    if not np.isclose(total_cost, final_cost):
        raise ValueError("path accounting must match dynamic-programming cost")
    return RegimePathResult(
        states=states,
        total_cost=total_cost,
        fit_cost=fit_cost,
        switch_cost=switch_cost,
        n_switches=n_switches,
    )


def _validate_fit_costs(fit_costs: np.ndarray) -> np.ndarray:
    costs = np.asarray(fit_costs, dtype=float)
    if costs.ndim != 2:
        raise ValueError("fit_costs must be a 2-D array")
    if costs.shape[0] == 0 or costs.shape[1] == 0:
        raise ValueError("fit_costs must have non-empty time and state dimensions")
    if not np.isfinite(costs).all():
        raise ValueError("fit_costs must be finite")
    return costs


def _validate_switch_penalty(switch_penalty: float | np.ndarray, n_steps: int) -> np.ndarray:
    penalty = np.asarray(switch_penalty, dtype=float)
    if penalty.ndim == 0:
        value = float(penalty)
        if not np.isfinite(value):
            raise ValueError("switch_penalty must be finite")
        if value < 0.0:
            raise ValueError("switch_penalty must be nonnegative")
        out = np.zeros(n_steps, dtype=float)
        out[1:] = value
        return out
    if penalty.ndim != 1:
        raise ValueError("switch_penalty must be a scalar or 1-D array")
    if len(penalty) != n_steps:
        raise ValueError("1-D switch_penalty must have length equal to fit_costs rows")
    if not np.isfinite(penalty).all():
        raise ValueError("switch_penalty must be finite")
    if (penalty < 0.0).any():
        raise ValueError("switch_penalty must be nonnegative")
    return penalty


def _score_path(costs: np.ndarray, penalties: np.ndarray, states: np.ndarray) -> tuple[float, float, int]:
    rows = np.arange(len(states))
    fit_cost = float(costs[rows, states].sum())
    if len(states) == 1:
        return fit_cost, 0.0, 0
    switches = states[1:] != states[:-1]
    switch_cost = float(penalties[1:][switches].sum())
    return fit_cost, switch_cost, int(switches.sum())
