"""Fixed and adaptive jump-model fitting using the DP path solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from adaptive_jump.dp import solve_regime_path


@dataclass(frozen=True)
class JumpModelResult:
    """Fitted jump model and decoded state path."""

    states: np.ndarray
    centers: np.ndarray
    total_cost: float
    fit_cost: float
    switch_cost: float
    n_switches: int
    n_iter: int
    converged: bool
    switch_penalty: float | np.ndarray


def fit_jump_model(
    features: np.ndarray,
    switch_penalty: float | np.ndarray,
    n_states: int = 2,
    max_iter: int = 25,
    n_init: int = 10,
    random_state: int | None = 0,
    tol: float = 1e-8,
    standardize: bool = True,
) -> JumpModelResult:
    """Fit a jump model by coordinate descent over centers and state path."""
    x = _validate_features(features)
    penalty = _validate_switch_penalty(switch_penalty, len(x))
    if n_states < 1:
        raise ValueError("n_states must be positive")
    if len(x) < n_states:
        raise ValueError("features must have at least n_states rows")
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    if n_init < 1:
        raise ValueError("n_init must be positive")
    if tol < 0.0 or not np.isfinite(tol):
        raise ValueError("tol must be finite and nonnegative")
    x_model, mean, scale = _standardize_features(x) if standardize else (x, np.zeros(x.shape[1]), np.ones(x.shape[1]))

    rng = np.random.default_rng(random_state)
    best: JumpModelResult | None = None
    for centers in _initial_centers(x_model, n_states, n_init, rng):
        candidate = _fit_from_centers(x_model, centers, penalty, max_iter=max_iter, tol=tol)
        centers_original = candidate.centers * scale + mean
        candidate = JumpModelResult(
            states=candidate.states,
            centers=centers_original,
            total_cost=candidate.total_cost,
            fit_cost=candidate.fit_cost,
            switch_cost=candidate.switch_cost,
            n_switches=candidate.n_switches,
            n_iter=candidate.n_iter,
            converged=candidate.converged,
            switch_penalty=switch_penalty,
        )
        if best is None or candidate.total_cost < best.total_cost:
            best = candidate
    if best is None:
        raise ValueError("failed to fit jump model")
    return best


def _fit_from_centers(
    x: np.ndarray,
    centers: np.ndarray,
    penalty: float | np.ndarray,
    max_iter: int,
    tol: float,
) -> JumpModelResult:
    previous_states: np.ndarray | None = None
    previous_cost = np.inf
    current_centers = centers.copy()
    path = None
    converged = False
    for iteration in range(1, max_iter + 1):
        fit_costs = _squared_distances(x, current_centers)
        path = solve_regime_path(fit_costs, penalty)
        if previous_states is not None and np.array_equal(path.states, previous_states):
            converged = True
            break
        if previous_cost - path.total_cost >= 0.0 and previous_cost - path.total_cost < tol:
            converged = True
            break
        previous_states = path.states.copy()
        previous_cost = path.total_cost
        current_centers = _update_centers(x, path.states, current_centers)

    if path is None:
        raise ValueError("jump model coordinate descent did not run")
    final_fit_costs = _squared_distances(x, current_centers)
    final_path = solve_regime_path(final_fit_costs, penalty)
    return JumpModelResult(
        states=final_path.states,
        centers=current_centers,
        total_cost=final_path.total_cost,
        fit_cost=final_path.fit_cost,
        switch_cost=final_path.switch_cost,
        n_switches=final_path.n_switches,
        n_iter=iteration,
        converged=converged or np.array_equal(final_path.states, previous_states),
        switch_penalty=penalty,
    )


def _validate_features(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=float)
    if x.ndim != 2:
        raise ValueError("features must be a 2-D array")
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("features must have non-empty row and column dimensions")
    if not np.isfinite(x).all():
        raise ValueError("features must be finite")
    return x


def _validate_switch_penalty(switch_penalty: float | np.ndarray, n_obs: int) -> float | np.ndarray:
    penalty = np.asarray(switch_penalty, dtype=float)
    if penalty.ndim == 0:
        value = float(penalty)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("switch_penalty must be finite and nonnegative")
        return value
    if penalty.ndim != 1:
        raise ValueError("switch_penalty must be a scalar or 1-D array")
    if len(penalty) != n_obs:
        raise ValueError("1-D switch_penalty must have length equal to feature rows")
    if not np.isfinite(penalty).all() or (penalty < 0.0).any():
        raise ValueError("switch_penalty must be finite and nonnegative")
    return penalty


def _standardize_features(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return (x - mean) / scale, mean, scale


def _initial_centers(
    x: np.ndarray,
    n_states: int,
    n_init: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    centers = [_quantile_centers(x, n_states)]
    for _ in range(n_init - 1):
        rows = rng.choice(len(x), size=n_states, replace=False)
        centers.append(x[rows].copy())
    return centers


def _quantile_centers(x: np.ndarray, n_states: int) -> np.ndarray:
    if x.shape[1] == 1:
        score = x[:, 0]
    else:
        centered = x - x.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        score = centered @ vh[0]
    order = np.argsort(score, kind="mergesort")
    centers = []
    for chunk in np.array_split(order, n_states):
        centers.append(x[chunk].mean(axis=0))
    return np.vstack(centers)


def _squared_distances(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    diff = x[:, None, :] - centers[None, :, :]
    return np.sum(diff * diff, axis=2)


def _update_centers(x: np.ndarray, states: np.ndarray, previous_centers: np.ndarray) -> np.ndarray:
    centers = previous_centers.copy()
    for state in range(len(previous_centers)):
        mask = states == state
        if mask.any():
            centers[state] = x[mask].mean(axis=0)
    return centers
