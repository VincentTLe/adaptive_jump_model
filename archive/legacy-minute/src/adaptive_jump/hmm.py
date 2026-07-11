"""Gaussian HMM baseline for one-dimensional return series."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from hmmlearn.hmm import GaussianHMM


@dataclass(frozen=True)
class GaussianHMMResult:
    """Fitted Gaussian HMM and decoded state path."""

    states: np.ndarray
    means: np.ndarray
    variances: np.ndarray
    startprob: np.ndarray
    transmat: np.ndarray
    loglik: float
    n_iter: int
    converged: bool


def fit_gaussian_hmm(
    x: np.ndarray,
    n_states: int = 2,
    n_init: int = 10,
    max_iter: int = 100,
    tol: float = 1e-6,
    random_state: int | None = 0,
    min_variance: float = 1e-8,
) -> GaussianHMMResult:
    """Fit a one-dimensional Gaussian HMM and relabel by variance."""
    values = _validate_1d_input(x)
    if n_states < 1:
        raise ValueError("n_states must be positive")
    if len(values) < n_states:
        raise ValueError("x must contain at least n_states observations")
    if n_init < 1:
        raise ValueError("n_init must be positive")
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    if tol <= 0.0 or not np.isfinite(tol):
        raise ValueError("tol must be finite and positive")
    if min_variance <= 0.0 or not np.isfinite(min_variance):
        raise ValueError("min_variance must be finite and positive")

    x2 = values.reshape(-1, 1)
    rng = np.random.default_rng(random_state)
    best_model: GaussianHMM | None = None
    best_loglik = -np.inf
    for labels in _initial_labels(values, n_states, n_init, rng):
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=max_iter,
            tol=tol,
            random_state=int(rng.integers(0, 2**31 - 1)),
            init_params="",
            params="stmc",
            min_covar=min_variance,
        )
        startprob, transmat, means, variances = _parameters_from_labels(values, labels, n_states, min_variance)
        model.startprob_ = startprob
        model.transmat_ = transmat
        model.means_ = means.reshape(-1, 1)
        model.covars_ = variances.reshape(-1, 1)
        model.fit(x2)
        loglik = float(model.score(x2))
        if loglik > best_loglik:
            best_loglik = loglik
            best_model = model

    if best_model is None:
        raise ValueError("failed to fit Gaussian HMM")
    states = best_model.predict(x2)
    result = GaussianHMMResult(
        states=states,
        means=best_model.means_.reshape(n_states),
        variances=_model_variances(best_model, min_variance),
        startprob=np.asarray(best_model.startprob_, dtype=float),
        transmat=np.asarray(best_model.transmat_, dtype=float),
        loglik=best_loglik,
        n_iter=int(best_model.monitor_.iter),
        converged=bool(best_model.monitor_.converged),
    )
    return _relabel_by_variance(result)


def _gaussian_logpdf(x: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=float).reshape(-1, 1)
    means = np.asarray(means, dtype=float).reshape(1, -1)
    variances = np.asarray(variances, dtype=float).reshape(1, -1)
    if (variances <= 0.0).any():
        raise ValueError("variances must be positive")
    return -0.5 * (np.log(2.0 * np.pi * variances) + (values - means) ** 2 / variances)


def _validate_1d_input(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    if values.ndim != 1:
        raise ValueError("x must be a 1-D array")
    if len(values) == 0:
        raise ValueError("x must be non-empty")
    if not np.isfinite(values).all():
        raise ValueError("x must be finite")
    return values


def _initial_labels(values: np.ndarray, n_states: int, n_init: int, rng: np.random.Generator) -> list[np.ndarray]:
    labels = [_quantile_labels(values, n_states)]
    for _ in range(n_init - 1):
        labels.append(_balanced_random_labels(len(values), n_states, rng))
    return labels


def _quantile_labels(values: np.ndarray, n_states: int) -> np.ndarray:
    volatility_score = np.abs(values - np.median(values))
    order = np.argsort(volatility_score, kind="mergesort")
    labels = np.empty(len(values), dtype=int)
    for state, chunk in enumerate(np.array_split(order, n_states)):
        labels[chunk] = state
    return labels


def _balanced_random_labels(n_obs: int, n_states: int, rng: np.random.Generator) -> np.ndarray:
    labels = np.arange(n_obs) % n_states
    rng.shuffle(labels)
    return labels.astype(int)


def _parameters_from_labels(
    values: np.ndarray,
    labels: np.ndarray,
    n_states: int,
    min_variance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    means = np.empty(n_states, dtype=float)
    variances = np.empty(n_states, dtype=float)
    global_mean = float(np.mean(values))
    global_var = max(float(np.var(values)), min_variance)
    for state in range(n_states):
        state_values = values[labels == state]
        if len(state_values) == 0:
            means[state] = global_mean
            variances[state] = global_var
        else:
            means[state] = float(np.mean(state_values))
            variances[state] = max(float(np.var(state_values)), min_variance)

    startprob = np.full(n_states, 1e-3, dtype=float)
    startprob[labels[0]] += 1.0
    startprob /= startprob.sum()

    transmat = np.full((n_states, n_states), 1e-3, dtype=float)
    for left, right in zip(labels[:-1], labels[1:]):
        transmat[left, right] += 1.0
    transmat /= transmat.sum(axis=1, keepdims=True)
    return startprob, transmat, means, variances


def _model_variances(model: GaussianHMM, min_variance: float) -> np.ndarray:
    variances = np.asarray(model.covars_, dtype=float).reshape(model.n_components, -1)[:, 0]
    return np.maximum(variances, min_variance)


def _relabel_by_variance(result: GaussianHMMResult) -> GaussianHMMResult:
    order = np.argsort(result.variances, kind="mergesort")
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    states = inverse[result.states]
    transmat = result.transmat[np.ix_(order, order)]
    transmat = transmat / transmat.sum(axis=1, keepdims=True)
    return GaussianHMMResult(
        states=states.astype(int),
        means=result.means[order],
        variances=result.variances[order],
        startprob=result.startprob[order] / result.startprob[order].sum(),
        transmat=transmat,
        loglik=result.loglik,
        n_iter=result.n_iter,
        converged=result.converged,
    )
