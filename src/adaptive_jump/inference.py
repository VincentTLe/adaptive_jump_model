"""Deterministic paired uncertainty calculations for strategy comparisons."""

from __future__ import annotations

import math
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from adaptive_jump.backtest import annualized_excess_sharpe


class InferenceError(ValueError):
    """Raised when an uncertainty calculation violates its frozen contract."""


@dataclass(frozen=True)
class SharpeDeltaBootstrap:
    """Observed Sharpe delta and deterministic stationary-bootstrap bounds."""

    observed: float
    lower_one_sided: float
    confidence_low: float
    confidence_high: float
    replications: int
    mean_block_length: int


@dataclass(frozen=True)
class BootstrapProgress:
    """Completed draw prefix and RNG state at an exact batch boundary."""

    draws: np.ndarray
    rng_state: dict[str, Any]


def stationary_bootstrap_indices(
    observations: int,
    replications: int,
    mean_block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw paired circular stationary-bootstrap indices."""
    if observations < 2 or replications < 1:
        raise InferenceError("bootstrap dimensions must be positive and nontrivial")
    if not 1 <= mean_block_length <= observations:
        raise InferenceError("mean block length must be within the sample")
    indices = np.empty((replications, observations), dtype=np.int64)
    indices[:, 0] = rng.integers(0, observations, size=replications)
    restart_probability = 1.0 / mean_block_length
    for column in range(1, observations):
        restart = rng.random(replications) < restart_probability
        continued = (indices[:, column - 1] + 1) % observations
        fresh = rng.integers(0, observations, size=replications)
        indices[:, column] = np.where(restart, fresh, continued)
    return indices


def bootstrap_sharpe_delta(
    challenger_return: pd.Series,
    baseline_return: pd.Series,
    cash_return: pd.Series,
    *,
    replications: int,
    mean_block_length: int,
    seed: int,
    confidence_level: float = 0.95,
    periods_per_year: int = 252,
    volatility_ddof: int = 1,
    batch_size: int = 500,
    initial: BootstrapProgress | None = None,
    progress: Callable[[BootstrapProgress], None] | None = None,
) -> SharpeDeltaBootstrap:
    """Bootstrap challenger-minus-baseline Sharpe on aligned daily rows."""
    if not (
        challenger_return.index.equals(baseline_return.index)
        and challenger_return.index.equals(cash_return.index)
    ):
        raise InferenceError("bootstrap returns must have identical aligned indices")
    values = np.column_stack([challenger_return, baseline_return, cash_return]).astype(
        float
    )
    if len(values) < 2 or not np.isfinite(values).all():
        raise InferenceError("bootstrap returns must be aligned, finite, and nonempty")
    if replications < 1 or batch_size < 1:
        raise InferenceError("bootstrap replication and batch counts must be positive")
    if not 0 < confidence_level < 1:
        raise InferenceError("confidence level must be between zero and one")
    if periods_per_year < 1 or not 0 <= volatility_ddof < len(values):
        raise InferenceError("Sharpe annualization or ddof is invalid")

    observed = annualized_excess_sharpe(
        pd.Series(values[:, 0]),
        pd.Series(values[:, 2]),
        periods_per_year=periods_per_year,
        volatility_ddof=volatility_ddof,
    ) - annualized_excess_sharpe(
        pd.Series(values[:, 1]),
        pd.Series(values[:, 2]),
        periods_per_year=periods_per_year,
        volatility_ddof=volatility_ddof,
    )
    if not math.isfinite(observed):
        raise InferenceError("observed Sharpe delta is not finite")

    rng = np.random.default_rng(seed)
    draws = np.empty(replications, dtype=float)
    offset = 0
    if initial is not None:
        prefix, rng = _resume_bootstrap(initial, replications, batch_size, seed)
        offset = len(prefix)
        draws[:offset] = prefix
    while offset < replications:
        size = min(batch_size, replications - offset)
        indices = stationary_bootstrap_indices(
            len(values), size, mean_block_length, rng
        )
        sampled = values[indices]
        challenger = _sharpe(
            sampled[:, :, 0], sampled[:, :, 2], periods_per_year, volatility_ddof
        )
        baseline = _sharpe(
            sampled[:, :, 1], sampled[:, :, 2], periods_per_year, volatility_ddof
        )
        draws[offset : offset + size] = challenger - baseline
        offset += size
        if progress is not None:
            progress(
                BootstrapProgress(
                    draws=draws[:offset].copy(),
                    rng_state=deepcopy(rng.bit_generator.state),
                )
            )
    if not np.isfinite(draws).all():
        raise InferenceError("bootstrap produced a non-finite Sharpe delta")

    alpha = 1.0 - confidence_level
    low, high = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return SharpeDeltaBootstrap(
        observed=float(observed),
        lower_one_sided=float(np.quantile(draws, alpha)),
        confidence_low=float(low),
        confidence_high=float(high),
        replications=replications,
        mean_block_length=mean_block_length,
    )


def _resume_bootstrap(
    initial: BootstrapProgress, replications: int, batch_size: int, seed: int
) -> tuple[np.ndarray, np.random.Generator]:
    try:
        prefix = np.asarray(initial.draws, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InferenceError("bootstrap checkpoint draws are invalid") from exc
    if prefix.ndim != 1 or not 1 <= prefix.size <= replications:
        raise InferenceError("bootstrap checkpoint draw count is invalid")
    if prefix.size < replications and prefix.size % batch_size:
        raise InferenceError("bootstrap checkpoint is not at a batch boundary")
    if not np.isfinite(prefix).all():
        raise InferenceError("bootstrap checkpoint draws are not finite")
    if not isinstance(initial.rng_state, dict):
        raise InferenceError("bootstrap checkpoint RNG state is invalid")
    rng = np.random.default_rng(seed)
    try:
        rng.bit_generator.state = deepcopy(initial.rng_state)
    except (KeyError, TypeError, ValueError) as exc:
        raise InferenceError("bootstrap checkpoint RNG state is invalid") from exc
    return prefix.copy(), rng


def _sharpe(
    strategy: np.ndarray,
    cash: np.ndarray,
    periods_per_year: int,
    volatility_ddof: int,
) -> np.ndarray:
    volatility = strategy.std(axis=1, ddof=volatility_ddof)
    return math.sqrt(periods_per_year) * (strategy - cash).mean(axis=1) / volatility
