"""Return-aware discrete Jump Model for the frozen simple-JM experiment.

The challenger adds one standardized, matured ``t+2`` excess-return coordinate
to the in-sample state loss.  Online decoding deliberately uses features only.
The experiment's ``gamma=0`` control is the canonical fixed-JM pipeline, so this
estimator refuses to refit that control through a second implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from jumpmodels.jump import (
    JumpModel,
    dp,
    empirical_trans_mx,
    init_centers_kmeans_plusplus,
    is_same_clustering,
    jump_penalty_to_mx,
    raise_JM_labels_to_proba,
    raise_JM_proba_to_df,
    reduce_proba_to_labels,
    weighted_mean_cluster,
)
from scipy.spatial.distance import cdist


class ReturnAwareError(ValueError):
    """Raised when a return-aware fit would violate the frozen contract."""


@dataclass(frozen=True)
class MaturedTargetAlignment:
    """Raw ``t+2`` targets aligned to feature rows at one fit cutoff."""

    feature_dates: pd.DatetimeIndex
    target_dates: pd.DatetimeIndex
    values: np.ndarray
    matured_mask: np.ndarray


@dataclass(frozen=True)
class StandardizedTargets:
    """Past-only population standardization of matured targets."""

    values: np.ndarray
    matured_mask: np.ndarray
    mean: float
    scale: float


def align_matured_targets(
    return_dates,
    excess_returns,
    feature_dates,
    cutoff,
    *,
    offset: int = 2,
) -> MaturedTargetAlignment:
    """Align feature rows to observed forward returns on the full calendar.

    The return calendar is first truncated at ``cutoff``.  Consequently neither
    a future value nor even a future calendar row can affect the returned mask.
    ``offset=2`` matches the project's signal-at-t to return-at-t+2 protocol.
    """
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 1:
        raise ReturnAwareError("target offset must be a positive integer")
    calendar = pd.DatetimeIndex(pd.to_datetime(return_dates, errors="raise"))
    features = pd.DatetimeIndex(pd.to_datetime(feature_dates, errors="raise"))
    values = np.asarray(excess_returns, dtype=float)
    cutoff = pd.Timestamp(cutoff)
    if len(calendar) != len(values):
        raise ReturnAwareError("return dates and excess returns must have equal length")
    if calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ReturnAwareError("return calendar must be increasing and unique")
    if features.has_duplicates or not features.is_monotonic_increasing:
        raise ReturnAwareError("feature dates must be increasing and unique")
    if pd.isna(cutoff):
        raise ReturnAwareError("cutoff must be a valid date")
    if len(features) and features[-1] > cutoff:
        raise ReturnAwareError("feature dates must not exceed the fit cutoff")

    observed = calendar <= cutoff
    past_calendar = calendar[observed]
    past_values = values[observed]
    positions = past_calendar.get_indexer(features)
    if (positions < 0).any():
        raise ReturnAwareError("every feature date must occur on the return calendar")

    target_dates = np.full(len(features), np.datetime64("NaT"), dtype="datetime64[ns]")
    targets = np.full(len(features), np.nan, dtype=float)
    matured = np.zeros(len(features), dtype=bool)
    target_positions = positions + offset
    available = target_positions < len(past_calendar)
    if available.any():
        destination = target_positions[available]
        candidate_values = past_values[destination]
        finite = np.isfinite(candidate_values)
        rows = np.flatnonzero(available)[finite]
        target_dates[rows] = past_calendar[destination[finite]].to_numpy()
        targets[rows] = candidate_values[finite]
        matured[rows] = True
    return MaturedTargetAlignment(
        feature_dates=features,
        target_dates=pd.DatetimeIndex(target_dates),
        values=targets,
        matured_mask=matured,
    )


def standardize_matured_targets(target, matured_mask) -> StandardizedTargets:
    """Use the mean and population standard deviation of matured targets only."""
    values = np.asarray(target, dtype=float)
    mask = np.asarray(matured_mask)
    if values.ndim != 1 or mask.ndim != 1 or values.shape != mask.shape:
        raise ReturnAwareError("target and matured_mask must be equal-length vectors")
    if mask.dtype.kind != "b":
        raise ReturnAwareError("matured_mask must be boolean")
    matured = values[mask]
    if len(matured) < 2 or not np.isfinite(matured).all():
        raise ReturnAwareError("at least two finite matured targets are required")
    mean = float(matured.mean())
    scale = float(matured.std(ddof=0))
    if not np.isfinite(scale) or scale <= 0:
        raise ReturnAwareError("matured target scale must be finite and positive")
    standardized = np.zeros_like(values, dtype=float)
    standardized[mask] = (matured - mean) / scale
    return StandardizedTargets(standardized, mask.copy(), mean, scale)


def feature_loss_matrix(X, centers) -> np.ndarray:
    """Return ``0.5`` times squared Euclidean feature distance."""
    values = np.asarray(X, dtype=float)
    means = np.asarray(centers, dtype=float)
    if values.ndim != 2 or means.ndim != 2 or values.shape[1] != means.shape[1]:
        raise ReturnAwareError(
            "X and centers must be compatible two-dimensional arrays"
        )
    if len(values) == 0 or len(means) < 2 or not np.isfinite(values).all():
        raise ReturnAwareError(
            "X must be finite and states must contain at least two centers"
        )
    return 0.5 * cdist(values, means, metric="sqeuclidean")


def return_aware_loss_matrix(
    X,
    centers,
    standardized_target,
    target_means,
    matured_mask,
    *,
    gamma: float,
) -> np.ndarray:
    """Build feature plus masked standardized-target loss for each state."""
    if gamma not in (0, 0.0, 1, 1.0):
        raise ReturnAwareError("gamma must be exactly 0 or 1")
    feature_loss = feature_loss_matrix(X, centers)
    target = np.asarray(standardized_target, dtype=float)
    target_means = np.asarray(target_means, dtype=float)
    mask = np.asarray(matured_mask)
    if target.ndim != 1 or len(target) != len(feature_loss):
        raise ReturnAwareError("standardized_target must align with X")
    if target_means.shape != (feature_loss.shape[1],):
        raise ReturnAwareError("target_means must contain one value per state")
    if mask.shape != target.shape or mask.dtype.kind != "b":
        raise ReturnAwareError("matured_mask must be a boolean vector aligned with X")
    if not np.isfinite(target[mask]).all():
        raise ReturnAwareError("matured standardized targets must be finite")
    if float(gamma) == 0.0:
        return feature_loss
    target_loss = np.zeros_like(feature_loss)
    target_loss[mask] = 0.5 * (target[mask, None] - target_means[None, :]) ** 2
    return feature_loss + target_loss


def dp_return_aware(
    X,
    centers,
    standardized_target,
    target_means,
    matured_mask,
    *,
    gamma: float,
    jump_penalty: float,
    return_value_mx: bool = False,
):
    """Solve the exact discrete path for fixed return-aware parameters."""
    if not np.isfinite(jump_penalty) or jump_penalty < 0:
        raise ReturnAwareError("jump_penalty must be finite and nonnegative")
    loss = return_aware_loss_matrix(
        X,
        centers,
        standardized_target,
        target_means,
        matured_mask,
        gamma=gamma,
    )
    penalty = jump_penalty_to_mx(float(jump_penalty), loss.shape[1])
    return dp(loss, penalty, return_value_mx=return_value_mx)


def _m_step(
    X: np.ndarray,
    target: np.ndarray,
    matured_mask: np.ndarray,
    proba: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    centers = weighted_mean_cluster(X, proba)
    target_means = np.full(proba.shape[1], np.nan, dtype=float)
    matured_weights = proba[matured_mask]
    counts = matured_weights.sum(axis=0)
    used = counts > 0
    if used.any():
        target_means[used] = (
            matured_weights[:, used].T @ target[matured_mask]
        ) / counts[used]
    # A state containing only the final, unlabeled rows has no identified target
    # location.  Zero is the global standardized mean and a deterministic
    # minimizer of its empty target term.
    active = proba.sum(axis=0) > 0
    target_means[active & ~used] = 0.0
    return centers, target_means


def _e_step(
    X: np.ndarray,
    centers: np.ndarray,
    target: np.ndarray,
    target_means: np.ndarray,
    matured_mask: np.ndarray,
    gamma: float,
    penalty: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    loss = return_aware_loss_matrix(
        X,
        centers,
        target,
        target_means,
        matured_mask,
        gamma=gamma,
    )
    labels, value = dp(loss, penalty)
    proba = raise_JM_labels_to_proba(labels, len(centers), None)
    return proba, labels, float(value)


class ReturnAwareJumpModel(JumpModel):
    """Two-state squared-loss JM trained with matured ``t+2`` returns."""

    def __init__(
        self,
        *,
        jump_penalty: float,
        gamma: float = 1.0,
        random_state=0,
        max_iter: int = 1000,
        tol: float = 1e-8,
        n_init: int = 10,
    ):
        if gamma not in (0, 0.0, 1, 1.0):
            raise ReturnAwareError("gamma must be exactly 0 or 1")
        super().__init__(
            n_components=2,
            jump_penalty=jump_penalty,
            cont=False,
            random_state=random_state,
            max_iter=max_iter,
            tol=tol,
            n_init=n_init,
        )
        self.gamma = float(gamma)

    def fit(self, X, target, matured_mask):
        """Fit gamma one; gamma zero belongs to the canonical fixed-JM path."""
        if self.gamma == 0.0:
            raise ReturnAwareError(
                "gamma=0 must route through the canonical fixed-JM control"
            )
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or len(values) == 0 or not np.isfinite(values).all():
            raise ReturnAwareError("X must be a finite non-empty two-dimensional array")
        standardized = standardize_matured_targets(target, matured_mask)
        if len(standardized.values) != len(values):
            raise ReturnAwareError("target must align with X")
        if (
            not np.isfinite(self.jump_penalty)
            or float(self.jump_penalty) < 0
            or self.n_init < 1
            or self.max_iter < 1
            or not np.isfinite(self.tol)
            or self.tol < 0
        ):
            raise ReturnAwareError("invalid fit parameters")

        self.feat_weights = None
        self.prob_vecs = None
        penalty = jump_penalty_to_mx(float(self.jump_penalty), self.n_components)
        self.jump_penalty_mx = penalty
        initial_centers = init_centers_kmeans_plusplus(
            values, self.n_components, self.n_init, self.random_state
        )
        best_value = np.inf
        best = None
        histories: list[tuple[float, ...]] = []
        for init_index, centers in enumerate(initial_centers):
            feature_labels, _ = dp(feature_loss_matrix(values, centers), penalty)
            proba = raise_JM_labels_to_proba(feature_labels, self.n_components, None)
            centers, target_means = _m_step(
                values,
                standardized.values,
                standardized.matured_mask,
                proba,
            )
            proba, labels, value = _e_step(
                values,
                centers,
                standardized.values,
                target_means,
                standardized.matured_mask,
                self.gamma,
                penalty,
            )
            history = [value]
            previous_labels = None
            previous_value = np.inf
            iterations = 0
            while (
                iterations < self.max_iter
                and not is_same_clustering(labels, previous_labels)
                and previous_value - value > self.tol
            ):
                previous_labels = labels
                previous_value = value
                centers, target_means = _m_step(
                    values,
                    standardized.values,
                    standardized.matured_mask,
                    proba,
                )
                proba, labels, value = _e_step(
                    values,
                    centers,
                    standardized.values,
                    target_means,
                    standardized.matured_mask,
                    self.gamma,
                    penalty,
                )
                history.append(value)
                iterations += 1
            histories.append(tuple(history))
            if np.isfinite(value) and value < best_value:
                best_value = value
                best = {
                    "centers": centers,
                    "target_means": target_means,
                    "proba": proba,
                    "labels": labels,
                    "iterations": iterations,
                    "init_index": init_index,
                    "history": tuple(history),
                }
        if best is None:
            raise ReturnAwareError("all deterministic initializations failed")

        target_means = np.asarray(best["target_means"])
        if np.isfinite(target_means).all() and target_means[0] == target_means[1]:
            raise ReturnAwareError("fitted target means are exactly tied")
        criterion = np.where(np.isfinite(target_means), -target_means, np.inf)
        order = np.argsort(criterion, kind="stable")
        centers = np.asarray(best["centers"])[order]
        target_means = target_means[order]
        proba = np.asarray(best["proba"])[:, order]

        self.val_ = float(best_value)
        self.centers_ = centers
        self.proba_ = raise_JM_proba_to_df(proba, X)
        self.labels_ = reduce_proba_to_labels(self.proba_)
        self.transmat_ = empirical_trans_mx(
            self.labels_, n_components=self.n_components
        )
        self.target_standardizer_mean_ = standardized.mean
        self.target_standardizer_scale_ = standardized.scale
        self.target_means_standardized_ = target_means
        self.target_means_ = standardized.mean + standardized.scale * target_means
        self.matured_target_count_ = int(standardized.matured_mask.sum())
        self.matured_state_counts_ = (
            np.asarray(proba)[standardized.matured_mask].sum(axis=0).astype(int)
        )
        self.n_iter_ = int(best["iterations"])
        self.best_init_ = int(best["init_index"])
        self.objective_history_ = best["history"]
        self.all_objective_histories_ = tuple(histories)

        labels = np.asarray(self.labels_, dtype=int)
        feature_loss = feature_loss_matrix(values, self.centers_)
        self.feature_value_ = float(feature_loss[np.arange(len(values)), labels].sum())
        target_rows = np.flatnonzero(standardized.matured_mask)
        residual = (
            standardized.values[target_rows]
            - self.target_means_standardized_[labels[target_rows]]
        )
        self.target_value_ = float(0.5 * np.square(residual).sum())
        self.transition_value_ = float(
            self.jump_penalty * np.count_nonzero(np.diff(labels))
        )
        recomputed = (
            self.feature_value_
            + self.gamma * self.target_value_
            + self.transition_value_
        )
        if not np.isclose(recomputed, self.val_, rtol=1e-12, atol=1e-12):
            raise ReturnAwareError("stored objective components do not match fit value")
        return self
