"""Two-state fixed-penalty jump model with raw cityblock feature loss."""

from __future__ import annotations

import math

import numpy as np
from jumpmodels.jump import (
    JumpModel,
    dp,
    empirical_trans_mx,
    is_same_clustering,
    jump_penalty_to_mx,
    raise_JM_labels_to_proba,
    raise_JM_proba_to_df,
    reduce_proba_to_labels,
    sort_states_from_ret,
)
from scipy.spatial.distance import cdist


def l1_loss_matrix(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Return raw cityblock loss for every observation-state pair."""
    values = np.asarray(X, dtype=float)
    locations = np.asarray(centers, dtype=float)
    if values.ndim != 2 or not len(values) or not values.shape[1]:
        raise ValueError("X must be a non-empty 2-d matrix")
    if locations.ndim != 2 or locations.shape[1] != values.shape[1]:
        raise ValueError("centers must be a 2-d matrix matching X features")
    if not np.isfinite(values).all():
        raise ValueError("X must be finite")
    valid_center = np.isfinite(locations).all(axis=1) | np.isnan(locations).all(axis=1)
    if not valid_center.all():
        raise ValueError("each center must be entirely finite or unavailable")
    return cdist(values, locations, "cityblock")


def componentwise_median_centers(
    X: np.ndarray, labels: np.ndarray, n_components: int = 2
) -> np.ndarray:
    """Minimize hard-cluster L1 loss; empty-state centers remain unavailable."""
    values = np.asarray(X, dtype=float)
    assignments = np.asarray(labels)
    if values.ndim != 2 or not len(values) or not values.shape[1]:
        raise ValueError("X must be a non-empty 2-d matrix")
    if not np.isfinite(values).all():
        raise ValueError("X must be finite")
    if assignments.shape != (len(values),):
        raise ValueError("labels must contain one assignment per observation")
    if not isinstance(n_components, int) or isinstance(n_components, bool):
        raise ValueError("n_components must be a positive integer")
    if n_components < 1 or not np.isin(assignments, np.arange(n_components)).all():
        raise ValueError("labels must identify a configured component")

    centers = np.full((n_components, values.shape[1]), np.nan)
    for component in range(n_components):
        members = values[assignments == component]
        if len(members):
            centers[component] = np.median(members, axis=0)
    return centers


def solve_l1_path(
    X: np.ndarray,
    centers: np.ndarray,
    jump_penalty: float,
    *,
    return_value_mx: bool = False,
):
    """Solve the exact fixed-cost path for frozen L1 centers."""
    try:
        penalty = float(jump_penalty)
    except (TypeError, ValueError) as exc:
        raise ValueError("jump_penalty must be a real scalar") from exc
    if not math.isfinite(penalty) or penalty < 0:
        raise ValueError("jump_penalty must be finite and nonnegative")
    loss = l1_loss_matrix(X, centers)
    penalty_mx = jump_penalty_to_mx(penalty, len(centers))
    return dp(loss, penalty_mx, return_value_mx=return_value_mx)


class L1JumpModel(JumpModel):
    """Discrete two-state JM with L1 loss and median center updates."""

    def __init__(
        self,
        n_components: int = 2,
        jump_penalty: float = 0.0,
        random_state=0,
        max_iter: int = 1000,
        tol: float = 1e-8,
        n_init: int = 10,
        verbose: int = 0,
    ) -> None:
        if n_components != 2:
            raise ValueError("L1JumpModel supports exactly two components")
        super().__init__(
            n_components=n_components,
            jump_penalty=jump_penalty,
            cont=False,
            random_state=random_state,
            max_iter=max_iter,
            tol=tol,
            n_init=n_init,
            verbose=verbose,
        )

    def fit(self, X, ret_ser, sort_by: str = "cumret"):
        """Fit by deterministic multi-start coordinate descent."""
        if sort_by != "cumret":
            raise ValueError("L1JumpModel states must be sorted by cumulative return")
        values = np.asarray(X, dtype=float)
        if (
            values.ndim != 2
            or len(values) < self.n_components
            or not values.shape[1]
            or not np.isfinite(values).all()
        ):
            raise ValueError("X must be a finite 2-d matrix with at least two rows")
        try:
            penalty = float(self.jump_penalty)
        except (TypeError, ValueError) as exc:
            raise ValueError("jump_penalty must be a real scalar") from exc
        if not math.isfinite(penalty) or penalty < 0:
            raise ValueError("jump_penalty must be finite and nonnegative")

        self.prob_vecs = None
        self.feat_weights = None
        self.jump_penalty_mx = jump_penalty_to_mx(penalty, self.n_components)
        best_value = math.inf
        best: dict[str, np.ndarray | None] = {"labels_": None}
        for initial_centers in self.init_centers(values):
            centers = initial_centers
            previous_labels = None
            previous_value = math.inf
            labels, value = self._e_step(values, centers)
            iterations = 0
            while (
                iterations < self.max_iter
                and not is_same_clustering(labels, previous_labels)
                and previous_value - value > self.tol
            ):
                iterations += 1
                previous_labels, previous_value = labels, value
                centers = componentwise_median_centers(
                    values, labels, self.n_components
                )
                labels, value = self._e_step(values, centers)
            if not is_same_clustering(best["labels_"], labels) and value < best_value:
                best_value = float(value)
                best = {
                    "centers_": centers,
                    "labels_": labels,
                    "proba_": raise_JM_labels_to_proba(labels, self.n_components, None),
                }

        sort_states_from_ret(ret_ser, X, best, sort_by="cumret")
        self.val_ = best_value
        self.ret_ = best["ret_"]
        self.vol_ = best["vol_"]
        self.centers_ = best["centers_"]
        self.proba_ = raise_JM_proba_to_df(best["proba_"], X)
        self.labels_ = reduce_proba_to_labels(self.proba_)
        self.transmat_ = empirical_trans_mx(
            self.labels_, n_components=self.n_components
        )
        return self

    def _e_step(self, X: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, float]:
        labels, value = dp(l1_loss_matrix(X, centers), self.jump_penalty_mx)
        return np.asarray(labels, dtype=int), float(value)

    def predict_proba_online(self, X):
        """Return states whose row t uses only observations through t."""
        values = self.check_X_predict_func(X)
        value_mx = solve_l1_path(
            values,
            self.centers_,
            self.jump_penalty,
            return_value_mx=True,
        )
        labels = np.asarray(value_mx).argmin(axis=1)
        proba = raise_JM_labels_to_proba(labels, self.n_components, None)
        return raise_JM_proba_to_df(proba, X)

    def predict_online(self, X):
        """Return the exact online-DP label sequence."""
        return reduce_proba_to_labels(self.predict_proba_online(X))

    def predict_proba(self, X):
        """Return probabilities for the exact full-window optimal path."""
        values = self.check_X_predict_func(X)
        labels, _ = solve_l1_path(values, self.centers_, self.jump_penalty)
        proba = raise_JM_labels_to_proba(labels, self.n_components, None)
        return raise_JM_proba_to_df(proba, X)

    def predict(self, X):
        """Return the exact full-window optimal path."""
        return reduce_proba_to_labels(self.predict_proba(X))
