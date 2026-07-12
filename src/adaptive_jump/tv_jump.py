"""Time-varying-penalty extension of the statistical jump model.

Extends ``jumpmodels.jump.JumpModel`` (Shu/Kolm/Mulvey reference library) with a
per-period jump penalty: the transition cost between t-1 and t is ``lam_seq[t]``
instead of a single scalar. With a constant sequence the model reproduces the
original JM exactly (oracle-tested), so the fixed-penalty baseline is nested.

Only the discrete model (``cont=False``) is supported. The DP stays exact and
O(T * K^2); ``penalty_seq[t]`` simply replaces the constant penalty matrix at
step t. ``penalty_seq[0]`` is never read (no incoming transition at t=0).

Design note: the penalty sequence is an argument of ``fit_tv`` /
``predict_online_tv`` (not of ``__init__``), because in the walk-forward
protocol the training window and each evaluation window carry their own
aligned ``lam_seq``. The DP core is deliberately a standalone function so a
batched (GPU) backend can replace it later without touching the estimator.
"""

from __future__ import annotations

import numpy as np
from jumpmodels.jump import (
    JumpModel,
    empirical_trans_mx,
    is_same_clustering,
    raise_JM_labels_to_proba,
    raise_JM_proba_to_df,
    reduce_proba_to_labels,
    replace_nan_by_inf,
    sort_states_from_ret,
    weighted_mean_cluster,
)
from scipy.spatial.distance import cdist


def lam_to_penalty_seq(lam_seq: np.ndarray, n_c: int) -> np.ndarray:
    """Expand a per-period scalar penalty ``lam_seq`` (T,) into (T, n_c, n_c)
    matrices with ``lam_seq[t]`` off-diagonal and 0 on the diagonal."""
    lam = np.asarray(lam_seq, dtype=float)
    if lam.ndim != 1:
        raise ValueError(f"lam_seq must be 1-d, got shape {lam.shape}")
    if not np.isfinite(lam).all():
        raise ValueError("lam_seq must be finite")
    if (lam < 0).any():
        raise ValueError("lam_seq must be nonnegative")
    off = 1.0 - np.eye(n_c)
    return lam[:, None, None] * off[None, :, :]


def dp_tv(loss_mx: np.ndarray, penalty_seq: np.ndarray, return_value_mx: bool = False):
    """Exact DP for a time-varying penalty.

    Minimizes ``sum_t L(t, s_t) + sum_{t>=1} penalty_seq[t][s_{t-1}, s_t]``.
    Mirrors ``jumpmodels.jump.dp`` with ``penalty_seq[t]`` in place of the
    constant matrix. Returns (assignments, optimal value), or the DP value
    matrix if ``return_value_mx`` (row t uses data up to t only -> online).
    """
    n_s, n_c = loss_mx.shape
    if penalty_seq.shape != (n_s, n_c, n_c):
        raise ValueError(f"penalty_seq shape {penalty_seq.shape} != {(n_s, n_c, n_c)}")
    loss_mx = replace_nan_by_inf(loss_mx)
    values = np.empty((n_s, n_c))
    values[0] = loss_mx[0]
    for t in range(1, n_s):
        values[t] = loss_mx[t] + (values[t - 1][:, None] + penalty_seq[t]).min(axis=0)
    if return_value_mx:
        return values
    assign = np.empty(n_s, dtype=int)
    assign[-1] = values[-1].argmin()
    value_opt = values[-1, assign[-1]]
    for t in range(n_s - 1, 0, -1):
        assign[t - 1] = (values[t - 1] + penalty_seq[t][:, assign[t]]).argmin()
    return assign, value_opt


def _do_E_step_tv(
    X: np.ndarray,
    centers_: np.ndarray,
    penalty_seq: np.ndarray,
    return_value_mx: bool = False,
):
    """E-step (loss matrix + DP) for the discrete time-varying model."""
    loss_mx = 0.5 * cdist(X, centers_, "sqeuclidean")
    if return_value_mx:
        return dp_tv(loss_mx, penalty_seq, return_value_mx=True)
    labels_, val_ = dp_tv(loss_mx, penalty_seq)
    proba_ = raise_JM_labels_to_proba(labels_, len(centers_), None)
    return proba_, labels_, val_


class TVJumpModel(JumpModel):
    """Discrete jump model with a per-period jump penalty sequence.

    ``jump_penalty`` from the parent is ignored by the ``*_tv`` methods; the
    penalty enters through ``lam_seq`` (per-period scalar, expanded to
    matrices) or ``penalty_seq`` (full (T, K, K), for asymmetric costs later).
    """

    def __init__(
        self,
        n_components: int = 2,
        random_state=0,
        max_iter: int = 1000,
        tol: float = 1e-8,
        n_init: int = 10,
    ):
        super().__init__(
            n_components=n_components,
            jump_penalty=0.0,
            cont=False,
            random_state=random_state,
            max_iter=max_iter,
            tol=tol,
            n_init=n_init,
        )

    # ---- helpers -------------------------------------------------------

    def _resolve_penalty_seq(self, n_s: int, lam_seq, penalty_seq) -> np.ndarray:
        if (lam_seq is None) == (penalty_seq is None):
            raise ValueError("pass exactly one of lam_seq / penalty_seq")
        if lam_seq is not None:
            penalty_seq = lam_to_penalty_seq(np.asarray(lam_seq), self.n_components)
        penalty_seq = np.asarray(penalty_seq, dtype=float)
        if penalty_seq.shape != (n_s, self.n_components, self.n_components):
            raise ValueError(
                f"penalty length {penalty_seq.shape} does not match n_samples {n_s}"
            )
        return penalty_seq

    # ---- estimation ----------------------------------------------------

    def fit_tv(
        self, X, ret_ser=None, lam_seq=None, penalty_seq=None, sort_by: str = "cumret"
    ):
        """Coordinate-descent fit with a time-varying penalty.

        Mirrors ``JumpModel.fit`` (same inits, convergence rule, state
        sorting) with the constant penalty matrix replaced by
        ``penalty_seq[t]`` in every E-step.
        """
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be 2-d")
        penalty_seq = self._resolve_penalty_seq(len(X_arr), lam_seq, penalty_seq)
        self.prob_vecs = None  # discrete model only
        self.feat_weights = None
        init_centers_values = self.init_centers(X_arr)
        best_val, best_res = np.inf, {"labels_": None}
        for centers_ in init_centers_values:
            labels_pre, val_pre = None, np.inf
            proba_, labels_, val_ = _do_E_step_tv(X_arr, centers_, penalty_seq)
            num_iter = 0
            while (
                num_iter < self.max_iter
                and not is_same_clustering(labels_, labels_pre)
                and val_pre - val_ > self.tol
            ):
                num_iter += 1
                labels_pre, val_pre = labels_, val_
                centers_ = weighted_mean_cluster(X_arr, proba_)
                proba_, labels_, val_ = _do_E_step_tv(X_arr, centers_, penalty_seq)
            if not is_same_clustering(best_res["labels_"], labels_) and val_ < best_val:
                best_val = val_
                best_res = {"centers_": centers_, "labels_": labels_, "proba_": proba_}
        self.val_ = best_val
        sort_states_from_ret(ret_ser, X, best_res, sort_by=sort_by)
        if ret_ser is not None:
            self.ret_ = best_res["ret_"]
            self.vol_ = best_res["vol_"]
        self.centers_ = best_res["centers_"]
        self.proba_ = raise_JM_proba_to_df(best_res["proba_"], X)
        self.labels_ = reduce_proba_to_labels(self.proba_)
        self.transmat_ = empirical_trans_mx(
            self.labels_, n_components=self.n_components
        )
        return self

    # ---- inference -----------------------------------------------------

    def predict_online_tv(self, X, lam_seq=None, penalty_seq=None):
        """Online state labels: row t uses data up to t only."""
        X_arr = self.check_X_predict_func(X)
        penalty_seq = self._resolve_penalty_seq(len(X_arr), lam_seq, penalty_seq)
        value_mx = _do_E_step_tv(
            X_arr, self.centers_, penalty_seq, return_value_mx=True
        )
        labels_ = value_mx.argmin(axis=1)
        proba_ = raise_JM_labels_to_proba(labels_, self.n_components, None)
        return reduce_proba_to_labels(raise_JM_proba_to_df(proba_, X))

    def predict_tv(self, X, lam_seq=None, penalty_seq=None):
        """Full-window (offline) decoding with a time-varying penalty."""
        X_arr = self.check_X_predict_func(X)
        penalty_seq = self._resolve_penalty_seq(len(X_arr), lam_seq, penalty_seq)
        _, labels_, _ = _do_E_step_tv(X_arr, self.centers_, penalty_seq)
        proba_ = raise_JM_labels_to_proba(labels_, self.n_components, None)
        return reduce_proba_to_labels(raise_JM_proba_to_df(proba_, X))
