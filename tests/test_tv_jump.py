"""Oracle tests for the time-varying-penalty jump model extension."""

from itertools import product

import numpy as np
import pytest
from jumpmodels.jump import JumpModel, dp, jump_penalty_to_mx

from adaptive_jump.tv_jump import TVJumpModel, dp_tv, lam_to_penalty_seq

RNG = np.random.default_rng(7)


def brute_force_tv(loss_mx: np.ndarray, penalty_seq: np.ndarray) -> float:
    """Enumerate all K^T paths; return the optimal objective value."""
    n_s, n_c = loss_mx.shape
    best = np.inf
    for path in product(range(n_c), repeat=n_s):
        val = sum(loss_mx[t, path[t]] for t in range(n_s))
        val += sum(penalty_seq[t][path[t - 1], path[t]] for t in range(1, n_s))
        best = min(best, val)
    return best


def path_cost_tv(loss_mx, penalty_seq, path) -> float:
    val = sum(loss_mx[t, path[t]] for t in range(len(path)))
    val += sum(penalty_seq[t][path[t - 1], path[t]] for t in range(1, len(path)))
    return float(val)


# ---------- dp_tv correctness ----------

@pytest.mark.parametrize("n_s,n_c", [(5, 2), (7, 2), (6, 3)])
def test_dp_tv_matches_brute_force(n_s, n_c):
    for _ in range(20):
        loss = RNG.uniform(0, 5, size=(n_s, n_c))
        lam = RNG.uniform(0, 3, size=n_s)
        pen = lam_to_penalty_seq(lam, n_c)
        assign, val = dp_tv(loss, pen)
        assert val == pytest.approx(brute_force_tv(loss, pen))
        # the returned path must itself achieve the optimal value
        assert path_cost_tv(loss, pen, assign) == pytest.approx(val)


def test_dp_tv_asymmetric_matrix_matches_brute_force():
    n_s, n_c = 6, 2
    for _ in range(20):
        loss = RNG.uniform(0, 5, size=(n_s, n_c))
        pen = RNG.uniform(0, 3, size=(n_s, n_c, n_c))
        pen[:, np.arange(n_c), np.arange(n_c)] = 0.0     # zero diagonal
        assign, val = dp_tv(loss, pen)
        assert val == pytest.approx(brute_force_tv(loss, pen))
        assert path_cost_tv(loss, pen, assign) == pytest.approx(val)


def test_dp_tv_constant_lambda_equals_reference_dp():
    """With a constant sequence, dp_tv must reproduce jumpmodels.dp exactly."""
    for lam in [0.0, 0.7, 5.0, 50.0]:
        loss = RNG.uniform(0, 5, size=(40, 2))
        pen_mx = jump_penalty_to_mx(lam, 2)
        ref_assign, ref_val = dp(loss, pen_mx)
        tv_assign, tv_val = dp_tv(loss, lam_to_penalty_seq(np.full(40, lam), 2))
        assert tv_val == pytest.approx(ref_val)
        assert np.array_equal(tv_assign, ref_assign)
        # online value matrices agree too
        ref_vals = dp(loss, pen_mx, return_value_mx=True)
        tv_vals = dp_tv(loss, lam_to_penalty_seq(np.full(40, lam), 2),
                        return_value_mx=True)
        assert np.allclose(ref_vals, tv_vals)


def test_dp_tv_zero_lambda_is_pointwise_argmin():
    loss = RNG.uniform(0, 5, size=(30, 2))
    assign, val = dp_tv(loss, lam_to_penalty_seq(np.zeros(30), 2))
    assert np.array_equal(assign, loss.argmin(axis=1))
    assert val == pytest.approx(loss.min(axis=1).sum())


def test_dp_tv_huge_lambda_forbids_switching():
    loss = RNG.uniform(0, 5, size=(30, 2))
    assign, _ = dp_tv(loss, lam_to_penalty_seq(np.full(30, 1e9), 2))
    assert len(np.unique(assign)) == 1


# ---------- input validation ----------

def test_lam_to_penalty_seq_rejects_bad_input():
    with pytest.raises(ValueError):
        lam_to_penalty_seq(np.array([[1.0]]), 2)          # not 1-d
    with pytest.raises(ValueError):
        lam_to_penalty_seq(np.array([1.0, -0.1]), 2)      # negative
    with pytest.raises(ValueError):
        lam_to_penalty_seq(np.array([1.0, np.inf]), 2)    # non-finite


def test_tv_model_rejects_wrong_length_and_double_spec():
    X = RNG.normal(size=(50, 3))
    m = TVJumpModel(n_init=2)
    with pytest.raises(ValueError):
        m.fit_tv(X, lam_seq=np.ones(49))                  # length mismatch
    with pytest.raises(ValueError):
        m.fit_tv(X)                                       # neither given
    with pytest.raises(ValueError):
        m.fit_tv(X, lam_seq=np.ones(50),
                 penalty_seq=lam_to_penalty_seq(np.ones(50), 2))  # both given


# ---------- TVJumpModel nests JumpModel ----------

def _regime_data(n=400, seed=0):
    """Two-state synthetic features + returns with persistent blocks."""
    rng = np.random.default_rng(seed)
    state = np.zeros(n, dtype=int)
    for t in range(1, n):
        state[t] = state[t - 1] if rng.uniform() > 0.02 else 1 - state[t - 1]
    mu = np.where(state == 0, 0.5, -0.5)
    X = mu[:, None] + rng.normal(scale=1.0, size=(n, 3))
    ret = np.where(state == 0, 0.001, -0.001) + rng.normal(scale=0.01, size=n)
    return X, ret


def test_tv_model_with_constant_lambda_equals_jump_model():
    X, ret = _regime_data()
    for lam in [5.0, 50.0]:
        ref = JumpModel(n_components=2, jump_penalty=lam, cont=False,
                        n_init=4, random_state=3).fit(X, ret, sort_by="cumret")
        tv = TVJumpModel(n_components=2, n_init=4, random_state=3)
        tv.fit_tv(X, ret, lam_seq=np.full(len(X), lam), sort_by="cumret")
        assert np.array_equal(np.asarray(tv.labels_), np.asarray(ref.labels_))
        assert tv.val_ == pytest.approx(ref.val_)
        # online inference agrees as well
        Xo, _ = _regime_data(n=120, seed=9)
        ref_lab = np.asarray(ref.predict_online(Xo))
        tv_lab = np.asarray(tv.predict_online_tv(Xo, lam_seq=np.full(120, lam)))
        assert np.array_equal(ref_lab, tv_lab)


def test_tv_lambda_actually_changes_the_path():
    """A penalty dropped to ~0 in a window must allow extra switches there."""
    X, ret = _regime_data(seed=4)
    tv = TVJumpModel(n_components=2, n_init=4, random_state=3)
    tv.fit_tv(X, ret, lam_seq=np.full(len(X), 50.0), sort_by="cumret")
    Xo, _ = _regime_data(n=200, seed=11)
    lab_const = np.asarray(tv.predict_online_tv(Xo, lam_seq=np.full(200, 50.0)))
    lam_var = np.full(200, 50.0)
    lam_var[100:140] = 0.0
    lab_var = np.asarray(tv.predict_online_tv(Xo, lam_seq=lam_var))
    n_sw = lambda a: int((np.diff(a) != 0).sum())  # noqa: E731
    assert n_sw(lab_var) >= n_sw(lab_const)
    assert not np.array_equal(lab_const, lab_var)
