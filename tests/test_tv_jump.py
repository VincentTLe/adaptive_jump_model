"""Oracle tests for the time-varying-penalty jump model extension."""

from itertools import product

import numpy as np
import pytest
from jumpmodels.jump import JumpModel, dp, jump_penalty_to_mx

from adaptive_jump.tv_jump import (
    TVJumpModel,
    dp_tv,
    evidence_penalty_seq,
    lam_to_penalty_seq,
    loss_matrix,
    robust_loss_scale,
)

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


def test_loss_matrix_uses_half_squared_euclidean_distance():
    X = np.array([[0.0, 0.0], [2.0, 1.0]])
    centers = np.array([[0.0, 0.0], [2.0, 0.0]])

    actual = loss_matrix(X, centers)

    assert np.array_equal(actual, np.array([[0.0, 2.0], [2.5, 0.5]]))


def test_robust_loss_scale_is_raw_mad_of_finite_training_losses():
    loss = np.array([[0.0, np.nan], [1.0, np.inf], [4.0, np.nan], [9.0, np.inf]])

    # Finite losses are [0, 1, 4, 9]: median 2.5 and raw MAD 2.
    assert robust_loss_scale(loss) == 2.0


@pytest.mark.parametrize(
    "loss",
    [
        np.empty((0, 2)),
        np.ones((3, 1)),
        np.array([[0.0, np.nan], [1.0, 0.0]]),
        np.full((2, 2), np.nan),
        np.array([[0.0, -np.inf], [1.0, 0.0]]),
        np.ones((4, 2)),
    ],
)
def test_robust_loss_scale_rejects_invalid_or_zero_scale(loss):
    with pytest.raises(ValueError):
        robust_loss_scale(loss)


def test_evidence_penalty_uses_arrival_loss_and_previous_to_destination_direction():
    loss = np.array([[0.0, 10.0], [2.0, 0.0], [0.0, 0.2]])
    lambda0 = 4.0
    beta = np.log(4.0)

    penalty = evidence_penalty_seq(loss, lambda0, beta, q_train=1.0)

    assert np.array_equal(penalty[:, np.arange(2), np.arange(2)], np.zeros((3, 2)))
    assert penalty[1, 0, 1] == pytest.approx(lambda0 * np.exp(-beta * np.tanh(2.0)))
    assert penalty[1, 1, 0] == lambda0
    assert penalty[2, 1, 0] == pytest.approx(lambda0 * np.exp(-beta * np.tanh(0.2)))
    assert penalty[2, 0, 1] == lambda0
    off_diagonal = penalty[:, ~np.eye(2, dtype=bool)]
    assert (off_diagonal >= lambda0 * np.exp(-beta)).all()
    assert (off_diagonal <= lambda0).all()


def test_evidence_penalty_beta_zero_is_exact_fixed_penalty():
    loss = np.random.default_rng(19).uniform(0.0, 5.0, size=(12, 3))
    q_train = robust_loss_scale(loss[:8])

    actual = evidence_penalty_seq(loss, lambda0=5.0, beta=0.0, q_train=q_train)
    expected = lam_to_penalty_seq(np.full(len(loss), 5.0), n_c=3)

    assert np.array_equal(actual, expected)


@pytest.mark.parametrize(
    "lambda0,beta,q_train",
    [
        (-1.0, 0.0, 1.0),
        (np.inf, 0.0, 1.0),
        (1.0, -0.1, 1.0),
        (1.0, np.nan, 1.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.0, np.inf),
    ],
)
def test_evidence_penalty_rejects_invalid_parameters(lambda0, beta, q_train):
    loss = np.array([[0.0, 1.0], [1.0, 0.0]])

    with pytest.raises(ValueError):
        evidence_penalty_seq(loss, lambda0, beta, q_train)


def test_collapsed_center_has_finite_scale_penalties_and_path():
    X = np.array([[0.0], [1.0], [2.0], [4.0]])
    loss = loss_matrix(X, np.array([[0.0], [np.nan]]))
    q_train = robust_loss_scale(loss)
    lambda0 = 4.0
    beta = np.log(4.0)

    penalty = evidence_penalty_seq(loss, lambda0, beta, q_train)
    beta_zero = evidence_penalty_seq(loss, lambda0, 0.0, q_train)
    path, value = dp_tv(loss, penalty)

    assert q_train == 1.0
    assert np.isfinite(penalty).all()
    assert np.array_equal(penalty[:, 0, 1], np.full(len(loss), lambda0))
    assert np.array_equal(penalty[:, 1, 0], np.full(len(loss), lambda0 * np.exp(-beta)))
    assert np.array_equal(
        beta_zero, lam_to_penalty_seq(np.full(len(loss), lambda0), n_c=2)
    )
    assert np.array_equal(path, np.zeros(len(loss), dtype=int))
    assert np.isfinite(value)


def test_evidence_penalty_rejects_negative_infinity():
    loss = np.array([[0.0, -np.inf], [1.0, 0.0]])

    with pytest.raises(ValueError, match="negative infinity"):
        evidence_penalty_seq(loss, lambda0=1.0, beta=np.log(2.0), q_train=1.0)


def test_evidence_penalty_requires_finite_state_loss_in_each_row():
    loss = np.array([[np.nan, np.inf], [1.0, 0.0]])

    with pytest.raises(ValueError, match="each loss_mx row"):
        evidence_penalty_seq(loss, lambda0=1.0, beta=np.log(2.0), q_train=1.0)


@pytest.mark.parametrize("beta", [0.0, np.log(2.0), np.log(4.0)])
def test_adaptive_penalty_dp_matches_brute_force_objective(beta):
    loss = np.random.default_rng(23).uniform(0.0, 5.0, size=(6, 2))
    q_train = robust_loss_scale(loss[:4])
    penalty = evidence_penalty_seq(loss, lambda0=3.0, beta=beta, q_train=q_train)

    assign, value = dp_tv(loss, penalty)

    assert value == pytest.approx(brute_force_tv(loss, penalty))
    assert path_cost_tv(loss, penalty, assign) == pytest.approx(value)


def test_adaptive_penalty_toy_switches_on_strong_evidence_not_weak_reversal():
    loss = np.array([[0.0, 10.0], [2.0, 0.0], [0.0, 0.2]])
    fixed = evidence_penalty_seq(loss, lambda0=4.0, beta=0.0, q_train=1.0)
    adaptive = evidence_penalty_seq(loss, lambda0=4.0, beta=np.log(4.0), q_train=1.0)

    fixed_path, fixed_value = dp_tv(loss, fixed)
    adaptive_path, adaptive_value = dp_tv(loss, adaptive)
    fixed_online = dp_tv(loss, fixed, return_value_mx=True).argmin(axis=1)
    adaptive_online = dp_tv(loss, adaptive, return_value_mx=True).argmin(axis=1)

    assert np.array_equal(fixed_path, [0, 0, 0])
    assert np.array_equal(fixed_online, [0, 0, 0])
    assert fixed_value == 2.0
    assert np.array_equal(adaptive_path, [0, 1, 1])
    assert np.array_equal(adaptive_online, [0, 1, 1])
    expected = 4.0 * np.exp(-np.log(4.0) * np.tanh(2.0)) + 0.2
    assert adaptive_value == pytest.approx(expected)
    assert adaptive_value == pytest.approx(brute_force_tv(loss, adaptive))


def test_two_state_directed_cost_identity_holds_each_arrival_day():
    loss = np.array([[0.0, 4.0], [2.0, 0.0], [0.1, 0.3], [0.0, 2.0], [3.0, 0.0]])
    penalty = evidence_penalty_seq(loss, lambda0=4.0, beta=np.log(4.0), q_train=1.0)
    path = np.array([0, 1, 1, 0, 1])

    directed = 0.0
    decomposed = 0.0
    for t in range(1, len(path)):
        previous, current = path[t - 1], path[t]
        c01, c10 = penalty[t, 0, 1], penalty[t, 1, 0]
        directed += penalty[t, previous, current]
        decomposed += 0.5 * (c01 + c10) * (previous != current)
        decomposed += 0.5 * (c01 - c10) * (current - previous)

    assert directed == pytest.approx(decomposed)


def test_adaptive_online_values_are_prefix_invariant():
    prefix_loss = np.array([[0.0, 4.0], [0.2, 1.0], [2.0, 0.1], [0.4, 0.6], [3.0, 0.0]])
    future_loss = np.array([[1e6, 0.0], [0.0, 1e6]])
    full_loss = np.vstack([prefix_loss, future_loss])
    q_train = robust_loss_scale(prefix_loss[:3])
    beta = np.log(4.0)
    prefix_penalty = evidence_penalty_seq(prefix_loss, 4.0, beta, q_train)
    full_penalty = evidence_penalty_seq(full_loss, 4.0, beta, q_train)

    prefix_values = dp_tv(prefix_loss, prefix_penalty, return_value_mx=True)
    full_values = dp_tv(full_loss, full_penalty, return_value_mx=True)

    assert np.array_equal(prefix_penalty, full_penalty[: len(prefix_loss)])
    assert np.array_equal(prefix_values, full_values[: len(prefix_loss)])
    assert np.array_equal(
        prefix_values.argmin(axis=1),
        full_values[: len(prefix_loss)].argmin(axis=1),
    )


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
        pen[:, np.arange(n_c), np.arange(n_c)] = 0.0  # zero diagonal
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
        assert tv_val == ref_val
        assert np.array_equal(tv_assign, ref_assign)
        # online value matrices agree too
        ref_vals = dp(loss, pen_mx, return_value_mx=True)
        tv_vals = dp_tv(
            loss, lam_to_penalty_seq(np.full(40, lam), 2), return_value_mx=True
        )
        assert np.array_equal(ref_vals, tv_vals)


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
        lam_to_penalty_seq(np.array([[1.0]]), 2)  # not 1-d
    with pytest.raises(ValueError):
        lam_to_penalty_seq(np.array([1.0, -0.1]), 2)  # negative
    with pytest.raises(ValueError):
        lam_to_penalty_seq(np.array([1.0, np.inf]), 2)  # non-finite


def test_tv_model_rejects_wrong_length_and_double_spec():
    X = RNG.normal(size=(50, 3))
    m = TVJumpModel(n_init=2)
    with pytest.raises(ValueError):
        m.fit_tv(X, lam_seq=np.ones(49))  # length mismatch
    with pytest.raises(ValueError):
        m.fit_tv(X)  # neither given
    with pytest.raises(ValueError):
        m.fit_tv(
            X, lam_seq=np.ones(50), penalty_seq=lam_to_penalty_seq(np.ones(50), 2)
        )  # both given


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
        ref = JumpModel(
            n_components=2, jump_penalty=lam, cont=False, n_init=4, random_state=3
        ).fit(X, ret, sort_by="cumret")
        tv = TVJumpModel(n_components=2, n_init=4, random_state=3)
        tv.fit_tv(X, ret, lam_seq=np.full(len(X), lam), sort_by="cumret")
        assert np.array_equal(tv.centers_, ref.centers_, equal_nan=True)
        assert np.array_equal(np.asarray(tv.labels_), np.asarray(ref.labels_))
        assert tv.val_ == ref.val_
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
