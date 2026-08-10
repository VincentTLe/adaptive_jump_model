"""Focused regressions for the studentized Sharpe-difference helpers.

`scripts/diagnostics/studentized_sharpe_difference.py` re-derives the repo's
Sharpe from a vector of uncentered moments so that the difference of two Sharpe
ratios is a smooth function of means (Ledoit & Wolf 2008). Every test here pins
a property
whose violation silently produces a number that no other artifact in this
project would reproduce:

* the moment form must BE `performance_metrics`' estimator, including its
  ``ddof=1`` volatility, not an asymptotically equivalent approximation of it
  (the defect this file was written to close);
* the analytic gradient must be the gradient of the function actually used;
* the repo's 5-moment extension must collapse onto ddof_scale(T) times
  Ledoit-Wolf Eq. (4) when the cash leg is zero -- a scaled relationship, not
  equality, and only under that condition -- or it is a different method
  rather than a generalization;
* the block-sum variance estimator must reduce to the i.i.d. delta method at
  block length 1, which is the reduction their footnote 9 requires;
* the resamplers must stay inside the sample and wrap only where the circular
  bootstrap is supposed to wrap.

Nothing here refits a model, reads an artifact, or runs a full bootstrap: all
inputs are synthetic and every assertion is arithmetic.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adaptive_jump.backtest import annualized_excess_sharpe, performance_metrics

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def module():
    """Import the script without executing ``main()``."""
    path = ROOT / "scripts/diagnostics/studentized_sharpe_difference.py"
    spec = importlib.util.spec_from_file_location("studentized_under_test", path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def _arms(observations: int = 500, seed: int = 7):
    """Two correlated daily return series plus a positive cash leg."""
    rng = np.random.default_rng(seed)
    common = rng.normal(0.0004, 0.011, observations)
    challenger = common + rng.normal(0.0, 0.0015, observations)
    baseline = common + rng.normal(0.0, 0.0015, observations)
    cash = np.full(observations, 0.00012) + rng.normal(0.0, 1e-5, observations)
    return challenger, baseline, cash


def _series(module, observations: int = 500, seed: int = 7):
    challenger, baseline, cash = _arms(observations, seed)
    return module.Series(
        market="synthetic",
        challenger=challenger,
        baseline=baseline,
        cash=cash,
        challenger_sharpe=annualized_excess_sharpe(
            pd.Series(challenger), pd.Series(cash)
        ),
        baseline_sharpe=annualized_excess_sharpe(pd.Series(baseline), pd.Series(cash)),
    )


def _trades(strategy: np.ndarray, cash: np.ndarray) -> pd.DataFrame:
    """A minimal frame `performance_metrics` accepts, with a fixed position."""
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2000-01-03", periods=strategy.size),
            "cash_return": cash,
            "position": np.ones(strategy.size),
            "one_way_turnover": np.zeros(strategy.size),
            "strategy_return": strategy,
        }
    )


# ---------------------------------------------------------------------------
# The estimand: moment form == performance_metrics, exactly
# ---------------------------------------------------------------------------


def test_ddof_scale_is_the_bessel_correction(module):
    for observations in (2, 10, 8565):
        assert module.ddof_scale(observations) ** 2 == pytest.approx(
            (observations - 1) / observations, rel=0, abs=1e-15
        )


def test_ddof_scale_rejects_a_degenerate_sample(module):
    with pytest.raises(ValueError):
        module.ddof_scale(1)


def test_moment_form_matches_performance_metrics_sharpe_difference(module):
    """The defect this file closes: ddof=0 moments vs the repo's ddof=1 sd."""
    series = _series(module)
    moments = module.moment_matrix(series, "repo")
    moment_delta = float(
        module.sharpe_difference(moments.mean(axis=0), "repo", series.observations)
    )

    challenger = performance_metrics(_trades(series.challenger, series.cash))
    baseline = performance_metrics(_trades(series.baseline, series.cash))
    repo_delta = float(challenger["sharpe"]) - float(baseline["sharpe"])

    assert moment_delta == pytest.approx(repo_delta, rel=0, abs=1e-14)


def test_moment_form_reproduces_each_arm_not_only_the_difference(module):
    """A difference can match by cancelling two equal errors; the levels cannot.

    This must go THROUGH the module -- an earlier version rebuilt the scale
    constant locally and so passed even when `estimator_scale` was wrong.
    Each arm is recovered by zeroing the other arm's contribution.
    """
    series = _series(module)
    m1, m2, g1, g2, k = module.moment_matrix(series, "repo").mean(axis=0)
    observations = series.observations
    inert = k**2 + 1.0  # any second moment giving a positive variance at mean k

    # Setting an arm's mean equal to the cash mean makes its term exactly zero,
    # so f() returns the other arm alone -- computed by the module, not here.
    challenger_only = float(
        module.sharpe_difference(np.array([m1, k, g1, inert, k]), "repo", observations)
    )
    baseline_only = float(
        module.sharpe_difference(np.array([k, m2, inert, g2, k]), "repo", observations)
    )
    assert challenger_only == pytest.approx(
        annualized_excess_sharpe(pd.Series(series.challenger), pd.Series(series.cash)),
        rel=1e-13,
        abs=0,
    )
    assert -baseline_only == pytest.approx(
        annualized_excess_sharpe(pd.Series(series.baseline), pd.Series(series.cash)),
        rel=1e-13,
        abs=0,
    )
    # and the two arms really are different, so this is not two ways of zero
    assert abs(challenger_only + baseline_only) > 1e-6


def test_sigma_rejects_a_non_positive_variance(module):
    with pytest.raises(ValueError):
        module._sigma(np.array(1.0), np.array(0.5))


def test_sigma_rejects_a_variance_of_exactly_zero(module):
    """Pins `<= 0`, not `< 0`: a degenerate arm has an undefined Sharpe."""
    with pytest.raises(ValueError):
        module._sigma(np.array(1.0), np.array(1.0))


# ---------------------------------------------------------------------------
# The gradient
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("estimator", ["repo", "lw_excess"])
def test_gradient_matches_a_central_finite_difference(module, estimator):
    assert module.check_gradient(_series(module), estimator) < 1e-6


def _zero_cash_series(module):
    challenger, baseline, _ = _arms()
    return module.Series(
        market="synthetic",
        challenger=challenger,
        baseline=baseline,
        cash=np.zeros_like(challenger),
        challenger_sharpe=math.nan,
        baseline_sharpe=math.nan,
    )


def test_repo_gradient_collapses_onto_ledoit_wolf_with_a_zero_cash_leg(module):
    """At k = 0 the two gradients differ by the Bessel constant and nothing else.

    Note the SCOPE: this is the k = 0 case only. The two are never exactly
    equal -- asserting equality here would be asserting that the canonical
    estimator carries the repo's ddof=1 convention, which it must not. At
    k != 0 the relation below does not hold either, because the two also use
    different denominators; that case is covered by the report's own numbers,
    not by this test.
    """
    series = _zero_cash_series(module)
    observations = series.observations
    repo = module.gradient(
        module.moment_matrix(series, "repo").mean(axis=0), "repo", observations
    )
    canonical = module.gradient(
        module.moment_matrix(series, "lw_excess").mean(axis=0),
        "lw_excess",
        observations,
    )
    scale = module.ddof_scale(observations)
    # RELATIVE tolerance: these components are O(1e3), where one float64 ulp is
    # already ~5e-13, so an absolute 1e-15 bound would test the association
    # order of the multiplication rather than the mathematics.
    assert repo[:4] == pytest.approx(scale * canonical, rel=1e-13, abs=0)
    # and the scaling is real, not a no-op that would make the test vacuous
    assert repo[:4] != pytest.approx(canonical, rel=1e-13, abs=0)


def test_delta_at_zero_cash_is_the_canonical_delta_times_the_bessel_factor(module):
    """The same relationship at the level of the statistic, not just its gradient."""
    series = _zero_cash_series(module)
    observations = series.observations
    repo = float(
        module.sharpe_difference(
            module.moment_matrix(series, "repo").mean(axis=0), "repo", observations
        )
    )
    canonical = float(
        module.sharpe_difference(
            module.moment_matrix(series, "lw_excess").mean(axis=0),
            "lw_excess",
            observations,
        )
    )
    assert repo == pytest.approx(
        module.ddof_scale(observations) * canonical, rel=1e-14, abs=0
    )
    assert abs(repo) < abs(canonical)


def test_canonical_estimator_carries_no_bessel_correction(module):
    """lw_excess carries the annualization and NOT the Bessel factor.

    Its scale must be sqrt(252) alone: the Eq. (1)-(2) functional form plus
    this repo's reporting annualization, with no ddof=1 correction.
    """
    for observations in (50, 500, 8565):
        assert module.estimator_scale("lw_excess", observations) == pytest.approx(
            math.sqrt(module.ANNUALIZATION), rel=1e-15, abs=0
        )
        assert module.estimator_scale("repo", observations) == pytest.approx(
            math.sqrt(module.ANNUALIZATION) * module.ddof_scale(observations),
            rel=1e-15,
            abs=0,
        )
    # the canonical gradient does not depend on T at all
    series = _series(module)
    v = module.moment_matrix(series, "lw_excess").mean(axis=0)
    assert module.gradient(v, "lw_excess", 100) == pytest.approx(
        module.gradient(v, "lw_excess", 10_000), rel=1e-15, abs=0
    )


def test_canonical_moment_matrix_is_built_from_excess_returns(module):
    """lw_excess must net cash BEFORE forming its moments, or it is not LW's f.

    Without this the cash subtraction can be dropped entirely and every other
    test in this file still passes -- the defining feature of the "canonical
    excess-return cross-check" would be untested.
    """
    series = _series(module)
    moments = module.moment_matrix(series, "lw_excess")
    excess_challenger = series.challenger - series.cash
    excess_baseline = series.baseline - series.cash
    assert moments[:, 0] == pytest.approx(excess_challenger, rel=1e-15, abs=0)
    assert moments[:, 1] == pytest.approx(excess_baseline, rel=1e-15, abs=0)
    assert moments[:, 2] == pytest.approx(excess_challenger**2, rel=1e-15, abs=0)
    assert moments[:, 3] == pytest.approx(excess_baseline**2, rel=1e-15, abs=0)
    # and the cash leg is non-trivial, so dropping it would be a real change
    assert np.abs(series.cash).mean() > 1e-6
    assert moments[:, 0] != pytest.approx(series.challenger, rel=1e-9, abs=0)


def test_canonical_delta_is_the_excess_return_sharpe_difference(module):
    """The end-to-end statement of the same thing, in the repo's own terms."""
    series = _series(module)
    observations = series.observations
    delta = float(
        module.sharpe_difference(
            module.moment_matrix(series, "lw_excess").mean(axis=0),
            "lw_excess",
            observations,
        )
    )
    expected = math.sqrt(module.ANNUALIZATION) * (
        (series.challenger - series.cash).mean()
        / (series.challenger - series.cash).std(ddof=0)
        - (series.baseline - series.cash).mean()
        / (series.baseline - series.cash).std(ddof=0)
    )
    assert delta == pytest.approx(expected, rel=1e-13, abs=0)


def test_cash_derivative_equals_its_analytic_value_and_is_not_zero(module):
    """d/dk = sqrt(252) sqrt((T-1)/T) (1/sig_2 - 1/sig_1).

    This coordinate has no counterpart in Ledoit-Wolf Eq. (4) -- it is the whole
    content of the 5-moment extension -- so it is pinned to its closed form and
    to a finite difference of the function actually used. It vanishes only when
    the two arms have equal volatility, which is why an "is it zero?" assertion
    would be testing a coincidence rather than the derivative.
    """
    series = _series(module)
    observations = series.observations
    v = module.moment_matrix(series, "repo").mean(axis=0)
    m1, m2, g1, g2, _ = v

    sigma_1 = math.sqrt(g1 - m1**2)
    sigma_2 = math.sqrt(g2 - m2**2)
    expected = (
        math.sqrt(module.ANNUALIZATION)
        * module.ddof_scale(observations)
        * (1.0 / sigma_2 - 1.0 / sigma_1)
    )

    analytic = module.gradient(v, "repo", observations)[4]
    assert analytic == pytest.approx(expected, rel=1e-13, abs=0)

    # the two arms are deliberately not equal-volatility, so it must not be zero
    assert sigma_1 != pytest.approx(sigma_2, rel=1e-6)
    assert abs(analytic) > 0.5 * abs(expected)  # not merely "bigger than tiny"

    step = 1e-8
    up, down = v.copy(), v.copy()
    up[4] += step
    down[4] -= step
    numeric = float(
        module.sharpe_difference(up, "repo", observations)
        - module.sharpe_difference(down, "repo", observations)
    ) / (2.0 * step)
    assert analytic == pytest.approx(numeric, rel=1e-6)


def test_cash_derivative_vanishes_exactly_when_the_volatilities_coincide(module):
    """The one case where zero is the right answer -- stated as a consequence."""
    series = _series(module)
    v = module.moment_matrix(series, "repo").mean(axis=0).copy()
    m1, m2, g1, _, _ = v
    v[3] = g1 - m1**2 + m2**2  # force sigma_2 == sigma_1
    assert module.gradient(v, "repo", series.observations)[4] == pytest.approx(
        0.0, abs=1e-12
    )


@pytest.mark.parametrize("estimator", ["repo", "lw_excess"])
def test_gradient_carries_the_same_scale_constant_as_the_statistic(module, estimator):
    """Delta and grad f must carry the SAME constant, or the HAC SE is wrong.

    Comparing gradient(T=100) with gradient(T=10000) does not establish this --
    a build where the two functions used different constants would pass that.
    Euler's theorem does not apply (f is not homogeneous), so the constant is
    recovered directly: f and grad f are each linear in `root`, so the ratio
    between two sample sizes must agree between them.
    """
    series = _series(module)
    v = module.moment_matrix(series, estimator).mean(axis=0)
    ratio_statistic = float(module.sharpe_difference(v, estimator, 100)) / float(
        module.sharpe_difference(v, estimator, 10_000)
    )
    grad_small = module.gradient(v, estimator, 100)
    grad_large = module.gradient(v, estimator, 10_000)
    ratio_gradient = grad_small / grad_large
    assert ratio_gradient == pytest.approx(ratio_statistic, rel=1e-13, abs=0)
    expected = module.estimator_scale(estimator, 100) / module.estimator_scale(
        estimator, 10_000
    )
    assert ratio_statistic == pytest.approx(expected, rel=1e-13, abs=0)


def test_unknown_estimator_is_rejected(module):
    series = _series(module)
    with pytest.raises(ValueError):
        module.sharpe_difference(np.zeros(5), "not_an_estimator", series.observations)
    with pytest.raises(ValueError):
        module.moment_matrix(series, "not_an_estimator")


# ---------------------------------------------------------------------------
# HAC and block variance estimators
# ---------------------------------------------------------------------------


def test_parzen_kernel_matches_its_definition(module):
    assert module.parzen_kernel(np.array(0.0)) == pytest.approx(1.0)
    assert module.parzen_kernel(np.array(0.5)) == pytest.approx(0.25)
    assert module.parzen_kernel(np.array(1.0)) == pytest.approx(0.0)
    assert module.parzen_kernel(np.array(1.5)) == pytest.approx(0.0)
    # even, and never negative -- the property that makes Psi_hat PSD
    grid = np.linspace(-2.0, 2.0, 41)
    assert module.parzen_kernel(grid) == pytest.approx(module.parzen_kernel(-grid))
    assert (module.parzen_kernel(grid) >= 0.0).all()


def test_block_standard_error_at_block_one_is_the_iid_delta_method(module):
    """Ledoit-Wolf footnote 9: Psi* must reduce to the sample covariance at b = 1."""
    series = _series(module)
    moments = module.moment_matrix(series, "repo")
    observations, width = moments.shape
    v = moments.mean(axis=0)
    centred = moments - v
    psi = centred.T @ centred / observations * (observations / (observations - width))
    grad = module.gradient(v, "repo", observations)
    expected = math.sqrt(float(grad @ psi @ grad) / observations)
    assert module.block_standard_error(moments, 1, "repo") == pytest.approx(
        expected, rel=1e-12
    )


def test_block_sum_normalises_by_sqrt_block_at_a_real_block_length(module):
    """Psi* = (1/l) sum_j zeta_j zeta_j' with zeta_j = b^-0.5 sum_{t in j} y_t.

    At b = 1 the sqrt is invisible (sqrt(1) = 1), so the b = 1 reduction test
    cannot see this normalisation at all. Any other exponent rescales Psi* by
    a power of b and so rescales every bootstrap standard error.
    """
    series = _series(module, observations=400, seed=11)
    moments = module.moment_matrix(series, "repo")
    block = 20
    observations, width = moments.shape
    blocks = observations // block

    v = moments.mean(axis=0)
    centred = moments - v
    zeta = centred[: blocks * block].reshape(blocks, block, width).sum(
        axis=1
    ) / math.sqrt(block)
    psi = (zeta.T @ zeta / blocks) * (observations / (observations - width))
    grad = module.gradient(v, "repo", observations)
    expected = math.sqrt(float(grad @ psi @ grad) / observations)
    assert module.block_standard_error(moments, block, "repo") == pytest.approx(
        expected, rel=1e-12, abs=0
    )

    # dividing by b instead of sqrt(b) shrinks it by exactly sqrt(b) -- far
    # outside the tolerance, so the exponent is genuinely pinned
    assert expected / math.sqrt(block) != pytest.approx(expected, rel=1e-6, abs=0)


def test_andrews_alpha_matches_its_closed_form_on_a_known_ar1(module):
    """Andrews (1991) alpha(2) = 4 rho^2 sig^4/(1-rho)^8 / [sig^4/(1-rho)^4].

    The automatic bandwidth 2.6614 (alpha T)^0.2 sets the HAC standard error,
    hence the width of the published interval, and nothing else in this file
    touched it. Pinned on a single AR(1) column where the answer is analytic.
    """
    rho = 0.6
    rng = np.random.default_rng(3)
    innovations = rng.normal(0.0, 1.0, 20_000)
    series = np.empty_like(innovations)
    series[0] = innovations[0]
    for index in range(1, series.size):
        series[index] = rho * series[index - 1] + innovations[index]

    alpha = module.andrews_alpha(series[:, None])
    # with one column the variance terms cancel: alpha = 4 rho^2 / (1-rho)^4
    assert alpha == pytest.approx(4.0 * rho**2 / (1.0 - rho) ** 4, rel=0.05)
    # a white-noise column must give a much smaller alpha
    assert module.andrews_alpha(innovations[:, None]) < 0.1 * alpha


def test_kernel_psi_uses_the_andrews_bandwidth_and_both_lag_directions(module):
    """S_T = 2.6614 (alpha(2) T)^0.2, Parzen weights, Gamma(j) + Gamma(j)'.

    The constant 2.6614 is Andrews (1991) Table I for the Parzen kernel and it
    sets how many lags enter, hence the HAC standard error and the width of the
    published interval. Rebuilt here from the formula rather than trusted.
    """
    rng = np.random.default_rng(5)
    innovations = rng.normal(0.0, 1.0, (3000, 2))
    y = np.empty_like(innovations)
    y[0] = innovations[0]
    for index in range(1, y.shape[0]):
        y[index] = 0.5 * y[index - 1] + innovations[index]
    y = y - y.mean(axis=0)

    observations, width = y.shape
    bandwidth = 2.6614 * (module.andrews_alpha(y) * observations) ** 0.2
    bandwidth = float(min(max(bandwidth, 1.0), observations - 1.0))
    expected = y.T @ y / observations
    for lag in range(1, int(math.floor(bandwidth)) + 1):
        weight = float(module.parzen_kernel(np.array(lag / bandwidth)))
        if weight == 0.0:
            continue
        gamma = y[lag:].T @ y[:-lag] / observations
        expected = expected + weight * (gamma + gamma.T)
    expected = expected * (observations / (observations - width))

    assert module.kernel_psi(y) == pytest.approx(expected, rel=1e-12, abs=0)
    assert bandwidth > 1.0  # the lag loop actually runs on this series

    # a wrong constant changes the answer materially, so it is genuinely pinned
    narrow = 1.0 * (module.andrews_alpha(y) * observations) ** 0.2
    assert narrow < bandwidth - 1.0


def test_hac_standard_error_is_the_delta_method_formula(module):
    """Pins sqrt(g' Psi g / T). Comparing its delta to `sharpe_difference` is
    self-comparison -- `hac_standard_error` returns that very call -- so the
    standard error itself is rebuilt here from its published definition."""
    series = _series(module)
    moments = module.moment_matrix(series, "repo")
    observations = moments.shape[0]
    delta, error, prewhitened = module.hac_standard_error(moments, "repo")

    v = moments.mean(axis=0)
    psi, _ = module.prewhitened_psi(moments - v)
    grad = module.gradient(v, "repo", observations)
    expected = math.sqrt(float(grad @ psi @ grad) / observations)
    assert error == pytest.approx(expected, rel=1e-13, abs=0)
    assert isinstance(prewhitened, (bool, np.bool_))

    # divide by T, not T-1: at this sample size the two differ by 1e-3 relative,
    # far above the tolerance above, so the convention is genuinely pinned
    wrong = math.sqrt(float(grad @ psi @ grad) / (observations - 1))
    assert error != pytest.approx(wrong, rel=1e-6, abs=0)
    assert delta == pytest.approx(
        float(module.sharpe_difference(v, "repo", observations)), rel=0, abs=1e-15
    )


# ---------------------------------------------------------------------------
# Resamplers
# ---------------------------------------------------------------------------


def test_circular_block_indices_actually_wrap_around_the_end(module):
    """The wrap is the whole difference from a moving block, so it is observed.

    Asserting only "contiguous modulo T" is satisfied by a moving block too.
    A circular bootstrap draws starts uniformly on 0..T-1, so with 4000 draws
    at T=50, b=7 some block MUST straddle the end -- and every index must still
    be inside the sample.
    """
    drawn = module.circular_block_indices(50, 4000, 7, np.random.default_rng(0))
    assert drawn.shape == (4000, 50)
    assert drawn.min() >= 0 and drawn.max() < 50
    blocks = drawn[:, :7]
    steps = np.diff(blocks, axis=1) % 50
    assert (steps == 1).all()
    wrapped = (np.diff(blocks, axis=1) < 0).any(axis=1)
    assert wrapped.any(), "no block wrapped: this is a moving block, not circular"
    # starts should cover the whole index range, including the last b-1 positions
    assert blocks[:, 0].max() > 50 - 7


def test_moving_block_indices_never_wrap_even_when_many_are_drawn(module):
    """The complementary property: a moving block must never straddle the end."""
    drawn = module.moving_block_indices(50, 4000, 7, np.random.default_rng(1))
    assert drawn.shape == (4000, 50)
    assert drawn.min() >= 0 and drawn.max() < 50
    blocks = drawn[:, :7]
    assert (np.diff(blocks, axis=1) == 1).all()  # strictly increasing => no wrap
    assert blocks[:, 0].max() <= 50 - 7  # and starts are restricted, unlike above


@pytest.mark.parametrize("scheme", ["circular_block", "moving_block"])
def test_resamplers_reject_a_block_longer_than_the_sample(module, scheme):
    rng = np.random.default_rng(2)
    with pytest.raises(ValueError):
        module._indices(scheme, 10, 3, 11, rng)


def test_unknown_bootstrap_scheme_is_rejected(module):
    with pytest.raises(ValueError):
        module._indices("not_a_scheme", 10, 3, 2, np.random.default_rng(3))


# ---------------------------------------------------------------------------
# End to end on a synthetic sample, small enough to stay a unit test
# ---------------------------------------------------------------------------


def test_symmetric_interval_uses_the_95th_percentile_of_the_absolute_statistic(
    module,
):
    """LW Eq. (7): half-width = z*_{|.|,0.95} * s(Delta_hat), not any quantile.

    "Midpoint == delta" and "0 < p <= 1" hold for EVERY quantile and every
    standard error, so they pin nothing. The half-width is reconstructed here
    from the module's own CONFIDENCE constant and checked to be sensitive to it.
    """
    series = _series(module, observations=400, seed=11)
    result = module.studentized_bootstrap(
        series, "repo", "circular_block", 20, 200, seed=123
    )
    half_width = 0.5 * (result.symmetric_high - result.symmetric_low)
    implied_quantile = half_width / result.standard_error

    # recompute the studentized draws independently and take the same quantile
    moments = module.moment_matrix(series, "repo")
    delta, standard_error, _ = module.hac_standard_error(moments, "repo")
    rng = np.random.default_rng(123)
    indices = module._indices("circular_block", series.observations, 200, 20, rng)
    draws, errors = module._bootstrap_standard_errors(moments[indices], 20, "repo")
    absolute = np.abs((draws - delta) / errors)
    assert implied_quantile == pytest.approx(
        float(np.quantile(absolute, module.CONFIDENCE)), rel=1e-12, abs=0
    )
    # and it is genuinely the 95th, distinguishable from the median
    assert implied_quantile > float(np.quantile(absolute, 0.5))
    assert result.standard_error == pytest.approx(standard_error, rel=1e-13, abs=0)
    assert result.delta == pytest.approx(delta, rel=0, abs=1e-15)


def test_symmetric_p_value_counts_the_correct_tail(module):
    """LW Eq. (9): fraction of |t*| at least as large as |t_observed|, +1/+1.

    A flipped comparison (<= instead of >=) gives a number that is still a
    probability, so a range check cannot catch it. The count is reproduced here.
    """
    series = _series(module, observations=400, seed=11)
    replications = 200
    result = module.studentized_bootstrap(
        series, "repo", "circular_block", 20, replications, seed=123
    )
    moments = module.moment_matrix(series, "repo")
    delta, standard_error, _ = module.hac_standard_error(moments, "repo")
    rng = np.random.default_rng(123)
    indices = module._indices(
        "circular_block", series.observations, replications, 20, rng
    )
    draws, errors = module._bootstrap_standard_errors(moments[indices], 20, "repo")
    absolute = np.abs((draws - delta) / errors)
    observed = abs(delta) / standard_error
    expected = (np.count_nonzero(absolute >= observed) + 1) / (replications + 1)
    assert result.p_value_symmetric == pytest.approx(expected, rel=0, abs=1e-15)
    # the opposite tail is a different number here, so the direction is pinned
    flipped = (np.count_nonzero(absolute <= observed) + 1) / (replications + 1)
    assert expected != pytest.approx(flipped, rel=1e-6, abs=0)


def test_equal_tailed_endpoints_invert_the_studentized_quantiles(module):
    """The upper quantile builds the LOWER endpoint -- the inversion is the point.

    Getting this backwards still yields an interval containing delta, so an
    ordering assertion cannot catch it.
    """
    series = _series(module, observations=400, seed=11)
    result = module.studentized_bootstrap(
        series, "repo", "circular_block", 20, 200, seed=123
    )
    moments = module.moment_matrix(series, "repo")
    delta, standard_error, _ = module.hac_standard_error(moments, "repo")
    rng = np.random.default_rng(123)
    indices = module._indices("circular_block", series.observations, 200, 20, rng)
    draws, errors = module._bootstrap_standard_errors(moments[indices], 20, "repo")
    studentized = (draws - delta) / errors
    alpha = 1.0 - module.CONFIDENCE
    low_q, high_q = np.quantile(studentized, [alpha / 2.0, 1.0 - alpha / 2.0])
    assert result.equal_tailed_low == pytest.approx(
        delta - float(high_q) * standard_error, rel=1e-12, abs=0
    )
    assert result.equal_tailed_high == pytest.approx(
        delta - float(low_q) * standard_error, rel=1e-12, abs=0
    )
    assert result.equal_tailed_low < result.equal_tailed_high
    # the two quantiles are distinct, so swapping them would be detected
    assert float(high_q) != pytest.approx(float(low_q), rel=1e-6, abs=0)


def test_bootstrap_result_reports_the_sample_it_was_given(module):
    series = _series(module, observations=400, seed=11)
    result = module.studentized_bootstrap(
        series, "repo", "circular_block", 20, 60, seed=123
    )
    assert result.observations == series.observations
    assert result.replications == 60
    assert result.block == 20
    assert 0.0 <= result.positive_fraction <= 1.0


def test_studentized_bootstrap_is_deterministic_given_its_seed(module):
    series = _series(module, observations=400, seed=11)
    first = module.studentized_bootstrap(
        series, "repo", "circular_block", 20, 40, seed=99
    )
    second = module.studentized_bootstrap(
        series, "repo", "circular_block", 20, 40, seed=99
    )
    assert first == second
