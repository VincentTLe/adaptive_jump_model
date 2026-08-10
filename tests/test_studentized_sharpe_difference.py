"""Focused regressions for the studentized Sharpe-difference helpers.

`scripts/studentized_sharpe_difference.py` re-derives the repo's Sharpe from a
vector of uncentered moments so that the difference of two Sharpe ratios is a
smooth function of means (Ledoit & Wolf 2008). Every test here pins a property
whose violation silently produces a number that no other artifact in this
project would reproduce:

* the moment form must BE `performance_metrics`' estimator, including its
  ``ddof=1`` volatility, not an asymptotically equivalent approximation of it
  (the defect this file was written to close);
* the analytic gradient must be the gradient of the function actually used;
* the repo's 5-moment extension must collapse onto Ledoit-Wolf Eq. (4) when
  the cash leg is zero, or it is a different method rather than a
  generalization;
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
    path = ROOT / "scripts/studentized_sharpe_difference.py"
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
    """A difference can match by cancelling two equal errors; the levels cannot."""
    series = _series(module)
    root = math.sqrt(module.ANNUALIZATION) * module.ddof_scale(series.observations)
    for arm in ("challenger", "baseline"):
        values = getattr(series, arm)
        mean, second = values.mean(), (values**2).mean()
        moment_sharpe = root * (mean - series.cash.mean()) / math.sqrt(second - mean**2)
        assert moment_sharpe == pytest.approx(
            annualized_excess_sharpe(pd.Series(values), pd.Series(series.cash)),
            rel=0,
            abs=1e-14,
        )


def test_population_form_is_biased_by_exactly_the_bessel_factor(module):
    """The old (ddof=0) number is recoverable, so the correction is a constant."""
    series = _series(module)
    v = module.moment_matrix(series, "repo").mean(axis=0)
    corrected = float(module.sharpe_difference(v, "repo", series.observations))
    population = corrected / module.ddof_scale(series.observations)
    assert abs(population) > abs(corrected)
    assert corrected / population == pytest.approx(
        math.sqrt((series.observations - 1) / series.observations), rel=0, abs=1e-15
    )


def test_sigma_rejects_a_non_positive_variance(module):
    with pytest.raises(ValueError):
        module._sigma(np.array(1.0), np.array(0.5))


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

    They are NOT equal, and asserting equality would be asserting that the
    canonical estimator carries the repo's ddof=1 convention -- which it must
    not, or it stops being Ledoit-Wolf Eq. (1)-(2). The relationship is what
    makes the repo form a scaled generalization rather than a different method.
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
    assert repo[:4] == pytest.approx(scale * canonical, rel=0, abs=1e-15)
    # and the scaling is real, not a no-op that would make the test vacuous
    assert repo[:4] != pytest.approx(canonical, rel=0, abs=1e-15)


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
    """lw_excess must be Eq. (1)-(2) verbatim: its scale is sqrt(252), full stop."""
    for observations in (50, 500, 8565):
        assert module.estimator_scale("lw_excess", observations) == pytest.approx(
            math.sqrt(module.ANNUALIZATION), rel=0, abs=1e-15
        )
        assert module.estimator_scale("repo", observations) == pytest.approx(
            math.sqrt(module.ANNUALIZATION) * module.ddof_scale(observations),
            rel=0,
            abs=1e-15,
        )
    # the canonical gradient does not depend on T at all
    series = _series(module)
    v = module.moment_matrix(series, "lw_excess").mean(axis=0)
    assert module.gradient(v, "lw_excess", 100) == pytest.approx(
        module.gradient(v, "lw_excess", 10_000), rel=0, abs=1e-15
    )


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
    assert abs(analytic) > 1e-3

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


def test_gradient_carries_the_same_bessel_factor_as_the_statistic(module):
    """Delta and grad f must scale together or the HAC standard error is wrong."""
    series = _series(module)
    v = module.moment_matrix(series, "repo").mean(axis=0)
    small = module.gradient(v, "repo", 100)
    large = module.gradient(v, "repo", 10_000)
    expected = module.ddof_scale(100) / module.ddof_scale(10_000)
    assert small == pytest.approx(large * expected, rel=1e-12)


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


def test_hac_standard_error_is_positive_and_delta_matches_the_moment_form(module):
    series = _series(module)
    moments = module.moment_matrix(series, "repo")
    delta, error, _ = module.hac_standard_error(moments, "repo")
    assert error > 0.0
    assert delta == pytest.approx(
        float(module.sharpe_difference(moments.mean(axis=0), "repo", len(moments))),
        rel=0,
        abs=1e-15,
    )


# ---------------------------------------------------------------------------
# Resamplers
# ---------------------------------------------------------------------------


def test_circular_block_indices_have_the_right_shape_and_wrap(module):
    rng = np.random.default_rng(0)
    drawn = module.circular_block_indices(50, 200, 7, rng)
    assert drawn.shape == (200, 50)
    assert drawn.min() >= 0 and drawn.max() < 50
    # every block is a contiguous run modulo T
    blocks = module.circular_block_indices(50, 200, 7, np.random.default_rng(0))[:, :7]
    steps = np.diff(blocks, axis=1) % 50
    assert (steps == 1).all()


def test_moving_block_indices_never_wrap(module):
    rng = np.random.default_rng(1)
    drawn = module.moving_block_indices(50, 200, 7, rng)
    assert drawn.shape == (200, 50)
    assert drawn.min() >= 0 and drawn.max() < 50
    blocks = drawn[:, :7]
    assert (np.diff(blocks, axis=1) == 1).all()


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


def test_studentized_interval_is_centred_on_delta_and_p_value_is_a_probability(
    module,
):
    series = _series(module, observations=400, seed=11)
    result = module.studentized_bootstrap(
        series, "repo", "circular_block", 20, 60, seed=123
    )
    midpoint = 0.5 * (result.symmetric_low + result.symmetric_high)
    assert midpoint == pytest.approx(result.delta, rel=0, abs=1e-12)
    assert result.symmetric_low < result.symmetric_high
    assert 0.0 < result.p_value_symmetric <= 1.0
    assert 0.0 <= result.positive_fraction <= 1.0
    assert result.observations == series.observations


def test_studentized_bootstrap_is_deterministic_given_its_seed(module):
    series = _series(module, observations=400, seed=11)
    first = module.studentized_bootstrap(
        series, "repo", "circular_block", 20, 40, seed=99
    )
    second = module.studentized_bootstrap(
        series, "repo", "circular_block", 20, 40, seed=99
    )
    assert first == second
