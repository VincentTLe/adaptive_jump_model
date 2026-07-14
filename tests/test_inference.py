import numpy as np
import pandas as pd
import pytest

from adaptive_jump.backtest import annualized_excess_sharpe
from adaptive_jump.inference import (
    InferenceError,
    bootstrap_sharpe_delta,
    stationary_bootstrap_indices,
)


def _returns() -> tuple[pd.Series, pd.Series, pd.Series]:
    rng = np.random.default_rng(42)
    baseline = pd.Series(rng.normal(0.0003, 0.01, 320))
    challenger = baseline + pd.Series(rng.normal(0.0001, 0.002, 320))
    cash = pd.Series(np.full(320, 0.00005))
    return challenger, baseline, cash


def test_stationary_bootstrap_indices_are_paired_circular_paths() -> None:
    indices = stationary_bootstrap_indices(7, 4, 7, np.random.default_rng(3))

    assert indices.shape == (4, 7)
    assert ((0 <= indices) & (indices < 7)).all()
    continued = (indices[:, :-1] + 1) % 7
    assert (indices[:, 1:] == continued).any()


def test_bootstrap_observed_delta_uses_frozen_sharpe_definition() -> None:
    challenger, baseline, cash = _returns()

    result = bootstrap_sharpe_delta(
        challenger,
        baseline,
        cash,
        replications=200,
        mean_block_length=20,
        seed=9,
    )

    expected = annualized_excess_sharpe(challenger, cash) - annualized_excess_sharpe(
        baseline, cash
    )
    assert result.observed == pytest.approx(expected)
    assert result.confidence_low <= result.confidence_high
    assert result.replications == 200


def test_bootstrap_is_deterministic_and_preserves_zero_paired_delta() -> None:
    _, baseline, cash = _returns()
    arguments = dict(
        challenger_return=baseline,
        baseline_return=baseline,
        cash_return=cash,
        replications=120,
        mean_block_length=12,
        seed=17,
    )

    first = bootstrap_sharpe_delta(**arguments)
    second = bootstrap_sharpe_delta(**arguments)

    assert first == second
    assert first.observed == pytest.approx(0.0)
    assert first.lower_one_sided == pytest.approx(0.0)
    assert first.confidence_low == pytest.approx(0.0)
    assert first.confidence_high == pytest.approx(0.0)


def test_bootstrap_uses_requested_volatility_ddof_in_every_draw() -> None:
    challenger, baseline, cash = _returns()
    arguments = dict(
        challenger_return=challenger,
        baseline_return=baseline,
        cash_return=cash,
        replications=120,
        mean_block_length=12,
        seed=17,
    )

    population = bootstrap_sharpe_delta(**arguments, volatility_ddof=0)
    sample = bootstrap_sharpe_delta(**arguments, volatility_ddof=1)

    assert population.observed != pytest.approx(sample.observed)
    assert population.confidence_low != pytest.approx(sample.confidence_low)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"replications": 0}, "replication"),
        ({"mean_block_length": 500}, "block length"),
        ({"confidence_level": 1.0}, "confidence level"),
    ],
)
def test_bootstrap_rejects_invalid_contract(overrides, message) -> None:
    challenger, baseline, cash = _returns()
    arguments = dict(
        challenger_return=challenger,
        baseline_return=baseline,
        cash_return=cash,
        replications=20,
        mean_block_length=10,
        seed=1,
    )
    arguments.update(overrides)

    with pytest.raises(InferenceError, match=message):
        bootstrap_sharpe_delta(**arguments)


def test_bootstrap_rejects_nonfinite_or_constant_strategy() -> None:
    challenger, baseline, cash = _returns()
    challenger.iloc[0] = np.nan
    with pytest.raises(InferenceError, match="finite"):
        bootstrap_sharpe_delta(
            challenger,
            baseline,
            cash,
            replications=20,
            mean_block_length=10,
            seed=1,
        )

    constant = pd.Series(np.zeros(30))
    with pytest.raises(InferenceError, match="observed Sharpe delta"):
        bootstrap_sharpe_delta(
            constant,
            constant,
            pd.Series(np.zeros(30)),
            replications=20,
            mean_block_length=10,
            seed=1,
        )


def test_bootstrap_rejects_misaligned_indices() -> None:
    challenger, baseline, cash = _returns()
    baseline.index = baseline.index + 1

    with pytest.raises(InferenceError, match="identical aligned indices"):
        bootstrap_sharpe_delta(
            challenger,
            baseline,
            cash,
            replications=20,
            mean_block_length=10,
            seed=1,
        )
