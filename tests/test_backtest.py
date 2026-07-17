import numpy as np
import pandas as pd
import pytest

from adaptive_jump.backtest import (
    BacktestError,
    annualized_excess_sharpe,
    apply_signal,
    buy_and_hold,
    performance_metrics,
)


def return_frame(rows: int = 8) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2023-01-02", periods=rows),
            "equity_simple": np.linspace(0.01, 0.02, rows),
            "cash_return": 0.001,
        }
    )


def test_one_day_delay_applies_signal_to_t_plus_two() -> None:
    returns = return_frame(6)
    signal = pd.Series([1, 0, 1, 0, 1, 0])

    result = apply_signal(returns, signal, delay_trading_days=1)

    expected = pd.Series([np.nan, np.nan, 1.0, 0.0, 1.0, 0.0])
    pd.testing.assert_series_equal(result["position"], expected, check_names=False)
    assert result.loc[2, "gross_return"] == returns.loc[2, "equity_simple"]
    assert result.loc[3, "gross_return"] == returns.loc[3, "cash_return"]


def test_first_allocation_is_free_then_each_switch_costs_ten_bps() -> None:
    returns = return_frame(6)
    signal = pd.Series([1, 0, 1, 0, 1, 0])

    result = apply_signal(returns, signal)

    assert result.loc[2, "one_way_turnover"] == 0.0
    assert result.loc[2, "transaction_cost"] == 0.0
    assert result.loc[3:, "transaction_cost"].eq(0.001).all()
    assert result.loc[3, "strategy_return"] == pytest.approx(0.0)


def test_initial_allocation_cost_can_be_enabled_explicitly() -> None:
    result = apply_signal(
        return_frame(4),
        pd.Series([1, 1, 1, 1]),
        charge_initial_allocation=True,
    )

    assert result.loc[2, "transaction_cost"] == 0.001


def test_five_day_delay_uses_six_observation_offset() -> None:
    result = apply_signal(
        return_frame(8), pd.Series([1, 0, 0, 0, 0, 0, 0, 0]), delay_trading_days=5
    )

    assert result["position"].first_valid_index() == 6
    assert result.loc[6, "position"] == 1


def test_missing_inputs_produce_no_strategy_observation() -> None:
    returns = return_frame(6)
    returns.loc[4, "cash_return"] = np.nan
    signal = pd.Series([1, 0, np.nan, 1, 0, 1])

    result = apply_signal(returns, signal)

    assert pd.isna(result.loc[4, "position"])
    assert pd.isna(result.loc[4, "strategy_return"])


def test_future_signal_change_does_not_change_past_accounting() -> None:
    returns = return_frame(8)
    baseline_signal = pd.Series([1, 1, 1, 1, 1, 1, 1, 1])
    changed_signal = baseline_signal.copy()
    changed_signal.loc[4] = 0

    baseline = apply_signal(returns, baseline_signal)
    changed = apply_signal(returns, changed_signal)

    accounting = [
        "position",
        "gross_return",
        "one_way_turnover",
        "transaction_cost",
        "strategy_return",
    ]
    pd.testing.assert_frame_equal(
        baseline.loc[:5, accounting], changed.loc[:5, accounting]
    )
    assert baseline.loc[6, "position"] != changed.loc[6, "position"]


@pytest.mark.parametrize(
    ("signal", "message"),
    [
        (pd.Series([1, 2, 1]), "signal values"),
        (pd.Series([1, 0]), "lengths must match"),
    ],
)
def test_invalid_signal_fails(signal, message) -> None:
    with pytest.raises(BacktestError, match=message):
        apply_signal(return_frame(3), signal)


def test_buy_and_hold_is_fully_invested_without_cost() -> None:
    returns = return_frame(4)

    result = buy_and_hold(returns)

    pd.testing.assert_series_equal(
        result["strategy_return"], returns["equity_simple"], check_names=False
    )
    assert result["position"].eq(1.0).all()
    assert result["transaction_cost"].eq(0.0).all()


def test_frozen_performance_metric_definitions() -> None:
    result = pd.DataFrame(
        {
            "date": pd.bdate_range("2023-01-02", periods=4),
            "cash_return": 0.0,
            "position": [1.0, 0.0, 0.0, 1.0],
            "one_way_turnover": [0.0, 1.0, 0.0, 1.0],
            "strategy_return": [0.10, -0.10, 0.05, 0.0],
        }
    )

    metrics = performance_metrics(
        result, periods_per_year=4, expected_shortfall_quantile=0.25
    )

    expected_cagr = np.prod(1.0 + result["strategy_return"]) - 1.0
    expected_vol = result["strategy_return"].std(ddof=1) * 2
    assert metrics["cagr"] == pytest.approx(expected_cagr)
    assert metrics["volatility"] == pytest.approx(expected_vol)
    assert metrics["sharpe"] == pytest.approx(2 * 0.0125 / (expected_vol / 2))
    assert metrics["maximum_drawdown"] == pytest.approx(-0.10)
    assert metrics["calmar"] == pytest.approx(0.5)
    assert metrics["expected_shortfall_5pct"] == pytest.approx(-0.10)
    assert metrics["turnover"] == pytest.approx(1.0)
    assert metrics["leverage"] == pytest.approx(0.5)


def test_turnover_convention_does_not_change_full_switch_costs() -> None:
    result = apply_signal(
        return_frame(6),
        pd.Series([1, 0, 1, 0, 1, 0]),
        delay_trading_days=1,
        one_way_cost_bps=10,
    )

    paper = performance_metrics(result, periods_per_year=4)
    legacy = performance_metrics(result, periods_per_year=4, turnover_scale=1.0)

    assert result["one_way_turnover"].sum() == pytest.approx(3.0)
    assert result["transaction_cost"].sum() == pytest.approx(0.003)
    assert paper["turnover"] == pytest.approx(1.5)
    assert legacy["turnover"] == pytest.approx(3.0)


def test_sharpe_uses_strategy_volatility_not_excess_volatility() -> None:
    strategy = pd.Series([0.01, 0.03, -0.01])
    cash = pd.Series([0.00, 0.01, 0.00])

    value = annualized_excess_sharpe(strategy, cash, periods_per_year=4)

    expected = 2 * (strategy - cash).mean() / strategy.std(ddof=1)
    assert value == pytest.approx(expected)


def test_zero_strategy_volatility_has_no_sharpe() -> None:
    value = annualized_excess_sharpe(pd.Series([0.01, 0.01]), pd.Series([0.0, 0.0]))

    assert np.isnan(value)


def test_metrics_reject_missing_position_on_a_return_date() -> None:
    result = buy_and_hold(return_frame(3))
    result.loc[1, "position"] = np.nan

    with pytest.raises(BacktestError, match="metric inputs must be finite"):
        performance_metrics(result)
