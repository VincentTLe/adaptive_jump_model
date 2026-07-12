import numpy as np
import pandas as pd
import pytest

from adaptive_jump.backtest import BacktestError, apply_signal


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
