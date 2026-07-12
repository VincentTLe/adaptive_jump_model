from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adaptive_jump.config import load_config
from adaptive_jump.features import (
    FeatureError,
    align_cash_returns,
    effective_oos_start,
    equity_returns,
    make_features,
    prepare_market,
)

CONFIG = load_config(Path(__file__).resolve().parents[1] / "research.toml")


def canonical(dates, values) -> pd.DataFrame:
    return pd.DataFrame({"date": dates, "value": values})


def test_equity_returns_drop_missing_level_and_record_gap() -> None:
    levels = canonical(
        ["2023-01-02", "2023-01-03", "2023-01-05", "2023-01-06"],
        [100.0, None, 110.0, 121.0],
    )

    result = equity_returns(levels)

    assert result["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2023-01-05",
        "2023-01-06",
    ]
    np.testing.assert_allclose(result["equity_simple"], [0.1, 0.1])
    np.testing.assert_allclose(result["equity_log"], np.log([1.1, 1.1]))
    assert result["gap_calendar_days"].tolist() == [3.0, 1.0]


def test_equity_returns_reject_nonpositive_level() -> None:
    levels = canonical(["2023-01-02", "2023-01-03"], [100.0, 0.0])

    with pytest.raises(FeatureError, match="positive observations"):
        equity_returns(levels)


def test_daily_cash_alignment_is_lagged_and_staleness_bounded() -> None:
    source = CONFIG.markets[0].cash
    cash = canonical(["2023-01-01"], [2.52])
    dates = pd.Series(
        pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-12", "2023-01-13"])
    )

    result = align_cash_returns(dates, cash, source, 252)

    assert pd.isna(result.loc[0, "cash_return"])
    assert result.loc[1, "cash_return"] == pytest.approx(0.0001)
    assert result.loc[2, "cash_age_days"] == 10
    assert result.loc[2, "cash_return"] == pytest.approx(0.0001)
    assert pd.isna(result.loc[3, "cash_return"])


def test_monthly_cash_alignment_uses_second_following_month() -> None:
    source = CONFIG.markets[1].cash
    cash = canonical(["2023-01-01"], [-2.52])
    dates = pd.Series(
        pd.to_datetime(["2023-02-28", "2023-03-01", "2023-06-29", "2023-06-30"])
    )

    result = align_cash_returns(dates, cash, source, 252)

    assert pd.isna(result.loc[0, "cash_return"])
    assert result.loc[1, "cash_return"] == pytest.approx(-0.0001)
    assert result.loc[2, "cash_age_days"] == 120
    assert result.loc[2, "cash_return"] == pytest.approx(-0.0001)
    assert pd.isna(result.loc[3, "cash_return"])


def test_features_match_frozen_pandas_ewm_formulas() -> None:
    excess = pd.Series([-0.02, 0.01, -0.01, 0.03])

    actual = make_features(excess)
    negative_squared = excess.clip(upper=0).pow(2)
    expected_dd = np.sqrt(
        negative_squared.ewm(halflife=10, adjust=True, ignore_na=False).mean()
    )
    expected_dd20 = np.sqrt(
        negative_squared.ewm(halflife=20, adjust=True, ignore_na=False).mean()
    )
    expected_dd60 = np.sqrt(
        negative_squared.ewm(halflife=60, adjust=True, ignore_na=False).mean()
    )

    np.testing.assert_allclose(actual["dd_10"], expected_dd)
    np.testing.assert_allclose(
        actual["sortino_20"],
        excess.ewm(halflife=20, adjust=True, ignore_na=False).mean() / expected_dd20,
    )
    np.testing.assert_allclose(
        actual["sortino_60"],
        excess.ewm(halflife=60, adjust=True, ignore_na=False).mean() / expected_dd60,
    )


def test_zero_downside_and_missing_excess_never_create_infinity() -> None:
    result = make_features(pd.Series([0.01, 0.02, None]))

    assert result["dd_10"].iloc[:2].eq(0).all()
    assert result[["sortino_20", "sortino_60"]].isna().all().all()
    assert result.iloc[2].isna().all()


def test_future_changes_do_not_alter_past_market_features() -> None:
    dates = pd.date_range("2023-01-01", periods=12, freq="D")
    equity = canonical(dates.strftime("%Y-%m-%d"), np.linspace(100, 111, 12))
    cash = canonical(dates.strftime("%Y-%m-%d"), np.linspace(1, 2, 12))
    market = CONFIG.markets[0]
    baseline = prepare_market(equity, cash, market, CONFIG)

    changed_equity = equity.copy()
    changed_cash = cash.copy()
    changed_equity.loc[changed_equity.index[-1], "value"] = 999.0
    changed_cash.loc[changed_cash.index[-1], "value"] = 99.0
    changed = prepare_market(changed_equity, changed_cash, market, CONFIG)

    pd.testing.assert_frame_equal(baseline.iloc[:-1], changed.iloc[:-1])


def test_effective_oos_start_uses_3000_rows_plus_eight_years() -> None:
    dates = pd.bdate_range("1990-01-01", periods=9000)
    frame = pd.DataFrame(
        {
            "date": dates,
            "dd_10": 1.0,
            "sortino_20": 1.0,
            "sortino_60": 1.0,
        }
    )
    target = dates[2999] + pd.DateOffset(years=8)
    expected = dates[dates >= target][0].date()

    assert effective_oos_start(frame) == expected
    assert effective_oos_start(frame.iloc[:2000]) is None
