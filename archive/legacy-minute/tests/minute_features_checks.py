import math

import numpy as np
import pandas as pd
import pytest

from adaptive_jump.features import aggregate_tick_to_minutes, make_minute_features_from_minute_bidask, zscore_series


def test_aggregate_tick_to_minutes_computes_minute_bars_and_standard_rv():
    df_tick = pd.DataFrame(
        {
            "price": [100.0, 101.0, 99.0, 102.0],
            "bid": [99.9, 100.9, 98.8, 101.8],
            "ask": [100.1, 101.1, 99.2, 102.2],
            "size": [10, 20, 30, 40],
            "mid": [100.0, 101.0, 99.0, 102.0],
            "spread": [0.2, 0.2, 0.4, 0.4],
            "rel_spread": [0.002, 0.001980198, 0.004040404, 0.003921569],
        },
        index=pd.to_datetime(
            [
                "2024-01-02 09:30:00",
                "2024-01-02 09:30:15",
                "2024-01-02 09:30:45",
                "2024-01-02 09:31:05",
            ]
        ),
    )

    result = aggregate_tick_to_minutes(df_tick)

    first = result.loc[pd.Timestamp("2024-01-02 09:30:00")]
    assert first["open"] == pytest.approx(100.0)
    assert first["high"] == pytest.approx(101.0)
    assert first["low"] == pytest.approx(99.0)
    assert first["close"] == pytest.approx(99.0)
    assert first["volume"] == pytest.approx(60)
    assert first["trade_count"] == 3
    assert first["bid_close"] == pytest.approx(98.8)
    assert first["ask_close"] == pytest.approx(99.2)
    assert first["mid_close"] == pytest.approx(99.0)
    assert first["spread_close"] == pytest.approx(0.4)
    assert first["rel_spread_close"] == pytest.approx(0.004040404)

    first_rv = math.log(101.0 / 100.0) ** 2 + math.log(99.0 / 101.0) ** 2
    assert first["realized_var"] == pytest.approx(first_rv)

    second = result.loc[pd.Timestamp("2024-01-02 09:31:00")]
    assert second["trade_count"] == 1
    assert second["realized_var"] == pytest.approx(math.log(102.0 / 99.0) ** 2)


def test_aggregate_tick_to_minutes_resets_rv_across_dates():
    df_tick = _tick_frame(
        prices=[100.0, 110.0, 121.0],
        index=[
            "2024-01-02 16:00:00",
            "2024-01-03 09:30:00",
            "2024-01-03 09:30:30",
        ],
    )

    result = aggregate_tick_to_minutes(df_tick)

    assert result.loc[pd.Timestamp("2024-01-02 16:00:00"), "realized_var"] == pytest.approx(0.0)
    assert result.loc[pd.Timestamp("2024-01-03 09:30:00"), "realized_var"] == pytest.approx(
        math.log(121.0 / 110.0) ** 2
    )


def test_aggregate_tick_to_minutes_preserves_equal_timestamp_order():
    df_tick = _tick_frame(
        prices=[100.0, 101.0, 99.0],
        index=[
            "2024-01-02 09:30:00",
            "2024-01-02 09:30:00",
            "2024-01-02 09:30:30",
        ],
    )

    result = aggregate_tick_to_minutes(df_tick)

    row = result.loc[pd.Timestamp("2024-01-02 09:30:00")]
    assert row["open"] == pytest.approx(100.0)
    assert row["close"] == pytest.approx(99.0)
    expected_rv = math.log(101.0 / 100.0) ** 2 + math.log(99.0 / 101.0) ** 2
    assert row["realized_var"] == pytest.approx(expected_rv)


def test_aggregate_tick_to_minutes_requires_expected_columns():
    df_tick = pd.DataFrame({"price": [100.0]}, index=pd.to_datetime(["2024-01-02 09:30:00"]))

    with pytest.raises(ValueError, match="missing required columns"):
        aggregate_tick_to_minutes(df_tick)


def test_aggregate_tick_to_minutes_requires_valid_numeric_inputs():
    df_tick = _tick_frame(prices=[100.0, 0.0], index=["2024-01-02 09:30:00", "2024-01-02 09:30:10"])

    with pytest.raises(ValueError, match="must be positive"):
        aggregate_tick_to_minutes(df_tick)


def test_aggregate_tick_to_minutes_requires_datetime_index():
    df_tick = _tick_frame(prices=[100.0], index=["2024-01-02 09:30:00"])
    df_tick.index = [0]

    with pytest.raises(TypeError, match="DatetimeIndex"):
        aggregate_tick_to_minutes(df_tick)


def test_zscore_series_standardizes_values():
    result = zscore_series(pd.Series([1.0, 2.0, 3.0]))

    assert result.mean() == pytest.approx(0.0)
    assert result.std() == pytest.approx(1.0)


def test_zscore_series_handles_flat_values_and_preserves_nan():
    result = zscore_series(pd.Series([np.nan, 5.0, 5.0]))

    assert pd.isna(result.iloc[0])
    assert list(result.iloc[1:]) == [0.0, 0.0]


def test_zscore_series_preserves_nan_with_nonflat_values():
    result = zscore_series(pd.Series([np.nan, 1.0, 2.0, 3.0]))

    assert pd.isna(result.iloc[0])
    assert result.dropna().mean() == pytest.approx(0.0)
    assert result.dropna().std() == pytest.approx(1.0)


def test_make_minute_features_from_minute_bidask_computes_core_formulas():
    index = pd.date_range("2024-01-02 09:30", periods=25, freq="min")
    df = pd.DataFrame(
        {
            "mid_close": [100.0 + i + (i % 3) * 0.1 for i in range(25)],
            "volume": [10 + i for i in range(25)],
            "spread_close": [0.1 + i * 0.01 for i in range(25)],
            "rel_spread_close": [0.001 + i * 0.0001 for i in range(25)],
        },
        index=index,
    )

    result = make_minute_features_from_minute_bidask(df)
    expected_mid_return = np.log(df["mid_close"] / df["mid_close"].shift(1))
    expected_log_volume = np.log1p(df["volume"])
    expected_rolling_vol_5 = expected_mid_return.rolling(5).std()
    expected_rolling_vol_20 = expected_mid_return.rolling(20).std()
    expected_noise = _manual_zscore(df["rel_spread_close"]) - _manual_zscore(expected_log_volume)
    expected_shock = _manual_zscore(expected_mid_return).abs() + _manual_zscore(expected_rolling_vol_20)

    assert list(result.columns) == [
        "mid_return",
        "abs_mid_return",
        "volume",
        "log_volume",
        "spread_close",
        "rel_spread_close",
        "rolling_vol_5",
        "rolling_vol_20",
        "noise_score_raw",
        "shock_score_raw",
    ]
    pd.testing.assert_series_equal(result["mid_return"], expected_mid_return, check_names=False)
    pd.testing.assert_series_equal(result["rolling_vol_5"], expected_rolling_vol_5, check_names=False)
    pd.testing.assert_series_equal(result["rolling_vol_20"], expected_rolling_vol_20, check_names=False)
    pd.testing.assert_series_equal(result["noise_score_raw"], expected_noise, check_names=False)
    pd.testing.assert_series_equal(result["shock_score_raw"], expected_shock, check_names=False)


def test_make_minute_features_from_minute_bidask_resets_returns_across_dates():
    df = pd.DataFrame(
        {
            "mid_close": [100.0, 110.0, 121.0],
            "volume": [10, 20, 30],
            "spread_close": [0.1, 0.1, 0.2],
            "rel_spread_close": [0.001, 0.001, 0.002],
        },
        index=pd.to_datetime(["2024-01-02 16:00", "2024-01-03 09:30", "2024-01-03 09:31"]),
    )

    result = make_minute_features_from_minute_bidask(df)

    assert pd.isna(result.iloc[0]["mid_return"])
    assert pd.isna(result.iloc[1]["mid_return"])
    assert result.iloc[2]["mid_return"] == pytest.approx(math.log(121.0 / 110.0))


def test_make_minute_features_from_minute_bidask_requires_columns():
    df = pd.DataFrame({"mid_close": [100.0]})

    with pytest.raises(ValueError, match="missing required columns"):
        make_minute_features_from_minute_bidask(df)


def test_make_minute_features_from_minute_bidask_requires_valid_inputs():
    df = pd.DataFrame(
        {
            "mid_close": [100.0, 0.0],
            "volume": [10, 20],
            "spread_close": [0.1, 0.2],
            "rel_spread_close": [0.001, 0.002],
        },
        index=pd.date_range("2024-01-02 09:30", periods=2, freq="min"),
    )

    with pytest.raises(ValueError, match="mid_close must be positive"):
        make_minute_features_from_minute_bidask(df)


def _tick_frame(prices: list[float], index: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "price": prices,
            "bid": [price - 0.1 for price in prices],
            "ask": [price + 0.1 for price in prices],
            "size": [10] * len(prices),
            "mid": prices,
            "spread": [0.2] * len(prices),
            "rel_spread": [0.002] * len(prices),
        },
        index=pd.to_datetime(index),
    )


def _manual_zscore(s: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=s.index, dtype=float)
    valid = s.dropna()
    std = valid.std()
    if pd.isna(std) or std < 1e-12:
        result.loc[valid.index] = 0.0
    else:
        result.loc[valid.index] = (valid - valid.mean()) / std
    return result
