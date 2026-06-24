import math

import pandas as pd
import pytest

from adaptive_jump.features import aggregate_tick_to_minutes, make_minute_features_from_minute_bidask, zscore_series


def test_aggregate_tick_to_minutes_computes_minute_bars():
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

    assert list(result.index) == [
        pd.Timestamp("2024-01-02 09:30:00"),
        pd.Timestamp("2024-01-02 09:31:00"),
    ]
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

    expected_rv = math.log(101.0 / 100.0) ** 2 + math.log(99.0 / 101.0) ** 2
    assert first["realized_var"] == pytest.approx(expected_rv)

    second = result.loc[pd.Timestamp("2024-01-02 09:31:00")]
    assert second["trade_count"] == 1
    assert second["realized_var"] == pytest.approx(0.0)


def test_aggregate_tick_to_minutes_requires_expected_columns():
    df_tick = pd.DataFrame({"price": [100.0]}, index=pd.to_datetime(["2024-01-02 09:30:00"]))

    with pytest.raises(ValueError, match="missing required columns"):
        aggregate_tick_to_minutes(df_tick)


def test_aggregate_tick_to_minutes_requires_datetime_index():
    df_tick = pd.DataFrame(
        {
            "price": [100.0],
            "bid": [99.9],
            "ask": [100.1],
            "size": [10],
            "mid": [100.0],
            "spread": [0.2],
            "rel_spread": [0.002],
        }
    )

    with pytest.raises(TypeError, match="DatetimeIndex"):
        aggregate_tick_to_minutes(df_tick)


def test_zscore_series_standardizes_values():
    result = zscore_series(pd.Series([1.0, 2.0, 3.0]))

    assert result.mean() == pytest.approx(0.0)
    assert result.std() == pytest.approx(1.0)


def test_zscore_series_handles_flat_values():
    result = zscore_series(pd.Series([5.0, 5.0, 5.0]))

    assert list(result) == [0.0, 0.0, 0.0]


def test_make_minute_features_from_minute_bidask_computes_core_columns():
    df = pd.DataFrame(
        {
            "mid_close": [100.0, 101.0, 99.0, 102.0, 103.0],
            "volume": [0, 10, 20, 30, 40],
            "spread_close": [0.1, 0.2, 0.2, 0.3, 0.4],
            "rel_spread_close": [0.001, 0.002, 0.0021, 0.003, 0.004],
        },
        index=pd.date_range("2024-01-02 09:30", periods=5, freq="min"),
    )

    result = make_minute_features_from_minute_bidask(df)

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
    assert pd.isna(result.iloc[0]["mid_return"])
    assert result.iloc[1]["mid_return"] == pytest.approx(math.log(101.0 / 100.0))
    assert result.iloc[2]["abs_mid_return"] == pytest.approx(abs(math.log(99.0 / 101.0)))
    assert result.iloc[1]["volume"] == pytest.approx(10)
    assert result.iloc[1]["log_volume"] == pytest.approx(math.log1p(10))
    assert "rolling_vol_5" in result.columns
    assert "rolling_vol_20" in result.columns
    assert "noise_score_raw" in result.columns
    assert "shock_score_raw" in result.columns


def test_make_minute_features_from_minute_bidask_requires_columns():
    df = pd.DataFrame({"mid_close": [100.0]})

    with pytest.raises(ValueError, match="missing required columns"):
        make_minute_features_from_minute_bidask(df)
