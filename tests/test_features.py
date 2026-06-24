import math

import pandas as pd
import pytest

from adaptive_jump.features import aggregate_tick_to_minutes


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
