import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adaptive_jump.features import aggregate_tick_to_minutes
from scripts.prepare_processed_data import (
    _aggregate_tick_chunk,
    _combine_minute_partials,
    _inventory_row,
    _json_ready,
    _ordered_output,
    _process_tick_bidask_chunked,
)


def test_chunked_tick_aggregation_matches_in_memory_aggregation():
    df_tick = _tick_frame(
        prices=[100.0, 101.0, 99.0, 110.0, 121.0],
        index=[
            "2024-01-02 09:30:00",
            "2024-01-02 09:30:30",
            "2024-01-02 09:31:00",
            "2024-01-03 09:30:00",
            "2024-01-03 09:30:30",
        ],
    )

    first, prev_price, prev_timestamp = _aggregate_tick_chunk(df_tick.iloc[:2], float("nan"), pd.NaT)
    second, prev_price, prev_timestamp = _aggregate_tick_chunk(df_tick.iloc[2:4], prev_price, prev_timestamp)
    third, _, _ = _aggregate_tick_chunk(df_tick.iloc[4:], prev_price, prev_timestamp)

    result = _combine_minute_partials([first, second, third])
    expected = aggregate_tick_to_minutes(df_tick)

    pd.testing.assert_frame_equal(result, expected)
    assert result.loc[pd.Timestamp("2024-01-03 09:30:00"), "realized_var"] == pytest.approx(
        math.log(121.0 / 110.0) ** 2
    )


def test_tick_processing_raises_if_chunks_move_backward(tmp_path):
    csv_path = tmp_path / "bad_tick_order.csv"
    csv_path.write_text(
        "\n".join(
            [
                "01/02/2024,09:31:00,101.0,100.9,101.1,10",
                "01/02/2024,09:32:00,102.0,101.9,102.1,10",
                "01/02/2024,09:30:00,100.0,99.9,100.1,10",
            ]
        )
    )

    with pytest.raises(ValueError, match="moved backward across chunks"):
        _process_tick_bidask_chunked(csv_path, "TEST", chunksize=2)


def test_inventory_row_flattens_loader_and_chunk_drop_counts(tmp_path):
    raw_path = tmp_path / "raw.csv"
    raw_path.write_text("raw")
    index = pd.date_range("2024-01-02 09:30", periods=2, freq="min")
    df = pd.DataFrame(
        {
            "has_bidask": [1, 1],
            "return": [float("nan"), 0.01],
            "mid_return": [float("nan"), 0.01],
            "rolling_vol_20": [0.0, 0.1],
            "noise_score_raw": [1.0, 2.0],
            "shock_score_raw": [0.5, 0.6],
        },
        index=index,
    )
    df.attrs["dropped_rows"] = {"crossed_tick_quotes": 2}
    df.attrs["dropped_outside_regular_session"] = 3

    row = _inventory_row("TEST", "tick_bidask", raw_path, Path(tmp_path / "out.csv.gz"), False, 1.5, df)

    assert row["dropped_crossed_tick_quotes"] == 2
    assert row["dropped_outside_regular_session"] == 3
    assert row["raw_bytes"] == 3


def test_ordered_output_adds_session_fields_and_gap_minutes():
    df = pd.DataFrame(
        {
            "symbol": ["TEST"] * 4,
            "data_type": ["adjusted_ohlcv"] * 4,
            "has_bidask": [0] * 4,
        },
        index=pd.to_datetime(
            [
                "2024-01-02 09:30:00",
                "2024-01-02 09:31:00",
                "2024-01-02 09:34:00",
                "2024-01-03 09:30:00",
            ]
        ),
    )

    result = _ordered_output(df)

    assert list(result["session_date"]) == ["2024-01-02", "2024-01-02", "2024-01-02", "2024-01-03"]
    assert list(result["bar_index"]) == [0, 1, 2, 0]
    assert math.isnan(result.iloc[0]["gap_minutes"])
    assert result.iloc[1]["gap_minutes"] == pytest.approx(1.0)
    assert result.iloc[2]["gap_minutes"] == pytest.approx(3.0)
    assert math.isnan(result.iloc[3]["gap_minutes"])


def test_json_ready_converts_nonfinite_values_to_strict_json_null():
    payload = {
        "nan": float("nan"),
        "inf": np.float64(np.inf),
        "missing": pd.NA,
        "nested": [{"ok": np.int64(3), "bad": np.float64(np.nan)}],
    }

    result = _json_ready(payload)

    assert result == {"nan": None, "inf": None, "missing": None, "nested": [{"ok": 3, "bad": None}]}
    json.dumps(result, allow_nan=False)


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
