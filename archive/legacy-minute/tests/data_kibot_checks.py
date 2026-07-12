import pandas as pd
import pytest

from adaptive_jump.data_kibot import (
    load_kibot_adjusted_ohlcv_csv,
    load_kibot_minute_bidask_csv,
    load_kibot_tick_bidask_csv,
)


def test_load_kibot_tick_bidask_csv_sorts_stably_and_computes_fields(tmp_path):
    csv_path = tmp_path / "IVE_tick_bidask.csv"
    csv_path.write_text(
        "\n".join(
            [
                "09/28/2009,09:31:00,10.00,9.90,10.10,100",
                "09/28/2009,09:30:00,20.00,19.80,20.20,200",
                "09/28/2009,09:30:00,21.00,20.90,21.10,300",
                "09/28/2009,08:00:00,9.00,8.90,9.10,50",
            ]
        )
    )

    df = load_kibot_tick_bidask_csv(str(csv_path))

    assert list(df.index) == [
        pd.Timestamp("2009-09-28 09:30:00"),
        pd.Timestamp("2009-09-28 09:30:00"),
        pd.Timestamp("2009-09-28 09:31:00"),
    ]
    assert list(df["price"]) == [20.0, 21.0, 10.0]
    assert df.index.name == "timestamp"
    assert list(df.columns) == ["price", "bid", "ask", "size", "mid", "spread", "rel_spread"]
    assert df.iloc[0]["mid"] == pytest.approx(20.0)
    assert df.iloc[0]["spread"] == pytest.approx(0.4)
    assert df.iloc[0]["rel_spread"] == pytest.approx(0.02)


def test_load_kibot_tick_bidask_csv_records_dropped_market_rows(tmp_path):
    csv_path = tmp_path / "tick_with_bad_market_rows.csv"
    csv_path.write_text(
        "\n".join(
            [
                "09/28/2009,09:30:00,10.00,9.90,10.10,100",
                "09/28/2009,09:31:00,15.00,15.20,15.10,100",
                "09/28/2009,09:32:00,0.00,9.90,10.10,100",
            ]
        )
    )

    df = load_kibot_tick_bidask_csv(str(csv_path))

    assert list(df["price"]) == [10.0]
    assert df.attrs["dropped_rows"] == {
        "nonpositive_tick_values": 1,
        "crossed_tick_quotes": 1,
    }


def test_load_kibot_minute_bidask_csv_sorts_filters_session_and_computes_fields(tmp_path):
    csv_path = tmp_path / "IVE_minute_bidask.csv"
    csv_path.write_text(
        "\n".join(
            [
                "09/28/2009,09:31,10.00,10.40,9.90,10.20,1000,10.01,10.31,9.81,10.11,10.03,10.33,9.83,10.13",
                "09/28/2009,09:30,20.00,20.40,19.90,20.20,2000,20.01,20.31,19.81,20.11,20.03,20.33,19.83,20.13",
                "09/28/2009,08:00,9.00,9.40,8.90,9.20,500,9.01,9.31,8.81,9.11,9.03,9.33,8.83,9.13",
            ]
        )
    )

    df = load_kibot_minute_bidask_csv(str(csv_path))

    assert list(df.index) == [
        pd.Timestamp("2009-09-28 09:30:00"),
        pd.Timestamp("2009-09-28 09:31:00"),
    ]
    row = df.loc[pd.Timestamp("2009-09-28 09:30:00")]
    assert row["mid_close"] == pytest.approx(20.12)
    assert row["spread_close"] == pytest.approx(0.02)
    assert row["rel_spread_close"] == pytest.approx(0.02 / 20.12)
    assert row["volume"] == pytest.approx(2000)


def test_load_kibot_minute_bidask_csv_records_dropped_market_rows(tmp_path):
    csv_path = tmp_path / "minute_with_bad_market_rows.csv"
    csv_path.write_text(
        "\n".join(
            [
                "09/28/2009,09:30,10.00,10.40,9.90,10.20,1000,10.01,10.31,9.81,10.11,10.03,10.33,9.83,10.13",
                "09/28/2009,09:31,10.00,10.40,9.90,10.20,-1,10.01,10.31,9.81,10.11,10.03,10.33,9.83,10.13",
                "09/28/2009,09:32,10.00,10.40,9.90,10.20,1000,10.01,10.31,9.81,10.11,10.03,10.33,9.80,10.13",
            ]
        )
    )

    df = load_kibot_minute_bidask_csv(str(csv_path))

    assert list(df.index) == [pd.Timestamp("2009-09-28 09:30:00")]
    assert df.attrs["dropped_rows"] == {
        "negative_minute_bidask_volume": 1,
        "crossed_minute_low_quotes": 1,
    }


def test_load_kibot_adjusted_ohlcv_csv_sorts_filters_session(tmp_path):
    csv_path = tmp_path / "IBM_adjusted_ohlcv.csv"
    csv_path.write_text(
        "\n".join(
            [
                "01/02/1998,09:31,25.196,25.226,25.196,25.196,44739",
                "01/02/1998,09:30,25.226,25.226,25.226,25.226,277549",
                "01/02/1998,08:00,25.196,25.226,25.196,25.196,19888",
            ]
        )
    )

    df = load_kibot_adjusted_ohlcv_csv(str(csv_path))

    assert list(df.index) == [
        pd.Timestamp("1998-01-02 09:30:00"),
        pd.Timestamp("1998-01-02 09:31:00"),
    ]
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.iloc[0]["open"] == pytest.approx(25.226)
    assert df.iloc[0]["volume"] == pytest.approx(277549)


def test_load_kibot_adjusted_ohlcv_csv_records_dropped_market_rows(tmp_path):
    csv_path = tmp_path / "ohlcv_with_bad_market_rows.csv"
    csv_path.write_text(
        "\n".join(
            [
                "01/02/1998,09:30,25.226,25.226,25.226,25.226,277549",
                "01/02/1998,09:31,0.000,25.226,25.196,25.226,55096",
                "01/02/1998,09:32,25.211,25.226,25.196,25.196,-1",
            ]
        )
    )

    df = load_kibot_adjusted_ohlcv_csv(str(csv_path))

    assert list(df.index) == [pd.Timestamp("1998-01-02 09:30:00")]
    assert df.attrs["dropped_rows"] == {
        "nonpositive_adjusted_ohlc_prices": 1,
        "negative_adjusted_ohlcv_volume": 1,
    }


def test_loader_raises_for_wrong_column_count(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("09/28/2009,09:30:00,42.91\n")

    with pytest.raises(ValueError, match="Expected 6 columns"):
        load_kibot_tick_bidask_csv(str(csv_path))


def test_loader_raises_for_non_numeric_values(tmp_path):
    csv_path = tmp_path / "bad_numeric.csv"
    csv_path.write_text("09/28/2009,09:30:00,not-a-price,9.90,10.10,100\n")

    with pytest.raises(ValueError, match="non-numeric"):
        load_kibot_tick_bidask_csv(str(csv_path))


def test_loader_raises_for_bad_timestamp(tmp_path):
    csv_path = tmp_path / "bad_timestamp.csv"
    csv_path.write_text("not-a-date,09:30:00,10.00,9.90,10.10,100\n")

    with pytest.raises(ValueError):
        load_kibot_tick_bidask_csv(str(csv_path))
