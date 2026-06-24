import pandas as pd
import pytest

from adaptive_jump.data_kibot import load_kibot_minute_bidask_csv, load_kibot_tick_bidask_csv


def test_load_kibot_tick_bidask_csv_cleans_and_computes_fields(tmp_path):
    csv_path = tmp_path / "IVE_tick_bidask.csv"
    csv_path.write_text(
        "\n".join(
            [
                "09/28/2009,09:31:00,10.00,9.90,10.10,100",
                "09/28/2009,09:30:00,20.00,19.80,20.20,200",
                "09/28/2009,09:32:00,15.00,15.20,15.10,100",
                "09/28/2009,08:00:00,9.00,8.90,9.10,50",
                "09/28/2009,09:33:00,0.00,9.90,10.10,100",
            ]
        )
    )

    df = load_kibot_tick_bidask_csv(str(csv_path))

    assert list(df.index) == [
        pd.Timestamp("2009-09-28 09:30:00"),
        pd.Timestamp("2009-09-28 09:31:00"),
    ]
    assert df.index.name == "timestamp"
    assert list(df.columns) == ["price", "bid", "ask", "size", "mid", "spread", "rel_spread"]
    assert df.loc[pd.Timestamp("2009-09-28 09:30:00"), "mid"] == pytest.approx(20.0)
    assert df.loc[pd.Timestamp("2009-09-28 09:30:00"), "spread"] == pytest.approx(0.4)
    assert df.loc[pd.Timestamp("2009-09-28 09:30:00"), "rel_spread"] == pytest.approx(0.02)
    assert (df["ask"] >= df["bid"]).all()


def test_load_kibot_minute_bidask_csv_cleans_and_computes_fields(tmp_path):
    csv_path = tmp_path / "IVE_minute_bidask.csv"
    csv_path.write_text(
        "\n".join(
            [
                "09/28/2009,09:31,10.00,10.40,9.90,10.20,1000,10.01,10.31,9.81,10.11,10.03,10.33,9.83,10.13",
                "09/28/2009,09:30,20.00,20.40,19.90,20.20,2000,20.01,20.31,19.81,20.11,20.03,20.33,19.83,20.13",
                "09/28/2009,09:32,15.00,15.40,14.90,15.20,1500,15.01,15.31,14.81,15.21,15.03,15.33,14.83,15.19",
                "09/28/2009,09:33,0.00,10.40,9.90,10.20,1000,10.01,10.31,9.81,10.11,10.03,10.33,9.83,10.13",
            ]
        )
    )

    df = load_kibot_minute_bidask_csv(str(csv_path))

    assert list(df.index) == [
        pd.Timestamp("2009-09-28 09:30:00"),
        pd.Timestamp("2009-09-28 09:31:00"),
    ]
    assert df.index.name == "timestamp"
    assert "mid_close" in df.columns
    assert "spread_close" in df.columns
    assert "rel_spread_close" in df.columns
    row = df.loc[pd.Timestamp("2009-09-28 09:30:00")]
    assert row["mid_close"] == pytest.approx(20.12)
    assert row["spread_close"] == pytest.approx(0.02)
    assert row["rel_spread_close"] == pytest.approx(0.02 / 20.12)
    assert row["volume"] == pytest.approx(2000)
    assert (df["ask_close"] >= df["bid_close"]).all()


def test_loader_raises_for_wrong_column_count(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("09/28/2009,09:30:00,42.91\n")

    with pytest.raises(ValueError, match="Expected 6 columns"):
        load_kibot_tick_bidask_csv(str(csv_path))
