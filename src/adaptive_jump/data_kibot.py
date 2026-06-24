"""Loaders for Kibot free sample bid/ask CSV files."""

from pathlib import Path

import pandas as pd


TICK_BIDASK_COLUMNS = ["date", "time", "price", "bid", "ask", "size"]

MINUTE_BIDASK_COLUMNS = [
    "date",
    "time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "bid_open",
    "bid_high",
    "bid_low",
    "bid_close",
    "ask_open",
    "ask_high",
    "ask_low",
    "ask_close",
]

ADJUSTED_OHLCV_COLUMNS = ["date", "time", "open", "high", "low", "close", "volume"]


def load_kibot_tick_bidask_csv(path: str) -> pd.DataFrame:
    """Load a headerless Kibot tick-with-bid-ask CSV file.

    Kibot's standard tick bid/ask format is:
    Date,Time,Price,Bid,Ask,Size.
    """
    raw = _read_headerless_csv(path, TICK_BIDASK_COLUMNS)
    timestamp = _parse_timestamp(raw)

    df = pd.DataFrame(index=timestamp)
    df.index.name = "timestamp"
    for column in ["price", "bid", "ask", "size"]:
        df[column] = _parse_numeric(raw[column], column).to_numpy()

    valid = (
        (df["price"] > 0)
        & (df["bid"] > 0)
        & (df["ask"] > 0)
        & (df["size"] > 0)
        & (df["ask"] >= df["bid"])
    )
    df = df.loc[valid].sort_index()
    df = df.between_time("09:30", "16:00")

    df["mid"] = (df["bid"] + df["ask"]) / 2
    df["spread"] = df["ask"] - df["bid"]
    df["rel_spread"] = df["spread"] / df["mid"]

    return df[["price", "bid", "ask", "size", "mid", "spread", "rel_spread"]]


def load_kibot_minute_bidask_csv(path: str) -> pd.DataFrame:
    """Load a headerless Kibot 1-minute bid/ask companion CSV file.

    Kibot's free IVE companion format is standard OHLCV plus bid and ask OHLC.
    """
    raw = _read_headerless_csv(path, MINUTE_BIDASK_COLUMNS)
    timestamp = _parse_timestamp(raw)

    df = pd.DataFrame(index=timestamp)
    df.index.name = "timestamp"
    numeric_columns = [column for column in MINUTE_BIDASK_COLUMNS if column not in {"date", "time"}]
    for column in numeric_columns:
        df[column] = _parse_numeric(raw[column], column).to_numpy()

    positive_columns = [
        "open",
        "high",
        "low",
        "close",
        "bid_open",
        "bid_high",
        "bid_low",
        "bid_close",
        "ask_open",
        "ask_high",
        "ask_low",
        "ask_close",
    ]
    valid = (df[positive_columns] > 0).all(axis=1) & (df["ask_close"] >= df["bid_close"])
    df = df.loc[valid].sort_index()

    df["mid_close"] = (df["bid_close"] + df["ask_close"]) / 2
    df["spread_close"] = df["ask_close"] - df["bid_close"]
    df["rel_spread_close"] = df["spread_close"] / df["mid_close"]

    output_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "bid_open",
        "bid_high",
        "bid_low",
        "bid_close",
        "ask_open",
        "ask_high",
        "ask_low",
        "ask_close",
        "mid_close",
        "spread_close",
        "rel_spread_close",
    ]
    return df[output_columns]


def load_kibot_adjusted_ohlcv_csv(path: str) -> pd.DataFrame:
    """Load a headerless Kibot adjusted OHLCV intraday CSV file.

    The expected format is: Date,Time,Open,High,Low,Close,Volume.
    """
    raw = _read_headerless_csv(path, ADJUSTED_OHLCV_COLUMNS)
    timestamp = _parse_timestamp(raw)

    df = pd.DataFrame(index=timestamp)
    df.index.name = "timestamp"
    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = _parse_numeric(raw[column], column).to_numpy()

    price_columns = ["open", "high", "low", "close"]
    valid = (df[price_columns] > 0).all(axis=1) & (df["volume"] >= 0)
    df = df.loc[valid].sort_index()
    df = df.between_time("09:30", "16:00")

    return df[["open", "high", "low", "close", "volume"]]


def _read_headerless_csv(path: str, columns: list[str]) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Kibot CSV file does not exist: {csv_path}")

    raw = pd.read_csv(csv_path, header=None)
    if raw.shape[1] != len(columns):
        raise ValueError(f"Expected {len(columns)} columns in {csv_path}, found {raw.shape[1]}")

    raw.columns = columns
    return raw


def _parse_timestamp(df: pd.DataFrame) -> pd.DatetimeIndex:
    timestamp = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="raise")
    return pd.DatetimeIndex(timestamp)


def _parse_numeric(series: pd.Series, column: str) -> pd.Series:
    try:
        return pd.to_numeric(series, errors="raise")
    except ValueError as exc:
        raise ValueError(f"Column {column!r} contains non-numeric values") from exc
