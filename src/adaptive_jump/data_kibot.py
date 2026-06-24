"""Loaders for Kibot free sample bid/ask CSV files."""

from pathlib import Path

import numpy as np
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

    dropped_rows = {}
    _raise_if_invalid(_finite(df, ["price", "bid", "ask", "size"]), "tick rows must be finite", df.index)
    df = _drop_invalid(df, (df[["price", "bid", "ask", "size"]] > 0).all(axis=1), "nonpositive_tick_values", dropped_rows)
    df = _drop_invalid(df, df["ask"] >= df["bid"], "crossed_tick_quotes", dropped_rows)
    df = df.sort_index(kind="mergesort")
    before_session = len(df)
    df = df.between_time("09:30", "16:00")
    _record_drop(dropped_rows, "outside_regular_session", before_session - len(df))

    df["mid"] = (df["bid"] + df["ask"]) / 2
    df["spread"] = df["ask"] - df["bid"]
    df["rel_spread"] = df["spread"] / df["mid"]

    result = df[["price", "bid", "ask", "size", "mid", "spread", "rel_spread"]].copy()
    result.attrs["dropped_rows"] = dropped_rows
    return result


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

    dropped_rows = {}
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
    _raise_if_invalid(_finite(df, numeric_columns), "minute bid/ask rows must be finite", df.index)
    df = _drop_invalid(df, (df[positive_columns] > 0).all(axis=1), "nonpositive_minute_bidask_prices", dropped_rows)
    df = _drop_invalid(df, df["volume"] >= 0, "negative_minute_bidask_volume", dropped_rows)
    for suffix in ["open", "high", "low", "close"]:
        df = _drop_invalid(
            df,
            df[f"ask_{suffix}"] >= df[f"bid_{suffix}"],
            f"crossed_minute_{suffix}_quotes",
            dropped_rows,
        )
    df = df.sort_index(kind="mergesort")
    before_session = len(df)
    df = df.between_time("09:30", "16:00")
    _record_drop(dropped_rows, "outside_regular_session", before_session - len(df))

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
    result = df[output_columns].copy()
    result.attrs["dropped_rows"] = dropped_rows
    return result


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

    dropped_rows = {}
    price_columns = ["open", "high", "low", "close"]
    _raise_if_invalid(_finite(df, ["open", "high", "low", "close", "volume"]), "adjusted OHLCV rows must be finite", df.index)
    df = _drop_invalid(df, (df[price_columns] > 0).all(axis=1), "nonpositive_adjusted_ohlc_prices", dropped_rows)
    df = _drop_invalid(df, df["volume"] >= 0, "negative_adjusted_ohlcv_volume", dropped_rows)
    df = df.sort_index(kind="mergesort")
    before_session = len(df)
    df = df.between_time("09:30", "16:00")
    _record_drop(dropped_rows, "outside_regular_session", before_session - len(df))

    result = df[["open", "high", "low", "close", "volume"]].copy()
    result.attrs["dropped_rows"] = dropped_rows
    return result


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


def _finite(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    return pd.Series(np.isfinite(df[columns].to_numpy()).all(axis=1), index=df.index)


def _raise_if_invalid(valid: pd.Series, rule: str, index: pd.Index) -> None:
    invalid = ~valid
    if invalid.any():
        first = index[invalid.to_numpy()][0]
        raise ValueError(f"{rule}: {int(invalid.sum())} invalid rows; first at {first}")


def _drop_invalid(df: pd.DataFrame, valid: pd.Series, reason: str, dropped_rows: dict[str, int]) -> pd.DataFrame:
    dropped = int((~valid).sum())
    _record_drop(dropped_rows, reason, dropped)
    if dropped == 0:
        return df
    return df.loc[valid].copy()


def _record_drop(dropped_rows: dict[str, int], reason: str, count: int) -> None:
    if count > 0:
        dropped_rows[reason] = dropped_rows.get(reason, 0) + int(count)
