"""Feature construction utilities for intraday market data."""

import numpy as np
import pandas as pd


def aggregate_tick_to_minutes(df_tick: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cleaned tick bid/ask data to one row per minute."""
    required_columns = ["price", "bid", "ask", "size", "mid", "spread", "rel_spread"]
    missing_columns = [column for column in required_columns if column not in df_tick.columns]
    if missing_columns:
        raise ValueError(f"df_tick is missing required columns: {missing_columns}")
    if not isinstance(df_tick.index, pd.DatetimeIndex):
        raise TypeError("df_tick index must be a pandas.DatetimeIndex")

    df = df_tick.sort_index()
    minute = df.index.floor("min")
    grouped = df.groupby(minute, sort=True)

    out = pd.DataFrame(index=grouped.size().index)
    out.index.name = "timestamp"
    out["open"] = grouped["price"].first()
    out["high"] = grouped["price"].max()
    out["low"] = grouped["price"].min()
    out["close"] = grouped["price"].last()
    out["volume"] = grouped["size"].sum()
    out["trade_count"] = grouped["price"].size()
    out["bid_close"] = grouped["bid"].last()
    out["ask_close"] = grouped["ask"].last()
    out["mid_close"] = grouped["mid"].last()
    out["spread_close"] = grouped["spread"].last()
    out["rel_spread_close"] = grouped["rel_spread"].last()
    out["realized_var"] = grouped["price"].apply(_minute_realized_var)

    return out[
        [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "trade_count",
            "bid_close",
            "ask_close",
            "mid_close",
            "spread_close",
            "rel_spread_close",
            "realized_var",
        ]
    ]


def _minute_realized_var(price: pd.Series) -> float:
    log_returns = np.log(price / price.shift(1))
    return float((log_returns.dropna() ** 2).sum())
