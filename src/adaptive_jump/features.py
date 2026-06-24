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


def make_minute_features_from_minute_bidask(df: pd.DataFrame) -> pd.DataFrame:
    """Construct basic minute-level microstructure features."""
    required_columns = ["mid_close", "volume", "spread_close", "rel_spread_close"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"df is missing required columns: {missing_columns}")

    out = pd.DataFrame(index=df.index)
    out.index.name = df.index.name
    out["mid_return"] = np.log(df["mid_close"] / df["mid_close"].shift(1))
    out["abs_mid_return"] = out["mid_return"].abs()
    out["volume"] = df["volume"]
    out["log_volume"] = np.log1p(df["volume"])
    out["spread_close"] = df["spread_close"]
    out["rel_spread_close"] = df["rel_spread_close"]
    out["rolling_vol_5"] = out["mid_return"].rolling(5).std()
    out["rolling_vol_20"] = out["mid_return"].rolling(20).std()

    z_log_volume = zscore_series(out["log_volume"])
    z_mid_return = zscore_series(out["mid_return"])
    z_rolling_vol_20 = zscore_series(out["rolling_vol_20"])
    out["noise_score_raw"] = out["rel_spread_close"] - z_log_volume
    out["shock_score_raw"] = z_mid_return.abs() + z_rolling_vol_20

    return out[
        [
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
    ]


def zscore_series(s: pd.Series, min_std: float = 1e-12) -> pd.Series:
    """Return a z-scored series, avoiding division by near-zero standard deviation."""
    std = s.std()
    if pd.isna(std) or std < min_std:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def _minute_realized_var(price: pd.Series) -> float:
    log_returns = np.log(price / price.shift(1))
    return float((log_returns.dropna() ** 2).sum())
