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
    _raise_if_not_finite(df_tick, required_columns, "df_tick")
    _raise_if_invalid((df_tick[["price", "bid", "ask", "size", "mid"]] > 0).all(axis=1), "df_tick prices and size must be positive")
    _raise_if_invalid((df_tick[["spread", "rel_spread"]] >= 0).all(axis=1), "df_tick spreads must be nonnegative")

    df = df_tick.sort_index(kind="mergesort")
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
    out["realized_var"] = _realized_var_by_later_tick_minute(df["price"], minute).reindex(out.index, fill_value=0.0)

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
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df index must be a pandas.DatetimeIndex")
    _raise_if_not_finite(df, required_columns, "df")
    _raise_if_invalid(df["mid_close"] > 0, "mid_close must be positive")
    _raise_if_invalid(df["volume"] >= 0, "volume must be nonnegative")
    _raise_if_invalid((df[["spread_close", "rel_spread_close"]] >= 0).all(axis=1), "spreads must be nonnegative")

    out = pd.DataFrame(index=df.index)
    out.index.name = df.index.name
    same_date = _same_date_as_previous(df.index)
    out["mid_return"] = np.log(df["mid_close"] / df["mid_close"].shift(1)).where(same_date)
    out["abs_mid_return"] = out["mid_return"].abs()
    out["volume"] = df["volume"]
    out["log_volume"] = np.log1p(df["volume"])
    out["spread_close"] = df["spread_close"]
    out["rel_spread_close"] = df["rel_spread_close"]
    out["rolling_vol_5"] = out["mid_return"].rolling(5).std()
    out["rolling_vol_20"] = out["mid_return"].rolling(20).std()

    z_rel_spread = zscore_series(out["rel_spread_close"])
    z_log_volume = zscore_series(out["log_volume"])
    z_mid_return = zscore_series(out["mid_return"])
    z_rolling_vol_20 = zscore_series(out["rolling_vol_20"])
    out["noise_score_raw"] = z_rel_spread - z_log_volume
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
    values = s.astype(float)
    valid = values.dropna()
    result = pd.Series(np.nan, index=s.index, dtype=float)
    std = valid.std()
    if pd.isna(std) or std < min_std:
        result.loc[valid.index] = 0.0
        return result
    result.loc[valid.index] = (valid - valid.mean()) / std
    return result


def _realized_var_by_later_tick_minute(price: pd.Series, minute: pd.DatetimeIndex) -> pd.Series:
    same_date = _same_date_as_previous(price.index)
    log_returns = np.log(price / price.shift(1)).where(same_date)
    squared_returns = log_returns.pow(2)
    return squared_returns.groupby(minute).sum(min_count=1).fillna(0.0)


def _same_date_as_previous(index: pd.DatetimeIndex) -> pd.Series:
    dates = pd.Series(index.normalize(), index=index)
    return dates == dates.shift(1)


def _raise_if_not_finite(df: pd.DataFrame, columns: list[str], name: str) -> None:
    valid = pd.Series(np.isfinite(df[columns].to_numpy()).all(axis=1), index=df.index)
    _raise_if_invalid(valid, f"{name} columns must be finite: {columns}")


def _raise_if_invalid(valid: pd.Series, rule: str) -> None:
    invalid = ~valid
    if invalid.any():
        first = invalid[invalid].index[0]
        raise ValueError(f"{rule}: {int(invalid.sum())} invalid rows; first at {first}")
