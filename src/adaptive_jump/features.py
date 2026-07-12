"""Causal return alignment and paper feature construction."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd

from adaptive_jump.config import MarketConfig, ResearchConfig, SourceConfig


class FeatureError(ValueError):
    """Raised when observations cannot satisfy the frozen feature contract."""


def equity_returns(levels: pd.DataFrame) -> pd.DataFrame:
    """Calculate simple/log returns between consecutive valid index levels."""
    _require_columns(levels)
    frame = levels.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    _require_ordered_unique(frame, "equity")
    frame = frame.loc[frame["value"].notna()].copy()
    if frame.empty or (frame["value"] <= 0).any():
        raise FeatureError("equity levels must contain positive observations")
    frame["gap_calendar_days"] = frame["date"].diff().dt.days
    frame["equity_simple"] = frame["value"].pct_change(fill_method=None)
    frame["equity_log"] = np.log(frame["value"] / frame["value"].shift(1))
    return frame.loc[
        frame["equity_simple"].notna(),
        ["date", "equity_simple", "equity_log", "gap_calendar_days"],
    ].reset_index(drop=True)


def align_cash_returns(
    return_dates: pd.Series,
    cash: pd.DataFrame,
    source: SourceConfig,
    trading_days_per_year: int,
) -> pd.DataFrame:
    """Align only causally available annual yields to equity return dates."""
    _require_columns(cash)
    left = pd.DataFrame(
        {"date": pd.to_datetime(return_dates, errors="raise")}
    ).sort_values("date")
    right = cash.copy()
    right["cash_observation_date"] = pd.to_datetime(right.pop("date"), errors="raise")
    _require_ordered_unique(
        right.rename(columns={"cash_observation_date": "date"}), "cash"
    )
    right = right.loc[right["value"].notna()].copy()
    if source.frequency == "daily":
        lag = _positive_setting(source, "availability_lag_calendar_days")
        right["cash_available_date"] = right["cash_observation_date"] + pd.to_timedelta(
            lag, unit="D"
        )
    elif source.frequency == "monthly":
        lag = _positive_setting(source, "availability_lag_month_starts")
        right["cash_available_date"] = right["cash_observation_date"] + pd.DateOffset(
            months=lag
        )
    else:
        raise FeatureError(f"unsupported cash frequency: {source.frequency}")
    right = right.sort_values("cash_available_date")
    aligned = pd.merge_asof(
        left,
        right[["cash_available_date", "cash_observation_date", "value"]],
        left_on="date",
        right_on="cash_available_date",
        direction="backward",
        allow_exact_matches=True,
    )
    aligned["cash_age_days"] = (
        aligned["date"] - aligned["cash_available_date"]
    ).dt.days
    max_age = _positive_setting(source, "max_staleness_calendar_days")
    stale = aligned["cash_age_days"] > max_age
    aligned.loc[stale, ["cash_observation_date", "value"]] = np.nan
    aligned["cash_yield_percent"] = aligned.pop("value")
    aligned["cash_return"] = (
        aligned["cash_yield_percent"] / 100.0 / trading_days_per_year
    )
    return aligned


def make_features(
    excess_return: pd.Series,
    *,
    downside_halflife: int = 10,
    sortino_halflives: tuple[int, ...] = (20, 60),
    adjust: bool = True,
    ignore_na: bool = False,
) -> pd.DataFrame:
    """Compute the three EWM downside/Sortino features from the paper."""
    excess = pd.Series(excess_return, dtype=float)
    negative_squared = excess.clip(upper=0).pow(2)

    def downside(halflife: int) -> pd.Series:
        return np.sqrt(
            negative_squared.ewm(
                halflife=halflife,
                adjust=adjust,
                ignore_na=ignore_na,
                min_periods=0,
            ).mean()
        )

    features = pd.DataFrame(index=excess.index)
    features[f"dd_{downside_halflife}"] = downside(downside_halflife)
    for halflife in sortino_halflives:
        mean = excess.ewm(
            halflife=halflife,
            adjust=adjust,
            ignore_na=ignore_na,
            min_periods=0,
        ).mean()
        denominator = downside(halflife)
        features[f"sortino_{halflife}"] = mean.div(denominator.where(denominator > 0))
    features.loc[excess.isna(), :] = np.nan
    return features.replace([np.inf, -np.inf], np.nan)


def prepare_market(
    equity: pd.DataFrame,
    cash: pd.DataFrame,
    market: MarketConfig,
    config: ResearchConfig,
) -> pd.DataFrame:
    """Build causal returns and unscaled JM features for one market."""
    returns = equity_returns(equity)
    aligned = align_cash_returns(
        returns["date"], cash, market.cash, config.trading_days_per_year
    )
    frame = returns.merge(aligned, on="date", how="left", validate="one_to_one")
    frame["excess_return"] = frame["equity_simple"] - frame["cash_return"]
    protocol = config.feature_protocol
    features = make_features(
        frame["excess_return"],
        downside_halflife=protocol.downside_halflife,
        sortino_halflives=protocol.sortino_halflives,
        adjust=protocol.ewm_adjust,
        ignore_na=protocol.ewm_ignore_na,
    )
    return pd.concat([frame, features], axis=1)


def effective_oos_start(
    frame: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] = ("dd_10", "sortino_20", "sortino_60"),
    requested: date = date(1990, 1, 1),
    fit_window: int = 3000,
    validation_years: int = 8,
) -> date | None:
    """Return the first feature date eligible after fit and validation history."""
    complete = frame.loc[frame[list(feature_columns)].notna().all(axis=1)].copy()
    if len(complete) < fit_window:
        return None
    fit_terminal = pd.Timestamp(complete.iloc[fit_window - 1]["date"])
    target = max(
        pd.Timestamp(requested), fit_terminal + pd.DateOffset(years=validation_years)
    )
    eligible = complete.loc[pd.to_datetime(complete["date"]) >= target]
    if eligible.empty:
        return None
    return pd.Timestamp(eligible.iloc[0]["date"]).date()


def _positive_setting(source: SourceConfig, key: str) -> int:
    value = source.settings.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise FeatureError(f"{source.source_id}: {key} must be positive")
    return value


def _require_columns(frame: pd.DataFrame) -> None:
    if list(frame.columns) != ["date", "value"]:
        raise FeatureError("canonical observations must have date,value columns")


def _require_ordered_unique(frame: pd.DataFrame, label: str) -> None:
    if frame["date"].duplicated().any():
        raise FeatureError(f"{label} dates must be unique")
    if not frame["date"].is_monotonic_increasing:
        raise FeatureError(f"{label} dates must be increasing")
    if frame["date"].isna().any():
        raise FeatureError(f"{label} dates must not be missing")
    finite = frame["value"].dropna().map(math.isfinite)
    if not finite.all():
        raise FeatureError(f"{label} values must be finite")
