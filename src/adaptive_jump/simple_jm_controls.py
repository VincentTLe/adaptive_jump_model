"""Causal control paths for the prespecified simple-JM study."""

from __future__ import annotations

import hmac
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_jump.artifacts import sha256_file
from adaptive_jump.backtest import apply_signal

DEVELOPMENT_CUTOFF = date(2023, 12, 31)
LAMBDA50_COLUMN = "50.0"
MARKETS = ("us", "de", "jp")
PRIMARY_DELAY_TRADING_DAYS = 1
ONE_WAY_COST_BPS = 10.0
SIGNAL_TO_RETURN_OFFSET = PRIMARY_DELAY_TRADING_DAYS + 1


class SimpleJMControlError(ValueError):
    """Raised when a simple-JM control violates its frozen contract."""


@dataclass(frozen=True)
class ControlPath:
    """An emitted state, its risky-asset signal, and causal accounting path."""

    state: pd.Series
    signal: pd.Series
    trades: pd.DataFrame


def confirm_two_observations(states: pd.Series) -> pd.Series:
    """Accept a new binary state only after two consecutive raw observations."""
    raw = _binary_series(states, "raw state")
    values = raw.to_numpy(dtype=float)
    first = int(np.flatnonzero(~np.isnan(values))[0])
    confirmed = np.full(len(values), np.nan, dtype=float)
    confirmed[first] = values[first]
    for row in range(first + 1, len(values)):
        confirmed[row] = (
            values[row] if values[row] == values[row - 1] else confirmed[row - 1]
        )
    return pd.Series(
        confirmed,
        index=raw.index.copy(),
        name="confirmed_state",
        dtype=float,
    )


def states_from_signal(signal: pd.Series) -> pd.Series:
    """Convert the canonical risky-asset signal to its selected JM state."""
    values = _binary_series(signal, "signal")
    state = 1.0 - values
    state.name = "raw_state"
    return state


def signal_from_states(states: pd.Series) -> pd.Series:
    """Map favorable state 0 to equity and unfavorable state 1 to cash."""
    values = _binary_series(states, "state")
    signal = 1.0 - values
    signal.name = "signal"
    return signal


def load_static_lambda50_states(
    source_root: str | Path,
    market: str,
    *,
    expected_sha256: str,
    cutoff: date = DEVELOPMENT_CUTOFF,
) -> pd.Series:
    """Load the hash-pinned lambda-50 column from the accepted source run."""
    if market not in MARKETS:
        raise SimpleJMControlError(f"unknown market: {market}")
    expected = _validated_sha256(expected_sha256)
    root = Path(source_root)
    market_root = root / market
    source = market_root / "jm-missing-states.csv"
    if any(path.is_symlink() for path in (root, market_root, source)):
        raise SimpleJMControlError("lambda50 source path may not contain symlinks")
    if not source.is_file():
        raise SimpleJMControlError(f"lambda50 source file is missing: {source}")
    try:
        observed = sha256_file(source)
    except OSError as exc:
        raise SimpleJMControlError(f"cannot hash lambda50 source: {source}") from exc
    if not hmac.compare_digest(observed, expected):
        raise SimpleJMControlError(f"lambda50 source hash changed: {source}")
    try:
        frame = pd.read_csv(
            source,
            usecols=["date", LAMBDA50_COLUMN],
            dtype={LAMBDA50_COLUMN: "string"},
        )
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise SimpleJMControlError(f"cannot read lambda50 source: {source}") from exc
    if frame.empty:
        raise SimpleJMControlError("lambda50 source is empty")
    try:
        dates = pd.DatetimeIndex(pd.to_datetime(frame["date"], errors="raise"))
    except (TypeError, ValueError) as exc:
        raise SimpleJMControlError("lambda50 dates are invalid") from exc
    _validate_dates(dates, cutoff, "lambda50 source")
    raw = frame[LAMBDA50_COLUMN]
    numeric = pd.to_numeric(raw, errors="coerce")
    if (raw.notna() & numeric.isna()).any():
        raise SimpleJMControlError("lambda50 states contain a non-numeric value")
    state = pd.Series(
        numeric.to_numpy(dtype=float),
        index=dates,
        name="state",
        dtype=float,
    )
    return _binary_series(state, "lambda50 state").rename("state")


def build_control_path(
    returns: pd.DataFrame,
    states: pd.Series,
    *,
    cutoff: date = DEVELOPMENT_CUTOFF,
) -> ControlPath:
    """Map one past-only state path through the frozen t+2, 10-bps accounting."""
    prepared = _prepare_returns(returns, cutoff)
    state = _binary_series(states, "state").rename("state")
    try:
        state_dates = pd.DatetimeIndex(pd.to_datetime(state.index, errors="raise"))
    except (TypeError, ValueError) as exc:
        raise SimpleJMControlError("state dates are invalid") from exc
    _validate_dates(state_dates, cutoff, "state")
    return_dates = pd.DatetimeIndex(prepared["date"], name="date")
    if not state_dates.equals(return_dates):
        raise SimpleJMControlError("state and return dates must match exactly")
    state.index = return_dates
    signal = signal_from_states(state)
    trades = apply_signal(
        prepared,
        signal.reset_index(drop=True),
        delay_trading_days=PRIMARY_DELAY_TRADING_DAYS,
        one_way_cost_bps=ONE_WAY_COST_BPS,
    )
    _validate_accounting(trades, signal)
    return ControlPath(state=state, signal=signal, trades=trades)


def build_confirmed_control_path(
    returns: pd.DataFrame,
    canonical_signal: pd.Series,
    *,
    cutoff: date = DEVELOPMENT_CUTOFF,
) -> ControlPath:
    """Post-filter the sealed selected fixed-JM state, then account identically."""
    raw_state = states_from_signal(canonical_signal)
    confirmed = confirm_two_observations(raw_state)
    return build_control_path(returns, confirmed, cutoff=cutoff)


def build_static_lambda50_path(
    returns: pd.DataFrame,
    source_root: str | Path,
    market: str,
    *,
    expected_sha256: str,
    cutoff: date = DEVELOPMENT_CUTOFF,
) -> ControlPath:
    """Load and account the sealed static-lambda-50 state without monthly CV."""
    state = load_static_lambda50_states(
        source_root,
        market,
        expected_sha256=expected_sha256,
        cutoff=cutoff,
    )
    return build_control_path(returns, state, cutoff=cutoff)


def _binary_series(values: pd.Series, label: str) -> pd.Series:
    try:
        series = pd.Series(values, copy=True, dtype=float)
    except (TypeError, ValueError) as exc:
        raise SimpleJMControlError(f"{label} must be numeric") from exc
    raw = series.to_numpy(dtype=float)
    observed = ~np.isnan(raw)
    if not observed.any():
        raise SimpleJMControlError(f"{label} has no finite observation")
    if (
        not np.isfinite(raw[observed]).all()
        or not np.isin(raw[observed], [0.0, 1.0]).all()
    ):
        raise SimpleJMControlError(
            f"{label} must contain only 0, 1, or leading missing"
        )
    first = int(np.flatnonzero(observed)[0])
    if np.isnan(raw[first:]).any():
        raise SimpleJMControlError(f"{label} contains missing values after it starts")
    return series


def _validated_sha256(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SimpleJMControlError("expected source SHA256 must be 64 lowercase hex")
    return value


def _validate_dates(dates: pd.DatetimeIndex, cutoff: date, label: str) -> None:
    try:
        cutoff_timestamp = pd.Timestamp(cutoff)
    except (TypeError, ValueError) as exc:
        raise SimpleJMControlError("cutoff date is invalid") from exc
    if dates.empty or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise SimpleJMControlError(
            f"{label} dates must be non-empty, unique, and sorted"
        )
    if dates.tz is not None or cutoff_timestamp.tz is not None:
        raise SimpleJMControlError(f"{label} dates must be timezone-naive")
    if dates.max() > cutoff_timestamp:
        raise SimpleJMControlError(f"{label} contains a post-cutoff date")


def _prepare_returns(returns: pd.DataFrame, cutoff: date) -> pd.DataFrame:
    required = ["date", "equity_simple", "cash_return"]
    missing = [column for column in required if column not in returns]
    if missing:
        raise SimpleJMControlError(f"missing return columns: {missing}")
    prepared = returns.loc[:, required].copy()
    try:
        prepared["date"] = pd.to_datetime(prepared["date"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise SimpleJMControlError("return dates are invalid") from exc
    dates = pd.DatetimeIndex(prepared["date"], name="date")
    _validate_dates(dates, cutoff, "return")
    try:
        numeric = prepared[["equity_simple", "cash_return"]].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise SimpleJMControlError("returns must be numeric") from exc
    if np.isinf(numeric).any():
        raise SimpleJMControlError("returns must be finite when present")
    return prepared.reset_index(drop=True)


def _validate_accounting(trades: pd.DataFrame, signal: pd.Series) -> None:
    expected_position = signal.reset_index(drop=True).shift(SIGNAL_TO_RETURN_OFFSET)
    if not np.array_equal(
        trades["position"].to_numpy(dtype=float),
        expected_position.to_numpy(dtype=float),
        equal_nan=True,
    ):
        raise SimpleJMControlError("t+2 position identity failed")
    positioned = trades["position"].notna()
    earned_returns = trades.loc[positioned, ["equity_simple", "cash_return"]].to_numpy(
        dtype=float
    )
    if not np.isfinite(earned_returns).all():
        raise SimpleJMControlError("returns must be finite wherever position is finite")
    expected_cost = trades["one_way_turnover"] * (ONE_WAY_COST_BPS / 10_000.0)
    expected_gross = (
        trades["position"] * trades["equity_simple"]
        + (1.0 - trades["position"]) * trades["cash_return"]
    )
    controls = (
        np.allclose(
            trades["transaction_cost"],
            expected_cost,
            rtol=0,
            atol=1e-15,
            equal_nan=True,
        ),
        np.allclose(
            trades["gross_return"], expected_gross, rtol=0, atol=1e-15, equal_nan=True
        ),
        np.allclose(
            trades["strategy_return"],
            trades["gross_return"] - trades["transaction_cost"],
            rtol=0,
            atol=1e-15,
            equal_nan=True,
        ),
    )
    if not all(controls):
        raise SimpleJMControlError("control-path accounting identity failed")
    finite_cost = trades["transaction_cost"].dropna()
    if not finite_cost.map(math.isfinite).all():
        raise SimpleJMControlError("control-path cost is non-finite")
