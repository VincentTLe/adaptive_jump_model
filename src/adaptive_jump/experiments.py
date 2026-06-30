"""Experiment utilities for regime paths and leakage-safe backtests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureStats:
    """Train-only feature normalization statistics."""

    means: pd.Series
    stds: pd.Series
    min_std: float


@dataclass(frozen=True)
class TimeSplit:
    """Integer train/test split for walk-forward experiments."""

    train: np.ndarray
    test: np.ndarray


DEFAULT_SCORE_COLUMNS = ["log_volume", "mid_return", "rolling_vol_20"]


def make_time_splits(
    index,
    scheme: str,
    train_size: int | None = None,
    test_size: int | None = None,
    step_size: int | None = None,
    min_train_size: int | None = None,
    gap: int = 0,
) -> list[TimeSplit]:
    """Create ordered train/test splits for leakage-safe experiments."""
    n_obs = len(pd.Index(index))
    if n_obs == 0:
        raise ValueError("index must be non-empty")
    if gap < 0:
        raise ValueError("gap must be nonnegative")
    if test_size is None or test_size <= 0:
        raise ValueError("test_size must be positive")
    if step_size is None:
        step_size = test_size
    if step_size <= 0:
        raise ValueError("step_size must be positive")

    if scheme == "train_only":
        if train_size is None or train_size <= 0:
            raise ValueError("train_size must be positive for train_only")
        test_start = train_size + gap
        test_end = min(test_start + test_size, n_obs)
        return [_split(np.arange(0, train_size), np.arange(test_start, test_end))]

    if scheme not in {"rolling", "expanding"}:
        raise ValueError("scheme must be one of {'train_only', 'rolling', 'expanding'}")
    if min_train_size is None or min_train_size <= 0:
        raise ValueError("min_train_size must be positive for rolling/expanding")
    if scheme == "rolling" and (train_size is None or train_size <= 0):
        raise ValueError("train_size must be positive for rolling")

    splits: list[TimeSplit] = []
    test_start = min_train_size + gap
    while test_start < n_obs:
        if scheme == "expanding":
            train_start = 0
        else:
            train_start = max(0, test_start - gap - train_size)
        train_end = test_start - gap
        test_end = min(test_start + test_size, n_obs)
        splits.append(_split(np.arange(train_start, train_end), np.arange(test_start, test_end)))
        test_start += step_size
    return splits


def fit_feature_stats(
    train_df: pd.DataFrame,
    columns: list[str] | None = None,
    min_std: float = 1e-12,
) -> FeatureStats:
    """Fit normalization statistics on training rows only."""
    if min_std <= 0.0:
        raise ValueError("min_std must be positive")
    if columns is None:
        columns = DEFAULT_SCORE_COLUMNS.copy()
        if "rel_spread_close" in train_df and train_df["rel_spread_close"].notna().any():
            columns.append("rel_spread_close")
    if not columns:
        raise ValueError("columns must be non-empty")
    _require_columns(train_df, columns)
    values = train_df[columns].apply(pd.to_numeric, errors="raise").astype(float)
    means = values.mean(skipna=True)
    stds = values.std(skipna=True)
    if means.isna().any():
        missing = means[means.isna()].index.tolist()
        raise ValueError(f"training columns have no finite rows: {missing}")
    stds = stds.mask(stds.isna() | (stds < min_std), 1.0)
    return FeatureStats(means=means, stds=stds, min_std=min_std)


def apply_feature_stats(df: pd.DataFrame, stats: FeatureStats) -> pd.DataFrame:
    """Apply train-only feature stats to any frame without refitting."""
    _require_columns(df, list(stats.means.index))
    values = df[list(stats.means.index)].apply(pd.to_numeric, errors="raise").astype(float)
    z = (values - stats.means) / stats.stds
    return z.replace([np.inf, -np.inf], np.nan)


def make_train_only_adaptive_scores(df: pd.DataFrame, stats: FeatureStats) -> pd.DataFrame:
    """Recompute adaptive scores using train-only fitted normalization."""
    z = apply_feature_stats(df, stats)
    _require_columns(z, DEFAULT_SCORE_COLUMNS)
    if "rel_spread_close" in z.columns:
        noise = z["rel_spread_close"] - z["log_volume"]
    else:
        noise = -z["log_volume"]
    out = pd.DataFrame(index=df.index)
    out["noise_score_raw"] = noise
    out["shock_score_raw"] = z["mid_return"].abs() + z["rolling_vol_20"]
    return out


def make_train_only_feature_frame(
    df: pd.DataFrame,
    feature_columns: list[str],
    stats: FeatureStats,
) -> pd.DataFrame:
    """Build model features without reading full-sample processed scores."""
    if not feature_columns:
        raise ValueError("feature_columns must be non-empty")
    scores = make_train_only_adaptive_scores(df, stats)
    z = apply_feature_stats(df, stats)
    out = pd.DataFrame(index=df.index)
    for column in feature_columns:
        if column in {"noise_score_raw", "shock_score_raw"}:
            out[column] = scores[column]
        elif column in z.columns:
            out[column] = z[column]
        else:
            _require_columns(df, [column])
            out[column] = pd.to_numeric(df[column], errors="raise").astype(float)
    return out.replace([np.inf, -np.inf], np.nan).dropna()


def expanding_zscore_series(s: pd.Series, min_periods: int = 20, min_std: float = 1e-12) -> pd.Series:
    """Z-score a series using only observations available up to each row."""
    if min_periods < 2:
        raise ValueError("min_periods must be at least 2")
    if min_std <= 0.0:
        raise ValueError("min_std must be positive")
    values = pd.to_numeric(s, errors="raise").astype(float)
    values = values.where(np.isfinite(values))
    mean = values.expanding(min_periods=min_periods).mean()
    std = values.expanding(min_periods=min_periods).std()
    result = (values - mean) / std
    result = result.where(std >= min_std, 0.0)
    return result.where(std.notna())


def make_leakage_safe_adaptive_scores(
    df: pd.DataFrame,
    min_periods: int = 20,
    min_std: float = 1e-12,
) -> pd.DataFrame:
    """Build adaptive penalty scores without full-sample standardization.

    The processed cache stores full-sample diagnostic scores for inspection.
    This function recomputes the scores with expanding statistics so row ``t``
    never uses information from rows after ``t``.
    """
    _require_columns(df, ["mid_return", "log_volume", "rolling_vol_20"])
    z_log_volume = expanding_zscore_series(df["log_volume"], min_periods=min_periods, min_std=min_std)
    z_mid_return = expanding_zscore_series(df["mid_return"], min_periods=min_periods, min_std=min_std)
    z_rolling_vol_20 = expanding_zscore_series(df["rolling_vol_20"], min_periods=min_periods, min_std=min_std)

    rel_spread = df.get("rel_spread_close")
    if rel_spread is not None and np.isfinite(pd.to_numeric(rel_spread, errors="raise").to_numpy(dtype=float)).any():
        z_rel_spread = expanding_zscore_series(rel_spread, min_periods=min_periods, min_std=min_std)
        noise = z_rel_spread - z_log_volume
    else:
        noise = -z_log_volume

    out = pd.DataFrame(index=df.index)
    out["noise_score_raw"] = noise
    out["shock_score_raw"] = z_mid_return.abs() + z_rolling_vol_20
    return out


def make_leakage_safe_feature_frame(
    df: pd.DataFrame,
    columns: list[str],
    min_periods: int = 20,
    standardize: bool = True,
) -> pd.DataFrame:
    """Return model features using expanding score replacements.

    Requests for ``noise_score_raw`` and ``shock_score_raw`` are satisfied by
    recomputing leakage-safe expanding scores, not by copying the full-sample
    diagnostic columns from the processed cache.
    """
    if not columns:
        raise ValueError("columns must be non-empty")
    safe_scores = make_leakage_safe_adaptive_scores(df, min_periods=min_periods)
    out = pd.DataFrame(index=df.index)
    for column in columns:
        if column in {"noise_score_raw", "shock_score_raw"}:
            out[column] = safe_scores[column]
        else:
            _require_columns(df, [column])
            out[column] = pd.to_numeric(df[column], errors="raise").astype(float)
    if standardize:
        for column in out.columns:
            out[column] = expanding_zscore_series(out[column], min_periods=min_periods)
    return out.replace([np.inf, -np.inf], np.nan).dropna()


def positions_from_states(
    states: np.ndarray | pd.Series,
    delay_bars: int = 1,
    invested_state: int = 0,
) -> np.ndarray | pd.Series:
    """Convert states to 0/1 positions with one close-to-next-bar lag.

    ``delay_bars=0`` means the state estimated after bar ``t`` first affects
    the position for bar ``t+1``. Larger delays add more bars of waiting.
    """
    if delay_bars < 1:
        raise ValueError("delay_bars must be at least 1 for leakage-safe backtests")
    is_series = isinstance(states, pd.Series)
    index = states.index if is_series else None
    state_array = _validate_states(states)
    signal = (state_array == invested_state).astype(float)
    lag = delay_bars
    position = np.zeros(len(signal), dtype=float)
    if lag < len(signal):
        position[lag:] = signal[:-lag]
    if is_series:
        return pd.Series(position, index=index, name="position")
    return position


def backtest_regime_01(
    returns: np.ndarray | pd.Series,
    states: np.ndarray | pd.Series,
    delay_bars: int = 1,
    transaction_cost: float = 0.0,
    periods_per_year: int = 252 * 390,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Vectorized 0/1 regime backtest with delay and one-way costs."""
    if transaction_cost < 0.0 or not np.isfinite(transaction_cost):
        raise ValueError("transaction_cost must be finite and nonnegative")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    r = _as_series(returns, "return")
    state_series = _as_series(states, "state", index=r.index)
    position = positions_from_states(state_series, delay_bars=delay_bars)
    turnover = position.diff().abs().fillna(position.abs())
    gross = position * r
    costs = transaction_cost * turnover
    net = gross - costs
    equity = (1.0 + net).cumprod()
    result = pd.DataFrame(
        {
            "return": r,
            "state": state_series.astype(int),
            "position": position,
            "gross_return": gross,
            "turnover": turnover,
            "cost": costs,
            "net_return": net,
            "equity": equity,
        },
        index=r.index,
    )
    return result, backtest_metrics(result["net_return"], result["position"], periods_per_year=periods_per_year)


def backtest_metrics(
    strategy_returns: np.ndarray | pd.Series,
    positions: np.ndarray | pd.Series,
    periods_per_year: int = 252 * 390,
) -> dict[str, float]:
    """Compute standard vectorized backtest metrics."""
    r = _as_series(strategy_returns, "strategy_returns").dropna()
    p = _as_series(positions, "positions").reindex(r.index).fillna(0.0)
    if len(r) == 0:
        raise ValueError("strategy_returns must contain at least one finite row")
    equity = (1.0 + r).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    if total_return <= -1.0:
        annualized_return = -1.0
    else:
        annualized_return = float((1.0 + total_return) ** (periods_per_year / len(r)) - 1.0)
    volatility = float(r.std(ddof=1) * np.sqrt(periods_per_year)) if len(r) > 1 else 0.0
    sharpe = float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year)) if len(r) > 1 and r.std(ddof=1) > 0 else np.nan
    drawdown = equity / equity.cummax() - 1.0
    max_drawdown = float(-drawdown.min())
    calmar = float(annualized_return / max_drawdown) if max_drawdown > 0.0 else np.nan
    cutoff = r.quantile(0.05)
    expected_shortfall = float(r[r <= cutoff].mean())
    turnover = float(p.diff().abs().fillna(p.abs()).sum())
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "expected_shortfall_5pct": expected_shortfall,
        "turnover": turnover,
        "n_trades": int((p.diff().abs().fillna(p.abs()) > 0).sum()),
        "exposure": float(p.mean()),
    }


def path_diagnostics(states: np.ndarray | pd.Series) -> dict[str, float]:
    """Summarize switching behavior for a regime path."""
    state_array = _validate_states(states)
    values, lengths = _runs(state_array)
    return {
        "n_obs": int(len(state_array)),
        "n_switches": int(np.sum(state_array[1:] != state_array[:-1])) if len(state_array) > 1 else 0,
        "average_duration": float(np.mean(lengths)),
        "min_duration": int(np.min(lengths)),
        "max_duration": int(np.max(lengths)),
        "fraction_state_0": float(np.mean(state_array == 0)),
        "fraction_state_1": float(np.mean(state_array == 1)),
    }


def summarize_regime_path(index, returns, states) -> pd.DataFrame:
    """Return one row of return and duration diagnostics per state."""
    idx = pd.Index(index)
    r = _as_series(returns, "returns", index=idx)
    state_array = _validate_states(states)
    if len(r) != len(state_array):
        raise ValueError("returns and states must have the same length")
    values, lengths = _runs(state_array)
    rows = []
    for state in sorted(np.unique(state_array)):
        mask = state_array == state
        run_lengths = lengths[values == state]
        state_returns = r.iloc[mask].dropna()
        rows.append(
            {
                "state": int(state),
                "count": int(mask.sum()),
                "fraction": float(mask.mean()),
                "mean_return": float(state_returns.mean()) if len(state_returns) else np.nan,
                "return_volatility": float(state_returns.std(ddof=1)) if len(state_returns) > 1 else 0.0,
                "average_duration": float(np.mean(run_lengths)),
                "min_duration": int(np.min(run_lengths)),
                "max_duration": int(np.max(run_lengths)),
            }
        )
    return pd.DataFrame(rows)


def relabel_states_by_realized_volatility(states, returns) -> np.ndarray:
    """Relabel a two-state path so state 0 has lower realized volatility."""
    mapping = state_mapping_by_realized_volatility(states, returns)
    return apply_state_mapping(states, mapping)


def state_mapping_by_realized_volatility(states, returns) -> dict[int, int]:
    """Create a train-only mapping where state 0 has lower return volatility."""
    state_array = _validate_states(states)
    r = _as_series(returns, "returns")
    if len(r) != len(state_array):
        raise ValueError("returns and states must have the same length")
    unique = sorted(np.unique(state_array))
    if len(unique) != 2:
        raise ValueError("realized-volatility relabeling currently requires exactly two states")
    vols = []
    for state in unique:
        state_returns = r.iloc[state_array == state].dropna()
        if len(state_returns) < 2:
            raise ValueError("each state must have at least two finite returns for volatility relabeling")
        vols.append((float(state_returns.std(ddof=1)), state))
    order = [state for _, state in sorted(vols)]
    return {old: new for new, old in enumerate(order)}


def apply_state_mapping(states, mapping: dict[int, int]) -> np.ndarray:
    """Apply an explicit state label mapping."""
    state_array = _validate_states(states)
    missing = sorted(set(state_array) - set(mapping))
    if missing:
        raise ValueError(f"mapping is missing states: {missing}")
    return np.array([mapping[state] for state in state_array], dtype=int)


def compare_state_paths(paths: dict[str, np.ndarray]) -> pd.DataFrame:
    """Return pairwise agreement and disagreement rates between paths."""
    if len(paths) < 2:
        raise ValueError("paths must contain at least two entries")
    validated = {name: _validate_states(path) for name, path in paths.items()}
    lengths = {len(path) for path in validated.values()}
    if len(lengths) != 1:
        raise ValueError("all paths must have the same length")
    names = list(validated)
    rows = []
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            agreement = float(np.mean(validated[left] == validated[right]))
            rows.append(
                {
                    "path_a": left,
                    "path_b": right,
                    "agreement": agreement,
                    "disagreement": 1.0 - agreement,
                }
            )
    return pd.DataFrame(rows)


def predict_causal_states(fit_costs: np.ndarray, switch_penalty: float | np.ndarray) -> np.ndarray:
    """Return the best terminal state after each prefix of the sequence.

    This is the causal counterpart to a retrospective DP path. The output at
    row ``t`` uses fit costs only from rows ``0..t``.
    """
    costs = _validate_fit_costs(fit_costs)
    penalties = _penalty_vector(switch_penalty, len(costs))
    n_steps, n_states = costs.shape
    dp = costs[0].copy()
    ranks = np.arange(n_states, dtype=int)
    states = np.empty(n_steps, dtype=int)
    states[0] = _best_terminal_state(dp, ranks)

    for t in range(1, n_steps):
        old_dp = dp
        old_ranks = ranks
        next_dp = np.empty(n_states, dtype=float)
        rank_keys: list[tuple[int, int]] = []
        for state in range(n_states):
            best_cost = old_dp[0] + (0.0 if state == 0 else penalties[t])
            best_rank = old_ranks[0]
            for candidate_prev in range(1, n_states):
                transition_cost = 0.0 if candidate_prev == state else penalties[t]
                candidate_cost = old_dp[candidate_prev] + transition_cost
                candidate_rank = old_ranks[candidate_prev]
                if candidate_cost < best_cost or (candidate_cost == best_cost and candidate_rank < best_rank):
                    best_cost = candidate_cost
                    best_rank = candidate_rank
            next_dp[state] = costs[t, state] + best_cost
            rank_keys.append((best_rank, state))
        ranks = np.empty(n_states, dtype=int)
        for rank, state in enumerate(sorted(range(n_states), key=lambda k: rank_keys[k])):
            ranks[state] = rank
        dp = next_dp
        states[t] = _best_terminal_state(dp, ranks)
    return states


def _as_series(values, name: str, index: pd.Index | None = None) -> pd.Series:
    if isinstance(values, pd.Series):
        result = values.astype(float)
        if index is not None:
            result = result.reindex(index)
    else:
        result = pd.Series(values, index=index, name=name, dtype=float)
    if not np.isfinite(result.dropna().to_numpy()).all():
        raise ValueError(f"{name} must be finite apart from NaN rows")
    return result


def _validate_states(states) -> np.ndarray:
    values = np.asarray(states, dtype=int)
    if values.ndim != 1:
        raise ValueError("states must be a 1-D array")
    if len(values) == 0:
        raise ValueError("states must be non-empty")
    if not np.isfinite(np.asarray(states, dtype=float)).all():
        raise ValueError("states must be finite")
    if (values < 0).any():
        raise ValueError("states must be nonnegative")
    return values


def _runs(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts = np.r_[0, np.flatnonzero(states[1:] != states[:-1]) + 1]
    ends = np.r_[starts[1:], len(states)]
    return states[starts], ends - starts


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"df is missing required columns: {missing}")


def _split(train: np.ndarray, test: np.ndarray) -> TimeSplit:
    if len(train) == 0 or len(test) == 0:
        raise ValueError("train and test splits must be non-empty")
    if train.max() >= test.min():
        raise ValueError("train rows must be strictly before test rows")
    return TimeSplit(train=train, test=test)


def _validate_fit_costs(fit_costs: np.ndarray) -> np.ndarray:
    costs = np.asarray(fit_costs, dtype=float)
    if costs.ndim != 2:
        raise ValueError("fit_costs must be a 2-D array")
    if costs.shape[0] == 0 or costs.shape[1] == 0:
        raise ValueError("fit_costs must have non-empty time and state dimensions")
    if not np.isfinite(costs).all():
        raise ValueError("fit_costs must be finite")
    return costs


def _penalty_vector(switch_penalty: float | np.ndarray, n_steps: int) -> np.ndarray:
    penalty = np.asarray(switch_penalty, dtype=float)
    if penalty.ndim == 0:
        value = float(penalty)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("switch_penalty must be finite and nonnegative")
        out = np.zeros(n_steps, dtype=float)
        out[1:] = value
        return out
    if penalty.ndim != 1:
        raise ValueError("switch_penalty must be a scalar or 1-D array")
    if len(penalty) != n_steps:
        raise ValueError("1-D switch_penalty must have length equal to fit_costs rows")
    if not np.isfinite(penalty).all() or (penalty < 0.0).any():
        raise ValueError("switch_penalty must be finite and nonnegative")
    return penalty


def _best_terminal_state(costs: np.ndarray, ranks: np.ndarray) -> int:
    best = 0
    for state in range(1, len(costs)):
        if costs[state] < costs[best] or (costs[state] == costs[best] and ranks[state] < ranks[best]):
            best = state
    return int(best)
