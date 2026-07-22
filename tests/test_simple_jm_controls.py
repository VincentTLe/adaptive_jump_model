from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adaptive_jump.artifacts import sha256_file
from adaptive_jump.backtest import BacktestError
from adaptive_jump.simple_jm_controls import (
    SIGNAL_TO_RETURN_OFFSET,
    SimpleJMControlError,
    build_confirmed_control_path,
    build_control_path,
    build_static_lambda50_path,
    confirm_two_observations,
    load_static_lambda50_states,
    signal_from_states,
    states_from_signal,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ([0, 1, 0], [0, 0, 0]),
        ([0, 1, 1], [0, 0, 1]),
        ([1, 0, 0, 1, 1], [1, 1, 0, 0, 1]),
    ],
)
def test_two_observation_confirmation_toy_paths(raw, expected) -> None:
    observed = confirm_two_observations(pd.Series(raw, dtype=float))

    np.testing.assert_array_equal(observed.to_numpy(), expected)


def test_confirmation_preserves_leading_missing_and_is_prefix_invariant() -> None:
    raw = pd.Series([np.nan, np.nan, 0, 1, 1, 0, 0], dtype=float)

    full = confirm_two_observations(raw)
    prefix = confirm_two_observations(raw.iloc[:5])

    assert full.iloc[:2].isna().all()
    np.testing.assert_array_equal(full.iloc[2:].to_numpy(), [0, 0, 1, 1, 0])
    pd.testing.assert_series_equal(full.iloc[:5], prefix)


def test_control_error_uses_backtest_hierarchy() -> None:
    assert issubclass(SimpleJMControlError, BacktestError)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([np.nan, np.nan], "no finite observation"),
        ([0.0, np.nan, 1.0], "missing values after it starts"),
        ([0.0, 2.0], "only 0, 1"),
        ([0.0, np.inf], "only 0, 1"),
    ],
)
def test_confirmation_rejects_undefined_paths(values, message) -> None:
    with pytest.raises(SimpleJMControlError, match=message):
        confirm_two_observations(pd.Series(values))


def test_state_signal_mapping_is_exact_and_symmetric() -> None:
    state = pd.Series([np.nan, 0.0, 1.0, 0.0])

    signal = signal_from_states(state)
    roundtrip = states_from_signal(signal)

    np.testing.assert_array_equal(signal.to_numpy(), [np.nan, 1.0, 0.0, 1.0])
    np.testing.assert_array_equal(roundtrip.to_numpy(), state.to_numpy())


def _returns(rows: int = 7) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2023-01-02", periods=rows),
            "equity_simple": np.linspace(-0.01, 0.02, rows),
            "cash_return": 0.0001,
        }
    )


def test_control_path_uses_t_plus_two_and_full_one_way_cost() -> None:
    returns = _returns()
    dates = pd.DatetimeIndex(returns["date"], name="date")
    state = pd.Series([0, 1, 1, 0, 0, 1, 1], index=dates, dtype=float)

    result = build_control_path(returns, state)

    expected_signal = pd.Series([1, 0, 0, 1, 1, 0, 0], dtype=float)
    expected_position = expected_signal.shift(SIGNAL_TO_RETURN_OFFSET)
    np.testing.assert_array_equal(
        result.trades["position"].to_numpy(),
        expected_position.to_numpy(),
    )
    assert result.trades.loc[2, "one_way_turnover"] == 0.0
    assert result.trades.loc[3, "one_way_turnover"] == 1.0
    assert result.trades.loc[3, "transaction_cost"] == pytest.approx(0.001)
    assert result.trades.loc[4, "transaction_cost"] == 0.0


def test_control_paths_keep_leading_missing_return_calendar(tmp_path: Path) -> None:
    returns = _returns()
    returns.loc[:1, "cash_return"] = np.nan
    dates = pd.DatetimeIndex(returns["date"], name="date")
    raw_state = pd.Series([np.nan, np.nan, 0, 0, 1, 1, 0], index=dates, dtype=float)
    canonical_signal = 1.0 - raw_state
    _, digest = _write_lambda50_source(tmp_path, raw_state.to_list())

    direct = build_control_path(returns, raw_state)
    static = build_static_lambda50_path(returns, tmp_path, "us", expected_sha256=digest)
    confirmed = build_confirmed_control_path(returns, canonical_signal)

    assert direct.trades["date"].equals(returns["date"])
    assert static.trades["date"].equals(returns["date"])
    assert confirmed.trades["date"].equals(returns["date"])
    assert direct.trades.loc[:3, "position"].isna().all()
    assert static.trades.loc[4, "position"] == static.signal.iloc[2]
    assert direct.trades.loc[4, "position"] == direct.signal.iloc[2]
    assert confirmed.trades.loc[4, "position"] == confirmed.signal.iloc[2]


def test_control_path_rejects_missing_return_where_position_is_finite() -> None:
    returns = _returns()
    dates = pd.DatetimeIndex(returns["date"], name="date")
    state = pd.Series([np.nan, np.nan, 0, 0, 1, 1, 0], index=dates, dtype=float)
    returns.loc[4, "cash_return"] = np.nan

    with pytest.raises(SimpleJMControlError, match="wherever position is finite"):
        build_control_path(returns, state)


def test_control_path_rejects_infinite_return_even_before_position() -> None:
    returns = _returns()
    dates = pd.DatetimeIndex(returns["date"], name="date")
    state = pd.Series([np.nan, np.nan, 0, 0, 1, 1, 0], index=dates, dtype=float)
    returns.loc[0, "cash_return"] = np.inf

    with pytest.raises(SimpleJMControlError, match="finite when present"):
        build_control_path(returns, state)


def test_confirmed_control_filters_state_before_unchanged_accounting() -> None:
    returns = _returns()
    dates = pd.DatetimeIndex(returns["date"], name="date")
    raw_state = pd.Series([0, 1, 1, 0, 0, 1, 1], index=dates, dtype=float)
    canonical_signal = 1.0 - raw_state

    result = build_confirmed_control_path(returns, canonical_signal)

    np.testing.assert_array_equal(result.state.to_numpy(), [0, 0, 1, 1, 0, 0, 1])
    np.testing.assert_array_equal(result.signal.to_numpy(), [1, 1, 0, 0, 1, 1, 0])
    assert result.trades.loc[4, "position"] == result.signal.iloc[2]


def _write_lambda50_source(
    root: Path, values: list[float], *, market: str = "us", start: str = "2023-01-02"
) -> tuple[Path, str]:
    target = root / market / "jm-missing-states.csv"
    target.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": pd.bdate_range(start, periods=len(values)),
            "10.0": 0.0,
            "50.0": values,
            "100.0": 1.0,
        }
    ).to_csv(target, index=False)
    return target, sha256_file(target)


def test_static_lambda50_load_is_hash_pinned_and_builds_no_selection_layer(
    tmp_path: Path,
) -> None:
    values = [np.nan, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
    source, digest = _write_lambda50_source(tmp_path, values)
    returns = _returns()

    state = load_static_lambda50_states(tmp_path, "us", expected_sha256=digest)
    result = build_static_lambda50_path(returns, tmp_path, "us", expected_sha256=digest)

    assert source.is_file()
    assert state.name == "state"
    np.testing.assert_array_equal(state.to_numpy(), values)
    pd.testing.assert_series_equal(result.state, state)
    assert not hasattr(result, "choices")
    assert not hasattr(result, "surface")


def test_static_lambda50_rejects_hash_schema_and_cutoff_violations(
    tmp_path: Path,
) -> None:
    _, digest = _write_lambda50_source(tmp_path, [np.nan, 0.0, 1.0])

    with pytest.raises(SimpleJMControlError, match="hash changed"):
        load_static_lambda50_states(tmp_path, "us", expected_sha256="0" * 64)
    with pytest.raises(SimpleJMControlError, match="64 lowercase hex"):
        load_static_lambda50_states(tmp_path, "us", expected_sha256="bad")
    with pytest.raises(SimpleJMControlError, match="unknown market"):
        load_static_lambda50_states(tmp_path, "xx", expected_sha256=digest)

    late_root = tmp_path / "late"
    _, late_digest = _write_lambda50_source(late_root, [0.0, 1.0], start="2024-01-02")
    with pytest.raises(SimpleJMControlError, match="post-cutoff"):
        load_static_lambda50_states(
            late_root,
            "us",
            expected_sha256=late_digest,
            cutoff=date(2023, 12, 31),
        )


def test_static_lambda50_rejects_internal_missing_and_nonbinary_values(
    tmp_path: Path,
) -> None:
    missing_root = tmp_path / "missing"
    _, missing_digest = _write_lambda50_source(missing_root, [np.nan, 0.0, np.nan])
    with pytest.raises(SimpleJMControlError, match="missing values after it starts"):
        load_static_lambda50_states(missing_root, "us", expected_sha256=missing_digest)

    invalid_root = tmp_path / "invalid"
    _, invalid_digest = _write_lambda50_source(invalid_root, [0.0, 2.0])
    with pytest.raises(SimpleJMControlError, match="only 0, 1"):
        load_static_lambda50_states(invalid_root, "us", expected_sha256=invalid_digest)


def test_control_path_rejects_misaligned_and_post_cutoff_dates() -> None:
    returns = _returns(4)
    dates = pd.DatetimeIndex(returns["date"], name="date")
    shifted = pd.Series([0.0] * 4, index=dates + pd.Timedelta(days=1))
    with pytest.raises(SimpleJMControlError, match="must match exactly"):
        build_control_path(returns, shifted)

    late_returns = returns.copy()
    late_returns.loc[3, "date"] = pd.Timestamp("2024-01-02")
    late_state = pd.Series(
        [0.0] * 4,
        index=pd.DatetimeIndex(late_returns["date"], name="date"),
    )
    with pytest.raises(SimpleJMControlError, match="post-cutoff"):
        build_control_path(late_returns, late_state)
