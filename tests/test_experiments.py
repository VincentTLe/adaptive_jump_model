import numpy as np
import pandas as pd
import pytest

import adaptive_jump.library_checks as library_checks
from adaptive_jump.dp import solve_regime_path
from adaptive_jump.backtesting import (
    backtest_regime_01,
    positions_from_states,
    round_trips_from_backtest_frame,
    trade_events_from_backtest_frame,
)
from adaptive_jump.experiments import (
    apply_feature_stats,
    compare_state_paths,
    fit_feature_stats,
    make_time_splits,
    make_train_only_adaptive_scores,
    make_train_only_feature_frame,
    path_diagnostics,
    predict_causal_states,
    relabel_states_by_realized_volatility,
    apply_state_mapping,
    state_mapping_by_realized_volatility,
    summarize_regime_path,
)
from adaptive_jump.library_checks import quantstats_metric_check, vectorbt_signal_check


def test_time_splits_are_ordered_and_non_overlapping():
    splits = make_time_splits(range(20), scheme="rolling", train_size=8, test_size=3, step_size=3, min_train_size=8, gap=1)

    assert len(splits) == 4
    for split in splits:
        assert split.train.max() < split.test.min()
        assert len(split.train) <= 8
        assert len(split.test) <= 3


def test_train_only_scaling_ignores_test_distribution():
    train = _feature_frame(0.0)
    test_a = _feature_frame(10.0)
    test_b = _feature_frame(10_000.0)

    stats = fit_feature_stats(train)
    z_a = apply_feature_stats(test_a, stats)
    z_b = apply_feature_stats(test_b, stats)

    pd.testing.assert_series_equal(stats.means, fit_feature_stats(train).means)
    assert not np.allclose(z_a["log_volume"], z_b["log_volume"])


def test_processed_raw_scores_are_not_used_for_live_feature_frame():
    df = _feature_frame(0.0)
    df["noise_score_raw"] = 1_000_000.0
    df["shock_score_raw"] = -1_000_000.0
    stats = fit_feature_stats(df.iloc[:10])

    first = make_train_only_feature_frame(df, ["noise_score_raw", "shock_score_raw"], stats)
    df["noise_score_raw"] = -1_000_000.0
    df["shock_score_raw"] = 1_000_000.0
    second = make_train_only_feature_frame(df, ["noise_score_raw", "shock_score_raw"], stats)

    pd.testing.assert_frame_equal(first, second)


def test_train_only_adaptive_scores_use_spread_when_available():
    df = _feature_frame(0.0)
    df["rel_spread_close"] = np.linspace(0.001, 0.003, len(df))
    stats = fit_feature_stats(df.iloc[:10])

    result = make_train_only_adaptive_scores(df, stats)

    assert list(result.columns) == ["noise_score_raw", "shock_score_raw"]
    assert result["noise_score_raw"].notna().any()


def test_causal_states_match_prefix_dp_and_are_future_invariant():
    fit_costs = np.array(
        [
            [0.0, 5.0],
            [1.0, 4.0],
            [4.0, 1.0],
            [5.0, 0.0],
        ]
    )

    result = predict_causal_states(fit_costs, switch_penalty=0.5)
    oracle = [solve_regime_path(fit_costs[: i + 1], 0.5).states[-1] for i in range(len(fit_costs))]
    extended = np.vstack([fit_costs, [100.0, 0.0], [100.0, 0.0]])

    assert result.tolist() == oracle
    assert predict_causal_states(extended, 0.5)[: len(fit_costs)].tolist() == result.tolist()


def test_backtest_delay_uses_prior_signal_without_extra_shift():
    returns = pd.Series([0.10, 0.20, 0.30])
    states = pd.Series([0, 1, 0])

    frame, metrics = backtest_regime_01(returns, states, delay_bars=1, transaction_cost=0.0, periods_per_year=3)

    assert frame["position"].tolist() == [0.0, 1.0, 0.0]
    assert np.isnan(frame["signal_state"].iloc[0])
    assert frame["signal_state"].iloc[1:].tolist() == [0.0, 1.0]
    assert frame["net_return"].tolist() == pytest.approx([0.0, 0.20, 0.0])
    assert metrics["total_return"] == pytest.approx(0.20)


def test_positions_require_positive_delay_for_backtest_safety():
    with pytest.raises(ValueError, match="at least 1"):
        positions_from_states(np.array([0, 1, 0]), delay_bars=0)


def test_transaction_cost_is_one_way_turnover_cost():
    returns = pd.Series([0.0, 0.0, 0.0, 0.0])
    states = pd.Series([0, 1, 0, 1])

    frame, metrics = backtest_regime_01(returns, states, delay_bars=1, transaction_cost=0.001)

    assert frame["position"].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert frame["turnover"].sum() == pytest.approx(3.0)
    assert frame["cost"].sum() == pytest.approx(0.003)
    assert metrics["n_trades"] == 3


def test_trade_event_log_records_delayed_source_state():
    index = pd.date_range("2026-01-01 09:30", periods=4, freq="min")
    returns = pd.Series([0.0, 0.01, -0.02, 0.03], index=index)
    states = pd.Series([0, 1, 0, 1], index=index)
    frame, _ = backtest_regime_01(returns, states, delay_bars=1, transaction_cost=0.001, periods_per_year=4)

    events = trade_events_from_backtest_frame(frame, "TST", "Fixed JM", delay_bars=1, transaction_cost=0.001)

    assert events["timestamp"].tolist() == index[1:].tolist()
    assert events["source_state_timestamp"].tolist() == index[:-1].tolist()
    assert events["source_state"].tolist() == [0.0, 1.0, 0.0]
    assert events["side"].tolist() == ["buy", "sell", "buy"]
    assert events["position_timing"].unique().tolist() == ["delayed_by_1_bars"]


def test_round_trip_log_pairs_closed_and_open_trades():
    index = pd.date_range("2026-01-01 09:30", periods=4, freq="min")
    returns = pd.Series([0.0, 0.01, -0.02, 0.03], index=index)
    states = pd.Series([0, 1, 0, 1], index=index)
    frame, _ = backtest_regime_01(returns, states, delay_bars=1, transaction_cost=0.001, periods_per_year=4)

    trips = round_trips_from_backtest_frame(frame, "TST", "Adaptive JM", delay_bars=1, transaction_cost=0.001)

    assert trips["status"].tolist() == ["closed", "open"]
    assert trips.loc[0, "entry_timestamp"] == index[1]
    assert trips.loc[0, "exit_timestamp"] == index[2]
    assert trips.loc[0, "holding_bars"] == 1
    assert trips.loc[1, "entry_timestamp"] == index[3]
    assert pd.isna(trips.loc[1, "exit_timestamp"])


def test_library_backtest_checks_return_finite_metrics():
    returns = pd.Series(np.linspace(-0.02, 0.02, 100))
    positions = pd.Series(([0.0] * 10 + [1.0] * 60 + [0.0] * 30))
    price = (1.0 + returns).cumprod()

    qs_metrics = quantstats_metric_check(returns, periods_per_year=4)
    vbt_metrics = vectorbt_signal_check(price, positions, transaction_cost=0.001, periods_per_year=4)

    assert qs_metrics["quantstats_status"] in {"ok", "missing"}
    assert vbt_metrics["vectorbt_status"] in {"ok", "missing"}
    if qs_metrics["quantstats_status"] == "ok":
        assert np.isfinite(qs_metrics["quantstats_sharpe"])
    if vbt_metrics["vectorbt_status"] == "ok":
        assert vbt_metrics["vectorbt_trades"] == 1
        assert np.isfinite(vbt_metrics["vectorbt_total_return"])


def test_library_backtest_checks_are_optional(monkeypatch):
    def missing_package(name):
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(library_checks, "import_module", missing_package)
    returns = pd.Series(np.linspace(-0.01, 0.01, 20))
    positions = pd.Series(([0.0] * 5 + [1.0] * 10 + [0.0] * 5))
    price = (1.0 + returns).cumprod()

    qs_metrics = quantstats_metric_check(returns, periods_per_year=4)
    vbt_metrics = vectorbt_signal_check(price, positions, transaction_cost=0.001, periods_per_year=4)

    assert qs_metrics["quantstats_status"] == "missing"
    assert vbt_metrics["vectorbt_status"] == "missing"
    assert np.isnan(qs_metrics["quantstats_sharpe"])
    assert np.isnan(vbt_metrics["vectorbt_total_return"])


def test_path_diagnostics_and_summary_are_correct():
    states = np.array([0, 0, 1, 1, 1, 0])
    returns = pd.Series([0.01, 0.02, -0.01, -0.02, -0.03, 0.01])

    diagnostics = path_diagnostics(states)
    summary = summarize_regime_path(returns.index, returns, states)

    assert diagnostics["n_obs"] == 6
    assert diagnostics["n_switches"] == 2
    assert diagnostics["average_duration"] == pytest.approx(2.0)
    assert set(summary.columns) == {
        "state",
        "count",
        "fraction",
        "mean_return",
        "return_volatility",
        "average_duration",
        "min_duration",
        "max_duration",
    }


def test_relabel_states_by_realized_volatility():
    states = np.array([1, 1, 1, 0, 0, 0])
    returns = pd.Series([0.001, -0.001, 0.001, 0.10, -0.10, 0.12])

    relabeled = relabel_states_by_realized_volatility(states, returns)

    assert relabeled.tolist() == [0, 0, 0, 1, 1, 1]


def test_state_mapping_is_fit_on_train_only():
    train_states = np.array([1, 1, 1, 0, 0, 0])
    train_returns = pd.Series([0.001, -0.001, 0.001, 0.10, -0.10, 0.12])
    test_states = np.array([1, 0])
    test_returns_extreme = pd.Series([100.0, 0.001])

    mapping = state_mapping_by_realized_volatility(train_states, train_returns)
    relabeled_test = apply_state_mapping(test_states, mapping)

    assert mapping == {1: 0, 0: 1}
    assert relabeled_test.tolist() == [0, 1]
    assert test_returns_extreme.iloc[0] > test_returns_extreme.iloc[1]


def test_compare_state_paths_returns_pairwise_agreement():
    result = compare_state_paths({"a": np.array([0, 1, 1]), "b": np.array([0, 0, 1]), "c": np.array([1, 1, 1])})

    assert len(result) == 3
    assert result.loc[(result["path_a"] == "a") & (result["path_b"] == "b"), "agreement"].iloc[0] == pytest.approx(2 / 3)


def _feature_frame(offset: float) -> pd.DataFrame:
    n = 30
    return pd.DataFrame(
        {
            "log_volume": np.linspace(1.0 + offset, 2.0 + offset, n),
            "mid_return": np.linspace(-0.01 + offset, 0.01 + offset, n),
            "rolling_vol_20": np.linspace(0.001 + offset, 0.003 + offset, n),
        }
    )
