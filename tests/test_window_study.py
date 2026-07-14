from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adaptive_jump.backtest import apply_signal, buy_and_hold
from adaptive_jump.config import load_config
from adaptive_jump.window_spec import load_window_spec
from adaptive_jump.window_study import (
    COMPARISON_MODELS,
    WindowStudyError,
    align_comparison_paths,
    bootstrap_rows,
    comparison_metrics,
    window_claim,
)

ROOT = Path(__file__).resolve().parents[1]


def _contract():
    config = load_config(ROOT / "research.toml")
    spec = load_window_spec(ROOT / "research/jm-train-window-sensitivity.toml", config)
    return config, spec


def _paths(periods: int = 320) -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2020-01-02", periods=periods)
    rng = np.random.default_rng(8)
    returns = pd.DataFrame(
        {
            "date": dates,
            "equity_simple": rng.normal(0.0004, 0.01, periods),
            "cash_return": 0.00005,
        }
    )
    signals = {
        "hmm_3000": pd.Series((np.arange(periods) % 19 != 0).astype(float)),
        "jm_3000": pd.Series((np.arange(periods) % 23 != 0).astype(float)),
        "jm_4000": pd.Series((np.arange(periods) % 29 != 0).astype(float)),
    }
    return {
        "buy_and_hold": buy_and_hold(returns),
        **{
            model: apply_signal(returns, signal, delay_trading_days=1)
            for model, signal in signals.items()
        },
    }


def test_alignment_uses_identical_complete_post_eligibility_rows() -> None:
    paths = _paths()
    paths["buy_and_hold"] = paths["buy_and_hold"].iloc[5:].reset_index(drop=True)
    start = paths["jm_4000"].loc[12, "date"].date()

    aligned = align_comparison_paths(paths, oos_start=start)

    dates = [path["date"] for path in aligned.values()]
    assert all(values.equals(dates[0]) for values in dates[1:])
    assert dates[0].iloc[0].date() >= start
    assert all(not path.isna().any().any() for path in aligned.values())


def test_alignment_rejects_different_market_returns() -> None:
    paths = _paths()
    paths["jm_4000"].loc[20, "equity_simple"] += 0.01

    with pytest.raises(WindowStudyError, match="market returns differ"):
        align_comparison_paths(paths, oos_start=paths["jm_4000"].loc[10, "date"].date())


def test_metrics_add_cash_fraction_and_switch_count() -> None:
    config, _ = _contract()
    paths = align_comparison_paths(
        _paths(), oos_start=pd.Timestamp("2020-01-20").date()
    )

    metrics = comparison_metrics(paths, config)

    assert tuple(metrics["model"]) == COMPARISON_MODELS
    assert metrics["cash_fraction"].between(0, 1).all()
    assert (metrics["switch_count"] >= 0).all()


def test_bootstrap_rows_use_all_frozen_blocks_and_are_deterministic() -> None:
    config, spec = _contract()
    spec = replace(spec, bootstrap_replications=40)
    paths = align_comparison_paths(
        _paths(), oos_start=pd.Timestamp("2020-01-20").date()
    )

    first = bootstrap_rows(paths, spec, config)
    second = bootstrap_rows(paths, spec, config)

    pd.testing.assert_frame_equal(first, second)
    assert tuple(first["block_length"]) == (60, 20, 120)
    assert (first["replications"] == 40).all()


@pytest.mark.parametrize(
    ("deltas", "outcome"),
    [
        ((0.2, 0.1, 0.3), "consistent improvement"),
        ((0.2, -0.1, 0.3), "mixed"),
        ((-0.2, -0.1, -0.3), "not supported"),
    ],
)
def test_claim_applies_frozen_three_market_rule(deltas, outcome) -> None:
    markets = ("us", "de", "jp")
    metric_rows = []
    bootstrap_rows_fixture = []
    for market, delta in zip(markets, deltas, strict=True):
        for model in COMPARISON_MODELS:
            sharpe = 0.5 + delta if model == "jm_4000" else 0.5
            metric_rows.append(
                {"market": market, "model": model, "delay": 1, "sharpe": sharpe}
            )
        bootstrap_rows_fixture.append(
            {
                "market": market,
                "block_length": 60,
                "observed_delta": delta,
                "lower_one_sided": delta - 0.05,
            }
        )

    claim = window_claim(
        pd.DataFrame(metric_rows),
        pd.DataFrame(bootstrap_rows_fixture),
        market_ids=markets,
        primary_delay=1,
        primary_block=60,
    )

    assert claim["directional_outcome"] == outcome
    assert claim["uncertainty_supported"] is (outcome == "consistent improvement")
