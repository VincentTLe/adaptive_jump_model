"""Tests for the frozen holdout window readout mechanics."""

from pathlib import Path

import pandas as pd
import pytest

import adaptive_jump.holdout_runner as holdout
from adaptive_jump.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def _trades(dates: pd.DatetimeIndex, position: list[float]) -> pd.DataFrame:
    equity = pd.Series(0.01, index=range(len(dates)))
    cash = pd.Series(0.0001, index=range(len(dates)))
    pos = pd.Series(position, dtype=float)
    turnover = pos.diff().abs().fillna(pos.abs())
    gross = pos * equity + (1 - pos) * cash
    cost = turnover * 0.001
    return pd.DataFrame(
        {
            "date": dates,
            "equity_simple": equity,
            "cash_return": cash,
            "signal": pos,
            "position": pos,
            "gross_return": gross,
            "one_way_turnover": turnover,
            "transaction_cost": cost,
            "strategy_return": gross - cost,
        }
    )


def test_window_slices_inclusive_bounds() -> None:
    dates = pd.bdate_range("2023-12-27", periods=8)
    frame = _trades(dates, [1.0] * 8)
    window = holdout._window(
        frame, pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-04")
    )
    assert window["date"].min() >= pd.Timestamp("2024-01-02")
    assert window["date"].max() <= pd.Timestamp("2024-01-04")
    with pytest.raises(holdout.HoldoutError):
        holdout._window(frame, pd.Timestamp("2030-01-01"), pd.Timestamp("2030-02-01"))


def test_metric_row_counts_switches_and_cash_fraction() -> None:
    config = load_config(ROOT / "research.toml")
    dates = pd.bdate_range("2024-01-02", periods=6)
    frame = _trades(dates, [0.0, 1.0, 1.0, 0.0, 0.0, 1.0])
    row = holdout._metric_row(frame, config)
    assert row["switch_count"] == 3
    assert row["cash_fraction"] == pytest.approx(0.5)
    assert "sharpe" in row and "maximum_drawdown" in row


def test_spec_requires_frozen_registration(tmp_path: Path) -> None:
    (tmp_path / "research").mkdir()
    (tmp_path / "research" / holdout.SPEC_NAME).write_text("schema_version = 1\n")
    (tmp_path / "research" / "experiment_registry.jsonl").write_text("")
    with pytest.raises(holdout.HoldoutError):
        holdout.load_holdout_spec(tmp_path)


def test_render_holdout_figure(tmp_path: Path) -> None:
    metrics = pd.DataFrame(
        [
            {"market": m, "model": model, "window": "holdout", "sharpe": s}
            for m, model, s in [
                ("us", "buy_and_hold", 1.05),
                ("us", "hmm", 0.53),
                ("us", "fixed_jm", 0.57),
                ("us", "dd_only", 0.78),
                ("de", "buy_and_hold", 0.90),
                ("de", "hmm", 0.90),
                ("de", "fixed_jm", 0.90),
                ("de", "dd_only", 0.90),
                ("jp", "buy_and_hold", 1.27),
                ("jp", "hmm", 1.27),
                ("jp", "fixed_jm", 1.27),
                ("jp", "dd_only", 1.17),
            ]
        ]
    )
    metrics.to_csv(tmp_path / "holdout-metrics.csv", index=False)
    from adaptive_jump.holdout_runner import render_holdout_figure

    target = render_holdout_figure(tmp_path)
    assert target.exists()
    assert target.stat().st_size > 1000
