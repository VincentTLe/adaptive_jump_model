"""Tests for the lagged-evidence whipsaw mechanism figure."""

from pathlib import Path

import pandas as pd
import pytest

import adaptive_jump.lagged_figures as figures


def _summary(tmp_path: Path) -> Path:
    rows = []
    data = {
        "us": {"arrival": (12, 6, 6), "lagged": (6, 2, 4)},
        "de": {"arrival": (14, 6, 8), "lagged": (5, 3, 2)},
        "jp": {"arrival": (12, 5, 7), "lagged": (6, 1, 5)},
    }
    for market, rules in data.items():
        for rule, (events, whipsaw, persistent) in rules.items():
            rows.append(
                {
                    "market": market,
                    "beta": figures.LOG4,
                    "rule": rule,
                    "event_count": events,
                    "whipsaw_count": whipsaw,
                    "persistent_count": persistent,
                }
            )
        # a log2 row that must be excluded
        rows.append(
            {
                "market": market,
                "beta": 0.6931471805599453,
                "rule": "arrival",
                "event_count": 1,
                "whipsaw_count": 0,
                "persistent_count": 1,
            }
        )
    pd.DataFrame(rows).to_csv(tmp_path / "mechanism-summary.csv", index=False)
    return tmp_path


def test_load_log4_isolates_beta_and_sums_pooled(tmp_path: Path) -> None:
    pivot = figures.load_log4_mechanism(_summary(tmp_path))
    markets = ("us", "de", "jp")
    arrival = [int(pivot.loc[m, ("whipsaw_count", "arrival")]) for m in markets]
    lagged = [int(pivot.loc[m, ("whipsaw_count", "lagged")]) for m in markets]
    assert sum(arrival) == 17
    assert sum(lagged) == 6


def test_render_writes_nonempty_figure(tmp_path: Path) -> None:
    run = _summary(tmp_path)
    out = figures.render_whipsaw_figure(run, tmp_path / "out" / "whipsaw.png")
    assert out.exists()
    assert out.stat().st_size > 1000


def test_missing_columns_raise(tmp_path: Path) -> None:
    pd.DataFrame({"market": ["us"], "beta": [figures.LOG4]}).to_csv(
        tmp_path / "mechanism-summary.csv", index=False
    )
    with pytest.raises(figures.LaggedFigureError):
        figures.load_log4_mechanism(tmp_path)
