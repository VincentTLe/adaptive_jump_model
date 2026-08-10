"""Audit regressions for the held-out-delay experiment (heldout-delay-001).

Written by the independent auditor of that experiment. Each test pins a
property whose violation was reachable by a single edit during fault injection
and was caught by NOTHING that existed at the time:

* the Table-5 transcription is never compared with the paper, so a transposed
  or column-shifted block scores silently (fault F11);
* the grade bands are never compared with the scale they cite, so reversing
  them scores silently (fault F09);
* the delay is only asserted to *change the answer*, never asserted to reach
  the cross-validation and the execution separately (faults F01/F02);
* the sealed row that supplies the evaluation window is keyed by the delay,
  but every sealed window happens to be identical, so pinning that lookup to
  delay 1 is undetectable on the real data (fault F10);
* nothing binds the arms actually run to the arms in the frozen spec, or the
  frozen spec to its registry hash (fault F04).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "data/external/inputs/shu_paper.txt"
SPEC = ROOT / "research/heldout-delay-001.toml"
REGISTRY = ROOT / "research/experiment_registry.jsonl"
MARKET_HEADER = {"S&P 500": "us", "DAX": "de", "Nikkei 225": "jp"}
ROW_LABEL = {"Return": "cagr", "Sharpe": "sharpe", "Calmar": "calmar"}


def _module(name: str):
    """Import a script by path without running its ``main()``."""
    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        f"{name}_under_audit", ROOT / "scripts" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def table5():
    return _module("_shu_table5")


@pytest.fixture(scope="module")
def scorer():
    return _module("score_grid")


# --------------------------------------------------------------------------
# 1. the transcription, read back out of the paper
# --------------------------------------------------------------------------


def _parse_table5_from_paper() -> dict[str, dict[int, dict[str, float]]]:
    """Re-read Table 5 out of the extracted paper text.

    The block is three market panels; each panel is a header naming the index,
    a ``delay = 1  5  10`` line, and three metric rows. Every metric row prints
    seven figures: buy-and-hold, then HMM at delays 1/5/10, then JM at delays
    1/5/10. The JM block is therefore the LAST three, and reading it from any
    other offset is the transposition this parser exists to expose.
    """
    lines = PAPER.read_text(encoding="utf-8").splitlines()
    parsed: dict[str, dict[int, dict[str, float]]] = {}
    market: str | None = None
    for line in lines:
        for header, key in MARKET_HEADER.items():
            if re.match(rf"\s*{re.escape(header)}\s+HMM\s+JM\s*$", line):
                market = key
                parsed.setdefault(market, {1: {}, 5: {}, 10: {}})
        if market is None:
            continue
        label = line.strip().split(" ")[0] if line.strip() else ""
        if label not in ROW_LABEL:
            continue
        figures = re.findall(r"-?\d+\.\d+%?", line)
        assert len(figures) == 7, f"{market} {label}: {len(figures)} figures, want 7"
        jm = figures[-3:]
        for delay, text in zip((1, 5, 10), jm, strict=True):
            value = float(text.rstrip("%")) / (100.0 if text.endswith("%") else 1.0)
            parsed[market][delay][ROW_LABEL[label]] = value
    return parsed


def test_table5_transcription_matches_the_paper_text(table5):
    """Every figure in TABLE5_JM is the figure the paper prints, in its place."""
    parsed = _parse_table5_from_paper()
    assert set(parsed) == set(table5.TABLE5_JM) == {"us", "de", "jp"}
    for market, by_delay in table5.TABLE5_JM.items():
        for delay, cells in by_delay.items():
            for cell, value in cells.items():
                assert parsed[market][delay][cell] == pytest.approx(value, abs=1e-12), (
                    f"{market} delay {delay} {cell}: transcribed {value}, "
                    f"paper prints {parsed[market][delay][cell]}"
                )


def test_table5_delay_1_column_duplicates_table_4(table5):
    """The docstring's in-sample claim, checked rather than asserted in prose."""
    table4 = _module("_shu_table4")
    for market, by_delay in table5.TABLE5_JM.items():
        published = table4.TABLE4[market]["fixed_jm"]
        for cell, value in by_delay[1].items():
            assert published[cell] == pytest.approx(value, abs=1e-12), (
                f"{market} {cell}: Table 5 delay 1 is {value}, Table 4 is "
                f"{published[cell]}; they must be the same strategy"
            )


def test_delay_1_can_never_enter_the_held_out_set(table5):
    assert 1 not in table5.HELD_OUT
    assert set(table5.HELD_OUT) == {5, 10}
    assert set(table5.TABLE5_JM["us"]) == {1, *table5.HELD_OUT}


# --------------------------------------------------------------------------
# 2. the grading scale
# --------------------------------------------------------------------------


def test_grade_bands_are_the_scale_they_cite(table5):
    """A/B/C/D/F at 2/20/40/60 percent, ordered, with F for anything else."""
    assert table5.GRADE_BANDS == (("A", 0.02), ("B", 0.20), ("C", 0.40), ("D", 0.60))
    letters = [letter for letter, _ in table5.GRADE_BANDS]
    ceilings = [ceiling for _, ceiling in table5.GRADE_BANDS]
    assert letters == sorted(letters), "bands must run best grade first"
    assert ceilings == sorted(ceilings), "band ceilings must increase"
    assert [table5.grade(x) for x in (0.0, 0.02, 0.021, 0.2, 0.5, 0.61)] == [
        "A",
        "A",
        "B",
        "B",
        "D",
        "F",
    ]
    assert table5.grade(float("nan")) == "F"
    assert table5.grade(float("inf")) == "F"


# --------------------------------------------------------------------------
# 3. the delay has to reach both halves of the pipeline, and the sealed row
# --------------------------------------------------------------------------


@pytest.fixture
def synthetic_base(tmp_path: Path) -> Path:
    """A miniature sealed run: one market, two penalties, three delay rows.

    The three rows carry DIFFERENT start dates, which the real sealed run does
    not: every real window is identical, so a scorer that reads the delay-1 row
    at every delay cannot be caught on the real data.
    """
    dates = pd.bdate_range("2005-01-03", periods=3400)
    rng = np.random.default_rng(11)
    market = tmp_path / "xx"
    market.mkdir()
    pd.DataFrame(
        {
            "date": dates,
            "equity_simple": rng.normal(0.0004, 0.011, len(dates)),
            "cash_return": 0.00006,
        }
    ).to_csv(market / "features.csv", index=False)
    states = pd.DataFrame(
        {
            "0.0": (np.arange(len(dates)) // 23 % 2).astype(float),
            "70.0": (np.arange(len(dates)) // 260 % 2).astype(float),
        },
        index=dates,
    )
    states.index.name = "date"
    states.to_csv(market / "jm-states.csv")
    pd.DataFrame(
        [
            {
                "model": "fixed_jm",
                "delay": delay,
                "start": start,
                "end": "2018-01-01",
                "market": "xx",
            }
            for delay, start in ((1, "2014-01-02"), (5, "2015-01-02"), (10, "2016-01-04"))
        ]
    ).to_csv(tmp_path / "metrics.csv", index=False)
    return tmp_path


class _Reached(Exception):
    """Raised by a recorder so the test stops at the call it is inspecting."""


@pytest.mark.parametrize("delay", [1, 5, 10])
def test_delay_reaches_the_cross_validation(scorer, monkeypatch, delay):
    """Table 5's caption applies the delay in the CV, not only in execution."""
    seen: list[int] = []

    def recorder(*args, **kwargs):
        seen.append(kwargs["delay_trading_days"])
        raise _Reached

    monkeypatch.setattr(scorer, "select_monthly_candidate", recorder)
    with pytest.raises(_Reached):
        scorer.score("us", [0.0, 21.544346900318832, 70.0], delay=delay)
    assert seen == [delay]


@pytest.mark.parametrize("delay", [1, 5, 10])
def test_delay_reaches_the_execution(scorer, monkeypatch, synthetic_base, delay):
    """And the same delay reaches apply_signal, on the same call."""
    seen: list[int] = []

    def recorder(*args, **kwargs):
        seen.append(kwargs["delay_trading_days"])
        raise _Reached

    monkeypatch.setattr(scorer, "BASE", synthetic_base)
    monkeypatch.setattr(scorer, "apply_signal", recorder)
    with pytest.raises(_Reached):
        scorer.score("xx", [0.0, 70.0], delay=delay)
    # select_monthly_candidate calls the backtest module's own apply_signal, so
    # the only call this recorder can intercept is the execution one.
    assert seen == [delay]


def test_the_sealed_window_row_is_keyed_by_the_delay(scorer, monkeypatch, synthetic_base):
    """Pin the lookup that every real sealed window makes untestable."""
    monkeypatch.setattr(scorer, "BASE", synthetic_base)
    observed = {
        delay: scorer.score("xx", [0.0, 70.0], delay=delay)["observations"]
        for delay in (1, 5, 10)
    }
    assert observed[1] > observed[5] > observed[10], (
        "the three sealed rows start a year apart, so the observation counts "
        f"must fall with the delay; got {observed}"
    )


def test_an_unpublished_delay_is_refused(scorer):
    for delay in (0, 2, 3, 7, 11):
        with pytest.raises(SystemExit, match="not published"):
            scorer.score("us", [0.0, 70.0], delay=delay)


# --------------------------------------------------------------------------
# 4. what was run is what was frozen
# --------------------------------------------------------------------------


def _spec() -> dict:
    return tomllib.loads(SPEC.read_text(encoding="utf-8"))


def test_frozen_spec_matches_its_registry_hash():
    digest = hashlib.sha256(SPEC.read_bytes()).hexdigest()
    rows = [
        json.loads(line)
        for line in REGISTRY.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("experiment_id") == "heldout-delay-001"
    ]
    assert rows, "heldout-delay-001 has no registry row"
    hashed = [row for row in rows if row.get("spec_sha256")]
    assert hashed, "no registry row carries the spec hash"
    for row in hashed:
        assert row["spec_sha256"] == digest, (
            f"spec on disk hashes to {digest}, registry row "
            f"({row.get('status')}) froze {row['spec_sha256']}"
        )


def test_the_driver_runs_the_arms_the_spec_declares():
    """The driver must not carry its own copy of a grid, an arm or a delay."""
    driver = _module("run_heldout_delay")
    spec = _spec()
    assert set(spec["arms"]) == {"A_sealed_default", "B_minimax_search"}
    assert spec["protocol"]["held_out_delays"] == [5, 10]
    assert spec["protocol"]["in_sample_delay"] == 1
    assert tuple(spec["protocol"]["cells"]) == driver.CELLS
    source = (ROOT / "scripts/run_heldout_delay.py").read_text(encoding="utf-8")
    for arm in spec["arms"].values():
        for market in spec["sources"]["markets"]:
            for value in arm[market]:
                assert f"{value}" not in source, (
                    f"the driver hardcodes the penalty {value}; grids must come "
                    "from the frozen spec only"
                )


def test_arm_a_is_the_sealed_configs_own_grid():
    """Arm A must be the v10 contract's grid, not a hand-typed copy of it."""
    sys.path.insert(0, str(ROOT / "src"))
    from adaptive_jump.config import load_config

    config = load_config(ROOT / "configs/baselines/legacy/research-calibrated-v10.toml")
    arm = _spec()["arms"]["A_sealed_default"]
    for market in ("us", "de", "jp"):
        assert tuple(arm[market]) == tuple(config.jm_protocol_for(market).lambda_grid)
