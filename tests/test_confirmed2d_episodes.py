"""Focused regressions for the episode-anatomy helpers.

`scripts/probe_confirmed2d_episodes.py` reduces a paired daily return
difference to divergence EPISODES, and the conclusions drawn from it (every
episode is one day; one US day carries more than the whole net; the frozen
partition is not cost-complete and its residual is exactly the settlement days)
rest entirely on four small pure functions. This file pins them.

The load-bearing one is `runs_of_true` together with the settlement convention:
the frozen episode partition covers only the days on which the two arms HOLD
different positions, while the day after an episode still differs in
transaction cost. `test_episodes_plus_settlement_days_are_exhaustive` states
that identity arithmetically on a synthetic path, which is the property the
probe checks numerically on the real one.

Nothing here reads an artifact or fits a model.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def module():
    """Import the probe without executing ``main()``."""
    path = ROOT / "scripts/probe_confirmed2d_episodes.py"
    spec = importlib.util.spec_from_file_location("confirmed2d_under_test", path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


# ---------------------------------------------------------------------------
# runs_of_true: the episode partition itself
# ---------------------------------------------------------------------------


def test_runs_of_true_returns_inclusive_maximal_runs(module):
    flag = np.array([False, True, True, False, True, False, False, True])
    assert module.runs_of_true(flag) == [(1, 2), (4, 4), (7, 7)]


def test_runs_of_true_handles_both_edges(module):
    assert module.runs_of_true(np.array([True, True, False, True])) == [(0, 1), (3, 3)]
    assert module.runs_of_true(np.array([True])) == [(0, 0)]


def test_runs_of_true_is_empty_when_nothing_differs(module):
    assert module.runs_of_true(np.zeros(9, dtype=bool)) == []


def test_runs_of_true_covers_every_true_day_exactly_once(module):
    rng = np.random.default_rng(4)
    flag = rng.random(200) < 0.15
    covered = [
        index
        for first, last in module.runs_of_true(flag)
        for index in range(first, last + 1)
    ]
    assert covered == sorted(covered)
    assert len(covered) == len(set(covered))
    assert set(covered) == set(np.flatnonzero(flag).tolist())


def test_episodes_plus_settlement_days_are_exhaustive(module):
    """The frozen partition misses one settlement day per episode -- by exactly that.

    Days 2-3 and 7 hold different positions; on days 4 and 8 the arms hold the
    same position but arrive from different ones, so their cost still differs.
    Summing over episodes alone must therefore leave a residual equal to the
    settlement days, which is precisely the arithmetic the probe reports.
    """
    differ = np.zeros(10, dtype=bool)
    differ[2:4] = True
    differ[7] = True
    diff = np.zeros(10)
    diff[2], diff[3], diff[7] = 0.03, -0.01, 0.02
    diff[4], diff[8] = -0.001, -0.001  # settlement-day cost differences

    spans = module.runs_of_true(differ)
    assert spans == [(2, 3), (7, 7)]
    episode_sum = sum(float(diff[first : last + 1].sum()) for first, last in spans)
    settlement_sum = sum(float(diff[last + 1]) for _, last in spans)
    closed_sum = sum(
        float(diff[first : last + 1].sum()) + float(diff[last + 1])
        for first, last in spans
    )

    total = float(diff.sum())
    assert episode_sum != pytest.approx(total)
    assert total - episode_sum == pytest.approx(settlement_sum, abs=1e-15)
    assert closed_sum == pytest.approx(total, abs=1e-15)


# ---------------------------------------------------------------------------
# merge_padded: the secondary robustness view
# ---------------------------------------------------------------------------


def test_merge_padded_clips_to_the_window(module):
    assert module.merge_padded([(0, 0), (9, 9)], 5, 10) == [(0, 9)]


def test_merge_padded_merges_overlapping_and_adjacent_spans(module):
    assert module.merge_padded([(10, 10), (13, 13)], 1, 100) == [(9, 14)]


def test_merge_padded_keeps_distant_spans_separate(module):
    assert module.merge_padded([(10, 10), (40, 41)], 2, 100) == [(8, 12), (38, 43)]


def test_merge_padded_output_is_disjoint_and_ordered(module):
    rng = np.random.default_rng(5)
    flag = rng.random(300) < 0.05
    merged = module.merge_padded(module.runs_of_true(flag), 5, 300)
    assert all(low <= high for low, high in merged)
    assert all(
        merged[index][1] + 1 < merged[index + 1][0] for index in range(len(merged) - 1)
    )


# ---------------------------------------------------------------------------
# concentration: the headline "one day is 177% of the net" statistic
# ---------------------------------------------------------------------------


def test_concentration_ranks_by_absolute_size_but_sums_signed_values(module):
    values = np.array([0.05, -0.04, 0.01, -0.002])
    shares = module.concentration(values, float(values.sum()))
    assert shares["top1_sum"] == pytest.approx(0.05)
    assert shares["top3_sum"] == pytest.approx(0.05 - 0.04 + 0.01)
    assert shares["top1_share"] == pytest.approx(0.05 / values.sum())


def test_concentration_can_exceed_one_when_the_net_is_a_small_residual(module):
    """The US reading: a top-1 share above 100% is arithmetic, not a bug."""
    values = np.array([0.047476, -0.030000, 0.009306])
    total = float(values.sum())
    shares = module.concentration(values, total)
    assert 0.0 < total < values.max()
    assert shares["top1_share"] > 1.0


def test_concentration_reports_nan_rather_than_dividing_by_a_zero_net(module):
    shares = module.concentration(np.array([0.01, -0.01]), 0.0)
    assert math.isnan(shares["top1_share"])
    assert shares["top1_sum"] == pytest.approx(0.01)


def test_concentration_of_fewer_entries_than_k_uses_all_of_them(module):
    values = np.array([0.02, -0.01])
    shares = module.concentration(values, float(values.sum()))
    assert shares["top5_sum"] == pytest.approx(values.sum())
    assert shares["top5_share"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# describe and crisis_label
# ---------------------------------------------------------------------------


def test_describe_uses_the_sample_standard_deviation(module):
    values = np.array([1.0, 2.0, 4.0])
    stats = module.describe(values)
    assert stats["mean"] == pytest.approx(7.0 / 3.0)
    assert stats["median"] == pytest.approx(2.0)
    assert stats["min"] == pytest.approx(1.0)
    assert stats["max"] == pytest.approx(4.0)
    assert stats["std"] == pytest.approx(float(np.std(values, ddof=1)))


def test_describe_is_all_nan_on_an_empty_input(module):
    stats = module.describe(np.array([]))
    assert all(math.isnan(value) for value in stats.values())


def test_describe_reports_nan_std_for_a_single_episode(module):
    stats = module.describe(np.array([0.03]))
    assert stats["mean"] == pytest.approx(0.03)
    assert math.isnan(stats["std"])


def test_crisis_label_is_inclusive_at_both_window_edges(module):
    assert module.crisis_label(pd.Timestamp("2007-10-01")) == "gfc"
    assert module.crisis_label(pd.Timestamp("2009-03-31")) == "gfc"
    assert module.crisis_label(pd.Timestamp("2009-04-01")) == "normal"
    assert module.crisis_label(pd.Timestamp("2020-03-20")) == "covid"
    assert module.crisis_label(pd.Timestamp("1995-06-15")) == "normal"


def test_crisis_windows_do_not_overlap(module):
    windows = [
        (pd.Timestamp(low), pd.Timestamp(high))
        for _, low, high in module.CRISIS_WINDOWS
    ]
    ordered = sorted(windows)
    assert ordered == windows
    assert all(
        ordered[index][1] < ordered[index + 1][0] for index in range(len(ordered) - 1)
    )
