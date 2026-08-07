"""Pure regime-path comparison metrics, pinned on hand-computed toy paths.

Every expected value below was derived by hand on 20-business-day paths
before the module existed. Conventions under test: state 1.0 = bear;
switch direction +1 = entering bear; lag = our position minus their
position on the joint calendar, so positive lag means we switch later.
"""

import numpy as np
import pandas as pd
import pytest

from adaptive_jump.regime_comparison import (
    RegimeComparisonError,
    concordance,
    covering,
    disagreement_decomposition,
    era_slices,
    match_switches,
    switch_f1,
    switches,
)

DAYS = pd.bdate_range("2020-01-01", periods=20)


def path(bear_positions: list[int]) -> pd.Series:
    values = np.zeros(len(DAYS))
    values[bear_positions] = 1.0
    return pd.Series(values, index=DAYS)


THEIRS = path([5, 6, 7, 8, 9])
OURS_LAG2 = path([7, 8, 9, 10, 11])
OURS_ALL_BULL = path([])
OURS_EXTRA = path([5, 6, 7, 8, 9, 15, 16])


def test_concordance_counts_matching_days() -> None:
    assert concordance(THEIRS, THEIRS) == 1.0
    assert concordance(THEIRS, OURS_LAG2) == pytest.approx(16 / 20)
    assert concordance(THEIRS, OURS_ALL_BULL) == pytest.approx(15 / 20)


def test_concordance_inner_joins_and_drops_nan() -> None:
    shorter = OURS_LAG2.drop(DAYS[[0, 19]])
    with_nan = THEIRS.copy()
    with_nan.iloc[1] = np.nan
    # 18 shared dates, one dropped for NaN -> 17 pairs, 13 of them equal
    assert concordance(with_nan, shorter) == pytest.approx(13 / 17)


def test_switches_dates_and_directions() -> None:
    events = switches(THEIRS)
    assert list(events["date"]) == [DAYS[5], DAYS[10]]
    assert list(events["direction"]) == [1, -1]


def test_switches_first_valid_day_is_never_a_switch() -> None:
    starts_in_bear = path([0, 1, 2])
    events = switches(starts_in_bear)
    assert list(events["date"]) == [DAYS[3]]
    assert list(events["direction"]) == [-1]
    assert switches(OURS_ALL_BULL).empty


def test_switches_rejects_non_binary_values() -> None:
    bad = THEIRS.copy()
    bad.iloc[3] = 0.5
    with pytest.raises(RegimeComparisonError):
        switches(bad)


def test_match_switches_signed_lag_positive_when_we_are_late() -> None:
    matches, unmatched_theirs, unmatched_ours = match_switches(
        THEIRS, OURS_LAG2, margin=10
    )
    assert list(matches["lag"]) == [2, 2]
    assert list(matches["direction"]) == [1, -1]
    assert unmatched_theirs.empty and unmatched_ours.empty


def test_match_switches_is_one_to_one_greedy_by_smallest_lag() -> None:
    matches, unmatched_theirs, unmatched_ours = match_switches(
        THEIRS, OURS_EXTRA, margin=10
    )
    # the exact-lag pairs win; the spurious episode at 15..16 stays unmatched
    # even though |15 - 5| = 10 lies within the margin, because 1-1 matching
    # already consumed both of their switches at lag 0
    assert list(matches["lag"]) == [0, 0]
    assert unmatched_theirs.empty
    assert list(unmatched_ours["date"]) == [DAYS[15], DAYS[17]]


def test_match_switches_respects_margin() -> None:
    matches, unmatched_theirs, unmatched_ours = match_switches(
        THEIRS, OURS_LAG2, margin=1
    )
    assert matches.empty
    assert len(unmatched_theirs) == 2 and len(unmatched_ours) == 2


def test_switch_f1_perfect_partial_and_degenerate() -> None:
    assert switch_f1(THEIRS, OURS_LAG2, margin=10)["f1"] == 1.0
    partial = switch_f1(THEIRS, OURS_EXTRA, margin=10)
    assert partial["precision"] == pytest.approx(0.5)
    assert partial["recall"] == 1.0
    assert partial["f1"] == pytest.approx(2 / 3)
    none_found = switch_f1(THEIRS, OURS_ALL_BULL, margin=10)
    assert none_found == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    both_empty = switch_f1(OURS_ALL_BULL, OURS_ALL_BULL, margin=10)
    assert both_empty == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_one_day_mis_shift_costs_one_day_per_switch_but_keeps_f1() -> None:
    shifted = path([6, 7, 8, 9, 10])
    assert concordance(THEIRS, shifted) == pytest.approx(18 / 20)
    matches, _, _ = match_switches(THEIRS, shifted, margin=10)
    assert list(matches["lag"]) == [1, 1]
    assert switch_f1(THEIRS, shifted, margin=10)["f1"] == 1.0


def test_covering_hand_computed_value() -> None:
    # gt segments 5/5/10 days; best Jaccards 5/7, 3/7, 8/10
    assert covering(THEIRS, OURS_LAG2) == pytest.approx(96 / 140)
    assert covering(THEIRS, THEIRS) == 1.0
    assert covering(OURS_ALL_BULL, OURS_ALL_BULL) == 1.0


def test_decomposition_classes_and_identity() -> None:
    timing = disagreement_decomposition(THEIRS, OURS_LAG2)
    assert set(timing["kind"]) == {"timing"}
    assert len(timing) == 4

    missing = disagreement_decomposition(THEIRS, OURS_ALL_BULL)
    assert set(missing["kind"]) == {"missing"}
    assert list(missing["date"]) == list(DAYS[5:10])

    extra = disagreement_decomposition(THEIRS, OURS_EXTRA)
    assert set(extra["kind"]) == {"extra"}
    assert list(extra["date"]) == [DAYS[15], DAYS[16]]

    for ours in (OURS_LAG2, OURS_ALL_BULL, OURS_EXTRA):
        n_disagree = len(disagreement_decomposition(THEIRS, ours))
        assert n_disagree == round((1.0 - concordance(THEIRS, ours)) * 20)


def test_era_slices_partitions_exactly() -> None:
    eras = [("early", DAYS[0], DAYS[9]), ("late", DAYS[10], DAYS[19])]
    slices = dict(era_slices(DAYS, eras))
    assert len(slices["early"]) == 10 and len(slices["late"]) == 10

    overlapping = [("early", DAYS[0], DAYS[10]), ("late", DAYS[10], DAYS[19])]
    with pytest.raises(RegimeComparisonError, match="overlap"):
        era_slices(DAYS, overlapping)

    gappy = [("early", DAYS[0], DAYS[8]), ("late", DAYS[10], DAYS[19])]
    with pytest.raises(RegimeComparisonError, match="uncovered"):
        era_slices(DAYS, gappy)
