"""Synthetic production-path tests for the balanced mechanism analysis."""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from adaptive_jump.balanced_analysis import (
    ANCHOR_COLUMNS,
    MATCHED_CATEGORIES,
    classify,
    matched_response,
    summarize,
)
from adaptive_jump.balanced_events import EVENT_COLUMNS, extract_events
from adaptive_jump.balanced_model import (
    load_balanced_spec,
)
from adaptive_jump.config import load_config
from adaptive_jump.lagged_model import LockedStateEvidence
from adaptive_jump.models import FEATURE_COLUMNS
from adaptive_jump.separation_analysis import MarketInputs

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def spec():
    config = load_config(ROOT / "research.toml")
    return load_balanced_spec(
        ROOT / "research/balanced-lagged-mechanism-001.toml", config
    )


def _matched_inputs(spec):
    dates = pd.date_range("2020-01-01", periods=340, name="date")
    states = pd.DataFrame(0, index=dates, columns=[5.0], dtype=int)
    fixed = states.copy()
    categories = list(MATCHED_CATEGORIES)
    positions = [20 + 45 * index for index in range(len(categories))]
    fixed_h = [5.0, 5.0, math.nan, math.nan, math.nan, 5.0, math.nan]
    confirmed = [False, True, False, False, False, True, False]
    lagged_whipsaw = [True, False, False, True, True, False, False]
    lagged_unconfirmed = [False, False, True, False, False, False, True]

    for category, position in zip(categories, positions, strict=True):
        if category.startswith("already_destination"):
            states.iloc[position - 1 : position + spec.matched_followup, 0] = 1
        if category == "already_destination_whipsaw":
            states.iloc[position + 2, 0] = 0
        elif category == "enters_then_whipsaw":
            states.iloc[position + 2 : position + 2 + spec.matched_followup + 1, 0] = 1
            states.iloc[position + 4, 0] = 0
        elif category == "enters_persistent_confirmed":
            states.iloc[position + 5 : position + 5 + spec.matched_followup + 1, 0] = 1
        elif category == "enters_persistent_unconfirmed":
            states.iloc[position + 2 : position + 2 + spec.matched_followup + 1, 0] = 1

    rows = []
    for index, (_category, position) in enumerate(
        zip(categories, positions, strict=True)
    ):
        if math.isfinite(fixed_h[index]):
            fixed.iloc[position + int(fixed_h[index]), 0] = 1
        rows.append(
            {
                "market": "us",
                "rule": "lagged",
                "beta": spec.decision_beta,
                "beta_label": "log4",
                "lambda0": 5.0,
                "signal_date": dates[position],
                "fit_date": dates[0],
                "source_state": 0,
                "destination_state": 1,
                "whipsaw_20": lagged_whipsaw[index],
                "confirmed_early": confirmed[index],
                "unconfirmed_persistent_20": lagged_unconfirmed[index],
                "fixed_confirmation_h": fixed_h[index],
            }
        )
    refits = pd.DataFrame({"fit_date": [dates[0]], "lambda0": [5.0]})
    return pd.DataFrame.from_records(rows), states, fixed, refits, categories


def test_all_seven_matched_categories_partition_every_anchor(spec):
    parent, states, fixed, refits, expected = _matched_inputs(spec)

    result, audit = matched_response(parent, states, fixed, refits, spec)

    assert result.columns.tolist() == ANCHOR_COLUMNS
    assert result["matched_category"].tolist() == expected
    assert len(result) == len(parent) == 7
    assert result["matched_category"].value_counts().to_dict() == {
        category: 1 for category in MATCHED_CATEGORIES
    }
    assert result["matched_whipsaw_20"].tolist() == [
        True,
        False,
        False,
        False,
        True,
        False,
        False,
    ]
    assert not result.loc[
        result["matched_category"] == "suppressed_no_entry",
        "matched_whipsaw_20",
    ].item()
    assert result["matched_follow_end_h"].tolist()[:3] == [19, 19, 19]
    suppressed = result.loc[result["matched_category"] == "suppressed_no_entry"].iloc[0]
    assert math.isnan(suppressed["matched_follow_end_h"])
    assert math.isnan(suppressed["matched_fixed_confirmation_h"])
    assert audit == {
        "original_lagged_admitted_events": 7,
        "matched_anchor_censored": 0,
        "eligible_matched_anchors": 7,
    }


def test_matched_latency_retention_is_strictly_early_and_no_return(spec):
    parent, states, fixed, refits, _ = _matched_inputs(spec)

    result, _ = matched_response(parent, states, fixed, refits, spec)
    result = result.set_index("matched_category")

    assert result.loc[
        "already_destination_persistent_confirmed",
        "matched_retained_confirmed_early",
    ]
    assert not result.loc[
        "enters_persistent_confirmed",
        "matched_retained_confirmed_early",
    ]
    assert (
        result.loc["enters_persistent_confirmed", "first_destination_h"]
        == result.loc["enters_persistent_confirmed", "fixed_confirmation_h"]
    )


def test_empty_own_event_path_has_stable_schema(spec):
    mini = replace(
        spec,
        markets=("us",),
        rules=("lagged", "balanced"),
        lambdas=(0.0, 5.0),
        event_lambdas=(5.0,),
        fit_window=3,
        horizon=2,
        evaluation_starts={"us": date(2020, 1, 1)},
    )
    dates = pd.date_range("2020-01-01", periods=6, name="date")
    state = pd.DataFrame(0, index=dates, columns=[5.0], dtype=float)
    features = pd.DataFrame(0.0, index=dates, columns=FEATURE_COLUMNS)
    inputs = MarketInputs(
        market="us",
        features=features,
        model_dates=dates,
        candidates={0.0: state.copy()},
        refits=pd.DataFrame(columns=["fit_date", "lambda0"]),
    )
    empty = pd.DataFrame(index=dates)
    evidence = {
        rule: LockedStateEvidence(
            states={mini.decision_beta: state.copy()},
            loss0=empty.copy(),
            loss1=empty.copy(),
            q_train=empty.copy(),
            c01={},
            c10={},
            refits=inputs.refits.copy(),
        )
        for rule in mini.rules
    }

    events, audit = extract_events(inputs, evidence, mini)

    assert events.empty
    assert events.columns.tolist() == EVENT_COLUMNS
    assert set(audit) == set(mini.rules)
    assert all(values["admitted_events"] == 0 for values in audit.values())
    for values in audit.values():
        assert values["minimum_terminal_state_margin"] is None
        assert values["minimum_terminal_predecessor_margin"] is None
        assert values["minimum_ablated_state_margin"] is None
        assert values["nonfinite_terminal_state_margin_count"] == 0
        assert values["nonfinite_terminal_predecessor_margin_count"] == 0
        assert values["nonfinite_ablated_state_margin_count"] == 0


def _decision_summary(spec) -> pd.DataFrame:
    rows = []
    for market in spec.markets:
        for rule in spec.rules:
            balanced = rule == "balanced"
            rows.append(
                {
                    "market": market,
                    "rule": rule,
                    "beta": spec.decision_beta,
                    "beta_label": "log4",
                    "event_count": 4,
                    "whipsaw_count": 1 if balanced else 2,
                    "persistent_count": 3 if balanced else 2,
                    "confirmed_early_count": 1,
                    "unconfirmed_persistent_count": 1 if balanced else 2,
                    "switch_count": 8 if balanced else 10,
                    "state_differences_vs_fixed": 4,
                    "state_differences_vs_lagged": 3 if balanced else 0,
                }
            )
    return pd.DataFrame.from_records(rows)


def _decision_anchors(spec) -> pd.DataFrame:
    rows = []
    for market_index, market in enumerate(spec.markets):
        base = pd.Timestamp("2020-01-01") + pd.Timedelta(days=100 * market_index)
        rows.extend(
            [
                {
                    "market": market,
                    "beta": spec.decision_beta,
                    "beta_label": "log4",
                    "lambda0": 5.0,
                    "signal_date": base,
                    "source_state": 0,
                    "destination_state": 1,
                    "lagged_whipsaw_20": True,
                    "lagged_confirmed_early": False,
                    "lagged_unconfirmed_persistent_20": False,
                    "fixed_confirmation_h": math.nan,
                    "first_destination_h": math.nan,
                    "matched_follow_end_h": math.nan,
                    "matched_fixed_confirmation_h": math.nan,
                    "matched_category": "suppressed_no_entry",
                    "matched_whipsaw_20": False,
                    "matched_persistent_20": False,
                    "matched_unconfirmed_persistent_20": False,
                    "matched_retained_confirmed_early": False,
                },
                {
                    "market": market,
                    "beta": spec.decision_beta,
                    "beta_label": "log4",
                    "lambda0": 15.0,
                    "signal_date": base + pd.Timedelta(days=25),
                    "source_state": 0,
                    "destination_state": 1,
                    "lagged_whipsaw_20": False,
                    "lagged_confirmed_early": True,
                    "lagged_unconfirmed_persistent_20": False,
                    "fixed_confirmation_h": 5.0,
                    "first_destination_h": 2.0,
                    "matched_follow_end_h": 22.0,
                    "matched_fixed_confirmation_h": 5.0,
                    "matched_category": "enters_persistent_confirmed",
                    "matched_whipsaw_20": False,
                    "matched_persistent_20": True,
                    "matched_unconfirmed_persistent_20": False,
                    "matched_retained_confirmed_early": True,
                },
                {
                    "market": market,
                    "beta": spec.decision_beta,
                    "beta_label": "log4",
                    "lambda0": 35.0,
                    "signal_date": base + pd.Timedelta(days=50),
                    "source_state": 0,
                    "destination_state": 1,
                    "lagged_whipsaw_20": False,
                    "lagged_confirmed_early": False,
                    "lagged_unconfirmed_persistent_20": True,
                    "fixed_confirmation_h": math.nan,
                    "first_destination_h": math.nan,
                    "matched_follow_end_h": math.nan,
                    "matched_fixed_confirmation_h": math.nan,
                    "matched_category": "suppressed_no_entry",
                    "matched_whipsaw_20": False,
                    "matched_persistent_20": False,
                    "matched_unconfirmed_persistent_20": False,
                    "matched_retained_confirmed_early": False,
                },
            ]
        )
    return pd.DataFrame.from_records(rows, columns=ANCHOR_COLUMNS)


def test_decision_supported_and_switch_edge_fails(spec):
    summary = _decision_summary(spec)
    anchors = _decision_anchors(spec)

    supported = classify(summary, anchors, spec, mechanical_passed=True)
    assert supported["result"] == "supported"
    assert all(supported["conditions"].values())
    assert supported["pooled"]["matched_category_counts"] == {
        category: int((anchors["matched_category"] == category).sum())
        for category in MATCHED_CATEGORIES
    }

    failed = summary.copy()
    failed.loc[
        (failed["market"] == "us") & (failed["rule"] == "balanced"),
        "switch_count",
    ] = 11
    rejected = classify(failed, anchors, spec, mechanical_passed=True)
    assert rejected["result"] == "not_supported"
    assert rejected["conditions"]["market_switch_guard"] is False


def test_both_own_and_matched_lock_in_guards_fail_closed(spec):
    summary = _decision_summary(spec)
    anchors = _decision_anchors(spec)

    own = summary.copy()
    own.loc[own["rule"] == "balanced", "unconfirmed_persistent_count"] = 3
    own_result = classify(own, anchors, spec, mechanical_passed=True)
    assert own_result["conditions"]["own_lock_in_guard"] is False
    assert own_result["result"] == "not_supported"

    matched = anchors.copy()
    matched["matched_unconfirmed_persistent_20"] = True
    matched["matched_category"] = "enters_persistent_unconfirmed"
    matched_result = classify(summary, matched, spec, mechanical_passed=True)
    assert matched_result["conditions"]["matched_lock_in_guard"] is False
    assert matched_result["result"] == "not_supported"


def _anchor_row(spec, market, lambda0, signal_date, fit_date, fixed_h=math.nan):
    return {
        "market": market,
        "rule": "lagged",
        "beta": spec.decision_beta,
        "beta_label": "log4",
        "lambda0": lambda0,
        "signal_date": signal_date,
        "fit_date": fit_date,
        "source_state": 0,
        "destination_state": 1,
        "whipsaw_20": False,
        "confirmed_early": math.isfinite(fixed_h),
        "unconfirmed_persistent_20": not math.isfinite(fixed_h),
        "fixed_confirmation_h": fixed_h,
    }


def test_late_entries_and_baseline_have_exact_equal_follow_exposure(spec):
    dates = pd.date_range("2020-01-01", periods=190, name="date")
    lambdas = [5.0, 15.0, 35.0]
    balanced = pd.DataFrame(0, index=dates, columns=lambdas, dtype=int)
    fixed = balanced.copy()
    positions = [20, 75, 130]

    # Baseline destination: observe h=0..19; a source at h=20 is outside.
    balanced.loc[dates[positions[0] - 1] : dates[positions[0] + 19], 5.0] = 1
    balanced.loc[dates[positions[0] + 20], 5.0] = 0
    fixed.loc[dates[positions[0] + 20], 5.0] = 1

    # Entry h=19: observe h=20..39; a source at h=40 is outside.
    balanced.loc[dates[positions[1] + 19] : dates[positions[1] + 39], 15.0] = 1
    balanced.loc[dates[positions[1] + 40], 15.0] = 0
    fixed.loc[dates[positions[1] + 20], 15.0] = 1

    # Entry h=20: observe h=21..40; source at h=40 is a whipsaw.
    balanced.loc[dates[positions[2] + 20] : dates[positions[2] + 40], 35.0] = 1
    balanced.loc[dates[positions[2] + 40], 35.0] = 0
    fixed.loc[dates[positions[2] + 20], 35.0] = 1

    events = pd.DataFrame.from_records(
        [
            _anchor_row(spec, "us", 5.0, dates[positions[0]], dates[0], 20.0),
            _anchor_row(spec, "us", 15.0, dates[positions[1]], dates[0], 20.0),
            _anchor_row(spec, "us", 35.0, dates[positions[2]], dates[0], 20.0),
        ]
    )
    refits = pd.DataFrame({"fit_date": [dates[0]] * 3, "lambda0": lambdas})

    result, audit = matched_response(events, balanced, fixed, refits, spec)

    assert result["first_destination_h"].tolist() == [-1, 19, 20]
    assert result["matched_follow_end_h"].tolist() == [19, 39, 40]
    assert result["matched_category"].tolist() == [
        "already_destination_persistent_unconfirmed",
        "enters_persistent_confirmed",
        "enters_then_whipsaw",
    ]
    assert math.isnan(result.loc[0, "matched_fixed_confirmation_h"])
    assert result.loc[1, "matched_fixed_confirmation_h"] == 20
    assert result["matched_retained_confirmed_early"].tolist() == [
        True,
        True,
        False,
    ]
    assert audit["eligible_matched_anchors"] == 3


def test_matched_eligibility_checks_t40_before_same_refit(spec):
    dates = pd.date_range("2020-01-01", periods=150, name="date")
    lambdas = [5.0, 15.0, 35.0]
    balanced = pd.DataFrame(0, index=dates, columns=lambdas, dtype=int)
    fixed = balanced.copy()
    events = pd.DataFrame.from_records(
        [
            # Invalid event fit would fail if refit were inspected before t+40.
            _anchor_row(spec, "us", 5.0, dates[120], dates[10]),
            _anchor_row(spec, "us", 15.0, dates[20], dates[0]),
            _anchor_row(spec, "us", 35.0, dates[70], dates[0]),
        ]
    )
    refits = pd.DataFrame.from_records(
        [
            {"fit_date": dates[0], "lambda0": 5.0},
            {"fit_date": dates[0], "lambda0": 15.0},
            {"fit_date": dates[50], "lambda0": 15.0},
            {"fit_date": dates[0], "lambda0": 35.0},
        ]
    )

    result, audit = matched_response(events, balanced, fixed, refits, spec)

    assert len(result) == 1
    assert result.iloc[0]["lambda0"] == 35.0
    assert result.iloc[0]["matched_category"] == "suppressed_no_entry"
    assert audit == {
        "original_lagged_admitted_events": 3,
        "matched_anchor_censored": 2,
        "eligible_matched_anchors": 1,
    }


def test_summary_reports_minimum_admitted_margins(spec):
    mini = replace(
        spec, markets=("us",), event_lambdas=(5.0,), rules=("lagged", "balanced")
    )
    behavior = pd.DataFrame.from_records(
        [
            {
                "market": "us",
                "rule": rule,
                "lambda0": 5.0,
                "switch_count": 2,
                "state_differences_vs_fixed": 1,
                "state_differences_vs_lagged": int(rule == "balanced"),
            }
            for rule in mini.rules
        ]
    )
    rows = []
    for rule, margins in (
        ("lagged", (3.0, 2.0, 4.0)),
        ("balanced", (1.0, 5.0, 2.0)),
    ):
        row = {column: math.nan for column in EVENT_COLUMNS}
        row.update(
            {
                "market": "us",
                "rule": rule,
                "whipsaw_20": False,
                "persistent_20": True,
                "confirmed_early": True,
                "unconfirmed_persistent_20": False,
                "terminal_state_margin": margins[0],
                "terminal_predecessor_margin": margins[1],
                "ablated_state_margin": margins[2],
            }
        )
        rows.append(row)
    events = pd.DataFrame.from_records(rows, columns=EVENT_COLUMNS)

    summary = summarize(events, behavior, mini).set_index("rule")

    assert summary.loc["lagged", "minimum_terminal_state_margin"] == 3.0
    assert summary.loc["lagged", "minimum_terminal_predecessor_margin"] == 2.0
    assert summary.loc["lagged", "minimum_ablated_state_margin"] == 4.0
    assert summary.loc["balanced", "minimum_terminal_state_margin"] == 1.0
    assert summary.loc["balanced", "minimum_terminal_predecessor_margin"] == 5.0
    assert summary.loc["balanced", "minimum_ablated_state_margin"] == 2.0
