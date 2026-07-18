"""Performance-free paths, events, matched anchors, and decision logic."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from adaptive_jump.balanced_events import EVENT_COLUMNS, extract_events
from adaptive_jump.balanced_model import (
    BalancedSpec,
    BalancedStudyError,
    beta_label,
)
from adaptive_jump.lagged_model import LockedStateEvidence
from adaptive_jump.separation_analysis import MarketInputs, _refit_for_date

MATCHED_CATEGORIES = (
    "already_destination_whipsaw",
    "already_destination_persistent_confirmed",
    "already_destination_persistent_unconfirmed",
    "suppressed_no_entry",
    "enters_then_whipsaw",
    "enters_persistent_confirmed",
    "enters_persistent_unconfirmed",
)

ANCHOR_COLUMNS = [
    "market",
    "beta",
    "beta_label",
    "lambda0",
    "signal_date",
    "source_state",
    "destination_state",
    "lagged_whipsaw_20",
    "lagged_confirmed_early",
    "lagged_unconfirmed_persistent_20",
    "fixed_confirmation_h",
    "first_destination_h",
    "matched_follow_end_h",
    "matched_fixed_confirmation_h",
    "matched_category",
    "matched_whipsaw_20",
    "matched_persistent_20",
    "matched_unconfirmed_persistent_20",
    "matched_retained_confirmed_early",
]


@dataclass(frozen=True)
class MechanismAnalysis:
    market: str
    behavior: pd.DataFrame
    events: pd.DataFrame
    anchors: pd.DataFrame
    penalties: pd.DataFrame
    audit: dict[str, Any]


def path_behavior(
    inputs: MarketInputs,
    evidence: dict[str, LockedStateEvidence],
    spec: BalancedSpec,
) -> pd.DataFrame:
    fixed = inputs.candidates[0.0]
    parent = inputs.candidates[spec.decision_beta]
    start = pd.Timestamp(spec.evaluation_starts[inputs.market])
    rows: list[dict[str, Any]] = []
    for rule in spec.rules:
        states = evidence[rule].states[spec.decision_beta]
        for lambda0 in spec.event_lambdas:
            complete = states[lambda0].dropna().astype(int)
            first = int(complete.index.searchsorted(start, side="left"))
            path = complete.iloc[first:]
            if path.empty:
                raise BalancedStudyError(f"{inputs.market}/{rule}: empty path")
            fixed_path = fixed[lambda0].reindex(path.index).astype(int)
            parent_path = parent[lambda0].reindex(path.index).astype(int)
            switch_values = complete.iloc[max(0, first - 1) :].to_numpy(dtype=int)
            rows.append(
                {
                    "market": inputs.market,
                    "rule": rule,
                    "beta": spec.decision_beta,
                    "beta_label": beta_label(spec.decision_beta),
                    "lambda0": lambda0,
                    "start": path.index[0],
                    "end": path.index[-1],
                    "observations": len(path),
                    "switch_count": int(np.count_nonzero(np.diff(switch_values))),
                    "state_0_count": int((path == 0).sum()),
                    "state_1_count": int((path == 1).sum()),
                    "state_differences_vs_fixed": int(
                        (path.to_numpy() != fixed_path.to_numpy()).sum()
                    ),
                    "state_differences_vs_lagged": int(
                        (path.to_numpy() != parent_path.to_numpy()).sum()
                    ),
                }
            )
    return pd.DataFrame.from_records(rows)


def penalty_summary(
    market: str,
    evidence: dict[str, LockedStateEvidence],
    spec: BalancedSpec,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rule in spec.rules:
        item = evidence[rule]
        for lambda0 in spec.event_lambdas:
            c01 = item.c01[spec.decision_beta][lambda0]
            c10 = item.c10[spec.decision_beta][lambda0]
            valid = c01.notna() & c10.notna()
            left = c01.loc[valid].to_numpy(dtype=float)
            right = c10.loc[valid].to_numpy(dtype=float)
            pair_ratio = (left + right) / (2.0 * lambda0)
            rows.append(
                {
                    "market": market,
                    "rule": rule,
                    "beta": spec.decision_beta,
                    "beta_label": beta_label(spec.decision_beta),
                    "lambda0": lambda0,
                    "observations": len(left),
                    "minimum_cost_ratio": float(np.min(np.r_[left, right]) / lambda0),
                    "maximum_cost_ratio": float(np.max(np.r_[left, right]) / lambda0),
                    "median_pair_mean_ratio": float(np.median(pair_ratio)),
                    "maximum_pair_sum_abs_error": float(
                        np.max(np.abs(left + right - 2.0 * lambda0))
                    ),
                    "discount_cells": int((np.r_[left, right] < lambda0).sum()),
                    "surcharge_cells": int((np.r_[left, right] > lambda0).sum()),
                }
            )
    return pd.DataFrame.from_records(rows)


def matched_response(
    parent_events: pd.DataFrame,
    balanced_states: pd.DataFrame,
    fixed_states: pd.DataFrame,
    refits: pd.DataFrame,
    spec: BalancedSpec,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Classify equal-exposure balanced responses on eligible lagged anchors."""
    lagged_events = parent_events.loc[parent_events["rule"] == "lagged"]
    counters = {
        "original_lagged_admitted_events": len(lagged_events),
        "matched_anchor_censored": 0,
        "eligible_matched_anchors": 0,
    }
    refits_by_lambda = {
        float(lambda0): rows.sort_values("fit_date").reset_index(drop=True)
        for lambda0, rows in refits.groupby("lambda0")
        if float(lambda0) in spec.event_lambdas
    }
    records: list[dict[str, Any]] = []
    for anchor in lagged_events.itertuples():
        lambda0 = float(anchor.lambda0)
        if lambda0 not in fixed_states or lambda0 not in refits_by_lambda:
            raise BalancedStudyError("matched anchor lambda coverage changed")
        fixed_path = fixed_states[lambda0].dropna().astype(int)
        signal_date = pd.Timestamp(anchor.signal_date)
        try:
            position = int(fixed_path.index.get_loc(signal_date))
        except KeyError as exc:
            raise BalancedStudyError("matched anchor date is not mapped") from exc
        censor_position = position + spec.matched_anchor_censor
        if position < 1 or censor_position >= len(fixed_path):
            counters["matched_anchor_censored"] += 1
            continue
        rows = refits_by_lambda[lambda0]
        event_fit = pd.Timestamp(anchor.fit_date)
        if pd.Timestamp(_refit_for_date(rows, signal_date)["fit_date"]) != event_fit:
            raise BalancedStudyError("matched anchor event refit changed")
        censor_date = pd.Timestamp(fixed_path.index[censor_position])
        if pd.Timestamp(_refit_for_date(rows, censor_date)["fit_date"]) != event_fit:
            counters["matched_anchor_censored"] += 1
            continue
        counters["eligible_matched_anchors"] += 1

        path = balanced_states[lambda0].dropna().astype(int)
        if not path.index.equals(fixed_path.index):
            raise BalancedStudyError("matched candidate indexes changed")
        source, destination = int(anchor.source_state), int(anchor.destination_state)
        if int(path.iloc[position - 1]) == destination:
            first_h: int | float = -1
        else:
            search = path.iloc[
                position : position + spec.matched_entry_search + 1
            ].to_numpy(dtype=int)
            entries = np.flatnonzero(search == destination)
            first_h = int(entries[0]) if len(entries) else math.nan

        if not math.isfinite(first_h):
            follow_end_h: int | float = math.nan
            matched_fixed_h: int | float = math.nan
            category = "suppressed_no_entry"
            whipsaw = False
            persistent = False
        else:
            follow_start = position if first_h == -1 else position + int(first_h) + 1
            follow_end_h = (
                spec.matched_followup - 1
                if first_h == -1
                else int(first_h) + spec.matched_followup
            )
            follow_end = position + int(follow_end_h)
            follow = path.iloc[follow_start : follow_end + 1].to_numpy(dtype=int)
            if len(follow) != spec.matched_followup:
                raise BalancedStudyError("matched follow window length changed")
            returned = bool((follow == source).any())
            fixed_search = fixed_path.iloc[position : follow_end + 1].to_numpy(
                dtype=int
            )
            confirmations = np.flatnonzero(fixed_search == destination)
            matched_fixed_h = int(confirmations[0]) if len(confirmations) else math.nan
            if returned:
                category = (
                    "already_destination_whipsaw"
                    if first_h == -1
                    else "enters_then_whipsaw"
                )
                whipsaw, persistent = True, False
            else:
                confirmed = math.isfinite(matched_fixed_h)
                prefix = "already_destination" if first_h == -1 else "enters"
                category = (
                    f"{prefix}_persistent_{'confirmed' if confirmed else 'unconfirmed'}"
                )
                whipsaw, persistent = False, True
        unconfirmed = persistent and not math.isfinite(matched_fixed_h)
        original_fixed_h = float(anchor.fixed_confirmation_h)
        retained = bool(
            anchor.confirmed_early
            and math.isfinite(first_h)
            and math.isfinite(original_fixed_h)
            and float(first_h) < original_fixed_h
            and not whipsaw
        )
        records.append(
            {
                "market": anchor.market,
                "beta": anchor.beta,
                "beta_label": anchor.beta_label,
                "lambda0": anchor.lambda0,
                "signal_date": anchor.signal_date,
                "source_state": source,
                "destination_state": destination,
                "lagged_whipsaw_20": bool(anchor.whipsaw_20),
                "lagged_confirmed_early": bool(anchor.confirmed_early),
                "lagged_unconfirmed_persistent_20": bool(
                    anchor.unconfirmed_persistent_20
                ),
                "fixed_confirmation_h": anchor.fixed_confirmation_h,
                "first_destination_h": first_h,
                "matched_follow_end_h": follow_end_h,
                "matched_fixed_confirmation_h": matched_fixed_h,
                "matched_category": category,
                "matched_whipsaw_20": whipsaw,
                "matched_persistent_20": persistent,
                "matched_unconfirmed_persistent_20": unconfirmed,
                "matched_retained_confirmed_early": retained,
            }
        )
    result = pd.DataFrame.from_records(records, columns=ANCHOR_COLUMNS)
    allowed = set(MATCHED_CATEGORIES)
    if (
        counters["original_lagged_admitted_events"]
        != counters["matched_anchor_censored"] + counters["eligible_matched_anchors"]
        or len(result) != counters["eligible_matched_anchors"]
        or (len(result) and not set(result["matched_category"]).issubset(allowed))
    ):
        raise BalancedStudyError("matched anchors do not form the frozen partition")
    return result, counters


def summarize(
    events: pd.DataFrame, behavior: pd.DataFrame, spec: BalancedSpec
) -> pd.DataFrame:
    expected_paths = {
        (market, rule, float(lambda0))
        for market in spec.markets
        for rule in spec.rules
        for lambda0 in spec.event_lambdas
    }
    observed_paths = {
        (str(row.market), str(row.rule), float(row.lambda0))
        for row in behavior.itertuples()
    }
    if len(behavior) != len(expected_paths) or observed_paths != expected_paths:
        raise BalancedStudyError("path behavior coverage changed")
    if not set(EVENT_COLUMNS).issubset(events):
        raise BalancedStudyError("event table schema changed")
    rows: list[dict[str, Any]] = []
    for market in spec.markets:
        for rule in spec.rules:
            event = events.loc[(events["market"] == market) & (events["rule"] == rule)]
            paths = behavior.loc[
                (behavior["market"] == market) & (behavior["rule"] == rule)
            ]
            finite_margins = {}
            for column in (
                "terminal_state_margin",
                "terminal_predecessor_margin",
                "ablated_state_margin",
            ):
                values = pd.to_numeric(event[column], errors="coerce").to_numpy(
                    dtype=float
                )
                finite = values[np.isfinite(values)]
                finite_margins[f"minimum_{column}"] = (
                    float(finite.min()) if len(finite) else math.nan
                )
            rows.append(
                {
                    "market": market,
                    "rule": rule,
                    "beta": spec.decision_beta,
                    "beta_label": beta_label(spec.decision_beta),
                    "event_count": len(event),
                    "whipsaw_count": int(event["whipsaw_20"].astype(bool).sum()),
                    "persistent_count": int(event["persistent_20"].astype(bool).sum()),
                    "confirmed_early_count": int(
                        event["confirmed_early"].astype(bool).sum()
                    ),
                    "unconfirmed_persistent_count": int(
                        event["unconfirmed_persistent_20"].astype(bool).sum()
                    ),
                    **finite_margins,
                    "switch_count": int(paths["switch_count"].sum()),
                    "state_differences_vs_fixed": int(
                        paths["state_differences_vs_fixed"].sum()
                    ),
                    "state_differences_vs_lagged": int(
                        paths["state_differences_vs_lagged"].sum()
                    ),
                }
            )
    return pd.DataFrame.from_records(rows)


def classify(
    summary: pd.DataFrame,
    anchors: pd.DataFrame,
    spec: BalancedSpec,
    *,
    mechanical_passed: bool,
) -> dict[str, Any]:
    required_summary = {
        "market",
        "rule",
        "whipsaw_count",
        "switch_count",
        "state_differences_vs_fixed",
        "state_differences_vs_lagged",
        "unconfirmed_persistent_count",
    }
    if not required_summary.issubset(summary) or not set(ANCHOR_COLUMNS).issubset(
        anchors
    ):
        raise BalancedStudyError("decision evidence schema changed")
    expected_summary = {
        (market, rule) for market in spec.markets for rule in spec.rules
    }
    observed_summary = {
        (str(row.market), str(row.rule)) for row in summary.itertuples()
    }
    if len(summary) != len(expected_summary) or observed_summary != expected_summary:
        raise BalancedStudyError("decision summary coverage changed")
    by_market: dict[str, dict[str, Any]] = {}
    coverage: dict[str, bool] = {}
    switch_guard: dict[str, bool] = {}
    own_guard: dict[str, bool] = {}
    matched_guard: dict[str, bool] = {}
    latency_guard: dict[str, bool] = {}
    for market in spec.markets:
        rows = summary.loc[summary["market"] == market].set_index("rule")
        matched = anchors.loc[anchors["market"] == market]
        lagged_anchor_count = len(matched)
        lagged_confirmed = int(matched["lagged_confirmed_early"].astype(bool).sum())
        lagged_whipsaw = int(matched["lagged_whipsaw_20"].astype(bool).sum())
        balanced_whipsaw = int(matched["matched_whipsaw_20"].astype(bool).sum())
        retained = int(matched["matched_retained_confirmed_early"].astype(bool).sum())
        coverage[market] = lagged_anchor_count >= 1 and lagged_confirmed >= 1
        switch_guard[market] = int(rows.loc["balanced", "switch_count"]) <= int(
            rows.loc["lagged", "switch_count"]
        )
        own_guard[market] = int(rows.loc["balanced", "whipsaw_count"]) <= int(
            rows.loc["lagged", "whipsaw_count"]
        )
        matched_guard[market] = balanced_whipsaw <= lagged_whipsaw
        latency_guard[market] = retained >= 1
        by_market[market] = {
            "lagged_anchor_count": lagged_anchor_count,
            "lagged_confirmed_early_count": lagged_confirmed,
            "lagged_whipsaw_anchor_count": lagged_whipsaw,
            "balanced_matched_whipsaw_count": balanced_whipsaw,
            "balanced_retained_confirmed_early_count": retained,
            "matched_category_counts": {
                category: int((matched["matched_category"] == category).sum())
                for category in MATCHED_CATEGORIES
            },
        }
    lagged = summary.loc[summary["rule"] == "lagged"]
    balanced = summary.loc[summary["rule"] == "balanced"]
    pooled_lagged_whipsaw = int(lagged["whipsaw_count"].sum())
    pooled_balanced_whipsaw = int(balanced["whipsaw_count"].sum())
    matched_lagged_whipsaw = int(anchors["lagged_whipsaw_20"].astype(bool).sum())
    matched_balanced_whipsaw = int(anchors["matched_whipsaw_20"].astype(bool).sum())
    lagged_confirmed = int(anchors["lagged_confirmed_early"].astype(bool).sum())
    retained = int(anchors["matched_retained_confirmed_early"].astype(bool).sum())
    lagged_matched_lock = int(
        anchors["lagged_unconfirmed_persistent_20"].astype(bool).sum()
    )
    balanced_matched_lock = int(
        anchors["matched_unconfirmed_persistent_20"].astype(bool).sum()
    )
    pooled_category_counts = {
        category: int((anchors["matched_category"] == category).sum())
        for category in MATCHED_CATEGORIES
    }
    conditions = {
        "mechanical_prerequisites": bool(mechanical_passed),
        "nontrivial_vs_fixed": int(balanced["state_differences_vs_fixed"].sum()) > 0,
        "nontrivial_vs_lagged": int(balanced["state_differences_vs_lagged"].sum()) > 0,
        "anchor_coverage": all(coverage.values()),
        "market_switch_guard": all(switch_guard.values()),
        "own_market_whipsaw": all(own_guard.values()),
        "own_pooled_whipsaw": pooled_balanced_whipsaw < pooled_lagged_whipsaw,
        "matched_market_whipsaw": all(matched_guard.values()),
        "matched_pooled_whipsaw": matched_balanced_whipsaw < matched_lagged_whipsaw,
        "matched_latency_by_market": all(latency_guard.values()),
        "matched_latency_fraction": lagged_confirmed > 0
        and retained / lagged_confirmed >= 0.5,
        "matched_lock_in_guard": balanced_matched_lock <= lagged_matched_lock,
        "own_lock_in_guard": int(balanced["unconfirmed_persistent_count"].sum())
        <= int(lagged["unconfirmed_persistent_count"].sum()),
    }
    supported = all(conditions.values())
    return {
        "experiment_id": spec.experiment_id,
        "claim_class": "EXPLORATORY",
        "decision_beta_label": "log4",
        "performance_claim_allowed": False,
        "result": "supported" if supported else "not_supported",
        "conditions": conditions,
        "by_market": by_market,
        "pooled": {
            "lagged_own_whipsaw_count": pooled_lagged_whipsaw,
            "balanced_own_whipsaw_count": pooled_balanced_whipsaw,
            "lagged_matched_whipsaw_count": matched_lagged_whipsaw,
            "balanced_matched_whipsaw_count": matched_balanced_whipsaw,
            "lagged_confirmed_early_count": lagged_confirmed,
            "balanced_retained_confirmed_early_count": retained,
            "latency_fraction": retained / lagged_confirmed
            if lagged_confirmed
            else None,
            "lagged_matched_unconfirmed_count": lagged_matched_lock,
            "balanced_matched_unconfirmed_count": balanced_matched_lock,
            "matched_category_counts": pooled_category_counts,
        },
        "interpretation": (
            "Performance-free mechanism behavior on the repeatedly inspected "
            "development sample; no same-sample P&L is authorized."
        ),
    }


def analyze_market(
    inputs: MarketInputs,
    evidence: dict[str, LockedStateEvidence],
    spec: BalancedSpec,
) -> MechanismAnalysis:
    behavior = path_behavior(inputs, evidence, spec)
    events, own_audit = extract_events(inputs, evidence, spec)
    anchors, matched_audit = matched_response(
        events,
        evidence["balanced"].states[spec.decision_beta],
        inputs.candidates[0.0],
        inputs.refits,
        spec,
    )
    return MechanismAnalysis(
        market=inputs.market,
        behavior=behavior,
        events=events,
        anchors=anchors,
        penalties=penalty_summary(inputs.market, evidence, spec),
        audit={"own_events": own_audit, "matched_anchors": matched_audit},
    )
