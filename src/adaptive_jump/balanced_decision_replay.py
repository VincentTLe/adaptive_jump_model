"""Independent summary, dated-audit, and decision replay."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from adaptive_jump.balanced_model import (
    BalancedSpec,
    BalancedStudyError,
    beta_label,
)

MATCHED_CATEGORIES = (
    "already_destination_whipsaw",
    "already_destination_persistent_confirmed",
    "already_destination_persistent_unconfirmed",
    "suppressed_no_entry",
    "enters_then_whipsaw",
    "enters_persistent_confirmed",
    "enters_persistent_unconfirmed",
)


def summarize(
    events: pd.DataFrame, behavior: pd.DataFrame, spec: BalancedSpec
) -> pd.DataFrame:
    required_event = {
        "market",
        "rule",
        "whipsaw_20",
        "persistent_20",
        "confirmed_early",
        "unconfirmed_persistent_20",
        "terminal_state_margin",
        "terminal_predecessor_margin",
        "ablated_state_margin",
    }
    required_behavior = {
        "market",
        "rule",
        "lambda0",
        "switch_count",
        "state_differences_vs_fixed",
        "state_differences_vs_lagged",
    }
    if not required_event.issubset(events) or not required_behavior.issubset(behavior):
        raise BalancedStudyError("summary inputs are incomplete")
    expected_paths = {
        (market, rule, lambda0)
        for market in spec.markets
        for rule in spec.rules
        for lambda0 in spec.event_lambdas
    }
    observed_paths = {
        (row.market, row.rule, float(row.lambda0))
        for row in behavior.itertuples(index=False)
    }
    if len(behavior) != len(expected_paths) or observed_paths != expected_paths:
        raise BalancedStudyError("path behavior coverage changed")
    rows: list[dict[str, Any]] = []
    for market in spec.markets:
        for rule in spec.rules:
            event = events.loc[(events["market"] == market) & (events["rule"] == rule)]
            paths = behavior.loc[
                (behavior["market"] == market) & (behavior["rule"] == rule)
            ]
            minimum_margins: dict[str, float] = {}
            for column in (
                "terminal_state_margin",
                "terminal_predecessor_margin",
                "ablated_state_margin",
            ):
                values = pd.to_numeric(event[column], errors="coerce").tolist()
                finite = [float(value) for value in values if math.isfinite(value)]
                minimum_margins[f"minimum_{column}"] = (
                    min(finite) if finite else math.nan
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
                    **minimum_margins,
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
    expected = {(market, rule) for market in spec.markets for rule in spec.rules}
    observed = {(row.market, row.rule) for row in summary.itertuples(index=False)}
    if len(summary) != len(expected) or observed != expected:
        raise BalancedStudyError("decision summary coverage changed")
    required_anchor = {
        "market",
        "lagged_confirmed_early",
        "lagged_whipsaw_20",
        "matched_whipsaw_20",
        "matched_retained_confirmed_early",
        "lagged_unconfirmed_persistent_20",
        "matched_unconfirmed_persistent_20",
    }
    if not required_anchor.issubset(anchors):
        raise BalancedStudyError("decision anchor columns changed")
    by_market: dict[str, dict[str, Any]] = {}
    coverage: dict[str, bool] = {}
    switch_guard: dict[str, bool] = {}
    own_guard: dict[str, bool] = {}
    matched_guard: dict[str, bool] = {}
    latency_guard: dict[str, bool] = {}
    for market in spec.markets:
        rows = summary.loc[summary["market"] == market].set_index("rule")
        matched = anchors.loc[anchors["market"] == market]
        anchor_count = len(matched)
        lagged_confirmed = int(matched["lagged_confirmed_early"].astype(bool).sum())
        lagged_whipsaw = int(matched["lagged_whipsaw_20"].astype(bool).sum())
        balanced_whipsaw = int(matched["matched_whipsaw_20"].astype(bool).sum())
        retained = int(matched["matched_retained_confirmed_early"].astype(bool).sum())
        coverage[market] = anchor_count >= 1 and lagged_confirmed >= 1
        switch_guard[market] = int(rows.loc["balanced", "switch_count"]) <= int(
            rows.loc["lagged", "switch_count"]
        )
        own_guard[market] = int(rows.loc["balanced", "whipsaw_count"]) <= int(
            rows.loc["lagged", "whipsaw_count"]
        )
        matched_guard[market] = balanced_whipsaw <= lagged_whipsaw
        latency_guard[market] = retained >= 1
        by_market[market] = {
            "lagged_anchor_count": anchor_count,
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
            "latency_fraction": (
                retained / lagged_confirmed if lagged_confirmed else None
            ),
            "lagged_matched_unconfirmed_count": lagged_matched_lock,
            "balanced_matched_unconfirmed_count": balanced_matched_lock,
            "matched_category_counts": pooled_category_counts,
        },
        "interpretation": (
            "Performance-free mechanism behavior on the repeatedly inspected "
            "development sample; no same-sample P&L is authorized."
        ),
    }


def dated_audit(events: pd.DataFrame) -> pd.DataFrame:
    return (
        events.sort_values(["market", "rule", "signal_date", "lambda0"])
        .groupby(["market", "rule"], sort=False, as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
