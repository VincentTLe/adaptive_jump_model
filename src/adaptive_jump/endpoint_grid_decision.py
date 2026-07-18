"""Descriptive endpoint effects and the frozen cell-D rescue rule."""

from __future__ import annotations

from typing import Any

import pandas as pd

from adaptive_jump.endpoint_grid_types import (
    MDD_ABSOLUTE_DEADBAND,
    PRIMARY_DELAY,
    EndpointGridError,
)

MARKETS = ("us", "de", "jp")


def endpoint_effects(metrics: pd.DataFrame) -> pd.DataFrame:
    fields = ("sharpe", "maximum_drawdown", "turnover", "cash_fraction", "switch_count")
    rows = []
    for (market, delay), values in metrics.groupby(["market", "delay"]):
        indexed = values.set_index("path")
        for model, baseline, endpoint in (
            ("fixed_jm", "J0", "J1"),
            ("hmm", "K0", "K1"),
        ):
            rows.append(
                {
                    "market": market,
                    "delay": delay,
                    "model": model,
                    "baseline_path": baseline,
                    "endpoint_path": endpoint,
                    **{
                        f"delta_{field}": float(
                            indexed.loc[endpoint, field] - indexed.loc[baseline, field]
                        )
                        for field in fields
                    },
                }
            )
    return pd.DataFrame.from_records(rows)


def d_rescue_decision(metrics: pd.DataFrame) -> dict[str, Any]:
    primary = metrics.loc[metrics["delay"] == PRIMARY_DELAY]
    if set(primary["market"]) != set(MARKETS):
        raise EndpointGridError("D rescue gate does not cover all markets")
    markets = []
    for market in MARKETS:
        indexed = primary.loc[primary["market"] == market].set_index("path")
        if not {"buy_and_hold", "J1", "K1"}.issubset(indexed.index):
            raise EndpointGridError(f"{market}: D rescue paths are incomplete")
        j1, k1, hold = indexed.loc["J1"], indexed.loc["K1"], indexed.loc["buy_and_hold"]
        improvement = abs(float(hold.maximum_drawdown)) - abs(
            float(j1.maximum_drawdown)
        )
        checks = {
            "j1_sharpe_gt_k1": bool(j1.sharpe > k1.sharpe),
            "j1_sharpe_gt_buy_and_hold": bool(j1.sharpe > hold.sharpe),
            "j1_abs_mdd_better_than_buy_and_hold": bool(
                improvement > MDD_ABSOLUTE_DEADBAND
            ),
        }
        markets.append(
            {
                "market": market,
                **checks,
                "mdd_absolute_improvement": improvement,
                "passed": all(checks.values()),
            }
        )
    passed = all(row["passed"] for row in markets)
    return {
        "schema_version": 1,
        "cell": "D",
        "primary_delay": PRIMARY_DELAY,
        "mdd_absolute_deadband": MDD_ABSOLUTE_DEADBAND,
        "markets": markets,
        "all_markets_passed": passed,
        "decision": "passed" if passed else "failed",
        "descriptive_only": True,
        "claim_class": "EXPLORATORY",
        "performance_claim_allowed": False,
        "paper_replication_claim_allowed": False,
    }
