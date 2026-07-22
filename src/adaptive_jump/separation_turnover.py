"""Exploratory link between decision-time regime separation and next-month switching."""

from __future__ import annotations

import argparse
import json
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from adaptive_jump.artifacts import ArtifactError, sha256_file, write_json

EXPERIMENT_ID = "separation-turnover-001"
SPEC_NAME = "separation-turnover-001.toml"
DEVELOPMENT_CUTOFF = pd.Timestamp("2023-12-31")
PERMUTATION_DRAWS = 10_000
PERMUTATION_SEED = 20260722


class SeparationTurnoverError(ArtifactError):
    """Raised when the frozen separation-turnover contract cannot be satisfied."""


@dataclass(frozen=True)
class SeparationSpec:
    spec_sha256: str
    run_id: str
    run_inventory_sha256: str
    variant: str
    markets: tuple[str, ...]


def load_spec(repo_root: Path) -> SeparationSpec:
    """Load the frozen contract and require its registered FROZEN hash."""
    spec_path = repo_root / "research" / SPEC_NAME
    digest = sha256_file(spec_path)
    registered = False
    registry = repo_root / "research" / "experiment_registry.jsonl"
    for line in registry.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        if (
            entry.get("experiment_id") == EXPERIMENT_ID
            and entry.get("status") == "FROZEN"
            and entry.get("frozen_spec_hash") == digest
        ):
            registered = True
    if not registered:
        raise SeparationTurnoverError("frozen registration is missing or stale")
    document = tomllib.loads(spec_path.read_text(encoding="utf-8"))
    source = document["source"]
    return SeparationSpec(
        spec_sha256=digest,
        run_id=source["run_id"],
        run_inventory_sha256=source["run_inventory_sha256"],
        variant=source["variant"],
        markets=tuple(source["markets"]),
    )


def parse_centers(text: str) -> tuple[float, float]:
    """Parse the stored two-state single-feature centers."""
    value = json.loads(text)
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(not isinstance(row, list) or len(row) != 1 for row in value)
    ):
        raise SeparationTurnoverError(
            f"centers are not two single-feature states: {text}"
        )
    return float(value[0][0]), float(value[1][0])


def separation_table(run_dir: Path, market: str, variant: str) -> pd.DataFrame:
    """Build the per-decision separation and window-outcome table for one market."""
    target = run_dir / market / variant
    choices = pd.read_csv(target / "choices.csv")
    refits = pd.read_csv(target / "refits.csv")
    trades = pd.read_csv(target / "trades.csv")
    choices["decision_date"] = pd.to_datetime(choices["decision_date"], errors="raise")
    refits["fit_date"] = pd.to_datetime(refits["fit_date"], errors="raise")
    trades["date"] = pd.to_datetime(trades["date"], errors="raise")
    if choices["decision_date"].max() > DEVELOPMENT_CUTOFF:
        raise SeparationTurnoverError(f"{market}: decision date beyond the cutoff")
    if trades["date"].max() > DEVELOPMENT_CUTOFF:
        raise SeparationTurnoverError(f"{market}: trade date beyond the cutoff")

    rows: list[dict[str, Any]] = []
    for decision, selected in zip(
        choices["decision_date"], choices["selected"], strict=True
    ):
        active = refits.loc[
            (refits["fit_date"] <= decision) & (refits["lambda"] == float(selected))
        ]
        if active.empty:
            raise SeparationTurnoverError(
                f"{market}: no refit for {decision.date()} lambda={selected}"
            )
        active = active.sort_values("fit_date").iloc[-1]
        collapsed = int(active["active_state_count"]) < 2
        if collapsed:
            separation = 0.0
        else:
            center_low, center_high = parse_centers(active["centers"])
            separation = abs(center_high - center_low)
        rows.append(
            {
                "market": market,
                "decision_date": decision,
                "selected_lambda": float(selected),
                "fit_date": active["fit_date"],
                "active_state_count": int(active["active_state_count"]),
                "collapsed": collapsed,
                "separation": separation,
            }
        )
    table = pd.DataFrame(rows)

    boundaries = table["decision_date"].to_numpy()
    dates = trades["date"].to_numpy()
    position = trades["position"].to_numpy()
    turnover = trades["one_way_turnover"].to_numpy()
    window = np.searchsorted(boundaries, dates, side="left") - 1
    in_scope = window >= 0
    if not in_scope.any():
        raise SeparationTurnoverError(f"{market}: no trade dates after first decision")
    covered = int(in_scope.sum())
    switches_next = np.zeros(len(table), dtype=int)
    turnover_next = np.zeros(len(table))
    days_next = np.zeros(len(table), dtype=int)
    previous_position = 0.0
    for index in range(len(dates)):
        if not in_scope[index]:
            previous_position = position[index]
            continue
        slot = window[index]
        if position[index] != previous_position:
            switches_next[slot] += 1
        turnover_next[slot] += turnover[index]
        days_next[slot] += 1
        previous_position = position[index]
    if int(days_next.sum()) != covered:
        raise SeparationTurnoverError(f"{market}: window partition is not exact")
    total_changes = int(
        (np.diff(position[in_scope]) != 0).sum()
        + (
            position[in_scope][0]
            != (position[~in_scope][-1] if (~in_scope).any() else 0.0)
        )
    )
    if int(switches_next.sum()) != total_changes:
        raise SeparationTurnoverError(f"{market}: switch totals are inconsistent")

    table["switches_next"] = switches_next
    table["turnover_next"] = turnover_next
    table["days_next"] = days_next
    table["switches_prev"] = table["switches_next"].shift(1)
    table["turnover_prev"] = table["turnover_next"].shift(1)
    return table.loc[table["days_next"] > 0].reset_index(drop=True)


def association(table: pd.DataFrame, outcome: str, seed: int) -> dict[str, float]:
    """One-sided Spearman permutation association for one market table."""
    separation = table["separation"].to_numpy()
    observed_outcome = table[outcome].to_numpy()
    rho = float(spearmanr(separation, observed_outcome).statistic)
    generator = np.random.default_rng(seed)
    draws = np.empty(PERMUTATION_DRAWS)
    for index in range(PERMUTATION_DRAWS):
        draws[index] = spearmanr(
            generator.permutation(separation), observed_outcome
        ).statistic
    p_value = float((draws <= rho).mean())
    return {"rho": rho, "p_one_sided": p_value, "n": int(len(table))}


def decide(primary: dict[str, dict[str, float]]) -> str:
    """Apply the frozen decision rule to per-market primary associations."""
    rhos = [values["rho"] for values in primary.values()]
    p_values = [values["p_one_sided"] for values in primary.values()]
    if all(rho < 0 for rho in rhos) and sum(p < 0.05 for p in p_values) >= 2:
        return "supported"
    if sum(rho >= 0 for rho in rhos) >= 2:
        return "not_supported"
    return "inconclusive"


def run_separation_turnover(repo_root: Path, run_dir: Path) -> Path:
    """Execute the frozen exploratory and seal its table, summary, and chart."""
    spec = load_spec(repo_root)
    if run_dir.name != spec.run_id:
        raise SeparationTurnoverError("run directory does not match the contract")
    inventory_digest = sha256_file(run_dir / "inventory.json")
    if inventory_digest != spec.run_inventory_sha256:
        raise SeparationTurnoverError("source inventory hash changed")

    tables = [
        separation_table(run_dir, market, spec.variant) for market in spec.markets
    ]
    table = pd.concat(tables, ignore_index=True)
    primary: dict[str, dict[str, float]] = {}
    secondary: dict[str, dict[str, float]] = {}
    for market, market_table in zip(spec.markets, tables, strict=True):
        primary[market] = association(market_table, "switches_next", PERMUTATION_SEED)
        secondary[market] = association(market_table, "turnover_next", PERMUTATION_SEED)
    conclusion = decide(primary)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = (
        repo_root
        / "artifacts"
        / EXPERIMENT_ID
        / f"separation-turnover-{spec.spec_sha256[:12]}-{stamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=False)
    table_out = table.copy()
    table_out["decision_date"] = table_out["decision_date"].dt.date.astype(str)
    table_out["fit_date"] = pd.to_datetime(table_out["fit_date"]).dt.date.astype(str)
    table_out.to_csv(out_dir / "separation-turnover.csv", index=False)
    write_json(
        out_dir / "summary.json",
        {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "spec_sha256": spec.spec_sha256,
            "source_run_id": spec.run_id,
            "source_inventory_sha256": inventory_digest,
            "conclusion": conclusion,
            "primary_switches_next": primary,
            "secondary_turnover_next": secondary,
            "collapsed_rows": {
                market: int(market_table["collapsed"].sum())
                for market, market_table in zip(spec.markets, tables, strict=True)
            },
            "permutation": {
                "draws": PERMUTATION_DRAWS,
                "seed": PERMUTATION_SEED,
                "sided": "one-sided toward negative",
            },
            "created_at_utc": datetime.now(UTC).isoformat(),
        },
    )
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", required=True, help="sealed dd-loss-scale run directory"
    )
    arguments = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = run_separation_turnover(repo_root, Path(arguments.run).resolve())
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    print(json.dumps({"out_dir": str(out_dir), "conclusion": summary["conclusion"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
