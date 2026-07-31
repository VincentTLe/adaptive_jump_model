"""Execute the frozen pair-balanced mechanism study without opening P&L."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import pandas as pd
from threadpoolctl import threadpool_limits

from adaptive_jump.artifacts import read_json, write_inventory, write_json
from adaptive_jump.balanced_analysis import (
    MechanismAnalysis,
    analyze_market,
    classify,
    summarize,
)
from adaptive_jump.balanced_model import (
    BalancedSpec,
    beta_label,
    load_balanced_spec,
    load_market_inputs,
)
from adaptive_jump.balanced_smoke import (
    balanced_penalty_checks,
    candidate_parity,
    generate_evidence,
    run_us_smoke,
)
from adaptive_jump.balanced_sources import (
    implementation_lock,
    verify_source_inputs,
)
from adaptive_jump.config import ResearchConfig, load_config


def _verify(run_dir: Path) -> dict[str, Any]:
    from adaptive_jump.balanced_verifier import verify_balanced_run

    return verify_balanced_run(run_dir)


def _run_market(
    market: str,
    fixed_dir: Path,
    parent_dir: Path,
    target: Path,
    config: ResearchConfig,
    spec: BalancedSpec,
) -> tuple[MechanismAnalysis, dict[str, Any]]:
    with threadpool_limits(limits=1):
        inputs, fixed = load_market_inputs(market, fixed_dir, parent_dir, spec)
        evidence = generate_evidence(inputs, fixed, config, spec)
        parent_cells = sum(
            candidate_parity(
                evidence["lagged"].states[beta],
                inputs.candidates[beta],
                f"{market}/lagged/{beta_label(beta)}",
            )
            for beta in spec.betas
        )
        beta_zero_cells = sum(
            candidate_parity(
                evidence[rule].states[0.0],
                fixed,
                f"{market}/{rule}/beta-zero",
            )
            for rule in spec.rules
        )
        discounts, surcharges, pair_error = balanced_penalty_checks(
            evidence["balanced"], spec
        )
        terminal_rows = int(fixed.notna().all(axis=1).sum())
        cells_per_path = terminal_rows * len(spec.lambdas)
        all_candidate_cells = sum(
            int(evidence[rule].states[beta].notna().sum().sum())
            for rule in spec.rules
            for beta in spec.betas
        )
        replay = {
            "parent_lagged_exact": parent_cells == cells_per_path * len(spec.betas),
            "beta_zero_exact": beta_zero_cells == cells_per_path * len(spec.rules),
            "candidate_coverage_exact": all_candidate_cells
            == cells_per_path * len(spec.rules) * len(spec.betas),
            "pair_balance_exact": pair_error <= spec.numerical_tolerance,
            "terminal_rows": terminal_rows,
            "parent_lagged_state_cells_checked": parent_cells,
            "beta_zero_state_cells_checked": beta_zero_cells,
            "all_candidate_state_cells_checked": all_candidate_cells,
            "balanced_discount_cells": discounts,
            "balanced_surcharge_cells": surcharges,
            "maximum_pair_sum_abs_error": pair_error,
            "return_columns_accessed": False,
        }
        target.mkdir(parents=True, exist_ok=True)
        for rule in spec.rules:
            for beta in spec.betas:
                evidence[rule].states[beta].to_csv(
                    target / f"candidate-states-{rule}-beta-{beta_label(beta)}.csv"
                )
        result = analyze_market(inputs, evidence, spec)
        result.behavior.to_csv(target / "path-behavior.csv", index=False)
        result.events.to_csv(target / "discount-events.csv", index=False)
        result.anchors.to_csv(target / "matched-lagged-anchors.csv", index=False)
        result.penalties.to_csv(target / "penalty-summary.csv", index=False)
        write_json(target / "audit.json", result.audit)
        return result, replay


def _finalize_verified_run(
    run_dir: Path, metadata_path: Path, metadata: dict[str, Any]
) -> None:
    """Expose no complete lifecycle state until sealed replay succeeds."""
    metadata.update(
        {
            "status": "verifying",
            "verification_started_at_utc": datetime.now(UTC).isoformat(),
        }
    )
    write_json(metadata_path, metadata)
    write_inventory(run_dir)
    first = _verify(run_dir)
    if first.get("lifecycle") != "verifying":
        raise RuntimeError("balanced pre-completion verification status changed")
    metadata.update(
        {
            "status": "complete",
            "finished_at_utc": datetime.now(UTC).isoformat(),
        }
    )
    write_json(metadata_path, metadata)
    try:
        final = _verify(run_dir)
        if final.get("lifecycle") != "complete":
            raise RuntimeError("balanced final verification status changed")
    except Exception as exc:
        metadata.pop("finished_at_utc", None)
        metadata.update(
            {
                "status": "invalid_verification",
                "verification_error": (
                    f"final complete-status verification failed ({type(exc).__name__})"
                ),
            }
        )
        write_json(metadata_path, metadata)
        raise


def _dated_audit(events: pd.DataFrame) -> pd.DataFrame:
    """Select the earliest event per market and rule, with lambda as tie-break."""
    required = {"market", "rule", "signal_date", "lambda0"}
    if not required.issubset(events):
        raise ValueError("balanced dated-audit event schema changed")
    return (
        events.sort_values(["market", "rule", "signal_date", "lambda0"])
        .groupby(["market", "rule"], sort=False, as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def _mechanical_checks(
    prerequisites: dict[str, Any],
    smoke: dict[str, Any],
    market_replays: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    expected_smoke = {
        "parent_lagged_exact": True,
        "beta_zero_exact": True,
        "short_long_prefix_exact": True,
        "future_mutation_prefix_invariant": True,
        "prefix_invariant": True,
        "future_mutation_effect_present": True,
        "actual_formula_exact": True,
        "actual_bounds_exact": True,
        "formula_through_second_refit": True,
        "pair_balance_exact": True,
        "balanced_discounts_present": True,
        "balanced_surcharges_present": True,
        "refit_convention_numeric": True,
        "performance_files_accessed": False,
        "return_columns_accessed": False,
        "post_2023_accessed": False,
    }
    smoke_checks = {key: smoke.get(key) for key in expected_smoke}
    smoke_passed = all(
        smoke_checks[key] is expected for key, expected in expected_smoke.items()
    )
    replays_passed = set(market_replays) == {"us", "de", "jp"} and all(
        replay.get("parent_lagged_exact") is True
        and replay.get("beta_zero_exact") is True
        and replay.get("candidate_coverage_exact") is True
        and replay.get("pair_balance_exact") is True
        and replay.get("return_columns_accessed") is False
        for replay in market_replays.values()
    )
    return {
        "schema_version": 1,
        "mechanical_prerequisites": prerequisites,
        "smoke_checks": smoke_checks,
        "market_replays": market_replays,
        "passed": bool(
            prerequisites.get("passed") is True and smoke_passed and replays_passed
        ),
    }


def run_balanced_study(config: ResearchConfig, spec: BalancedSpec) -> Path:
    """Run US smoke, then all three performance-free market paths in parallel."""
    root = config.path.parent
    sources = verify_source_inputs(root, config, spec)
    smoke = run_us_smoke(config, spec)
    implementation = implementation_lock(root, spec)
    run_id = (
        f"balanced-lagged-{spec.sha256[:12]}-"
        f"{spec.parent_inventory_sha256[:12]}-"
        f"{implementation['implementation_sha256'][:12]}"
    )
    run_dir = root / config.artifact_root / spec.artifact_subdir / run_id
    metadata_path = run_dir / "run.json"
    if metadata_path.exists():
        metadata = read_json(metadata_path)
        if (
            metadata.get("status") == "complete"
            and metadata.get("spec_sha256") == spec.sha256
            and metadata.get("implementation_sha256")
            == implementation["implementation_sha256"]
        ):
            _verify(run_dir)
            return run_dir

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "study.lock.toml").write_bytes(spec.path.read_bytes())
    (run_dir / "config.lock.toml").write_bytes(config.path.read_bytes())
    write_json(run_dir / "source-lock.json", sources.source_lock)
    write_json(run_dir / "implementation-lock.json", implementation)
    write_json(run_dir / "smoke.json", smoke)
    write_json(
        metadata_path,
        {
            "schema_version": 1,
            "study_kind": "balanced_lagged_mechanism",
            "experiment_id": spec.experiment_id,
            "run_id": run_id,
            "status": "running",
            "claim_class": "EXPLORATORY",
            "performance_claim_allowed": False,
            "paper_replication_claim_allowed": False,
            "new_model_claim_allowed": False,
            "performance_files_accessed": False,
            "return_columns_accessed": False,
            "post_2023_accessed": False,
            "provider_accessed": False,
            "monthly_selection_performed": False,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "spec_sha256": spec.sha256,
            "config_sha256": config.sha256,
            "fixed_inventory_sha256": spec.fixed_inventory_sha256,
            "parent_inventory_sha256": spec.parent_inventory_sha256,
            "parent_spec_sha256": spec.parent_spec_sha256,
            "data_manifest_sha256": spec.data_manifest_sha256,
            "implementation_sha256": implementation["implementation_sha256"],
            "git_head": implementation["git_head"],
        },
    )

    results: dict[str, MechanismAnalysis] = {}
    market_replays: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(
        max_workers=len(spec.markets), mp_context=get_context("forkserver")
    ) as executor:
        futures = {
            executor.submit(
                _run_market,
                market,
                sources.fixed_markets[market],
                sources.parent_markets[market],
                run_dir / market,
                config,
                spec,
            ): market
            for market in spec.markets
        }
        for future in as_completed(futures):
            market = futures[future]
            results[market], market_replays[market] = future.result()
            print(f"{market}: balanced mechanism complete", flush=True)

    behavior = pd.concat(
        [results[market].behavior for market in spec.markets], ignore_index=True
    )
    events = pd.concat(
        [results[market].events for market in spec.markets], ignore_index=True
    )
    anchors = pd.concat(
        [results[market].anchors for market in spec.markets], ignore_index=True
    )
    penalties = pd.concat(
        [results[market].penalties for market in spec.markets], ignore_index=True
    )
    summary = summarize(events, behavior, spec)
    dated = _dated_audit(events)
    mechanics = _mechanical_checks(
        smoke["mechanical_prerequisites"], smoke, market_replays
    )
    conclusion = classify(summary, anchors, spec, mechanical_passed=mechanics["passed"])
    behavior.to_csv(run_dir / "path-behavior.csv", index=False)
    events.to_csv(run_dir / "discount-events.csv", index=False)
    anchors.to_csv(run_dir / "matched-lagged-anchors.csv", index=False)
    penalties.to_csv(run_dir / "penalty-summary.csv", index=False)
    summary.to_csv(run_dir / "mechanism-summary.csv", index=False)
    dated.to_csv(run_dir / "dated-audit.csv", index=False)
    write_json(run_dir / "mechanical-checks.json", mechanics)
    write_json(run_dir / "conclusion.json", conclusion)

    metadata = read_json(metadata_path)
    metadata.update(
        {
            "result": conclusion["result"],
            "decision_beta_label": conclusion["decision_beta_label"],
            "mechanical_prerequisites_passed": mechanics["passed"],
            "path_rows": len(behavior),
            "event_rows": len(events),
            "matched_anchor_rows": len(anchors),
            "penalty_rows": len(penalties),
            "summary_rows": len(summary),
            "dated_audit_rows": len(dated),
        }
    )
    _finalize_verified_run(run_dir, metadata_path, metadata)
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(prog="balanced-lagged-mechanism")
    parser.add_argument("--config", required=True)
    parser.add_argument("--spec", default="research/balanced-lagged-mechanism-001.toml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--verify")
    arguments = parser.parse_args()
    if arguments.verify:
        print(json.dumps(_verify(Path(arguments.verify)), sort_keys=True))
        return 0
    config = load_config(arguments.config)
    spec_path = Path(arguments.spec)
    if not spec_path.is_absolute():
        spec_path = config.path.parent / spec_path
    spec = load_balanced_spec(spec_path, config)
    if arguments.smoke:
        print(json.dumps(run_us_smoke(config, spec), sort_keys=True))
    else:
        print(run_balanced_study(config, spec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
