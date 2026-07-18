"""Execute and verify lagged-evidence-mechanism-001 without opening P&L."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from adaptive_jump.artifacts import read_json, write_inventory, write_json
from adaptive_jump.config import ResearchConfig, load_config
from adaptive_jump.lagged_analysis import (
    MechanismAnalysis,
    _input_spec,
    analyze_market_mechanism,
)
from adaptive_jump.lagged_mechanics import run_locked_smoke
from adaptive_jump.lagged_model import generate_locked_candidates
from adaptive_jump.lagged_sources import implementation_lock, verify_source_inputs
from adaptive_jump.lagged_study import (
    LaggedMechanismSpec,
    LaggedStudyError,
    beta_label,
    classify_mechanism,
    load_lagged_spec,
    summarize_mechanism,
)
from adaptive_jump.lagged_verifier import verify_lagged_run
from adaptive_jump.separation_analysis import MarketInputs, load_market_inputs
from adaptive_jump.tv_jump import evidence_penalty_seq, lagged_evidence_penalty_seq

BUILDERS = {"arrival": evidence_penalty_seq, "lagged": lagged_evidence_penalty_seq}


def _load_inputs(
    market: str, fixed_dir: Path, arrival_dir: Path, spec: LaggedMechanismSpec
) -> tuple[MarketInputs, pd.DataFrame]:
    inputs = load_market_inputs(
        market,
        fixed_dir / "features.csv",
        arrival_dir,
        _input_spec(spec),
        include_fixed_objective=False,
    )
    names = ["date", *(str(value) for value in spec.lambdas)]
    fixed = pd.read_csv(fixed_dir / "jm-states.csv", usecols=names)
    fixed["date"] = pd.to_datetime(fixed["date"], errors="raise")
    fixed = fixed.set_index("date")
    fixed.columns = tuple(float(column) for column in fixed.columns)
    fixed = fixed.reindex(columns=spec.lambdas)
    sealed_fixed = inputs.candidates[0.0].reindex(columns=spec.lambdas)
    if not fixed.index.equals(inputs.features.index) or not np.array_equal(
        fixed, sealed_fixed, equal_nan=True
    ):
        raise LaggedStudyError(f"{market}: fixed and sealed beta-zero paths differ")
    return inputs, fixed


def _generate_locked(
    market: str,
    inputs: MarketInputs,
    fixed: pd.DataFrame,
    config: ResearchConfig,
    spec: LaggedMechanismSpec,
    *,
    terminal_limit: int | None = None,
    features: pd.DataFrame | None = None,
):
    feature_frame = inputs.features if features is None else features
    if not feature_frame.index.equals(inputs.features.index):
        raise LaggedStudyError("feature override changed the source dates")
    if terminal_limit is not None:
        position = spec.fit_window + terminal_limit - 2
        if terminal_limit <= 0 or position >= len(inputs.model_dates):
            raise LaggedStudyError("terminal smoke limit is outside model coverage")
    return generate_locked_candidates(
        feature_frame.reset_index(),
        fixed,
        inputs.refits,
        config,
        spec,
        market=market,
        penalty_builders=BUILDERS,
        terminal_limit=terminal_limit,
    )


def _parity(generated: pd.DataFrame, expected: pd.DataFrame, label: str) -> int:
    expected = expected.reindex(index=generated.index, columns=generated.columns)
    if not np.array_equal(generated, expected, equal_nan=True):
        raise LaggedStudyError(f"{label}: generated states differ from sealed source")
    return int(np.isfinite(generated.to_numpy(dtype=float)).sum())


def run_us_smoke(
    config: ResearchConfig,
    spec: LaggedMechanismSpec,
) -> dict[str, Any]:
    """Exercise 20 terminal dates without reading a return column or fitting."""
    sources = verify_source_inputs(config.path.parent, config, spec)
    inputs, fixed = _load_inputs(
        "us", sources.fixed_markets["us"], sources.arrival_markets["us"], spec
    )
    return run_locked_smoke(inputs, fixed, config, spec, BUILDERS)


def _run_market(
    market: str,
    fixed_dir: Path,
    arrival_dir: Path,
    target: Path,
    config: ResearchConfig,
    spec: LaggedMechanismSpec,
) -> tuple[MechanismAnalysis, dict[str, Any]]:
    with threadpool_limits(limits=1):
        inputs, fixed = _load_inputs(market, fixed_dir, arrival_dir, spec)
        evidence = _generate_locked(market, inputs, fixed, config, spec)
        arrival_cells = sum(
            _parity(
                evidence["arrival"].states[beta],
                inputs.candidates[beta],
                f"{market}/arrival/{beta_label(beta)}",
            )
            for beta in spec.betas
        )
        beta_cells = sum(
            _parity(evidence[rule].states[0.0], fixed, f"{market}/{rule}/beta-zero")
            for rule in spec.rules
        )
        terminal_rows = int(fixed.notna().all(axis=1).sum())
        replay = {
            "sealed_arrival_exact": arrival_cells
            == terminal_rows * len(spec.lambdas) * len(spec.betas),
            "beta_zero_exact": beta_cells
            == terminal_rows * len(spec.lambdas) * len(spec.rules),
            "sealed_arrival_state_cells_checked": arrival_cells,
            "beta_zero_state_cells_checked": beta_cells,
            "return_columns_accessed": False,
        }

        target.mkdir(parents=True, exist_ok=True)
        lagged = evidence["lagged"]
        for beta, values in lagged.states.items():
            values.to_csv(target / f"candidate-states-beta-{beta_label(beta)}.csv")
        lagged.refits.to_csv(target / "refits-and-scales.csv", index=False)
        result = analyze_market_mechanism(
            market,
            fixed_dir / "features.csv",
            arrival_dir,
            lagged.states,
            spec,
        )
        result.behavior.to_csv(target / "path-behavior.csv", index=False)
        result.events.to_csv(target / "discount-events.csv", index=False)
        write_json(target / "audit.json", result.audit)
        return result, replay


def _mechanical_checks(
    mechanics: dict[str, Any],
    smoke: dict[str, Any],
    market_replays: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    expected_smoke = {
        "sealed_arrival_exact": True,
        "beta_zero_exact": True,
        "prefix_invariant": True,
        "refit_convention_numeric": True,
        "lagged_discounts_present": True,
        "performance_files_accessed": False,
        "return_columns_accessed": False,
        "future_mutation_effect_present": True,
        "post_2023_accessed": False,
    }
    smoke_checks = {key: smoke[key] for key in expected_smoke}
    smoke_passed = all(
        smoke_checks[key] is expected for key, expected in expected_smoke.items()
    )
    replays_passed = bool(market_replays) and all(
        replay["sealed_arrival_exact"] is True
        and replay["beta_zero_exact"] is True
        and replay["return_columns_accessed"] is False
        for replay in market_replays.values()
    )
    return {
        "schema_version": 1,
        "mechanical_prerequisites": mechanics,
        "smoke_checks": smoke_checks,
        "market_replays": market_replays,
        "passed": bool(
            mechanics.get("passed") is True and smoke_passed and replays_passed
        ),
    }


def run_lagged_study(
    config: ResearchConfig,
    spec: LaggedMechanismSpec,
) -> Path:
    """Run US smoke, then US/DE/JP candidate-path analyses in parallel."""
    root = config.path.parent
    sources = verify_source_inputs(root, config, spec)
    smoke = run_us_smoke(config, spec)
    implementation = implementation_lock(root, spec)
    run_id = (
        f"lagged-evidence-{spec.sha256[:12]}-"
        f"{spec.arrival_inventory_sha256[:12]}-"
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
            verify_lagged_run(run_dir)
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
            "study_kind": "lagged_evidence_mechanism",
            "experiment_id": spec.experiment_id,
            "run_id": run_id,
            "status": "running",
            "claim_class": "EXPLORATORY",
            "performance_files_accessed": False,
            "return_columns_accessed": False,
            "post_2023_accessed": False,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "spec_sha256": spec.sha256,
            "config_sha256": config.sha256,
            "fixed_inventory_sha256": spec.fixed_inventory_sha256,
            "arrival_inventory_sha256": spec.arrival_inventory_sha256,
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
                sources.arrival_markets[market],
                run_dir / market,
                config,
                spec,
            ): market
            for market in spec.markets
        }
        for future in as_completed(futures):
            market = futures[future]
            results[market], market_replays[market] = future.result()
            print(f"{market}: lagged mechanism complete", flush=True)

    behavior = pd.concat(
        [results[market].behavior for market in spec.markets],
        ignore_index=True,
    )
    events = pd.concat(
        [results[market].events for market in spec.markets],
        ignore_index=True,
    )
    summary = summarize_mechanism(events, behavior, spec)
    dated = (
        events.sort_values(["market", "rule", "beta_label", "signal_date"])
        .groupby(["market", "rule", "beta_label"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    mechanics = _mechanical_checks(
        smoke["mechanical_prerequisites"], smoke, market_replays
    )
    conclusion = classify_mechanism(
        summary,
        spec,
        mechanical_prerequisites_passed=mechanics["passed"],
    )
    behavior.to_csv(run_dir / "path-behavior.csv", index=False)
    events.to_csv(run_dir / "discount-events.csv", index=False)
    summary.to_csv(run_dir / "mechanism-summary.csv", index=False)
    dated.to_csv(run_dir / "dated-audit.csv", index=False)
    write_json(run_dir / "mechanical-checks.json", mechanics)
    write_json(run_dir / "conclusion.json", conclusion)

    metadata = read_json(metadata_path)
    metadata.update(
        {
            "status": "complete",
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "result": conclusion["result"],
            "selected_beta_label": conclusion["selected_beta_label"],
            "mechanical_prerequisites_passed": mechanics["passed"],
            "events": len(events),
        }
    )
    write_json(metadata_path, metadata)
    write_inventory(run_dir)
    verify_lagged_run(run_dir)
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(prog="lagged-evidence-mechanism")
    parser.add_argument("--config", required=True)
    parser.add_argument("--spec", default="research/lagged-evidence-mechanism-001.toml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--verify")
    arguments = parser.parse_args()
    if arguments.verify:
        print(json.dumps(verify_lagged_run(arguments.verify), sort_keys=True))
        return 0
    config = load_config(arguments.config)
    spec_path = Path(arguments.spec)
    if not spec_path.is_absolute():
        spec_path = config.path.parent / spec_path
    spec = load_lagged_spec(spec_path, config)
    if arguments.smoke:
        print(json.dumps(run_us_smoke(config, spec), sort_keys=True))
    else:
        print(run_lagged_study(config, spec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
