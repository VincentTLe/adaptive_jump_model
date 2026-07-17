"""Execute the frozen adaptive-separation-001 mechanism diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from adaptive_jump.artifacts import (
    read_json,
    sha256_file,
    verify_inventory,
    write_inventory,
    write_json,
)
from adaptive_jump.config import ResearchConfig, load_config
from adaptive_jump.separation_analysis import MarketAnalysis, analyze_market
from adaptive_jump.separation_evaluation import (
    build_conclusion,
    evaluate_leave_one_market_out,
    summarize_mechanism,
)
from adaptive_jump.separation_study import (
    SeparationSpec,
    SeparationStudyError,
    classify_decision,
    load_separation_spec,
)


@dataclass(frozen=True)
class SourcePaths:
    fixed_dir: Path
    adaptive_dir: Path
    fixed_features: dict[str, Path]
    adaptive_markets: dict[str, Path]
    source_lock: dict[str, Any]


def _registry_lock(root: Path, spec: SeparationSpec) -> None:
    records = []
    for line in (
        (root / "research/experiment_registry.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ):
        record = json.loads(line)
        if record.get("experiment_id") == spec.experiment_id:
            records.append(record)
    if (
        not records
        or records[-1].get("frozen_spec_hash") != spec.sha256
        or records[-1].get("status") not in {"FROZEN", "EXPERIMENT_COMPLETE"}
    ):
        raise SeparationStudyError("separation spec is not the latest registry lock")


def _inventory_files(run_dir: Path, expected_hash: str) -> dict[str, str]:
    inventory_path = run_dir / "inventory.json"
    if sha256_file(inventory_path) != expected_hash:
        raise SeparationStudyError(f"source inventory changed: {run_dir}")
    inventory = read_json(inventory_path)
    files = inventory.get("files")
    if not isinstance(files, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in files.items()
    ):
        raise SeparationStudyError(f"source inventory schema changed: {run_dir}")
    return files


def _verified_file(run_dir: Path, files: dict[str, str], relative: str) -> Path:
    path = run_dir / relative
    expected = files.get(relative)
    if not isinstance(expected, str) or sha256_file(path) != expected:
        raise SeparationStudyError(f"source file changed: {relative}")
    return path


def verify_source_inputs(
    root: Path, config: ResearchConfig, spec: SeparationSpec
) -> SourcePaths:
    """Verify only source metadata and allowlisted files; never open performance."""
    _registry_lock(root, spec)
    fixed_dir = root / config.artifact_root / "fixed-baselines" / spec.fixed_run_id
    adaptive_dir = (
        root / config.artifact_root / "adaptive-confidence-001" / spec.adaptive_run_id
    )
    fixed_meta = read_json(fixed_dir / "run.json")
    adaptive_meta = read_json(adaptive_dir / "run.json")
    if (
        fixed_meta.get("status") != "complete"
        or fixed_meta.get("run_id") != spec.fixed_run_id
        or fixed_meta.get("config_sha256") != config.sha256
        or adaptive_meta.get("status") != "complete"
        or adaptive_meta.get("run_id") != spec.adaptive_run_id
        or adaptive_meta.get("spec_sha256") != spec.adaptive_spec_sha256
        or adaptive_meta.get("data_manifest_sha256") != spec.data_manifest_sha256
    ):
        raise SeparationStudyError("source run identity changed")
    manifest_path = fixed_dir / "data-manifest.json"
    if sha256_file(manifest_path) != spec.data_manifest_sha256:
        raise SeparationStudyError("fixed parent data manifest changed")

    fixed_inventory = _inventory_files(fixed_dir, spec.fixed_inventory_sha256)
    adaptive_inventory = _inventory_files(adaptive_dir, spec.adaptive_inventory_sha256)
    fixed_features: dict[str, Path] = {}
    adaptive_markets: dict[str, Path] = {}
    allowed_hashes: dict[str, str] = {}
    for market in spec.markets:
        fixed_relative = f"{market}/features.csv"
        fixed_features[market] = _verified_file(
            fixed_dir, fixed_inventory, fixed_relative
        )
        allowed_hashes[f"fixed/{fixed_relative}"] = fixed_inventory[fixed_relative]
        adaptive_markets[market] = adaptive_dir / market
        for filename in spec.adaptive_allowed_files:
            relative = f"{market}/{filename}"
            _verified_file(adaptive_dir, adaptive_inventory, relative)
            allowed_hashes[f"adaptive/{relative}"] = adaptive_inventory[relative]
    return SourcePaths(
        fixed_dir=fixed_dir,
        adaptive_dir=adaptive_dir,
        fixed_features=fixed_features,
        adaptive_markets=adaptive_markets,
        source_lock={
            "schema_version": 1,
            "fixed_run_id": spec.fixed_run_id,
            "fixed_inventory_sha256": spec.fixed_inventory_sha256,
            "adaptive_run_id": spec.adaptive_run_id,
            "adaptive_inventory_sha256": spec.adaptive_inventory_sha256,
            "data_manifest_sha256": spec.data_manifest_sha256,
            "allowed_file_hashes": allowed_hashes,
            "performance_files_accessed": False,
            "post_2023_accessed": False,
        },
    )


def _implementation_lock(root: Path, spec: SeparationSpec) -> dict[str, Any]:
    paths = (
        spec.path,
        root / "research.toml",
        root / "pyproject.toml",
        root / "uv.lock",
        root / "src/adaptive_jump/tv_jump.py",
        root / "src/adaptive_jump/separation_study.py",
        root / "src/adaptive_jump/separation_analysis.py",
        root / "src/adaptive_jump/separation_evaluation.py",
        Path(__file__).resolve(),
    )
    files = {str(path.relative_to(root)): sha256_file(path) for path in paths}
    digest = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "schema_version": 1,
        "implementation_sha256": digest,
        "git_head": revision,
        "files": files,
    }


def _worker(
    market: str, feature_path: Path, adaptive_dir: Path, spec: SeparationSpec
) -> MarketAnalysis:
    with threadpool_limits(limits=1):
        return analyze_market(market, feature_path, adaptive_dir, spec)


def run_us_smoke(config: ResearchConfig, spec: SeparationSpec) -> dict[str, Any]:
    """Exercise all US source, refit, DP, ablation, and event paths first."""
    root = config.path.parent
    sources = verify_source_inputs(root, config, spec)
    result = _worker(
        "us",
        sources.fixed_features["us"],
        sources.adaptive_markets["us"],
        spec,
    )
    return {
        "status": "passed",
        "market": "us",
        "refit_lambda_rows": len(result.separation),
        "valid_separation_rows": int(result.separation["reliability_valid"].sum()),
        **result.audit,
    }


def run_separation_study(config: ResearchConfig, spec: SeparationSpec) -> Path:
    """Run all markets concurrently and write ignored auditable evidence."""
    root = config.path.parent
    sources = verify_source_inputs(root, config, spec)
    implementation = _implementation_lock(root, spec)
    run_id = (
        f"adaptive-separation-{spec.sha256[:12]}-"
        f"{spec.adaptive_inventory_sha256[:12]}-"
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
            verify_separation_run(run_dir)
            return run_dir

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "study.lock.toml").write_bytes(spec.path.read_bytes())
    (run_dir / "config.lock.toml").write_bytes(config.path.read_bytes())
    write_json(run_dir / "source-lock.json", sources.source_lock)
    write_json(run_dir / "implementation-lock.json", implementation)
    write_json(
        metadata_path,
        {
            "schema_version": 1,
            "study_kind": "adaptive_separation",
            "experiment_id": spec.experiment_id,
            "run_id": run_id,
            "status": "running",
            "claim_class": "EXPLORATORY",
            "performance_files_accessed": False,
            "post_2023_accessed": False,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "spec_sha256": spec.sha256,
            "config_sha256": config.sha256,
            "fixed_inventory_sha256": spec.fixed_inventory_sha256,
            "adaptive_inventory_sha256": spec.adaptive_inventory_sha256,
            "data_manifest_sha256": spec.data_manifest_sha256,
            "implementation_sha256": implementation["implementation_sha256"],
            "git_head": implementation["git_head"],
        },
    )

    results: dict[str, MarketAnalysis] = {}
    with ProcessPoolExecutor(
        max_workers=len(spec.markets), mp_context=get_context("forkserver")
    ) as executor:
        futures = {
            executor.submit(
                _worker,
                market,
                sources.fixed_features[market],
                sources.adaptive_markets[market],
                spec,
            ): market
            for market in spec.markets
        }
        for future in as_completed(futures):
            market = futures[future]
            results[market] = future.result()
            print(f"{market}: separation diagnostic complete", flush=True)

    for market in spec.markets:
        target = run_dir / market
        target.mkdir(exist_ok=True)
        results[market].separation.to_csv(target / "refit-separation.csv", index=False)
        results[market].events.to_csv(target / "discount-events.csv", index=False)
        write_json(target / "audit.json", results[market].audit)
    separation = pd.concat(
        [results[market].separation for market in spec.markets], ignore_index=True
    )
    events = pd.concat(
        [results[market].events for market in spec.markets], ignore_index=True
    )
    folds = evaluate_leave_one_market_out(events, spec)
    summary = summarize_mechanism(separation, events, spec)
    separation.to_csv(run_dir / "refit-separation.csv", index=False)
    events.to_csv(run_dir / "discount-events.csv", index=False)
    folds.to_csv(run_dir / "leave-one-market-out.csv", index=False)
    summary.to_csv(run_dir / "summary.csv", index=False)
    conclusion = build_conclusion(folds, summary, spec)
    write_json(run_dir / "conclusion.json", conclusion)
    write_inventory(run_dir)
    metadata = read_json(metadata_path)
    metadata.update(
        {
            "status": "complete",
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "result": conclusion["result"],
            "admitted_events": len(events),
            "valid_reliability_events": int(events["reliability_valid"].sum()),
        }
    )
    write_json(metadata_path, metadata)
    return run_dir


def verify_separation_run(run: str | Path) -> dict[str, Any]:
    """Verify locks, inventory, cutoff, fold decision, and conclusion."""
    run_dir = Path(run).resolve()
    verify_inventory(run_dir)
    metadata = read_json(run_dir / "run.json")
    spec = load_separation_spec(run_dir / "study.lock.toml")
    implementation = read_json(run_dir / "implementation-lock.json")
    source = read_json(run_dir / "source-lock.json")
    if (
        metadata.get("status") != "complete"
        or metadata.get("study_kind") != "adaptive_separation"
        or metadata.get("spec_sha256") != spec.sha256
        or metadata.get("implementation_sha256")
        != implementation.get("implementation_sha256")
        or source.get("performance_files_accessed") is not False
        or source.get("post_2023_accessed") is not False
    ):
        raise SeparationStudyError("separation run identity changed")
    separation = pd.read_csv(run_dir / "refit-separation.csv")
    events = pd.read_csv(run_dir / "discount-events.csv")
    separation_required = {
        "market",
        "fit_date",
        "training_end",
        "q_train",
        "q_train_reconstructed",
        "objective_abs_error",
        "reliability_valid",
        "reliability_train",
    }
    event_required = {
        "market",
        "beta",
        "lambda0",
        "signal_date",
        "fit_date",
        "source_state",
        "destination_state",
        "normalized_gap",
        "transition_penalty",
        "reverse_penalty",
        "log_discount",
        "terminal_predecessor",
        "ablated_state",
        "discount_attributable",
        "persistent_20",
        "whipsaw_20",
    }
    if not separation_required.issubset(separation) or not event_required.issubset(
        events
    ):
        raise SeparationStudyError("separation evidence table is incomplete")
    if (
        set(separation["market"]) != set(spec.markets)
        or (separation["objective_abs_error"] > spec.objective_tolerance).any()
        or not np.array_equal(
            separation["q_train"].to_numpy(),
            separation["q_train_reconstructed"].to_numpy(),
        )
    ):
        raise SeparationStudyError("stored refit reconstruction changed")
    valid_reliability = separation.loc[separation["reliability_valid"].astype(bool)]
    if not valid_reliability["reliability_train"].between(0.0, 1.0).all():
        raise SeparationStudyError("stored reliability left [0,1]")
    for column in ("signal_date", "fit_date"):
        dates = pd.to_datetime(events[column], errors="raise")
        if len(dates) and dates.max().date() > spec.data_cutoff:
            raise SeparationStudyError("separation output crossed cutoff")
    signal_dates = pd.to_datetime(events["signal_date"], errors="raise")
    for market, start in spec.evaluation_starts.items():
        market_dates = signal_dates.loc[events["market"] == market]
        if len(market_dates) and market_dates.min().date() < start:
            raise SeparationStudyError(f"{market}: event preceded outer sample")
    if len(events):
        expected_penalty = events["lambda0"] * np.exp(
            -events["beta"] * np.tanh(events["normalized_gap"])
        )
        expected_log_discount = np.log(events["lambda0"] / events["transition_penalty"])
        if (
            not np.allclose(
                events["transition_penalty"], expected_penalty, rtol=0, atol=1e-12
            )
            or not np.allclose(
                events["log_discount"], expected_log_discount, rtol=0, atol=1e-12
            )
            or not (events["transition_penalty"] < events["lambda0"]).all()
            or not np.array_equal(events["reverse_penalty"], events["lambda0"])
            or not np.array_equal(
                events["terminal_predecessor"], events["source_state"]
            )
            or not np.array_equal(events["ablated_state"], events["source_state"])
            or not (events["source_state"] != events["destination_state"]).all()
            or not events["discount_attributable"].astype(bool).all()
            or not (
                events["persistent_20"].astype(bool)
                != events["whipsaw_20"].astype(bool)
            ).all()
            or events.duplicated(["market", "beta", "lambda0", "signal_date"]).any()
        ):
            raise SeparationStudyError("stored discount-event invariant changed")
    folds = pd.read_csv(run_dir / "leave-one-market-out.csv")
    conclusion = read_json(run_dir / "conclusion.json")
    reconstructed = classify_decision(folds, spec.score_tolerance)
    if conclusion.get("result") != reconstructed:
        raise SeparationStudyError("separation conclusion changed")
    return {
        "status": "verified",
        "run_id": metadata["run_id"],
        "result": reconstructed,
        "events": len(events),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="adaptive-separation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--spec", default="research/adaptive-separation-001.toml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--verify")
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    if arguments.verify:
        print(json.dumps(verify_separation_run(arguments.verify), sort_keys=True))
        return 0
    config = load_config(arguments.config)
    spec_path = Path(arguments.spec)
    if not spec_path.is_absolute():
        spec_path = config.path.parent / spec_path
    spec = load_separation_spec(spec_path)
    if arguments.smoke:
        print(json.dumps(run_us_smoke(config, spec), sort_keys=True))
    else:
        print(run_separation_study(config, spec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
