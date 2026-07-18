"""Allowlisted source and implementation locks for the lagged study."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adaptive_jump.artifacts import read_json, sha256_file
from adaptive_jump.config import ResearchConfig
from adaptive_jump.lagged_study import (
    LaggedMechanismSpec,
    LaggedStudyError,
)


@dataclass(frozen=True)
class SourcePaths:
    fixed_dir: Path
    arrival_dir: Path
    fixed_markets: dict[str, Path]
    arrival_markets: dict[str, Path]
    source_lock: dict[str, Any]


def _registry_lock(root: Path, spec: LaggedMechanismSpec) -> None:
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
        raise LaggedStudyError("lagged spec is not the latest registry lock")


def _inventory_files(run_dir: Path, expected_hash: str) -> dict[str, str]:
    inventory_path = run_dir / "inventory.json"
    if sha256_file(inventory_path) != expected_hash:
        raise LaggedStudyError(f"source inventory changed: {run_dir}")
    files = read_json(inventory_path).get("files")
    if not isinstance(files, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in files.items()
    ):
        raise LaggedStudyError(f"source inventory schema changed: {run_dir}")
    return files


def _verified_file(run_dir: Path, files: dict[str, str], relative: str) -> Path:
    path = run_dir / relative
    expected = files.get(relative)
    if not isinstance(expected, str) or sha256_file(path) != expected:
        raise LaggedStudyError(f"source file changed: {relative}")
    return path


def verify_source_inputs(
    root: Path,
    config: ResearchConfig,
    spec: LaggedMechanismSpec,
) -> SourcePaths:
    """Verify metadata and allowlisted state inputs without opening performance."""
    _registry_lock(root, spec)
    fixed_dir = root / config.artifact_root / "fixed-baselines" / spec.fixed_run_id
    arrival_dir = (
        root / config.artifact_root / "adaptive-confidence-001" / spec.arrival_run_id
    )
    if sha256_file(fixed_dir / "data-manifest.json") != spec.data_manifest_sha256:
        raise LaggedStudyError("lagged source identity changed")

    fixed_inventory = _inventory_files(fixed_dir, spec.fixed_inventory_sha256)
    arrival_inventory = _inventory_files(arrival_dir, spec.arrival_inventory_sha256)
    fixed_markets: dict[str, Path] = {}
    arrival_markets: dict[str, Path] = {}
    allowed_hashes: dict[str, str] = {}
    for market in spec.markets:
        fixed_markets[market] = fixed_dir / market
        arrival_markets[market] = arrival_dir / market
        for filename in spec.fixed_allowed_files:
            relative = f"{market}/{filename}"
            _verified_file(fixed_dir, fixed_inventory, relative)
            allowed_hashes[f"fixed/{relative}"] = fixed_inventory[relative]
        for filename in spec.arrival_allowed_files:
            relative = f"{market}/{filename}"
            _verified_file(arrival_dir, arrival_inventory, relative)
            allowed_hashes[f"arrival/{relative}"] = arrival_inventory[relative]

    if set(spec.arrival_allowed_files) & set(spec.performance_files_forbidden):
        raise LaggedStudyError("performance file entered the arrival allowlist")
    return SourcePaths(
        fixed_dir=fixed_dir,
        arrival_dir=arrival_dir,
        fixed_markets=fixed_markets,
        arrival_markets=arrival_markets,
        source_lock={
            "schema_version": 1,
            "fixed_run_id": spec.fixed_run_id,
            "fixed_inventory_sha256": spec.fixed_inventory_sha256,
            "arrival_run_id": spec.arrival_run_id,
            "arrival_inventory_sha256": spec.arrival_inventory_sha256,
            "data_manifest_sha256": spec.data_manifest_sha256,
            "allowed_file_hashes": allowed_hashes,
            "columns_read": {
                "fixed/features.csv": ["date", "dd_10", "sortino_20", "sortino_60"],
                "fixed/jm-states.csv": [
                    "date",
                    *[str(value) for value in spec.lambdas],
                ],
                "arrival/candidate-states": [
                    "date",
                    *[str(value) for value in spec.lambdas],
                ],
                "arrival/refits-and-scales.csv": [
                    "market",
                    "fit_date",
                    "training_start",
                    "training_end",
                    "lambda0",
                    "q_train",
                    "scaler_mean",
                    "scaler_scale",
                    "centers",
                ],
            },
            "performance_files_accessed": False,
            "post_2023_accessed": False,
        },
    )


def implementation_lock(root: Path, spec: LaggedMechanismSpec) -> dict[str, Any]:
    paths = (
        spec.path,
        root / "research.toml",
        root / "pyproject.toml",
        root / "uv.lock",
        root / "src/adaptive_jump/artifacts.py",
        root / "src/adaptive_jump/config.py",
        root / "src/adaptive_jump/models.py",
        root / "src/adaptive_jump/tv_jump.py",
        root / "src/adaptive_jump/lagged_model.py",
        root / "src/adaptive_jump/lagged_mechanics.py",
        root / "src/adaptive_jump/lagged_study.py",
        root / "src/adaptive_jump/lagged_analysis.py",
        root / "src/adaptive_jump/lagged_sources.py",
        root / "src/adaptive_jump/lagged_runner.py",
        root / "src/adaptive_jump/lagged_verifier.py",
        root / "src/adaptive_jump/separation_analysis.py",
        root / "src/adaptive_jump/separation_study.py",
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
