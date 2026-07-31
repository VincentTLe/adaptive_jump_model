"""Allowlisted source and implementation locks for the balanced study."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adaptive_jump.artifacts import read_json, sha256_file
from adaptive_jump.balanced_model import BalancedSpec, BalancedStudyError
from adaptive_jump.config import ResearchConfig
from adaptive_jump.models import FEATURE_COLUMNS


@dataclass(frozen=True)
class SourcePaths:
    fixed_dir: Path
    parent_dir: Path
    fixed_markets: dict[str, Path]
    parent_markets: dict[str, Path]
    source_lock: dict[str, Any]


def _registry_lock(root: Path, spec: BalancedSpec) -> None:
    rows = []
    lines = (root / "research/experiment_registry.jsonl").read_text(encoding="utf-8")
    for line in lines.splitlines():
        row = json.loads(line)
        if row.get("experiment_id") == spec.experiment_id:
            rows.append(row)
    if (
        not rows
        or rows[-1].get("frozen_spec_hash") != spec.sha256
        or rows[-1].get("status") not in {"FROZEN", "EXPERIMENT_COMPLETE"}
    ):
        raise BalancedStudyError("balanced spec is not the latest registry lock")


def _inventory_files(run_dir: Path, expected_hash: str) -> dict[str, str]:
    inventory_path = run_dir / "inventory.json"
    if sha256_file(inventory_path) != expected_hash:
        raise BalancedStudyError(f"source inventory changed: {run_dir}")
    files = read_json(inventory_path).get("files")
    if not isinstance(files, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in files.items()
    ):
        raise BalancedStudyError("source inventory schema changed")
    return files


def _verified(run_dir: Path, files: dict[str, str], relative: str) -> Path:
    path = run_dir / relative
    if files.get(relative) != sha256_file(path):
        raise BalancedStudyError(f"source file changed: {relative}")
    return path


def verify_source_inputs(
    root: Path, config: ResearchConfig, spec: BalancedSpec
) -> SourcePaths:
    """Verify source identities while opening only performance-free files."""
    _registry_lock(root, spec)
    fixed_dir = root / config.artifact_root / "fixed-baselines" / spec.fixed_run_id
    parent_dir = (
        root
        / config.artifact_root
        / "lagged-evidence-mechanism-001"
        / spec.parent_run_id
    )
    if sha256_file(fixed_dir / "data-manifest.json") != spec.data_manifest_sha256:
        raise BalancedStudyError("fixed data manifest changed")
    fixed_files = _inventory_files(fixed_dir, spec.fixed_inventory_sha256)
    parent_files = _inventory_files(parent_dir, spec.parent_inventory_sha256)
    parent_lock = _verified(parent_dir, parent_files, "study.lock.toml")
    if sha256_file(parent_lock) != spec.parent_spec_sha256:
        raise BalancedStudyError("parent lagged spec changed")
    parent_meta = read_json(parent_dir / "run.json")
    if (
        parent_meta.get("experiment_id") != "lagged-evidence-mechanism-001"
        or parent_meta.get("run_id") != spec.parent_run_id
        or parent_meta.get("status") != "complete"
        or parent_meta.get("spec_sha256") != spec.parent_spec_sha256
        or parent_meta.get("result") != "supported"
        or parent_meta.get("selected_beta_label") != "log4"
    ):
        raise BalancedStudyError("parent lagged metadata changed")

    fixed_markets: dict[str, Path] = {}
    parent_markets: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for market in spec.markets:
        fixed_markets[market] = fixed_dir / market
        parent_markets[market] = parent_dir / market
        for filename in spec.fixed_allowed_files:
            relative = f"{market}/{filename}"
            _verified(fixed_dir, fixed_files, relative)
            hashes[f"fixed/{relative}"] = fixed_files[relative]
        for filename in spec.parent_allowed_files:
            relative = f"{market}/{filename}"
            _verified(parent_dir, parent_files, relative)
            hashes[f"parent/{relative}"] = parent_files[relative]
    if set(spec.parent_allowed_files) & set(spec.forbidden_files):
        raise BalancedStudyError("forbidden parent file entered allowlist")
    return SourcePaths(
        fixed_dir=fixed_dir,
        parent_dir=parent_dir,
        fixed_markets=fixed_markets,
        parent_markets=parent_markets,
        source_lock={
            "schema_version": 1,
            "fixed_run_id": spec.fixed_run_id,
            "fixed_inventory_sha256": spec.fixed_inventory_sha256,
            "parent_run_id": spec.parent_run_id,
            "parent_inventory_sha256": spec.parent_inventory_sha256,
            "parent_spec_sha256": spec.parent_spec_sha256,
            "data_manifest_sha256": spec.data_manifest_sha256,
            "allowed_file_hashes": hashes,
            "columns_read": {
                "fixed/features.csv": ["date", *FEATURE_COLUMNS],
                "fixed/jm-states.csv": ["date", *[str(x) for x in spec.lambdas]],
                "parent/candidate-states": [
                    "date",
                    *[str(x) for x in spec.lambdas],
                ],
                "parent/refits-and-scales.csv": [
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
            "return_columns_accessed": False,
            "post_2023_accessed": False,
        },
    )


def implementation_lock(root: Path, spec: BalancedSpec) -> dict[str, Any]:
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
        root / "src/adaptive_jump/separation_analysis.py",
        root / "src/adaptive_jump/separation_study.py",
        root / "src/adaptive_jump/balanced_model.py",
        root / "src/adaptive_jump/balanced_sources.py",
        root / "src/adaptive_jump/balanced_mechanics.py",
        root / "src/adaptive_jump/balanced_analysis.py",
        root / "src/adaptive_jump/balanced_events.py",
        root / "src/adaptive_jump/balanced_smoke.py",
        root / "src/adaptive_jump/balanced_runner.py",
        root / "src/adaptive_jump/balanced_verifier.py",
        root / "src/adaptive_jump/balanced_replay.py",
        root / "src/adaptive_jump/balanced_event_replay.py",
        root / "src/adaptive_jump/balanced_decision_replay.py",
        root / "src/adaptive_jump/balanced_smoke_replay.py",
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
