"""Integrity helpers for immutable research-run artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


class ArtifactError(RuntimeError):
    """Raised when frozen study inputs or run artifacts are invalid."""


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object with a stable research-facing error."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        raise ArtifactError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ArtifactError(f"JSON must contain an object: {path}")
    return document


def sha256_file(path: Path) -> str:
    """Hash a file without trusting stored metadata."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, document: dict[str, Any]) -> None:
    """Write canonical human-readable JSON without NaN values."""
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _inventory_files(run_dir: Path) -> dict[str, str]:
    return {
        str(path.relative_to(run_dir)): sha256_file(path)
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name not in {"inventory.json", "run.json"}
    }


def write_inventory(run_dir: Path) -> None:
    """Seal every immutable run file; mutable lifecycle metadata is separate."""
    write_json(
        run_dir / "inventory.json",
        {"schema_version": 1, "files": _inventory_files(run_dir)},
    )


def verify_inventory(run_dir: Path) -> None:
    """Reject missing, extra, or modified immutable run files."""
    inventory = read_json(run_dir / "inventory.json")
    expected = inventory.get("files")
    if not isinstance(expected, dict):
        raise ArtifactError("invalid artifact inventory")
    if _inventory_files(run_dir) != expected:
        raise ArtifactError("artifact inventory mismatch")


def directional_gate(metrics: pd.DataFrame, primary_delay: int) -> dict[str, Any]:
    """Evaluate the frozen three-condition replication gate."""
    rows = []
    primary = metrics.loc[metrics["delay"] == primary_delay]
    for market, values in primary.groupby("market"):
        indexed = values.set_index("model")
        required = {"fixed_jm", "hmm", "buy_and_hold"}
        if set(indexed.index) != required:
            raise ArtifactError(f"{market}: incomplete primary metrics")
        jm = indexed.loc["fixed_jm"]
        hmm = indexed.loc["hmm"]
        hold = indexed.loc["buy_and_hold"]
        checks = {
            "sharpe_above_hmm": bool(jm["sharpe"] > hmm["sharpe"]),
            "sharpe_above_buy_and_hold": bool(jm["sharpe"] > hold["sharpe"]),
            "mdd_below_buy_and_hold": bool(
                abs(jm["maximum_drawdown"]) < abs(hold["maximum_drawdown"])
            ),
        }
        rows.append({"market": market, **checks, "passed": all(checks.values())})
    passed = len(rows) == 3 and all(row["passed"] for row in rows)
    return {
        "claim_label": "proxy replication",
        "primary_delay": primary_delay,
        "markets": rows,
        "passed": passed,
        "conclusion": (
            "directional proxy replication"
            if passed
            else "non-replication; adaptive work remains blocked"
        ),
    }
