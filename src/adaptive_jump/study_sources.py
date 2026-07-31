"""Source-run identity checks shared by the rerun harnesses.

The restored studies used to pin their upstream runs with literals -- the
``fixed-baselines-001-v7`` experiment id, a v7 data-manifest digest, and the
``adaptive-confidence-001`` artifact directory name. A rerun against a different
sealed baseline then required editing the implementation, which is exactly the
thing a frozen spec is supposed to make impossible.

Here the spec names its sources and this module checks that the run on disk
agrees. ``run.json`` is the authority: a spec that names an experiment id the
run does not carry is rejected before any evidence is read.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adaptive_jump.artifacts import read_json


@dataclass(frozen=True)
class SourceReference:
    """One upstream run named by a frozen spec."""

    experiment_id: str
    artifact_subdir: Path
    run_id: str
    inventory_sha256: str

    def directory(self, root: Path, artifact_root: Path) -> Path:
        return root / artifact_root / self.artifact_subdir / self.run_id


def read_source_reference(
    table: Any,
    *,
    error: type[Exception],
    label: str,
) -> SourceReference:
    """Parse ``experiment_id`` / ``artifact_subdir`` / ``run_id`` from a spec table."""
    if not isinstance(table, dict):
        raise error(f"{label} source table is missing")
    experiment_id = table.get("experiment_id")
    artifact_subdir = table.get("artifact_subdir")
    run_id = table.get("run_id")
    inventory = table.get("run_inventory_sha256")
    if not all(
        isinstance(value, str) and value
        for value in (experiment_id, artifact_subdir, run_id, inventory)
    ):
        raise error(f"{label} source must name experiment_id, artifact_subdir, run_id")
    subdir = Path(str(artifact_subdir))
    if subdir.is_absolute() or not subdir.parts or ".." in subdir.parts:
        raise error(f"{label} artifact_subdir must stay inside the artifact root")
    return SourceReference(
        experiment_id=str(experiment_id),
        artifact_subdir=subdir,
        run_id=str(run_id),
        inventory_sha256=str(inventory),
    )


def verify_source_identity(
    run_dir: Path,
    reference: SourceReference,
    *,
    error: type[Exception],
    label: str,
    config_sha256: str | None = None,
) -> dict[str, Any]:
    """The run on disk must be the complete run the spec names."""
    metadata_path = run_dir / "run.json"
    if not metadata_path.is_file():
        raise error(f"{label} source run is missing: {run_dir}")
    metadata = read_json(metadata_path)
    if metadata.get("experiment_id") != reference.experiment_id:
        raise error(
            f"{label} source experiment_id is not {reference.experiment_id}: {run_dir}"
        )
    if metadata.get("run_id") != reference.run_id:
        raise error(f"{label} source run_id changed: {run_dir}")
    if metadata.get("status") != "complete":
        raise error(f"{label} source run is not complete: {run_dir}")
    if config_sha256 is not None and metadata.get("config_sha256") != config_sha256:
        raise error(f"{label} source was produced under another contract: {run_dir}")
    return metadata
