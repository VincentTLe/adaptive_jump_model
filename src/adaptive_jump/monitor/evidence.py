"""Path-allowlisted, verifier-gated access to sealed research evidence."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from adaptive_jump import artifacts
from adaptive_jump.window_verifier import verify_window_run

Verifier = Callable[[str | Path], dict[str, Any]]


class EvidenceError(RuntimeError):
    """Raised when sealed evidence is unavailable, invalid, or unauthorized."""


class OutcomeLocked(EvidenceError):
    """Raised when a valid run has not opened conclusion-bearing outcomes."""


@dataclass(frozen=True)
class EvidenceDefinition:
    run_id: str
    title: str
    relative_path: Path
    verifier: Verifier


SEALED_RUNS = {
    definition.run_id: definition
    for definition in (
        EvidenceDefinition(
            "fixed-baselines-8adb330565d6-3636939b525d-e9614112b234",
            "Fixed baseline proxy replication",
            Path(
                "artifacts/fixed-baselines/"
                "fixed-baselines-8adb330565d6-3636939b525d-e9614112b234"
            ),
            artifacts.verify_run,
        ),
        EvidenceDefinition(
            "jm-window-cd9ac0b9d7a6-3636939b525d-6c19911401ad",
            "JM 4,000-day training-window sensitivity",
            Path(
                "artifacts/jm-train-window-sensitivity/"
                "jm-window-cd9ac0b9d7a6-3636939b525d-6c19911401ad"
            ),
            verify_window_run,
        ),
    )
}


class EvidenceStore:
    """Verify exact code-registered runs before exposing any artifact content."""

    def __init__(
        self,
        project_root: Path,
        definitions: Mapping[str, EvidenceDefinition] = SEALED_RUNS,
    ) -> None:
        self.project_root = project_root.resolve()
        self.artifact_root = self.project_root / "artifacts"
        self.definitions = dict(definitions)
        self._receipts: dict[str, tuple[str, dict[str, Any]]] = {}
        self._lock = threading.RLock()
        for run_id, definition in self.definitions.items():
            run_dir = (self.project_root / definition.relative_path).resolve()
            if (
                run_id != definition.run_id
                or run_dir.name != run_id
                or not run_dir.is_relative_to(self.artifact_root)
            ):
                raise EvidenceError(
                    "sealed evidence catalog violates its path allowlist"
                )

    def catalog(self) -> tuple[dict[str, Any], ...]:
        """List registered identities without reading unverified result content."""
        return tuple(
            {
                "run_id": definition.run_id,
                "title": definition.title,
                "available": self._run_dir(definition).is_dir(),
            }
            for definition in self.definitions.values()
        )

    def evidence(self, run_id: str) -> dict[str, Any]:
        """Return verified identity and boundary evidence, never outcome metrics."""
        definition, run_dir, receipt, seal = self._verify(run_id)
        metadata = artifacts.read_json(run_dir / "run.json")
        boundaries = _read_records(run_dir / "boundaries.csv")
        safe_receipt = {
            key: value for key, value in receipt.items() if key not in {"conclusion"}
        }
        result = {
            "run_id": definition.run_id,
            "title": definition.title,
            "status": metadata.get("status"),
            "metrics_opened": metadata.get("metrics_opened") is True,
            "claim_label": metadata.get("claim_label"),
            "verification": safe_receipt,
            "boundaries": boundaries,
        }
        self._require_unchanged(run_dir, seal)
        return result

    def outcome(self, run_id: str) -> dict[str, Any]:
        """Return verified outcomes only when the run explicitly opened them."""
        definition, run_dir, receipt, seal = self._verify(run_id)
        metadata = artifacts.read_json(run_dir / "run.json")
        if (
            metadata.get("metrics_opened") is not True
            or metadata.get("status") != "complete"
        ):
            raise OutcomeLocked(f"outcomes remain locked for {definition.run_id}")
        metrics_path = run_dir / "metrics.csv"
        claim_path = run_dir / "claim.json"
        if not metrics_path.is_file() or not claim_path.is_file():
            raise EvidenceError("opened outcome files are incomplete")
        result = {
            "run_id": definition.run_id,
            "title": definition.title,
            "verification": receipt,
            "metrics": _read_records(metrics_path),
            "claim": artifacts.read_json(claim_path),
        }
        self._require_unchanged(run_dir, seal)
        return result

    def _verify(
        self, run_id: str
    ) -> tuple[EvidenceDefinition, Path, dict[str, Any], str]:
        definition = self._definition(run_id)
        run_dir = self._run_dir(definition)
        inventory = run_dir / "inventory.json"
        metadata_path = run_dir / "run.json"
        if (
            not run_dir.is_dir()
            or not inventory.is_file()
            or not metadata_path.is_file()
        ):
            raise EvidenceError(f"sealed evidence is unavailable: {run_id}")
        with self._lock:
            try:
                seal = self._seal_identity(run_dir)
                metadata = artifacts.read_json(metadata_path)
            except (artifacts.ArtifactError, OSError, ValueError) as exc:
                raise EvidenceError(
                    f"sealed evidence verification failed: {run_id}"
                ) from exc
            cached = self._receipts.get(run_id)
            if cached is not None and cached[0] == seal:
                return definition, run_dir, cached[1], seal
            try:
                receipt = definition.verifier(run_dir)
            except (artifacts.ArtifactError, OSError, ValueError) as exc:
                raise EvidenceError(
                    f"sealed evidence verification failed: {run_id}"
                ) from exc
            if (
                metadata.get("run_id") != run_id
                or receipt.get("run_id") != run_id
                or receipt.get("status") != metadata.get("status")
            ):
                raise EvidenceError("verifier returned a different run identity")
            safe_receipt = json.loads(json.dumps(receipt, allow_nan=False))
            self._receipts[run_id] = (seal, safe_receipt)
            return definition, run_dir, safe_receipt, seal

    def _definition(self, run_id: str) -> EvidenceDefinition:
        try:
            return self.definitions[run_id]
        except (KeyError, TypeError) as exc:
            raise EvidenceError(f"run is not registered: {run_id}") from exc

    def _run_dir(self, definition: EvidenceDefinition) -> Path:
        return (self.project_root / definition.relative_path).resolve()

    @staticmethod
    def _seal_identity(run_dir: Path) -> str:
        artifacts.verify_inventory(run_dir)
        return ":".join(
            artifacts.sha256_file(run_dir / name)
            for name in ("inventory.json", "run.json")
        )

    def _require_unchanged(self, run_dir: Path, expected: str) -> None:
        try:
            actual = self._seal_identity(run_dir)
        except (artifacts.ArtifactError, OSError) as exc:
            raise EvidenceError("sealed evidence changed while being read") from exc
        if actual != expected:
            raise EvidenceError("sealed evidence changed while being read")


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise EvidenceError(f"verified evidence file is missing: {path.name}")
    frame = pd.read_csv(path)
    return json.loads(frame.to_json(orient="records", date_format="iso"))
