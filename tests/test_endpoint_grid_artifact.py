from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from test_endpoint_grid_audit import _synthetic_artifact

import adaptive_jump.endpoint_grid_artifact_verifier as verifier
from adaptive_jump.artifacts import (
    ArtifactError,
    read_json,
    verify_run,
    write_inventory,
    write_json,
)


@pytest.mark.parametrize(
    "relative",
    (
        "claim.json",
        "cell-metrics.csv",
        "us/A-delay-1/choices.csv",
        "us/J0-delay-1/extra.csv",
    ),
)
def test_recursive_allowlist_rejects_extra_files_after_reinventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative: str,
) -> None:
    run_dir, _ = _synthetic_artifact(tmp_path, monkeypatch)
    path = run_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("unexpected\n", encoding="utf-8")
    write_inventory(run_dir)

    with pytest.raises(ArtifactError, match="allowlist"):
        verifier.verify_endpoint_grid_run(run_dir)


def test_recursive_allowlist_rejects_empty_directory_and_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _ = _synthetic_artifact(tmp_path, monkeypatch)
    (run_dir / "unexpected-empty").mkdir()
    with pytest.raises(ArtifactError, match="allowlist"):
        verifier.verify_endpoint_grid_run(run_dir)

    (run_dir / "unexpected-empty").rmdir()
    (run_dir / "unexpected-link").symlink_to(run_dir / "metrics.csv")
    with pytest.raises(ArtifactError, match="symlink"):
        verifier.verify_endpoint_grid_run(run_dir)


def test_inventory_json_extra_claim_key_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _ = _synthetic_artifact(tmp_path, monkeypatch)
    inventory = read_json(run_dir / "inventory.json")
    inventory["conclusion"] = "not allowed"
    write_json(run_dir / "inventory.json", inventory)

    with pytest.raises(ArtifactError, match="inventory schema"):
        verifier.verify_endpoint_grid_run(run_dir)


def test_run_json_extra_claim_key_is_rejected_even_though_inventory_ignores_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _ = _synthetic_artifact(tmp_path, monkeypatch)
    metadata = read_json(run_dir / "run.json")
    metadata["conclusion"] = "not allowed"
    write_json(run_dir / "run.json", metadata)

    with pytest.raises(ArtifactError, match="metadata keys"):
        verifier.verify_endpoint_grid_run(run_dir)


def test_stored_git_sha_is_bound_to_actual_repository_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, git_sha = _synthetic_artifact(tmp_path, monkeypatch)
    monkeypatch.setattr(verifier, "research_git_sha", lambda _root: "8" * 40)

    with pytest.raises(ArtifactError, match="not current repository SHA"):
        verifier.verify_endpoint_grid_run(run_dir)

    monkeypatch.setattr(verifier, "research_git_sha", lambda _root: git_sha)
    assert verifier.verify_endpoint_grid_run(run_dir)["status"] == "complete"


@pytest.mark.parametrize(
    ("kind", "message"),
    (
        ("behavior", "parity receipt changed"),
        ("decision", "decision does not match"),
        ("path_changes", "path changes do not match"),
        ("trace", "change traces do not match"),
    ),
)
def test_independent_verifier_rejects_decision_evidence_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    message: str,
) -> None:
    run_dir, _ = _synthetic_artifact(tmp_path, monkeypatch)
    if kind == "behavior":
        path = run_dir / "behavior-control.json"
        payload = read_json(path)
        payload["all_markets_passed"] = False
        write_json(path, payload)
    elif kind == "decision":
        path = run_dir / "decision.json"
        payload = read_json(path)
        payload["all_markets_passed"] = not payload["all_markets_passed"]
        write_json(path, payload)
    elif kind == "path_changes":
        path = run_dir / "path-changes.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "binding"] = not bool(frame.loc[0, "binding"])
        frame.to_csv(path, index=False)
    else:
        path = run_dir / "change-traces.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "signal_change_date"] = "1999-01-01"
        frame.to_csv(path, index=False)
    write_inventory(run_dir)

    with pytest.raises(ArtifactError, match=message):
        verifier.verify_endpoint_grid_run(run_dir)


def test_artifact_dispatches_endpoint_grid_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_json(run_dir / "run.json", {"study_kind": "endpoint_grid_audit"})
    monkeypatch.setattr(
        verifier,
        "verify_endpoint_grid_run",
        lambda path: {"status": "complete", "path": str(path)},
    )
    assert verify_run(run_dir)["status"] == "complete"
