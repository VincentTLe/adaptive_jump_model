"""Identity checks on the source runs a -002 study is allowed to read."""

import json
from pathlib import Path

import pytest

from adaptive_jump.study_sources import SourceReference, verify_source_identity


def _reference() -> SourceReference:
    return SourceReference(
        experiment_id="baseline-reseal-v10",
        artifact_subdir=Path("fixed-baselines"),
        run_id="fixed-baselines-abc",
        inventory_sha256="d" * 64,
    )


def _run(tmp_path: Path, **overrides: object) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    document: dict[str, object] = {
        "run_id": "fixed-baselines-abc",
        "status": "complete",
        "config_sha256": "c" * 64,
    }
    document.update(overrides)
    (run / "run.json").write_text(json.dumps(document), encoding="utf-8")
    return run


def test_identity_accepts_replication_run_without_experiment_id(
    tmp_path: Path,
) -> None:
    """A fixed-baselines run identifies itself by run id and contract hash."""
    run = _run(tmp_path)

    metadata = verify_source_identity(
        run, _reference(), error=ValueError, label="fixed", config_sha256="c" * 64
    )

    assert metadata["run_id"] == "fixed-baselines-abc"


def test_identity_still_rejects_a_foreign_contract(tmp_path: Path) -> None:
    run = _run(tmp_path)

    with pytest.raises(ValueError, match="another contract"):
        verify_source_identity(
            run, _reference(), error=ValueError, label="fixed", config_sha256="e" * 64
        )


def test_identity_refuses_when_neither_anchor_is_available(tmp_path: Path) -> None:
    """No experiment id and no contract hash leaves only the run id: refuse."""
    run = _run(tmp_path)

    with pytest.raises(ValueError, match="no experiment_id"):
        verify_source_identity(run, _reference(), error=ValueError, label="fixed")


def test_identity_still_enforces_a_present_experiment_id(tmp_path: Path) -> None:
    run = _run(tmp_path, experiment_id="some-other-study")

    with pytest.raises(ValueError, match="experiment_id is not"):
        verify_source_identity(
            run, _reference(), error=ValueError, label="fixed", config_sha256="c" * 64
        )


def test_identity_requires_a_complete_run(tmp_path: Path) -> None:
    run = _run(tmp_path, status="running")

    with pytest.raises(ValueError, match="not complete"):
        verify_source_identity(
            run, _reference(), error=ValueError, label="fixed", config_sha256="c" * 64
        )
