"""Every frozen spec that is run must be the spec the registry froze.

Adversarial verification on 2026-08-01 found that four restored test modules
had each dropped their registry-hash assertion, and that for the arrival
family no such check existed anywhere else -- so nothing in the repository
verified "the spec I am running is the one that was preregistered". The
defect survived a self-review that counted assertions instead of reading
them: two files kept or raised their count while swapping the check out.

These tests bind the property at two levels: repository state (every -002
spec on disk matches its latest registry row) and per-family behaviour (each
loader rejects a spec whose bytes changed).
"""

import json
from pathlib import Path

import pytest

from adaptive_jump.study_sources import registry_lock

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "research/experiment_registry.jsonl"
# Every study contract this repository can execute today. A new -002 spec that
# is runnable but missing here is itself the bug this list exists to catch.
RUNNABLE_SPECS = (
    "simple-jm-suite-002",
    "dd-loss-scale-002",
    "adaptive-confidence-002",
    "holdout-2026-001",
    "separation-turnover-001",
)


def _rows(experiment_id: str) -> list[dict]:
    rows = []
    for line in REGISTRY.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("experiment_id") == experiment_id:
                rows.append(record)
    return rows


@pytest.mark.parametrize("experiment_id", RUNNABLE_SPECS)
def test_runnable_spec_matches_its_latest_registry_row(experiment_id: str) -> None:
    spec = ROOT / "research" / f"{experiment_id}.toml"
    assert spec.is_file(), f"{experiment_id}: spec file is missing"
    rows = _rows(experiment_id)
    assert rows, f"{experiment_id}: no registry row"
    latest = rows[-1]
    assert latest["status"] in {"FROZEN", "EXPERIMENT_COMPLETE"}

    import hashlib

    digest = hashlib.sha256(spec.read_bytes()).hexdigest()
    assert latest["frozen_spec_hash"] == digest, (
        f"{experiment_id}: spec bytes differ from the frozen registry hash"
    )


def test_every_runnable_spec_is_listed() -> None:
    """A contract the code can execute but this list forgets is a hole.

    Runnable means: some module names it, either as a spec filename or as the
    experiment-id constant a runner turns into one. Probe specs that no runner
    loads (the jm-* forensic series) are deliberately out of scope.
    """
    import re

    named: set[str] = set()
    for source in (ROOT / "src/adaptive_jump").glob("*.py"):
        text = source.read_text(encoding="utf-8")
        named |= set(re.findall(r'"([a-z0-9-]+-\d{3})\.toml"', text))
        named |= {
            value
            for value in re.findall(r'EXPERIMENT_ID = "([a-z0-9-]+-\d{3})"', text)
        }
    runnable = {
        name
        for name in named
        if (ROOT / "research" / f"{name}.toml").is_file()
    }

    assert runnable == set(RUNNABLE_SPECS), (
        "the specs the code can run and RUNNABLE_SPECS disagree; add the new "
        "spec (and its registry row) to the list"
    )


def _registry(tmp_path: Path, experiment_id: str, digest: str, status: str) -> Path:
    root = tmp_path / "repo"
    (root / "research").mkdir(parents=True)
    (root / "research/experiment_registry.jsonl").write_text(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "status": status,
                "frozen_spec_hash": digest,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def test_registry_lock_accepts_the_frozen_hash(tmp_path: Path) -> None:
    root = _registry(tmp_path, "some-study-002", "a" * 64, "FROZEN")

    registry_lock(root, "some-study-002", "a" * 64, error=ValueError)


def test_registry_lock_rejects_changed_bytes(tmp_path: Path) -> None:
    root = _registry(tmp_path, "some-study-002", "a" * 64, "FROZEN")

    with pytest.raises(ValueError, match="latest registry lock"):
        registry_lock(root, "some-study-002", "b" * 64, error=ValueError)


def test_registry_lock_rejects_an_unregistered_study(tmp_path: Path) -> None:
    root = _registry(tmp_path, "some-study-002", "a" * 64, "FROZEN")

    with pytest.raises(ValueError, match="latest registry lock"):
        registry_lock(root, "other-study-002", "a" * 64, error=ValueError)


def test_registry_lock_rejects_a_non_frozen_status(tmp_path: Path) -> None:
    root = _registry(tmp_path, "some-study-002", "a" * 64, "NOTE")

    with pytest.raises(ValueError, match="latest registry lock"):
        registry_lock(root, "some-study-002", "a" * 64, error=ValueError)


def test_registry_lock_reads_the_latest_row_not_the_first(tmp_path: Path) -> None:
    """An amended spec must be rejected until its new hash is registered."""
    root = _registry(tmp_path, "some-study-002", "a" * 64, "FROZEN")
    with (root / "research/experiment_registry.jsonl").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(
            json.dumps(
                {
                    "experiment_id": "some-study-002",
                    "status": "FROZEN",
                    "frozen_spec_hash": "c" * 64,
                }
            )
            + "\n"
        )

    registry_lock(root, "some-study-002", "c" * 64, error=ValueError)
    with pytest.raises(ValueError, match="latest registry lock"):
        registry_lock(root, "some-study-002", "a" * 64, error=ValueError)
