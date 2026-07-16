from __future__ import annotations

import json
from pathlib import Path

import pytest

from adaptive_jump.config import load_config
from adaptive_jump.grid_spec import GridSpecError, load_grid_spec

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "research/persistence-grid-evaluation.toml"
REGISTRY = ROOT / "research/experiment_registry.jsonl"


def test_frozen_grid_spec_matches_registry() -> None:
    config = load_config(ROOT / "research.toml")
    spec = load_grid_spec(SPEC, config)
    latest = [
        json.loads(line)
        for line in REGISTRY.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["experiment_id"] == spec.experiment_id
    ][-1]

    assert spec.sha256 == latest["frozen_spec_hash"]
    assert latest["status"] == "FROZEN"
    assert spec.jm_grid == (
        0.0,
        0.3535533905932738,
        1.0,
        5.656854249492381,
        16.0,
        32.0,
        64.0,
        181.01933598375618,
        256.0,
    )
    assert spec.hmm_grid == (0, 3, 9, 32, 54, 114, 166, 402, 1115)


def test_grid_spec_rejects_provider_access(tmp_path: Path) -> None:
    changed = tmp_path / "study.toml"
    changed.write_text(
        SPEC.read_text(encoding="utf-8").replace(
            "provider_access = false", "provider_access = true"
        ),
        encoding="utf-8",
    )
    with pytest.raises(GridSpecError, match="provider access"):
        load_grid_spec(changed, load_config(ROOT / "research.toml"))
