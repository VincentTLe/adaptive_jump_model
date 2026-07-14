import hashlib
from pathlib import Path

import pytest

from adaptive_jump.config import load_config
from adaptive_jump.window_spec import WindowSpecError, load_window_spec

ROOT = Path(__file__).resolve().parents[1]


def _mutable_contract(tmp_path: Path) -> tuple[Path, Path]:
    config_path = tmp_path / "research.toml"
    config_path.write_bytes((ROOT / "research.toml").read_bytes())
    spec_path = tmp_path / "research/jm-train-window-sensitivity.toml"
    spec_path.parent.mkdir()
    spec_path.write_bytes(
        (ROOT / "research/jm-train-window-sensitivity.toml").read_bytes()
    )
    return config_path, spec_path


def test_window_spec_binds_only_longer_jm_window_to_exact_v7() -> None:
    config = load_config(ROOT / "research.toml")
    path = ROOT / "research/jm-train-window-sensitivity.toml"

    spec = load_window_spec(path, config)

    assert spec.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert spec.baseline_window == 3000
    assert spec.challenger_window == 4000
    assert spec.models == ("buy_and_hold", "hmm_3000", "jm_3000", "jm_4000")
    assert spec.delays == (1, 5, 10)
    assert spec.bootstrap_blocks == (60, 20, 120)
    assert spec.data_cutoff.isoformat() == "2023-12-31"


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("extension_access = false", "extension_access = true", "extension access"),
        ("challenger_observations = 4000", "challenger_observations = 3000", "longer"),
        (
            "validation_calendar_years = 8",
            "validation_calendar_years = 7",
            "validation",
        ),
        ("hmm_observations = 3000", "hmm_observations = 4000", "HMM window"),
        ("primary_delay = 1", "primary_delay = 2", "primary delay"),
        (
            "grid_expansion_after_results = false",
            "grid_expansion_after_results = true",
            "grid expansion",
        ),
    ],
)
def test_window_spec_rejects_result_affecting_protocol_changes(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    config_path, spec_path = _mutable_contract(tmp_path)
    spec_path.write_text(
        spec_path.read_text(encoding="utf-8").replace(old, new), encoding="utf-8"
    )

    with pytest.raises(WindowSpecError, match=message):
        load_window_spec(spec_path, load_config(config_path))


def test_window_spec_rejects_parent_hash_mismatch(tmp_path: Path) -> None:
    config_path, spec_path = _mutable_contract(tmp_path)
    payload = spec_path.read_text(encoding="utf-8")
    spec_path.write_text(
        payload.replace(
            "8adb330565d64f8ed6edd986f0422dbba72585eda4efd34b0c1b41b95450d81b",
            "0" * 64,
        ),
        encoding="utf-8",
    )

    with pytest.raises(WindowSpecError, match="config hash"):
        load_window_spec(spec_path, load_config(config_path))


def test_window_spec_rejects_unsafe_artifact_path(tmp_path: Path) -> None:
    config_path, spec_path = _mutable_contract(tmp_path)
    spec_path.write_text(
        spec_path.read_text(encoding="utf-8").replace(
            'artifact_subdir = "jm-train-window-sensitivity"',
            'artifact_subdir = "../outside"',
        ),
        encoding="utf-8",
    )

    with pytest.raises(WindowSpecError, match="safe relative path"):
        load_window_spec(spec_path, load_config(config_path))
