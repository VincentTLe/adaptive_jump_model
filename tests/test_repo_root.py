"""Repository-root discovery must not depend on where a config file sits.

Protocol configs live under ``configs/baselines/`` — the current comparator at
that level, superseded ones in ``legacy/`` below it. These tests pin the
behaviour that lets them sit there without silently redirecting artifacts,
data, or the experiment registry into the config's own directory.
"""

from pathlib import Path

import pytest

from adaptive_jump.config import load_config, resolve_repo_root

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_CONFIG = "configs/baselines/research-calibrated-v11.toml"
LEGACY_CONFIGS = (
    "configs/baselines/legacy/research.toml",
    "configs/baselines/legacy/research-expanding-v8-5.toml",
    "configs/baselines/legacy/research-expanding-v9-3.toml",
    "configs/baselines/legacy/research-expanding-v9-4.toml",
    "configs/baselines/legacy/research-calibrated-v10.toml",
    "configs/baselines/legacy/research-calibrated-v10-ninit60.toml",
)
CONFIG_NAMES = (CANONICAL_CONFIG, *LEGACY_CONFIGS)


def _repo(tmp_path: Path) -> Path:
    """A directory that looks like a repository: it carries the root marker."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'fixture'\n")
    return root


def _config_at(target: Path, source: str = CANONICAL_CONFIG) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((ROOT / source).read_bytes())
    return target


@pytest.mark.parametrize("relative", CONFIG_NAMES)
def test_every_protocol_config_resolves_the_real_repository(relative: str) -> None:
    """Each config really sits under configs/ and still resolves the same root."""
    config_path = ROOT / relative
    assert config_path.is_file()
    assert load_config(config_path).repo_root == ROOT


def test_the_canonical_comparator_sits_above_the_superseded_ones() -> None:
    """v11 is the current comparator; the six it replaced sit one level down."""
    baselines = ROOT / "configs" / "baselines"

    assert [path.name for path in sorted(baselines.glob("*.toml"))] == [
        "research-calibrated-v11.toml"
    ]
    assert len(list((baselines / "legacy").glob("*.toml"))) == len(LEGACY_CONFIGS)


def test_no_protocol_config_is_left_at_the_repository_root() -> None:
    """A stray root copy would be a second writable version of a sealed config."""
    assert list(ROOT.glob("research*.toml")) == []


@pytest.mark.parametrize(
    "relative",
    [
        "research-calibrated-v11.toml",
        "configs/research-calibrated-v11.toml",
        "configs/baselines/research-calibrated-v11.toml",
        "configs/baselines/legacy/research-calibrated-v11.toml",
    ],
)
def test_config_depth_does_not_move_the_repository_root(
    tmp_path: Path, relative: str
) -> None:
    root = _repo(tmp_path)
    config = _config_at(root / relative)

    assert resolve_repo_root(config) == root
    assert load_config(config).repo_root == root


def test_config_outside_any_repository_keeps_the_legacy_parent(
    tmp_path: Path,
) -> None:
    """Temporary configs written by tests have no marker above them."""
    config = _config_at(tmp_path / "loose" / "research.toml")

    assert resolve_repo_root(config) == config.parent
    assert load_config(config).repo_root == config.parent


def test_nearest_marker_wins_over_an_outer_one(tmp_path: Path) -> None:
    """A marker above the repository must not capture a config inside it."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'outer'\n")
    root = _repo(tmp_path)
    config = _config_at(root / "configs/baselines/research-calibrated-v11.toml")

    assert resolve_repo_root(config) == root


def test_resolution_is_independent_of_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repo(tmp_path)
    config = _config_at(root / "configs/baselines/research-calibrated-v11.toml")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    monkeypatch.chdir(elsewhere)
    from_elsewhere = load_config(config).repo_root
    monkeypatch.chdir(root / "configs")
    from_inside = load_config(config).repo_root

    assert from_elsewhere == from_inside == root


def test_relative_config_path_resolves_the_same_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A config named relative to the cwd must not resolve a different root."""
    root = _repo(tmp_path)
    _config_at(root / "configs/baselines/research-calibrated-v11.toml")

    monkeypatch.chdir(root)
    config = load_config(Path("configs/baselines/research-calibrated-v11.toml"))

    assert config.repo_root == root


def test_nested_config_keeps_derived_paths_at_the_repository_root(
    tmp_path: Path,
) -> None:
    """The Phase-B2 move must not drag artifacts/ and data/ under configs/."""
    root = _repo(tmp_path)
    nested = _config_at(root / "configs/baselines/research-calibrated-v11.toml")
    flat = _config_at(root / "research-calibrated-v11.toml")

    nested_config = load_config(nested)
    flat_config = load_config(flat)

    for derived in ("artifact_root", "raw_root", "processed_root"):
        nested_path = nested_config.repo_root / getattr(nested_config, derived)
        assert nested_path == flat_config.repo_root / getattr(flat_config, derived)
        assert nested_path.is_relative_to(root)
        assert not nested_path.is_relative_to(nested.parent)
    registry = nested_config.repo_root / "research" / "experiment_registry.jsonl"
    assert registry == root / "research" / "experiment_registry.jsonl"


def test_repo_root_follows_a_replaced_config_path(tmp_path: Path) -> None:
    """`replace(config, path=...)` retargets the root, as callers rely on."""
    from dataclasses import replace

    config = load_config(ROOT / "configs/baselines/legacy/research.toml")
    redirected = replace(config, path=tmp_path / "research.toml")

    assert config.repo_root == ROOT
    assert redirected.repo_root == tmp_path
