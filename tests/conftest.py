"""Shared fixtures for the restored -002 rerun harnesses.

The arrival, lagged and balanced studies were frozen against the v7 proxy
baseline: one global lambda grid, and source runs pinned by literals inside the
implementation. They are restored here as reruns against the calibrated-v10
contract, where every market carries its own grid and every source is named by
the spec.

No -002 spec is committed yet, so the tests synthesise one from the archived
-001 contract: the same frozen prose and the same decision rules, with the
identity fields moved to -002 and the grids expanded into per-market tables.
That rewrite is also the checklist a real -002 spec has to satisfy.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from adaptive_jump.config import ResearchConfig, load_config
from adaptive_jump.study_grids import market_grids, positive_grids

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
CALIBRATED_CONFIG_PATH = ROOT / "research-calibrated-v10.toml"
MARKETS = ("us", "de", "jp")

# -001 -> -002 for every study restored here. The sealed fixed baseline keeps
# whatever id its own run.json carries; the spec names it, so nothing here has
# to know it.
STUDY_RENAMES = {
    "adaptive-confidence-001": "adaptive-confidence-002",
    "lagged-evidence-mechanism-001": "lagged-evidence-mechanism-002",
    "lagged-evidence-performance-001": "lagged-evidence-performance-002",
    "lagged-selection-attribution-001": "lagged-selection-attribution-002",
    "balanced-lagged-mechanism-001": "balanced-lagged-mechanism-002",
    "balanced-lagged-performance-001": "balanced-lagged-performance-002",
}
# Where each named source run lives under the artifact root.
SOURCE_SUBDIRS = {
    "fixed-baselines-001-v7": "fixed-baselines",
    "adaptive-confidence-002": "adaptive-confidence-002",
    "lagged-evidence-mechanism-002": "lagged-evidence-mechanism-002",
    "lagged-evidence-performance-002": "lagged-evidence-performance-002",
    "balanced-lagged-mechanism-002": "balanced-lagged-mechanism-002",
}
V7_CONFIG_SHA = "8adb330565d64f8ed6edd986f0422dbba72585eda4efd34b0c1b41b95450d81b"


def calibrated_config() -> ResearchConfig:
    """The calibrated-v10 contract: three markets, three different grids."""
    return load_config(CALIBRATED_CONFIG_PATH)


def _grid_table(grids: dict[str, tuple[float, ...]]) -> str:
    body = ", ".join(
        f"{market} = [{', '.join(repr(value) for value in grids[market])}]"
        for market in MARKETS
    )
    return "{ " + body + " }"


def calibrated_spec_text(name: str, config: ResearchConfig) -> str:
    """Rewrite an archived -001 contract into its -002 rerun form.

    Four things move, and nothing else:

    * the study id and its artifact subdirectory become -002;
    * every source table gains ``artifact_subdir``, so the harness no longer
      hardcodes the directory a source run lives in;
    * ``config_sha256`` becomes the calibrated contract's digest;
    * ``raw_lambda_grid`` / ``event_lambda_grid`` become per-market tables read
      from ``config.jm_protocol_for(market)``.
    """
    text = (RESEARCH / name).read_text(encoding="utf-8")
    for old, new in STUDY_RENAMES.items():
        text = text.replace(old, new)
    text = text.replace(V7_CONFIG_SHA, config.sha256)

    lines: list[str] = []
    for line in text.splitlines():
        lines.append(line)
        match = re.fullmatch(r'experiment_id = "([^"]+)"', line)
        if match and match.group(1) in SOURCE_SUBDIRS:
            lines.append(f'artifact_subdir = "{SOURCE_SUBDIRS[match.group(1)]}"')
    text = "\n".join(lines) + "\n"

    grids = market_grids(config, MARKETS)
    text = re.sub(
        r"^raw_lambda_grid = \[[^\]]*\]$",
        "raw_lambda_grid = " + _grid_table(grids),
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^event_lambda_grid = \[[^\]]*\]$",
        "event_lambda_grid = " + _grid_table(positive_grids(grids)),
        text,
        flags=re.MULTILINE,
    )
    # balanced-lagged-performance names its parameter source in prose; the
    # loader now templates it on the declared fixed-source experiment id.
    text = text.replace(
        'fitted_parameters = "sealed v7 scalers and centers; no refit or state '
        'regeneration in the P&L runner"',
        'fitted_parameters = "sealed fixed-baselines-001-v7 scalers and centers; '
        'no refit or state regeneration in the P&L runner"',
    )
    return text


def write_calibrated_spec(
    directory: Path, name: str, config: ResearchConfig
) -> Path:
    """Materialise the -002 form of an archived contract under ``directory``."""
    target = directory / STUDY_RENAMES.get(name[: -len(".toml")], name[: -len(".toml")])
    path = target.with_suffix(".toml")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(calibrated_spec_text(name, config), encoding="utf-8")
    return path


@pytest.fixture
def config_v10() -> ResearchConfig:
    return calibrated_config()


def registered_config(
    tmp_path: Path, spec_path: Path, experiment_id: str
) -> ResearchConfig:
    """A contract rooted at ``tmp_path`` whose registry froze exactly this spec.

    Loaders check that the spec bytes they are handed are the bytes the
    registry froze. A synthetic spec therefore needs a synthetic registry, or
    the check would only ever be satisfiable by the committed contracts.
    """
    research = tmp_path / "research"
    research.mkdir(exist_ok=True)
    (research / "experiment_registry.jsonl").write_text(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "status": "FROZEN",
                "frozen_spec_hash": hashlib.sha256(
                    spec_path.read_bytes()
                ).hexdigest(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = calibrated_config()
    return replace(config, path=tmp_path / config.path.name)
