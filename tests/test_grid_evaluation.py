from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import adaptive_jump.grid_runner as grid_runner
from adaptive_jump.config import load_config
from adaptive_jump.grid_runner import (
    COMPARISON_MODELS,
    _grid_claim,
    _path_metrics,
    _run_bootstrap,
)
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


def test_grid_control_gate_uses_sealed_calibration_sha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT / "research.toml")
    spec = load_grid_spec(SPEC, config)
    observed = {}
    monkeypatch.setattr(
        grid_runner,
        "_verify_parents",
        lambda *_args: ({}, {"git_sha": "parent"}, {"git_sha": "calibration"}),
    )
    monkeypatch.setattr(grid_runner, "research_git_sha", lambda _root: "current")

    def stop_after_gate(_root: Path, baseline: str, current: str) -> None:
        observed.update(baseline=baseline, current=current)
        raise RuntimeError("stop after control gate")

    monkeypatch.setattr(grid_runner, "_verify_control_source", stop_after_gate)

    with pytest.raises(RuntimeError, match="stop after control gate"):
        grid_runner.run_grid_evaluation(config, spec)

    assert observed == {"baseline": "calibration", "current": "current"}


def _toy_path(scale: float) -> pd.DataFrame:
    observations = 200
    equity = 0.001 + 0.004 * np.sin(np.arange(observations) / 7)
    cash = np.full(observations, 0.0001)
    strategy = cash + scale * (equity - cash)
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2000-01-03", periods=observations),
            "equity_simple": equity,
            "cash_return": cash,
            "signal": np.ones(observations),
            "position": np.ones(observations),
            "gross_return": strategy,
            "one_way_turnover": np.zeros(observations),
            "transaction_cost": np.zeros(observations),
            "strategy_return": strategy,
        }
    )


def test_grid_bootstrap_and_claim_smoke(tmp_path: Path) -> None:
    config = load_config(ROOT / "research.toml")
    frozen = load_grid_spec(SPEC, config)
    spec = replace(
        frozen,
        bootstrap_replications=20,
        bootstrap_blocks=(2, 3, 4),
    )
    scales = {
        "buy_and_hold": 1.0,
        "hmm_v7": 0.4,
        "hmm_new_grid": 0.45,
        "fixed_jm_v7": 0.5,
        "fixed_jm_new_grid": 0.6,
    }
    one_market = {model: _toy_path(scales[model]) for model in COMPARISON_MODELS}
    paths = {market: one_market for market in ("us", "de", "jp")}
    metrics = pd.concat(
        [
            _path_metrics(one_market, config).assign(
                market=market, delay=spec.primary_delay
            )
            for market in paths
        ],
        ignore_index=True,
    )

    bootstrap = _run_bootstrap(
        paths,
        config,
        spec,
        tmp_path,
        {"test_identity": "fixture"},
        None,
    )
    claim = _grid_claim(metrics, bootstrap, spec)

    assert len(bootstrap) == 18
    assert bootstrap["holm_adjusted_p"].between(0, 1).all()
    assert claim["claim_class"] == "EXPLORATORY"
    assert len(claim["fixed_jm"]) == 3
