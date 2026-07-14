import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adaptive_jump.artifacts import ArtifactError, verify_inventory, write_json
from adaptive_jump.backtest import apply_signal, buy_and_hold
from adaptive_jump.config import load_config
from adaptive_jump.models import FixedJMResult
from adaptive_jump.walkforward import SelectionResult
from adaptive_jump.window_runner import _verify_parent, run_window_sensitivity
from adaptive_jump.window_spec import load_window_spec
from adaptive_jump.window_study import WindowMarketStudy, comparison_metrics

ROOT = Path(__file__).resolve().parents[1]


def _fixture_run(tmp_path: Path, monkeypatch, *, boundary_passed: bool) -> Path:
    config_path = tmp_path / "research.toml"
    config_path.write_bytes((ROOT / "research.toml").read_bytes())
    spec_path = tmp_path / "research/jm-train-window-sensitivity.toml"
    spec_path.parent.mkdir()
    spec_path.write_bytes(
        (ROOT / "research/jm-train-window-sensitivity.toml").read_bytes()
    )
    config = load_config(config_path)
    spec = load_window_spec(spec_path, config)
    manifest = b'{"fixture": true}\n'
    spec = replace(spec, data_manifest_sha256=hashlib.sha256(manifest).hexdigest())
    parent = tmp_path / "artifacts/fixed-baselines" / spec.parent_run_id
    parent.mkdir(parents=True)
    (parent / "data-manifest.json").write_bytes(manifest)

    dates = pd.bdate_range("2018-01-02", periods=90)
    rng = np.random.default_rng(4)
    equity = rng.normal(0.0005, 0.01, len(dates))
    base_frame = pd.DataFrame(
        {
            "date": dates,
            "equity_simple": equity,
            "cash_return": 0.00005,
            "excess_return": equity - 0.00005,
            "dd_10": np.abs(equity),
            "sortino_20": 1.0,
            "sortino_60": 1.0,
        }
    )
    signal = pd.Series((np.arange(len(dates)) % 17 != 0).astype(float), index=dates)
    choices = pd.DataFrame({"decision_date": [dates[0]], "selected": [5.0]})
    selection = SelectionResult(
        signal=signal,
        choices=choices,
        surface=pd.DataFrame(
            {
                "decision_date": [dates[0]],
                "candidate": [5.0],
                "valid_returns": [30],
                "sharpe": [0.1],
                "eligible": [True],
            }
        ),
        candidate_returns=pd.DataFrame({5.0: equity}, index=dates),
    )
    boundaries = pd.DataFrame(
        [
            {
                "model": "jm_4000",
                "delay": delay,
                "upper_candidate": 1200.0,
                "selected_months": 0 if boundary_passed else 6,
                "total_months": 100,
                "fraction": 0.0 if boundary_passed else 0.06,
                "limit": 0.05,
                "passed": boundary_passed,
            }
            for delay in spec.delays
        ]
    )
    study = WindowMarketStudy(
        oos_start=dates[0].date(),
        jm=FixedJMResult(
            pd.DataFrame({5.0: signal}, index=dates),
            pd.DataFrame(
                {
                    "fit_date": [dates[0]],
                    "training_start": [dates[0]],
                    "training_end": [dates[0]],
                    "observations": [4000],
                    "lambda": [5.0],
                    "objective": [1.0],
                }
            ),
        ),
        selections={delay: selection for delay in spec.delays},
        boundaries=boundaries,
    )

    for market in config.markets:
        market_dir = parent / market.id
        market_dir.mkdir()
        base_frame.to_csv(market_dir / "features.csv", index=False)
        trades = market_dir / "trades"
        trades.mkdir()
        returns = base_frame[["date", "equity_simple", "cash_return"]]
        for delay in spec.delays:
            paths = {
                "buy_and_hold": buy_and_hold(returns),
                "hmm": apply_signal(
                    returns,
                    pd.Series((np.arange(len(dates)) % 13 != 0).astype(float)),
                    delay_trading_days=delay,
                ),
                "fixed_jm": apply_signal(
                    returns,
                    pd.Series((np.arange(len(dates)) % 11 != 0).astype(float)),
                    delay_trading_days=delay,
                ),
            }
            for model, path in paths.items():
                path.to_csv(trades / f"{model}-delay-{delay}.csv", index=False)

    monkeypatch.setattr(
        "adaptive_jump.window_runner._verify_parent",
        lambda *_: (
            {"run_id": spec.parent_run_id, "status": "complete"},
            {"git_sha": "a" * 40},
        ),
    )
    monkeypatch.setattr(
        "adaptive_jump.window_runner.research_git_sha", lambda _root: "b" * 40
    )
    monkeypatch.setattr(
        "adaptive_jump.window_runner._verify_control_source", lambda *_: None
    )
    monkeypatch.setattr(
        "adaptive_jump.window_runner.effective_oos_start",
        lambda *_args, **_kwargs: dates[0].date(),
    )
    monkeypatch.setattr(
        "adaptive_jump.window_runner.build_window_market_study",
        lambda *_args, **_kwargs: study,
    )

    def fake_bootstrap(paths, frozen, research_config):
        metrics = comparison_metrics(paths, research_config).set_index("model")
        delta = float(
            metrics.loc["jm_4000", "sharpe"] - metrics.loc["jm_3000", "sharpe"]
        )
        return pd.DataFrame(
            [
                {
                    "block_length": block,
                    "observed_delta": delta,
                    "lower_one_sided": delta - 0.1,
                    "confidence_low": delta - 0.2,
                    "confidence_high": delta + 0.2,
                    "replications": frozen.bootstrap_replications,
                }
                for block in frozen.bootstrap_blocks
            ]
        )

    monkeypatch.setattr("adaptive_jump.window_runner.bootstrap_rows", fake_bootstrap)
    return run_window_sensitivity(config, spec)


def test_window_runner_seals_complete_four_model_comparison(
    tmp_path: Path, monkeypatch
) -> None:
    run_dir = _fixture_run(tmp_path, monkeypatch, boundary_passed=True)

    metadata = json.loads((run_dir / "run.json").read_text())
    assert metadata["status"] == "complete"
    assert metadata["metrics_opened"] is True
    assert len(pd.read_csv(run_dir / "metrics.csv")) == 36
    assert len(pd.read_csv(run_dir / "bootstrap.csv")) == 9
    assert (run_dir / "us/trades/jm_4000-delay-1.csv").is_file()
    assert (
        json.loads((run_dir / "claim.json").read_text())["claim_class"] == "EXPLORATORY"
    )
    verify_inventory(run_dir)


def test_window_runner_keeps_metrics_closed_after_boundary_failure(
    tmp_path: Path, monkeypatch
) -> None:
    run_dir = _fixture_run(tmp_path, monkeypatch, boundary_passed=False)

    metadata = json.loads((run_dir / "run.json").read_text())
    assert metadata["status"] == "boundary_failed"
    assert metadata["metrics_opened"] is False
    assert not (run_dir / "metrics.csv").exists()
    assert not list(run_dir.glob("*/trades/*.csv"))
    verify_inventory(run_dir)


def test_parent_verification_rejects_wrong_inventory_hash(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "research.toml"
    config_path.write_bytes((ROOT / "research.toml").read_bytes())
    config = load_config(config_path)
    spec = load_window_spec(
        ROOT / "research/jm-train-window-sensitivity.toml",
        load_config(ROOT / "research.toml"),
    )
    parent = tmp_path / spec.parent_run_id
    parent.mkdir()
    write_json(
        parent / "run.json",
        {
            "config_sha256": config.sha256,
            "data_manifest_sha256": spec.data_manifest_sha256,
        },
    )
    write_json(parent / "inventory.json", {"schema_version": 1, "files": {}})
    monkeypatch.setattr(
        "adaptive_jump.window_runner.verify_run",
        lambda _path: {"status": "complete", "run_id": spec.parent_run_id},
    )

    with pytest.raises(ArtifactError, match="does not match"):
        _verify_parent(parent, config, spec)
