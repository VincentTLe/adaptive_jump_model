import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adaptive_jump.artifacts import (
    ArtifactError,
    verify_inventory,
    verify_run,
    write_inventory,
    write_json,
)
from adaptive_jump.backtest import apply_signal, buy_and_hold
from adaptive_jump.config import load_config
from adaptive_jump.inference import BootstrapProgress
from adaptive_jump.models import FixedJMResult
from adaptive_jump.reporting import build_report
from adaptive_jump.walkforward import SelectionResult
from adaptive_jump.window_evidence import (
    verify_window_bootstrap,
    verify_window_metrics,
)
from adaptive_jump.window_runner import _verify_parent, run_window_sensitivity
from adaptive_jump.window_spec import load_window_spec
from adaptive_jump.window_study import WindowMarketStudy, comparison_metrics
from adaptive_jump.window_verifier import (
    _read_states,
    _verify_refits,
    _verify_selection,
    verify_window_run,
)

ROOT = Path(__file__).resolve().parents[1]


def _fixture_run(
    tmp_path: Path,
    monkeypatch,
    *,
    boundary_passed: bool,
    interrupt_bootstrap: bool = False,
) -> Path:
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

    bootstrap_attempt = 0
    resumed_progress: list[BootstrapProgress] = []

    def fake_bootstrap(paths, frozen, research_config, *, initial=None, progress=None):
        nonlocal bootstrap_attempt
        if interrupt_bootstrap and bootstrap_attempt == 0:
            checkpoint = BootstrapProgress(
                np.zeros(500), np.random.default_rng(11).bit_generator.state
            )
            progress(frozen.bootstrap_blocks[0], checkpoint)
            bootstrap_attempt += 1
            raise RuntimeError("simulated bootstrap interruption")
        if interrupt_bootstrap:
            checkpoint = initial(frozen.bootstrap_blocks[0])
            if checkpoint is not None:
                resumed_progress.append(checkpoint)
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
    if interrupt_bootstrap:
        with pytest.raises(RuntimeError, match="bootstrap interruption"):
            run_window_sensitivity(config, spec)
        runtime = tmp_path / "artifacts/.monitor/checkpoints"
        assert len(list(runtime.rglob("bootstrap-us-block-60.json"))) == 1
        run_dir = run_window_sensitivity(config, spec)
        assert len(resumed_progress) == 1
        assert len(resumed_progress[0].draws) == 500
        assert not list(runtime.rglob("bootstrap-*.json"))
        assert not list(runtime.rglob("bootstrap-*.pkl"))
        return run_dir
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


def test_window_runner_resumes_identity_bound_bootstrap_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    run_dir = _fixture_run(
        tmp_path,
        monkeypatch,
        boundary_passed=True,
        interrupt_bootstrap=True,
    )

    assert json.loads((run_dir / "run.json").read_text())["status"] == "complete"


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


def test_window_evidence_is_recomputed_and_detects_metric_tampering(
    tmp_path: Path, monkeypatch
) -> None:
    run_dir = _fixture_run(tmp_path, monkeypatch, boundary_passed=True)
    config = load_config(tmp_path / "research.toml")
    spec = load_window_spec(
        tmp_path / "research/jm-train-window-sensitivity.toml", config
    )

    metrics, paths, metric_difference = verify_window_metrics(run_dir, config, spec)
    assert len(metrics) == 36
    assert metric_difference < 1e-12

    stored_bootstrap = pd.read_csv(run_dir / "bootstrap.csv")
    expected_bootstrap = stored_bootstrap.loc[stored_bootstrap["market"] == "us"].drop(
        columns="market"
    )
    monkeypatch.setattr(
        "adaptive_jump.window_evidence.bootstrap_rows",
        lambda *_: expected_bootstrap.copy(),
    )
    bootstrap, bootstrap_difference = verify_window_bootstrap(
        run_dir, paths, config, spec
    )
    assert len(bootstrap) == 9
    assert bootstrap_difference == 0.0

    metrics_path = run_dir / "metrics.csv"
    tampered = pd.read_csv(metrics_path)
    tampered.loc[0, "sharpe"] += 1.0
    tampered.to_csv(metrics_path, index=False)
    with pytest.raises(ArtifactError, match="evidence mismatch: sharpe"):
        verify_window_metrics(run_dir, config, spec)


def test_window_verifier_rebuilds_complete_claim(tmp_path: Path, monkeypatch) -> None:
    run_dir = _fixture_run(tmp_path, monkeypatch, boundary_passed=True)
    metadata = json.loads((run_dir / "run.json").read_text())
    config = load_config(tmp_path / "research.toml")
    spec = load_window_spec(
        tmp_path / "research/jm-train-window-sensitivity.toml", config
    )
    monkeypatch.setattr(
        "adaptive_jump.window_verifier._verify_identity",
        lambda _run: (metadata, config, spec, tmp_path / "parent"),
    )
    monkeypatch.setattr(
        "adaptive_jump.window_verifier._verify_model_evidence",
        lambda *_: pd.read_csv(run_dir / "boundaries.csv"),
    )
    monkeypatch.setattr(
        "adaptive_jump.window_verifier.verify_window_bootstrap",
        lambda *_: (pd.read_csv(run_dir / "bootstrap.csv"), 0.0),
    )

    receipt = verify_window_run(run_dir)

    assert receipt["metric_rows"] == 36
    assert receipt["bootstrap_rows"] == 9
    claim_path = run_dir / "claim.json"
    claim = json.loads(claim_path.read_text())
    claim["positive_markets"] = 99
    write_json(claim_path, claim)
    write_inventory(run_dir)
    with pytest.raises(ArtifactError, match="claim does not match"):
        verify_window_run(run_dir)


def test_window_verifier_accepts_closed_boundary_failure(
    tmp_path: Path, monkeypatch
) -> None:
    run_dir = _fixture_run(tmp_path, monkeypatch, boundary_passed=False)
    metadata = json.loads((run_dir / "run.json").read_text())
    config = load_config(tmp_path / "research.toml")
    spec = load_window_spec(
        tmp_path / "research/jm-train-window-sensitivity.toml", config
    )
    monkeypatch.setattr(
        "adaptive_jump.window_verifier._verify_identity",
        lambda _run: (metadata, config, spec, tmp_path / "parent"),
    )
    monkeypatch.setattr(
        "adaptive_jump.window_verifier._verify_model_evidence",
        lambda *_: pd.read_csv(run_dir / "boundaries.csv"),
    )

    receipt = verify_window_run(run_dir)

    assert receipt["status"] == "boundary_failed"
    assert receipt["metric_rows"] == 0


def test_generic_verifier_dispatches_window_study(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_json(
        run_dir / "run.json",
        {"study_kind": "jm_train_window_sensitivity"},
    )
    expected = {"status": "complete", "study_kind": "jm_train_window_sensitivity"}
    monkeypatch.setattr(
        "adaptive_jump.window_verifier.verify_window_run", lambda _run: expected
    )

    assert verify_run(run_dir) == expected


def test_window_report_is_english_verified_and_deterministic(
    tmp_path: Path, monkeypatch
) -> None:
    run_dir = _fixture_run(tmp_path, monkeypatch, boundary_passed=True)
    receipt = {
        "status": "complete",
        "inventory_files": 60,
        "metric_rows": 36,
        "bootstrap_rows": 9,
        "maximum_metric_absolute_difference": 0.0,
    }
    monkeypatch.setattr("adaptive_jump.window_reporting.verify_run", lambda _: receipt)

    report_path = build_report(run_dir)
    first = report_path.read_bytes()
    assert build_report(run_dir).read_bytes() == first
    report = first.decode()
    assert '<html lang="en">' in report
    assert "Does a longer JM training window help?" in report
    assert "not caused by v7 using a shorter JM window" in report
    assert "Not 2024&ndash;2026 evidence" in report
    assert "JM (4,000)" in report


def test_window_verifier_recomputes_model_evidence_files(tmp_path: Path) -> None:
    config = load_config(ROOT / "research.toml")
    spec = load_window_spec(ROOT / "research/jm-train-window-sensitivity.toml", config)
    dates = pd.DatetimeIndex(pd.bdate_range("2020-01-02", periods=4), name="date")
    states = pd.DataFrame(0.0, index=dates, columns=config.jm_protocol.lambda_grid)
    states_path = tmp_path / "jm-4000-states.csv"
    states.to_csv(states_path)
    pd.testing.assert_frame_equal(
        _read_states(states_path, config), states, check_freq=False
    )

    initial = pd.Timestamp("2019-11-12")
    scheduled = dates[0]
    refits = pd.DataFrame(
        [
            {
                "fit_date": fit_date,
                "training_start": fit_date - pd.DateOffset(years=16),
                "training_end": fit_date,
                "observations": spec.challenger_window,
                "lambda": penalty,
                "objective": 1.0,
            }
            for fit_date in (initial, scheduled)
            for penalty in config.jm_protocol.lambda_grid
        ]
    )
    refits_path = tmp_path / "jm-4000-refits.csv"
    refits.to_csv(refits_path, index=False)
    _verify_refits(refits_path, config, spec)

    target = tmp_path / "selection"
    target.mkdir()
    expected = SelectionResult(
        signal=pd.Series([np.nan, 1.0, 1.0, 0.0], index=dates, name="selected_signal"),
        choices=pd.DataFrame({"decision_date": [dates[1]], "selected": [5.0]}),
        surface=pd.DataFrame(
            {"decision_date": [dates[1]], "candidate": [5.0], "sharpe": [0.2]}
        ),
        candidate_returns=pd.DataFrame({5.0: [np.nan, 0.1, 0.2, 0.3]}, index=dates),
    )
    expected.choices.to_csv(target / "choices.csv", index=False)
    expected.surface.to_csv(target / "cv-surface.csv", index=False)
    expected.candidate_returns.to_csv(target / "candidate-returns.csv")
    expected.signal.to_csv(target / "selected-signal.csv", header=True)
    choices = pd.read_csv(target / "choices.csv")
    _verify_selection(target, choices, expected)

    choices.loc[0, "selected"] = 15.0
    with pytest.raises(ArtifactError, match="recomputed choices differs"):
        _verify_selection(target, choices, expected)


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
