import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from adaptive_jump import data
from adaptive_jump.artifacts import ArtifactError, verify_run, write_inventory
from adaptive_jump.cli import RunError, load_frozen_data, main, run_replication
from adaptive_jump.config import load_config
from adaptive_jump.data import HttpResult
from adaptive_jump.models import FixedJMResult, HMMResult
from adaptive_jump.walkforward import BaselineStudy, SelectionProgress, SelectionResult

ROOT = Path(__file__).resolve().parents[1]


def test_fetch_cli_runs_complete_fixture_pipeline(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config = tmp_path / "research.toml"
    config.write_bytes((ROOT / "research.toml").read_bytes())

    def yahoo_loader(_source, _start, _end):
        return pd.DataFrame(
            {"Close": [100.0, 101.0]},
            index=pd.to_datetime(["2023-01-03", "2023-12-29"]),
        ), {"adapter": "fixture"}

    def http_get(url, _params):
        if "fredgraph" in url:
            source_id = "DTB3" if "DTB3" in url else "IR3TIB01DEM156N"
            content = (
                f"observation_date,{source_id}\n1970-01-02,1.0\n2023-12-01,2.0\n"
            ).encode()
        else:
            content = (
                b"STATUS,200\nNEXTPOSITION,\n"
                b"SERIES_CODE,NAME_OF_TIME_SERIES,UNIT,FREQUENCY,CATEGORY,"
                b"LAST_UPDATE,SURVEY_DATES,VALUES\n"
                b"STRACLUC3M,Call,percent,MONTHLY,Call,20240101,198901,1.0\n"
                b"STRACLUC3M,Call,percent,MONTHLY,Call,20240101,202312,2.0\n"
            )
        return HttpResult(content, url, 200, "text/csv")

    monkeypatch.setattr(data, "_download_yahoo", yahoo_loader)
    monkeypatch.setattr(data, "_get_http", http_get)
    monkeypatch.setattr(data, "research_git_sha", lambda _root: "abc123")

    assert main(["fetch", "--config", str(config)]) == 0

    manifest_path = Path(capsys.readouterr().out.strip())
    manifest = json.loads(manifest_path.read_text())
    assert manifest["config_sha256"] == (
        "8adb330565d64f8ed6edd986f0422dbba72585eda4efd34b0c1b41b95450d81b"
    )
    assert len(manifest["sources"]) == 6
    assert manifest_path.parent.parent == tmp_path / "data/raw"


def test_fetch_cli_reports_missing_config(capsys) -> None:
    assert main(["fetch", "--config", "missing.toml"]) == 2
    assert "missing.toml" in capsys.readouterr().err


def test_monitor_cli_delegates_to_the_loopback_server(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        "adaptive_jump.monitor.server.run_monitor_server",
        lambda config: calls.append(config) or 0,
    )

    assert main(["monitor", "--config", "research.toml"]) == 0
    assert calls == ["research.toml"]


def test_figures_cli_delegates_and_prints_each_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    run = tmp_path / "sealed-run"
    output_root = tmp_path / "reports"
    outputs = (output_root / "regimes.png", output_root / "wealth.pdf")
    calls = []

    def render(run_dir, destination):
        calls.append((run_dir, destination))
        return outputs

    monkeypatch.setattr("adaptive_jump.cli.render_figures", render)

    assert (
        main(
            [
                "figures",
                "--run",
                str(run),
                "--output-root",
                str(output_root),
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.splitlines() == [str(path) for path in outputs]
    assert calls == [(str(run), str(output_root))]


def test_figures_cli_uses_shared_artifact_error_handling(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    def reject(_run, _output_root):
        raise ArtifactError("invalid figure artifact")

    monkeypatch.setattr("adaptive_jump.cli.render_figures", reject)

    assert main(["figures", "--run", "bad-run"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "adaptive-jump: invalid figure artifact\n"


def test_calibration_cli_uses_frozen_spec(monkeypatch, capsys) -> None:
    expected = ROOT / "artifacts/calibration-fixture"
    calls = []

    def fake_run(config, spec):
        calls.append((config, spec))
        return expected

    monkeypatch.setattr("adaptive_jump.cli.run_calibration_study", fake_run)
    monkeypatch.setattr(
        "adaptive_jump.cli._artifacts.verify_run",
        lambda artifact: {"run_id": artifact.name, "status": "complete"},
    )
    arguments = [
        "run",
        "--study",
        "persistence-calibration",
        "--config",
        str(ROOT / "research.toml"),
    ]

    assert main(arguments) == 0
    assert Path(capsys.readouterr().out.strip()) == expected
    assert calls[0][1].name == "persistence-calibrated-search.toml"


def test_grid_evaluation_cli_uses_frozen_spec(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    expected = ROOT / "artifacts/grid-fixture"
    calls = []

    def fake_run(config, spec, observer):
        calls.append((config, spec, observer))
        return expected

    monkeypatch.setattr("adaptive_jump.cli.run_grid_evaluation", fake_run)
    monkeypatch.setattr(
        "adaptive_jump.cli._artifacts.verify_run",
        lambda artifact: {"run_id": artifact.name, "status": "boundary_failed"},
    )
    arguments = [
        "run",
        "--study",
        "persistence-grid-evaluation",
        "--config",
        str(ROOT / "research.toml"),
    ]

    assert main(arguments) == 0
    assert Path(capsys.readouterr().out.strip()) == expected
    assert calls[0][1].jm_grid[-1] == 256.0
    assert calls[0][1].hmm_grid[-1] == 1115


@pytest.mark.parametrize(
    ("study", "spec_name", "loader_name", "runner_name"),
    [
        (
            "simple-jm-suite",
            "simple-jm-suite-001.toml",
            "load_simple_jm_spec",
            "run_simple_jm_study",
        ),
        (
            "dd-loss-scale",
            "dd-loss-scale-001.toml",
            "load_dd_loss_scale_spec",
            "run_dd_loss_scale_study",
        ),
    ],
)
def test_simple_jm_cli_uses_shared_runner(
    monkeypatch: pytest.MonkeyPatch,
    capsys,
    study: str,
    spec_name: str,
    loader_name: str,
    runner_name: str,
) -> None:
    expected = ROOT / "artifacts" / f"{study}-fixture"
    calls = []
    events = []
    loaded_spec = object()

    def observer(event):
        events.append(event)

    def fake_load(path, config):
        calls.append(("load", config, path))
        return loaded_spec

    def fake_run(config, spec, selected_observer):
        calls.append(("run", config, spec, selected_observer))
        return expected

    monkeypatch.setattr(f"adaptive_jump.cli.{loader_name}", fake_load)
    monkeypatch.setattr(f"adaptive_jump.cli.{runner_name}", fake_run)
    monkeypatch.setattr(
        "adaptive_jump.cli._artifacts.verify_run",
        lambda artifact: {"run_id": artifact.name, "status": "complete"},
    )
    monkeypatch.setattr(
        "adaptive_jump.cli.child_observer_from_environment", lambda: observer
    )

    assert (
        main(
            [
                "run",
                "--study",
                study,
                "--config",
                str(ROOT / "research.toml"),
            ]
        )
        == 0
    )
    assert Path(capsys.readouterr().out.strip()) == expected
    assert calls[0][0] == "load"
    assert calls[0][2].name == spec_name
    assert calls[1][0] == "run"
    assert calls[1][2] is loaded_spec
    assert calls[1][3] is observer
    assert events[0].kind == "artifact_verified"
    assert events[0].payload == {
        "run_id": expected.name,
        "status": "complete",
    }


def test_window_study_cli_uses_frozen_spec_without_manifest(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    expected = ROOT / "artifacts/window-fixture"
    calls = []
    events = []

    def observer(event):
        events.append(event)

    def verify(artifact):
        assert artifact == expected
        return {"run_id": "window-fixture", "status": "boundary_failed"}

    def fake_run(config, spec, observer):
        calls.append((config, spec, observer))
        return expected

    monkeypatch.setattr("adaptive_jump.cli.run_window_sensitivity", fake_run)
    monkeypatch.setattr("adaptive_jump.cli._artifacts.verify_run", verify)
    monkeypatch.setattr(
        "adaptive_jump.cli.child_observer_from_environment", lambda: observer
    )
    arguments = [
        "run",
        "--study",
        "train-window-sensitivity",
        "--config",
        str(ROOT / "research.toml"),
    ]
    assert main(arguments) == 0
    assert Path(capsys.readouterr().out.strip()) == expected
    assert calls[0][1].challenger_window == 4000
    assert calls[0][2] is observer
    assert events[0].kind == "artifact_verified"
    assert events[0].visibility == "decision"
    assert events[0].payload == {
        "run_id": "window-fixture",
        "status": "boundary_failed",
    }

    def reject(_artifact):
        raise ArtifactError("verification failed")

    events.clear()
    monkeypatch.setattr("adaptive_jump.cli._artifacts.verify_run", reject)
    assert main(arguments) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "verification failed" in captured.err
    assert events == []


def test_window_study_cli_rejects_manifest_override(capsys) -> None:
    result = main(
        [
            "run",
            "--study",
            "train-window-sensitivity",
            "--config",
            str(ROOT / "research.toml"),
            "--manifest",
            "other.json",
        ]
    )

    assert result == 2
    assert "only valid for replication" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("study_kind", "verifier_name"),
    [
        ("simple-jm-suite-001", "verify_simple_jm_run"),
        ("dd-loss-scale-001", "verify_dd_loss_scale_run"),
    ],
)
def test_verify_run_dispatches_simple_jm_suite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    study_kind: str,
    verifier_name: str,
) -> None:
    run = tmp_path / f"{study_kind}-fixture"
    run.mkdir()
    (run / "run.json").write_text(
        json.dumps({"study_kind": study_kind}), encoding="utf-8"
    )
    expected = {"run_id": run.name, "status": "complete"}
    monkeypatch.setattr(
        f"adaptive_jump.simple_jm_suite.{verifier_name}",
        lambda selected: expected if selected == run.resolve() else None,
    )

    assert verify_run(run) == expected


def _manifest_fixture(tmp_path: Path) -> tuple[Path, Path]:
    config_path = tmp_path / "research.toml"
    config_path.write_bytes((ROOT / "research.toml").read_bytes())
    config = load_config(config_path)
    sources = []
    for market in config.markets:
        for kind, source in (("equity", market.equity), ("cash", market.cash)):
            path = tmp_path / "data/processed/run" / f"{market.id}_{kind}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("date,value\n2023-01-02,1.0\n", encoding="utf-8")
            sources.append(
                {
                    "market": market.id,
                    "kind": kind,
                    "source_id": source.source_id,
                    "canonical": {
                        "path": str(path.relative_to(tmp_path)),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    },
                }
            )
    manifest = tmp_path / "data/raw/shu-proxy-replication-v7-run/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "config_id": config.config_id,
                "config_sha256": config.sha256,
                "replication_cutoff": "2023-12-31",
                "sources": sources,
            }
        ),
        encoding="utf-8",
    )
    return config_path, manifest


def test_load_frozen_data_recomputes_every_canonical_hash(tmp_path: Path) -> None:
    config_path, manifest_path = _manifest_fixture(tmp_path)
    config = load_config(config_path)

    frozen = load_frozen_data(config)

    assert frozen.path == manifest_path
    assert frozen.sha256 == hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    canonical = tmp_path / frozen.document["sources"][0]["canonical"]["path"]
    canonical.write_text("date,value\n2023-01-02,2.0\n", encoding="utf-8")
    with pytest.raises(RunError, match="canonical hash mismatch"):
        load_frozen_data(config, manifest_path)


def test_ambiguous_matching_manifests_require_explicit_path(tmp_path: Path) -> None:
    config_path, manifest_path = _manifest_fixture(tmp_path)
    duplicate = (
        manifest_path.parent.parent / "shu-proxy-replication-v7-two/manifest.json"
    )
    duplicate.parent.mkdir()
    duplicate.write_bytes(manifest_path.read_bytes())

    with pytest.raises(RunError, match="found 2"):
        load_frozen_data(load_config(config_path))


def _runner_fixture(config, *, boundary_passed: bool = True):
    dates = pd.bdate_range("2021-01-04", periods=40)
    equity = [0.01 if index % 3 else -0.005 for index in range(len(dates))]
    frame = pd.DataFrame(
        {
            "date": dates,
            "equity_simple": equity,
            "equity_log": equity,
            "cash_return": 0.0001,
            "excess_return": pd.Series(equity) - 0.0001,
            "dd_10": 1.0,
            "sortino_20": 1.0,
            "sortino_60": 1.0,
        }
    )
    signal = pd.Series(1.0, index=dates, name="selected_signal")
    choices = pd.DataFrame({"decision_date": [dates[0]], "selected": [0.0]})
    selection = SelectionResult(
        signal=signal,
        choices=choices,
        surface=pd.DataFrame(),
        candidate_returns=pd.DataFrame(index=dates),
    )
    selections = {
        model: {
            delay: selection for delay in config.backtest_protocol.robustness_delays
        }
        for model in ("fixed_jm", "hmm")
    }
    boundary_fraction = 0.0 if boundary_passed else 0.06
    boundaries = pd.DataFrame(
        [
            {
                "model": model,
                "delay": delay,
                "upper_candidate": max(
                    config.jm_protocol.lambda_grid
                    if model == "fixed_jm"
                    else config.hmm_protocol.smoothing_grid
                ),
                "selected_months": int(boundary_fraction * 100),
                "total_months": 100,
                "fraction": boundary_fraction,
                "limit": config.selection_protocol.boundary_fraction_limit,
                "passed": boundary_passed,
            }
            for model in selections
            for delay in config.backtest_protocol.robustness_delays
        ]
    )
    empty_states = pd.DataFrame(index=dates)
    study = BaselineStudy(
        oos_start=dates[0].date(),
        jm=FixedJMResult(empty_states, pd.DataFrame()),
        hmm=HMMResult(pd.Series(index=dates, dtype=float), pd.DataFrame()),
        hmm_candidates=empty_states,
        selections=selections,
        boundaries=boundaries,
    )
    return frame, study


def test_replication_runner_writes_and_verifies_complete_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    config_path, manifest_path = _manifest_fixture(tmp_path)
    config = load_config(config_path)
    frozen = load_frozen_data(config, manifest_path)
    frame, study = _runner_fixture(config)
    monkeypatch.setattr(
        "adaptive_jump.cli.prepare_manifest_market",
        lambda *_: type("Input", (), {"frame": frame, "oos_start": study.oos_start})(),
    )
    monkeypatch.setattr(
        "adaptive_jump.cli.build_baseline_study", lambda *_args, **_kwargs: study
    )
    monkeypatch.setattr("adaptive_jump.cli.research_git_sha", lambda _root: "a" * 40)

    run_dir = run_replication(config, frozen)

    metadata = json.loads((run_dir / "run.json").read_text())
    assert metadata["status"] == "complete"
    assert metadata["metrics_opened"] is True
    assert len(pd.read_csv(run_dir / "metrics.csv")) == 27
    assert json.loads((run_dir / "claim.json").read_text())["passed"] is False
    assert (run_dir / "us/trades/fixed_jm-delay-1.csv").is_file()
    assert not list(run_dir.rglob("*.pkl"))
    runtime_root = tmp_path / "artifacts/.monitor/checkpoints"
    assert len(list(runtime_root.rglob("baseline-study.json"))) == 3
    assert run_replication(config, frozen) == run_dir
    assert main(["verify", "--run", str(run_dir)]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["boundary_rows"] == 18
    assert receipt["metric_rows"] == 27

    assert main(["report", "--run", str(run_dir)]) == 0
    report_path = Path(capsys.readouterr().out.strip())
    report = report_path.read_text(encoding="utf-8")
    assert report_path.parent.name == run_dir.name
    assert '<html lang="en">' in report
    assert "Fixed-baseline proxy replication" in report
    assert "non-replication; adaptive work remains blocked" in report
    first_report = report_path.read_bytes()
    assert main(["report", "--run", str(run_dir)]) == 0
    assert report_path.read_bytes() == first_report
    capsys.readouterr()

    metrics_path = run_dir / "metrics.csv"
    original_metrics = metrics_path.read_bytes()
    metrics = pd.read_csv(metrics_path)
    metrics.loc[0, "sharpe"] += 1.0
    metrics.to_csv(metrics_path, index=False)
    write_inventory(run_dir)
    with pytest.raises(ArtifactError, match="metric mismatch"):
        verify_run(run_dir)
    metrics_path.write_bytes(original_metrics)
    write_inventory(run_dir)

    metrics_path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RunError, match="inventory mismatch"):
        run_replication(config, frozen)


def test_replication_runner_does_not_open_metrics_after_boundary_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, manifest_path = _manifest_fixture(tmp_path)
    config = load_config(config_path)
    frozen = load_frozen_data(config, manifest_path)
    frame, study = _runner_fixture(config, boundary_passed=False)
    monkeypatch.setattr(
        "adaptive_jump.cli.prepare_manifest_market",
        lambda *_: type("Input", (), {"frame": frame, "oos_start": study.oos_start})(),
    )
    monkeypatch.setattr(
        "adaptive_jump.cli.build_baseline_study", lambda *_args, **_kwargs: study
    )
    monkeypatch.setattr("adaptive_jump.cli.research_git_sha", lambda _root: "b" * 40)

    run_dir = run_replication(config, frozen)

    metadata = json.loads((run_dir / "run.json").read_text())
    assert metadata["status"] == "boundary_failed"
    assert metadata["metrics_opened"] is False
    assert not (run_dir / "metrics.csv").exists()
    assert verify_run(run_dir)["metric_rows"] == 0


def test_replication_runner_resumes_hmm_progress_after_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, manifest_path = _manifest_fixture(tmp_path)
    config = load_config(config_path)
    frozen = load_frozen_data(config, manifest_path)
    frame, study = _runner_fixture(config)
    monkeypatch.setattr(
        "adaptive_jump.cli.prepare_manifest_market",
        lambda *_: type("Input", (), {"frame": frame, "oos_start": study.oos_start})(),
    )
    monkeypatch.setattr(
        "adaptive_jump.cli.build_baseline_study",
        lambda *_args, **_kwargs: study,
    )
    monkeypatch.setattr("adaptive_jump.cli.research_git_sha", lambda _root: "c" * 40)

    def interrupted(*_args, progress, **_kwargs):
        progress(study.hmm)
        raise RuntimeError("interrupted")

    monkeypatch.setattr("adaptive_jump.cli.hmm_states", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        run_replication(config, frozen)
    runtime_root = tmp_path / "artifacts/.monitor/checkpoints"
    assert len(list(runtime_root.rglob("hmm-progress.json"))) == 1

    resumed_inputs = []

    def resumed(*_args, initial, **_kwargs):
        if initial is not None:
            resumed_inputs.append(initial)
        return initial or study.hmm

    monkeypatch.setattr("adaptive_jump.cli.hmm_states", resumed)
    run_dir = run_replication(config, frozen)

    assert json.loads((run_dir / "run.json").read_text())["status"] == "complete"
    assert isinstance(resumed_inputs[0], HMMResult)
    assert not list(runtime_root.rglob("hmm-progress.json"))
    assert not list(runtime_root.rglob("hmm-progress.*.pkl"))


def test_replication_runner_resumes_fixed_jm_progress_after_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, manifest_path = _manifest_fixture(tmp_path)
    config = load_config(config_path)
    frozen = load_frozen_data(config, manifest_path)
    frame, study = _runner_fixture(config)
    monkeypatch.setattr(
        "adaptive_jump.cli.prepare_manifest_market",
        lambda *_: type("Input", (), {"frame": frame, "oos_start": study.oos_start})(),
    )
    monkeypatch.setattr(
        "adaptive_jump.cli.build_baseline_study", lambda *_args, **_kwargs: study
    )
    monkeypatch.setattr("adaptive_jump.cli.research_git_sha", lambda _root: "d" * 40)

    def interrupted(*_args, progress, **_kwargs):
        progress(study.jm)
        raise RuntimeError("interrupted")

    monkeypatch.setattr("adaptive_jump.cli.fixed_jm_states", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        run_replication(config, frozen)
    runtime_root = tmp_path / "artifacts/.monitor/checkpoints"
    assert len(list(runtime_root.rglob("jm-progress.json"))) == 1

    resumed_inputs = []

    def resumed(*_args, initial, **_kwargs):
        if initial is not None:
            resumed_inputs.append(initial)
        return initial or study.jm

    monkeypatch.setattr("adaptive_jump.cli.fixed_jm_states", resumed)
    run_dir = run_replication(config, frozen)

    assert json.loads((run_dir / "run.json").read_text())["status"] == "complete"
    assert isinstance(resumed_inputs[0], FixedJMResult)
    assert not list(runtime_root.rglob("jm-progress.json"))
    assert not list(runtime_root.rglob("jm-progress.*.pkl"))


def test_replication_runner_resumes_monthly_selection_after_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, manifest_path = _manifest_fixture(tmp_path)
    config = load_config(config_path)
    frozen = load_frozen_data(config, manifest_path)
    frame, study = _runner_fixture(config)
    progress = SelectionProgress(
        choices=pd.DataFrame(
            {"decision_date": [pd.Timestamp("2021-01-29")], "selected": [0.0]}
        ),
        surface=pd.DataFrame(
            {
                "decision_date": [pd.Timestamp("2021-01-29")],
                "candidate": [0.0],
                "valid_returns": [20],
                "sharpe": [1.0],
                "eligible": [True],
            }
        ),
    )
    monkeypatch.setattr(
        "adaptive_jump.cli.prepare_manifest_market",
        lambda *_: type("Input", (), {"frame": frame, "oos_start": study.oos_start})(),
    )
    monkeypatch.setattr("adaptive_jump.cli.research_git_sha", lambda _root: "e" * 40)

    def interrupted(*_args, selection_progress, **_kwargs):
        selection_progress("fixed_jm", 1, progress)
        raise RuntimeError("interrupted")

    monkeypatch.setattr("adaptive_jump.cli.build_baseline_study", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        run_replication(config, frozen)
    runtime_root = tmp_path / "artifacts/.monitor/checkpoints"
    assert len(list(runtime_root.rglob("selection-fixed_jm-delay-1.json"))) == 1

    resumed_inputs = []

    def resumed(*_args, selection_initial, **_kwargs):
        initial = selection_initial("fixed_jm", 1)
        if initial is not None:
            resumed_inputs.append(initial)
        return study

    monkeypatch.setattr("adaptive_jump.cli.build_baseline_study", resumed)
    run_replication(config, frozen)

    pd.testing.assert_frame_equal(resumed_inputs[0].choices, progress.choices)
    pd.testing.assert_frame_equal(resumed_inputs[0].surface, progress.surface)
    assert not list(runtime_root.rglob("selection-*.json"))
    assert not list(runtime_root.rglob("selection-*.pkl"))
