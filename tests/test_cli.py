import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from adaptive_jump import data
from adaptive_jump.cli import RunError, load_frozen_data, main, run_replication
from adaptive_jump.config import load_config
from adaptive_jump.data import HttpResult
from adaptive_jump.models import FixedJMResult, HMMResult
from adaptive_jump.walkforward import BaselineStudy, SelectionResult

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
        "553ff3fc0969eb9515fba546f135ad207c0578f0b2d2f812affc41c872101337"
    )
    assert len(manifest["sources"]) == 6
    assert manifest_path.parent.parent == tmp_path / "data/raw"


def test_fetch_cli_reports_missing_config(capsys) -> None:
    assert main(["fetch", "--config", "missing.toml"]) == 2
    assert "missing.toml" in capsys.readouterr().err


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
    manifest = tmp_path / "data/raw/shu-proxy-replication-v5-run/manifest.json"
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
        manifest_path.parent.parent / "shu-proxy-replication-v5-two/manifest.json"
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
    boundaries = pd.DataFrame(
        [
            {"model": model, "delay": delay, "passed": boundary_passed}
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
    monkeypatch.setattr("adaptive_jump.cli.research_git_sha", lambda _root: "a" * 40)

    run_dir = run_replication(config, frozen)

    metadata = json.loads((run_dir / "run.json").read_text())
    assert metadata["status"] == "complete"
    assert metadata["metrics_opened"] is True
    assert len(pd.read_csv(run_dir / "metrics.csv")) == 27
    assert json.loads((run_dir / "claim.json").read_text())["passed"] is False
    assert (run_dir / "us/trades/fixed_jm-delay-1.csv").is_file()
    assert run_replication(config, frozen) == run_dir

    (run_dir / "metrics.csv").write_text("tampered\n", encoding="utf-8")
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

    resumed_inputs = []

    def resumed(*_args, initial, **_kwargs):
        if initial is not None:
            resumed_inputs.append(initial)
        return initial or study.hmm

    monkeypatch.setattr("adaptive_jump.cli.hmm_states", resumed)
    run_dir = run_replication(config, frozen)

    assert json.loads((run_dir / "run.json").read_text())["status"] == "complete"
    assert isinstance(resumed_inputs[0], HMMResult)
    assert not (run_dir / "us/hmm-progress.pkl").exists()
