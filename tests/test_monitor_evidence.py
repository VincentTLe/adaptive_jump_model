import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from adaptive_jump import artifacts
from adaptive_jump.monitor.evidence import (
    EvidenceDefinition,
    EvidenceError,
    EvidenceStore,
    OutcomeLocked,
)


def _fixture(
    tmp_path: Path, *, metrics_opened: bool
) -> tuple[EvidenceStore, Path, list[Path]]:
    run_id = "sealed-run-001"
    relative = Path("artifacts/test") / run_id
    run_dir = tmp_path / relative
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": "complete" if metrics_opened else "boundary_failed",
                "metrics_opened": metrics_opened,
                "claim_label": "fixture replication",
            }
        )
    )
    pd.DataFrame(
        [{"model": "fixed_jm", "delay": 1, "passed": not metrics_opened}]
    ).to_csv(run_dir / "boundaries.csv", index=False)
    if metrics_opened:
        pd.DataFrame([{"model": "fixed_jm", "sharpe": 0.7}]).to_csv(
            run_dir / "metrics.csv", index=False
        )
        (run_dir / "claim.json").write_text('{"passed":false}\n')
    artifacts.write_inventory(run_dir)
    calls = []

    def verify(path):
        calls.append(Path(path))
        status = "complete" if metrics_opened else "boundary_failed"
        return {"run_id": run_id, "status": status, "conclusion": "hidden"}

    definition = EvidenceDefinition(run_id, "Fixture", relative, verify)
    return EvidenceStore(tmp_path, {run_id: definition}), run_dir, calls


def test_verified_evidence_excludes_conclusion_and_caches_by_inventory(
    tmp_path: Path,
) -> None:
    store, run_dir, calls = _fixture(tmp_path, metrics_opened=True)

    first = store.evidence("sealed-run-001")
    second = store.evidence("sealed-run-001")

    assert first == second
    assert first["metrics_opened"] is True
    assert "conclusion" not in first["verification"]
    assert first["boundaries"] == [{"model": "fixed_jm", "delay": 1, "passed": False}]
    assert calls == [run_dir]


def test_outcome_requires_open_flag_and_successful_verification(tmp_path: Path) -> None:
    opened, opened_dir, _calls = _fixture(tmp_path / "open", metrics_opened=True)
    locked, _run_dir, _calls = _fixture(tmp_path / "locked", metrics_opened=False)

    outcome = opened.outcome("sealed-run-001")

    assert outcome["metrics"][0]["sharpe"] == 0.7
    assert outcome["claim"] == {"passed": False}
    assert outcome["verification"]["conclusion"] == "hidden"
    with pytest.raises(OutcomeLocked, match="locked"):
        locked.outcome("sealed-run-001")

    (opened_dir / "metrics.csv").write_text("model,sharpe\nfixed_jm,99\n")
    with pytest.raises(EvidenceError, match="verification"):
        opened.outcome("sealed-run-001")


def test_unknown_paths_and_wrong_verifier_identity_fail_closed(tmp_path: Path) -> None:
    store, _run_dir, _calls = _fixture(tmp_path, metrics_opened=True)
    with pytest.raises(EvidenceError, match="registered"):
        store.evidence("../secret")

    definition = EvidenceDefinition(
        "known-run",
        "Bad verifier",
        Path("artifacts/test/known-run"),
        lambda _path: {"run_id": "other-run"},
    )
    run_dir = tmp_path / "bad/artifacts/test/known-run"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text('{"run_id":"known-run","status":"complete"}\n')
    artifacts.write_inventory(run_dir)
    with pytest.raises(EvidenceError, match="different"):
        EvidenceStore(tmp_path / "bad", {"known-run": definition}).evidence("known-run")


def test_catalog_reports_missing_ignored_artifacts_without_reading_them(
    tmp_path: Path,
) -> None:
    definition = EvidenceDefinition(
        "missing-run",
        "Missing",
        Path("artifacts/test/missing-run"),
        lambda _path: {},
    )
    store = EvidenceStore(tmp_path, {"missing-run": definition})

    assert store.catalog() == (
        {"run_id": "missing-run", "title": "Missing", "available": False},
    )
    with pytest.raises(EvidenceError, match="unavailable"):
        store.evidence("missing-run")


def test_market_data_is_bound_to_verified_manifest_and_raw_hash(
    tmp_path: Path, monkeypatch
) -> None:
    run_id = "fixed-baselines-aaaaaaaaaaaa-bbbbbbbbbbbb-cccccccccccc"
    run_dir = tmp_path / "artifacts/fixed-baselines" / run_id
    raw_path = tmp_path / "data/raw/acquisition/us_equity.csv"
    run_dir.mkdir(parents=True)
    raw_path.parent.mkdir(parents=True)
    raw_payload = (
        b"Date,Adj Close,Close,High,Low,Open,Volume\n"
        b"2023-12-28 00:00:00-05:00,100,100,102,99,101,0\n"
        b"2023-12-29 00:00:00-05:00,101,101,101,101,101,0\n"
    )
    raw_path.write_bytes(raw_payload)
    (run_dir / "run.json").write_text(
        json.dumps({"run_id": run_id, "status": "complete"})
    )
    (run_dir / "data-manifest.json").write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "market": "us",
                        "kind": "equity",
                        "provider": "yahoo",
                        "source_id": "^SP500TR",
                        "currency": "USD",
                        "frequency": "daily",
                        "source_classification": "proxy_candidate",
                        "deviations": ["short fixture"],
                        "quality": {"rows": 2, "valid_rows": 2},
                        "raw": {
                            "path": "data/raw/acquisition/us_equity.csv",
                            "bytes": len(raw_payload),
                            "sha256": hashlib.sha256(raw_payload).hexdigest(),
                        },
                    }
                ]
            }
        )
    )
    artifacts.write_inventory(run_dir)
    monkeypatch.setattr(
        "adaptive_jump.monitor.evidence.artifacts.verify_run",
        lambda _path: {"run_id": run_id, "status": "complete"},
    )
    store = EvidenceStore(tmp_path, {})

    result = store.market_data(run_id, "us")

    assert result["source"]["source_id"] == "^SP500TR"
    assert result["quality"]["complete_ohlc_rows"] == 2
    assert result["quality"]["distinct_ohlc_rows"] == 1
    assert result["quality"]["nonzero_volume_rows"] == 0
    assert result["rows"][0] == {
        "date": "2023-12-28",
        "open": 101.0,
        "high": 102.0,
        "low": 99.0,
        "close": 100.0,
        "volume": 0,
    }

    raw_path.write_bytes(raw_payload.replace(b",100,100", b",999,100", 1))
    with pytest.raises(EvidenceError, match="hash"):
        store.market_data(run_id, "us")

    raw_path.write_bytes(raw_payload)
    manifest_path = run_dir / "data-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sources"][0]["raw"]["path"] = "../outside.csv"
    manifest_path.write_text(json.dumps(manifest))
    artifacts.write_inventory(run_dir)
    with pytest.raises(EvidenceError, match="outside"):
        store.market_data(run_id, "us")


def _trade_frame(
    dates: list[str],
    equity: list[float],
    signal: list[float],
    position: list[float],
) -> pd.DataFrame:
    cash = pd.Series([0.001] * len(dates))
    position_series = pd.Series(position)
    equity_series = pd.Series(equity)
    turnover = position_series.diff().abs().fillna(0.0)
    gross = position_series * equity_series + (1.0 - position_series) * cash
    cost = turnover * 10 / 10_000
    return pd.DataFrame(
        {
            "date": dates,
            "equity_simple": equity,
            "cash_return": cash,
            "signal": signal,
            "position": position,
            "gross_return": gross,
            "one_way_turnover": turnover,
            "transaction_cost": cost,
            "strategy_return": gross - cost,
        }
    )


def test_market_story_uses_opened_verified_trade_and_feature_paths(
    tmp_path: Path, monkeypatch
) -> None:
    run_id = "fixed-baselines-aaaaaaaaaaaa-bbbbbbbbbbbb-cccccccccccc"
    run_dir = tmp_path / "artifacts/fixed-baselines" / run_id
    market_dir = run_dir / "us"
    trades = market_dir / "trades"
    trades.mkdir(parents=True)
    dates = ["2023-12-25", "2023-12-26", "2023-12-27", "2023-12-28"]
    equity = [0.01, -0.02, 0.03, -0.01]
    strategy = _trade_frame(dates, equity, [0, 1, 1, 0], [0, 0, 0, 1])
    hold = _trade_frame(dates, equity, [1, 1, 1, 1], [1, 1, 1, 1])
    strategy.to_csv(trades / "fixed_jm-delay-1.csv", index=False)
    hold.to_csv(trades / "buy_and_hold-delay-1.csv", index=False)
    pd.DataFrame(
        {
            "date": dates,
            "excess_return": [0.009, -0.021, 0.029, -0.011],
            "dd_10": [0.0, 0.01, 0.009, 0.012],
            "sortino_20": [None, -0.4, 0.3, 0.2],
            "sortino_60": [None, -0.2, 0.1, 0.1],
        }
    ).to_csv(market_dir / "features.csv", index=False)
    (run_dir / "config.lock.toml").write_bytes(
        (Path(__file__).resolve().parents[1] / "research.toml").read_bytes()
    )
    metadata_path = run_dir / "run.json"
    metadata_path.write_text(
        json.dumps({"run_id": run_id, "status": "complete", "metrics_opened": True})
    )
    artifacts.write_inventory(run_dir)
    monkeypatch.setattr(
        "adaptive_jump.monitor.evidence.artifacts.verify_run",
        lambda _path: {"run_id": run_id, "status": "complete"},
    )
    store = EvidenceStore(tmp_path, {})

    result = store.market_story(run_id, "us", "fixed_jm", 1)

    assert result["protocol"] == {
        "delay_trading_days": 1,
        "effective_return_offset": 2,
        "one_way_cost_bps": 10,
    }
    assert result["coverage"] == {
        "first_date": dates[0],
        "last_date": dates[-1],
        "rows": 4,
    }
    assert result["rows"][0]["sortino_20"] is None
    assert result["rows"][2]["position"] == 0.0
    assert result["rows"][3]["transaction_cost"] == pytest.approx(0.001)
    assert result["rows"][0]["strategy_wealth_100"] == pytest.approx(100.1)
    assert result["rows"][1]["buy_hold_drawdown"] < 0

    metadata_path.write_text(
        json.dumps({"run_id": run_id, "status": "complete", "metrics_opened": False})
    )
    with pytest.raises(OutcomeLocked, match="locked"):
        store.market_story(run_id, "us", "fixed_jm", 1)

    metadata_path.write_text(
        json.dumps({"run_id": run_id, "status": "complete", "metrics_opened": 1})
    )
    with pytest.raises(OutcomeLocked, match="locked"):
        store.market_story(run_id, "us", "fixed_jm", 1)

    metadata_path.write_text(
        json.dumps({"run_id": run_id, "status": "complete", "metrics_opened": True})
    )
    strategy.loc[3, "strategy_return"] += 0.01
    strategy.to_csv(trades / "fixed_jm-delay-1.csv", index=False)
    artifacts.write_inventory(run_dir)
    with pytest.raises(EvidenceError, match="accounting"):
        store.market_story(run_id, "us", "fixed_jm", 1)

    with pytest.raises(EvidenceError, match="model"):
        store.market_story(run_id, "us", "p2", 1)
    with pytest.raises(EvidenceError, match="delay"):
        store.market_story(run_id, "us", "fixed_jm", 2)
