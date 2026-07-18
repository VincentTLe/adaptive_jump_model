from __future__ import annotations

import hashlib
from concurrent.futures import Future
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import adaptive_jump.endpoint_grid_artifact_verifier as artifact_verifier
import adaptive_jump.endpoint_grid_audit as audit
import adaptive_jump.endpoint_grid_replay as replay
import adaptive_jump.endpoint_grid_smoke as runner_smoke
import adaptive_jump.endpoint_grid_verifier as input_verifier
from adaptive_jump.config import PAPER_TURNOVER_DEFINITION, load_config
from adaptive_jump.endpoint_grid_types import MarketSource
from adaptive_jump.models import FixedJMResult, smoothed_hmm_states
from adaptive_jump.walkforward import boundary_diagnostic, select_monthly_candidate

ROOT = Path(__file__).resolve().parents[1]


def _write_witness(
    target: Path,
    source: MarketSource,
    config,
    jm_grid: tuple[float, ...],
    hmm_grid: tuple[int, ...],
    fit: FixedJMResult,
) -> None:
    target.mkdir(parents=True)
    base_jm = fit.states.loc[:, list(jm_grid)]
    base_refits = fit.refits.loc[fit.refits["lambda"].isin(jm_grid)]
    base_hmm = smoothed_hmm_states(source.raw_hmm, hmm_grid)
    base_jm.to_csv(target / "jm-states.csv")
    base_refits.to_csv(target / "jm-refits.csv", index=False)
    base_hmm.to_csv(target / "hmm-candidates.csv")
    returns = source.frame[["date", "equity_simple", "cash_return"]]
    rows = []
    for delay in config.backtest_protocol.robustness_delays:
        for model, states in (("fixed_jm", base_jm), ("hmm", base_hmm)):
            selected = select_monthly_candidate(
                returns,
                states,
                config.selection_protocol,
                delay_trading_days=delay,
                one_way_cost_bps=config.backtest_protocol.one_way_cost_bps,
                periods_per_year=config.metrics_protocol.periods_per_year,
                volatility_ddof=config.metrics_protocol.volatility_ddof,
            )
            directory = target / f"{model}-delay-{delay}"
            directory.mkdir()
            selected.choices.to_csv(directory / "choices.csv", index=False)
            selected.surface.to_csv(directory / "cv-surface.csv", index=False)
            selected.candidate_returns.to_csv(directory / "candidate-returns.csv")
            selected.signal.to_csv(directory / "selected-signal.csv", header=True)
            diagnostic = boundary_diagnostic(
                selected.choices,
                tuple(float(value) for value in states.columns),
                oos_start=source.oos_start,
                fraction_limit=config.selection_protocol.boundary_fraction_limit,
            )
            rows.append({"model": model, "delay": delay, **diagnostic.__dict__})
    pd.DataFrame.from_records(rows).to_csv(target / "boundaries.csv", index=False)


def _small_fixture(tmp_path: Path, market: str = "us"):
    base = load_config(ROOT / "research.toml")
    config = replace(
        base,
        model_protocol=replace(base.model_protocol, fit_window=3),
        backtest_protocol=replace(
            base.backtest_protocol, primary_delay=1, robustness_delays=(1,)
        ),
        selection_protocol=replace(
            base.selection_protocol,
            validation_years=1,
            minimum_valid_returns=5,
        ),
        metrics_protocol=replace(
            base.metrics_protocol,
            turnover_definition=PAPER_TURNOVER_DEFINITION,
        ),
    )
    dates = pd.bdate_range("2020-01-02", periods=420, name="date")
    sequence = np.arange(len(dates))
    wave = np.sin(sequence / 4.0)
    frame = pd.DataFrame(
        {
            "date": dates,
            "equity_simple": 0.001 + 0.01 * wave,
            "cash_return": 0.0001,
            "excess_return": 0.0009 + 0.01 * wave,
            "dd_10": wave,
            "sortino_20": np.cos(sequence / 5.0),
            "sortino_60": np.sin(sequence / 7.0),
        }
    )
    raw = pd.Series((sequence // 3) % 2, index=dates, name="hmm_state", dtype=float)
    parent = tmp_path / "parent" / market
    parent.mkdir(parents=True)
    feature_path, raw_path = parent / "features.csv", parent / "hmm-states.csv"
    frame.to_csv(feature_path, index=False)
    raw.to_csv(raw_path)
    source = MarketSource(
        market,
        frame,
        raw,
        dates[300].date(),
        feature_path,
        raw_path,
    )
    jm_grid = tuple(float(value) for value in range(9))
    hmm_grid = tuple(range(9))
    endpoints = audit.EndpointEvidence(9.0, 10.0, 0, 9, 10)
    states = pd.DataFrame(index=dates, columns=(*jm_grid, 9.0), dtype=float)
    for column, candidate in enumerate(states.columns):
        states.iloc[2:, column] = (sequence[2:] // (column + 2) + int(candidate)) % 2
    refits = pd.DataFrame(
        [
            {
                "fit_date": dates[2],
                "training_start": dates[0],
                "training_end": dates[2],
                "observations": 3,
                "scaler_mean": [0.0, 0.0, 0.0],
                "scaler_scale": [1.0, 1.0, 1.0],
                "lambda": candidate,
                "objective": 1.0 + candidate / 100.0,
            }
            for candidate in (*jm_grid, endpoints.jm_endpoint)
        ]
    )
    fit = FixedJMResult(states, refits)
    witness = tmp_path / "base" / market
    _write_witness(witness, source, config, jm_grid, hmm_grid, fit)
    prepared = audit.prepare_market(
        source,
        config,
        endpoints,
        jm_grid,
        hmm_grid,
        witness,
        "7" * 40,
        current_jm=fit,
    )
    prepared_by_market = {
        name: replace(
            prepared,
            market=name,
            behavior_control={**prepared.behavior_control, "market": name},
        )
        for name in audit.MARKETS
    }
    result = audit.finalize_markets(prepared_by_market, config)["us"]
    return (
        config,
        source,
        endpoints,
        jm_grid,
        hmm_grid,
        fit,
        witness,
        prepared,
        result,
    )


def test_current_code_recomputes_full_base_plus_endpoint_before_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, source, endpoints, jm_grid, hmm_grid, fit, witness, _, _ = _small_fixture(
        tmp_path
    )
    calls = []

    def fake_fit(_frame, _model, protocol):
        calls.append(protocol.lambda_grid)
        return fit

    monkeypatch.setattr(audit, "fixed_jm_states", fake_fit)
    prepared = audit.prepare_market(
        source,
        config,
        endpoints,
        jm_grid,
        hmm_grid,
        witness,
        "7" * 40,
    )

    assert calls == [(*jm_grid, endpoints.jm_endpoint)]
    assert prepared.behavior_control["passed"] is True
    assert prepared.behavior_control["counts"]["jm_base_candidates"] == 9
    assert prepared.behavior_control["counts"]["hmm_base_candidates"] == 9
    assert "base_jm_refits" in prepared.behavior_control["current_hashes"]
    assert all(
        component in prepared.behavior_control["current_hashes"]
        for component in (
            "fixed_jm_delay_1_cv_surface",
            "fixed_jm_delay_1_candidate_returns",
            "hmm_delay_1_selected_signal",
            "base_boundaries",
        )
    )


def test_global_gate_precedes_any_accounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, *_, prepared, _result = _small_fixture(tmp_path)
    markets = {
        market: replace(
            prepared,
            market=market,
            behavior_control={**prepared.behavior_control, "market": market},
        )
        for market in audit.MARKETS
    }
    failed = markets["jp"].behavior_control.copy()
    failed["passed"] = False
    markets["jp"] = replace(markets["jp"], behavior_control=failed)
    opened = []
    monkeypatch.setattr(
        audit,
        "_finalize_market",
        lambda *_args: opened.append(True),
    )

    with pytest.raises(audit.EndpointGridError, match="all-market"):
        audit.finalize_markets(markets, config)
    assert opened == []


def test_five_paths_t_plus_2_ten_bps_and_paper_turnover(tmp_path: Path) -> None:
    config, *_, result = _small_fixture(tmp_path)
    assert tuple(result.paths[1]) == audit.PATHS
    path = result.paths[1]["J1"]
    assert (
        path["position"]
        .iloc[2:]
        .reset_index(drop=True)
        .equals(path["signal"].iloc[:-2].reset_index(drop=True))
    )
    assert np.allclose(
        path["transaction_cost"],
        path["one_way_turnover"] * 10 / 10_000,
        rtol=0,
        atol=1e-15,
    )
    metric = result.metrics.set_index("path").loc["J1"]
    assert metric["turnover"] == pytest.approx(
        0.5 * path["one_way_turnover"].mean() * 252
    )


def test_smoke_uses_frozen_numerical_thread_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, source, endpoints, _, _, fit, _, _, _ = _small_fixture(tmp_path)
    seen = []
    endpoint = FixedJMResult(fit.states[[endpoints.jm_endpoint]], fit.refits.tail(1))

    def smoke_fit(frame, *_args):
        dates = pd.DatetimeIndex(pd.to_datetime(frame["date"]), name="date")
        return FixedJMResult(endpoint.states.reindex(dates), endpoint.refits)

    monkeypatch.setattr(runner_smoke, "fixed_jm_states", smoke_fit)
    monkeypatch.setattr(
        runner_smoke,
        "threadpool_limits",
        lambda *, limits: seen.append(limits) or nullcontext(),
    )
    runner_smoke.run_us_smoke(source, config, endpoints, 20, numerical_threads=1)
    assert seen == [1]


def test_parallel_executor_uses_frozen_context_and_three_submissions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}
    context = object()

    class Executor:
        def __init__(self, *, max_workers, mp_context):
            captured["init"] = (max_workers, mp_context)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def submit(self, fn, task):
            captured.setdefault("tasks", []).append(task[0])
            future = Future()
            future.set_result(fn(task))
            return future

    def fake_context(method):
        captured["method"] = method
        return context

    monkeypatch.setattr(audit, "ProcessPoolExecutor", Executor)
    monkeypatch.setattr(audit, "get_context", fake_context)
    monkeypatch.setattr(audit, "_market_worker", lambda task: task[0])
    spec = SimpleNamespace(market_workers=3, process_start_method="forkserver")
    tasks = [(market, None, None, None, "", 1) for market in audit.MARKETS]

    parallel = audit._prepare_markets_parallel(tasks, spec)

    assert captured["init"] == (3, context)
    assert captured["method"] == "forkserver"
    assert captured["tasks"] == list(audit.MARKETS)
    assert parallel == {market: market for market in audit.MARKETS}


def test_worker_applies_one_thread_and_matches_serial_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, source, endpoints, jm_grid, hmm_grid, fit, witness, prepared, _ = (
        _small_fixture(tmp_path)
    )
    lineage = audit._Lineage(
        tmp_path / "parent", tmp_path / "base", jm_grid, hmm_grid, endpoints
    )
    seen = []
    monkeypatch.setattr(
        input_verifier,
        "load_market_source",
        lambda *_args: source,
    )
    monkeypatch.setattr(
        audit,
        "threadpool_limits",
        lambda *, limits: seen.append(limits) or nullcontext(),
    )
    monkeypatch.setattr(
        audit,
        "prepare_market",
        lambda *_args, **_kwargs: prepared,
    )
    task = ("us", lineage.parent_dir, config, lineage, "7" * 40, 1)

    parallel = audit._market_worker(task)

    assert seen == [1]
    pd.testing.assert_frame_equal(parallel.endpoint_jm, prepared.endpoint_jm)


def _synthetic_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    (
        config,
        source,
        endpoints,
        jm_grid,
        hmm_grid,
        fit,
        witness,
        prepared,
        _,
    ) = _small_fixture(tmp_path)
    config_lock = tmp_path / "config.lock.toml"
    config_lock.write_bytes(b"synthetic config")
    (tmp_path / "research.toml").write_bytes(config_lock.read_bytes())
    config_hash = hashlib.sha256(config_lock.read_bytes()).hexdigest()
    config = replace(config, path=config_lock, sha256=config_hash)
    study_lock = tmp_path / "study.lock.toml"
    study_lock.write_bytes(b"synthetic spec")
    spec_hash = hashlib.sha256(study_lock.read_bytes()).hexdigest()
    spec = audit.EndpointGridSpec(
        study_lock,
        spec_hash,
        "FROZEN",
        "endpoint-grid-audit-test",
        "parent",
        "1" * 64,
        "2" * 64,
        "calibration",
        "3" * 64,
        "4" * 64,
        "5" * 64,
        "base",
        "6" * 64,
        "8" * 64,
        Path("base"),
        20,
        Path("endpoint-grid-audit"),
        "forkserver",
        3,
        1,
    )
    base_dir = tmp_path / "base"
    for market in audit.MARKETS:
        target = base_dir / market
        if target != witness:
            import shutil

            shutil.copytree(witness, target)
    lineage = audit._Lineage(
        tmp_path / "parent", base_dir, jm_grid, hmm_grid, endpoints
    )
    sources = {market: replace(source, market=market) for market in audit.MARKETS}
    prepared_markets = {
        market: replace(
            prepared,
            market=market,
            behavior_control={**prepared.behavior_control, "market": market},
        )
        for market in audit.MARKETS
    }
    results = audit.finalize_markets(prepared_markets, config)
    control = audit.behavior_control_receipt(prepared_markets)
    git_sha = "7" * 40
    run_id = f"endpoint-grid-audit-{spec_hash[:12]}-{'4' * 12}-{git_sha[:12]}"
    run_dir = tmp_path / config.artifact_root / spec.artifact_subdir / run_id
    endpoint_states = fit.states[[endpoints.jm_endpoint]]
    prefix = endpoint_states[endpoints.jm_endpoint].dropna().iloc[:20]
    hmm_prefix = smoothed_hmm_states(source.raw_hmm, (endpoints.hmm_endpoint,))[
        endpoints.hmm_endpoint
    ].reindex(prefix.index)
    smoke = {
        "status": "passed",
        "market": "us",
        "terminal_dates": 20,
        "jm_endpoint": endpoints.jm_endpoint,
        "hmm_endpoint": endpoints.hmm_endpoint,
        "state_dates": [value.date().isoformat() for value in prefix.index],
        "jm_states": [int(value) for value in prefix],
        "hmm_states": [int(value) for value in hmm_prefix],
        "jm_observations": 20,
        "hmm_observations": 20,
        "performance_metrics_opened": False,
        "strategy_return_columns_accessed": False,
    }
    audit._write_run(
        run_dir,
        config,
        spec,
        lineage,
        git_sha,
        smoke,
        control,
        results,
        config,
    )
    monkeypatch.setattr(artifact_verifier, "load_config", lambda _path: config)
    monkeypatch.setattr(
        artifact_verifier, "load_endpoint_grid_spec", lambda *_args: spec
    )
    monkeypatch.setattr(artifact_verifier, "verify_lineage", lambda *_args: lineage)
    monkeypatch.setattr(
        artifact_verifier,
        "load_market_source",
        lambda _parent, market, *_args: sources[market],
    )
    monkeypatch.setattr(artifact_verifier, "research_git_sha", lambda _root: git_sha)

    def fake_fit(frame, _model, protocol):
        dates = pd.DatetimeIndex(pd.to_datetime(frame["date"]), name="date")
        columns = list(protocol.lambda_grid)
        states = fit.states.loc[:, columns].reindex(dates)
        refits = fit.refits.loc[fit.refits["lambda"].isin(columns)].copy()
        return FixedJMResult(states, refits)

    monkeypatch.setattr(replay, "fixed_jm_states", fake_fit)
    return run_dir, git_sha


def test_independent_verifier_replays_synthetic_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _ = _synthetic_artifact(tmp_path, monkeypatch)
    receipt = artifact_verifier.verify_endpoint_grid_run(run_dir)
    assert receipt["metric_rows"] == 15
    assert receipt["boundary_rows"] == 12
