"""One-shot audit of the first invalid-bracket endpoint for fixed JM and HMM."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import pandas as pd
from threadpoolctl import threadpool_limits

from adaptive_jump.artifacts import read_json, write_inventory, write_json
from adaptive_jump.config import PAPER_TURNOVER_DEFINITION, ResearchConfig
from adaptive_jump.data import research_git_sha
from adaptive_jump.endpoint_grid_accounting import (
    accounting_paths,
    path_metrics,
    write_market,
)
from adaptive_jump.endpoint_grid_control import (
    verify_selection_parity,
    verify_state_parity,
)
from adaptive_jump.endpoint_grid_decision import (
    d_rescue_decision,
    endpoint_effects,
)
from adaptive_jump.endpoint_grid_evidence import classify_path_changes
from adaptive_jump.endpoint_grid_smoke import (
    SMOKE_TERMINAL_DATES,
    run_us_smoke,
    verify_full_smoke_prefix,
)
from adaptive_jump.endpoint_grid_types import (
    EndpointEvidence,
    EndpointGridError,
    EndpointGridSpec,
    MarketSource,
    _Lineage,
)
from adaptive_jump.models import (
    FEATURE_COLUMNS,
    FixedJMResult,
    fixed_jm_states,
    smoothed_hmm_states,
)
from adaptive_jump.walkforward import (
    SelectionResult,
    boundary_diagnostic,
    select_monthly_candidate,
)

MARKETS = ("us", "de", "jp")
PATHS = ("buy_and_hold", "J0", "J1", "K0", "K1")
SELECTION_PATHS = PATHS[1:]
CELL_PATHS = {
    "A": {"fixed_jm": "J0", "hmm": "K0"},
    "B": {"fixed_jm": "J1", "hmm": "K0"},
    "C": {"fixed_jm": "J0", "hmm": "K1"},
    "D": {"fixed_jm": "J1", "hmm": "K1"},
}


@dataclass(frozen=True)
class PreparedMarket:
    """Performance-free current-code reconstruction for one market."""

    market: str
    endpoint_jm: pd.DataFrame
    endpoint_refits: pd.DataFrame
    selections: dict[str, dict[int, SelectionResult]]
    boundaries: pd.DataFrame
    returns: pd.DataFrame
    oos_start: date
    behavior_control: dict[str, Any]


@dataclass(frozen=True)
class MarketResult:
    endpoint_jm: pd.DataFrame
    endpoint_refits: pd.DataFrame
    selections: dict[str, dict[int, SelectionResult]]
    paths: dict[int, dict[str, pd.DataFrame]]
    boundaries: pd.DataFrame
    metrics: pd.DataFrame
    behavior_control: dict[str, Any]


def prepare_market(
    source: MarketSource,
    config: ResearchConfig,
    endpoints: EndpointEvidence,
    jm_grid: tuple[float, ...],
    hmm_grid: tuple[int, ...],
    witness_dir: Path,
    current_git_sha: str,
    *,
    current_jm: FixedJMResult | None = None,
) -> PreparedMarket:
    """Recompute current base+endpoint behavior without constructing P&L paths."""
    if (
        len(jm_grid) != 9
        or len(hmm_grid) != 9
        or endpoints.jm_endpoint in jm_grid
        or endpoints.hmm_endpoint in hmm_grid
    ):
        raise EndpointGridError(
            "endpoint preparation requires two nine-point base grids"
        )
    full_jm_grid = (*jm_grid, endpoints.jm_endpoint)
    if current_jm is None:
        model_frame = source.frame.loc[:, ("date", *FEATURE_COLUMNS, "excess_return")]
        protocol = replace(config.jm_protocol, lambda_grid=full_jm_grid)
        current_jm = fixed_jm_states(model_frame, config.model_protocol, protocol)
    dates = pd.DatetimeIndex(source.frame["date"], name="date")
    expected_columns = tuple(float(value) for value in full_jm_grid)
    if tuple(
        current_jm.states.columns
    ) != expected_columns or not current_jm.states.index.equals(dates):
        raise EndpointGridError("current JM state matrix does not cover the full grid")
    complete = current_jm.states.dropna(how="all").stack()
    if complete.empty or not complete.isin((0.0, 1.0)).all():
        raise EndpointGridError("current JM state matrix is invalid")
    base_jm = current_jm.states.loc[:, list(jm_grid)]
    endpoint_jm = current_jm.states.loc[:, [endpoints.jm_endpoint]]
    lambdas = pd.to_numeric(current_jm.refits["lambda"], errors="raise")
    base_refits = current_jm.refits.loc[lambdas.isin(jm_grid)].reset_index(drop=True)
    endpoint_refits = current_jm.refits.loc[
        lambdas.eq(endpoints.jm_endpoint)
    ].reset_index(drop=True)
    if base_refits.empty or endpoint_refits.empty:
        raise EndpointGridError("current JM refits do not cover base and endpoint")

    base_hmm = smoothed_hmm_states(source.raw_hmm, hmm_grid).reindex(dates)
    endpoint_hmm = smoothed_hmm_states(
        source.raw_hmm, (endpoints.hmm_endpoint,)
    ).reindex(dates)
    receipt = verify_state_parity(
        source,
        witness_dir,
        base_jm,
        base_refits,
        base_hmm,
        current_git_sha,
    )
    candidates = {
        "J0": base_jm,
        "J1": pd.concat([base_jm, endpoint_jm], axis=1),
        "K0": base_hmm,
        "K1": pd.concat([base_hmm, endpoint_hmm], axis=1),
    }
    returns = source.frame.loc[:, ["date", "equity_simple", "cash_return"]]
    selections = {path: {} for path in SELECTION_PATHS}
    boundary_rows: list[dict[str, Any]] = []
    for delay in config.backtest_protocol.robustness_delays:
        for path, states in candidates.items():
            selection = select_monthly_candidate(
                returns,
                states,
                config.selection_protocol,
                delay_trading_days=delay,
                one_way_cost_bps=config.backtest_protocol.one_way_cost_bps,
                periods_per_year=config.metrics_protocol.periods_per_year,
                volatility_ddof=config.metrics_protocol.volatility_ddof,
            )
            selections[path][delay] = selection
            diagnostic = boundary_diagnostic(
                selection.choices,
                tuple(float(value) for value in states.columns),
                oos_start=source.oos_start,
                fraction_limit=config.selection_protocol.boundary_fraction_limit,
            )
            boundary_rows.append(
                {
                    "path": path,
                    "delay": delay,
                    **diagnostic.__dict__,
                    "descriptive_only": True,
                }
            )
    boundaries = pd.DataFrame.from_records(boundary_rows)
    receipt = verify_selection_parity(
        receipt,
        witness_dir,
        selections,
        boundaries,
        config.backtest_protocol.robustness_delays,
    )
    return PreparedMarket(
        source.market,
        endpoint_jm,
        endpoint_refits,
        selections,
        boundaries,
        returns,
        source.oos_start,
        receipt,
    )


def finalize_markets(
    prepared: dict[str, PreparedMarket], config: ResearchConfig
) -> dict[str, MarketResult]:
    """Open accounting only after all three market parity receipts pass."""
    if set(prepared) != set(MARKETS) or any(
        item.market != market or item.behavior_control.get("passed") is not True
        for market, item in prepared.items()
    ):
        raise EndpointGridError(
            "all-market selection-behavior parity must pass before accounting"
        )
    return {market: _finalize_market(prepared[market], config) for market in MARKETS}


def _finalize_market(item: PreparedMarket, config: ResearchConfig) -> MarketResult:
    paths = accounting_paths(item.returns, item.selections, item.oos_start, config)
    return MarketResult(
        item.endpoint_jm,
        item.endpoint_refits,
        item.selections,
        paths,
        item.boundaries,
        path_metrics(paths, config),
        item.behavior_control,
    )


def behavior_control_receipt(
    prepared: dict[str, PreparedMarket],
) -> dict[str, Any]:
    markets = [prepared[market].behavior_control for market in MARKETS]
    passed = len(markets) == len(MARKETS) and all(
        row.get("passed") is True for row in markets
    )
    if not passed:
        raise EndpointGridError(
            "cannot seal a failed selection-behavior parity receipt"
        )
    return {
        "schema_version": 1,
        "mode": "global-current-code-selection-behavior-exact-parity",
        "markets": markets,
        "all_markets_passed": True,
        "accounting_allowed_only_after_all_markets_passed": True,
    }


def run_endpoint_grid_audit(config: ResearchConfig, spec: EndpointGridSpec) -> Path:
    """Run US smoke, all-market parity, then accounting in that order."""
    from adaptive_jump.endpoint_grid_verifier import load_market_source, verify_lineage

    if (
        spec.protocol_status != "FROZEN"
        or spec.smoke_terminal_dates != SMOKE_TERMINAL_DATES
        or (spec.process_start_method, spec.market_workers, spec.numerical_threads)
        != ("forkserver", 3, 1)
    ):
        raise EndpointGridError("endpoint-grid audit execution contract is not frozen")
    lineage = verify_lineage(config, spec)
    root = config.path.parent
    git_sha = research_git_sha(root)
    evaluated = replace(
        config,
        metrics_protocol=replace(
            config.metrics_protocol,
            turnover_definition=PAPER_TURNOVER_DEFINITION,
        ),
    )
    us_source = load_market_source(lineage.parent_dir, "us", evaluated, lineage)
    smoke = run_us_smoke(
        us_source,
        evaluated,
        lineage.endpoints,
        spec.smoke_terminal_dates,
        numerical_threads=spec.numerical_threads,
    )
    run_id = "endpoint-grid-audit-" + "-".join(
        (spec.sha256[:12], spec.calibration_inventory_sha256[:12], git_sha[:12])
    )
    run_dir = root / config.artifact_root / spec.artifact_subdir / run_id
    if run_dir.exists():
        from adaptive_jump.endpoint_grid_artifact_verifier import (
            verify_endpoint_grid_run,
        )

        verify_endpoint_grid_run(run_dir)
        return run_dir
    tasks = [
        (
            market,
            lineage.parent_dir,
            evaluated,
            lineage,
            git_sha,
            spec.numerical_threads,
        )
        for market in MARKETS
    ]
    prepared = _prepare_markets_parallel(tasks, spec)
    verify_full_smoke_prefix(
        us_source,
        prepared["us"].endpoint_jm,
        lineage.endpoints,
        smoke,
        spec.smoke_terminal_dates,
    )
    results = finalize_markets(prepared, evaluated)
    control = behavior_control_receipt(prepared)
    _write_run(
        run_dir, config, spec, lineage, git_sha, smoke, control, results, evaluated
    )
    from adaptive_jump.endpoint_grid_artifact_verifier import verify_endpoint_grid_run

    verify_endpoint_grid_run(run_dir)
    return run_dir


def _prepare_markets_parallel(
    tasks: list[tuple[str, Path, ResearchConfig, _Lineage, str, int]],
    spec: EndpointGridSpec,
) -> dict[str, PreparedMarket]:
    results: dict[str, PreparedMarket] = {}
    with ProcessPoolExecutor(
        max_workers=spec.market_workers,
        mp_context=get_context(spec.process_start_method),
    ) as pool:
        futures = {pool.submit(_market_worker, task): task[0] for task in tasks}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return results


def _market_worker(
    task: tuple[str, Path, ResearchConfig, _Lineage, str, int],
) -> PreparedMarket:
    from adaptive_jump.endpoint_grid_verifier import load_market_source

    market, parent_dir, config, lineage, git_sha, numerical_threads = task
    with threadpool_limits(limits=numerical_threads):
        source = load_market_source(parent_dir, market, config, lineage)
        return prepare_market(
            source,
            config,
            lineage.endpoints,
            lineage.jm_grid,
            lineage.hmm_grid,
            lineage.base_dir / market,
            git_sha,
        )


def _write_run(
    run_dir: Path,
    config: ResearchConfig,
    spec: EndpointGridSpec,
    lineage: _Lineage,
    git_sha: str,
    smoke: dict[str, Any],
    control: dict[str, Any],
    results: dict[str, MarketResult],
    evaluated: ResearchConfig,
) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "study.lock.toml").write_bytes(spec.path.read_bytes())
    (run_dir / "config.lock.toml").write_bytes(config.path.read_bytes())
    write_json(run_dir / "endpoint-provenance.json", lineage.endpoints.as_dict())
    write_json(run_dir / "behavior-control.json", control)
    write_json(run_dir / "us-smoke.json", smoke)
    write_json(
        run_dir / "run.json",
        {
            "schema_version": 1,
            "study_kind": "endpoint_grid_audit",
            "experiment_id": spec.experiment_id,
            "run_id": run_dir.name,
            "status": "running",
            "claim_class": "EXPLORATORY",
            "performance_claim_allowed": False,
            "post_2023_accessed": False,
            "boundary_descriptive_only": True,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "spec_sha256": spec.sha256,
            "config_sha256": config.sha256,
            "calibration_inventory_sha256": spec.calibration_inventory_sha256,
            "data_manifest_sha256": spec.data_manifest_sha256,
            "git_sha": git_sha,
            "execution": {
                "process_start_method": spec.process_start_method,
                "market_workers": spec.market_workers,
                "numerical_threads": spec.numerical_threads,
            },
        },
    )
    metrics, boundaries, changes, traces = [], [], [], []
    for market in MARKETS:
        result = results[market]
        write_market(run_dir / market, result)
        metrics.append(result.metrics.assign(market=market))
        boundaries.append(result.boundaries.assign(market=market))
        change, trace = classify_path_changes(
            result.selections,
            result.paths,
            result.metrics,
            market,
            evaluated.backtest_protocol.return_offset,
        )
        changes.append(change)
        traces.append(trace)
    all_metrics = pd.concat(metrics, ignore_index=True)
    all_boundaries = pd.concat(boundaries, ignore_index=True)
    all_metrics.to_csv(run_dir / "metrics.csv", index=False)
    all_boundaries.to_csv(run_dir / "boundaries.csv", index=False)
    endpoint_effects(all_metrics).to_csv(run_dir / "endpoint-effects.csv", index=False)
    pd.concat(changes, ignore_index=True).to_csv(
        run_dir / "path-changes.csv", index=False
    )
    pd.concat(traces, ignore_index=True).to_csv(
        run_dir / "change-traces.csv", index=False
    )
    write_json(
        run_dir / "composition.json",
        {"cells": CELL_PATHS, "materialized_paths": list(PATHS), "passed": True},
    )
    binding = all_boundaries.loc[
        all_boundaries["path"].isin(("J1", "K1")), "passed"
    ].eq(False)
    decision = d_rescue_decision(all_metrics)
    decision.update(
        {
            "endpoint_concentration_present": bool(binding.any()),
            "finite_optimum_identified": False if binding.any() else None,
        }
    )
    write_json(run_dir / "decision.json", decision)
    metadata = read_json(run_dir / "run.json")
    metadata.update(status="complete", finished_at_utc=datetime.now(UTC).isoformat())
    write_json(run_dir / "run.json", metadata)
    write_inventory(run_dir)
