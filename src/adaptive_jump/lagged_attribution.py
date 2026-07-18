"""Post-result 2x2 attribution of lagged state paths and lambda choices."""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, date, datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from adaptive_jump.artifacts import (
    read_json,
    read_trade_path,
    sha256_file,
    verify_inventory,
    write_inventory,
    write_json,
)
from adaptive_jump.confidence_evaluation import _active_lambda, _align_parent_sample
from adaptive_jump.config import ResearchConfig, load_config
from adaptive_jump.lagged_performance import (
    LAMBDAS,
    MARKETS,
    TURNOVER_SCALE,
    LaggedPerformanceSpec,
    SourcePaths,
    _assert_frame_close,
    _full_path,
    _git_head,
    _load_market,
    _metric_row,
    _verify_sources,
    load_lagged_performance_spec,
)
from adaptive_jump.walkforward import SelectionResult, _compose_selected_signal

CELLS = ("FF", "FL", "LF", "LL")
METRICS = (
    "sharpe",
    "maximum_drawdown",
    "turnover",
    "cash_fraction",
    "switch_count",
)
CELL_SOURCES = {
    "FF": ("fixed", "fixed"),
    "FL": ("fixed", "lagged_log4"),
    "LF": ("lagged_log4", "fixed"),
    "LL": ("lagged_log4", "lagged_log4"),
}


class AttributionError(RuntimeError):
    """Raised when the frozen attribution contract is violated."""


@dataclass(frozen=True)
class AttributionSpec:
    path: Path
    sha256: str
    experiment_id: str
    parent_run_id: str
    parent_inventory_sha256: str
    parent_choices_sha256: str
    parent_spec_sha256: str
    parent_implementation_sha256: str
    fixed_inventory_sha256: str
    lagged_inventory_sha256: str
    cutoff: date
    markets: tuple[str, ...]
    lambdas: tuple[float, ...]
    cells: tuple[str, ...]
    artifact_subdir: Path
    identity_tolerance: float


@dataclass(frozen=True)
class AttributionInputs:
    parent: Path
    parent_spec: LaggedPerformanceSpec
    sources: SourcePaths
    choices: pd.DataFrame


def load_attribution_spec(path: str | Path, config: ResearchConfig) -> AttributionSpec:
    spec_path = Path(path).resolve()
    payload = spec_path.read_bytes()
    try:
        doc = tomllib.loads(payload.decode())
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise AttributionError(f"invalid attribution spec: {exc}") from exc
    parent = doc.get("parent", {})
    sources = doc.get("state_sources", {})
    cells = doc.get("cells", {})
    protocol = doc.get("protocol", {})
    attribution = doc.get("attribution", {})
    decision = doc.get("decision", {})
    verification = doc.get("verification", {})
    storage_doc = doc.get("storage", {})
    required = (
        doc.get("schema_version") == 1,
        doc.get("experiment_id") == "lagged-selection-attribution-001",
        doc.get("claim_class") == "EXPLORATORY",
        doc.get("stage") == "POST_RESULT_MECHANICAL_ATTRIBUTION",
        doc.get("performance_claim_allowed") is False,
        doc.get("paper_replication_claim_allowed") is False,
        doc.get("causal_claim_allowed") is False,
        doc.get("post_2023_access") is False,
        parent.get("config_sha256") == config.sha256,
        parent.get("data_cutoff") == "2023-12-31",
        tuple(float(value) for value in sources.get("raw_lambda_grid", ())) == LAMBDAS,
        tuple(key for key in cells if key in CELL_SOURCES) == CELLS,
        cells.get("first_letter") == "candidate-state family: F=fixed, L=lagged-log4",
        cells.get("second_letter") == "monthly-choice schedule: F=fixed, L=lagged-log4",
        tuple(protocol.get("markets", ())) == MARKETS,
        protocol.get("primary_delay_trading_days")
        == config.backtest_protocol.primary_delay,
        protocol.get("signal_to_return_offset")
        == config.backtest_protocol.return_offset,
        protocol.get("one_way_cost_bps") == config.backtest_protocol.one_way_cost_bps,
        TURNOVER_SCALE == 0.5,
        protocol.get("no_new_selection") is True,
        protocol.get("no_grid_expansion") is True,
        attribution.get("percentage_attribution") is False,
        decision.get("result") == "diagnostic_complete",
        decision.get("supported_or_not_supported_forbidden") is True,
        decision.get("cell_winner_selection_forbidden") is True,
        verification.get("independent_source_replay") is True,
        storage_doc.get("artifact_subdir") == "lagged-selection-attribution-001",
    )
    if not all(required):
        raise AttributionError("attribution controls changed")
    artifact_subdir = Path(storage_doc["artifact_subdir"])
    return AttributionSpec(
        path=spec_path,
        sha256=hashlib.sha256(payload).hexdigest(),
        experiment_id=doc["experiment_id"],
        parent_run_id=str(parent["run_id"]),
        parent_inventory_sha256=str(parent["run_inventory_sha256"]),
        parent_choices_sha256=str(parent["choices_sha256"]),
        parent_spec_sha256=str(parent["spec_sha256"]),
        parent_implementation_sha256=str(parent["implementation_sha256"]),
        fixed_inventory_sha256=str(sources["fixed_inventory_sha256"]),
        lagged_inventory_sha256=str(sources["lagged_inventory_sha256"]),
        cutoff=date.fromisoformat(parent["data_cutoff"]),
        markets=tuple(protocol["markets"]),
        lambdas=tuple(float(value) for value in sources["raw_lambda_grid"]),
        cells=tuple(key for key in cells if key in CELL_SOURCES),
        artifact_subdir=artifact_subdir,
        identity_tolerance=float(verification["shapley_identity_absolute_tolerance"]),
    )


def _registry_lock(root: Path, spec: AttributionSpec) -> None:
    rows = [
        json.loads(line)
        for line in (root / "research/experiment_registry.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if json.loads(line).get("experiment_id") == spec.experiment_id
    ]
    if (
        not rows
        or rows[-1].get("status") not in {"FROZEN", "EXPERIMENT_COMPLETE"}
        or rows[-1].get("frozen_spec_hash") != spec.sha256
    ):
        raise AttributionError("attribution registry lock changed")


def _load_inputs(
    root: Path, config: ResearchConfig, spec: AttributionSpec
) -> AttributionInputs:
    _registry_lock(root, spec)
    parent_spec = load_lagged_performance_spec(
        root / "research/lagged-evidence-performance-001.toml", config
    )
    if (
        parent_spec.sha256 != spec.parent_spec_sha256
        or parent_spec.fixed_inventory_sha256 != spec.fixed_inventory_sha256
        or parent_spec.lagged_inventory_sha256 != spec.lagged_inventory_sha256
    ):
        raise AttributionError("parent study or inherited state sources changed")
    parent = (
        root / config.artifact_root / parent_spec.artifact_subdir / spec.parent_run_id
    )
    if (
        sha256_file(parent / "inventory.json") != spec.parent_inventory_sha256
        or sha256_file(parent / "choices.csv") != spec.parent_choices_sha256
    ):
        raise AttributionError("parent attribution source identity changed")
    metadata = read_json(parent / "run.json")
    if (
        metadata.get("study_kind") != "lagged_evidence_performance"
        or metadata.get("experiment_id") != "lagged-evidence-performance-001"
        or metadata.get("run_id") != spec.parent_run_id
        or metadata.get("implementation_sha256") != spec.parent_implementation_sha256
        or metadata.get("spec_sha256") != spec.parent_spec_sha256
        or metadata.get("config_sha256") != config.sha256
        or metadata.get("status") != "complete"
        or metadata.get("decision") != "supported"
        or metadata.get("post_2023_accessed") is not False
        or metadata.get("performance_claim_allowed") is not False
        or metadata.get("paper_replication_claim_allowed") is not False
    ):
        raise AttributionError("parent metadata changed")
    verify_inventory(parent)
    sources = _verify_sources(root, config, parent_spec)
    choices = pd.read_csv(parent / "choices.csv")
    required_columns = ("decision_date", "selected", "market", "model")
    if tuple(choices.columns) != required_columns or choices.empty:
        raise AttributionError("parent choice schema changed")
    choices["decision_date"] = pd.to_datetime(choices["decision_date"], errors="raise")
    expected = {
        (market, model) for market in MARKETS for model in ("fixed", "lagged_log4")
    }
    actual = set(
        choices.loc[
            choices["model"].isin(("fixed", "lagged_log4")), ["market", "model"]
        ].itertuples(index=False, name=None)
    )
    if actual != expected:
        raise AttributionError("parent choices lack a frozen state/choice source")
    return AttributionInputs(parent, parent_spec, sources, choices)


def _schedule(inputs: AttributionInputs, market: str, model: str) -> pd.DataFrame:
    rows = inputs.choices.loc[
        (inputs.choices["market"] == market) & (inputs.choices["model"] == model),
        ["decision_date", "selected"],
    ].reset_index(drop=True)
    if (
        rows.empty
        or rows["decision_date"].duplicated().any()
        or not rows["decision_date"].is_monotonic_increasing
        or not rows["selected"].isin(LAMBDAS).all()
    ):
        raise AttributionError(f"{market}/{model}: invalid frozen choice schedule")
    return rows


def _cell_selections(
    frame: pd.DataFrame,
    states: dict[str, pd.DataFrame],
    inputs: AttributionInputs,
    market: str,
) -> dict[str, SelectionResult]:
    dates = pd.DatetimeIndex(pd.to_datetime(frame["date"], errors="raise"))
    schedules = {
        model: _schedule(inputs, market, model) for model in ("fixed", "lagged_log4")
    }
    selections: dict[str, SelectionResult] = {}
    for cell, (state_source, choice_source) in CELL_SOURCES.items():
        signal = _compose_selected_signal(
            dates, states[state_source], schedules[choice_source]
        )
        selections[cell] = SelectionResult(
            signal=signal,
            choices=schedules[choice_source].copy(),
            surface=pd.DataFrame(),
            candidate_returns=pd.DataFrame(index=dates),
        )
    return selections


def _add_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    baseline = metrics.loc[metrics["cell"] == "FF"]
    if len(baseline) != 1:
        raise AttributionError("market lacks a unique FF metric row")
    output = metrics.copy()
    for metric in METRICS:
        output[f"delta_{metric}"] = output[metric] - baseline.iloc[0][metric]
    return output


def _attribution_rows(summary: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for market in MARKETS:
        indexed = summary.loc[summary["market"] == market].set_index("cell")
        if set(indexed.index) != set(CELLS) or len(indexed) != len(CELLS):
            raise AttributionError(f"{market}: incomplete 2x2 metrics")
        for metric in METRICS:
            values = {cell: float(indexed.loc[cell, metric]) for cell in CELLS}
            total = values["LL"] - values["FF"]
            path_fixed = values["LF"] - values["FF"]
            choice_fixed = values["FL"] - values["FF"]
            interaction = values["LL"] - values["LF"] - values["FL"] + values["FF"]
            path_shapley = 0.5 * (
                values["LF"] - values["FF"] + values["LL"] - values["FL"]
            )
            choice_shapley = 0.5 * (
                values["FL"] - values["FF"] + values["LL"] - values["LF"]
            )
            error = path_shapley + choice_shapley - total
            if abs(error) > tolerance:
                raise AttributionError(f"{market}/{metric}: Shapley identity failed")
            records.append(
                {
                    "market": market,
                    "metric": metric,
                    "total": total,
                    "path_at_fixed_choices": path_fixed,
                    "choice_at_fixed_path": choice_fixed,
                    "interaction": interaction,
                    "path_shapley": path_shapley,
                    "choice_shapley": choice_shapley,
                    "identity_error": error,
                }
            )
    frame = pd.DataFrame.from_records(records)
    means = (
        frame.groupby("metric", sort=False)[
            [
                "total",
                "path_at_fixed_choices",
                "choice_at_fixed_path",
                "interaction",
                "path_shapley",
                "choice_shapley",
                "identity_error",
            ]
        ]
        .mean()
        .reset_index()
    )
    means.insert(0, "market", "equal_market_mean")
    return pd.concat([frame, means], ignore_index=True)


def _change_traces(
    market: str,
    frame: pd.DataFrame,
    selections: dict[str, SelectionResult],
    full: dict[str, pd.DataFrame],
    aligned: dict[str, pd.DataFrame],
    offset: int,
) -> pd.DataFrame:
    dates = pd.DatetimeIndex(pd.to_datetime(frame["date"], errors="raise"))
    active = {cell: _active_lambda(dates, selections[cell].choices) for cell in CELLS}
    first_execution = pd.Timestamp(aligned["FF"]["date"].iloc[0])
    records: list[dict[str, Any]] = []
    for cell in CELLS[1:]:
        seen: set[str] = set()
        for row in range(len(dates) - offset):
            execution = row + offset
            execution_date = pd.Timestamp(full[cell].iloc[execution]["date"])
            if execution_date < first_execution:
                continue
            ff_trade = (
                full["FF"].iloc[execution]["position"]
                - full["FF"].iloc[execution - 1]["position"]
            )
            cell_trade = (
                full[cell].iloc[execution]["position"]
                - full[cell].iloc[execution - 1]["position"]
            )
            flags = {
                "choice": active[cell].iloc[row] != active["FF"].iloc[row],
                "state": selections[cell].signal.iloc[row]
                != selections["FF"].signal.iloc[row],
                "position": full[cell].iloc[execution]["position"]
                != full["FF"].iloc[execution]["position"],
                "trade": (
                    np.isfinite(cell_trade)
                    and np.isfinite(ff_trade)
                    and cell_trade != ff_trade
                ),
            }
            for event, changed in flags.items():
                if not changed or event in seen:
                    continue
                base = full["FF"].iloc[execution]
                counterfactual = full[cell].iloc[execution]
                records.append(
                    {
                        "market": market,
                        "cell": cell,
                        "event_type": event,
                        "signal_date": dates[row],
                        "execution_date": execution_date,
                        "offset_observations": offset,
                        "ff_lambda": float(active["FF"].iloc[row]),
                        "cell_lambda": float(active[cell].iloc[row]),
                        "ff_state": int(1.0 - selections["FF"].signal.iloc[row]),
                        "cell_state": int(1.0 - selections[cell].signal.iloc[row]),
                        "ff_signal": float(selections["FF"].signal.iloc[row]),
                        "cell_signal": float(selections[cell].signal.iloc[row]),
                        "ff_position": float(base["position"]),
                        "cell_position": float(counterfactual["position"]),
                        "ff_trade": float(ff_trade),
                        "cell_trade": float(cell_trade),
                        "ff_turnover": float(base["one_way_turnover"]),
                        "cell_turnover": float(counterfactual["one_way_turnover"]),
                        "ff_cost": float(base["transaction_cost"]),
                        "cell_cost": float(counterfactual["transaction_cost"]),
                    }
                )
                seen.add(event)
            if len(seen) == 4:
                break
    return pd.DataFrame.from_records(records)


def _replay_market(
    market: str,
    inputs: AttributionInputs,
    config: ResearchConfig,
    spec: AttributionSpec,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, pd.DataFrame],
    dict[str, SelectionResult],
]:
    frame, source_states = _load_market(market, inputs.sources, inputs.parent_spec)
    states = {
        "fixed": source_states["fixed"],
        "lagged_log4": source_states["lagged_log4"],
    }
    selections = _cell_selections(frame, states, inputs, market)
    full = {cell: _full_path(frame, selections[cell], config) for cell in CELLS}
    aligned = {
        cell: _align_parent_sample(
            inputs.sources.fixed,
            market,
            full[cell],
            beta_zero=cell == "FF",
        )
        for cell in CELLS
    }
    for cell, parent_model in (("FF", "fixed"), ("LL", "lagged_log4")):
        parent_trade = read_trade_path(
            inputs.parent / market / "trades" / f"{parent_model}.csv",
            config.backtest_protocol.primary_delay,
            config.backtest_protocol.one_way_cost_bps,
        )
        _assert_frame_close(
            aligned[cell], parent_trade, f"{market}/{cell} parent parity"
        )
    summary = _add_deltas(
        pd.DataFrame(
            [
                {**_metric_row(market, cell, aligned[cell], config), "cell": cell}
                for cell in CELLS
            ]
        ).drop(columns="model")
    )
    traces = _change_traces(
        market,
        frame,
        selections,
        full,
        aligned,
        config.backtest_protocol.return_offset,
    )
    return summary, traces, aligned, selections


def _market_run(
    market: str,
    inputs: AttributionInputs,
    target: Path,
    config: ResearchConfig,
    spec: AttributionSpec,
) -> dict[str, Any]:
    with threadpool_limits(limits=1):
        summary, traces, aligned, _ = _replay_market(market, inputs, config, spec)
        (target / "trades").mkdir(parents=True, exist_ok=True)
        for cell in CELLS:
            aligned[cell].to_csv(target / "trades" / f"{cell}.csv", index=False)
        summary.to_csv(target / "summary.csv", index=False)
        traces.to_csv(target / "change-traces.csv", index=False)
        return {
            "summary": summary.to_dict("records"),
            "traces": traces.to_dict("records"),
        }


def run_us_smoke(
    config: ResearchConfig,
    spec: AttributionSpec,
    inputs: AttributionInputs | None = None,
) -> dict[str, Any]:
    root = config.path.parent
    inputs = inputs or _load_inputs(root, config, spec)
    frame, source_states = _load_market("us", inputs.sources, inputs.parent_spec)
    states = {
        "fixed": source_states["fixed"],
        "lagged_log4": source_states["lagged_log4"],
    }
    selections = _cell_selections(frame, states, inputs, "us")
    full = {cell: _full_path(frame, selections[cell], config) for cell in CELLS}
    aligned = {
        cell: _align_parent_sample(
            inputs.sources.fixed,
            "us",
            full[cell],
            beta_zero=cell == "FF",
        )
        for cell in CELLS
    }
    for cell, parent_model in (("FF", "fixed"), ("LL", "lagged_log4")):
        parent_trade = read_trade_path(
            inputs.parent / "us" / "trades" / f"{parent_model}.csv",
            config.backtest_protocol.primary_delay,
            config.backtest_protocol.one_way_cost_bps,
        )
        _assert_frame_close(aligned[cell], parent_trade, f"US smoke {cell}")
    offset = config.backtest_protocol.return_offset
    valid = np.flatnonzero(selections["FL"].signal.notna().to_numpy())
    if len(valid) == 0 or valid[0] + offset >= len(frame):
        raise AttributionError("US attribution smoke has no t+2 row")
    row = int(valid[0])
    if full["FL"].iloc[row + offset]["position"] != selections["FL"].signal.iloc[row]:
        raise AttributionError("US attribution smoke t+2 mismatch")
    return {
        "status": "passed",
        "market": "us",
        "cells": list(CELLS),
        "choice_months_fixed": len(selections["FF"].choices),
        "choice_months_lagged": len(selections["FL"].choices),
        "first_signal_date": pd.Timestamp(frame.iloc[row]["date"]).date().isoformat(),
        "first_execution_date": pd.Timestamp(full["FL"].iloc[row + offset]["date"])
        .date()
        .isoformat(),
        "metrics_opened": False,
    }


def _decision(attribution: pd.DataFrame) -> dict[str, Any]:
    sharpe = attribution.loc[
        (attribution["market"] == "equal_market_mean")
        & (attribution["metric"] == "sharpe")
    ]
    if (
        len(sharpe) != 1
        or not np.isfinite(
            sharpe[["total", "path_shapley", "choice_shapley", "interaction"]].to_numpy(
                dtype=float
            )
        ).all()
    ):
        raise AttributionError("equal-market Sharpe attribution is incomplete")
    row = sharpe.iloc[0]
    return {
        "schema_version": 1,
        "experiment_id": "lagged-selection-attribution-001",
        "claim_class": "EXPLORATORY",
        "result": "diagnostic_complete",
        "equal_market_mean_sharpe_total": float(row["total"]),
        "equal_market_mean_sharpe_path_shapley": float(row["path_shapley"]),
        "equal_market_mean_sharpe_choice_shapley": float(row["choice_shapley"]),
        "equal_market_mean_sharpe_interaction": float(row["interaction"]),
        "supported_or_not_supported": None,
        "cell_winner_selected": False,
        "causal_claim_allowed": False,
        "performance_claim_allowed": False,
        "paper_replication_claim_allowed": False,
    }


def _implementation_sha(root: Path, spec: AttributionSpec) -> str:
    files = (
        spec.path,
        root / "research.toml",
        root / "uv.lock",
        root / "src/adaptive_jump/lagged_attribution.py",
        root / "src/adaptive_jump/lagged_performance.py",
        root / "src/adaptive_jump/artifacts.py",
        root / "src/adaptive_jump/backtest.py",
        root / "src/adaptive_jump/config.py",
        root / "src/adaptive_jump/confidence_evaluation.py",
        root / "src/adaptive_jump/confidence_model.py",
        root / "src/adaptive_jump/walkforward.py",
    )
    payload = {str(file.relative_to(root)): sha256_file(file) for file in files}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _expected_files(spec: AttributionSpec) -> set[str]:
    files = {
        "attribution.csv",
        "change-traces.csv",
        "config.lock.toml",
        "decision.json",
        "inventory.json",
        "run.json",
        "smoke.json",
        "study.lock.toml",
        "summary.csv",
    }
    for market in spec.markets:
        files.update(
            {
                f"{market}/change-traces.csv",
                f"{market}/summary.csv",
                *(f"{market}/trades/{cell}.csv" for cell in CELLS),
            }
        )
    return files


def verify_attribution_run(
    run_dir: str | Path,
    config: ResearchConfig,
    spec: AttributionSpec,
) -> dict[str, Any]:
    raw_path = Path(run_dir)
    if raw_path.is_symlink():
        raise AttributionError("attribution run may not be a symlink")
    path = raw_path.resolve()
    if any(item.is_symlink() for item in path.rglob("*")):
        raise AttributionError("attribution artifacts may not contain symlinks")
    actual = {str(item.relative_to(path)) for item in path.rglob("*") if item.is_file()}
    if actual != _expected_files(spec):
        raise AttributionError("attribution artifact coverage changed")
    verify_inventory(path)
    root = config.path.parent
    if (
        sha256_file(path / "study.lock.toml") != spec.sha256
        or sha256_file(path / "config.lock.toml") != config.sha256
    ):
        raise AttributionError("attribution run locks changed")
    implementation = _implementation_sha(root, spec)
    run_id = (
        f"lagged-attribution-{spec.sha256[:12]}-"
        f"{spec.parent_inventory_sha256[:12]}-{implementation[:12]}"
    )
    expected_path = root / config.artifact_root / spec.artifact_subdir / run_id
    metadata = read_json(path / "run.json")
    controls = (
        path == expected_path.resolve(),
        path.name == run_id,
        metadata.get("schema_version") == 1,
        metadata.get("study_kind") == "lagged_selection_attribution",
        metadata.get("experiment_id") == spec.experiment_id,
        metadata.get("run_id") == run_id,
        metadata.get("status") == "complete",
        metadata.get("claim_class") == "EXPLORATORY",
        metadata.get("spec_sha256") == spec.sha256,
        metadata.get("config_sha256") == config.sha256,
        metadata.get("implementation_sha256") == implementation,
        metadata.get("git_worktree_clean") is False,
        metadata.get("metrics_opened") is True,
        metadata.get("post_2023_accessed") is False,
        metadata.get("causal_claim_allowed") is False,
        metadata.get("performance_claim_allowed") is False,
        metadata.get("paper_replication_claim_allowed") is False,
    )
    if not all(controls):
        raise AttributionError("attribution run identity changed")
    inputs = _load_inputs(root, config, spec)
    if read_json(path / "smoke.json") != run_us_smoke(config, spec, inputs):
        raise AttributionError("attribution smoke changed")
    summaries: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    maximum_error = 0.0
    for market in spec.markets:
        summary, trace, aligned, _ = _replay_market(market, inputs, config, spec)
        for cell in CELLS:
            stored_trade = read_trade_path(
                path / market / "trades" / f"{cell}.csv",
                config.backtest_protocol.primary_delay,
                config.backtest_protocol.one_way_cost_bps,
            )
            if stored_trade["date"].max().date() > spec.cutoff:
                raise AttributionError(f"{market}/{cell}: post-2023 trade")
            _assert_frame_close(
                stored_trade, aligned[cell], f"{market}/{cell} source replay"
            )
        maximum_error = max(
            maximum_error,
            _assert_frame_close(
                pd.read_csv(path / market / "summary.csv"),
                summary,
                f"{market} summary",
                tolerance=1e-12,
            ),
        )
        _assert_frame_close(
            pd.read_csv(path / market / "change-traces.csv"),
            trace,
            f"{market} traces",
        )
        summaries.extend(summary.to_dict("records"))
        traces.extend(trace.to_dict("records"))
    summary = pd.DataFrame(summaries)
    trace = pd.DataFrame(traces)
    attribution = _attribution_rows(summary, spec.identity_tolerance)
    maximum_error = max(
        maximum_error,
        _assert_frame_close(
            pd.read_csv(path / "summary.csv"),
            summary,
            "root summary",
            tolerance=1e-12,
        ),
        _assert_frame_close(
            pd.read_csv(path / "attribution.csv"),
            attribution,
            "root attribution",
            tolerance=1e-12,
        ),
    )
    _assert_frame_close(pd.read_csv(path / "change-traces.csv"), trace, "root traces")
    decision = _decision(attribution)
    if read_json(path / "decision.json") != decision:
        raise AttributionError("attribution decision changed")
    if metadata.get("decision") != "diagnostic_complete":
        raise AttributionError("attribution metadata result changed")
    return {
        "status": "passed",
        "run_id": run_id,
        "metric_rows": len(summary),
        "attribution_rows": len(attribution),
        "maximum_absolute_error": maximum_error,
        "result": "diagnostic_complete",
    }


def run_attribution_study(config: ResearchConfig, spec: AttributionSpec) -> Path:
    root = config.path.parent
    inputs = _load_inputs(root, config, spec)
    smoke = run_us_smoke(config, spec, inputs)
    implementation = _implementation_sha(root, spec)
    run_id = (
        f"lagged-attribution-{spec.sha256[:12]}-"
        f"{spec.parent_inventory_sha256[:12]}-{implementation[:12]}"
    )
    run_dir = root / config.artifact_root / spec.artifact_subdir / run_id
    if (run_dir / "run.json").is_file():
        metadata = read_json(run_dir / "run.json")
        if metadata.get("status") == "complete":
            verify_attribution_run(run_dir, config, spec)
            return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "study.lock.toml").write_bytes(spec.path.read_bytes())
    (run_dir / "config.lock.toml").write_bytes(config.path.read_bytes())
    write_json(run_dir / "smoke.json", smoke)
    write_json(
        run_dir / "run.json",
        {
            "schema_version": 1,
            "study_kind": "lagged_selection_attribution",
            "experiment_id": spec.experiment_id,
            "run_id": run_id,
            "status": "running",
            "claim_class": "EXPLORATORY",
            "spec_sha256": spec.sha256,
            "config_sha256": config.sha256,
            "implementation_sha256": implementation,
            "git_sha": _git_head(root),
            "git_worktree_clean": False,
            "metrics_opened": False,
            "post_2023_accessed": False,
            "causal_claim_allowed": False,
            "performance_claim_allowed": False,
            "paper_replication_claim_allowed": False,
            "created_at_utc": datetime.now(UTC).isoformat(),
        },
    )
    results: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(
        max_workers=len(spec.markets), mp_context=get_context("forkserver")
    ) as executor:
        futures = {
            executor.submit(
                _market_run,
                market,
                inputs,
                run_dir / market,
                config,
                spec,
            ): market
            for market in spec.markets
        }
        for future in as_completed(futures):
            market = futures[future]
            results[market] = future.result()
            print(f"{market}: complete", flush=True)
    summary = pd.DataFrame(
        [row for market in spec.markets for row in results[market]["summary"]]
    )
    traces = pd.DataFrame(
        [row for market in spec.markets for row in results[market]["traces"]]
    )
    attribution = _attribution_rows(summary, spec.identity_tolerance)
    summary.to_csv(run_dir / "summary.csv", index=False)
    traces.to_csv(run_dir / "change-traces.csv", index=False)
    attribution.to_csv(run_dir / "attribution.csv", index=False)
    decision = _decision(attribution)
    write_json(run_dir / "decision.json", decision)
    metadata = read_json(run_dir / "run.json")
    metadata.update(
        {
            "status": "complete",
            "metrics_opened": True,
            "decision": "diagnostic_complete",
            "finished_at_utc": datetime.now(UTC).isoformat(),
        }
    )
    write_json(run_dir / "run.json", metadata)
    write_inventory(run_dir)
    verify_attribution_run(run_dir, config, spec)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lagged-selection-attribution")
    parser.add_argument("--config", default="research.toml")
    parser.add_argument(
        "--spec", default="research/lagged-selection-attribution-001.toml"
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--verify")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    spec_path = Path(args.spec)
    if not spec_path.is_absolute():
        spec_path = config.path.parent / spec_path
    spec = load_attribution_spec(spec_path, config)
    if args.verify:
        print(
            json.dumps(
                verify_attribution_run(args.verify, config, spec), sort_keys=True
            )
        )
    elif args.smoke:
        print(json.dumps(run_us_smoke(config, spec), sort_keys=True))
    else:
        print(run_attribution_study(config, spec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
