"""Frozen development-sample P&L readout for lagged-evidence JM."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import tomllib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, date, datetime
from io import StringIO
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
    verify_run,
    write_inventory,
    write_json,
)
from adaptive_jump.backtest import apply_signal, performance_metrics
from adaptive_jump.confidence_evaluation import (
    _active_lambda,
    _align_parent_sample,
    _assert_beta_zero_selection,
)
from adaptive_jump.confidence_model import _load_parent_frame, _parent_states
from adaptive_jump.config import ResearchConfig, load_config
from adaptive_jump.walkforward import (
    SelectionResult,
    boundary_diagnostic,
    select_monthly_candidate,
)

MODELS = ("fixed", "arrival_log4", "lagged_log4")
MARKETS = ("us", "de", "jp")
LAMBDAS = (0.0, 5.0, 15.0, 35.0, 70.0, 150.0, 300.0, 600.0, 1200.0)
BETA = math.log(4.0)
TURNOVER_SCALE = 0.5
REFIT_BOUNDARY_CONVENTION = (
    "on Jan/Jul refit dates, L_(t-1) is recomputed under the scaler and centers "
    "fitted through t, exactly as sealed by lagged-evidence-mechanism-001; the "
    "rule is causal at t but is not strictly F_(t-1)-predictable on those refit dates"
)


class LaggedPerformanceError(RuntimeError):
    """Raised when the frozen study contract or evidence is violated."""


@dataclass(frozen=True)
class LaggedPerformanceSpec:
    path: Path
    sha256: str
    experiment_id: str
    fixed_run_id: str
    fixed_inventory_sha256: str
    data_manifest_sha256: str
    arrival_run_id: str
    arrival_inventory_sha256: str
    arrival_spec_sha256: str
    lagged_run_id: str
    lagged_inventory_sha256: str
    lagged_spec_sha256: str
    cutoff: date
    markets: tuple[str, ...]
    lambdas: tuple[float, ...]
    beta: float
    artifact_subdir: Path


@dataclass(frozen=True)
class SourcePaths:
    fixed: Path
    arrival: Path
    lagged: Path


def load_lagged_performance_spec(
    path: str | Path, config: ResearchConfig
) -> LaggedPerformanceSpec:
    spec_path = Path(path).resolve()
    payload = spec_path.read_bytes()
    try:
        doc = tomllib.loads(payload.decode())
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise LaggedPerformanceError(f"invalid lagged performance spec: {exc}") from exc
    fixed, arrival, lagged = (
        doc.get("fixed_source", {}),
        doc.get("arrival_source", {}),
        doc.get("lagged_source", {}),
    )
    model, protocol, decision = (
        doc.get("model", {}),
        doc.get("protocol", {}),
        doc.get("decision", {}),
    )
    verification = doc.get("verification", {})
    storage_doc = doc.get("storage", {})
    required = (
        doc.get("schema_version") == 1,
        doc.get("experiment_id") == "lagged-evidence-performance-001",
        doc.get("claim_class") == "EXPLORATORY",
        doc.get("performance_claim_allowed") is False,
        doc.get("paper_replication_claim_allowed") is False,
        doc.get("post_2023_access") is False,
        fixed.get("config_sha256") == config.sha256,
        fixed.get("data_cutoff") == "2023-12-31",
        arrival.get("role")
        == (
            "secondary mechanism control only; cannot determine or rescue the "
            "primary result"
        ),
        lagged.get("required_result") == "supported",
        lagged.get("required_selected_beta_label") == "log4",
        tuple(float(x) for x in model.get("raw_lambda_grid", ())) == LAMBDAS,
        float(model.get("beta", math.nan)) == BETA,
        model.get("refit_boundary_convention") == REFIT_BOUNDARY_CONVENTION,
        model.get("refit_or_state_regeneration") is False,
        tuple(protocol.get("markets", ())) == MARKETS,
        protocol.get("primary_delay_trading_days")
        == config.backtest_protocol.primary_delay,
        protocol.get("signal_to_return_offset")
        == config.backtest_protocol.return_offset,
        protocol.get("one_way_cost_bps") == config.backtest_protocol.one_way_cost_bps,
        protocol.get("minimum_valid_returns")
        == config.selection_protocol.minimum_valid_returns,
        decision.get("primary_contrast") == "lagged_log4_minus_fixed",
        decision.get("arrival_control_cannot_change_primary_decision") is True,
        decision.get("raw_metrics_always_reported") is True,
        decision.get("non_finite_primary_metrics")
        == "error; never silently omit a market from the three-market mean",
        verification.get("independent_source_replay") is True,
        verification.get("artifact_schema_and_allowlist_exact") is True,
        storage_doc.get("boundaries_csv") == "boundaries.csv",
    )
    if not all(required):
        raise LaggedPerformanceError("lagged performance controls changed")
    storage = Path(str(storage_doc.get("artifact_subdir", "")))
    if not storage.parts or storage.is_absolute() or ".." in storage.parts:
        raise LaggedPerformanceError("invalid lagged performance artifact path")
    return LaggedPerformanceSpec(
        path=spec_path,
        sha256=hashlib.sha256(payload).hexdigest(),
        experiment_id=doc["experiment_id"],
        fixed_run_id=str(fixed["run_id"]),
        fixed_inventory_sha256=str(fixed["run_inventory_sha256"]),
        data_manifest_sha256=str(fixed["data_manifest_sha256"]),
        arrival_run_id=str(arrival["run_id"]),
        arrival_inventory_sha256=str(arrival["run_inventory_sha256"]),
        arrival_spec_sha256=str(arrival["spec_sha256"]),
        lagged_run_id=str(lagged["run_id"]),
        lagged_inventory_sha256=str(lagged["run_inventory_sha256"]),
        lagged_spec_sha256=str(lagged["spec_sha256"]),
        cutoff=date.fromisoformat(fixed["data_cutoff"]),
        markets=tuple(protocol["markets"]),
        lambdas=tuple(float(x) for x in model["raw_lambda_grid"]),
        beta=float(model["beta"]),
        artifact_subdir=storage,
    )


def _registry_lock(root: Path, spec: LaggedPerformanceSpec) -> None:
    rows = [
        json.loads(line)
        for line in (root / "research/experiment_registry.jsonl")
        .read_text()
        .splitlines()
        if json.loads(line).get("experiment_id") == spec.experiment_id
    ]
    if (
        not rows
        or rows[-1].get("status") not in {"FROZEN", "EXPERIMENT_COMPLETE"}
        or rows[-1].get("frozen_spec_hash") != spec.sha256
    ):
        raise LaggedPerformanceError("lagged performance registry lock changed")


def _verify_sources(
    root: Path, config: ResearchConfig, spec: LaggedPerformanceSpec
) -> SourcePaths:
    _registry_lock(root, spec)
    paths = SourcePaths(
        fixed=root / config.artifact_root / "fixed-baselines" / spec.fixed_run_id,
        arrival=root
        / config.artifact_root
        / "adaptive-confidence-001"
        / spec.arrival_run_id,
        lagged=root
        / config.artifact_root
        / "lagged-evidence-mechanism-001"
        / spec.lagged_run_id,
    )
    expected = {
        paths.fixed: spec.fixed_inventory_sha256,
        paths.arrival: spec.arrival_inventory_sha256,
        paths.lagged: spec.lagged_inventory_sha256,
    }
    for run, digest in expected.items():
        if sha256_file(run / "inventory.json") != digest:
            raise LaggedPerformanceError(
                f"source inventory identity changed: {run.name}"
            )
        verify_inventory(run)
    receipt = verify_run(paths.fixed)
    if receipt.get("status") != "complete":
        raise LaggedPerformanceError("fixed parent is not complete")
    fixed_meta, arrival_meta, lagged_meta = (
        read_json(paths.fixed / "run.json"),
        read_json(paths.arrival / "run.json"),
        read_json(paths.lagged / "run.json"),
    )
    lagged_conclusion = read_json(paths.lagged / "conclusion.json")
    if (
        fixed_meta.get("config_sha256") != config.sha256
        or sha256_file(paths.fixed / "data-manifest.json") != spec.data_manifest_sha256
        or arrival_meta.get("status") != "complete"
        or arrival_meta.get("spec_sha256") != spec.arrival_spec_sha256
        or lagged_meta.get("status") != "complete"
        or lagged_meta.get("spec_sha256") != spec.lagged_spec_sha256
        or lagged_conclusion.get("result") != "supported"
        or lagged_conclusion.get("selected_beta_label") != "log4"
    ):
        raise LaggedPerformanceError("source run metadata changed")
    return paths


def _read_states(path: Path, spec: LaggedPerformanceSpec) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if (
        "date" not in frame
        or tuple(float(c) for c in frame.columns[1:]) != spec.lambdas
    ):
        raise LaggedPerformanceError(f"candidate-state schema changed: {path}")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    dates = pd.DatetimeIndex(frame.pop("date"))
    frame.index = dates
    frame.columns = spec.lambdas
    if (
        dates.has_duplicates
        or not dates.is_monotonic_increasing
        or dates.max().date() > spec.cutoff
        or not frame.stack().isin([0.0, 1.0]).all()
    ):
        raise LaggedPerformanceError(f"candidate-state values changed: {path}")
    frame.index.name = "date"
    return frame


def _load_market(
    market: str,
    paths: SourcePaths,
    spec: LaggedPerformanceSpec,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    frame = _load_parent_frame(paths.fixed, market, spec.cutoff)
    fixed = _parent_states(paths.fixed, market, spec.lambdas)
    arrival0 = _read_states(
        paths.arrival / market / "candidate-states-beta-0.csv", spec
    )
    lagged0 = _read_states(paths.lagged / market / "candidate-states-beta-0.csv", spec)
    arrival4 = _read_states(
        paths.arrival / market / "candidate-states-beta-log4.csv", spec
    )
    lagged4 = _read_states(
        paths.lagged / market / "candidate-states-beta-log4.csv", spec
    )
    for candidate in (arrival0, lagged0, arrival4, lagged4):
        if not candidate.index.equals(fixed.index):
            raise LaggedPerformanceError(f"{market}: candidate dates changed")
    for beta0 in (arrival0, lagged0):
        if not np.array_equal(fixed.to_numpy(), beta0.to_numpy(), equal_nan=True):
            raise LaggedPerformanceError(f"{market}: beta0 states differ from fixed")
    return frame, {"fixed": fixed, "arrival_log4": arrival4, "lagged_log4": lagged4}


def _select_paths(
    frame: pd.DataFrame,
    states: dict[str, pd.DataFrame],
    config: ResearchConfig,
) -> dict[str, SelectionResult]:
    returns = frame[["date", "equity_simple", "cash_return"]]
    return {
        model: select_monthly_candidate(
            returns,
            candidate,
            config.selection_protocol,
            delay_trading_days=config.backtest_protocol.primary_delay,
            one_way_cost_bps=config.backtest_protocol.one_way_cost_bps,
            periods_per_year=config.metrics_protocol.periods_per_year,
            volatility_ddof=config.metrics_protocol.volatility_ddof,
        )
        for model, candidate in states.items()
    }


def _full_path(
    frame: pd.DataFrame, selected: SelectionResult, config: ResearchConfig
) -> pd.DataFrame:
    return apply_signal(
        frame[["date", "equity_simple", "cash_return"]],
        selected.signal.reset_index(drop=True),
        delay_trading_days=config.backtest_protocol.primary_delay,
        one_way_cost_bps=config.backtest_protocol.one_way_cost_bps,
    )


def _metric_row(
    market: str, model: str, path: pd.DataFrame, config: ResearchConfig
) -> dict[str, Any]:
    values = performance_metrics(
        path,
        periods_per_year=config.metrics_protocol.periods_per_year,
        volatility_ddof=config.metrics_protocol.volatility_ddof,
        expected_shortfall_quantile=config.metrics_protocol.expected_shortfall_quantile,
        turnover_scale=TURNOVER_SCALE,
    )
    return {
        "market": market,
        "model": model,
        **values,
        "cash_fraction": float(1.0 - path["position"].mean()),
        "switch_count": int((path["one_way_turnover"] > 0).sum()),
    }


def _add_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    fixed = metrics.loc[metrics["model"] == "fixed"]
    if len(fixed) != 1:
        raise LaggedPerformanceError("market has no unique fixed metric row")
    output = metrics.copy()
    for name in (
        "sharpe",
        "maximum_drawdown",
        "turnover",
        "cash_fraction",
        "switch_count",
    ):
        output[f"delta_{name}"] = output[name] - fixed.iloc[0][name]
    return output


def _decision(summary: pd.DataFrame) -> dict[str, Any]:
    expected = {(market, model) for market in MARKETS for model in MODELS}
    actual = list(summary[["market", "model"]].itertuples(index=False, name=None))
    if len(actual) != len(expected) or set(actual) != expected:
        raise LaggedPerformanceError("market/model coverage is incomplete")
    deltas = pd.to_numeric(summary["delta_sharpe"], errors="coerce")
    if not np.isfinite(deltas.to_numpy()).all():
        raise LaggedPerformanceError("Sharpe deltas must be finite in all markets")
    lagged = summary.loc[summary["model"] == "lagged_log4"].sort_values("market")
    mean_delta = float(lagged["delta_sharpe"].mean())
    positive = int((lagged["delta_sharpe"] > 0).sum())
    supported = mean_delta > 0 and positive >= 2
    arrival = summary.loc[summary["model"] == "arrival_log4"]
    return {
        "schema_version": 1,
        "experiment_id": "lagged-evidence-performance-001",
        "claim_class": "EXPLORATORY",
        "primary_contrast": "lagged_log4_minus_fixed",
        "primary_mean_delta_sharpe": mean_delta,
        "positive_market_count": positive,
        "result": "supported" if supported else "not_supported",
        "market_delta_sharpe": {
            row.market: float(row.delta_sharpe)
            for row in lagged.itertuples(index=False)
        },
        "arrival_control_mean_delta_sharpe": float(arrival["delta_sharpe"].mean()),
        "performance_claim_allowed": False,
        "paper_replication_claim_allowed": False,
    }


def _verify_fixed_metrics(
    market: str,
    row: dict[str, Any],
    paths: SourcePaths,
) -> None:
    parent = pd.read_csv(paths.fixed / "metrics.csv")
    expected = parent.loc[
        (parent["market"] == market)
        & (parent["model"] == "fixed_jm")
        & (parent["delay"] == 1)
    ]
    if len(expected) != 1:
        raise LaggedPerformanceError(f"{market}: parent fixed metrics missing")
    expected_row = expected.iloc[0]
    exact = ("start", "end", "observations")
    numeric = (
        "cagr",
        "volatility",
        "sharpe",
        "maximum_drawdown",
        "calmar",
        "expected_shortfall_5pct",
        "leverage",
    )
    if any(row[key] != expected_row[key] for key in exact) or any(
        not math.isclose(
            float(row[key]), float(expected_row[key]), rel_tol=0, abs_tol=1e-12
        )
        for key in numeric
    ):
        raise LaggedPerformanceError(f"{market}: fixed non-turnover metrics changed")
    if not math.isclose(
        2.0 * float(row["turnover"]),
        float(expected_row["turnover"]),
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise LaggedPerformanceError(f"{market}: paper turnover is not parent/2")


def _governing_choice(choices: pd.DataFrame, signal_date: pd.Timestamp) -> pd.Series:
    dates = pd.to_datetime(choices["decision_date"], errors="raise")
    prior = choices.loc[dates <= signal_date]
    if prior.empty:
        raise LaggedPerformanceError("signal has no governing monthly choice")
    return prior.iloc[-1]


def _change_traces(
    market: str,
    frame: pd.DataFrame,
    selections: dict[str, SelectionResult],
    full_paths: dict[str, pd.DataFrame],
    aligned: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    dates = pd.DatetimeIndex(pd.to_datetime(frame["date"], errors="raise"))
    fixed_active = _active_lambda(dates, selections["fixed"].choices)
    fixed_state = 1.0 - selections["fixed"].signal.reindex(dates)
    first_execution = pd.Timestamp(aligned["fixed"]["date"].iloc[0])
    records: list[dict[str, Any]] = []
    for challenger in ("arrival_log4", "lagged_log4"):
        active = _active_lambda(dates, selections[challenger].choices)
        state = 1.0 - selections[challenger].signal.reindex(dates)
        seen: set[str] = set()
        for row in range(len(dates) - 2):
            execution = row + 2
            execution_date = pd.Timestamp(
                full_paths[challenger].iloc[execution]["date"]
            )
            if execution_date < first_execution:
                continue
            flags = {
                "choice": pd.notna(active.iloc[row])
                and pd.notna(fixed_active.iloc[row])
                and active.iloc[row] != fixed_active.iloc[row],
                "state": pd.notna(state.iloc[row])
                and pd.notna(fixed_state.iloc[row])
                and state.iloc[row] != fixed_state.iloc[row],
                "position": full_paths[challenger].iloc[execution]["position"]
                != full_paths["fixed"].iloc[execution]["position"],
                "trade": full_paths[challenger].iloc[execution]["one_way_turnover"]
                != full_paths["fixed"].iloc[execution]["one_way_turnover"],
            }
            for event_type, changed in flags.items():
                if not changed or event_type in seen:
                    continue
                fixed_choice = _governing_choice(
                    selections["fixed"].choices, dates[row]
                )
                challenger_choice = _governing_choice(
                    selections[challenger].choices, dates[row]
                )
                ftrade, ctrade = (
                    full_paths["fixed"].iloc[execution],
                    full_paths[challenger].iloc[execution],
                )
                records.append(
                    {
                        "market": market,
                        "challenger": challenger,
                        "event_type": event_type,
                        "signal_date": dates[row],
                        "execution_date": execution_date,
                        "offset_observations": 2,
                        "fixed_decision_date": fixed_choice["decision_date"],
                        "challenger_decision_date": challenger_choice["decision_date"],
                        "fixed_lambda": float(fixed_active.iloc[row]),
                        "challenger_lambda": float(active.iloc[row]),
                        "fixed_state": int(fixed_state.iloc[row]),
                        "challenger_state": int(state.iloc[row]),
                        "fixed_signal": float(
                            selections["fixed"].signal.loc[dates[row]]
                        ),
                        "challenger_signal": float(
                            selections[challenger].signal.loc[dates[row]]
                        ),
                        "fixed_position": float(ftrade["position"]),
                        "challenger_position": float(ctrade["position"]),
                        "fixed_turnover": float(ftrade["one_way_turnover"]),
                        "challenger_turnover": float(ctrade["one_way_turnover"]),
                        "fixed_cost": float(ftrade["transaction_cost"]),
                        "challenger_cost": float(ctrade["transaction_cost"]),
                    }
                )
                seen.add(event_type)
            if len(seen) == 4:
                break
    return pd.DataFrame.from_records(records)


def _boundary_rows(market, selections, aligned, config, spec) -> pd.DataFrame:
    oos_start = pd.Timestamp(aligned["fixed"]["date"].iloc[0]).date()
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        diagnostic = boundary_diagnostic(
            selections[model].choices,
            spec.lambdas,
            oos_start=oos_start,
            fraction_limit=config.selection_protocol.boundary_fraction_limit,
        )
        rows.append({"market": market, "model": model, **diagnostic.__dict__})
    return pd.DataFrame.from_records(rows)


def _market_run(
    market: str,
    paths: SourcePaths,
    target: Path,
    config: ResearchConfig,
    spec: LaggedPerformanceSpec,
) -> dict[str, Any]:
    with threadpool_limits(limits=1):
        frame, states = _load_market(market, paths, spec)
        selections = _select_paths(frame, states, config)
        _assert_beta_zero_selection(paths.fixed, market, selections["fixed"])
        full = {
            model: _full_path(frame, selection, config)
            for model, selection in selections.items()
        }
        aligned = {
            model: _align_parent_sample(
                paths.fixed,
                market,
                path,
                beta_zero=model == "fixed",
            )
            for model, path in full.items()
        }
        metrics = _add_deltas(
            pd.DataFrame(
                [_metric_row(market, model, aligned[model], config) for model in MODELS]
            )
        )
        _verify_fixed_metrics(
            market,
            metrics.loc[metrics["model"] == "fixed"].iloc[0].to_dict(),
            paths,
        )
        choices = pd.concat(
            [
                selection.choices.assign(market=market, model=model)
                for model, selection in selections.items()
            ],
            ignore_index=True,
        )
        traces = _change_traces(market, frame, selections, full, aligned)
        boundaries = _boundary_rows(market, selections, aligned, config, spec)
        (target / "trades").mkdir(parents=True, exist_ok=True)
        for model in MODELS:
            aligned[model].to_csv(target / "trades" / f"{model}.csv", index=False)
        choices.to_csv(target / "choices.csv", index=False)
        traces.to_csv(target / "change-traces.csv", index=False)
        metrics.to_csv(target / "summary.csv", index=False)
        boundaries.to_csv(target / "boundaries.csv", index=False)
        return {
            "market": market,
            "summary": metrics.to_dict("records"),
            "choices": choices.to_dict("records"),
            "traces": traces.to_dict("records"),
            "boundaries": boundaries.to_dict("records"),
        }


def run_us_smoke(
    config: ResearchConfig,
    spec: LaggedPerformanceSpec,
    paths: SourcePaths | None = None,
) -> dict[str, Any]:
    root = config.path.parent
    paths = paths or _verify_sources(root, config, spec)
    frame, states = _load_market("us", paths, spec)
    if np.array_equal(
        states["fixed"].to_numpy(), states["lagged_log4"].to_numpy(), equal_nan=True
    ):
        raise LaggedPerformanceError("US lagged candidate paths are vacuous")
    selections = _select_paths(frame, states, config)
    _assert_beta_zero_selection(paths.fixed, "us", selections["fixed"])
    full = {
        model: _full_path(frame, selection, config)
        for model, selection in selections.items()
    }
    for model in ("fixed", "lagged_log4"):
        _align_parent_sample(paths.fixed, "us", full[model], beta_zero=model == "fixed")
    valid = np.flatnonzero(selections["lagged_log4"].signal.notna().to_numpy())
    if len(valid) == 0 or valid[0] + 2 >= len(frame):
        raise LaggedPerformanceError("US smoke has no t+2 accounting row")
    row = int(valid[0])
    if (
        full["lagged_log4"].iloc[row + 2]["position"]
        != selections["lagged_log4"].signal.iloc[row]
    ):
        raise LaggedPerformanceError("US smoke t+2 position mismatch")
    return {
        "status": "passed",
        "market": "us",
        "candidate_rows": len(states["lagged_log4"]),
        "choice_months": len(selections["lagged_log4"].choices),
        "first_signal_date": pd.Timestamp(frame.iloc[row]["date"]).date().isoformat(),
        "first_execution_date": pd.Timestamp(full["lagged_log4"].iloc[row + 2]["date"])
        .date()
        .isoformat(),
        "metrics_opened": False,
    }


def _git_head(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _implementation_sha(root: Path, spec: LaggedPerformanceSpec) -> str:
    files = (
        spec.path,
        root / "research.toml",
        root / "uv.lock",
        root / "src/adaptive_jump/lagged_performance.py",
        root / "src/adaptive_jump/artifacts.py",
        root / "src/adaptive_jump/backtest.py",
        root / "src/adaptive_jump/config.py",
        root / "src/adaptive_jump/confidence_evaluation.py",
        root / "src/adaptive_jump/confidence_model.py",
        root / "src/adaptive_jump/walkforward.py",
    )
    payload = {str(path.relative_to(root)): sha256_file(path) for path in files}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _csv_roundtrip(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.read_csv(StringIO(frame.to_csv(index=False)))


def _assert_frame_close(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    label: str,
    *,
    tolerance: float = 1e-15,
) -> float:
    left = _csv_roundtrip(observed.reset_index(drop=True))
    right = _csv_roundtrip(expected.reset_index(drop=True))
    if tuple(left.columns) != tuple(right.columns) or left.shape != right.shape:
        raise LaggedPerformanceError(f"{label}: table schema or shape changed")
    try:
        pd.testing.assert_frame_equal(
            left,
            right,
            check_dtype=False,
            check_exact=False,
            rtol=0,
            atol=tolerance,
        )
    except AssertionError as exc:
        raise LaggedPerformanceError(f"{label}: stored values changed") from exc
    maximum = 0.0
    for column in left.columns:
        left_numeric = pd.to_numeric(left[column], errors="coerce")
        right_numeric = pd.to_numeric(right[column], errors="coerce")
        if pd.api.types.is_bool_dtype(left_numeric) or pd.api.types.is_bool_dtype(
            right_numeric
        ):
            continue
        left_values = left_numeric.to_numpy(dtype=float)
        right_values = right_numeric.to_numpy(dtype=float)
        finite = np.isfinite(left_values) & np.isfinite(right_values)
        if finite.any():
            maximum = max(
                maximum,
                float(np.max(np.abs(left_values[finite] - right_values[finite]))),
            )
    return maximum


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, OSError, pd.errors.ParserError) as exc:
        raise LaggedPerformanceError(f"cannot read artifact CSV: {path}") from exc


def _expected_artifact_files(spec: LaggedPerformanceSpec) -> set[str]:
    files = {
        "boundaries.csv",
        "change-traces.csv",
        "choices.csv",
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
                f"{market}/boundaries.csv",
                f"{market}/change-traces.csv",
                f"{market}/choices.csv",
                f"{market}/summary.csv",
                *(f"{market}/trades/{model}.csv" for model in MODELS),
            }
        )
    return files


def _verify_run_identity(
    path: Path,
    config: ResearchConfig,
    spec: LaggedPerformanceSpec,
) -> tuple[dict[str, Any], SourcePaths]:
    root = config.path.parent
    if any(item.is_symlink() for item in (path, *path.rglob("*"))):
        raise LaggedPerformanceError("run artifacts may not contain symlinks")
    actual_files = {
        str(item.relative_to(path)) for item in path.rglob("*") if item.is_file()
    }
    if actual_files != _expected_artifact_files(spec):
        raise LaggedPerformanceError("run artifact file coverage changed")
    verify_inventory(path)
    if (
        sha256_file(path / "study.lock.toml") != spec.sha256
        or sha256_file(path / "config.lock.toml") != config.sha256
    ):
        raise LaggedPerformanceError("run locks differ from the frozen study")
    implementation = _implementation_sha(root, spec)
    run_id = (
        f"lagged-pnl-{spec.sha256[:12]}-"
        f"{spec.lagged_inventory_sha256[:12]}-{implementation[:12]}"
    )
    expected_path = root / config.artifact_root / spec.artifact_subdir / run_id
    metadata = read_json(path / "run.json")
    controls = (
        path == expected_path.resolve(),
        path.name == run_id,
        metadata.get("schema_version") == 1,
        metadata.get("study_kind") == "lagged_evidence_performance",
        metadata.get("experiment_id") == spec.experiment_id,
        metadata.get("run_id") == run_id,
        metadata.get("status") == "complete",
        metadata.get("claim_class") == "EXPLORATORY",
        metadata.get("metrics_opened") is True,
        metadata.get("spec_sha256") == spec.sha256,
        metadata.get("config_sha256") == config.sha256,
        metadata.get("implementation_sha256") == implementation,
        metadata.get("git_worktree_clean") is False,
        metadata.get("post_2023_accessed") is False,
        metadata.get("performance_claim_allowed") is False,
        metadata.get("paper_replication_claim_allowed") is False,
    )
    git_sha = metadata.get("git_sha")
    try:
        valid_git_sha = isinstance(git_sha, str) and len(git_sha) >= 12
        if valid_git_sha:
            int(git_sha, 16)
    except ValueError:
        valid_git_sha = False
    if not all(controls) or not valid_git_sha:
        raise LaggedPerformanceError("run identity or controls changed")
    paths = _verify_sources(root, config, spec)
    if read_json(path / "smoke.json") != run_us_smoke(config, spec, paths):
        raise LaggedPerformanceError("stored US smoke changed")
    return metadata, paths


def verify_lagged_performance_run(
    run_dir: str | Path,
    config: ResearchConfig,
    spec: LaggedPerformanceSpec | None = None,
) -> dict[str, Any]:
    raw_path = Path(run_dir)
    if raw_path.is_symlink():
        raise LaggedPerformanceError("run directory may not be a symlink")
    path = raw_path.resolve()
    spec = spec or load_lagged_performance_spec(
        config.path.parent / "research/lagged-evidence-performance-001.toml",
        config,
    )
    metadata, paths = _verify_run_identity(path, config, spec)
    all_metrics: list[dict[str, Any]] = []
    all_choices: list[dict[str, Any]] = []
    all_traces: list[dict[str, Any]] = []
    all_boundaries: list[dict[str, Any]] = []
    maximum_error = 0.0
    for market in spec.markets:
        frame, states = _load_market(market, paths, spec)
        selections = _select_paths(frame, states, config)
        _assert_beta_zero_selection(paths.fixed, market, selections["fixed"])
        full = {
            model: _full_path(frame, selection, config)
            for model, selection in selections.items()
        }
        aligned = {
            model: _align_parent_sample(
                paths.fixed,
                market,
                full[model],
                beta_zero=model == "fixed",
            )
            for model in MODELS
        }
        market_rows: list[dict[str, Any]] = []
        for model in MODELS:
            trade_path = path / market / "trades" / f"{model}.csv"
            observed_trade = read_trade_path(
                trade_path,
                config.backtest_protocol.primary_delay,
                config.backtest_protocol.one_way_cost_bps,
            )
            if observed_trade["date"].max().date() > spec.cutoff:
                raise LaggedPerformanceError(f"{market}: post-2023 trade row")
            _assert_frame_close(
                observed_trade,
                aligned[model],
                f"{market}/{model} source replay",
            )
            market_rows.append(_metric_row(market, model, aligned[model], config))
        metrics = _add_deltas(pd.DataFrame(market_rows))
        _verify_fixed_metrics(
            market,
            metrics.loc[metrics["model"] == "fixed"].iloc[0].to_dict(),
            paths,
        )
        choices = pd.concat(
            [
                selections[model].choices.assign(market=market, model=model)
                for model in MODELS
            ],
            ignore_index=True,
        )
        traces = _change_traces(market, frame, selections, full, aligned)
        boundaries = _boundary_rows(market, selections, aligned, config, spec)
        maximum_error = max(
            maximum_error,
            _assert_frame_close(
                _read_csv(path / market / "summary.csv"),
                metrics,
                f"{market} summary",
                tolerance=1e-12,
            ),
        )
        _assert_frame_close(
            _read_csv(path / market / "choices.csv"), choices, f"{market} choices"
        )
        _assert_frame_close(
            _read_csv(path / market / "change-traces.csv"),
            traces,
            f"{market} change traces",
        )
        _assert_frame_close(
            _read_csv(path / market / "boundaries.csv"),
            boundaries,
            f"{market} boundaries",
        )
        all_metrics.extend(metrics.to_dict("records"))
        all_choices.extend(choices.to_dict("records"))
        all_traces.extend(traces.to_dict("records"))
        all_boundaries.extend(boundaries.to_dict("records"))
    summary = pd.DataFrame(all_metrics)
    choices = pd.DataFrame(all_choices)
    traces = pd.DataFrame(all_traces)
    boundaries = pd.DataFrame(all_boundaries)
    maximum_error = max(
        maximum_error,
        _assert_frame_close(
            _read_csv(path / "summary.csv"),
            summary,
            "root summary",
            tolerance=1e-12,
        ),
    )
    _assert_frame_close(_read_csv(path / "choices.csv"), choices, "root choices")
    _assert_frame_close(
        _read_csv(path / "change-traces.csv"), traces, "root change traces"
    )
    _assert_frame_close(
        _read_csv(path / "boundaries.csv"), boundaries, "root boundaries"
    )
    expected_decision = _decision(summary)
    if read_json(path / "decision.json") != expected_decision:
        raise LaggedPerformanceError("stored decision changed")
    if metadata.get("decision") != expected_decision["result"]:
        raise LaggedPerformanceError("run metadata decision changed")
    return {
        "status": "passed",
        "run_id": metadata["run_id"],
        "metric_rows": len(summary),
        "maximum_metric_absolute_error": maximum_error,
        "decision": expected_decision["result"],
    }


def run_lagged_performance_study(
    config: ResearchConfig, spec: LaggedPerformanceSpec
) -> Path:
    root = config.path.parent
    paths = _verify_sources(root, config, spec)
    smoke = run_us_smoke(config, spec, paths)
    implementation = _implementation_sha(root, spec)
    run_id = (
        f"lagged-pnl-{spec.sha256[:12]}-"
        f"{spec.lagged_inventory_sha256[:12]}-{implementation[:12]}"
    )
    run_dir = root / config.artifact_root / spec.artifact_subdir / run_id
    if (run_dir / "run.json").is_file():
        metadata = read_json(run_dir / "run.json")
        if metadata.get("status") == "complete":
            verify_lagged_performance_run(run_dir, config, spec)
            return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "study.lock.toml").write_bytes(spec.path.read_bytes())
    (run_dir / "config.lock.toml").write_bytes(config.path.read_bytes())
    write_json(run_dir / "smoke.json", smoke)
    write_json(
        run_dir / "run.json",
        {
            "schema_version": 1,
            "study_kind": "lagged_evidence_performance",
            "experiment_id": spec.experiment_id,
            "run_id": run_id,
            "status": "running",
            "claim_class": "EXPLORATORY",
            "metrics_opened": False,
            "spec_sha256": spec.sha256,
            "config_sha256": config.sha256,
            "implementation_sha256": implementation,
            "git_sha": _git_head(root),
            "git_worktree_clean": False,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "post_2023_accessed": False,
            "performance_claim_allowed": False,
            "paper_replication_claim_allowed": False,
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
                paths,
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
    choices = pd.DataFrame(
        [row for market in spec.markets for row in results[market]["choices"]]
    )
    traces = pd.DataFrame(
        [row for market in spec.markets for row in results[market]["traces"]]
    )
    boundaries = pd.DataFrame(
        [row for market in spec.markets for row in results[market]["boundaries"]]
    )
    summary.to_csv(run_dir / "summary.csv", index=False)
    choices.to_csv(run_dir / "choices.csv", index=False)
    traces.to_csv(run_dir / "change-traces.csv", index=False)
    boundaries.to_csv(run_dir / "boundaries.csv", index=False)
    decision = _decision(summary)
    write_json(run_dir / "decision.json", decision)
    metadata = read_json(run_dir / "run.json")
    metadata.update(
        {
            "status": "complete",
            "metrics_opened": True,
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "decision": decision["result"],
        }
    )
    write_json(run_dir / "run.json", metadata)
    write_inventory(run_dir)
    verify_lagged_performance_run(run_dir, config, spec)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lagged-evidence-performance")
    parser.add_argument("--config", default="research.toml")
    parser.add_argument(
        "--spec", default="research/lagged-evidence-performance-001.toml"
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
    spec = load_lagged_performance_spec(spec_path, config)
    if args.verify:
        print(
            json.dumps(
                verify_lagged_performance_run(args.verify, config, spec), sort_keys=True
            )
        )
    elif args.smoke:
        print(json.dumps(run_us_smoke(config, spec), sort_keys=True))
    else:
        print(run_lagged_performance_study(config, spec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
