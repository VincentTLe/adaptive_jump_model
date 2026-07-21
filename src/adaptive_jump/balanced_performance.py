"""Frozen development-sample P&L readout for the pair-balanced lagged JM."""

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

SPEC_SHA256 = "3ae665413a01622be6bdda19a4a9eba6b063a65ef75607fdbb6153396f97f7cc"
MODELS = ("fixed", "lagged_log4", "balanced_log4")
BASELINES = ("fixed", "lagged_log4")
MARKETS = ("us", "de", "jp")
LAMBDAS = (0.0, 5.0, 15.0, 35.0, 70.0, 150.0, 300.0, 600.0, 1200.0)
BETA = math.log(4.0)
TURNOVER_SCALE = 0.5
MDD_DEADBAND = 1e-9
METRIC_NAMES = (
    "cagr",
    "volatility",
    "sharpe",
    "maximum_drawdown",
    "calmar",
    "expected_shortfall_5pct",
    "turnover",
    "leverage",
    "cash_fraction",
    "switch_count",
)
DELTA_METRICS = (
    "sharpe",
    "maximum_drawdown",
    "turnover",
    "cash_fraction",
    "switch_count",
)


class BalancedPerformanceError(RuntimeError):
    """Raised when the frozen P&L contract or its evidence is violated."""


@dataclass(frozen=True)
class BalancedPerformanceSpec:
    path: Path
    sha256: str
    experiment_id: str
    fixed_run_id: str
    fixed_inventory_sha256: str
    data_manifest_sha256: str
    lagged_run_id: str
    lagged_inventory_sha256: str
    lagged_spec_sha256: str
    balanced_run_id: str
    balanced_inventory_sha256: str
    balanced_run_json_sha256: str
    balanced_spec_sha256: str
    balanced_candidate_sha256: dict[str, str]
    oracle_run_id: str
    oracle_inventory_sha256: str
    oracle_run_json_sha256: str
    oracle_spec_sha256: str
    cutoff: date
    markets: tuple[str, ...]
    outer_start: dict[str, date]
    outer_end: date
    lambdas: tuple[float, ...]
    beta: float
    artifact_subdir: Path


@dataclass(frozen=True)
class PerformanceSources:
    fixed: Path
    lagged: Path
    balanced: Path
    oracle: Path
    source_lock: dict[str, Any]


@dataclass(frozen=True)
class MarketEvidence:
    summary: pd.DataFrame
    choices: pd.DataFrame
    surface: pd.DataFrame
    boundaries: pd.DataFrame
    timeline: pd.DataFrame
    traces: pd.DataFrame
    trades: dict[str, pd.DataFrame]


def _is_hash(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def load_balanced_performance_spec(
    path: str | Path, config: ResearchConfig
) -> BalancedPerformanceSpec:
    """Load the exact frozen post-result P&L contract."""
    spec_path = Path(path).resolve()
    payload = spec_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    try:
        doc = tomllib.loads(payload.decode())
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise BalancedPerformanceError(
            f"invalid balanced performance spec: {exc}"
        ) from exc

    fixed = doc.get("fixed_source", {})
    lagged = doc.get("lagged_source", {})
    balanced = doc.get("balanced_source", {})
    oracle = doc.get("lagged_performance_oracle", {})
    model = doc.get("model", {})
    protocol = doc.get("protocol", {})
    decision = doc.get("decision", {})
    verification = doc.get("verification", {})
    execution = doc.get("execution", {})
    storage = doc.get("storage", {})
    candidate_hashes = balanced.get("candidate_sha256", {})
    expected_outer = {"us": "2007-12-04", "de": "2008-01-03", "jp": "2009-05-07"}
    controls = (
        digest == SPEC_SHA256,
        doc.get("schema_version") == 1,
        doc.get("experiment_id") == "balanced-lagged-performance-001",
        doc.get("claim_class") == "EXPLORATORY",
        doc.get("stage") == "POST_RESULT_DEVELOPMENT_SAMPLE_PNL_STUDY",
        doc.get("performance_claim_allowed") is False,
        doc.get("paper_replication_claim_allowed") is False,
        doc.get("holdout_claim_allowed") is False,
        doc.get("post_2023_access") is False,
        doc.get("provider_access") is False,
        fixed.get("config_sha256") == config.sha256,
        fixed.get("data_cutoff") == "2023-12-31",
        balanced.get("required_result") == "not_supported",
        balanced.get("required_beta_label") == "log4",
        oracle.get("role")
        == (
            "fixed and lagged choices, signals, trades, and metrics parity oracle "
            "only; it cannot determine or rescue the balanced result"
        ),
        tuple(model.get("models", ())) == MODELS,
        model.get("primary_challenger") == "balanced_log4",
        model.get("primary_baseline") == "lagged_log4",
        model.get("required_benchmark") == "fixed",
        tuple(float(value) for value in model.get("raw_lambda_grid", ())) == LAMBDAS,
        float(model.get("beta", math.nan)) == BETA,
        model.get("fitted_parameters")
        == "sealed v7 scalers and centers; no refit or state regeneration in the P&L runner",  # noqa: E501
        tuple(protocol.get("markets", ())) == MARKETS,
        protocol.get("outer_start") == expected_outer,
        protocol.get("outer_end") == "2023-12-29",
        protocol.get("selection")
        == "separate monthly lambda selection for fixed, lagged_log4, and balanced_log4 candidates",  # noqa: E501
        protocol.get("selection_objective")
        == "annualized net strategy excess Sharpe over the previous eight calendar years",  # noqa: E501
        protocol.get("minimum_valid_returns")
        == config.selection_protocol.minimum_valid_returns,
        protocol.get("tie_rule")
        == "lower lambda within the existing numerical tie tolerance",
        protocol.get("selection_surface_required") is True,
        protocol.get("primary_delay_trading_days")
        == config.backtest_protocol.primary_delay
        == 1,
        protocol.get("signal_to_return_offset")
        == config.backtest_protocol.return_offset
        == 2,
        protocol.get("one_way_cost_bps")
        == config.backtest_protocol.one_way_cost_bps
        == 10,
        protocol.get("turnover") == "0.5*252*mean(abs(position change))",
        tuple(protocol.get("metrics", ())) == METRIC_NAMES,
        decision.get("primary_contrast") == "balanced_log4_minus_lagged_log4",
        decision.get("required_fixed_contrast") == "balanced_log4_minus_fixed",
        decision.get("maximum_drawdown_reporting_deadband") == MDD_DEADBAND,
        decision.get("raw_metrics_always_reported") is True,
        decision.get("selection_or_grid_change_after_results") is False,
        decision.get("non_finite_primary_metrics") == "error; never omit a market",
        all(
            verification.get(key) is True
            for key in (
                "registry_and_spec_lock_exact",
                "source_inventory_and_run_hashes_exact",
                "balanced_candidate_hashes_exact",
                "fixed_and_lagged_choices_signals_trades_metrics_equal_oracle",
                "same_dates_information_selection_cost_and_delay",
                "signal_equals_one_minus_selected_state",
                "t_plus_2_position_identity",
                "ten_bps_cost_identity",
                "paper_turnover_recomputed",
                "full_selection_surface_saved",
                "concrete_state_signal_position_trade_cost_return_trace",
                "all_dates_no_later_than_2023",
                "independent_metric_and_decision_recomputation",
                "artifact_allowlist_and_inventory_exact",
                "git_head_is_informational_only",
            )
        ),
        execution.get("us_smoke_before_metrics") is True,
        execution.get("full_markets_parallel") is True,
        execution.get("market_workers") == 3,
        execution.get("threadpool_limit_per_worker") == 1,
        execution.get("gpu_required") is False,
        storage.get("summary_csv") == "summary.csv",
        storage.get("choices_csv") == "choices.csv",
        storage.get("selection_surface_csv") == "selection-surface.csv",
        storage.get("boundaries_csv") == "boundaries.csv",
        storage.get("change_traces_csv") == "change-traces.csv",
        storage.get("selected_timeline_csv") == "selected-timeline.csv",
        storage.get("decision_json") == "decision.json",
        storage.get("source_lock_json") == "source-lock.json",
        storage.get("implementation_lock_json") == "implementation-lock.json",
        storage.get("generated_outputs_tracked") is False,
        set(candidate_hashes) == set(MARKETS),
        all(_is_hash(value) for value in candidate_hashes.values()),
        all(
            _is_hash(value)
            for value in (
                fixed.get("run_inventory_sha256"),
                fixed.get("data_manifest_sha256"),
                fixed.get("config_sha256"),
                lagged.get("run_inventory_sha256"),
                lagged.get("spec_sha256"),
                balanced.get("run_inventory_sha256"),
                balanced.get("run_json_sha256"),
                balanced.get("spec_sha256"),
                oracle.get("run_inventory_sha256"),
                oracle.get("run_json_sha256"),
                oracle.get("spec_sha256"),
            )
        ),
    )
    if not all(controls):
        raise BalancedPerformanceError("balanced performance controls changed")
    artifact_subdir = Path(str(storage.get("artifact_subdir", "")))
    if (
        artifact_subdir != Path("balanced-lagged-performance-001")
        or artifact_subdir.is_absolute()
        or ".." in artifact_subdir.parts
    ):
        raise BalancedPerformanceError("invalid balanced performance artifact path")
    return BalancedPerformanceSpec(
        path=spec_path,
        sha256=digest,
        experiment_id=str(doc["experiment_id"]),
        fixed_run_id=str(fixed["run_id"]),
        fixed_inventory_sha256=str(fixed["run_inventory_sha256"]),
        data_manifest_sha256=str(fixed["data_manifest_sha256"]),
        lagged_run_id=str(lagged["run_id"]),
        lagged_inventory_sha256=str(lagged["run_inventory_sha256"]),
        lagged_spec_sha256=str(lagged["spec_sha256"]),
        balanced_run_id=str(balanced["run_id"]),
        balanced_inventory_sha256=str(balanced["run_inventory_sha256"]),
        balanced_run_json_sha256=str(balanced["run_json_sha256"]),
        balanced_spec_sha256=str(balanced["spec_sha256"]),
        balanced_candidate_sha256={
            key: str(value) for key, value in candidate_hashes.items()
        },
        oracle_run_id=str(oracle["run_id"]),
        oracle_inventory_sha256=str(oracle["run_inventory_sha256"]),
        oracle_run_json_sha256=str(oracle["run_json_sha256"]),
        oracle_spec_sha256=str(oracle["spec_sha256"]),
        cutoff=date.fromisoformat(str(fixed["data_cutoff"])),
        markets=tuple(protocol["markets"]),
        outer_start={
            key: date.fromisoformat(value)
            for key, value in protocol["outer_start"].items()
        },
        outer_end=date.fromisoformat(str(protocol["outer_end"])),
        lambdas=tuple(float(value) for value in model["raw_lambda_grid"]),
        beta=float(model["beta"]),
        artifact_subdir=artifact_subdir,
    )


def _registry_lock(root: Path, spec: BalancedPerformanceSpec) -> None:
    rows: list[dict[str, Any]] = []
    for line in (
        (root / "research/experiment_registry.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ):
        row = json.loads(line)
        if row.get("experiment_id") == spec.experiment_id:
            rows.append(row)
    if (
        not rows
        or rows[-1].get("frozen_spec_hash") != spec.sha256
        or rows[-1].get("status") not in {"FROZEN", "EXPERIMENT_COMPLETE"}
    ):
        raise BalancedPerformanceError("balanced performance registry lock changed")


def _inventory_files(run_dir: Path, expected_hash: str) -> dict[str, str]:
    inventory = run_dir / "inventory.json"
    if sha256_file(inventory) != expected_hash:
        raise BalancedPerformanceError(
            f"source inventory identity changed: {run_dir.name}"
        )
    verify_inventory(run_dir)
    files = read_json(inventory).get("files")
    if not isinstance(files, dict) or not all(
        isinstance(key, str) and _is_hash(value) for key, value in files.items()
    ):
        raise BalancedPerformanceError(
            f"source inventory schema changed: {run_dir.name}"
        )
    return files


def _namespaced_inventory_entries(
    inventories: dict[str, dict[str, str]],
) -> dict[str, str]:
    return dict(
        sorted(
            (f"{label}/{relative}", digest)
            for label, files in inventories.items()
            for relative, digest in files.items()
        )
    )


def _locked_file(
    run_dir: Path,
    inventory: dict[str, str],
    relative: str,
    used: dict[str, str],
    label: str,
) -> Path:
    path = run_dir / relative
    digest = inventory.get(relative)
    if not isinstance(digest, str) or sha256_file(path) != digest:
        raise BalancedPerformanceError(f"source file changed: {label}/{relative}")
    used[f"{label}/{relative}"] = digest
    return path


def _assert_run_metadata(
    metadata: dict[str, Any],
    *,
    experiment_id: str | None,
    run_id: str,
    spec_sha256: str | None = None,
    result: str | None = None,
    required_false: tuple[str, ...] = (),
) -> None:
    controls = (
        metadata.get("schema_version") == 1,
        experiment_id is None or metadata.get("experiment_id") == experiment_id,
        metadata.get("run_id") == run_id,
        metadata.get("status") == "complete",
        spec_sha256 is None or metadata.get("spec_sha256") == spec_sha256,
        result is None or metadata.get("result", metadata.get("decision")) == result,
        all(metadata.get(field) is False for field in required_false),
    )
    if not all(controls):
        raise BalancedPerformanceError(f"source run metadata changed: {run_id}")


def verify_performance_sources(
    root: Path, config: ResearchConfig, spec: BalancedPerformanceSpec
) -> PerformanceSources:
    """Verify every frozen source identity before opening balanced metrics."""
    _registry_lock(root, spec)
    paths = PerformanceSources(
        fixed=root / config.artifact_root / "fixed-baselines" / spec.fixed_run_id,
        lagged=root
        / config.artifact_root
        / "lagged-evidence-mechanism-001"
        / spec.lagged_run_id,
        balanced=root
        / config.artifact_root
        / "balanced-lagged-mechanism-001"
        / spec.balanced_run_id,
        oracle=root
        / config.artifact_root
        / "lagged-evidence-performance-001"
        / spec.oracle_run_id,
        source_lock={},
    )
    if any(
        item.is_symlink()
        for run in (paths.fixed, paths.lagged, paths.balanced, paths.oracle)
        for item in (run, *run.rglob("*"))
    ):
        raise BalancedPerformanceError("source runs may not contain symlinks")
    inventories = {
        "fixed": _inventory_files(paths.fixed, spec.fixed_inventory_sha256),
        "lagged": _inventory_files(paths.lagged, spec.lagged_inventory_sha256),
        "balanced": _inventory_files(paths.balanced, spec.balanced_inventory_sha256),
        "oracle": _inventory_files(paths.oracle, spec.oracle_inventory_sha256),
    }
    if sha256_file(paths.fixed / "data-manifest.json") != spec.data_manifest_sha256:
        raise BalancedPerformanceError("fixed data manifest changed")
    if sha256_file(paths.fixed / "config.lock.toml") != config.sha256:
        raise BalancedPerformanceError("fixed config lock changed")
    if sha256_file(paths.lagged / "study.lock.toml") != spec.lagged_spec_sha256:
        raise BalancedPerformanceError("lagged mechanism spec changed")
    if sha256_file(paths.balanced / "study.lock.toml") != spec.balanced_spec_sha256:
        raise BalancedPerformanceError("balanced mechanism spec changed")
    if sha256_file(paths.oracle / "study.lock.toml") != spec.oracle_spec_sha256:
        raise BalancedPerformanceError("lagged performance oracle spec changed")
    if sha256_file(paths.oracle / "config.lock.toml") != config.sha256:
        raise BalancedPerformanceError("lagged performance oracle config changed")
    if sha256_file(paths.balanced / "run.json") != spec.balanced_run_json_sha256:
        raise BalancedPerformanceError("balanced mechanism run metadata hash changed")
    if sha256_file(paths.oracle / "run.json") != spec.oracle_run_json_sha256:
        raise BalancedPerformanceError(
            "lagged performance oracle metadata hash changed"
        )

    fixed_meta = read_json(paths.fixed / "run.json")
    lagged_meta = read_json(paths.lagged / "run.json")
    balanced_meta = read_json(paths.balanced / "run.json")
    oracle_meta = read_json(paths.oracle / "run.json")
    _assert_run_metadata(
        fixed_meta,
        experiment_id=None,
        run_id=spec.fixed_run_id,
    )
    if (
        fixed_meta.get("config_sha256") != config.sha256
        or fixed_meta.get("data_manifest_sha256") != spec.data_manifest_sha256
    ):
        raise BalancedPerformanceError("fixed source metadata changed")
    _assert_run_metadata(
        lagged_meta,
        experiment_id="lagged-evidence-mechanism-001",
        run_id=spec.lagged_run_id,
        spec_sha256=spec.lagged_spec_sha256,
        result="supported",
        required_false=("post_2023_accessed",),
    )
    if lagged_meta.get("selected_beta_label") != "log4":
        raise BalancedPerformanceError("lagged source selected beta changed")
    _assert_run_metadata(
        balanced_meta,
        experiment_id="balanced-lagged-mechanism-001",
        run_id=spec.balanced_run_id,
        spec_sha256=spec.balanced_spec_sha256,
        result="not_supported",
        required_false=(
            "post_2023_accessed",
            "provider_accessed",
            "performance_claim_allowed",
            "paper_replication_claim_allowed",
        ),
    )
    if balanced_meta.get("decision_beta_label") != "log4":
        raise BalancedPerformanceError("balanced source beta changed")
    _assert_run_metadata(
        oracle_meta,
        experiment_id="lagged-evidence-performance-001",
        run_id=spec.oracle_run_id,
        spec_sha256=spec.oracle_spec_sha256,
        result="supported",
        required_false=(
            "post_2023_accessed",
            "performance_claim_allowed",
            "paper_replication_claim_allowed",
        ),
    )
    if oracle_meta.get("metrics_opened") is not True:
        raise BalancedPerformanceError("lagged performance oracle is incomplete")
    conclusion = read_json(paths.balanced / "conclusion.json")
    if (
        conclusion.get("result") != "not_supported"
        or conclusion.get("decision_beta_label") != "log4"
    ):
        raise BalancedPerformanceError("balanced mechanism conclusion changed")

    used: dict[str, str] = {}
    for relative in ("config.lock.toml", "data-manifest.json", "metrics.csv"):
        _locked_file(paths.fixed, inventories["fixed"], relative, used, "fixed")
    for relative in ("study.lock.toml", "conclusion.json"):
        _locked_file(paths.lagged, inventories["lagged"], relative, used, "lagged")
    for relative in ("study.lock.toml", "conclusion.json"):
        _locked_file(
            paths.balanced, inventories["balanced"], relative, used, "balanced"
        )
    for relative in ("study.lock.toml", "config.lock.toml", "summary.csv"):
        _locked_file(paths.oracle, inventories["oracle"], relative, used, "oracle")
    for market in spec.markets:
        for relative in (
            f"{market}/features.csv",
            f"{market}/jm-states.csv",
            f"{market}/fixed_jm-delay-1/choices.csv",
            f"{market}/fixed_jm-delay-1/selected-signal.csv",
            f"{market}/trades/fixed_jm-delay-1.csv",
        ):
            _locked_file(paths.fixed, inventories["fixed"], relative, used, "fixed")
        lagged_relative = f"{market}/candidate-states-beta-log4.csv"
        _locked_file(
            paths.lagged, inventories["lagged"], lagged_relative, used, "lagged"
        )
        balanced_relative = f"{market}/candidate-states-balanced-beta-log4.csv"
        balanced_path = _locked_file(
            paths.balanced,
            inventories["balanced"],
            balanced_relative,
            used,
            "balanced",
        )
        if sha256_file(balanced_path) != spec.balanced_candidate_sha256[market]:
            raise BalancedPerformanceError(f"{market}: balanced candidate hash changed")
        for relative in (
            f"{market}/choices.csv",
            f"{market}/summary.csv",
            f"{market}/trades/fixed.csv",
            f"{market}/trades/lagged_log4.csv",
        ):
            _locked_file(paths.oracle, inventories["oracle"], relative, used, "oracle")
    explicitly_locked = dict(sorted(used.items()))
    integrity_hashed = _namespaced_inventory_entries(inventories)
    source_lock = {
        "schema_version": 2,
        "fixed_run_id": spec.fixed_run_id,
        "fixed_inventory_sha256": spec.fixed_inventory_sha256,
        "lagged_run_id": spec.lagged_run_id,
        "lagged_inventory_sha256": spec.lagged_inventory_sha256,
        "lagged_spec_sha256": spec.lagged_spec_sha256,
        "balanced_run_id": spec.balanced_run_id,
        "balanced_inventory_sha256": spec.balanced_inventory_sha256,
        "balanced_run_json_sha256": spec.balanced_run_json_sha256,
        "balanced_spec_sha256": spec.balanced_spec_sha256,
        "balanced_candidate_sha256": spec.balanced_candidate_sha256,
        "oracle_run_id": spec.oracle_run_id,
        "oracle_inventory_sha256": spec.oracle_inventory_sha256,
        "oracle_run_json_sha256": spec.oracle_run_json_sha256,
        "oracle_spec_sha256": spec.oracle_spec_sha256,
        "data_manifest_sha256": spec.data_manifest_sha256,
        "files_explicitly_locked_count": len(explicitly_locked),
        "files_explicitly_locked_sha256": explicitly_locked,
        "inventory_entries_integrity_hashed_count": len(integrity_hashed),
        "inventory_entries_integrity_hashed_sha256": integrity_hashed,
        "columns_read": {
            "fixed/features.csv": [
                "date",
                "equity_simple",
                "equity_log",
                "gap_calendar_days",
                "cash_available_date",
                "cash_observation_date",
                "cash_age_days",
                "cash_yield_percent",
                "cash_return",
                "excess_return",
                "dd_10",
                "sortino_20",
                "sortino_60",
            ],
            "candidate_states": ["date", *[str(value) for value in spec.lambdas]],
            "oracle": ["choices", "signals", "trades", "metrics"],
        },
        "columns_used_for_selection_and_accounting": {
            "fixed/features.csv": ["date", "equity_simple", "cash_return"],
        },
        "post_2023_accessed": False,
        "provider_accessed": False,
    }
    return PerformanceSources(
        fixed=paths.fixed,
        lagged=paths.lagged,
        balanced=paths.balanced,
        oracle=paths.oracle,
        source_lock=source_lock,
    )


def _git_information(root: Path) -> tuple[str, str]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    descriptive = subprocess.run(
        ["git", "describe", "--always", "--dirty", "--broken"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return head, descriptive


def implementation_lock(root: Path, spec: BalancedPerformanceSpec) -> dict[str, Any]:
    """Hash scientific content; Git fields are descriptive, never identity gates."""
    paths = (
        spec.path,
        root / "research.toml",
        root / "pyproject.toml",
        root / "uv.lock",
        root / "src/adaptive_jump/artifacts.py",
        root / "src/adaptive_jump/backtest.py",
        root / "src/adaptive_jump/confidence_evaluation.py",
        root / "src/adaptive_jump/confidence_model.py",
        root / "src/adaptive_jump/config.py",
        root / "src/adaptive_jump/walkforward.py",
        root / "src/adaptive_jump/balanced_performance.py",
    )
    files = {str(path.relative_to(root)): sha256_file(path) for path in paths}
    digest = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    head, descriptive = _git_information(root)
    return {
        "schema_version": 1,
        "implementation_sha256": digest,
        "git_head_informational": head,
        "git_describe_informational": descriptive,
        "files": files,
    }


def _read_states(path: Path, spec: BalancedPerformanceSpec) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except (FileNotFoundError, OSError, pd.errors.ParserError) as exc:
        raise BalancedPerformanceError(f"cannot read candidate states: {path}") from exc
    if (
        "date" not in frame
        or tuple(float(column) for column in frame.columns[1:]) != spec.lambdas
    ):
        raise BalancedPerformanceError(f"candidate-state schema changed: {path}")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    dates = pd.DatetimeIndex(frame.pop("date"), name="date")
    frame.index = dates
    frame.columns = spec.lambdas
    values = frame.stack()
    if (
        dates.empty
        or dates.has_duplicates
        or not dates.is_monotonic_increasing
        or dates.max().date() > spec.cutoff
        or not values.isin([0.0, 1.0]).all()
    ):
        raise BalancedPerformanceError(f"candidate-state values changed: {path}")
    return frame


def _load_market(
    market: str,
    sources: PerformanceSources,
    spec: BalancedPerformanceSpec,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    frame = _load_parent_frame(sources.fixed, market, spec.cutoff)
    fixed = _parent_states(sources.fixed, market, spec.lambdas)
    lagged = _read_states(
        sources.lagged / market / "candidate-states-beta-log4.csv", spec
    )
    balanced = _read_states(
        sources.balanced / market / "candidate-states-balanced-beta-log4.csv",
        spec,
    )
    for model, candidate in (("lagged_log4", lagged), ("balanced_log4", balanced)):
        if not candidate.index.equals(fixed.index):
            raise BalancedPerformanceError(f"{market}/{model}: candidate dates changed")
    frame_dates = pd.DatetimeIndex(pd.to_datetime(frame["date"], errors="raise"))
    if not fixed.index.isin(frame_dates).all():
        raise BalancedPerformanceError(
            f"{market}: candidate dates are outside source frame"
        )
    return frame, {"fixed": fixed, "lagged_log4": lagged, "balanced_log4": balanced}


def _select_paths(
    frame: pd.DataFrame,
    states: dict[str, pd.DataFrame],
    config: ResearchConfig,
) -> dict[str, SelectionResult]:
    returns = frame[["date", "equity_simple", "cash_return"]]
    selections = {
        model: select_monthly_candidate(
            returns,
            states[model],
            config.selection_protocol,
            delay_trading_days=config.backtest_protocol.primary_delay,
            one_way_cost_bps=config.backtest_protocol.one_way_cost_bps,
            periods_per_year=config.metrics_protocol.periods_per_year,
            volatility_ddof=config.metrics_protocol.volatility_ddof,
        )
        for model in MODELS
    }
    reference_choices = pd.to_datetime(
        selections["fixed"].choices["decision_date"], errors="raise"
    ).to_numpy()
    reference_surface = pd.to_datetime(
        selections["fixed"].surface["decision_date"], errors="raise"
    ).to_numpy()
    for model in MODELS:
        selected = selections[model]
        if selected.surface.empty or tuple(selected.surface.columns) != (
            "decision_date",
            "candidate",
            "valid_returns",
            "sharpe",
            "eligible",
        ):
            raise BalancedPerformanceError(f"{model}: full selection surface missing")
        if tuple(sorted(selected.surface["candidate"].unique())) != LAMBDAS:
            raise BalancedPerformanceError(f"{model}: selection grid changed")
        if not np.array_equal(
            pd.to_datetime(
                selected.choices["decision_date"], errors="raise"
            ).to_numpy(),
            reference_choices,
        ) or not np.array_equal(
            pd.to_datetime(
                selected.surface["decision_date"], errors="raise"
            ).to_numpy(),
            reference_surface,
        ):
            raise BalancedPerformanceError("model selection budgets differ")
    return selections


def _full_path(
    frame: pd.DataFrame, selection: SelectionResult, config: ResearchConfig
) -> pd.DataFrame:
    return apply_signal(
        frame[["date", "equity_simple", "cash_return"]],
        selection.signal.reset_index(drop=True),
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
    output = metrics.copy()
    if list(output["model"]) != list(MODELS):
        raise BalancedPerformanceError("market metric model order changed")
    for baseline in BASELINES:
        rows = output.loc[output["model"] == baseline]
        if len(rows) != 1:
            raise BalancedPerformanceError(f"market has no unique {baseline} row")
        base = rows.iloc[0]
        for metric in DELTA_METRICS:
            column = f"delta_vs_{baseline}_{metric}"
            output[column] = output[metric] - base[metric]
        raw = output[f"delta_vs_{baseline}_maximum_drawdown"]
        output[f"delta_vs_{baseline}_maximum_drawdown_reported"] = raw.mask(
            raw.abs() <= MDD_DEADBAND, 0.0
        )
    return output


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
        raise BalancedPerformanceError(f"{label}: table schema or shape changed")
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
        raise BalancedPerformanceError(f"{label}: stored values changed") from exc
    maximum = 0.0
    for column in left.columns:
        left_numeric = pd.to_numeric(left[column], errors="coerce")
        right_numeric = pd.to_numeric(right[column], errors="coerce")
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
        raise BalancedPerformanceError(f"cannot read artifact CSV: {path}") from exc


def _oracle_choices_and_trades(
    market: str,
    selections: dict[str, SelectionResult],
    aligned: dict[str, pd.DataFrame],
    sources: PerformanceSources,
    config: ResearchConfig,
) -> dict[str, int]:
    stored_choices = _read_csv(sources.oracle / market / "choices.csv")
    choice_rows = 0
    trade_rows = 0
    for model in BASELINES:
        observed_choices = selections[model].choices.assign(market=market, model=model)
        expected_choices = stored_choices.loc[
            stored_choices["model"] == model, observed_choices.columns
        ].reset_index(drop=True)
        _assert_frame_close(
            observed_choices,
            expected_choices,
            f"{market}/{model} oracle choices",
            tolerance=0,
        )
        expected_trade = read_trade_path(
            sources.oracle / market / "trades" / f"{model}.csv",
            config.backtest_protocol.primary_delay,
            config.backtest_protocol.one_way_cost_bps,
        )
        _assert_frame_close(
            aligned[model],
            expected_trade,
            f"{market}/{model} oracle trades",
            tolerance=1e-15,
        )
        choice_rows += len(observed_choices)
        trade_rows += len(aligned[model])
    return {"choice_rows": choice_rows, "trade_rows": trade_rows}


def _oracle_metrics(
    market: str,
    metrics: pd.DataFrame,
    sources: PerformanceSources,
) -> int:
    expected = _read_csv(sources.oracle / market / "summary.csv")
    columns = ["market", "model", "start", "end", "observations", *METRIC_NAMES]
    checked = 0
    for model in BASELINES:
        observed_row = metrics.loc[metrics["model"] == model, columns].reset_index(
            drop=True
        )
        expected_row = expected.loc[expected["model"] == model, columns].reset_index(
            drop=True
        )
        _assert_frame_close(
            observed_row,
            expected_row,
            f"{market}/{model} oracle metrics",
            tolerance=1e-12,
        )
        checked += 1
    return checked


def _governing_choice(choices: pd.DataFrame, signal_date: pd.Timestamp) -> pd.Series:
    dates = pd.to_datetime(choices["decision_date"], errors="raise")
    prior = choices.loc[dates <= signal_date]
    if prior.empty:
        raise BalancedPerformanceError("signal has no governing monthly choice")
    return prior.iloc[-1]


TIMELINE_COLUMNS = (
    "market",
    "model",
    "signal_date",
    "execution_date",
    "offset_observations",
    "selection_decision_date",
    "active_lambda",
    "previous_active_lambda",
    "lambda_changed",
    "candidate_previous_same_lambda_state",
    "previous_emitted_state",
    "state",
    "same_lambda_candidate_changed",
    "emitted_state_changed",
    "signal",
    "position",
    "previous_position",
    "signed_trade",
    "one_way_turnover",
    "transaction_cost",
    "equity_simple",
    "cash_return",
    "gross_return",
    "strategy_return",
)


def _selected_timeline(
    market: str,
    model: str,
    frame: pd.DataFrame,
    states: pd.DataFrame,
    selection: SelectionResult,
    full: pd.DataFrame,
    aligned: pd.DataFrame,
    config: ResearchConfig,
) -> pd.DataFrame:
    dates = pd.DatetimeIndex(pd.to_datetime(frame["date"], errors="raise"), name="date")
    active = _active_lambda(dates, selection.choices)
    selected_state = 1.0 - selection.signal.reindex(dates)
    full_frame = full.copy()
    full_frame["date"] = pd.to_datetime(full_frame["date"], errors="raise")
    full_dates = pd.DatetimeIndex(full_frame["date"])
    if not full_dates.equals(dates):
        raise BalancedPerformanceError(f"{market}/{model}: accounting dates changed")
    previous_position = full_frame["position"].ffill().shift(1)
    signed_trade = full_frame["position"] - previous_position
    first_valid = full_frame["position"].first_valid_index()
    if first_valid is not None:
        signed_trade.loc[first_valid] = 0.0
    positions = {timestamp: row for row, timestamp in enumerate(dates)}
    offset = config.backtest_protocol.return_offset
    records: list[dict[str, Any]] = []
    for execution_date in pd.to_datetime(aligned["date"], errors="raise"):
        execution_row = positions[pd.Timestamp(execution_date)]
        signal_row = execution_row - offset
        if signal_row < 0:
            raise BalancedPerformanceError(
                f"{market}/{model}: execution lacks t+2 signal"
            )
        signal_date = dates[signal_row]
        penalty = active.iloc[signal_row]
        current_state = selected_state.iloc[signal_row]
        if pd.isna(penalty) or pd.isna(current_state):
            raise BalancedPerformanceError(
                f"{market}/{model}: selected state is missing"
            )
        previous_lambda = active.iloc[signal_row - 1] if signal_row else math.nan
        previous_emitted = (
            selected_state.iloc[signal_row - 1] if signal_row else math.nan
        )
        candidate_previous = (
            states.loc[dates[signal_row - 1], float(penalty)]
            if signal_row
            else math.nan
        )
        lambda_changed = pd.notna(previous_lambda) and float(previous_lambda) != float(
            penalty
        )
        same_lambda_changed = pd.notna(candidate_previous) and int(
            candidate_previous
        ) != int(current_state)
        emitted_changed = pd.notna(previous_emitted) and int(previous_emitted) != int(
            current_state
        )
        trade = full_frame.iloc[execution_row]
        prior_position = previous_position.iloc[execution_row]
        signed = signed_trade.iloc[execution_row]
        if pd.isna(prior_position):
            prior_position = trade["position"]
        if pd.isna(signed):
            signed = 0.0
        expected_gross = float(trade["position"]) * float(trade["equity_simple"]) + (
            1.0 - float(trade["position"])
        ) * float(trade["cash_return"])
        controls = (
            float(trade["position"]) == float(selection.signal.iloc[signal_row]),
            math.isclose(
                abs(float(signed)),
                float(trade["one_way_turnover"]),
                rel_tol=0,
                abs_tol=1e-15,
            ),
            math.isclose(
                float(trade["transaction_cost"]),
                float(trade["one_way_turnover"]) * 0.001,
                rel_tol=0,
                abs_tol=1e-15,
            ),
            math.isclose(
                float(trade["gross_return"]), expected_gross, rel_tol=0, abs_tol=1e-15
            ),
            math.isclose(
                float(trade["strategy_return"]),
                float(trade["gross_return"]) - float(trade["transaction_cost"]),
                rel_tol=0,
                abs_tol=1e-15,
            ),
        )
        if not all(controls):
            raise BalancedPerformanceError(
                f"{market}/{model}: t+2 accounting identity failed"
            )
        choice = _governing_choice(selection.choices, signal_date)
        records.append(
            {
                "market": market,
                "model": model,
                "signal_date": signal_date,
                "execution_date": pd.Timestamp(execution_date),
                "offset_observations": offset,
                "selection_decision_date": choice["decision_date"],
                "active_lambda": float(penalty),
                "previous_active_lambda": previous_lambda,
                "lambda_changed": bool(lambda_changed),
                "candidate_previous_same_lambda_state": candidate_previous,
                "previous_emitted_state": previous_emitted,
                "state": int(current_state),
                "same_lambda_candidate_changed": bool(same_lambda_changed),
                "emitted_state_changed": bool(emitted_changed),
                "signal": float(selection.signal.iloc[signal_row]),
                "position": float(trade["position"]),
                "previous_position": float(prior_position),
                "signed_trade": float(signed),
                "one_way_turnover": float(trade["one_way_turnover"]),
                "transaction_cost": float(trade["transaction_cost"]),
                "equity_simple": float(trade["equity_simple"]),
                "cash_return": float(trade["cash_return"]),
                "gross_return": float(trade["gross_return"]),
                "strategy_return": float(trade["strategy_return"]),
            }
        )
    return pd.DataFrame.from_records(records, columns=TIMELINE_COLUMNS)


TRACE_COLUMNS = (
    "market",
    "baseline",
    "challenger",
    "event_type",
    "signal_date",
    "execution_date",
    "offset_observations",
    "baseline_lambda",
    "challenger_lambda",
    "baseline_lambda_changed",
    "challenger_lambda_changed",
    "baseline_candidate_previous_same_lambda_state",
    "challenger_candidate_previous_same_lambda_state",
    "baseline_state",
    "challenger_state",
    "baseline_signal",
    "challenger_signal",
    "baseline_position",
    "challenger_position",
    "baseline_signed_trade",
    "challenger_signed_trade",
    "baseline_turnover",
    "challenger_turnover",
    "baseline_cost",
    "challenger_cost",
    "baseline_gross_return",
    "challenger_gross_return",
    "baseline_strategy_return",
    "challenger_strategy_return",
)


def _change_traces(market: str, timelines: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    challenger = timelines["balanced_log4"].set_index("execution_date")
    for baseline_name in BASELINES:
        baseline = timelines[baseline_name].set_index("execution_date")
        if not challenger.index.equals(baseline.index):
            raise BalancedPerformanceError(f"{market}: timeline dates differ")
        seen: set[str] = set()
        for execution_date in challenger.index:
            left = baseline.loc[execution_date]
            right = challenger.loc[execution_date]
            flags = {
                "choice": left["active_lambda"] != right["active_lambda"],
                "state": left["state"] != right["state"],
                "position": left["position"] != right["position"],
                "trade": left["signed_trade"] != right["signed_trade"],
            }
            for event_type, changed in flags.items():
                if not changed or event_type in seen:
                    continue
                records.append(
                    {
                        "market": market,
                        "baseline": baseline_name,
                        "challenger": "balanced_log4",
                        "event_type": event_type,
                        "signal_date": right["signal_date"],
                        "execution_date": execution_date,
                        "offset_observations": int(right["offset_observations"]),
                        "baseline_lambda": float(left["active_lambda"]),
                        "challenger_lambda": float(right["active_lambda"]),
                        "baseline_lambda_changed": bool(left["lambda_changed"]),
                        "challenger_lambda_changed": bool(right["lambda_changed"]),
                        "baseline_candidate_previous_same_lambda_state": left[
                            "candidate_previous_same_lambda_state"
                        ],
                        "challenger_candidate_previous_same_lambda_state": right[
                            "candidate_previous_same_lambda_state"
                        ],
                        "baseline_state": int(left["state"]),
                        "challenger_state": int(right["state"]),
                        "baseline_signal": float(left["signal"]),
                        "challenger_signal": float(right["signal"]),
                        "baseline_position": float(left["position"]),
                        "challenger_position": float(right["position"]),
                        "baseline_signed_trade": float(left["signed_trade"]),
                        "challenger_signed_trade": float(right["signed_trade"]),
                        "baseline_turnover": float(left["one_way_turnover"]),
                        "challenger_turnover": float(right["one_way_turnover"]),
                        "baseline_cost": float(left["transaction_cost"]),
                        "challenger_cost": float(right["transaction_cost"]),
                        "baseline_gross_return": float(left["gross_return"]),
                        "challenger_gross_return": float(right["gross_return"]),
                        "baseline_strategy_return": float(left["strategy_return"]),
                        "challenger_strategy_return": float(right["strategy_return"]),
                    }
                )
                seen.add(event_type)
            if len(seen) == 4:
                break
    return pd.DataFrame.from_records(records, columns=TRACE_COLUMNS)


def _boundary_rows(
    market: str,
    selections: dict[str, SelectionResult],
    spec: BalancedPerformanceSpec,
    config: ResearchConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        diagnostic = boundary_diagnostic(
            selections[model].choices,
            spec.lambdas,
            oos_start=spec.outer_start[market],
            fraction_limit=config.selection_protocol.boundary_fraction_limit,
        )
        rows.append({"market": market, "model": model, **diagnostic.__dict__})
    return pd.DataFrame.from_records(rows)


def _market_evidence(
    market: str,
    sources: PerformanceSources,
    config: ResearchConfig,
    spec: BalancedPerformanceSpec,
) -> MarketEvidence:
    frame, states = _load_market(market, sources, spec)
    selections = _select_paths(frame, states, config)
    _assert_beta_zero_selection(sources.fixed, market, selections["fixed"])
    full = {model: _full_path(frame, selections[model], config) for model in MODELS}
    aligned = {
        model: _align_parent_sample(
            sources.fixed,
            market,
            full[model],
            beta_zero=model == "fixed",
        )
        for model in MODELS
    }
    for model in MODELS:
        start = pd.Timestamp(aligned[model]["date"].iloc[0]).date()
        end = pd.Timestamp(aligned[model]["date"].iloc[-1]).date()
        if start != spec.outer_start[market] or end != spec.outer_end:
            raise BalancedPerformanceError(f"{market}/{model}: outer sample changed")
    _oracle_choices_and_trades(market, selections, aligned, sources, config)
    metrics = _add_deltas(
        pd.DataFrame(
            [_metric_row(market, model, aligned[model], config) for model in MODELS]
        )
    )
    _oracle_metrics(market, metrics, sources)
    choices = pd.concat(
        [
            selections[model].choices.assign(market=market, model=model)
            for model in MODELS
        ],
        ignore_index=True,
    )
    surface = pd.concat(
        [
            selections[model].surface.assign(market=market, model=model)
            for model in MODELS
        ],
        ignore_index=True,
    )
    timelines = {
        model: _selected_timeline(
            market,
            model,
            frame,
            states[model],
            selections[model],
            full[model],
            aligned[model],
            config,
        )
        for model in MODELS
    }
    timeline = pd.concat([timelines[model] for model in MODELS], ignore_index=True)
    if (
        pd.to_datetime(timeline["execution_date"], errors="raise").max().date()
        > spec.cutoff
    ):
        raise BalancedPerformanceError(f"{market}: post-2023 timeline row")
    return MarketEvidence(
        summary=metrics,
        choices=choices,
        surface=surface,
        boundaries=_boundary_rows(market, selections, spec, config),
        timeline=timeline,
        traces=_change_traces(market, timelines),
        trades=aligned,
    )


def _accounting_smoke_checks(
    market: str,
    states: dict[str, pd.DataFrame],
    selections: dict[str, SelectionResult],
    full: dict[str, pd.DataFrame],
    aligned: dict[str, pd.DataFrame],
    config: ResearchConfig,
) -> dict[str, int]:
    offset = config.backtest_protocol.return_offset
    t2_cells = 0
    cost_cells = 0
    for model in MODELS:
        signal = selections[model].signal.reset_index(drop=True)
        expected_position = signal.shift(offset)
        observed_position = full[model]["position"]
        valid = expected_position.notna()
        if not np.array_equal(
            expected_position.loc[valid].to_numpy(),
            observed_position.loc[valid].to_numpy(),
        ):
            raise BalancedPerformanceError(f"{market}/{model}: smoke t+2 mismatch")
        dates = pd.DatetimeIndex(pd.to_datetime(full[model]["date"], errors="raise"))
        active = _active_lambda(dates, selections[model].choices)
        selected_state = 1.0 - selections[model].signal.reindex(dates)
        reconstructed = pd.Series(np.nan, index=dates, dtype=float)
        for candidate in states[model].columns:
            mask = active == candidate
            reconstructed.loc[mask] = states[model].reindex(dates).loc[mask, candidate]
        if not np.array_equal(
            selected_state.to_numpy(), reconstructed.to_numpy(), equal_nan=True
        ):
            raise BalancedPerformanceError(f"{market}/{model}: signal/state mismatch")
        path = aligned[model]
        expected_cost = path["one_way_turnover"] * 0.001
        expected_gross = (
            path["position"] * path["equity_simple"]
            + (1.0 - path["position"]) * path["cash_return"]
        )
        if not np.allclose(path["transaction_cost"], expected_cost, rtol=0, atol=1e-15):
            raise BalancedPerformanceError(f"{market}/{model}: smoke 10-bps mismatch")
        if not np.allclose(path["gross_return"], expected_gross, rtol=0, atol=1e-15):
            raise BalancedPerformanceError(
                f"{market}/{model}: smoke gross return mismatch"
            )
        if not np.allclose(
            path["strategy_return"],
            path["gross_return"] - path["transaction_cost"],
            rtol=0,
            atol=1e-15,
        ):
            raise BalancedPerformanceError(
                f"{market}/{model}: smoke net return mismatch"
            )
        t2_cells += int(valid.sum())
        cost_cells += len(path)
    return {"t_plus_2_cells_checked": t2_cells, "cost_cells_checked": cost_cells}


def run_us_smoke(
    config: ResearchConfig,
    spec: BalancedPerformanceSpec,
    sources: PerformanceSources | None = None,
) -> dict[str, Any]:
    """Open no aggregate metric; check source, selection, timing, cost, and parity."""
    root = config.path.parent
    sources = sources or verify_performance_sources(root, config, spec)
    frame, states = _load_market("us", sources, spec)
    if np.array_equal(
        states["balanced_log4"].to_numpy(),
        states["lagged_log4"].to_numpy(),
        equal_nan=True,
    ) or np.array_equal(
        states["balanced_log4"].to_numpy(),
        states["fixed"].to_numpy(),
        equal_nan=True,
    ):
        raise BalancedPerformanceError("US balanced candidate paths are vacuous")
    selections = _select_paths(frame, states, config)
    _assert_beta_zero_selection(sources.fixed, "us", selections["fixed"])
    full = {model: _full_path(frame, selections[model], config) for model in MODELS}
    aligned = {
        model: _align_parent_sample(
            sources.fixed, "us", full[model], beta_zero=model == "fixed"
        )
        for model in MODELS
    }
    parity = _oracle_choices_and_trades("us", selections, aligned, sources, config)
    accounting = _accounting_smoke_checks(
        "us", states, selections, full, aligned, config
    )
    return {
        "schema_version": 1,
        "status": "passed",
        "market": "us",
        "models": list(MODELS),
        "candidate_rows": len(states["balanced_log4"]),
        "surface_rows": {model: len(selections[model].surface) for model in MODELS},
        "choice_rows": {model: len(selections[model].choices) for model in MODELS},
        "oracle_choice_rows_checked": parity["choice_rows"],
        "oracle_trade_rows_checked": parity["trade_rows"],
        **accounting,
        "signal_to_return_offset": 2,
        "one_way_cost_bps": 10,
        "fixed_and_lagged_oracle_parity": True,
        "metrics_opened": False,
        "post_2023_accessed": False,
        "provider_accessed": False,
    }


def _contrast(summary: pd.DataFrame, baseline: str) -> dict[str, Any]:
    balanced = summary.loc[summary["model"] == "balanced_log4"].sort_values("market")
    if len(balanced) != len(MARKETS) or set(balanced["market"]) != set(MARKETS):
        raise BalancedPerformanceError("balanced decision market coverage changed")
    column = f"delta_vs_{baseline}_sharpe"
    deltas = pd.to_numeric(balanced[column], errors="coerce")
    if not np.isfinite(deltas.to_numpy()).all():
        raise BalancedPerformanceError("Sharpe deltas must be finite in all markets")
    mean = float(deltas.mean())
    positive = int((deltas > 0).sum())
    return {
        "equal_market_mean_delta_sharpe": mean,
        "positive_market_count": positive,
        "market_delta_sharpe": {
            row.market: float(getattr(row, column))
            for row in balanced.itertuples(index=False)
        },
        "passed": bool(mean > 0 and positive >= 2),
    }


def _decision(summary: pd.DataFrame) -> dict[str, Any]:
    expected = {(market, model) for market in MARKETS for model in MODELS}
    actual = list(summary[["market", "model"]].itertuples(index=False, name=None))
    if len(actual) != len(expected) or set(actual) != expected:
        raise BalancedPerformanceError("market/model coverage is incomplete")
    primary = _contrast(summary, "lagged_log4")
    fixed = _contrast(summary, "fixed")
    supported = primary["passed"] and fixed["passed"]
    return {
        "schema_version": 1,
        "experiment_id": "balanced-lagged-performance-001",
        "claim_class": "EXPLORATORY",
        "primary_contrast": "balanced_log4_minus_lagged_log4",
        "required_fixed_contrast": "balanced_log4_minus_fixed",
        "contrasts": {
            "balanced_log4_minus_lagged_log4": primary,
            "balanced_log4_minus_fixed": fixed,
        },
        "result": "supported" if supported else "not_supported",
        "performance_claim_allowed": False,
        "paper_replication_claim_allowed": False,
        "holdout_claim_allowed": False,
        "development_sample_repeatedly_inspected": True,
    }


def _write_market(target: Path, evidence: MarketEvidence) -> None:
    (target / "trades").mkdir(parents=True, exist_ok=True)
    evidence.summary.to_csv(target / "summary.csv", index=False)
    evidence.choices.to_csv(target / "choices.csv", index=False)
    evidence.surface.to_csv(target / "selection-surface.csv", index=False)
    evidence.boundaries.to_csv(target / "boundaries.csv", index=False)
    evidence.timeline.to_csv(target / "selected-timeline.csv", index=False)
    evidence.traces.to_csv(target / "change-traces.csv", index=False)
    for model in MODELS:
        evidence.trades[model].to_csv(target / "trades" / f"{model}.csv", index=False)


def _market_run(
    market: str,
    sources: PerformanceSources,
    target: Path,
    config: ResearchConfig,
    spec: BalancedPerformanceSpec,
) -> MarketEvidence:
    with threadpool_limits(limits=1):
        evidence = _market_evidence(market, sources, config, spec)
        _write_market(target, evidence)
        return evidence


def _expected_artifact_files(spec: BalancedPerformanceSpec) -> set[str]:
    files = {
        "boundaries.csv",
        "change-traces.csv",
        "choices.csv",
        "config.lock.toml",
        "decision.json",
        "implementation-lock.json",
        "inventory.json",
        "run.json",
        "selected-timeline.csv",
        "selection-surface.csv",
        "smoke.json",
        "source-lock.json",
        "study.lock.toml",
        "summary.csv",
    }
    for market in spec.markets:
        files.update(
            {
                f"{market}/boundaries.csv",
                f"{market}/change-traces.csv",
                f"{market}/choices.csv",
                f"{market}/selected-timeline.csv",
                f"{market}/selection-surface.csv",
                f"{market}/summary.csv",
                *(f"{market}/trades/{model}.csv" for model in MODELS),
            }
        )
    return files


def _verify_run_identity(
    path: Path,
    config: ResearchConfig,
    spec: BalancedPerformanceSpec,
) -> tuple[dict[str, Any], PerformanceSources]:
    root = config.path.parent
    if any(item.is_symlink() for item in (path, *path.rglob("*"))):
        raise BalancedPerformanceError("run artifacts may not contain symlinks")
    actual = {str(item.relative_to(path)) for item in path.rglob("*") if item.is_file()}
    if actual != _expected_artifact_files(spec):
        raise BalancedPerformanceError("run artifact file coverage changed")
    verify_inventory(path)
    if (
        sha256_file(path / "study.lock.toml") != spec.sha256
        or sha256_file(path / "config.lock.toml") != config.sha256
    ):
        raise BalancedPerformanceError("run locks differ from frozen study")
    stored_implementation = read_json(path / "implementation-lock.json")
    current_implementation = implementation_lock(root, spec)
    if (
        stored_implementation.get("schema_version") != 1
        or stored_implementation.get("implementation_sha256")
        != current_implementation["implementation_sha256"]
        or stored_implementation.get("files") != current_implementation["files"]
    ):
        raise BalancedPerformanceError("implementation content lock changed")
    implementation = str(stored_implementation["implementation_sha256"])
    run_id = (
        f"balanced-pnl-{spec.sha256[:12]}-"
        f"{spec.balanced_inventory_sha256[:12]}-{implementation[:12]}"
    )
    expected_path = root / config.artifact_root / spec.artifact_subdir / run_id
    metadata = read_json(path / "run.json")
    controls = (
        path == expected_path.resolve(),
        path.name == run_id,
        metadata.get("schema_version") == 1,
        metadata.get("study_kind") == "balanced_lagged_performance",
        metadata.get("experiment_id") == spec.experiment_id,
        metadata.get("run_id") == run_id,
        metadata.get("status") in {"verifying", "complete"},
        metadata.get("claim_class") == "EXPLORATORY",
        metadata.get("metrics_opened") is True,
        metadata.get("spec_sha256") == spec.sha256,
        metadata.get("config_sha256") == config.sha256,
        metadata.get("implementation_sha256") == implementation,
        metadata.get("git_head_informational")
        == stored_implementation.get("git_head_informational"),
        metadata.get("git_describe_informational")
        == stored_implementation.get("git_describe_informational"),
        metadata.get("post_2023_accessed") is False,
        metadata.get("provider_accessed") is False,
        metadata.get("performance_claim_allowed") is False,
        metadata.get("paper_replication_claim_allowed") is False,
        metadata.get("holdout_claim_allowed") is False,
    )
    if not all(controls):
        raise BalancedPerformanceError("run identity or controls changed")
    sources = verify_performance_sources(root, config, spec)
    if read_json(path / "source-lock.json") != sources.source_lock:
        raise BalancedPerformanceError("stored source lock changed")
    if read_json(path / "smoke.json") != run_us_smoke(config, spec, sources):
        raise BalancedPerformanceError("stored US smoke changed")
    return metadata, sources


def verify_balanced_performance_run(
    run_dir: str | Path,
    config: ResearchConfig,
    spec: BalancedPerformanceSpec | None = None,
) -> dict[str, Any]:
    """Independently replay sources, selection, accounting, metrics, and decision."""
    raw = Path(run_dir)
    if raw.is_symlink():
        raise BalancedPerformanceError("run directory may not be a symlink")
    path = raw.resolve()
    spec = spec or load_balanced_performance_spec(
        config.path.parent / "research/balanced-lagged-performance-001.toml",
        config,
    )
    metadata, sources = _verify_run_identity(path, config, spec)
    tables: dict[str, list[pd.DataFrame]] = {
        "summary": [],
        "choices": [],
        "selection-surface": [],
        "boundaries": [],
        "selected-timeline": [],
        "change-traces": [],
    }
    maximum_error = 0.0
    for market in spec.markets:
        with threadpool_limits(limits=1):
            evidence = _market_evidence(market, sources, config, spec)
        expected_frames = {
            "summary": evidence.summary,
            "choices": evidence.choices,
            "selection-surface": evidence.surface,
            "boundaries": evidence.boundaries,
            "selected-timeline": evidence.timeline,
            "change-traces": evidence.traces,
        }
        for name, expected in expected_frames.items():
            tolerance = 1e-12 if name == "summary" else 1e-15
            maximum_error = max(
                maximum_error,
                _assert_frame_close(
                    _read_csv(path / market / f"{name}.csv"),
                    expected,
                    f"{market}/{name}",
                    tolerance=tolerance,
                ),
            )
            tables[name].append(expected)
        for model in MODELS:
            observed_trade = read_trade_path(
                path / market / "trades" / f"{model}.csv",
                config.backtest_protocol.primary_delay,
                config.backtest_protocol.one_way_cost_bps,
            )
            _assert_frame_close(
                observed_trade,
                evidence.trades[model],
                f"{market}/{model} source replay",
                tolerance=1e-15,
            )
    combined = {
        name: pd.concat(frames, ignore_index=True) for name, frames in tables.items()
    }
    for name, expected in combined.items():
        tolerance = 1e-12 if name == "summary" else 1e-15
        maximum_error = max(
            maximum_error,
            _assert_frame_close(
                _read_csv(path / f"{name}.csv"),
                expected,
                f"root/{name}",
                tolerance=tolerance,
            ),
        )
    decision = _decision(combined["summary"])
    if read_json(path / "decision.json") != decision:
        raise BalancedPerformanceError("stored decision changed")
    if metadata.get("decision") != decision["result"]:
        raise BalancedPerformanceError("run metadata decision changed")
    return {
        "schema_version": 1,
        "status": "passed",
        "lifecycle": metadata["status"],
        "run_id": metadata["run_id"],
        "metric_rows": len(combined["summary"]),
        "surface_rows": len(combined["selection-surface"]),
        "timeline_rows": len(combined["selected-timeline"]),
        "maximum_absolute_replay_error": maximum_error,
        "decision": decision["result"],
        "fixed_and_lagged_oracle_parity": True,
    }


def _finalize_verified_run(
    run_dir: Path,
    metadata: dict[str, Any],
    config: ResearchConfig,
    spec: BalancedPerformanceSpec,
) -> None:
    metadata_path = run_dir / "run.json"
    metadata.update(
        {
            "status": "verifying",
            "verification_started_at_utc": datetime.now(UTC).isoformat(),
        }
    )
    write_json(metadata_path, metadata)
    write_inventory(run_dir)
    first = verify_balanced_performance_run(run_dir, config, spec)
    if first.get("lifecycle") != "verifying":
        raise BalancedPerformanceError("pre-completion lifecycle changed")
    metadata.update(
        {"status": "complete", "finished_at_utc": datetime.now(UTC).isoformat()}
    )
    write_json(metadata_path, metadata)
    try:
        final = verify_balanced_performance_run(run_dir, config, spec)
        if final.get("lifecycle") != "complete":
            raise BalancedPerformanceError("final lifecycle changed")
    except Exception as exc:
        metadata.pop("finished_at_utc", None)
        metadata.update(
            {
                "status": "invalid_verification",
                "verification_error": f"final verification failed ({type(exc).__name__})",  # noqa: E501
            }
        )
        write_json(metadata_path, metadata)
        raise


def run_balanced_performance_study(
    config: ResearchConfig, spec: BalancedPerformanceSpec
) -> Path:
    """Run the US accounting smoke, then three market selections in parallel."""
    root = config.path.parent
    sources = verify_performance_sources(root, config, spec)
    smoke = run_us_smoke(config, spec, sources)
    if smoke.get("metrics_opened") is not False:
        raise BalancedPerformanceError("US smoke opened aggregate metrics")
    implementation = implementation_lock(root, spec)
    run_id = (
        f"balanced-pnl-{spec.sha256[:12]}-"
        f"{spec.balanced_inventory_sha256[:12]}-"
        f"{implementation['implementation_sha256'][:12]}"
    )
    run_dir = root / config.artifact_root / spec.artifact_subdir / run_id
    metadata_path = run_dir / "run.json"
    if metadata_path.is_file():
        metadata = read_json(metadata_path)
        if metadata.get("status") == "complete":
            verify_balanced_performance_run(run_dir, config, spec)
            return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "study.lock.toml").write_bytes(spec.path.read_bytes())
    (run_dir / "config.lock.toml").write_bytes(config.path.read_bytes())
    write_json(run_dir / "source-lock.json", sources.source_lock)
    write_json(run_dir / "implementation-lock.json", implementation)
    write_json(run_dir / "smoke.json", smoke)
    metadata = {
        "schema_version": 1,
        "study_kind": "balanced_lagged_performance",
        "experiment_id": spec.experiment_id,
        "run_id": run_id,
        "status": "running",
        "claim_class": "EXPLORATORY",
        "metrics_opened": False,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "spec_sha256": spec.sha256,
        "config_sha256": config.sha256,
        "implementation_sha256": implementation["implementation_sha256"],
        "git_head_informational": implementation["git_head_informational"],
        "git_describe_informational": implementation["git_describe_informational"],
        "post_2023_accessed": False,
        "provider_accessed": False,
        "performance_claim_allowed": False,
        "paper_replication_claim_allowed": False,
        "holdout_claim_allowed": False,
    }
    write_json(metadata_path, metadata)
    results: dict[str, MarketEvidence] = {}
    with ProcessPoolExecutor(
        max_workers=3, mp_context=get_context("forkserver")
    ) as executor:
        futures = {
            executor.submit(
                _market_run,
                market,
                sources,
                run_dir / market,
                config,
                spec,
            ): market
            for market in spec.markets
        }
        for future in as_completed(futures):
            market = futures[future]
            results[market] = future.result()
            print(f"{market}: balanced performance complete", flush=True)
    summary = pd.concat(
        [results[market].summary for market in spec.markets], ignore_index=True
    )
    choices = pd.concat(
        [results[market].choices for market in spec.markets], ignore_index=True
    )
    surface = pd.concat(
        [results[market].surface for market in spec.markets], ignore_index=True
    )
    boundaries = pd.concat(
        [results[market].boundaries for market in spec.markets], ignore_index=True
    )
    timeline = pd.concat(
        [results[market].timeline for market in spec.markets], ignore_index=True
    )
    traces = pd.concat(
        [results[market].traces for market in spec.markets], ignore_index=True
    )
    summary.to_csv(run_dir / "summary.csv", index=False)
    choices.to_csv(run_dir / "choices.csv", index=False)
    surface.to_csv(run_dir / "selection-surface.csv", index=False)
    boundaries.to_csv(run_dir / "boundaries.csv", index=False)
    timeline.to_csv(run_dir / "selected-timeline.csv", index=False)
    traces.to_csv(run_dir / "change-traces.csv", index=False)
    decision = _decision(summary)
    write_json(run_dir / "decision.json", decision)
    metadata.update(
        {
            "metrics_opened": True,
            "decision": decision["result"],
            "metric_rows": len(summary),
            "surface_rows": len(surface),
            "timeline_rows": len(timeline),
        }
    )
    _finalize_verified_run(run_dir, metadata, config, spec)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="balanced-lagged-performance")
    parser.add_argument("--config", default="research.toml")
    parser.add_argument(
        "--spec", default="research/balanced-lagged-performance-001.toml"
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
    spec = load_balanced_performance_spec(spec_path, config)
    if args.verify:
        print(
            json.dumps(
                verify_balanced_performance_run(args.verify, config, spec),
                sort_keys=True,
            )
        )
    elif args.smoke:
        print(json.dumps(run_us_smoke(config, spec), sort_keys=True))
    else:
        print(run_balanced_performance_study(config, spec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
