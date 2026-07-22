"""End-to-end runner for the frozen simple-JM challenger suite."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import subprocess
import tomllib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime
from multiprocessing import get_context
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np
import pandas as pd
from jumpmodels.jump import dp, jump_penalty_to_mx
from threadpoolctl import threadpool_limits

from adaptive_jump.artifacts import (
    TRADE_COLUMNS,
    ArtifactError,
    read_json,
    read_trade_path,
    sha256_file,
    verify_inventory,
    write_inventory,
    write_json,
)
from adaptive_jump.backtest import apply_signal, performance_metrics
from adaptive_jump.config import ResearchConfig, load_config
from adaptive_jump.runtime import study_runtime
from adaptive_jump.runtime.events import EventObserver, emit_event
from adaptive_jump.simple_jm_controls import (
    ControlPath,
    build_confirmed_control_path,
    build_control_path,
    build_static_lambda50_path,
)
from adaptive_jump.simple_jm_fitting import (
    canonical_complete_mask,
    custom_variant_states,
    dd_only_states,
    fixed_jm_trace_receipt,
    run_us_prefix_smoke,
)
from adaptive_jump.simple_jm_l1 import l1_loss_matrix, solve_l1_path
from adaptive_jump.simple_jm_return import (
    dp_return_aware,
    feature_loss_matrix,
    return_aware_loss_matrix,
)
from adaptive_jump.walkforward import (
    SelectionResult,
    boundary_diagnostic,
    select_monthly_candidate,
)

EXPERIMENT_ID = "simple-jm-suite-001"
LOSS_SCALE_EXPERIMENT_ID = "dd-loss-scale-001"
MARKETS = ("us", "de", "jp")
CONTROLS = ("buy_and_hold", "hmm", "fixed_jm")
CHALLENGERS = (
    "static_lambda50",
    "dd_only",
    "confirmed_2d",
    "return_aware",
    "robust_l1",
)
ALL_MODELS = (*CONTROLS, *CHALLENGERS)
FITTED_VARIANTS = ("dd_only", "return_aware", "robust_l1")
SCALED_DD_VARIANT = "dd_scaled_3x"
LOSS_SCALE_MODELS = (*CONTROLS, "dd_only", SCALED_DD_VARIANT)
DD_OBSERVATION_LOSS_SCALE = 3.0
FEATURE_COLUMNS = (
    "date",
    "equity_simple",
    "cash_return",
    "excess_return",
    "dd_10",
    "sortino_20",
    "sortino_60",
)
REFIT_COLUMNS = (
    "fit_date",
    "training_start",
    "training_end",
    "observations",
    "scaler_mean",
    "scaler_scale",
    "lambda",
    "objective",
)

METRIC_REQUIRED = (
    "cash_return",
    "position",
    "one_way_turnover",
    "strategy_return",
)
DEVELOPMENT_CUTOFF = pd.Timestamp("2023-12-31")
PAPER_TURNOVER_SCALE = 0.5


class SimpleJMSuiteError(ArtifactError):
    """Raised when the frozen suite cannot be run or verified exactly."""


@dataclass(frozen=True)
class SuiteSpec:
    path: Path
    sha256: str
    document: dict[str, Any]
    canonical_root: Path
    lambda50_root: Path


@dataclass(frozen=True)
class LossScaleSpec:
    path: Path
    sha256: str
    document: dict[str, Any]
    canonical_root: Path
    dd_parent_root: Path


@dataclass(frozen=True)
class LossScaleMarketSource:
    market: str
    features: pd.DataFrame
    controls: dict[str, pd.DataFrame]


@dataclass(frozen=True)
class MarketSource:
    market: str
    features: pd.DataFrame
    controls: dict[str, pd.DataFrame]
    canonical_signal: pd.Series
    canonical_choices: pd.DataFrame

    canonical_refits: pd.DataFrame
    lambda50_refits: pd.DataFrame


@dataclass(frozen=True)
class VariantOutput:
    market: str
    variant: str
    states: pd.DataFrame
    refits: pd.DataFrame
    selection: SelectionResult
    selected_state: pd.Series
    signal: pd.Series
    full_trades: pd.DataFrame
    boundary: dict[str, Any]


def load_simple_jm_spec(path: str | Path, config: ResearchConfig) -> SuiteSpec:
    """Load the immutable study contract and prove it was registered frozen."""
    repo_root = config.path.parent.resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    resolved = candidate.resolve()
    payload = resolved.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    try:
        document = tomllib.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise SimpleJMSuiteError(f"invalid suite TOML: {exc}") from exc
    sources = document.get("sources")
    variants = document.get("variants")
    if not isinstance(sources, dict) or not isinstance(variants, dict):
        raise SimpleJMSuiteError("suite contract does not match the frozen identity")
    canonical_name = sources.get("canonical_run_root")
    lambda50_name = sources.get("lambda50_run_root")
    required = (
        document.get("schema_version") == 1,
        document.get("experiment_id") == EXPERIMENT_ID,
        document.get("status") == "FROZEN_BEFORE_RESULTS",
        document.get("claim_class") == "EXPLORATORY",
        sources.get("markets") == list(MARKETS),
        sources.get("cutoff") == "2023-12-31",
        sources.get("post_2023_access") is False,
        isinstance(canonical_name, str) and bool(canonical_name),
        isinstance(lambda50_name, str) and bool(lambda50_name),
        tuple(variants) == CHALLENGERS,
    )
    if not all(required):
        raise SimpleJMSuiteError("suite contract does not match the frozen identity")
    _require_frozen_registration(repo_root, EXPERIMENT_ID, digest)
    canonical_root = (repo_root / canonical_name).resolve()
    lambda50_root = (repo_root / lambda50_name).resolve()
    for root in (canonical_root, lambda50_root):
        if not root.is_dir() or root.is_symlink() or repo_root not in root.parents:
            raise SimpleJMSuiteError(f"unsafe or missing source root: {root}")
    return SuiteSpec(resolved, digest, document, canonical_root, lambda50_root)


def load_dd_loss_scale_spec(path: str | Path, config: ResearchConfig) -> LossScaleSpec:
    """Load the registered frozen DD loss-scale control."""
    repo_root = config.path.parent.resolve()
    candidate = Path(path)
    resolved = (
        repo_root / candidate if not candidate.is_absolute() else candidate
    ).resolve()
    payload = resolved.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    try:
        document = tomllib.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise SimpleJMSuiteError(f"invalid loss-scale TOML: {exc}") from exc
    sources = document.get("sources")
    model = document.get("model")
    execution = document.get("execution")
    valid_sections = all(
        isinstance(section, dict) for section in (sources, model, execution)
    )
    if not valid_sections:
        raise SimpleJMSuiteError("loss-scale contract does not match frozen identity")
    canonical_name = sources.get("canonical_run_root")
    parent_name = sources.get("dd_parent_run_root")
    required = (
        document.get("schema_version") == 1,
        document.get("experiment_id") == LOSS_SCALE_EXPERIMENT_ID,
        document.get("parent_id") == EXPERIMENT_ID,
        document.get("status") == "FROZEN_BEFORE_RESULTS",
        document.get("claim_class") == "EXPLORATORY",
        sources.get("markets") == list(MARKETS),
        sources.get("cutoff") == "2023-12-31",
        sources.get("post_2023_access") is False,
        isinstance(canonical_name, str) and bool(canonical_name),
        isinstance(parent_name, str) and bool(parent_name),
        model.get("label") == SCALED_DD_VARIANT,
        model.get("features") == ["dd_10"],
        model.get("observation_loss_scale") == DD_OBSERVATION_LOSS_SCALE,
        execution.get("primary_delay_trading_days") == 1,
        execution.get("signal_to_return_offset") == 2,
        execution.get("one_way_cost_bps") == 10,
    )
    if not all(required):
        raise SimpleJMSuiteError("loss-scale contract does not match frozen identity")
    _require_frozen_registration(repo_root, LOSS_SCALE_EXPERIMENT_ID, digest)
    roots = (
        (repo_root / canonical_name).resolve(),
        (repo_root / parent_name).resolve(),
    )
    if any(
        not root.is_dir() or root.is_symlink() or repo_root not in root.parents
        for root in roots
    ):
        raise SimpleJMSuiteError("unsafe or missing loss-scale source root")
    return LossScaleSpec(resolved, digest, document, *roots)


def _require_frozen_registration(
    repo_root: Path, experiment_id: str, digest: str
) -> None:
    registry = repo_root / "research" / "experiment_registry.jsonl"
    try:
        rows = registry.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise SimpleJMSuiteError(
            f"cannot read experiment registry: {registry}"
        ) from exc
    frozen = False
    for number, raw in enumerate(rows, start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SimpleJMSuiteError(
                f"invalid experiment registry row {number}"
            ) from exc
        if not isinstance(row, dict):
            raise SimpleJMSuiteError(f"invalid experiment registry row {number}")
        registered_id, status = row.get("experiment_id"), row.get("status")
        if (
            not isinstance(registered_id, str)
            or not registered_id
            or not isinstance(status, str)
        ):
            raise SimpleJMSuiteError(f"invalid experiment registry row {number}")
        frozen |= (
            registered_id == experiment_id
            and status == "FROZEN"
            and row.get("frozen_spec_hash") == digest
        )
    if not frozen:
        raise SimpleJMSuiteError("no matching pre-result FROZEN registry row")


def run_simple_jm_study(
    config: ResearchConfig,
    spec: SuiteSpec,
    observer: EventObserver | None = None,
) -> Path:
    """Run A, then DD-only, then the remaining frozen variants end to end."""
    repo_root = config.path.parent.resolve()
    verify_inventory(spec.canonical_root)
    lambda_inventory = _verify_custom_inventory(spec.lambda50_root)
    canonical_inventory = read_json(spec.canonical_root / "inventory.json")["files"]
    if sha256_file(spec.canonical_root / "config.lock.toml") != config.sha256:
        raise SimpleJMSuiteError(
            "active config does not match the sealed source config"
        )
    _validate_protocol(config, spec)
    sources, explicit = _load_sources(
        spec, config, canonical_inventory, lambda_inventory
    )

    code_hashes = _implementation_hashes(repo_root)
    code_digest = _mapping_digest(code_hashes)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"simple-jm-suite-{spec.sha256[:12]}-{code_digest[:12]}-{timestamp}"
    run_dir = repo_root / config.artifact_root / EXPERIMENT_ID / run_id
    started = monotonic()
    run_dir.mkdir(parents=True)
    (run_dir / "study.lock.toml").write_bytes(spec.path.read_bytes())
    (run_dir / "config.lock.toml").write_bytes(
        (spec.canonical_root / "config.lock.toml").read_bytes()
    )
    source_lock = {
        "schema_version": 2,
        "explicitly_read_scientific_inputs": explicit,
        "explicit_file_count": len(explicit),
        "inventory_integrity_hashed": {
            "canonical": canonical_inventory,
            "lambda50": lambda_inventory,
        },
        "inventory_file_count": len(canonical_inventory) + len(lambda_inventory),
    }
    write_json(run_dir / "source-lock.json", source_lock)
    write_json(
        run_dir / "gamma-zero-route.json",
        _gamma_zero_route(spec, canonical_inventory),
    )
    write_json(
        run_dir / "implementation-lock.json",
        {"schema_version": 1, "files": code_hashes, "bundle_sha256": code_digest},
    )
    _start_run(
        run_dir / "run.json",
        study_kind=EXPERIMENT_ID,
        run_id=run_id,
        spec_sha256=spec.sha256,
        implementation_sha256=code_digest,
    )

    _emit_stage(observer, "stage_started", "fixed_jm", completed=0, total=15)
    outputs: dict[tuple[str, str], VariantOutput | ControlPath] = {}
    for completed, market in enumerate(MARKETS, start=1):
        expected = lambda_inventory[f"{market}/jm-missing-states.csv"]
        control = build_static_lambda50_path(
            sources[market].features.loc[:, ["date", "equity_simple", "cash_return"]],
            spec.lambda50_root,
            market,
            expected_sha256=expected,
        )
        outputs[(market, "static_lambda50")] = control
        _emit_variant_completed(
            observer, market, "static_lambda50", completed=completed, total=15
        )
    stage_a = _stage_summary(sources, outputs, ("static_lambda50",), config)
    stage_a.to_csv(run_dir / "stage-a-static-summary.csv", index=False)

    smoke = []
    for variant in ("dd_only", "return_aware", "robust_l1"):
        evidence = run_us_prefix_smoke(
            sources["us"].features,
            config.model_protocol,
            config.jm_protocol,
            variant=variant,
        )
        smoke.append(asdict(evidence))
    pd.DataFrame.from_records(smoke).to_csv(run_dir / "us-smoke.csv", index=False)

    dd_outputs = _parallel_fit(
        spec,
        config,
        ("dd_only",),
        workers=3,
        observer=observer,
        progress_offset=3,
        progress_total=15,
    )
    outputs.update(dd_outputs)
    stage_b = _stage_summary(sources, outputs, ("dd_only",), config)
    stage_b.to_csv(run_dir / "stage-b-dd-only-summary.csv", index=False)

    for completed, market in enumerate(MARKETS, start=7):
        source = sources[market]
        control = build_confirmed_control_path(
            source.features.loc[:, ["date", "equity_simple", "cash_return"]],
            source.canonical_signal,
        )
        outputs[(market, "confirmed_2d")] = control
        _emit_variant_completed(
            observer, market, "confirmed_2d", completed=completed, total=15
        )

    custom = _parallel_fit(
        spec,
        config,
        ("return_aware", "robust_l1"),
        workers=6,
        observer=observer,
        progress_offset=9,
        progress_total=15,
    )
    outputs.update(custom)
    _emit_stage(observer, "stage_completed", "fixed_jm", completed=15, total=15)
    _emit_stage(observer, "stage_started", "selection", completed=0, total=9)
    _emit_variant_events(observer, dd_outputs)
    _emit_variant_events(observer, custom)
    _emit_stage(observer, "stage_completed", "selection", completed=9, total=9)

    math_receipt = _verify_math_contracts()
    aligned, summary = _finalize_paths(sources, outputs, config)
    summary.to_csv(run_dir / "summary.csv", index=False, float_format="%.17g")
    degeneracy = _build_fit_degeneracy(outputs)
    degeneracy.to_csv(run_dir / "fit-degeneracy.csv", index=False)
    decisions = _decision(summary)
    write_json(run_dir / "decision.json", decisions)
    traces = _build_traces(sources, outputs, aligned, config)
    _validate_traces(traces)
    traces.to_csv(run_dir / "traces.csv", index=False, float_format="%.17g")
    _write_market_artifacts(run_dir, sources, outputs, aligned)
    write_json(
        run_dir / "verification.json",
        {
            "schema_version": 1,
            "math_contracts": math_receipt,
            "us_smoke": smoke,
            "gamma_zero_route": "sealed canonical fixed_jm artifacts",
            "cutoff": DEVELOPMENT_CUTOFF.date().isoformat(),
            "one_state_fits": "reported without post-result exclusion",
            "paper_turnover_scale": PAPER_TURNOVER_SCALE,
            "t_plus_2_offset": 2,
            "one_way_cost_bps": 10,
        },
    )
    write_inventory(run_dir)
    _finish_run(run_dir / "run.json", decisions["conclusion"], started)
    verify_simple_jm_run(run_dir)
    return run_dir


def run_dd_loss_scale_study(
    config: ResearchConfig,
    spec: LossScaleSpec,
    observer: EventObserver | None = None,
) -> Path:
    """Run US smoke, then the frozen three-market scale-three DD control."""
    repo_root = config.path.parent.resolve()
    _validate_loss_scale_protocol(config, spec)
    sources = _load_loss_scale_sources(spec, config)
    code_hashes = _implementation_hashes(repo_root)
    code_digest = _mapping_digest(code_hashes)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"dd-loss-scale-{spec.sha256[:12]}-{code_digest[:12]}-{timestamp}"
    run_dir = repo_root / config.artifact_root / LOSS_SCALE_EXPERIMENT_ID / run_id
    started = monotonic()
    run_dir.mkdir(parents=True)
    (run_dir / "study.lock.toml").write_bytes(spec.path.read_bytes())
    (run_dir / "config.lock.toml").write_bytes(
        (spec.canonical_root / "config.lock.toml").read_bytes()
    )
    write_json(
        run_dir / "implementation-lock.json",
        {"schema_version": 1, "files": code_hashes, "bundle_sha256": code_digest},
    )
    _start_run(
        run_dir / "run.json",
        study_kind=LOSS_SCALE_EXPERIMENT_ID,
        run_id=run_id,
        spec_sha256=spec.sha256,
        implementation_sha256=code_digest,
    )

    _emit_stage(observer, "stage_started", "fixed_jm", completed=0, total=3)
    smoke = asdict(
        run_us_prefix_smoke(
            sources["us"].features,
            config.model_protocol,
            config.jm_protocol,
            variant="dd_only",
            observation_loss_scale=DD_OBSERVATION_LOSS_SCALE,
        )
    )
    smoke["variant"] = SCALED_DD_VARIANT
    pd.DataFrame.from_records([smoke]).to_csv(run_dir / "us-smoke.csv", index=False)
    outputs = _parallel_fit(
        spec,
        config,
        (SCALED_DD_VARIANT,),
        workers=3,
        observer=observer,
        progress_total=3,
    )
    _emit_stage(observer, "stage_completed", "fixed_jm", completed=3, total=3)
    _emit_stage(observer, "stage_started", "selection", completed=0, total=3)
    _emit_variant_events(observer, outputs, (SCALED_DD_VARIANT,))
    _emit_stage(observer, "stage_completed", "selection", completed=3, total=3)

    aligned, summary = _finalize_paths(
        sources,
        outputs,
        config,
        (SCALED_DD_VARIANT,),
    )
    summary.to_csv(run_dir / "summary.csv", index=False, float_format="%.17g")
    _loss_scale_contrasts(summary).to_csv(
        run_dir / "dd-scale-contrast.csv",
        index=False,
        float_format="%.17g",
    )
    _build_fit_degeneracy(outputs, (SCALED_DD_VARIANT,)).to_csv(
        run_dir / "fit-degeneracy.csv", index=False
    )
    decision = _decision(summary, (SCALED_DD_VARIANT,))
    write_json(run_dir / "decision.json", decision)
    traces = _build_traces(
        sources,
        outputs,
        aligned,
        config,
        (SCALED_DD_VARIANT,),
    )
    _validate_traces(traces)
    traces.to_csv(run_dir / "traces.csv", index=False, float_format="%.17g")
    _write_market_artifacts(
        run_dir,
        sources,
        outputs,
        aligned,
        LOSS_SCALE_MODELS,
    )
    write_json(
        run_dir / "verification.json",
        {
            "schema_version": 1,
            "math_contracts": _verify_loss_scale_math(),
            "us_smoke": [smoke],
            "observation_loss_scale": DD_OBSERVATION_LOSS_SCALE,
            "cutoff": DEVELOPMENT_CUTOFF.date().isoformat(),
            "paper_turnover_scale": PAPER_TURNOVER_SCALE,
            "t_plus_2_offset": 2,
            "one_way_cost_bps": 10,
        },
    )
    write_inventory(run_dir)
    _finish_run(run_dir / "run.json", decision["conclusion"], started)
    verify_dd_loss_scale_run(run_dir)
    return run_dir


def verify_simple_jm_run(run_dir: Path) -> dict[str, Any]:
    """Independently replay metrics, decisions, source hashes, and timing."""
    run_dir = run_dir.resolve()
    verify_inventory(run_dir)
    metadata = read_json(run_dir / "run.json")
    if (
        metadata.get("status") != "complete"
        or metadata.get("study_kind") != EXPERIMENT_ID
    ):
        raise SimpleJMSuiteError("run metadata is not a completed simple-JM suite")
    spec_hash = sha256_file(run_dir / "study.lock.toml")
    if metadata.get("spec_sha256") != spec_hash:
        raise SimpleJMSuiteError("run spec hash mismatch")
    implementation = read_json(run_dir / "implementation-lock.json")
    implementation_files = implementation.get("files")
    if not isinstance(implementation_files, dict):
        raise SimpleJMSuiteError("implementation lock is invalid")
    repo_root = run_dir.parents[2]
    implementation_digest = _mapping_digest(implementation_files)
    if (
        implementation.get("bundle_sha256") != implementation_digest
        or metadata.get("implementation_sha256") != implementation_digest
    ):
        raise SimpleJMSuiteError("implementation digest mismatch")
    implementation_source_commit = _implementation_source_commit(
        repo_root, implementation_files
    )
    source_lock = read_json(run_dir / "source-lock.json")
    for path_text, evidence in source_lock["explicitly_read_scientific_inputs"].items():
        path = Path(path_text)
        if not path.is_file() or sha256_file(path) != evidence["sha256"]:
            raise SimpleJMSuiteError(f"source changed after run: {path}")
    gamma_route = read_json(run_dir / "gamma-zero-route.json")
    if gamma_route.get("route") != "sealed canonical fixed_jm":
        raise SimpleJMSuiteError("gamma-zero route is not the sealed fixed JM")
    explicit_paths = source_lock["explicitly_read_scientific_inputs"]
    for market, files in gamma_route["markets"].items():
        for evidence in files.values():
            if (
                evidence["path"] not in explicit_paths
                or explicit_paths[evidence["path"]]["sha256"] != evidence["sha256"]
            ):
                raise SimpleJMSuiteError(f"{market}: incomplete gamma-zero source lock")

    config = load_config(run_dir / "config.lock.toml")
    stored = pd.read_csv(run_dir / "summary.csv")
    rows = []
    max_difference = 0.0
    for market in MARKETS:
        paths = {
            model: read_trade_path(run_dir / market / model / "trades.csv", 1, 10)
            for model in ALL_MODELS
        }
        dates = paths[ALL_MODELS[0]]["date"]
        sealed_fixed = read_trade_path(
            Path(gamma_route["markets"][market]["positions_costs_returns"]["path"]),
            1,
            10,
        )
        routed_fixed = (
            sealed_fixed.set_index("date").reindex(dates).reset_index()
        ).loc[:, TRADE_COLUMNS]
        if not _trade_route_equal(routed_fixed, paths["fixed_jm"]):
            raise SimpleJMSuiteError(f"{market}: gamma-zero trade route changed")
        for model, path in paths.items():
            if (
                not path["date"].equals(dates)
                or path["date"].max() > DEVELOPMENT_CUTOFF
            ):
                raise SimpleJMSuiteError(f"{market}/{model}: invalid common dates")
        recalculated = _metric_rows(market, paths, config)
        rows.extend(recalculated)
    expected = pd.DataFrame.from_records(rows)
    for column in (
        "sharpe",
        "maximum_drawdown",
        "turnover",
        "cash_fraction",
        "switch_count",
        "gap_vs_stronger_control",
    ):
        left = pd.to_numeric(stored[column], errors="coerce")
        right = pd.to_numeric(expected[column], errors="coerce")
        difference = np.abs(left - right)
        finite = difference[np.isfinite(difference)]
        if len(finite):
            max_difference = max(max_difference, float(finite.max()))
        if not np.allclose(left, right, rtol=0, atol=1e-12, equal_nan=True):
            raise SimpleJMSuiteError(f"stored metric mismatch: {column}")
    if read_json(run_dir / "decision.json") != _decision(expected):
        raise SimpleJMSuiteError("stored decision does not match recomputed metrics")
    trace = pd.read_csv(run_dir / "traces.csv")
    _validate_traces(trace)
    _verify_trace_trade_rows(run_dir, trace)
    degeneracy = _verify_fit_degeneracy(run_dir)
    return {
        "schema_version": 1,
        "run_id": metadata["run_id"],
        "status": metadata["status"],
        "implementation_source_commit": implementation_source_commit,
        "metric_rows": len(expected),
        "trace_rows": len(trace),
        "degeneracy_rows": len(degeneracy),
        "maximum_metric_absolute_difference": max_difference,
        "conclusion": metadata["conclusion"],
    }


def verify_dd_loss_scale_run(run_dir: Path) -> dict[str, Any]:
    """Replay source routes, accounting, metrics, traces, and decisions."""
    run_dir = run_dir.resolve()
    verify_inventory(run_dir)
    metadata = read_json(run_dir / "run.json")
    if (
        metadata.get("status") != "complete"
        or metadata.get("study_kind") != LOSS_SCALE_EXPERIMENT_ID
        or metadata.get("spec_sha256") != sha256_file(run_dir / "study.lock.toml")
    ):
        raise SimpleJMSuiteError("run metadata is not a completed loss-scale study")
    implementation = read_json(run_dir / "implementation-lock.json")
    files = implementation.get("files")
    digest = _mapping_digest(files) if isinstance(files, dict) else ""
    if (
        not isinstance(files, dict)
        or implementation.get("bundle_sha256") != digest
        or metadata.get("implementation_sha256") != digest
    ):
        raise SimpleJMSuiteError("loss-scale implementation lock is invalid")
    repo_root = run_dir.parents[2]
    source_commit = _implementation_source_commit(repo_root, files)

    config = replace(
        load_config(run_dir / "config.lock.toml"),
        path=repo_root / "research.toml",
    )
    spec = load_dd_loss_scale_spec(run_dir / "study.lock.toml", config)
    _validate_loss_scale_protocol(config, spec)
    sources = _load_loss_scale_sources(spec, config)
    rows = []
    replayed_outputs: dict[tuple[str, str], VariantOutput | ControlPath] = {}
    aligned_paths: dict[str, dict[str, pd.DataFrame]] = {}
    for market in MARKETS:
        paths = {
            model: read_trade_path(run_dir / market / model / "trades.csv", 1, 10)
            for model in LOSS_SCALE_MODELS
        }
        dates = paths[LOSS_SCALE_MODELS[0]]["date"]
        replayed_outputs[(market, SCALED_DD_VARIANT)] = _replay_scaled_selector(
            run_dir,
            market,
            sources[market],
            dates,
            paths[SCALED_DD_VARIANT],
            config,
        )
        for model in (*CONTROLS, "dd_only"):
            sealed = (
                sources[market]
                .controls[model]
                .set_index("date")
                .reindex(dates)
                .reset_index()
                .loc[:, TRADE_COLUMNS]
            )
            if not _trade_route_equal(sealed, paths[model]):
                raise SimpleJMSuiteError(f"{market}/{model}: source route changed")
        if any(
            not path["date"].equals(dates) or path["date"].max() > DEVELOPMENT_CUTOFF
            for path in paths.values()
        ):
            raise SimpleJMSuiteError(f"{market}: invalid common dates")
        aligned_paths[market] = paths
        rows.extend(
            _metric_rows(
                market,
                paths,
                config,
                challengers=(SCALED_DD_VARIANT,),
            )
        )
    expected = pd.DataFrame.from_records(rows)
    stored = pd.read_csv(run_dir / "summary.csv")
    expected_contrast = _loss_scale_contrasts(expected)
    stored_contrast = pd.read_csv(run_dir / "dd-scale-contrast.csv")
    try:
        pd.testing.assert_frame_equal(
            stored,
            expected,
            check_dtype=False,
            check_exact=False,
            rtol=0,
            atol=1e-12,
        )
        pd.testing.assert_frame_equal(
            stored_contrast,
            expected_contrast,
            check_dtype=False,
            check_exact=False,
            rtol=0,
            atol=1e-12,
        )
    except AssertionError as exc:
        raise SimpleJMSuiteError("loss-scale stored metrics changed") from exc
    decision = _decision(expected, (SCALED_DD_VARIANT,))
    if (
        read_json(run_dir / "decision.json") != decision
        or metadata.get("conclusion") != decision["conclusion"]
    ):
        raise SimpleJMSuiteError("loss-scale decision changed")
    verification = read_json(run_dir / "verification.json")
    if (
        verification.get("math_contracts") != _verify_loss_scale_math()
        or verification.get("observation_loss_scale") != DD_OBSERVATION_LOSS_SCALE
        or not verification["us_smoke"][0]["prefix_invariant"]
    ):
        raise SimpleJMSuiteError("loss-scale verification receipt is invalid")
    traces = pd.read_csv(run_dir / "traces.csv")
    _validate_traces(traces)
    _verify_trace_trade_rows(run_dir, traces)
    expected_traces = _build_traces(
        sources,
        replayed_outputs,
        aligned_paths,
        config,
        (SCALED_DD_VARIANT,),
    )
    for frame in (traces, expected_traces):
        for column in ("signal_date", "trade_date", "fit_date"):
            frame[column] = pd.to_datetime(frame[column], errors="raise")
    try:
        pd.testing.assert_frame_equal(
            traces,
            expected_traces,
            check_dtype=False,
            check_exact=False,
            rtol=0,
            atol=1e-12,
        )
    except AssertionError as exc:
        raise SimpleJMSuiteError("loss-scale trace replay changed") from exc
    if set(traces["variant"]) != {SCALED_DD_VARIANT}:
        raise SimpleJMSuiteError("loss-scale trace variant is invalid")
    degeneracy = _verify_fit_degeneracy(run_dir, (SCALED_DD_VARIANT,))
    differences = [
        np.abs(
            pd.to_numeric(stored[column], errors="raise")
            - pd.to_numeric(expected[column], errors="raise")
        ).max()
        for column in (
            "sharpe",
            "maximum_drawdown",
            "turnover",
            "cash_fraction",
            "switch_count",
        )
    ]
    return {
        "schema_version": 1,
        "run_id": metadata["run_id"],
        "status": metadata["status"],
        "implementation_source_commit": source_commit,
        "metric_rows": len(expected),
        "trace_rows": len(traces),
        "degeneracy_rows": len(degeneracy),
        "maximum_metric_absolute_difference": float(max(differences)),
        "conclusion": metadata["conclusion"],
    }


def _replay_scaled_selector(
    run_dir: Path,
    market: str,
    source: LossScaleMarketSource,
    dates: pd.Series,
    stored_trades: pd.DataFrame,
    config: ResearchConfig,
) -> VariantOutput:
    """Replay candidate states through monthly selection and t+2 accounting."""
    target = run_dir / market / SCALED_DD_VARIANT
    states = pd.read_csv(target / "candidate-states.csv")
    if tuple(states.columns[:1]) != ("date",):
        raise SimpleJMSuiteError(f"{market}: invalid scaled candidate states")
    states["date"] = pd.to_datetime(states["date"], errors="raise")
    states = states.set_index("date")
    try:
        states.columns = [float(column) for column in states.columns]
    except (TypeError, ValueError) as exc:
        raise SimpleJMSuiteError(f"{market}: invalid scaled candidate grid") from exc
    if tuple(states.columns) != config.jm_protocol.lambda_grid:
        raise SimpleJMSuiteError(f"{market}: scaled candidate grid changed")
    returns = source.features.loc[:, ["date", "equity_simple", "cash_return"]]
    selection = select_monthly_candidate(
        returns,
        states,
        config.selection_protocol,
        delay_trading_days=1,
        one_way_cost_bps=10,
        periods_per_year=252,
        volatility_ddof=1,
    )
    full_trades = apply_signal(
        returns,
        selection.signal.reset_index(drop=True),
        delay_trading_days=1,
        one_way_cost_bps=10,
    )
    replayed_trades = (full_trades.set_index("date").reindex(dates).reset_index()).loc[
        :, TRADE_COLUMNS
    ]
    if not _trade_route_equal(replayed_trades, stored_trades):
        raise SimpleJMSuiteError(f"{market}: scaled t+2 trade replay changed")

    stored_choices = pd.read_csv(target / "choices.csv")
    stored_signal = pd.read_csv(target / "selected-signal.csv")
    expected_signal = selection.signal.rename("selected_signal").reset_index()
    for frame, column in (
        (stored_choices, "decision_date"),
        (selection.choices, "decision_date"),
        (stored_signal, "date"),
        (expected_signal, "date"),
    ):
        frame[column] = pd.to_datetime(frame[column], errors="raise")
    try:
        pd.testing.assert_frame_equal(
            stored_choices,
            selection.choices,
            check_dtype=False,
            check_exact=True,
        )
        pd.testing.assert_frame_equal(
            stored_signal,
            expected_signal,
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError as exc:
        raise SimpleJMSuiteError(f"{market}: scaled selector replay changed") from exc
    selected_state = (1.0 - selection.signal).rename("selected_state")
    return VariantOutput(
        market,
        SCALED_DD_VARIANT,
        states,
        pd.read_csv(target / "refits.csv"),
        selection,
        selected_state,
        selection.signal,
        full_trades,
        read_json(target / "boundary.json"),
    )


def _validate_protocol(config: ResearchConfig, spec: SuiteSpec) -> None:
    common = spec.document["common_protocol"]
    checks = (
        config.replication_cutoff == date(2023, 12, 31),
        config.model_protocol.fit_window == 3000,
        config.model_protocol.n_states == 2,
        config.jm_protocol.lambda_grid == tuple(common["lambda_grid"]),
        config.jm_protocol.refit_months == (1, 7),
        common["primary_delay_trading_days"] == 1,
        common["signal_to_return_offset"] == 2,
        common["one_way_cost_bps"] == 10,
    )
    if not all(checks):
        raise SimpleJMSuiteError("canonical config and frozen suite protocol disagree")


def _validate_loss_scale_protocol(config: ResearchConfig, spec: LossScaleSpec) -> None:
    model = spec.document["model"]
    checks = (
        config.replication_cutoff == date(2023, 12, 31),
        config.model_protocol.n_states == model["states"] == 2,
        config.model_protocol.fit_window == model["fit_window_observations"] == 3000,
        config.jm_protocol.lambda_grid == tuple(model["lambda_grid"]),
        config.jm_protocol.refit_months == tuple(model["refit_months"]) == (1, 7),
        config.jm_protocol.n_init == model["n_init"] == 10,
        config.jm_protocol.random_state == model["random_state"] == 0,
        config.jm_protocol.max_iter == model["max_iter"] == 1000,
        config.jm_protocol.tol == model["normalized_tol"] == 1e-8,
        math.isclose(
            model["optimizer_tol"], model["normalized_tol"] * DD_OBSERVATION_LOSS_SCALE
        ),
        config.selection_protocol.validation_years == 8,
    )
    if not all(checks):
        raise SimpleJMSuiteError(
            "canonical config and frozen loss-scale protocol disagree"
        )


def _load_loss_scale_sources(
    spec: LossScaleSpec, config: ResearchConfig
) -> dict[str, LossScaleMarketSource]:
    source_rows = spec.document["sources"]
    roots = {
        "canonical": (
            spec.canonical_root,
            source_rows["canonical_inventory_sha256"],
        ),
        "dd_parent": (
            spec.dd_parent_root,
            source_rows["dd_parent_inventory_sha256"],
        ),
    }
    for label, (root, digest) in roots.items():
        if sha256_file(root / "inventory.json") != digest:
            raise SimpleJMSuiteError(f"{label} source inventory hash changed")
        verify_inventory(root)
        files = read_json(root / "inventory.json").get("files")
        if not isinstance(files, dict):
            raise SimpleJMSuiteError(f"{label} source inventory is invalid")
    canonical_metadata = read_json(spec.canonical_root / "run.json")
    parent_metadata = read_json(spec.dd_parent_root / "run.json")
    if (
        canonical_metadata.get("run_id") != source_rows["canonical_run_id"]
        or canonical_metadata.get("status") != "complete"
        or canonical_metadata.get("config_sha256") != config.sha256
        or parent_metadata.get("run_id") != source_rows["dd_parent_run_id"]
        or parent_metadata.get("study_kind") != EXPERIMENT_ID
        or parent_metadata.get("status") != "complete"
        or sha256_file(spec.canonical_root / "config.lock.toml") != config.sha256
        or sha256_file(spec.dd_parent_root / "config.lock.toml") != config.sha256
    ):
        raise SimpleJMSuiteError("loss-scale source identity changed")

    output = {}
    for market in MARKETS:
        feature_path = spec.canonical_root / market / "features.csv"
        features = pd.read_csv(feature_path, usecols=list(FEATURE_COLUMNS))
        features["date"] = pd.to_datetime(features["date"], errors="raise")
        controls = {
            model: read_trade_path(
                spec.canonical_root / market / "trades" / f"{model}-delay-1.csv",
                1,
                10,
            )
            for model in CONTROLS
        }
        controls["dd_only"] = _replay_parent_dd_control(
            features, spec.dd_parent_root, market
        )
        if features["date"].max() > DEVELOPMENT_CUTOFF or any(
            path["date"].max() > DEVELOPMENT_CUTOFF for path in controls.values()
        ):
            raise SimpleJMSuiteError(f"{market}: source crosses the cutoff")
        output[market] = LossScaleMarketSource(market, features, controls)
    return output


def _replay_parent_dd_control(
    features: pd.DataFrame, parent_root: Path, market: str
) -> pd.DataFrame:
    """Replay the sealed DD signal on canonical returns after CSV validation."""
    target = parent_root / market / "dd_only"
    signal_path = target / "selected-signal.csv"
    try:
        frame = pd.read_csv(signal_path, usecols=["date", "selected_signal"])
        dates = pd.DatetimeIndex(
            pd.to_datetime(frame["date"], errors="raise"), name="date"
        )
        raw = frame["selected_signal"]
        signal = pd.to_numeric(raw, errors="coerce")
        if (raw.notna() & signal.isna()).any():
            raise ValueError("selected signal contains a non-numeric value")
        state = pd.Series(
            1.0 - signal.to_numpy(dtype=float),
            index=dates,
            name="state",
        )
        replayed = build_control_path(
            features.loc[:, ["date", "equity_simple", "cash_return"]],
            state,
        ).trades
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise SimpleJMSuiteError(
            f"{market}: invalid parent DD selected signal"
        ) from exc
    sealed = read_trade_path(target / "trades.csv", 1, 10)
    routed = (
        replayed.set_index("date")
        .reindex(sealed["date"])
        .reset_index()
        .loc[:, TRADE_COLUMNS]
    )
    if not _trade_route_equal(sealed, routed):
        raise SimpleJMSuiteError(f"{market}: parent DD trade route changed")
    return replayed


def _loss_scale_contrasts(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = (
        "sharpe",
        "maximum_drawdown",
        "turnover",
        "cash_fraction",
        "switch_count",
    )
    for market in MARKETS:
        indexed = summary.loc[summary["market"] == market].set_index("model")
        if not {"dd_only", SCALED_DD_VARIANT}.issubset(indexed.index):
            raise SimpleJMSuiteError(f"{market}: loss-scale contrast is incomplete")
        ordinary, scaled = indexed.loc["dd_only"], indexed.loc[SCALED_DD_VARIANT]
        row = {
            "market": market,
            "primary_gap": float(scaled["gap_vs_stronger_control"]),
            "market_pass": bool(scaled["market_pass"]),
        }
        row.update(
            {
                f"delta_{metric}": float(scaled[metric]) - float(ordinary[metric])
                for metric in metrics
            }
        )
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _verify_loss_scale_math() -> dict[str, bool]:
    X = np.asarray([[-2.0], [-1.8], [1.7], [2.0]])
    centers = np.asarray([[-2.0], [2.0]])
    base = feature_loss_matrix(X, centers)
    root = math.sqrt(DD_OBSERVATION_LOSS_SCALE)
    scaled = feature_loss_matrix(X * root, centers * root)
    penalty = 0.9
    scaled_path, scaled_dp_value = dp(scaled, jump_penalty_to_mx(penalty, 2))
    base_path, base_dp_value = dp(
        base,
        jump_penalty_to_mx(penalty / DD_OBSERVATION_LOSS_SCALE, 2),
    )
    scaled_brute = _brute_force_value(scaled, penalty)
    base_brute = _brute_force_value(base, penalty / DD_OBSERVATION_LOSS_SCALE)
    every_path = True
    for path in itertools.product(range(2), repeat=len(base)):
        switches = sum(
            left != right for left, right in zip(path[:-1], path[1:], strict=True)
        )
        scaled_value = sum(scaled[row, state] for row, state in enumerate(path))
        scaled_value += penalty * switches
        base_value = sum(base[row, state] for row, state in enumerate(path))
        base_value += penalty / DD_OBSERVATION_LOSS_SCALE * switches
        every_path &= math.isclose(
            scaled_value,
            DD_OBSERVATION_LOSS_SCALE * base_value,
            rel_tol=0,
            abs_tol=1e-12,
        )
    equivalent = (
        np.allclose(
            scaled,
            DD_OBSERVATION_LOSS_SCALE * base,
            rtol=0,
            atol=1e-14,
        )
        and every_path
        and np.array_equal(scaled_path, base_path)
        and math.isclose(float(scaled_dp_value), scaled_brute, rel_tol=0, abs_tol=1e-12)
        and math.isclose(float(base_dp_value), base_brute, rel_tol=0, abs_tol=1e-12)
        and math.isclose(
            scaled_brute / DD_OBSERVATION_LOSS_SCALE,
            base_brute,
            rel_tol=0,
            abs_tol=1e-12,
        )
    )
    if not equivalent:
        raise SimpleJMSuiteError("scaled DD mathematical identity failed")
    return {
        "loss_scale_three_formula": True,
        "every_toy_path_objective_identity": True,
        "lambda_one_third_path_equivalence": True,
        "brute_force_equivalence": True,
    }


def _verify_custom_inventory(run_root: Path) -> dict[str, str]:
    inventory = read_json(run_root / "inventory.json")
    expected = inventory.get("files")
    if not isinstance(expected, dict):
        raise SimpleJMSuiteError("lambda50 inventory schema is invalid")
    actual = {
        str(path.relative_to(run_root))
        for path in run_root.rglob("*")
        if path.is_file() and path.name != "inventory.json"
    }
    if actual != set(expected):
        raise SimpleJMSuiteError("lambda50 inventory file set changed")
    for relative, digest in expected.items():
        if sha256_file(run_root / relative) != digest:
            raise SimpleJMSuiteError(f"lambda50 source changed: {relative}")
    return expected


def _load_sources(
    spec: SuiteSpec,
    config: ResearchConfig,
    canonical_inventory: dict[str, str],
    lambda_inventory: dict[str, str],
) -> tuple[dict[str, MarketSource], dict[str, dict[str, Any]]]:
    sources = {}
    explicit: dict[str, dict[str, Any]] = {}
    _record_access(
        explicit,
        spec.canonical_root / "config.lock.toml",
        canonical_inventory["config.lock.toml"],
        ["TOML configuration"],
    )
    for market in MARKETS:
        feature_path = spec.canonical_root / market / "features.csv"
        _record_access(
            explicit,
            feature_path,
            canonical_inventory[f"{market}/features.csv"],
            list(FEATURE_COLUMNS),
        )
        features = pd.read_csv(feature_path, usecols=list(FEATURE_COLUMNS))
        features["date"] = pd.to_datetime(features["date"], errors="raise")
        if features["date"].max() > DEVELOPMENT_CUTOFF:
            raise SimpleJMSuiteError(f"{market}: features cross the cutoff")
        controls = {}
        for model in CONTROLS:
            relative = f"{market}/trades/{model}-delay-1.csv"
            path = spec.canonical_root / relative
            _record_access(
                explicit, path, canonical_inventory[relative], list(TRADE_COLUMNS)
            )
            controls[model] = read_trade_path(path, 1, 10)
        reference = controls["fixed_jm"][["date", "equity_simple", "cash_return"]]
        if any(
            not control[["date", "equity_simple", "cash_return"]].equals(reference)
            for control in controls.values()
        ):
            raise SimpleJMSuiteError(f"{market}: sealed controls use different samples")

        signal_relative = f"{market}/fixed_jm-delay-1/selected-signal.csv"
        signal_path = spec.canonical_root / signal_relative
        _record_access(
            explicit,
            signal_path,
            canonical_inventory[signal_relative],
            ["date", "selected_signal"],
        )
        signal_frame = pd.read_csv(signal_path, usecols=["date", "selected_signal"])
        signal_dates = pd.DatetimeIndex(
            pd.to_datetime(signal_frame["date"], errors="raise"), name="date"
        )
        canonical_signal = pd.Series(
            signal_frame["selected_signal"].to_numpy(dtype=float),
            index=signal_dates,
            name="selected_signal",
        )
        choice_relative = f"{market}/fixed_jm-delay-1/choices.csv"
        choice_path = spec.canonical_root / choice_relative
        _record_access(
            explicit,
            choice_path,
            canonical_inventory[choice_relative],
            ["decision_date", "selected"],
        )
        choices = pd.read_csv(choice_path, usecols=["decision_date", "selected"])
        choices["decision_date"] = pd.to_datetime(
            choices["decision_date"], errors="raise"
        )
        state_relative = f"{market}/jm-states.csv"
        state_path = spec.canonical_root / state_relative
        state_columns = [
            "date",
            *[str(value) for value in config.jm_protocol.lambda_grid],
        ]
        _record_access(
            explicit,
            state_path,
            canonical_inventory[state_relative],
            state_columns,
        )
        gamma_zero_states = pd.read_csv(state_path, usecols=state_columns)
        state_dates = pd.to_datetime(gamma_zero_states.pop("date"), errors="raise")
        state_values = gamma_zero_states.apply(pd.to_numeric, errors="coerce")
        if (
            not state_dates.equals(features["date"])
            or not state_values.stack().isin([0.0, 1.0]).all()
        ):
            raise SimpleJMSuiteError(f"{market}: invalid sealed gamma-zero states")
        canonical_refit_relative = f"{market}/jm-refits.csv"
        canonical_refit_path = spec.canonical_root / canonical_refit_relative
        _record_access(
            explicit,
            canonical_refit_path,
            canonical_inventory[canonical_refit_relative],
            list(REFIT_COLUMNS),
        )
        canonical_refits = _read_refit_source(canonical_refit_path)
        lambda_refit_relative = f"{market}/jm-missing-refits.csv"
        lambda_refit_path = spec.lambda50_root / lambda_refit_relative
        _record_access(
            explicit,
            lambda_refit_path,
            lambda_inventory[lambda_refit_relative],
            list(REFIT_COLUMNS),
        )
        lambda50_refits = _read_refit_source(lambda_refit_path)
        if not np.isclose(
            pd.to_numeric(lambda50_refits["lambda"], errors="raise"), 50.0
        ).any():
            raise SimpleJMSuiteError(
                f"{market}: lambda50 source refits do not contain lambda=50"
            )

        lambda_relative = f"{market}/jm-missing-states.csv"
        _record_access(
            explicit,
            spec.lambda50_root / lambda_relative,
            lambda_inventory[lambda_relative],
            ["date", "50.0"],
        )
        sources[market] = MarketSource(
            market,
            features,
            controls,
            canonical_signal,
            choices,
            canonical_refits,
            lambda50_refits,
        )
    return sources, explicit


def _gamma_zero_route(
    spec: SuiteSpec, canonical_inventory: dict[str, str]
) -> dict[str, Any]:
    relative_files = {
        "candidate_states": "{market}/jm-states.csv",
        "monthly_choices": "{market}/fixed_jm-delay-1/choices.csv",
        "selected_signal": "{market}/fixed_jm-delay-1/selected-signal.csv",
        "positions_costs_returns": "{market}/trades/fixed_jm-delay-1.csv",
    }
    markets = {}
    for market in MARKETS:
        markets[market] = {}
        for label, template in relative_files.items():
            relative = template.format(market=market)
            markets[market][label] = {
                "path": str((spec.canonical_root / relative).resolve()),
                "sha256": canonical_inventory[relative],
            }
    return {
        "schema_version": 1,
        "gamma": 0,
        "route": "sealed canonical fixed_jm",
        "markets": markets,
    }


def _record_access(
    evidence: dict[str, dict[str, Any]],
    path: Path,
    expected_sha256: str,
    columns: list[str],
) -> None:
    if path.is_symlink() or sha256_file(path) != expected_sha256:
        raise SimpleJMSuiteError(f"explicit source lock failed: {path}")
    evidence[str(path.resolve())] = {
        "sha256": expected_sha256,
        "columns_physically_read": columns,
    }


def _read_refit_source(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=list(REFIT_COLUMNS))
    if frame.empty:
        raise SimpleJMSuiteError(f"empty refit source: {path}")
    for column in ("fit_date", "training_start", "training_end"):
        frame[column] = pd.to_datetime(frame[column], errors="raise")
    numeric = ("observations", "lambda", "objective")
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if (
        frame["fit_date"].max() > DEVELOPMENT_CUTOFF
        or frame["training_end"].max() > DEVELOPMENT_CUTOFF
        or not (frame["fit_date"] == frame["training_end"]).all()
        or not (frame["observations"] == 3000).all()
        or not np.isfinite(frame.loc[:, numeric].to_numpy(dtype=float)).all()
        or (frame["lambda"] < 0).any()
    ):
        raise SimpleJMSuiteError(f"invalid refit source: {path}")
    return frame


def _implementation_hashes(repo_root: Path) -> dict[str, str]:
    paths = (
        "src/adaptive_jump/simple_jm_controls.py",
        "src/adaptive_jump/simple_jm_l1.py",
        "src/adaptive_jump/simple_jm_return.py",
        "src/adaptive_jump/simple_jm_fitting.py",
        "src/adaptive_jump/models.py",
        "src/adaptive_jump/artifacts.py",
        "src/adaptive_jump/backtest.py",
        "src/adaptive_jump/config.py",
        "src/adaptive_jump/walkforward.py",
        "src/adaptive_jump/simple_jm_suite.py",
        "pyproject.toml",
        "uv.lock",
    )
    return {path: sha256_file(repo_root / path) for path in paths}


def _mapping_digest(mapping: dict[str, str]) -> str:
    payload = json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _implementation_source_commit(
    repo_root: Path, implementation_files: dict[str, Any]
) -> str | None:
    """Resolve the lock from current files or one complete Git snapshot."""
    if not implementation_files:
        raise SimpleJMSuiteError("implementation lock contains no files")
    expected: dict[str, str] = {}
    for relative, digest in implementation_files.items():
        path = Path(relative) if isinstance(relative, str) else Path()
        valid_path = (
            isinstance(relative, str)
            and relative == path.as_posix()
            and not path.is_absolute()
            and ".." not in path.parts
            and ":" not in relative
        )
        valid_digest = (
            isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
        )
        if not valid_path or not valid_digest:
            raise SimpleJMSuiteError(
                "implementation lock contains an invalid file entry"
            )
        expected[relative] = digest

    if all(
        (repo_root / relative).is_file() and sha256_file(repo_root / relative) == digest
        for relative, digest in expected.items()
    ):
        return None

    paths = sorted(expected)
    try:
        history = subprocess.run(
            ["git", "log", "--all", "--full-history", "--format=%H", "--", *paths],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise SimpleJMSuiteError("cannot inspect implementation history") from exc
    if history.returncode != 0:
        raise SimpleJMSuiteError("cannot inspect implementation history")

    try:
        for commit in dict.fromkeys(history.stdout.splitlines()):
            matched = True
            for relative, digest in expected.items():
                blob = subprocess.run(
                    ["git", "show", f"{commit}:{relative}"],
                    cwd=repo_root,
                    check=False,
                    capture_output=True,
                )
                if (
                    blob.returncode != 0
                    or hashlib.sha256(blob.stdout).hexdigest() != digest
                ):
                    matched = False
                    break
            if matched:
                return commit
    except OSError as exc:
        raise SimpleJMSuiteError("cannot inspect implementation history") from exc
    raise SimpleJMSuiteError(
        "no single Git commit contains the complete locked implementation"
    )


def _emit_stage(
    observer: EventObserver | None,
    kind: str,
    stage: str,
    *,
    completed: int,
    total: int,
) -> None:
    emit_event(
        observer,
        kind=kind,
        stage=stage,
        completed=completed,
        total=total,
    )


def _emit_variant_completed(
    observer: EventObserver | None,
    market: str,
    variant: str,
    *,
    completed: int,
    total: int,
) -> None:
    emit_event(
        observer,
        kind="variant_completed",
        stage="fixed_jm",
        market=market,
        model=variant,
        completed=completed,
        total=total,
    )


def _emit_variant_events(
    observer: EventObserver | None,
    outputs: dict[tuple[str, str], VariantOutput],
    variants: tuple[str, ...] = FITTED_VARIANTS,
) -> None:
    if observer is None:
        return
    for variant in variants:
        for market in MARKETS:
            output = outputs.get((market, variant))
            if output is None:
                continue
            study_runtime.emit_selected_signal(
                observer,
                output.selection,
                variant,
                delay=1,
                market=market,
            )
            boundary = pd.DataFrame.from_records(
                [{"model": variant, "delay": 1, **output.boundary}]
            )
            study_runtime.emit_boundary_rows(observer, boundary, market)


def _parallel_fit(
    spec: SuiteSpec | LossScaleSpec,
    config: ResearchConfig,
    variants: tuple[str, ...],
    *,
    workers: int,
    observer: EventObserver | None = None,
    progress_offset: int = 0,
    progress_total: int | None = None,
) -> dict[tuple[str, str], VariantOutput]:
    tasks = [
        (
            str(spec.canonical_root),
            market,
            variant,
            config,
        )
        for variant, market in itertools.product(variants, MARKETS)
    ]
    total = len(tasks) if progress_total is None else progress_total
    if progress_offset < 0 or progress_offset + len(tasks) > total:
        raise SimpleJMSuiteError("parallel fit progress range is invalid")
    output = {}
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=get_context("forkserver")
    ) as executor:
        futures = {executor.submit(_fit_market_task, task): task for task in tasks}
        for future in as_completed(futures):
            _, market, variant, _ = futures[future]
            result = future.result()
            output[(market, variant)] = result
            _emit_variant_completed(
                observer,
                market,
                variant,
                completed=progress_offset + len(output),
                total=total,
            )
    if len(output) != len(tasks):
        raise SimpleJMSuiteError("parallel fit did not return every market/variant")
    return output


def _start_run(
    metadata_path: Path,
    *,
    study_kind: str,
    run_id: str,
    spec_sha256: str,
    implementation_sha256: str,
) -> None:
    write_json(
        metadata_path,
        {
            "schema_version": 1,
            "study_kind": study_kind,
            "run_id": run_id,
            "status": "running",
            "spec_sha256": spec_sha256,
            "implementation_sha256": implementation_sha256,
            "created_at_utc": datetime.now(UTC).isoformat(),
        },
    )


def _finish_run(metadata_path: Path, conclusion: str, started: float) -> None:
    metadata = read_json(metadata_path)
    if metadata.get("status") != "running":
        raise SimpleJMSuiteError("run metadata is not in the running state")
    metadata.update(
        {
            "status": "complete",
            "conclusion": conclusion,
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "runtime_seconds": monotonic() - started,
        }
    )
    write_json(metadata_path, metadata)


def _fit_market_task(
    task: tuple[str, str, str, ResearchConfig],
) -> VariantOutput:
    canonical_root_text, market, variant, config = task
    with threadpool_limits(limits=1):
        path = Path(canonical_root_text) / market / "features.csv"
        frame = pd.read_csv(path, usecols=list(FEATURE_COLUMNS))
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        if variant in ("dd_only", SCALED_DD_VARIANT):
            fitted = dd_only_states(
                frame,
                config.model_protocol,
                config.jm_protocol,
                observation_loss_scale=(
                    DD_OBSERVATION_LOSS_SCALE if variant == SCALED_DD_VARIANT else 1.0
                ),
            )
        elif variant in ("return_aware", "robust_l1"):
            fitted = custom_variant_states(
                frame,
                config.model_protocol,
                config.jm_protocol,
                variant=variant,
            )
        else:
            raise SimpleJMSuiteError(f"unsupported fitted variant: {variant}")
        returns = frame.loc[:, ["date", "equity_simple", "cash_return"]]
        selection = select_monthly_candidate(
            returns,
            fitted.states,
            config.selection_protocol,
            delay_trading_days=1,
            one_way_cost_bps=10,
            periods_per_year=252,
            volatility_ddof=1,
        )
        signal = selection.signal.copy()
        selected_state = (1.0 - signal).rename("selected_state")
        trades = apply_signal(
            returns,
            signal.reset_index(drop=True),
            delay_trading_days=1,
            one_way_cost_bps=10,
        )
        control = read_trade_path(
            Path(canonical_root_text) / market / "trades" / "fixed_jm-delay-1.csv",
            1,
            10,
        )
        diagnostic = boundary_diagnostic(
            selection.choices,
            tuple(float(value) for value in fitted.states.columns),
            oos_start=control["date"].iloc[0].date(),
            fraction_limit=config.selection_protocol.boundary_fraction_limit,
        )
        return VariantOutput(
            market,
            variant,
            fitted.states,
            fitted.refits,
            selection,
            selected_state,
            signal,
            trades,
            {**asdict(diagnostic), "descriptive_only": True},
        )


def _stage_summary(
    sources: dict[str, MarketSource],
    outputs: dict[tuple[str, str], VariantOutput | ControlPath],
    variants: tuple[str, ...],
    config: ResearchConfig,
) -> pd.DataFrame:
    rows = []
    for market in MARKETS:
        paths = dict(sources[market].controls)
        for variant in variants:
            item = outputs[(market, variant)]
            paths[variant] = (
                item.full_trades if isinstance(item, VariantOutput) else item.trades
            )
        rows.extend(_metric_rows(market, _align_paths(paths), config))
    return pd.DataFrame.from_records(rows)


def _finalize_paths(
    sources: dict[str, MarketSource] | dict[str, LossScaleMarketSource],
    outputs: dict[tuple[str, str], VariantOutput | ControlPath],
    config: ResearchConfig,
    variants: tuple[str, ...] = CHALLENGERS,
) -> tuple[dict[str, dict[str, pd.DataFrame]], pd.DataFrame]:
    aligned = {}
    rows = []
    for market in MARKETS:
        paths = dict(sources[market].controls)
        for variant in variants:
            item = outputs[(market, variant)]
            paths[variant] = (
                item.full_trades if isinstance(item, VariantOutput) else item.trades
            )
        aligned[market] = _align_paths(paths)
        rows.extend(_metric_rows(market, aligned[market], config, challengers=variants))
    return aligned, pd.DataFrame.from_records(rows)


def _align_paths(paths: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    valid_dates = []
    indexed = {}
    for model, raw in paths.items():
        frame = raw.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        if (
            frame["date"].duplicated().any()
            or not frame["date"].is_monotonic_increasing
        ):
            raise SimpleJMSuiteError(f"{model}: invalid path dates")
        complete = frame.loc[:, METRIC_REQUIRED].notna().all(axis=1)
        dates = pd.DatetimeIndex(frame.loc[complete, "date"])
        valid_dates.append(dates)
        indexed[model] = frame.set_index("date")
    common = valid_dates[0]
    for dates in valid_dates[1:]:
        common = common.intersection(dates, sort=False)
    common = common.sort_values()
    if common.empty or common.max() > DEVELOPMENT_CUTOFF:
        raise SimpleJMSuiteError("no valid through-2023 common comparison sample")
    output = {
        model: frame.reindex(common).reset_index().loc[:, TRADE_COLUMNS]
        for model, frame in indexed.items()
    }
    reference = output[next(iter(output))][["date", "equity_simple", "cash_return"]]
    if any(
        not frame[["date", "equity_simple", "cash_return"]].equals(reference)
        for frame in output.values()
    ):
        raise SimpleJMSuiteError("aligned paths have different market returns")
    return output


def _trade_route_equal(sealed: pd.DataFrame, routed: pd.DataFrame) -> bool:
    if tuple(sealed.columns) != TRADE_COLUMNS or tuple(routed.columns) != TRADE_COLUMNS:
        return False
    if not sealed["date"].equals(routed["date"]):
        return False
    discrete = ["signal", "position", "one_way_turnover"]
    if not np.array_equal(
        sealed[discrete].to_numpy(dtype=float),
        routed[discrete].to_numpy(dtype=float),
    ):
        return False
    continuous = [column for column in TRADE_COLUMNS[1:] if column not in discrete]
    return bool(
        np.allclose(
            sealed[continuous],
            routed[continuous],
            rtol=0,
            atol=1e-15,
        )
    )


def _metric_rows(
    market: str,
    paths: dict[str, pd.DataFrame],
    config: ResearchConfig,
    *,
    challengers: tuple[str, ...] = CHALLENGERS,
) -> list[dict[str, Any]]:
    calculated = {}
    for model, path in paths.items():
        metrics = performance_metrics(
            path,
            periods_per_year=252,
            volatility_ddof=1,
            expected_shortfall_quantile=(
                config.metrics_protocol.expected_shortfall_quantile
            ),
            turnover_scale=PAPER_TURNOVER_SCALE,
        )
        calculated[model] = {
            "market": market,
            "model": model,
            **metrics,
            "cash_fraction": 1.0 - float(metrics["leverage"]),
            "switch_count": int((path["one_way_turnover"] > 0).sum()),
        }
    if not set(CONTROLS).issubset(calculated):
        raise SimpleJMSuiteError("metric set is missing sealed controls")
    stronger = max(
        float(calculated["buy_and_hold"]["sharpe"]),
        float(calculated["hmm"]["sharpe"]),
    )
    rows = []
    for model, row in calculated.items():
        gap = float(row["sharpe"]) - stronger if model not in CONTROLS[:2] else math.nan
        rows.append(
            {
                **row,
                "stronger_control_sharpe": stronger,
                "gap_vs_stronger_control": gap,
                "market_pass": bool(gap > 0) if model in challengers else False,
            }
        )
    return rows


def _decision(
    summary: pd.DataFrame, variants: tuple[str, ...] = CHALLENGERS
) -> dict[str, Any]:
    decisions = []
    for variant in variants:
        rows = summary.loc[summary["model"] == variant].sort_values("market")
        if len(rows) != len(MARKETS):
            raise SimpleJMSuiteError(f"incomplete decision rows for {variant}")
        passes = {row.market: bool(row.market_pass) for row in rows.itertuples()}
        decisions.append(
            {
                "variant": variant,
                "market_pass": passes,
                "cross_market_support": all(passes.values()),
            }
        )
    supported = [row["variant"] for row in decisions if row["cross_market_support"]]
    return {
        "schema_version": 1,
        "rule": "G_m(v) > 0 in US, DE, and JP for the same frozen variant",
        "variants": decisions,
        "supported_variants": supported,
        "conclusion": "supported" if supported else "not_supported",
        "claim_restriction": "repeatedly inspected exploratory development evidence",
    }


def _active_choice(choices: pd.DataFrame, observation_date: pd.Timestamp) -> float:
    dated = choices.copy()
    dated["decision_date"] = pd.to_datetime(dated["decision_date"], errors="raise")
    eligible = dated.loc[dated["decision_date"] <= observation_date]
    if eligible.empty:
        raise SimpleJMSuiteError("trace has no active monthly choice")
    penalty = float(eligible.iloc[-1]["selected"])
    if not math.isfinite(penalty) or penalty < 0:
        raise SimpleJMSuiteError("active monthly choice is invalid")
    return penalty


def _active_refit(
    refits: pd.DataFrame, observation_date: pd.Timestamp, penalty: float
) -> pd.Series:
    dated = refits.copy()
    dated["fit_date"] = pd.to_datetime(dated["fit_date"], errors="raise")
    selected = pd.to_numeric(dated["lambda"], errors="raise")
    eligible = dated.loc[
        (dated["fit_date"] <= observation_date)
        & np.isclose(selected, penalty, rtol=0, atol=1e-12)
    ]
    if eligible.empty:
        raise SimpleJMSuiteError("observation has no active refit")
    return eligible.sort_values("fit_date").iloc[-1]


def _strict_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value in ("True", "False"):
        return value == "True"
    raise SimpleJMSuiteError("collapse flag is not boolean")


def _fit_degeneracy_row(
    market: str,
    variant: str,
    refits: pd.DataFrame,
    choices: pd.DataFrame,
) -> dict[str, Any]:
    required = {"fit_date", "lambda", "active_state_count", "collapsed_to_one_state"}
    if not required.issubset(refits):
        raise SimpleJMSuiteError(f"{market}/{variant}: missing collapse diagnostics")
    active = pd.to_numeric(refits["active_state_count"], errors="raise")
    collapsed = refits["collapsed_to_one_state"].map(_strict_bool)
    if not active.isin([1, 2]).all() or not (collapsed == (active == 1)).all():
        raise SimpleJMSuiteError(f"{market}/{variant}: invalid collapse diagnostics")
    selected_collapsed = 0
    for choice in choices.itertuples(index=False):
        record = _active_refit(
            refits,
            pd.Timestamp(choice.decision_date),
            float(choice.selected),
        )
        selected_collapsed += int(_strict_bool(record["collapsed_to_one_state"]))
    total_fit_rows = len(refits)
    selected_months = len(choices)
    return {
        "market": market,
        "variant": variant,
        "refit_dates": int(pd.to_datetime(refits["fit_date"]).nunique()),
        "total_fit_rows": total_fit_rows,
        "collapsed_fit_rows": int(collapsed.sum()),
        "collapsed_fit_fraction": float(collapsed.mean()),
        "selected_months": selected_months,
        "selected_collapsed_months": selected_collapsed,
        "selected_collapsed_fraction": (
            float(selected_collapsed / selected_months) if selected_months else math.nan
        ),
        "treatment": "retained; no post-result exclusion",
    }


def _build_fit_degeneracy(
    outputs: dict[tuple[str, str], VariantOutput | ControlPath],
    variants: tuple[str, ...] = FITTED_VARIANTS,
) -> pd.DataFrame:
    rows = []
    for market, variant in itertools.product(MARKETS, variants):
        item = outputs[(market, variant)]
        if not isinstance(item, VariantOutput):
            raise SimpleJMSuiteError(f"{market}/{variant}: fitted output missing")
        rows.append(
            _fit_degeneracy_row(
                market,
                variant,
                item.refits,
                item.selection.choices,
            )
        )
    return pd.DataFrame.from_records(rows)


def _verify_fit_degeneracy(
    run_dir: Path, variants: tuple[str, ...] = FITTED_VARIANTS
) -> pd.DataFrame:
    expected_rows = []
    for market, variant in itertools.product(MARKETS, variants):
        target = run_dir / market / variant
        expected_rows.append(
            _fit_degeneracy_row(
                market,
                variant,
                pd.read_csv(target / "refits.csv"),
                pd.read_csv(target / "choices.csv"),
            )
        )
    expected = (
        pd.DataFrame.from_records(expected_rows)
        .sort_values(["market", "variant"])
        .reset_index(drop=True)
    )
    stored = (
        pd.read_csv(run_dir / "fit-degeneracy.csv")
        .sort_values(["market", "variant"])
        .reset_index(drop=True)
    )
    if tuple(stored.columns) != tuple(expected.columns) or len(stored) != len(expected):
        raise SimpleJMSuiteError("fit-degeneracy artifact has invalid schema")
    floating = {
        "collapsed_fit_fraction",
        "selected_collapsed_fraction",
    }
    for column in expected:
        if column in floating:
            if not np.allclose(
                pd.to_numeric(stored[column], errors="raise"),
                pd.to_numeric(expected[column], errors="raise"),
                rtol=0,
                atol=1e-15,
                equal_nan=True,
            ):
                raise SimpleJMSuiteError(f"fit-degeneracy mismatch: {column}")
        elif not stored[column].equals(expected[column]):
            raise SimpleJMSuiteError(f"fit-degeneracy mismatch: {column}")
    return expected


def _write_market_artifacts(
    run_dir: Path,
    sources: dict[str, MarketSource] | dict[str, LossScaleMarketSource],
    outputs: dict[tuple[str, str], VariantOutput | ControlPath],
    aligned: dict[str, dict[str, pd.DataFrame]],
    models: tuple[str, ...] = ALL_MODELS,
) -> None:
    for market in MARKETS:
        for model in models:
            target = run_dir / market / model
            target.mkdir(parents=True)
            aligned[market][model].to_csv(
                target / "trades.csv", index=False, float_format="%.17g"
            )
            if model in CONTROLS:
                continue
            item = outputs.get((market, model))
            if item is None:
                continue
            if isinstance(item, VariantOutput):
                item.states.reset_index().to_csv(
                    target / "candidate-states.csv",
                    index=False,
                    float_format="%.17g",
                )
                item.refits.to_csv(target / "refits.csv", index=False)
                item.selection.choices.to_csv(target / "choices.csv", index=False)
                item.selection.surface.to_csv(
                    target / "cv-surface.csv", index=False, float_format="%.17g"
                )
                item.selection.candidate_returns.reset_index().to_csv(
                    target / "candidate-returns.csv",
                    index=False,
                    float_format="%.17g",
                )
                item.signal.rename("selected_signal").reset_index().to_csv(
                    target / "selected-signal.csv", index=False
                )
                write_json(target / "boundary.json", item.boundary)
            else:
                pd.DataFrame(
                    {"date": item.state.index, "state": item.state.to_numpy()}
                ).to_csv(target / "candidate-states.csv", index=False)
                item.signal.rename("selected_signal").reset_index().to_csv(
                    target / "selected-signal.csv", index=False
                )
                if model == "confirmed_2d":
                    sources[market].canonical_choices.to_csv(
                        target / "reused-canonical-choices.csv", index=False
                    )


def _build_traces(
    sources: dict[str, MarketSource] | dict[str, LossScaleMarketSource],
    outputs: dict[tuple[str, str], VariantOutput | ControlPath],
    aligned: dict[str, dict[str, pd.DataFrame]],
    config: ResearchConfig,
    variants: tuple[str, ...] = CHALLENGERS,
) -> pd.DataFrame:
    rows = []
    for market, variant in itertools.product(MARKETS, variants):
        item = outputs[(market, variant)]
        state = item.selected_state if isinstance(item, VariantOutput) else item.state
        raw_state = (
            1.0 - sources[market].canonical_signal
            if variant == "confirmed_2d"
            else state
        )
        signal = item.signal
        full_trades = (
            item.full_trades if isinstance(item, VariantOutput) else item.trades
        )
        dates = pd.DatetimeIndex(pd.to_datetime(full_trades["date"]), name="date")
        state = state.reindex(dates)
        raw_state = raw_state.reindex(dates)
        signal = signal.reindex(dates)
        aligned_dates = set(pd.DatetimeIndex(aligned[market][variant]["date"]))
        changes = signal.ne(signal.shift()).fillna(False) & signal.notna()
        candidates = []
        for signal_row in np.flatnonzero(changes.to_numpy()):
            trade_row = signal_row + 2
            if trade_row < len(dates) and dates[trade_row] in aligned_dates:
                candidates.append((signal_row, trade_row))
        if not candidates:
            raise SimpleJMSuiteError(f"no concrete trace for {market}/{variant}")
        chosen = sorted(set((0, len(candidates) // 2, len(candidates) - 1)))
        for trace_number, candidate_index in enumerate(chosen, start=1):
            signal_row, trade_row = candidates[candidate_index]
            trade = full_trades.iloc[trade_row]
            if float(signal.iloc[signal_row]) != float(trade["position"]):
                raise SimpleJMSuiteError("trace signal-position timing mismatch")
            evidence = _trace_evidence(
                sources[market],
                item,
                variant,
                dates[signal_row],
                float(raw_state.iloc[signal_row]),
                config,
            )
            rows.append(
                {
                    "market": market,
                    "variant": variant,
                    "trace_number": trace_number,
                    "signal_date": dates[signal_row],
                    "trade_date": dates[trade_row],
                    "signal_row": signal_row,
                    "trade_row": trade_row,
                    **evidence,
                    "raw_state": float(raw_state.iloc[signal_row]),
                    "state": float(state.iloc[signal_row]),
                    "signal": float(signal.iloc[signal_row]),
                    "position": float(trade["position"]),
                    "one_way_turnover": float(trade["one_way_turnover"]),
                    "transaction_cost": float(trade["transaction_cost"]),
                    "gross_return": float(trade["gross_return"]),
                    "strategy_return": float(trade["strategy_return"]),
                }
            )
    return pd.DataFrame.from_records(rows)


def _trace_evidence(
    source: MarketSource | LossScaleMarketSource,
    item: VariantOutput | ControlPath,
    variant: str,
    signal_date: pd.Timestamp,
    expected_raw_state: float,
    config: ResearchConfig,
) -> dict[str, Any]:
    full_features = ("dd_10", "sortino_20", "sortino_60")
    observation_loss_scale = 1.0
    if variant == "static_lambda50":
        penalty = 50.0
        record = _active_refit(source.lambda50_refits, signal_date, penalty)
        feature_columns = full_features
    elif variant == "confirmed_2d":
        penalty = _active_choice(source.canonical_choices, signal_date)
        record = _active_refit(source.canonical_refits, signal_date, penalty)
        feature_columns = full_features
    elif variant in ("dd_only", SCALED_DD_VARIANT):
        if not isinstance(item, VariantOutput):
            raise SimpleJMSuiteError("DD-only trace is missing its fitted output")
        penalty = _active_choice(item.selection.choices, signal_date)
        record = _active_refit(item.refits, signal_date, penalty)
        feature_columns = ("dd_10",)
        if variant == SCALED_DD_VARIANT:
            observation_loss_scale = DD_OBSERVATION_LOSS_SCALE
    else:
        return _custom_trace_evidence(
            source,
            item,
            variant,
            signal_date,
            expected_raw_state,
            config,
        )

    receipt = fixed_jm_trace_receipt(
        source.features,
        config.model_protocol,
        config.jm_protocol,
        feature_columns=feature_columns,
        penalty=penalty,
        refit_record=record,
        signal_date=signal_date,
        observation_loss_scale=observation_loss_scale,
        expected_state=expected_raw_state,
    )
    return {
        "loss_family": (
            "3 * 0.5 squared standardized DD distance used online"
            if variant == SCALED_DD_VARIANT
            else "0.5 squared standardized feature distance used online"
        ),
        "fit_date": receipt.fit_date,
        "fit_objective": receipt.objective,
        "scaler_mean": _encode_trace_array(receipt.scaler_mean),
        "scaler_scale": _encode_trace_array(receipt.scaler_scale),
        "centers": _encode_trace_array(receipt.centers),
        "active_state_count": receipt.active_state_count,
        "collapsed_to_one_state": receipt.collapsed_to_one_state,
        "loss_state_0": receipt.point_loss[0],
        "loss_state_1": receipt.point_loss[1],
        "terminal_value_state_0": receipt.terminal_value[0],
        "terminal_value_state_1": receipt.terminal_value[1],
        "reachable_state_0": math.isfinite(receipt.terminal_value[0]),
        "reachable_state_1": math.isfinite(receipt.terminal_value[1]),
        "transition_penalty": receipt.penalty,
    }


def _custom_trace_evidence(
    source: MarketSource,
    item: VariantOutput | ControlPath,
    variant: str,
    signal_date: pd.Timestamp,
    expected_raw_state: float,
    config: ResearchConfig,
) -> dict[str, Any]:
    if not isinstance(item, VariantOutput) or variant not in (
        "return_aware",
        "robust_l1",
    ):
        raise SimpleJMSuiteError(f"unsupported trace variant: {variant}")
    penalty = _active_choice(item.selection.choices, signal_date)
    record = _active_refit(item.refits, signal_date, penalty)
    mean = _decode_trace_array(record["scaler_mean"], ndim=1)
    scale = _decode_trace_array(record["scaler_scale"], ndim=1)
    centers = _decode_trace_array(record["centers"], ndim=2, allow_nan=True)
    if (
        mean.shape != scale.shape
        or centers.shape != (2, len(mean))
        or (scale <= 0).any()
    ):
        raise SimpleJMSuiteError("custom trace parameters have invalid shapes")

    mask = canonical_complete_mask(source.features)
    complete = source.features.loc[
        mask, ["date", "dd_10", "sortino_20", "sortino_60"]
    ].copy()
    complete["date"] = pd.to_datetime(complete["date"], errors="raise")
    positions = np.flatnonzero(
        complete["date"].to_numpy() == np.datetime64(signal_date)
    )
    if len(positions) != 1:
        raise SimpleJMSuiteError("custom trace date is not a canonical complete row")
    terminal = int(positions[0])
    first = terminal - config.model_protocol.fit_window + 1
    if first < 0:
        raise SimpleJMSuiteError("custom trace online window is incomplete")
    raw = complete.iloc[first : terminal + 1].loc[
        :, ["dd_10", "sortino_20", "sortino_60"]
    ]
    scaled = (raw.to_numpy(dtype=float) - mean) / scale
    if variant == "robust_l1":
        loss = l1_loss_matrix(scaled, centers)
        family = "L1 standardized feature distance used online"
    else:
        loss = feature_loss_matrix(scaled, centers)
        family = "0.5 squared standardized feature distance used online"
    safe_loss = np.where(np.isnan(loss), np.inf, loss)
    values = np.asarray(
        dp(
            safe_loss,
            jump_penalty_to_mx(penalty, 2),
            return_value_mx=True,
        )
    )
    online_state = int(values[-1].argmin())
    if online_state != int(expected_raw_state):
        raise SimpleJMSuiteError("custom trace DP state differs from emitted state")
    active_state_count = int(record["active_state_count"])
    collapsed = _strict_bool(record["collapsed_to_one_state"])
    reachable_centers = np.isfinite(centers).all(axis=1)
    unavailable_centers = np.isnan(centers).all(axis=1)
    if (
        not (reachable_centers | unavailable_centers).all()
        or int(reachable_centers.sum()) != active_state_count
        or collapsed != (active_state_count == 1)
    ):
        raise SimpleJMSuiteError("custom trace collapse metadata is inconsistent")
    return {
        "loss_family": family,
        "fit_date": pd.Timestamp(record["fit_date"]),
        "fit_objective": float(record["objective"]),
        "scaler_mean": _encode_trace_array(mean),
        "scaler_scale": _encode_trace_array(scale),
        "centers": _encode_trace_array(centers),
        "active_state_count": active_state_count,
        "collapsed_to_one_state": collapsed,
        "loss_state_0": float(safe_loss[-1, 0]),
        "loss_state_1": float(safe_loss[-1, 1]),
        "terminal_value_state_0": float(values[-1, 0]),
        "terminal_value_state_1": float(values[-1, 1]),
        "reachable_state_0": bool(np.isfinite(values[-1, 0])),
        "reachable_state_1": bool(np.isfinite(values[-1, 1])),
        "transition_penalty": penalty,
    }


def _decode_trace_array(
    value: object, *, ndim: int, allow_nan: bool = False
) -> np.ndarray:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise SimpleJMSuiteError("trace parameter is invalid JSON") from exc
    array = np.asarray(value, dtype=float)
    if array.ndim != ndim or not array.size:
        raise SimpleJMSuiteError("trace parameter has invalid dimensions")
    valid = np.isfinite(array) | np.isnan(array) if allow_nan else np.isfinite(array)
    if not valid.all():
        raise SimpleJMSuiteError("trace parameter contains an invalid value")
    return array


def _encode_trace_array(value: object) -> str:
    array = np.asarray(value, dtype=float)

    def clean(item: object) -> object:
        if isinstance(item, list):
            return [clean(child) for child in item]
        number = float(item)
        return number if math.isfinite(number) else None

    return json.dumps(clean(array.tolist()), separators=(",", ":"), allow_nan=False)


def _validate_traces(traces: pd.DataFrame) -> None:
    """Require a complete loss-to-t+2 trade chain for every concrete trace."""
    required = {
        "market",
        "variant",
        "signal_date",
        "trade_date",
        "signal_row",
        "trade_row",
        "loss_family",
        "loss_state_0",
        "loss_state_1",
        "transition_penalty",
        "terminal_value_state_0",
        "terminal_value_state_1",
        "active_state_count",
        "collapsed_to_one_state",
        "raw_state",
        "state",
        "signal",
        "position",
        "one_way_turnover",
        "transaction_cost",
        "gross_return",
        "strategy_return",
    }
    missing = sorted(required.difference(traces.columns))
    if traces.empty or missing:
        raise SimpleJMSuiteError(f"trace evidence columns are missing: {missing}")
    signal_dates = pd.to_datetime(traces["signal_date"], errors="raise")
    trade_dates = pd.to_datetime(traces["trade_date"], errors="raise")
    if (trade_dates <= signal_dates).any() or trade_dates.max() > DEVELOPMENT_CUTOFF:
        raise SimpleJMSuiteError("trace dates violate causal ordering or cutoff")
    numeric_columns = (
        "signal_row",
        "trade_row",
        "loss_state_0",
        "loss_state_1",
        "transition_penalty",
        "terminal_value_state_0",
        "terminal_value_state_1",
        "active_state_count",
        "raw_state",
        "state",
        "signal",
        "position",
        "one_way_turnover",
        "transaction_cost",
        "gross_return",
        "strategy_return",
    )
    numeric = {
        column: pd.to_numeric(traces[column], errors="raise").to_numpy(dtype=float)
        for column in numeric_columns
    }
    penalty = numeric["transition_penalty"]
    if not np.isfinite(penalty).all() or (penalty < 0).any():
        raise SimpleJMSuiteError("trace transition penalty is invalid")
    accounting = np.column_stack(
        [
            numeric["position"],
            numeric["one_way_turnover"],
            numeric["transaction_cost"],
            numeric["gross_return"],
            numeric["strategy_return"],
        ]
    )
    if not np.isfinite(accounting).all():
        raise SimpleJMSuiteError("trace trade accounting is invalid")
    if not (numeric["trade_row"] == numeric["signal_row"] + 2).all():
        raise SimpleJMSuiteError("trace t+2 row identity failed")
    if (
        traces["loss_family"].isna().any()
        or (traces["loss_family"].astype(str).str.len() == 0).any()
    ):
        raise SimpleJMSuiteError("trace loss family is missing")

    collapsed = traces["collapsed_to_one_state"].map(_strict_bool).to_numpy()
    active = numeric["active_state_count"]
    if not np.isin(active, [1.0, 2.0]).all() or not np.array_equal(
        collapsed, active == 1.0
    ):
        raise SimpleJMSuiteError("trace collapse metadata is inconsistent")
    raw_state = numeric["raw_state"]
    state = numeric["state"]
    signal = numeric["signal"]
    position = numeric["position"]
    if not np.isin(raw_state, [0.0, 1.0]).all():
        raise SimpleJMSuiteError("trace raw state is not binary")
    if not np.isin(state, [0.0, 1.0]).all():
        raise SimpleJMSuiteError("trace state is not binary")

    for row in range(len(traces)):
        losses = np.asarray(
            [numeric["loss_state_0"][row], numeric["loss_state_1"][row]]
        )
        terminal = np.asarray(
            [
                numeric["terminal_value_state_0"][row],
                numeric["terminal_value_state_1"][row],
            ]
        )
        if np.isnan(losses).any() or np.isneginf(losses).any():
            raise SimpleJMSuiteError("trace loss contains an invalid value")
        if np.isnan(terminal).any() or np.isneginf(terminal).any():
            raise SimpleJMSuiteError("trace terminal value contains an invalid value")
        reachable = np.isfinite(terminal)
        finite_loss = np.isfinite(losses)
        if not np.array_equal(reachable, finite_loss):
            raise SimpleJMSuiteError("trace reachable state has nonfinite loss")
        if int(reachable.sum()) != int(active[row]):
            raise SimpleJMSuiteError("trace loss availability contradicts collapse")
        decoded = int(np.argmin(terminal))
        if decoded != int(raw_state[row]):
            raise SimpleJMSuiteError("trace raw state differs from online DP")
        variant = str(traces.iloc[row]["variant"])
        if variant != "confirmed_2d" and raw_state[row] != state[row]:
            raise SimpleJMSuiteError("trace state differs from raw state")
        if state[row] != 1.0 - signal[row]:
            raise SimpleJMSuiteError("trace signal differs from state mapping")
        if signal[row] != position[row]:
            raise SimpleJMSuiteError("trace position differs from t+2 signal")
        for state_number in (0, 1):
            column = f"reachable_state_{state_number}"
            if column in traces and _strict_bool(traces.iloc[row][column]) != bool(
                reachable[state_number]
            ):
                raise SimpleJMSuiteError("trace reachable-state flag is inconsistent")
    if "fit_date" in traces:
        fit_dates = pd.to_datetime(traces["fit_date"], errors="raise")
        if (fit_dates > signal_dates).any():
            raise SimpleJMSuiteError("trace fit date follows its signal date")


def _verify_trace_trade_rows(run_dir: Path, traces: pd.DataFrame) -> None:
    """Link every trace's t+2 accounting fields to its verified trade path."""
    cache: dict[tuple[str, str], pd.DataFrame] = {}
    fields = (
        "position",
        "one_way_turnover",
        "transaction_cost",
        "gross_return",
        "strategy_return",
    )
    for trace in traces.itertuples(index=False):
        market = str(trace.market)
        variant = str(trace.variant)
        if market not in MARKETS or not variant:
            raise SimpleJMSuiteError("trace market or variant is invalid")
        key = (market, variant)
        if key not in cache:
            cache[key] = read_trade_path(
                run_dir / market / variant / "trades.csv", 1, 10
            ).set_index("date")
        trade_date = pd.Timestamp(trace.trade_date)
        matches = cache[key].loc[cache[key].index == trade_date]
        if len(matches) != 1:
            raise SimpleJMSuiteError("trace t+2 trade row is missing")
        trade = matches.iloc[0]
        for field in fields:
            if not math.isclose(
                float(getattr(trace, field)),
                float(trade[field]),
                rel_tol=0,
                abs_tol=1e-15,
            ):
                raise SimpleJMSuiteError(f"trace trade field changed: {field}")


def _verify_math_contracts() -> dict[str, bool]:
    X = np.asarray([[-1.0], [-0.5], [1.0]])
    centers = np.asarray([[-0.75], [1.0]])
    penalty = 0.7
    l1_labels, l1_value = solve_l1_path(X, centers, penalty)
    l1_brute = _brute_force_value(l1_loss_matrix(X, centers), penalty)
    if not math.isclose(float(l1_value), l1_brute, rel_tol=0, abs_tol=1e-12):
        raise SimpleJMSuiteError("L1 DP failed brute-force equivalence")
    target = np.asarray([-1.0, 0.25, 1.0])
    target_means = np.asarray([-0.5, 0.75])
    mask = np.asarray([True, True, False])
    gamma_zero = return_aware_loss_matrix(
        X, centers, target, target_means, mask, gamma=0
    )
    fixed = feature_loss_matrix(X, centers)
    if not np.array_equal(gamma_zero, fixed):
        raise SimpleJMSuiteError("return-aware gamma zero differs from fixed loss")
    _, return_value = dp_return_aware(
        X,
        centers,
        target,
        target_means,
        mask,
        gamma=1,
        jump_penalty=penalty,
    )
    combined = return_aware_loss_matrix(X, centers, target, target_means, mask, gamma=1)
    return_brute = _brute_force_value(combined, penalty)
    if not math.isclose(float(return_value), return_brute, rel_tol=0, abs_tol=1e-12):
        raise SimpleJMSuiteError("return-aware DP failed brute-force equivalence")
    if not np.isin(l1_labels, [0, 1]).all():
        raise SimpleJMSuiteError("L1 DP emitted invalid state")
    return {
        "l1_formula": True,
        "l1_brute_force": True,
        "return_gamma_zero_exact": True,
        "return_formula": True,
        "return_brute_force": True,
    }


def _brute_force_value(loss: np.ndarray, penalty: float) -> float:
    return min(
        sum(loss[row, state] for row, state in enumerate(path))
        + penalty
        * sum(left != right for left, right in zip(path, path[1:], strict=False))
        for path in itertools.product(range(loss.shape[1]), repeat=len(loss))
    )
