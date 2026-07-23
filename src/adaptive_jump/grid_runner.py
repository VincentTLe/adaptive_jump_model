"""Run, report, and verify the frozen persistence-grid evaluation."""

# ruff: noqa: E501 - embedded report HTML stays readable as source

from __future__ import annotations

import math
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from html import escape
from importlib.metadata import PackageNotFoundError, version
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from adaptive_jump.artifacts import (
    TRADE_COLUMNS,
    ArtifactError,
    finish_run_metadata,
    read_json,
    sha256_file,
    verify_inventory,
    verify_run,
    write_inventory,
    write_json,
)
from adaptive_jump.backtest import performance_metrics
from adaptive_jump.config import ResearchConfig, load_config
from adaptive_jump.data import research_git_sha
from adaptive_jump.features import effective_oos_start
from adaptive_jump.grid_spec import GridStudySpec, load_grid_spec
from adaptive_jump.inference import BootstrapProgress, bootstrap_sharpe_delta
from adaptive_jump.models import FixedJMResult, HMMResult, fixed_jm_states
from adaptive_jump.runtime import checkpoints as checkpoint_store
from adaptive_jump.runtime import study_runtime
from adaptive_jump.runtime.events import EventObserver
from adaptive_jump.walkforward import (
    BaselineStudy,
    SelectionProgress,
    baseline_paths,
    boundary_diagnostic,
    build_baseline_study,
)

MARKETS = ("us", "de", "jp")
COMPARISON_MODELS = (
    "buy_and_hold",
    "hmm_v7",
    "hmm_new_grid",
    "fixed_jm_v7",
    "fixed_jm_new_grid",
)
CONTROL_SCOPE = (
    "research.toml",
    "src/adaptive_jump/config.py",
    "src/adaptive_jump/features.py",
    "src/adaptive_jump/models.py",
    "src/adaptive_jump/walkforward.py",
    "src/adaptive_jump/backtest.py",
)
GRID_WORKERS = 3


@dataclass(frozen=True)
class ParentMarket:
    """Sealed parent inputs needed by the changed-grid study."""

    frame: pd.DataFrame
    oos_start: date
    hmm: HMMResult


def run_grid_evaluation(
    config: ResearchConfig,
    spec: GridStudySpec,
    observer: EventObserver | None = None,
) -> Path:
    """Run or resume the boundary-first changed-grid evaluation."""
    root = config.path.parent
    parent_dir = root / config.artifact_root / "fixed-baselines" / spec.parent_run_id
    calibration_dir = (
        root
        / config.artifact_root
        / "persistence-calibrated-search"
        / spec.calibration_run_id
    )
    parent_receipt, parent_metadata, calibration_metadata = _verify_parents(
        parent_dir, calibration_dir, config, spec
    )
    git_sha = research_git_sha(root)
    control_git_sha = str(calibration_metadata["git_sha"])
    _verify_control_source(root, control_git_sha, git_sha)
    identity = {
        "spec_sha256": spec.sha256,
        "config_sha256": config.sha256,
        "data_manifest_sha256": spec.data_manifest_sha256,
        "parent_inventory_sha256": spec.parent_inventory_sha256,
        "calibration_inventory_sha256": spec.calibration_inventory_sha256,
        "git_sha": git_sha,
    }
    run_id = "grid-eval-" + "-".join(
        identity[key][:12] for key in ("spec_sha256", "data_manifest_sha256", "git_sha")
    )
    run_dir = root / config.artifact_root / spec.artifact_subdir / run_id
    checkpoint_root = root / config.artifact_root / ".monitor" / "checkpoints" / run_id
    metadata_path = run_dir / "run.json"
    if metadata_path.exists():
        metadata = read_json(metadata_path)
        if any(metadata.get(key) != value for key, value in identity.items()):
            raise ArtifactError("existing grid-evaluation identity does not match")
        _verify_locks(run_dir, config, spec)
        if metadata.get("status") in {"complete", "boundary_failed"}:
            verify_grid_run(run_dir)
            checkpoint_store.clear_checkpoint_tree(checkpoint_root)
            return run_dir
    else:
        _create_run(
            run_dir,
            config,
            spec,
            parent_dir,
            parent_receipt,
            run_id,
            identity,
            control_git_sha,
        )

    evaluated = replace(
        config,
        jm_protocol=replace(config.jm_protocol, lambda_grid=spec.jm_grid),
        hmm_protocol=replace(config.hmm_protocol, smoothing_grid=spec.hmm_grid),
    )
    inputs = {
        market: _load_parent_market(parent_dir, market, config, spec)
        for market in MARKETS
    }
    studies: dict[str, BaselineStudy] = {}
    pending: dict[str, tuple[Path, FixedJMResult | None]] = {}
    for market in MARKETS:
        market_root = checkpoint_root / market
        cached = _load_checkpoint(
            market_root / "baseline-study",
            "grid_study",
            identity,
            BaselineStudy,
        )
        if cached is not None:
            if cached.oos_start != inputs[market].oos_start:
                raise ArtifactError(f"{market}: cached OOS start changed")
            studies[market] = cached
            continue
        pending[market] = (
            market_root / "jm-grid",
            _load_checkpoint(
                market_root / "jm-grid",
                "grid_jm",
                identity,
                FixedJMResult,
            ),
        )

    fitted = _fit_pending_jm(inputs, evaluated, pending, identity)
    for market in MARKETS:
        if market in studies:
            study = studies[market]
        else:
            market_root = checkpoint_root / market

            def selection_loader(
                model: str,
                delay: int,
                root: Path = market_root,
            ) -> SelectionProgress | None:
                return _load_selection(root, identity, model, delay)

            def save_selection(
                model: str,
                delay: int,
                value: SelectionProgress,
                root: Path = market_root,
            ) -> None:
                _save_selection(root, identity, model, delay, value)

            selection_saver = study_runtime.baseline_selection_recorder(
                save_selection,
                observer,
                market,
            )
            study = build_baseline_study(
                inputs[market].frame,
                evaluated,
                oos_start=inputs[market].oos_start,
                precomputed_jm=fitted[market],
                precomputed_hmm=inputs[market].hmm,
                selection_initial=selection_loader,
                selection_progress=selection_saver,
            )
            _save_checkpoint(
                market_root / "baseline-study",
                study,
                "grid_study",
                identity,
            )
        studies[market] = study
        study_runtime.emit_selected_signals(observer, study.selections, market)
        study_runtime.emit_boundary_rows(observer, study.boundaries, market)
        _write_market_evidence(run_dir / market, inputs[market].frame, study)

    boundaries = pd.concat(
        [study.boundaries.assign(market=market) for market, study in studies.items()],
        ignore_index=True,
    )
    boundaries.to_csv(run_dir / "boundaries.csv", index=False)
    if len(boundaries) != 18:
        raise ArtifactError("grid evaluation did not produce 18 boundary rows")
    if not boundaries["passed"].all():
        finish_run_metadata(
            metadata_path,
            status="boundary_failed",
            metrics_opened=False,
            conclusion="locked changed grid failed the preregistered boundary gate",
        )
        _write_report(run_dir)
        write_inventory(run_dir)
        checkpoint_store.clear_checkpoint_tree(checkpoint_root)
        return run_dir

    metric_frames: list[pd.DataFrame] = []
    primary_paths: dict[str, dict[str, pd.DataFrame]] = {}
    for market in MARKETS:
        new_paths = baseline_paths(inputs[market].frame, studies[market], evaluated)
        for delay in spec.delays:
            compared = _comparison_paths(
                parent_dir,
                market,
                delay,
                new_paths[delay],
            )
            _write_trade_paths(run_dir / market / "trades", delay, compared)
            metric_frames.append(
                _path_metrics(compared, config).assign(market=market, delay=delay)
            )
            if delay == spec.primary_delay:
                primary_paths[market] = compared

    metrics = pd.concat(metric_frames, ignore_index=True)
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    bootstrap = _run_bootstrap(
        primary_paths,
        config,
        spec,
        checkpoint_root,
        identity,
        observer,
    )
    bootstrap.to_csv(run_dir / "bootstrap.csv", index=False)
    claim = _grid_claim(metrics, bootstrap, spec)
    write_json(run_dir / "claim.json", claim)
    finish_run_metadata(
        metadata_path,
        status="complete",
        metrics_opened=True,
        conclusion=claim["conclusion"],
    )
    _write_report(run_dir)
    write_inventory(run_dir)
    checkpoint_store.clear_checkpoint_tree(checkpoint_root)
    return run_dir


def _verify_parents(
    parent_dir: Path,
    calibration_dir: Path,
    config: ResearchConfig,
    spec: GridStudySpec,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    parent_receipt = verify_run(parent_dir)
    parent_metadata = read_json(parent_dir / "run.json")
    if (
        parent_receipt.get("status") != "complete"
        or parent_receipt.get("run_id") != spec.parent_run_id
        or parent_metadata.get("config_sha256") != config.sha256
        or sha256_file(parent_dir / "inventory.json") != spec.parent_inventory_sha256
        or sha256_file(parent_dir / "data-manifest.json") != spec.data_manifest_sha256
    ):
        raise ArtifactError("sealed v7 parent does not match the grid contract")

    calibration_receipt = verify_run(calibration_dir)
    calibration_metadata = read_json(calibration_dir / "run.json")
    if (
        calibration_receipt.get("status") != "complete"
        or calibration_receipt.get("metrics_opened") is not False
        or sha256_file(calibration_dir / "inventory.json")
        != spec.calibration_inventory_sha256
        or sha256_file(calibration_dir / "selection.json")
        != spec.calibration_selection_sha256
    ):
        raise ArtifactError("sealed calibration does not match the grid contract")
    return parent_receipt, parent_metadata, calibration_metadata


def _verify_control_source(root: Path, baseline_sha: str, current_sha: str) -> None:
    result = subprocess.run(
        ["git", "diff", "--quiet", baseline_sha, current_sha, "--", *CONTROL_SCOPE],
        cwd=root,
        check=False,
    )
    if result.returncode != 0:
        raise ArtifactError("v7 model, feature, walk-forward, or backtest code changed")


def _create_run(
    run_dir: Path,
    config: ResearchConfig,
    spec: GridStudySpec,
    parent_dir: Path,
    parent_receipt: dict[str, Any],
    run_id: str,
    identity: dict[str, str],
    control_git_sha: str,
) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "study.lock.toml").write_bytes(spec.path.read_bytes())
    (run_dir / "config.lock.toml").write_bytes(config.path.read_bytes())
    (run_dir / "data-manifest.json").write_bytes(
        (parent_dir / "data-manifest.json").read_bytes()
    )
    write_json(run_dir / "parent-verification.json", parent_receipt)
    write_json(
        run_dir / "control.json",
        {
            "parent_run_id": spec.parent_run_id,
            "calibration_run_id": spec.calibration_run_id,
            "control_scope": list(CONTROL_SCOPE),
            "control_baseline_git_sha": control_git_sha,
            "control_source_unchanged": True,
            "hmm_refit": False,
        },
    )
    write_json(
        run_dir / "run.json",
        {
            "schema_version": 1,
            "study_kind": "persistence_grid_evaluation",
            "run_id": run_id,
            "experiment_id": spec.experiment_id,
            "parent_run_id": spec.parent_run_id,
            "status": "running",
            "claim_class": "EXPLORATORY",
            "metrics_opened": False,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "packages": _package_versions(),
            **identity,
        },
    )


def _verify_locks(run_dir: Path, config: ResearchConfig, spec: GridStudySpec) -> None:
    if (
        sha256_file(run_dir / "study.lock.toml") != spec.sha256
        or sha256_file(run_dir / "config.lock.toml") != config.sha256
        or sha256_file(run_dir / "data-manifest.json") != spec.data_manifest_sha256
    ):
        raise ArtifactError("grid-evaluation locks changed")


def _load_parent_market(
    parent_dir: Path,
    market: str,
    config: ResearchConfig,
    spec: GridStudySpec,
) -> ParentMarket:
    frame = pd.read_csv(parent_dir / market / "features.csv")
    required = {
        "date",
        "equity_simple",
        "cash_return",
        "excess_return",
        "dd_10",
        "sortino_20",
        "sortino_60",
    }
    if not required.issubset(frame):
        raise ArtifactError(f"{market}: parent features are incomplete")
    dates = pd.to_datetime(frame["date"], errors="raise")
    if (
        dates.isna().any()
        or dates.duplicated().any()
        or not dates.is_monotonic_increasing
        or dates.max().date() > spec.data_cutoff
    ):
        raise ArtifactError(f"{market}: parent feature dates are invalid")
    frame["date"] = dates
    requested = date.fromisoformat(config.document["oos_start"]["requested"])
    oos_start = effective_oos_start(
        frame,
        requested=requested,
        fit_window=config.model_protocol.fit_window,
        validation_years=config.selection_protocol.validation_years,
    )
    if oos_start is None:
        raise ArtifactError(f"{market}: parent has no eligible OOS start")

    raw = pd.read_csv(parent_dir / market / "hmm-states.csv")
    if list(raw.columns) != ["date", "hmm_state"]:
        raise ArtifactError(f"{market}: parent HMM state schema changed")
    raw_dates = pd.DatetimeIndex(
        pd.to_datetime(raw["date"], errors="raise"), name="date"
    )
    if raw_dates.has_duplicates or not raw_dates.is_monotonic_increasing:
        raise ArtifactError(f"{market}: parent HMM dates are invalid")
    states = pd.Series(
        pd.to_numeric(raw["hmm_state"], errors="coerce").to_numpy(),
        index=raw_dates,
        name="hmm_state",
        dtype=float,
    )
    if not states.dropna().isin((0.0, 1.0)).all():
        raise ArtifactError(f"{market}: parent HMM states are invalid")
    fits = pd.read_csv(parent_dir / market / "hmm-fits.csv")
    return ParentMarket(frame, oos_start, HMMResult(states, fits))


_JMTask = tuple[
    str,
    pd.DataFrame,
    ResearchConfig,
    FixedJMResult | None,
    Path,
    dict[str, str],
]


def _fit_pending_jm(
    inputs: dict[str, ParentMarket],
    config: ResearchConfig,
    pending: dict[str, tuple[Path, FixedJMResult | None]],
    identity: dict[str, str],
) -> dict[str, FixedJMResult]:
    results: dict[str, FixedJMResult] = {}
    tasks: list[_JMTask] = []
    for market, (stem, initial) in pending.items():
        tasks.append((market, inputs[market].frame, config, initial, stem, identity))
    if tasks:
        workers = min(GRID_WORKERS, len(tasks))
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context("forkserver"),
        ) as executor:
            futures = {executor.submit(_fit_jm_worker, task): task[0] for task in tasks}
            for future in as_completed(futures):
                market = futures[future]
                result = future.result()
                _save_checkpoint(pending[market][0], result, "grid_jm", identity)
                results[market] = result
    for market, (_, initial) in pending.items():
        if market not in results:
            if initial is None:
                raise ArtifactError(f"{market}: JM fit did not return")
            results[market] = initial
    return results


def _fit_jm_worker(task: _JMTask) -> FixedJMResult:
    market, frame, config, initial, stem, identity = task

    def save(value: FixedJMResult) -> None:
        _save_checkpoint(stem, value, "grid_jm", identity)
        completed = int(value.states.notna().all(axis=1).sum())
        print(f"{market}: changed-grid JM {completed}", file=sys.stderr, flush=True)

    with threadpool_limits(limits=1):
        return fixed_jm_states(
            frame,
            config.model_protocol,
            config.jm_protocol,
            initial=initial,
            checkpoint_every=50,
            progress=save,
        )


def _write_market_evidence(
    target: Path,
    frame: pd.DataFrame,
    study: BaselineStudy,
) -> None:
    target.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target / "features.csv", index=False)
    study.jm.states.to_csv(target / "jm-states.csv")
    study.jm.refits.to_csv(target / "jm-refits.csv", index=False)
    study.hmm.states.to_csv(target / "hmm-states.csv", header=True)
    study.hmm.fits.to_csv(target / "hmm-fits.csv", index=False)
    study.hmm_candidates.to_csv(target / "hmm-candidates.csv")
    study.boundaries.to_csv(target / "boundaries.csv", index=False)
    for model, by_delay in study.selections.items():
        for delay, selection in by_delay.items():
            directory = target / f"{model}-delay-{delay}"
            directory.mkdir(exist_ok=True)
            selection.choices.to_csv(directory / "choices.csv", index=False)
            selection.surface.to_csv(directory / "cv-surface.csv", index=False)
            selection.candidate_returns.to_csv(directory / "candidate-returns.csv")
            selection.signal.to_csv(directory / "selected-signal.csv", header=True)


def _comparison_paths(
    parent_dir: Path,
    market: str,
    delay: int,
    new: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    parent = {
        model: pd.read_csv(
            parent_dir / market / "trades" / f"{model}-delay-{delay}.csv"
        )
        for model in ("buy_and_hold", "hmm", "fixed_jm")
    }
    candidates = {
        "buy_and_hold": parent["buy_and_hold"],
        "hmm_v7": parent["hmm"],
        "hmm_new_grid": new["hmm"],
        "fixed_jm_v7": parent["fixed_jm"],
        "fixed_jm_new_grid": new["fixed_jm"],
    }
    reference_dates: pd.DatetimeIndex | None = None
    reference_returns: pd.DataFrame | None = None
    output: dict[str, pd.DataFrame] = {}
    for model in COMPARISON_MODELS:
        path = candidates[model].copy()
        if tuple(path.columns) != TRADE_COLUMNS or path.empty:
            raise ArtifactError(f"{market}/{model}: trade schema changed")
        path["date"] = pd.to_datetime(path["date"], errors="raise")
        dates = pd.DatetimeIndex(path["date"])
        if dates.has_duplicates or not dates.is_monotonic_increasing:
            raise ArtifactError(f"{market}/{model}: trade dates are invalid")
        returns = path[["equity_simple", "cash_return"]].reset_index(drop=True)
        if reference_dates is None:
            reference_dates = dates
            reference_returns = returns
        elif not dates.equals(reference_dates):
            raise ArtifactError(f"{market}/{model}: OOS sample differs from v7")
        elif not np.allclose(returns, reference_returns, rtol=0, atol=1e-15):
            raise ArtifactError(f"{market}/{model}: market returns differ from v7")
        numeric = path[list(TRADE_COLUMNS[1:])].to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ArtifactError(f"{market}/{model}: trade path is incomplete")
        output[model] = path
    return output


def _write_trade_paths(
    target: Path,
    delay: int,
    paths: dict[str, pd.DataFrame],
) -> None:
    if tuple(paths) != COMPARISON_MODELS:
        raise ArtifactError("comparison path order changed")
    target.mkdir(parents=True, exist_ok=True)
    for model, path in paths.items():
        path.to_csv(target / f"{model}-delay-{delay}.csv", index=False)


def _path_metrics(
    paths: dict[str, pd.DataFrame],
    config: ResearchConfig,
) -> pd.DataFrame:
    if tuple(paths) != COMPARISON_MODELS:
        raise ArtifactError("metric path order changed")
    rows = []
    protocol = config.metrics_protocol
    for model, path in paths.items():
        values = performance_metrics(
            path,
            periods_per_year=protocol.periods_per_year,
            volatility_ddof=protocol.volatility_ddof,
            expected_shortfall_quantile=protocol.expected_shortfall_quantile,
            turnover_scale=protocol.turnover_scale,
        )
        rows.append(
            {
                "model": model,
                **values,
                "cash_fraction": float(1.0 - path["position"].mean()),
                "switch_count": int((path["one_way_turnover"] > 0).sum()),
            }
        )
    return pd.DataFrame.from_records(rows)


def _run_bootstrap(
    paths: dict[str, dict[str, pd.DataFrame]],
    config: ResearchConfig,
    spec: GridStudySpec,
    checkpoint_root: Path,
    identity: dict[str, str],
    observer: EventObserver | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pairs = {
        "fixed_jm": ("fixed_jm_new_grid", "fixed_jm_v7"),
        "hmm": ("hmm_new_grid", "hmm_v7"),
    }
    for market in MARKETS:
        for model, (challenger, baseline) in pairs.items():
            for block in spec.bootstrap_blocks:
                stem = checkpoint_root / f"bootstrap-{market}-{model}-{block}"
                initial = _load_checkpoint(
                    stem,
                    "grid_bootstrap",
                    identity,
                    BootstrapProgress,
                )
                final = [initial]
                event_observer = study_runtime.bootstrap_recorder(
                    lambda _block, value, path=stem: _save_checkpoint(
                        path, value, "grid_bootstrap", identity
                    ),
                    observer,
                    spec.bootstrap_replications,
                )

                def save(
                    value: BootstrapProgress,
                    state: list[BootstrapProgress | None] = final,
                    emit: Any = event_observer,
                    current_block: int = block,
                ) -> None:
                    state[0] = value
                    emit(current_block, value)

                compared = paths[market]
                result = bootstrap_sharpe_delta(
                    compared[challenger]["strategy_return"],
                    compared[baseline]["strategy_return"],
                    compared[challenger]["cash_return"],
                    replications=spec.bootstrap_replications,
                    mean_block_length=block,
                    seed=spec.bootstrap_seed,
                    confidence_level=spec.confidence_level,
                    periods_per_year=config.metrics_protocol.periods_per_year,
                    volatility_ddof=config.metrics_protocol.volatility_ddof,
                    initial=initial,
                    progress=save,
                )
                if (
                    final[0] is None
                    or len(final[0].draws) != spec.bootstrap_replications
                ):
                    raise ArtifactError("bootstrap did not preserve its final draws")
                draws = final[0].draws
                one_sided_p = float(
                    (np.count_nonzero(draws <= 0.0) + 1) / (len(draws) + 1)
                )
                rows.append(
                    {
                        "market": market,
                        "model": model,
                        "block_length": block,
                        "observed_delta": result.observed,
                        "lower_one_sided": result.lower_one_sided,
                        "confidence_low": result.confidence_low,
                        "confidence_high": result.confidence_high,
                        "one_sided_p": one_sided_p,
                        "replications": result.replications,
                    }
                )
    output = pd.DataFrame.from_records(rows)
    output["holm_adjusted_p"] = np.nan
    for (_, _), indexes in output.groupby(["model", "block_length"]).groups.items():
        ordered = sorted(indexes, key=lambda index: output.loc[index, "one_sided_p"])
        running = 0.0
        count = len(ordered)
        for rank, index in enumerate(ordered):
            adjusted = min(1.0, (count - rank) * output.loc[index, "one_sided_p"])
            running = max(running, adjusted)
            output.loc[index, "holm_adjusted_p"] = running
    if output["holm_adjusted_p"].isna().any():
        raise ArtifactError("Holm adjustment is incomplete")
    return output


def _grid_claim(
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    spec: GridStudySpec,
) -> dict[str, Any]:
    primary = metrics.loc[metrics["delay"] == spec.primary_delay]
    uncertainty = bootstrap.loc[bootstrap["block_length"] == spec.bootstrap_blocks[0]]
    markets = []
    hmm_rows = []
    directional = []
    for market in MARKETS:
        values = primary.loc[primary["market"] == market].set_index("model")
        if set(values.index) != set(COMPARISON_MODELS):
            raise ArtifactError(f"{market}: primary comparison is incomplete")
        fixed = uncertainty.loc[
            (uncertainty["market"] == market) & (uncertainty["model"] == "fixed_jm")
        ]
        hmm = uncertainty.loc[
            (uncertainty["market"] == market) & (uncertainty["model"] == "hmm")
        ]
        if len(fixed) != 1 or len(hmm) != 1:
            raise ArtifactError(f"{market}: primary bootstrap is incomplete")
        fixed_delta = float(
            values.loc["fixed_jm_new_grid", "sharpe"]
            - values.loc["fixed_jm_v7", "sharpe"]
        )
        hmm_delta = float(
            values.loc["hmm_new_grid", "sharpe"] - values.loc["hmm_v7", "sharpe"]
        )
        if not math.isclose(
            fixed_delta,
            float(fixed.iloc[0]["observed_delta"]),
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ArtifactError(f"{market}: fixed-JM delta disagrees")
        if not math.isclose(
            hmm_delta,
            float(hmm.iloc[0]["observed_delta"]),
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ArtifactError(f"{market}: HMM delta disagrees")
        adjusted = float(fixed.iloc[0]["holm_adjusted_p"])
        markets.append(
            {
                "market": market,
                "sharpe_v7": float(values.loc["fixed_jm_v7", "sharpe"]),
                "sharpe_new_grid": float(values.loc["fixed_jm_new_grid", "sharpe"]),
                "delta_sharpe": fixed_delta,
                "positive": fixed_delta > 0,
                "holm_adjusted_p": adjusted,
                "holm_significant": adjusted < 0.05,
            }
        )
        hmm_rows.append(
            {
                "market": market,
                "delta_sharpe": hmm_delta,
                "positive": hmm_delta > 0,
                "holm_adjusted_p": float(hmm.iloc[0]["holm_adjusted_p"]),
            }
        )
        new_jm = values.loc["fixed_jm_new_grid"]
        new_hmm = values.loc["hmm_new_grid"]
        hold = values.loc["buy_and_hold"]
        checks = {
            "sharpe_above_hmm": bool(new_jm["sharpe"] > new_hmm["sharpe"]),
            "sharpe_above_buy_and_hold": bool(new_jm["sharpe"] > hold["sharpe"]),
            "mdd_below_buy_and_hold": bool(
                abs(new_jm["maximum_drawdown"]) < abs(hold["maximum_drawdown"])
            ),
        }
        directional.append({"market": market, **checks, "passed": all(checks.values())})

    positives = sum(row["positive"] for row in markets)
    significant = sum(row["holm_significant"] for row in markets)
    direction = (
        "consistent improvement"
        if positives == 3
        else ("mixed" if positives else "not supported")
    )
    uncertainty_supported = significant >= 2
    return {
        "claim_class": "EXPLORATORY",
        "paper_replication_claim_allowed": False,
        "primary_delay": spec.primary_delay,
        "primary_block_length": spec.bootstrap_blocks[0],
        "fixed_jm": markets,
        "hmm_secondary": hmm_rows,
        "positive_fixed_jm_markets": positives,
        "holm_significant_fixed_jm_markets": significant,
        "directional_outcome": direction,
        "uncertainty_supported": uncertainty_supported,
        "new_grid_directional_gate": {
            "markets": directional,
            "passed": all(row["passed"] for row in directional),
        },
        "conclusion": (
            f"{direction}; Holm uncertainty support "
            + ("present" if uncertainty_supported else "absent")
        ),
    }


def verify_grid_run(run: str | Path) -> dict[str, Any]:
    """Independently verify a terminal changed-grid artifact."""
    run_dir = Path(run).resolve()
    metadata = read_json(run_dir / "run.json")
    if (
        metadata.get("schema_version") != 1
        or metadata.get("study_kind") != "persistence_grid_evaluation"
        or metadata.get("status") not in {"complete", "boundary_failed"}
    ):
        raise ArtifactError("grid run is not in a verifiable terminal state")
    verify_inventory(run_dir)
    config = load_config(run_dir / "config.lock.toml")
    spec = load_grid_spec(run_dir / "study.lock.toml", config)
    _verify_locks(run_dir, config, spec)
    expected_id = "grid-eval-" + "-".join(
        str(metadata[key])[:12]
        for key in ("spec_sha256", "data_manifest_sha256", "git_sha")
    )
    if run_dir.name != expected_id or metadata.get("run_id") != expected_id:
        raise ArtifactError("grid run identity changed")
    if metadata.get("spec_sha256") != spec.sha256:
        raise ArtifactError("grid spec identity changed")

    evaluated = replace(
        config,
        jm_protocol=replace(config.jm_protocol, lambda_grid=spec.jm_grid),
        hmm_protocol=replace(config.hmm_protocol, smoothing_grid=spec.hmm_grid),
    )
    expected_boundaries = []
    for market in MARKETS:
        frame = pd.read_csv(run_dir / market / "features.csv")
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        requested = date.fromisoformat(config.document["oos_start"]["requested"])
        oos_start = effective_oos_start(
            frame,
            requested=requested,
            fit_window=config.model_protocol.fit_window,
            validation_years=config.selection_protocol.validation_years,
        )
        if oos_start is None:
            raise ArtifactError(f"{market}: verifier has no OOS start")
        grids = {
            "fixed_jm": evaluated.jm_protocol.lambda_grid,
            "hmm": tuple(
                float(value) for value in evaluated.hmm_protocol.smoothing_grid
            ),
        }
        for model, grid in grids.items():
            for delay in spec.delays:
                choices = pd.read_csv(
                    run_dir / market / f"{model}-delay-{delay}" / "choices.csv"
                )
                diagnostic = boundary_diagnostic(
                    choices,
                    grid,
                    oos_start=oos_start,
                    fraction_limit=spec.boundary_fraction_limit,
                )
                expected_boundaries.append(
                    {
                        "model": model,
                        "delay": delay,
                        **diagnostic.__dict__,
                        "market": market,
                    }
                )
    observed_boundaries = pd.read_csv(run_dir / "boundaries.csv")
    recomputed_boundaries = pd.DataFrame.from_records(expected_boundaries)
    _assert_frame_equal(
        observed_boundaries,
        recomputed_boundaries,
        ("market", "model", "delay"),
        "boundaries",
    )
    all_passed = bool(recomputed_boundaries["passed"].all())
    if metadata["status"] == "boundary_failed":
        if all_passed or metadata.get("metrics_opened") is not False:
            raise ArtifactError("boundary-failed metadata is inconsistent")
        forbidden = ("metrics.csv", "bootstrap.csv", "claim.json")
        if any((run_dir / name).exists() for name in forbidden):
            raise ArtifactError("boundary-failed run exposed performance")
        return {
            "schema_version": 1,
            "study_kind": "persistence_grid_evaluation",
            "run_id": expected_id,
            "status": "boundary_failed",
            "boundary_rows": len(recomputed_boundaries),
            "metrics_opened": False,
        }

    if not all_passed or metadata.get("metrics_opened") is not True:
        raise ArtifactError("complete grid run bypassed its boundary gate")
    recomputed_metrics = []
    for market in MARKETS:
        for delay in spec.delays:
            paths = {
                model: pd.read_csv(
                    run_dir / market / "trades" / f"{model}-delay-{delay}.csv"
                )
                for model in COMPARISON_MODELS
            }
            recomputed_metrics.append(
                _path_metrics(paths, config).assign(market=market, delay=delay)
            )
    metrics = pd.read_csv(run_dir / "metrics.csv")
    expected_metrics = pd.concat(recomputed_metrics, ignore_index=True)
    _assert_frame_equal(
        metrics,
        expected_metrics,
        ("market", "delay", "model"),
        "metrics",
    )
    bootstrap = pd.read_csv(run_dir / "bootstrap.csv")
    _validate_bootstrap(bootstrap, spec)
    claim = read_json(run_dir / "claim.json")
    expected_claim = _grid_claim(metrics, bootstrap, spec)
    if claim != expected_claim or metadata.get("conclusion") != claim["conclusion"]:
        raise ArtifactError("grid claim does not match its evidence")
    return {
        "schema_version": 1,
        "study_kind": "persistence_grid_evaluation",
        "run_id": expected_id,
        "status": "complete",
        "boundary_rows": len(recomputed_boundaries),
        "metric_rows": len(metrics),
        "bootstrap_rows": len(bootstrap),
        "metrics_opened": True,
        "conclusion": claim["conclusion"],
    }


def _validate_bootstrap(frame: pd.DataFrame, spec: GridStudySpec) -> None:
    required = {
        "market",
        "model",
        "block_length",
        "observed_delta",
        "lower_one_sided",
        "confidence_low",
        "confidence_high",
        "one_sided_p",
        "replications",
        "holm_adjusted_p",
    }
    if set(frame.columns) != required:
        raise ArtifactError("bootstrap schema changed")
    expected = {
        (market, model, block)
        for market in MARKETS
        for model in ("fixed_jm", "hmm")
        for block in spec.bootstrap_blocks
    }
    observed = set(
        zip(frame["market"], frame["model"], frame["block_length"], strict=True)
    )
    numeric = frame[list(required - {"market", "model"})].to_numpy(dtype=float)
    if (
        observed != expected
        or not np.isfinite(numeric).all()
        or not (frame["replications"] == spec.bootstrap_replications).all()
        or not frame["one_sided_p"].between(0, 1).all()
        or not frame["holm_adjusted_p"].between(0, 1).all()
    ):
        raise ArtifactError("bootstrap evidence is invalid")


def _assert_frame_equal(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    keys: tuple[str, ...],
    label: str,
) -> None:
    left = observed.sort_values(list(keys)).reset_index(drop=True)
    right = expected.sort_values(list(keys)).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            left,
            right,
            check_dtype=False,
            check_exact=False,
            rtol=0,
            atol=1e-12,
        )
    except AssertionError as exc:
        raise ArtifactError(f"{label} do not match recomputation") from exc


def build_grid_report(run: str | Path) -> Path:
    """Verify and return the sealed English grid-evaluation report."""
    run_dir = Path(run).resolve()
    verify_grid_run(run_dir)
    target = run_dir / "report.html"
    if not target.is_file():
        raise ArtifactError("grid report is missing")
    return target


def _write_report(run_dir: Path) -> None:
    metadata = read_json(run_dir / "run.json")
    boundaries = pd.read_csv(run_dir / "boundaries.csv")
    failed = boundaries.loc[~boundaries["passed"]]
    if metadata["status"] == "complete":
        metrics = pd.read_csv(run_dir / "metrics.csv")
        claim = read_json(run_dir / "claim.json")
        primary = metrics.loc[
            (metrics["delay"] == claim["primary_delay"])
            & metrics["model"].isin(
                (
                    "buy_and_hold",
                    "hmm_v7",
                    "hmm_new_grid",
                    "fixed_jm_v7",
                    "fixed_jm_new_grid",
                )
            ),
            ["market", "model", "sharpe", "maximum_drawdown", "turnover"],
        ]
        result = escape(claim["conclusion"])
        metric_table = primary.to_html(
            index=False, border=0, float_format="{:.4f}".format
        )
        evidence = (
            "All boundary checks passed, so OOS paths and metrics were opened once."
        )
    else:
        result = "Boundary gate failed; performance remains sealed"
        metric_table = ""
        evidence = (
            "At least one upper-grid candidate was selected too often. No trades, "
            "Sharpe, bootstrap output or performance claim was produced."
        )
    boundary_table = boundaries[
        [
            "market",
            "model",
            "delay",
            "upper_candidate",
            "selected_months",
            "total_months",
            "fraction",
            "passed",
        ]
    ].to_html(index=False, border=0, float_format="{:.4f}".format)
    failed_count = len(failed)
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Persistence Grid Evaluation</title>
<style>
:root{{color-scheme:dark;--bg:#0b0d10;--panel:#15191f;--line:#303640;--text:#f4f6f8;--muted:#aab3bf;--green:#63d59a;--amber:#f0bd64}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 system-ui,sans-serif}}
main{{width:min(1120px,calc(100% - 28px));margin:auto;padding:38px 0 64px}}h1{{font-size:clamp(2rem,6vw,4rem);line-height:1.05;margin:.2em 0}}
.eyebrow{{color:var(--green);font-weight:800;text-transform:uppercase}}.lead{{color:var(--muted);max-width:78ch}}
section{{padding:26px 0;border-top:1px solid var(--line)}}.verdict{{padding:16px;border-left:4px solid var(--amber);background:#201a0e}}
.table{{overflow:auto;border:1px solid var(--line)}}table{{border-collapse:collapse;width:100%;white-space:nowrap}}
th,td{{padding:9px 11px;border-bottom:1px solid var(--line);text-align:left}}th{{background:#1b2027;color:var(--muted)}}
code{{overflow-wrap:anywhere}}@media(max-width:600px){{main{{padding-top:22px}}}}
</style></head><body><main>
<div class="eyebrow">Exploratory grid attribution</div>
<h1>Persistence-calibrated grid evaluation</h1>
<p class="lead">Only the fixed-JM lambda grid and HMM smoothing grid differ from sealed v7. This is not a fresh paper replication.</p>
<div class="verdict"><strong>{result}</strong><p>{escape(evidence)}</p></div>
<section><h2>Boundary-first gate</h2><p>{failed_count} of 18 checks failed.</p><div class="table">{boundary_table}</div></section>
{"<section><h2>Primary-delay metrics</h2><div class='table'>" + metric_table + "</div></section>" if metric_table else ""}
<section><h2>Run identity</h2><p><code>{escape(str(metadata["run_id"]))}</code></p>
<p>Study hash: <code>{escape(str(metadata["spec_sha256"]))}</code></p></section>
</main></body></html>"""
    (run_dir / "report.html").write_text(html, encoding="utf-8")


def _load_checkpoint(
    stem: Path,
    kind: str,
    identity: dict[str, str],
    expected: type[Any],
) -> Any:
    try:
        value = checkpoint_store.load_checkpoint(stem, kind=kind, identity=identity)
    except checkpoint_store.CheckpointStoreError as exc:
        raise ArtifactError(str(exc)) from exc
    if value is not None and not isinstance(value, expected):
        raise ArtifactError(f"invalid {kind} checkpoint type")
    return value


def _save_checkpoint(
    stem: Path,
    value: Any,
    kind: str,
    identity: dict[str, str],
) -> None:
    try:
        checkpoint_store.save_checkpoint(stem, value, kind=kind, identity=identity)
    except checkpoint_store.CheckpointStoreError as exc:
        raise ArtifactError(str(exc)) from exc


def _load_selection(
    root: Path,
    identity: dict[str, str],
    model: str,
    delay: int,
) -> SelectionProgress | None:
    return _load_checkpoint(
        root / f"selection-{model}-{delay}",
        "selection",
        identity,
        SelectionProgress,
    )


def _save_selection(
    root: Path,
    identity: dict[str, str],
    model: str,
    delay: int,
    value: SelectionProgress,
) -> None:
    _save_checkpoint(
        root / f"selection-{model}-{delay}",
        value,
        "selection",
        identity,
    )


def _package_versions() -> dict[str, str]:
    output = {}
    for package in (
        "adaptive-jump-model",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "hmmlearn",
        "jumpmodels",
    ):
        try:
            output[package] = version(package)
        except PackageNotFoundError:
            output[package] = "not-installed"
    return output
