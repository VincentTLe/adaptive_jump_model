"""Command-line entry point for reproducible research workflows."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd

from adaptive_jump import artifacts as _artifacts
from adaptive_jump.calibration_runner import run_calibration_study
from adaptive_jump.config import ConfigError, ResearchConfig, load_config
from adaptive_jump.data import AcquisitionError, acquire, research_git_sha
from adaptive_jump.features import effective_oos_start, prepare_market
from adaptive_jump.grid_runner import run_grid_evaluation
from adaptive_jump.grid_spec import load_grid_spec
from adaptive_jump.models import FixedJMResult, HMMResult, fixed_jm_states, hmm_states
from adaptive_jump.monitor import checkpoints as checkpoint_store
from adaptive_jump.monitor import study_runtime
from adaptive_jump.monitor.child_events import (
    ChildEventError,
    child_observer_from_environment,
)
from adaptive_jump.monitor.events import EventObserver, emit_artifact_verified
from adaptive_jump.reporting import build_report
from adaptive_jump.simple_jm_figures import render_figures
from adaptive_jump.simple_jm_suite import (
    load_dd_loss_scale_spec,
    load_simple_jm_spec,
    run_dd_loss_scale_study,
    run_simple_jm_study,
)
from adaptive_jump.walkforward import (
    BaselineStudy,
    SelectionProgress,
    baseline_paths,
    build_baseline_study,
    open_baseline_metrics,
)
from adaptive_jump.window_runner import run_window_sensitivity
from adaptive_jump.window_spec import load_window_spec

RunError = _artifacts.ArtifactError


@dataclass(frozen=True)
class FrozenData:
    path: Path
    document: dict[str, Any]
    sha256: str


@dataclass(frozen=True)
class MarketInput:
    frame: pd.DataFrame
    oos_start: date


HMM_WORKERS, MODEL_CHECKPOINT_DAYS = 16, 50


def load_frozen_data(
    config: ResearchConfig, manifest_path: str | Path | None = None
) -> FrozenData:
    """Resolve and independently verify one exact acquisition manifest."""
    root = config.path.parent
    if manifest_path is None:
        candidates = sorted(
            (root / config.raw_root).glob(f"{config.config_id}-*/manifest.json")
        )
        matching = []
        for candidate in candidates:
            document = _artifacts.read_json(candidate)
            if document.get("config_sha256") == config.sha256:
                matching.append(candidate)
        if len(matching) != 1:
            raise RunError(
                f"expected one matching data manifest, found {len(matching)}; "
                "pass --manifest explicitly"
            )
        path = matching[0]
    else:
        path = Path(manifest_path)
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
    document = _artifacts.read_json(path)
    _verify_manifest(config, document, root)
    return FrozenData(path, document, _artifacts.sha256_file(path))


def prepare_manifest_market(
    config: ResearchConfig, frozen: FrozenData, market_id: str
) -> MarketInput:
    """Load one market's canonical pair and build its causal feature frame."""
    markets = {market.id: market for market in config.markets}
    if market_id not in markets:
        raise RunError(f"unknown market: {market_id}")
    records = {
        (source["market"], source["kind"]): source
        for source in frozen.document["sources"]
    }
    equity = pd.read_csv(
        config.path.parent / records[(market_id, "equity")]["canonical"]["path"]
    )
    cash = pd.read_csv(
        config.path.parent / records[(market_id, "cash")]["canonical"]["path"]
    )
    frame = prepare_market(equity, cash, markets[market_id], config)
    requested = date.fromisoformat(config.document["oos_start"]["requested"])
    start = effective_oos_start(
        frame,
        requested=requested,
        fit_window=config.model_protocol.fit_window,
        validation_years=config.selection_protocol.validation_years,
    )
    if start is None:
        raise RunError(f"{market_id}: insufficient observations for OOS")
    return MarketInput(frame, start)


def _verify_manifest(
    config: ResearchConfig, document: dict[str, Any], root: Path
) -> None:
    expected = {
        (market.id, "equity", market.equity.source_id) for market in config.markets
    } | {(market.id, "cash", market.cash.source_id) for market in config.markets}
    actual = {
        (source.get("market"), source.get("kind"), source.get("source_id"))
        for source in document.get("sources", [])
    }
    if (
        document.get("config_id") != config.config_id
        or document.get("config_sha256") != config.sha256
        or document.get("replication_cutoff") != config.replication_cutoff.isoformat()
        or actual != expected
    ):
        raise RunError("data manifest identity violates the frozen config")
    processed = (root / config.processed_root).resolve()
    for source in document["sources"]:
        record = source.get("canonical", {})
        path = (root / str(record.get("path", ""))).resolve()
        if not path.is_relative_to(processed) or not path.is_file():
            raise RunError(f"invalid canonical path: {path}")
        if _artifacts.sha256_file(path) != record.get("sha256"):
            raise RunError(f"canonical hash mismatch: {path}")
        frame = pd.read_csv(path)
        if frame.empty or list(frame.columns) != ["date", "value"]:
            raise RunError(f"invalid canonical columns: {path}")
        dates = pd.to_datetime(frame["date"], errors="raise")
        if (
            dates.duplicated().any()
            or not dates.is_monotonic_increasing
            or dates.min().date() < config.sample_start
            or dates.max().date() > config.replication_cutoff
        ):
            raise RunError(f"invalid canonical dates: {path}")
        numeric = pd.to_numeric(frame["value"], errors="coerce")
        invalid = frame["value"].notna() & numeric.isna()
        finite = numeric.dropna().map(math.isfinite)
        if invalid.any() or not finite.all():
            raise RunError(f"invalid canonical values: {path}")


def run_replication(
    config: ResearchConfig, frozen: FrozenData, observer: EventObserver | None = None
) -> Path:
    """Run or exactly resume the sealed three-market baseline study."""
    root = config.path.parent
    git_sha = research_git_sha(root)
    identity = {
        "config_sha256": config.sha256,
        "data_manifest_sha256": frozen.sha256,
        "git_sha": git_sha,
    }
    run_id = "fixed-baselines-" + "-".join(
        identity[key][:12]
        for key in ("config_sha256", "data_manifest_sha256", "git_sha")
    )
    run_dir = root / config.artifact_root / "fixed-baselines" / run_id
    checkpoint_root = root / config.artifact_root / ".monitor" / "checkpoints" / run_id
    metadata_path = run_dir / "run.json"
    if metadata_path.exists():
        metadata = _artifacts.read_json(metadata_path)
        if any(metadata.get(key) != value for key, value in identity.items()):
            raise RunError("existing run identity does not match")
        if (
            _artifacts.sha256_file(run_dir / "config.lock.toml") != config.sha256
            or _artifacts.sha256_file(run_dir / "data-manifest.json") != frozen.sha256
        ):
            raise RunError("existing run locks do not match")
        if metadata.get("status") in {"complete", "boundary_failed"}:
            _artifacts.verify_inventory(run_dir)
            return run_dir
    else:
        run_dir.mkdir(parents=True)
        (run_dir / "config.lock.toml").write_bytes(config.path.read_bytes())
        (run_dir / "data-manifest.json").write_bytes(frozen.path.read_bytes())
        _artifacts.write_json(
            metadata_path,
            {
                "schema_version": 1,
                "run_id": run_id,
                "status": "running",
                "claim_label": "proxy replication",
                "metrics_opened": False,
                "created_at_utc": datetime.now(UTC).isoformat(),
                "packages": _package_versions(),
                **identity,
            },
        )

    inputs: dict[str, MarketInput] = {}
    studies: dict[str, BaselineStudy] = {}
    for market in config.markets:
        market_input = prepare_manifest_market(config, frozen, market.id)
        inputs[market.id] = market_input
        market_dir = run_dir / market.id
        checkpoint_dir = checkpoint_root / market.id
        checkpoint = _load_cache(
            checkpoint_dir / "baseline-study",
            "baseline_study",
            identity,
            BaselineStudy,
        )
        if checkpoint is None:
            initial_hmm = _load_cache(
                checkpoint_dir / "hmm-progress", "hmm", identity, HMMResult
            )
            total_hmm_days = max(
                0,
                market_input.frame["equity_log"].notna().sum()
                - config.model_protocol.fit_window
                + 1,
            )

            def save_hmm_progress(
                result: HMMResult,
                target: Path = checkpoint_dir / "hmm-progress",
                market_id: str = market.id,
                total: int = total_hmm_days,
            ) -> None:
                _write_cache(target, result, "hmm", identity)
                print(
                    f"{market_id}: HMM {len(result.fits)}/{total}",
                    file=sys.stderr,
                    flush=True,
                )

            fitted_hmm = hmm_states(
                market_input.frame,
                config.model_protocol,
                config.hmm_protocol,
                initial=initial_hmm,
                n_jobs=HMM_WORKERS,
                checkpoint_every=MODEL_CHECKPOINT_DAYS,
                progress=save_hmm_progress,
                observer=study_runtime.model_observer(
                    observer, market.id, "hmm", market_input.frame
                ),
            )
            initial_jm = _load_cache(
                checkpoint_dir / "jm-progress", "fixed_jm", identity, FixedJMResult
            )
            jm_columns = ("dd_10", "sortino_20", "sortino_60", "excess_return")
            jm_complete = market_input.frame.loc[:, jm_columns].notna().all(axis=1)
            total_jm_days = max(
                0, int(jm_complete.sum()) - config.model_protocol.fit_window + 1
            )

            def save_jm_progress(
                result: FixedJMResult,
                target: Path = checkpoint_dir / "jm-progress",
                market_id: str = market.id,
                total: int = total_jm_days,
            ) -> None:
                _write_cache(target, result, "fixed_jm", identity)
                completed = int(result.states.notna().all(axis=1).sum())
                print(
                    f"{market_id}: JM {completed}/{total}",
                    file=sys.stderr,
                    flush=True,
                )

            fitted_jm = fixed_jm_states(
                market_input.frame,
                config.model_protocol,
                config.jm_protocol,
                initial=initial_jm,
                checkpoint_every=MODEL_CHECKPOINT_DAYS,
                progress=save_jm_progress,
                observer=study_runtime.model_observer(
                    observer, market.id, "fixed_jm", market_input.frame
                ),
            )

            checkpoint = build_baseline_study(
                market_input.frame,
                config,
                oos_start=market_input.oos_start,
                precomputed_jm=fitted_jm,
                precomputed_hmm=fitted_hmm,
                selection_initial=partial(_load_selection, checkpoint_dir, identity),
                selection_progress=study_runtime.baseline_selection_recorder(
                    partial(_save_selection, checkpoint_dir, identity),
                    observer,
                    market.id,
                ),
            )
        elif checkpoint.oos_start != market_input.oos_start:
            raise RunError(f"{market.id}: checkpoint OOS start mismatch")
        study_runtime.emit_selected_signals(observer, checkpoint.selections, market.id)
        _write_checkpoint(market_dir, market_input.frame, checkpoint)
        study_runtime.emit_boundary_rows(observer, checkpoint.boundaries, market.id)
        _write_cache(
            checkpoint_dir / "baseline-study", checkpoint, "baseline_study", identity
        )
        checkpoint_store.clear_checkpoint(checkpoint_dir / "hmm-progress")
        checkpoint_store.clear_checkpoint(checkpoint_dir / "jm-progress")
        for metadata in checkpoint_dir.glob("selection-*.json"):
            checkpoint_store.clear_checkpoint(metadata.with_suffix(""))
        studies[market.id] = checkpoint

    boundaries = pd.concat(
        [
            study.boundaries.assign(market=market_id)
            for market_id, study in studies.items()
        ],
        ignore_index=True,
    )
    boundaries.to_csv(run_dir / "boundaries.csv", index=False)
    if not boundaries["passed"].all():
        _artifacts.write_inventory(run_dir)
        _finish_run(
            metadata_path,
            status="boundary_failed",
            metrics_opened=False,
            conclusion="grid expansion required before OOS metrics",
        )
        return run_dir

    metric_frames = []
    for market_id, study in studies.items():
        market_input = inputs[market_id]
        metrics = open_baseline_metrics(market_input.frame, study, config)
        metric_frames.append(metrics.assign(market=market_id))
        for delay, models in baseline_paths(market_input.frame, study, config).items():
            trades = run_dir / market_id / "trades"
            trades.mkdir(exist_ok=True)
            for model_name, path in models.items():
                path.to_csv(trades / f"{model_name}-delay-{delay}.csv", index=False)
    metrics = pd.concat(metric_frames, ignore_index=True)
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    gate = _artifacts.directional_gate(metrics, config.backtest_protocol.primary_delay)
    _artifacts.write_json(run_dir / "claim.json", gate)
    _artifacts.write_inventory(run_dir)
    _finish_run(
        metadata_path,
        status="complete",
        metrics_opened=True,
        conclusion=gate["conclusion"],
    )
    return run_dir


def _write_checkpoint(
    market_dir: Path, frame: pd.DataFrame, study: BaselineStudy
) -> None:
    market_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(market_dir / "features.csv", index=False)
    study.jm.states.to_csv(market_dir / "jm-states.csv")
    study.jm.refits.to_csv(market_dir / "jm-refits.csv", index=False)
    study.hmm.states.to_csv(market_dir / "hmm-states.csv", header=True)
    study.hmm.fits.to_csv(market_dir / "hmm-fits.csv", index=False)
    study.hmm_candidates.to_csv(market_dir / "hmm-candidates.csv")
    study.boundaries.to_csv(market_dir / "boundaries.csv", index=False)
    for model_name, by_delay in study.selections.items():
        for delay, selection in by_delay.items():
            target = market_dir / f"{model_name}-delay-{delay}"
            target.mkdir(exist_ok=True)
            selection.choices.to_csv(target / "choices.csv", index=False)
            selection.surface.to_csv(target / "cv-surface.csv", index=False)
            selection.candidate_returns.to_csv(target / "candidate-returns.csv")
            selection.signal.to_csv(target / "selected-signal.csv", header=True)


def _write_cache(stem: Path, value: Any, kind: str, identity: dict[str, str]) -> None:
    try:
        checkpoint_store.save_checkpoint(stem, value, kind=kind, identity=identity)
    except checkpoint_store.CheckpointStoreError as exc:
        raise RunError(str(exc)) from exc


def _load_cache(
    stem: Path, kind: str, identity: dict[str, str], expected: type[Any]
) -> Any:
    try:
        cached = checkpoint_store.load_checkpoint(stem, kind=kind, identity=identity)
    except checkpoint_store.CheckpointStoreError as exc:
        raise RunError(str(exc)) from exc
    if cached is not None and not isinstance(cached, expected):
        raise RunError(f"invalid {kind} checkpoint payload: {stem}")
    return cached


def _load_selection(root, identity, model_name, delay):
    stem = root / f"selection-{model_name}-delay-{delay}"
    return _load_cache(stem, "selection", identity, SelectionProgress)


def _save_selection(root, identity, model_name, delay, result) -> None:
    stem = root / f"selection-{model_name}-delay-{delay}"
    _write_cache(stem, result, "selection", identity)


def _finish_run(metadata_path: Path, **updates: Any) -> None:
    metadata = _artifacts.read_json(metadata_path)
    metadata.update(updates)
    metadata["finished_at_utc"] = datetime.now(UTC).isoformat()
    _artifacts.write_json(metadata_path, metadata)


def _package_versions() -> dict[str, str]:
    packages = (
        "adaptive-jump-model numpy pandas scikit-learn scipy "
        "matplotlib hmmlearn jumpmodels"
    ).split()
    output = {}
    for package in packages:
        try:
            output[package] = version(package)
        except PackageNotFoundError:
            output[package] = "not-installed"
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="adaptive-jump")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="acquire the frozen source bundle")
    fetch.add_argument("--config", required=True, help="path to research.toml")
    run = commands.add_parser("run", help="execute a frozen research study")
    run.add_argument(
        "--study",
        required=True,
        choices=[
            "replication",
            "train-window-sensitivity",
            "persistence-calibration",
            "persistence-grid-evaluation",
            "simple-jm-suite",
            "dd-loss-scale",
        ],
    )
    run.add_argument("--config", required=True, help="path to research.toml")
    run.add_argument("--manifest", help="exact acquisition manifest path")
    verify = commands.add_parser("verify", help="verify a sealed research run")
    verify.add_argument("--run", required=True, help="path to one run directory")
    report = commands.add_parser("report", help="report a verified sealed run")
    report.add_argument("--run", required=True, help="path to one run directory")
    figures = commands.add_parser(
        "figures", help="render figures from a completed simple-JM run"
    )
    figures.add_argument("--run", required=True, help="path to one run directory")
    figures.add_argument("--output-root", help="base output directory")
    commands.add_parser("monitor").add_argument("--config", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "fetch":
            manifest = acquire(load_config(arguments.config))
            print(manifest)
            return 0
        if arguments.command == "run":
            config = load_config(arguments.config)
            observer = child_observer_from_environment()
            research = config.path.parent / "research"
            if arguments.study == "replication":
                frozen = load_frozen_data(config, arguments.manifest)
                artifact = run_replication(config, frozen, observer)
            elif arguments.manifest:
                raise RunError("--manifest is only valid for replication")
            elif arguments.study == "train-window-sensitivity":
                spec = load_window_spec(
                    research / "jm-train-window-sensitivity.toml", config
                )
                artifact = run_window_sensitivity(config, spec, observer)
            elif arguments.study == "persistence-grid-evaluation":
                spec = load_grid_spec(
                    research / "persistence-grid-evaluation.toml", config
                )
                artifact = run_grid_evaluation(config, spec, observer)
            elif arguments.study == "simple-jm-suite":
                spec = load_simple_jm_spec(
                    research / "simple-jm-suite-001.toml", config
                )
                artifact = run_simple_jm_study(config, spec, observer)
            elif arguments.study == "dd-loss-scale":
                spec = load_dd_loss_scale_spec(
                    research / "dd-loss-scale-001.toml", config
                )
                artifact = run_dd_loss_scale_study(config, spec, observer)
            else:
                artifact = run_calibration_study(
                    config, research / "persistence-calibrated-search.toml"
                )
            emit_artifact_verified(observer, _artifacts.verify_run(artifact))
            print(artifact)
            return 0
        if arguments.command == "verify":
            print(json.dumps(_artifacts.verify_run(arguments.run), sort_keys=True))
            return 0
        if arguments.command == "report":
            print(build_report(arguments.run))
            return 0
        if arguments.command == "figures":
            for output in render_figures(arguments.run, arguments.output_root):
                print(output)
            return 0
        if arguments.command == "monitor":
            from adaptive_jump.monitor.server import run_monitor_server

            return run_monitor_server(arguments.config)
    except (
        AcquisitionError,
        ChildEventError,
        ConfigError,
        RunError,
        FileNotFoundError,
        OSError,
        ValueError,
    ) as exc:
        print(f"adaptive-jump: {exc}", file=sys.stderr)
        return 2
    parser.error(f"unsupported command: {arguments.command}")
