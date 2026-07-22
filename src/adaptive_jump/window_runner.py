"""Artifact-producing runner for the frozen JM-window sensitivity."""

from __future__ import annotations

import subprocess
from datetime import UTC, date, datetime
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd

from adaptive_jump.artifacts import (
    ArtifactError,
    finish_run_metadata,
    read_json,
    sha256_file,
    verify_run,
    write_inventory,
    write_json,
)
from adaptive_jump.backtest import apply_signal
from adaptive_jump.config import ResearchConfig
from adaptive_jump.data import research_git_sha
from adaptive_jump.features import effective_oos_start
from adaptive_jump.inference import BootstrapProgress
from adaptive_jump.models import FixedJMResult
from adaptive_jump.runtime import checkpoints as checkpoint_store
from adaptive_jump.runtime import study_runtime
from adaptive_jump.runtime.events import EventObserver, bind_event_context
from adaptive_jump.walkforward import SelectionProgress
from adaptive_jump.window_spec import WindowStudySpec
from adaptive_jump.window_study import (
    COMPARISON_MODELS,
    WindowMarketStudy,
    align_comparison_paths,
    bootstrap_rows,
    build_window_market_study,
    comparison_metrics,
    window_claim,
)

CONTROL_SCOPE = (
    "research.toml",
    "src/adaptive_jump/config.py",
    "src/adaptive_jump/features.py",
    "src/adaptive_jump/models.py",
    "src/adaptive_jump/walkforward.py",
    "src/adaptive_jump/backtest.py",
)


def run_window_sensitivity(
    config: ResearchConfig,
    spec: WindowStudySpec,
    observer: EventObserver | None = None,
) -> Path:
    """Run JM-4000 against exact sealed v7 controls through 2023."""
    root = config.path.parent
    parent_dir = root / config.artifact_root / "fixed-baselines" / spec.parent_run_id
    parent_receipt, parent_metadata = _verify_parent(parent_dir, config, spec)
    git_sha = research_git_sha(root)
    _verify_control_source(root, str(parent_metadata["git_sha"]), git_sha)
    identity = {
        "spec_sha256": spec.sha256,
        "config_sha256": config.sha256,
        "data_manifest_sha256": spec.data_manifest_sha256,
        "parent_inventory_sha256": spec.parent_inventory_sha256,
        "git_sha": git_sha,
    }
    run_id = "jm-window-" + "-".join(
        identity[key][:12] for key in ("spec_sha256", "data_manifest_sha256", "git_sha")
    )
    run_dir = root / config.artifact_root / spec.artifact_subdir / run_id
    checkpoint_root = root / config.artifact_root / ".monitor" / "checkpoints" / run_id
    metadata_path = run_dir / "run.json"
    if metadata_path.exists():
        metadata = read_json(metadata_path)
        if any(metadata.get(key) != value for key, value in identity.items()):
            raise ArtifactError("existing JM-window run identity does not match")
        _verify_run_locks(run_dir, config, spec)
        if metadata.get("status") in {"complete", "boundary_failed"}:
            from adaptive_jump.window_verifier import verify_window_run

            verify_window_run(run_dir)
            checkpoint_store.clear_checkpoint_tree(checkpoint_root)
            return run_dir
    else:
        _create_run(
            run_dir,
            config,
            spec,
            parent_dir,
            parent_receipt,
            parent_metadata,
            run_id,
            identity,
        )

    studies: dict[str, WindowMarketStudy] = {}
    frames: dict[str, pd.DataFrame] = {}
    for market in config.markets:
        frame = _read_parent_features(parent_dir, market.id, spec.data_cutoff)
        oos_start = effective_oos_start(
            frame,
            requested=date.fromisoformat(config.document["oos_start"]["requested"]),
            fit_window=spec.challenger_window,
            validation_years=config.selection_protocol.validation_years,
        )
        if oos_start is None:
            raise ArtifactError(f"{market.id}: JM-4000 has no eligible OOS sample")
        market_checkpoints = checkpoint_root / market.id
        study = build_window_market_study(
            frame,
            config,
            spec,
            oos_start=oos_start,
            jm_initial=_load_checkpoint(
                market_checkpoints / "jm-progress",
                "fixed_jm",
                identity,
                FixedJMResult,
            ),
            jm_progress=partial(
                _save_checkpoint,
                market_checkpoints / "jm-progress",
                "fixed_jm",
                identity,
            ),
            selection_initial=partial(_load_selection, market_checkpoints, identity),
            selection_progress=partial(_save_selection, market_checkpoints, identity),
            observer=study_runtime.model_observer(
                observer, market.id, "jm_4000", frame
            ),
        )
        frames[market.id] = frame
        studies[market.id] = study
        study_runtime.emit_boundary_rows(observer, study.boundaries, market.id)
        _write_market_evidence(run_dir / market.id, study)

    boundaries = pd.concat(
        [
            study.boundaries.assign(market=market_id)
            for market_id, study in studies.items()
        ],
        ignore_index=True,
    )
    boundaries.to_csv(run_dir / "boundaries.csv", index=False)
    if not boundaries["passed"].all():
        write_inventory(run_dir)
        finish_run_metadata(
            metadata_path,
            status="boundary_failed",
            metrics_opened=False,
            conclusion="JM-4000 upper-lambda boundary requires a new experiment",
        )
        checkpoint_store.clear_checkpoint_tree(checkpoint_root)
        return run_dir

    metric_frames = []
    bootstrap_frames = []
    for market in config.markets:
        market_id = market.id
        frame = frames[market_id]
        returns = frame[["date", "equity_simple", "cash_return"]]
        for delay in spec.delays:
            selection = studies[market_id].selections[delay]
            challenger = apply_signal(
                returns,
                selection.signal.reset_index(drop=True),
                delay_trading_days=delay,
                one_way_cost_bps=config.backtest_protocol.one_way_cost_bps,
                charge_initial_allocation=(
                    config.backtest_protocol.charge_initial_allocation
                ),
            )
            parent_paths = _read_parent_paths(parent_dir, market_id, delay)
            paths = align_comparison_paths(
                {
                    "buy_and_hold": parent_paths["buy_and_hold"],
                    "hmm_3000": parent_paths["hmm"],
                    "jm_3000": parent_paths["fixed_jm"],
                    "jm_4000": challenger,
                },
                oos_start=studies[market_id].oos_start,
            )
            _write_trade_paths(run_dir / market_id / "trades", delay, paths)
            metric_frames.append(
                comparison_metrics(paths, config).assign(market=market_id, delay=delay)
            )
            if delay == spec.primary_delay:
                bootstrap_observer = bind_event_context(
                    observer,
                    market=market_id,
                    model="jm_4000_vs_jm_3000",
                    delay=delay,
                )
                bootstrap_frames.append(
                    bootstrap_rows(
                        paths,
                        spec,
                        config,
                        initial=partial(
                            _load_bootstrap,
                            checkpoint_root,
                            identity,
                            market_id,
                        ),
                        progress=study_runtime.bootstrap_recorder(
                            partial(
                                _save_bootstrap,
                                checkpoint_root,
                                identity,
                                market_id,
                            ),
                            bootstrap_observer,
                            spec.bootstrap_replications,
                        ),
                    ).assign(market=market_id)
                )

    metrics = pd.concat(metric_frames, ignore_index=True)
    bootstrap = pd.concat(bootstrap_frames, ignore_index=True)
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    bootstrap.to_csv(run_dir / "bootstrap.csv", index=False)
    claim = window_claim(
        pd.read_csv(run_dir / "metrics.csv"),
        pd.read_csv(run_dir / "bootstrap.csv"),
        market_ids=tuple(market.id for market in config.markets),
        primary_delay=spec.primary_delay,
        primary_block=spec.bootstrap_blocks[0],
    )
    write_json(run_dir / "claim.json", claim)
    write_inventory(run_dir)
    finish_run_metadata(
        metadata_path,
        status="complete",
        metrics_opened=True,
        conclusion=claim["conclusion"],
    )
    checkpoint_store.clear_checkpoint_tree(checkpoint_root)
    return run_dir


def _verify_parent(
    parent_dir: Path, config: ResearchConfig, spec: WindowStudySpec
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = verify_run(parent_dir)
    metadata = read_json(parent_dir / "run.json")
    if (
        receipt.get("status") != "complete"
        or receipt.get("run_id") != spec.parent_run_id
        or metadata.get("config_sha256") != config.sha256
        or metadata.get("data_manifest_sha256") != spec.data_manifest_sha256
        or sha256_file(parent_dir / "inventory.json") != spec.parent_inventory_sha256
    ):
        raise ArtifactError("sealed v7 parent does not match the window contract")
    return receipt, metadata


def _verify_control_source(root: Path, parent_sha: str, current_sha: str) -> None:
    result = subprocess.run(
        ["git", "diff", "--quiet", parent_sha, current_sha, "--", *CONTROL_SCOPE],
        cwd=root,
        check=False,
    )
    if result.returncode != 0:
        raise ArtifactError("v7 control implementation changed after its sealed run")


def _create_run(
    run_dir: Path,
    config: ResearchConfig,
    spec: WindowStudySpec,
    parent_dir: Path,
    parent_receipt: dict[str, Any],
    parent_metadata: dict[str, Any],
    run_id: str,
    identity: dict[str, str],
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
            "parent_git_sha": parent_metadata["git_sha"],
            "control_scope": list(CONTROL_SCOPE),
            "control_source_unchanged": True,
        },
    )
    write_json(
        run_dir / "run.json",
        {
            "schema_version": 1,
            "study_kind": "jm_train_window_sensitivity",
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


def _verify_run_locks(
    run_dir: Path, config: ResearchConfig, spec: WindowStudySpec
) -> None:
    if (
        sha256_file(run_dir / "study.lock.toml") != spec.sha256
        or sha256_file(run_dir / "config.lock.toml") != config.sha256
        or sha256_file(run_dir / "data-manifest.json") != spec.data_manifest_sha256
    ):
        raise ArtifactError("existing JM-window run locks do not match")


def _read_parent_features(parent_dir: Path, market: str, cutoff: date) -> pd.DataFrame:
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
        or dates.max().date() > cutoff
    ):
        raise ArtifactError(f"{market}: parent feature dates violate the cutoff")
    frame["date"] = dates
    return frame


def _read_parent_paths(
    parent_dir: Path, market: str, delay: int
) -> dict[str, pd.DataFrame]:
    return {
        model: pd.read_csv(
            parent_dir / market / "trades" / f"{model}-delay-{delay}.csv"
        )
        for model in ("buy_and_hold", "hmm", "fixed_jm")
    }


def _write_market_evidence(market_dir: Path, study: WindowMarketStudy) -> None:
    market_dir.mkdir(parents=True, exist_ok=True)
    study.jm.states.to_csv(market_dir / "jm-4000-states.csv")
    study.jm.refits.to_csv(market_dir / "jm-4000-refits.csv", index=False)
    study.boundaries.to_csv(market_dir / "boundaries.csv", index=False)
    for delay, selection in study.selections.items():
        target = market_dir / f"jm-4000-delay-{delay}"
        target.mkdir(exist_ok=True)
        selection.choices.to_csv(target / "choices.csv", index=False)
        selection.surface.to_csv(target / "cv-surface.csv", index=False)
        selection.candidate_returns.to_csv(target / "candidate-returns.csv")
        selection.signal.to_csv(target / "selected-signal.csv", header=True)


def _write_trade_paths(
    target: Path, delay: int, paths: dict[str, pd.DataFrame]
) -> None:
    if tuple(paths) != COMPARISON_MODELS:
        raise ArtifactError("cannot write incomplete comparison paths")
    target.mkdir(exist_ok=True)
    for model, path in paths.items():
        path.to_csv(target / f"{model}-delay-{delay}.csv", index=False)


def _load_checkpoint(stem, kind, identity, expected):
    try:
        cached = checkpoint_store.load_checkpoint(stem, kind=kind, identity=identity)
    except checkpoint_store.CheckpointStoreError as exc:
        raise ArtifactError(str(exc)) from exc
    if cached is not None and not isinstance(cached, expected):
        raise ArtifactError(f"invalid {kind} checkpoint payload: {stem}")
    return cached


def _save_checkpoint(stem, kind, identity, result) -> None:
    try:
        checkpoint_store.save_checkpoint(stem, result, kind=kind, identity=identity)
    except checkpoint_store.CheckpointStoreError as exc:
        raise ArtifactError(str(exc)) from exc


def _load_selection(root, identity, delay):
    return _load_checkpoint(
        root / f"selection-delay-{delay}", "selection", identity, SelectionProgress
    )


def _save_selection(root, identity, delay, result) -> None:
    _save_checkpoint(root / f"selection-delay-{delay}", "selection", identity, result)


def _load_bootstrap(root, identity, market, block):
    return _load_checkpoint(
        root / f"bootstrap-{market}-block-{block}",
        "bootstrap",
        identity,
        BootstrapProgress,
    )


def _save_bootstrap(root, identity, market, block, result) -> None:
    _save_checkpoint(
        root / f"bootstrap-{market}-block-{block}", "bootstrap", identity, result
    )


def _package_versions() -> dict[str, str]:
    output = {}
    for package in (
        "adaptive-jump-model",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "jumpmodels",
    ):
        try:
            output[package] = version(package)
        except PackageNotFoundError:
            output[package] = "not-installed"
    return output
