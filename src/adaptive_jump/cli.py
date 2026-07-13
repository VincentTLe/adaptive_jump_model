"""Command-line entry point for reproducible research workflows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd

from adaptive_jump.artifacts import (
    ArtifactError as RunError,
)
from adaptive_jump.artifacts import (
    directional_gate as _directional_gate,
)
from adaptive_jump.artifacts import (
    read_json as _read_json,
)
from adaptive_jump.artifacts import (
    sha256_file as _sha256,
)
from adaptive_jump.artifacts import (
    verify_inventory as _verify_inventory,
)
from adaptive_jump.artifacts import (
    verify_run,
)
from adaptive_jump.artifacts import (
    write_inventory as _write_inventory,
)
from adaptive_jump.artifacts import (
    write_json as _write_json,
)
from adaptive_jump.config import ConfigError, ResearchConfig, load_config
from adaptive_jump.data import AcquisitionError, acquire, research_git_sha
from adaptive_jump.features import effective_oos_start, prepare_market
from adaptive_jump.models import HMMResult, hmm_states
from adaptive_jump.reporting import build_report
from adaptive_jump.walkforward import (
    BaselineStudy,
    baseline_paths,
    build_baseline_study,
    open_baseline_metrics,
)


@dataclass(frozen=True)
class FrozenData:
    path: Path
    document: dict[str, Any]
    sha256: str


@dataclass(frozen=True)
class MarketInput:
    frame: pd.DataFrame
    oos_start: date


HMM_WORKERS, HMM_CHECKPOINT_DAYS = 16, 50


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
            document = _read_json(candidate)
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
    document = _read_json(path)
    _verify_manifest(config, document, root)
    return FrozenData(path, document, _sha256(path))


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
        if _sha256(path) != record.get("sha256"):
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


def run_replication(config: ResearchConfig, frozen: FrozenData) -> Path:
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
    metadata_path = run_dir / "run.json"
    if metadata_path.exists():
        metadata = _read_json(metadata_path)
        if any(metadata.get(key) != value for key, value in identity.items()):
            raise RunError("existing run identity does not match")
        if (
            _sha256(run_dir / "config.lock.toml") != config.sha256
            or _sha256(run_dir / "data-manifest.json") != frozen.sha256
        ):
            raise RunError("existing run locks do not match")
        if metadata.get("status") in {"complete", "boundary_failed"}:
            _verify_inventory(run_dir)
            return run_dir
    else:
        run_dir.mkdir(parents=True)
        (run_dir / "config.lock.toml").write_bytes(config.path.read_bytes())
        (run_dir / "data-manifest.json").write_bytes(frozen.path.read_bytes())
        _write_json(
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
        checkpoint = _load_checkpoint(market_dir, identity)
        if checkpoint is None:
            initial_hmm = _load_hmm_progress(market_dir, identity)
            total_hmm_days = max(
                0,
                market_input.frame["equity_log"].notna().sum()
                - config.model_protocol.fit_window
                + 1,
            )

            def save_hmm_progress(
                result: HMMResult,
                target: Path = market_dir / "hmm-progress",
                market_id: str = market.id,
                total: int = total_hmm_days,
            ) -> None:
                _write_cache(target, result, identity)
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
                checkpoint_every=HMM_CHECKPOINT_DAYS,
                progress=save_hmm_progress,
            )
            checkpoint = build_baseline_study(
                market_input.frame,
                config,
                oos_start=market_input.oos_start,
                precomputed_hmm=fitted_hmm,
            )
        elif checkpoint.oos_start != market_input.oos_start:
            raise RunError(f"{market.id}: checkpoint OOS start mismatch")
        _write_checkpoint(market_dir, market_input.frame, checkpoint, identity)
        _clear_cache(market_dir / "hmm-progress")
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
        _write_inventory(run_dir)
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
    gate = _directional_gate(metrics, config.backtest_protocol.primary_delay)
    _write_json(run_dir / "claim.json", gate)
    _write_inventory(run_dir)
    _finish_run(
        metadata_path,
        status="complete",
        metrics_opened=True,
        conclusion=gate["conclusion"],
    )
    return run_dir


def _write_checkpoint(
    market_dir: Path,
    frame: pd.DataFrame,
    study: BaselineStudy,
    identity: dict[str, str],
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
    _write_cache(market_dir / "checkpoint", study, identity)


def _load_checkpoint(
    market_dir: Path, identity: dict[str, str]
) -> BaselineStudy | None:
    cached = _load_cache(market_dir / "checkpoint", identity)
    if cached is not None and not isinstance(cached, BaselineStudy):
        raise RunError(f"invalid checkpoint payload: {market_dir}")
    return cached


def _load_hmm_progress(market_dir: Path, identity: dict[str, str]) -> HMMResult | None:
    cached = _load_cache(market_dir / "hmm-progress", identity)
    if cached is not None and not isinstance(cached, HMMResult):
        raise RunError(f"invalid HMM progress payload: {market_dir}")
    return cached


def _write_cache(stem: Path, value: Any, identity: dict[str, str]) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    payload = pickle.dumps(value, protocol=5)
    stem.with_suffix(".pkl").write_bytes(payload)
    _write_json(
        stem.with_suffix(".json"),
        {"payload_sha256": hashlib.sha256(payload).hexdigest(), **identity},
    )


def _load_cache(stem: Path, identity: dict[str, str]) -> Any:
    metadata_path = stem.with_suffix(".json")
    payload_path = stem.with_suffix(".pkl")
    if not metadata_path.exists() and not payload_path.exists():
        return None
    if not metadata_path.exists() or not payload_path.exists():
        raise RunError(f"incomplete cache: {stem}")
    metadata = _read_json(metadata_path)
    if any(metadata.get(key) != value for key, value in identity.items()):
        raise RunError(f"cache identity mismatch: {stem}")
    if _sha256(payload_path) != metadata.get("payload_sha256"):
        raise RunError(f"cache hash mismatch: {stem}")
    return pickle.loads(payload_path.read_bytes())  # noqa: S301 - local hashed cache


def _clear_cache(stem: Path) -> None:
    stem.with_suffix(".json").unlink(missing_ok=True)
    stem.with_suffix(".pkl").unlink(missing_ok=True)


def _finish_run(metadata_path: Path, **updates: Any) -> None:
    metadata = _read_json(metadata_path)
    metadata.update(updates)
    metadata["finished_at_utc"] = datetime.now(UTC).isoformat()
    _write_json(metadata_path, metadata)


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
    run.add_argument("--study", required=True, choices=["replication"])
    run.add_argument("--config", required=True, help="path to research.toml")
    run.add_argument("--manifest", help="exact acquisition manifest path")
    verify = commands.add_parser("verify", help="verify a sealed research run")
    verify.add_argument("--run", required=True, help="path to one run directory")
    report = commands.add_parser("report", help="report a verified sealed run")
    report.add_argument("--run", required=True, help="path to one run directory")
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
            artifact = run_replication(
                config, load_frozen_data(config, arguments.manifest)
            )
            print(artifact)
            return 0
        if arguments.command == "verify":
            print(json.dumps(verify_run(arguments.run), sort_keys=True))
            return 0
        if arguments.command == "report":
            print(build_report(arguments.run))
            return 0
    except (
        AcquisitionError,
        ConfigError,
        RunError,
        FileNotFoundError,
        OSError,
        ValueError,
    ) as exc:
        print(f"adaptive-jump: {exc}", file=sys.stderr)
        return 2
    parser.error(f"unsupported command: {arguments.command}")
    return 2
