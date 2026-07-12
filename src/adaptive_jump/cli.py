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

from adaptive_jump.config import ConfigError, ResearchConfig, load_config
from adaptive_jump.data import AcquisitionError, acquire, research_git_sha
from adaptive_jump.features import effective_oos_start, prepare_market
from adaptive_jump.walkforward import (
    BaselineStudy,
    baseline_paths,
    build_baseline_study,
    open_baseline_metrics,
)


class RunError(RuntimeError):
    """Raised when frozen study inputs or run artifacts are invalid."""


@dataclass(frozen=True)
class FrozenData:
    path: Path
    document: dict[str, Any]
    sha256: str


@dataclass(frozen=True)
class MarketInput:
    frame: pd.DataFrame
    oos_start: date


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


def _read_json(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        raise RunError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise RunError(f"manifest must contain an object: {path}")
    return document


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
            checkpoint = build_baseline_study(
                market_input.frame, config, oos_start=market_input.oos_start
            )
        elif checkpoint.oos_start != market_input.oos_start:
            raise RunError(f"{market.id}: checkpoint OOS start mismatch")
        _write_checkpoint(market_dir, market_input.frame, checkpoint, identity)
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
    payload = pickle.dumps(study, protocol=5)
    payload_path = market_dir / "checkpoint.pkl"
    payload_path.write_bytes(payload)
    _write_json(
        market_dir / "checkpoint.json",
        {
            "schema_version": 1,
            "oos_start": study.oos_start.isoformat(),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            **identity,
        },
    )


def _load_checkpoint(
    market_dir: Path, identity: dict[str, str]
) -> BaselineStudy | None:
    metadata_path = market_dir / "checkpoint.json"
    payload_path = market_dir / "checkpoint.pkl"
    if not metadata_path.exists() and not payload_path.exists():
        return None
    if not metadata_path.exists() or not payload_path.exists():
        raise RunError(f"incomplete checkpoint: {market_dir}")
    metadata = _read_json(metadata_path)
    if any(metadata.get(key) != value for key, value in identity.items()):
        raise RunError(f"checkpoint identity mismatch: {market_dir}")
    if _sha256(payload_path) != metadata.get("payload_sha256"):
        raise RunError(f"checkpoint hash mismatch: {market_dir}")
    study = pickle.loads(payload_path.read_bytes())  # noqa: S301 - local hashed cache
    if not isinstance(study, BaselineStudy):
        raise RunError(f"invalid checkpoint payload: {market_dir}")
    return study


def _directional_gate(metrics: pd.DataFrame, primary_delay: int) -> dict[str, Any]:
    rows = []
    primary = metrics.loc[metrics["delay"] == primary_delay]
    for market, values in primary.groupby("market"):
        indexed = values.set_index("model")
        required = {"fixed_jm", "hmm", "buy_and_hold"}
        if set(indexed.index) != required:
            raise RunError(f"{market}: incomplete primary metrics")
        jm = indexed.loc["fixed_jm"]
        hmm = indexed.loc["hmm"]
        hold = indexed.loc["buy_and_hold"]
        checks = {
            "sharpe_above_hmm": bool(jm["sharpe"] > hmm["sharpe"]),
            "sharpe_above_buy_and_hold": bool(jm["sharpe"] > hold["sharpe"]),
            "mdd_below_buy_and_hold": bool(
                abs(jm["maximum_drawdown"]) < abs(hold["maximum_drawdown"])
            ),
        }
        rows.append({"market": market, **checks, "passed": all(checks.values())})
    passed = len(rows) == 3 and all(row["passed"] for row in rows)
    return {
        "claim_label": "proxy replication",
        "primary_delay": primary_delay,
        "markets": rows,
        "passed": passed,
        "conclusion": (
            "directional proxy replication"
            if passed
            else "non-replication; adaptive work remains blocked"
        ),
    }


def _finish_run(metadata_path: Path, **updates: Any) -> None:
    metadata = _read_json(metadata_path)
    metadata.update(updates)
    metadata["finished_at_utc"] = datetime.now(UTC).isoformat()
    _write_json(metadata_path, metadata)


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_inventory(run_dir: Path) -> None:
    files = {
        str(path.relative_to(run_dir)): _sha256(path)
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name not in {"inventory.json", "run.json"}
    }
    _write_json(run_dir / "inventory.json", {"schema_version": 1, "files": files})


def _verify_inventory(run_dir: Path) -> None:
    inventory = _read_json(run_dir / "inventory.json")
    expected = inventory.get("files")
    if not isinstance(expected, dict):
        raise RunError("invalid artifact inventory")
    actual = {
        str(path.relative_to(run_dir)): _sha256(path)
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name not in {"inventory.json", "run.json"}
    }
    if actual != expected:
        raise RunError("artifact inventory mismatch")


def _package_versions() -> dict[str, str]:
    packages = (
        "adaptive-jump-model",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "matplotlib",
        "hmmlearn",
        "jumpmodels",
    )
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
