"""Command-line entry point for reproducible research workflows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from adaptive_jump.config import ConfigError, ResearchConfig, load_config
from adaptive_jump.data import AcquisitionError, acquire
from adaptive_jump.features import effective_oos_start, prepare_market


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
        raise RunError(f"cannot read manifest {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise RunError(f"manifest must contain an object: {path}")
    return document


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="adaptive-jump")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="acquire the frozen source bundle")
    fetch.add_argument("--config", required=True, help="path to research.toml")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "fetch":
            manifest = acquire(load_config(arguments.config))
            print(manifest)
            return 0
    except (AcquisitionError, ConfigError, FileNotFoundError, OSError) as exc:
        print(f"adaptive-jump: {exc}", file=sys.stderr)
        return 2
    parser.error(f"unsupported command: {arguments.command}")
    return 2
