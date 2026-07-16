"""Safe input loading and bracket orchestration for pre-OOS calibration."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from adaptive_jump.calibration import (
    CalibrationResult,
    CalibrationRules,
    calibrate_paths,
    diagnose_paths,
    generate_calibration_paths,
    jm_penalty,
    next_jm_index,
)
from adaptive_jump.config import ResearchConfig

MARKETS = ("us", "de", "jp")
FEATURE_COLUMNS = ("date", "dd_10", "sortino_20", "sortino_60", "excess_return")


class CalibrationRunError(ValueError):
    """Invalid frozen parent input or calibration run state."""


@dataclass(frozen=True)
class CalibrationSearchResult:
    """Complete pre-OOS paths, diagnostics, and attempted JM penalties."""

    paths: Mapping[str, Mapping[str, pd.DataFrame]]
    diagnostics: CalibrationResult
    attempted_jm: tuple[float, ...]


def read_pre_oos_csv(
    path: str | Path,
    exclusive_end: date,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    """Read and convert only rows strictly before one frozen OOS boundary."""
    source = Path(path)
    if not columns or columns[0] != "date" or len(set(columns)) != len(columns):
        raise CalibrationRunError("CSV columns must be unique and start with date")
    try:
        handle = source.open(encoding="utf-8", newline="")
    except OSError as exc:
        raise CalibrationRunError(f"cannot open parent input: {source}") from exc
    rows: list[dict[str, str | None]] = []
    previous: date | None = None
    reached_boundary = False
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not set(columns).issubset(reader.fieldnames):
            raise CalibrationRunError(f"parent input columns changed: {source}")
        for row in reader:
            try:
                current = date.fromisoformat(str(row["date"]))
            except (KeyError, ValueError) as exc:
                raise CalibrationRunError(f"invalid parent date: {source}") from exc
            if previous is not None and current <= previous:
                raise CalibrationRunError(f"parent dates are not increasing: {source}")
            previous = current
            if current >= exclusive_end:
                reached_boundary = True
                break
            rows.append({column: row[column] for column in columns})
    if not rows or not reached_boundary:
        raise CalibrationRunError(f"parent input does not bracket OOS: {source}")
    frame = pd.DataFrame.from_records(rows, columns=columns)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    for column in columns[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame


def load_parent_inputs(
    parent_dir: str | Path,
    rules: CalibrationRules,
    expected_inventory_sha256: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series]]:
    """Verify frozen parent files and load only each market's pre-OOS rows."""
    parent = Path(parent_dir)
    inventory_path = parent / "inventory.json"
    if _sha256(inventory_path) != expected_inventory_sha256:
        raise CalibrationRunError("parent inventory hash changed")
    try:
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CalibrationRunError("cannot read parent inventory") from exc
    files = inventory.get("files") if isinstance(inventory, dict) else None
    if not isinstance(files, dict):
        raise CalibrationRunError("parent inventory schema changed")

    frames: dict[str, pd.DataFrame] = {}
    raw_hmm: dict[str, pd.Series] = {}
    for market in MARKETS:
        feature_path = _verified_path(parent, files, f"{market}/features.csv")
        hmm_path = _verified_path(parent, files, f"{market}/hmm-states.csv")
        end = rules.exclusive_ends[market]
        frames[market] = read_pre_oos_csv(feature_path, end, FEATURE_COLUMNS)
        hmm_frame = read_pre_oos_csv(hmm_path, end, ("date", "hmm_state"))
        raw_hmm[market] = hmm_frame.set_index("date")["hmm_state"]
    return frames, raw_hmm


def run_calibration_search(
    frames: Mapping[str, pd.DataFrame],
    raw_hmm: Mapping[str, pd.Series],
    config: ResearchConfig,
    rules: CalibrationRules,
    checkpoint_dir: str | Path,
    identity: Mapping[str, str],
    *,
    workers: int | None = None,
) -> CalibrationSearchResult:
    """Expand the frozen JM upper bracket, then compress both model domains."""
    initial = (
        0.0,
        *(
            jm_penalty(j)
            for j in range(rules.jm_initial_j_min, rules.jm_initial_j_max + 1)
        ),
    )
    paths = generate_calibration_paths(
        frames,
        raw_hmm,
        config,
        rules,
        initial,
        checkpoint_dir,
        identity,
        workers=workers,
    )
    attempted = list(initial)
    while True:
        _, candidates = diagnose_paths(paths, rules)
        jm = candidates.loc[candidates["model"] == "fixed_jm"]
        validity = {
            float(row.candidate): bool(row.globally_valid)
            for row in jm.itertuples(index=False)
        }
        next_index = next_jm_index(validity, rules)
        if next_index is None:
            break
        penalty = jm_penalty(next_index)
        extra = generate_calibration_paths(
            frames,
            raw_hmm,
            config,
            rules,
            (penalty,),
            checkpoint_dir,
            identity,
            workers=workers,
        )
        for market in MARKETS:
            paths["fixed_jm"][market][penalty] = extra["fixed_jm"][market][penalty]
        attempted.append(penalty)
    return CalibrationSearchResult(
        paths=paths,
        diagnostics=calibrate_paths(paths, rules),
        attempted_jm=tuple(attempted),
    )


def _verified_path(parent: Path, files: dict, relative: str) -> Path:
    path = parent / relative
    expected = files.get(relative)
    if not isinstance(expected, str) or _sha256(path) != expected:
        raise CalibrationRunError(f"parent file hash changed: {relative}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CalibrationRunError(f"cannot hash parent input: {path}") from exc
    return digest.hexdigest()
