"""Safe input loading and bracket orchestration for pre-OOS calibration."""

from __future__ import annotations

import csv
import json
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from time import monotonic

import pandas as pd

from adaptive_jump.artifacts import (
    read_json,
    sha256_file,
    verify_inventory,
    write_inventory,
    write_json,
)
from adaptive_jump.calibration import (
    CalibrationResult,
    CalibrationRules,
    calibrate_paths,
    diagnose_paths,
    generate_calibration_paths,
    jm_penalty,
    load_calibration_rules,
    next_jm_index,
)
from adaptive_jump.config import ResearchConfig, load_config
from adaptive_jump.data import research_git_sha

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
    if sha256_file(inventory_path) != expected_inventory_sha256:
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
    if not isinstance(expected, str) or sha256_file(path) != expected:
        raise CalibrationRunError(f"parent file hash changed: {relative}")
    return path


def run_calibration_study(
    config: ResearchConfig,
    spec_path: str | Path,
    *,
    workers: int | None = None,
) -> Path:
    """Run or resume the frozen pre-OOS persistence calibration."""
    spec = Path(spec_path).resolve()
    rules = load_calibration_rules(spec, config)
    document = _study_document(spec)
    root = config.path.parent
    parent_id = _field(document, "parent", "run_id")
    parent_hash = _field(document, "parent", "run_inventory_sha256")
    data_hash = _field(document, "parent", "data_manifest_sha256")
    parent_dir = root / config.artifact_root / "fixed-baselines" / parent_id
    _verify_parent(root, parent_dir, config, document)

    git_sha = research_git_sha(root)
    identity = {
        "spec_sha256": rules.sha256,
        "config_sha256": config.sha256,
        "parent_inventory_sha256": parent_hash,
        "data_manifest_sha256": data_hash,
        "git_sha": git_sha,
    }
    run_id = "persistence-calibration-" + "-".join(
        identity[key][:12]
        for key in ("spec_sha256", "parent_inventory_sha256", "git_sha")
    )
    subdir = _field(document, "storage", "artifact_subdir")
    run_dir = root / config.artifact_root / subdir / run_id
    checkpoint_dir = root / config.artifact_root / ".monitor/checkpoints" / run_id
    metadata_path = run_dir / "run.json"
    if metadata_path.exists():
        metadata = read_json(metadata_path)
        if any(metadata.get(key) != value for key, value in identity.items()):
            raise CalibrationRunError("existing calibration identity changed")
        _verify_run_locks(run_dir, config, rules, data_hash, parent_hash)
        if metadata.get("status") == "complete":
            verify_calibration_run(run_dir)
            return run_dir
    else:
        run_dir.mkdir(parents=True)
        (run_dir / "study.lock.toml").write_bytes(spec.read_bytes())
        (run_dir / "config.lock.toml").write_bytes(config.path.read_bytes())
        (run_dir / "data-manifest.json").write_bytes(
            (parent_dir / "data-manifest.json").read_bytes()
        )
        write_json(
            metadata_path,
            {
                "schema_version": 1,
                "study_kind": "persistence_calibration",
                "experiment_id": document["experiment_id"],
                "run_id": run_id,
                "status": "running",
                "claim_class": "EXPLORATORY",
                "metrics_opened": False,
                "post_2023_accessed": False,
                "workers": rules.process_workers if workers is None else workers,
                "created_at_utc": datetime.now(UTC).isoformat(),
                "command": (
                    "adaptive-jump run --study persistence-calibration "
                    f"--config {config.path}"
                ),
                **identity,
            },
        )

    started = monotonic()
    frames, raw_hmm = load_parent_inputs(parent_dir, rules, parent_hash)
    result = run_calibration_search(
        frames,
        raw_hmm,
        config,
        rules,
        checkpoint_dir,
        {"code_sha": git_sha, "data_sha256": data_hash},
        workers=workers,
    )
    states_dir = run_dir / "states"
    states_dir.mkdir(exist_ok=True)
    for model in ("fixed_jm", "hmm"):
        for market in MARKETS:
            result.paths[model][market].to_csv(
                states_dir / f"{model}-{market}.csv", index_label="date"
            )
    result.diagnostics.market_diagnostics.to_csv(
        run_dir / "market-diagnostics.csv", index=False
    )
    result.diagnostics.candidate_diagnostics.to_csv(
        run_dir / "candidate-diagnostics.csv", index=False
    )
    write_json(
        run_dir / "selection.json",
        {
            "attempted_jm": list(result.attempted_jm),
            "selected_grids": {
                model: list(grid) for model, grid in result.diagnostics.grids.items()
            },
            "common_budget": len(result.diagnostics.grids["fixed_jm"]),
        },
    )
    write_inventory(run_dir)
    metadata = read_json(metadata_path)
    metadata.update(
        {
            "status": "complete",
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "runtime_seconds": monotonic() - started,
        }
    )
    write_json(metadata_path, metadata)
    return run_dir


def verify_calibration_run(run: str | Path) -> dict[str, object]:
    """Independently recompute behavior diagnostics from sealed state paths."""
    run_dir = Path(run).resolve()
    if not run_dir.is_dir():
        raise CalibrationRunError(f"run directory does not exist: {run_dir}")
    verify_inventory(run_dir)
    metadata = read_json(run_dir / "run.json")
    if (
        metadata.get("schema_version") != 1
        or metadata.get("study_kind") != "persistence_calibration"
        or metadata.get("status") != "complete"
        or metadata.get("claim_class") != "EXPLORATORY"
        or metadata.get("metrics_opened") is not False
        or metadata.get("post_2023_accessed") is not False
    ):
        raise CalibrationRunError("invalid calibration run metadata")
    config = load_config(run_dir / "config.lock.toml")
    rules = load_calibration_rules(run_dir / "study.lock.toml", config)
    document = _study_document(run_dir / "study.lock.toml")
    root = run_dir.parents[2]
    parent_hash = _field(document, "parent", "run_inventory_sha256")
    data_hash = _field(document, "parent", "data_manifest_sha256")
    parent_dir = (
        root
        / config.artifact_root
        / "fixed-baselines"
        / _field(document, "parent", "run_id")
    )
    _verify_parent(root, parent_dir, config, document)
    git_sha = str(metadata.get("git_sha", ""))
    expected_id = "persistence-calibration-" + "-".join(
        value[:12] for value in (rules.sha256, parent_hash, git_sha)
    )
    identity = {
        "spec_sha256": rules.sha256,
        "config_sha256": config.sha256,
        "parent_inventory_sha256": parent_hash,
        "data_manifest_sha256": data_hash,
    }
    if (
        len(git_sha) != 40
        or any(metadata.get(key) != value for key, value in identity.items())
        or metadata.get("experiment_id") != document.get("experiment_id")
        or metadata.get("run_id") != expected_id
        or run_dir.name != expected_id
    ):
        raise CalibrationRunError("calibration run identity is inconsistent")
    _verify_run_locks(run_dir, config, rules, data_hash, parent_hash)
    forbidden = ("metrics.csv", "claim.json")
    if any((run_dir / name).exists() for name in forbidden) or list(
        run_dir.rglob("trades")
    ):
        raise CalibrationRunError("calibration artifact contains performance output")

    paths = _read_paths(run_dir, rules)
    result = calibrate_paths(paths, rules)
    try:
        for name, expected in (
            ("market-diagnostics.csv", result.market_diagnostics),
            ("candidate-diagnostics.csv", result.candidate_diagnostics),
        ):
            pd.testing.assert_frame_equal(
                pd.read_csv(run_dir / name),
                expected,
                check_dtype=False,
                check_exact=False,
                rtol=1e-12,
                atol=1e-12,
            )
    except AssertionError as exc:
        raise CalibrationRunError(
            "stored diagnostics do not match recomputation"
        ) from exc
    attempted = tuple(float(value) for value in paths["fixed_jm"]["us"].columns)
    validity = {
        float(row.candidate): bool(row.globally_valid)
        for row in result.candidate_diagnostics.loc[
            result.candidate_diagnostics["model"] == "fixed_jm"
        ].itertuples(index=False)
    }
    if next_jm_index(validity, rules) is not None:
        raise CalibrationRunError("JM upper bracket did not reach its frozen stop")
    expected_selection = {
        "attempted_jm": list(attempted),
        "selected_grids": {model: list(grid) for model, grid in result.grids.items()},
        "common_budget": len(result.grids["fixed_jm"]),
    }
    if read_json(run_dir / "selection.json") != expected_selection:
        raise CalibrationRunError("stored calibration selection is inconsistent")
    return {
        "schema_version": 1,
        "study_kind": "persistence_calibration",
        "run_id": expected_id,
        "status": "complete",
        "attempted_jm": len(attempted),
        "attempted_hmm": len(paths["hmm"]["us"].columns),
        "selected_budget": len(result.grids["fixed_jm"]),
        "metrics_opened": False,
    }


def _read_paths(
    run_dir: Path, rules: CalibrationRules
) -> dict[str, dict[str, pd.DataFrame]]:
    target = run_dir / "states"
    expected = {
        target / f"{model}-{market}.csv"
        for model in ("fixed_jm", "hmm")
        for market in MARKETS
    }
    if set(target.glob("*.csv")) != expected:
        raise CalibrationRunError("calibration state-file coverage is invalid")
    paths: dict[str, dict[str, pd.DataFrame]] = {"fixed_jm": {}, "hmm": {}}
    for model in paths:
        for market in MARKETS:
            frame = pd.read_csv(target / f"{model}-{market}.csv")
            if frame.empty or frame.columns[0] != "date":
                raise CalibrationRunError("calibration state schema is invalid")
            dates = pd.DatetimeIndex(pd.to_datetime(frame.pop("date"), errors="raise"))
            frame.columns = [float(value) for value in frame.columns]
            frame = frame.apply(pd.to_numeric, errors="raise").set_axis(dates)
            values = frame.stack().to_numpy(dtype=float)
            if (
                dates.has_duplicates
                or not dates.is_monotonic_increasing
                or (dates.date >= rules.exclusive_ends[market]).any()
                or not values.size
                or not pd.Series(values).isin((0.0, 1.0)).all()
            ):
                raise CalibrationRunError("calibration states violate pre-OOS rules")
            paths[model][market] = frame
    hmm = tuple(paths["hmm"]["us"].columns)
    expected_hmm = tuple(
        float(value)
        for value in range(rules.hmm_k_min, rules.hmm_k_max + 1, rules.hmm_k_step)
    )
    if hmm != expected_hmm:
        raise CalibrationRunError("HMM calibration path is incomplete")
    return paths


def _study_document(path: Path) -> dict[str, object]:
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise CalibrationRunError("cannot read calibration study lock") from exc
    return document


def _field(document: dict[str, object], table: str, key: str) -> str:
    section = document.get(table)
    value = section.get(key) if isinstance(section, dict) else None
    if not isinstance(value, str) or not value:
        raise CalibrationRunError(f"missing calibration field: {table}.{key}")
    return value


def _verify_parent(
    root: Path,
    parent_dir: Path,
    config: ResearchConfig,
    document: dict[str, object],
) -> None:
    checks = {
        parent_dir / "inventory.json": _field(
            document, "parent", "run_inventory_sha256"
        ),
        parent_dir / "config.lock.toml": config.sha256,
        parent_dir / "data-manifest.json": _field(
            document, "parent", "data_manifest_sha256"
        ),
        root / "2402.05272v3.pdf": _field(document, "source", "paper_sha256"),
    }
    if any(sha256_file(path) != expected for path, expected in checks.items()):
        raise CalibrationRunError("frozen calibration lineage hash changed")


def _verify_run_locks(
    run_dir: Path,
    config: ResearchConfig,
    rules: CalibrationRules,
    data_hash: str,
    parent_hash: str,
) -> None:
    checks = {
        run_dir / "study.lock.toml": rules.sha256,
        run_dir / "config.lock.toml": config.sha256,
        run_dir / "data-manifest.json": data_hash,
    }
    if any(sha256_file(path) != expected for path, expected in checks.items()):
        raise CalibrationRunError("calibration run lock changed")
