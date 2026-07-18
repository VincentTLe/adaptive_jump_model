"""Independent artifact replay for the one-shot endpoint-grid audit."""

from __future__ import annotations

import ast
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from adaptive_jump.artifacts import (
    TRADE_COLUMNS,
    ArtifactError,
    read_json,
    read_trade_path,
    sha256_file,
    verify_inventory,
)
from adaptive_jump.config import PAPER_TURNOVER_DEFINITION, load_config
from adaptive_jump.data import research_git_sha
from adaptive_jump.endpoint_grid_replay import (
    REPLAY_CELL_PATHS,
    REPLAY_MARKETS,
    REPLAY_PATHS,
    REPLAY_SELECTION_PATHS,
    ReplayMarket,
    ReplayPrepared,
    refit_current_jm,
    replay_behavior_control,
    replay_d_rescue_decision,
    replay_endpoint_effects,
    replay_finalize_markets,
    replay_prepare_market,
    replay_us_smoke,
    verify_replay_smoke_prefix,
)
from adaptive_jump.endpoint_grid_replay_evidence import replay_path_changes
from adaptive_jump.endpoint_grid_verifier import (
    load_endpoint_grid_spec,
    load_market_source,
    verify_lineage,
)
from adaptive_jump.models import FEATURE_COLUMNS, FixedJMResult

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
ROOT_FILES = set(
    (
        "behavior-control.json boundaries.csv change-traces.csv composition.json "
        "config.lock.toml decision.json endpoint-effects.csv "
        "endpoint-provenance.json inventory.json metrics.csv path-changes.csv "
        "run.json study.lock.toml us-smoke.json"
    ).split()
)
RUN_KEYS = {
    "schema_version",
    "study_kind",
    "experiment_id",
    "run_id",
    "status",
    "claim_class",
    "performance_claim_allowed",
    "post_2023_accessed",
    "boundary_descriptive_only",
    "created_at_utc",
    "finished_at_utc",
    "spec_sha256",
    "config_sha256",
    "calibration_inventory_sha256",
    "data_manifest_sha256",
    "git_sha",
    "execution",
}


def verify_endpoint_grid_run(run: str | Path) -> dict[str, Any]:
    """Replay the two-phase run and reject anything outside its allowlist."""
    supplied = Path(run)
    if supplied.is_symlink():
        raise ArtifactError("endpoint-grid run cannot be a symlink")
    run_dir = supplied.resolve()
    _reject_symlinks(run_dir)
    metadata = read_json(run_dir / "run.json")
    if set(metadata) != RUN_KEYS:
        raise ArtifactError("endpoint-grid run metadata keys changed")
    if (
        metadata.get("schema_version") != 1
        or metadata.get("study_kind") != "endpoint_grid_audit"
        or metadata.get("status") != "complete"
        or metadata.get("claim_class") != "EXPLORATORY"
        or metadata.get("performance_claim_allowed") is not False
        or metadata.get("post_2023_accessed") is not False
        or metadata.get("boundary_descriptive_only") is not True
    ):
        raise ArtifactError("endpoint-grid metadata is invalid")
    _verify_timestamps(metadata)
    _verify_inventory_schema(run_dir)
    verify_inventory(run_dir)
    config = load_config(run_dir / "config.lock.toml")
    spec = load_endpoint_grid_spec(run_dir / "study.lock.toml", config)
    if spec.protocol_status != "FROZEN":
        raise ArtifactError("completed endpoint-grid run used a draft contract")
    _verify_allowlist(run_dir, config.backtest_protocol.robustness_delays)
    expected_execution = {
        "process_start_method": spec.process_start_method,
        "market_workers": spec.market_workers,
        "numerical_threads": spec.numerical_threads,
    }
    if metadata.get("execution") != expected_execution:
        raise ArtifactError("endpoint-grid execution metadata changed")
    git_sha = _git_sha(metadata.get("git_sha"))
    expected_id = "endpoint-grid-audit-" + "-".join(
        (spec.sha256[:12], spec.calibration_inventory_sha256[:12], git_sha[:12])
    )
    identity = {
        "run_id": expected_id,
        "experiment_id": spec.experiment_id,
        "spec_sha256": spec.sha256,
        "config_sha256": config.sha256,
        "calibration_inventory_sha256": spec.calibration_inventory_sha256,
        "data_manifest_sha256": spec.data_manifest_sha256,
    }
    if run_dir.name != expected_id or any(
        metadata.get(key) != value for key, value in identity.items()
    ):
        raise ArtifactError("endpoint-grid identity is inconsistent")
    root = _repository_root(run_dir, config, spec, expected_id)
    if research_git_sha(root) != git_sha:
        raise ArtifactError(
            "stored endpoint-grid Git SHA is not current repository SHA"
        )
    if (
        sha256_file(run_dir / "study.lock.toml") != spec.sha256
        or sha256_file(run_dir / "config.lock.toml") != config.sha256
        or sha256_file(root / "research.toml") != config.sha256
    ):
        raise ArtifactError("endpoint-grid locks changed")
    source_config = replace(config, path=root / "research.toml")
    lineage = verify_lineage(source_config, spec)
    evaluated = replace(
        source_config,
        metrics_protocol=replace(
            config.metrics_protocol,
            turnover_definition=PAPER_TURNOVER_DEFINITION,
        ),
    )
    if read_json(run_dir / "endpoint-provenance.json") != lineage.endpoints.as_dict():
        raise ArtifactError("derived endpoint provenance changed")
    sources = {
        market: load_market_source(lineage.parent_dir, market, evaluated, lineage)
        for market in REPLAY_MARKETS
    }
    smoke = replay_us_smoke(
        sources["us"],
        evaluated,
        lineage.endpoints,
        spec.smoke_terminal_dates,
        numerical_threads=spec.numerical_threads,
    )
    if read_json(run_dir / "us-smoke.json") != smoke:
        raise ArtifactError("US performance-free smoke does not replay")

    prepared: dict[str, ReplayPrepared] = {}
    fits: dict[str, FixedJMResult] = {}
    for market in REPLAY_MARKETS:
        fit = refit_current_jm(
            sources[market],
            evaluated,
            lineage.endpoints,
            lineage.jm_grid,
            numerical_threads=spec.numerical_threads,
        )
        fits[market] = fit
        item = replay_prepare_market(
            sources[market],
            evaluated,
            lineage.endpoints,
            lineage.jm_grid,
            lineage.hmm_grid,
            lineage.base_dir / market,
            git_sha,
            fit,
        )
        prepared[market] = item
        _verify_endpoint_fit(
            run_dir / market,
            FixedJMResult(item.endpoint_jm, item.endpoint_refits),
            lineage.endpoints.jm_endpoint,
        )
        _verify_selection_evidence(run_dir / market, item)
    expected_control = replay_behavior_control(prepared)
    if read_json(run_dir / "behavior-control.json") != expected_control:
        raise ArtifactError("selection-behavior parity receipt changed")
    verify_replay_smoke_prefix(
        sources["us"],
        fits["us"],
        lineage.endpoints,
        smoke,
        spec.smoke_terminal_dates,
    )

    rebuilt = replay_finalize_markets(prepared, evaluated)
    metrics, boundaries, changes, traces = [], [], [], []
    for market in REPLAY_MARKETS:
        _verify_market_paths(run_dir / market, rebuilt[market], evaluated)
        metrics.append(rebuilt[market].metrics.assign(market=market))
        boundaries.append(rebuilt[market].boundaries.assign(market=market))
        change, trace = replay_path_changes(
            rebuilt[market].selections,
            rebuilt[market].paths,
            rebuilt[market].metrics,
            market,
            evaluated.backtest_protocol.return_offset,
        )
        changes.append(change)
        traces.append(trace)
    all_metrics = pd.concat(metrics, ignore_index=True)
    all_boundaries = pd.concat(boundaries, ignore_index=True)
    _compare_csv(run_dir / "metrics.csv", all_metrics, "metrics")
    _compare_csv(run_dir / "boundaries.csv", all_boundaries, "boundaries")
    _compare_csv(
        run_dir / "endpoint-effects.csv",
        replay_endpoint_effects(all_metrics),
        "endpoint effects",
    )
    _compare_csv(
        run_dir / "path-changes.csv",
        pd.concat(changes, ignore_index=True),
        "path changes",
    )
    _compare_csv(
        run_dir / "change-traces.csv",
        pd.concat(traces, ignore_index=True),
        "change traces",
    )
    composition = {
        "cells": REPLAY_CELL_PATHS,
        "materialized_paths": list(REPLAY_PATHS),
        "passed": True,
    }
    if read_json(run_dir / "composition.json") != composition:
        raise ArtifactError("A-D composition invariants changed")
    binding = all_boundaries.loc[
        all_boundaries["path"].isin(("J1", "K1")), "passed"
    ].eq(False)
    decision = replay_d_rescue_decision(all_metrics)
    decision.update(
        {
            "endpoint_concentration_present": bool(binding.any()),
            "finite_optimum_identified": False if binding.any() else None,
        }
    )
    if read_json(run_dir / "decision.json") != decision:
        raise ArtifactError("endpoint decision does not match independent replay")
    return {
        "schema_version": 1,
        "study_kind": "endpoint_grid_audit",
        "run_id": expected_id,
        "status": "complete",
        "materialized_paths": len(REPLAY_PATHS),
        "metric_rows": len(all_metrics),
        "boundary_rows": len(all_boundaries),
        "performance_claim_allowed": False,
    }


def _reject_symlinks(run_dir: Path) -> None:
    for path in run_dir.rglob("*"):
        if path.is_symlink():
            raise ArtifactError(
                f"endpoint-grid artifact contains symlink: {path.relative_to(run_dir)}"
            )


def _verify_inventory_schema(run_dir: Path) -> None:
    inventory = read_json(run_dir / "inventory.json")
    files = inventory.get("files")
    if (
        set(inventory) != {"schema_version", "files"}
        or inventory.get("schema_version") != 1
        or not isinstance(files, dict)
        or any(
            not isinstance(name, str)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for name, digest in files.items()
        )
    ):
        raise ArtifactError("endpoint-grid inventory schema changed")


def _verify_allowlist(run_dir: Path, delays: tuple[int, ...]) -> None:
    expected_files = {Path(name) for name in ROOT_FILES}
    expected_dirs = {Path(market) for market in REPLAY_MARKETS}
    for market in REPLAY_MARKETS:
        root = Path(market)
        expected_files.update(
            {
                root / "endpoint-jm-states.csv",
                root / "endpoint-jm-refits.csv",
            }
        )
        expected_dirs.add(root / "trades")
        for path in REPLAY_SELECTION_PATHS:
            for delay in delays:
                directory = root / f"{path}-delay-{delay}"
                expected_dirs.add(directory)
                expected_files.update(
                    {
                        directory / "choices.csv",
                        directory / "cv-surface.csv",
                        directory / "selected-signal.csv",
                    }
                )
        expected_files.update(
            root / "trades" / f"{path}-delay-{delay}.csv"
            for path in REPLAY_PATHS
            for delay in delays
        )
    observed_files, observed_dirs = set(), set()
    for path in run_dir.rglob("*"):
        relative = path.relative_to(run_dir)
        if path.is_symlink():
            raise ArtifactError(f"endpoint-grid artifact contains symlink: {relative}")
        if path.is_file():
            observed_files.add(relative)
        elif path.is_dir():
            observed_dirs.add(relative)
        else:
            raise ArtifactError(f"unsupported endpoint-grid artifact: {relative}")
    if observed_files != expected_files or observed_dirs != expected_dirs:
        raise ArtifactError("endpoint-grid recursive artifact allowlist changed")


def _verify_timestamps(metadata: dict[str, Any]) -> None:
    try:
        created = datetime.fromisoformat(str(metadata["created_at_utc"]))
        finished = datetime.fromisoformat(str(metadata["finished_at_utc"]))
    except ValueError as exc:
        raise ArtifactError("endpoint-grid timestamps are invalid") from exc
    if created.tzinfo is None or finished.tzinfo is None or finished < created:
        raise ArtifactError("endpoint-grid timestamps are invalid")


def _git_sha(value: Any) -> str:
    try:
        valid = isinstance(value, str) and len(value) == 40 and int(value, 16) >= 0
    except ValueError:
        valid = False
    if not valid:
        raise ArtifactError("endpoint-grid Git identity is invalid")
    return value


def _repository_root(run_dir: Path, config, spec, run_id: str) -> Path:
    depth = len(config.artifact_root.parts) + len(spec.artifact_subdir.parts)
    try:
        root = run_dir.parents[depth]
    except IndexError as exc:
        raise ArtifactError("endpoint-grid artifact layout is invalid") from exc
    expected = root / config.artifact_root / spec.artifact_subdir / run_id
    if expected.resolve() != run_dir:
        raise ArtifactError("endpoint-grid artifact layout is invalid")
    return root


def _verify_selection_evidence(target: Path, rebuilt: ReplayPrepared) -> None:
    for path in REPLAY_SELECTION_PATHS:
        for delay, selection in rebuilt.selections[path].items():
            directory = target / f"{path}-delay-{delay}"
            _compare_csv(
                directory / "choices.csv", selection.choices, f"{path} choices"
            )
            _compare_csv(
                directory / "cv-surface.csv", selection.surface, f"{path} CV surface"
            )
            _compare_csv(
                directory / "selected-signal.csv",
                selection.signal.to_frame().reset_index(),
                f"{path} signal",
            )


def _verify_market_paths(target: Path, rebuilt: ReplayMarket, config) -> None:
    for delay, paths in rebuilt.paths.items():
        for path, expected in paths.items():
            observed = read_trade_path(
                target / "trades" / f"{path}-delay-{delay}.csv",
                delay,
                config.backtest_protocol.one_way_cost_bps,
            )
            if tuple(observed.columns) != TRADE_COLUMNS:
                raise ArtifactError("trade schema changed")
            _compare_frames(observed, expected, f"{path} trades")


def _verify_endpoint_fit(
    target: Path, expected: FixedJMResult, endpoint: float
) -> None:
    states = pd.read_csv(target / "endpoint-jm-states.csv")
    if states.empty or states.columns[0] != "date" or len(states.columns) != 2:
        raise ArtifactError("endpoint JM state schema changed")
    observed_dates = pd.DatetimeIndex(
        pd.to_datetime(states.pop("date"), errors="raise"), name="date"
    )
    states.columns = [float(states.columns[0])]
    states = states.apply(pd.to_numeric, errors="raise").set_axis(observed_dates)
    if (
        tuple(states.columns) != (endpoint,)
        or tuple(expected.states.columns) != (endpoint,)
        or not observed_dates.equals(expected.states.index)
    ):
        raise ArtifactError("endpoint JM states are invalid")
    _compare_exact_frames(states, expected.states, "endpoint JM states")
    observed_refits = _canonical_refits(
        pd.read_csv(target / "endpoint-jm-refits.csv", float_precision="round_trip"),
        endpoint,
    )
    expected_refits = _canonical_refits(expected.refits.copy(), endpoint)
    _compare_exact_frames(observed_refits, expected_refits, "endpoint JM refits")


def _canonical_refits(frame: pd.DataFrame, endpoint: float) -> pd.DataFrame:
    if tuple(frame.columns) != REFIT_COLUMNS or frame.empty:
        raise ArtifactError("endpoint JM refit schema changed")
    output = frame.copy()
    for column in ("fit_date", "training_start", "training_end"):
        output[column] = pd.to_datetime(output[column], errors="raise")
    observations = pd.to_numeric(output["observations"], errors="raise")
    if not np.equal(observations, observations.astype(np.int64)).all():
        raise ArtifactError("endpoint JM refit observations are invalid")
    output["observations"] = observations.astype(np.int64)
    for column in ("scaler_mean", "scaler_scale"):
        output[column] = output[column].map(_refit_vector)
    for column in ("lambda", "objective"):
        output[column] = pd.to_numeric(output[column], errors="raise").astype(float)
    if not np.isfinite(output[["lambda", "objective"]].to_numpy()).all():
        raise ArtifactError("endpoint JM refit scalars are not finite")
    if not output["lambda"].eq(endpoint).all():
        raise ArtifactError("endpoint JM refits use another lambda")
    return output


def _refit_vector(value: object) -> tuple[float, ...]:
    parsed = ast.literal_eval(value) if isinstance(value, str) else value
    if not isinstance(parsed, (list, tuple)) or len(parsed) != len(FEATURE_COLUMNS):
        raise ArtifactError("endpoint JM refit scaler vector is invalid")
    vector = tuple(float(item) for item in parsed)
    if not np.isfinite(vector).all():
        raise ArtifactError("endpoint JM refit scaler vector is not finite")
    return vector


def _compare_exact_frames(
    observed: pd.DataFrame, expected: pd.DataFrame, label: str
) -> None:
    try:
        pd.testing.assert_frame_equal(
            observed, expected, check_dtype=True, check_exact=True, check_freq=False
        )
    except AssertionError as exc:
        raise ArtifactError(f"{label} do not exactly match independent refit") from exc


def _compare_csv(path: Path, expected: pd.DataFrame, label: str) -> None:
    try:
        observed = pd.read_csv(path, float_precision="round_trip")
    except (FileNotFoundError, OSError, pd.errors.ParserError) as exc:
        raise ArtifactError(f"cannot read {label}") from exc
    for column in expected.columns:
        if pd.api.types.is_datetime64_any_dtype(expected[column]):
            observed[column] = pd.to_datetime(observed[column], errors="raise")
    _compare_frames(observed, expected.reset_index(drop=True), label)


def _compare_frames(observed: pd.DataFrame, expected: pd.DataFrame, label: str) -> None:
    try:
        pd.testing.assert_frame_equal(
            observed.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=0,
            atol=1e-12,
        )
    except AssertionError as exc:
        raise ArtifactError(f"{label} do not match independent replay") from exc
