"""Independent verification for sealed JM training-window runs."""

from __future__ import annotations

import math
import subprocess
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from adaptive_jump.artifacts import (
    ArtifactError,
    read_json,
    sha256_file,
    verify_inventory,
)
from adaptive_jump.config import ResearchConfig, load_config
from adaptive_jump.features import effective_oos_start
from adaptive_jump.walkforward import boundary_diagnostic, select_monthly_candidate
from adaptive_jump.window_evidence import (
    verify_window_bootstrap,
    verify_window_metrics,
)
from adaptive_jump.window_runner import CONTROL_SCOPE
from adaptive_jump.window_spec import WindowStudySpec, load_window_spec
from adaptive_jump.window_study import window_claim


def verify_window_run(run: str | Path) -> dict[str, Any]:
    """Recompute all exposed evidence in one sealed window experiment."""
    run_dir = Path(run).resolve()
    if not run_dir.is_dir():
        raise ArtifactError(f"run directory does not exist: {run_dir}")
    verify_inventory(run_dir)
    metadata, config, spec, parent_dir = _verify_identity(run_dir)
    boundaries = _verify_model_evidence(run_dir, parent_dir, config, spec)
    all_passed = bool(boundaries["passed"].all())
    if metadata["status"] == "boundary_failed":
        _verify_boundary_failure(run_dir, metadata, all_passed)
        metric_rows = bootstrap_count = 0
        metric_difference = bootstrap_difference = 0.0
    else:
        if not all_passed or metadata.get("metrics_opened") is not True:
            raise ArtifactError("complete window run has an invalid boundary state")
        metrics, paths, metric_difference = verify_window_metrics(run_dir, config, spec)
        bootstrap, bootstrap_difference = verify_window_bootstrap(
            run_dir, paths, config, spec
        )
        claim = window_claim(
            metrics,
            bootstrap,
            market_ids=tuple(market.id for market in config.markets),
            primary_delay=spec.primary_delay,
            primary_block=spec.bootstrap_blocks[0],
        )
        if read_json(run_dir / "claim.json") != claim:
            raise ArtifactError("window claim does not match recomputed evidence")
        if metadata.get("conclusion") != claim["conclusion"]:
            raise ArtifactError("window conclusion does not match its claim")
        metric_rows, bootstrap_count = len(metrics), len(bootstrap)
    inventory = read_json(run_dir / "inventory.json")["files"]
    return {
        "schema_version": 1,
        "study_kind": "jm_train_window_sensitivity",
        "run_id": metadata["run_id"],
        "status": metadata["status"],
        "inventory_files": len(inventory),
        "boundary_rows": len(boundaries),
        "metric_rows": metric_rows,
        "bootstrap_rows": bootstrap_count,
        "maximum_metric_absolute_difference": metric_difference,
        "maximum_bootstrap_absolute_difference": bootstrap_difference,
        "conclusion": metadata["conclusion"],
    }


def _verify_identity(
    run_dir: Path,
) -> tuple[dict[str, Any], ResearchConfig, WindowStudySpec, Path]:
    metadata = read_json(run_dir / "run.json")
    if (
        metadata.get("schema_version") != 1
        or metadata.get("study_kind") != "jm_train_window_sensitivity"
        or metadata.get("status") not in {"complete", "boundary_failed"}
        or metadata.get("claim_class") != "EXPLORATORY"
    ):
        raise ArtifactError("unsupported JM-window run identity")
    config_path = run_dir / "config.lock.toml"
    spec_path = run_dir / "study.lock.toml"
    config = load_config(config_path)
    spec = load_window_spec(spec_path, config)
    identity = {
        "spec_sha256": spec.sha256,
        "config_sha256": config.sha256,
        "data_manifest_sha256": spec.data_manifest_sha256,
        "parent_inventory_sha256": spec.parent_inventory_sha256,
    }
    if any(metadata.get(key) != value for key, value in identity.items()):
        raise ArtifactError("JM-window metadata disagrees with its locks")
    if (
        metadata.get("experiment_id") != spec.experiment_id
        or metadata.get("parent_run_id") != spec.parent_run_id
    ):
        raise ArtifactError("JM-window experiment lineage is invalid")
    git_sha = _hex(metadata.get("git_sha"), "Git SHA")
    expected_id = "jm-window-" + "-".join(
        value[:12] for value in (spec.sha256, spec.data_manifest_sha256, git_sha)
    )
    if metadata.get("run_id") != expected_id or run_dir.name != expected_id:
        raise ArtifactError("JM-window directory and run identity disagree")
    manifest_path = run_dir / "data-manifest.json"
    if (
        sha256_file(config_path) != config.sha256
        or sha256_file(spec_path) != spec.sha256
        or sha256_file(manifest_path) != spec.data_manifest_sha256
    ):
        raise ArtifactError("JM-window lock hash mismatch")
    manifest = read_json(manifest_path)
    if (
        manifest.get("config_sha256") != config.sha256
        or manifest.get("config_id") != config.config_id
        or manifest.get("replication_cutoff") != spec.data_cutoff.isoformat()
    ):
        raise ArtifactError("JM-window data manifest disagrees with its locks")

    parent_dir = run_dir.parent.parent / "fixed-baselines" / spec.parent_run_id
    from adaptive_jump.artifacts import verify_run

    parent_receipt = verify_run(parent_dir)
    parent_metadata = read_json(parent_dir / "run.json")
    if (
        parent_receipt.get("status") != "complete"
        or read_json(run_dir / "parent-verification.json") != parent_receipt
        or sha256_file(parent_dir / "inventory.json") != spec.parent_inventory_sha256
        or parent_metadata.get("config_sha256") != config.sha256
        or parent_metadata.get("data_manifest_sha256") != spec.data_manifest_sha256
    ):
        raise ArtifactError("JM-window parent verification failed")
    control = read_json(run_dir / "control.json")
    if control != {
        "control_scope": list(CONTROL_SCOPE),
        "control_source_unchanged": True,
        "parent_git_sha": parent_metadata["git_sha"],
        "parent_run_id": spec.parent_run_id,
    }:
        raise ArtifactError("JM-window control declaration is invalid")
    _verify_control_commit(run_dir.parents[2], str(parent_metadata["git_sha"]), git_sha)
    return metadata, config, spec, parent_dir


def _verify_control_commit(root: Path, parent_sha: str, run_sha: str) -> None:
    for revision in (parent_sha, run_sha):
        found = subprocess.run(
            ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
            cwd=root,
            check=False,
            capture_output=True,
        )
        if found.returncode != 0:
            raise ArtifactError("run references an unavailable Git commit")
    unchanged = subprocess.run(
        ["git", "diff", "--quiet", parent_sha, run_sha, "--", *CONTROL_SCOPE],
        cwd=root,
        check=False,
    )
    if unchanged.returncode != 0:
        raise ArtifactError("v7 control source changed before the window run")


def _verify_model_evidence(
    run_dir: Path,
    parent_dir: Path,
    config: ResearchConfig,
    spec: WindowStudySpec,
) -> pd.DataFrame:
    boundaries = _read_csv(run_dir / "boundaries.csv")
    expected_columns = set("market model delay upper_candidate selected_months".split())
    expected_columns.update("total_months fraction limit passed".split())
    expected_keys = {
        (market.id, "jm_4000", delay)
        for market in config.markets
        for delay in spec.delays
    }
    actual_keys = (
        set(boundaries[["market", "model", "delay"]].itertuples(index=False, name=None))
        if expected_columns.issubset(boundaries)
        else set()
    )
    if (
        set(boundaries.columns) != expected_columns
        or actual_keys != expected_keys
        or len(boundaries) != len(expected_keys)
    ):
        raise ArtifactError("JM-window boundary coverage is invalid")
    for market in config.markets:
        market_dir = run_dir / market.id
        states = _read_states(market_dir / "jm-4000-states.csv", config)
        _verify_refits(market_dir / "jm-4000-refits.csv", config, spec)
        features = _read_csv(parent_dir / market.id / "features.csv")
        feature_dates = pd.DatetimeIndex(
            pd.to_datetime(features["date"], errors="raise"), name="date"
        )
        if not states.index.equals(feature_dates):
            raise ArtifactError(f"{market.id}: JM-4000 state dates differ from inputs")
        oos_start = effective_oos_start(
            features,
            requested=date.fromisoformat(config.document["oos_start"]["requested"]),
            fit_window=spec.challenger_window,
            validation_years=config.selection_protocol.validation_years,
        )
        if oos_start is None:
            raise ArtifactError(f"{market.id}: cannot reproduce JM-4000 eligibility")
        local_boundaries = _read_csv(market_dir / "boundaries.csv")
        root_boundaries = boundaries.loc[boundaries["market"] == market.id].drop(
            columns="market"
        )
        _assert_frame_close(
            local_boundaries.reset_index(drop=True),
            root_boundaries.reset_index(drop=True),
            f"{market.id} boundaries",
        )
        for delay in spec.delays:
            target = market_dir / f"jm-4000-delay-{delay}"
            choices = _read_csv(target / "choices.csv")
            expected_selection = select_monthly_candidate(
                features[["date", "equity_simple", "cash_return"]],
                states,
                config.selection_protocol,
                delay_trading_days=delay,
                one_way_cost_bps=config.backtest_protocol.one_way_cost_bps,
                periods_per_year=config.metrics_protocol.periods_per_year,
                volatility_ddof=config.metrics_protocol.volatility_ddof,
            )
            _verify_selection(target, choices, expected_selection)
            diagnostic = boundary_diagnostic(
                expected_selection.choices,
                config.jm_protocol.lambda_grid,
                oos_start=oos_start,
                fraction_limit=spec.boundary_fraction_limit,
            )
            row = boundaries.loc[
                (boundaries["market"] == market.id) & (boundaries["delay"] == delay)
            ].iloc[0]
            _compare_boundary(row, diagnostic.__dict__)
    return boundaries


def _read_states(path: Path, config: ResearchConfig) -> pd.DataFrame:
    states = _read_csv(path)
    if states.empty or states.columns[0] != "date":
        raise ArtifactError("JM-4000 state schema is invalid")
    dates = pd.to_datetime(states.pop("date"), errors="raise")
    try:
        states.columns = [float(value) for value in states.columns]
    except ValueError as exc:
        raise ArtifactError("JM-4000 state candidates are invalid") from exc
    values = states.stack().to_numpy(dtype=float)
    if (
        tuple(states.columns) != config.jm_protocol.lambda_grid
        or not len(values)
        or dates.isna().any()
        or dates.duplicated().any()
        or not dates.is_monotonic_increasing
        or not np.isin(values, [0.0, 1.0]).all()
    ):
        raise ArtifactError("JM-4000 states violate the model contract")
    return states.set_axis(dates)


def _verify_refits(path: Path, config: ResearchConfig, spec: WindowStudySpec) -> None:
    refits = _read_csv(path)
    required = {
        "fit_date",
        "training_start",
        "training_end",
        "observations",
        "lambda",
        "objective",
    }
    if not required.issubset(refits) or refits.empty:
        raise ArtifactError("JM-4000 refit records are incomplete")
    if not (pd.to_numeric(refits["observations"]) == spec.challenger_window).all():
        raise ArtifactError("JM-4000 refit window is inconsistent")
    fit_dates = pd.to_datetime(refits["fit_date"], errors="raise")
    starts = pd.to_datetime(refits["training_start"], errors="raise")
    ends = pd.to_datetime(refits["training_end"], errors="raise")
    if (
        fit_dates.isna().any()
        or starts.isna().any()
        or ends.isna().any()
        or not ends.equals(fit_dates)
        or (starts > ends).any()
        or not fit_dates.dt.month.isin(config.jm_protocol.refit_months).all()
    ):
        raise ArtifactError("JM-4000 refit dates are inconsistent")
    expected = set(config.jm_protocol.lambda_grid)
    if any(
        set(group["lambda"].astype(float)) != expected
        for _, group in refits.groupby("fit_date")
    ):
        raise ArtifactError("JM-4000 refit lambda coverage is incomplete")
    if not np.isfinite(pd.to_numeric(refits["objective"])).all():
        raise ArtifactError("JM-4000 refit objective is invalid")


def _verify_selection(target: Path, choices: pd.DataFrame, expected: Any) -> None:
    expected_files = set(
        "choices.csv cv-surface.csv candidate-returns.csv selected-signal.csv".split()
    )
    if {path.name for path in target.iterdir() if path.is_file()} != expected_files:
        raise ArtifactError(f"selection artifact coverage is invalid: {target}")
    comparisons = (
        (choices, expected.choices.reset_index(drop=True), "choices"),
        (_read_csv(target / "cv-surface.csv"), expected.surface, "CV surface"),
        (
            _read_csv(target / "candidate-returns.csv"),
            expected.candidate_returns.reset_index(),
            "candidate returns",
        ),
        (
            _read_csv(target / "selected-signal.csv"),
            expected.signal.to_frame().reset_index(),
            "selected signal",
        ),
    )
    for stored, calculated, label in comparisons:
        _assert_frame_close(stored, calculated, label)


def _assert_frame_close(
    stored: pd.DataFrame, calculated: pd.DataFrame, label: str
) -> None:
    stored = stored.copy()
    calculated = calculated.copy()
    stored.columns = stored.columns.map(str)
    calculated.columns = calculated.columns.map(str)
    for column in stored.columns.intersection(calculated.columns):
        if "date" in column:
            stored[column] = pd.to_datetime(stored[column], errors="raise")
            calculated[column] = pd.to_datetime(calculated[column], errors="raise")
    try:
        pd.testing.assert_frame_equal(
            stored,
            calculated,
            check_dtype=False,
            check_exact=False,
            rtol=0,
            atol=1e-12,
        )
    except AssertionError as exc:
        raise ArtifactError(f"recomputed {label} differs") from exc


def _compare_boundary(row: Any, expected: dict[str, Any]) -> None:
    exact = ("selected_months", "total_months", "passed")
    numeric = ("upper_candidate", "fraction", "limit")
    if any(row[key] != expected[key] for key in exact) or any(
        not math.isclose(
            float(row[key]), float(expected[key]), rel_tol=0, abs_tol=1e-12
        )
        for key in numeric
    ):
        raise ArtifactError("JM-4000 boundary row does not match its choices")


def _verify_boundary_failure(
    run_dir: Path, metadata: dict[str, Any], all_passed: bool
) -> None:
    forbidden = ("metrics.csv", "bootstrap.csv", "claim.json")
    if (
        all_passed
        or metadata.get("metrics_opened") is not False
        or metadata.get("conclusion")
        != "JM-4000 upper-lambda boundary requires a new experiment"
        or any((run_dir / name).exists() for name in forbidden)
        or any(run_dir.glob("*/trades/*.csv"))
    ):
        raise ArtifactError("boundary-failed JM-window run exposes invalid evidence")


def _hex(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) < 12:
        raise ArtifactError(f"invalid {label}")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ArtifactError(f"invalid {label}") from exc
    return value


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, OSError, pd.errors.ParserError) as exc:
        raise ArtifactError(f"cannot read CSV {path}: {exc}") from exc
