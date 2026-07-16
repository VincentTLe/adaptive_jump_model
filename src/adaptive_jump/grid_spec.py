"""Strict loader for the frozen persistence-grid evaluation."""

from __future__ import annotations

import hashlib
import math
import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from adaptive_jump.config import ResearchConfig


class GridSpecError(ValueError):
    """Raised when the grid-evaluation contract differs from its freeze."""


@dataclass(frozen=True)
class GridStudySpec:
    """Validated study values allowed to differ from canonical v7."""

    path: Path
    sha256: str
    experiment_id: str
    parent_run_id: str
    parent_inventory_sha256: str
    data_manifest_sha256: str
    data_cutoff: date
    calibration_run_id: str
    calibration_inventory_sha256: str
    calibration_selection_sha256: str
    jm_grid: tuple[float, ...]
    hmm_grid: tuple[int, ...]
    primary_delay: int
    delays: tuple[int, ...]
    boundary_fraction_limit: float
    bootstrap_replications: int
    bootstrap_seed: int
    bootstrap_blocks: tuple[int, ...]
    confidence_level: float
    artifact_subdir: Path


def load_grid_spec(path: str | Path, config: ResearchConfig) -> GridStudySpec:
    """Load and bind the frozen grid study to the exact canonical config."""
    spec_path = Path(path).resolve()
    payload = spec_path.read_bytes()
    try:
        document = tomllib.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise GridSpecError(f"invalid grid study TOML: {exc}") from exc

    _require(document.get("schema_version") == 1, "schema_version must be 1")
    _require(document.get("claim_class") == "EXPLORATORY", "claim must be exploratory")
    _require(document.get("stage") == "OOS_GRID_ATTRIBUTION", "study stage changed")
    for key in ("extension_access", "post_2023_access", "adaptive_experiment"):
        _require(document.get(key) is False, f"{key} must be false")

    parent = _table(document, "parent")
    parent_config = _safe_relative(parent, "config_path")
    source_matches = (config.path.parent / parent_config).resolve() == config.path
    sealed_matches = (
        config.path.name == "config.lock.toml"
        and spec_path.name == "study.lock.toml"
        and parent_config == Path("research.toml")
    )
    _require(source_matches or sealed_matches, "parent config path changed")
    _require(parent.get("config_sha256") == config.sha256, "config hash changed")
    cutoff = _date(parent, "data_cutoff")
    _require(cutoff == config.replication_cutoff, "data cutoff changed")
    _require(cutoff <= date(2023, 12, 31), "post-2023 access is forbidden")

    calibration = _table(document, "calibration")
    _require(calibration.get("metrics_opened") is False, "calibration opened metrics")
    _require(calibration.get("reviewed_by_owner") is True, "grid was not reviewed")

    grids = _table(document, "grids")
    jm_grid = _float_tuple(grids, "fixed_jm")
    hmm_grid = _int_tuple(grids, "hmm")
    budget = _positive_int(grids, "equal_budget")
    _require(len(jm_grid) == len(hmm_grid) == budget == 9, "grid budget changed")
    _require(jm_grid == tuple(sorted(set(jm_grid))), "JM grid is not increasing")
    _require(hmm_grid == tuple(sorted(set(hmm_grid))), "HMM grid is not increasing")
    _require(jm_grid[0] == 0 and hmm_grid[0] == 0, "zero smoothing is required")
    _require(
        grids.get("selection_source")
        == "pre-OOS behavior spacing; no strategy performance",
        "grid selection source changed",
    )
    _require(
        grids.get("result_driven_expansion_allowed") is False,
        "result-driven expansion is forbidden",
    )

    controls = _table(document, "controls")
    _require(
        _positive_int(controls, "fit_window_observations")
        == config.model_protocol.fit_window,
        "fit window changed",
    )
    _require(
        _positive_int(controls, "validation_calendar_years")
        == config.selection_protocol.validation_years,
        "validation window changed",
    )
    _require(
        _int_tuple(controls, "jm_refit_months") == config.jm_protocol.refit_months,
        "JM refit schedule changed",
    )
    _require(
        _positive_int(controls, "signal_to_return_offset")
        == config.backtest_protocol.return_offset,
        "return offset changed",
    )
    delays = _int_tuple(controls, "robustness_delays")
    _require(delays == config.backtest_protocol.robustness_delays, "delays changed")
    _require(
        _positive_int(controls, "one_way_cost_bps")
        == config.backtest_protocol.one_way_cost_bps,
        "cost changed",
    )
    _require(controls.get("provider_access") is False, "provider access is forbidden")

    evaluation = _table(document, "evaluation")
    expected_flags = {
        "fit_new_jm_candidates": True,
        "refit_hmm": False,
        "reuse_parent_features": True,
        "reuse_parent_raw_hmm_states": True,
        "reuse_parent_buy_and_hold": True,
        "reuse_parent_oos_dates": True,
        "reuse_parent_candidate_choices": False,
        "checkpointing": True,
    }
    _require(
        all(evaluation.get(key) is value for key, value in expected_flags.items()),
        "evaluation reuse contract changed",
    )

    boundary = _table(document, "boundary_gate")
    boundary_limit = _number(boundary, "upper_candidate_month_fraction_limit")
    _require(
        math.isclose(
            boundary_limit,
            config.selection_protocol.boundary_fraction_limit,
            rel_tol=0,
            abs_tol=1e-15,
        ),
        "boundary threshold changed",
    )
    _require(boundary.get("checked_before_metrics") is True, "boundary order changed")
    _require(boundary.get("expected_rows") == 18, "boundary row count changed")
    _require(boundary.get("expansion_after_failure") is False, "expansion forbidden")

    comparison = _table(document, "comparison")
    primary_delay = _positive_int(comparison, "primary_delay")
    _require(
        primary_delay == config.backtest_protocol.primary_delay,
        "primary delay changed",
    )
    _require(
        _int_tuple(comparison, "robustness_delays") == delays,
        "comparison delays changed",
    )
    _require(
        comparison.get("primary_metric") == "annualized_excess_sharpe",
        "primary metric changed",
    )

    bootstrap = _table(document, "bootstrap")
    _require(
        bootstrap.get("method") == "paired_stationary_block",
        "bootstrap method changed",
    )
    primary_block = _positive_int(bootstrap, "mean_block_length")
    blocks = (primary_block, *_int_tuple(bootstrap, "sensitivity_block_lengths"))
    _require(len(set(blocks)) == len(blocks), "bootstrap blocks must be unique")
    confidence = _number(bootstrap, "confidence_level")
    _require(0 < confidence < 1, "confidence level is invalid")
    _require(bootstrap.get("holm_markets") is True, "Holm correction is required")

    decision = _table(document, "decision")
    _require(
        decision.get("paper_replication_claim_allowed") is False,
        "paper replication claim is forbidden",
    )
    storage = _table(document, "storage")
    return GridStudySpec(
        path=spec_path,
        sha256=hashlib.sha256(payload).hexdigest(),
        experiment_id=_text(document, "experiment_id"),
        parent_run_id=_text(parent, "run_id"),
        parent_inventory_sha256=_hex(parent, "run_inventory_sha256"),
        data_manifest_sha256=_hex(parent, "data_manifest_sha256"),
        data_cutoff=cutoff,
        calibration_run_id=_text(calibration, "run_id"),
        calibration_inventory_sha256=_hex(calibration, "run_inventory_sha256"),
        calibration_selection_sha256=_hex(calibration, "selection_sha256"),
        jm_grid=jm_grid,
        hmm_grid=hmm_grid,
        primary_delay=primary_delay,
        delays=delays,
        boundary_fraction_limit=boundary_limit,
        bootstrap_replications=_positive_int(bootstrap, "replications"),
        bootstrap_seed=_positive_int(bootstrap, "seed"),
        bootstrap_blocks=blocks,
        confidence_level=confidence,
        artifact_subdir=_safe_relative(storage, "artifact_subdir"),
    )


def _table(document: dict[str, Any], key: str) -> dict[str, Any]:
    value = document.get(key)
    if not isinstance(value, dict):
        raise GridSpecError(f"{key} must be a table")
    return value


def _text(document: dict[str, Any], key: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value:
        raise GridSpecError(f"{key} must be non-empty text")
    return value


def _positive_int(document: dict[str, Any], key: str) -> int:
    value = document.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise GridSpecError(f"{key} must be a positive integer")
    return value


def _number(document: dict[str, Any], key: str) -> float:
    value = document.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GridSpecError(f"{key} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise GridSpecError(f"{key} must be finite")
    return value


def _float_tuple(document: dict[str, Any], key: str) -> tuple[float, ...]:
    value = document.get(key)
    if not isinstance(value, list) or not value:
        raise GridSpecError(f"{key} must be a non-empty list")
    values = tuple(_number({"value": item}, "value") for item in value)
    if any(item < 0 for item in values):
        raise GridSpecError(f"{key} must be non-negative")
    return values


def _int_tuple(document: dict[str, Any], key: str) -> tuple[int, ...]:
    value = document.get(key)
    if (
        not isinstance(value, list)
        or not value
        or any(
            not isinstance(item, int) or isinstance(item, bool) or item < 0
            for item in value
        )
    ):
        raise GridSpecError(f"{key} must contain non-negative integers")
    return tuple(value)


def _date(document: dict[str, Any], key: str) -> date:
    try:
        return date.fromisoformat(_text(document, key))
    except ValueError as exc:
        raise GridSpecError(f"{key} must be an ISO date") from exc


def _hex(document: dict[str, Any], key: str) -> str:
    value = _text(document, key)
    if len(value) != 64:
        raise GridSpecError(f"{key} must be a SHA-256 hash")
    try:
        int(value, 16)
    except ValueError as exc:
        raise GridSpecError(f"{key} must be a SHA-256 hash") from exc
    return value


def _safe_relative(document: dict[str, Any], key: str) -> Path:
    value = Path(_text(document, key))
    if value.is_absolute() or ".." in value.parts:
        raise GridSpecError(f"{key} must be a safe relative path")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GridSpecError(message)
