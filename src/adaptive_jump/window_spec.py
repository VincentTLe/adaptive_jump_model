"""Strict contract loader for the exploratory JM-window sensitivity."""

from __future__ import annotations

import hashlib
import math
import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from adaptive_jump.config import ResearchConfig


class WindowSpecError(ValueError):
    """Raised when the frozen window experiment contract is inconsistent."""


@dataclass(frozen=True)
class WindowStudySpec:
    """Validated values that are allowed to differ from the parent study."""

    path: Path
    sha256: str
    experiment_id: str
    parent_run_id: str
    parent_inventory_sha256: str
    data_manifest_sha256: str
    data_cutoff: date
    baseline_window: int
    challenger_window: int
    models: tuple[str, ...]
    primary_delay: int
    delays: tuple[int, ...]
    bootstrap_replications: int
    bootstrap_seed: int
    bootstrap_blocks: tuple[int, ...]
    confidence_level: float
    boundary_fraction_limit: float
    artifact_subdir: Path
    report_subdir: Path


def load_window_spec(path: str | Path, config: ResearchConfig) -> WindowStudySpec:
    """Load the preregistered sensitivity and bind it to the exact v7 config."""
    spec_path = Path(path).resolve()
    payload = spec_path.read_bytes()
    try:
        document = tomllib.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise WindowSpecError(f"invalid window study TOML: {exc}") from exc

    _require(document.get("schema_version") == 1, "schema_version must be 1")
    _require(document.get("claim_class") == "EXPLORATORY", "claim must be exploratory")
    _require(document.get("extension_access") is False, "extension access is forbidden")
    _require(document.get("adaptive_experiment") is False, "adaptive work is forbidden")
    experiment_id = _text(document, "experiment_id")

    parent = _table(document, "parent")
    parent_config = _safe_relative(parent, "config_path")
    _require(
        (config.path.parent / parent_config).resolve() == config.path,
        "parent config path does not identify the loaded config",
    )
    _require(
        _hex(parent, "config_sha256") == config.sha256,
        "parent config hash does not match v7",
    )
    cutoff = _date(parent, "data_cutoff")
    _require(cutoff == config.replication_cutoff, "data cutoff does not match v7")
    _require(cutoff <= date(2023, 12, 31), "post-2023 access is forbidden")

    windows = _table(document, "windows")
    baseline = _positive_int(windows, "baseline_observations")
    challenger = _positive_int(windows, "challenger_observations")
    _require(baseline == config.model_protocol.fit_window, "baseline window changed")
    _require(challenger > baseline, "challenger window must be longer")
    _require(
        _positive_int(windows, "validation_calendar_years")
        == config.selection_protocol.validation_years,
        "validation window changed",
    )
    _require(
        _positive_int(windows, "hmm_observations") == baseline,
        "HMM window must remain at the parent value",
    )
    _require(
        _int_tuple(windows, "jm_refit_months") == config.jm_protocol.refit_months,
        "JM refit schedule changed",
    )
    _require(
        windows.get("online_lookback_equals_fit_window") is True,
        "online JM lookback must equal its fit window",
    )

    comparison = _table(document, "comparison")
    models = _text_tuple(comparison, "models")
    _require(
        models == ("buy_and_hold", "hmm_3000", "jm_3000", "jm_4000"),
        "comparison models changed",
    )
    primary_delay = _positive_int(comparison, "primary_delay")
    delays = _int_tuple(comparison, "robustness_delays")
    _require(
        primary_delay == config.backtest_protocol.primary_delay, "primary delay changed"
    )
    _require(delays == config.backtest_protocol.robustness_delays, "delay grid changed")
    _require(
        comparison.get("sample")
        == "per_market_delay_intersection_after_challenger_eligibility",
        "comparison sample changed",
    )
    _require(
        comparison.get("primary_metric") == "annualized_excess_sharpe",
        "primary metric changed",
    )

    bootstrap = _table(document, "bootstrap")
    _require(
        bootstrap.get("method") == "paired_stationary_block", "bootstrap method changed"
    )
    primary_block = _positive_int(bootstrap, "mean_block_length")
    sensitivity_blocks = _int_tuple(bootstrap, "sensitivity_block_lengths")
    blocks = (primary_block, *sensitivity_blocks)
    _require(len(set(blocks)) == len(blocks), "bootstrap blocks must be unique")
    confidence = _number(bootstrap, "confidence_level")
    _require(0 < confidence < 1, "confidence level must be between zero and one")

    decision = _table(document, "decision")
    _require(
        decision.get("consistent_improvement")
        == "primary_delta_positive_in_all_three_markets",
        "consistent-improvement rule changed",
    )
    _require(
        decision.get("mixed") == "primary_delta_positive_in_one_or_two_markets",
        "mixed rule changed",
    )
    _require(
        decision.get("not_supported") == "primary_delta_positive_in_zero_markets",
        "null rule changed",
    )
    _require(
        decision.get("uncertainty_supported")
        == "one_sided_95pct_lower_bound_positive_in_all_three_markets",
        "uncertainty rule changed",
    )
    _require(
        decision.get("grid_expansion_after_results") is False,
        "post-result grid expansion is forbidden",
    )
    boundary_limit = _number(decision, "upper_lambda_boundary_fraction_limit")
    _require(
        math.isclose(
            boundary_limit,
            config.selection_protocol.boundary_fraction_limit,
            rel_tol=0,
            abs_tol=1e-15,
        ),
        "boundary threshold changed",
    )

    storage = _table(document, "storage")
    return WindowStudySpec(
        path=spec_path,
        sha256=hashlib.sha256(payload).hexdigest(),
        experiment_id=experiment_id,
        parent_run_id=_text(parent, "run_id"),
        parent_inventory_sha256=_hex(parent, "run_inventory_sha256"),
        data_manifest_sha256=_hex(parent, "data_manifest_sha256"),
        data_cutoff=cutoff,
        baseline_window=baseline,
        challenger_window=challenger,
        models=models,
        primary_delay=primary_delay,
        delays=delays,
        bootstrap_replications=_positive_int(bootstrap, "replications"),
        bootstrap_seed=_positive_int(bootstrap, "seed"),
        bootstrap_blocks=blocks,
        confidence_level=confidence,
        boundary_fraction_limit=boundary_limit,
        artifact_subdir=_safe_relative(storage, "artifact_subdir"),
        report_subdir=_safe_relative(storage, "report_subdir"),
    )


def _table(document: dict[str, Any], key: str) -> dict[str, Any]:
    value = document.get(key)
    if not isinstance(value, dict):
        raise WindowSpecError(f"{key} must be a table")
    return value


def _text(document: dict[str, Any], key: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value:
        raise WindowSpecError(f"{key} must be non-empty text")
    return value


def _positive_int(document: dict[str, Any], key: str) -> int:
    value = document.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise WindowSpecError(f"{key} must be a positive integer")
    return value


def _number(document: dict[str, Any], key: str) -> float:
    value = document.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise WindowSpecError(f"{key} must be finite")
    return float(value)


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
        raise WindowSpecError(f"{key} must contain non-negative integers")
    return tuple(value)


def _text_tuple(document: dict[str, Any], key: str) -> tuple[str, ...]:
    value = document.get(key)
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise WindowSpecError(f"{key} must contain non-empty text")
    return tuple(value)


def _date(document: dict[str, Any], key: str) -> date:
    try:
        return date.fromisoformat(_text(document, key))
    except ValueError as exc:
        raise WindowSpecError(f"{key} must be an ISO date") from exc


def _hex(document: dict[str, Any], key: str) -> str:
    value = _text(document, key)
    if len(value) != 64:
        raise WindowSpecError(f"{key} must be a SHA-256 hash")
    try:
        int(value, 16)
    except ValueError as exc:
        raise WindowSpecError(f"{key} must be a SHA-256 hash") from exc
    return value


def _safe_relative(document: dict[str, Any], key: str) -> Path:
    value = Path(_text(document, key))
    if value.is_absolute() or ".." in value.parts:
        raise WindowSpecError(f"{key} must be a safe relative path")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise WindowSpecError(message)
