"""Strict contract loader for the one-shot endpoint-grid audit."""

from __future__ import annotations

import hashlib
import math
import tomllib
from datetime import date, datetime
from pathlib import Path
from typing import Any

from adaptive_jump.config import ResearchConfig
from adaptive_jump.endpoint_grid_types import (
    MDD_ABSOLUTE_DEADBAND,
    METRIC_CHANGE_TOLERANCE,
    PRIMARY_DELAY,
    EndpointGridError,
    EndpointGridSpec,
)
from adaptive_jump.models import FEATURE_COLUMNS

PARENT_EXPERIMENT_ID = "fixed-baselines-001-v7"
CALIBRATION_EXPERIMENT_ID = "persistence-calibrated-search-001"
BASE_EXPERIMENT_ID = "persistence-grid-evaluation-001"


def load_endpoint_grid_spec(
    path: str | Path, config: ResearchConfig
) -> EndpointGridSpec:
    """Strictly load the draft/frozen one-shot endpoint audit contract."""
    spec_path = Path(path).resolve()
    payload = spec_path.read_bytes()
    try:
        doc = tomllib.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise EndpointGridError(f"invalid endpoint-grid TOML: {exc}") from exc
    status = doc.get("protocol_status")
    _need(status in {"DRAFT", "FROZEN"}, "protocol status is invalid")
    if status == "DRAFT":
        _need("frozen_at_utc" not in doc, "draft must not have a freeze timestamp")
    else:
        value = doc.get("frozen_at_utc")
        _need(isinstance(value, str) and value, "frozen study needs a timestamp")
        try:
            datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError as exc:
            raise EndpointGridError("freeze timestamp is invalid") from exc
    expected = {
        "schema_version": 1,
        "claim_class": "EXPLORATORY",
        "stage": "ENDPOINT_GRID_AUDIT",
        "post_2023_access": False,
        "adaptive_experiment": False,
        "performance_claim_allowed": False,
    }
    _need(
        all(doc.get(key) == value for key, value in expected.items()),
        "study header changed",
    )
    parent = _table(doc, "parent")
    calibration = _table(doc, "calibration")
    base = _table(doc, "base_grid")
    derivation = _table(doc, "endpoint_derivation")
    protocol = _table(doc, "protocol")
    cells = _table(doc, "cells")
    boundary = _table(doc, "boundary")
    smoke = _table(doc, "smoke")
    control = _table(doc, "control")
    execution = _table(doc, "execution")
    _need(
        parent.get("experiment_id") == PARENT_EXPERIMENT_ID,
        "parent experiment changed",
    )
    parent_config = _relative(parent, "config_path")
    source_matches = (config.path.parent / parent_config).resolve() == config.path
    sealed_matches = (
        config.path.name == "config.lock.toml"
        and spec_path.name == "study.lock.toml"
        and parent_config == Path("research.toml")
    )
    _need(source_matches or sealed_matches, "parent config path changed")
    _need(parent.get("config_sha256") == config.sha256, "config hash changed")
    cutoff = date.fromisoformat(_text(parent, "data_cutoff"))
    _need(cutoff == config.replication_cutoff <= date(2023, 12, 31), "cutoff changed")
    _need(
        calibration.get("experiment_id") == CALIBRATION_EXPERIMENT_ID
        and calibration.get("metrics_opened") is False,
        "calibration contract changed",
    )
    _need(
        base.get("experiment_id") == BASE_EXPERIMENT_ID
        and base.get("required_status") == "boundary_failed"
        and base.get("metrics_opened") is False
        and base.get("candidate_budget_per_model") == 9,
        "base grid contract changed",
    )
    base_inventory = base.get("run_inventory_sha256")
    if status == "DRAFT":
        _need(
            base_inventory == "DRAFT_REQUIRED_BEFORE_FREEZE",
            "draft base inventory placeholder changed",
        )
        base_inventory = None
    else:
        base_inventory = _hex(base, "run_inventory_sha256")
    _need(
        derivation
        == {
            "selection_rule": "maximum candidate with globally_valid and eligible",
            "jm_path": "lambda_j = 2^(j/2)",
            "jm_next_candidate": "endpoint * sqrt(2)",
            "hmm_path": "integer k with unit steps",
            "hmm_next_candidate": "endpoint + 1",
            "first_higher_candidate_must_be_globally_invalid": True,
            "hand_entered_fallback_allowed": False,
        },
        "endpoint derivation changed",
    )
    expected_protocol = {
        "features": list(FEATURE_COLUMNS),
        "fit_window_observations": config.model_protocol.fit_window,
        "validation_calendar_years": config.selection_protocol.validation_years,
        "jm_refit_months": list(config.jm_protocol.refit_months),
        "hmm_raw_fit": "reuse exact frozen-v7 raw states",
        "monthly_objective": "annualized strategy excess Sharpe",
        "tie_rule": "lower_smoothing",
        "signal_to_return_offset": config.backtest_protocol.return_offset,
        "primary_delay": config.backtest_protocol.primary_delay,
        "robustness_delays": list(config.backtest_protocol.robustness_delays),
        "one_way_cost_bps": config.backtest_protocol.one_way_cost_bps,
        "comparison_sample": (
            "per-market intersection across all five paths and all delays"
        ),
        "turnover": "half_mean_one_way_turnover_times_252",
        "turnover_scale_source": "config.metrics_protocol.turnover_scale",
    }
    _need(protocol == expected_protocol, "protocol changed")
    expected_paths = {
        "materialized": ["B&H", "J0", "J1", "K0", "K1"],
        "J0": "fixed JM on the selected nine-candidate calibration grid",
        "J1": "J0 plus the derived JM endpoint",
        "K0": "HMM smoothing on the selected nine-candidate calibration grid",
        "K1": "K0 plus the derived HMM endpoint",
    }
    _need(_table(doc, "paths") == expected_paths, "path definitions changed")
    expected_cells = {
        "A": ["J0", "K0"],
        "B": ["J1", "K0"],
        "C": ["J0", "K1"],
        "D": ["J1", "K1"],
    }
    _need(
        all(cells.get(key) == value for key, value in expected_cells.items()),
        "cell composition changed",
    )
    _need(
        cells.get("composition_only") is True
        and cells.get("fit_only_one_new_jm_candidate") is True
        and cells.get("refit_hmm") is False,
        "cell execution changed",
    )
    _need(
        boundary.get("descriptive_only") is True
        and boundary.get("seals_metrics") is False
        and boundary.get("additional_expansion_allowed") is False
        and math.isclose(
            float(boundary.get("upper_candidate_month_fraction_limit", math.nan)),
            config.selection_protocol.boundary_fraction_limit,
            rel_tol=0,
            abs_tol=1e-15,
        ),
        "boundary diagnostic changed",
    )
    _need(
        smoke
        == {
            "market": "us",
            "terminal_dates": 20,
            "performance_metrics_opened": False,
            "run_before_market_evaluation": True,
            "require_full_run_prefix_match": True,
        },
        "smoke changed",
    )
    decision = _table(doc, "decision")
    _need(
        decision
        == {
            "endpoint_effect": (
                "report J1 minus J0 and K1 minus K0 for Sharpe, maximum drawdown, "
                "turnover, cash fraction, and switch count"
            ),
            "finite_optimum_identified": (
                "false if either endpoint remains the most-selected candidate above "
                "the descriptive boundary limit"
            ),
            "primary_delay": PRIMARY_DELAY,
            "d_rescue_gate": (
                "J1 Sharpe > K1 Sharpe and J1 Sharpe > buy-and-hold Sharpe and "
                "abs(J1 MDD) < abs(buy-and-hold MDD) in every market"
            ),
            "mdd_absolute_deadband": MDD_ABSOLUTE_DEADBAND,
            "mdd_deadband_rule": (
                "improvement less than or equal to deadband is neutral and not better"
            ),
            "all_markets_required": True,
            "binding_pairs": ["J1-J0", "K1-K0"],
            "binding_components": [
                "monthly_choices",
                "signal_days",
                "comparable_state_days",
                "delayed_position_days",
                "trade_turnover_days",
                "reported_metrics",
            ],
            "binding_rule": "binding iff any component changes",
            "metric_change_tolerance": METRIC_CHANGE_TOLERANCE,
            "change_trace_delay": PRIMARY_DELAY,
            "change_trace_signal_to_position_offset": (
                config.backtest_protocol.return_offset
            ),
            "winner_selection_allowed": False,
            "paper_replication_claim_allowed": False,
        },
        "decision rule changed",
    )
    _need(
        control
        == {
            "parity_scope_label": (
                "selection-behavior parity; base witness contains no trades or metrics"
            ),
            "base_grid_artifact_role": (
                "behavior-parity witness only; never a candidate-state or signal input"
            ),
            "jm_current_recomputation": (
                "recompute all nine J0 candidates plus one derived endpoint"
            ),
            "hmm_current_recomputation": (
                "recompute all nine K0 smoothings plus one derived endpoint from "
                "locked parent raw HMM states"
            ),
            "parity_components": [
                "complete J0 candidate-state matrix",
                "complete J0 refit evidence",
                "complete K0 candidate-state matrix",
                "J0 and K0 monthly choices",
                "J0 and K0 CV surfaces",
                "J0 and K0 candidate returns",
                "J0 and K0 selected signals",
                "J0 and K0 boundary diagnostics",
            ],
            "exact_comparison": True,
            "metrics_before_parity_allowed": False,
            "receipt_fields": [
                "source_hashes",
                "artifact_hashes",
                "current_hashes",
                "counts",
            ],
        },
        "control contract changed",
    )
    _need(
        execution
        == {
            "process_start_method": "forkserver",
            "market_workers": 3,
            "numerical_threads": 1,
        },
        "execution contract changed",
    )
    return EndpointGridSpec(
        spec_path,
        hashlib.sha256(payload).hexdigest(),
        str(status),
        _text(doc, "experiment_id"),
        _text(parent, "run_id"),
        _hex(parent, "run_inventory_sha256"),
        _hex(parent, "data_manifest_sha256"),
        _text(calibration, "run_id"),
        _hex(calibration, "spec_sha256"),
        _hex(calibration, "run_inventory_sha256"),
        _hex(calibration, "selection_sha256"),
        _text(base, "run_id"),
        _hex(base, "spec_sha256"),
        base_inventory,
        _relative(base, "artifact_subdir"),
        20,
        _relative(_table(doc, "storage"), "artifact_subdir"),
        "forkserver",
        3,
        1,
    )


def _table(document: dict[str, Any], key: str) -> dict[str, Any]:
    value = document.get(key)
    if not isinstance(value, dict):
        raise EndpointGridError(f"{key} must be a table")
    return value


def _text(document: dict[str, Any], key: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value:
        raise EndpointGridError(f"{key} must be non-empty text")
    return value


def _hex(document: dict[str, Any], key: str) -> str:
    value = _text(document, key)
    try:
        valid = len(value) == 64 and int(value, 16) >= 0
    except ValueError:
        valid = False
    if not valid:
        raise EndpointGridError(f"{key} must be a SHA-256")
    return value


def _relative(document: dict[str, Any], key: str) -> Path:
    value = Path(_text(document, key))
    if value.is_absolute() or ".." in value.parts:
        raise EndpointGridError(f"{key} must be a safe relative path")
    return value


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise EndpointGridError(message)
