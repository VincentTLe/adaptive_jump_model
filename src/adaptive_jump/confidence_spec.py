"""Frozen contract for adaptive-confidence-001."""

from __future__ import annotations

import hashlib
import math
import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from adaptive_jump.config import ResearchConfig
from adaptive_jump.models import FEATURE_COLUMNS

BETAS = (0.0, math.log(2.0), math.log(4.0))
MARKETS = ("us", "de", "jp")


class ConfidenceStudyError(ValueError):
    """Raised when the frozen study or its v7 nesting contract is violated."""


@dataclass(frozen=True)
class ConfidenceSpec:
    path: Path
    sha256: str
    experiment_id: str
    parent_run_id: str
    parent_inventory_sha256: str
    data_manifest_sha256: str
    data_cutoff: date
    betas: tuple[float, ...]
    lambdas: tuple[float, ...]
    markets: tuple[str, ...]
    artifact_subdir: Path


def load_confidence_spec(path: str | Path, config: ResearchConfig) -> ConfidenceSpec:
    """Load the compact frozen contract and bind it to canonical v7."""
    spec_path = Path(path).resolve()
    payload = spec_path.read_bytes()
    try:
        document = tomllib.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ConfidenceStudyError(f"invalid confidence study TOML: {exc}") from exc

    required_flags = (
        document.get("schema_version") == 1,
        document.get("experiment_id") == "adaptive-confidence-001",
        document.get("claim_class") == "EXPLORATORY",
        document.get("performance_claim_allowed") is False,
        document.get("extension_access") is False,
        document.get("post_2023_access") is False,
    )
    if not all(required_flags):
        raise ConfidenceStudyError("confidence study identity or evidence lane changed")

    parent = document.get("parent", {})
    if (
        parent.get("config_sha256") != config.sha256
        or parent.get("data_cutoff") != config.replication_cutoff.isoformat()
        or date.fromisoformat(parent["data_cutoff"]) > date(2023, 12, 31)
    ):
        raise ConfidenceStudyError("confidence study parent or cutoff changed")

    penalty = document.get("penalty", {})
    betas = tuple(float(value) for value in penalty.get("beta", ()))
    candidates = document.get("candidates", {})
    lambdas = tuple(float(value) for value in candidates.get("raw_lambda_grid", ()))
    controls = document.get("controls", {})
    comparison = document.get("comparison", {})
    if (
        betas != BETAS
        or penalty.get("q_train")
        != (
            "raw median absolute deviation about the median of all finite "
            "state-loss entries on the training prefix; require finite and >0"
        )
        or penalty.get("missing_center_loss")
        != (
            "+infinity, matching the existing fixed-JM DP treatment of an "
            "unoccupied fitted state"
        )
        or penalty.get("q_train_fallback") != "none"
        or lambdas != config.jm_protocol.lambda_grid
        or candidates.get("raw_grid_expansion") is not False
        or candidates.get("calibration_framework") is not False
        or candidates.get("beta_selected") is not False
        or tuple(controls.get("features", ())) != FEATURE_COLUMNS
        or controls.get("fit_window_observations") != config.model_protocol.fit_window
        or tuple(controls.get("jm_refit_months", ())) != config.jm_protocol.refit_months
        or controls.get("validation_calendar_years")
        != config.selection_protocol.validation_years
        or controls.get("primary_delay_trading_days")
        != config.backtest_protocol.primary_delay
        or controls.get("signal_to_return_offset")
        != config.backtest_protocol.return_offset
        or controls.get("one_way_cost_bps") != config.backtest_protocol.one_way_cost_bps
        or controls.get("provider_access") is not False
        or tuple(comparison.get("markets", ())) != MARKETS
    ):
        raise ConfidenceStudyError("confidence study controls changed")

    storage = document.get("storage", {})
    artifact_subdir = Path(str(storage.get("artifact_subdir", "")))
    if (
        not artifact_subdir.parts
        or artifact_subdir.is_absolute()
        or ".." in artifact_subdir.parts
    ):
        raise ConfidenceStudyError("invalid confidence artifact subdirectory")
    return ConfidenceSpec(
        path=spec_path,
        sha256=hashlib.sha256(payload).hexdigest(),
        experiment_id=document["experiment_id"],
        parent_run_id=str(parent["run_id"]),
        parent_inventory_sha256=str(parent["run_inventory_sha256"]),
        data_manifest_sha256=str(parent["data_manifest_sha256"]),
        data_cutoff=date.fromisoformat(parent["data_cutoff"]),
        betas=betas,
        lambdas=lambdas,
        markets=tuple(comparison["markets"]),
        artifact_subdir=artifact_subdir,
    )
