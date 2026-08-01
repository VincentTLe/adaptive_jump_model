"""Frozen contract for the arrival-rule refit study.

-002 reruns the -001 question against the calibrated-v10 baseline: the betas,
the q_train definition, the timing conventions and the output schemas are
unchanged, but the candidate grid is now one grid per market and the sealed
parent is named by the spec rather than by a literal in this file.
"""

from __future__ import annotations

import hashlib
import math
import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from adaptive_jump.config import ResearchConfig
from adaptive_jump.models import FEATURE_COLUMNS
from adaptive_jump.study_grids import (
    MarketGrids,
    grid_for,
    grids_equal,
    market_grids,
    parse_grid_table,
)
from adaptive_jump.study_sources import (
    SourceReference,
    read_source_reference,
    registry_lock,
)

# The rerun of adaptive-confidence-001 against the calibrated-v10 baseline.
# Verifying the archived -001 run requires the pre-restoration commit.
EXPERIMENT_ID = "adaptive-confidence-002"
BETAS = (0.0, math.log(2.0), math.log(4.0))
MARKETS = ("us", "de", "jp")


class ConfidenceStudyError(ValueError):
    """Raised when the frozen study or its parent nesting contract is violated."""


@dataclass(frozen=True)
class ConfidenceSpec:
    path: Path
    sha256: str
    experiment_id: str
    parent: SourceReference
    data_manifest_sha256: str
    data_cutoff: date
    betas: tuple[float, ...]
    lambdas: MarketGrids
    markets: tuple[str, ...]
    artifact_subdir: Path

    @property
    def parent_run_id(self) -> str:
        return self.parent.run_id

    @property
    def parent_inventory_sha256(self) -> str:
        return self.parent.inventory_sha256

    def lambdas_for(self, market: str) -> tuple[float, ...]:
        """The candidate lambda grid governing one market."""
        return grid_for(self.lambdas, market)


def load_confidence_spec(path: str | Path, config: ResearchConfig) -> ConfidenceSpec:
    """Load the compact frozen contract and bind it to its sealed parent."""
    spec_path = Path(path).resolve()
    payload = spec_path.read_bytes()
    try:
        document = tomllib.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ConfidenceStudyError(f"invalid confidence study TOML: {exc}") from exc

    required_flags = (
        document.get("schema_version") == 1,
        document.get("experiment_id") == EXPERIMENT_ID,
        document.get("claim_class") == "EXPLORATORY",
        document.get("performance_claim_allowed") is False,
        document.get("extension_access") is False,
        document.get("post_2023_access") is False,
    )
    if not all(required_flags):
        raise ConfidenceStudyError("confidence study identity or evidence lane changed")

    parent_table = document.get("parent", {})
    parent = read_source_reference(
        parent_table, error=ConfidenceStudyError, label="fixed"
    )
    manifest = parent_table.get("data_manifest_sha256")
    if (
        parent_table.get("config_sha256") != config.sha256
        or not isinstance(manifest, str)
        or not manifest
        or parent_table.get("data_cutoff") != config.replication_cutoff.isoformat()
        or date.fromisoformat(parent_table["data_cutoff"]) > date(2023, 12, 31)
    ):
        raise ConfidenceStudyError("confidence study parent or cutoff changed")

    penalty = document.get("penalty", {})
    betas = tuple(float(value) for value in penalty.get("beta", ()))
    candidates = document.get("candidates", {})
    comparison = document.get("comparison", {})
    markets = tuple(comparison.get("markets", ()))
    lambdas = parse_grid_table(candidates.get("raw_lambda_grid"), markets or MARKETS)
    controls = document.get("controls", {})
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
        or markets != MARKETS
        or not grids_equal(lambdas, market_grids(config, markets))
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
    assert lambdas is not None
    digest = hashlib.sha256(payload).hexdigest()
    registry_lock(
        config.path.parent,
        str(document["experiment_id"]),
        digest,
        error=ConfidenceStudyError,
    )
    return ConfidenceSpec(
        path=spec_path,
        sha256=digest,
        experiment_id=document["experiment_id"],
        parent=parent,
        data_manifest_sha256=str(manifest),
        data_cutoff=date.fromisoformat(parent_table["data_cutoff"]),
        betas=betas,
        lambdas=lambdas,
        markets=markets,
        artifact_subdir=artifact_subdir,
    )
