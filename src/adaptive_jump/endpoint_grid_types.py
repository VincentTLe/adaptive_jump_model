"""Shared contracts and verified inputs for the endpoint-grid audit."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

PRIMARY_DELAY = 1
MDD_ABSOLUTE_DEADBAND = 1e-9
METRIC_CHANGE_TOLERANCE = 1e-12
REPORTED_METRICS = (
    "sharpe",
    "maximum_drawdown",
    "turnover",
    "cash_fraction",
    "switch_count",
)


class EndpointGridError(ValueError):
    """The draft contract, sealed lineage, or endpoint audit is inconsistent."""


@dataclass(frozen=True)
class EndpointGridSpec:
    path: Path
    sha256: str
    protocol_status: str
    experiment_id: str
    parent_run_id: str
    parent_inventory_sha256: str
    data_manifest_sha256: str
    calibration_run_id: str
    calibration_spec_sha256: str
    calibration_inventory_sha256: str
    calibration_selection_sha256: str
    base_run_id: str
    base_spec_sha256: str
    base_inventory_sha256: str | None
    base_artifact_subdir: Path
    smoke_terminal_dates: int
    artifact_subdir: Path
    process_start_method: str
    market_workers: int
    numerical_threads: int


@dataclass(frozen=True)
class EndpointEvidence:
    jm_endpoint: float
    jm_first_invalid: float
    jm_index: int
    hmm_endpoint: int
    hmm_first_invalid: int

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "selection_rule": "maximum candidate with globally_valid and eligible",
            "jm_endpoint": self.jm_endpoint,
            "jm_first_invalid": self.jm_first_invalid,
            "jm_index": self.jm_index,
            "hmm_endpoint": self.hmm_endpoint,
            "hmm_first_invalid": self.hmm_first_invalid,
        }


@dataclass(frozen=True)
class MarketSource:
    market: str
    frame: pd.DataFrame
    raw_hmm: pd.Series
    oos_start: date
    feature_path: Path
    raw_hmm_path: Path


@dataclass(frozen=True)
class _Lineage:
    parent_dir: Path
    base_dir: Path
    jm_grid: tuple[float, ...]
    hmm_grid: tuple[int, ...]
    endpoints: EndpointEvidence
