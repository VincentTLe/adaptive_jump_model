"""Contract and data loading for the AJM-EXT-001 external runner.

This module is the runner's trust boundary. It refuses to proceed unless the
contract bytes hash to the registry-frozen value, reads each Fama-French region
straight out of its hash-pinned zip (never from a loose extracted file), and
refuses to open the confirmation region at all — that seal is released only by
the transport gate, not by code in this module.
"""

from __future__ import annotations

import hashlib
import io
import math
import re
import tomllib
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# Pinned by the FROZEN registry event of 2026-08-05T23:20:47Z. An edited
# contract must not run; amend via a new experiment id instead.
FROZEN_CONTRACT_SHA256 = (
    "e331b96d662ca703a3fa5140d4ba9d92544c9eed553eb36df1525fafbc0b6a49"
)
MISSING_MARKER = -99.99


class ExtSourceError(ValueError):
    """Raised when the contract or a region payload violates the frozen seal."""


@dataclass(frozen=True)
class ExtContract:
    path: Path
    sha256: str
    experiment_id: str
    beta: float
    grids: tuple[tuple[float, ...], ...]
    grid_names: tuple[str, ...]
    standardizers: tuple[str, ...]
    hmm_grid: tuple[int, ...]
    fit_window: int
    refit_months: tuple[int, ...]
    validation_years: int
    one_way_cost_bps: int
    primary_delay: int
    descriptive_delays: tuple[int, ...]
    available_start: date
    evaluation_cutoff: date
    transport_regions: tuple[str, ...]
    required_passing_region: str
    minimum_positive_regions: int


@dataclass(frozen=True)
class LockedSource:
    role: str
    region: str
    zip_name: str
    sha256: str
    csv_name: str | None


def load_ext_contract(path: str | Path) -> ExtContract:
    """Parse the frozen contract, refusing any byte drift from the registry."""
    contract_path = Path(path).resolve()
    payload = contract_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != FROZEN_CONTRACT_SHA256:
        raise ExtSourceError(
            "contract bytes do not match the registry-frozen sha256 "
            f"(expected {FROZEN_CONTRACT_SHA256}, found {digest}); "
            "an amended question needs a new experiment id"
        )
    document = tomllib.loads(payload.decode("utf-8"))

    def _need(table: dict, key: str):
        if key not in table:
            raise ExtSourceError(f"contract is missing {key}")
        return table[key]

    baseline = _need(document, "baseline_family")
    challenger = _need(document, "challenger")
    protocol = _need(document, "protocol")
    external = _need(document, "external_data")
    transport = _need(document, "transport_gate")

    grid_names: list[str] = []
    grids: list[tuple[float, ...]] = []
    for entry in _need(baseline, "grids"):
        name, _, values = str(entry).partition(":")
        grid_names.append(name)
        grids.append(tuple(float(v) for v in values.split(",")))
    beta = float(_need(challenger, "beta"))
    if beta != math.log(4.0):
        raise ExtSourceError("challenger beta must be exactly ln(4)")
    contract = ExtContract(
        path=contract_path,
        sha256=digest,
        experiment_id=str(_need(document, "experiment_id")),
        beta=beta,
        grids=tuple(grids),
        grid_names=tuple(grid_names),
        standardizers=tuple(_need(baseline, "standardizers")),
        hmm_grid=tuple(int(k) for k in _need(baseline, "hmm_grid")),
        fit_window=int(_need(protocol, "fit_window_observations")),
        refit_months=tuple(int(m) for m in _need(protocol, "refit_months")),
        validation_years=int(_need(protocol, "validation_calendar_years")),
        one_way_cost_bps=int(_need(protocol, "one_way_cost_bps")),
        primary_delay=int(_need(protocol, "primary_delay")),
        descriptive_delays=tuple(int(d) for d in _need(protocol, "descriptive_delays")),
        available_start=date.fromisoformat(str(_need(external, "available_start"))),
        evaluation_cutoff=date.fromisoformat(str(_need(external, "evaluation_cutoff"))),
        transport_regions=tuple(_need(document["data_roles"], "transport")),
        required_passing_region=str(_need(transport, "required_passing_region")),
        minimum_positive_regions=int(_need(transport, "minimum_positive_regions")),
    )
    if contract.fit_window != 3000 or contract.validation_years != 8:
        raise ExtSourceError("protocol windows violate the frozen contract")
    if len(contract.grids) != 2 or len(contract.standardizers) != 2:
        raise ExtSourceError("baseline family must be exactly two grids x two scalers")
    return contract


def load_data_lock(path: str | Path) -> dict[str, LockedSource]:
    """Read the download lock: region -> pinned zip and csv names."""
    document = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    sources: dict[str, LockedSource] = {}
    for row in document.get("source", []):
        source = LockedSource(
            role=str(row["role"]),
            region=str(row["region"]),
            zip_name=str(row["zip"]),
            sha256=str(row["sha256"]),
            csv_name=str(row["csv"]) if "csv" in row else None,
        )
        sources[source.region] = source
    if not sources:
        raise ExtSourceError("data lock names no sources")
    return sources


def load_region_frame(
    region: str,
    contract: ExtContract,
    lock: dict[str, LockedSource],
    data_dir: str | Path,
) -> pd.DataFrame:
    """Load one transport region from its pinned zip into the model frame.

    Output columns: date, equity_simple, cash_return, excess_return,
    equity_log — decimal daily returns on [available_start, evaluation_cutoff].
    """
    if region not in lock:
        raise ExtSourceError(f"region is not in the data lock: {region}")
    source = lock[region]
    if source.role == "confirmation":
        raise ExtSourceError(
            f"{region} is the sealed confirmation region; it opens only after "
            "the transport gate passes, never through this loader"
        )
    if region not in contract.transport_regions:
        raise ExtSourceError(f"region is not a transport region: {region}")
    if source.csv_name is None:
        raise ExtSourceError(f"{region}: lock entry names no csv")

    zip_path = Path(data_dir) / source.zip_name
    if not zip_path.is_file():
        raise ExtSourceError(f"{region}: missing pinned archive {zip_path.name}")
    raw = zip_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != source.sha256:
        raise ExtSourceError(
            f"{region}: archive sha256 mismatch "
            f"(expected {source.sha256}, found {digest})"
        )
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        text = archive.read(source.csv_name).decode("utf-8", errors="strict")

    rows = [line for line in text.splitlines() if re.match(r"^\d{8}\s*,", line)]
    if not rows:
        raise ExtSourceError(f"{region}: no data rows in {source.csv_name}")
    parsed = [[field.strip() for field in line.split(",")] for line in rows]
    frame = pd.DataFrame(parsed, columns=["date", "mkt_rf", "smb", "hml", "rf"])
    dates = pd.to_datetime(frame["date"], format="%Y%m%d", errors="raise")
    mkt_rf = pd.to_numeric(frame["mkt_rf"], errors="raise")
    rf = pd.to_numeric(frame["rf"], errors="raise")
    if (mkt_rf == MISSING_MARKER).any() or (rf == MISSING_MARKER).any():
        raise ExtSourceError(f"{region}: -99.99 missing markers present")

    keep = (dates.dt.date >= contract.available_start) & (
        dates.dt.date <= contract.evaluation_cutoff
    )
    dates, mkt_rf, rf = dates[keep], mkt_rf[keep], rf[keep]
    if dates.empty:
        raise ExtSourceError(f"{region}: no rows inside the contract window")
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ExtSourceError(f"{region}: dates must be unique and increasing")

    equity_simple = (mkt_rf + rf) / 100.0
    cash_return = rf / 100.0
    result = pd.DataFrame(
        {
            "date": dates.to_numpy(),
            "equity_simple": equity_simple.to_numpy(dtype=float),
            "cash_return": cash_return.to_numpy(dtype=float),
        }
    )
    result["excess_return"] = result["equity_simple"] - result["cash_return"]
    if (result["equity_simple"] <= -1.0).any():
        raise ExtSourceError(f"{region}: a daily return at or below -100%")
    result["equity_log"] = np.log1p(result["equity_simple"])
    if not np.isfinite(result.drop(columns="date").to_numpy()).all():
        raise ExtSourceError(f"{region}: non-finite values after conversion")
    return result.reset_index(drop=True)
