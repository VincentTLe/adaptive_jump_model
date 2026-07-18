"""Independent replay verifier for the one-shot endpoint-grid audit."""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_jump.artifacts import (
    read_json,
    sha256_file,
)
from adaptive_jump.calibration import jm_penalty
from adaptive_jump.calibration_runner import verify_calibration_run
from adaptive_jump.config import ResearchConfig
from adaptive_jump.endpoint_grid_spec import (
    BASE_EXPERIMENT_ID,
    CALIBRATION_EXPERIMENT_ID,
    PARENT_EXPERIMENT_ID,
)
from adaptive_jump.endpoint_grid_spec import (
    load_endpoint_grid_spec as load_endpoint_grid_spec,
)
from adaptive_jump.endpoint_grid_types import (
    EndpointEvidence,
    EndpointGridError,
    EndpointGridSpec,
    MarketSource,
    _Lineage,
)
from adaptive_jump.features import effective_oos_start
from adaptive_jump.grid_runner import verify_grid_run
from adaptive_jump.grid_spec import load_grid_spec
from adaptive_jump.models import FEATURE_COLUMNS


def derive_endpoints(diagnostics: pd.DataFrame) -> EndpointEvidence:
    """Derive, never supply, the last eligible endpoint and first invalid point."""
    required = {"model", "candidate", "globally_valid", "eligible"}
    if not required.issubset(diagnostics):
        raise EndpointGridError("calibration diagnostics are incomplete")
    rows = diagnostics.loc[:, list(required)].copy()
    if (
        set(rows["model"]) != {"fixed_jm", "hmm"}
        or rows.duplicated(["model", "candidate"]).any()
    ):
        raise EndpointGridError("calibration candidate coverage is invalid")
    rows["candidate"] = pd.to_numeric(rows["candidate"], errors="raise")
    if not np.isfinite(rows["candidate"]).all() or (rows["candidate"] < 0).any():
        raise EndpointGridError("calibration candidates are invalid")
    if not pd.api.types.is_bool_dtype(
        rows["globally_valid"]
    ) or not pd.api.types.is_bool_dtype(rows["eligible"]):
        raise EndpointGridError("calibration flags are invalid")
    values: dict[str, tuple[float, float]] = {}
    for model in ("fixed_jm", "hmm"):
        selected = rows.loc[rows["model"] == model].sort_values("candidate")
        eligible = selected.loc[selected["globally_valid"] & selected["eligible"]]
        if eligible.empty:
            raise EndpointGridError(f"{model}: no globally valid eligible endpoint")
        endpoint = float(eligible["candidate"].max())
        higher = selected.loc[selected["candidate"] > endpoint]
        if higher.empty:
            raise EndpointGridError(
                f"{model}: first invalid candidate was not observed"
            )
        first = higher.iloc[0]
        if bool(first["globally_valid"]) or bool(first["eligible"]):
            raise EndpointGridError(f"{model}: first higher candidate is not invalid")
        values[model] = endpoint, float(first["candidate"])
    jm, jm_invalid = values["fixed_jm"]
    index = int(round(2.0 * math.log2(jm)))
    if not math.isclose(
        jm, jm_penalty(index), rel_tol=0, abs_tol=1e-12
    ) or not math.isclose(jm_invalid, jm * math.sqrt(2.0), rel_tol=0, abs_tol=1e-12):
        raise EndpointGridError(
            "JM endpoint does not follow its sealed half-power path"
        )
    hmm, hmm_invalid = values["hmm"]
    if (
        not hmm.is_integer()
        or not hmm_invalid.is_integer()
        or int(hmm_invalid) != int(hmm) + 1
    ):
        raise EndpointGridError(
            "HMM endpoint does not have a unit-step invalid neighbor"
        )
    return EndpointEvidence(jm, jm_invalid, index, int(hmm), int(hmm_invalid))


def _read_candidate_diagnostics(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


def verify_lineage(config: ResearchConfig, spec: EndpointGridSpec) -> _Lineage:
    """Verify both performance-free parents before deriving either endpoint."""
    root = config.path.parent
    parent_dir = root / config.artifact_root / "fixed-baselines" / spec.parent_run_id
    parent_metadata = read_json(parent_dir / "run.json")
    if (
        sha256_file(parent_dir / "inventory.json") != spec.parent_inventory_sha256
        or parent_metadata.get("run_id") != spec.parent_run_id
        or (
            "experiment_id" in parent_metadata
            and parent_metadata["experiment_id"] != PARENT_EXPERIMENT_ID
        )
        or parent_metadata.get("config_sha256") != config.sha256
        or parent_metadata.get("data_manifest_sha256") != spec.data_manifest_sha256
    ):
        raise EndpointGridError("sealed parent lineage changed")
    calibration = (
        root
        / config.artifact_root
        / "persistence-calibrated-search"
        / spec.calibration_run_id
    )
    receipt = verify_calibration_run(calibration)
    calibration_metadata = read_json(calibration / "run.json")
    checks = {
        calibration / "inventory.json": spec.calibration_inventory_sha256,
        calibration / "study.lock.toml": spec.calibration_spec_sha256,
        calibration / "selection.json": spec.calibration_selection_sha256,
    }
    if (
        receipt.get("run_id") != spec.calibration_run_id
        or calibration_metadata.get("run_id") != spec.calibration_run_id
        or calibration_metadata.get("experiment_id") != CALIBRATION_EXPERIMENT_ID
        or any(sha256_file(path) != expected for path, expected in checks.items())
    ):
        raise EndpointGridError("sealed calibration lineage changed")
    base_dir = (
        root / config.artifact_root / spec.base_artifact_subdir / spec.base_run_id
    )
    base_receipt = verify_grid_run(base_dir)
    base_metadata = read_json(base_dir / "run.json")
    if (
        spec.base_inventory_sha256 is None
        or sha256_file(base_dir / "inventory.json") != spec.base_inventory_sha256
        or base_receipt.get("run_id") != spec.base_run_id
        or base_receipt.get("status") != "boundary_failed"
        or base_metadata.get("experiment_id") != BASE_EXPERIMENT_ID
        or sha256_file(base_dir / "study.lock.toml") != spec.base_spec_sha256
        or base_metadata.get("parent_run_id") != spec.parent_run_id
        or base_metadata.get("parent_inventory_sha256") != spec.parent_inventory_sha256
        or base_metadata.get("calibration_inventory_sha256")
        != spec.calibration_inventory_sha256
        or base_metadata.get("data_manifest_sha256") != spec.data_manifest_sha256
    ):
        raise EndpointGridError("sealed base-grid run changed")
    base_spec = load_grid_spec(base_dir / "study.lock.toml", config)
    selection = read_json(calibration / "selection.json").get("selected_grids")
    if (
        not isinstance(selection, dict)
        or tuple(selection.get("fixed_jm", ())) != base_spec.jm_grid
        or tuple(selection.get("hmm", ())) != base_spec.hmm_grid
    ):
        raise EndpointGridError("base grids do not match calibrated selection")
    endpoints = derive_endpoints(
        _read_candidate_diagnostics(calibration / "candidate-diagnostics.csv")
    )
    if (
        endpoints.jm_endpoint <= max(base_spec.jm_grid)
        or endpoints.jm_endpoint in base_spec.jm_grid
        or endpoints.hmm_endpoint <= max(base_spec.hmm_grid)
        or endpoints.hmm_endpoint in base_spec.hmm_grid
    ):
        raise EndpointGridError("derived endpoints do not extend each base grid once")
    return _Lineage(
        parent_dir,
        base_dir,
        base_spec.jm_grid,
        base_spec.hmm_grid,
        endpoints,
    )


def load_market_source(
    parent_dir: Path, market: str, config: ResearchConfig, lineage: _Lineage
) -> MarketSource:
    """Load and individually hash-check the two sealed parent inputs."""
    if parent_dir != lineage.parent_dir:
        raise EndpointGridError("market source is not the sealed parent")
    feature_path = parent_dir / market / "features.csv"
    raw_hmm_path = parent_dir / market / "hmm-states.csv"
    _verify_parent_file(parent_dir, feature_path)
    _verify_parent_file(parent_dir, raw_hmm_path)
    frame = pd.read_csv(feature_path)
    required = {
        "date",
        "equity_simple",
        "cash_return",
        "excess_return",
        *FEATURE_COLUMNS,
    }
    if not required.issubset(frame):
        raise EndpointGridError(f"{market}: features are incomplete")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if (
        frame["date"].duplicated().any()
        or not frame["date"].is_monotonic_increasing
        or frame["date"].max().date() > config.replication_cutoff
    ):
        raise EndpointGridError(f"{market}: feature dates are invalid")
    dates = pd.DatetimeIndex(frame["date"], name="date")
    raw_frame = pd.read_csv(raw_hmm_path)
    if tuple(raw_frame.columns) != ("date", "hmm_state"):
        raise EndpointGridError(f"{market}: raw HMM schema changed")
    raw_hmm = pd.Series(
        pd.to_numeric(raw_frame["hmm_state"], errors="raise").to_numpy(),
        index=pd.DatetimeIndex(
            pd.to_datetime(raw_frame["date"], errors="raise"), name="date"
        ),
        dtype=float,
    )
    values = raw_hmm.dropna()
    if (
        raw_hmm.index.has_duplicates
        or not raw_hmm.index.is_monotonic_increasing
        or not raw_hmm.index.equals(dates)
        or not values.isin((0.0, 1.0)).all()
    ):
        raise EndpointGridError(f"{market}: raw HMM states are invalid")
    requested = date.fromisoformat(config.document["oos_start"]["requested"])
    oos = effective_oos_start(
        frame,
        requested=requested,
        fit_window=config.model_protocol.fit_window,
        validation_years=config.selection_protocol.validation_years,
    )
    if oos is None:
        raise EndpointGridError(f"{market}: no eligible OOS start")
    return MarketSource(market, frame, raw_hmm, oos, feature_path, raw_hmm_path)


def _verify_parent_file(parent_dir: Path, path: Path) -> None:
    inventory = read_json(parent_dir / "inventory.json").get("files")
    relative = str(path.relative_to(parent_dir))
    if (
        not isinstance(inventory, dict)
        or not path.is_file()
        or path.is_symlink()
        or inventory.get(relative) != sha256_file(path)
    ):
        raise EndpointGridError(f"sealed parent source changed: {relative}")
