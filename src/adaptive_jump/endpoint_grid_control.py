"""Exact selection-behavior parity control for the endpoint-grid audit."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from adaptive_jump.artifacts import sha256_file
from adaptive_jump.endpoint_grid_types import EndpointGridError, MarketSource
from adaptive_jump.models import FEATURE_COLUMNS
from adaptive_jump.walkforward import SelectionResult

BASE_SELECTIONS = (("fixed_jm", "J0"), ("hmm", "K0"))
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


def verify_state_parity(
    source: MarketSource,
    witness_dir: Path,
    base_jm: pd.DataFrame,
    base_refits: pd.DataFrame,
    base_hmm: pd.DataFrame,
    current_git_sha: str,
) -> dict[str, Any]:
    """Abort unless all current base states and JM refits equal the witness."""
    if len(base_jm.columns) != 9 or len(base_hmm.columns) != 9:
        raise EndpointGridError("behavior parity requires nine base candidates")
    paths = {
        "base_jm_states": witness_dir / "jm-states.csv",
        "base_jm_refits": witness_dir / "jm-refits.csv",
        "base_hmm_candidates": witness_dir / "hmm-candidates.csv",
    }
    witness_jm = _state_frame(paths["base_jm_states"], base_jm)
    witness_hmm = _state_frame(paths["base_hmm_candidates"], base_hmm)
    witness_refits = _canonical_refits(_read_csv(paths["base_jm_refits"]))
    current_refits = _canonical_refits(base_refits.copy())
    _assert_exact(witness_jm, base_jm, "J0 candidate states")
    _assert_exact(witness_hmm, base_hmm, "K0 candidate states")
    _assert_exact(witness_refits, current_refits, "J0 refits")
    return {
        "schema_version": 1,
        "mode": "current-code-selection-behavior-exact-parity",
        "market": source.market,
        "current_git_sha": current_git_sha,
        "source_hashes": {
            "parent_features": sha256_file(source.feature_path),
            "parent_raw_hmm_states": sha256_file(source.raw_hmm_path),
        },
        "artifact_hashes": {key: sha256_file(path) for key, path in paths.items()},
        "current_hashes": {
            "base_jm_states": _frame_hash(base_jm),
            "base_jm_refits": _frame_hash(current_refits, index=False),
            "base_hmm_candidates": _frame_hash(base_hmm),
        },
        "counts": {
            "source_rows": len(source.frame),
            "raw_hmm_rows": len(source.raw_hmm),
            "jm_state_rows": len(base_jm),
            "jm_base_candidates": len(base_jm.columns),
            "jm_refit_rows": len(current_refits),
            "hmm_state_rows": len(base_hmm),
            "hmm_base_candidates": len(base_hmm.columns),
        },
        "state_and_refit_parity_passed": True,
        "selection_behavior_parity_passed": False,
        "passed": False,
    }


def verify_selection_parity(
    receipt: dict[str, Any],
    witness_dir: Path,
    selections: dict[str, dict[int, SelectionResult]],
    boundaries: pd.DataFrame,
    delays: tuple[int, ...],
) -> dict[str, Any]:
    """Compare choices, CV, candidate returns, signals, and boundaries exactly."""
    output = {
        **receipt,
        "artifact_hashes": dict(receipt["artifact_hashes"]),
        "current_hashes": dict(receipt["current_hashes"]),
        "counts": dict(receipt["counts"]),
    }
    for witness_model, path in BASE_SELECTIONS:
        for delay in delays:
            selected = selections[path][delay]
            prefix = f"{witness_model}_delay_{delay}"
            directory = witness_dir / f"{witness_model}-delay-{delay}"
            components = {
                "choices": _choices(selected),
                "cv_surface": _surface(selected),
                "candidate_returns": _candidate_returns(selected),
                "selected_signal": _signal(selected),
            }
            names = {
                "choices": "choices.csv",
                "cv_surface": "cv-surface.csv",
                "candidate_returns": "candidate-returns.csv",
                "selected_signal": "selected-signal.csv",
            }
            for component, expected in components.items():
                artifact = directory / names[component]
                observed = _evidence_frame(artifact, expected)
                _assert_exact(observed, expected, f"{path} {component}")
                key = f"{prefix}_{component}"
                output["artifact_hashes"][key] = sha256_file(artifact)
                output["current_hashes"][key] = _frame_hash(expected, index=False)
                output["counts"][f"{key}_rows"] = len(expected)
    witness_boundary_path = witness_dir / "boundaries.csv"
    current_boundaries = _base_boundaries(boundaries)
    witness_boundaries = _evidence_frame(witness_boundary_path, current_boundaries)
    _assert_exact(witness_boundaries, current_boundaries, "base boundaries")
    output["artifact_hashes"]["base_boundaries"] = sha256_file(witness_boundary_path)
    output["current_hashes"]["base_boundaries"] = _frame_hash(
        current_boundaries, index=False
    )
    output["counts"]["base_boundary_rows"] = len(current_boundaries)
    output["selection_behavior_parity_passed"] = True
    output["passed"] = True
    return output


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, float_precision="round_trip")
    except (FileNotFoundError, OSError, pd.errors.ParserError) as exc:
        raise EndpointGridError(f"cannot read behavior witness {path}") from exc


def _state_frame(path: Path, expected: pd.DataFrame) -> pd.DataFrame:
    frame = _read_csv(path)
    if frame.empty or frame.columns[0] != "date":
        raise EndpointGridError(f"behavior witness state schema changed: {path}")
    dates = pd.DatetimeIndex(
        pd.to_datetime(frame.pop("date"), errors="raise"), name="date"
    )
    try:
        candidates = tuple(float(value) for value in frame.columns)
    except ValueError as exc:
        raise EndpointGridError(f"behavior witness candidates changed: {path}") from exc
    if candidates != tuple(float(value) for value in expected.columns):
        raise EndpointGridError(f"behavior witness candidates changed: {path}")
    frame = frame.apply(pd.to_numeric, errors="raise")
    frame.columns = expected.columns
    frame.index = dates
    return frame


def _canonical_refits(frame: pd.DataFrame) -> pd.DataFrame:
    if tuple(frame.columns) != REFIT_COLUMNS or frame.empty:
        raise EndpointGridError("base JM refit schema changed")
    output = frame.copy()
    for column in ("fit_date", "training_start", "training_end"):
        output[column] = pd.to_datetime(output[column], errors="raise")
    output["observations"] = pd.to_numeric(
        output["observations"], errors="raise"
    ).astype(np.int64)
    for column in ("scaler_mean", "scaler_scale"):
        output[column] = output[column].map(_vector)
    for column in ("lambda", "objective"):
        output[column] = pd.to_numeric(output[column], errors="raise").astype(float)
    return output


def _vector(value: object) -> tuple[float, ...]:
    parsed = ast.literal_eval(value) if isinstance(value, str) else value
    if not isinstance(parsed, (list, tuple)) or len(parsed) != len(FEATURE_COLUMNS):
        raise EndpointGridError("base JM refit vector changed")
    return tuple(float(item) for item in parsed)


def _choices(selection: SelectionResult) -> pd.DataFrame:
    frame = selection.choices.copy()
    frame["decision_date"] = pd.to_datetime(frame["decision_date"], errors="raise")
    return frame.reset_index(drop=True)


def _surface(selection: SelectionResult) -> pd.DataFrame:
    frame = selection.surface.copy()
    frame["decision_date"] = pd.to_datetime(frame["decision_date"], errors="raise")
    return frame.reset_index(drop=True)


def _candidate_returns(selection: SelectionResult) -> pd.DataFrame:
    return selection.candidate_returns.rename_axis("date").reset_index()


def _signal(selection: SelectionResult) -> pd.DataFrame:
    return selection.signal.rename_axis("date").to_frame().reset_index()


def _evidence_frame(path: Path, expected: pd.DataFrame) -> pd.DataFrame:
    observed = _read_csv(path)
    if tuple(str(value) for value in observed.columns) != tuple(
        str(value) for value in expected.columns
    ):
        raise EndpointGridError(f"behavior witness schema changed: {path}")
    observed.columns = expected.columns
    for column in expected.columns:
        if pd.api.types.is_datetime64_any_dtype(expected[column]):
            observed[column] = pd.to_datetime(observed[column], errors="raise")
    return observed


def _base_boundaries(boundaries: pd.DataFrame) -> pd.DataFrame:
    mapping = {"J0": "fixed_jm", "K0": "hmm"}
    output = boundaries.loc[boundaries["path"].isin(mapping)].copy()
    output["model"] = output.pop("path").map(mapping)
    output = output.drop(columns=["descriptive_only"])
    columns = (
        "model",
        "delay",
        "upper_candidate",
        "selected_months",
        "total_months",
        "fraction",
        "limit",
        "passed",
    )
    return output.loc[:, columns].reset_index(drop=True)


def _frame_hash(frame: pd.DataFrame, *, index: bool = True) -> str:
    payload = frame.to_csv(index=index, lineterminator="\n").encode()
    return hashlib.sha256(payload).hexdigest()


def _assert_exact(observed: pd.DataFrame, expected: pd.DataFrame, label: str) -> None:
    try:
        pd.testing.assert_frame_equal(
            observed,
            expected,
            check_dtype=False,
            check_exact=True,
            check_freq=False,
            check_column_type=False,
            check_index_type=False,
        )
    except AssertionError as exc:
        raise EndpointGridError(
            f"{label} do not exactly match selection-behavior witness"
        ) from exc
