"""Independent selection-behavior parity replay for endpoint-grid artifacts."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from adaptive_jump.artifacts import ArtifactError, sha256_file
from adaptive_jump.endpoint_grid_types import MarketSource
from adaptive_jump.models import FEATURE_COLUMNS
from adaptive_jump.walkforward import SelectionResult

REPLAY_BASE_SELECTIONS = (("fixed_jm", "J0"), ("hmm", "K0"))
REPLAY_REFIT_COLUMNS = (
    "fit_date",
    "training_start",
    "training_end",
    "observations",
    "scaler_mean",
    "scaler_scale",
    "lambda",
    "objective",
)


def replay_state_parity(
    source: MarketSource,
    witness_dir: Path,
    base_jm: pd.DataFrame,
    base_refits: pd.DataFrame,
    base_hmm: pd.DataFrame,
    current_git_sha: str,
) -> dict[str, Any]:
    """Independently require complete base states and refits to match."""
    if len(base_jm.columns) != 9 or len(base_hmm.columns) != 9:
        raise ArtifactError("parity replay requires nine base candidates")
    paths = {
        "base_jm_states": witness_dir / "jm-states.csv",
        "base_jm_refits": witness_dir / "jm-refits.csv",
        "base_hmm_candidates": witness_dir / "hmm-candidates.csv",
    }
    observed_jm = _replay_state_frame(paths["base_jm_states"], base_jm)
    observed_hmm = _replay_state_frame(paths["base_hmm_candidates"], base_hmm)
    observed_refits = _replay_refits(_replay_read(paths["base_jm_refits"]))
    current_refits = _replay_refits(base_refits.copy())
    _replay_assert(observed_jm, base_jm, "J0 candidate states")
    _replay_assert(observed_hmm, base_hmm, "K0 candidate states")
    _replay_assert(observed_refits, current_refits, "J0 refits")
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
            "base_jm_states": _replay_hash(base_jm),
            "base_jm_refits": _replay_hash(current_refits, index=False),
            "base_hmm_candidates": _replay_hash(base_hmm),
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


def replay_selection_parity(
    receipt: dict[str, Any],
    witness_dir: Path,
    selections: dict[str, dict[int, SelectionResult]],
    boundaries: pd.DataFrame,
    delays: tuple[int, ...],
) -> dict[str, Any]:
    """Independently replay all available base selection behavior."""
    output = {
        **receipt,
        "artifact_hashes": dict(receipt["artifact_hashes"]),
        "current_hashes": dict(receipt["current_hashes"]),
        "counts": dict(receipt["counts"]),
    }
    for witness_model, path in REPLAY_BASE_SELECTIONS:
        for delay in delays:
            selection = selections[path][delay]
            prefix = f"{witness_model}_delay_{delay}"
            directory = witness_dir / f"{witness_model}-delay-{delay}"
            frames = {
                "choices": _replay_choices(selection),
                "cv_surface": _replay_surface(selection),
                "candidate_returns": _replay_returns(selection),
                "selected_signal": _replay_signal(selection),
            }
            files = {
                "choices": "choices.csv",
                "cv_surface": "cv-surface.csv",
                "candidate_returns": "candidate-returns.csv",
                "selected_signal": "selected-signal.csv",
            }
            for component, expected in frames.items():
                artifact = directory / files[component]
                observed = _replay_evidence(artifact, expected)
                _replay_assert(observed, expected, f"{path} {component}")
                key = f"{prefix}_{component}"
                output["artifact_hashes"][key] = sha256_file(artifact)
                output["current_hashes"][key] = _replay_hash(expected, index=False)
                output["counts"][f"{key}_rows"] = len(expected)
    boundary_path = witness_dir / "boundaries.csv"
    current = _replay_base_boundaries(boundaries)
    observed = _replay_evidence(boundary_path, current)
    _replay_assert(observed, current, "base boundaries")
    output["artifact_hashes"]["base_boundaries"] = sha256_file(boundary_path)
    output["current_hashes"]["base_boundaries"] = _replay_hash(current, index=False)
    output["counts"]["base_boundary_rows"] = len(current)
    output["selection_behavior_parity_passed"] = True
    output["passed"] = True
    return output


def _replay_read(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, float_precision="round_trip")
    except (FileNotFoundError, OSError, pd.errors.ParserError) as exc:
        raise ArtifactError(f"cannot read selection-behavior witness {path}") from exc


def _replay_state_frame(path: Path, expected: pd.DataFrame) -> pd.DataFrame:
    frame = _replay_read(path)
    if frame.empty or frame.columns[0] != "date":
        raise ArtifactError("state behavior witness schema changed")
    dates = pd.DatetimeIndex(
        pd.to_datetime(frame.pop("date"), errors="raise"), name="date"
    )
    try:
        candidates = tuple(float(value) for value in frame.columns)
    except ValueError as exc:
        raise ArtifactError("state behavior witness candidates changed") from exc
    if candidates != tuple(float(value) for value in expected.columns):
        raise ArtifactError("state behavior witness candidates changed")
    frame = frame.apply(pd.to_numeric, errors="raise")
    frame.columns = expected.columns
    frame.index = dates
    return frame


def _replay_refits(frame: pd.DataFrame) -> pd.DataFrame:
    if tuple(frame.columns) != REPLAY_REFIT_COLUMNS or frame.empty:
        raise ArtifactError("base JM refit replay schema changed")
    output = frame.copy()
    for column in ("fit_date", "training_start", "training_end"):
        output[column] = pd.to_datetime(output[column], errors="raise")
    output["observations"] = pd.to_numeric(
        output["observations"], errors="raise"
    ).astype(np.int64)
    for column in ("scaler_mean", "scaler_scale"):
        output[column] = output[column].map(_replay_vector)
    for column in ("lambda", "objective"):
        output[column] = pd.to_numeric(output[column], errors="raise").astype(float)
    return output


def _replay_vector(value: object) -> tuple[float, ...]:
    parsed = ast.literal_eval(value) if isinstance(value, str) else value
    if not isinstance(parsed, (list, tuple)) or len(parsed) != len(FEATURE_COLUMNS):
        raise ArtifactError("base JM refit replay vector changed")
    return tuple(float(item) for item in parsed)


def _replay_choices(selection: SelectionResult) -> pd.DataFrame:
    output = selection.choices.copy()
    output["decision_date"] = pd.to_datetime(output["decision_date"], errors="raise")
    return output.reset_index(drop=True)


def _replay_surface(selection: SelectionResult) -> pd.DataFrame:
    output = selection.surface.copy()
    output["decision_date"] = pd.to_datetime(output["decision_date"], errors="raise")
    return output.reset_index(drop=True)


def _replay_returns(selection: SelectionResult) -> pd.DataFrame:
    return selection.candidate_returns.rename_axis("date").reset_index()


def _replay_signal(selection: SelectionResult) -> pd.DataFrame:
    return selection.signal.rename_axis("date").to_frame().reset_index()


def _replay_evidence(path: Path, expected: pd.DataFrame) -> pd.DataFrame:
    observed = _replay_read(path)
    if tuple(str(value) for value in observed.columns) != tuple(
        str(value) for value in expected.columns
    ):
        raise ArtifactError(f"selection-behavior witness schema changed: {path}")
    observed.columns = expected.columns
    for column in expected.columns:
        if pd.api.types.is_datetime64_any_dtype(expected[column]):
            observed[column] = pd.to_datetime(observed[column], errors="raise")
    return observed


def _replay_base_boundaries(boundaries: pd.DataFrame) -> pd.DataFrame:
    mapping = {"J0": "fixed_jm", "K0": "hmm"}
    output = boundaries.loc[boundaries["path"].isin(mapping)].copy()
    output["model"] = output.pop("path").map(mapping)
    output = output.drop(columns=["descriptive_only"])
    return output.loc[
        :,
        (
            "model",
            "delay",
            "upper_candidate",
            "selected_months",
            "total_months",
            "fraction",
            "limit",
            "passed",
        ),
    ].reset_index(drop=True)


def _replay_hash(frame: pd.DataFrame, *, index: bool = True) -> str:
    return hashlib.sha256(
        frame.to_csv(index=index, lineterminator="\n").encode()
    ).hexdigest()


def _replay_assert(observed: pd.DataFrame, expected: pd.DataFrame, label: str) -> None:
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
        raise ArtifactError(
            f"{label} do not exactly match selection-behavior witness"
        ) from exc
