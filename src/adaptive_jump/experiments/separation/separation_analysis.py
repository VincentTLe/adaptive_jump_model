"""Shared sealed-input loading for the evidence-penalty studies.

Only the reader the arrival, lagged and balanced harnesses import survives from
adaptive-separation-001: it opens a sealed feature file plus a sealed arrival
run's candidate states and refit table, checks them against the frozen scope,
and reconstructs the model dates.

The spec argument is structural, not a class: any object exposing ``markets``,
``lambdas`` (this market's full ordered candidate grid), ``data_cutoff`` and
``fit_window`` works. ``lambdas`` used to be the positive-only global grid with
a literal ``"0.0"`` column prepended, which is wrong for a calibrated contract
whose per-market grid need not contain zero at all.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from adaptive_jump.experiments.separation.separation_study import SeparationStudyError
from adaptive_jump.models import FEATURE_COLUMNS

BETA_FILES = {
    0.0: "candidate-states-beta-0.csv",
    math.log(2.0): "candidate-states-beta-log2.csv",
    math.log(4.0): "candidate-states-beta-log4.csv",
}


class MarketInputSpec(Protocol):
    """The frozen scope one market's sealed inputs are read against."""

    markets: tuple[str, ...]
    lambdas: tuple[float, ...]
    data_cutoff: Any
    fit_window: int


@dataclass(frozen=True)
class MarketInputs:
    """Sealed features, candidate states and refits for one market."""

    market: str
    features: pd.DataFrame
    model_dates: pd.DatetimeIndex
    candidates: dict[float, pd.DataFrame]
    refits: pd.DataFrame


def _read_candidates(path: Path, spec: MarketInputSpec) -> pd.DataFrame:
    # The market's own grid, in contract order, is exactly the column set a
    # sealed arrival run writes for that market.
    names = ["date", *(str(value) for value in spec.lambdas)]
    frame = pd.read_csv(path, usecols=names)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame["date"].duplicated().any() or not frame["date"].is_monotonic_increasing:
        raise SeparationStudyError(f"candidate dates changed: {path}")
    if frame["date"].max().date() > spec.data_cutoff:
        raise SeparationStudyError(f"post-cutoff candidate state: {path}")
    frame = frame.set_index("date")
    frame.columns = tuple(float(column) for column in frame.columns)
    values = frame.to_numpy(dtype=float)
    present = np.isfinite(values)
    if not np.array_equal(present.any(axis=1), present.all(axis=1)):
        raise SeparationStudyError(f"partial candidate state row: {path}")
    observed = values[present]
    if observed.size and not np.isin(observed, (0.0, 1.0)).all():
        raise SeparationStudyError(f"non-binary candidate state: {path}")
    return frame


def load_market_inputs(
    market: str,
    feature_path: Path,
    adaptive_market_dir: Path,
    spec: MarketInputSpec,
    *,
    include_fixed_objective: bool = True,
) -> MarketInputs:
    """Read only the frozen allowlisted columns and reconstruct model dates."""
    if market not in spec.markets:
        raise SeparationStudyError(f"unknown market: {market}")
    features = pd.read_csv(feature_path, usecols=["date", *FEATURE_COLUMNS])
    features["date"] = pd.to_datetime(features["date"], errors="raise")
    if (
        features["date"].duplicated().any()
        or not features["date"].is_monotonic_increasing
        or features["date"].max().date() > spec.data_cutoff
    ):
        raise SeparationStudyError(f"{market}: feature dates changed")
    for column in FEATURE_COLUMNS:
        features[column] = pd.to_numeric(features[column], errors="raise")
    observed = features.loc[:, FEATURE_COLUMNS].dropna().to_numpy(dtype=float)
    if not np.isfinite(observed).all():
        raise SeparationStudyError(f"{market}: nonfinite observed feature")
    features = features.set_index("date")

    candidates = {
        beta: _read_candidates(adaptive_market_dir / filename, spec)
        for beta, filename in BETA_FILES.items()
    }
    indexes = [frame.index for frame in candidates.values()]
    if any(not index.equals(features.index) for index in indexes):
        raise SeparationStudyError(f"{market}: candidate and feature dates differ")
    validity = [frame.notna().all(axis=1) for frame in candidates.values()]
    if any(not mask.equals(validity[0]) for mask in validity[1:]):
        raise SeparationStudyError(f"{market}: candidate coverage differs by beta")

    refit_columns = [
        "market",
        "fit_date",
        "training_start",
        "training_end",
        "lambda0",
        "q_train",
        "scaler_mean",
        "scaler_scale",
        "centers",
    ]
    numeric_columns = ["lambda0", "q_train"]
    if include_fixed_objective:
        refit_columns.append("fixed_objective")
        numeric_columns.append("fixed_objective")
    refits = pd.read_csv(
        adaptive_market_dir / "refits-and-scales.csv", usecols=refit_columns
    )
    for column in ("fit_date", "training_start", "training_end"):
        refits[column] = pd.to_datetime(refits[column], errors="raise")
    for column in numeric_columns:
        refits[column] = pd.to_numeric(refits[column], errors="raise")
    expected_lambdas = set(spec.lambdas)
    if (
        set(refits["market"]) != {market}
        or refits.duplicated(["fit_date", "lambda0"]).any()
        or set(refits["lambda0"]) != expected_lambdas
        or (refits["training_end"] != refits["fit_date"]).any()
        or refits["fit_date"].max().date() > spec.data_cutoff
        or not np.isfinite(refits[numeric_columns]).all().all()
        or (refits["q_train"] <= 0).any()
    ):
        raise SeparationStudyError(f"{market}: refit table changed")
    counts = refits.groupby("fit_date")["lambda0"].nunique()
    if not (counts == len(expected_lambdas)).all():
        raise SeparationStudyError(f"{market}: incomplete refit lambda grid")

    first_fit = refits["fit_date"].min()
    first_start = refits.loc[refits["fit_date"] == first_fit, "training_start"].unique()
    if len(first_start) != 1:
        raise SeparationStudyError(f"{market}: first training start differs by lambda")
    complete_features = features.dropna(subset=list(FEATURE_COLUMNS))
    initial = complete_features.loc[pd.Timestamp(first_start[0]) : first_fit]
    if len(initial) != spec.fit_window:
        raise SeparationStudyError(f"{market}: initial feature prefix is not exact")
    state_dates = candidates[0.0].index[validity[0]]
    if len(state_dates) == 0 or state_dates[0] != first_fit:
        raise SeparationStudyError(f"{market}: candidate terminal start changed")
    if features.loc[state_dates, FEATURE_COLUMNS].isna().any().any():
        raise SeparationStudyError(f"{market}: candidate date has missing feature")
    model_dates = initial.index.append(state_dates[state_dates > first_fit])
    if model_dates.has_duplicates or not model_dates.is_monotonic_increasing:
        raise SeparationStudyError(f"{market}: reconstructed model dates invalid")
    return MarketInputs(
        market=market,
        features=features,
        model_dates=model_dates,
        candidates=candidates,
        refits=refits.sort_values(["fit_date", "lambda0"]).reset_index(drop=True),
    )


def _decode_parameters(row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        mean = np.asarray(json.loads(row["scaler_mean"]), dtype=float)
        scale = np.asarray(json.loads(row["scaler_scale"]), dtype=float)
        centers = np.asarray(json.loads(row["centers"]), dtype=float)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SeparationStudyError("stored refit parameters are invalid") from exc
    if (
        mean.shape != (len(FEATURE_COLUMNS),)
        or scale.shape != mean.shape
        or centers.shape != (2, len(FEATURE_COLUMNS))
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or (scale <= 0).any()
    ):
        raise SeparationStudyError("stored scaler or center shape changed")
    return mean, scale, centers


def _refit_for_date(rows: pd.DataFrame, current: pd.Timestamp) -> pd.Series:
    eligible = rows.loc[rows["fit_date"] <= current]
    if eligible.empty:
        raise SeparationStudyError(f"no refit available for {current.date()}")
    return eligible.iloc[-1]
