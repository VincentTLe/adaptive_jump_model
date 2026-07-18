"""Causal refit geometry and exact discount-event extraction."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_jump.models import FEATURE_COLUMNS
from adaptive_jump.separation_study import (
    SeparationSpec,
    SeparationStudyError,
    arrival_ablation_state,
    reliability_from_geometry,
    terminal_decision,
)
from adaptive_jump.tv_jump import (
    dp_tv,
    evidence_penalty_seq,
    lam_to_penalty_seq,
    loss_matrix,
    robust_loss_scale,
)

BETA_FILES = {
    0.0: "candidate-states-beta-0.csv",
    math.log(2.0): "candidate-states-beta-log2.csv",
    math.log(4.0): "candidate-states-beta-log4.csv",
}


@dataclass(frozen=True)
class MarketInputs:
    market: str
    features: pd.DataFrame
    model_dates: pd.DatetimeIndex
    candidates: dict[float, pd.DataFrame]
    refits: pd.DataFrame


@dataclass(frozen=True)
class MarketAnalysis:
    market: str
    separation: pd.DataFrame
    events: pd.DataFrame
    audit: dict[str, int]


def _read_candidates(path: Path, spec: SeparationSpec) -> pd.DataFrame:
    names = ["date", "0.0", *(str(value) for value in spec.lambdas)]
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
    spec: SeparationSpec,
    *,
    include_fixed_objective: bool = True,
) -> MarketInputs:
    """Read only the frozen allowlisted columns and reconstruct model dates."""
    if market not in spec.markets:
        raise SeparationStudyError(f"unknown separation market: {market}")
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
    expected_lambdas = {0.0, *spec.lambdas}
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


def compute_refit_separation(
    inputs: MarketInputs, spec: SeparationSpec
) -> pd.DataFrame:
    """Reconstruct every positive-lambda objective and compute R_train."""
    positions = {date: index for index, date in enumerate(inputs.model_dates)}
    records: list[dict[str, object]] = []
    positive = inputs.refits.loc[inputs.refits["lambda0"].isin(spec.lambdas)]
    for _, row in positive.iterrows():
        fit_date = pd.Timestamp(row["fit_date"])
        end = positions.get(fit_date)
        if end is None or end + 1 < spec.fit_window:
            raise SeparationStudyError(f"{inputs.market}/{fit_date}: refit not mapped")
        dates = inputs.model_dates[end - spec.fit_window + 1 : end + 1]
        if (
            dates[0] != row["training_start"]
            or dates[-1] != row["training_end"]
            or len(dates) != spec.fit_window
        ):
            raise SeparationStudyError(
                f"{inputs.market}/{fit_date}: training prefix changed"
            )
        raw = inputs.features.loc[dates, FEATURE_COLUMNS].to_numpy(dtype=float)
        mean, scale, centers = _decode_parameters(row)
        scaled = (raw - mean) / scale
        losses = loss_matrix(scaled, centers)
        q_reconstructed = robust_loss_scale(losses)
        if not math.isclose(
            q_reconstructed, float(row["q_train"]), rel_tol=0.0, abs_tol=1e-12
        ):
            raise SeparationStudyError(
                f"{inputs.market}/{fit_date}/{row['lambda0']:g}: q_train changed"
            )
        penalty = lam_to_penalty_seq(np.full(spec.fit_window, float(row["lambda0"])), 2)
        _, objective = dp_tv(losses, penalty)
        objective_error = abs(float(objective) - float(row["fixed_objective"]))
        if not np.isfinite(objective) or objective_error > spec.objective_tolerance:
            raise SeparationStudyError(
                f"{inputs.market}/{fit_date}/{row['lambda0']:g}: objective changed"
            )
        result = reliability_from_geometry(scaled, centers)
        records.append(
            {
                "market": inputs.market,
                "fit_date": fit_date,
                "training_start": dates[0],
                "training_end": dates[-1],
                "lambda0": float(row["lambda0"]),
                "q_train": float(row["q_train"]),
                "q_train_reconstructed": q_reconstructed,
                "fixed_objective": float(row["fixed_objective"]),
                "fixed_objective_reconstructed": float(objective),
                "objective_abs_error": objective_error,
                "center_distance": result.center_distance,
                "preferred_count_0": result.preferred_count_0,
                "preferred_count_1": result.preferred_count_1,
                "tie_count": result.tie_count,
                "median_radius_0": result.median_radius_0,
                "median_radius_1": result.median_radius_1,
                "reliability_valid": result.valid,
                "reliability_train": result.reliability_train,
            }
        )
    result = pd.DataFrame.from_records(records).sort_values(["lambda0", "fit_date"])
    result["next_fit_date"] = result.groupby("lambda0")["fit_date"].shift(-1)
    return result.sort_values(["fit_date", "lambda0"]).reset_index(drop=True)


def _refit_for_date(rows: pd.DataFrame, current: pd.Timestamp) -> pd.Series:
    eligible = rows.loc[rows["fit_date"] <= current]
    if eligible.empty:
        raise SeparationStudyError(f"no refit available for {current.date()}")
    return eligible.iloc[-1]


def extract_discount_events(
    inputs: MarketInputs,
    separation: pd.DataFrame,
    spec: SeparationSpec,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Extract locally necessary arrival-discount events at fixed lambda."""
    positions = {date: index for index, date in enumerate(inputs.model_dates)}
    refit_source = inputs.refits.set_index(["fit_date", "lambda0"], drop=False)
    separation_by_lambda = {
        value: rows.sort_values("fit_date").reset_index(drop=True)
        for value, rows in separation.groupby("lambda0")
    }
    counters = {
        "candidate_divergences": 0,
        "horizon_censored": 0,
        "state_reconstructions": 0,
        "terminal_tie_exclusions": 0,
        "terminal_transition_matches": 0,
        "discounted_terminal_transitions": 0,
        "ablation_attributable": 0,
        "admitted_events": 0,
        "invalid_reliability_events": 0,
    }
    records: list[dict[str, object]] = []

    for beta in spec.betas:
        adaptive_frame = inputs.candidates[beta]
        baseline_frame = inputs.candidates[0.0]
        for lambda0 in spec.lambdas:
            adaptive = adaptive_frame[lambda0].dropna().astype(int)
            baseline = baseline_frame[lambda0].reindex(adaptive.index).astype(int)
            refit_rows = separation_by_lambda[lambda0]
            start = pd.Timestamp(spec.evaluation_starts[inputs.market])
            first_eligible = int(adaptive.index.searchsorted(start, side="left"))
            suppress_through = -1
            for index in range(max(1, first_eligible), len(adaptive)):
                if index <= suppress_through:
                    continue
                source = int(adaptive.iloc[index - 1])
                destination = int(adaptive.iloc[index])
                if (
                    source != int(baseline.iloc[index - 1])
                    or destination == source
                    or int(baseline.iloc[index]) != source
                ):
                    continue
                counters["candidate_divergences"] += 1
                if index + spec.horizon >= len(adaptive):
                    counters["horizon_censored"] += 1
                    continue
                signal_date = pd.Timestamp(adaptive.index[index])
                horizon_end = pd.Timestamp(adaptive.index[index + spec.horizon])
                refit = _refit_for_date(refit_rows, signal_date)
                horizon_refit = _refit_for_date(refit_rows, horizon_end)
                if horizon_refit["fit_date"] != refit["fit_date"]:
                    counters["horizon_censored"] += 1
                    continue

                terminal = positions.get(signal_date)
                if terminal is None or terminal + 1 < spec.fit_window:
                    raise SeparationStudyError(
                        f"{inputs.market}/{signal_date}: model window not mapped"
                    )
                dates = inputs.model_dates[
                    terminal - spec.fit_window + 1 : terminal + 1
                ]
                source_row = refit_source.loc[
                    (pd.Timestamp(refit["fit_date"]), float(lambda0))
                ]
                mean, scale, centers = _decode_parameters(source_row)
                raw = inputs.features.loc[dates, FEATURE_COLUMNS].to_numpy(dtype=float)
                scaled = (raw - mean) / scale
                losses = loss_matrix(scaled, centers)
                q_train = float(source_row["q_train"])
                adaptive_penalty = evidence_penalty_seq(
                    losses, lambda0=lambda0, beta=beta, q_train=q_train
                )
                fixed_penalty = lam_to_penalty_seq(np.full(spec.fit_window, lambda0), 2)
                adaptive_decision = terminal_decision(losses, adaptive_penalty)
                fixed_decision = terminal_decision(losses, fixed_penalty)
                counters["state_reconstructions"] += 1
                if (
                    adaptive_decision.state != destination
                    or fixed_decision.state != int(baseline.iloc[index])
                ):
                    raise SeparationStudyError(
                        f"{inputs.market}/{signal_date}/{beta:g}/{lambda0:g}: "
                        "candidate state changed"
                    )
                if adaptive_decision.state_tied or adaptive_decision.predecessor_tied:
                    counters["terminal_tie_exclusions"] += 1
                    continue
                if adaptive_decision.predecessor != source:
                    continue
                counters["terminal_transition_matches"] += 1
                transition_penalty = float(adaptive_penalty[-1, source, destination])
                reverse_penalty = float(adaptive_penalty[-1, destination, source])
                if not transition_penalty < lambda0:
                    continue
                if reverse_penalty != lambda0:
                    raise SeparationStudyError("discounted reverse penalty changed")
                counters["discounted_terminal_transitions"] += 1

                ablated_penalty = adaptive_penalty.copy()
                ablated_penalty[-1] = lambda0 * (1.0 - np.eye(2))
                ablated_decision = terminal_decision(losses, ablated_penalty)
                if ablated_decision.state_tied:
                    counters["terminal_tie_exclusions"] += 1
                    continue
                ablated_state = arrival_ablation_state(
                    losses, adaptive_penalty, lambda0
                )
                if ablated_state != ablated_decision.state:
                    raise SeparationStudyError(
                        "arrival ablation implementations differ"
                    )
                if ablated_state != source:
                    continue
                counters["ablation_attributable"] += 1

                future_adaptive = adaptive.iloc[
                    index + 1 : index + spec.horizon + 1
                ].to_numpy(dtype=int)
                future_baseline = baseline.iloc[
                    index + 1 : index + spec.horizon + 1
                ].to_numpy(dtype=int)
                reversals = np.flatnonzero(future_adaptive == source)
                confirmations = np.flatnonzero(future_baseline == destination)
                reliability_valid = bool(refit["reliability_valid"])
                records.append(
                    {
                        "market": inputs.market,
                        "beta": beta,
                        "beta_label": "log2" if beta == math.log(2.0) else "log4",
                        "lambda0": lambda0,
                        "signal_date": signal_date,
                        "fit_date": pd.Timestamp(refit["fit_date"]),
                        "source_state": source,
                        "destination_state": destination,
                        "loss_source": float(losses[-1, source]),
                        "loss_destination": float(losses[-1, destination]),
                        "q_train": q_train,
                        "normalized_gap": float(
                            (losses[-1, source] - losses[-1, destination]) / q_train
                        ),
                        "transition_penalty": transition_penalty,
                        "reverse_penalty": reverse_penalty,
                        "log_discount": float(math.log(lambda0 / transition_penalty)),
                        "terminal_predecessor": adaptive_decision.predecessor,
                        "terminal_state_margin": adaptive_decision.state_margin,
                        "terminal_predecessor_margin": (
                            adaptive_decision.predecessor_margin
                        ),
                        "ablated_state": ablated_state,
                        "ablated_state_margin": ablated_decision.state_margin,
                        "discount_attributable": True,
                        "horizon_signal_days": spec.horizon,
                        "persistent_20": len(reversals) == 0,
                        "whipsaw_20": len(reversals) > 0,
                        "first_reversal_h": (
                            int(reversals[0] + 1) if len(reversals) else math.nan
                        ),
                        "fixed_confirmation_h": (
                            int(confirmations[0] + 1)
                            if len(confirmations)
                            else math.nan
                        ),
                        "reliability_valid": reliability_valid,
                        "reliability_train": float(refit["reliability_train"]),
                        "center_distance": float(refit["center_distance"]),
                        "preferred_count_0": int(refit["preferred_count_0"]),
                        "preferred_count_1": int(refit["preferred_count_1"]),
                    }
                )
                suppress_through = index + spec.horizon
                counters["admitted_events"] += 1
                if not reliability_valid:
                    counters["invalid_reliability_events"] += 1

    columns = [
        "market",
        "beta",
        "beta_label",
        "lambda0",
        "signal_date",
        "fit_date",
        "source_state",
        "destination_state",
        "loss_source",
        "loss_destination",
        "q_train",
        "normalized_gap",
        "transition_penalty",
        "reverse_penalty",
        "log_discount",
        "terminal_predecessor",
        "terminal_state_margin",
        "terminal_predecessor_margin",
        "ablated_state",
        "ablated_state_margin",
        "discount_attributable",
        "horizon_signal_days",
        "persistent_20",
        "whipsaw_20",
        "first_reversal_h",
        "fixed_confirmation_h",
        "reliability_valid",
        "reliability_train",
        "center_distance",
        "preferred_count_0",
        "preferred_count_1",
    ]
    events = pd.DataFrame.from_records(records, columns=columns)
    return events, counters


def analyze_market(
    market: str,
    feature_path: Path,
    adaptive_market_dir: Path,
    spec: SeparationSpec,
) -> MarketAnalysis:
    inputs = load_market_inputs(market, feature_path, adaptive_market_dir, spec)
    separation = compute_refit_separation(inputs, spec)
    events, audit = extract_discount_events(inputs, separation, spec)
    return MarketAnalysis(
        market=market,
        separation=separation,
        events=events,
        audit=audit,
    )
