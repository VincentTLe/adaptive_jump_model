"""Candidate-path and event analysis for the lagged-evidence mechanism study."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from adaptive_jump.lagged_study import (
    LaggedMechanismSpec,
    LaggedStudyError,
    beta_label,
)
from adaptive_jump.models import FEATURE_COLUMNS
from adaptive_jump.separation_analysis import (
    MarketInputs,
    _decode_parameters,
    _refit_for_date,
    load_market_inputs,
)
from adaptive_jump.separation_study import arrival_ablation_state, terminal_decision
from adaptive_jump.tv_jump import (
    evidence_penalty_seq,
    lagged_evidence_penalty_seq,
    lam_to_penalty_seq,
    loss_matrix,
)


@dataclass(frozen=True)
class _InputSpec:
    markets: tuple[str, ...]
    lambdas: tuple[float, ...]
    data_cutoff: Any
    fit_window: int


@dataclass(frozen=True)
class MechanismAnalysis:
    market: str
    behavior: pd.DataFrame
    events: pd.DataFrame
    audit: dict[str, Any]


def _input_spec(spec: LaggedMechanismSpec) -> _InputSpec:
    return _InputSpec(
        markets=spec.markets,
        lambdas=spec.event_lambdas,
        data_cutoff=spec.data_cutoff,
        fit_window=spec.fit_window,
    )


def _validate_lagged_states(
    inputs: MarketInputs,
    states: dict[float, pd.DataFrame],
    spec: LaggedMechanismSpec,
) -> dict[float, pd.DataFrame]:
    if set(states) != set(spec.betas):
        raise LaggedStudyError(f"{inputs.market}: lagged beta set changed")
    validated: dict[float, pd.DataFrame] = {}
    for beta in spec.betas:
        frame = states[beta].copy()
        if not frame.index.equals(inputs.features.index):
            raise LaggedStudyError(f"{inputs.market}/{beta:g}: lagged dates changed")
        if tuple(float(column) for column in frame.columns) != spec.lambdas:
            raise LaggedStudyError(f"{inputs.market}/{beta:g}: lagged grid changed")
        frame.columns = spec.lambdas
        values = frame.to_numpy(dtype=float)
        present = np.isfinite(values)
        if not np.array_equal(present.any(axis=1), present.all(axis=1)):
            raise LaggedStudyError(
                f"{inputs.market}/{beta:g}: partial lagged candidate row"
            )
        observed = values[present]
        if observed.size and not np.isin(observed, (0.0, 1.0)).all():
            raise LaggedStudyError(
                f"{inputs.market}/{beta:g}: nonbinary lagged candidate state"
            )
        validated[beta] = frame

    fixed = inputs.candidates[0.0].reindex(columns=spec.lambdas)
    if not np.array_equal(
        validated[0.0].to_numpy(),
        fixed.to_numpy(),
        equal_nan=True,
    ):
        raise LaggedStudyError(f"{inputs.market}: lagged beta zero differs from fixed")
    return validated


def _path_behavior(
    inputs: MarketInputs,
    candidates: dict[float, pd.DataFrame],
    rule: str,
    spec: LaggedMechanismSpec,
) -> list[dict[str, Any]]:
    fixed = inputs.candidates[0.0]
    start = pd.Timestamp(spec.evaluation_starts[inputs.market])
    records: list[dict[str, Any]] = []
    for beta in spec.event_betas:
        for lambda0 in spec.event_lambdas:
            complete = candidates[beta][lambda0].dropna().astype(int)
            first = int(complete.index.searchsorted(start, side="left"))
            model = complete.iloc[first:]
            if model.empty:
                raise LaggedStudyError(
                    f"{inputs.market}/{rule}/{beta:g}/{lambda0:g}: empty outer path"
                )
            baseline = fixed[lambda0].reindex(model.index).astype(int)
            # Count transitions by their destination date, matching event admission.
            switch_values = complete.iloc[max(0, first - 1) :].to_numpy(dtype=int)
            records.append(
                {
                    "market": inputs.market,
                    "rule": rule,
                    "beta": beta,
                    "beta_label": beta_label(beta),
                    "lambda0": lambda0,
                    "start": model.index[0],
                    "end": model.index[-1],
                    "observations": len(model),
                    "switch_count": int(np.count_nonzero(np.diff(switch_values))),
                    "state_differences_vs_fixed": int(
                        (model.to_numpy() != baseline.to_numpy()).sum()
                    ),
                }
            )
    return records


def _penalty_builder(rule: str):
    if rule == "arrival":
        return evidence_penalty_seq
    if rule == "lagged":
        return lagged_evidence_penalty_seq
    raise LaggedStudyError(f"unknown mechanism rule: {rule}")


def _extract_events(
    inputs: MarketInputs,
    candidates: dict[float, pd.DataFrame],
    rule: str,
    spec: LaggedMechanismSpec,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    positions = {current: index for index, current in enumerate(inputs.model_dates)}
    refit_source = inputs.refits.set_index(["fit_date", "lambda0"], drop=False)
    refits_by_lambda = {
        value: rows.sort_values("fit_date").reset_index(drop=True)
        for value, rows in inputs.refits.groupby("lambda0")
        if value in spec.event_lambdas
    }
    builder = _penalty_builder(rule)
    counters: dict[str, Any] = {
        "candidate_divergences": 0,
        "horizon_censored": 0,
        "state_reconstructions": 0,
        "terminal_tie_exclusions": 0,
        "terminal_transition_matches": 0,
        "discounted_terminal_transitions": 0,
        "ablation_attributable": 0,
        "admitted_events": 0,
        "max_penalty_abs_error": 0.0,
    }
    records: list[dict[str, Any]] = []
    fixed_frame = inputs.candidates[0.0]

    for beta in spec.event_betas:
        model_frame = candidates[beta]
        for lambda0 in spec.event_lambdas:
            model = model_frame[lambda0].dropna().astype(int)
            fixed = fixed_frame[lambda0].reindex(model.index).astype(int)
            refit_rows = refits_by_lambda[lambda0]
            first = int(
                model.index.searchsorted(
                    pd.Timestamp(spec.evaluation_starts[inputs.market]), side="left"
                )
            )
            suppress_through = -1
            for index in range(max(1, first), len(model)):
                if index <= suppress_through:
                    continue
                source = int(model.iloc[index - 1])
                destination = int(model.iloc[index])
                if (
                    source != int(fixed.iloc[index - 1])
                    or destination == source
                    or int(fixed.iloc[index]) != source
                ):
                    continue
                counters["candidate_divergences"] += 1
                if index + spec.horizon >= len(model):
                    counters["horizon_censored"] += 1
                    continue

                signal_date = pd.Timestamp(model.index[index])
                horizon_end = pd.Timestamp(model.index[index + spec.horizon])
                refit = _refit_for_date(refit_rows, signal_date)
                horizon_refit = _refit_for_date(refit_rows, horizon_end)
                if horizon_refit["fit_date"] != refit["fit_date"]:
                    counters["horizon_censored"] += 1
                    continue

                terminal = positions.get(signal_date)
                if terminal is None or terminal + 1 < spec.fit_window:
                    raise LaggedStudyError(
                        f"{inputs.market}/{signal_date.date()}: model window not mapped"
                    )
                dates = inputs.model_dates[
                    terminal - spec.fit_window + 1 : terminal + 1
                ]
                row = refit_source.loc[
                    (pd.Timestamp(refit["fit_date"]), float(lambda0))
                ]
                mean, scale, centers = _decode_parameters(row)
                raw = inputs.features.loc[dates, FEATURE_COLUMNS].to_numpy(dtype=float)
                scaled = (raw - mean) / scale
                losses = loss_matrix(scaled, centers)
                q_train = float(row["q_train"])
                penalties = builder(losses, lambda0, beta, q_train)
                fixed_penalties = lam_to_penalty_seq(
                    np.full(spec.fit_window, lambda0), 2
                )
                model_decision = terminal_decision(losses, penalties)
                fixed_decision = terminal_decision(losses, fixed_penalties)
                counters["state_reconstructions"] += 1
                if model_decision.state != destination or fixed_decision.state != int(
                    fixed.iloc[index]
                ):
                    raise LaggedStudyError(
                        f"{inputs.market}/{rule}/{signal_date.date()}/"
                        f"{beta:g}/{lambda0:g}: candidate state changed"
                    )
                if model_decision.state_tied or model_decision.predecessor_tied:
                    counters["terminal_tie_exclusions"] += 1
                    continue
                if model_decision.predecessor != source:
                    continue
                counters["terminal_transition_matches"] += 1

                transition_penalty = float(penalties[-1, source, destination])
                reverse_penalty = float(penalties[-1, destination, source])
                if not transition_penalty < lambda0:
                    continue
                if reverse_penalty != lambda0:
                    raise LaggedStudyError("reverse transition penalty changed")
                counters["discounted_terminal_transitions"] += 1

                evidence_index = -1 if rule == "arrival" else -2
                gap = max(
                    float(
                        losses[evidence_index, source]
                        - losses[evidence_index, destination]
                    ),
                    0.0,
                )
                expected_penalty = lambda0 * math.exp(-beta * math.tanh(gap / q_train))
                penalty_error = abs(expected_penalty - transition_penalty)
                counters["max_penalty_abs_error"] = max(
                    float(counters["max_penalty_abs_error"]), penalty_error
                )
                if penalty_error > spec.numerical_tolerance:
                    raise LaggedStudyError("rule-specific transition formula changed")

                ablated = penalties.copy()
                ablated[-1] = lambda0 * (1.0 - np.eye(2))
                ablated_decision = terminal_decision(losses, ablated)
                if ablated_decision.state_tied:
                    counters["terminal_tie_exclusions"] += 1
                    continue
                ablated_state = arrival_ablation_state(losses, penalties, lambda0)
                if ablated_state != ablated_decision.state:
                    raise LaggedStudyError("local ablation implementations differ")
                if ablated_state != source:
                    continue
                counters["ablation_attributable"] += 1

                future_model = model.iloc[
                    index + 1 : index + spec.horizon + 1
                ].to_numpy(dtype=int)
                future_fixed = fixed.iloc[
                    index + 1 : index + spec.horizon + 1
                ].to_numpy(dtype=int)
                reversals = np.flatnonzero(future_model == source)
                confirmations = np.flatnonzero(future_fixed == destination)
                persistent = len(reversals) == 0
                records.append(
                    {
                        "market": inputs.market,
                        "rule": rule,
                        "beta": beta,
                        "beta_label": beta_label(beta),
                        "lambda0": lambda0,
                        "signal_date": signal_date,
                        "evidence_date": pd.Timestamp(dates[evidence_index]),
                        "fit_date": pd.Timestamp(refit["fit_date"]),
                        "source_state": source,
                        "destination_state": destination,
                        "loss_source": float(losses[evidence_index, source]),
                        "loss_destination": float(losses[evidence_index, destination]),
                        "q_train": q_train,
                        "normalized_gap": gap / q_train,
                        "transition_penalty": transition_penalty,
                        "expected_transition_penalty": expected_penalty,
                        "reverse_penalty": reverse_penalty,
                        "terminal_predecessor": model_decision.predecessor,
                        "terminal_state_margin": model_decision.state_margin,
                        "terminal_predecessor_margin": (
                            model_decision.predecessor_margin
                        ),
                        "ablated_state": ablated_state,
                        "ablated_state_margin": ablated_decision.state_margin,
                        "discount_attributable": True,
                        "horizon_signal_days": spec.horizon,
                        "persistent_20": persistent,
                        "whipsaw_20": not persistent,
                        "first_reversal_h": (
                            int(reversals[0] + 1) if len(reversals) else math.nan
                        ),
                        "fixed_confirmation_h": (
                            int(confirmations[0] + 1)
                            if len(confirmations)
                            else math.nan
                        ),
                        "confirmed_early": persistent and len(confirmations) > 0,
                    }
                )
                suppress_through = index + spec.horizon
                counters["admitted_events"] += 1

    columns = [
        "market",
        "rule",
        "beta",
        "beta_label",
        "lambda0",
        "signal_date",
        "evidence_date",
        "fit_date",
        "source_state",
        "destination_state",
        "loss_source",
        "loss_destination",
        "q_train",
        "normalized_gap",
        "transition_penalty",
        "expected_transition_penalty",
        "reverse_penalty",
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
        "confirmed_early",
    ]
    return pd.DataFrame.from_records(records, columns=columns), counters


def analyze_market_mechanism(
    market: str,
    fixed_feature_path: Path,
    arrival_market_dir: Path,
    lagged_states: dict[float, pd.DataFrame],
    spec: LaggedMechanismSpec,
) -> MechanismAnalysis:
    """Compare fixed, sealed arrival, and generated lagged candidate paths."""
    try:
        inputs = load_market_inputs(
            market,
            fixed_feature_path,
            arrival_market_dir,
            _input_spec(spec),
            include_fixed_objective=False,
        )
    except Exception as exc:
        if isinstance(exc, LaggedStudyError):
            raise
        raise LaggedStudyError(f"{market}: source input reconstruction failed") from exc
    lagged = _validate_lagged_states(inputs, lagged_states, spec)
    candidates = {"arrival": inputs.candidates, "lagged": lagged}

    behavior = pd.DataFrame.from_records(
        [
            row
            for rule in spec.rules
            for row in _path_behavior(inputs, candidates[rule], rule, spec)
        ]
    )
    event_frames: list[pd.DataFrame] = []
    audit: dict[str, Any] = {}
    for rule in spec.rules:
        events, counters = _extract_events(inputs, candidates[rule], rule, spec)
        event_frames.append(events)
        audit[rule] = counters
    events = pd.concat(event_frames, ignore_index=True)
    return MechanismAnalysis(
        market=market,
        behavior=behavior,
        events=events,
        audit=audit,
    )
