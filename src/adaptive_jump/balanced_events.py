"""Exact own-rule discount-attributable event reconstruction."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from adaptive_jump.balanced_model import (
    BUILDERS,
    BalancedSpec,
    BalancedStudyError,
    beta_label,
)
from adaptive_jump.lagged_model import LockedStateEvidence
from adaptive_jump.models import FEATURE_COLUMNS
from adaptive_jump.separation_analysis import (
    MarketInputs,
    _decode_parameters,
    _refit_for_date,
)
from adaptive_jump.separation_study import arrival_ablation_state, terminal_decision
from adaptive_jump.tv_jump import lam_to_penalty_seq, loss_matrix

EVENT_COLUMNS = [
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
    "pair_sum_abs_error",
    "terminal_predecessor",
    "terminal_state_margin",
    "terminal_predecessor_margin",
    "ablated_state",
    "ablated_state_margin",
    "discount_attributable",
    "horizon_candidate_dates",
    "persistent_20",
    "whipsaw_20",
    "first_reversal_h",
    "fixed_confirmation_h",
    "confirmed_early",
    "unconfirmed_persistent_20",
]


def _expected_cost(
    rule: str, gap: float, lambda0: float, beta: float, q_train: float
) -> float:
    evidence = math.tanh(gap / q_train)
    if rule == "lagged":
        return lambda0 * math.exp(-beta * evidence)
    return lambda0 * (1.0 - (1.0 - math.exp(-beta)) * evidence)


def extract_events(
    inputs: MarketInputs,
    evidence: dict[str, LockedStateEvidence],
    spec: BalancedSpec,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Regenerate every admitted own-rule event under the frozen rules."""
    positions = {value: index for index, value in enumerate(inputs.model_dates)}
    refit_source = inputs.refits.set_index(["fit_date", "lambda0"], drop=False)
    refits = {
        value: rows.sort_values("fit_date").reset_index(drop=True)
        for value, rows in inputs.refits.groupby("lambda0")
        if value in spec.event_lambdas
    }
    fixed_frame = inputs.candidates[0.0]
    records: list[dict[str, Any]] = []
    audits: dict[str, Any] = {}
    for rule in spec.rules:
        counters = {
            "candidate_divergences": 0,
            "horizon_censored": 0,
            "state_reconstructions": 0,
            "terminal_tie_exclusions": 0,
            "terminal_transition_matches": 0,
            "discounted_terminal_transitions": 0,
            "ablation_attributable": 0,
            "admitted_events": 0,
            "max_penalty_abs_error": 0.0,
            "max_pair_sum_abs_error": 0.0,
            "minimum_terminal_state_margin": None,
            "minimum_terminal_predecessor_margin": None,
            "minimum_ablated_state_margin": None,
            "nonfinite_terminal_state_margin_count": 0,
            "nonfinite_terminal_predecessor_margin_count": 0,
            "nonfinite_ablated_state_margin_count": 0,
        }
        model_frame = evidence[rule].states[spec.decision_beta]
        for lambda0 in spec.event_lambdas:
            model = model_frame[lambda0].dropna().astype(int)
            fixed = fixed_frame[lambda0].reindex(model.index).astype(int)
            first = int(
                model.index.searchsorted(
                    pd.Timestamp(spec.evaluation_starts[inputs.market]), side="left"
                )
            )
            suppress_through = -1
            for index in range(max(1, first), len(model)):
                if index <= suppress_through:
                    continue
                source, destination = int(model.iloc[index - 1]), int(model.iloc[index])
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
                refit = _refit_for_date(refits[lambda0], signal_date)
                if (
                    _refit_for_date(refits[lambda0], horizon_end)["fit_date"]
                    != refit["fit_date"]
                ):
                    counters["horizon_censored"] += 1
                    continue
                terminal = positions.get(signal_date)
                if terminal is None or terminal + 1 < spec.fit_window:
                    raise BalancedStudyError("event terminal is not mapped")
                dates = inputs.model_dates[
                    terminal - spec.fit_window + 1 : terminal + 1
                ]
                row = refit_source.loc[
                    (pd.Timestamp(refit["fit_date"]), float(lambda0))
                ]
                mean, scale, centers = _decode_parameters(row)
                raw = inputs.features.loc[dates, FEATURE_COLUMNS].to_numpy(dtype=float)
                losses = loss_matrix((raw - mean) / scale, centers)
                q_train = float(row["q_train"])
                penalties = BUILDERS[rule](losses, lambda0, spec.decision_beta, q_train)
                fixed_penalties = lam_to_penalty_seq(
                    np.full(spec.fit_window, lambda0), 2
                )
                decision = terminal_decision(losses, penalties)
                fixed_decision = terminal_decision(losses, fixed_penalties)
                counters["state_reconstructions"] += 1
                if decision.state != destination or fixed_decision.state != int(
                    fixed.iloc[index]
                ):
                    raise BalancedStudyError("candidate state reconstruction changed")
                if decision.state_tied or decision.predecessor_tied:
                    counters["terminal_tie_exclusions"] += 1
                    continue
                if decision.predecessor != source:
                    continue
                counters["terminal_transition_matches"] += 1
                transition = float(penalties[-1, source, destination])
                reverse = float(penalties[-1, destination, source])
                if not transition < lambda0:
                    continue
                pair_error = abs(transition + reverse - 2.0 * lambda0)
                if rule == "lagged":
                    if abs(reverse - lambda0) > spec.numerical_tolerance:
                        raise BalancedStudyError("lagged reverse cost changed")
                elif pair_error > spec.numerical_tolerance or not reverse > lambda0:
                    raise BalancedStudyError("balanced reverse cost changed")
                counters["max_pair_sum_abs_error"] = max(
                    counters["max_pair_sum_abs_error"], pair_error
                )
                counters["discounted_terminal_transitions"] += 1
                gap = max(float(losses[-2, source] - losses[-2, destination]), 0.0)
                expected = _expected_cost(
                    rule, gap, lambda0, spec.decision_beta, q_train
                )
                error = abs(expected - transition)
                counters["max_penalty_abs_error"] = max(
                    counters["max_penalty_abs_error"], error
                )
                if error > spec.numerical_tolerance:
                    raise BalancedStudyError("transition formula changed")
                ablated = penalties.copy()
                ablated[-1] = lambda0 * (1.0 - np.eye(2))
                ablated_decision = terminal_decision(losses, ablated)
                if ablated_decision.state_tied:
                    counters["terminal_tie_exclusions"] += 1
                    continue
                ablated_state = arrival_ablation_state(losses, penalties, lambda0)
                if ablated_state != ablated_decision.state:
                    raise BalancedStudyError("ablation implementations differ")
                if ablated_state != source:
                    continue
                counters["ablation_attributable"] += 1
                for key, value in (
                    ("terminal_state_margin", decision.state_margin),
                    ("terminal_predecessor_margin", decision.predecessor_margin),
                    ("ablated_state_margin", ablated_decision.state_margin),
                ):
                    minimum_key = f"minimum_{key}"
                    if math.isfinite(value):
                        current_minimum = counters[minimum_key]
                        counters[minimum_key] = (
                            float(value)
                            if current_minimum is None
                            else min(float(current_minimum), float(value))
                        )
                    else:
                        counters[f"nonfinite_{key}_count"] += 1
                future_model = model.iloc[index + 1 : index + spec.horizon + 1]
                future_fixed = fixed.iloc[index + 1 : index + spec.horizon + 1]
                reversals = np.flatnonzero(future_model.to_numpy() == source)
                confirmations = np.flatnonzero(future_fixed.to_numpy() == destination)
                persistent = len(reversals) == 0
                confirmed = persistent and len(confirmations) > 0
                records.append(
                    {
                        "market": inputs.market,
                        "rule": rule,
                        "beta": spec.decision_beta,
                        "beta_label": beta_label(spec.decision_beta),
                        "lambda0": lambda0,
                        "signal_date": signal_date,
                        "evidence_date": pd.Timestamp(dates[-2]),
                        "fit_date": pd.Timestamp(refit["fit_date"]),
                        "source_state": source,
                        "destination_state": destination,
                        "loss_source": float(losses[-2, source]),
                        "loss_destination": float(losses[-2, destination]),
                        "q_train": q_train,
                        "normalized_gap": gap / q_train,
                        "transition_penalty": transition,
                        "expected_transition_penalty": expected,
                        "reverse_penalty": reverse,
                        "pair_sum_abs_error": pair_error,
                        "terminal_predecessor": decision.predecessor,
                        "terminal_state_margin": decision.state_margin,
                        "terminal_predecessor_margin": decision.predecessor_margin,
                        "ablated_state": ablated_state,
                        "ablated_state_margin": ablated_decision.state_margin,
                        "discount_attributable": True,
                        "horizon_candidate_dates": spec.horizon,
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
                        "confirmed_early": confirmed,
                        "unconfirmed_persistent_20": persistent
                        and not len(confirmations),
                    }
                )
                suppress_through = index + spec.horizon
                counters["admitted_events"] += 1
        audits[rule] = counters
    return pd.DataFrame.from_records(records, columns=EVENT_COLUMNS), audits
