"""Independent event, ablation, overlap, and matched-anchor replay."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd

from adaptive_jump.balanced_model import (
    BalancedSpec,
    BalancedStudyError,
    beta_label,
)
from adaptive_jump.balanced_replay import INDEPENDENT_BUILDERS
from adaptive_jump.lagged_model import LockedStateEvidence
from adaptive_jump.models import FEATURE_COLUMNS
from adaptive_jump.separation_analysis import MarketInputs
from adaptive_jump.tv_jump import dp_tv, loss_matrix

EVENT_COLUMNS = (
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
)
MATCHED_COLUMNS = (
    "market",
    "beta",
    "beta_label",
    "lambda0",
    "signal_date",
    "source_state",
    "destination_state",
    "lagged_whipsaw_20",
    "lagged_confirmed_early",
    "lagged_unconfirmed_persistent_20",
    "fixed_confirmation_h",
    "first_destination_h",
    "matched_follow_end_h",
    "matched_fixed_confirmation_h",
    "matched_category",
    "matched_whipsaw_20",
    "matched_persistent_20",
    "matched_unconfirmed_persistent_20",
    "matched_retained_confirmed_early",
)
MATCHED_CATEGORIES = frozenset(
    {
        "already_destination_whipsaw",
        "already_destination_persistent_confirmed",
        "already_destination_persistent_unconfirmed",
        "suppressed_no_entry",
        "enters_then_whipsaw",
        "enters_persistent_confirmed",
        "enters_persistent_unconfirmed",
    }
)


def _argmin(values: np.ndarray) -> tuple[int, float, bool]:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or not np.isfinite(vector).any():
        raise BalancedStudyError("terminal cost vector is invalid")
    minimum = float(np.nanmin(vector))
    state = int(np.nanargmin(vector))
    tied = int(np.count_nonzero(vector == minimum)) > 1
    ordered = np.sort(vector[np.isfinite(vector)])
    margin = float(ordered[1] - ordered[0]) if len(ordered) > 1 else math.inf
    return state, margin, tied


def _terminal_decision(
    losses: np.ndarray, penalties: np.ndarray
) -> tuple[int, int, float, float, bool, bool]:
    values = dp_tv(losses, penalties, return_value_mx=True)
    state, state_margin, state_tied = _argmin(values[-1])
    incoming = values[-2] + penalties[-1, :, state]
    predecessor, predecessor_margin, predecessor_tied = _argmin(incoming)
    return (
        state,
        predecessor,
        state_margin,
        predecessor_margin,
        state_tied,
        predecessor_tied,
    )


def _decode_parameters(row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        mean = np.asarray(json.loads(row["scaler_mean"]), dtype=float)
        scale = np.asarray(json.loads(row["scaler_scale"]), dtype=float)
        centers = np.asarray(json.loads(row["centers"]), dtype=float)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise BalancedStudyError("refit parameters cannot be decoded") from exc
    if (
        mean.shape != (len(FEATURE_COLUMNS),)
        or scale.shape != mean.shape
        or centers.shape != (2, len(FEATURE_COLUMNS))
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or (scale <= 0).any()
    ):
        raise BalancedStudyError("refit parameter shape changed")
    return mean, scale, centers


def _refit_for_date(rows: pd.DataFrame, current: pd.Timestamp) -> pd.Series:
    dates = pd.DatetimeIndex(rows["fit_date"])
    position = int(dates.searchsorted(current, side="right")) - 1
    if position < 0:
        raise BalancedStudyError(f"{current.date()}: no refit in replay")
    return rows.iloc[position]


def _fixed_penalties(length: int, lambda0: float) -> np.ndarray:
    result = np.full((length, 2, 2), float(lambda0))
    result[:, 0, 0] = 0.0
    result[:, 1, 1] = 0.0
    return result


def _expected_cost(
    rule: str, gap: float, lambda0: float, beta: float, q_train: float
) -> float:
    evidence = math.tanh(gap / q_train)
    if rule == "lagged":
        return lambda0 * math.exp(-beta * evidence)
    if rule == "balanced":
        return lambda0 * (1.0 - (1.0 - math.exp(-beta)) * evidence)
    raise BalancedStudyError(f"unknown replay rule: {rule}")


def extract_events(
    inputs: MarketInputs,
    evidence: dict[str, LockedStateEvidence],
    spec: BalancedSpec,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Replay all own-rule events without the production event module."""
    positions = {value: index for index, value in enumerate(inputs.model_dates)}
    refit_source = inputs.refits.set_index(["fit_date", "lambda0"], drop=False)
    refits = {
        float(value): rows.sort_values("fit_date").reset_index(drop=True)
        for value, rows in inputs.refits.groupby("lambda0")
        if float(value) in spec.event_lambdas
    }
    fixed_frame = inputs.candidates[0.0]
    records: list[dict[str, Any]] = []
    audits: dict[str, Any] = {}
    for rule in spec.rules:
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
                refit = _refit_for_date(refits[lambda0], signal_date)
                horizon_refit = _refit_for_date(refits[lambda0], horizon_end)
                if horizon_refit["fit_date"] != refit["fit_date"]:
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
                penalties = INDEPENDENT_BUILDERS[rule](
                    losses, lambda0, spec.decision_beta, q_train
                )
                fixed_penalty = _fixed_penalties(spec.fit_window, lambda0)
                decision = _terminal_decision(losses, penalties)
                fixed_decision = _terminal_decision(losses, fixed_penalty)
                counters["state_reconstructions"] += 1
                if decision[0] != destination or fixed_decision[0] != int(
                    fixed.iloc[index]
                ):
                    raise BalancedStudyError("candidate state replay changed")
                if decision[4] or decision[5]:
                    counters["terminal_tie_exclusions"] += 1
                    continue
                if decision[1] != source:
                    continue
                counters["terminal_transition_matches"] += 1
                transition = float(penalties[-1, source, destination])
                reverse = float(penalties[-1, destination, source])
                if not transition < lambda0:
                    continue
                pair_error = abs(transition + reverse - 2.0 * lambda0)
                if rule == "lagged":
                    if abs(reverse - lambda0) > spec.numerical_tolerance:
                        raise BalancedStudyError("lagged reverse penalty changed")
                elif pair_error > spec.numerical_tolerance or not reverse > lambda0:
                    raise BalancedStudyError("balanced reverse penalty changed")
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
                    raise BalancedStudyError("transition formula replay changed")
                ablated = penalties.copy()
                ablated[-1] = _fixed_penalties(1, lambda0)[0]
                ablated_decision = _terminal_decision(losses, ablated)
                if ablated_decision[4]:
                    counters["terminal_tie_exclusions"] += 1
                    continue
                if ablated_decision[0] != source:
                    continue
                counters["ablation_attributable"] += 1
                for key, value in (
                    ("terminal_state_margin", decision[2]),
                    ("terminal_predecessor_margin", decision[3]),
                    ("ablated_state_margin", ablated_decision[2]),
                ):
                    if math.isfinite(value):
                        minimum_key = f"minimum_{key}"
                        current = counters[minimum_key]
                        counters[minimum_key] = (
                            float(value)
                            if current is None
                            else min(float(current), float(value))
                        )
                    else:
                        counters[f"nonfinite_{key}_count"] += 1
                future_model = model.iloc[index + 1 : index + spec.horizon + 1]
                future_fixed = fixed.iloc[index + 1 : index + spec.horizon + 1]
                reversals = np.flatnonzero(future_model.to_numpy() == source)
                confirmations = np.flatnonzero(future_fixed.to_numpy() == destination)
                persistent = len(reversals) == 0
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
                        "terminal_predecessor": decision[1],
                        "terminal_state_margin": decision[2],
                        "terminal_predecessor_margin": decision[3],
                        "ablated_state": ablated_decision[0],
                        "ablated_state_margin": ablated_decision[2],
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
                        "confirmed_early": persistent and len(confirmations) > 0,
                        "unconfirmed_persistent_20": persistent
                        and len(confirmations) == 0,
                    }
                )
                suppress_through = index + spec.horizon
                counters["admitted_events"] += 1
        audits[rule] = counters
    return pd.DataFrame.from_records(records, columns=EVENT_COLUMNS), audits


def matched_response(
    events: pd.DataFrame,
    balanced_states: pd.DataFrame,
    fixed_states: pd.DataFrame,
    refits: pd.DataFrame,
    spec: BalancedSpec,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Classify balanced behavior on every independently regenerated lagged anchor."""
    records: list[dict[str, Any]] = []
    lagged = events.loc[events["rule"] == "lagged"]
    counters = {
        "original_lagged_admitted_events": len(lagged),
        "matched_anchor_censored": 0,
        "eligible_matched_anchors": 0,
    }
    refits_by_lambda = {
        float(lambda0): rows.sort_values("fit_date").reset_index(drop=True)
        for lambda0, rows in refits.groupby("lambda0")
        if float(lambda0) in spec.event_lambdas
    }
    for anchor in lagged.itertuples(index=False):
        lambda0 = float(anchor.lambda0)
        if lambda0 not in fixed_states or lambda0 not in refits_by_lambda:
            raise BalancedStudyError("matched anchor lambda coverage changed")
        fixed = fixed_states[lambda0].dropna().astype(int)
        signal_date = pd.Timestamp(anchor.signal_date)
        try:
            position = int(fixed.index.get_loc(signal_date))
        except KeyError as exc:
            raise BalancedStudyError("matched anchor date is missing") from exc
        censor_position = position + spec.matched_anchor_censor
        if position < 1 or censor_position >= len(fixed):
            counters["matched_anchor_censored"] += 1
            continue
        event_refit = _refit_for_date(refits_by_lambda[lambda0], signal_date)
        if pd.Timestamp(event_refit["fit_date"]) != pd.Timestamp(anchor.fit_date):
            raise BalancedStudyError("matched anchor event refit changed")
        censor_date = pd.Timestamp(fixed.index[censor_position])
        censor_refit = _refit_for_date(refits_by_lambda[lambda0], censor_date)
        if pd.Timestamp(censor_refit["fit_date"]) != pd.Timestamp(anchor.fit_date):
            counters["matched_anchor_censored"] += 1
            continue
        counters["eligible_matched_anchors"] += 1
        path = balanced_states[lambda0].reindex(fixed.index)
        if path.isna().any() or not np.isin(path.to_numpy(), (0, 1)).all():
            raise BalancedStudyError("matched path coverage changed")
        path = path.astype(int)
        source = int(anchor.source_state)
        destination = int(anchor.destination_state)
        if int(path.iloc[position - 1]) == destination:
            first_h: int | float = -1
        else:
            search = path.iloc[
                position : position + spec.matched_entry_search + 1
            ].to_numpy(dtype=int)
            entries = np.flatnonzero(search == destination)
            first_h = int(entries[0]) if len(entries) else math.nan
        if not math.isfinite(first_h):
            follow_end_h: int | float = math.nan
            matched_fixed_h: int | float = math.nan
            category = "suppressed_no_entry"
            whipsaw, persistent = False, False
        else:
            follow_start_h = 0 if first_h == -1 else int(first_h) + 1
            follow_end_h = (
                spec.matched_followup - 1
                if first_h == -1
                else int(first_h) + spec.matched_followup
            )
            follow = path.iloc[
                position + follow_start_h : position + int(follow_end_h) + 1
            ].to_numpy(dtype=int)
            if len(follow) != spec.matched_followup:
                raise BalancedStudyError("matched follow-up exposure changed")
            returned = bool((follow == source).any())
            fixed_through_follow = fixed.iloc[
                position : position + int(follow_end_h) + 1
            ].to_numpy(dtype=int)
            confirmations = np.flatnonzero(fixed_through_follow == destination)
            matched_fixed_h = int(confirmations[0]) if len(confirmations) else math.nan
            if returned:
                category = (
                    "already_destination_whipsaw"
                    if first_h == -1
                    else "enters_then_whipsaw"
                )
                whipsaw, persistent = True, False
            else:
                prefix = "already_destination" if first_h == -1 else "enters"
                suffix = (
                    "confirmed" if math.isfinite(matched_fixed_h) else "unconfirmed"
                )
                category = f"{prefix}_persistent_{suffix}"
                whipsaw, persistent = False, True
        fixed_h = float(anchor.fixed_confirmation_h)
        unconfirmed = persistent and not math.isfinite(matched_fixed_h)
        retained = bool(
            anchor.confirmed_early
            and math.isfinite(first_h)
            and math.isfinite(fixed_h)
            and first_h < fixed_h
            and not whipsaw
        )
        records.append(
            {
                "market": anchor.market,
                "beta": anchor.beta,
                "beta_label": anchor.beta_label,
                "lambda0": anchor.lambda0,
                "signal_date": signal_date,
                "source_state": source,
                "destination_state": destination,
                "lagged_whipsaw_20": bool(anchor.whipsaw_20),
                "lagged_confirmed_early": bool(anchor.confirmed_early),
                "lagged_unconfirmed_persistent_20": bool(
                    anchor.unconfirmed_persistent_20
                ),
                "fixed_confirmation_h": anchor.fixed_confirmation_h,
                "first_destination_h": first_h,
                "matched_follow_end_h": follow_end_h,
                "matched_fixed_confirmation_h": matched_fixed_h,
                "matched_category": category,
                "matched_whipsaw_20": whipsaw,
                "matched_persistent_20": persistent,
                "matched_unconfirmed_persistent_20": unconfirmed,
                "matched_retained_confirmed_early": retained,
            }
        )
    result = pd.DataFrame.from_records(records, columns=MATCHED_COLUMNS)
    if (
        counters["original_lagged_admitted_events"]
        != counters["matched_anchor_censored"] + counters["eligible_matched_anchors"]
        or len(result) != counters["eligible_matched_anchors"]
        or (
            len(result)
            and not set(result["matched_category"]).issubset(MATCHED_CATEGORIES)
        )
    ):
        raise BalancedStudyError("matched anchors violate the frozen partition")
    return result, counters
