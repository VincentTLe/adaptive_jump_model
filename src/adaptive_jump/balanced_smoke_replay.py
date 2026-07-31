"""Independent prefix, future-mutation, and current-fit smoke replay."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from adaptive_jump.balanced_mechanics import (
    independent_balanced_penalty,
    mechanical_prerequisites,
)
from adaptive_jump.balanced_model import (
    BalancedSpec,
    BalancedStudyError,
    beta_label,
    load_market_inputs,
)
from adaptive_jump.balanced_replay import INDEPENDENT_BUILDERS
from adaptive_jump.balanced_sources import SourcePaths
from adaptive_jump.config import ResearchConfig
from adaptive_jump.lagged_mechanics import _mutated_fixed_states, _sealed_parameters
from adaptive_jump.lagged_model import LockedStateEvidence, generate_locked_candidates
from adaptive_jump.models import FEATURE_COLUMNS
from adaptive_jump.separation_analysis import MarketInputs
from adaptive_jump.tv_jump import loss_matrix


def _parity(generated: pd.DataFrame, expected: pd.DataFrame, label: str) -> int:
    expected = expected.reindex(index=generated.index, columns=generated.columns)
    if not np.array_equal(generated, expected, equal_nan=True):
        raise BalancedStudyError(f"{label}: independent states changed")
    return int(np.isfinite(generated.to_numpy(dtype=float)).sum())


def _generate(
    inputs: MarketInputs,
    fixed: pd.DataFrame,
    config: ResearchConfig,
    spec: BalancedSpec,
    *,
    terminal_limit: int,
    features: pd.DataFrame | None = None,
) -> dict[str, LockedStateEvidence]:
    frame = inputs.features if features is None else features
    return generate_locked_candidates(
        frame.reset_index(),
        fixed,
        inputs.refits,
        config,
        spec,
        market="us",
        penalty_builders=INDEPENDENT_BUILDERS,
        terminal_limit=terminal_limit,
    )


def _penalty_checks(
    evidence: LockedStateEvidence, spec: BalancedSpec
) -> tuple[int, int, float]:
    discounts = 0
    surcharges = 0
    pair_error = 0.0
    for lambda0 in spec.event_lambdas:
        c01 = evidence.c01[spec.decision_beta][lambda0]
        c10 = evidence.c10[spec.decision_beta][lambda0]
        valid = c01.notna() & c10.notna()
        left = c01.loc[valid].to_numpy(dtype=float)
        right = c10.loc[valid].to_numpy(dtype=float)
        if len(left) == 0:
            raise BalancedStudyError("independent smoke penalties are empty")
        joined = np.concatenate((left, right))
        discounts += int((joined < lambda0).sum())
        surcharges += int((joined > lambda0).sum())
        pair_error = max(
            pair_error, float(np.max(np.abs(left + right - 2.0 * lambda0)))
        )
    return discounts, surcharges, pair_error


def _row_for_date(rows: pd.DataFrame, current: pd.Timestamp) -> pd.Series:
    dates = pd.DatetimeIndex(rows["fit_date"])
    position = int(dates.searchsorted(current, side="right")) - 1
    if position < 0:
        raise BalancedStudyError("independent smoke has no prior refit")
    return rows.iloc[position]


def _actual_formula_checks(
    inputs: MarketInputs,
    evidence: LockedStateEvidence,
    spec: BalancedSpec,
) -> dict[str, Any]:
    fit_dates = pd.DatetimeIndex(sorted(evidence.refits["fit_date"].unique()))
    if len(fit_dates) < 2:
        raise BalancedStudyError("independent smoke needs a second refit")
    previous_fit, second_fit = pd.Timestamp(fit_dates[0]), pd.Timestamp(fit_dates[1])
    terminal_dates = evidence.states[spec.decision_beta][spec.lambdas[0]].dropna().index
    terminal_dates = terminal_dates[terminal_dates <= second_fit]
    if len(terminal_dates) == 0 or pd.Timestamp(terminal_dates[-1]) != second_fit:
        raise BalancedStudyError("independent formula replay missed second refit")
    rows_by_lambda = {
        float(value): rows.sort_values("fit_date").reset_index(drop=True)
        for value, rows in evidence.refits.groupby("lambda0")
        if float(value) in spec.lambdas
    }
    if set(rows_by_lambda) != set(spec.lambdas):
        raise BalancedStudyError("independent smoke refit lambda coverage changed")
    formula_error = 0.0
    pair_error = 0.0
    second_error = 0.0
    bounds_exact = True
    directed_cells = 0
    for current_date in terminal_dates:
        current_date = pd.Timestamp(current_date)
        terminal = int(inputs.model_dates.get_loc(current_date))
        dates = inputs.model_dates[terminal - spec.fit_window + 1 : terminal + 1]
        raw = inputs.features.loc[dates, list(FEATURE_COLUMNS)].to_numpy(float)
        for lambda0 in spec.lambdas:
            row = _row_for_date(rows_by_lambda[lambda0], current_date)
            mean, scale, centers = _sealed_parameters(row)
            losses = loss_matrix((raw - mean) / scale, centers)
            expected = independent_balanced_penalty(
                losses,
                lambda0,
                spec.decision_beta,
                float(row["q_train"]),
            )[-1]
            observed = np.array(
                [
                    [0.0, evidence.c01[spec.decision_beta].loc[current_date, lambda0]],
                    [evidence.c10[spec.decision_beta].loc[current_date, lambda0], 0.0],
                ]
            )
            error = float(np.max(np.abs(observed - expected)))
            formula_error = max(formula_error, error)
            if current_date == second_fit:
                second_error = max(second_error, error)
            directed = observed[~np.eye(2, dtype=bool)]
            lower = lambda0 * math.exp(-spec.decision_beta)
            upper = lambda0 * (2.0 - math.exp(-spec.decision_beta))
            bounds_exact = bool(
                bounds_exact
                and (directed >= lower - spec.numerical_tolerance).all()
                and (directed <= upper + spec.numerical_tolerance).all()
            )
            pair_error = max(
                pair_error,
                abs(float(observed[0, 1] + observed[1, 0] - 2.0 * lambda0)),
            )
            directed_cells += 2
    second_terminal = int(inputs.model_dates.get_loc(second_fit))
    previous_raw = inputs.features.loc[
        inputs.model_dates[second_terminal - 1], list(FEATURE_COLUMNS)
    ].to_numpy(dtype=float, copy=True)[None, :]
    positive_event_lambdas = tuple(
        lambda0 for lambda0 in spec.event_lambdas if lambda0 > 0.0
    )
    if len(positive_event_lambdas) != len(spec.event_lambdas):
        raise BalancedStudyError("independent smoke event lambda coverage changed")
    stale_distances: list[float] = []
    informative_lambdas = 0
    for lambda0 in positive_event_lambdas:
        rows = rows_by_lambda[lambda0].set_index("fit_date")
        if second_fit not in rows.index or previous_fit not in rows.index:
            raise BalancedStudyError("independent smoke refit lambda coverage changed")
        stale_mean, stale_scale, stale_centers = _sealed_parameters(
            rows.loc[previous_fit]
        )
        current_mean, current_scale, current_centers = _sealed_parameters(
            rows.loc[second_fit]
        )
        stale_loss = loss_matrix(
            (previous_raw - stale_mean) / stale_scale, stale_centers
        )
        current_loss = loss_matrix(
            (previous_raw - current_mean) / current_scale, current_centers
        )
        if not (
            np.isfinite(stale_loss[-1]).all() and np.isfinite(current_loss[-1]).all()
        ):
            # A missing sealed center saturates the signed evidence at +-1 under
            # both fits, so the penalty is parameter-independent by construction
            # and cannot distinguish the stale from the current convention.
            continue
        informative_lambdas += 1
        stale = independent_balanced_penalty(
            np.repeat(stale_loss, 2, axis=0),
            lambda0,
            spec.decision_beta,
            float(rows.loc[second_fit, "q_train"]),
        )[-1]
        observed = np.array(
            [
                [0.0, evidence.c01[spec.decision_beta].loc[second_fit, lambda0]],
                [evidence.c10[spec.decision_beta].loc[second_fit, lambda0], 0.0],
            ]
        )
        stale_distances.append(float(np.max(np.abs(observed - stale))))
    if not stale_distances:
        raise BalancedStudyError("independent smoke has no informative stale lambda")
    return {
        "first_terminal_date": pd.Timestamp(terminal_dates[0]),
        "second_refit_date": second_fit,
        "terminal_dates_checked": len(terminal_dates),
        "lambda_values_checked": len(spec.lambdas),
        "directed_cells_checked": directed_cells,
        "maximum_formula_abs_error": formula_error,
        "maximum_second_refit_formula_abs_error": second_error,
        "maximum_pair_sum_abs_error": pair_error,
        "bounds_exact": bounds_exact,
        "minimum_stale_fit_distance": min(stale_distances),
        "maximum_stale_fit_distance": max(stale_distances),
        "stale_fit_lambdas_checked": len(positive_event_lambdas),
        "stale_fit_lambdas_informative": informative_lambdas,
        "stale_fit_lambdas_distinct": sum(
            distance > spec.numerical_tolerance for distance in stale_distances
        ),
    }


def run_independent_smoke(
    config: ResearchConfig,
    spec: BalancedSpec,
    sources: SourcePaths,
) -> dict[str, Any]:
    """Re-run the exact smoke schema with independent penalty builders."""
    inputs, fixed = load_market_inputs(
        "us", sources.fixed_markets["us"], sources.parent_markets["us"], spec
    )
    prefix_terminal_dates = 20
    fit_dates = pd.DatetimeIndex(sorted(inputs.refits["fit_date"].unique()))
    if len(fit_dates) < 2:
        raise BalancedStudyError("independent smoke needs a genuine second refit")
    refit_date = pd.Timestamp(fit_dates[1])
    refit_limit = int(inputs.model_dates.get_loc(refit_date)) - spec.fit_window + 2
    generation_limit = max(prefix_terminal_dates, refit_limit)
    if spec.fit_window + generation_limit - 2 >= len(inputs.model_dates):
        raise BalancedStudyError("independent smoke terminal coverage is incomplete")
    prefix_end = inputs.model_dates[spec.fit_window + prefix_terminal_dates - 2]
    generated_end = inputs.model_dates[spec.fit_window + generation_limit - 2]
    evidence = _generate(inputs, fixed, config, spec, terminal_limit=generation_limit)
    short = _generate(inputs, fixed, config, spec, terminal_limit=prefix_terminal_dates)
    short_long_cells = sum(
        _parity(
            short[rule].states[beta].loc[:prefix_end],
            evidence[rule].states[beta].loc[:prefix_end],
            f"us/{rule}/{beta_label(beta)}/short-long",
        )
        for rule in spec.rules
        for beta in spec.betas
    )
    parent_cells = sum(
        _parity(
            evidence["lagged"].states[beta].loc[:generated_end],
            inputs.candidates[beta].loc[:generated_end],
            f"us/lagged/{beta_label(beta)}",
        )
        for beta in spec.betas
    )
    beta_zero_cells = sum(
        _parity(
            evidence[rule].states[0.0].loc[:generated_end],
            fixed.loc[:generated_end],
            f"us/{rule}/beta-zero",
        )
        for rule in spec.rules
    )
    first_mutated_terminal = spec.fit_window + prefix_terminal_dates - 1
    mutated = inputs.features.copy()
    mutated.loc[
        inputs.model_dates[first_mutated_terminal] :, list(FEATURE_COLUMNS)
    ] += 1_000_000.0
    mutated_fixed = _mutated_fixed_states(
        inputs,
        fixed,
        mutated,
        config,
        spec,
        first_mutated_terminal=first_mutated_terminal,
        terminal_limit=generation_limit,
    )
    future = _generate(
        inputs,
        mutated_fixed,
        config,
        spec,
        terminal_limit=generation_limit,
        features=mutated,
    )
    future_prefix_cells = sum(
        _parity(
            evidence[rule].states[beta].loc[:prefix_end],
            future[rule].states[beta].loc[:prefix_end],
            f"us/{rule}/{beta_label(beta)}/future-prefix",
        )
        for rule in spec.rules
        for beta in spec.betas
    )
    original_losses = np.stack(
        [
            evidence["balanced"].loss0.loc[prefix_end:].iloc[1:].to_numpy(float),
            evidence["balanced"].loss1.loc[prefix_end:].iloc[1:].to_numpy(float),
        ]
    )
    mutated_losses = np.stack(
        [
            future["balanced"].loss0.loc[prefix_end:].iloc[1:].to_numpy(float),
            future["balanced"].loss1.loc[prefix_end:].iloc[1:].to_numpy(float),
        ]
    )
    finite = np.isfinite(original_losses) & np.isfinite(mutated_losses)
    changes = np.abs(original_losses[finite] - mutated_losses[finite])
    changed_cells = int((changes > spec.numerical_tolerance).sum())
    max_change = float(changes.max()) if changes.size else 0.0
    discounts, surcharges, summary_pair_error = _penalty_checks(
        evidence["balanced"], spec
    )
    actual = _actual_formula_checks(inputs, evidence["balanced"], spec)
    pair_error = max(summary_pair_error, float(actual["maximum_pair_sum_abs_error"]))
    mechanics = mechanical_prerequisites(spec)
    prefix_candidate_cells = prefix_terminal_dates * len(spec.lambdas)
    generated_candidate_cells = generation_limit * len(spec.lambdas)
    checks = {
        "parent_lagged_exact": parent_cells
        == generated_candidate_cells * len(spec.betas),
        "beta_zero_exact": beta_zero_cells
        == generated_candidate_cells * len(spec.rules),
        "short_long_prefix_exact": short_long_cells
        == prefix_candidate_cells * len(spec.rules) * len(spec.betas),
        "future_mutation_prefix_invariant": future_prefix_cells
        == prefix_candidate_cells * len(spec.rules) * len(spec.betas),
        "prefix_invariant": short_long_cells
        == prefix_candidate_cells * len(spec.rules) * len(spec.betas)
        and future_prefix_cells
        == prefix_candidate_cells * len(spec.rules) * len(spec.betas),
        "future_mutation_effect_present": changed_cells > 0,
        "actual_formula_exact": float(actual["maximum_formula_abs_error"])
        <= spec.numerical_tolerance,
        "actual_bounds_exact": actual["bounds_exact"] is True,
        "formula_through_second_refit": actual["second_refit_date"] == refit_date
        and int(actual["terminal_dates_checked"]) == refit_limit
        and int(actual["terminal_dates_checked"]) >= 2
        and int(actual["lambda_values_checked"]) == len(spec.lambdas),
        "pair_balance_exact": pair_error <= spec.numerical_tolerance,
        "balanced_discounts_present": discounts > 0,
        "balanced_surcharges_present": surcharges > 0,
        "refit_convention_numeric": float(
            actual["maximum_second_refit_formula_abs_error"]
        )
        <= spec.numerical_tolerance
        and float(actual["minimum_stale_fit_distance"]) > spec.numerical_tolerance
        and int(actual["stale_fit_lambdas_checked"]) == len(spec.event_lambdas)
        and int(actual["stale_fit_lambdas_informative"]) >= 1
        and int(actual["stale_fit_lambdas_distinct"])
        == int(actual["stale_fit_lambdas_informative"]),
    }
    if not all(checks.values()) or mechanics.get("passed") is not True:
        raise BalancedStudyError(f"independent balanced smoke failed: {checks}")
    return {
        "status": "passed",
        "market": "us",
        "terminal_dates": prefix_terminal_dates,
        "generated_terminal_dates": generation_limit,
        "generated_terminal_end_date": generated_end.date().isoformat(),
        "refit_probe_date": actual["second_refit_date"].date().isoformat(),
        **checks,
        "mechanical_prerequisites": mechanics,
        "parent_lagged_state_cells_checked": parent_cells,
        "beta_zero_state_cells_checked": beta_zero_cells,
        "short_long_prefix_state_cells_checked": short_long_cells,
        "future_mutation_prefix_state_cells_checked": future_prefix_cells,
        "future_mutation_loss_cells_changed": changed_cells,
        "future_mutation_max_abs_loss_change": max_change,
        "balanced_discount_cells": discounts,
        "balanced_surcharge_cells": surcharges,
        "actual_formula_terminal_dates_checked": int(actual["terminal_dates_checked"]),
        "actual_formula_lambda_values_checked": int(actual["lambda_values_checked"]),
        "actual_formula_directed_cells_checked": int(actual["directed_cells_checked"]),
        "actual_formula_first_terminal_date": actual["first_terminal_date"]
        .date()
        .isoformat(),
        "actual_formula_max_abs_error": float(actual["maximum_formula_abs_error"]),
        "actual_second_refit_formula_max_abs_error": float(
            actual["maximum_second_refit_formula_abs_error"]
        ),
        "maximum_pair_sum_abs_error": pair_error,
        "refit_convention_min_stale_distance": float(
            actual["minimum_stale_fit_distance"]
        ),
        "refit_convention_max_stale_distance": float(
            actual["maximum_stale_fit_distance"]
        ),
        "refit_convention_lambdas_checked": int(actual["stale_fit_lambdas_checked"]),
        "refit_convention_informative_lambdas": int(
            actual["stale_fit_lambdas_informative"]
        ),
        "refit_convention_distinct_lambdas": int(actual["stale_fit_lambdas_distinct"]),
        "refit_convention_max_abs_error": float(
            actual["maximum_second_refit_formula_abs_error"]
        ),
        "performance_files_accessed": False,
        "return_columns_accessed": False,
        "post_2023_accessed": False,
    }
