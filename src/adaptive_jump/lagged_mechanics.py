"""Deterministic mechanical prerequisites for evidence-penalty candidates."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from adaptive_jump.config import ResearchConfig
from adaptive_jump.lagged_model import (
    LockedModelError,
    PenaltyBuilder,
    _decode_array,
    generate_locked_candidates,
)
from adaptive_jump.lagged_study import LaggedMechanismSpec, LaggedStudyError, beta_label
from adaptive_jump.models import FEATURE_COLUMNS
from adaptive_jump.separation_analysis import MarketInputs, _refit_for_date
from adaptive_jump.tv_jump import dp_tv, lam_to_penalty_seq, loss_matrix

EXPECTED_TOY_PATHS = {
    "isolated": {
        "fixed": [0, 0, 0, 0, 0],
        "arrival": [0, 0, 1, 0, 0],
        "lagged": [0, 0, 0, 0, 0],
    },
    "alternating": {
        "fixed": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "arrival": [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        "lagged": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    },
    "persistent": {
        "fixed": [0, 0, 0, 0, 0, 0, 1, 1],
        "arrival": [0, 0, 0, 1, 1, 1, 1, 1],
        "lagged": [0, 0, 0, 0, 1, 1, 1, 1],
    },
}


def _parity(generated: pd.DataFrame, expected: pd.DataFrame, label: str) -> int:
    expected = expected.reindex(index=generated.index, columns=generated.columns)
    if not np.array_equal(generated, expected, equal_nan=True):
        raise LaggedStudyError(f"{label}: generated states differ from sealed source")
    return int(np.isfinite(generated.to_numpy(dtype=float)).sum())


def _sealed_parameters(row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode parameters already validated by locked candidate generation."""
    return (
        _decode_array(row["scaler_mean"]),
        _decode_array(row["scaler_scale"]),
        _decode_array(row["centers"]),
    )


def _mutated_fixed_states(
    inputs: MarketInputs,
    fixed: pd.DataFrame,
    features: pd.DataFrame,
    config: ResearchConfig,
    spec: LaggedMechanismSpec,
    *,
    first_mutated_terminal: int,
    terminal_limit: int,
) -> pd.DataFrame:
    """Recompute only mutation-affected beta-zero states without fitting."""
    result = fixed.copy()
    final_terminal = min(len(inputs.model_dates), spec.fit_window - 1 + terminal_limit)
    refits = inputs.refits.copy()
    refits["fit_date"] = pd.to_datetime(refits["fit_date"], errors="raise")
    by_lambda = {
        lambda0: rows.sort_values("fit_date").reset_index(drop=True)
        for lambda0, rows in refits.groupby("lambda0")
    }
    for terminal in range(first_mutated_terminal, final_terminal):
        dates = inputs.model_dates[terminal - spec.fit_window + 1 : terminal + 1]
        current_date = pd.Timestamp(dates[-1])
        raw = features.loc[dates, list(FEATURE_COLUMNS)].to_numpy(dtype=float)
        for lambda0 in spec.lambdas:
            rows = by_lambda.get(lambda0)
            if rows is None:
                raise LaggedStudyError("US smoke refit lambda coverage changed")
            row = _refit_for_date(rows, current_date)
            mean, scale, centers = _sealed_parameters(row)
            losses = loss_matrix((raw - mean) / scale, centers)
            penalty = lam_to_penalty_seq(
                np.full(spec.fit_window, lambda0), config.model_protocol.n_states
            )
            result.loc[current_date, lambda0] = int(
                dp_tv(losses, penalty, return_value_mx=True)[-1].argmin()
            )
    return result


def _refit_convention_errors(
    inputs: MarketInputs,
    lagged: Any,
    spec: LaggedMechanismSpec,
) -> tuple[float, float, pd.Timestamp]:
    fit_dates = pd.DatetimeIndex(sorted(lagged.refits["fit_date"].unique()))
    if len(fit_dates) < 2:
        raise LaggedStudyError("US smoke needs a genuine second refit")
    previous_fit, fit_date = pd.Timestamp(fit_dates[0]), pd.Timestamp(fit_dates[1])
    terminal = int(inputs.model_dates.get_loc(fit_date))
    if terminal < 1:
        raise LaggedStudyError("second refit has no preceding feature row")
    raw_previous = inputs.features.loc[
        inputs.model_dates[terminal - 1], list(FEATURE_COLUMNS)
    ].to_numpy(dtype=float, copy=True)[None, :]
    current_errors: list[float] = []
    stale_distances: list[float] = []

    for lambda0 in spec.event_lambdas:
        rows = lagged.refits.loc[lagged.refits["lambda0"] == lambda0].set_index(
            "fit_date"
        )
        if fit_date not in rows.index or previous_fit not in rows.index:
            raise LaggedStudyError("US refit lambda coverage changed")
        current_mean, current_scale, current_centers = _sealed_parameters(
            rows.loc[fit_date]
        )
        stale_mean, stale_scale, stale_centers = _sealed_parameters(
            rows.loc[previous_fit]
        )
        current_loss = loss_matrix(
            (raw_previous - current_mean) / current_scale, current_centers
        )
        stale_loss = loss_matrix(
            (raw_previous - stale_mean) / stale_scale, stale_centers
        )
        q_train = float(rows.loc[fit_date, "q_train"])
        for beta in spec.event_betas:
            current_penalty = _formula_penalty(
                np.repeat(current_loss, 2, axis=0),
                lambda0,
                beta,
                q_train,
                "lagged",
            )[-1]
            stale_penalty = _formula_penalty(
                np.repeat(stale_loss, 2, axis=0),
                lambda0,
                beta,
                q_train,
                "lagged",
            )[-1]
            observed = np.array(
                [
                    [0.0, lagged.c01[beta].loc[fit_date, lambda0]],
                    [lagged.c10[beta].loc[fit_date, lambda0], 0.0],
                ]
            )
            if np.isfinite([current_penalty, stale_penalty, observed]).all():
                current_errors.append(float(np.max(np.abs(observed - current_penalty))))
                stale_distances.append(float(np.max(np.abs(observed - stale_penalty))))
    if not current_errors:
        raise LaggedStudyError("US refit convention probe has no finite comparison")
    return max(current_errors), max(stale_distances), fit_date


def _formula_penalty(
    loss: np.ndarray, lambda0: float, beta: float, q_train: float, rule: str
) -> np.ndarray:
    evidence_loss = np.asarray(loss, dtype=float)
    if rule == "lagged":
        evidence_loss = np.vstack([np.zeros((1, loss.shape[1])), loss[:-1]])
    evidence_loss = np.where(np.isnan(evidence_loss), np.inf, evidence_loss)
    previous_loss = evidence_loss[:, :, None]
    destination_loss = evidence_loss[:, None, :]
    with np.errstate(invalid="ignore", over="ignore"):
        gap = np.maximum(previous_loss - destination_loss, 0.0)
    gap = np.where(np.isposinf(destination_loss), 0.0, gap)
    result = lambda0 * np.exp(-beta * np.tanh(gap / q_train))
    states = np.arange(loss.shape[1])
    result[:, states, states] = 0.0
    return result


def run_locked_smoke(
    inputs: MarketInputs,
    fixed: pd.DataFrame,
    config: ResearchConfig,
    spec: LaggedMechanismSpec,
    penalty_builders: Mapping[str, PenaltyBuilder],
) -> dict[str, Any]:
    """Recompute the nonvacuous US prefix and second-refit smoke checks."""
    sealed_fixed = inputs.candidates[0.0].reindex(columns=spec.lambdas)
    if not fixed.index.equals(inputs.features.index) or not np.array_equal(
        fixed, sealed_fixed, equal_nan=True
    ):
        raise LaggedStudyError("US smoke fixed and sealed beta-zero paths differ")

    prefix_terminal_limit = 20
    fit_dates = pd.DatetimeIndex(sorted(inputs.refits["fit_date"].unique()))
    if len(fit_dates) < 2:
        raise LaggedStudyError("US smoke needs a genuine second refit")
    refit_date = pd.Timestamp(fit_dates[1])
    refit_terminal_limit = (
        int(inputs.model_dates.get_loc(refit_date)) - spec.fit_window + 2
    )
    generation_limit = max(prefix_terminal_limit, refit_terminal_limit)
    if spec.fit_window + generation_limit - 2 >= len(inputs.model_dates):
        raise LaggedStudyError("US smoke terminal coverage is incomplete")
    prefix_end = inputs.model_dates[spec.fit_window + prefix_terminal_limit - 2]

    def generate(features: pd.DataFrame, expected_fixed: pd.DataFrame):
        return generate_locked_candidates(
            features.reset_index(),
            expected_fixed,
            inputs.refits,
            config,
            spec,
            market="us",
            penalty_builders=penalty_builders,
            terminal_limit=generation_limit,
        )

    evidence = generate(inputs.features, fixed)
    arrival_cells = sum(
        _parity(
            evidence["arrival"].states[beta].loc[:prefix_end],
            inputs.candidates[beta].loc[:prefix_end],
            f"us/arrival/{beta_label(beta)}",
        )
        for beta in spec.betas
    )
    beta_cells = sum(
        _parity(
            evidence[rule].states[0.0].loc[:prefix_end],
            fixed.loc[:prefix_end],
            f"us/{rule}/beta-zero",
        )
        for rule in spec.rules
    )

    first_mutated_terminal = spec.fit_window + prefix_terminal_limit - 1
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
    future = generate(mutated, mutated_fixed)
    prefix_cells = sum(
        _parity(
            evidence[rule].states[beta].loc[:prefix_end],
            future[rule].states[beta].loc[:prefix_end],
            f"us/{rule}/{beta_label(beta)}/prefix",
        )
        for rule in spec.rules
        for beta in spec.betas
    )
    original_losses = np.stack(
        [
            evidence["lagged"].loss0.loc[prefix_end:].iloc[1:].to_numpy(dtype=float),
            evidence["lagged"].loss1.loc[prefix_end:].iloc[1:].to_numpy(dtype=float),
        ]
    )
    mutated_losses = np.stack(
        [
            future["lagged"].loss0.loc[prefix_end:].iloc[1:].to_numpy(dtype=float),
            future["lagged"].loss1.loc[prefix_end:].iloc[1:].to_numpy(dtype=float),
        ]
    )
    finite = np.isfinite(original_losses) & np.isfinite(mutated_losses)
    changes = np.abs(original_losses[finite] - mutated_losses[finite])
    changed_cells = int((changes > spec.numerical_tolerance).sum())
    max_change = float(changes.max()) if changes.size else 0.0

    discount_cells = sum(
        int((evidence["lagged"].c01[beta][lambda0].dropna() < lambda0).sum())
        + int((evidence["lagged"].c10[beta][lambda0].dropna() < lambda0).sum())
        for beta in spec.event_betas
        for lambda0 in spec.event_lambdas
    )
    refit_error, stale_distance, refit_date = _refit_convention_errors(
        inputs, evidence["lagged"], spec
    )
    candidate_cells = prefix_terminal_limit * len(spec.lambdas)
    checks = {
        "sealed_arrival_exact": arrival_cells == candidate_cells * len(spec.betas),
        "beta_zero_exact": beta_cells == candidate_cells * len(spec.rules),
        "prefix_invariant": prefix_cells
        == candidate_cells * len(spec.rules) * len(spec.betas),
        "future_mutation_effect_present": changed_cells > 0,
        "refit_convention_numeric": refit_error <= spec.numerical_tolerance
        and stale_distance > spec.numerical_tolerance,
        "lagged_discounts_present": discount_cells > 0,
    }
    if not all(checks.values()):
        raise LaggedStudyError(f"US locked smoke failed: {checks}")
    mechanics = mechanical_prerequisites(
        penalty_builders, atol=spec.numerical_tolerance
    )
    return {
        "status": "passed",
        "market": "us",
        "terminal_dates": prefix_terminal_limit,
        "generated_terminal_dates": generation_limit,
        "refit_probe_date": refit_date.date().isoformat(),
        **checks,
        "mechanical_prerequisites": mechanics,
        "refit_convention": "current-fit parameters applied to previous-row loss",
        "sealed_arrival_state_cells_checked": arrival_cells,
        "beta_zero_state_cells_checked": beta_cells,
        "prefix_state_cells_checked": prefix_cells,
        "future_mutation_loss_cells_changed": changed_cells,
        "future_mutation_max_abs_loss_change": max_change,
        "lagged_discount_cells": discount_cells,
        "refit_convention_stale_distance": stale_distance,
        "refit_convention_max_abs_error": refit_error,
        "performance_files_accessed": False,
        "return_columns_accessed": False,
        "post_2023_accessed": False,
    }


def _brute_force_value(loss: np.ndarray, penalty: np.ndarray) -> float:
    values = []
    for path in itertools.product(range(loss.shape[1]), repeat=len(loss)):
        value = sum(loss[t, state] for t, state in enumerate(path))
        value += sum(penalty[t, path[t - 1], path[t]] for t in range(1, len(path)))
        values.append(float(value))
    return min(values)


def _online_path(loss: np.ndarray, penalty: np.ndarray) -> list[int]:
    return dp_tv(loss, penalty, return_value_mx=True).argmin(axis=1).tolist()


def _rule_evidence(
    rule: str,
    builder: PenaltyBuilder,
    loss: np.ndarray,
    lambda0: float,
    beta: float,
    q_train: float,
    atol: float,
) -> dict[str, Any]:
    checks = {
        "formula": False,
        "bounds": False,
        "beta_zero": False,
        "scale_invariance": False,
        "brute_force": False,
    }
    formula_error: float | None = None
    try:
        penalty = np.asarray(builder(loss, lambda0, beta, q_train), dtype=float)
        expected = _formula_penalty(loss, lambda0, beta, q_train, rule)
        if penalty.shape == expected.shape:
            measured_error = float(np.max(np.abs(penalty - expected)))
            formula_error = measured_error if math.isfinite(measured_error) else None
            off_diagonal = penalty[:, ~np.eye(2, dtype=bool)]
            diagonal = np.diagonal(penalty, axis1=1, axis2=2)
            checks["formula"] = formula_error is not None and formula_error <= atol
            checks["bounds"] = bool(
                np.isfinite(penalty).all()
                and (off_diagonal >= lambda0 * math.exp(-beta) - atol).all()
                and (off_diagonal <= lambda0 + atol).all()
                and (np.abs(diagonal) <= atol).all()
            )
            beta_zero = np.asarray(builder(loss, lambda0, 0.0, q_train), dtype=float)
            checks["beta_zero"] = np.array_equal(
                beta_zero, lam_to_penalty_seq(np.full(len(loss), lambda0), 2)
            )
            scaled = np.asarray(
                builder(loss * 8.0, lambda0, beta, q_train * 8.0), dtype=float
            )
            checks["scale_invariance"] = bool(
                scaled.shape == penalty.shape
                and np.allclose(scaled, penalty, rtol=0.0, atol=atol)
            )
            path, value = dp_tv(loss, penalty)
            path_value = sum(loss[t, state] for t, state in enumerate(path))
            path_value += sum(
                penalty[t, path[t - 1], path[t]] for t in range(1, len(path))
            )
            brute = _brute_force_value(loss, penalty)
            checks["brute_force"] = bool(
                abs(float(value) - brute) <= atol
                and abs(float(path_value) - brute) <= atol
            )
    except (TypeError, ValueError, IndexError):
        pass
    return {"checks": checks, "max_formula_abs_error": formula_error}


def mechanical_prerequisites(
    penalty_builders: Mapping[str, PenaltyBuilder], *, atol: float = 1e-12
) -> dict[str, Any]:
    """Compute formula, objective, limiting cases, and exact toy paths."""
    required = {"arrival", "lagged"}
    if set(penalty_builders) != required or not math.isfinite(atol) or atol < 0:
        raise LockedModelError("mechanical prerequisite controls are invalid")
    loss = np.array([[0.2, 3.1], [2.4, 0.1], [0.7, 1.6], [3.0, 0.3]], dtype=float)
    lambda0, beta, q_train = 4.0, math.log(4.0), 1.7
    by_rule = {
        rule: _rule_evidence(rule, builder, loss, lambda0, beta, q_train, atol)
        for rule, builder in penalty_builders.items()
    }

    toy_losses = {
        "isolated": np.array(
            [[0.0, 8.0], [0.0, 8.0], [2.0, 0.0], [0.0, 2.0], [0.0, 8.0]]
        ),
        "alternating": np.array(
            [[0.0, 8.0], [0.0, 8.0]] + [[2.0, 0.0], [0.0, 2.0]] * 5
        ),
        "persistent": np.array([[0.0, 8.0], [0.0, 8.0]] + [[1.0, 0.0]] * 6),
    }
    toy_paths: dict[str, dict[str, list[int]]] = {}
    try:
        for name, toy_loss in toy_losses.items():
            toy_paths[name] = {
                "fixed": _online_path(
                    toy_loss,
                    lam_to_penalty_seq(np.full(len(toy_loss), 4.0), 2),
                ),
                "arrival": _online_path(
                    toy_loss,
                    penalty_builders["arrival"](toy_loss, 4.0, math.log(4.0), 1.0),
                ),
                "lagged": _online_path(
                    toy_loss,
                    penalty_builders["lagged"](toy_loss, 4.0, math.log(4.0), 1.0),
                ),
            }
    except (TypeError, ValueError, IndexError):
        toy_paths = {}

    checks = {
        name: all(rule["checks"][name] for rule in by_rule.values())
        for name in (
            "formula",
            "bounds",
            "beta_zero",
            "scale_invariance",
            "brute_force",
        )
    }
    checks["toy_paths"] = toy_paths == EXPECTED_TOY_PATHS
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "by_rule": by_rule,
        "toy_paths": toy_paths,
    }
