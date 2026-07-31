"""Independent formula, candidate, path, and penalty replay for verification."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from adaptive_jump.balanced_mechanics import independent_balanced_penalty
from adaptive_jump.balanced_model import BalancedSpec, BalancedStudyError, beta_label
from adaptive_jump.config import ResearchConfig
from adaptive_jump.lagged_model import LockedStateEvidence, generate_locked_candidates
from adaptive_jump.separation_analysis import MarketInputs


def _assert_frame_exact(
    actual: pd.DataFrame, expected: pd.DataFrame, label: str
) -> None:
    try:
        pd.testing.assert_frame_equal(
            actual,
            expected,
            check_exact=True,
            check_dtype=False,
            check_freq=False,
        )
    except AssertionError as exc:
        raise BalancedStudyError(f"{label} changed") from exc


def independent_lagged_penalty(
    loss_mx: np.ndarray, lambda0: float, beta: float, q_train: float
) -> np.ndarray:
    """Reconstruct the sealed one-sided lagged formula without its builder."""
    loss = np.asarray(loss_mx, dtype=float)
    if loss.ndim != 2 or loss.shape[0] == 0 or loss.shape[1] < 2:
        raise ValueError("independent lagged loss has invalid shape")
    if np.isneginf(loss).any() or not np.isfinite(loss).any(axis=1).all():
        raise ValueError("independent lagged loss has an unavailable row")
    if not np.isfinite(lambda0) or lambda0 < 0:
        raise ValueError("independent lagged lambda is invalid")
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("independent lagged beta is invalid")
    if not np.isfinite(q_train) or q_train <= 0:
        raise ValueError("independent lagged scale is invalid")
    clean = np.where(np.isnan(loss), np.inf, loss)
    lagged = np.zeros_like(clean)
    lagged[1:] = clean[:-1]
    source = lagged[:, :, None]
    destination = lagged[:, None, :]
    with np.errstate(invalid="ignore"):
        gap = source - destination
    gap[np.isposinf(source) & np.isposinf(destination)] = 0.0
    gap = np.maximum(gap, 0.0)
    penalty = float(lambda0) * np.exp(-float(beta) * np.tanh(gap / float(q_train)))
    states = np.arange(loss.shape[1])
    penalty[:, states, states] = 0.0
    return penalty


INDEPENDENT_BUILDERS = {
    "lagged": independent_lagged_penalty,
    "balanced": independent_balanced_penalty,
}


def independent_candidates(
    inputs: MarketInputs,
    fixed: pd.DataFrame,
    config: ResearchConfig,
    spec: BalancedSpec,
) -> dict[str, LockedStateEvidence]:
    return generate_locked_candidates(
        inputs.features.reset_index(),
        fixed,
        inputs.refits,
        config,
        spec,
        market=inputs.market,
        penalty_builders=INDEPENDENT_BUILDERS,
    )


def _assert_and_count(actual: pd.DataFrame, expected: pd.DataFrame, label: str) -> int:
    _assert_frame_exact(actual, expected, label)
    return int(actual.notna().sum().sum())


def candidate_checks(
    inputs: MarketInputs,
    fixed: pd.DataFrame,
    evidence: dict[str, LockedStateEvidence],
    spec: BalancedSpec,
) -> dict[str, Any]:
    parent_cells = sum(
        _assert_and_count(
            evidence["lagged"].states[beta],
            inputs.candidates[beta],
            f"{inputs.market}/{beta_label(beta)} parent",
        )
        for beta in spec.betas
    )
    beta_zero_cells = sum(
        _assert_and_count(
            evidence[rule].states[0.0],
            fixed,
            f"{inputs.market}/{rule} beta zero",
        )
        for rule in spec.rules
    )
    discounts = 0
    surcharges = 0
    pair_error = 0.0
    for lambda0 in spec.event_lambdas:
        c01 = evidence["balanced"].c01[spec.decision_beta][lambda0]
        c10 = evidence["balanced"].c10[spec.decision_beta][lambda0]
        valid = c01.notna() & c10.notna()
        left = c01.loc[valid].to_numpy(dtype=float)
        right = c10.loc[valid].to_numpy(dtype=float)
        joined = np.concatenate((left, right))
        discounts += int((joined < lambda0).sum())
        surcharges += int((joined > lambda0).sum())
        if len(left):
            pair_error = max(
                pair_error,
                float(np.max(np.abs(left + right - 2.0 * lambda0))),
            )
    terminal_rows = int(fixed.notna().all(axis=1).sum())
    cells_per_path = terminal_rows * len(spec.lambdas)
    all_candidate_cells = sum(
        int(evidence[rule].states[beta].notna().sum().sum())
        for rule in spec.rules
        for beta in spec.betas
    )
    return {
        "parent_lagged_exact": parent_cells == cells_per_path * len(spec.betas),
        "beta_zero_exact": beta_zero_cells == cells_per_path * len(spec.rules),
        "candidate_coverage_exact": all_candidate_cells
        == cells_per_path * len(spec.rules) * len(spec.betas),
        "pair_balance_exact": pair_error <= spec.numerical_tolerance,
        "terminal_rows": terminal_rows,
        "parent_lagged_state_cells_checked": parent_cells,
        "beta_zero_state_cells_checked": beta_zero_cells,
        "all_candidate_state_cells_checked": all_candidate_cells,
        "balanced_discount_cells": discounts,
        "balanced_surcharge_cells": surcharges,
        "maximum_pair_sum_abs_error": pair_error,
        "return_columns_accessed": False,
    }


def path_behavior(
    inputs: MarketInputs,
    evidence: dict[str, LockedStateEvidence],
    spec: BalancedSpec,
) -> pd.DataFrame:
    fixed = inputs.candidates[0.0]
    lagged = inputs.candidates[spec.decision_beta]
    start = pd.Timestamp(spec.evaluation_starts[inputs.market])
    rows: list[dict[str, Any]] = []
    for rule in spec.rules:
        states = evidence[rule].states[spec.decision_beta]
        for lambda0 in spec.event_lambdas:
            complete = states[lambda0].dropna().astype(int)
            first = int(complete.index.searchsorted(start, side="left"))
            selected = complete.iloc[first:]
            if selected.empty:
                raise BalancedStudyError(f"{inputs.market}/{rule}: empty replay path")
            fixed_path = fixed[lambda0].reindex(selected.index).astype(int)
            lagged_path = lagged[lambda0].reindex(selected.index).astype(int)
            switch_path = complete.iloc[max(0, first - 1) :].to_numpy(dtype=int)
            rows.append(
                {
                    "market": inputs.market,
                    "rule": rule,
                    "beta": spec.decision_beta,
                    "beta_label": beta_label(spec.decision_beta),
                    "lambda0": lambda0,
                    "start": selected.index[0],
                    "end": selected.index[-1],
                    "observations": len(selected),
                    "switch_count": int(np.count_nonzero(np.diff(switch_path))),
                    "state_0_count": int((selected == 0).sum()),
                    "state_1_count": int((selected == 1).sum()),
                    "state_differences_vs_fixed": int(
                        np.count_nonzero(selected.to_numpy() != fixed_path.to_numpy())
                    ),
                    "state_differences_vs_lagged": int(
                        np.count_nonzero(selected.to_numpy() != lagged_path.to_numpy())
                    ),
                }
            )
    return pd.DataFrame.from_records(rows)


def penalty_summary(
    market: str,
    evidence: dict[str, LockedStateEvidence],
    spec: BalancedSpec,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rule in spec.rules:
        for lambda0 in spec.event_lambdas:
            c01 = evidence[rule].c01[spec.decision_beta][lambda0]
            c10 = evidence[rule].c10[spec.decision_beta][lambda0]
            valid = c01.notna() & c10.notna()
            left = c01.loc[valid].to_numpy(dtype=float)
            right = c10.loc[valid].to_numpy(dtype=float)
            if len(left) == 0:
                raise BalancedStudyError(f"{market}/{rule}: empty penalty replay")
            joined = np.concatenate((left, right))
            rows.append(
                {
                    "market": market,
                    "rule": rule,
                    "beta": spec.decision_beta,
                    "beta_label": beta_label(spec.decision_beta),
                    "lambda0": lambda0,
                    "observations": len(left),
                    "minimum_cost_ratio": float(np.min(joined) / lambda0),
                    "maximum_cost_ratio": float(np.max(joined) / lambda0),
                    "median_pair_mean_ratio": float(
                        np.median((left + right) / (2.0 * lambda0))
                    ),
                    "maximum_pair_sum_abs_error": float(
                        np.max(np.abs(left + right - 2.0 * lambda0))
                    ),
                    "discount_cells": int((joined < lambda0).sum()),
                    "surcharge_cells": int((joined > lambda0).sum()),
                }
            )
    return pd.DataFrame.from_records(rows)
