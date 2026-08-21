"""Arm construction for the AJM-EXT-001 external runner.

One arm = one paired specification: a lambda grid times a standardizer. The
fixed leg reuses the sealed walk-forward engine (`fixed_jm_states`, which
carries the objective gate). The challenger leg re-decodes every day with the
lagged-evidence penalty, holding the fixed leg's scaler and centers sealed —
and every day is also decoded at beta zero, which must reproduce the fixed leg
exactly. That equality is the `beta_zero_nests_fixed` health gate executed on
the real data, not only in unit tests.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from adaptive_jump.config import ModelProtocol
from adaptive_jump.features import make_features, standardize_expanding
from adaptive_jump.infrastructure.runtime.processes import process_pool_context
from adaptive_jump.models import FEATURE_COLUMNS, FixedJMResult, fixed_jm_states
from adaptive_jump.tv_jump import (
    dp_tv,
    lagged_evidence_penalty_seq,
    loss_matrix,
    robust_loss_scale,
)

TRAILING = "trailing_3000_ddof0"
EXPANDING = "causal_expanding_ddof1_min63"


class ArmError(ValueError):
    """Raised when an arm violates the frozen construction."""


@dataclass(frozen=True)
class ArmStates:
    fixed: pd.DataFrame
    challenger: pd.DataFrame
    refits: pd.DataFrame
    q_train: pd.DataFrame


def build_spec_frame(
    region_frame: pd.DataFrame,
    standardizer: str,
    *,
    downside_halflife: int = 10,
    sortino_halflives: tuple[int, ...] = (20, 60),
    min_observations: int = 63,
) -> pd.DataFrame:
    """Attach the paper's three features under one standardizer variant."""
    features = make_features(
        region_frame["excess_return"],
        downside_halflife=downside_halflife,
        sortino_halflives=sortino_halflives,
        adjust=True,
        ignore_na=False,
    )
    if standardizer == EXPANDING:
        features = standardize_expanding(features, min_observations)
    elif standardizer != TRAILING:
        raise ArmError(f"unknown standardizer: {standardizer}")
    return pd.concat([region_frame.reset_index(drop=True), features], axis=1)


def model_protocol_for(standardizer: str, fit_window: int) -> ModelProtocol:
    if standardizer == TRAILING:
        return ModelProtocol(2, fit_window, 0, 1)
    if standardizer == EXPANDING:
        return ModelProtocol(
            2, fit_window, 0, 1, "expanding_full_history_ddof1", 63
        )
    raise ArmError(f"unknown standardizer: {standardizer}")


def _complete_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """The same completeness rule fixed_jm_states applies, kept in lockstep.

    Any drift between this and the sealed engine is caught by the full-sample
    beta-zero equality gate in challenger_states, so the duplication cannot
    silently diverge.
    """
    required = ("date", *FEATURE_COLUMNS, "excess_return")
    prepared = frame.loc[:, list(required)].copy()
    dates = pd.to_datetime(prepared["date"], errors="raise")
    prepared["date"] = dates
    all_dates = pd.DatetimeIndex(dates, name="date")
    complete = prepared.dropna().reset_index(drop=True)
    return complete, all_dates


def _decode_chunk(
    raw: np.ndarray,
    terminals: list[int],
    fit_positions: list[int],
    parameters: list[dict[float, tuple[np.ndarray, np.ndarray, np.ndarray, float]]],
    lambdas: tuple[float, ...],
    beta: float,
    fit_window: int,
) -> list[list[tuple[int, int]]]:
    """Decode (beta-zero, beta) terminal states for a chunk of days."""
    with threadpool_limits(1):
        results: list[list[tuple[int, int]]] = []
        for terminal, fit_position in zip(terminals, fit_positions, strict=True):
            window = raw[terminal - fit_window + 1 : terminal + 1]
            per_lambda: list[tuple[int, int]] = []
            for lambda0 in lambdas:
                mean, scale, centers, q_train = parameters[fit_position][lambda0]
                losses = loss_matrix((window - mean) / scale, centers)
                zero = lagged_evidence_penalty_seq(losses, lambda0, 0.0, q_train)
                lagged = lagged_evidence_penalty_seq(losses, lambda0, beta, q_train)
                state_zero = int(dp_tv(losses, zero, return_value_mx=True)[-1].argmin())
                state_beta = int(
                    dp_tv(losses, lagged, return_value_mx=True)[-1].argmin()
                )
                per_lambda.append((state_zero, state_beta))
            results.append(per_lambda)
        return results


def challenger_states(
    spec_frame: pd.DataFrame,
    fixed: FixedJMResult,
    lambdas: tuple[float, ...],
    beta: float,
    *,
    fit_window: int,
    n_jobs: int = 1,
    chunk_days: int = 128,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decode the lagged-evidence challenger against sealed fixed parameters.

    Returns (challenger states, q_train per fit_date x lambda). Raises if any
    beta-zero decode differs from the sealed fixed state on any day.
    """
    if beta <= 0 or not np.isfinite(beta):
        raise ArmError("challenger beta must be positive and finite")
    complete, all_dates = _complete_rows(spec_frame)
    if len(complete) < fit_window:
        raise ArmError("insufficient complete observations")
    raw = complete.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    model_dates = pd.DatetimeIndex(complete["date"])

    refits = fixed.refits
    needed = {"fit_date", "lambda", "scaler_mean", "scaler_scale", "centers"}
    if not needed <= set(refits.columns):
        raise ArmError("fixed refits lack challenger inputs; enable diagnostics")
    fit_dates = pd.DatetimeIndex(sorted(pd.to_datetime(refits["fit_date"]).unique()))
    parameters: list[dict[float, tuple]] = []
    q_rows: list[dict[str, object]] = []
    for fit_date in fit_dates:
        rows = refits.loc[pd.to_datetime(refits["fit_date"]) == fit_date]
        end = int(model_dates.searchsorted(fit_date, side="right")) - 1
        window = raw[end - fit_window + 1 : end + 1]
        per_lambda: dict[float, tuple] = {}
        for row in rows.to_dict("records"):
            mean = np.asarray(row["scaler_mean"], dtype=float)
            scale = np.asarray(row["scaler_scale"], dtype=float)
            centers = np.asarray(row["centers"], dtype=float)
            losses = loss_matrix((window - mean) / scale, centers)
            q_train = robust_loss_scale(losses)
            penalty = float(row["lambda"])
            per_lambda[penalty] = (mean, scale, centers, q_train)
            q_rows.append(
                {"fit_date": fit_date, "lambda": penalty, "q_train": q_train}
            )
        if set(per_lambda) != set(float(v) for v in lambdas):
            raise ArmError(f"{fit_date.date()}: refit lambdas differ from the grid")
        parameters.append(per_lambda)

    first_terminal = fit_window - 1
    terminals = list(range(first_terminal, len(complete)))
    positions = [
        int(fit_dates.searchsorted(model_dates[t], side="right")) - 1
        for t in terminals
    ]
    if any(position < 0 for position in positions):
        raise ArmError("a decode day precedes the first refit")

    chunks = [
        (terminals[i : i + chunk_days], positions[i : i + chunk_days])
        for i in range(0, len(terminals), chunk_days)
    ]
    if n_jobs > 1 and len(chunks) > 1:
        with ProcessPoolExecutor(
            max_workers=n_jobs, mp_context=process_pool_context()
        ) as pool:
            decoded = list(
                pool.map(
                    _decode_chunk,
                    [raw] * len(chunks),
                    [c[0] for c in chunks],
                    [c[1] for c in chunks],
                    [parameters] * len(chunks),
                    [tuple(lambdas)] * len(chunks),
                    [beta] * len(chunks),
                    [fit_window] * len(chunks),
                )
            )
    else:
        decoded = [
            _decode_chunk(
                raw, days, pos, parameters, tuple(lambdas), beta, fit_window
            )
            for days, pos in chunks
        ]

    states = pd.DataFrame(index=all_dates, columns=list(lambdas), dtype=float)
    mismatches: list[str] = []
    flat = [item for chunk in decoded for item in chunk]
    for terminal, per_lambda in zip(terminals, flat, strict=True):
        day = model_dates[terminal]
        for lambda0, (state_zero, state_beta) in zip(lambdas, per_lambda, strict=True):
            sealed = fixed.states.loc[day, lambda0]
            if not np.isfinite(sealed) or int(sealed) != state_zero:
                mismatches.append(f"{day.date()}/{lambda0:g}")
            states.loc[day, lambda0] = float(state_beta)
    if mismatches:
        raise ArmError(
            "beta-zero decode differs from the sealed fixed path on "
            f"{len(mismatches)} cells (first: {mismatches[0]}); the challenger "
            "is not consuming the sealed parameters faithfully"
        )
    states.index.name = "date"
    return states, pd.DataFrame.from_records(q_rows)


def build_arm(
    spec_frame: pd.DataFrame,
    model_protocol: ModelProtocol,
    jm_protocol,
    beta: float,
    *,
    n_jobs: int = 1,
    progress=None,
    checkpoint_every: int = 50,
) -> ArmStates:
    """Fixed leg via the sealed engine, then the paired challenger decode."""
    fixed = fixed_jm_states(
        spec_frame,
        model_protocol,
        jm_protocol,
        include_fit_diagnostics=True,
        n_jobs=n_jobs,
        progress=progress,
        checkpoint_every=checkpoint_every,
    )
    challenger, q_train = challenger_states(
        spec_frame,
        fixed,
        tuple(jm_protocol.lambda_grid),
        beta,
        fit_window=model_protocol.fit_window,
        n_jobs=n_jobs,
    )
    return ArmStates(
        fixed=fixed.states,
        challenger=challenger,
        refits=fixed.refits,
        q_train=q_train,
    )
