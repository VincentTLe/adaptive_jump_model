"""Pure locked-parameter candidate generation for evidence-penalty studies."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from adaptive_jump.config import ResearchConfig
from adaptive_jump.lagged_study import LaggedMechanismSpec
from adaptive_jump.models import FEATURE_COLUMNS
from adaptive_jump.tv_jump import dp_tv, lam_to_penalty_seq, loss_matrix

PenaltyBuilder = Callable[[np.ndarray, float, float, float], np.ndarray]


class LockedModelError(ValueError):
    """Raised when sealed inputs cannot reproduce the locked model path."""


@dataclass
class LockedStateEvidence:
    """StateEvidence-compatible evidence generated without fitting or returns."""

    states: dict[float, pd.DataFrame]
    loss0: pd.DataFrame
    loss1: pd.DataFrame
    q_train: pd.DataFrame
    c01: dict[float, pd.DataFrame]
    c10: dict[float, pd.DataFrame]
    refits: pd.DataFrame


def _feature_inputs(
    frame: pd.DataFrame, spec: LaggedMechanismSpec
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    required = ("date", *FEATURE_COLUMNS)
    missing = [column for column in required if column not in frame]
    if missing:
        raise LockedModelError(f"missing locked-model columns: {missing}")
    prepared = frame.loc[:, required].copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="raise")
    dates = pd.DatetimeIndex(prepared["date"], name="date")
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise LockedModelError("feature dates must be increasing and unique")
    if len(dates) == 0 or dates.max().date() > spec.data_cutoff:
        raise LockedModelError("feature dates exceed the frozen scope")
    for column in FEATURE_COLUMNS:
        prepared[column] = pd.to_numeric(prepared[column], errors="raise")
    values = prepared.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise LockedModelError("observed feature values must be finite")
    complete = prepared.dropna(subset=list(FEATURE_COLUMNS)).reset_index(drop=True)
    return complete, dates


def _fixed_inputs(
    frame: pd.DataFrame,
    all_dates: pd.DatetimeIndex,
    model_dates: pd.DatetimeIndex,
    lambdas: tuple[float, ...],
    fit_window: int,
) -> pd.DataFrame:
    fixed = frame.copy()
    if "date" in fixed:
        fixed["date"] = pd.to_datetime(fixed["date"], errors="raise")
        fixed = fixed.set_index("date")
    else:
        fixed.index = pd.to_datetime(fixed.index, errors="raise")
    fixed.index = pd.DatetimeIndex(fixed.index, name="date")
    if not fixed.index.equals(all_dates):
        raise LockedModelError("fixed-state and feature date indexes differ")
    try:
        fixed.columns = tuple(float(column) for column in fixed.columns)
    except (TypeError, ValueError) as exc:
        raise LockedModelError("fixed-state lambda columns are invalid") from exc
    if tuple(fixed.columns) != lambdas or fixed.columns.has_duplicates:
        raise LockedModelError("fixed-state lambda grid changed")
    for column in fixed:
        fixed[column] = pd.to_numeric(fixed[column], errors="raise")
    present = fixed.notna().to_numpy()
    if not np.array_equal(present.any(axis=1), present.all(axis=1)):
        raise LockedModelError("fixed-state table has partial candidate rows")
    observed = fixed.to_numpy(dtype=float)[present]
    if observed.size and not np.isin(observed, (0.0, 1.0)).all():
        raise LockedModelError("fixed states must be binary")
    expected = model_dates[fit_window - 1 :]
    actual = fixed.index[present.all(axis=1)]
    if not actual.equals(expected):
        raise LockedModelError("fixed-state terminal coverage changed")
    return fixed


def _decode_array(value: Any) -> np.ndarray:
    try:
        decoded = json.loads(value) if isinstance(value, str) else value
        return np.asarray(decoded, dtype=float)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise LockedModelError("sealed refit parameters are invalid") from exc


def _sealed_inputs(
    frame: pd.DataFrame,
    model_dates: pd.DatetimeIndex,
    lambdas: tuple[float, ...],
    fit_window: int,
    spec: LaggedMechanismSpec,
    market: str,
) -> tuple[
    pd.DataFrame,
    dict[tuple[pd.Timestamp, float], tuple[np.ndarray, np.ndarray, np.ndarray, float]],
]:
    required = {
        "fit_date",
        "training_start",
        "training_end",
        "lambda0",
        "q_train",
        "scaler_mean",
        "scaler_scale",
        "centers",
    }
    if not required.issubset(frame):
        raise LockedModelError("sealed refit table is incomplete")
    refits = frame.copy()
    for column in ("fit_date", "training_start", "training_end"):
        refits[column] = pd.to_datetime(refits[column], errors="raise")
    for column in ("lambda0", "q_train"):
        refits[column] = pd.to_numeric(refits[column], errors="raise")
    if "market" in refits and set(refits["market"]) != {market}:
        raise LockedModelError("sealed refit market changed")
    if (
        len(refits) == 0
        or refits.duplicated(["fit_date", "lambda0"]).any()
        or (refits["training_end"] != refits["fit_date"]).any()
        or refits["fit_date"].max().date() > spec.data_cutoff
        or not np.isfinite(refits["q_train"]).all()
        or (refits["q_train"] <= 0).any()
    ):
        raise LockedModelError("sealed refit rows are invalid")

    fit_dates = pd.DatetimeIndex(sorted(refits["fit_date"].unique()))
    state_dates = model_dates[fit_window - 1 :]
    if len(state_dates) == 0 or fit_dates[0] != state_dates[0]:
        raise LockedModelError("first sealed fit is not the first terminal date")
    if not fit_dates.isin(state_dates).all():
        raise LockedModelError("sealed fit date is outside terminal coverage")
    for fit_date, rows in refits.groupby("fit_date", sort=True):
        if tuple(sorted(rows["lambda0"].astype(float))) != tuple(sorted(lambdas)):
            raise LockedModelError(f"{fit_date.date()}: refit lambda coverage changed")
        terminal = int(model_dates.searchsorted(fit_date))
        dates = model_dates[terminal - fit_window + 1 : terminal + 1]
        if (
            len(dates) != fit_window
            or dates[-1] != fit_date
            or (rows["training_start"] != dates[0]).any()
            or (rows["training_end"] != dates[-1]).any()
        ):
            raise LockedModelError(f"{fit_date.date()}: training window changed")

    parameters: dict[
        tuple[pd.Timestamp, float], tuple[np.ndarray, np.ndarray, np.ndarray, float]
    ] = {}
    for _, row in refits.iterrows():
        mean = _decode_array(row["scaler_mean"])
        scale = _decode_array(row["scaler_scale"])
        centers = _decode_array(row["centers"])
        centers_have_shape = centers.shape == (2, len(FEATURE_COLUMNS))
        finite_center_rows = (
            np.isfinite(centers).all(axis=1)
            if centers_have_shape
            else np.zeros(2, dtype=bool)
        )
        empty_center_rows = (
            np.isnan(centers).all(axis=1)
            if centers_have_shape
            else np.zeros(2, dtype=bool)
        )
        if (
            mean.shape != (len(FEATURE_COLUMNS),)
            or scale.shape != mean.shape
            or not centers_have_shape
            or not np.isfinite(mean).all()
            or not np.isfinite(scale).all()
            or (scale <= 0).any()
            or not np.logical_or(finite_center_rows, empty_center_rows).all()
            or not finite_center_rows.any()
        ):
            raise LockedModelError("sealed scaler or centers are invalid")
        key = (pd.Timestamp(row["fit_date"]), float(row["lambda0"]))
        parameters[key] = (mean, scale, centers, float(row["q_train"]))
    return refits.sort_values(["fit_date", "lambda0"]).reset_index(
        drop=True
    ), parameters


def _empty_evidence(
    dates: pd.DatetimeIndex,
    lambdas: tuple[float, ...],
    betas: tuple[float, ...],
) -> LockedStateEvidence:
    def frame() -> pd.DataFrame:
        return pd.DataFrame(index=dates, columns=lambdas, dtype=float)

    return LockedStateEvidence(
        states={beta: frame() for beta in betas},
        loss0=frame(),
        loss1=frame(),
        q_train=frame(),
        c01={beta: frame() for beta in betas},
        c10={beta: frame() for beta in betas},
        refits=pd.DataFrame(),
    )


def generate_locked_candidates(
    feature_frame: pd.DataFrame,
    fixed_states: pd.DataFrame,
    sealed_refits: pd.DataFrame,
    config: ResearchConfig,
    spec: LaggedMechanismSpec,
    *,
    market: str,
    penalty_builders: Mapping[str, PenaltyBuilder],
    terminal_limit: int | None = None,
) -> dict[str, LockedStateEvidence]:
    """Generate candidate states using only features and sealed fit parameters."""
    lambdas = tuple(float(value) for value in spec.lambdas)
    betas = tuple(float(value) for value in spec.betas)
    rules = tuple(spec.rules)
    fit_window = int(spec.fit_window)
    if (
        market not in spec.markets
        or config.model_protocol.n_states != 2
        or config.model_protocol.fit_window != fit_window
        or tuple(config.jm_protocol.lambda_grid) != lambdas
        or set(penalty_builders) != set(rules)
        or len(set(betas)) != len(betas)
        or 0.0 not in betas
        or not np.isfinite(betas).all()
        or min(betas) < 0
    ):
        raise LockedModelError("locked model controls changed")
    if terminal_limit is not None and (
        isinstance(terminal_limit, bool)
        or not isinstance(terminal_limit, int)
        or terminal_limit < 1
    ):
        raise LockedModelError("terminal_limit must be a positive integer")

    complete, all_dates = _feature_inputs(feature_frame, spec)
    model_dates = pd.DatetimeIndex(complete["date"], name="date")
    if len(model_dates) < fit_window:
        raise LockedModelError("insufficient complete feature observations")
    fixed = _fixed_inputs(fixed_states, all_dates, model_dates, lambdas, fit_window)
    refits, parameters = _sealed_inputs(
        sealed_refits, model_dates, lambdas, fit_window, spec, market
    )
    fit_dates = pd.DatetimeIndex(sorted(refits["fit_date"].unique()))
    outputs = {rule: _empty_evidence(all_dates, lambdas, betas) for rule in rules}
    shared_loss0 = next(iter(outputs.values())).loss0
    shared_loss1 = next(iter(outputs.values())).loss1
    shared_q = next(iter(outputs.values())).q_train

    final_terminal = len(complete)
    if terminal_limit is not None:
        final_terminal = min(final_terminal, fit_window - 1 + terminal_limit)
    for terminal in range(fit_window - 1, final_terminal):
        window = complete.iloc[terminal - fit_window + 1 : terminal + 1]
        current_date = pd.Timestamp(window.iloc[-1]["date"])
        fit_position = int(fit_dates.searchsorted(current_date, side="right")) - 1
        if fit_position < 0:
            raise LockedModelError(f"{current_date.date()}: no sealed fit available")
        fit_date = pd.Timestamp(fit_dates[fit_position])
        raw = window.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
        for lambda0 in lambdas:
            mean, scale, centers, q_train = parameters[(fit_date, lambda0)]
            losses = loss_matrix((raw - mean) / scale, centers)
            if not np.isfinite(losses).any(axis=1).all():
                raise LockedModelError(
                    f"{current_date.date()}/{lambda0:g}: invalid loss"
                )
            shared_loss0.loc[current_date, lambda0] = losses[-1, 0]
            shared_loss1.loc[current_date, lambda0] = losses[-1, 1]
            shared_q.loc[current_date, lambda0] = q_train
            fixed_penalty = lam_to_penalty_seq(
                np.full(fit_window, lambda0), config.model_protocol.n_states
            )
            for rule, builder in penalty_builders.items():
                evidence = outputs[rule]
                for beta in betas:
                    penalty = np.asarray(
                        builder(losses, lambda0, beta, q_train), dtype=float
                    )
                    if (
                        penalty.shape != fixed_penalty.shape
                        or not np.isfinite(penalty).all()
                        or (penalty < 0).any()
                        or not np.array_equal(
                            np.diagonal(penalty, axis1=1, axis2=2),
                            np.zeros((fit_window, 2)),
                        )
                    ):
                        raise LockedModelError(f"{rule}: penalty builder is invalid")
                    if beta == 0.0 and not np.array_equal(penalty, fixed_penalty):
                        raise LockedModelError(f"{rule}: beta zero is not fixed JM")
                    values = dp_tv(losses, penalty, return_value_mx=True)
                    evidence.states[beta].loc[current_date, lambda0] = int(
                        values[-1].argmin()
                    )
                    evidence.c01[beta].loc[current_date, lambda0] = penalty[-1, 0, 1]
                    evidence.c10[beta].loc[current_date, lambda0] = penalty[-1, 1, 0]

    for evidence in outputs.values():
        evidence.loss0 = shared_loss0.copy()
        evidence.loss1 = shared_loss1.copy()
        evidence.q_train = shared_q.copy()
        evidence.refits = refits.copy()
        for beta, states in evidence.states.items():
            states.index.name = "date"
            populated = states.notna().any(axis=1)
            if beta == 0.0 and not np.array_equal(
                states.loc[populated].to_numpy(),
                fixed.loc[populated].to_numpy(),
                equal_nan=True,
            ):
                raise LockedModelError("sealed beta-zero path differs from fixed JM")
    return outputs
