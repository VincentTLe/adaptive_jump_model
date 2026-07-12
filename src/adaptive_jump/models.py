"""Causal adapters for the frozen fixed JM and HMM baselines."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from jumpmodels.jump import JumpModel
from sklearn.preprocessing import StandardScaler

from adaptive_jump.config import JMProtocol, ModelProtocol

FEATURE_COLUMNS = ("dd_10", "sortino_20", "sortino_60")


class ModelError(ValueError):
    """Raised when model inputs or fitted outputs violate the protocol."""


@dataclass(frozen=True)
class FixedJMResult:
    """Daily candidate states and auditable semiannual fit records."""

    states: pd.DataFrame
    refits: pd.DataFrame


@dataclass
class _FixedJMFit:
    scaler: StandardScaler
    models: dict[float, JumpModel]


def fixed_jm_states(
    frame: pd.DataFrame,
    model_protocol: ModelProtocol,
    jm_protocol: JMProtocol,
    *,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> FixedJMResult:
    """Generate causal terminal online states for every frozen lambda."""
    complete, all_dates = _complete_model_frame(
        frame, (*feature_columns, "excess_return")
    )
    fit_window = model_protocol.fit_window
    penalties = jm_protocol.lambda_grid
    states = pd.DataFrame(index=all_dates, columns=penalties, dtype=float)
    fit: _FixedJMFit | None = None
    last_anchor: tuple[int, int] | None = None
    records: list[dict[str, object]] = []

    for terminal in range(fit_window - 1, len(complete)):
        window = complete.iloc[terminal - fit_window + 1 : terminal + 1]
        current_date = pd.Timestamp(window.iloc[-1]["date"])
        anchor = (current_date.year, current_date.month)
        scheduled = current_date.month in jm_protocol.refit_months
        if fit is None or (scheduled and anchor != last_anchor):
            fit = _fit_fixed_jm(
                window,
                model_protocol,
                jm_protocol,
                feature_columns=feature_columns,
            )
            last_anchor = anchor
            records.extend(_jm_fit_records(fit, window, current_date))

        scaled = fit.scaler.transform(window.loc[:, feature_columns])
        for penalty, fitted_model in fit.models.items():
            states.loc[current_date, penalty] = terminal_online_state(
                fitted_model, scaled
            )

    states.index.name = "date"
    refits = pd.DataFrame.from_records(records)
    return FixedJMResult(states=states, refits=refits)


def terminal_online_state(model: JumpModel, scaled_window: np.ndarray) -> int:
    """Return and validate the final upstream online-DP state."""
    values = np.asarray(scaled_window, dtype=float)
    if values.ndim != 2 or len(values) == 0 or not np.isfinite(values).all():
        raise ModelError("scaled JM window must be a finite non-empty matrix")
    labels = np.asarray(model.predict_online(values))
    if labels.shape != (len(values),):
        raise ModelError("upstream JM returned an invalid label shape")
    terminal = int(labels[-1])
    if terminal not in (0, 1):
        raise ModelError("upstream JM state must be 0 or 1")
    return terminal


def _fit_fixed_jm(
    window: pd.DataFrame,
    model_protocol: ModelProtocol,
    jm_protocol: JMProtocol,
    *,
    feature_columns: tuple[str, ...],
) -> _FixedJMFit:
    if len(window) != model_protocol.fit_window:
        raise ModelError("JM fit window length violates the protocol")
    features = window.loc[:, feature_columns]
    returns = window.loc[:, "excess_return"]
    scaler = StandardScaler().fit(features)
    scaled = pd.DataFrame(
        scaler.transform(features), index=features.index, columns=features.columns
    )
    models: dict[float, JumpModel] = {}
    for penalty in jm_protocol.lambda_grid:
        fitted = JumpModel(
            n_components=model_protocol.n_states,
            jump_penalty=penalty,
            random_state=jm_protocol.random_state,
            max_iter=jm_protocol.max_iter,
            tol=jm_protocol.tol,
            n_init=jm_protocol.n_init,
        ).fit(scaled, ret_ser=returns, sort_by="cumret")
        if not math.isfinite(float(fitted.val_)):
            raise ModelError(f"JM lambda {penalty:g} produced a non-finite objective")
        labels = np.asarray(fitted.labels_)
        if not np.isin(labels, [0, 1]).all():
            raise ModelError(f"JM lambda {penalty:g} produced invalid states")
        models[penalty] = fitted
    return _FixedJMFit(scaler=scaler, models=models)


def _jm_fit_records(
    fit: _FixedJMFit, window: pd.DataFrame, fit_date: pd.Timestamp
) -> list[dict[str, object]]:
    common: dict[str, object] = {
        "fit_date": fit_date,
        "training_start": pd.Timestamp(window.iloc[0]["date"]),
        "training_end": pd.Timestamp(window.iloc[-1]["date"]),
        "observations": len(window),
        "scaler_mean": fit.scaler.mean_.tolist(),
        "scaler_scale": fit.scaler.scale_.tolist(),
    }
    return [
        {**common, "lambda": penalty, "objective": float(model.val_)}
        for penalty, model in fit.models.items()
    ]


def _complete_model_frame(
    frame: pd.DataFrame, required_values: tuple[str, ...]
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    required = ("date", *required_values)
    missing = [column for column in required if column not in frame]
    if missing:
        raise ModelError(f"missing model columns: {missing}")
    prepared = frame.loc[:, required].copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="raise")
    if (
        prepared["date"].duplicated().any()
        or not prepared["date"].is_monotonic_increasing
    ):
        raise ModelError("model dates must be increasing and unique")
    observed = prepared.loc[:, required_values].dropna()
    if not np.isfinite(observed.to_numpy(dtype=float)).all():
        raise ModelError("model observations must be finite when present")
    complete = prepared.dropna(subset=list(required_values)).reset_index(drop=True)
    dates = pd.DatetimeIndex(prepared["date"], name="date")
    return complete, dates
