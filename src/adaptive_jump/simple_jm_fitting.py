"""Causal fitting adapters for the frozen simple-JM challenger suite."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from jumpmodels.jump import dp, jump_penalty_to_mx
from sklearn.preprocessing import StandardScaler

from adaptive_jump.config import JMProtocol, ModelProtocol
from adaptive_jump.models import (
    FEATURE_COLUMNS,
    FixedJMResult,
    ModelError,
    fit_fixed_jm_window,
    fixed_jm_states,
    terminal_online_state,
)
from adaptive_jump.monitor import model_runtime as runtime
from adaptive_jump.monitor.events import EventObserver
from adaptive_jump.simple_jm_l1 import L1JumpModel
from adaptive_jump.simple_jm_return import (
    ReturnAwareJumpModel,
    align_matured_targets,
    feature_loss_matrix,
)

VariantKind = Literal["robust_l1", "return_aware"]
CANONICAL_REQUIRED = (*FEATURE_COLUMNS, "excess_return")


class SimpleJMFitError(ModelError):
    """Raised when a challenger fit violates its frozen causal protocol."""


@dataclass(frozen=True)
class SmokeEvidence:
    """Small US real-data fit used before the three-market jobs."""

    variant: str
    penalty: float
    complete_rows: int
    first_state_date: str
    last_state_date: str
    prefix_rows_compared: int
    future_rows_appended: int
    prefix_invariant: bool


@dataclass
class _CustomFit:
    scaler: StandardScaler
    models: dict[float, L1JumpModel | ReturnAwareJumpModel]


@dataclass(frozen=True)
class FixedJMTraceReceipt:
    """Self-contained online-DP evidence for one fixed-JM observation."""

    fit_date: pd.Timestamp
    penalty: float
    objective: float
    scaler_mean: tuple[float, ...]
    scaler_scale: tuple[float, ...]
    centers: tuple[tuple[float, ...], ...]
    point_loss: tuple[float, float]
    terminal_value: tuple[float, float]
    online_state: int
    active_state_count: int
    collapsed_to_one_state: bool


def canonical_complete_mask(frame: pd.DataFrame) -> pd.Series:
    """Return the full-feature complete-row mask shared by every challenger."""
    missing = [
        column for column in ("date", *CANONICAL_REQUIRED) if column not in frame
    ]
    if missing:
        raise SimpleJMFitError(f"missing canonical columns: {missing}")
    dates = pd.to_datetime(frame["date"], errors="raise")
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise SimpleJMFitError("feature dates must be increasing and unique")
    observed = frame.loc[:, CANONICAL_REQUIRED]
    finite_or_missing = observed.apply(
        lambda column: column.isna() | np.isfinite(column.astype(float))
    )
    if not finite_or_missing.all().all():
        raise SimpleJMFitError("canonical observations must be finite when present")
    return observed.notna().all(axis=1)


def dd_only_states(
    frame: pd.DataFrame,
    model_protocol: ModelProtocol,
    jm_protocol: JMProtocol,
    *,
    observation_loss_scale: float = 1.0,
    observer: EventObserver | None = None,
) -> FixedJMResult:
    """Fit upstream squared-loss JM on DD10 using the canonical row calendar."""
    mask = canonical_complete_mask(frame)
    prepared = frame.loc[:, ["date", "dd_10", "excess_return"]].copy()
    prepared.loc[~mask, ["dd_10", "excess_return"]] = np.nan
    result = fixed_jm_states(
        prepared,
        model_protocol,
        jm_protocol,
        feature_columns=("dd_10",),
        observation_loss_scale=observation_loss_scale,
        include_fit_diagnostics=True,
        observer=observer,
    )
    expected_first = pd.Timestamp(
        frame.loc[mask, "date"].iloc[model_protocol.fit_window - 1]
    )
    if result.states.dropna(how="all").first_valid_index() != expected_first:
        raise SimpleJMFitError(
            "DD-only did not preserve the canonical complete-row calendar"
        )
    return result


def fixed_jm_trace_receipt(
    frame: pd.DataFrame,
    model_protocol: ModelProtocol,
    jm_protocol: JMProtocol,
    *,
    feature_columns: tuple[str, ...],
    penalty: float,
    refit_record: pd.Series | dict[str, object],
    signal_date: pd.Timestamp,
    expected_state: float,
    observation_loss_scale: float = 1.0,
) -> FixedJMTraceReceipt:
    """Replay one fixed-JM refit and its online state for an auditable trace."""
    if model_protocol.n_states != 2 or not feature_columns:
        raise SimpleJMFitError("trace receipt requires two states and features")
    penalty = float(penalty)
    if not math.isfinite(penalty) or penalty < 0:
        raise SimpleJMFitError("trace penalty must be finite and nonnegative")
    expected_state = float(expected_state)
    if expected_state not in (0.0, 1.0):
        raise SimpleJMFitError("trace expected state must be binary")
    signal_date = pd.Timestamp(signal_date)

    mask = canonical_complete_mask(frame)
    prepared = frame.loc[:, ["date", *CANONICAL_REQUIRED]].copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="raise")
    complete = prepared.loc[mask].reset_index(drop=True)
    dates = pd.DatetimeIndex(complete["date"])
    fit_date = pd.Timestamp(refit_record["fit_date"])
    training_start = pd.Timestamp(refit_record["training_start"])
    training_end = pd.Timestamp(refit_record["training_end"])
    if fit_date != training_end or fit_date > signal_date:
        raise SimpleJMFitError("trace refit dates are inconsistent")
    fit_positions = np.flatnonzero(dates == fit_date)
    signal_positions = np.flatnonzero(dates == signal_date)
    if len(fit_positions) != 1 or len(signal_positions) != 1:
        raise SimpleJMFitError("trace dates must be canonical complete rows")
    fit_terminal = int(fit_positions[0])
    signal_terminal = int(signal_positions[0])
    first_fit_row = fit_terminal - model_protocol.fit_window + 1
    first_online_row = signal_terminal - model_protocol.fit_window + 1
    if first_fit_row < 0 or first_online_row < 0:
        raise SimpleJMFitError("trace window is shorter than the fit protocol")
    fit_window = complete.iloc[first_fit_row : fit_terminal + 1]
    online_window = complete.iloc[first_online_row : signal_terminal + 1]
    if (
        pd.Timestamp(fit_window.iloc[0]["date"]) != training_start
        or len(fit_window) != int(refit_record["observations"])
        or len(fit_window) != model_protocol.fit_window
    ):
        raise SimpleJMFitError("trace training window does not match the refit record")

    one_penalty = JMProtocol(
        lambda_grid=(penalty,),
        n_init=jm_protocol.n_init,
        random_state=jm_protocol.random_state,
        max_iter=jm_protocol.max_iter,
        tol=jm_protocol.tol,
        refit_months=jm_protocol.refit_months,
    )
    fitted = fit_fixed_jm_window(
        fit_window,
        model_protocol,
        one_penalty,
        feature_columns=feature_columns,
        observation_loss_scale=observation_loss_scale,
    )
    model = fitted.models[penalty]
    objective = float(model.val_)
    if not math.isclose(
        objective,
        float(refit_record["objective"]),
        rel_tol=1e-12,
        abs_tol=1e-9,
    ):
        raise SimpleJMFitError("trace refit objective does not match sealed evidence")
    expected_mean = _decode_refit_array(refit_record["scaler_mean"])
    expected_scale = _decode_refit_array(refit_record["scaler_scale"])
    if not (
        np.allclose(fitted.scaler.mean_, expected_mean, rtol=0, atol=1e-12)
        and np.allclose(fitted.scaler.scale_, expected_scale, rtol=0, atol=1e-12)
    ):
        raise SimpleJMFitError("trace scaler does not match sealed evidence")

    scaled = fitted.transform(online_window.loc[:, feature_columns])
    loss = feature_loss_matrix(scaled, model.centers_)
    safe_loss = np.where(np.isnan(loss), np.inf, loss)
    values = dp(
        safe_loss,
        jump_penalty_to_mx(penalty, model_protocol.n_states),
        return_value_mx=True,
    )
    online_state = int(np.asarray(values)[-1].argmin())
    if online_state != int(expected_state):
        raise SimpleJMFitError("trace online state does not match the emitted state")
    active_state_count = int(np.unique(np.asarray(model.labels_, dtype=int)).size)
    centers = np.asarray(model.centers_, dtype=float)
    return FixedJMTraceReceipt(
        fit_date=fit_date,
        penalty=penalty,
        objective=objective,
        scaler_mean=tuple(float(value) for value in fitted.scaler.mean_),
        scaler_scale=tuple(float(value) for value in fitted.scaler.scale_),
        centers=tuple(tuple(float(value) for value in row) for row in centers),
        point_loss=tuple(float(value) for value in safe_loss[-1]),
        terminal_value=tuple(float(value) for value in np.asarray(values)[-1]),
        online_state=online_state,
        active_state_count=active_state_count,
        collapsed_to_one_state=active_state_count == 1,
    )


def custom_variant_states(
    frame: pd.DataFrame,
    model_protocol: ModelProtocol,
    jm_protocol: JMProtocol,
    *,
    variant: VariantKind,
    observer: EventObserver | None = None,
) -> FixedJMResult:
    """Generate causal terminal states for L1 or return-aware JM."""
    if variant not in ("robust_l1", "return_aware"):
        raise SimpleJMFitError(f"unknown custom variant: {variant}")
    mask = canonical_complete_mask(frame)
    prepared = frame.loc[:, ["date", *CANONICAL_REQUIRED]].copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="raise")
    complete = prepared.loc[mask].reset_index(drop=True)
    if len(complete) < model_protocol.fit_window:
        raise SimpleJMFitError("not enough canonical rows for the frozen fit window")

    all_dates = pd.DatetimeIndex(prepared["date"], name="date")
    states = pd.DataFrame(index=all_dates, columns=jm_protocol.lambda_grid, dtype=float)
    records: list[dict[str, object]] = []
    fitted: _CustomFit | None = None
    last_anchor: tuple[int, int] | None = None
    first_terminal = model_protocol.fit_window - 1
    total = len(complete) - first_terminal
    runtime.emit_fixed_jm_started(
        observer,
        model_protocol.fit_window,
        jm_protocol.lambda_grid,
        0,
        total,
    )
    for terminal in range(first_terminal, len(complete)):
        window = complete.iloc[terminal - first_terminal : terminal + 1]
        current_date = pd.Timestamp(window.iloc[-1]["date"])
        anchor = (current_date.year, current_date.month)
        scheduled = current_date.month in jm_protocol.refit_months
        if fitted is None or (scheduled and anchor != last_anchor):
            fitted = _fit_custom_window(
                window,
                prepared,
                model_protocol,
                jm_protocol,
                variant,
            )
            last_anchor = anchor
            records.extend(_custom_fit_records(fitted, window, current_date, variant))
            runtime.emit_fixed_jm_refit(
                observer,
                current_date.date(),
                terminal - first_terminal,
                total,
            )

        scaled = fitted.scaler.transform(window.loc[:, FEATURE_COLUMNS])
        for penalty, model in fitted.models.items():
            states.loc[current_date, penalty] = terminal_online_state(model, scaled)
        completed = terminal - first_terminal + 1
        runtime.emit_fixed_jm_terminal(
            observer,
            current_date.date(),
            completed,
            total,
            [
                (penalty, int(states.loc[current_date, penalty]))
                for penalty in jm_protocol.lambda_grid
            ],
        )

    states.index.name = "date"
    observed = states.dropna(how="all")
    expected_first = pd.Timestamp(complete.iloc[first_terminal]["date"])
    if observed.empty or observed.index[0] != expected_first:
        raise SimpleJMFitError("custom JM state calendar is incomplete")
    runtime.emit_stage_completed(observer, "fixed_jm", total)
    return FixedJMResult(states=states, refits=pd.DataFrame.from_records(records))


def run_us_prefix_smoke(
    frame: pd.DataFrame,
    model_protocol: ModelProtocol,
    jm_protocol: JMProtocol,
    *,
    variant: Literal["dd_only", "robust_l1", "return_aware"],
    observation_loss_scale: float = 1.0,
) -> SmokeEvidence:
    """Fit two real US prefixes and require identical overlapping outputs."""
    mask = canonical_complete_mask(frame)
    complete_dates = pd.DatetimeIndex(pd.to_datetime(frame.loc[mask, "date"]))
    available_states = len(complete_dates) - model_protocol.fit_window + 1
    if available_states < 4:
        raise SimpleJMFitError("US smoke source is too short")
    future_rows = min(128, available_states // 2)
    prefix_rows = min(128, available_states - future_rows)
    short_complete_rows = model_protocol.fit_window + prefix_rows - 1
    long_complete_rows = short_complete_rows + future_rows
    penalty = float(jm_protocol.lambda_grid[len(jm_protocol.lambda_grid) // 2])
    one_penalty = JMProtocol(
        lambda_grid=(penalty,),
        n_init=jm_protocol.n_init,
        random_state=jm_protocol.random_state,
        max_iter=jm_protocol.max_iter,
        tol=jm_protocol.tol,
        refit_months=jm_protocol.refit_months,
    )
    short_end = complete_dates[short_complete_rows - 1]
    long_end = complete_dates[long_complete_rows - 1]
    short = frame.loc[pd.to_datetime(frame["date"]) <= short_end].copy()
    long = frame.loc[pd.to_datetime(frame["date"]) <= long_end].copy()
    short_result = _fit_smoke_variant(
        short,
        model_protocol,
        one_penalty,
        variant,
        observation_loss_scale,
    )
    long_result = _fit_smoke_variant(
        long,
        model_protocol,
        one_penalty,
        variant,
        observation_loss_scale,
    )
    short_states = short_result.states[penalty].dropna()
    overlap = long_result.states.loc[short_states.index, penalty]
    invariant = short_states.equals(overlap)
    if not invariant:
        raise SimpleJMFitError(f"{variant} failed real-data prefix invariance")
    long_states = long_result.states[penalty].dropna()
    return SmokeEvidence(
        variant=variant,
        penalty=penalty,
        complete_rows=long_complete_rows,
        first_state_date=long_states.index[0].date().isoformat(),
        last_state_date=long_states.index[-1].date().isoformat(),
        prefix_rows_compared=len(short_states),
        future_rows_appended=future_rows,
        prefix_invariant=True,
    )


def _fit_smoke_variant(
    frame: pd.DataFrame,
    model_protocol: ModelProtocol,
    jm_protocol: JMProtocol,
    variant: str,
    observation_loss_scale: float,
) -> FixedJMResult:
    if variant == "dd_only":
        return dd_only_states(
            frame,
            model_protocol,
            jm_protocol,
            observation_loss_scale=observation_loss_scale,
        )
    return custom_variant_states(
        frame,
        model_protocol,
        jm_protocol,
        variant=variant,  # type: ignore[arg-type]
    )


def _decode_refit_array(value: object) -> np.ndarray:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise SimpleJMFitError("trace refit array is invalid JSON") from exc
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise SimpleJMFitError("trace refit array must be a finite vector")
    return array


def _fit_custom_window(
    window: pd.DataFrame,
    return_calendar: pd.DataFrame,
    model_protocol: ModelProtocol,
    jm_protocol: JMProtocol,
    variant: VariantKind,
) -> _CustomFit:
    if len(window) != model_protocol.fit_window:
        raise SimpleJMFitError("custom JM fit window violates the frozen length")
    features = window.loc[:, FEATURE_COLUMNS]
    scaler = StandardScaler().fit(features)
    scaled = pd.DataFrame(
        scaler.transform(features), index=features.index, columns=features.columns
    )
    models: dict[float, L1JumpModel | ReturnAwareJumpModel] = {}
    target = None
    if variant == "return_aware":
        target = align_matured_targets(
            return_calendar["date"],
            return_calendar["excess_return"],
            window["date"],
            pd.Timestamp(window.iloc[-1]["date"]),
            offset=2,
        )
    for penalty in jm_protocol.lambda_grid:
        if variant == "robust_l1":
            model = L1JumpModel(
                n_components=model_protocol.n_states,
                jump_penalty=penalty,
                random_state=jm_protocol.random_state,
                max_iter=jm_protocol.max_iter,
                tol=jm_protocol.tol,
                n_init=jm_protocol.n_init,
            ).fit(scaled, ret_ser=window["excess_return"], sort_by="cumret")
        else:
            assert target is not None
            model = ReturnAwareJumpModel(
                jump_penalty=penalty,
                gamma=1,
                random_state=jm_protocol.random_state,
                max_iter=jm_protocol.max_iter,
                tol=jm_protocol.tol,
                n_init=jm_protocol.n_init,
            ).fit(scaled, target.values, target.matured_mask)
        if not math.isfinite(float(model.val_)):
            raise SimpleJMFitError(
                f"{variant} lambda {penalty:g} has invalid objective"
            )
        labels = np.asarray(model.labels_)
        if not np.isin(labels, [0, 1]).all():
            raise SimpleJMFitError(f"{variant} lambda {penalty:g} has invalid state")
        models[float(penalty)] = model
    return _CustomFit(scaler=scaler, models=models)


def _custom_fit_records(
    fitted: _CustomFit,
    window: pd.DataFrame,
    fit_date: pd.Timestamp,
    variant: VariantKind,
) -> list[dict[str, object]]:
    encode = lambda values: json.dumps(  # noqa: E731
        np.asarray(values, dtype=float).tolist(), separators=(",", ":")
    )
    common: dict[str, object] = {
        "variant": variant,
        "fit_date": fit_date,
        "training_start": pd.Timestamp(window.iloc[0]["date"]),
        "training_end": fit_date,
        "observations": len(window),
        "scaler_mean": encode(fitted.scaler.mean_),
        "scaler_scale": encode(fitted.scaler.scale_),
    }
    rows = []
    for penalty, model in fitted.models.items():
        active_state_count = int(np.unique(np.asarray(model.labels_, dtype=int)).size)
        collapsed_to_one_state = active_state_count == 1
        row = {
            **common,
            "lambda": penalty,
            "objective": float(model.val_),
            "centers": encode(model.centers_),
            "active_state_count": active_state_count,
            "collapsed_to_one_state": collapsed_to_one_state,
        }
        if variant == "return_aware":
            assert isinstance(model, ReturnAwareJumpModel)
            row.update(
                {
                    "feature_objective": model.feature_value_,
                    "target_objective": model.target_value_,
                    "transition_objective": model.transition_value_,
                    "target_mean": model.target_standardizer_mean_,
                    "target_scale": model.target_standardizer_scale_,
                    "state_target_means": encode(model.target_means_),
                    "matured_targets": model.matured_target_count_,
                    "matured_state_counts": encode(model.matured_state_counts_),
                }
            )
        rows.append(row)
    return rows
