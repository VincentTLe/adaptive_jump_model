"""Causal adapters for the frozen fixed JM and HMM baselines."""

from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from jumpmodels.jump import JumpModel
from sklearn.preprocessing import StandardScaler

from adaptive_jump.config import HMMProtocol, JMProtocol, ModelProtocol

FEATURE_COLUMNS = ("dd_10", "sortino_20", "sortino_60")


class ModelError(ValueError):
    """Raised when model inputs or fitted outputs violate the protocol."""


@dataclass(frozen=True)
class FixedJMResult:
    """Daily candidate states and auditable semiannual fit records."""

    states: pd.DataFrame
    refits: pd.DataFrame


@dataclass(frozen=True)
class HMMFit:
    """Accepted best daily HMM fit reduced to its auditable terminal output."""

    terminal_state: int
    seed: int
    log_likelihood: float
    variances: tuple[float, float]
    accepted_starts: int
    failed_starts: tuple[str, ...]


@dataclass(frozen=True)
class HMMResult:
    """Daily volatility states and restart diagnostics."""

    states: pd.Series
    fits: pd.DataFrame


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


def hmm_states(
    frame: pd.DataFrame,
    model_protocol: ModelProtocol,
    hmm_protocol: HMMProtocol,
) -> HMMResult:
    """Fit the frozen HMM daily and retain each Viterbi terminal state."""
    complete, all_dates = _complete_model_frame(frame, ("equity_log",))
    fit_window = model_protocol.fit_window
    states = pd.Series(np.nan, index=all_dates, name="hmm_state")
    records: list[dict[str, object]] = []
    for terminal in range(fit_window - 1, len(complete)):
        window = complete.iloc[terminal - fit_window + 1 : terminal + 1]
        fit = best_hmm_terminal_fit(window["equity_log"], model_protocol, hmm_protocol)
        fit_date = pd.Timestamp(window.iloc[-1]["date"])
        states.loc[fit_date] = fit.terminal_state
        records.append(
            {
                "fit_date": fit_date,
                "training_start": pd.Timestamp(window.iloc[0]["date"]),
                "training_end": fit_date,
                "observations": len(window),
                "seed": fit.seed,
                "log_likelihood": fit.log_likelihood,
                "low_variance": fit.variances[0],
                "high_variance": fit.variances[1],
                "accepted_starts": fit.accepted_starts,
                "failed_starts": list(fit.failed_starts),
            }
        )
    return HMMResult(states=states, fits=pd.DataFrame.from_records(records))


def best_hmm_terminal_fit(
    log_returns: pd.Series,
    model_protocol: ModelProtocol,
    hmm_protocol: HMMProtocol,
) -> HMMFit:
    """Select the best accepted deterministic HMM restart."""
    values = np.asarray(log_returns, dtype=float).reshape(-1, 1)
    if len(values) != model_protocol.fit_window or not np.isfinite(values).all():
        raise ModelError("HMM window must contain the frozen number of finite returns")

    accepted: list[tuple[float, int, int, tuple[float, float]]] = []
    failures: list[str] = []
    for seed in hmm_protocol.seeds:
        try:
            with _quiet_hmmlearn():
                model = GaussianHMM(
                    n_components=model_protocol.n_states,
                    covariance_type="diag",
                    min_covar=hmm_protocol.min_covar,
                    n_iter=hmm_protocol.n_iter,
                    tol=hmm_protocol.tol,
                    algorithm="viterbi",
                    random_state=seed,
                ).fit(values)
            _require_strict_hmm_convergence(model, hmm_protocol)
            score = float(model.score(values))
            variances = np.asarray(model.covars_, dtype=float).reshape(2, -1).mean(1)
            if not math.isfinite(score) or not np.isfinite(variances).all():
                raise ModelError("non-finite score or variance")
            order = np.argsort(variances, kind="stable")
            if variances[order[0]] == variances[order[1]]:
                raise ModelError("conditional variances are tied")
            raw_terminal = int(np.asarray(model.predict(values))[-1])
            label_by_raw = {int(order[0]): 0, int(order[1]): 1}
            terminal = label_by_raw[raw_terminal]
            ordered_variances = (float(variances[order[0]]), float(variances[order[1]]))
            accepted.append((score, seed, terminal, ordered_variances))
        except (ArithmeticError, KeyError, ValueError, np.linalg.LinAlgError) as exc:
            failures.append(f"seed={seed}: {type(exc).__name__}: {exc}")

    if not accepted:
        detail = "; ".join(failures)
        raise ModelError(f"all HMM restarts failed: {detail}")
    score, seed, terminal, variances = max(accepted, key=lambda item: item[0])
    return HMMFit(
        terminal_state=terminal,
        seed=seed,
        log_likelihood=score,
        variances=variances,
        accepted_starts=len(accepted),
        failed_starts=tuple(failures),
    )


def _require_strict_hmm_convergence(model: GaussianHMM, protocol: HMMProtocol) -> None:
    monitor = model.monitor_
    history = tuple(monitor.history)
    delta = history[-1] - history[-2] if len(history) >= 2 else math.nan
    precision = math.sqrt(np.finfo(float).eps)
    accepted = (
        monitor.converged and len(history) >= 2 and -precision <= delta < protocol.tol
    )
    if not accepted:
        raise ModelError(
            f"strict convergence failed (iter={monitor.iter}, delta={delta})"
        )


@contextmanager
def _quiet_hmmlearn():
    logger = logging.getLogger("hmmlearn.base")
    previous = logger.level
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(previous)


def smoothed_hmm_states(
    states: pd.Series,
    smoothing_grid: tuple[int, ...],
    *,
    threshold: float = 0.5,
    min_periods: int = 1,
) -> pd.DataFrame:
    """Apply the preregistered causal majority filter to HMM states."""
    values = pd.Series(states, dtype=float)
    if not values.dropna().isin([0.0, 1.0]).all():
        raise ModelError("HMM states must be 0, 1, or missing")
    if not smoothing_grid or any(k < 0 for k in smoothing_grid):
        raise ModelError("HMM smoothing windows must be non-negative")
    candidates = pd.DataFrame(index=values.index)
    for window in smoothing_grid:
        if window == 0:
            candidates[window] = values
            continue
        mean = values.rolling(window=window, min_periods=min_periods).mean()
        candidates[window] = (mean > threshold).astype(float).where(mean.notna())
    candidates.columns.name = "k"
    return candidates


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
