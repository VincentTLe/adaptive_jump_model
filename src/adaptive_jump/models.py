"""Causal adapters for the frozen fixed JM and HMM baselines."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing import get_context

import numpy as np
import pandas as pd
from hmmlearn.base import ConvergenceMonitor
from hmmlearn.hmm import GaussianHMM
from jumpmodels.jump import JumpModel
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from adaptive_jump.config import HMMProtocol, JMProtocol, ModelProtocol
from adaptive_jump.monitor import model_runtime as runtime
from adaptive_jump.monitor.events import EventObserver

FEATURE_COLUMNS = ("dd_10", "sortino_20", "sortino_60")


class ModelError(ValueError):
    """Raised when model inputs or fitted outputs violate the protocol."""


class _SymmetricConvergenceMonitor(ConvergenceMonitor):
    @property
    def converged(self) -> bool:
        if len(self.history) < 2:
            return False
        return abs(self.history[-1] - self.history[-2]) < self.tol


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
class FixedJMFit:
    """Fitted fixed-JM candidates sharing one past-only feature scaler."""

    scaler: StandardScaler
    models: dict[float, JumpModel]
    observation_loss_scale: float = 1.0

    def transform(self, features: pd.DataFrame) -> np.ndarray:
        """Apply the fitted standardization and declared loss scaling."""
        scaled = self.scaler.transform(features)
        if self.observation_loss_scale != 1.0:
            scaled = scaled * math.sqrt(self.observation_loss_scale)
        return scaled


def fixed_jm_states(
    frame: pd.DataFrame,
    model_protocol: ModelProtocol,
    jm_protocol: JMProtocol,
    *,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
    observation_loss_scale: float = 1.0,
    include_fit_diagnostics: bool = False,
    initial: FixedJMResult | None = None,
    checkpoint_every: int = 50,
    progress: Callable[[FixedJMResult], None] | None = None,
    observer: EventObserver | None = None,
) -> FixedJMResult:
    """Generate causal terminal online states for every frozen lambda."""
    if checkpoint_every < 1:
        raise ModelError("JM checkpoint interval must be positive")
    observation_loss_scale = _validated_observation_loss_scale(observation_loss_scale)
    if initial is not None and observation_loss_scale != 1.0:
        raise ModelError("scaled JM does not support checkpoint resume")
    complete, all_dates = _complete_model_frame(
        frame, (*feature_columns, "excess_return")
    )
    fit_window = model_protocol.fit_window
    penalties = jm_protocol.lambda_grid
    states = pd.DataFrame(index=all_dates, columns=penalties, dtype=float)
    fit: FixedJMFit | None = None
    last_anchor: tuple[int, int] | None = None
    records: list[dict[str, object]] = []
    first_terminal = fit_window - 1
    last_refit_terminal: int | None = None
    if initial is not None:
        try:
            resume = runtime.prepare_fixed_jm_resume(
                initial.states,
                initial.refits,
                complete,
                all_dates,
                fit_window,
                penalties,
                jm_protocol.refit_months,
            )
        except runtime.CheckpointError as exc:
            raise ModelError(str(exc)) from exc
        states, records = resume.states, resume.records
        first_terminal = resume.first_terminal
        last_refit_terminal = resume.last_refit_terminal
    total = max(0, len(complete) - fit_window + 1)
    completed = max(0, first_terminal - fit_window + 1)
    runtime.emit_fixed_jm_started(observer, fit_window, penalties, completed, total)
    if last_refit_terminal is not None and first_terminal < len(complete):
        refit_window = complete.iloc[
            last_refit_terminal - fit_window + 1 : last_refit_terminal + 1
        ]
        fit = fit_fixed_jm_window(
            refit_window,
            model_protocol,
            jm_protocol,
            feature_columns=feature_columns,
            observation_loss_scale=observation_loss_scale,
        )
        refit_date = pd.Timestamp(refit_window.iloc[-1]["date"])
        last_anchor = (refit_date.year, refit_date.month)
    for terminal in range(first_terminal, len(complete)):
        window = complete.iloc[terminal - fit_window + 1 : terminal + 1]
        current_date = pd.Timestamp(window.iloc[-1]["date"])
        anchor = (current_date.year, current_date.month)
        scheduled = current_date.month in jm_protocol.refit_months
        if fit is None or (scheduled and anchor != last_anchor):
            fit = fit_fixed_jm_window(
                window,
                model_protocol,
                jm_protocol,
                feature_columns=feature_columns,
                observation_loss_scale=observation_loss_scale,
            )
            last_anchor = anchor
            records.extend(
                _jm_fit_records(
                    fit,
                    window,
                    current_date,
                    include_fit_diagnostics=include_fit_diagnostics,
                )
            )
            runtime.emit_fixed_jm_refit(
                observer,
                current_date.date(),
                terminal - fit_window + 1,
                total,
            )

        scaled = fit.transform(window.loc[:, feature_columns])
        for penalty, fitted_model in fit.models.items():
            states.loc[current_date, penalty] = terminal_online_state(
                fitted_model, scaled
            )
        completed = terminal - fit_window + 2
        runtime.emit_fixed_jm_terminal(
            observer,
            current_date.date(),
            completed,
            total,
            [
                (penalty, int(states.loc[current_date, penalty]))
                for penalty in penalties
            ],
        )
        if progress is not None and (
            completed % checkpoint_every == 0 or completed == total
        ):
            progress(FixedJMResult(states.copy(), pd.DataFrame.from_records(records)))

    states.index.name = "date"
    refits = pd.DataFrame.from_records(records)
    runtime.emit_stage_completed(observer, "fixed_jm", total)
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
    *,
    initial: HMMResult | None = None,
    n_jobs: int = 1,
    checkpoint_every: int = 50,
    progress: Callable[[HMMResult], None] | None = None,
    observer: EventObserver | None = None,
) -> HMMResult:
    """Fit the frozen HMM daily and retain each Viterbi terminal state."""
    if n_jobs < 1 or checkpoint_every < 1:
        raise ModelError("HMM workers and checkpoint interval must be positive")
    complete, all_dates = _complete_model_frame(frame, ("equity_log",))
    fit_window = model_protocol.fit_window
    states = pd.Series(np.nan, index=all_dates, name="hmm_state")
    records: list[dict[str, object]] = []
    first_terminal = fit_window - 1
    if initial is not None:
        _validate_hmm_initial(initial, complete, all_dates, first_terminal)
        states = initial.states.copy()
        records = initial.fits.to_dict("records")
        first_terminal += len(records)
    total = max(0, len(complete) - fit_window + 1)
    runtime.emit_hmm_started(
        observer,
        fit_window,
        len(hmm_protocol.seeds),
        n_jobs,
        len(records),
        total,
    )

    executor = (
        ProcessPoolExecutor(max_workers=n_jobs, mp_context=get_context("forkserver"))
        if n_jobs > 1
        else None
    )
    try:
        for batch_start in range(first_terminal, len(complete), checkpoint_every):
            terminals = range(
                batch_start, min(batch_start + checkpoint_every, len(complete))
            )
            tasks = [
                (
                    complete.iloc[terminal - fit_window + 1 : terminal + 1][
                        "equity_log"
                    ].to_numpy(),
                    model_protocol,
                    hmm_protocol,
                )
                for terminal in terminals
            ]
            fits = (
                list(executor.map(_fit_hmm_task, tasks))
                if executor is not None
                else [_fit_hmm_task(task) for task in tasks]
            )
            for terminal, fit in zip(terminals, fits, strict=True):
                window = complete.iloc[terminal - fit_window + 1 : terminal + 1]
                fit_date = pd.Timestamp(window.iloc[-1]["date"])
                states.loc[fit_date] = fit.terminal_state
                records.append(_hmm_fit_record(window, fit, fit_date))
                runtime.emit_hmm_terminal(
                    observer,
                    fit_date.date(),
                    len(records),
                    total,
                    fit.terminal_state,
                    fit.seed,
                    fit.log_likelihood,
                    fit.variances,
                    fit.accepted_starts,
                    fit.failed_starts,
                )
            if progress is not None:
                progress(HMMResult(states.copy(), pd.DataFrame.from_records(records)))
    finally:
        if executor is not None:
            executor.shutdown(cancel_futures=True)
    result = HMMResult(states=states, fits=pd.DataFrame.from_records(records))
    runtime.emit_stage_completed(observer, "hmm", total)
    return result


def _fit_hmm_task(
    task: tuple[np.ndarray, ModelProtocol, HMMProtocol],
) -> HMMFit:
    values, model_protocol, hmm_protocol = task
    with threadpool_limits(limits=1):
        return best_hmm_terminal_fit(pd.Series(values), model_protocol, hmm_protocol)


def _hmm_fit_record(
    window: pd.DataFrame, fit: HMMFit, fit_date: pd.Timestamp
) -> dict[str, object]:
    return {
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


def _validate_hmm_initial(
    initial: HMMResult,
    complete: pd.DataFrame,
    all_dates: pd.DatetimeIndex,
    first_terminal: int,
) -> None:
    if not initial.states.index.equals(all_dates):
        raise ModelError("HMM checkpoint dates do not match inputs")
    if initial.fits.empty:
        return
    fit_dates = pd.DatetimeIndex(pd.to_datetime(initial.fits["fit_date"]))
    expected = pd.DatetimeIndex(
        complete.iloc[first_terminal : first_terminal + len(fit_dates)]["date"]
    )
    if not fit_dates.equals(expected) or initial.states.loc[fit_dates].isna().any():
        raise ModelError("HMM checkpoint is not a contiguous causal prefix")


def best_hmm_terminal_fit(
    log_returns: pd.Series,
    model_protocol: ModelProtocol,
    hmm_protocol: HMMProtocol,
) -> HMMFit:
    """Select the best accepted deterministic HMM restart."""
    values = np.asarray(log_returns, dtype=float).reshape(-1, 1)
    if len(values) != model_protocol.fit_window or not np.isfinite(values).all():
        raise ModelError("HMM window must contain the frozen number of finite returns")
    if (
        not isinstance(hmm_protocol.kmeans_n_init, int)
        or hmm_protocol.kmeans_n_init < 1
    ):
        raise ModelError("HMM KMeans n_init must be a positive integer")
    if not math.isfinite(hmm_protocol.covars_prior) or hmm_protocol.covars_prior < 0:
        raise ModelError("HMM covariance prior must be finite and non-negative")

    accepted: list[tuple[float, int, int, tuple[float, float]]] = []
    failures: list[str] = []
    for seed in hmm_protocol.seeds:
        try:
            with _quiet_hmmlearn():
                means = (
                    KMeans(
                        n_clusters=model_protocol.n_states,
                        init="k-means++",
                        n_init=hmm_protocol.kmeans_n_init,
                        random_state=seed,
                    )
                    .fit(values)
                    .cluster_centers_
                )
                model = GaussianHMM(
                    n_components=model_protocol.n_states,
                    covariance_type="diag",
                    min_covar=hmm_protocol.min_covar,
                    startprob_prior=1.0,
                    transmat_prior=1.0,
                    means_prior=0.0,
                    means_weight=0.0,
                    covars_prior=hmm_protocol.covars_prior,
                    covars_weight=1.0,
                    n_iter=hmm_protocol.n_iter,
                    tol=hmm_protocol.tol,
                    algorithm="viterbi",
                    random_state=seed,
                    verbose=False,
                    params="stmc",
                    init_params="stc",
                    implementation="log",
                )
                model.means_ = means
                model.monitor_ = _SymmetricConvergenceMonitor(
                    hmm_protocol.tol, hmm_protocol.n_iter, False
                )
                model.fit(values)
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
    accepted = (
        monitor.converged
        and len(history) >= 2
        and math.isfinite(delta)
        and abs(delta) < protocol.tol
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
    require_full_window: bool = False,
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
        required = window if require_full_window else min_periods
        mean = values.rolling(window=window, min_periods=required).mean()
        candidates[window] = (mean > threshold).astype(float).where(mean.notna())
    candidates.columns.name = "k"
    return candidates


def fit_fixed_jm_window(
    window: pd.DataFrame,
    model_protocol: ModelProtocol,
    jm_protocol: JMProtocol,
    *,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
    observation_loss_scale: float = 1.0,
) -> FixedJMFit:
    """Fit every fixed-JM penalty on one past-only training window."""
    if len(window) != model_protocol.fit_window:
        raise ModelError("JM fit window length violates the protocol")
    observation_loss_scale = _validated_observation_loss_scale(observation_loss_scale)
    features = window.loc[:, feature_columns]
    returns = window.loc[:, "excess_return"]
    scaler = StandardScaler().fit(features)
    scaled_values = scaler.transform(features)
    if observation_loss_scale != 1.0:
        scaled_values = scaled_values * math.sqrt(observation_loss_scale)
    scaled = pd.DataFrame(scaled_values, index=features.index, columns=features.columns)
    models: dict[float, JumpModel] = {}
    for penalty in jm_protocol.lambda_grid:
        fitted = JumpModel(
            n_components=model_protocol.n_states,
            jump_penalty=penalty,
            random_state=jm_protocol.random_state,
            max_iter=jm_protocol.max_iter,
            tol=jm_protocol.tol * observation_loss_scale,
            n_init=jm_protocol.n_init,
        ).fit(scaled, ret_ser=returns, sort_by="cumret")
        if not math.isfinite(float(fitted.val_)):
            raise ModelError(f"JM lambda {penalty:g} produced a non-finite objective")
        labels = np.asarray(fitted.labels_)
        if not np.isin(labels, [0, 1]).all():
            raise ModelError(f"JM lambda {penalty:g} produced invalid states")
        models[penalty] = fitted
    return FixedJMFit(
        scaler=scaler,
        models=models,
        observation_loss_scale=observation_loss_scale,
    )


def _validated_observation_loss_scale(value: float) -> float:
    if isinstance(value, bool):
        raise ModelError("JM observation loss scale must be finite and positive")
    try:
        scale = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelError("JM observation loss scale must be a real scalar") from exc
    if not math.isfinite(scale) or scale <= 0:
        raise ModelError("JM observation loss scale must be finite and positive")
    return scale


def _jm_fit_records(
    fit: FixedJMFit,
    window: pd.DataFrame,
    fit_date: pd.Timestamp,
    *,
    include_fit_diagnostics: bool = False,
) -> list[dict[str, object]]:
    common: dict[str, object] = {
        "fit_date": fit_date,
        "training_start": pd.Timestamp(window.iloc[0]["date"]),
        "training_end": pd.Timestamp(window.iloc[-1]["date"]),
        "observations": len(window),
        "scaler_mean": fit.scaler.mean_.tolist(),
        "scaler_scale": fit.scaler.scale_.tolist(),
    }
    records = []
    for penalty, model in fit.models.items():
        record = {
            **common,
            "lambda": penalty,
            "objective": float(model.val_),
        }
        if include_fit_diagnostics:
            active_state_count = int(
                np.unique(np.asarray(model.labels_, dtype=int)).size
            )
            record.update(
                {
                    "centers": np.asarray(model.centers_, dtype=float).tolist(),
                    "active_state_count": active_state_count,
                    "collapsed_to_one_state": active_state_count == 1,
                }
            )
        records.append(record)
    return records


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
