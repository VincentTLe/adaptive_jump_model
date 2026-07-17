"""Fitted-center reconstruction and adaptive state generation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from adaptive_jump.confidence_spec import ConfidenceSpec, ConfidenceStudyError
from adaptive_jump.config import ResearchConfig
from adaptive_jump.models import (
    FEATURE_COLUMNS,
    _complete_model_frame,
    _fit_fixed_jm,
)
from adaptive_jump.tv_jump import (
    TVJumpModel,
    evidence_penalty_seq,
    loss_matrix,
    robust_loss_scale,
)


@dataclass
class StateEvidence:
    states: dict[float, pd.DataFrame]
    loss0: pd.DataFrame
    loss1: pd.DataFrame
    q_train: pd.DataFrame
    c01: dict[float, pd.DataFrame]
    c10: dict[float, pd.DataFrame]
    refits: pd.DataFrame


def _load_parent_frame(parent: Path, market: str, cutoff: date) -> pd.DataFrame:
    frame = pd.read_csv(parent / market / "features.csv")
    required = {
        "date",
        "equity_simple",
        "cash_return",
        "excess_return",
        *FEATURE_COLUMNS,
    }
    if not required.issubset(frame):
        raise ConfidenceStudyError(f"{market}: parent features are incomplete")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    dates = pd.DatetimeIndex(frame["date"])
    if (
        dates.has_duplicates
        or not dates.is_monotonic_increasing
        or dates.max().date() > cutoff
        or dates.max().date() > date(2023, 12, 31)
    ):
        raise ConfidenceStudyError(f"{market}: parent feature dates are invalid")
    return frame


def _parent_states(
    parent: Path, market: str, lambdas: tuple[float, ...]
) -> pd.DataFrame:
    stored = pd.read_csv(parent / market / "jm-states.csv")
    stored["date"] = pd.to_datetime(stored["date"], errors="raise")
    stored = stored.set_index("date")
    stored.columns = tuple(float(column) for column in stored.columns)
    if tuple(stored.columns) != lambdas:
        raise ConfidenceStudyError(f"{market}: parent lambda grid changed")
    return stored


def _validate_refit(
    market: str,
    fit_date: pd.Timestamp,
    window: pd.DataFrame,
    fit: Any,
    expected: pd.DataFrame,
) -> None:
    rows = expected.loc[expected["fit_date"] == fit_date]
    if len(rows) != len(fit.models):
        raise ConfidenceStudyError(f"{market}/{fit_date.date()}: missing v7 refit rows")
    for penalty, model in fit.models.items():
        row = rows.loc[np.isclose(rows["lambda"], penalty, rtol=0, atol=1e-15)]
        if len(row) != 1:
            raise ConfidenceStudyError(
                f"{market}/{fit_date.date()}/{penalty:g}: v7 refit row changed"
            )
        item = row.iloc[0]
        if (
            pd.Timestamp(item["training_start"]) != pd.Timestamp(window.iloc[0]["date"])
            or pd.Timestamp(item["training_end"])
            != pd.Timestamp(window.iloc[-1]["date"])
            or int(item["observations"]) != len(window)
            or not math.isclose(
                float(item["objective"]),
                float(model.val_),
                rel_tol=0,
                abs_tol=1e-8,
            )
            or not np.allclose(
                json.loads(item["scaler_mean"]),
                fit.scaler.mean_,
                rtol=0,
                atol=1e-14,
            )
            or not np.allclose(
                json.loads(item["scaler_scale"]),
                fit.scaler.scale_,
                rtol=0,
                atol=1e-14,
            )
        ):
            raise ConfidenceStudyError(
                f"{market}/{fit_date.date()}/{penalty:g}: reconstructed fit differs"
            )


def generate_adaptive_states(
    frame: pd.DataFrame,
    parent_refits: pd.DataFrame,
    config: ResearchConfig,
    spec: ConfidenceSpec,
    *,
    market: str,
    terminal_limit: int | None = None,
) -> StateEvidence:
    """Reconstruct v7 fits and emit terminal states for every beta/lambda."""
    complete, all_dates = _complete_model_frame(
        frame, (*FEATURE_COLUMNS, "excess_return")
    )
    fit_window = config.model_protocol.fit_window
    if len(complete) < fit_window:
        raise ConfidenceStudyError(f"{market}: insufficient fixed-JM observations")

    parent_refits = parent_refits.copy()
    for column in ("fit_date", "training_start", "training_end"):
        parent_refits[column] = pd.to_datetime(parent_refits[column], errors="raise")
    parent_refits["lambda"] = pd.to_numeric(parent_refits["lambda"], errors="raise")

    states = {
        beta: pd.DataFrame(index=all_dates, columns=spec.lambdas, dtype=float)
        for beta in spec.betas
    }
    loss0 = pd.DataFrame(index=all_dates, columns=spec.lambdas, dtype=float)
    loss1 = pd.DataFrame(index=all_dates, columns=spec.lambdas, dtype=float)
    q_frame = pd.DataFrame(index=all_dates, columns=spec.lambdas, dtype=float)
    c01 = {
        beta: pd.DataFrame(index=all_dates, columns=spec.lambdas, dtype=float)
        for beta in spec.betas
    }
    c10 = {
        beta: pd.DataFrame(index=all_dates, columns=spec.lambdas, dtype=float)
        for beta in spec.betas
    }

    fit = None
    tv_models: dict[float, TVJumpModel] = {}
    scales: dict[float, float] = {}
    last_anchor: tuple[int, int] | None = None
    refit_rows: list[dict[str, Any]] = []
    first_terminal = fit_window - 1
    final_terminal = len(complete)
    if terminal_limit is not None:
        if terminal_limit < 1:
            raise ConfidenceStudyError("terminal_limit must be positive")
        final_terminal = min(final_terminal, first_terminal + terminal_limit)
    total_terminals = final_terminal - first_terminal

    for terminal in range(first_terminal, final_terminal):
        window = complete.iloc[terminal - fit_window + 1 : terminal + 1]
        current_date = pd.Timestamp(window.iloc[-1]["date"])
        anchor = (current_date.year, current_date.month)
        scheduled = current_date.month in config.jm_protocol.refit_months
        if fit is None or (scheduled and anchor != last_anchor):
            fit = _fit_fixed_jm(
                window,
                config.model_protocol,
                config.jm_protocol,
                feature_columns=FEATURE_COLUMNS,
            )
            _validate_refit(market, current_date, window, fit, parent_refits)
            last_anchor = anchor
            training_scaled = fit.scaler.transform(window.loc[:, FEATURE_COLUMNS])
            tv_models = {}
            scales = {}
            for penalty, fixed_model in fit.models.items():
                train_loss = loss_matrix(training_scaled, fixed_model.centers_)
                q_train = robust_loss_scale(train_loss)
                tv = TVJumpModel(
                    n_components=config.model_protocol.n_states,
                    random_state=config.jm_protocol.random_state,
                    max_iter=config.jm_protocol.max_iter,
                    tol=config.jm_protocol.tol,
                    n_init=config.jm_protocol.n_init,
                )
                tv.centers_ = np.asarray(fixed_model.centers_, dtype=float).copy()
                tv.feat_weights = None
                tv_models[penalty] = tv
                scales[penalty] = q_train
                refit_rows.append(
                    {
                        "market": market,
                        "fit_date": current_date,
                        "training_start": pd.Timestamp(window.iloc[0]["date"]),
                        "training_end": pd.Timestamp(window.iloc[-1]["date"]),
                        "lambda0": penalty,
                        "q_train": q_train,
                        "fixed_objective": float(fixed_model.val_),
                        "scaler_mean": json.dumps(fit.scaler.mean_.tolist()),
                        "scaler_scale": json.dumps(fit.scaler.scale_.tolist()),
                        "centers": json.dumps(tv.centers_.tolist()),
                    }
                )

        scaled = fit.scaler.transform(window.loc[:, FEATURE_COLUMNS])
        for penalty in spec.lambdas:
            losses = loss_matrix(scaled, tv_models[penalty].centers_)
            q_train = scales[penalty]
            loss0.loc[current_date, penalty] = losses[-1, 0]
            loss1.loc[current_date, penalty] = losses[-1, 1]
            q_frame.loc[current_date, penalty] = q_train
            for beta in spec.betas:
                penalty_seq = evidence_penalty_seq(
                    losses,
                    lambda0=penalty,
                    beta=beta,
                    q_train=q_train,
                )
                labels = np.asarray(
                    tv_models[penalty].predict_online_tv(
                        scaled, penalty_seq=penalty_seq
                    )
                )
                states[beta].loc[current_date, penalty] = int(labels[-1])
                c01[beta].loc[current_date, penalty] = penalty_seq[-1, 0, 1]
                c10[beta].loc[current_date, penalty] = penalty_seq[-1, 1, 0]

        completed = terminal - first_terminal + 1
        if completed % 100 == 0 or terminal + 1 == final_terminal:
            print(
                f"{market}: adaptive states {completed}/{total_terminals}",
                flush=True,
            )

    for beta, values in states.items():
        values.index.name = "date"
        if not np.array_equal(
            values[0.0].to_numpy(),
            states[0.0][0.0].to_numpy(),
            equal_nan=True,
        ):
            raise ConfidenceStudyError(f"{market}: lambda0=0 differs at beta={beta:g}")
    return StateEvidence(
        states=states,
        loss0=loss0,
        loss1=loss1,
        q_train=q_frame,
        c01=c01,
        c10=c10,
        refits=pd.DataFrame.from_records(refit_rows),
    )


def _assert_beta_zero_states(
    generated: pd.DataFrame,
    stored: pd.DataFrame,
    *,
    market: str,
) -> None:
    expected = stored.reindex(generated.index)
    mask = generated.notna().any(axis=1)
    if not np.array_equal(
        generated.loc[mask].to_numpy(),
        expected.loc[mask].to_numpy(),
        equal_nan=True,
    ):
        mismatch = generated.loc[mask].ne(expected.loc[mask]).stack()
        first = mismatch[mismatch].index[0]
        raise ConfidenceStudyError(f"{market}: beta0 state mismatch at {first}")
