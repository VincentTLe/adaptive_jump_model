"""Penalty calibration utilities for adaptive jump models."""

import numpy as np
import pandas as pd


def lambda_from_expected_duration(expected_duration: float) -> float:
    """Convert expected regime duration to a nonnegative switch penalty.

    The calibration uses the log-odds relation
    ``lambda = log((1 - p_switch) / p_switch)`` with
    ``p_switch = 1 / expected_duration``. In this project ``lambda`` is a
    persistence penalty, so durations below two bars are rejected because they
    imply a negative switch cost.
    """
    duration = _finite_scalar(expected_duration, "expected_duration")
    if duration < 2.0:
        raise ValueError("expected_duration must be at least 2.0 for a nonnegative switch penalty")
    return float(np.log(duration - 1.0))


def expected_duration_from_lambda(lambda_value: float) -> float:
    """Convert a nonnegative switch penalty back to expected duration."""
    penalty = _finite_scalar(lambda_value, "lambda_value")
    if penalty < 0.0:
        raise ValueError("lambda_value must be nonnegative")
    with np.errstate(over="ignore"):
        duration = float(1.0 + np.exp(penalty))
    if not np.isfinite(duration):
        raise ValueError("expected duration must be finite")
    return duration


def make_adaptive_lambda(
    df: pd.DataFrame,
    base_lambda: float,
    noise_scale: float = 0.5,
    shock_scale: float = 0.5,
    min_lambda: float = 0.0,
    max_lambda: float | None = None,
    noise_column: str = "noise_score_raw",
    shock_column: str = "shock_score_raw",
) -> pd.Series:
    """Build a time-varying switch penalty from feature diagnostics.

    ``noise_score_raw`` raises the penalty because noisy quotes can create false
    switches. ``shock_score_raw`` lowers the penalty because abrupt price moves
    are candidates for regime changes. The scores are added directly on the
    lambda scale, preserving the log-odds duration calibration.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas.DataFrame")
    _require_columns(df, [noise_column, shock_column])
    base = _nonnegative_scalar(base_lambda, "base_lambda")
    noise_scale = _nonnegative_scalar(noise_scale, "noise_scale")
    shock_scale = _nonnegative_scalar(shock_scale, "shock_scale")
    min_lambda = _nonnegative_scalar(min_lambda, "min_lambda")
    if max_lambda is not None:
        max_lambda = _nonnegative_scalar(max_lambda, "max_lambda")
        if max_lambda < min_lambda:
            raise ValueError("max_lambda must be greater than or equal to min_lambda")

    scores = df[[noise_column, shock_column]].astype(float)
    valid = np.isfinite(scores.to_numpy()).all(axis=1)
    if not valid.all():
        first = scores.index[~valid][0]
        raise ValueError(f"adaptive lambda scores must be finite; first invalid row at {first}")

    values = base + noise_scale * scores[noise_column] - shock_scale * scores[shock_column]
    result = pd.Series(values, index=df.index, name="lambda")
    result = result.clip(lower=min_lambda, upper=max_lambda)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("adaptive lambda values must be finite after clipping")
    return result


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"df is missing required columns: {missing}")


def _finite_scalar(value: float, name: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _nonnegative_scalar(value: float, name: str) -> float:
    number = _finite_scalar(value, name)
    if number < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return number
