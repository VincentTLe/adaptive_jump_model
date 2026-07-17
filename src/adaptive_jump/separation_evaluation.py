"""Leave-one-market-out scoring and descriptive mechanism summaries."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd

from adaptive_jump.separation_study import (
    MARKETS,
    SeparationSpec,
    SeparationStudyError,
    classify_decision,
    fit_logistic,
    prediction_scores,
)


def _equal_market_weights(markets: pd.Series) -> np.ndarray:
    unique = tuple(sorted(markets.unique()))
    if len(unique) != 2:
        raise SeparationStudyError("a training fold must contain two markets")
    counts = markets.value_counts()
    return markets.map({market: 0.5 / counts[market] for market in unique}).to_numpy()


def _standardize_continuous(
    train: pd.DataFrame,
    test: pd.DataFrame,
    weights: np.ndarray,
    columns: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_values = train.loc[:, columns].to_numpy(dtype=float)
    test_values = test.loc[:, columns].to_numpy(dtype=float)
    mean = np.sum(weights[:, None] * train_values, axis=0)
    scale = np.sqrt(np.sum(weights[:, None] * (train_values - mean) ** 2, axis=0))
    if (
        not np.isfinite(train_values).all()
        or not np.isfinite(test_values).all()
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or (scale <= 0).any()
    ):
        raise SeparationStudyError("continuous fold scale is invalid")
    return (train_values - mean) / scale, (test_values - mean) / scale, mean, scale


def evaluate_leave_one_market_out(
    events: pd.DataFrame, spec: SeparationSpec
) -> pd.DataFrame:
    """Evaluate the baseline and reliability challenger on held-out markets."""
    required = {
        "market",
        "beta_label",
        "lambda0",
        "source_state",
        "log_discount",
        "whipsaw_20",
        "reliability_valid",
        "reliability_train",
    }
    if not required.issubset(events):
        raise SeparationStudyError("event table is incomplete")
    prepared = events.copy()
    prepared["log_lambda0"] = np.log(pd.to_numeric(prepared["lambda0"]))
    prepared["beta_log4_indicator"] = (
        prepared["beta_label"].astype(str) == "log4"
    ).astype(float)
    prepared["source_state_indicator"] = pd.to_numeric(prepared["source_state"])
    prepared["whipsaw_20"] = prepared["whipsaw_20"].astype(float)
    numeric_columns = (
        "log_discount",
        "log_lambda0",
        "reliability_train",
        "beta_log4_indicator",
        "source_state_indicator",
        "whipsaw_20",
    )
    finite = np.isfinite(prepared.loc[:, numeric_columns].to_numpy(dtype=float)).all(
        axis=1
    )
    scored = prepared.loc[prepared["reliability_valid"].astype(bool) & finite].copy()
    records: list[dict[str, Any]] = []

    for held_out in spec.markets:
        train = scored.loc[scored["market"] != held_out].copy()
        test = scored.loc[scored["market"] == held_out].copy()
        total_held_out = int((prepared["market"] == held_out).sum())
        base: dict[str, Any] = {
            "held_out_market": held_out,
            "fold_valid": False,
            "failure_reason": "",
            "training_events": len(train),
            "admitted_events": len(test),
            "total_admitted_events": total_held_out,
            "training_whipsaws": int(train["whipsaw_20"].sum()),
            "held_out_whipsaws": int(test["whipsaw_20"].sum()),
            "reliability_coefficient": math.nan,
            "baseline_brier": math.nan,
            "challenger_brier": math.nan,
            "delta_brier": math.nan,
            "baseline_log_loss": math.nan,
            "challenger_log_loss": math.nan,
            "baseline_gradient_inf": math.nan,
            "challenger_gradient_inf": math.nan,
            "baseline_coefficients": "",
            "challenger_coefficients": "",
            "continuous_mean": "",
            "continuous_scale": "",
        }
        try:
            if len(test) == 0:
                raise SeparationStudyError("held-out market has zero valid events")
            target_train = train["whipsaw_20"].to_numpy(dtype=float)
            if len(np.unique(target_train)) != 2:
                raise SeparationStudyError("training fold has one outcome class")
            weights = _equal_market_weights(train["market"])
            continuous = ("log_discount", "log_lambda0", "reliability_train")
            train_cont, test_cont, mean, scale = _standardize_continuous(
                train, test, weights, continuous
            )
            train_indicators = train.loc[
                :, ("beta_log4_indicator", "source_state_indicator")
            ].to_numpy(dtype=float)
            test_indicators = test.loc[
                :, ("beta_log4_indicator", "source_state_indicator")
            ].to_numpy(dtype=float)
            baseline_train = np.column_stack([train_cont[:, :2], train_indicators])
            baseline_test = np.column_stack([test_cont[:, :2], test_indicators])
            challenger_train = np.column_stack([baseline_train, train_cont[:, 2]])
            challenger_test = np.column_stack([baseline_test, test_cont[:, 2]])
            baseline_fit = fit_logistic(
                baseline_train,
                target_train,
                weights,
                max_iterations=spec.optimizer_max_iterations,
                gradient_tolerance=spec.optimizer_gradient_tolerance,
            )
            challenger_fit = fit_logistic(
                challenger_train,
                target_train,
                weights,
                max_iterations=spec.optimizer_max_iterations,
                gradient_tolerance=spec.optimizer_gradient_tolerance,
            )
            target_test = test["whipsaw_20"].to_numpy(dtype=float)
            baseline_probability = baseline_fit.predict_proba(baseline_test)
            challenger_probability = challenger_fit.predict_proba(challenger_test)
            baseline_brier, baseline_log_loss = prediction_scores(
                target_test, baseline_probability
            )
            challenger_brier, challenger_log_loss = prediction_scores(
                target_test, challenger_probability
            )
            base.update(
                {
                    "fold_valid": True,
                    "reliability_coefficient": float(challenger_fit.coef[-1]),
                    "baseline_brier": baseline_brier,
                    "challenger_brier": challenger_brier,
                    "delta_brier": challenger_brier - baseline_brier,
                    "baseline_log_loss": baseline_log_loss,
                    "challenger_log_loss": challenger_log_loss,
                    "baseline_gradient_inf": baseline_fit.gradient_inf,
                    "challenger_gradient_inf": challenger_fit.gradient_inf,
                    "baseline_coefficients": json.dumps(baseline_fit.coef.tolist()),
                    "challenger_coefficients": json.dumps(challenger_fit.coef.tolist()),
                    "continuous_mean": json.dumps(mean.tolist()),
                    "continuous_scale": json.dumps(scale.tolist()),
                }
            )
        except SeparationStudyError as exc:
            base["failure_reason"] = str(exc)
        records.append(base)
    return pd.DataFrame.from_records(records)


def summarize_mechanism(
    separation: pd.DataFrame, events: pd.DataFrame, spec: SeparationSpec
) -> pd.DataFrame:
    """Build small market/beta tables without returns or selected paths."""
    records: list[dict[str, Any]] = []
    for market in spec.markets:
        market_separation = separation.loc[separation["market"] == market]
        market_events = events.loc[events["market"] == market]
        for label in ("all", "log2", "log4"):
            values = (
                market_events
                if label == "all"
                else market_events.loc[market_events["beta_label"] == label]
            )
            valid = values.loc[values["reliability_valid"].astype(bool)]
            whipsaw = valid.loc[valid["whipsaw_20"].astype(bool)]
            persistent = valid.loc[~valid["whipsaw_20"].astype(bool)]
            records.append(
                {
                    "market": market,
                    "beta_label": label,
                    "refit_lambda_rows": len(market_separation),
                    "valid_refit_lambda_rows": int(
                        market_separation["reliability_valid"].sum()
                    ),
                    "admitted_events": len(values),
                    "valid_reliability_events": len(valid),
                    "whipsaw_events": int(values["whipsaw_20"].sum()),
                    "persistent_events": int(
                        (~values["whipsaw_20"].astype(bool)).sum()
                    ),
                    "whipsaw_fraction": (
                        float(values["whipsaw_20"].mean()) if len(values) else math.nan
                    ),
                    "median_reliability_all": (
                        float(valid["reliability_train"].median())
                        if len(valid)
                        else math.nan
                    ),
                    "median_reliability_whipsaw": (
                        float(whipsaw["reliability_train"].median())
                        if len(whipsaw)
                        else math.nan
                    ),
                    "median_reliability_persistent": (
                        float(persistent["reliability_train"].median())
                        if len(persistent)
                        else math.nan
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def _optional_finite(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def build_conclusion(
    folds: pd.DataFrame, summary: pd.DataFrame, spec: SeparationSpec
) -> dict[str, Any]:
    result = classify_decision(folds, spec.score_tolerance)
    valid_folds = folds.loc[folds["fold_valid"].astype(bool)]
    mean_baseline = (
        float(valid_folds["baseline_brier"].mean())
        if len(valid_folds) == len(MARKETS)
        else math.nan
    )
    mean_challenger = (
        float(valid_folds["challenger_brier"].mean())
        if len(valid_folds) == len(MARKETS)
        else math.nan
    )
    market_rows = summary.loc[summary["beta_label"] == "all"]
    return {
        "experiment_id": spec.experiment_id,
        "claim_class": "EXPLORATORY",
        "performance_claim_allowed": False,
        "new_model_claim_allowed": False,
        "result": result,
        "valid_folds": len(valid_folds),
        "mean_baseline_brier": _optional_finite(mean_baseline),
        "mean_challenger_brier": _optional_finite(mean_challenger),
        "mean_delta_brier": _optional_finite(mean_challenger - mean_baseline),
        "market_event_counts": {
            row.market: {
                "admitted": int(row.admitted_events),
                "valid_reliability": int(row.valid_reliability_events),
                "whipsaw": int(row.whipsaw_events),
            }
            for row in market_rows.itertuples(index=False)
        },
        "interpretation": (
            "This development-sample diagnostic tests one causal training-prefix "
            "reliability statistic. It does not establish regime truth, causal "
            "switch quality, profitability, or a new model claim."
        ),
    }
