"""Mathematical core and frozen contract for adaptive-separation-001."""

from __future__ import annotations

import hashlib
import math
import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from adaptive_jump.tv_jump import dp_tv, loss_matrix

MARKETS = ("us", "de", "jp")
BETAS = (math.log(2.0), math.log(4.0))
LAMBDAS = (5.0, 15.0, 35.0, 70.0, 150.0, 300.0, 600.0, 1200.0)
ADAPTIVE_FILES = (
    "candidate-states-beta-0.csv",
    "candidate-states-beta-log2.csv",
    "candidate-states-beta-log4.csv",
    "refits-and-scales.csv",
)
FIXED_FILES = ("features.csv",)
FORBIDDEN_FILES = (
    "summary.csv",
    "selected-timeline.csv",
    "choices.csv",
    "conclusion.json",
)


class SeparationStudyError(ValueError):
    """Raised when the frozen diagnostic or one of its invariants fails."""


@dataclass(frozen=True)
class SeparationSpec:
    path: Path
    sha256: str
    experiment_id: str
    adaptive_run_id: str
    adaptive_inventory_sha256: str
    adaptive_spec_sha256: str
    fixed_run_id: str
    fixed_inventory_sha256: str
    data_manifest_sha256: str
    data_cutoff: date
    evaluation_starts: dict[str, date]
    markets: tuple[str, ...]
    betas: tuple[float, ...]
    lambdas: tuple[float, ...]
    horizon: int
    fit_window: int
    objective_tolerance: float
    score_tolerance: float
    coefficient_tolerance: float
    optimizer_max_iterations: int
    optimizer_gradient_tolerance: float
    adaptive_allowed_files: tuple[str, ...]
    fixed_allowed_files: tuple[str, ...]
    performance_files_forbidden: tuple[str, ...]
    artifact_subdir: Path


@dataclass(frozen=True)
class SeparationResult:
    valid: bool
    center_distance: float
    preferred_count_0: int
    preferred_count_1: int
    tie_count: int
    median_radius_0: float
    median_radius_1: float
    reliability_train: float


@dataclass(frozen=True)
class TerminalDecision:
    state: int
    predecessor: int
    state_margin: float
    predecessor_margin: float
    state_tied: bool
    predecessor_tied: bool


@dataclass(frozen=True)
class LogisticFit:
    """Unpenalized logistic MLE; ``coef`` is intercept first."""

    coef: np.ndarray
    converged: bool
    gradient_inf: float
    iterations: int

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=float)
        if values.ndim != 2 or values.shape[1] + 1 != len(self.coef):
            raise SeparationStudyError("logistic prediction shape changed")
        if not np.isfinite(values).all():
            raise SeparationStudyError("logistic prediction features must be finite")
        return expit(self.coef[0] + values @ self.coef[1:])


def load_separation_spec(path: str | Path) -> SeparationSpec:
    """Load and strictly validate the final frozen mechanism diagnostic."""
    spec_path = Path(path).resolve()
    payload = spec_path.read_bytes()
    try:
        document = tomllib.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise SeparationStudyError(f"invalid separation study TOML: {exc}") from exc

    if (
        document.get("schema_version") != 1
        or document.get("experiment_id") != "adaptive-separation-001"
        or document.get("claim_class") != "EXPLORATORY"
        or document.get("performance_claim_allowed") is not False
        or document.get("new_model_claim_allowed") is not False
        or document.get("extension_access") is not False
        or document.get("post_2023_access") is not False
    ):
        raise SeparationStudyError("separation study identity or evidence lane changed")

    adaptive = document.get("adaptive_source", {})
    fixed = document.get("fixed_parent", {})
    scope = document.get("scope", {})
    separation = document.get("separation", {})
    events = document.get("events", {})
    prediction = document.get("prediction", {})
    decision = document.get("decision", {})
    storage = document.get("storage", {})
    try:
        cutoff = date.fromisoformat(str(fixed["data_cutoff"]))
        evaluation_starts = {
            market: date.fromisoformat(str(scope["evaluation_start"][market]))
            for market in MARKETS
        }
        artifact_subdir = Path(str(storage["artifact_subdir"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise SeparationStudyError(
            "separation source or storage fields missing"
        ) from exc

    adaptive_files = tuple(adaptive.get("allowed_market_files", ()))
    fixed_files = tuple(fixed.get("allowed_market_files", ()))
    forbidden = tuple(adaptive.get("performance_files_forbidden", ()))
    betas = tuple(float(value) for value in events.get("betas", ()))
    lambdas = tuple(float(value) for value in events.get("raw_lambda_grid", ()))
    if (
        adaptive.get("experiment_id") != "adaptive-confidence-001"
        or fixed.get("experiment_id") != "fixed-baselines-001-v7"
        or fixed.get("config_sha256")
        != "8adb330565d64f8ed6edd986f0422dbba72585eda4efd34b0c1b41b95450d81b"
        or cutoff != date(2023, 12, 31)
        or tuple(scope.get("markets", ())) != MARKETS
        or evaluation_starts
        != {
            "us": date(2007, 12, 4),
            "de": date(2008, 1, 3),
            "jp": date(2009, 5, 7),
        }
        or adaptive_files != ADAPTIVE_FILES
        or fixed_files != FIXED_FILES
        or forbidden != FORBIDDEN_FILES
        or betas != BETAS
        or lambdas != LAMBDAS
        or int(events.get("horizon_signal_days", 0)) != 20
        or int(separation.get("training_prefix_observations", 0)) != 3000
        or prediction.get("log_discount")
        != (
            "natural log(lambda0 / C_t(i,j)) = beta*tanh((L_t(i)-L_t(j))/"
            "q_train), nonnegative on admitted events"
        )
        or prediction.get("estimator")
        != "unpenalized logistic regression with intercept and analytic gradient"
        or float(decision.get("numerical_score_tolerance", math.nan)) != 1e-12
        or float(decision.get("standardized_coefficient_tolerance", math.nan)) != 1e-12
        or artifact_subdir.is_absolute()
        or not artifact_subdir.parts
        or ".." in artifact_subdir.parts
    ):
        raise SeparationStudyError("separation study controls changed")

    return SeparationSpec(
        path=spec_path,
        sha256=hashlib.sha256(payload).hexdigest(),
        experiment_id=document["experiment_id"],
        adaptive_run_id=str(adaptive["run_id"]),
        adaptive_inventory_sha256=str(adaptive["run_inventory_sha256"]),
        adaptive_spec_sha256=str(adaptive["spec_sha256"]),
        fixed_run_id=str(fixed["run_id"]),
        fixed_inventory_sha256=str(fixed["run_inventory_sha256"]),
        data_manifest_sha256=str(fixed["data_manifest_sha256"]),
        data_cutoff=cutoff,
        evaluation_starts=evaluation_starts,
        markets=MARKETS,
        betas=betas,
        lambdas=lambdas,
        horizon=int(events["horizon_signal_days"]),
        fit_window=int(separation["training_prefix_observations"]),
        objective_tolerance=float(separation["fixed_objective_abs_tolerance"]),
        score_tolerance=float(decision["numerical_score_tolerance"]),
        coefficient_tolerance=float(decision["standardized_coefficient_tolerance"]),
        optimizer_max_iterations=int(prediction["optimizer_max_iterations"]),
        optimizer_gradient_tolerance=float(prediction["optimizer_gradient_tolerance"]),
        adaptive_allowed_files=adaptive_files,
        fixed_allowed_files=fixed_files,
        performance_files_forbidden=forbidden,
        artifact_subdir=artifact_subdir,
    )


def _invalid_separation(
    *,
    center_distance: float = math.nan,
    counts: tuple[int, int] = (0, 0),
    ties: int = 0,
) -> SeparationResult:
    return SeparationResult(
        valid=False,
        center_distance=center_distance,
        preferred_count_0=counts[0],
        preferred_count_1=counts[1],
        tie_count=ties,
        median_radius_0=math.nan,
        median_radius_1=math.nan,
        reliability_train=math.nan,
    )


def reliability_from_geometry(
    features: np.ndarray, centers: np.ndarray
) -> SeparationResult:
    """Compute the frozen robust geometric center-separation statistic."""
    values = np.asarray(features, dtype=float)
    center_values = np.asarray(centers, dtype=float)
    if values.ndim != 2 or len(values) == 0:
        raise SeparationStudyError("separation features must be a nonempty matrix")
    if center_values.shape != (2, values.shape[1]):
        raise SeparationStudyError("separation centers must have shape (2, n_features)")
    if not np.isfinite(values).all():
        raise SeparationStudyError("separation features must be finite")
    if not np.isfinite(center_values).all():
        return _invalid_separation()

    losses = loss_matrix(values, center_values)
    mask0 = losses[:, 0] < losses[:, 1]
    mask1 = losses[:, 1] < losses[:, 0]
    ties = int((~mask0 & ~mask1).sum())
    counts = (int(mask0.sum()), int(mask1.sum()))
    distance = float(np.linalg.norm(center_values[0] - center_values[1]))
    if counts[0] == 0 or counts[1] == 0:
        return _invalid_separation(center_distance=distance, counts=counts, ties=ties)

    euclidean = np.sqrt(2.0 * losses)
    radius0 = float(np.median(euclidean[mask0, 0]))
    radius1 = float(np.median(euclidean[mask1, 1]))
    denominator = distance + radius0 + radius1
    if not np.isfinite(denominator) or denominator <= 0:
        return _invalid_separation(center_distance=distance, counts=counts, ties=ties)
    reliability = distance / denominator
    if not np.isfinite(reliability) or not 0.0 <= reliability <= 1.0:
        raise SeparationStudyError("separation reliability left [0,1]")
    return SeparationResult(
        valid=True,
        center_distance=distance,
        preferred_count_0=counts[0],
        preferred_count_1=counts[1],
        tie_count=ties,
        median_radius_0=radius0,
        median_radius_1=radius1,
        reliability_train=reliability,
    )


def _argmin_with_margin(values: np.ndarray) -> tuple[int, float, bool]:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or len(vector) < 2 or not np.isfinite(vector).any():
        raise SeparationStudyError("terminal decision vector is invalid")
    minimum = float(np.nanmin(vector))
    state = int(np.nanargmin(vector))
    tied = int(np.count_nonzero(vector == minimum)) > 1
    ordered = np.sort(vector[np.isfinite(vector)])
    margin = float(ordered[1] - ordered[0]) if len(ordered) > 1 else math.inf
    return state, margin, tied


def terminal_decision(loss: np.ndarray, penalty_seq: np.ndarray) -> TerminalDecision:
    """Return the terminal online state and its exact last-step predecessor."""
    losses = np.asarray(loss, dtype=float)
    penalties = np.asarray(penalty_seq, dtype=float)
    if losses.ndim != 2 or len(losses) < 2:
        raise SeparationStudyError("terminal attribution needs at least two rows")
    values = dp_tv(losses, penalties, return_value_mx=True)
    state, state_margin, state_tied = _argmin_with_margin(values[-1])
    predecessor_cost = values[-2] + penalties[-1, :, state]
    predecessor, predecessor_margin, predecessor_tied = _argmin_with_margin(
        predecessor_cost
    )
    return TerminalDecision(
        state=state,
        predecessor=predecessor,
        state_margin=state_margin,
        predecessor_margin=predecessor_margin,
        state_tied=state_tied,
        predecessor_tied=predecessor_tied,
    )


def arrival_ablation_state(
    loss: np.ndarray, penalty_seq: np.ndarray, lambda0: float
) -> int:
    """Reset only the final arrival costs to fixed lambda and emit the state."""
    losses = np.asarray(loss, dtype=float)
    penalties = np.asarray(penalty_seq, dtype=float)
    if penalties.ndim != 3 or penalties.shape[:2] != losses.shape:
        raise SeparationStudyError("ablation loss and penalty shapes differ")
    if not np.isfinite(lambda0) or lambda0 < 0:
        raise SeparationStudyError("ablation lambda must be finite and nonnegative")
    ablated = penalties.copy()
    n_states = losses.shape[1]
    ablated[-1] = lambda0 * (1.0 - np.eye(n_states))
    return int(dp_tv(losses, ablated, return_value_mx=True)[-1].argmin())


def fit_logistic(
    features: np.ndarray,
    outcome: np.ndarray,
    weights: np.ndarray,
    *,
    max_iterations: int = 10_000,
    gradient_tolerance: float = 1e-9,
) -> LogisticFit:
    """Fit the frozen unpenalized weighted logistic model."""
    values = np.asarray(features, dtype=float)
    target = np.asarray(outcome, dtype=float)
    sample_weight = np.asarray(weights, dtype=float)
    if (
        values.ndim != 2
        or target.shape != (len(values),)
        or sample_weight.shape != (len(values),)
        or len(values) == 0
        or not np.isfinite(values).all()
        or not np.isin(target, (0.0, 1.0)).all()
        or not np.isfinite(sample_weight).all()
        or (sample_weight <= 0).any()
        or len(np.unique(target)) != 2
    ):
        raise SeparationStudyError("invalid logistic training data")
    design = np.column_stack([np.ones(len(values)), values])
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise SeparationStudyError("logistic design is not full rank")
    normalized_weight = sample_weight / sample_weight.sum()

    def objective(coef: np.ndarray) -> tuple[float, np.ndarray]:
        linear = design @ coef
        value = float(
            np.sum(normalized_weight * (np.logaddexp(0.0, linear) - target * linear))
        )
        gradient = design.T @ (normalized_weight * (expit(linear) - target))
        return value, gradient

    result = minimize(
        objective,
        np.zeros(design.shape[1]),
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": int(max_iterations),
            "gtol": float(gradient_tolerance),
            "ftol": 1e-15,
            "maxls": 100,
        },
    )
    _, gradient = objective(np.asarray(result.x, dtype=float))
    gradient_inf = float(np.max(np.abs(gradient)))
    converged = bool(
        result.success
        and np.isfinite(result.x).all()
        and np.isfinite(gradient_inf)
        and gradient_inf <= gradient_tolerance
    )
    if not converged:
        raise SeparationStudyError(
            "logistic optimizer did not satisfy gradient tolerance "
            f"(gradient_inf={gradient_inf:.17g}, tolerance="
            f"{gradient_tolerance:.17g}): {result.message}"
        )
    return LogisticFit(
        coef=np.asarray(result.x, dtype=float),
        converged=True,
        gradient_inf=gradient_inf,
        iterations=int(result.nit),
    )


def prediction_scores(
    outcome: np.ndarray, probability: np.ndarray
) -> tuple[float, float]:
    """Return mean Brier score and binary log loss."""
    target = np.asarray(outcome, dtype=float)
    predicted = np.asarray(probability, dtype=float)
    if (
        target.shape != predicted.shape
        or target.ndim != 1
        or len(target) == 0
        or not np.isin(target, (0.0, 1.0)).all()
        or not np.isfinite(predicted).all()
        or (predicted < 0).any()
        or (predicted > 1).any()
    ):
        raise SeparationStudyError("invalid prediction score inputs")
    clipped = np.clip(predicted, np.finfo(float).eps, 1.0 - np.finfo(float).eps)
    brier = float(np.mean((predicted - target) ** 2))
    log_loss = float(
        -np.mean(target * np.log(clipped) + (1.0 - target) * np.log1p(-clipped))
    )
    return brier, log_loss


def classify_decision(folds: pd.DataFrame, tol: float) -> str:
    """Apply the mutually exclusive frozen support/falsification rule."""
    required = {
        "held_out_market",
        "fold_valid",
        "admitted_events",
        "reliability_coefficient",
        "baseline_brier",
        "challenger_brier",
    }
    if not required.issubset(folds) or len(folds) != len(MARKETS):
        raise SeparationStudyError("decision fold table is incomplete")
    if set(folds["held_out_market"]) != set(MARKETS):
        raise SeparationStudyError("decision markets changed")
    if (
        not folds["fold_valid"].astype(bool).all()
        or (pd.to_numeric(folds["admitted_events"]) <= 0).any()
    ):
        return "inconclusive"
    numeric = folds.loc[
        :, ["reliability_coefficient", "baseline_brier", "challenger_brier"]
    ].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy()).all():
        return "inconclusive"
    coefficient = numeric["reliability_coefficient"].to_numpy()
    delta = (numeric["challenger_brier"] - numeric["baseline_brier"]).to_numpy()
    if (
        (coefficient < -tol).all()
        and float(delta.mean()) < -tol
        and int((delta < -tol).sum()) >= 2
    ):
        return "supported"
    if int((coefficient >= -tol).sum()) >= 2 and float(delta.mean()) >= -tol:
        return "falsified"
    return "inconclusive"
