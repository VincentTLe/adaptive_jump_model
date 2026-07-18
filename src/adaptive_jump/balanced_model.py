"""Frozen inputs and pair-balanced penalty for the balanced lagged study."""

from __future__ import annotations

import hashlib
import math
import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from adaptive_jump.config import ResearchConfig
from adaptive_jump.lagged_model import LockedStateEvidence, generate_locked_candidates
from adaptive_jump.models import FEATURE_COLUMNS
from adaptive_jump.separation_analysis import MarketInputs
from adaptive_jump.tv_jump import lagged_evidence_penalty_seq

SPEC_SHA256 = "611301f9477c538b26374ce80425d8e7b684b0d531d450a786e6c8222ffda8ba"
MARKETS = ("us", "de", "jp")
BETAS = (0.0, math.log(4.0))
DECISION_BETA = math.log(4.0)
LAMBDAS = (0.0, 5.0, 15.0, 35.0, 70.0, 150.0, 300.0, 600.0, 1200.0)
POSITIVE_LAMBDAS = LAMBDAS[1:]
RULES = ("lagged", "balanced")
FIXED_FILES = ("features.csv", "jm-states.csv")
PARENT_FILES = (
    "candidate-states-beta-0.csv",
    "candidate-states-beta-log4.csv",
    "refits-and-scales.csv",
)
FORBIDDEN_FILES = (
    "path-behavior.csv",
    "discount-events.csv",
    "mechanism-summary.csv",
    "dated-audit.csv",
    "conclusion.json",
    "summary.csv",
    "choices.csv",
    "selected-timeline.csv",
    "trades.csv",
)


class BalancedStudyError(ValueError):
    """Raised when the frozen balanced study or an invariant changes."""


@dataclass(frozen=True)
class BalancedSpec:
    path: Path
    sha256: str
    experiment_id: str
    fixed_run_id: str
    fixed_inventory_sha256: str
    parent_run_id: str
    parent_inventory_sha256: str
    parent_spec_sha256: str
    data_manifest_sha256: str
    data_cutoff: date
    evaluation_starts: dict[str, date]
    markets: tuple[str, ...]
    betas: tuple[float, ...]
    decision_beta: float
    lambdas: tuple[float, ...]
    event_lambdas: tuple[float, ...]
    rules: tuple[str, ...]
    fit_window: int
    horizon: int
    matched_entry_search: int
    matched_followup: int
    matched_anchor_censor: int
    numerical_tolerance: float
    fixed_allowed_files: tuple[str, ...]
    parent_allowed_files: tuple[str, ...]
    forbidden_files: tuple[str, ...]
    artifact_subdir: Path
    toy_losses: dict[str, np.ndarray]
    toy_paths: dict[str, dict[str, list[int]]]


def beta_label(beta: float) -> str:
    if beta == 0.0:
        return "0"
    if beta == DECISION_BETA:
        return "log4"
    raise BalancedStudyError(f"unexpected beta: {beta}")


def _toy_contract(
    document: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, list[int]]]]:
    toys = document.get("toys", {})
    losses: dict[str, np.ndarray] = {}
    paths: dict[str, dict[str, list[int]]] = {}
    for name in ("isolated", "alternating", "persistent", "reversal"):
        try:
            loss = np.asarray(toys[f"{name}_loss"], dtype=float)
            by_rule = {
                rule: [int(value) for value in toys[f"{name}_{rule}_path"]]
                for rule in ("fixed", "lagged", "balanced")
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise BalancedStudyError(f"{name} toy contract is invalid") from exc
        if (
            loss.ndim != 2
            or loss.shape[1] != 2
            or any(len(path) != len(loss) for path in by_rule.values())
        ):
            raise BalancedStudyError(f"{name} toy shapes changed")
        losses[name] = loss
        paths[name] = by_rule
    return losses, paths


def load_balanced_spec(path: str | Path, config: ResearchConfig) -> BalancedSpec:
    """Load the exact corrected frozen study; any byte change requires refreeze."""
    spec_path = Path(path).resolve()
    payload = spec_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != SPEC_SHA256:
        raise BalancedStudyError("balanced study spec differs from its frozen hash")
    try:
        document = tomllib.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise BalancedStudyError("balanced study TOML is invalid") from exc

    fixed = document.get("fixed_source", {})
    parent = document.get("parent_lagged_source", {})
    scope = document.get("scope", {})
    penalty = document.get("penalty", {})
    candidates = document.get("candidates", {})
    events = document.get("events", {})
    decision = document.get("decision", {})
    execution = document.get("execution", {})
    storage = document.get("storage", {})
    try:
        cutoff = date.fromisoformat(str(fixed["data_cutoff"]))
        starts = {
            market: date.fromisoformat(str(scope["evaluation_start"][market]))
            for market in MARKETS
        }
        artifact_subdir = Path(str(storage["artifact_subdir"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise BalancedStudyError("balanced source or scope fields are invalid") from exc

    betas = tuple(float(value) for value in penalty.get("beta", ()))
    lambdas = tuple(float(value) for value in candidates.get("raw_lambda_grid", ()))
    event_lambdas = tuple(
        float(value) for value in candidates.get("event_lambda_grid", ())
    )
    rules = tuple(candidates.get("rules", ()))
    fixed_files = tuple(fixed.get("allowed_market_files", ()))
    parent_files = tuple(parent.get("allowed_market_files", ()))
    forbidden = tuple(parent.get("forbidden_files", ()))
    matched_entry_search = events.get("matched_entry_search_candidate_dates")
    matched_followup = events.get("matched_followup_candidate_dates")
    matched_anchor_censor = events.get("matched_anchor_censor_candidate_dates")
    losses, paths = _toy_contract(document)

    if (
        document.get("schema_version") != 1
        or document.get("experiment_id") != "balanced-lagged-mechanism-001"
        or document.get("claim_class") != "EXPLORATORY"
        or document.get("stage")
        != "DEVELOPMENT_SAMPLE_PERFORMANCE_FREE_MECHANISM_STUDY"
        or any(
            document.get(name) is not False
            for name in (
                "performance_claim_allowed",
                "paper_replication_claim_allowed",
                "new_model_claim_allowed",
                "post_2023_access",
                "provider_access",
                "monthly_selection_allowed",
            )
        )
        or fixed.get("experiment_id") != "fixed-baselines-001-v7"
        or fixed.get("config_sha256") != config.sha256
        or parent.get("experiment_id") != "lagged-evidence-mechanism-001"
        or parent.get("spec_sha256")
        != "6f964f5724b23cffff43c37ca050af1dd7eb37a3c7e7588462a86571e8825ed1"
        or cutoff != date(2023, 12, 31)
        or tuple(scope.get("markets", ())) != MARKETS
        or tuple(scope.get("features", ())) != FEATURE_COLUMNS
        or starts
        != {
            "us": date(2007, 12, 4),
            "de": date(2008, 1, 3),
            "jp": date(2009, 5, 7),
        }
        or betas != BETAS
        or float(penalty.get("decision_beta", math.nan)) != DECISION_BETA
        or lambdas != LAMBDAS
        or lambdas != config.jm_protocol.lambda_grid
        or event_lambdas != POSITIVE_LAMBDAS
        or rules != RULES
        or fixed_files != FIXED_FILES
        or parent_files != PARENT_FILES
        or forbidden != FORBIDDEN_FILES
        or int(scope.get("fit_window_observations", 0))
        != config.model_protocol.fit_window
        or tuple(scope.get("jm_refit_months", ())) != config.jm_protocol.refit_months
        or int(events.get("horizon_candidate_dates", 0)) != 20
        or not isinstance(matched_entry_search, int)
        or isinstance(matched_entry_search, bool)
        or matched_entry_search != 20
        or not isinstance(matched_followup, int)
        or isinstance(matched_followup, bool)
        or matched_followup != 20
        or not isinstance(matched_anchor_censor, int)
        or isinstance(matched_anchor_censor, bool)
        or matched_anchor_censor != 40
        or decision.get("decision_beta_label") != "log4"
        or float(decision.get("numerical_tolerance", math.nan)) != 1e-12
        or execution.get("full_markets_parallel") is not True
        or execution.get("market_workers") != len(MARKETS)
        or execution.get("threadpool_limit_per_worker") != 1
        or artifact_subdir != Path("balanced-lagged-mechanism-001")
        or artifact_subdir.is_absolute()
        or ".." in artifact_subdir.parts
    ):
        raise BalancedStudyError("balanced study controls changed")

    return BalancedSpec(
        path=spec_path,
        sha256=digest,
        experiment_id=str(document["experiment_id"]),
        fixed_run_id=str(fixed["run_id"]),
        fixed_inventory_sha256=str(fixed["run_inventory_sha256"]),
        parent_run_id=str(parent["run_id"]),
        parent_inventory_sha256=str(parent["run_inventory_sha256"]),
        parent_spec_sha256=str(parent["spec_sha256"]),
        data_manifest_sha256=str(fixed["data_manifest_sha256"]),
        data_cutoff=cutoff,
        evaluation_starts=starts,
        markets=MARKETS,
        betas=betas,
        decision_beta=DECISION_BETA,
        lambdas=lambdas,
        event_lambdas=event_lambdas,
        rules=rules,
        fit_window=int(scope["fit_window_observations"]),
        horizon=int(events["horizon_candidate_dates"]),
        matched_entry_search=matched_entry_search,
        matched_followup=matched_followup,
        matched_anchor_censor=matched_anchor_censor,
        numerical_tolerance=float(decision["numerical_tolerance"]),
        fixed_allowed_files=fixed_files,
        parent_allowed_files=parent_files,
        forbidden_files=forbidden,
        artifact_subdir=artifact_subdir,
        toy_losses=losses,
        toy_paths=paths,
    )


def balanced_lagged_penalty_seq(
    loss_mx: np.ndarray,
    lambda0: float,
    beta: float,
    q_train: float,
) -> np.ndarray:
    """Build the frozen pair-balanced signed lagged transition matrices."""
    loss = np.asarray(loss_mx, dtype=float)
    if loss.ndim != 2 or loss.shape[0] == 0 or loss.shape[1] < 2:
        raise ValueError("loss_mx must be non-empty with at least two states")
    if np.isneginf(loss).any():
        raise ValueError("loss_mx must not contain negative infinity")
    loss = np.where(np.isnan(loss), np.inf, loss)
    if not np.isfinite(loss).any(axis=1).all():
        raise ValueError("each loss row must contain a finite state")
    try:
        lambda0, beta, q_train = float(lambda0), float(beta), float(q_train)
    except (TypeError, ValueError) as exc:
        raise ValueError("lambda0, beta, and q_train must be scalars") from exc
    if not np.isfinite(lambda0) or lambda0 < 0:
        raise ValueError("lambda0 must be finite and nonnegative")
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative")
    if not np.isfinite(q_train) or q_train <= 0:
        raise ValueError("q_train must be finite and positive")

    lagged = np.zeros_like(loss)
    lagged[1:] = loss[:-1]
    source = lagged[:, :, None]
    destination = lagged[:, None, :]
    with np.errstate(invalid="ignore"):
        signed_gap = source - destination
    both_unavailable = np.isposinf(source) & np.isposinf(destination)
    signed_gap = np.where(both_unavailable, 0.0, signed_gap)
    alpha = 1.0 - math.exp(-beta)
    penalty = lambda0 * (1.0 - alpha * np.tanh(signed_gap / q_train))
    states = np.arange(loss.shape[1])
    penalty[:, states, states] = 0.0
    if not np.isfinite(penalty).all() or (penalty < 0).any():
        raise ValueError("balanced penalties must be finite and nonnegative")
    return penalty


BUILDERS = {
    "lagged": lagged_evidence_penalty_seq,
    "balanced": balanced_lagged_penalty_seq,
}


def _read_candidates(path: Path, spec: BalancedSpec) -> pd.DataFrame:
    columns = ["date", *(str(value) for value in spec.lambdas)]
    frame = pd.read_csv(path, usecols=columns)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if (
        frame["date"].duplicated().any()
        or not frame["date"].is_monotonic_increasing
        or frame["date"].max().date() > spec.data_cutoff
    ):
        raise BalancedStudyError(f"candidate dates changed: {path}")
    frame = frame.set_index("date")
    frame.columns = spec.lambdas
    values = frame.to_numpy(dtype=float)
    present = np.isfinite(values)
    if not np.array_equal(present.any(axis=1), present.all(axis=1)):
        raise BalancedStudyError(f"partial candidate row: {path}")
    if values[present].size and not np.isin(values[present], (0.0, 1.0)).all():
        raise BalancedStudyError(f"nonbinary candidate state: {path}")
    return frame


def load_market_inputs(
    market: str,
    fixed_market_dir: Path,
    parent_market_dir: Path,
    spec: BalancedSpec,
) -> tuple[MarketInputs, pd.DataFrame]:
    """Read only allowlisted features, parent states, refits, and fixed states."""
    if market not in spec.markets:
        raise BalancedStudyError(f"unknown market: {market}")
    features = pd.read_csv(
        fixed_market_dir / "features.csv", usecols=["date", *FEATURE_COLUMNS]
    )
    features["date"] = pd.to_datetime(features["date"], errors="raise")
    if (
        features["date"].duplicated().any()
        or not features["date"].is_monotonic_increasing
        or features["date"].max().date() > spec.data_cutoff
    ):
        raise BalancedStudyError(f"{market}: feature dates changed")
    for column in FEATURE_COLUMNS:
        features[column] = pd.to_numeric(features[column], errors="raise")
    observed = features.loc[:, FEATURE_COLUMNS].dropna().to_numpy(dtype=float)
    if not np.isfinite(observed).all():
        raise BalancedStudyError(f"{market}: nonfinite observed feature")
    features = features.set_index("date")

    candidates = {
        0.0: _read_candidates(parent_market_dir / "candidate-states-beta-0.csv", spec),
        DECISION_BETA: _read_candidates(
            parent_market_dir / "candidate-states-beta-log4.csv", spec
        ),
    }
    if any(not frame.index.equals(features.index) for frame in candidates.values()):
        raise BalancedStudyError(f"{market}: source dates differ")
    if (
        not candidates[0.0]
        .notna()
        .all(axis=1)
        .equals(candidates[DECISION_BETA].notna().all(axis=1))
    ):
        raise BalancedStudyError(f"{market}: source coverage differs")

    refit_columns = [
        "market",
        "fit_date",
        "training_start",
        "training_end",
        "lambda0",
        "q_train",
        "scaler_mean",
        "scaler_scale",
        "centers",
    ]
    refits = pd.read_csv(
        parent_market_dir / "refits-and-scales.csv", usecols=refit_columns
    )
    for column in ("fit_date", "training_start", "training_end"):
        refits[column] = pd.to_datetime(refits[column], errors="raise")
    for column in ("lambda0", "q_train"):
        refits[column] = pd.to_numeric(refits[column], errors="raise")
    if (
        set(refits["market"]) != {market}
        or refits.duplicated(["fit_date", "lambda0"]).any()
        or set(refits["lambda0"]) != set(spec.lambdas)
        or (refits["training_end"] != refits["fit_date"]).any()
        or refits["fit_date"].max().date() > spec.data_cutoff
        or not np.isfinite(refits[["lambda0", "q_train"]]).all().all()
        or (refits["q_train"] <= 0).any()
    ):
        raise BalancedStudyError(f"{market}: refit table changed")
    if not (refits.groupby("fit_date")["lambda0"].nunique() == len(spec.lambdas)).all():
        raise BalancedStudyError(f"{market}: refit lambda coverage changed")

    first_fit = pd.Timestamp(refits["fit_date"].min())
    first_starts = refits.loc[
        refits["fit_date"] == first_fit, "training_start"
    ].unique()
    complete = features.dropna(subset=list(FEATURE_COLUMNS))
    if len(first_starts) != 1:
        raise BalancedStudyError(f"{market}: first training start changed")
    initial = complete.loc[pd.Timestamp(first_starts[0]) : first_fit]
    if len(initial) != spec.fit_window:
        raise BalancedStudyError(f"{market}: initial training prefix changed")
    state_dates = candidates[0.0].index[candidates[0.0].notna().all(axis=1)]
    if len(state_dates) == 0 or state_dates[0] != first_fit:
        raise BalancedStudyError(f"{market}: candidate start changed")
    model_dates = initial.index.append(state_dates[state_dates > first_fit])
    if model_dates.has_duplicates or not model_dates.is_monotonic_increasing:
        raise BalancedStudyError(f"{market}: reconstructed model dates changed")
    if features.loc[state_dates, FEATURE_COLUMNS].isna().any().any():
        raise BalancedStudyError(f"{market}: candidate date has missing feature")

    fixed = _read_candidates(fixed_market_dir / "jm-states.csv", spec)
    if not fixed.index.equals(features.index) or not np.array_equal(
        fixed.to_numpy(), candidates[0.0].to_numpy(), equal_nan=True
    ):
        raise BalancedStudyError(f"{market}: fixed and parent beta-zero paths differ")
    inputs = MarketInputs(
        market=market,
        features=features,
        model_dates=model_dates,
        candidates=candidates,
        refits=refits.sort_values(["fit_date", "lambda0"]).reset_index(drop=True),
    )
    return inputs, fixed


def generate_candidates(
    inputs: MarketInputs,
    fixed: pd.DataFrame,
    config: ResearchConfig,
    spec: BalancedSpec,
    *,
    terminal_limit: int | None = None,
    features: pd.DataFrame | None = None,
) -> dict[str, LockedStateEvidence]:
    frame = inputs.features if features is None else features
    if not frame.index.equals(inputs.features.index):
        raise BalancedStudyError("feature override changed source dates")
    return generate_locked_candidates(
        frame.reset_index(),
        fixed,
        inputs.refits,
        config,
        spec,  # generic locked generator uses only the shared frozen fields
        market=inputs.market,
        penalty_builders=BUILDERS,
        terminal_limit=terminal_limit,
    )
