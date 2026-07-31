"""Frozen contract and decision logic for lagged-evidence-mechanism-001."""

from __future__ import annotations

import hashlib
import math
import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from adaptive_jump.config import ResearchConfig
from adaptive_jump.models import FEATURE_COLUMNS

MARKETS = ("us", "de", "jp")
BETAS = (0.0, math.log(2.0), math.log(4.0))
POSITIVE_BETAS = BETAS[1:]
LAMBDAS = (0.0, 5.0, 15.0, 35.0, 70.0, 150.0, 300.0, 600.0, 1200.0)
POSITIVE_LAMBDAS = LAMBDAS[1:]
RULES = ("arrival", "lagged")
FIXED_FILES = ("features.csv", "jm-states.csv")
ARRIVAL_FILES = (
    "candidate-states-beta-0.csv",
    "candidate-states-beta-log2.csv",
    "candidate-states-beta-log4.csv",
    "refits-and-scales.csv",
)
REQUIRED_VERIFICATION = frozenset(
    {
        "strict_spec_and_registry_lock",
        "source_inventory_exact",
        "no_performance_file_access",
        "all_dates_through_2023",
        "beta_zero_candidate_states_equal_parent",
        "full_sealed_arrival_candidate_states_reproduced",
        "mechanical_evidence_recomputed",
        "formula_reconstruction_from_rule_evidence_date",
        "candidate_state_reconstruction",
        "terminal_predecessor_and_tie_audit",
        "local_ablation_reconstruction",
        "event_horizon_and_overlap_audit",
        "decision_reconstruction",
        "dated_audit",
        "per_market_and_root_tables_equal",
        "source_and_implementation_locks_recomputed",
        "us_smoke_before_parallel_markets",
        "us_smoke_independently_recomputed",
    }
)
US_SMOKE_PROTOCOL = (
    "all lambdas and betas on the first 20 terminal dates while both runs receive "
    "full features through the genuine second refit; mutate only post-prefix "
    "features, require changed future losses and unchanged prefix states, and "
    "verify beta-zero, sealed arrival, lagged formula, and current-fit-versus-stale "
    "refit convention without fitting"
)

FORBIDDEN_FILES = (
    "summary.csv",
    "selected-timeline.csv",
    "choices.csv",
    "conclusion.json",
    "trades.csv",
)


class LaggedStudyError(ValueError):
    """Raised when the frozen mechanism study or an invariant changes."""


@dataclass(frozen=True)
class LaggedMechanismSpec:
    path: Path
    sha256: str
    experiment_id: str
    fixed_run_id: str
    fixed_inventory_sha256: str
    arrival_run_id: str
    arrival_inventory_sha256: str
    arrival_spec_sha256: str
    data_manifest_sha256: str
    data_cutoff: date
    evaluation_starts: dict[str, date]
    markets: tuple[str, ...]
    betas: tuple[float, ...]
    event_betas: tuple[float, ...]
    lambdas: tuple[float, ...]
    event_lambdas: tuple[float, ...]
    rules: tuple[str, ...]
    fit_window: int
    horizon: int
    numerical_tolerance: float
    fixed_allowed_files: tuple[str, ...]
    arrival_allowed_files: tuple[str, ...]
    performance_files_forbidden: tuple[str, ...]
    artifact_subdir: Path


def _dates(scope: dict[str, Any]) -> dict[str, date]:
    try:
        return {
            market: date.fromisoformat(str(scope["evaluation_start"][market]))
            for market in MARKETS
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise LaggedStudyError("lagged evaluation starts are invalid") from exc


def load_lagged_spec(path: str | Path, config: ResearchConfig) -> LaggedMechanismSpec:
    """Load and strictly bind the performance-free mechanism contract."""
    spec_path = Path(path).resolve()
    payload = spec_path.read_bytes()
    try:
        document = tomllib.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise LaggedStudyError(f"invalid lagged study TOML: {exc}") from exc

    if (
        document.get("schema_version") != 1
        or document.get("experiment_id") != "lagged-evidence-mechanism-001"
        or document.get("claim_class") != "EXPLORATORY"
        or document.get("stage")
        != "DEVELOPMENT_SAMPLE_PERFORMANCE_FREE_MECHANISM_STUDY"
        or document.get("performance_claim_allowed") is not False
        or document.get("new_model_claim_allowed") is not False
        or document.get("extension_access") is not False
        or document.get("post_2023_access") is not False
        or document.get("monthly_performance_selection_allowed") is not False
    ):
        raise LaggedStudyError("lagged study identity or evidence lane changed")

    fixed = document.get("fixed_source", {})
    arrival = document.get("arrival_source", {})
    scope = document.get("scope", {})
    penalty = document.get("penalty", {})
    candidates = document.get("candidates", {})
    controls = document.get("controls", {})
    events = document.get("events", {})
    decision = document.get("decision", {})
    verification = document.get("verification", {})
    execution = document.get("execution", {})
    storage = document.get("storage", {})

    try:
        cutoff = date.fromisoformat(str(fixed["data_cutoff"]))
        starts = _dates(scope)
        artifact_subdir = Path(str(storage["artifact_subdir"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise LaggedStudyError("lagged source or storage fields are missing") from exc

    betas = tuple(float(value) for value in penalty.get("beta", ()))
    event_betas = tuple(float(value) for value in events.get("betas", ()))
    lambdas = tuple(float(value) for value in candidates.get("raw_lambda_grid", ()))
    event_lambdas = tuple(
        float(value) for value in candidates.get("event_lambda_grid", ())
    )
    rules = tuple(events.get("rules", ()))
    fixed_files = tuple(fixed.get("allowed_market_files", ()))
    arrival_files = tuple(arrival.get("allowed_market_files", ()))
    forbidden = tuple(arrival.get("performance_files_forbidden", ()))

    if (
        fixed.get("experiment_id") != "fixed-baselines-001-v7"
        or fixed.get("config_sha256") != config.sha256
        or fixed.get("data_manifest_sha256")
        != "3636939b525d604c5c4180d7e3abb6192b53b81a068f009ad6ca83a945e53a84"
        or arrival.get("experiment_id") != "adaptive-confidence-001"
        or arrival.get("spec_sha256")
        != "1b0c327b2db44f39be183b153e6feaae6c53e2cad1e56e782f1ef7eda3849cc3"
        or cutoff != date(2023, 12, 31)
        or starts
        != {
            "us": date(2007, 12, 4),
            "de": date(2008, 1, 3),
            "jp": date(2009, 5, 7),
        }
        or tuple(scope.get("markets", ())) != MARKETS
        or scope.get("performance_files_accessed") is not False
        or betas != BETAS
        or event_betas != POSITIVE_BETAS
        or lambdas != LAMBDAS
        or lambdas != config.jm_protocol.lambda_grid
        or event_lambdas != POSITIVE_LAMBDAS
        or rules != RULES
        or fixed_files != FIXED_FILES
        or arrival_files != ARRIVAL_FILES
        or forbidden != FORBIDDEN_FILES
        or candidates.get("raw_grid_expansion") is not False
        or candidates.get("beta_selected_by_performance") is not False
        or candidates.get("lambda_selected_by_performance") is not False
        or candidates.get("center_refit") is not False
        or tuple(controls.get("features", ())) != FEATURE_COLUMNS
        or candidates.get("fitted_parameter_source")
        != "sealed adaptive-confidence-001 refits-and-scales.csv; no model refit"
        or controls.get("fit_window_observations") != config.model_protocol.fit_window
        or tuple(controls.get("jm_refit_months", ())) != config.jm_protocol.refit_months
        or controls.get("provider_access") is not False
        or penalty.get("q_train_fallback") != "none"
        or penalty.get("information_claim")
        != (
            "causal lagged-loss evidence; not strictly F_(t-1)-predictable "
            "on a refit date"
        )
        or int(events.get("horizon_signal_days", 0)) != 20
        or float(decision.get("numerical_tolerance", math.nan)) != 1e-12
        or decision.get("selection_if_both_advance")
        != "select log2 because it is the smaller deformation from fixed JM"
        or artifact_subdir.is_absolute()
        or not artifact_subdir.parts
        or set(verification) != REQUIRED_VERIFICATION
        or any(verification.get(name) is not True for name in REQUIRED_VERIFICATION)
        or execution.get("us_smoke") != US_SMOKE_PROTOCOL
        or execution.get("full_arrival_replay")
        != (
            "regenerate every arrival candidate state from sealed fitted parameters "
            "and compare to its sealed source before analysis"
        )
        or execution.get("full_markets_parallel") is not True
        or execution.get("market_workers") != len(MARKETS)
        or execution.get("threadpool_limit_per_worker") != 1
        or storage.get("mechanical_checks_json") != "mechanical-checks.json"
        or ".." in artifact_subdir.parts
    ):
        raise LaggedStudyError("lagged study controls changed")

    return LaggedMechanismSpec(
        path=spec_path,
        sha256=hashlib.sha256(payload).hexdigest(),
        experiment_id=str(document["experiment_id"]),
        fixed_run_id=str(fixed["run_id"]),
        fixed_inventory_sha256=str(fixed["run_inventory_sha256"]),
        arrival_run_id=str(arrival["run_id"]),
        arrival_inventory_sha256=str(arrival["run_inventory_sha256"]),
        arrival_spec_sha256=str(arrival["spec_sha256"]),
        data_manifest_sha256=str(fixed["data_manifest_sha256"]),
        data_cutoff=cutoff,
        evaluation_starts=starts,
        markets=MARKETS,
        betas=betas,
        event_betas=event_betas,
        lambdas=lambdas,
        event_lambdas=event_lambdas,
        rules=rules,
        fit_window=int(controls["fit_window_observations"]),
        horizon=int(events["horizon_signal_days"]),
        numerical_tolerance=float(decision["numerical_tolerance"]),
        fixed_allowed_files=fixed_files,
        arrival_allowed_files=arrival_files,
        performance_files_forbidden=forbidden,
        artifact_subdir=artifact_subdir,
    )


def beta_label(beta: float) -> str:
    if beta == 0.0:
        return "0"
    if beta == math.log(2.0):
        return "log2"
    if beta == math.log(4.0):
        return "log4"
    raise LaggedStudyError(f"unexpected beta: {beta}")


def summarize_mechanism(
    events: pd.DataFrame,
    behavior: pd.DataFrame,
    spec: LaggedMechanismSpec,
) -> pd.DataFrame:
    """Summarize own-rule event labels and complete candidate-path behavior."""
    required_events = {
        "market",
        "rule",
        "beta_label",
        "whipsaw_20",
        "persistent_20",
        "confirmed_early",
    }
    required_behavior = {
        "market",
        "rule",
        "beta_label",
        "lambda0",
        "switch_count",
        "state_differences_vs_fixed",
    }
    if not required_events.issubset(events) or not required_behavior.issubset(behavior):
        raise LaggedStudyError("mechanism evidence tables are incomplete")
    coverage_columns = ["market", "rule", "beta_label", "lambda0"]
    expected_coverage = {
        (market, rule, beta_label(beta), float(lambda0))
        for market in spec.markets
        for rule in spec.rules
        for beta in spec.event_betas
        for lambda0 in spec.event_lambdas
    }
    observed_coverage = {
        (market, rule, label, float(lambda0))
        for market, rule, label, lambda0 in behavior.loc[
            :, coverage_columns
        ].itertuples(index=False, name=None)
    }
    if (
        len(behavior) != len(expected_coverage)
        or observed_coverage != expected_coverage
    ):
        raise LaggedStudyError("mechanism path coverage changed")
    records: list[dict[str, Any]] = []
    for market in spec.markets:
        for beta in spec.event_betas:
            label = beta_label(beta)
            for rule in spec.rules:
                event_rows = events.loc[
                    (events["market"] == market)
                    & (events["beta_label"] == label)
                    & (events["rule"] == rule)
                ]
                path_rows = behavior.loc[
                    (behavior["market"] == market)
                    & (behavior["beta_label"] == label)
                    & (behavior["rule"] == rule)
                ]
                records.append(
                    {
                        "market": market,
                        "beta": beta,
                        "beta_label": label,
                        "rule": rule,
                        "event_count": len(event_rows),
                        "whipsaw_count": int(
                            event_rows["whipsaw_20"].astype(bool).sum()
                        ),
                        "persistent_count": int(
                            event_rows["persistent_20"].astype(bool).sum()
                        ),
                        "confirmed_early_count": int(
                            event_rows["confirmed_early"].astype(bool).sum()
                        ),
                        "switch_count": int(path_rows["switch_count"].sum()),
                        "state_differences_vs_fixed": int(
                            path_rows["state_differences_vs_fixed"].sum()
                        ),
                    }
                )
    return pd.DataFrame.from_records(records)


def classify_mechanism(
    summary: pd.DataFrame,
    spec: LaggedMechanismSpec,
    *,
    mechanical_prerequisites_passed: bool,
) -> dict[str, Any]:
    """Apply the frozen mechanism continuation gate for each beta."""
    required = {
        "market",
        "beta_label",
        "rule",
        "whipsaw_count",
        "confirmed_early_count",
        "switch_count",
        "state_differences_vs_fixed",
    }
    if not required.issubset(summary):
        raise LaggedStudyError("mechanism summary is incomplete")
    if set(summary["market"]) != set(spec.markets) or set(summary["rule"]) != set(
        spec.rules
    ):
        raise LaggedStudyError("mechanism summary scope changed")

    by_beta: dict[str, dict[str, Any]] = {}
    advancing: list[str] = []
    for beta in spec.event_betas:
        label = beta_label(beta)
        rows = summary.loc[summary["beta_label"] == label]
        comparisons: dict[str, bool] = {}
        for market in spec.markets:
            market_rows = rows.loc[rows["market"] == market].set_index("rule")
            if set(market_rows.index) != set(spec.rules):
                raise LaggedStudyError(f"{market}/{label}: rule summary missing")
            comparisons[market] = bool(
                int(market_rows.loc["lagged", "whipsaw_count"])
                <= int(market_rows.loc["arrival", "whipsaw_count"])
            )
        arrival = rows.loc[rows["rule"] == "arrival"]
        lagged = rows.loc[rows["rule"] == "lagged"]
        pooled_arrival_whipsaw = int(arrival["whipsaw_count"].sum())
        pooled_lagged_whipsaw = int(lagged["whipsaw_count"].sum())
        jp_arrival_switches = int(
            arrival.loc[arrival["market"] == "jp", "switch_count"].sum()
        )
        jp_lagged_switches = int(
            lagged.loc[lagged["market"] == "jp", "switch_count"].sum()
        )
        confirmed_early = int(lagged["confirmed_early_count"].sum())
        state_differences = int(lagged["state_differences_vs_fixed"].sum())
        conditions = {
            "mechanical_prerequisites": bool(mechanical_prerequisites_passed),
            "nontrivial": state_differences > 0,
            "market_whipsaw_nonincrease": all(comparisons.values()),
            "pooled_whipsaw_strict_reduction": (
                pooled_lagged_whipsaw < pooled_arrival_whipsaw
            ),
            "jp_switch_strict_reduction": jp_lagged_switches < jp_arrival_switches,
            "latency_retained": confirmed_early >= 1,
        }
        advances = all(conditions.values())
        if advances:
            advancing.append(label)
        by_beta[label] = {
            "advances": advances,
            "conditions": conditions,
            "market_whipsaw_nonincrease": comparisons,
            "arrival_whipsaw_count": pooled_arrival_whipsaw,
            "lagged_whipsaw_count": pooled_lagged_whipsaw,
            "jp_arrival_switch_count": jp_arrival_switches,
            "jp_lagged_switch_count": jp_lagged_switches,
            "lagged_confirmed_early_count": confirmed_early,
            "lagged_state_differences_vs_fixed": state_differences,
        }

    selected = "log2" if "log2" in advancing else (advancing[0] if advancing else None)
    return {
        "experiment_id": spec.experiment_id,
        "claim_class": "EXPLORATORY",
        "performance_claim_allowed": False,
        "result": "supported" if selected is not None else "not_supported",
        "selected_beta_label": selected,
        "advancing_beta_labels": advancing,
        "by_beta": by_beta,
        "interpretation": (
            "Performance-free development-sample mechanism evidence only; "
            "support authorizes only a separately frozen P&L study."
        ),
    }
