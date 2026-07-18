"""Independent artifact verifier for the pair-balanced lagged study."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from adaptive_jump.artifacts import read_json, verify_inventory
from adaptive_jump.balanced_decision_replay import classify, dated_audit, summarize
from adaptive_jump.balanced_event_replay import extract_events, matched_response
from adaptive_jump.balanced_mechanics import mechanical_prerequisites
from adaptive_jump.balanced_model import (
    BalancedSpec,
    BalancedStudyError,
    beta_label,
    load_balanced_spec,
    load_market_inputs,
)
from adaptive_jump.balanced_replay import (
    candidate_checks,
    independent_candidates,
    path_behavior,
    penalty_summary,
)
from adaptive_jump.balanced_smoke_replay import run_independent_smoke
from adaptive_jump.balanced_sources import (
    SourcePaths,
    implementation_lock,
    verify_source_inputs,
)
from adaptive_jump.config import ResearchConfig, load_config
from adaptive_jump.lagged_model import LockedStateEvidence

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_CONFIG = Path("research.toml")
CANONICAL_SPEC = Path("research/balanced-lagged-mechanism-001.toml")
DATE_COLUMNS = ("start", "end", "signal_date", "evidence_date", "fit_date")
ROOT_FILES = {
    "study.lock.toml",
    "config.lock.toml",
    "source-lock.json",
    "implementation-lock.json",
    "smoke.json",
    "run.json",
    "path-behavior.csv",
    "discount-events.csv",
    "matched-lagged-anchors.csv",
    "penalty-summary.csv",
    "mechanism-summary.csv",
    "dated-audit.csv",
    "mechanical-checks.json",
    "conclusion.json",
    "inventory.json",
}
SMOKE_EXPECTED = {
    "parent_lagged_exact": True,
    "beta_zero_exact": True,
    "short_long_prefix_exact": True,
    "future_mutation_prefix_invariant": True,
    "prefix_invariant": True,
    "future_mutation_effect_present": True,
    "actual_formula_exact": True,
    "actual_bounds_exact": True,
    "formula_through_second_refit": True,
    "pair_balance_exact": True,
    "balanced_discounts_present": True,
    "balanced_surcharges_present": True,
    "refit_convention_numeric": True,
    "performance_files_accessed": False,
    "return_columns_accessed": False,
    "post_2023_accessed": False,
}


def _smoke_coverage_exact(smoke: dict[str, Any], spec: BalancedSpec) -> bool:
    integer_fields = (
        "terminal_dates",
        "generated_terminal_dates",
        "parent_lagged_state_cells_checked",
        "beta_zero_state_cells_checked",
        "short_long_prefix_state_cells_checked",
        "future_mutation_prefix_state_cells_checked",
        "actual_formula_terminal_dates_checked",
        "actual_formula_lambda_values_checked",
        "actual_formula_directed_cells_checked",
        "refit_convention_lambdas_checked",
        "refit_convention_informative_lambdas",
        "refit_convention_distinct_lambdas",
    )
    if any(type(smoke.get(key)) is not int for key in integer_fields):
        return False
    prefix = smoke["terminal_dates"]
    generated = smoke["generated_terminal_dates"]
    formula_dates = smoke["actual_formula_terminal_dates_checked"]
    lambda_count = len(spec.lambdas)
    prefix_cells = prefix * lambda_count
    generated_cells = generated * lambda_count
    minimum_stale = smoke.get("refit_convention_min_stale_distance")
    maximum_stale = smoke.get("refit_convention_max_stale_distance")
    if type(minimum_stale) is not float or type(maximum_stale) is not float:
        return False
    return bool(
        prefix == 20
        and generated == max(prefix, formula_dates)
        and smoke["parent_lagged_state_cells_checked"]
        == generated_cells * len(spec.betas)
        and smoke["beta_zero_state_cells_checked"] == generated_cells * len(spec.rules)
        and smoke["short_long_prefix_state_cells_checked"]
        == prefix_cells * len(spec.rules) * len(spec.betas)
        and smoke["future_mutation_prefix_state_cells_checked"]
        == prefix_cells * len(spec.rules) * len(spec.betas)
        and formula_dates >= 2
        and smoke["actual_formula_lambda_values_checked"] == lambda_count
        and smoke["actual_formula_directed_cells_checked"]
        == formula_dates * lambda_count * 2
        and smoke["refit_convention_lambdas_checked"] == len(spec.event_lambdas)
        and 1
        <= smoke["refit_convention_informative_lambdas"]
        <= len(spec.event_lambdas)
        and smoke["refit_convention_distinct_lambdas"]
        == smoke["refit_convention_informative_lambdas"]
        and minimum_stale > spec.numerical_tolerance
        and maximum_stale >= minimum_stale
    )


@dataclass(frozen=True)
class MarketReplay:
    market: str
    evidence: dict[str, LockedStateEvidence]
    behavior: pd.DataFrame
    events: pd.DataFrame
    anchors: pd.DataFrame
    penalties: pd.DataFrame
    audit: dict[str, Any]
    checks: dict[str, Any]


def _json_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _json_equal(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _json_equal(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return bool(actual == expected)


def _assert_json_exact(actual: Any, expected: Any, label: str) -> None:
    if not _json_equal(actual, expected):
        raise BalancedStudyError(f"{label} changed")


def _assert_frame_exact(
    actual: pd.DataFrame, expected: pd.DataFrame, label: str
) -> None:
    try:
        pd.testing.assert_frame_equal(
            actual,
            expected,
            check_exact=True,
            check_dtype=False,
            check_freq=False,
        )
    except AssertionError as exc:
        raise BalancedStudyError(f"{label} changed") from exc


def _read_like(path: Path, expected: pd.DataFrame) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, float_precision="round_trip")
    except (OSError, ValueError) as exc:
        raise BalancedStudyError(f"cannot read balanced artifact: {path}") from exc
    if tuple(frame.columns) != tuple(expected.columns):
        raise BalancedStudyError(f"{path.name}: columns changed")
    for column in DATE_COLUMNS:
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="raise")
    return frame


def _read_state(path: Path, spec: BalancedSpec) -> pd.DataFrame:
    columns = ("date", *(str(value) for value in spec.lambdas))
    try:
        frame = pd.read_csv(path, float_precision="round_trip")
    except (OSError, ValueError) as exc:
        raise BalancedStudyError(f"cannot read balanced state table: {path}") from exc
    if tuple(frame.columns) != columns:
        raise BalancedStudyError(f"{path.name}: state columns changed")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.set_index("date")
    frame.index = pd.DatetimeIndex(frame.index, name="date")
    frame.columns = spec.lambdas
    return frame


def expected_files(spec: BalancedSpec) -> set[str]:
    files = set(ROOT_FILES)
    for market in spec.markets:
        files.update(
            {
                f"{market}/audit.json",
                f"{market}/path-behavior.csv",
                f"{market}/discount-events.csv",
                f"{market}/matched-lagged-anchors.csv",
                f"{market}/penalty-summary.csv",
                *(
                    f"{market}/candidate-states-{rule}-beta-{beta_label(beta)}.csv"
                    for rule in spec.rules
                    for beta in spec.betas
                ),
            }
        )
    return files


def _canonical_context() -> tuple[ResearchConfig, BalancedSpec]:
    config = load_config(REPOSITORY_ROOT / CANONICAL_CONFIG)
    spec = load_balanced_spec(REPOSITORY_ROOT / CANONICAL_SPEC, config)
    return config, spec


def _replay_market(
    market: str,
    sources: SourcePaths,
    config: ResearchConfig,
    spec: BalancedSpec,
) -> MarketReplay:
    inputs, fixed = load_market_inputs(
        market, sources.fixed_markets[market], sources.parent_markets[market], spec
    )
    evidence = independent_candidates(inputs, fixed, config, spec)
    checks = candidate_checks(inputs, fixed, evidence, spec)
    behavior = path_behavior(inputs, evidence, spec)
    events, own_audit = extract_events(inputs, evidence, spec)
    anchors, matched_audit = matched_response(
        events,
        evidence["balanced"].states[spec.decision_beta],
        inputs.candidates[0.0],
        inputs.refits,
        spec,
    )
    audit = {"own_events": own_audit, "matched_anchors": matched_audit}
    penalties = penalty_summary(market, evidence, spec)
    for rule in spec.rules:
        for beta in spec.betas:
            dates = evidence[rule].states[beta].dropna(how="all").index
            if len(dates) and dates.max().date() > spec.data_cutoff:
                raise BalancedStudyError(f"{market}: post-cutoff candidate state")
    for frame in (behavior, events, anchors):
        for column in DATE_COLUMNS:
            if column in frame and len(frame):
                maximum = pd.to_datetime(frame[column], errors="raise").max()
                if maximum.date() > spec.data_cutoff:
                    raise BalancedStudyError(f"{market}: post-cutoff {column}")
    return MarketReplay(
        market=market,
        evidence=evidence,
        behavior=behavior,
        events=events,
        anchors=anchors,
        penalties=penalties,
        audit=audit,
        checks=checks,
    )


def _verify_market(path: Path, replay: MarketReplay, spec: BalancedSpec) -> None:
    market = replay.market
    for rule in spec.rules:
        for beta in spec.betas:
            stored = _read_state(
                path / market / f"candidate-states-{rule}-beta-{beta_label(beta)}.csv",
                spec,
            )
            _assert_frame_exact(
                stored,
                replay.evidence[rule].states[beta],
                f"{market}/{rule}/{beta_label(beta)} states",
            )
    for filename, expected in (
        ("path-behavior.csv", replay.behavior),
        ("discount-events.csv", replay.events),
        ("matched-lagged-anchors.csv", replay.anchors),
        ("penalty-summary.csv", replay.penalties),
    ):
        stored = _read_like(path / market / filename, expected)
        _assert_frame_exact(stored, expected, f"{market}/{filename}")
    _assert_json_exact(
        read_json(path / market / "audit.json"), replay.audit, f"{market} audit"
    )


def _mechanical_checks(
    prerequisites: dict[str, Any],
    smoke: dict[str, Any],
    market_replays: dict[str, dict[str, Any]],
    spec: BalancedSpec,
) -> dict[str, Any]:
    smoke_checks = {key: smoke.get(key) for key in SMOKE_EXPECTED}
    smoke_passed = all(
        smoke_checks[key] is expected for key, expected in SMOKE_EXPECTED.items()
    ) and _smoke_coverage_exact(smoke, spec)
    replays_passed = set(market_replays) == set(spec.markets) and all(
        replay.get("parent_lagged_exact") is True
        and replay.get("beta_zero_exact") is True
        and replay.get("candidate_coverage_exact") is True
        and replay.get("pair_balance_exact") is True
        and replay.get("return_columns_accessed") is False
        for replay in market_replays.values()
    )
    return {
        "schema_version": 1,
        "mechanical_prerequisites": prerequisites,
        "smoke_checks": smoke_checks,
        "market_replays": market_replays,
        "passed": bool(
            prerequisites.get("passed") is True and smoke_passed and replays_passed
        ),
    }


def _verify_metadata(
    metadata: dict[str, Any],
    path: Path,
    config: ResearchConfig,
    spec: BalancedSpec,
    implementation: dict[str, Any],
    conclusion: dict[str, Any],
    counts: dict[str, int],
    mechanical_passed: bool,
) -> tuple[str, str]:
    implementation_sha = implementation["implementation_sha256"]
    run_id = (
        f"balanced-lagged-{spec.sha256[:12]}-"
        f"{spec.parent_inventory_sha256[:12]}-{implementation_sha[:12]}"
    )
    status = metadata.get("status")
    if status not in {"verifying", "complete"}:
        raise BalancedStudyError("balanced run is not in a verifiable state")
    expected = {
        "schema_version": 1,
        "study_kind": "balanced_lagged_mechanism",
        "experiment_id": spec.experiment_id,
        "run_id": run_id,
        "status": status,
        "claim_class": "EXPLORATORY",
        "performance_claim_allowed": False,
        "paper_replication_claim_allowed": False,
        "new_model_claim_allowed": False,
        "performance_files_accessed": False,
        "return_columns_accessed": False,
        "post_2023_accessed": False,
        "provider_accessed": False,
        "monthly_selection_performed": False,
        "spec_sha256": spec.sha256,
        "config_sha256": config.sha256,
        "fixed_inventory_sha256": spec.fixed_inventory_sha256,
        "parent_inventory_sha256": spec.parent_inventory_sha256,
        "parent_spec_sha256": spec.parent_spec_sha256,
        "data_manifest_sha256": spec.data_manifest_sha256,
        "implementation_sha256": implementation_sha,
        "git_head": implementation["git_head"],
        "result": conclusion["result"],
        "decision_beta_label": conclusion["decision_beta_label"],
        "mechanical_prerequisites_passed": mechanical_passed,
        **counts,
    }
    if path.name != run_id or any(
        not _json_equal(metadata.get(key), value) for key, value in expected.items()
    ):
        raise BalancedStudyError("balanced run metadata changed")
    common_time = metadata.get("created_at_utc")
    verification_time = metadata.get("verification_started_at_utc")
    if not isinstance(common_time, str) or not isinstance(verification_time, str):
        raise BalancedStudyError("balanced lifecycle timestamps changed")
    if status == "verifying":
        if "finished_at_utc" in metadata:
            raise BalancedStudyError("verifying run already has a finish timestamp")
        allowed = {*expected, "created_at_utc", "verification_started_at_utc"}
    else:
        if not isinstance(metadata.get("finished_at_utc"), str):
            raise BalancedStudyError("complete run lacks a finish timestamp")
        allowed = {
            *expected,
            "created_at_utc",
            "verification_started_at_utc",
            "finished_at_utc",
        }
    if set(metadata) != allowed:
        raise BalancedStudyError("balanced metadata fields changed")
    return run_id, status


def verify_balanced_run(run: str | Path) -> dict[str, Any]:
    """Replay formula, candidates, events, matched responses, and decision."""
    raw = Path(run)
    if raw.is_symlink():
        raise BalancedStudyError("balanced run may not be a symlink")
    path = raw.resolve()
    if not path.is_dir() or any(item.is_symlink() for item in path.rglob("*")):
        raise BalancedStudyError("balanced artifact tree is invalid")
    config, spec = _canonical_context()
    actual = {str(item.relative_to(path)) for item in path.rglob("*") if item.is_file()}
    if actual != expected_files(spec):
        raise BalancedStudyError("balanced artifact allowlist changed")
    verify_inventory(path)
    locked_config = load_config(path / "config.lock.toml")
    locked_spec = load_balanced_spec(path / "study.lock.toml", locked_config)
    if locked_config.sha256 != config.sha256 or locked_spec.sha256 != spec.sha256:
        raise BalancedStudyError("balanced canonical locks changed")
    sources = verify_source_inputs(REPOSITORY_ROOT, config, spec)
    _assert_json_exact(
        read_json(path / "source-lock.json"), sources.source_lock, "source lock"
    )
    implementation = implementation_lock(REPOSITORY_ROOT, spec)
    _assert_json_exact(
        read_json(path / "implementation-lock.json"),
        implementation,
        "implementation lock",
    )
    smoke = run_independent_smoke(config, spec, sources)
    _assert_json_exact(read_json(path / "smoke.json"), smoke, "independent smoke")
    prerequisites = mechanical_prerequisites(spec)
    if smoke.get("mechanical_prerequisites") != prerequisites:
        raise BalancedStudyError("smoke mechanical prerequisites changed")
    replays: dict[str, MarketReplay] = {}
    for market in spec.markets:
        replay = _replay_market(market, sources, config, spec)
        _verify_market(path, replay, spec)
        replays[market] = replay
    behavior = pd.concat(
        [replays[market].behavior for market in spec.markets], ignore_index=True
    )
    events = pd.concat(
        [replays[market].events for market in spec.markets], ignore_index=True
    )
    anchors = pd.concat(
        [replays[market].anchors for market in spec.markets], ignore_index=True
    )
    penalties = pd.concat(
        [replays[market].penalties for market in spec.markets], ignore_index=True
    )
    summary = summarize(events, behavior, spec)
    dated = dated_audit(events)
    for filename, expected in (
        ("path-behavior.csv", behavior),
        ("discount-events.csv", events),
        ("matched-lagged-anchors.csv", anchors),
        ("penalty-summary.csv", penalties),
        ("mechanism-summary.csv", summary),
        ("dated-audit.csv", dated),
    ):
        _assert_frame_exact(
            _read_like(path / filename, expected), expected, f"root {filename}"
        )
    mechanics = _mechanical_checks(
        prerequisites,
        smoke,
        {market: replays[market].checks for market in spec.markets},
        spec,
    )
    _assert_json_exact(
        read_json(path / "mechanical-checks.json"), mechanics, "mechanical checks"
    )
    conclusion = classify(summary, anchors, spec, mechanical_passed=mechanics["passed"])
    _assert_json_exact(
        read_json(path / "conclusion.json"), conclusion, "balanced conclusion"
    )
    counts = {
        "path_rows": len(behavior),
        "event_rows": len(events),
        "matched_anchor_rows": len(anchors),
        "penalty_rows": len(penalties),
        "summary_rows": len(summary),
        "dated_audit_rows": len(dated),
    }
    run_id, lifecycle = _verify_metadata(
        read_json(path / "run.json"),
        path,
        config,
        spec,
        implementation,
        conclusion,
        counts,
        mechanics["passed"],
    )
    return {
        "status": "verified",
        "lifecycle": lifecycle,
        "run_id": run_id,
        "result": conclusion["result"],
        "markets_reconstructed": len(replays),
        "candidate_paths_reconstructed": len(spec.markets)
        * len(spec.rules)
        * len(spec.betas)
        * len(spec.lambdas),
        **counts,
        "mechanical_prerequisites_passed": mechanics["passed"],
    }
