"""Independent artifact verifier for the lagged-evidence mechanism study."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from adaptive_jump.artifacts import read_json, verify_inventory
from adaptive_jump.config import load_config
from adaptive_jump.lagged_analysis import (
    MechanismAnalysis,
    _input_spec,
    analyze_market_mechanism,
)
from adaptive_jump.lagged_mechanics import mechanical_prerequisites, run_locked_smoke
from adaptive_jump.lagged_model import LockedStateEvidence, generate_locked_candidates
from adaptive_jump.lagged_sources import implementation_lock, verify_source_inputs
from adaptive_jump.lagged_study import (
    LaggedMechanismSpec,
    LaggedStudyError,
    beta_label,
    classify_mechanism,
    load_lagged_spec,
    summarize_mechanism,
)
from adaptive_jump.separation_analysis import load_market_inputs
from adaptive_jump.tv_jump import evidence_penalty_seq, lagged_evidence_penalty_seq

BUILDERS = {
    "arrival": evidence_penalty_seq,
    "lagged": lagged_evidence_penalty_seq,
}
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_CONFIG = Path("research.toml")
CANONICAL_SPEC = Path("research/lagged-evidence-mechanism-001.toml")
EVENT_DATE_COLUMNS = ("signal_date", "evidence_date", "fit_date")
REFIT_DATE_COLUMNS = ("fit_date", "training_start", "training_end")


def _assert_json_exact(actual: Any, expected: Any, *, label: str) -> None:
    if actual != expected:
        raise LaggedStudyError(f"{label} changed")


def _assert_frame_exact(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    label: str,
) -> None:
    """Compare scientific values exactly while ignoring CSV-inferred dtypes."""
    try:
        pd.testing.assert_frame_equal(
            actual,
            expected,
            check_exact=True,
            check_dtype=False,
            check_freq=False,
        )
    except AssertionError as exc:
        raise LaggedStudyError(f"{label} changed") from exc


def _read_csv(
    path: Path,
    *,
    date_columns: tuple[str, ...] = (),
    expected_columns: tuple[Any, ...] | None = None,
) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, float_precision="round_trip")
    except (OSError, ValueError) as exc:
        raise LaggedStudyError(f"cannot read lagged evidence table: {path}") from exc
    if expected_columns is not None and tuple(frame.columns) != expected_columns:
        raise LaggedStudyError(f"{path.name}: columns changed")
    for column in date_columns:
        if column not in frame:
            raise LaggedStudyError(f"{path.name}: date column {column} is missing")
        frame[column] = pd.to_datetime(frame[column], errors="raise")
    return frame


def _read_state_table(
    path: Path,
    spec: LaggedMechanismSpec,
) -> pd.DataFrame:
    columns = ("date", *(str(value) for value in spec.lambdas))
    frame = _read_csv(path, expected_columns=columns)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.set_index("date")
    frame.index = pd.DatetimeIndex(frame.index, name="date")
    frame.columns = spec.lambdas
    return frame


def _normalise_refits(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in REFIT_DATE_COLUMNS:
        if column not in result:
            raise LaggedStudyError(f"refit date column {column} is missing")
        result[column] = pd.to_datetime(result[column], errors="raise")
    for column in ("lambda0", "q_train"):
        if column not in result:
            raise LaggedStudyError(f"refit numeric column {column} is missing")
        result[column] = pd.to_numeric(result[column], errors="raise")
    return result.reset_index(drop=True)


def _read_like(path: Path, expected: pd.DataFrame) -> pd.DataFrame:
    date_columns = tuple(
        column
        for column in (*REFIT_DATE_COLUMNS, "start", "end", *EVENT_DATE_COLUMNS)
        if column in expected.columns
    )
    stored = _read_csv(
        path,
        date_columns=date_columns,
        expected_columns=tuple(expected.columns),
    )
    if "lambda0" in stored:
        stored["lambda0"] = pd.to_numeric(stored["lambda0"], errors="raise")
    if "fit_date" in stored and "training_start" in stored:
        stored = _normalise_refits(stored)
    return stored


def _verify_market_artifacts(
    market_dir: Path,
    lagged: LockedStateEvidence,
    analysis: MechanismAnalysis,
    spec: LaggedMechanismSpec,
) -> None:
    """Require every stored market output to equal its reconstruction."""
    for beta in spec.betas:
        stored = _read_state_table(
            market_dir / f"candidate-states-beta-{beta_label(beta)}.csv",
            spec,
        )
        _assert_frame_exact(
            stored,
            lagged.states[beta],
            label=f"{analysis.market}/{beta_label(beta)} lagged candidate states",
        )

    stored_refits = _read_like(
        market_dir / "refits-and-scales.csv",
        lagged.refits,
    )
    _assert_frame_exact(
        stored_refits,
        _normalise_refits(lagged.refits),
        label=f"{analysis.market} sealed refit copy",
    )
    stored_behavior = _read_like(
        market_dir / "path-behavior.csv",
        analysis.behavior,
    )
    stored_events = _read_like(
        market_dir / "discount-events.csv",
        analysis.events,
    )
    _assert_frame_exact(
        stored_behavior,
        analysis.behavior,
        label=f"{analysis.market} path behavior",
    )
    _assert_frame_exact(
        stored_events,
        analysis.events,
        label=f"{analysis.market} discount events",
    )
    _assert_json_exact(
        read_json(market_dir / "audit.json"),
        analysis.audit,
        label=f"{analysis.market} audit",
    )


def _dated_audit(events: pd.DataFrame) -> pd.DataFrame:
    return (
        events.sort_values(["market", "rule", "beta_label", "signal_date"])
        .groupby(["market", "rule", "beta_label"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def _verify_root_artifacts(
    run_dir: Path,
    behavior: pd.DataFrame,
    events: pd.DataFrame,
    spec: LaggedMechanismSpec,
) -> pd.DataFrame:
    """Verify exact root concatenation, summary, and dated evidence."""
    stored_behavior = _read_like(run_dir / "path-behavior.csv", behavior)
    stored_events = _read_like(run_dir / "discount-events.csv", events)
    _assert_frame_exact(stored_behavior, behavior, label="root path behavior")
    _assert_frame_exact(stored_events, events, label="root discount events")

    summary = summarize_mechanism(events, behavior, spec)
    stored_summary = _read_like(run_dir / "mechanism-summary.csv", summary)
    _assert_frame_exact(stored_summary, summary, label="mechanism summary")

    dated = _dated_audit(events)
    stored_dated = _read_like(run_dir / "dated-audit.csv", dated)
    _assert_frame_exact(stored_dated, dated, label="dated audit")
    return summary


def _verify_metadata(
    metadata: dict[str, Any],
    *,
    run_dir: Path,
    spec: LaggedMechanismSpec,
    config_sha256: str,
    implementation: dict[str, Any],
    conclusion: dict[str, Any],
    event_count: int,
    mechanical_prerequisites_passed: bool,
) -> str:
    implementation_sha = implementation["implementation_sha256"]
    expected_run_id = (
        f"lagged-evidence-{spec.sha256[:12]}-"
        f"{spec.arrival_inventory_sha256[:12]}-{implementation_sha[:12]}"
    )
    expected = {
        "schema_version": 1,
        "study_kind": "lagged_evidence_mechanism",
        "experiment_id": spec.experiment_id,
        "run_id": expected_run_id,
        "status": "complete",
        "claim_class": "EXPLORATORY",
        "performance_files_accessed": False,
        "post_2023_accessed": False,
        "return_columns_accessed": False,
        "spec_sha256": spec.sha256,
        "config_sha256": config_sha256,
        "fixed_inventory_sha256": spec.fixed_inventory_sha256,
        "arrival_inventory_sha256": spec.arrival_inventory_sha256,
        "data_manifest_sha256": spec.data_manifest_sha256,
        "implementation_sha256": implementation_sha,
        "git_head": implementation["git_head"],
        "result": conclusion["result"],
        "selected_beta_label": conclusion["selected_beta_label"],
        "events": event_count,
        "mechanical_prerequisites_passed": mechanical_prerequisites_passed,
    }

    def matches(key: str, value: Any) -> bool:
        actual = metadata.get(key)
        if type(value) is bool:
            return actual is value
        return actual == value

    if run_dir.name != expected_run_id or any(
        not matches(key, value) for key, value in expected.items()
    ):
        raise LaggedStudyError("lagged run metadata changed")
    return expected_run_id


def _combined_mechanical_prerequisite(
    mechanics: dict[str, Any],
    smoke: dict[str, Any],
) -> bool:
    checks = mechanics.get("checks")
    derived = (
        isinstance(checks, dict)
        and bool(checks)
        and all(value is True for value in checks.values())
    )
    if mechanics.get("passed") is not derived:
        raise LaggedStudyError("mechanical prerequisite aggregation changed")
    _assert_json_exact(
        smoke.get("mechanical_prerequisites"),
        mechanics,
        label="stored mechanical prerequisites",
    )
    real_checks = (
        smoke.get("status") == "passed",
        smoke.get("beta_zero_exact") is True,
        smoke.get("prefix_invariant") is True,
        smoke.get("sealed_arrival_exact") is True,
        smoke.get("refit_convention_numeric") is True,
        smoke.get("future_mutation_effect_present") is True,
        smoke.get("lagged_discounts_present") is True,
        smoke.get("refit_convention")
        == "current-fit parameters applied to previous-row loss",
        smoke.get("performance_files_accessed") is False,
        smoke.get("return_columns_accessed") is False,
        smoke.get("post_2023_accessed") is False,
    )
    return bool(derived and all(real_checks))


def _mechanical_checks(
    mechanics: dict[str, Any],
    smoke: dict[str, Any],
    market_replays: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    smoke_keys = (
        "sealed_arrival_exact",
        "beta_zero_exact",
        "prefix_invariant",
        "refit_convention_numeric",
        "future_mutation_effect_present",
        "lagged_discounts_present",
        "performance_files_accessed",
        "return_columns_accessed",
        "post_2023_accessed",
    )
    smoke_checks = {key: smoke.get(key) for key in smoke_keys}
    smoke_passed = _combined_mechanical_prerequisite(mechanics, smoke)
    replays_passed = bool(market_replays) and all(
        replay.get("sealed_arrival_exact") is True
        and replay.get("beta_zero_exact") is True
        and replay.get("return_columns_accessed") is False
        for replay in market_replays.values()
    )
    return {
        "schema_version": 1,
        "mechanical_prerequisites": mechanics,
        "smoke_checks": smoke_checks,
        "market_replays": market_replays,
        "passed": bool(smoke_passed and replays_passed),
    }


def verify_lagged_run(run: str | Path) -> dict[str, Any]:
    """Reconstruct locked paths, event analysis, and the frozen decision."""
    run_dir = Path(run).resolve()
    verify_inventory(run_dir)
    metadata = read_json(run_dir / "run.json")

    canonical_config = load_config(REPOSITORY_ROOT / CANONICAL_CONFIG)
    locked_config = load_config(run_dir / "config.lock.toml")
    if canonical_config.sha256 != locked_config.sha256:
        raise LaggedStudyError("locked config differs from the canonical config")
    canonical_spec = load_lagged_spec(
        REPOSITORY_ROOT / CANONICAL_SPEC,
        canonical_config,
    )
    locked_spec = load_lagged_spec(run_dir / "study.lock.toml", locked_config)
    if canonical_spec.sha256 != locked_spec.sha256:
        raise LaggedStudyError("locked study differs from the canonical study")
    spec = canonical_spec

    sources = verify_source_inputs(REPOSITORY_ROOT, canonical_config, spec)
    _assert_json_exact(
        read_json(run_dir / "source-lock.json"),
        sources.source_lock,
        label="source lock",
    )
    implementation = implementation_lock(REPOSITORY_ROOT, spec)
    _assert_json_exact(
        read_json(run_dir / "implementation-lock.json"),
        implementation,
        label="implementation lock",
    )

    mechanics = mechanical_prerequisites(
        BUILDERS,
        atol=spec.numerical_tolerance,
    )
    stored_smoke = read_json(run_dir / "smoke.json")
    smoke: dict[str, Any] | None = None

    results: dict[str, MechanismAnalysis] = {}
    market_replays: dict[str, dict[str, Any]] = {}
    for market in spec.markets:
        try:
            inputs = load_market_inputs(
                market,
                sources.fixed_markets[market] / "features.csv",
                sources.arrival_markets[market],
                _input_spec(spec),
                include_fixed_objective=False,
            )
            fixed = _read_state_table(
                sources.fixed_markets[market] / "jm-states.csv",
                spec,
            )
            if market == "us":
                smoke = run_locked_smoke(
                    inputs, fixed, canonical_config, spec, BUILDERS
                )
                _assert_json_exact(stored_smoke, smoke, label="US smoke")

            generated = generate_locked_candidates(
                inputs.features.reset_index(),
                fixed,
                inputs.refits,
                canonical_config,
                spec,
                market=market,
                penalty_builders=BUILDERS,
            )
            for beta in spec.betas:
                _assert_frame_exact(
                    generated["arrival"].states[beta],
                    inputs.candidates[beta],
                    label=f"{market}/{beta_label(beta)} sealed arrival states",
                )
            for rule in spec.rules:
                _assert_frame_exact(
                    generated[rule].states[0.0],
                    fixed,
                    label=f"{market}/{rule} beta-zero states",
                )
            arrival_cells = sum(
                int(generated["arrival"].states[beta].notna().sum().sum())
                for beta in spec.betas
            )
            beta_cells = sum(
                int(generated[rule].states[0.0].notna().sum().sum())
                for rule in spec.rules
            )
            terminal_rows = int(fixed.notna().all(axis=1).sum())
            market_replays[market] = {
                "sealed_arrival_exact": arrival_cells
                == terminal_rows * len(spec.lambdas) * len(spec.betas),
                "beta_zero_exact": beta_cells
                == terminal_rows * len(spec.lambdas) * len(spec.rules),
                "sealed_arrival_state_cells_checked": arrival_cells,
                "beta_zero_state_cells_checked": beta_cells,
                "return_columns_accessed": False,
            }
            result = analyze_market_mechanism(
                market,
                sources.fixed_markets[market] / "features.csv",
                sources.arrival_markets[market],
                generated["lagged"].states,
                spec,
            )
            _verify_market_artifacts(
                run_dir / market,
                generated["lagged"],
                result,
                spec,
            )
            results[market] = result
        except LaggedStudyError:
            raise
        except Exception as exc:
            raise LaggedStudyError(f"{market}: locked reconstruction failed") from exc

    if smoke is None:
        raise LaggedStudyError("US smoke was not independently reconstructed")
    _combined_mechanical_prerequisite(mechanics, smoke)
    expected_mechanical = _mechanical_checks(mechanics, smoke, market_replays)
    _assert_json_exact(
        read_json(run_dir / "mechanical-checks.json"),
        expected_mechanical,
        label="mechanical checks",
    )
    mechanical_passed = expected_mechanical["passed"]
    behavior = pd.concat(
        [results[market].behavior for market in spec.markets],
        ignore_index=True,
    )
    events = pd.concat(
        [results[market].events for market in spec.markets],
        ignore_index=True,
    )
    summary = _verify_root_artifacts(run_dir, behavior, events, spec)
    conclusion = classify_mechanism(
        summary,
        spec,
        mechanical_prerequisites_passed=mechanical_passed,
    )
    _assert_json_exact(
        read_json(run_dir / "conclusion.json"),
        conclusion,
        label="mechanism conclusion",
    )
    run_id = _verify_metadata(
        metadata,
        run_dir=run_dir,
        spec=spec,
        config_sha256=canonical_config.sha256,
        implementation=implementation,
        conclusion=conclusion,
        event_count=len(events),
        mechanical_prerequisites_passed=mechanical_passed,
    )
    return {
        "status": "verified",
        "run_id": run_id,
        "result": conclusion["result"],
        "selected_beta_label": conclusion["selected_beta_label"],
        "events": len(events),
        "path_behavior_rows": len(behavior),
        "markets_reconstructed": len(results),
        "mechanical_prerequisites_passed": mechanical_passed,
    }
