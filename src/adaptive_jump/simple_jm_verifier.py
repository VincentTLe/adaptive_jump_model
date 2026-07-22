"""Independent replay verification for sealed simple-JM and DD loss-scale runs."""

from __future__ import annotations

import hashlib
import itertools
import math
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from adaptive_jump.artifacts import (
    TRADE_COLUMNS,
    read_json,
    read_trade_path,
    sha256_file,
    verify_inventory,
)
from adaptive_jump.backtest import apply_signal
from adaptive_jump.config import ResearchConfig, load_config
from adaptive_jump.simple_jm_controls import ControlPath
from adaptive_jump.simple_jm_suite import (
    ALL_MODELS,
    CONTROLS,
    DD_OBSERVATION_LOSS_SCALE,
    DEVELOPMENT_CUTOFF,
    EXPERIMENT_ID,
    FITTED_VARIANTS,
    LOSS_SCALE_EXPERIMENT_ID,
    LOSS_SCALE_MODELS,
    MARKETS,
    SCALED_DD_VARIANT,
    LossScaleMarketSource,
    SimpleJMSuiteError,
    VariantOutput,
    build_decision,
    build_traces,
    fit_degeneracy_row,
    load_dd_loss_scale_spec,
    load_loss_scale_sources,
    loss_scale_contrasts,
    mapping_digest,
    metric_rows,
    trade_route_equal,
    validate_loss_scale_protocol,
    validate_traces,
    verify_loss_scale_math,
)
from adaptive_jump.walkforward import select_monthly_candidate


def _implementation_source_commit(
    repo_root: Path, implementation_files: dict[str, Any]
) -> str | None:
    """Resolve the lock from current files or one complete Git snapshot."""
    if not implementation_files:
        raise SimpleJMSuiteError("implementation lock contains no files")
    expected: dict[str, str] = {}
    for relative, digest in implementation_files.items():
        path = Path(relative) if isinstance(relative, str) else Path()
        valid_path = (
            isinstance(relative, str)
            and relative == path.as_posix()
            and not path.is_absolute()
            and ".." not in path.parts
            and ":" not in relative
        )
        valid_digest = (
            isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
        )
        if not valid_path or not valid_digest:
            raise SimpleJMSuiteError(
                "implementation lock contains an invalid file entry"
            )
        expected[relative] = digest

    if all(
        (repo_root / relative).is_file() and sha256_file(repo_root / relative) == digest
        for relative, digest in expected.items()
    ):
        return None

    paths = sorted(expected)
    try:
        history = subprocess.run(
            ["git", "log", "--all", "--full-history", "--format=%H", "--", *paths],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise SimpleJMSuiteError("cannot inspect implementation history") from exc
    if history.returncode != 0:
        raise SimpleJMSuiteError("cannot inspect implementation history")

    try:
        for commit in dict.fromkeys(history.stdout.splitlines()):
            matched = True
            for relative, digest in expected.items():
                blob = subprocess.run(
                    ["git", "show", f"{commit}:{relative}"],
                    cwd=repo_root,
                    check=False,
                    capture_output=True,
                )
                if (
                    blob.returncode != 0
                    or hashlib.sha256(blob.stdout).hexdigest() != digest
                ):
                    matched = False
                    break
            if matched:
                return commit
    except OSError as exc:
        raise SimpleJMSuiteError("cannot inspect implementation history") from exc
    raise SimpleJMSuiteError(
        "no single Git commit contains the complete locked implementation"
    )


def _replay_scaled_selector(
    run_dir: Path,
    market: str,
    source: LossScaleMarketSource,
    dates: pd.Series,
    stored_trades: pd.DataFrame,
    config: ResearchConfig,
) -> VariantOutput:
    """Replay candidate states through monthly selection and t+2 accounting."""
    target = run_dir / market / SCALED_DD_VARIANT
    states = pd.read_csv(target / "candidate-states.csv")
    if tuple(states.columns[:1]) != ("date",):
        raise SimpleJMSuiteError(f"{market}: invalid scaled candidate states")
    states["date"] = pd.to_datetime(states["date"], errors="raise")
    states = states.set_index("date")
    try:
        states.columns = [float(column) for column in states.columns]
    except (TypeError, ValueError) as exc:
        raise SimpleJMSuiteError(f"{market}: invalid scaled candidate grid") from exc
    if tuple(states.columns) != config.jm_protocol.lambda_grid:
        raise SimpleJMSuiteError(f"{market}: scaled candidate grid changed")
    returns = source.features.loc[:, ["date", "equity_simple", "cash_return"]]
    selection = select_monthly_candidate(
        returns,
        states,
        config.selection_protocol,
        delay_trading_days=1,
        one_way_cost_bps=10,
        periods_per_year=252,
        volatility_ddof=1,
    )
    full_trades = apply_signal(
        returns,
        selection.signal.reset_index(drop=True),
        delay_trading_days=1,
        one_way_cost_bps=10,
    )
    replayed_trades = (full_trades.set_index("date").reindex(dates).reset_index()).loc[
        :, TRADE_COLUMNS
    ]
    if not trade_route_equal(replayed_trades, stored_trades):
        raise SimpleJMSuiteError(f"{market}: scaled t+2 trade replay changed")

    stored_choices = pd.read_csv(target / "choices.csv")
    stored_signal = pd.read_csv(target / "selected-signal.csv")
    expected_signal = selection.signal.rename("selected_signal").reset_index()
    for frame, column in (
        (stored_choices, "decision_date"),
        (selection.choices, "decision_date"),
        (stored_signal, "date"),
        (expected_signal, "date"),
    ):
        frame[column] = pd.to_datetime(frame[column], errors="raise")
    try:
        pd.testing.assert_frame_equal(
            stored_choices,
            selection.choices,
            check_dtype=False,
            check_exact=True,
        )
        pd.testing.assert_frame_equal(
            stored_signal,
            expected_signal,
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError as exc:
        raise SimpleJMSuiteError(f"{market}: scaled selector replay changed") from exc
    selected_state = (1.0 - selection.signal).rename("selected_state")
    return VariantOutput(
        market,
        SCALED_DD_VARIANT,
        states,
        pd.read_csv(target / "refits.csv"),
        selection,
        selected_state,
        selection.signal,
        full_trades,
        read_json(target / "boundary.json"),
    )


def _verify_fit_degeneracy(
    run_dir: Path, variants: tuple[str, ...] = FITTED_VARIANTS
) -> pd.DataFrame:
    expected_rows = []
    for market, variant in itertools.product(MARKETS, variants):
        target = run_dir / market / variant
        expected_rows.append(
            fit_degeneracy_row(
                market,
                variant,
                pd.read_csv(target / "refits.csv"),
                pd.read_csv(target / "choices.csv"),
            )
        )
    expected = (
        pd.DataFrame.from_records(expected_rows)
        .sort_values(["market", "variant"])
        .reset_index(drop=True)
    )
    stored = (
        pd.read_csv(run_dir / "fit-degeneracy.csv")
        .sort_values(["market", "variant"])
        .reset_index(drop=True)
    )
    if tuple(stored.columns) != tuple(expected.columns) or len(stored) != len(expected):
        raise SimpleJMSuiteError("fit-degeneracy artifact has invalid schema")
    floating = {
        "collapsed_fit_fraction",
        "selected_collapsed_fraction",
    }
    for column in expected:
        if column in floating:
            if not np.allclose(
                pd.to_numeric(stored[column], errors="raise"),
                pd.to_numeric(expected[column], errors="raise"),
                rtol=0,
                atol=1e-15,
                equal_nan=True,
            ):
                raise SimpleJMSuiteError(f"fit-degeneracy mismatch: {column}")
        elif not stored[column].equals(expected[column]):
            raise SimpleJMSuiteError(f"fit-degeneracy mismatch: {column}")
    return expected


def _verify_trace_trade_rows(run_dir: Path, traces: pd.DataFrame) -> None:
    """Link every trace's t+2 accounting fields to its verified trade path."""
    cache: dict[tuple[str, str], pd.DataFrame] = {}
    fields = (
        "position",
        "one_way_turnover",
        "transaction_cost",
        "gross_return",
        "strategy_return",
    )
    for trace in traces.itertuples(index=False):
        market = str(trace.market)
        variant = str(trace.variant)
        if market not in MARKETS or not variant:
            raise SimpleJMSuiteError("trace market or variant is invalid")
        key = (market, variant)
        if key not in cache:
            cache[key] = read_trade_path(
                run_dir / market / variant / "trades.csv", 1, 10
            ).set_index("date")
        trade_date = pd.Timestamp(trace.trade_date)
        matches = cache[key].loc[cache[key].index == trade_date]
        if len(matches) != 1:
            raise SimpleJMSuiteError("trace t+2 trade row is missing")
        trade = matches.iloc[0]
        for field in fields:
            if not math.isclose(
                float(getattr(trace, field)),
                float(trade[field]),
                rel_tol=0,
                abs_tol=1e-15,
            ):
                raise SimpleJMSuiteError(f"trace trade field changed: {field}")


def verify_simple_jm_run(run_dir: Path) -> dict[str, Any]:
    """Independently replay metrics, decisions, source hashes, and timing."""
    run_dir = run_dir.resolve()
    verify_inventory(run_dir)
    metadata = read_json(run_dir / "run.json")
    if (
        metadata.get("status") != "complete"
        or metadata.get("study_kind") != EXPERIMENT_ID
    ):
        raise SimpleJMSuiteError("run metadata is not a completed simple-JM suite")
    spec_hash = sha256_file(run_dir / "study.lock.toml")
    if metadata.get("spec_sha256") != spec_hash:
        raise SimpleJMSuiteError("run spec hash mismatch")
    implementation = read_json(run_dir / "implementation-lock.json")
    implementation_files = implementation.get("files")
    if not isinstance(implementation_files, dict):
        raise SimpleJMSuiteError("implementation lock is invalid")
    repo_root = run_dir.parents[2]
    implementation_digest = mapping_digest(implementation_files)
    if (
        implementation.get("bundle_sha256") != implementation_digest
        or metadata.get("implementation_sha256") != implementation_digest
    ):
        raise SimpleJMSuiteError("implementation digest mismatch")
    implementation_source_commit = _implementation_source_commit(
        repo_root, implementation_files
    )
    source_lock = read_json(run_dir / "source-lock.json")
    for path_text, evidence in source_lock["explicitly_read_scientific_inputs"].items():
        path = Path(path_text)
        if not path.is_file() or sha256_file(path) != evidence["sha256"]:
            raise SimpleJMSuiteError(f"source changed after run: {path}")
    gamma_route = read_json(run_dir / "gamma-zero-route.json")
    if gamma_route.get("route") != "sealed canonical fixed_jm":
        raise SimpleJMSuiteError("gamma-zero route is not the sealed fixed JM")
    explicit_paths = source_lock["explicitly_read_scientific_inputs"]
    for market, files in gamma_route["markets"].items():
        for evidence in files.values():
            if (
                evidence["path"] not in explicit_paths
                or explicit_paths[evidence["path"]]["sha256"] != evidence["sha256"]
            ):
                raise SimpleJMSuiteError(f"{market}: incomplete gamma-zero source lock")

    config = load_config(run_dir / "config.lock.toml")
    stored = pd.read_csv(run_dir / "summary.csv")
    rows = []
    max_difference = 0.0
    for market in MARKETS:
        paths = {
            model: read_trade_path(run_dir / market / model / "trades.csv", 1, 10)
            for model in ALL_MODELS
        }
        dates = paths[ALL_MODELS[0]]["date"]
        sealed_fixed = read_trade_path(
            Path(gamma_route["markets"][market]["positions_costs_returns"]["path"]),
            1,
            10,
        )
        routed_fixed = (
            sealed_fixed.set_index("date").reindex(dates).reset_index()
        ).loc[:, TRADE_COLUMNS]
        if not trade_route_equal(routed_fixed, paths["fixed_jm"]):
            raise SimpleJMSuiteError(f"{market}: gamma-zero trade route changed")
        for model, path in paths.items():
            if (
                not path["date"].equals(dates)
                or path["date"].max() > DEVELOPMENT_CUTOFF
            ):
                raise SimpleJMSuiteError(f"{market}/{model}: invalid common dates")
        recalculated = metric_rows(market, paths, config)
        rows.extend(recalculated)
    expected = pd.DataFrame.from_records(rows)
    for column in (
        "sharpe",
        "maximum_drawdown",
        "turnover",
        "cash_fraction",
        "switch_count",
        "gap_vs_stronger_control",
    ):
        left = pd.to_numeric(stored[column], errors="coerce")
        right = pd.to_numeric(expected[column], errors="coerce")
        difference = np.abs(left - right)
        finite = difference[np.isfinite(difference)]
        if len(finite):
            max_difference = max(max_difference, float(finite.max()))
        if not np.allclose(left, right, rtol=0, atol=1e-12, equal_nan=True):
            raise SimpleJMSuiteError(f"stored metric mismatch: {column}")
    if read_json(run_dir / "decision.json") != build_decision(expected):
        raise SimpleJMSuiteError("stored decision does not match recomputed metrics")
    trace = pd.read_csv(run_dir / "traces.csv")
    validate_traces(trace)
    _verify_trace_trade_rows(run_dir, trace)
    degeneracy = _verify_fit_degeneracy(run_dir)
    return {
        "schema_version": 1,
        "run_id": metadata["run_id"],
        "status": metadata["status"],
        "implementation_source_commit": implementation_source_commit,
        "metric_rows": len(expected),
        "trace_rows": len(trace),
        "degeneracy_rows": len(degeneracy),
        "maximum_metric_absolute_difference": max_difference,
        "conclusion": metadata["conclusion"],
    }


def verify_dd_loss_scale_run(run_dir: Path) -> dict[str, Any]:
    """Replay source routes, accounting, metrics, traces, and decisions."""
    run_dir = run_dir.resolve()
    verify_inventory(run_dir)
    metadata = read_json(run_dir / "run.json")
    if (
        metadata.get("status") != "complete"
        or metadata.get("study_kind") != LOSS_SCALE_EXPERIMENT_ID
        or metadata.get("spec_sha256") != sha256_file(run_dir / "study.lock.toml")
    ):
        raise SimpleJMSuiteError("run metadata is not a completed loss-scale study")
    implementation = read_json(run_dir / "implementation-lock.json")
    files = implementation.get("files")
    digest = mapping_digest(files) if isinstance(files, dict) else ""
    if (
        not isinstance(files, dict)
        or implementation.get("bundle_sha256") != digest
        or metadata.get("implementation_sha256") != digest
    ):
        raise SimpleJMSuiteError("loss-scale implementation lock is invalid")
    repo_root = run_dir.parents[2]
    source_commit = _implementation_source_commit(repo_root, files)

    config = replace(
        load_config(run_dir / "config.lock.toml"),
        path=repo_root / "research.toml",
    )
    spec = load_dd_loss_scale_spec(run_dir / "study.lock.toml", config)
    validate_loss_scale_protocol(config, spec)
    sources = load_loss_scale_sources(spec, config)
    rows = []
    replayed_outputs: dict[tuple[str, str], VariantOutput | ControlPath] = {}
    aligned_paths: dict[str, dict[str, pd.DataFrame]] = {}
    for market in MARKETS:
        paths = {
            model: read_trade_path(run_dir / market / model / "trades.csv", 1, 10)
            for model in LOSS_SCALE_MODELS
        }
        dates = paths[LOSS_SCALE_MODELS[0]]["date"]
        replayed_outputs[(market, SCALED_DD_VARIANT)] = _replay_scaled_selector(
            run_dir,
            market,
            sources[market],
            dates,
            paths[SCALED_DD_VARIANT],
            config,
        )
        for model in (*CONTROLS, "dd_only"):
            sealed = (
                sources[market]
                .controls[model]
                .set_index("date")
                .reindex(dates)
                .reset_index()
                .loc[:, TRADE_COLUMNS]
            )
            if not trade_route_equal(sealed, paths[model]):
                raise SimpleJMSuiteError(f"{market}/{model}: source route changed")
        if any(
            not path["date"].equals(dates) or path["date"].max() > DEVELOPMENT_CUTOFF
            for path in paths.values()
        ):
            raise SimpleJMSuiteError(f"{market}: invalid common dates")
        aligned_paths[market] = paths
        rows.extend(
            metric_rows(
                market,
                paths,
                config,
                challengers=(SCALED_DD_VARIANT,),
            )
        )
    expected = pd.DataFrame.from_records(rows)
    stored = pd.read_csv(run_dir / "summary.csv")
    expected_contrast = loss_scale_contrasts(expected)
    stored_contrast = pd.read_csv(run_dir / "dd-scale-contrast.csv")
    try:
        pd.testing.assert_frame_equal(
            stored,
            expected,
            check_dtype=False,
            check_exact=False,
            rtol=0,
            atol=1e-12,
        )
        pd.testing.assert_frame_equal(
            stored_contrast,
            expected_contrast,
            check_dtype=False,
            check_exact=False,
            rtol=0,
            atol=1e-12,
        )
    except AssertionError as exc:
        raise SimpleJMSuiteError("loss-scale stored metrics changed") from exc
    decision = build_decision(expected, (SCALED_DD_VARIANT,))
    if (
        read_json(run_dir / "decision.json") != decision
        or metadata.get("conclusion") != decision["conclusion"]
    ):
        raise SimpleJMSuiteError("loss-scale decision changed")
    verification = read_json(run_dir / "verification.json")
    if (
        verification.get("math_contracts") != verify_loss_scale_math()
        or verification.get("observation_loss_scale") != DD_OBSERVATION_LOSS_SCALE
        or not verification["us_smoke"][0]["prefix_invariant"]
    ):
        raise SimpleJMSuiteError("loss-scale verification receipt is invalid")
    traces = pd.read_csv(run_dir / "traces.csv")
    validate_traces(traces)
    _verify_trace_trade_rows(run_dir, traces)
    expected_traces = build_traces(
        sources,
        replayed_outputs,
        aligned_paths,
        config,
        (SCALED_DD_VARIANT,),
    )
    for frame in (traces, expected_traces):
        for column in ("signal_date", "trade_date", "fit_date"):
            frame[column] = pd.to_datetime(frame[column], errors="raise")
    try:
        pd.testing.assert_frame_equal(
            traces,
            expected_traces,
            check_dtype=False,
            check_exact=False,
            rtol=0,
            atol=1e-12,
        )
    except AssertionError as exc:
        raise SimpleJMSuiteError("loss-scale trace replay changed") from exc
    if set(traces["variant"]) != {SCALED_DD_VARIANT}:
        raise SimpleJMSuiteError("loss-scale trace variant is invalid")
    degeneracy = _verify_fit_degeneracy(run_dir, (SCALED_DD_VARIANT,))
    differences = [
        np.abs(
            pd.to_numeric(stored[column], errors="raise")
            - pd.to_numeric(expected[column], errors="raise")
        ).max()
        for column in (
            "sharpe",
            "maximum_drawdown",
            "turnover",
            "cash_fraction",
            "switch_count",
        )
    ]
    return {
        "schema_version": 1,
        "run_id": metadata["run_id"],
        "status": metadata["status"],
        "implementation_source_commit": source_commit,
        "metric_rows": len(expected),
        "trace_rows": len(traces),
        "degeneracy_rows": len(degeneracy),
        "maximum_metric_absolute_difference": float(max(differences)),
        "conclusion": metadata["conclusion"],
    }
