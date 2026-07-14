"""Integrity helpers for immutable research-run artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from adaptive_jump.backtest import performance_metrics
from adaptive_jump.config import ResearchConfig, load_config

MODELS = ("buy_and_hold", "hmm", "fixed_jm")
SELECTION_MODELS = ("fixed_jm", "hmm")
METRIC_FIELDS = (
    "start",
    "end",
    "observations",
    "cagr",
    "volatility",
    "sharpe",
    "maximum_drawdown",
    "calmar",
    "expected_shortfall_5pct",
    "turnover",
    "leverage",
)
TRADE_COLUMNS = (
    "date",
    "equity_simple",
    "cash_return",
    "signal",
    "position",
    "gross_return",
    "one_way_turnover",
    "transaction_cost",
    "strategy_return",
)


class ArtifactError(RuntimeError):
    """Raised when frozen study inputs or run artifacts are invalid."""


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object with a stable research-facing error."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        raise ArtifactError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ArtifactError(f"JSON must contain an object: {path}")
    return document


def sha256_file(path: Path) -> str:
    """Hash a file without trusting stored metadata."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, document: dict[str, Any]) -> None:
    """Write canonical human-readable JSON without NaN values."""
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _inventory_files(run_dir: Path) -> dict[str, str]:
    return {
        str(path.relative_to(run_dir)): sha256_file(path)
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name not in {"inventory.json", "run.json"}
    }


def write_inventory(run_dir: Path) -> None:
    """Seal every immutable run file; mutable lifecycle metadata is separate."""
    write_json(
        run_dir / "inventory.json",
        {"schema_version": 1, "files": _inventory_files(run_dir)},
    )


def verify_inventory(run_dir: Path) -> None:
    """Reject missing, extra, or modified immutable run files."""
    inventory = read_json(run_dir / "inventory.json")
    expected = inventory.get("files")
    if not isinstance(expected, dict):
        raise ArtifactError("invalid artifact inventory")
    if _inventory_files(run_dir) != expected:
        raise ArtifactError("artifact inventory mismatch")


def directional_gate(metrics: pd.DataFrame, primary_delay: int) -> dict[str, Any]:
    """Evaluate the frozen three-condition replication gate."""
    rows = []
    primary = metrics.loc[metrics["delay"] == primary_delay]
    for market, values in primary.groupby("market"):
        indexed = values.set_index("model")
        required = {"fixed_jm", "hmm", "buy_and_hold"}
        if set(indexed.index) != required:
            raise ArtifactError(f"{market}: incomplete primary metrics")
        jm = indexed.loc["fixed_jm"]
        hmm = indexed.loc["hmm"]
        hold = indexed.loc["buy_and_hold"]
        checks = {
            "sharpe_above_hmm": bool(jm["sharpe"] > hmm["sharpe"]),
            "sharpe_above_buy_and_hold": bool(jm["sharpe"] > hold["sharpe"]),
            "mdd_below_buy_and_hold": bool(
                abs(jm["maximum_drawdown"]) < abs(hold["maximum_drawdown"])
            ),
        }
        rows.append({"market": market, **checks, "passed": all(checks.values())})
    passed = len(rows) == 3 and all(row["passed"] for row in rows)
    return {
        "claim_label": "proxy replication",
        "primary_delay": primary_delay,
        "markets": rows,
        "passed": passed,
        "conclusion": (
            "directional proxy replication"
            if passed
            else "non-replication; adaptive work remains blocked"
        ),
    }


def verify_run(run: str | Path) -> dict[str, Any]:
    """Independently verify a sealed fixed-baseline run from its artifacts."""
    run_dir = Path(run).resolve()
    if not run_dir.is_dir():
        raise ArtifactError(f"run directory does not exist: {run_dir}")
    metadata = read_json(run_dir / "run.json")
    study_kind = metadata.get("study_kind")
    if study_kind == "jm_train_window_sensitivity":
        from adaptive_jump.window_verifier import verify_window_run

        return verify_window_run(run_dir)
    if study_kind is not None:
        raise ArtifactError(f"unsupported study kind: {study_kind}")
    verify_inventory(run_dir)
    inventory = read_json(run_dir / "inventory.json").get("files")
    config = _verify_identity(run_dir, metadata)
    boundary_rows = _verify_boundaries(run_dir, metadata, config)

    metric_rows = 0
    maximum_difference = 0.0
    conclusion = str(metadata.get("conclusion", ""))
    if metadata["status"] == "complete":
        metrics, maximum_difference = _verify_metrics(run_dir, config)
        claim = read_json(run_dir / "claim.json")
        expected_claim = directional_gate(
            metrics, config.backtest_protocol.primary_delay
        )
        if claim != expected_claim:
            raise ArtifactError("claim does not match recomputed primary metrics")
        if metadata.get("conclusion") != claim["conclusion"]:
            raise ArtifactError("run conclusion does not match claim")
        metric_rows = len(metrics)
        conclusion = claim["conclusion"]

    return {
        "schema_version": 1,
        "run_id": metadata["run_id"],
        "status": metadata["status"],
        "inventory_files": len(inventory),
        "boundary_rows": boundary_rows,
        "metric_rows": metric_rows,
        "maximum_metric_absolute_difference": maximum_difference,
        "conclusion": conclusion,
    }


def _verify_identity(run_dir: Path, metadata: dict[str, Any]) -> ResearchConfig:
    if metadata.get("schema_version") != 1:
        raise ArtifactError("unsupported run schema")
    if metadata.get("status") not in {"complete", "boundary_failed"}:
        raise ArtifactError("run is not in a verifiable terminal state")
    config_hash = _require_hex(metadata.get("config_sha256"), 64, "config hash")
    data_hash = _require_hex(
        metadata.get("data_manifest_sha256"), 64, "data manifest hash"
    )
    git_sha = _require_hex(metadata.get("git_sha"), None, "Git SHA")
    expected_id = "fixed-baselines-" + "-".join(
        value[:12] for value in (config_hash, data_hash, git_sha)
    )
    if metadata.get("run_id") != expected_id or run_dir.name != expected_id:
        raise ArtifactError("run directory and identity lock disagree")

    config_path = run_dir / "config.lock.toml"
    manifest_path = run_dir / "data-manifest.json"
    if sha256_file(config_path) != config_hash:
        raise ArtifactError("config lock hash mismatch")
    if sha256_file(manifest_path) != data_hash:
        raise ArtifactError("data manifest lock hash mismatch")
    config = load_config(config_path)
    manifest = read_json(manifest_path)
    if (
        config.sha256 != config_hash
        or manifest.get("config_sha256") != config_hash
        or manifest.get("config_id") != config.config_id
        or manifest.get("replication_cutoff") != config.replication_cutoff.isoformat()
    ):
        raise ArtifactError("locked config and data manifest disagree")
    return config


def _require_hex(value: Any, length: int | None, label: str) -> str:
    if not isinstance(value, str) or (length is not None and len(value) != length):
        raise ArtifactError(f"invalid {label}")
    if len(value) < 12:
        raise ArtifactError(f"invalid {label}")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ArtifactError(f"invalid {label}") from exc
    return value


def _verify_boundaries(
    run_dir: Path, metadata: dict[str, Any], config: ResearchConfig
) -> int:
    frame = _read_csv(run_dir / "boundaries.csv")
    required = {
        "market",
        "model",
        "delay",
        "upper_candidate",
        "selected_months",
        "total_months",
        "fraction",
        "limit",
        "passed",
    }
    if set(frame.columns) != required:
        raise ArtifactError("boundary columns violate the frozen schema")
    expected = {
        (market.id, model, delay)
        for market in config.markets
        for model in SELECTION_MODELS
        for delay in config.backtest_protocol.robustness_delays
    }
    actual = set(frame[["market", "model", "delay"]].itertuples(index=False, name=None))
    if len(frame) != len(actual) or actual != expected:
        raise ArtifactError("boundary market/model/delay coverage is invalid")
    if not frame["passed"].isin([True, False]).all():
        raise ArtifactError("boundary pass flags are invalid")
    upper = {
        "fixed_jm": max(config.jm_protocol.lambda_grid),
        "hmm": max(config.hmm_protocol.smoothing_grid),
    }
    for row in frame.itertuples(index=False):
        total = int(row.total_months)
        selected = int(row.selected_months)
        if (
            float(row.total_months) != total
            or float(row.selected_months) != selected
            or total <= 0
            or selected < 0
            or selected > total
        ):
            raise ArtifactError("boundary month counts are invalid")
        expected_fraction = selected / total
        expected_passed = (
            expected_fraction <= config.selection_protocol.boundary_fraction_limit
        )
        if (
            not math.isclose(
                float(row.upper_candidate),
                float(upper[row.model]),
                rel_tol=0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(row.fraction), expected_fraction, rel_tol=0, abs_tol=1e-12
            )
            or not math.isclose(
                float(row.limit),
                config.selection_protocol.boundary_fraction_limit,
                rel_tol=0,
                abs_tol=1e-12,
            )
            or bool(row.passed) != expected_passed
        ):
            raise ArtifactError("boundary diagnostic values are inconsistent")
    all_passed = bool(frame["passed"].all())
    if metadata["status"] == "complete":
        if not all_passed or metadata.get("metrics_opened") is not True:
            raise ArtifactError("complete run has an invalid boundary state")
    elif all_passed or metadata.get("metrics_opened") is not False:
        raise ArtifactError("boundary-failed run has an invalid boundary state")
    if metadata["status"] == "boundary_failed":
        if metadata.get("conclusion") != "grid expansion required before OOS metrics":
            raise ArtifactError("boundary-failed run conclusion is invalid")
        if any(
            (run_dir / name).exists() for name in ("metrics.csv", "claim.json")
        ) or any(run_dir.glob("*/trades/*.csv")):
            raise ArtifactError("boundary-failed run exposes sealed metrics")
    return len(frame)


def _verify_metrics(
    run_dir: Path, config: ResearchConfig
) -> tuple[pd.DataFrame, float]:
    metrics = _read_csv(run_dir / "metrics.csv")
    expected_columns = {"market", "model", "delay", *METRIC_FIELDS}
    if set(metrics.columns) != expected_columns or metrics.isna().any().any():
        raise ArtifactError("metric table violates the frozen schema")
    expected_keys = {
        (market.id, model, delay)
        for market in config.markets
        for model in MODELS
        for delay in config.backtest_protocol.robustness_delays
    }
    actual_keys = set(
        metrics[["market", "model", "delay"]].itertuples(index=False, name=None)
    )
    if len(metrics) != len(actual_keys) or actual_keys != expected_keys:
        raise ArtifactError("metric market/model/delay coverage is invalid")

    expected_trade_files = {
        run_dir / market.id / "trades" / f"{model}-delay-{delay}.csv"
        for market in config.markets
        for model in MODELS
        for delay in config.backtest_protocol.robustness_delays
    }
    actual_trade_files = set(run_dir.glob("*/trades/*.csv"))
    if actual_trade_files != expected_trade_files:
        raise ArtifactError("trade artifact coverage is invalid")

    maximum_difference = 0.0
    tolerance = config.selection_protocol.tie_tolerance
    for market in config.markets:
        for delay in config.backtest_protocol.robustness_delays:
            paths = {
                model: read_trade_path(
                    run_dir / market.id / "trades" / f"{model}-delay-{delay}.csv",
                    delay,
                    config.backtest_protocol.one_way_cost_bps,
                )
                for model in MODELS
            }
            reference = paths[MODELS[0]][["date", "equity_simple", "cash_return"]]
            for model, path in paths.items():
                if not path[["date", "equity_simple", "cash_return"]].equals(reference):
                    raise ArtifactError(
                        f"{market.id} delay {delay}: trade samples differ"
                    )
                calculated = performance_metrics(
                    path,
                    periods_per_year=config.metrics_protocol.periods_per_year,
                    volatility_ddof=config.metrics_protocol.volatility_ddof,
                    expected_shortfall_quantile=(
                        config.metrics_protocol.expected_shortfall_quantile
                    ),
                )
                row = metrics.loc[
                    (metrics["market"] == market.id)
                    & (metrics["model"] == model)
                    & (metrics["delay"] == delay)
                ].iloc[0]
                for field, expected_value in calculated.items():
                    if field in {"start", "end", "observations"}:
                        if row[field] != expected_value:
                            raise ArtifactError(
                                f"metric mismatch: {market.id}/{model}/{field}"
                            )
                        continue
                    difference = abs(float(row[field]) - float(expected_value))
                    maximum_difference = max(maximum_difference, difference)
                    if difference > tolerance:
                        raise ArtifactError(
                            f"metric mismatch: {market.id}/{model}/{field}"
                        )
    return metrics, maximum_difference


def read_trade_path(path: Path, delay: int, cost_bps: float) -> pd.DataFrame:
    frame = _read_csv(path)
    if tuple(frame.columns) != TRADE_COLUMNS or frame.empty:
        raise ArtifactError(f"invalid trade path schema: {path}")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame["date"].duplicated().any() or not frame["date"].is_monotonic_increasing:
        raise ArtifactError(f"invalid trade dates: {path}")
    numeric_columns = list(TRADE_COLUMNS[1:])
    numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ArtifactError(f"incomplete trade path: {path}")
    frame[numeric_columns] = numeric
    if (
        not frame["signal"].isin([0.0, 1.0]).all()
        or not frame["position"].isin([0.0, 1.0]).all()
    ):
        raise ArtifactError(f"non-binary trade path: {path}")
    gross = (
        frame["position"] * frame["equity_simple"]
        + (1.0 - frame["position"]) * frame["cash_return"]
    )
    cost = frame["one_way_turnover"] * cost_bps / 10_000.0
    if (
        not np.allclose(frame["gross_return"], gross, rtol=0, atol=1e-15)
        or not np.allclose(frame["transaction_cost"], cost, rtol=0, atol=1e-15)
        or not np.allclose(frame["strategy_return"], gross - cost, rtol=0, atol=1e-15)
        or not np.allclose(
            frame["one_way_turnover"].iloc[1:],
            frame["position"].diff().abs().iloc[1:],
            rtol=0,
            atol=1e-15,
        )
    ):
        raise ArtifactError(f"trade accounting mismatch: {path}")
    offset = delay + 1
    if (
        not frame["position"]
        .iloc[offset:]
        .reset_index(drop=True)
        .equals(frame["signal"].iloc[:-offset].reset_index(drop=True))
    ):
        raise ArtifactError(f"trade delay mismatch: {path}")
    return frame


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, OSError, pd.errors.ParserError) as exc:
        raise ArtifactError(f"cannot read CSV {path}: {exc}") from exc
