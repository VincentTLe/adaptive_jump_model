"""Recompute performance and bootstrap tables for a JM-window artifact."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_jump.artifacts import METRIC_FIELDS, ArtifactError, read_trade_path
from adaptive_jump.config import ResearchConfig
from adaptive_jump.window_spec import WindowStudySpec
from adaptive_jump.window_study import (
    COMPARISON_MODELS,
    bootstrap_rows,
    comparison_metrics,
)

WINDOW_METRIC_FIELDS = (*METRIC_FIELDS, "cash_fraction", "switch_count")


def verify_window_metrics(
    run_dir: Path, config: ResearchConfig, spec: WindowStudySpec
) -> tuple[pd.DataFrame, dict[tuple[str, int], dict[str, pd.DataFrame]], float]:
    """Recalculate every metric row from validated trade paths."""
    metrics = _read_csv(run_dir / "metrics.csv")
    expected_keys = {
        (market.id, model, delay)
        for market in config.markets
        for model in COMPARISON_MODELS
        for delay in spec.delays
    }
    actual_keys = set(
        metrics[["market", "model", "delay"]].itertuples(index=False, name=None)
    )
    if (
        set(metrics.columns) != {"market", "model", "delay", *WINDOW_METRIC_FIELDS}
        or actual_keys != expected_keys
        or len(metrics) != len(expected_keys)
    ):
        raise ArtifactError("JM-window metric coverage is invalid")
    expected_files = {
        run_dir / market.id / "trades" / f"{model}-delay-{delay}.csv"
        for market in config.markets
        for model in COMPARISON_MODELS
        for delay in spec.delays
    }
    if set(run_dir.glob("*/trades/*.csv")) != expected_files:
        raise ArtifactError("JM-window trade coverage is invalid")
    paths_by_key = {}
    maximum = 0.0
    for market in config.markets:
        for delay in spec.delays:
            paths = {
                model: read_trade_path(
                    run_dir / market.id / "trades" / f"{model}-delay-{delay}.csv",
                    delay,
                    config.backtest_protocol.one_way_cost_bps,
                )
                for model in COMPARISON_MODELS
            }
            reference = paths[COMPARISON_MODELS[0]][
                ["date", "equity_simple", "cash_return"]
            ]
            if any(
                not path[["date", "equity_simple", "cash_return"]].equals(reference)
                for path in paths.values()
            ):
                raise ArtifactError(f"{market.id} delay {delay}: trade samples differ")
            calculated = comparison_metrics(paths, config).set_index("model")
            stored = metrics.loc[
                (metrics["market"] == market.id) & (metrics["delay"] == delay)
            ].set_index("model")
            maximum = max(
                maximum,
                _compare_tables(stored, calculated, WINDOW_METRIC_FIELDS),
            )
            paths_by_key[(market.id, delay)] = paths
    return metrics, paths_by_key, maximum


def verify_window_bootstrap(
    run_dir: Path,
    paths: dict[tuple[str, int], dict[str, pd.DataFrame]],
    config: ResearchConfig,
    spec: WindowStudySpec,
) -> tuple[pd.DataFrame, float]:
    """Rerun all paired stationary-bootstrap rows from sealed paths."""
    stored = _read_csv(run_dir / "bootstrap.csv")
    fields = (
        "observed_delta",
        "lower_one_sided",
        "confidence_low",
        "confidence_high",
        "replications",
    )
    expected_keys = {
        (market.id, block)
        for market in config.markets
        for block in spec.bootstrap_blocks
    }
    actual_keys = set(
        stored[["market", "block_length"]].itertuples(index=False, name=None)
    )
    if (
        set(stored.columns) != {"market", "block_length", *fields}
        or actual_keys != expected_keys
        or len(stored) != len(expected_keys)
    ):
        raise ArtifactError("JM-window bootstrap coverage is invalid")
    maximum = 0.0
    for market in config.markets:
        calculated = bootstrap_rows(
            paths[(market.id, spec.primary_delay)], spec, config
        ).set_index("block_length")
        actual = stored.loc[stored["market"] == market.id].set_index("block_length")
        maximum = max(maximum, _compare_tables(actual, calculated, fields))
    return stored, maximum


def _compare_tables(
    stored: pd.DataFrame, calculated: pd.DataFrame, fields: tuple[str, ...]
) -> float:
    if not stored.index.equals(calculated.index):
        raise ArtifactError("recomputed evidence row order differs")
    maximum = 0.0
    exact_fields = {
        "start",
        "end",
        "observations",
        "switch_count",
        "replications",
    }
    for field in fields:
        if field in exact_fields:
            if not stored[field].equals(calculated[field]):
                raise ArtifactError(f"evidence mismatch: {field}")
        else:
            difference = np.abs(
                stored[field].astype(float) - calculated[field].astype(float)
            )
            maximum = max(maximum, float(difference.max()))
            if (difference > 1e-12).any():
                raise ArtifactError(f"evidence mismatch: {field}")
    return maximum


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, OSError, pd.errors.ParserError) as exc:
        raise ArtifactError(f"cannot read CSV {path}: {exc}") from exc
