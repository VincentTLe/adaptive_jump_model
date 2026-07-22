"""One-shot 2024-2026 holdout: DD-only leg and the windowed multi-model readout."""

from __future__ import annotations

import argparse
import json
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from adaptive_jump.artifacts import (
    TRADE_COLUMNS,
    ArtifactError,
    read_trade_path,
    sha256_file,
    write_json,
)
from adaptive_jump.backtest import apply_signal, performance_metrics
from adaptive_jump.config import ResearchConfig, load_config
from adaptive_jump.inference import bootstrap_sharpe_delta
from adaptive_jump.simple_jm_fitting import dd_only_states
from adaptive_jump.walkforward import select_monthly_candidate

EXPERIMENT_ID = "holdout-2026-001"
SPEC_NAME = "holdout-2026-001.toml"
CONTROLS = ("buy_and_hold", "hmm", "fixed_jm")
BOOTSTRAP_REPLICATIONS = 10_000
BOOTSTRAP_MEAN_BLOCK = 60
BOOTSTRAP_SEED = 20260722


class HoldoutError(ArtifactError):
    """Raised when the frozen holdout contract cannot be satisfied."""


def load_holdout_spec(repo_root: Path) -> dict[str, Any]:
    """Load the frozen holdout contract and require its registered hash."""
    spec_path = repo_root / "research" / SPEC_NAME
    digest = sha256_file(spec_path)
    registered = False
    registry = repo_root / "research" / "experiment_registry.jsonl"
    for line in registry.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        if (
            entry.get("experiment_id") == EXPERIMENT_ID
            and entry.get("status") == "FROZEN"
            and entry.get("frozen_spec_hash") == digest
        ):
            registered = True
    if not registered:
        raise HoldoutError("frozen holdout registration is missing or stale")
    document = tomllib.loads(spec_path.read_text(encoding="utf-8"))
    document["spec_sha256"] = digest
    return document


def run_dd_only_leg(
    config: ResearchConfig, frozen: Any, run_dir: Path
) -> dict[str, Path]:
    """Fit, select, and trade the DD-only challenger over the full sample."""
    from adaptive_jump.cli import prepare_manifest_market

    backtest = config.backtest_protocol
    outputs: dict[str, Path] = {}
    for market in config.markets:
        market_input = prepare_manifest_market(config, frozen, market.id)
        frame = market_input.frame
        jm = dd_only_states(frame, config.model_protocol, config.jm_protocol)
        returns = frame.loc[:, ["date", "equity_simple", "cash_return"]]
        selection = select_monthly_candidate(
            returns,
            jm.states,
            config.selection_protocol,
            delay_trading_days=backtest.primary_delay,
            one_way_cost_bps=backtest.one_way_cost_bps,
            periods_per_year=config.metrics_protocol.periods_per_year,
            volatility_ddof=config.metrics_protocol.volatility_ddof,
        )
        trades = apply_signal(
            returns,
            selection.signal.reset_index(drop=True),
            delay_trading_days=backtest.primary_delay,
            one_way_cost_bps=backtest.one_way_cost_bps,
        )
        target = run_dir / market.id / "dd_only"
        target.mkdir(parents=True, exist_ok=True)
        jm.refits.to_csv(target / "refits.csv", index=False)
        selection.choices.to_csv(target / "choices.csv", index=False)
        selection.signal.rename("selected_signal").reset_index().to_csv(
            target / "selected-signal.csv", index=False
        )
        complete = trades.loc[:, TRADE_COLUMNS].notna().all(axis=1)
        trades.loc[complete, TRADE_COLUMNS].to_csv(
            target / "trades.csv", index=False, float_format="%.17g"
        )
        outputs[market.id] = target
    return outputs


def _window(trades: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp):
    frame = trades.loc[(trades["date"] >= start) & (trades["date"] <= end)]
    if frame.empty:
        raise HoldoutError("holdout window has no trade rows")
    return frame.reset_index(drop=True)


def _metric_row(trades: pd.DataFrame, config: ResearchConfig) -> dict[str, float]:
    metrics = performance_metrics(
        trades,
        periods_per_year=config.metrics_protocol.periods_per_year,
        volatility_ddof=config.metrics_protocol.volatility_ddof,
    )
    switches = int((trades["position"].diff().fillna(0.0) != 0).sum())
    cash_fraction = float((trades["position"] == 0).mean())
    return {**metrics, "switch_count": switches, "cash_fraction": cash_fraction}


def readout(
    repo_root: Path,
    config: ResearchConfig,
    baselines_run: Path,
    dd_run: Path,
    out_dir: Path,
    holdout_start: pd.Timestamp,
    holdout_end: pd.Timestamp,
) -> dict[str, Any]:
    """Compute the frozen windowed comparison for every declared model."""
    delay = config.backtest_protocol.primary_delay
    cost = config.backtest_protocol.one_way_cost_bps
    rows: list[dict[str, Any]] = []
    decisions: dict[str, dict[str, Any]] = {}
    for market in config.markets:
        trades = {
            model: read_trade_path(
                baselines_run / market.id / "trades" / f"{model}-delay-{delay}.csv",
                delay,
                cost,
            )
            for model in CONTROLS
        }
        trades["dd_only"] = read_trade_path(
            dd_run / market.id / "dd_only" / "trades.csv", delay, cost
        )
        for frame in trades.values():
            frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        windowed = {
            model: _window(frame, holdout_start, holdout_end)
            for model, frame in trades.items()
        }
        dates = {model: tuple(frame["date"]) for model, frame in windowed.items()}
        if len(set(dates.values())) != 1:
            raise HoldoutError(f"{market.id}: holdout windows are misaligned")
        sharpe = {}
        for model, frame in windowed.items():
            row = _metric_row(frame, config)
            sharpe[model] = row["sharpe"]
            rows.append(
                {"market": market.id, "model": model, "window": "holdout", **row}
            )
        for model, frame in trades.items():
            rows.append(
                {
                    "market": market.id,
                    "model": model,
                    "window": "full",
                    **_metric_row(frame, config),
                }
            )
        stronger = max(("buy_and_hold", "hmm"), key=lambda name: sharpe[name])
        gap = sharpe["dd_only"] - sharpe[stronger]
        bootstrap = bootstrap_sharpe_delta(
            windowed["dd_only"]["strategy_return"].reset_index(drop=True),
            windowed[stronger]["strategy_return"].reset_index(drop=True),
            windowed["dd_only"]["cash_return"].reset_index(drop=True),
            replications=BOOTSTRAP_REPLICATIONS,
            mean_block_length=BOOTSTRAP_MEAN_BLOCK,
            seed=BOOTSTRAP_SEED,
        )
        decisions[market.id] = {
            "stronger_control": stronger,
            "gap_dd_only": gap,
            "gap_fixed_jm": sharpe["fixed_jm"] - sharpe[stronger],
            "delta_vs_fixed": sharpe["dd_only"] - sharpe["fixed_jm"],
            "bootstrap_observed": bootstrap.observed,
            "bootstrap_ci_low": bootstrap.confidence_low,
            "bootstrap_ci_high": bootstrap.confidence_high,
            "bootstrap_lower_one_sided": bootstrap.lower_one_sided,
        }
    passes = sum(values["gap_dd_only"] > 0 for values in decisions.values())
    conclusion = "supported" if passes == len(config.markets) else "not_supported"
    table = pd.DataFrame.from_records(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "holdout-metrics.csv", index=False)
    summary = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "holdout_start": str(holdout_start.date()),
        "holdout_end": str(holdout_end.date()),
        "markets_passing_dd_only": int(passes),
        "conclusion_dd_only": conclusion,
        "per_market": decisions,
        "bootstrap": {
            "method": "paired_stationary_block",
            "replications": BOOTSTRAP_REPLICATIONS,
            "mean_block_length": BOOTSTRAP_MEAN_BLOCK,
            "seed": BOOTSTRAP_SEED,
        },
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    write_json(out_dir / "summary.json", summary)
    render_holdout_figure(out_dir)
    return summary


MODEL_ORDER = ("buy_and_hold", "hmm", "fixed_jm", "dd_only")
MODEL_LABELS = {
    "buy_and_hold": "Buy & hold",
    "hmm": "HMM",
    "fixed_jm": "Fixed JM",
    "dd_only": "DD-only JM",
}
MODEL_COLORS = {
    "buy_and_hold": "#52514e",
    "hmm": "#eda100",
    "fixed_jm": "#eb6834",
    "dd_only": "#2a78d6",
}
MARKET_LABELS = {"us": "US", "de": "Germany", "jp": "Japan"}


def render_holdout_figure(out_dir: Path) -> Path:
    """Render the holdout-window Sharpe comparison from the sealed metrics."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = pd.read_csv(out_dir / "holdout-metrics.csv")
    holdout = table.loc[table["window"] == "holdout"]
    markets = [m for m in ("us", "de", "jp") if m in set(holdout["market"])]
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": 0.55,
            "grid.color": "#d9d9d9",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "text.color": "#0b0b0b",
            "axes.labelcolor": "#52514e",
            "xtick.color": "#52514e",
            "ytick.color": "#52514e",
            "svg.fonttype": "none",
        }
    ):
        figure, axis = plt.subplots(figsize=(9.2, 4.6))
        width = 0.2
        base = range(len(markets))
        for index, model in enumerate(MODEL_ORDER):
            values = [
                float(
                    holdout.loc[
                        (holdout["market"] == market) & (holdout["model"] == model),
                        "sharpe",
                    ].iloc[0]
                )
                for market in markets
            ]
            offsets = [b + (index - 1.5) * width for b in base]
            bars = axis.bar(
                offsets,
                values,
                width,
                color=MODEL_COLORS[model],
                label=MODEL_LABELS[model],
            )
            axis.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
        axis.set_xticks(list(base), [MARKET_LABELS[m] for m in markets])
        axis.set_ylabel("Net Sharpe on the 2024-01 to 2026-06 holdout")
        axis.set_title(
            "One-shot holdout: no jump-model variant beats buy-and-hold\n"
            "on the untouched 2024-2026 window",
            loc="left",
            fontsize=12,
        )
        axis.legend(loc="upper left", frameon=False, fontsize=9, ncols=4)
        axis.margins(y=0.18)
        figure.tight_layout()
        target = out_dir / "holdout-sharpe.png"
        figure.savefig(target)
        plt.close(figure)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="holdout research config")
    parser.add_argument(
        "--baselines-run", required=True, help="completed 2026 fixed-baselines run"
    )
    arguments = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    spec = load_holdout_spec(repo_root)
    config = load_config(arguments.config)
    if config.sha256 != spec["source"]["config_sha256"]:
        raise HoldoutError("config does not match the frozen holdout contract")
    from adaptive_jump.cli import load_frozen_data

    frozen = load_frozen_data(config)
    if frozen.sha256 != spec["source"]["data_manifest_sha256"]:
        raise HoldoutError("data manifest does not match the frozen contract")
    baselines_run = Path(arguments.baselines_run).resolve()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = repo_root / "artifacts" / EXPERIMENT_ID / f"holdout-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    run_dd_only_leg(config, frozen, run_dir)
    summary = readout(
        repo_root,
        config,
        baselines_run,
        run_dir,
        run_dir,
        pd.Timestamp(spec["window"]["holdout_start"]),
        pd.Timestamp(spec["window"]["holdout_end"]),
    )
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "conclusion_dd_only": summary["conclusion_dd_only"],
                "markets_passing": summary["markets_passing_dd_only"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
