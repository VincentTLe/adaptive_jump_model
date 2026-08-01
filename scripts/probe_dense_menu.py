"""Denser lambda menu for Germany and Japan, then a greedy minimax search.

The 29-value menu was assembled from published and third-party grids. It is
sparse: the winning German grid uses five values below 40 and the space between
them is empty. This generates states on a dense log-spaced menu over the same
range, then searches it for the grid minimising the WORST RELATIVE deviation
across the eight published cells.

The menu spans the full range rather than the region the previous winner
occupied, so no region is privileged by having been seen to work. This remains
a calibration search against the published table and carries the same label.
"""

import itertools
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from multiprocessing import get_context  # noqa: E402

from _shu_table4 import TABLE4  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from adaptive_jump.backtest import apply_signal, performance_metrics  # noqa: E402
from adaptive_jump.config import load_config  # noqa: E402
from adaptive_jump.models import fixed_jm_states  # noqa: E402
from adaptive_jump.walkforward import select_monthly_candidate  # noqa: E402

_WORKER: dict = {}


def _init_worker(market, config, frame, states, window, target) -> None:
    """Load the shared inputs once per process instead of once per task."""
    _WORKER.update(
        market=market,
        config=config,
        frame=frame,
        states=states,
        window=window,
        target=target,
    )


def _score_task(grid):
    with threadpool_limits(limits=1):
        scored = score_grid(
            _WORKER["config"],
            _WORKER["market"],
            _WORKER["frame"],
            _WORKER["states"],
            grid,
            _WORKER["window"],
        )
        return grid, worst_relative(scored, _WORKER["target"])

BASE = (
    ROOT / "artifacts/fixed-baselines/"
    "fixed-baselines-36ca1ace131c-ed7abd7daea3-f9f3e0a93736"
)
OUT = ROOT / "artifacts/dense-menu/01-search"
METRICS = (
    "cagr", "volatility", "sharpe", "maximum_drawdown", "calmar",
    "expected_shortfall_5pct", "turnover", "leverage",
)


def dense_menu() -> tuple[float, ...]:
    """Zero plus 47 log-spaced values from 0.5 to 1000, the menu's own range."""
    return (0.0, *tuple(float(v) for v in np.logspace(np.log10(0.5), 3.0, 47)))


def worst_relative(scored: dict, target: dict) -> float:
    return max(
        abs(scored[m] - target[m]) / max(abs(target[m]), 1e-9) for m in METRICS
    )


def score_grid(config, market, frame, states, grid, window) -> dict:
    selection = select_monthly_candidate(
        frame[["date", "equity_simple", "cash_return"]],
        states.loc[:, list(grid)],
        config.selection_protocol,
        delay_trading_days=1,
        one_way_cost_bps=10.0,
        periods_per_year=config.metrics_protocol.periods_per_year,
        volatility_ddof=config.metrics_protocol.volatility_ddof,
    )
    signal = selection.signal.rename("s").reset_index()
    signal.columns = ["date", "s"]
    merged = frame.merge(signal, on="date", how="left")
    path = apply_signal(
        merged[["date", "equity_simple", "cash_return"]],
        merged["s"],
        delay_trading_days=1,
        one_way_cost_bps=10.0,
    )
    kept = path[
        (path["date"] >= window[0]) & (path["date"] <= window[1])
    ].dropna(
        subset=["cash_return", "position", "one_way_turnover", "strategy_return"]
    )
    scored = performance_metrics(
        kept,
        periods_per_year=config.metrics_protocol.periods_per_year,
        volatility_ddof=config.metrics_protocol.volatility_ddof,
        expected_shortfall_quantile=(
            config.metrics_protocol.expected_shortfall_quantile
        ),
        turnover_scale=config.metrics_protocol.turnover_scale,
        drawdown_basis="total_wealth",
    )
    scored["switches"] = int((kept["position"].diff().abs() > 0).sum())
    return scored


def main() -> int:
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 28
    markets = sys.argv[2].split(",") if len(sys.argv) > 2 else ["de", "jp"]
    OUT.mkdir(parents=True, exist_ok=True)
    config = load_config(ROOT / "research-calibrated-v10.toml")
    menu = dense_menu()
    reported = pd.read_csv(BASE / "metrics.csv", parse_dates=["start", "end"])

    rows = []
    for market in markets:
        frame = pd.read_csv(BASE / market / "features.csv", parse_dates=["date"])
        cache = OUT / f"states-{market}.csv"
        if cache.exists():
            states = pd.read_csv(cache, index_col=0, parse_dates=[0])
            states.columns = [float(c) for c in states.columns]
            print(f"{market}: reusing cached states", flush=True)
        else:
            print(f"{market}: fitting {len(menu)} penalties ...", flush=True)
            fitted = fixed_jm_states(
                frame,
                config.model_protocol,
                replace(config.jm_protocol, lambda_grid=menu),
                n_jobs=n_jobs,
            )
            states = fitted.states
            states.to_csv(cache)
        row = reported[
            (reported.market == market)
            & (reported.model == "fixed_jm")
            & (reported.delay == 1)
        ].iloc[0]
        window = (row["start"], row["end"])
        target = TABLE4[market]["fixed_jm"]

        # Seed with the best PAIR, exhaustively. Greedy cannot bootstrap from
        # an empty set here because a one-element grid gives the monthly
        # selection nothing to choose between, so the first move has to be a
        # pair and there are few enough of them to enumerate.
        # Every grid is scored independently, so the search fans out. The state
        # matrix is loaded once per worker rather than pickled per task.
        executor = ProcessPoolExecutor(
            max_workers=n_jobs,
            mp_context=get_context("forkserver"),
            initializer=_init_worker,
            initargs=(market, config, frame, states, window, target),
        )
        try:
            pairs = list(itertools.combinations(menu, 2))
            print(f"  {market}: scoring {len(pairs)} seed pairs ...", flush=True)
            seeded = list(executor.map(_score_task, pairs, chunksize=8))
            best_pair, best_score = min(seeded, key=lambda row: row[1])
            chosen = list(best_pair)
            print(
                f"  {market} seed: {sorted(chosen)} -> worst {best_score:.4f}",
                flush=True,
            )
            for step in range(6):
                trials = [
                    tuple(sorted([*chosen, value]))
                    for value in menu
                    if value not in chosen
                ]
                scored = list(executor.map(_score_task, trials, chunksize=4))
                trial, score = min(scored, key=lambda row: row[1])
                if score >= best_score - 1e-12:
                    print(f"  {market}: no further improvement", flush=True)
                    break
                added = next(v for v in trial if v not in chosen)
                chosen, best_score = list(trial), score
                print(
                    f"  {market} step {step + 1}: added {added:g} -> worst "
                    f"{best_score:.4f}  grid {sorted(chosen)}",
                    flush=True,
                )
        finally:
            executor.shutdown()
        final = tuple(sorted(chosen))
        scored = score_grid(config, market, frame, states, final, window)
        rows.append(
            {
                "market": market,
                "grid": "|".join(f"{v:g}" for v in final),
                "worst_relative_deviation": best_score,
                "switches": scored["switches"],
                **{m: scored[m] for m in METRICS},
            }
        )
        print(f"{market}: FINAL worst {best_score:.4f} with {final}\n", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "dense-grids.csv", index=False)
    print("wrote", OUT / "dense-grids.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
