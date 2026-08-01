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

from _shu_table4 import TABLE4  # noqa: E402

from adaptive_jump.backtest import apply_signal, performance_metrics  # noqa: E402
from adaptive_jump.config import load_config  # noqa: E402
from adaptive_jump.models import fixed_jm_states  # noqa: E402
from adaptive_jump.walkforward import select_monthly_candidate  # noqa: E402

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
        chosen: list[float] = []
        best_score = np.inf
        for left, right in itertools.combinations(menu, 2):
            score = worst_relative(
                score_grid(config, market, frame, states, (left, right), window),
                target,
            )
            if score < best_score - 1e-12:
                best_score, chosen = score, [left, right]
        if not chosen:
            raise SystemExit(f"{market}: no admissible pair in the dense menu")
        print(
            f"  {market} seed: {sorted(chosen)} -> worst {best_score:.4f}",
            flush=True,
        )
        for step in range(6):
            candidate, candidate_score = None, best_score
            for value in menu:
                if value in chosen:
                    continue
                trial = tuple(sorted([*chosen, value]))
                if len(trial) < 2:
                    continue
                score = worst_relative(
                    score_grid(config, market, frame, states, trial, window), target
                )
                if score < candidate_score - 1e-12:
                    candidate, candidate_score = value, score
            if candidate is None:
                break
            chosen.append(candidate)
            best_score = candidate_score
            print(
                f"  {market} step {step + 1}: added {candidate:g} -> worst "
                f"{best_score:.4f}  grid {sorted(chosen)}",
                flush=True,
            )
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
