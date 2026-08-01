"""The one way to score a lambda grid against the published table.

Four throwaway validation scripts were written in a single session, each
re-implementing this, and defects appeared in the copies rather than in the
pipeline. This is the shared implementation. It has one job: given a market and
a set of penalties, return the eight Table-4 cells and their deviations, using
the sealed pipeline's own selection, execution and metric code.

self_test() scores the grid the sealed v10 run actually used and asserts the
result reproduces that run's committed metrics.csv. That is a KNOWN ANSWER, so
a broken scorer fails here instead of in a result.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_jump.backtest import apply_signal, performance_metrics  # noqa: E402
from adaptive_jump.config import load_config  # noqa: E402
from adaptive_jump.walkforward import select_monthly_candidate  # noqa: E402

BASE = (
    ROOT / "artifacts/fixed-baselines/"
    "fixed-baselines-36ca1ace131c-ed7abd7daea3-f9f3e0a93736"
)
CELLS = (
    "cagr", "volatility", "sharpe", "maximum_drawdown", "calmar",
    "expected_shortfall_5pct", "turnover", "leverage",
)
COST, DELAY = 10.0, 1


def score(market: str, penalties, states_csv: Path | None = None) -> dict:
    """The eight cells for one grid, through the sealed pipeline."""
    config = load_config(ROOT / "research-calibrated-v10.toml")
    frame = pd.read_csv(BASE / market / "features.csv", parse_dates=["date"])
    source = states_csv or (BASE / market / "jm-states.csv")
    states = pd.read_csv(source, index_col=0, parse_dates=[0])
    states.columns = [float(c) for c in states.columns]
    columns = []
    for value in penalties:
        matches = [c for c in states.columns if np.isclose(c, float(value),
                                                           rtol=1e-9, atol=1e-9)]
        if len(matches) != 1:
            raise SystemExit(
                f"{market}: penalty {value!r} is not a column of {source.name}"
            )
        columns.append(matches[0])
    selection = select_monthly_candidate(
        frame[["date", "equity_simple", "cash_return"]],
        states.loc[:, columns],
        config.selection_protocol,
        delay_trading_days=DELAY,
        one_way_cost_bps=COST,
        periods_per_year=config.metrics_protocol.periods_per_year,
        volatility_ddof=config.metrics_protocol.volatility_ddof,
    )
    signal = selection.signal.rename("s").reset_index()
    signal.columns = ["date", "s"]
    merged = frame.merge(signal, on="date", how="left")
    path = apply_signal(
        merged[["date", "equity_simple", "cash_return"]],
        merged["s"],
        delay_trading_days=DELAY,
        one_way_cost_bps=COST,
    )
    reported = pd.read_csv(BASE / "metrics.csv", parse_dates=["start", "end"])
    row = reported[
        (reported.market == market)
        & (reported.model == "fixed_jm")
        & (reported.delay == DELAY)
    ].iloc[0]
    kept = path[
        (path["date"] >= row["start"]) & (path["date"] <= row["end"])
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


def self_test() -> None:
    """Score the grid the sealed run used; it must reproduce that run's metrics."""
    config = load_config(ROOT / "research-calibrated-v10.toml")
    reported = pd.read_csv(BASE / "metrics.csv")
    worst = 0.0
    for market in ("us", "de", "jp"):
        grid = config.jm_protocol_for(market).lambda_grid
        got = score(market, grid)
        row = reported[
            (reported.market == market)
            & (reported.model == "fixed_jm")
            & (reported.delay == DELAY)
        ].iloc[0]
        for cell in CELLS:
            gap = abs(float(got[cell]) - float(row[cell]))
            worst = max(worst, gap)
            if gap > 1e-9:
                raise SystemExit(
                    f"SELF TEST FAILED: {market} {cell} {got[cell]!r} vs sealed "
                    f"{row[cell]!r}"
                )
        print(f"  self test {market}: 8 cells reproduce the sealed run", flush=True)
    print(f"SELF TEST PASSED (worst absolute difference {worst:.2e})", flush=True)


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] == "--self-test":
        self_test()
        return 0
    from _shu_table4 import TABLE4  # noqa: PLC0415

    sys.path.insert(0, str(ROOT / "scripts"))
    market = sys.argv[1]
    penalties = [float(v) for v in sys.argv[2].split(",")]
    states = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    got = score(market, penalties, states)
    target = TABLE4[market]["fixed_jm"]
    worst = 0.0
    print(f"{market.upper()}  grid {sys.argv[2]}")
    for cell in CELLS:
        rel = abs(got[cell] - target[cell]) / max(abs(target[cell]), 1e-9)
        worst = max(worst, rel)
        print(
            f"   {cell:<24}{got[cell]:>9.3f}  Shu {target[cell]:>7.3f}  ({rel:>5.1%})"
        )
    print(f"   switches {got['switches']}   worst relative deviation {worst:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
