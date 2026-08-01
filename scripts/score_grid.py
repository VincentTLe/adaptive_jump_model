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


def resolve_columns(states: pd.DataFrame, penalties, source_name: str) -> list:
    """Exact column match for each penalty, or refuse and name the file.

    Kept separate from score() so the refusal can be tested without going
    through the scorer - an audit found the check unreachable when score() is
    stubbed, which is exactly how the reported numbers get produced.
    """
    columns = []
    for value in penalties:
        matches = [
            c for c in states.columns
            if np.isclose(c, float(value), rtol=1e-9, atol=1e-9)
        ]
        if len(matches) != 1:
            raise SystemExit(
                f"penalty {value!r} is not a column of {source_name}"
            )
        columns.append(matches[0])
    return columns


def score(market: str, penalties, states_csv: Path | None = None) -> dict:
    """The eight cells for one grid, through the sealed pipeline."""
    config = load_config(ROOT / "research-calibrated-v10.toml")
    frame = pd.read_csv(BASE / market / "features.csv", parse_dates=["date"])
    source = states_csv or (BASE / market / "jm-states.csv")
    states = pd.read_csv(source, index_col=0, parse_dates=[0])
    states.columns = [float(c) for c in states.columns]
    columns = resolve_columns(states, penalties, f"{market}: {source.name}")
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
    """Score the grid the sealed run used; it must reproduce that run's metrics.

    DELAY keys both the path and the row it is compared against, so changing it
    used to move the answer and its own known answer together and the test
    passed at a six-day holding delay. It is now pinned: the module must be at
    delay 1, asserted here, because Table 4 is a delay-1 table.
    """
    if DELAY != 1:
        raise SystemExit(
            f"SELF TEST FAILED: DELAY is {DELAY}, but Table 4 is a delay-1 table"
        )
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
            # `gap > tol` is False when gap is NaN, and max(0.0, nan) is 0.0,
            # so an all-NaN score used to print "SELF TEST PASSED (0.00e+00)".
            # Both comparisons are now written so NaN fails.
            if not (gap <= 1e-9):
                raise SystemExit(
                    f"SELF TEST FAILED: {market} {cell} {got[cell]!r} vs sealed "
                    f"{row[cell]!r}"
                )
            worst = max(worst, gap)
        print(f"  self test {market}: 8 cells reproduce the sealed run", flush=True)

    # The reported numbers all come through the states_csv path with grids the
    # sealed run never used, and the check above never touched that path: a
    # matcher that snapped to the nearest column, or a tolerance of 0.5, both
    # passed it. Exercise the path here, on a file whose answer is known
    # because it is bit-identical to the sealed states on the shared penalties.
    union = ROOT / "artifacts/jm-residual/01-grid-identification/us/union-states.csv"
    if union.exists():
        sealed_grid = config.jm_protocol_for("us").lambda_grid
        through_file = score("us", sealed_grid, union)
        direct = score("us", sealed_grid)
        for cell in CELLS:
            gap = abs(float(through_file[cell]) - float(direct[cell]))
            if not (gap <= 1e-9):
                raise SystemExit(
                    f"SELF TEST FAILED: the states_csv path disagrees with the "
                    f"sealed path on {cell}: {through_file[cell]!r} vs {direct[cell]!r}"
                )
        # and the matcher must REFUSE a penalty that is not in the file, rather
        # than snapping to a neighbour
        loaded = pd.read_csv(union, index_col=0, parse_dates=[0], nrows=1)
        loaded.columns = [float(c) for c in loaded.columns]
        missing = float(max(sealed_grid)) + 0.5
        try:
            resolve_columns(loaded, [*sealed_grid, missing], union.name)
        except SystemExit:
            pass
        else:
            raise SystemExit(
                "SELF TEST FAILED: the matcher accepted a penalty that is not a "
                "column; it is snapping instead of matching"
            )
        print("  self test us: states_csv path agrees and the matcher refuses "
              "an absent penalty", flush=True)
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
        # NaN must propagate into the headline. `max(0.0, nan)` is 0.0, so the
        # obvious accumulation reported an uncomputable cell as a perfect match.
        worst = rel if not np.isfinite(worst) or rel > worst else worst
        if not np.isfinite(rel):
            worst = float("nan")
        print(
            f"   {cell:<24}{got[cell]:>9.3f}  Shu {target[cell]:>7.3f}  ({rel:>5.1%})"
        )
    print(f"   switches {got['switches']}   worst relative deviation {worst:.1%}")
    if not np.isfinite(worst):
        print("   REFUSING: at least one cell is not finite", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
