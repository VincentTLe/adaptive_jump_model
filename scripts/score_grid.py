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

# Table 5 publishes the same strategy at delays 5 and 10, and no grid search has
# ever seen those columns. score() therefore takes the delay as an argument.
# The hazard this reopens is the one an audit already caught once: DELAY used to
# key BOTH the computed path and the sealed row it was checked against, so a
# wrong delay moved the answer and its own known answer together and passed.
# The rules that keep it shut, enforced in self_test():
#   - the Table-4 known answer is checked at delay 1 and only at delay 1;
#   - every delay is checked against the sealed row for THAT delay; and
#   - the delays are asserted to disagree with each other, so a delay argument
#     that silently fails to propagate cannot pass as a match.
DELAYS = (1, 5, 10)


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


def score(
    market: str,
    penalties,
    states_csv: Path | None = None,
    delay: int = DELAY,
) -> dict:
    """The eight cells for one grid at one trading delay, sealed pipeline.

    The delay enters BOTH the cross-validation that picks the penalty each month
    and the execution of the resulting signal, which is what the Table 5 caption
    requires at line 979: "A corresponding trading delay is also applied in the
    cross-validation when selecting the optimal smoothing hyperparameter."
    """
    if delay not in DELAYS:
        raise SystemExit(f"delay {delay!r} is not published; Table 5 has {DELAYS}")
    config = load_config(ROOT / "configs/baselines/legacy/research-calibrated-v10.toml")
    frame = pd.read_csv(BASE / market / "features.csv", parse_dates=["date"])
    source = states_csv or (BASE / market / "jm-states.csv")
    states = pd.read_csv(source, index_col=0, parse_dates=[0])
    states.columns = [float(c) for c in states.columns]
    columns = resolve_columns(states, penalties, f"{market}: {source.name}")
    selection = select_monthly_candidate(
        frame[["date", "equity_simple", "cash_return"]],
        states.loc[:, columns],
        config.selection_protocol,
        delay_trading_days=delay,
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
        delay_trading_days=delay,
        one_way_cost_bps=COST,
    )
    reported = pd.read_csv(BASE / "metrics.csv", parse_dates=["start", "end"])
    row = reported[
        (reported.market == market)
        & (reported.model == "fixed_jm")
        & (reported.delay == delay)
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
    config = load_config(ROOT / "configs/baselines/legacy/research-calibrated-v10.toml")
    reported = pd.read_csv(BASE / "metrics.csv")
    worst = 0.0
    for market in ("us", "de", "jp"):
        grid = config.jm_protocol_for(market).lambda_grid
        by_delay = {}
        for delay in DELAYS:
            got = score(market, grid, delay=delay)
            by_delay[delay] = got
            row = reported[
                (reported.market == market)
                & (reported.model == "fixed_jm")
                & (reported.delay == delay)
            ].iloc[0]
            for cell in CELLS:
                gap = abs(float(got[cell]) - float(row[cell]))
                # `gap > tol` is False when gap is NaN, and max(0.0, nan) is
                # 0.0, so an all-NaN score used to print "SELF TEST PASSED
                # (0.00e+00)". Both comparisons are written so NaN fails.
                if not (gap <= 1e-9):
                    raise SystemExit(
                        f"SELF TEST FAILED: {market} delay {delay} {cell} "
                        f"{got[cell]!r} vs sealed {row[cell]!r}"
                    )
                worst = max(worst, gap)
        # A delay argument that never reaches the pipeline would agree with the
        # sealed rows only if those rows were themselves identical. They are
        # not, so requiring the delays to disagree turns "the argument is
        # ignored" from a passing test into a failing one.
        for a, b in ((1, 5), (5, 10), (1, 10)):
            if abs(by_delay[a]["sharpe"] - by_delay[b]["sharpe"]) <= 1e-9:
                raise SystemExit(
                    f"SELF TEST FAILED: {market} delays {a} and {b} give the "
                    f"same sharpe; the delay argument is not reaching the run"
                )
        print(
            f"  self test {market}: 8 cells x 3 delays reproduce the sealed run",
            flush=True,
        )

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


def targets(market: str, delay: int) -> dict:
    """The published cells for this market and delay, and where they came from.

    Delay 1 is Table 4, eight cells, and it is the table every grid search in
    this project optimised against. Delays 5 and 10 are Table 5, three cells,
    and no search has ever seen them: they are the held-out set that decides
    whether a selected grid is a measurement or an artefact of the search.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    if delay == 1:
        from _shu_table4 import TABLE4  # noqa: PLC0415

        return TABLE4[market]["fixed_jm"]
    from _shu_table5 import TABLE5_JM  # noqa: PLC0415

    return TABLE5_JM[market][delay]


def main() -> int:
    flags = [a for a in sys.argv[1:] if a.startswith("--delay")]
    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    delay = int(flags[0].split("=")[1]) if flags else DELAY
    if not positional or "--self-test" in sys.argv[1:]:
        self_test()
        return 0
    sys.path.insert(0, str(ROOT / "scripts"))
    from _shu_table5 import grade  # noqa: PLC0415

    market = positional[0]
    penalties = [float(v) for v in positional[1].split(",")]
    states = Path(positional[2]) if len(positional) > 2 else None
    got = score(market, penalties, states, delay=delay)
    target = targets(market, delay)
    source = "Table 4" if delay == 1 else "Table 5 (HELD OUT)"
    worst = 0.0
    print(f"{market.upper()}  delay {delay}  {source}  grid {positional[1]}")
    for cell in target:
        rel = abs(got[cell] - target[cell]) / max(abs(target[cell]), 1e-9)
        # NaN must propagate into the headline. `max(0.0, nan)` is 0.0, so the
        # obvious accumulation reported an uncomputable cell as a perfect match.
        worst = rel if not np.isfinite(worst) or rel > worst else worst
        if not np.isfinite(rel):
            worst = float("nan")
        print(
            f"   {cell:<24}{got[cell]:>9.3f}  Shu {target[cell]:>7.3f}  "
            f"({rel:>5.1%})  {grade(rel)}"
        )
    print(
        f"   switches {got['switches']}   worst relative deviation {worst:.1%}"
        f"   GRADE {grade(worst)}"
    )
    if not np.isfinite(worst):
        print("   REFUSING: at least one cell is not finite", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
