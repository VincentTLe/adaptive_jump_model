"""How often does the model trade, as a function of the jump penalty?

Germany is the one market whose published cells we do not reproduce: Shu's DAX
JM turns over 170 percent per year and ours turns over 26 percent, while our US
and Japanese turnover match to within 3 percent. The v10 German grid is {150,
500}, two very large penalties, so the model almost never leaves the state it is
in. The question this answers is not "which grid scores best" - that is the
search that AGENTS.md section 7.3 forbids quoting - but the prior question,
which is diagnostic and has one answer per market:

    holding lambda FIXED, what does the strategy's turnover look like across the
    whole menu, and is there any lambda at which the DAX turns over 170 percent?

If no lambda produces the published turnover, the gap is not a grid problem and
no grid search can fix it. If some lambda does, the curve also shows what that
choice costs on the other seven cells. Either answer is useful; only one of them
is a grid.

This reports the WHOLE curve. It selects nothing.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from _shu_table4 import TABLE4  # noqa: E402
from score_grid import score  # noqa: E402

OUT = ROOT / "artifacts/lambda-turnover/01-curve"
CELLS = (
    "cagr", "volatility", "sharpe", "maximum_drawdown", "calmar",
    "expected_shortfall_5pct", "turnover", "leverage",
)
# The menus already fitted by earlier work; no new fits are needed because the
# state sequence for a given penalty does not depend on anything this asks.
MENUS = {
    "de": ROOT / "artifacts/dense-menu/01-search/states-de.csv",
    "jp": ROOT / "artifacts/dense-menu/01-search/states-jp.csv",
    "us": ROOT / "artifacts/jm-residual/01-grid-identification/us/union-states.csv",
}


def main() -> int:
    markets = sys.argv[1].split(",") if len(sys.argv) > 1 else ["de", "us", "jp"]
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for market in markets:
        source = MENUS[market]
        penalties = sorted(
            float(c) for c in pd.read_csv(source, index_col=0, nrows=1).columns
        )
        target = TABLE4[market]["fixed_jm"]
        print(f"\n{market.upper()}  {len(penalties)} penalties  "
              f"(Shu turnover {target['turnover']:.2f})", flush=True)
        for value in penalties:
            got = score(market, [value], source)
            rows.append(
                {"market": market, "lam": value, "switches": got["switches"],
                 **{c: got[c] for c in CELLS}}
            )
            print(
                f"   lam {value:>10.4f}  turnover {got['turnover']:>6.3f}  "
                f"switches {got['switches']:>4}  sharpe {got['sharpe']:>6.3f}  "
                f"cagr {got['cagr']:>7.4f}  leverage {got['leverage']:>5.3f}",
                flush=True,
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "curve.csv", index=False)
    print(f"\nwrote {OUT}/curve.csv")

    for market in markets:
        sub = frame[frame.market == market]
        want = TABLE4[market]["fixed_jm"]["turnover"]
        span = (sub.turnover.min(), sub.turnover.max())
        reachable = span[0] <= want <= span[1]
        print(
            f"{market}: turnover spans {span[0]:.3f} to {span[1]:.3f}; "
            f"Shu wants {want:.2f} -> {'reachable' if reachable else 'NOT REACHABLE'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
