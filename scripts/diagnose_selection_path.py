"""What does the monthly cross-validation actually choose, and does the grid bind?

The fixed-penalty sweep answered a question the pipeline does not ask. The
pipeline re-selects the penalty every month from whatever menu it is given, by
trailing validation Sharpe, so the object that decides German turnover is the
SELECTED PATH, not any single penalty.

This prints that path for a given menu: which penalty each month, how often the
choice sits on the menu's top or bottom value, and what turnover the resulting
mixture produces. A choice pinned to an endpoint means the menu is binding - the
cross-validation wanted something outside it and was not offered it - and that
is a different diagnosis from "the menu is in the wrong place", because it is
visible without knowing the right answer.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _shu_table4 import TABLE4  # noqa: E402

from adaptive_jump.config import load_config  # noqa: E402
from adaptive_jump.walkforward import select_monthly_candidate  # noqa: E402

BASE = (
    ROOT / "artifacts/fixed-baselines/"
    "fixed-baselines-36ca1ace131c-ed7abd7daea3-f9f3e0a93736"
)
UNION = ROOT / "artifacts/jm-residual/01-grid-identification"
OUT = ROOT / "artifacts/selection-path/01-choices"

MENUS = {
    # the authors' own candidate set, published in arXiv v1 Table 3 and dropped
    # from v3: "The jump penalty that leads to the highest Sharpe ratio achieved
    # during the validation period is selected."
    "authors_v1": (10.0, 22.0, 50.0, 100.0, 220.0, 500.0, 1000.0),
    # everything that has been fitted, so the choice is as unconstrained as this
    # project can make it. Not a candidate grid - a probe of what the CV wants.
    "widest_fitted": None,
}


def main() -> int:
    markets = sys.argv[1].split(",") if len(sys.argv) > 1 else ["us", "de", "jp"]
    OUT.mkdir(parents=True, exist_ok=True)
    config = load_config(ROOT / "research-calibrated-v10.toml")
    rows = []
    for market in markets:
        frame = pd.read_csv(BASE / market / "features.csv", parse_dates=["date"])
        states = pd.read_csv(
            UNION / market / "union-states.csv", index_col=0, parse_dates=[0]
        )
        states.columns = [float(c) for c in states.columns]
        for menu_name, menu in MENUS.items():
            values = sorted(states.columns) if menu is None else sorted(menu)
            columns = [
                next(c for c in states.columns if abs(c - v) < 1e-9) for v in values
            ]
            result = select_monthly_candidate(
                frame[["date", "equity_simple", "cash_return"]],
                states.loc[:, columns],
                config.selection_protocol,
                delay_trading_days=1,
                one_way_cost_bps=10.0,
                periods_per_year=config.metrics_protocol.periods_per_year,
                volatility_ddof=config.metrics_protocol.volatility_ddof,
            )
            chosen = result.choices["selected"].astype(float)
            lo, hi = min(values), max(values)
            at_lo = float((chosen <= lo + 1e-9).mean())
            at_hi = float((chosen >= hi - 1e-9).mean())
            counts = chosen.value_counts().sort_index()
            rows.append(
                {"market": market, "menu": menu_name, "size": len(values),
                 "months": len(chosen), "at_bottom": at_lo, "at_top": at_hi,
                 "median_choice": float(chosen.median()),
                 "distinct_chosen": int(chosen.nunique())}
            )
            shu = TABLE4[market]["fixed_jm"]["turnover"]
            print(
                f"\n{market.upper()}  menu {menu_name} ({len(values)} values, "
                f"{lo:g} to {hi:g})  Shu turnover {shu:.2f}"
            )
            print(f"   months {len(chosen)}   chose the BOTTOM value {at_lo:.1%} "
                  f"of months, the TOP value {at_hi:.1%}, median {chosen.median():g}")
            print("   " + "  ".join(
                f"{v:g}:{int(n)}" for v, n in counts.items() if n
            ))
            result.choices.assign(market=market, menu=menu_name).to_csv(
                OUT / f"choices-{market}-{menu_name}.csv", index=False
            )
    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
    print(f"\nwrote {OUT}/summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
