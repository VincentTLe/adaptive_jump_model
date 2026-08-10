"""The penalty-to-frequency map, free, from refit objectives already on disk.

The jump model minimises, over state sequences,

    L(lambda) = min_S [ sum_t 0.5||x_t - theta_{s_t}||^2 + lambda * jumps(S) ]

which is a minimum of affine functions of lambda, hence CONCAVE and piecewise
linear, with slope equal to jumps(S*) at the optimum. Two consequences, both
free:

  * the number of jumps in a training window at penalty lambda is the slope of
    the recorded objective, so it needs no refit and no state decoding; and
  * the fit degenerates to a single state exactly where that slope reaches zero,
    i.e. where the objective goes flat, so the degeneracy bound is READ OFF
    rather than searched for.

Inverting the slope gives the penalty required for a target shift frequency in
the training window. Every quantity comes from the trailing 3000 observations
that the refit already used, so the map is causal and can be recomputed inside
the walk-forward without looking ahead.

This script reports the map. It selects nothing and uses no published number.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
UNION = ROOT / "artifacts/jm-residual/01-grid-identification"
OUT = ROOT / "artifacts/penalty-frequency/01-map"

TRAINING_OBSERVATIONS = 3000
PERIODS_PER_YEAR = 252
TRAINING_YEARS = TRAINING_OBSERVATIONS / PERIODS_PER_YEAR
TARGETS = (8.0, 4.0, 2.0, 1.0, 0.5, 0.25)


def suboptimal_windows(refits: pd.DataFrame) -> pd.DataFrame:
    """Windows where the recorded objective FALLS as the penalty rises.

    L(lambda) is a minimum of affine functions with non-negative slopes, so it
    cannot decrease. Where it does, the fit at the smaller penalty failed to
    reach the optimum the larger penalty found: taking the larger penalty's
    state sequence and charging it the smaller penalty gives a strictly better
    objective. The drop is a LOWER BOUND on how suboptimal that fit is.

    The check costs nothing and needs no refit, so it runs on every window.
    """
    rows = []
    for fit_date, group in refits.groupby("fit_date"):
        group = group.sort_values("lambda")
        lam = group["lambda"].to_numpy(float)
        obj = group["objective"].to_numpy(float)
        steps = np.diff(obj)
        if (steps < -1e-9).any():
            worst = int(steps.argmin())
            rows.append(
                {
                    "fit_date": fit_date,
                    "lambda_suboptimal": lam[worst],
                    "lambda_better": lam[worst + 1],
                    "objective_suboptimal": obj[worst],
                    "objective_better": obj[worst + 1],
                    "shortfall": -steps[worst],
                    "shortfall_fraction": -steps[worst] / obj[worst],
                }
            )
    return pd.DataFrame(rows)


def window_table(refits: pd.DataFrame) -> pd.DataFrame:
    """Per training window: the loss gap and the penalty that degenerates it."""
    rows = []
    for fit_date, group in refits.groupby("fit_date"):
        group = group.sort_values("lambda")
        obj = group["objective"].to_numpy(float)
        lam = group["lambda"].to_numpy(float)
        one_state = obj.max()
        rows.append(
            {
                "fit_date": fit_date,
                "loss_kmeans": obj[0],
                "loss_one_state": one_state,
                "loss_gap": one_state - obj[0],
                # first penalty at which the objective is already flat; censored
                # from above by the menu, so report the menu top alongside it
                "lambda_degenerate": float(
                    lam[np.isclose(obj, one_state, rtol=1e-12)].min()
                ),
                "menu_top": lam[-1],
            }
        )
    return pd.DataFrame(rows)


def frequency_map(refits: pd.DataFrame) -> pd.DataFrame:
    """Per window, the penalty at which the jump rate first falls to each target."""
    rows = []
    for fit_date, group in refits.groupby("fit_date"):
        group = group.sort_values("lambda")
        lam = group["lambda"].to_numpy(float)
        obj = group["objective"].to_numpy(float)
        midpoints = (lam[:-1] + lam[1:]) / 2.0
        # slope of a concave piecewise-linear function = jumps at the optimum
        per_year = np.diff(obj) / np.diff(lam) / TRAINING_YEARS
        row = {"fit_date": fit_date}
        for target in TARGETS:
            reached = np.flatnonzero(per_year <= target)
            row[target] = midpoints[reached[0]] if reached.size else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    markets = sys.argv[1].split(",") if len(sys.argv) > 1 else ["us", "de", "jp"]
    OUT.mkdir(parents=True, exist_ok=True)
    medians, summaries = {}, []
    for market in markets:
        refits = pd.read_csv(UNION / market / "union-refits.csv")
        broken = suboptimal_windows(refits)
        if not broken.empty:
            broken.assign(market=market).to_csv(
                OUT / f"suboptimal-fits-{market}.csv", index=False
            )
            worst = broken.sort_values("shortfall_fraction").iloc[-1]
            print(
                f"{market.upper():>3}  WARNING: {len(broken)} of "
                f"{refits.fit_date.nunique()} windows have a DECREASING "
                f"objective, which is impossible at the optimum; the worst is "
                f"{worst.fit_date} where lambda {worst.lambda_suboptimal:g} "
                f"scores {worst.shortfall:.1f} "
                f"({worst.shortfall_fraction:.2%}) worse than lambda "
                f"{worst.lambda_better:g} already proves is reachable"
            )
        windows = window_table(refits)
        mapping = frequency_map(refits)
        windows.assign(market=market).to_csv(
            OUT / f"windows-{market}.csv", index=False
        )
        mapping.assign(market=market).to_csv(
            OUT / f"frequency-map-{market}.csv", index=False
        )
        medians[market] = {t: float(mapping[t].median()) for t in TARGETS}
        summaries.append(
            {
                "market": market,
                "windows": len(windows),
                "loss_gap_median": windows.loss_gap.median(),
                "lambda_degenerate_median": windows.lambda_degenerate.median(),
                "lambda_degenerate_min": windows.lambda_degenerate.min(),
                "menu_top": windows.menu_top.iloc[0],
            }
        )
        print(
            f"{market.upper():>3}  {len(windows)} windows   "
            f"loss gap median {windows.loss_gap.median():8.1f}   "
            f"degenerates at median {windows.lambda_degenerate.median():7.1f} "
            f"(min {windows.lambda_degenerate.min():6.1f}, menu top "
            f"{windows.menu_top.iloc[0]:g})"
        )

    print("\npenalty required for a target IN-SAMPLE shift frequency "
          "(median over windows)")
    print(f"{'shifts/yr':>10} " + "".join(f"{m.upper():>10}" for m in markets))
    for target in TARGETS:
        print(f"{target:>10} " + "".join(
            f"{medians[m][target]:>10.2f}" for m in markets
        ))
    if "us" in medians:
        print("\nratio to the US at each frequency")
        for target in TARGETS:
            ratios = "   ".join(
                f"{m}/us {medians[m][target] / medians['us'][target]:5.2f}"
                for m in markets if m != "us"
            )
            print(f"{target:>10}   {ratios}")

    frame = pd.DataFrame(
        [{"market": m, "shifts_per_year": t, "lambda_median": medians[m][t]}
         for m in markets for t in TARGETS]
    )
    frame.to_csv(OUT / "median-map.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT / "summary.csv", index=False)
    print(f"\nwrote {OUT}/median-map.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
