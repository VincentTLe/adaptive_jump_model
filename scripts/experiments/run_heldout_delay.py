"""Run the frozen held-out-delay test: research/contracts/heldout-delay-001.toml.

This is a DRIVER, not a scorer. Every number it prints comes from
score_grid.score(), the shared implementation whose known-answer self-test
covers three markets by three delays by eight cells against the sealed run.
Nothing here re-implements selection, execution or a metric; if it did, it would
be the ninth copy of a rule that already exists, which is where this project's
defects have always lived.

The question it answers is stated in the spec and repeated here because a driver
should say what it is for: the lambda grids were chosen by minimising deviation
on Shu's Table 4, a delay-1 table, so their Table-4 deviation is a selection
statistic. Table 5 prints the same strategy at delays 5 and 10, which no search
here has seen. Holding the grid fixed and moving only the delay turns the
selection back into a prediction.
"""

import sys
import tomllib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from _shu_table5 import HELD_OUT, TABLE5_JM, grade  # noqa: E402
from score_grid import score  # noqa: E402

SPEC = ROOT / "research/contracts/heldout-delay-001.toml"
OUT = ROOT / "artifacts/heldout-delay/01-table5"
CELLS = ("cagr", "sharpe", "calmar")

# score() reads the sealed run's own jm-states.csv unless given a file. The
# searched grids were fitted on wider menus, so each arm names where its states
# live; arm A's are the sealed ones and pass None.
STATES = {
    "us": ROOT / "artifacts/jm-residual/01-grid-identification/us/union-states.csv",
    "de": ROOT / "artifacts/dense-menu/01-search/states-de.csv",
    "jp": ROOT / "artifacts/dense-menu/01-search/states-jp.csv",
}


def main() -> int:
    spec = tomllib.loads(SPEC.read_text())
    if spec["status"] != "FROZEN_BEFORE_RESULTS":
        raise SystemExit(f"spec is {spec['status']}, refusing to run")
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for arm_id, arm in spec["arms"].items():
        for market in spec["sources"]["markets"]:
            grid = tuple(arm[market])
            states = None if arm_id.startswith("A_") else STATES[market]
            for delay in (spec["protocol"]["in_sample_delay"], *HELD_OUT):
                got = score(market, grid, states, delay=delay)
                target = TABLE5_JM[market][delay]
                for cell in CELLS:
                    rel = abs(got[cell] - target[cell]) / abs(target[cell])
                    rows.append(
                        {
                            "arm": arm_id,
                            "market": market,
                            "delay": delay,
                            "held_out": delay in HELD_OUT,
                            "cell": cell,
                            "ours": got[cell],
                            "shu": target[cell],
                            "relative_deviation": rel,
                            "grade": grade(rel),
                        }
                    )
                print(
                    f"{arm_id:<18} {market} delay {delay:>2} "
                    + "  ".join(
                        f"{c} {got[c]:.4f}/{target[c]:.2f}" for c in CELLS
                    ),
                    flush=True,
                )

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "cells.csv", index=False)

    held = frame[frame.held_out]
    summary = (
        held.groupby(["arm", "market"])["relative_deviation"].max().reset_index()
    )
    summary["grade"] = summary.relative_deviation.map(grade)
    summary["worst_cell"] = [
        held[(held.arm == r.arm) & (held.market == r.market)]
        .sort_values("relative_deviation")
        .iloc[-1]
        .pipe(lambda x: f"delay {x.delay} {x.cell}")
        for r in summary.itertuples()
    ]
    summary.to_csv(OUT / "heldout-summary.csv", index=False)

    print("\nHELD OUT (delays 5 and 10, six cells per market, never searched)")
    print(summary.to_string(index=False))
    print(f"\nwrote {OUT}/cells.csv and {OUT}/heldout-summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
