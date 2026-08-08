"""Direct Table-4 test of one new legitimately-sourced named grid.

Not a modification of jm-grid-identification-001 (frozen 2026-07-30) --
that spec's own sealed union/GRIDS/artifacts are untouched. This is a
narrow follow-up, registered separately: test ONE more named candidate
grid found via literature reading, using -001's own established, already-
approved methodology (evaluate() against Table 4, no search, no tuning,
winner_selection_allowed = false in spirit exactly as -001 states it).

Grid provenance: Luo & Mulvey (2026), "Regime-Aware Asset Allocation with
Dual-Regime Signals and Regime-Dependent Asset Selection" (SSRN 6933278),
page 13: "Algorithm 1 is evaluated over a grid of candidate jump penalties
lambda in [1, 100] (evenly spaced by 10) on the validation period" -- a
different application (12-asset US allocation, not the 3-market equity-
index replication target) but a real, disclosed, real-market grid from
the same lab. Read in full 2026-08-07 (workflow wg503b7pa); this is the
only grid among 7 companion-lab papers surveyed that (a) is a concrete
enumerable list and (b) was applied to real market data, not synthetic.

Frozen rule (registered in the registry NOTE before this ran): report the
same within_tol count and per-cell deviations -001 already reports for its
8 named grids, no new tolerance, no new estimand. Does NOT touch the
sealed jm-grid-identification-001 union, GRIDS dict, or artifacts.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402
from _shu_table4 import METRICS, TABLE4  # noqa: E402
from probe_jm_grid_identification import evaluate  # noqa: E402

from adaptive_jump.config import load_config  # noqa: E402
from adaptive_jump.models import fixed_jm_states  # noqa: E402

# v10's run directory substitutes for the deleted v9.4-hash run, per the
# same reasoning and verification already applied in
# probe_jm_grid_exhaustive2.py (features.csv byte-identical by reseal
# gate 2; rebuilt caches there reproduced sealed anchors to float
# precision).
RUN = ROOT / "artifacts" / "fixed-baselines" / (
    "fixed-baselines-36ca1ace131c-ed7abd7daea3-f9f3e0a93736"
)
OUT = ROOT / "artifacts" / "jm-residual" / "11-luo-mulvey-grid"
DELAY, COST, TOL, N_JOBS = 1, 10.0, 0.05, 30
NAMES = {"us": "S&P 500", "de": "DAX", "jp": "Nikkei 225"}

GRID_NAME = "luo_mulvey_2026"
GRID: tuple[float, ...] = tuple(float(v) for v in range(1, 100, 10))  # 1,11,...,91


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "research-expanding-v9-4.toml")
    reported = pd.read_csv(RUN / "metrics.csv", parse_dates=["start", "end"])
    print(f"grid {GRID_NAME}: " + "|".join(f"{v:g}" for v in GRID), flush=True)

    records = []
    for market in ("us", "de", "jp"):
        frame = pd.read_csv(RUN / market / "features.csv", parse_dates=["date"])
        # n_init raised from the sealed 10 to 60: these nine lambdas were
        # never fit before, and the standard n_init hit a coordinate-descent
        # local optimum (objective non-monotone lambda=61->71). Matches the
        # project's established remedy for exactly this failure mode (see
        # docs/audit/frequency-ladder-001-audit.md F-1: "refitting that
        # window with six times as many restarts lowers its objective").
        # This changes solver reliability, not the scientific question.
        protocol = dataclasses.replace(cfg.jm_protocol, lambda_grid=GRID, n_init=60)
        result = fixed_jm_states(frame, cfg.model_protocol, protocol, n_jobs=N_JOBS)
        states = result.states
        states.to_csv(OUT / f"{market}-states.csv", lineterminator="\n")

        row = reported[(reported.market == market)
                       & (reported.model == "fixed_jm")
                       & (reported.delay == DELAY)].iloc[0]
        got = evaluate(frame, states, GRID, cfg, row["start"], row["end"])
        target = TABLE4[market]["fixed_jm"]
        within = sum(abs(got[m] - target[m]) <= TOL for m in METRICS)
        records.append({
            "market": market, "grid": GRID_NAME,
            "candidates": "|".join(f"{v:g}" for v in GRID),
            **{m: got[m] for m in METRICS}, "shifts": got["shifts"],
            **{f"dev_{m}": abs(got[m] - target[m]) for m in METRICS},
            "within_tol": within,
        })
        print(
            f"{market} ({NAMES[market]}): sharpe {got['sharpe']:.3f} "
            f"(Shu {target['sharpe']:.2f}) turnover {got['turnover']:.2f} "
            f"(Shu {target['turnover']:.2f}) leverage {got['leverage']:.2f} "
            f"(Shu {target['leverage']:.2f}) shifts {got['shifts']} "
            f"within_tol {within}/8",
            flush=True,
        )

    out = pd.DataFrame.from_records(records)
    out.to_csv(OUT / "table4-cells.csv", index=False, lineterminator="\n")
    print("\nwrote", OUT / "table4-cells.csv")


if __name__ == "__main__":
    raise SystemExit(main())
