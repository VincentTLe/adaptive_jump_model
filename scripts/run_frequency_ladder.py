"""Run the frozen frequency-ladder experiment: research/frequency-ladder-001.toml.

Fits the state sequences for the derived per-market penalties, then scores both
declared arms through the shared scorer. It re-derives the menu from the refit
objectives and REFUSES if what it derives differs from what the spec froze, so a
menu cannot drift between being written down and being run.
"""

import sys
import tomllib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _shu_table4 import TABLE4  # noqa: E402
from _shu_table5 import grade  # noqa: E402
from score_grid import CELLS, score  # noqa: E402

from adaptive_jump.config import load_config  # noqa: E402
from adaptive_jump.models import fixed_jm_states  # noqa: E402

SPEC = ROOT / "research/frequency-ladder-001.toml"
BASE = (
    ROOT / "artifacts/fixed-baselines/"
    "fixed-baselines-36ca1ace131c-ed7abd7daea3-f9f3e0a93736"
)
UNION = ROOT / "artifacts/jm-residual/01-grid-identification"
OUT = ROOT / "artifacts/frequency-ladder/01-run"
TRAINING_YEARS = 3000 / 252
CUTOFF = pd.Timestamp("1990-01-01")


def derive(market: str, ladder) -> list[float]:
    """Re-derive the menu the spec froze, from the refit objectives alone."""
    refits = pd.read_csv(UNION / market / "union-refits.csv")
    refits["fit_date"] = pd.to_datetime(refits["fit_date"])
    pre = refits[refits["fit_date"] < CUTOFF]
    if pre.empty:
        raise SystemExit(f"{market}: no training window before {CUTOFF.date()}")
    if (refits["fit_date"] >= CUTOFF).all():
        raise SystemExit(f"{market}: every window is post-cutoff")
    per_target: dict[float, list[float]] = {t: [] for t in ladder}
    for _, group in pre.groupby("fit_date"):
        group = group.sort_values("lambda")
        lam = group["lambda"].to_numpy(float)
        obj = group["objective"].to_numpy(float)
        midpoints = (lam[:-1] + lam[1:]) / 2.0
        per_year = np.diff(obj) / np.diff(lam) / TRAINING_YEARS
        for target in ladder:
            reached = np.flatnonzero(per_year <= target)
            per_target[target].append(
                midpoints[reached[0]] if reached.size else np.nan
            )
    return [float(np.nanmedian(per_target[t])) for t in ladder]


def main() -> int:
    spec = tomllib.loads(SPEC.read_text())
    if spec["status"] != "FROZEN_BEFORE_RESULTS":
        raise SystemExit(f"spec is {spec['status']}, refusing to run")
    ladder = spec["derivation"]["ladder_shifts_per_year"]
    arms = spec["arms"]
    markets = spec["sources"]["markets"]
    OUT.mkdir(parents=True, exist_ok=True)
    config = load_config(ROOT / "configs/baselines/legacy/research-calibrated-v10.toml")
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 28

    # The frozen full ladder must equal what the objectives say today. If the
    # artifact ever changes under the spec, this stops the run rather than
    # quietly reporting a different menu than the one that was registered.
    for market in markets:
        rederived = derive(market, ladder)
        frozen = arms["L_full_ladder"][market]
        if not np.allclose(rederived, frozen, rtol=1e-9, atol=1e-9):
            raise SystemExit(
                f"{market}: the spec froze {frozen} but the refit objectives now "
                f"derive {rederived}; refusing to run"
            )
        print(f"  {market}: derived menu matches the frozen spec {frozen}", flush=True)

    rows = []
    for market in markets:
        needed = sorted({
            float(v) for arm in arms.values() for v in arm[market]
        })
        cache = OUT / f"states-{market}.csv"
        if cache.exists():
            states = pd.read_csv(cache, index_col=0, parse_dates=[0])
            states.columns = [float(c) for c in states.columns]
            print(f"{market}: reusing {len(states.columns)} fitted penalties",
                  flush=True)
        else:
            print(f"{market}: fitting {len(needed)} penalties {needed} ...",
                  flush=True)
            frame = pd.read_csv(BASE / market / "features.csv", parse_dates=["date"])
            fitted = fixed_jm_states(
                frame,
                config.model_protocol,
                replace(config.jm_protocol, lambda_grid=tuple(needed)),
                n_jobs=n_jobs,
            )
            states = fitted.states
            states.to_csv(cache)
        for arm_id, arm in arms.items():
            menu = [float(v) for v in arm[market]]
            got = score(market, menu, cache)
            target = TABLE4[market]["fixed_jm"]
            worst = 0.0
            print(f"\n{arm_id}  {market.upper()}  menu {menu}")
            for cell in CELLS:
                rel = abs(got[cell] - target[cell]) / abs(target[cell])
                worst = max(worst, rel)
                rows.append({"arm": arm_id, "market": market, "cell": cell,
                             "ours": got[cell], "shu": target[cell],
                             "relative_deviation": rel, "grade": grade(rel)})
                print(f"   {cell:<24}{got[cell]:>9.3f}  Shu {target[cell]:>7.3f}"
                      f"  ({rel:>6.1%})  {grade(rel)}")
            print(f"   switches {got['switches']}   worst {worst:.1%}"
                  f"   GRADE {grade(worst)}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "cells.csv", index=False)
    summary = frame.groupby(["arm", "market"])["relative_deviation"].max().reset_index()
    summary["grade"] = summary.relative_deviation.map(grade)
    summary["worst_cell"] = [
        frame[(frame.arm == r.arm) & (frame.market == r.market)]
        .sort_values("relative_deviation").iloc[-1].cell
        for r in summary.itertuples()
    ]
    summary.to_csv(OUT / "summary.csv", index=False)
    print("\n" + summary.to_string(index=False))
    print(f"\nwrote {OUT}/summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
