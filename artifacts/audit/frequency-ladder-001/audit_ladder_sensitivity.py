"""AUDIT step 3: SENSITIVITY ANALYSIS ONLY.

How much of the frequency-ladder result is carried by the one free choice, the
ladder itself? Alternative geometric ladders of the same shape are derived,
fitted and scored. THIS IS A SENSITIVITY ANALYSIS. It measures how load-bearing
the free choice is. Its outputs MUST NOT be used to pick a better ladder - that
would be exactly the search the frozen spec exists to forbid (decision_rule 4).
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/tle/adaptive_jump_model")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, "/tmp/claude-1017/-home-tle/69649cec-6fd3-40f9-9e01-42dd56f3559f/scratchpad")

from audit_recompute import BASE, CELLS, SHU, my_derive, my_score  # noqa: E402

from adaptive_jump.config import load_config  # noqa: E402
from adaptive_jump.models import fixed_jm_states  # noqa: E402

OUT = ROOT / "artifacts/audit/frequency-ladder-001"
LADDERS = {
    "frozen_8_to_0.25": [8, 4, 2, 1, 0.5, 0.25],
    "anchor12_12_to_0.375": [12, 6, 3, 1.5, 0.75, 0.375],
    "anchor6_6_to_0.1875": [6, 3, 1.5, 0.75, 0.375, 0.1875],
    "span_12_to_1": [12, 6, 3, 1.5, 1.0],
    "span_6_to_0.5": [6, 3, 1.5, 0.75, 0.5],
    "sqrt2_8_to_0.25_11rungs": [8, 5.657, 4, 2.828, 2, 1.414, 1, 0.707, 0.5,
                                0.354, 0.25],
}


def grade(rel):
    if not (rel >= 0.0):
        return "F"
    for letter, ceiling in (("A", .02), ("B", .20), ("C", .40), ("D", .60)):
        if rel <= ceiling:
            return letter
    return "F"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    menus = {n: {m: my_derive(m, lad) for m in ("us", "de", "jp")}
             for n, lad in LADDERS.items()}
    need = {m: sorted({round(float(v), 12)
                       for n in menus for v in menus[n][m]})
            for m in ("us", "de", "jp")}
    cfg = load_config(ROOT / "research-calibrated-v10.toml")
    for market in ("us", "de", "jp"):
        dest = OUT / f"states-{market}.csv"
        have = pd.read_csv(ROOT / f"artifacts/frequency-ladder/01-run/states-{market}.csv",
                           index_col=0, parse_dates=[0])
        have.columns = [float(c) for c in have.columns]
        if dest.exists():
            cur = pd.read_csv(dest, index_col=0, parse_dates=[0])
            cur.columns = [float(c) for c in cur.columns]
            have = cur
        missing = [v for v in need[market]
                   if not any(abs(c - v) < 1e-9 for c in have.columns)]
        if missing:
            print(f"{market}: fitting {len(missing)} extra penalties {missing}",
                  flush=True)
            frame = pd.read_csv(BASE / market / "features.csv", parse_dates=["date"])
            got = fixed_jm_states(frame, cfg.model_protocol,
                                  replace(cfg.jm_protocol,
                                          lambda_grid=tuple(missing)),
                                  n_jobs=14).states
            have = have.join(got, how="outer")
        have = have.loc[:, sorted(have.columns)]
        have.index.name = "date"
        have.to_csv(dest)
        print(f"{market}: states file has {len(have.columns)} penalties", flush=True)

    rows = []
    for name, lad in LADDERS.items():
        for arm, cut in (("full", 0), ("truncated", 1)):
            if cut and len(lad) < 3:
                continue
            for market in ("us", "de", "jp"):
                menu = menus[name][market][: len(lad) - cut]
                menu = sorted({round(float(v), 12) for v in menu})
                got = my_score(market, menu, OUT / f"states-{market}.csv")
                worst, worst_cell = 0.0, ""
                for c in CELLS:
                    rel = abs(got[c] - SHU[market][c]) / abs(SHU[market][c])
                    if rel > worst:
                        worst, worst_cell = rel, c
                rows.append({"ladder": name, "arm": arm, "market": market,
                             "menu": "|".join(f"{v:g}" for v in menu),
                             "turnover": got["turnover"],
                             "shu_turnover": SHU[market]["turnover"],
                             "worst_rel": worst, "worst_cell": worst_cell,
                             "grade": grade(worst)})
                print(f"{name:<24}{arm:<10}{market}  turnover "
                      f"{got['turnover']:.3f} (Shu {SHU[market]['turnover']:.2f})"
                      f"  worst {worst:.1%} {worst_cell} {grade(worst)}",
                      flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "ladder-sensitivity.csv", index=False)
    pd.DataFrame([{"ladder": n, "market": m,
                   "menu": "|".join(f"{v:g}" for v in menus[n][m])}
                  for n in menus for m in menus[n]]).to_csv(
        OUT / "ladder-menus.csv", index=False)
    print(f"\nwrote {OUT}/ladder-sensitivity.csv")


if __name__ == "__main__":
    main()
