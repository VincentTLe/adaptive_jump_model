"""-009 Q3 extension: DE/JP best-available grids (13/14 cells).

Best-available = 7 of 8 Table-4 cells at delay 1 (the frontier maximum for
DE/JP within the menu) AND all three Table-5 cells at delays 5 and 10.
Same scoping rule as -009; solutions are calibration artifacts.
"""

from __future__ import annotations

import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from math import comb
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from _shu_table4 import METRICS, TABLE4  # noqa: E402

from experiments.probe_jm_grid_exhaustive2 import (  # noqa: E402
    _arm_cache,
    arm_key,
    choice_matrix,
    digest_array,
)

# Overridable via env var, same mechanism as probe_jm_per_market_grids.py.
_DEFAULT_IN = ROOT / "artifacts" / "jm-residual" / "08-exhaustive-nine-arms"
_DEFAULT_OUT = ROOT / "artifacts" / "jm-residual" / "09-per-market-grids"
IN = Path(os.environ.get("AJM_PER_MARKET_IN", str(_DEFAULT_IN)))
OUT = Path(os.environ.get("AJM_PER_MARKET_OUT", str(_DEFAULT_OUT)))
N_JOBS, TOL = 30, 0.05
SIZES = range(2, 9)


def build_7of8_digests(market: str) -> None:
    z = np.load(IN / f"{market}-d1-results.npz")
    target = TABLE4[market]["fixed_jm"]
    count = sum((np.abs(z[m] - target[m]) <= TOL).astype(np.int8)
                for m in METRICS)
    keep = count >= 7
    np.save(OUT / f"{market}-d1-7of8-digests.npy", np.sort(z["digests"][keep]))
    print(f"{market}: {int(keep.sum())} vectors at >=7/8 (d1)", flush=True)


def _sweep(task: tuple[str, int, int, int]) -> tuple[int, list]:
    market, k, lo, hi = task
    keys = [arm_key(market, 1), arm_key(market, 5), arm_key(market, 10)]
    caches = {key: _arm_cache(key) for key in keys}
    sets = {
        keys[0]: np.load(OUT / f"{market}-d1-7of8-digests.npy"),
        keys[1]: np.load(IN / f"{keys[1]}-pass-digests.npy"),
        keys[2]: np.load(IN / f"{keys[2]}-pass-digests.npy"),
    }
    combos = np.array(list(itertools.islice(
        itertools.combinations(range(29), k), lo, hi)), dtype=np.int64)
    if not len(combos):
        return 0, []
    hits, examples = 0, []
    for start in range(0, len(combos), 4000):
        block = combos[start:start + 4000]
        good = np.ones(len(block), dtype=bool)
        for key in keys:
            choices, invalid = choice_matrix(caches[key], block)
            digests = digest_array([choices[i].tobytes()
                                    for i in range(len(block))])
            good &= np.isin(digests, sets[key]) & ~invalid
            if not good.any():
                break
        hits += int(good.sum())
        for idx in np.nonzero(good)[0][:5]:
            if len(examples) < 12:
                examples.append((k, block[idx].tolist()))
    return hits, examples


def main() -> None:
    for market in ("de", "jp"):
        build_7of8_digests(market)
    executor = ProcessPoolExecutor(max_workers=N_JOBS,
                                   mp_context=get_context("forkserver"))
    rows, ex_rows = [], []
    lambdas = np.load(IN / "us-d1-cache.npz")["lambdas"]
    try:
        for market in ("de", "jp"):
            tasks = []
            for k in SIZES:
                total = comb(29, k)
                step = max(1, total // (N_JOBS * 2))
                for lo in range(0, total, step):
                    tasks.append((market, k, lo, min(lo + step, total)))
            hits, examples = 0, []
            for h, ex in executor.map(_sweep, tasks):
                hits += h
                for k, combo in ex:
                    if len(examples) < 12:
                        examples.append((k, combo))
            rows.append({"market": market, "best13of14_count": hits})
            for k, combo in examples:
                ex_rows.append({"market": market, "size": k,
                                "grid": "|".join(f"{lambdas[i]:g}"
                                                 for i in combo)})
            print(f"{market}: 13/14 grids = {hits}", flush=True)
    finally:
        executor.shutdown(cancel_futures=True)
    pd.DataFrame(rows).to_csv(OUT / "best-13of14.csv", index=False,
                              lineterminator="\n")
    pd.DataFrame(ex_rows).to_csv(OUT / "best-13of14-examples.csv",
                                 index=False, lineterminator="\n")


if __name__ == "__main__":
    main()
