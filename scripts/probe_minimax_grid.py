"""Which lambda grid gets our model CLOSEST to every published cell at once.

The earlier searches asked "how many of the eight cells land within 0.05".
That is a bad question: it rewards a grid that nails seven cells and misses
the eighth by 84 percent (which is exactly what the adopted German grid does
on turnover) over a grid that is within 12 percent on all eight. This asks the
better question - minimise the WORST relative deviation across the eight cells
- and reports the grid that achieves it.

Still a calibration search against the published table, and labelled as such.
What changes is only that the objective now matches what a reader means by
"the replication is close".
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from multiprocessing import get_context  # noqa: E402

from _shu_table4 import TABLE4  # noqa: E402
from probe_jm_grid_exhaustive2 import _arm_cache, arm_key, choice_matrix  # noqa: E402

OUT = ROOT / "artifacts/minimax-grid/01-search"
MARKETS = ("us", "de", "jp")
SIZES = range(2, 9)
METRICS = (
    "cagr",
    "volatility",
    "sharpe",
    "maximum_drawdown",
    "calmar",
    "expected_shortfall_5pct",
    "turnover",
    "leverage",
)


def _score_task(task):
    """Worst relative deviation for every grid in one slice."""
    market, size, lo, hi = task
    cache = _arm_cache(arm_key(market, 1))
    lambdas = cache["lambdas"]
    results = np.load(
        ROOT / f"artifacts/jm-residual/08-exhaustive-nine-arms/{market}-d1-results.npz"
    )
    digests = results["digests"]
    order = np.argsort(digests)
    sorted_digests = digests[order]
    target = TABLE4[market]["fixed_jm"]
    stacked = np.column_stack(
        [
            np.abs(results[m] - target[m]) / max(abs(target[m]), 1e-9)
            for m in METRICS
        ]
    )
    worst = stacked.max(axis=1)

    from probe_jm_grid_exhaustive2 import digest_array

    combos = np.array(
        list(
            itertools.islice(
                itertools.combinations(range(len(lambdas)), size), lo, hi
            )
        ),
        dtype=np.int64,
    )
    if not len(combos):
        return None
    best = (np.inf, None)
    for start in range(0, len(combos), 4000):
        block = combos[start : start + 4000]
        choices, invalid = choice_matrix(cache, block)
        block_digests = digest_array(
            [choices[i].tobytes() for i in range(len(block))]
        )
        position = np.searchsorted(sorted_digests, block_digests)
        position = np.clip(position, 0, len(sorted_digests) - 1)
        found = sorted_digests[position] == block_digests
        rows = order[position]
        scores = np.where(found & ~invalid, worst[rows], np.inf)
        index = int(np.argmin(scores))
        if scores[index] < best[0]:
            best = (
                float(scores[index]),
                tuple(float(lambdas[j]) for j in block[index]),
            )
    return best


def main() -> int:
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 28
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for market in MARKETS:
        cache = _arm_cache(arm_key(market, 1))
        total_lambdas = len(cache["lambdas"])
        tasks = []
        for size in SIZES:
            count = len(list(itertools.combinations(range(total_lambdas), size)))
            step = max(1, count // 24)
            for lo in range(0, count, step):
                tasks.append((market, size, lo, min(lo + step, count)))
        executor = ProcessPoolExecutor(
            max_workers=n_jobs, mp_context=get_context("forkserver")
        )
        try:
            found = [r for r in executor.map(_score_task, tasks) if r]
        finally:
            executor.shutdown()
        score, grid = min(found, key=lambda r: r[0])
        rows.append(
            {
                "market": market,
                "worst_relative_deviation": score,
                "grid": "|".join(f"{v:g}" for v in grid),
                "size": len(grid),
            }
        )
        print(
            f"{market}: worst relative deviation {score:.4f} "
            f"with grid {'|'.join(f'{v:g}' for v in grid)}",
            flush=True,
        )
    pd.DataFrame(rows).to_csv(OUT / "minimax-grids.csv", index=False)
    print("\nwrote", OUT / "minimax-grids.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
