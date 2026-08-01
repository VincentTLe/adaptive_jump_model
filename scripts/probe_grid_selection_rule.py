"""grid-selection-rule-001: order the admissible grids by a stated rule.

The v10 contract picked its per-market grid by taking the first row of an
examples file. This replaces that with a rule declared before any agreement is
computed: among the grids ALREADY qualified by the published table, adopt the
one whose monthly-selected state sequence agrees most with the sequence the
authors themselves ran, read out of the published Figure 5 vectors.

No new grid search happens here. No strategy metric enters the selection.
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

from probe_jm_grid_exhaustive2 import (  # noqa: E402
    _arm_cache,
    arm_key,
    choice_matrix,
    digest_array,
)

IN = ROOT / "artifacts/jm-residual/08-exhaustive-nine-arms"
FIG = ROOT / "artifacts/hmm-residual/04-figure6-path"
OUT = ROOT / "artifacts/grid-selection-rule/01-rule"
MARKETS = ("us", "de", "jp")
DELAYS = (1, 5, 10)
SIZES = range(2, 9)
ADOPTED = {
    "us": (0.0, 21.544346900318832, 70.0),
    "de": (150.0, 500.0),
    "jp": (10.0, 220.0),
}


def authors_states(market: str) -> pd.Series:
    """Their regime sequence: the shading is the position, shifted by t+2."""
    path = pd.read_csv(FIG / f"position-fig5-{market}-jm.csv", parse_dates=["date"])
    position = path.set_index("date")["position"].astype(float)
    return (1.0 - position).shift(-2).dropna()


def _sweep(task):
    """Agreement for every admissible grid in one (market, size, slice)."""
    market, size, lo, hi, target_values, target_positions = task
    keys = [arm_key(market, delay) for delay in DELAYS]
    caches = {key: _arm_cache(key) for key in keys}
    passes = {key: np.load(IN / f"{key}-pass-digests.npy") for key in keys}
    primary = caches[keys[0]]
    lambdas = primary["lambdas"]
    states = primary["states"]
    governing = primary["governing"]

    combos = np.array(
        list(
            itertools.islice(
                itertools.combinations(range(len(lambdas)), size), lo, hi
            )
        ),
        dtype=np.int64,
    )
    if not len(combos):
        return []

    best: list = []
    seen: dict[bytes, float] = {}
    for start in range(0, len(combos), 4000):
        block = combos[start : start + 4000]
        good = np.ones(len(block), dtype=bool)
        block_choices = None
        for key in keys:
            choices, invalid = choice_matrix(caches[key], block)
            if key == keys[0]:
                block_choices = choices
            digests = digest_array(
                [choices[i].tobytes() for i in range(len(block))]
            )
            good &= np.isin(digests, passes[key]) & ~invalid
            if not good.any():
                break
        if not good.any():
            continue
        for index in np.nonzero(good)[0]:
            key_bytes = block_choices[index].tobytes()
            score = seen.get(key_bytes)
            if score is None:
                # day -> selected lambda column -> state
                chosen = block_choices[index][governing]
                ours = states[np.arange(len(states)), chosen]
                mine = ours[target_positions]
                both = ~np.isnan(mine)
                score = (
                    float((mine[both] == target_values[both]).mean())
                    if both.any()
                    else float("nan")
                )
                seen[key_bytes] = score
            best.append((score, tuple(float(lambdas[j]) for j in block[index])))
    best.sort(key=lambda row: (-row[0], len(row[1]), max(row[1]), row[1]))
    return best[:200]


def main() -> int:
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 28
    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    for market in MARKETS:
        cache = _arm_cache(arm_key(market, 1))
        dates = pd.DatetimeIndex(cache["dates"].astype("datetime64[ns]"))
        theirs = authors_states(market)
        common = dates.intersection(theirs.index)
        positions = np.searchsorted(dates.values, common.values)
        target = theirs.loc[common].to_numpy(float)
        print(
            f"{market}: comparing {len(common)} days against the authors' sequence "
            f"({common[0].date()} to {common[-1].date()})",
            flush=True,
        )

        tasks = []
        total = len(cache["lambdas"])
        for size in SIZES:
            count = len(list(itertools.combinations(range(total), size)))
            step = max(1, count // 24)
            for lo in range(0, count, step):
                tasks.append(
                    (market, size, lo, min(lo + step, count), target, positions)
                )
        executor = ProcessPoolExecutor(
            max_workers=n_jobs, mp_context=get_context("forkserver")
        )
        try:
            collected: list = []
            for chunk in executor.map(_sweep, tasks):
                collected.extend(chunk)
        finally:
            executor.shutdown()

        collected.sort(key=lambda row: (-row[0], len(row[1]), max(row[1]), row[1]))
        frame = pd.DataFrame(
            [
                {"agreement": s, "grid": "|".join(f"{v:g}" for v in g)}
                for s, g in collected
            ]
        )
        frame.to_csv(OUT / f"ranking-{market}.csv", index=False)
        adopted_key = "|".join(f"{v:g}" for v in ADOPTED[market])
        adopted_row = frame[frame.grid == adopted_key]
        winner_score, winner_grid = collected[0]
        ties = sum(1 for s, _ in collected if s == winner_score)
        adopted_score = (
            float(adopted_row["agreement"].iloc[0])
            if len(adopted_row)
            else float("nan")
        )
        print(
            f"  winner  {'|'.join(f'{v:g}' for v in winner_grid):<28} "
            f"agreement {winner_score:.4f}  ({ties} grids tie)"
        )
        print(
            f"  adopted {adopted_key:<28} agreement {adopted_score:.4f}"
            f"  -> {'SAME' if winner_grid == ADOPTED[market] else 'DIFFERS'}"
        )
        print(
            f"  admissible grids ranked: {len(frame)}; agreement spread "
            f"[{frame.agreement.min():.4f}, {frame.agreement.max():.4f}]",
            flush=True,
        )
        summary.append(
            {
                "market": market,
                "compared_days": len(common),
                "winner": "|".join(f"{v:g}" for v in winner_grid),
                "winner_agreement": winner_score,
                "winner_ties": ties,
                "adopted": adopted_key,
                "adopted_agreement": adopted_score,
                "differs": winner_grid != ADOPTED[market],
                "ranked": len(frame),
            }
        )
    pd.DataFrame(summary).to_csv(OUT / "summary.csv", index=False)
    print("\nwrote", OUT / "summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
