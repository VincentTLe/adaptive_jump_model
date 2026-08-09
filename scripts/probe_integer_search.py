"""Search the deduplicated integer menu for the minimax grid.

Strategy adapts to how many DISTINCT state paths survive deduplication:
  * every subset size whose count fits the enumeration budget is searched
    EXHAUSTIVELY, so those sizes are settled, not sampled;
  * larger sizes are searched by iterated local search - add, drop and swap
    one value at a time from many random restarts - which is what remains
    when C(n, k) stops being enumerable.
The two are reported separately so nobody reads a local-search result as a
proof of optimality.

Objective, unchanged: minimise the WORST RELATIVE deviation across the eight
published Table-4 cells. Still a calibration search, still labelled as one.
"""

import itertools
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from multiprocessing import get_context  # noqa: E402

from _shu_table4 import TABLE4  # noqa: E402

from adaptive_jump.config import load_config  # noqa: E402
from adaptive_jump.walkforward import select_monthly_candidate  # noqa: E402

BASE = (
    ROOT / "artifacts/fixed-baselines/"
    "fixed-baselines-36ca1ace131c-ed7abd7daea3-f9f3e0a93736"
)
STATES = ROOT / "artifacts/integer-menu/01-states"
OUT = ROOT / "artifacts/integer-menu/02-search"
BUDGET = 60_000_000
TIE, COST, DELAY = 1e-12, 10.0, 1
METRICS = (
    "cagr", "volatility", "sharpe", "maximum_drawdown", "calmar",
    "expected_shortfall_5pct", "turnover", "leverage",
)
_C: dict = {}


def build_cache(market: str) -> dict:
    config = load_config(ROOT / "research-calibrated-v10.toml")
    frame = pd.read_csv(BASE / market / "features.csv", parse_dates=["date"])
    states = pd.read_csv(STATES / f"states-{market}.csv", index_col=0, parse_dates=[0])
    states.columns = [float(c) for c in states.columns]
    keep_lambdas = pd.read_csv(STATES / f"distinct-{market}.csv")["lambda"].tolist()
    states = states.loc[:, keep_lambdas]
    lambdas = np.array(keep_lambdas, dtype=float)
    selection = select_monthly_candidate(
        frame[["date", "equity_simple", "cash_return"]], states,
        config.selection_protocol, delay_trading_days=DELAY, one_way_cost_bps=COST,
        periods_per_year=config.metrics_protocol.periods_per_year,
        volatility_ddof=config.metrics_protocol.volatility_ddof,
    )
    surface = selection.surface
    decisions = pd.DatetimeIndex(sorted(surface["decision_date"].unique()))
    score = surface.pivot(index="decision_date", columns="candidate",
                          values="sharpe").loc[decisions, lambdas].to_numpy()
    elig = (
        surface.pivot(index="decision_date", columns="candidate", values="eligible")
        .loc[decisions, lambdas]
        .to_numpy()
        .astype(bool)
    )
    dates = pd.DatetimeIndex(frame["date"])
    reported = pd.read_csv(BASE / "metrics.csv", parse_dates=["start", "end"])
    row = reported[(reported.market == market) & (reported.model == "fixed_jm")
                   & (reported.delay == DELAY)].iloc[0]
    return dict(
        score=np.where(np.isfinite(score), score, -np.inf), elig=elig,
        states=states.reindex(dates).to_numpy(), lambdas=lambdas,
        governing=np.searchsorted(decisions.values, dates.values, side="right") - 1,
        equity=frame["equity_simple"].to_numpy(), cash=frame["cash_return"].to_numpy(),
        keep=(dates >= row["start"]) & (dates <= row["end"]),
        target=TABLE4[market]["fixed_jm"],
    )


def _init(market):
    _C.update(build_cache(market))


def _worst(block: np.ndarray) -> np.ndarray:
    """Worst relative deviation for a (C, k) block of member indices."""
    c = _C
    sc = np.where(c["elig"][:, block], c["score"][:, block], -np.inf)
    top = sc.max(axis=2)
    first = (sc >= top[:, :, None] - TIE).argmax(axis=2)
    picked = np.take_along_axis(
        np.broadcast_to(block.T[None, :, :].transpose(0, 2, 1), sc.shape),
        first[:, :, None], axis=2)[:, :, 0]
    invalid = (
        ~np.isfinite(top) & (np.cumsum(np.isfinite(top), axis=0) > 0)
    ).any(axis=0)
    chosen = picked[c["governing"], :]
    signal = (c["states"][np.arange(len(chosen))[:, None], chosen] == 0)
    pos = pd.DataFrame(signal.astype(float)).shift(DELAY + 1).ffill().to_numpy()
    keep = c["keep"]
    pos = pos[keep]
    eq, cash = c["equity"][keep], c["cash"][keep]
    turn = np.abs(np.diff(pos, axis=0, prepend=pos[:1]))
    ret = pos * eq[:, None] + (1 - pos) * cash[:, None] - COST / 1e4 * turn
    ex = ret - cash[:, None]
    n = len(ret)
    with np.errstate(invalid="ignore", divide="ignore"):
        vol = np.nanstd(ret, axis=0, ddof=1) * np.sqrt(252)
        wealth = np.vstack([np.ones((1, ret.shape[1])),
                            np.nancumprod(1 + ret, axis=0)])
        mdd = (wealth / np.maximum.accumulate(wealth, axis=0) - 1).min(axis=0)
        es = np.nanquantile(ret, 0.05, axis=0)
        got = dict(
            cagr=np.nanprod(1 + ret, axis=0) ** (252 / n) - 1, volatility=vol,
            sharpe=np.nanmean(ex, axis=0) * 252 / vol, maximum_drawdown=mdd,
            calmar=np.nanmean(ex, axis=0) * 252 / np.abs(mdd),
            expected_shortfall_5pct=np.nanmean(
                np.where(ret <= es[None, :], ret, np.nan), axis=0),
            turnover=0.5 * np.nanmean(turn, axis=0) * 252,
            leverage=np.nanmean(pos, axis=0))
        worst = np.zeros(len(block))
        for m in METRICS:
            t = _C["target"][m]
            worst = np.maximum(worst, np.abs(got[m] - t) / max(abs(t), 1e-9))
    return np.where(invalid | ~np.isfinite(worst), np.inf, worst)


def _exhaustive(task):
    size, lo, hi, n = task
    combos = np.array(list(itertools.islice(
        itertools.combinations(range(n), size), lo, hi)), dtype=np.int64)
    if not len(combos):
        return np.inf, ()
    best = (np.inf, ())
    for start in range(0, len(combos), 2000):
        block = combos[start:start + 2000]
        w = _worst(block)
        i = int(np.argmin(w))
        if w[i] < best[0]:
            best = (float(w[i]), tuple(int(j) for j in block[i]))
    return best


# Grids already known to be good, from the sparse and dense searches. Local
# search that starts from random points would spend its budget rediscovering
# what is already known; seeding it means it starts ABOVE the current best and
# every reported improvement is a real improvement on the record.
KNOWN_GOOD = {
    "de": ([0, 2, 3, 848], [0, 4, 5, 10, 15, 40]),
    "jp": ([2, 37, 138, 162, 1000], [2, 15, 27, 35, 52, 220]),
}


def _local(task):
    """Iterated local search; the first restarts use the known-good grids."""
    seed, n, sizes, seconds, seeds_for_market = task
    rng = np.random.default_rng(seed)
    best = (np.inf, ())
    deadline = time.time() + seconds
    queue = list(seeds_for_market) if seed < len(seeds_for_market) else []
    while time.time() < deadline:
        if queue:
            current = queue.pop(0)
        else:
            size = int(rng.choice(sizes))
            current = tuple(sorted(rng.choice(n, size=size, replace=False).tolist()))
        score = float(_worst(np.array([current]))[0])
        improved = True
        while improved and time.time() < deadline:
            improved = False
            moves = []
            for out in current:                       # drop / swap
                rest = tuple(x for x in current if x != out)
                if len(rest) >= 2:
                    moves.append(rest)
                for new in range(n):
                    if new not in current:
                        moves.append(tuple(sorted(rest + (new,))))
            for new in range(n):                      # add
                if new not in current and len(current) < max(sizes):
                    moves.append(tuple(sorted(current + (new,))))
            if not moves:
                break
            moves = list(dict.fromkeys(moves))
            for chunk in range(0, len(moves), 2000):
                block = moves[chunk:chunk + 2000]
                by_size: dict[int, list] = {}
                for mv in block:
                    by_size.setdefault(len(mv), []).append(mv)
                for group in by_size.values():
                    w = _worst(np.array(group, dtype=np.int64))
                    i = int(np.argmin(w))
                    if w[i] < score - 1e-12:
                        score, current, improved = float(w[i]), group[i], True
        if score < best[0]:
            best = (score, current)
    return best


def main() -> int:
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 28
    markets = sys.argv[2].split(",") if len(sys.argv) > 2 else ["de", "jp"]
    minutes = float(sys.argv[3]) if len(sys.argv) > 3 else 45.0
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for market in markets:
        cache = build_cache(market)
        lambdas = cache["lambdas"]
        n = len(lambdas)
        print(f"\n=== {market.upper()}: {n} duong trang thai phan biet ===", flush=True)
        # `X or True and Y` parses as `X or Y` with X always truthy, so an
        # earlier version of this line silently ignored the budget and tried to
        # enumerate 5.7e12 subsets. Keep it a single comparison.
        exhaustive_sizes = [k for k in range(2, 9) if math.comb(n, k) <= BUDGET]
        if not exhaustive_sizes:
            raise SystemExit(f"{market}: even pairs exceed the budget")
        print(f"  quet can duoc cho co: {exhaustive_sizes}", flush=True)
        executor = ProcessPoolExecutor(
            max_workers=n_jobs, mp_context=get_context("forkserver"),
            initializer=_init, initargs=(market,))
        best = (np.inf, ())
        exhaustive_best = (np.inf, ())
        try:
            for size in exhaustive_sizes:
                count = __import__("math").comb(n, size)
                step = max(1, count // (n_jobs * 4))
                tasks = [(size, lo, min(lo + step, count), n)
                         for lo in range(0, count, step)]
                found = min(executor.map(_exhaustive, tasks), key=lambda r: r[0])
                if found[0] < exhaustive_best[0]:
                    exhaustive_best = found
                print(f"  co {size}: {count:,} tap con -> worst {found[0]:.4f}",
                      flush=True)
            best = exhaustive_best
            remaining = [k for k in range(2, 9) if k not in exhaustive_sizes]
            if remaining:
                print(f"  tim cuc bo cho co {remaining}, {minutes:g} phut ...",
                      flush=True)
                lookup = {float(v): i for i, v in enumerate(lambdas)}
                seeds = []
                for grid in KNOWN_GOOD.get(market, ()):
                    members = tuple(sorted(
                        lookup[float(v)] for v in grid if float(v) in lookup
                    ))
                    if len(members) >= 2:
                        seeds.append(members)
                print(f"  gieo mam tu {len(seeds)} luoi da biet la tot", flush=True)
                tasks = [
                    (s, n, remaining, minutes * 60, seeds) for s in range(n_jobs)
                ]
                found = min(executor.map(_local, tasks), key=lambda r: r[0])
                print(f"  tim cuc bo -> worst {found[0]:.4f}", flush=True)
                if found[0] < best[0]:
                    best = found
        finally:
            executor.shutdown()
        grid = tuple(float(lambdas[j]) for j in best[1])
        rows.append({
            "market": market, "worst_relative_deviation": best[0],
            "grid": "|".join(f"{v:g}" for v in grid),
            "exhaustive_best": exhaustive_best[0],
            "exhaustive_sizes": ",".join(map(str, exhaustive_sizes)),
            "distinct_paths": n,
        })
        print(f"  {market} FINAL worst {best[0]:.4f} "
              f"grid {'|'.join(f'{v:g}' for v in grid)}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "integer-grids.csv", index=False)
    print("\nwrote", OUT / "integer-grids.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
