"""Exhaustive minimax over the DENSE menu, sizes 2-6, for Germany and Japan.

The control run settled which factor binds. Holding the algorithm fixed, the
dense 48-value menu beats the sparse 29-value one (de 15.1 against 17.1 percent,
jp 18.2 against 18.6). Holding the menu fixed, exhaustive search beats greedy by
about five points in Germany. So greedy was the binding constraint, not the
menu, and the move that has never been made is exhaustive search ON the dense
menu.

Sizes 2 to 6 give 14,196,820 subsets, only 2.2 times the 6,474,511 already
enumerated on the sparse menu, and every winner found so far has been size 4 to
6. The scoring is vectorised the same way the sparse search was: the monthly
cross-validation surface is computed ONCE for all 48 candidates, and each
subset's monthly choice is then an argmax over columns of that surface.

Still a calibration search against the published table; the label is unchanged.
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

from adaptive_jump.config import load_config  # noqa: E402
from adaptive_jump.walkforward import select_monthly_candidate  # noqa: E402

BASE = (
    ROOT / "artifacts/fixed-baselines/"
    "fixed-baselines-36ca1ace131c-ed7abd7daea3-f9f3e0a93736"
)
DENSE = ROOT / "artifacts/dense-menu/01-search"
OUT = ROOT / "artifacts/dense-menu/02-exhaustive"
SIZES = range(2, 7)
TIE = 1e-12
COST, DELAY = 10.0, 1
METRICS = (
    "cagr", "volatility", "sharpe", "maximum_drawdown", "calmar",
    "expected_shortfall_5pct", "turnover", "leverage",
)
_C: dict = {}


def build_cache(market: str) -> dict:
    """One monthly-CV surface for all 48 candidates, plus the return series."""
    config = load_config(ROOT / "configs/baselines/legacy/research-calibrated-v10.toml")
    frame = pd.read_csv(BASE / market / "features.csv", parse_dates=["date"])
    states = pd.read_csv(DENSE / f"states-{market}.csv", index_col=0, parse_dates=[0])
    states.columns = [float(c) for c in states.columns]
    lambdas = np.array(sorted(states.columns), dtype=float)
    states = states.loc[:, list(lambdas)]
    selection = select_monthly_candidate(
        frame[["date", "equity_simple", "cash_return"]],
        states,
        config.selection_protocol,
        delay_trading_days=DELAY,
        one_way_cost_bps=COST,
        periods_per_year=config.metrics_protocol.periods_per_year,
        volatility_ddof=config.metrics_protocol.volatility_ddof,
    )
    surface = selection.surface
    decisions = pd.DatetimeIndex(sorted(surface["decision_date"].unique()))
    score = surface.pivot(
        index="decision_date", columns="candidate", values="sharpe"
    ).loc[decisions, lambdas].to_numpy()
    elig = surface.pivot(
        index="decision_date", columns="candidate", values="eligible"
    ).loc[decisions, lambdas].to_numpy().astype(bool)
    dates = pd.DatetimeIndex(frame["date"])
    governing = np.searchsorted(decisions.values, dates.values, side="right") - 1
    reported = pd.read_csv(BASE / "metrics.csv", parse_dates=["start", "end"])
    row = reported[
        (reported.market == market)
        & (reported.model == "fixed_jm")
        & (reported.delay == DELAY)
    ].iloc[0]
    keep = (dates >= row["start"]) & (dates <= row["end"])
    return dict(
        score=np.where(np.isfinite(score), score, -np.inf),
        elig=elig,
        states=states.reindex(dates).to_numpy(),
        lambdas=lambdas,
        governing=governing,
        equity=frame["equity_simple"].to_numpy(),
        cash=frame["cash_return"].to_numpy(),
        keep=keep,
        target=TABLE4[market]["fixed_jm"],
    )


def _init(market):
    _C.update(build_cache(market))


def _batch(task):
    """Worst relative deviation for a block of subsets, fully vectorised."""
    size, lo, hi = task
    cache = _C
    lambdas = cache["lambdas"]
    combos = np.array(
        list(
            itertools.islice(
                itertools.combinations(range(len(lambdas)), size), lo, hi
            )
        ),
        dtype=np.int64,
    )
    if not len(combos):
        return np.inf, ()
    best = (np.inf, ())
    keep = cache["keep"]
    eq = cache["equity"][keep]
    cash = cache["cash"][keep]
    target = cache["target"]
    n = int(keep.sum())
    for start in range(0, len(combos), 2000):
        block = combos[start : start + 2000]
        sc = np.where(cache["elig"][:, block], cache["score"][:, block], -np.inf)
        top = sc.max(axis=2)
        winners = sc >= top[:, :, None] - TIE
        first = winners.argmax(axis=2)
        picked = np.take_along_axis(
            np.broadcast_to(
                block.T[None, :, :].transpose(0, 2, 1), sc.shape
            ),
            first[:, :, None],
            axis=2,
        )[:, :, 0]
        invalid = (~np.isfinite(top) & (np.cumsum(np.isfinite(top), axis=0) > 0))
        invalid = invalid.any(axis=0)
        chosen = picked[cache["governing"], :]                       # (T, C)
        signal = (cache["states"][np.arange(len(chosen))[:, None], chosen] == 0)
        pos = pd.DataFrame(signal.astype(float)).shift(DELAY + 1).ffill().to_numpy()
        pos = pos[keep]
        turn = np.abs(np.diff(pos, axis=0, prepend=pos[:1]))
        ret = pos * eq[:, None] + (1 - pos) * cash[:, None] - COST / 1e4 * turn
        ex = ret - cash[:, None]
        vol = np.nanstd(ret, axis=0, ddof=1) * np.sqrt(252)
        with np.errstate(invalid="ignore", divide="ignore"):
            sharpe = np.nanmean(ex, axis=0) * 252 / vol
            wealth = np.vstack(
                [np.ones((1, ret.shape[1])), np.nancumprod(1 + ret, axis=0)]
            )
            mdd = (wealth / np.maximum.accumulate(wealth, axis=0) - 1).min(axis=0)
            cagr = np.nanprod(1 + ret, axis=0) ** (252 / n) - 1
            calmar = np.nanmean(ex, axis=0) * 252 / np.abs(mdd)
            es = np.nanquantile(ret, 0.05, axis=0)
            esm = np.where(ret <= es[None, :], ret, np.nan)
            es = np.nanmean(esm, axis=0)
            got = dict(
                cagr=cagr, volatility=vol, sharpe=sharpe, maximum_drawdown=mdd,
                calmar=calmar, expected_shortfall_5pct=es,
                turnover=0.5 * np.nanmean(turn, axis=0) * 252,
                leverage=np.nanmean(pos, axis=0),
            )
            worst = np.zeros(len(block))
            for m in METRICS:
                worst = np.maximum(
                    worst, np.abs(got[m] - target[m]) / max(abs(target[m]), 1e-9)
                )
        worst = np.where(invalid | ~np.isfinite(worst), np.inf, worst)
        index = int(np.argmin(worst))
        if worst[index] < best[0]:
            best = (float(worst[index]),
                    tuple(float(lambdas[j]) for j in block[index]))
    return best


def main() -> int:
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 28
    markets = sys.argv[2].split(",") if len(sys.argv) > 2 else ["de", "jp"]
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for market in markets:
        probe = build_cache(market)
        total = len(probe["lambdas"])
        tasks = []
        for size in SIZES:
            count = len(list(itertools.combinations(range(total), size)))
            step = max(1, count // 48)
            for lo in range(0, count, step):
                tasks.append((size, lo, min(lo + step, count)))
        print(f"{market}: {len(tasks)} khoi, menu {total} gia tri", flush=True)
        executor = ProcessPoolExecutor(
            max_workers=n_jobs, mp_context=get_context("forkserver"),
            initializer=_init, initargs=(market,),
        )
        try:
            found = list(executor.map(_batch, tasks))
        finally:
            executor.shutdown()
        score, grid = min(found, key=lambda r: r[0])
        rows.append({"market": market, "worst_relative_deviation": score,
                     "grid": "|".join(f"{v:g}" for v in grid)})
        print(f"{market}: worst {score:.4f} with {'|'.join(f'{v:g}' for v in grid)}",
              flush=True)
    pd.DataFrame(rows).to_csv(OUT / "dense-exhaustive.csv", index=False)
    print("wrote", OUT / "dense-exhaustive.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
