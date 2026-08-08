"""jm-grid-exhaustive-008: nine-arm exhaustive search (3 markets x 3 delays).

Frozen spec: research/jm-grid-exhaustive-008.toml — the honesty label there
governs everything: owner-instructed target-conditioned search; solutions are
calibration artifacts; the content is the existence/intersection map.

Design: per arm, one select_monthly_candidate pass gives the per-(decision,
lambda) surface; subsets map to monthly choice vectors (scores are
set-independent; tie 1e-12 to the lower lambda); unique vectors are scored by
a BATCHED numpy evaluator mirroring apply_signal + performance_metrics
formula-for-formula, gated per arm by (a) named-grid parity vs -001
grids.csv / the sealed metrics rows and (b) 50 random vectors re-scored
through the real code path, both to 1e-9, both hard stops. Intersections are
computed in a final sweep over the subset lattice against per-arm pass sets.
"""

from __future__ import annotations

import hashlib
import itertools
import sys
from concurrent.futures import ProcessPoolExecutor
from math import comb
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from _shu_table4 import METRICS, TABLE4  # noqa: E402

from adaptive_jump.backtest import apply_signal, performance_metrics  # noqa: E402
from adaptive_jump.config import load_config  # noqa: E402
from adaptive_jump.walkforward import select_monthly_candidate  # noqa: E402

# The original v9.4-hash run directory (fixed-baselines-34e51cd7a388-...) is
# gone from disk (2026-08-07). Substituted with the v10 run: its features.csv
# is byte-identical to v9.4's by reseal gate 2 (scripts/gate_v10_reseal.py),
# and rebuilding us-d1's cache from it reproduces the sealed v9.4 anchor
# (34 shifts / 0.21868067717454... bear / 8565 days) to float precision.
RUN = ROOT / "artifacts" / "fixed-baselines" / (
    "fixed-baselines-36ca1ace131c-ed7abd7daea3-f9f3e0a93736"
)
UNION_DIR = ROOT / "artifacts" / "jm-residual" / "01-grid-identification"
OUT = ROOT / "artifacts" / "jm-residual" / "08-exhaustive-nine-arms"
COST, TOL, N_JOBS, TIE = 10.0, 0.05, 30, 1e-12
SIZES = range(2, 9)
MARKETS = ("us", "de", "jp")
DELAYS = (1, 5, 10)
# Table 5 JM cells (paper lines 957-976): Return, Sharpe, Calmar per delay.
TABLE5_JM = {
    ("us", 5): {"cagr": 0.114, "sharpe": 0.71, "calmar": 0.39},
    ("us", 10): {"cagr": 0.117, "sharpe": 0.70, "calmar": 0.28},
    ("de", 5): {"cagr": 0.075, "sharpe": 0.38, "calmar": 0.13},
    ("de", 10): {"cagr": 0.059, "sharpe": 0.29, "calmar": 0.11},
    ("jp", 5): {"cagr": 0.040, "sharpe": 0.27, "calmar": 0.09},
    ("jp", 10): {"cagr": 0.034, "sharpe": 0.24, "calmar": 0.07},
}
NAMED_001 = {
    "table3_sealed": (0.0, 5.0, 15.0, 35.0, 70.0, 150.0),
    "v1_author_withdrawn": (10.0, 22.0, 50.0, 100.0, 220.0, 500.0, 1000.0),
    "li2025_citing": (0.0, 5.0, 10.0, 25.0, 50.0, 100.0),
    "bocconi_wild": (0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0,
                     150.0),
    "hackmd_wild": (0.0, 0.1, 1.0, 10.0, 100.0),
    "companion_log5": (0.0, *np.logspace(0, 2, 4)),
    "companion_log9": (0.0, *np.logspace(0, 2, 8)),
    "typical_range": (50.0, 70.0, 100.0),
}

_CACHE: dict[str, dict] = {}


def arm_key(market: str, delay: int) -> str:
    return f"{market}-d{delay}"


def _arm_cache(key: str) -> dict:
    if key not in _CACHE:
        data = np.load(OUT / f"{key}-cache.npz", allow_pickle=False)
        _CACHE[key] = {k: data[k] for k in data.files}
    return _CACHE[key]


def _build_arm(task: tuple[str, int]) -> str:
    market, delay = task
    cfg = load_config(ROOT / "research-expanding-v9-4.toml")
    frame = pd.read_csv(RUN / market / "features.csv", parse_dates=["date"])
    states = pd.read_csv(
        UNION_DIR / market / "union-states.csv", parse_dates=["date"]
    ).set_index("date")
    states.columns = [float(c) for c in states.columns]
    lambdas = np.array(sorted(states.columns), dtype=float)
    states = states.loc[:, list(lambdas)]
    selection = select_monthly_candidate(
        frame[["date", "equity_simple", "cash_return"]], states,
        cfg.selection_protocol, delay_trading_days=delay, one_way_cost_bps=COST,
        periods_per_year=cfg.metrics_protocol.periods_per_year,
        volatility_ddof=cfg.metrics_protocol.volatility_ddof)
    surface = selection.surface
    decisions = pd.DatetimeIndex(sorted(surface["decision_date"].unique()))
    score = surface.pivot(index="decision_date", columns="candidate",
                          values="sharpe").loc[decisions, lambdas].to_numpy()
    elig = surface.pivot(index="decision_date", columns="candidate",
                         values="eligible").loc[decisions, lambdas
                                                ].to_numpy().astype(bool)
    dates = pd.DatetimeIndex(frame["date"])
    governing = np.searchsorted(decisions.values, dates.values,
                                side="right") - 1
    # v10's metrics.csv carries the same market/model/delay/start/end schema
    # as the old run's metrics-exploratory.csv (protocol-level window bounds,
    # not data that could drift between the two runs).
    reported = pd.read_csv(RUN / "metrics.csv", parse_dates=["start", "end"])
    row = reported[(reported.market == market) & (reported.model == "fixed_jm")
                   & (reported.delay == delay)].iloc[0]
    np.savez_compressed(
        OUT / f"{arm_key(market, delay)}-cache.npz",
        score=np.where(np.isfinite(score), score, -np.inf), elig=elig,
        states=states.to_numpy(), lambdas=lambdas, governing=governing,
        equity=frame["equity_simple"].to_numpy(),
        cash=frame["cash_return"].to_numpy(),
        dates=dates.values.astype("datetime64[ns]").astype("int64"),
        window=np.array([row["start"].value, row["end"].value]),
        delay=np.array([delay]),
    )
    return f"{arm_key(market, delay)}: cache built ({len(decisions)} decisions)"


def choice_matrix(cache: dict, combos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(C, k) member indices -> (C, D) int8 choices; also per-combo validity."""
    scores = np.where(cache["elig"][:, combos], cache["score"][:, combos],
                      -np.inf)                       # (D, C, k)
    best = scores.max(axis=2)                        # (D, C)
    winners = scores >= best[:, :, None] - TIE
    first = winners.argmax(axis=2)                   # (D, C)
    choice = np.take_along_axis(
        np.broadcast_to(combos.T[None, :, :].transpose(0, 2, 1),
                        scores.shape), first[:, :, None], axis=2
    )[:, :, 0].astype(np.int8)                       # (D, C)
    none = ~np.isfinite(best)                        # (D, C)
    started = np.cumsum(~none, axis=0) > 0
    invalid = (none & started).any(axis=0)           # (C,)
    choice[none] = -1
    return choice.T.copy(), invalid                  # (C, D)


def _enumerate_task(task: tuple[str, int, int, int]) -> dict:
    key, k, lo, hi = task
    cache = _arm_cache(key)
    n = len(cache["lambdas"])
    combos = np.array(list(itertools.islice(
        itertools.combinations(range(n), k), lo, hi)), dtype=np.int64)
    if not len(combos):
        return {}
    out: dict[bytes, tuple[int, bytes]] = {}
    for start in range(0, len(combos), 4000):
        block = combos[start:start + 4000]
        choices, invalid = choice_matrix(cache, block)
        for i in range(len(block)):
            if invalid[i]:
                continue
            vec = choices[i].tobytes()
            prev = out.get(vec)
            if prev is None:
                out[vec] = (1, block[i].astype(np.int8).tobytes())
            else:
                out[vec] = (prev[0] + 1, prev[1])
    return out


def batch_metrics(cache: dict, choices: np.ndarray) -> dict[str, np.ndarray]:
    """Mirror apply_signal + performance_metrics for (V, D) choice matrices."""
    delay = int(cache["delay"][0])
    governing = cache["governing"]
    states = cache["states"]
    n_days = len(governing)
    active = np.where(governing >= 0,
                      choices[:, np.clip(governing, 0, None)], -1)  # (V, days)
    day_idx = np.broadcast_to(np.arange(n_days), active.shape)
    signal = np.where(active >= 0,
                      1.0 - states[day_idx, np.clip(active, 0, None)], np.nan)
    shift = delay + 1
    pos = np.full_like(signal, np.nan)
    pos[:, shift:] = signal[:, :-shift]
    prev = np.full_like(pos, np.nan)
    prev[:, 1:] = pos[:, :-1]
    turn = np.abs(pos - prev)
    first_valid = np.argmax(~np.isnan(pos), axis=1)
    turn[np.arange(len(pos)), first_valid] = 0.0     # charge_initial False
    strat = (pos * cache["equity"][None, :]
             + (1.0 - pos) * cache["cash"][None, :]
             - turn * COST / 10_000.0)
    lo, hi = cache["window"]
    in_window = (cache["dates"] >= lo) & (cache["dates"] <= hi)
    valid = in_window[None, :] & np.isfinite(strat) & np.isfinite(turn)
    n = valid.sum(axis=1).astype(float)              # (V,)

    s = np.where(valid, strat, np.nan)
    growth = np.where(valid, 1.0 + strat, 1.0)
    wealth = np.cumprod(growth, axis=1)
    final = wealth[:, -1]
    cagr = final ** (252.0 / n) - 1.0
    padded = np.concatenate([np.ones((len(wealth), 1)), wealth], axis=1)
    peaks = np.maximum.accumulate(padded, axis=1)
    mdd = (padded / peaks - 1.0).min(axis=1)
    var_s = np.nanvar(s, axis=1, ddof=1)
    vol = np.sqrt(var_s) * np.sqrt(252.0)
    excess = np.where(valid, strat - cache["cash"][None, :], np.nan)
    mean_ex = np.nanmean(excess, axis=1)
    sharpe = mean_ex * 252.0 / vol
    annual_ex = mean_ex * 252.0
    calmar = np.where(mdd < 0, annual_ex / np.abs(mdd), np.nan)
    q = np.nanquantile(s, 0.05, axis=1)
    below = np.where(s <= q[:, None], s, np.nan)
    es = np.nanmean(below, axis=1)
    turnover = np.nanmean(np.where(valid, turn, np.nan), axis=1) * 0.5 * 252.0
    leverage = np.nanmean(np.where(valid, pos, np.nan), axis=1)
    return {"cagr": cagr, "volatility": vol, "sharpe": sharpe,
            "maximum_drawdown": mdd, "calmar": calmar,
            "expected_shortfall_5pct": es, "turnover": turnover,
            "leverage": leverage}


def _evaluate_task(task: tuple[str, list[bytes]]) -> dict[str, np.ndarray]:
    key, vectors = task
    cache = _arm_cache(key)
    parts = []
    for start in range(0, len(vectors), 2000):
        block = vectors[start:start + 2000]
        choices = np.stack([np.frombuffer(v, dtype=np.int8) for v in block])
        parts.append(batch_metrics(cache, choices))
    return {m: np.concatenate([p[m] for p in parts]) for m in parts[0]}


def slow_metrics(cache: dict, vec: bytes, cfg) -> dict:
    """The real code path, for parity sampling."""
    choice = np.frombuffer(vec, dtype=np.int8)
    governing = cache["governing"]
    active = np.where(governing >= 0, choice[np.clip(governing, 0, None)], -1)
    signal = np.full(len(governing), np.nan)
    has = active >= 0
    signal[has] = 1.0 - cache["states"][np.nonzero(has)[0], active[has]]
    prepared = pd.DataFrame({
        "date": pd.to_datetime(cache["dates"]),
        "equity_simple": cache["equity"], "cash_return": cache["cash"],
    })
    path = apply_signal(prepared, pd.Series(signal),
                        delay_trading_days=int(cache["delay"][0]),
                        one_way_cost_bps=COST)
    lo, hi = pd.Timestamp(cache["window"][0]), pd.Timestamp(cache["window"][1])
    window = path[(path["date"] >= lo) & (path["date"] <= hi)].dropna(
        subset=["cash_return", "position", "one_way_turnover",
                "strategy_return"])
    return performance_metrics(
        window,
        periods_per_year=cfg.metrics_protocol.periods_per_year,
        volatility_ddof=cfg.metrics_protocol.volatility_ddof,
        expected_shortfall_quantile=(
            cfg.metrics_protocol.expected_shortfall_quantile),
        turnover_scale=cfg.metrics_protocol.turnover_scale,
        drawdown_basis="total_wealth",
    )


def pass_mask(key: str, metrics: dict[str, np.ndarray], market: str,
              delay: int) -> np.ndarray:
    if delay == 1:
        target = TABLE4[market]["fixed_jm"]
        ok = np.ones(len(metrics["cagr"]), dtype=bool)
        for m in METRICS:
            ok &= np.abs(metrics[m] - target[m]) <= TOL
        return ok
    target = TABLE5_JM[(market, delay)]
    ok = np.ones(len(metrics["cagr"]), dtype=bool)
    for m, value in target.items():
        ok &= np.abs(metrics[m] - value) <= TOL
    return ok


def digest_array(vectors: list[bytes]) -> np.ndarray:
    out = np.empty(len(vectors), dtype=np.uint64)
    for i, v in enumerate(vectors):
        out[i] = int.from_bytes(
            hashlib.blake2b(v, digest_size=8).digest(), "little")
    return out


def _intersect_task(task: tuple[int, int, int]) -> tuple[np.ndarray, list]:
    k, lo, hi = task
    keys = [arm_key(m, d) for m in MARKETS for d in DELAYS]
    caches = {key: _arm_cache(key) for key in keys}
    pass_sets = {key: np.load(OUT / f"{key}-pass-digests.npy")
                 for key in keys}
    n = len(caches[keys[0]]["lambdas"])
    combos = np.array(list(itertools.islice(
        itertools.combinations(range(n), k), lo, hi)), dtype=np.int64)
    bits = np.zeros((len(combos), len(keys)), dtype=bool)
    if not len(combos):
        return bits, []
    for start in range(0, len(combos), 4000):
        block = combos[start:start + 4000]
        for j, key in enumerate(keys):
            choices, invalid = choice_matrix(caches[key], block)
            digests = digest_array([choices[i].tobytes()
                                    for i in range(len(block))])
            found = np.isin(digests, pass_sets[key],
                            assume_unique=False)
            bits[start:start + len(block), j] = found & ~invalid
    all9 = bits.all(axis=1)

    def cols(suffix):
        return [j for j, key in enumerate(keys) if key.endswith(suffix)]

    d1_all = bits[:, cols("-d1")].all(axis=1)
    d5_all = bits[:, cols("-d5")].all(axis=1)
    d10_all = bits[:, cols("-d10")].all(axis=1)
    d510_all = d5_all & d10_all
    counts = {"d1_common": int(d1_all.sum()), "d5_common": int(d5_all.sum()),
              "d10_common": int(d10_all.sum()),
              "d5_and_d10_common": int(d510_all.sum()),
              "all_nine": int(all9.sum())}
    examples = []
    for tag, mask in (("d1_common", d1_all), ("d5_common", d5_all),
                      ("d10_common", d10_all), ("d5_and_d10", d510_all),
                      ("all_nine", all9)):
        for idx in np.nonzero(mask)[0][:8]:
            examples.append((tag, k, combos[idx].tolist()))
    return (bits.sum(axis=0), counts, examples)


def main() -> None:
    cfg = load_config(ROOT / "research-expanding-v9-4.toml")
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    executor = ProcessPoolExecutor(max_workers=N_JOBS,
                                   mp_context=get_context("forkserver"))
    parity_lines, arm_rows = [], []
    try:
        build_tasks = [(m, d) for m in MARKETS for d in DELAYS]
        todo = [t for t in build_tasks
                if not (OUT / f"{arm_key(*t)}-cache.npz").exists()]
        for line in executor.map(_build_arm, todo):
            print(line, flush=True)

        for market in MARKETS:
            for delay in DELAYS:
                key = arm_key(market, delay)
                if (OUT / f"{key}-pass-digests.npy").exists():
                    print(f"{key}: already evaluated, skipping", flush=True)
                    continue
                cache = _arm_cache(key)
                lambdas = list(cache["lambdas"])

                named_vecs, named_names = [], []
                for name, grid in NAMED_001.items():
                    idx = np.sort(np.array([lambdas.index(v) for v in grid],
                                           dtype=np.int64))
                    choices, invalid = choice_matrix(cache, idx[None, :])
                    named_vecs.append(choices[0].tobytes())
                    named_names.append(name)
                got = batch_metrics(
                    cache, np.stack([np.frombuffer(v, dtype=np.int8)
                                     for v in named_vecs]))
                # Every named grid checked against the real, unaccelerated
                # apply_signal + performance_metrics code path -- strictly
                # stronger than (and, since 2026-08-08, replaces) comparing
                # delay==1 against UNION_DIR/grids.csv, an external file
                # produced by a DIFFERENT script (probe_jm_grid_
                # identification.py) at whatever n_init IT used. That
                # cross-script dependency broke the moment this script's
                # own n_init could legitimately differ (the n_init=60
                # calibrated-only union, 2026-08-08): comparing an n_init=60
                # batch fit against an n_init=10 external reference fails
                # by design, not by defect. Delay 5/10 already used this
                # self-contained check (v9.4's metrics-exploratory.csv went
                # missing first); this makes delay==1 match and removes the
                # grids_001/UNION_DIR-file dependency entirely.
                worst = 0.0
                for i, _name in enumerate(named_names):
                    slow = slow_metrics(cache, named_vecs[i], cfg)
                    for m in METRICS:
                        worst = max(worst, abs(got[m][i] - slow[m]))
                if worst > 1e-9:
                    raise SystemExit(f"{key}: named parity FAILED {worst:.2e}")

                merged: dict[bytes, tuple[int, bytes]] = {}
                tasks = []
                n = len(lambdas)
                for k in SIZES:
                    total = comb(n, k)
                    step = max(1, total // (N_JOBS * 3))
                    for lo in range(0, total, step):
                        tasks.append((key, k, lo, min(lo + step, total)))
                for part in executor.map(_enumerate_task, tasks):
                    for vec, (count, example) in part.items():
                        prev = merged.get(vec)
                        if prev is None:
                            merged[vec] = (count, example)
                        else:
                            keep = (prev[1] if len(prev[1]) <= len(example)
                                    else example)
                            merged[vec] = (prev[0] + count, keep)
                uniques = list(merged.keys())
                print(f"{key}: {len(uniques)} unique vectors", flush=True)

                sample = rng.choice(len(uniques), size=min(50, len(uniques)),
                                    replace=False)
                fast = batch_metrics(
                    cache, np.stack([np.frombuffer(uniques[i], dtype=np.int8)
                                     for i in sample]))
                worst_s = 0.0
                for pos, i in enumerate(sample):
                    slow = slow_metrics(cache, uniques[i], cfg)
                    for m in METRICS:
                        worst_s = max(worst_s, abs(fast[m][pos] - slow[m]))
                if worst_s > 1e-9:
                    raise SystemExit(f"{key}: sample parity FAILED {worst_s:.2e}")
                line = (f"{key}: parity PASSED (named {worst:.2e}, "
                        f"50-sample {worst_s:.2e})")
                parity_lines.append(line)
                print(line, flush=True)

                step = max(2000, min(20000, len(uniques) // (N_JOBS * 3)))
                eval_tasks = [(key, uniques[i:i + step])
                              for i in range(0, len(uniques), step)]
                metric_parts = list(executor.map(_evaluate_task, eval_tasks))
                metrics = {m: np.concatenate([p[m] for p in metric_parts])
                           for m in METRICS}
                ok = pass_mask(key, metrics, market, delay)
                digests = digest_array(uniques)
                np.save(OUT / f"{key}-pass-digests.npy",
                        np.sort(digests[ok]))
                np.savez_compressed(
                    OUT / f"{key}-results.npz", digests=digests, ok=ok,
                    **{m: metrics[m] for m in METRICS})
                counts = int(ok.sum())
                arm_rows.append({"arm": key, "unique_vectors": len(uniques),
                                 "passing_vectors": counts})
                print(f"{key}: passing vectors {counts}", flush=True)
                _CACHE.clear()

        n = 29
        tasks = []
        for k in SIZES:
            total = comb(n, k)
            step = max(1, total // (N_JOBS * 2))
            for lo in range(0, total, step):
                tasks.append((k, lo, min(lo + step, total)))
        keys = [arm_key(m, d) for m in MARKETS for d in DELAYS]
        totals = np.zeros(len(keys), dtype=np.int64)
        examples = []
        agg = {"d1_common": 0, "d5_common": 0, "d10_common": 0,
               "d5_and_d10_common": 0, "all_nine": 0}
        for bits_sum, counts, ex in executor.map(_intersect_task, tasks):
            totals += bits_sum
            for key2, value in counts.items():
                agg[key2] += value
            for tag, k, combo in ex:
                if len(examples) < 200:
                    examples.append((tag, k, combo))
    finally:
        executor.shutdown(cancel_futures=True)

    (OUT / "parity-note.txt").write_text("\n".join(parity_lines) + "\n",
                                         encoding="utf-8")
    pd.DataFrame(arm_rows).to_csv(OUT / "arms.csv", index=False,
                                  lineterminator="\n")
    lambdas = np.load(OUT / "us-d1-cache.npz")["lambdas"]
    ex_rows = [{"criterion": tag, "size": k,
                "grid": "|".join(f"{lambdas[i]:g}" for i in combo)}
               for tag, k, combo in examples]
    pd.DataFrame(ex_rows).to_csv(OUT / "solutions.csv", index=False,
                                 lineterminator="\n")
    pd.DataFrame([agg]).to_csv(OUT / "intersections.csv", index=False,
                               lineterminator="\n")
    lines = ["jm-grid-exhaustive-008 — kết quả",
             "(tìm-kiếm-theo-đích theo lệnh owner; nghiệm = lưới hiệu chuẩn;"
             " cấm nhận nuôi)", ""]
    for r in arm_rows:
        lines.append(f"   {r['arm']}: {r['passing_vectors']} vector đạt /"
                     f" {r['unique_vectors']}")
    lines.append("")
    lines.append("Giao-tập trên TOÀN lattice 6.474.511 tập con:")
    for key2, value in agg.items():
        lines.append(f"   {key2}: {value}")
    report = "\n".join(lines) + "\n"
    (OUT / "report.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
