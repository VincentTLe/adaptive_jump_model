"""jm-per-market-grid-009: per-market grids across all three delays + frontiers.

Frozen spec: research/jm-per-market-grid-009.toml. Consumes the -008
persisted per-arm results (adversarially verified). Every nonexistence
statement is scoped to subsets (sizes 2-8) of the 29-lambda sourced menu.
"""

from __future__ import annotations

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
from probe_jm_grid_exhaustive2 import (  # noqa: E402
    TABLE5_JM,
    _arm_cache,
    arm_key,
    choice_matrix,
    digest_array,
)

IN = ROOT / "artifacts" / "jm-residual" / "08-exhaustive-nine-arms"
OUT = ROOT / "artifacts" / "jm-residual" / "09-per-market-grids"
N_JOBS, TOL = 30, 0.05
SIZES = range(2, 9)
MARKETS = ("us", "de", "jp")
DELAYS = (1, 5, 10)


def _market_sweep_task(task: tuple[str, int, int, int]) -> tuple[int, list]:
    market, k, lo, hi = task
    keys = [arm_key(market, d) for d in DELAYS]
    caches = {key: _arm_cache(key) for key in keys}
    pass_sets = {key: np.load(IN / f"{key}-pass-digests.npy") for key in keys}
    n = len(caches[keys[0]]["lambdas"])
    combos = np.array(list(itertools.islice(
        itertools.combinations(range(n), k), lo, hi)), dtype=np.int64)
    if not len(combos):
        return 0, []
    hits = 0
    examples = []
    for start in range(0, len(combos), 4000):
        block = combos[start:start + 4000]
        good = np.ones(len(block), dtype=bool)
        for key in keys:
            choices, invalid = choice_matrix(caches[key], block)
            digests = digest_array([choices[i].tobytes()
                                    for i in range(len(block))])
            good &= np.isin(digests, pass_sets[key]) & ~invalid
            if not good.any():
                break
        hits += int(good.sum())
        for idx in np.nonzero(good)[0][:5]:
            if len(examples) < 20:
                examples.append((k, block[idx].tolist()))
    return hits, examples


def frontier(market: str) -> list[dict]:
    key = arm_key(market, 1)
    z = np.load(IN / f"{key}-results.npz")
    target = TABLE4[market]["fixed_jm"]
    devs = {m: np.abs(z[m] - target[m]) for m in METRICS}
    ok_counts = sum((devs[m] <= TOL).astype(np.int8) for m in METRICS)
    rows = []
    for m in METRICS:
        passing_m = devs[m] <= TOL
        others = [x for x in METRICS if x != m]
        others_ok = np.ones(len(z[m]), dtype=bool)
        for x in others:
            others_ok &= devs[x] <= TOL
        rows.append({
            "market": market, "cell": m, "target": target[m],
            "pass_rate": float(passing_m.mean()),
            "lattice_min": float(np.nanmin(z[m])),
            "lattice_max": float(np.nanmax(z[m])),
            "best_dev_given_others_pass": (
                float(devs[m][others_ok].min()) if others_ok.any() else np.nan),
            "count_others_pass": int(others_ok.sum()),
        })
    rows.append({"market": market, "cell": "BEST_within_tol",
                 "target": 8, "pass_rate": float((ok_counts == 8).mean()),
                 "lattice_min": int(ok_counts.min()),
                 "lattice_max": int(ok_counts.max()),
                 "best_dev_given_others_pass": np.nan,
                 "count_others_pass": int((ok_counts == ok_counts.max()).sum())})
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    executor = ProcessPoolExecutor(max_workers=N_JOBS,
                                   mp_context=get_context("forkserver"))
    inter_rows, example_map = [], {}
    try:
        for market in MARKETS:
            n = 29
            tasks = []
            for k in SIZES:
                total = comb(n, k)
                step = max(1, total // (N_JOBS * 2))
                for lo in range(0, total, step):
                    tasks.append((market, k, lo, min(lo + step, total)))
            hits = 0
            examples = []
            for h, ex in executor.map(_market_sweep_task, tasks):
                hits += h
                for k, combo in ex:
                    if len(examples) < 20:
                        examples.append((k, combo))
            inter_rows.append({"market": market,
                               "d1_and_d5_and_d10_count": hits})
            example_map[market] = examples
            print(f"{market}: grids passing ALL THREE delays = {hits}",
                  flush=True)
    finally:
        executor.shutdown(cancel_futures=True)

    lambdas = np.load(IN / "us-d1-cache.npz")["lambdas"]
    ex_rows = []
    for market, examples in example_map.items():
        for k, combo in examples:
            ex_rows.append({"market": market, "size": k,
                            "grid": "|".join(f"{lambdas[i]:g}" for i in combo)})
    pd.DataFrame(inter_rows).to_csv(OUT / "per-market-intersections.csv",
                                    index=False, lineterminator="\n")
    pd.DataFrame(ex_rows).to_csv(OUT / "per-market-examples.csv",
                                 index=False, lineterminator="\n")

    frontier_rows = []
    for market in ("de", "jp"):
        frontier_rows.extend(frontier(market))
    fr = pd.DataFrame(frontier_rows)
    fr.to_csv(OUT / "frontier.csv", index=False, lineterminator="\n")

    lines = ["jm-per-market-grid-009 — kết quả",
             "(phạm vi MỌI phát biểu: tập con cỡ 2-8 của menu 29 λ có nguồn;"
             " nghiệm = lưới hiệu chuẩn; reseal là quyết định owner)", ""]
    lines.append("Q1. Lưới riêng từng thị trường đạt CẢ d1 (8 ô) + d5 + d10"
                 " (3 ô mỗi delay):")
    for r in inter_rows:
        lines.append(f"   {r['market']}: {r['d1_and_d5_and_d10_count']}"
                     " lưới trong menu")
    lines.append("")
    lines.append("Q2. Biên chặn tại delay-1 cho DE/JP (ô nào chặn, cách bao xa):")
    for market in ("de", "jp"):
        sub = fr[fr.market == market]
        best = sub[sub.cell == "BEST_within_tol"].iloc[0]
        lines.append(f"   {market}: tốt nhất {int(best.lattice_max)}/8 ô"
                     f" ({int(best.count_others_pass)} vector đạt mức đó)")
        for _, r in sub[sub.cell != "BEST_within_tol"].iterrows():
            flag = ""
            if r.pass_rate < 0.001:
                flag = "  <-- Ô CHẶN"
            lines.append(f"      {r.cell:24s} đích {r.target:8.3f} |"
                         f" phổ lattice [{r.lattice_min:8.3f},"
                         f" {r.lattice_max:8.3f}] | tỉ lệ đạt"
                         f" {r.pass_rate:7.2%} | lệch nhỏ nhất khi 7 ô kia đạt:"
                         f" {r.best_dev_given_others_pass}"
                         f" ({int(r.count_others_pass)} vector){flag}")
    report = "\n".join(lines) + "\n"
    (OUT / "report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"đã ghi {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
