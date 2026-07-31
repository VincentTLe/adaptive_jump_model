"""jm-grid-exhaustive-007: owner-instructed target-conditioned exhaustive search.

Frozen spec: research/jm-grid-exhaustive-007.toml — read its honesty label
first: solutions are CALIBRATION artifacts, never evidence about the authors'
grid; the scientific content is the existence map.

Machinery: one select_monthly_candidate pass per market on the 29-lambda
union gives a per-(decision, lambda) score/eligibility surface; per-candidate
scores are set-independent, so every subset's monthly choices follow from the
surface (tie 1e-12 toward the lower lambda, the contract rule). Subsets
sharing a choice vector are one outcome; each unique vector is scored through
the contract's own _compose_selected_signal semantics + apply_signal +
performance_metrics. A parity gate must reproduce -001's grids.csv rows to
1e-9 in all three markets before the search is read.
"""

from __future__ import annotations

import itertools
import sys
from concurrent.futures import ProcessPoolExecutor
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

RUN = ROOT / "artifacts" / "fixed-baselines" / (
    "fixed-baselines-34e51cd7a388-967806b961b4-e690dbe396f3"
)
UNION_DIR = ROOT / "artifacts" / "jm-residual" / "01-grid-identification"
OUT = ROOT / "artifacts" / "jm-residual" / "07-exhaustive-search"
DELAY, COST, TOL, N_JOBS = 1, 10.0, 0.05, 30
TIE = 1e-12
SIZES = range(2, 9)
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


def _market_cache(market: str) -> dict:
    if market not in _CACHE:
        data = np.load(OUT / f"{market}-cache.npz", allow_pickle=False)
        _CACHE[market] = {key: data[key] for key in data.files}
    return _CACHE[market]


def build_market(market: str, cfg) -> dict:
    """One selection pass; save the surface and composition arrays."""
    frame = pd.read_csv(RUN / market / "features.csv", parse_dates=["date"])
    states = pd.read_csv(
        UNION_DIR / market / "union-states.csv", parse_dates=["date"]
    ).set_index("date")
    states.columns = [float(c) for c in states.columns]
    lambdas = np.array(sorted(states.columns), dtype=float)
    states = states.loc[:, list(lambdas)]

    selection = select_monthly_candidate(
        frame[["date", "equity_simple", "cash_return"]], states,
        cfg.selection_protocol, delay_trading_days=DELAY, one_way_cost_bps=COST,
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
    if not dates.equals(states.index):
        raise SystemExit(f"{market}: frame and state dates differ")
    governing = np.searchsorted(decisions.values, dates.values,
                                side="right") - 1
    reported = pd.read_csv(RUN / "metrics-exploratory.csv",
                           parse_dates=["start", "end"])
    row = reported[(reported.market == market) & (reported.model == "fixed_jm")
                   & (reported.delay == DELAY)].iloc[0]
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / f"{market}-cache.npz",
        score=np.where(np.isfinite(score), score, -np.inf),
        raw_finite=np.isfinite(score),
        elig=elig,
        states=states.to_numpy(),
        lambdas=lambdas,
        governing=governing,
        equity=frame["equity_simple"].to_numpy(),
        cash=frame["cash_return"].to_numpy(),
        dates=dates.values.astype("datetime64[ns]").astype("int64"),
        window=np.array([row["start"].value, row["end"].value]),
    )
    return {"lambdas": lambdas, "n_decisions": len(decisions)}


def choice_vector(cache: dict, member_idx: np.ndarray) -> tuple[bytes, bool]:
    """Monthly choices for one subset (indices into lambdas, ascending)."""
    scores = np.where(cache["elig"][:, member_idx],
                      cache["score"][:, member_idx], -np.inf)
    best = scores.max(axis=1)
    none = ~np.isfinite(best)
    started = np.cumsum(~none) > 0
    invalid = bool((none & started).any())
    winners = scores >= (best[:, None] - TIE)
    first = winners.argmax(axis=1)
    choice = member_idx[first].astype(np.int8)
    choice[none] = -1
    return choice.tobytes(), invalid


def _enumerate_task(task: tuple[str, int, int, int]) -> dict:
    market, k, lo, hi = task
    cache = _market_cache(market)
    n = len(cache["lambdas"])
    out: dict[bytes, tuple[int, bytes]] = {}
    combos = itertools.islice(itertools.combinations(range(n), k), lo, hi)
    for combo in combos:
        member_idx = np.array(combo, dtype=np.int64)
        vec, invalid = choice_vector(cache, member_idx)
        if invalid:
            continue
        prev = out.get(vec)
        if prev is None:
            out[vec] = (1, np.array(combo, dtype=np.int8).tobytes())
        else:
            out[vec] = (prev[0] + 1, prev[1])
    return out


def _evaluate_task(task: tuple[str, list[bytes]]) -> list[dict]:
    market, vectors = task
    cache = _market_cache(market)
    cfg = load_config(ROOT / "research-expanding-v9-4.toml")
    dates = pd.to_datetime(cache["dates"])
    prepared = pd.DataFrame({
        "date": dates,
        "equity_simple": cache["equity"],
        "cash_return": cache["cash"],
    })
    governing = cache["governing"]
    states = cache["states"]
    lo = pd.Timestamp(cache["window"][0])
    hi = pd.Timestamp(cache["window"][1])
    rows = []
    for vec in vectors:
        choice = np.frombuffer(vec, dtype=np.int8)
        active = np.where(governing >= 0, choice[np.clip(governing, 0, None)],
                          -1)
        signal = np.full(len(dates), np.nan)
        has = active >= 0
        signal[has] = 1.0 - states[np.nonzero(has)[0], active[has]]
        path = apply_signal(prepared, pd.Series(signal),
                            delay_trading_days=DELAY, one_way_cost_bps=COST)
        window = path[(path["date"] >= lo) & (path["date"] <= hi)].dropna(
            subset=["cash_return", "position", "one_way_turnover",
                    "strategy_return"])
        scored = performance_metrics(
            window,
            periods_per_year=cfg.metrics_protocol.periods_per_year,
            volatility_ddof=cfg.metrics_protocol.volatility_ddof,
            expected_shortfall_quantile=(
                cfg.metrics_protocol.expected_shortfall_quantile),
            turnover_scale=cfg.metrics_protocol.turnover_scale,
            drawdown_basis="total_wealth",
        )
        target = TABLE4[market]["fixed_jm"]
        devs = {m: abs(scored[m] - target[m]) for m in METRICS}
        rows.append({
            "vector": vec,
            **{m: scored[m] for m in METRICS},
            "within_tol": sum(d <= TOL for d in devs.values()),
            "max_dev": max(devs.values()),
        })
    return rows


def parity_gate(market: str, cfg, executor) -> str:
    cache = _market_cache(market)
    lambdas = list(cache["lambdas"])
    grids = pd.read_csv(UNION_DIR / "grids.csv")
    vectors = []
    names = []
    for name, grid in NAMED_001.items():
        member_idx = np.array([lambdas.index(v) for v in grid], dtype=np.int64)
        vec, invalid = choice_vector(cache, np.sort(member_idx))
        if invalid:
            raise SystemExit(f"{market}: named grid {name} invalid?")
        vectors.append(vec)
        names.append(name)
    rows = _evaluate_task((market, vectors))
    worst = 0.0
    for name, row in zip(names, rows, strict=True):
        want = grids[(grids.market == market) & (grids.grid == name)].iloc[0]
        for m in METRICS:
            worst = max(worst, abs(row[m] - want[m]))
    if worst > 1e-9:
        raise SystemExit(f"{market}: parity gate FAILED, worst {worst:.2e}")
    return (f"{market}: parity gate PASSED — evaluator reproduces the eight "
            f"-001 grids to {worst:.2e}")


def main() -> None:
    cfg = load_config(ROOT / "research-expanding-v9-4.toml")
    OUT.mkdir(parents=True, exist_ok=True)
    parity_lines, frontier_rows, solution_rows = [], [], []
    executor = ProcessPoolExecutor(max_workers=N_JOBS,
                                   mp_context=get_context("forkserver"))
    try:
        for market in ("us", "de", "jp"):
            info = build_market(market, cfg)
            lambdas = info["lambdas"]
            n = len(lambdas)
            line = parity_gate(market, cfg, executor)
            parity_lines.append(line)
            print(line, flush=True)

            from math import comb
            merged: dict[bytes, tuple[int, bytes]] = {}
            tasks = []
            for k in SIZES:
                total = comb(n, k)
                step = max(1, total // (N_JOBS * 4))
                for lo in range(0, total, step):
                    tasks.append((market, k, lo, min(lo + step, total)))
            for part in executor.map(_enumerate_task, tasks):
                for vec, (count, example) in part.items():
                    prev = merged.get(vec)
                    if prev is None:
                        merged[vec] = (count, example)
                    else:
                        keep = prev[1] if len(prev[1]) <= len(example) else example
                        merged[vec] = (prev[0] + count, keep)
            uniques = list(merged.keys())
            print(f"{market}: {sum(c for c, _ in merged.values())} valid "
                  f"subsets -> {len(uniques)} unique choice vectors",
                  flush=True)

            step = max(1, len(uniques) // (N_JOBS * 4))
            eval_tasks = [(market, uniques[i:i + step])
                          for i in range(0, len(uniques), step)]
            results = []
            for part in executor.map(_evaluate_task, eval_tasks):
                results.extend(part)

            best = max(r["within_tol"] for r in results)
            counts = pd.Series([r["within_tol"] for r in results]
                               ).value_counts().sort_index()
            frontier_rows.append({
                "market": market, "unique_vectors": len(uniques),
                "best_within_tol": best,
                "distribution": ";".join(f"{k}:{v}" for k, v in counts.items()),
            })
            for r in sorted(results, key=lambda r: (-r["within_tol"],
                                                    r["max_dev"]))[:25]:
                count, example = merged[r["vector"]]
                grid = [float(lambdas[i]) for i in
                        np.frombuffer(example, dtype=np.int8)]
                solution_rows.append({
                    "market": market,
                    "within_tol": r["within_tol"],
                    "example_grid": "|".join(f"{v:g}" for v in sorted(grid)),
                    "subsets_mapping_here": count,
                    **{m: r[m] for m in METRICS},
                    "max_dev": r["max_dev"],
                })
            n_solutions = sum(r["within_tol"] == 8 for r in results)
            print(f"{market}: best {best}/8; full 8/8 vectors: {n_solutions}",
                  flush=True)
    finally:
        executor.shutdown(cancel_futures=True)

    (OUT / "parity-note.txt").write_text("\n".join(parity_lines) + "\n",
                                         encoding="utf-8")
    pd.DataFrame(frontier_rows).to_csv(OUT / "frontier.csv", index=False,
                                       lineterminator="\n")
    sol = pd.DataFrame(solution_rows)
    sol.to_csv(OUT / "solutions.csv", index=False, lineterminator="\n")

    lines = ["jm-grid-exhaustive-007 — kết quả",
             "(TÌM-KIẾM-THEO-ĐÍCH theo lệnh owner; nghiệm là lưới HIỆU CHUẨN,"
             " không phải lưới của tác giả; cấm nhận nuôi)", ""]
    for f in frontier_rows:
        lines.append(f"{f['market']}: {f['unique_vectors']} vector duy nhất |"
                     f" tốt nhất {f['best_within_tol']}/8 |"
                     f" phân bố {f['distribution']}")
    lines.append("")
    lines.append("Top nghiệm/biên (xem solutions.csv, 25 dòng đầu mỗi thị"
                 " trường):")
    for _, r in sol.groupby("market").head(3).iterrows():
        lines.append(f"   {r.market} {int(r.within_tol)}/8"
                     f" [{r.example_grid}] (cùng vector: "
                     f"{int(r.subsets_mapping_here)} tập con)"
                     f" sharpe {r.sharpe:.3f} turnover {r.turnover:.3f}")
    report = "\n".join(lines) + "\n"
    (OUT / "report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"đã ghi {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
