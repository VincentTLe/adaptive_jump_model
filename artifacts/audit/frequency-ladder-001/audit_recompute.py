"""AUDIT steps 2 + 4: independent re-derivation of the menu and of the 8 cells.

Nothing is imported from scripts/score_grid.py or scripts/run_frequency_ladder.py.
Only the library (walkforward, backtest, config) is used, plus the published
Table 4 numbers typed in below from the paper text.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/tle/adaptive_jump_model")
sys.path.insert(0, str(ROOT / "src"))

from adaptive_jump.backtest import apply_signal, performance_metrics  # noqa: E402
from adaptive_jump.config import load_config  # noqa: E402
from adaptive_jump.walkforward import select_monthly_candidate  # noqa: E402

BASE = ROOT / "artifacts/fixed-baselines/fixed-baselines-36ca1ace131c-ed7abd7daea3-f9f3e0a93736"
UNION = ROOT / "artifacts/jm-residual/01-grid-identification"
LADDER_STATES = ROOT / "artifacts/frequency-ladder/01-run"
DEST = Path("/tmp/claude-1017/-home-tle/69649cec-6fd3-40f9-9e01-42dd56f3559f/scratchpad")
CELLS = ("cagr", "volatility", "sharpe", "maximum_drawdown", "calmar",
         "expected_shortfall_5pct", "turnover", "leverage")
TY = 3000 / 252

# Shu, Yu & Mulvey (2024) Table 4, JM row, typed independently from the PDF text.
SHU = {
    "us": dict(cagr=0.112, volatility=0.131, sharpe=0.68, maximum_drawdown=-0.266,
               calmar=0.33, expected_shortfall_5pct=-0.020, turnover=0.44, leverage=0.80),
    "de": dict(cagr=0.086, volatility=0.164, sharpe=0.44, maximum_drawdown=-0.394,
               calmar=0.18, expected_shortfall_5pct=-0.025, turnover=1.70, leverage=0.84),
    "jp": dict(cagr=0.047, volatility=0.171, sharpe=0.31, maximum_drawdown=-0.453,
               calmar=0.12, expected_shortfall_5pct=-0.026, turnover=0.72, leverage=0.75),
}
ARMS = {
    "L_full_ladder": {
        "us": [0.55, 1.465348864441625, 5.459611390906074, 23.5, 37.5, 65.0],
        "de": [0.55, 2.829145724599095, 8.59842836500576, 23.5, 90.0, 185.0],
        "jp": [0.55, 2.829145724599095, 8.59842836500576, 17.5, 45.0, 75.0],
    },
    "M_no_freeze": {
        "us": [0.55, 1.465348864441625, 5.459611390906074, 23.5, 37.5],
        "de": [0.55, 2.829145724599095, 8.59842836500576, 23.5, 90.0],
        "jp": [0.55, 2.829145724599095, 8.59842836500576, 17.5, 45.0],
    },
}


def my_derive(market, ladder, cutoff="1990-01-01", stat="median", edge="mid",
              op="le"):
    """Independent re-implementation of the frozen inversion."""
    r = pd.read_csv(UNION / market / "union-refits.csv")
    r["fit_date"] = pd.to_datetime(r["fit_date"])
    pre = r[r.fit_date < pd.Timestamp(cutoff)] if cutoff else r
    per = {t: [] for t in ladder}
    for _, g in pre.groupby("fit_date"):
        g = g.sort_values("lambda")
        lam = g["lambda"].to_numpy(float)
        obj = g["objective"].to_numpy(float)
        rep = {"mid": (lam[:-1] + lam[1:]) / 2.0, "left": lam[:-1],
               "right": lam[1:]}[edge]
        rate = np.diff(obj) / np.diff(lam) / TY
        for t in ladder:
            hit = np.flatnonzero(rate <= t if op == "le" else rate >= t)
            per[t].append(rep[hit[0]] if hit.size else np.nan)
    agg = np.nanmedian if stat == "median" else np.nanmean
    return [float(agg(per[t])) for t in ladder]


def my_score(market, menu, states_csv):
    cfg = load_config(ROOT / "research-calibrated-v10.toml")
    frame = pd.read_csv(BASE / market / "features.csv", parse_dates=["date"])
    st = pd.read_csv(states_csv, index_col=0, parse_dates=[0])
    st.columns = [float(c) for c in st.columns]
    cols = []
    for v in menu:
        hit = [c for c in st.columns if abs(c - float(v)) < 1e-9]
        assert len(hit) == 1, (market, v, sorted(st.columns))
        cols.append(hit[0])
    sel = select_monthly_candidate(
        frame[["date", "equity_simple", "cash_return"]], st.loc[:, cols],
        cfg.selection_protocol, delay_trading_days=1, one_way_cost_bps=10.0,
        periods_per_year=cfg.metrics_protocol.periods_per_year,
        volatility_ddof=cfg.metrics_protocol.volatility_ddof)
    sig = sel.signal.rename("s").reset_index()
    sig.columns = ["date", "s"]
    merged = frame.merge(sig, on="date", how="left")
    path = apply_signal(merged[["date", "equity_simple", "cash_return"]],
                        merged["s"], delay_trading_days=1, one_way_cost_bps=10.0)
    rep = pd.read_csv(BASE / "metrics.csv", parse_dates=["start", "end"])
    row = rep[(rep.market == market) & (rep.model == "fixed_jm")
              & (rep.delay == 1)].iloc[0]
    kept = path[(path.date >= row.start) & (path.date <= row.end)].dropna(
        subset=["cash_return", "position", "one_way_turnover", "strategy_return"])
    got = performance_metrics(
        kept, periods_per_year=cfg.metrics_protocol.periods_per_year,
        volatility_ddof=cfg.metrics_protocol.volatility_ddof,
        expected_shortfall_quantile=cfg.metrics_protocol.expected_shortfall_quantile,
        turnover_scale=cfg.metrics_protocol.turnover_scale,
        drawdown_basis="total_wealth")
    got["switches"] = int((kept["position"].diff().abs() > 0).sum())
    got["window"] = f"{row.start.date()}..{row.end.date()}"
    got["n"] = len(kept)
    return got


if __name__ == "__main__":
    ladder = [8.0, 4.0, 2.0, 1.0, 0.5, 0.25]
    print("=== independent re-derivation, pre-1990 windows, median, midpoint ===")
    for m in ("us", "de", "jp"):
        mine = my_derive(m, ladder)
        frozen = ARMS["L_full_ladder"][m]
        ok = np.allclose(mine, frozen, rtol=1e-12, atol=1e-12)
        print(f"{m}: {mine}\n   frozen {frozen}\n   EXACT MATCH={ok}")

    print("\n=== independent recomputation of the eight cells ===")
    rows = []
    for arm, menus in ARMS.items():
        for m in ("us", "de", "jp"):
            got = my_score(m, menus[m], LADDER_STATES / f"states-{m}.csv")
            worst = 0.0
            for c in CELLS:
                rel = abs(got[c] - SHU[m][c]) / abs(SHU[m][c])
                worst = max(worst, rel)
                rows.append({"arm": arm, "market": m, "cell": c,
                             "mine": got[c], "shu": SHU[m][c], "rel": rel})
            print(f"{arm:>14} {m}  window {got['window']} n={got['n']} "
                  f"switches={got['switches']}  worst={worst:.4%}")
    out = pd.DataFrame(rows)
    out.to_csv(DEST / "audit-cells.csv", index=False)

    claimed = pd.read_csv(LADDER_STATES / "cells.csv")
    j = out.merge(claimed, on=["arm", "market", "cell"], suffixes=("_mine", "_claim"))
    j["abs_gap"] = (j["mine"] - j["ours"]).abs()
    j["rel_gap"] = (j["rel"] - j["relative_deviation"]).abs()
    print(f"\nrows compared: {len(j)}")
    print(f"largest absolute cell disagreement: {j.abs_gap.max():.3e}")
    print(j.sort_values("abs_gap").tail(3)[
        ["arm", "market", "cell", "mine", "ours", "abs_gap"]].to_string(index=False))
    j.to_csv(DEST / "audit-cells-vs-claim.csv", index=False)
