"""Run only the HMM arm of v9, and read it against Table 4.

The full replication spends about three quarters of its wall clock in the jump
model, which is not parallelised. Only the HMM is under investigation here, so
this fits the HMM alone and skips the rest.

Germany and Japan are untouched by v9 -- their equity series do not change, so
their HMM results are the sealed v8.5 ones by construction, and a test asserts
the two market definitions are byte-identical. Only the US is recomputed.

The scoring window is v8.5's reported sample, 1990-01-02..2023-12-29, so v8.5
and v9 are compared over exactly the same days. That is not the window a sealed
v9 run would derive for itself -- it intersects complete rows across the jump
model too -- so treat this as an exploratory readout, not a sealed result.

GUARD: before any v9 number is printed, the same selection-and-metrics path is
run on the sealed v8.5 states and must reproduce that run's published metric row.

Prediction, written before this ran, and already committed in c6a7889: the US
drawdown deepens from -19.72% toward the -28.9% Figure 6 confirms, regime shifts
fall from 128 toward the published 96, and Sharpe falls from 0.614 toward 0.54.
If the US does not move, the diagnosis in docs/audit/2026-07-full-audit.md is
wrong.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from adaptive_jump.backtest import apply_signal, performance_metrics  # noqa: E402
from adaptive_jump.cli import load_frozen_data, prepare_manifest_market  # noqa: E402
from adaptive_jump.config import load_config  # noqa: E402
from adaptive_jump.models import hmm_states, smoothed_hmm_states  # noqa: E402
from adaptive_jump.walkforward import select_monthly_candidate  # noqa: E402

SEALED = ROOT / ("artifacts/fixed-baselines/"
                 "fixed-baselines-7b95ec50dece-6bd27647967d-13641890668f")
DELAY, COST = 1, 10.0
SHU = dict(sharpe=.54, cagr=.085, volatility=.113, maximum_drawdown=-.289,
           calmar=.21, expected_shortfall_5pct=-.018, turnover=1.41,
           leverage=.72)
LBL = [("sharpe", "Sharpe"), ("cagr", "Return"), ("volatility", "Vol"),
       ("maximum_drawdown", "MDD"), ("calmar", "Calmar"),
       ("expected_shortfall_5pct", "ES 5%"), ("turnover", "Turnover"),
       ("leverage", "Leverage")]


def score(frame: pd.DataFrame, states: pd.Series, cfg, lo, hi) -> dict:
    cands = smoothed_hmm_states(states, cfg.hmm_protocol.smoothing_grid)
    sel = select_monthly_candidate(
        frame[["date", "equity_simple", "cash_return"]],
        cands,
        cfg.selection_protocol,
        delay_trading_days=DELAY,
        one_way_cost_bps=COST,
        periods_per_year=cfg.metrics_protocol.periods_per_year,
        volatility_ddof=cfg.metrics_protocol.volatility_ddof,
    )
    sig = sel.signal.rename("signal").reset_index()
    sig.columns = ["date", "signal"]
    merged = frame.merge(sig, on="date", how="left")
    path = apply_signal(
        merged[["date", "equity_simple", "cash_return"]],
        merged["signal"],
        delay_trading_days=DELAY,
        one_way_cost_bps=COST,
    )
    w = path[(path["date"] >= lo) & (path["date"] <= hi)].dropna(
        subset=["cash_return", "position", "one_way_turnover", "strategy_return"]
    )
    out = performance_metrics(
        w,
        periods_per_year=cfg.metrics_protocol.periods_per_year,
        volatility_ddof=cfg.metrics_protocol.volatility_ddof,
    )
    out["shifts"] = int((w["position"].diff().abs() > 0).sum())
    out["picks"] = sel.choices["selected"].value_counts(normalize=True).sort_index()
    return out


def main() -> None:
    workers = max(1, (os.cpu_count() or 4) - 2)
    reported = pd.read_csv(SEALED / "metrics-exploratory.csv",
                           parse_dates=["start", "end"])
    row = reported[(reported.market == "us") & (reported.model == "hmm")
                   & (reported.delay == DELAY)].iloc[0]
    lo, hi = row["start"], row["end"]

    # --- guard: reproduce the sealed v8.5 row from its own states -----------
    cfg85 = load_config(ROOT / "research-expanding-v8-5.toml")
    f85 = pd.read_csv(SEALED / "us" / "features.csv", parse_dates=["date"])
    s85 = pd.read_csv(SEALED / "us" / "hmm-states.csv",
                      parse_dates=["date"]).set_index("date")["hmm_state"]
    got = score(f85, s85, cfg85, lo, hi)
    drift = max(abs(got[k] - float(row[k])) for k, _ in LBL)
    print(f"guard: v8.5 tái lập hàng metric niêm phong, sai lệch tối đa {drift:.2e}")
    if drift > 1e-9:
        raise SystemExit("GUARD FAILED — không dùng được số nào bên dưới")

    # --- v9: refit the HMM on the S&P 500 series ----------------------------
    cfg9 = load_config(ROOT / "research-expanding-v9.toml")
    frozen = load_frozen_data(cfg9)
    us = prepare_manifest_market(cfg9, frozen, "us")
    print(f"v9 khung Mỹ: {len(us.frame)} dòng, OOS bắt đầu {us.oos_start}")
    print(f"khớp HMM trên {workers} worker …", flush=True)
    res = hmm_states(us.frame, cfg9.model_protocol, cfg9.hmm_protocol,
                     n_jobs=workers)
    new = score(us.frame, res.states, cfg9, lo, hi)

    print(f"\nHMM Mỹ, delay-1, {lo.date()}..{hi.date()}\n")
    print(f"  {'':<10}{'v8.5 (CRSP)':>13}{'v9 (S&P 500)':>14}{'Shu':>8}"
          f" | {'|lệch| v8.5':>12}{'|lệch| v9':>10}")
    for k, lab in LBL:
        da, db = abs(got[k] - SHU[k]), abs(new[k] - SHU[k])
        mark = "  tốt hơn" if db < da - 1e-9 else ("  tệ hơn" if db > da + 1e-9 else "")
        print(f"  {lab:<10}{got[k]:>13.4f}{new[k]:>14.4f}{SHU[k]:>8.3f}"
              f" | {da:>12.3f}{db:>10.3f}{mark}")
    print(f"  {'lần đổi':<10}{got['shifts']:>13d}{new['shifts']:>14d}{96:>8d}"
          f" | {abs(got['shifts'] - 96):>12d}{abs(new['shifts'] - 96):>10d}")

    for tag, r in (("v8.5", got), ("v9", new)):
        share = "  ".join(f"k{int(k)}:{v:.0%}" for k, v in r["picks"].items())
        print(f"\n  k được chọn, {tag}: {share}")

    n05 = sum(abs(new[k] - SHU[k]) <= 0.05 for k, _ in LBL)
    o05 = sum(abs(got[k] - SHU[k]) <= 0.05 for k, _ in LBL)
    print(f"\n  trong ngưỡng 0.05: v8.5 {o05}/8  ->  v9 {n05}/8")


if __name__ == "__main__":
    main()
