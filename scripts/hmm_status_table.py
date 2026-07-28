"""Where every HMM cell stands against Table 4, for all three markets at once.

Earlier readouts looked at one market at a time, which is how a deviation common
to all three can hide: each market gets explained by its own local story.

Every number is recomputed here from the stored position paths rather than read
out of a sealed metrics file, because the drawdown basis is now an explicit
protocol choice and both readings are worth printing side by side:

  total_wealth                  the cash leg earns the bill rate (v8.x, v9)
  risky_leg_wealth_flat_in_cash the cash leg contributes nothing (v9.1), which
                                Table 4's buy-and-hold row and the caption of
                                Figure 5 pin down together

Sources, all inside the repository:
  us      artifacts/hmm-residual/v9-us-hmm/  (the S&P 500 the paper names)
  de, jp  the sealed v8.5 run, whose market definitions v9 leaves untouched
  Shu     scripts/_shu_table4.py

Regime shifts are compared against the count Shu's turnover row implies through
the identity the paper states in words at line 781-783: turnover of 44% means
"44% of total allocation (a combined 88% trading) each year", so shifts per year
is twice the turnover.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402

from _shu_table4 import LABELS, METRICS, PRINTED_HALF_UNIT, TABLE4  # noqa: E402
from adaptive_jump.backtest import apply_signal, performance_metrics  # noqa: E402

SEALED = ROOT / ("artifacts/fixed-baselines/"
                 "fixed-baselines-7b95ec50dece-6bd27647967d-13641890668f")
V9 = ROOT / "artifacts" / "hmm-residual" / "v9-us-hmm"
OUT = ROOT / "artifacts" / "hmm-residual" / "01-status"
TOL = 0.05
DELAY, COST = 1, 10.0
PAPER_BASIS = "risky_leg_wealth_flat_in_cash"
LEGACY_BASIS = "total_wealth"
MARKETS = ("us", "de", "jp")
NAMES = {"us": "S&P 500", "de": "DAX", "jp": "Nikkei 225"}
SERIES = {"us": "v9 (S&P 500)", "de": "v8.5", "jp": "v8.5"}


def hmm_path(market: str) -> pd.DataFrame:
    if market == "us":
        return pd.read_csv(V9 / "hmm-delay-1" / "path.csv", parse_dates=["date"])
    feats = pd.read_csv(SEALED / market / "features.csv", parse_dates=["date"])
    sig = pd.read_csv(SEALED / market / "hmm-delay-1" / "selected-signal.csv",
                      parse_dates=["date"])
    merged = feats.merge(sig, on="date", how="left")
    return apply_signal(merged[["date", "equity_simple", "cash_return"]],
                        merged["selected_signal"], delay_trading_days=DELAY,
                        one_way_cost_bps=COST)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sealed = pd.read_csv(SEALED / "metrics-exploratory.csv",
                         parse_dates=["start", "end"])

    records = []
    for market in MARKETS:
        row = sealed[(sealed.market == market) & (sealed.model == "hmm")
                     & (sealed.delay == DELAY)].iloc[0]
        lo, hi = row["start"], row["end"]
        path = hmm_path(market)
        window = path[(path["date"] >= lo) & (path["date"] <= hi)].dropna(
            subset=["cash_return", "position", "one_way_turnover",
                    "strategy_return"])
        scored = {basis: performance_metrics(window, drawdown_basis=basis)
                  for basis in (PAPER_BASIS, LEGACY_BASIS)}
        shifts = int((window["position"].diff().abs() > 0).sum())
        years = (hi - lo).days / 365.25
        target = TABLE4[market]["hmm"]
        for basis, got in scored.items():
            for metric in METRICS:
                records.append({
                    "market": market, "series": SERIES[market], "basis": basis,
                    "metric": metric, "ours": got[metric], "shu": target[metric],
                    "deviation": abs(got[metric] - target[metric]),
                    "within_tol": abs(got[metric] - target[metric]) <= TOL,
                    "unresolvable": (abs(got[metric] - target[metric])
                                     <= PRINTED_HALF_UNIT[metric]),
                })
            implied = 2.0 * target["turnover"] * years
            records.append({
                "market": market, "series": SERIES[market], "basis": basis,
                "metric": "shifts", "ours": float(shifts), "shu": implied,
                "deviation": abs(shifts - implied), "within_tol": False,
                "unresolvable": False,
            })
    tidy = pd.DataFrame(records)
    tidy.to_csv(OUT / "hmm-vs-table4.csv", index=False, lineterminator="\n")

    lines = []
    for basis, title in ((PAPER_BASIS, "QUY ƯỚC CỦA PAPER (v9.1)"),
                         (LEGACY_BASIS, "quy ước cũ (v8.x/v9)")):
        sub = tidy[tidy.basis == basis]
        lines.append(f"\n{title} — HMM vs Table 4, delay 1, |lệch| tuyệt đối "
                     f"(ngưỡng {TOL:.2f}; * = dưới nửa chữ số cuối paper in ra)\n")
        head = f"{'metric':<12}"
        for market in MARKETS:
            head += f"{NAMES[market]:>22}"
        lines.append(head)
        for metric in (*METRICS, "shifts"):
            line = f"{LABELS.get(metric, metric):<12}"
            for market in MARKETS:
                r = sub[(sub.market == market) & (sub.metric == metric)].iloc[0]
                if metric == "shifts":
                    cell, flag = f"{r.ours:.0f}/{r.shu:.0f}", " "
                else:
                    cell = f"{r.ours:.4f}/{r.shu:.3f}"
                    flag = "*" if r.unresolvable else (" " if r.within_tol
                                                       else "!")
                line += f"{cell + ' ' + f'{r.deviation:.3f}' + flag:>22}"
            lines.append(line)
        lines.append("")
        for market in MARKETS:
            cells = sub[(sub.market == market) & (sub.metric != "shifts")]
            bad = cells[~cells.within_tol]
            lines.append(
                f"  {NAMES[market]:<11} {SERIES[market]:<13} trong ngưỡng "
                f"{int(cells.within_tol.sum())}/8"
                + (f"  — còn lệch: " + ", ".join(
                    f"{LABELS[m]} {d:.3f}" for m, d in zip(bad.metric,
                                                           bad.deviation))
                   if len(bad) else "  — khớp toàn bộ"))

    lines.append("\nChỉ số nào lệch ở nhiều thị trường nhất, theo quy ước paper:")
    sub = tidy[tidy.basis == PAPER_BASIS]
    for metric in METRICS:
        hits = [f"{m} {sub[(sub.market == m) & (sub.metric == metric)].iloc[0].deviation:.3f}"
                for m in MARKETS
                if not sub[(sub.market == m)
                           & (sub.metric == metric)].iloc[0].within_tol]
        if hits:
            lines.append(f"  {LABELS[metric]:<11} {len(hits)}/3  "
                         + "  ".join(hits))

    report = "\n".join(lines) + "\n"
    (OUT / "hmm-vs-table4.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"đã ghi {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
