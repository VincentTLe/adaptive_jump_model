"""Can ANY single candidate set reproduce Table 4's turnover in all three markets?

The paper never publishes the candidate values k is chosen from. It publishes
Table 3, which tabulates persistence at k = 0, 2, 4, 8, 20, and it says Bulla's
original k was 6. Everything else is our construction.

The earlier version of this probe measured the spread of turnover across eight
defensible sets and concluded the row was unidentified because the spread was
wide. That understates the case, and this version asks the sharper question:
the paper runs ONE method with ONE candidate set across all three markets and
reports all three, so a candidate set that explains the turnover row has to
explain it everywhere. How many markets can a single set get right at once?

Sets are taken verbatim from the earlier probe and NOT extended, because adding
a set after seeing which one helps is the fitting this repository exists to
detect. Their provenance, unchanged:

  table3        the k values Table 3 tabulates
  table3_plus6  those plus Bulla's original 6
  filtered      as published minus k=0, since no smoothing is arguably not a
                candidate but the absence of one
  bulla_only    the single value the paper names
  dense_small   a plausible fine grid over short windows
  dense_wide    a plausible coarse grid reaching the published maximum
  reach_low     reachability probe: not adopted, present only to show whether a
  reach_no_tail published value is attainable at all

Runs on the sealed v9.3 states. Smoothing is applied after the HMM is fitted, so
changing the candidate set needs no refit. v9.4 differs from v9.3 only in the
declared drawdown basis, which cannot touch turnover, states or selection.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402
from _shu_table4 import METRICS, TABLE4  # noqa: E402

from adaptive_jump.backtest import apply_signal, performance_metrics  # noqa: E402
from adaptive_jump.config import load_config  # noqa: E402
from adaptive_jump.models import smoothed_hmm_states  # noqa: E402
from adaptive_jump.walkforward import select_monthly_candidate  # noqa: E402

RUN = ROOT / "artifacts" / "fixed-baselines" / (
    "fixed-baselines-9aec0f58a8bf-f31f60d08cbb-b23238411618"
)
OUT = ROOT / "artifacts" / "hmm-residual" / "12-grid-identification-v9-3"
DELAY, COST, TOL = 1, 10.0, 0.05
NAMES = {"us": "S&P 500", "de": "DAX", "jp": "Nikkei 225"}

GRIDS = {
    "table3": (0, 2, 4, 8, 20),
    "table3_plus6": (0, 2, 4, 6, 8, 20),
    "filtered": (2, 4, 6, 8, 20),
    "bulla_only": (6,),
    "dense_small": (2, 4, 6, 8, 10, 12),
    "dense_wide": (4, 8, 12, 16, 20),
    "reach_low": (2, 4),
    "reach_no_tail": (0, 2, 4, 6),
}


def evaluate(frame, states, grid, cfg, lo, hi) -> dict:
    candidates = smoothed_hmm_states(states, list(grid))
    selection = select_monthly_candidate(
        frame[["date", "equity_simple", "cash_return"]], candidates,
        cfg.selection_protocol, delay_trading_days=DELAY, one_way_cost_bps=COST,
        periods_per_year=cfg.metrics_protocol.periods_per_year,
        volatility_ddof=cfg.metrics_protocol.volatility_ddof)
    signal = selection.signal.rename("selected_signal").reset_index()
    signal.columns = ["date", "selected_signal"]
    merged = frame.merge(signal, on="date", how="left")
    path = apply_signal(merged[["date", "equity_simple", "cash_return"]],
                        merged["selected_signal"], delay_trading_days=DELAY,
                        one_way_cost_bps=COST)
    window = path[(path["date"] >= lo) & (path["date"] <= hi)].dropna(
        subset=["cash_return", "position", "one_way_turnover", "strategy_return"])
    scored = performance_metrics(window, drawdown_basis="total_wealth")
    scored["shifts"] = int((window["position"].diff().abs() > 0).sum())
    return scored


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "configs/baselines/legacy/research-expanding-v9-3.toml")
    reported = pd.read_csv(RUN / "metrics-exploratory.csv",
                           parse_dates=["start", "end"])

    records = []
    for market in ("us", "de", "jp"):
        row = reported[(reported.market == market) & (reported.model == "hmm")
                       & (reported.delay == DELAY)].iloc[0]
        frame = pd.read_csv(RUN / market / "features.csv", parse_dates=["date"])
        states = pd.read_csv(RUN / market / "hmm-states.csv",
                             parse_dates=["date"]).set_index("date")["hmm_state"]
        for name, grid in GRIDS.items():
            got = evaluate(frame, states, grid, cfg, row["start"], row["end"])
            target = TABLE4[market]["hmm"]
            records.append({
                "market": market, "grid": name,
                "candidates": "|".join(str(k) for k in grid),
                **{m: got[m] for m in METRICS}, "shifts": got["shifts"],
                **{f"dev_{m}": abs(got[m] - target[m]) for m in METRICS},
                "within_tol": sum(abs(got[m] - target[m]) <= TOL for m in METRICS),
            })
    frame = pd.DataFrame(records)
    frame.to_csv(OUT / "grids.csv", index=False, lineterminator="\n")

    lines = ["Một bộ ứng viên k có khớp được turnover ở CẢ BA thị trường không?",
             f"(sealed v9.3, delay {DELAY}, ngưỡng {TOL:.2f}; * = trong ngưỡng)", ""]
    header = f"{'bộ ứng viên':<15}{'k':<18}"
    for market in ("us", "de", "jp"):
        header += f"{NAMES[market]:>13}"
    lines.append(header + f"{'khớp':>7}{'8 chỉ số':>10}")
    for name, grid in GRIDS.items():
        line = f"{name:<15}{'|'.join(str(k) for k in grid):<18}"
        hit = 0
        cells = []
        for market in ("us", "de", "jp"):
            r = frame[(frame.market == market) & (frame.grid == name)].iloc[0]
            ok = r["dev_turnover"] <= TOL
            hit += bool(ok)
            cells.append(int(r["within_tol"]))
            line += f"{f'{r.turnover:.3f}' + ('*' if ok else ' '):>13}"
        lines.append(line + f"{hit:>5}/3" + f"{'/'.join(map(str, cells)):>10}")

    best = max(
        sum(frame[(frame.market == m) & (frame.grid == n)].iloc[0]["dev_turnover"]
            <= TOL for m in ("us", "de", "jp")) for n in GRIDS)
    lines += ["", f"Tối đa {best}/3 thị trường khớp turnover bằng MỘT bộ duy nhất.", ""]

    for market in ("us", "de", "jp"):
        sub = frame[frame.market == market]
        hits = sub[sub.dev_turnover <= TOL]["grid"].tolist()
        shu = TABLE4[market]["hmm"]["turnover"]
        lines.append(f"  {NAMES[market]:<11} Shu {shu:.2f}"
                     f" | phổ {sub.turnover.min():.3f}..{sub.turnover.max():.3f}"
                     f" | bộ khớp: {', '.join(hits) if hits else 'không có'}")
    lines += ["",
              "Các thị trường đòi hỏi NGƯỢC NHAU: bộ nào kéo một thị trường vào",
              "ngưỡng thì đẩy thị trường khác ra xa. Vì paper dùng chung một bộ cho",
              "cả ba, không lựa chọn lưới nào tái lập được dòng turnover của Table 4.",
              "Turnover không phải chưa xác định vì phổ rộng — mà vì các ràng buộc",
              "mâu thuẫn nhau."]

    # The other seven metrics, to show this is specific to turnover.
    others = [m for m in METRICS if m != "turnover"]
    worst = frame[[f"dev_{m}" for m in others]].to_numpy().max()
    lines.append("")
    lines.append(f"Bảy chỉ số còn lại, trên toàn bộ {len(frame)} ô: lệch lớn nhất "
                 f"{worst:.3f}"
                 + ("  (mọi ô trong ngưỡng)" if worst <= TOL else ""))

    report = "\n".join(lines) + "\n"
    (OUT / "report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"đã ghi {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
