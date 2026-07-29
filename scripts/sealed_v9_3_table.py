"""The sealed v9.3 HMM result against Table 4, every delay, both drawdown bases.

This reads the first full sealed run of the v9.3 contract -- three markets,
delays 1/5/10, its own acquisition manifest, config lock and inventory -- and
scores it the way the audit ledger now says it must be scored.

Two bases are printed because the contract and the ledger disagree, and the
disagreement is documented rather than quietly resolved:

  total_wealth                  the a-priori default, restored after the
                                flat-in-cash basis was withdrawn (it had been
                                chosen by minimising error against Table 4, which
                                is fitting an unspecified knob to the target).
  risky_leg_wealth_flat_in_cash what research-expanding-v9-3.toml still declares,
                                inherited from v9.2, and therefore what the
                                sealed run's own metrics table reports.

Only maximum drawdown and Calmar can differ between them; the other six metrics
never touch the drawdown path.

The upper-boundary fractions are printed as a description of the fit, not as a
gate. The 5% limit is this project's own invention -- committed in 71a46d6 with a
threshold nobody justified -- and the paper never mentions any such check, so
failing it is not a replication failure. It is reported because the selection
sitting on the longest available smoothing window in a fifth to two fifths of
months is a real property of the paper's procedure on this data.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402
from _shu_table4 import LABELS, METRICS, PRINTED_HALF_UNIT, TABLE4  # noqa: E402

from adaptive_jump.backtest import (  # noqa: E402
    apply_signal,
    performance_metrics,
)

RUN = ROOT / "artifacts" / "fixed-baselines" / (
    "fixed-baselines-9aec0f58a8bf-f31f60d08cbb-b23238411618"
)
OUT = ROOT / "artifacts" / "hmm-residual" / "11-sealed-v9-3"
TOL, COST = 0.05, 10.0
DELAYS = (1, 5, 10)
A, D = "total_wealth", "risky_leg_wealth_flat_in_cash"
NAMES = {"us": "S&P 500", "de": "DAX", "jp": "Nikkei 225"}


def scored(market: str, delay: int, window: tuple[pd.Timestamp, pd.Timestamp]):
    """Rescore one arm from the sealed run's own stored selection.

    The run does not write trade paths: its boundary check failed, so the OOS
    accounting stayed sealed and only the monthly selection was recorded. The
    path is therefore rebuilt from the features and the selected signal through
    the same apply_signal the pipeline uses, which is what makes the six
    non-drawdown metrics reproduce the run's own table exactly.
    """
    features = pd.read_csv(RUN / market / "features.csv", parse_dates=["date"])
    signal = pd.read_csv(RUN / market / f"hmm-delay-{delay}" / "selected-signal.csv",
                         parse_dates=["date"])
    merged = features.merge(signal, on="date", how="left")
    column = "selected_signal" if "selected_signal" in merged else merged.columns[-1]
    path = apply_signal(merged[["date", "equity_simple", "cash_return"]],
                        merged[column], delay_trading_days=delay,
                        one_way_cost_bps=COST)
    lo, hi = window
    frame = path[(path["date"] >= lo) & (path["date"] <= hi)].dropna(
        subset=["cash_return", "position", "one_way_turnover", "strategy_return"])
    return {basis: performance_metrics(frame, drawdown_basis=basis)
            for basis in (A, D)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reported = pd.read_csv(RUN / "metrics-exploratory.csv",
                           parse_dates=["start", "end"])
    boundaries = pd.read_csv(RUN / "boundaries.csv")

    lines = [
        "SEALED RUN v9.3 — HMM vs Table 4, delay 1/5/10, hai quy ước drawdown",
        f"run: {RUN.name}",
        "",
        f"(ngưỡng {TOL:.2f}; * = dưới nửa chữ số cuối paper in ra; ! = ngoài ngưỡng)",
        "A = total_wealth (mặc định a-priori)   "
        "D = flat_in_cash (hợp đồng khai, đã rút)",
        "",
    ]
    records = []
    for market in ("us", "de", "jp"):
        lines.append(f"=== {NAMES[market]}")
        header = f"{'chỉ số':<12}"
        for delay in DELAYS:
            header += f"{f'd{delay} A':>14}{f'd{delay} D':>14}"
        lines.append(header + f"{'Shu':>8}")

        cells = {}
        for delay in DELAYS:
            row = reported[(reported.market == market) & (reported.model == "hmm")
                           & (reported.delay == delay)]
            if row.empty:
                continue
            row = row.iloc[0]
            cells[delay] = scored(market, delay, (row["start"], row["end"]))

        for metric in METRICS:
            line = f"{LABELS[metric]:<12}"
            target = TABLE4[market]["hmm"][metric]
            for delay in DELAYS:
                if delay not in cells:
                    line += f"{'—':>14}{'—':>14}"
                    continue
                for basis in (A, D):
                    got = cells[delay][basis][metric]
                    dev = abs(got - target)
                    flag = ("*" if dev <= PRINTED_HALF_UNIT[metric]
                            else (" " if dev <= TOL else "!"))
                    line += f"{f'{got:.4f}{flag}':>14}"
                    records.append({
                        "market": market, "delay": delay, "basis": basis,
                        "metric": metric, "ours": got, "shu": target,
                        "deviation": dev, "within_tol": dev <= TOL,
                    })
            lines.append(line + f"{target:>8.3f}")

        for delay in DELAYS:
            if delay not in cells:
                continue
            counts = {basis: sum(
                abs(cells[delay][basis][m] - TABLE4[market]["hmm"][m]) <= TOL
                for m in METRICS) for basis in (A, D)}
            bad = [LABELS[m] for m in METRICS
                   if abs(cells[delay][A][m] - TABLE4[market]["hmm"][m]) > TOL]
            gate = boundaries[(boundaries.market == market)
                              & (boundaries.model == "hmm")
                              & (boundaries.delay == delay)]
            share = (f"{int(gate.iloc[0]['selected_months'])}/"
                     f"{int(gate.iloc[0]['total_months'])}"
                     f" = {gate.iloc[0]['fraction']:.1%}") if not gate.empty else "—"
            lines.append(
                f"   delay {delay:<2}  A {counts[A]}/8   D {counts[D]}/8"
                f"   | ngoài ngưỡng: {', '.join(bad) if bad else 'không'}"
                f"   | chọn đỉnh lưới {share}")
        lines.append("")

    frame = pd.DataFrame(records)
    frame.to_csv(OUT / "sealed-v9-3-vs-table4.csv", index=False, lineterminator="\n")

    turnover = frame[(frame.metric == "turnover") & (frame.basis == A)]
    lines.append(f"Turnover ngoài ngưỡng ở {int((~turnover.within_tol).sum())}/"
                 f"{len(turnover)} ô (3 thị trường × 3 delay).")
    others = frame[(frame.metric != "turnover") & (frame.basis == A)]
    lines.append(f"Bảy chỉ số còn lại: {int((~others.within_tol).sum())}/{len(others)}"
                 " ô ngoài ngưỡng.")

    report = "\n".join(lines) + "\n"
    (OUT / "report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"đã ghi {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
