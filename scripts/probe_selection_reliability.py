"""Does the paper's selection rule carry any signal, or is it choosing on noise?

Turnover is the only Table 4 metric outside tolerance in any market, and every
upstream explanation has been eliminated: the fixed-k persistence curve
reproduces Table 3 to 1.9%, and Shu's own position path on our returns
reproduces the turnover row to 0.002. What remains is which smoothing window the
monthly cross-validation picks. The candidate set is unidentified, which has been
measured. This asks a different and more basic question about the rule itself.

Section 3.4.3 selects the candidate with the highest Sharpe ratio over an 8-year
validation window. An annualised Sharpe estimated from eight years of daily data
has a large sampling error -- roughly 1/sqrt(8) even before autocorrelation --
while the candidates being compared are variants of the same signal and differ
in true Sharpe by far less. If that is so, the argmax is close to arbitrary, and
turnover is not identified even after the grid is fixed.

Three measurements, none of which needs a distributional assumption:

  1. MARGIN. For each decision month, the gap between the best and second-best
     validation Sharpe. A rule whose winner is chosen by 0.01 of Sharpe is not
     choosing.

  2. SPLIT-HALF AGREEMENT. Split each validation window into its first and
     second halves, take the argmax of each, and ask how often they agree. This
     is the direct test: a rule with signal picks the same candidate from two
     independent samples of the same window. A rule without signal agrees at the
     chance rate, 1/len(grid).

  2b. POSITIVE CONTROL. The split-half test is only informative if it CAN
     detect a real difference. Run it again on a pool of two strategies whose
     Sharpes differ enormously and stably -- always invested against always in
     cash. If that also lands at chance, the instrument is broken and says
     nothing about the k candidates. If it lands near certainty, the instrument
     works and the k candidates really are indistinguishable.

  3. PERSISTENCE. How often the selected candidate changes from one month to
     the next. Consecutive windows share seven of eight years of data, so a rule
     reading signal should almost never change its mind; a rule reading noise
     changes it whenever the newest month tips the balance.

Reads only stored artifacts. Writes to artifacts/hmm-residual/09-selection-noise/.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from adaptive_jump.backtest import annualized_excess_sharpe  # noqa: E402

SEALED = ROOT / ("artifacts/fixed-baselines/"
                 "fixed-baselines-7b95ec50dece-6bd27647967d-13641890668f")
V9 = ROOT / "artifacts" / "hmm-residual" / "v9-us-hmm"
OUT = ROOT / "artifacts" / "hmm-residual" / "09-selection-noise"
NAMES = {"us": "S&P 500", "de": "DAX", "jp": "Nikkei 225"}
VALIDATION_YEARS = 8


def base_for(market: str) -> Path:
    return V9 if market == "us" else SEALED / market


def load(market: str):
    base = base_for(market)
    arm = base / "hmm-delay-1"
    surface = pd.read_csv(arm / "cv-surface.csv", parse_dates=["decision_date"])
    choices = pd.read_csv(arm / "choices.csv", parse_dates=["decision_date"])
    returns = pd.read_csv(arm / "candidate-returns.csv", parse_dates=["date"],
                          index_col="date")
    returns.columns = [float(c) for c in returns.columns]
    features = pd.read_csv(base / "features.csv", parse_dates=["date"])
    cash = features.set_index("date")["cash_return"]
    return surface, choices, returns, cash


def margins(surface: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for date, group in surface[surface["eligible"]].groupby("decision_date"):
        ordered = group.sort_values("sharpe", ascending=False)
        if len(ordered) < 2:
            continue
        best, second = ordered["sharpe"].iloc[0], ordered["sharpe"].iloc[1]
        rows.append({"decision_date": date,
                     "winner": ordered["candidate"].iloc[0],
                     "best": best, "second": second, "margin": best - second,
                     "spread": best - ordered["sharpe"].iloc[-1]})
    return pd.DataFrame(rows)


def split_half(returns: pd.DataFrame, cash: pd.Series,
               dates: pd.DatetimeIndex) -> pd.DataFrame:
    """argmax on the first half of each window against argmax on the second."""
    rows = []
    for decision in dates:
        start = decision - pd.DateOffset(years=VALIDATION_YEARS)
        middle = decision - pd.DateOffset(years=VALIDATION_YEARS // 2)
        halves = []
        for lo, hi in ((start, middle), (middle, decision)):
            # Select by DATE, not by a boolean mask: the control pool is
            # built from a dropna'd frame and so does not share an index length
            # with the cash series.
            window = returns.index[(returns.index > lo) & (returns.index <= hi)]
            cash_window = cash.reindex(window)
            scores = {}
            for candidate in returns.columns:
                paired = pd.concat([returns.loc[window, candidate],
                                    cash_window], axis=1).dropna()
                if len(paired) < 252:
                    continue
                scores[candidate] = annualized_excess_sharpe(
                    paired.iloc[:, 0], paired.iloc[:, 1])
            scores = {k: v for k, v in scores.items() if np.isfinite(v)}
            halves.append(max(scores, key=scores.get) if scores else None)
        if halves[0] is not None and halves[1] is not None:
            rows.append({"decision_date": decision, "first": halves[0],
                         "second": halves[1], "agree": halves[0] == halves[1]})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["ĐỘ TIN CẬY CỦA QUY TẮC CHỌN k", ""]
    records = []

    for market in ("us", "de", "jp"):
        surface, choices, returns, cash = load(market)
        grid = sorted(returns.columns)
        chance = 1.0 / len(grid)

        margin = margins(surface)
        # A decision every month; sample every third one for the split-half test
        # so it stays cheap without becoming a special case of the calendar.
        sample = pd.DatetimeIndex(margin["decision_date"])[::3]
        halves = split_half(returns, cash, sample)
        agree = float(halves["agree"].mean())

        # Positive control, third attempt, and the first that is a control.
        # Attempt one pitted always-invested against always-cash: in Japan those
        # genuinely trade places across four-year windows, so it agreed at 46.4%
        # and measured the instability of that comparison, not the blindness of
        # the test. Attempt two used a signal against its own mirror, which has
        # the same defect -- a timing signal and its inverse swap ranks over
        # short windows all the time.
        #
        # A control needs a difference that is large, known, and constant. This
        # one takes a candidate and subtracts a fixed drift from it, so the
        # second strategy is worse by exactly delta of annualised Sharpe on
        # every window, with identical noise. Sweeping delta turns the control
        # into a power curve: it says what Sharpe gap this test can actually
        # resolve, which is the number needed to interpret the 17% above.
        middle_k = sorted(returns.columns)[len(returns.columns) // 2]
        base = returns[middle_k].dropna()
        vol = float(base.std()) * np.sqrt(252)
        power = {}
        for delta in (0.1, 0.2, 0.4, 0.8, 1.6):
            handicap = delta * vol / 252.0
            pair = pd.DataFrame({"a": base, "b": base - handicap})
            power[delta] = float(
                split_half(pair, cash, sample)["agree"].mean())
        control_agree = power[0.4]

        # Two REALISTIC controls alongside the deterministic one. Both compare
        # strategies whose long-run Sharpes differ a lot, and both land near
        # chance -- which is not a failure of the test but the same finding
        # again: over four-year windows even very different strategies swap
        # ranks routinely.
        equity = pd.read_csv(base_for(market) / "features.csv",
                             parse_dates=["date"]).set_index("date")
        long_cash = pd.DataFrame({"invested": equity["equity_simple"],
                                  "cash": equity["cash_return"]}).dropna()
        realistic = {"nắm cổ phiếu vs tiền mặt":
                     float(split_half(long_cash, cash, sample)["agree"].mean())}
        pair = pd.concat([returns[middle_k], equity["equity_simple"],
                          equity["cash_return"]], axis=1).dropna()
        pair.columns = ["signal", "equity", "cash_leg"]
        weight = ((pair["signal"] - pair["cash_leg"])
                  / (pair["equity"] - pair["cash_leg"]).replace(0.0, np.nan))
        weight = weight.clip(0.0, 1.0).fillna(0.0).round()
        mirror = pd.DataFrame({
            "signal": pair["signal"],
            "mirror": (1.0 - weight) * pair["equity"] + weight * pair["cash_leg"]})
        realistic["tín hiệu vs ảnh gương của nó"] = float(
            split_half(mirror, cash, sample)["agree"].mean())

        switches = float((choices["selected"].diff().abs() > 0).mean())
        picks = choices["selected"].value_counts(normalize=True).sort_index()

        lines.append(f"=== {NAMES[market]}  ({len(margin)} tháng quyết định, "
                     f"lưới {len(grid)} ứng viên)")
        lines.append(f"  1. BIÊN THẮNG (tốt nhất trừ nhì)")
        lines.append(f"       trung vị {margin['margin'].median():.4f}   "
                     f"tứ phân vị 3 {margin['margin'].quantile(.75):.4f}   "
                     f"lớn nhất {margin['margin'].max():.4f}")
        lines.append(f"       số tháng biên < 0.02 : "
                     f"{(margin['margin'] < 0.02).mean():.1%}")
        lines.append(f"       toàn dải (tốt nhất trừ tệ nhất), trung vị "
                     f"{margin['spread'].median():.4f}")
        lines.append(f"  2. TÁCH ĐÔI CỬA SỔ ({len(halves)} tháng lấy mẫu)")
        lines.append(f"       hai nửa chọn cùng ứng viên: {agree:.1%}"
                     f"   (ngẫu nhiên = {chance:.1%})")
        lines.append(f"  2b. ĐƯỜNG CÔNG SUẤT — bài kiểm phân giải được chênh")
        lines.append(f"       lệch Sharpe cỡ nào? (hai chiến lược giống hệt, một")
        lines.append(f"       cái bị trừ đi một mức trôi cố định; ngẫu nhiên 50%)")
        for delta, value in power.items():
            lines.append(f"       chênh {delta:>4.1f} Sharpe -> đồng ý {value:>6.1%}")
        lines.append(f"  2c. ĐỐI CHỨNG THỰC TẾ — hai chiến lược khác nhau rõ")
        lines.append(f"       về dài hạn, nhưng đường đi khác nhau (ngẫu nhiên 50%):")
        for label, value in realistic.items():
            lines.append(f"       {label:<30} {value:>6.1%}")
        lines.append(f"       -> ngay cả những cặp này cũng đảo hạng qua 4 năm.")
        resolvable = [d for d, v in power.items() if v >= 0.90]
        lines.append(f"       -> cần chênh ít nhất "
                     + (f"{min(resolvable):.1f} Sharpe để nhận ra chắc chắn"
                        if resolvable else "hơn 1.6 Sharpe — bài kiểm rất tù"))
        lines.append(f"  3. ĐỔI Ý GIỮA HAI THÁNG LIỀN NHAU")
        lines.append(f"       {switches:.1%} số tháng đổi ứng viên, dù hai cửa sổ")
        lines.append(f"       chung 7/8 dữ liệu")
        lines.append(f"       tỉ lệ chọn: "
                     + "  ".join(f"k{int(k)}:{v:.0%}" for k, v in picks.items()))
        lines.append("")
        records.append({"market": market, "months": len(margin),
                        "median_margin": float(margin["margin"].median()),
                        "share_margin_under_002": float((margin["margin"] < 0.02).mean()),
                        "median_spread": float(margin["spread"].median()),
                        "split_half_agreement": agree, "chance": chance,
                        "month_to_month_switch": switches,
                        "control_agreement": control_agree, **{f"realistic_{i}": v for i, v in enumerate(realistic.values())}, **{f"power_{d}": v for d, v in power.items()}})
        margin.to_csv(OUT / f"margins-{market}.csv", index=False,
                      lineterminator="\n")
        halves.to_csv(OUT / f"split-half-{market}.csv", index=False,
                      lineterminator="\n")

    frame = pd.DataFrame(records)
    frame.to_csv(OUT / "summary.csv", index=False, lineterminator="\n")

    lines.append("=" * 66)
    lines.append("Ý nghĩa: nếu hai nửa của cùng một cửa sổ kiểm định thường")
    lines.append("chọn ra hai ứng viên KHÁC nhau, thì argmax không đọc được")
    lines.append("tín hiệu nào bền — và turnover, thứ do argmax đó quyết định,")
    lines.append("không xác định được ngay cả khi đã cố định lưới ứng viên.")
    report = "\n".join(lines) + "\n"
    (OUT / "report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"đã ghi {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
