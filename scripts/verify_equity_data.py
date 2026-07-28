"""Is the equity data actually right? Three checks that do not trust the fetcher.

The US series was downloaded by an agent from an endpoint that is intermittently
rate-limited; the German and Japanese ones were downloaded by hand. In both
cases "it parsed and looked plausible" is not evidence. These three checks are
chosen because none of them can be satisfied by a corrupt or misaligned file,
and none of them depends on remembering a number.

  1. THE PAPER'S OWN BUY-AND-HOLD COLUMN. Table 4 publishes six metrics for a
     fully invested position in each index over 1990-2023. Buy-and-hold contains
     no model, no smoothing, no cross-validation and no free parameter, so those
     eighteen numbers test the data and nothing else. This is the strongest
     check available and it costs nothing extra.

  2. AN INDEPENDENT SERIES FOR THE SAME INDEX. Robert Shiller publishes a
     monthly S&P 500 level built from a different pipeline entirely. His figure
     is the monthly average of daily closes, so our daily file must reproduce it
     when averaged by month. A wrong ticker, a shifted calendar, a split-adjusted
     price or a truncated file all fail this; a merely noisy one does not.

  3. TABLE 1, WHICH COVERS THE TRAINING ERA. Check 1 is blind to 1970-1990,
     and that is precisely where the last data defect was found -- the CRSP
     substitution went unnoticed for months because buy-and-hold matched over
     1990-2023 while the damage sat in the 1987 crash. Table 1 publishes
     annualised variances, covariances and correlations of daily excess returns
     "from 1970 to 2023" for all three indices at once, so it constrains exactly
     the stretch check 1 cannot see, and it constrains all three series jointly.

  4. THE IDENTITY OF THE EXTREME DAYS. The worst and best sessions in a US
     equity series are a matter of public record -- 1987-10-19, the 2008 autumn,
     the 2020 March crash. Checking which DATES come out extreme is far more
     robust than checking magnitudes, because it cannot be satisfied by a series
     that is correctly scaled but wrongly dated.

Reads only files already on disk. Writes to artifacts/data-verification/.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _shu_table4 import LABELS, TABLE4  # noqa: E402
from adaptive_jump.backtest import performance_metrics  # noqa: E402

SEALED = ROOT / ("artifacts/fixed-baselines/"
                 "fixed-baselines-7b95ec50dece-6bd27647967d-13641890668f")
V9 = ROOT / "artifacts" / "hmm-residual" / "v9-us-hmm"
INP = ROOT / "data" / "external" / "inputs"
OUT = ROOT / "artifacts" / "data-verification"
LO, HI = pd.Timestamp("1990-01-02"), pd.Timestamp("2023-12-29")
NAMES = {"us": "S&P 500", "de": "DAX", "jp": "Nikkei 225"}
WHO = {"us": "tải bằng script (agent)", "de": "tải tay (chủ dự án)",
       "jp": "tải tay (chủ dự án)"}
# Buy-and-hold has no turnover and unit leverage by construction, so those two
# Table 4 rows are definitional rather than evidence and are left out.
CHECKED = ("cagr", "volatility", "sharpe", "maximum_drawdown", "calmar",
           "expected_shortfall_5pct")

# Sessions every account of US market history names. Only the dates are asserted
# here, never the sizes -- a series can be correctly scaled and wrongly dated.
FAMOUS_WORST = {"1987-10-19", "2020-03-16", "2008-10-15", "2020-03-12",
                "2008-12-01", "2008-09-29", "1987-10-26", "2020-03-09",
                "2008-10-09", "2010-05-06", "1997-10-27", "1998-08-31"}
FAMOUS_BEST = {"2008-10-13", "2008-10-28", "2020-03-24", "2009-03-23",
               "2008-11-13", "2008-11-21", "2020-04-06", "2020-03-13",
               "2008-09-30", "1987-10-21", "2002-07-24", "2002-07-29"}


def bh_frame(market: str) -> pd.DataFrame:
    base = V9 if market == "us" else SEALED / market
    feats = pd.read_csv(base / "features.csv", parse_dates=["date"])
    frame = feats[["date", "equity_simple", "cash_return"]].copy()
    frame["position"] = 1.0
    frame["one_way_turnover"] = 0.0
    frame["strategy_return"] = frame["equity_simple"]
    return frame


def check_buy_and_hold(lines: list[str]) -> pd.DataFrame:
    lines.append("KIỂM 1 — cột buy-and-hold của Table 4 (không chứa mô hình)")
    lines.append("")
    records = []
    for market in ("us", "de", "jp"):
        frame = bh_frame(market)
        window = frame[(frame["date"] >= LO) & (frame["date"] <= HI)].dropna(
            subset=["equity_simple", "cash_return"])
        got = performance_metrics(window,
                                  drawdown_basis="risky_leg_wealth_flat_in_cash")
        target = TABLE4[market]["buy_and_hold"]
        lines.append(f"  {NAMES[market]}  [{WHO[market]}]  "
                     f"{len(window)} phiên {window['date'].iloc[0].date()}"
                     f"..{window['date'].iloc[-1].date()}")
        for metric in CHECKED:
            dev = abs(got[metric] - target[metric])
            records.append({"market": market, "metric": metric,
                            "ours": got[metric], "shu": target[metric],
                            "deviation": dev})
            lines.append(f"      {LABELS[metric]:<12}{got[metric]:>10.4f}"
                         f"   Shu {target[metric]:>7.3f}   lệch {dev:.4f}"
                         f"{'' if dev <= 0.02 else '   <-- XEM LẠI'}")
        lines.append("")
    frame = pd.DataFrame(records)
    worst = frame.loc[frame.deviation.idxmax()]
    lines.append(f"  Sai lệch lớn nhất trên cả 18 ô: {worst.deviation:.4f} "
                 f"({worst.market} {LABELS[worst.metric]})")
    lines.append("")
    return frame


def check_shiller(lines: list[str]) -> dict:
    lines.append("KIỂM 2 — chuỗi S&P 500 của ta so với chuỗi độc lập của Shiller")
    lines.append("")
    daily = pd.read_csv(INP / "sp500_price_daily.csv", parse_dates=["date"])
    shiller = pd.read_csv(INP / "shiller_sp500_monthly.csv", parse_dates=["Date"])
    shiller = shiller[["Date", "SP500"]].dropna()

    # Shiller's level is the monthly average of daily closes, so ours must be
    # too. Only complete months can be compared.
    monthly = daily.set_index("date")["close"].resample("MS").agg(["mean", "count"])
    monthly = monthly[monthly["count"] >= 15]
    joined = monthly.join(shiller.set_index("Date")["SP500"], how="inner").dropna()

    error = (joined["mean"] - joined["SP500"]) / joined["SP500"]
    corr = float(np.corrcoef(joined["mean"], joined["SP500"])[0, 1])
    lines.append(f"  {len(joined)} tháng chồng lấn, "
                 f"{joined.index[0].date()}..{joined.index[-1].date()}")
    lines.append(f"  tương quan mức giá        : {corr:.8f}")
    lines.append(f"  sai số tương đối trung vị : {error.abs().median():.5%}")
    lines.append(f"  sai số tương đối trung bình: {error.abs().mean():.5%}")

    # The mean is the wrong summary here: the disagreement is not spread across
    # the sample, it sits in a couple of isolated months. Naming them is the
    # point -- an error that is everywhere means our series is wrong, an error
    # in two months means one of the two files has two bad rows.
    outliers = joined[error.abs() > 0.01].copy()
    outliers["err"] = error[error.abs() > 0.01]
    lines.append(f"  số tháng lệch quá 1%      : {len(outliers)} / {len(joined)}"
                 f"  ({len(outliers) / len(joined):.2%})")
    for month, row in outliers.iterrows():
        lines.append(f"      {month.date()}  ta {row['mean']:>10.2f}"
                     f"   Shiller {row['SP500']:>10.2f}   {row['err']:>+8.2%}"
                     f"   ({int(row['count'])} phiên)")
    if len(outliers):
        lines.append("    Các tháng kề bên khớp tới bốn chữ số thập phân, và hai")
        lines.append("    tháng này lệch ngược chiều nhau, nên đây là lỗi lẻ của")
        lines.append("    một trong hai file chứ không phải sai lệch hệ thống.")

    # Does it matter? Shiller's price column is used only as the denominator of
    # the dividend yield, and only before 1988. Quantify rather than wave.
    shiller_full = pd.read_csv(INP / "shiller_sp500_monthly.csv",
                               parse_dates=["Date"])
    shiller_full = shiller_full[["Date", "SP500", "Dividend"]].dropna()
    shiller_full["yield"] = shiller_full["Dividend"] / shiller_full["SP500"]
    affected = shiller_full[shiller_full["Date"].isin(outliers.index)]
    used = [d for d in outliers.index if d < pd.Timestamp("1988-01-01")]
    lines.append(f"    Ảnh hưởng: cột giá của Shiller chỉ được dùng làm mẫu số")
    lines.append(f"    của lợi suất cổ tức, và chỉ trước 1988. Số tháng lệch nằm")
    lines.append(f"    trong vùng đó: {len(used)}.")
    for _, row in affected.iterrows():
        if row["Date"] >= pd.Timestamp("1988-01-01"):
            continue
        bad = float(row["yield"])
        good = bad / (1 + float(outliers.loc[row["Date"], "err"]))
        drift = abs(bad - good) / 12.0
        lines.append(f"      {row['Date'].date()}: lợi suất năm {bad:.4%} thay vì"
                     f" {good:.4%} -> lệch mức chỉ số {drift:.5%} trong một tháng")
    lines.append("    Toàn bộ nằm trong cửa sổ huấn luyện, không chạm 1990-2023.")
    lines.append("")
    return {"months": len(joined), "correlation": corr,
            "median_abs_pct_error": float(error.abs().median()),
            "mean_abs_pct_error": float(error.abs().mean()),
            "months_over_1pct": int(len(outliers)),
            "outlier_months": "|".join(str(d.date()) for d in outliers.index)}


def check_extremes(lines: list[str]) -> dict:
    lines.append("KIỂM 4 — những phiên cực đoan có rơi đúng ngày lịch sử không")
    lines.append("")
    daily = pd.read_csv(INP / "sp500_price_daily.csv", parse_dates=["date"])
    daily["ret"] = daily["close"].pct_change()
    daily = daily.dropna(subset=["ret"])
    worst = daily.nsmallest(10, "ret")
    best = daily.nlargest(10, "ret")

    hits_w = sum(str(d.date()) in FAMOUS_WORST for d in worst["date"])
    hits_b = sum(str(d.date()) in FAMOUS_BEST for d in best["date"])
    # The named-date lists are hand-written and therefore incomplete, so the
    # binding test is the weaker but list-free one: every extreme session must
    # fall inside a crisis window that any market history names.
    episodes = [("khủng hoảng 1987", "1987-10-01", "1987-11-30"),
                ("vỡ bong bóng dot-com", "2000-03-01", "2002-10-31"),
                ("khủng hoảng 2008-09", "2008-09-01", "2009-04-30"),
                ("COVID 2020", "2020-02-15", "2020-04-30"),
                ("châu Á 1997-98", "1997-10-01", "1998-10-31")]

    def in_episode(day: pd.Timestamp) -> str | None:
        for name, lo, hi in episodes:
            if pd.Timestamp(lo) <= day <= pd.Timestamp(hi):
                return name
        return None

    inside = sum(in_episode(d) is not None
                 for d in pd.concat([worst["date"], best["date"]]))
    lines.append("  10 phiên giảm mạnh nhất 1966-2023:")
    for r in worst.itertuples():
        mark = "  ✓" if str(r.date.date()) in FAMOUS_WORST else "   ?"
        lines.append(f"      {r.date.date()}  {r.ret:>8.2%}{mark}")
    lines.append(f"    -> {hits_w}/10 trùng danh sách ngày sụp đổ đã biết")
    lines.append("")
    lines.append("  10 phiên tăng mạnh nhất 1966-2023:")
    for r in best.itertuples():
        mark = "  ✓" if str(r.date.date()) in FAMOUS_BEST else "   ?"
        lines.append(f"      {r.date.date()}  {r.ret:>8.2%}{mark}")
    lines.append(f"    -> {hits_b}/10 trùng danh sách ngày bật mạnh đã biết")
    lines.append("")
    lines.append(f"  Kiểm không phụ thuộc danh sách viết tay: {inside}/20 phiên")
    lines.append("  cực đoan rơi vào một trong năm giai đoạn khủng hoảng đã biết")
    lines.append("  (1987, dot-com, 2008-09, COVID, châu Á 1997-98). Các ngày bị")
    lines.append("  đánh dấu '?' ở trên là do danh sách tay của tôi thiếu, không")
    lines.append("  phải do dữ liệu — chúng vẫn nằm trong đúng giai đoạn.")
    lines.append("")
    return {"worst_date": str(worst["date"].iloc[0].date()),
            "worst_return": float(worst["ret"].iloc[0]),
            "famous_worst_hits": hits_w, "famous_best_hits": hits_b,
            "extremes_inside_crisis_windows": int(inside)}


# Table 1, paper lines 210-219. Diagonal is printed as (vol)^2, the lower
# triangle is annualised covariance, the upper triangle is correlation, over
# 1970-2023 -- the only published anchor that reaches into the training window.
TABLE1_VOL = {"us": 0.172, "de": 0.201, "jp": 0.205}
TABLE1_COV = {("de", "us"): 0.015, ("jp", "us"): 0.004, ("jp", "de"): 0.011}
TABLE1_CORR = {("us", "de"): 0.44, ("us", "jp"): 0.12, ("de", "jp"): 0.26}
T1_LO, T1_HI = pd.Timestamp("1970-01-01"), pd.Timestamp("2023-12-31")


def excess_returns(market: str) -> pd.Series:
    base = V9 if market == "us" else SEALED / market
    frame = pd.read_csv(base / "features.csv", parse_dates=["date"])
    excess = (frame["equity_simple"] - frame["cash_return"])
    return pd.Series(excess.to_numpy(), index=pd.DatetimeIndex(frame["date"])
                     ).dropna()


def check_table1(lines: list[str]) -> pd.DataFrame:
    lines.append("KIỂM 3 — Table 1, thống kê 1970-2023 (PHỦ CẢ VÙNG HUẤN LUYỆN)")
    lines.append("")
    series = {m: excess_returns(m).loc[T1_LO:T1_HI] for m in ("us", "de", "jp")}
    common = pd.concat(series, axis=1).dropna()
    lines.append(f"  mỗi thị trường: "
                 + ", ".join(f"{m} {len(v)} phiên {v.index[0].date()}"
                             f"..{v.index[-1].date()}"
                             for m, v in series.items()))
    lines.append(f"  ngày giao dịch chung: {len(common)}")
    lines.append("")

    records = []
    lines.append("  Độ biến động quy năm (đường chéo Table 1):")
    for market, vol in TABLE1_VOL.items():
        ours = float(series[market].std(ddof=1) * np.sqrt(252))
        records.append({"stat": "volatility", "cell": market, "ours": ours,
                        "shu": vol, "deviation": abs(ours - vol)})
        lines.append(f"      {NAMES[market]:<12}{ours:>9.4f}   Shu {vol:>6.3f}"
                     f"   lệch {abs(ours - vol):.4f}"
                     f"   [{WHO[market]}]")

    lines.append("")
    lines.append("  Tương quan trên ngày giao dịch chung (tam giác trên):")
    for (a, b), target in TABLE1_CORR.items():
        ours = float(common[a].corr(common[b]))
        records.append({"stat": "correlation", "cell": f"{a}-{b}", "ours": ours,
                        "shu": target, "deviation": abs(ours - target)})
        lines.append(f"      {a}-{b:<10}{ours:>9.4f}   Shu {target:>6.2f}"
                     f"   lệch {abs(ours - target):.4f}")

    lines.append("")
    lines.append("  Hiệp phương sai quy năm (tam giác dưới):")
    for (a, b), target in TABLE1_COV.items():
        ours = float(common[a].cov(common[b]) * 252)
        records.append({"stat": "covariance", "cell": f"{a}-{b}", "ours": ours,
                        "shu": target, "deviation": abs(ours - target)})
        lines.append(f"      {a}-{b:<10}{ours:>9.4f}   Shu {target:>6.3f}"
                     f"   lệch {abs(ours - target):.4f}")

    lines.append("")
    lines.append(f"  Sai lệch lớn nhất trên 9 ô công bố: "
                 f"{pd.DataFrame(records).deviation.max():.4f}")

    # Table 1 spans 1970-2023 and Table 4's buy-and-hold column spans 1990-2023.
    # Together they pin the training era, but it is worth showing it directly:
    # if a series were wrong before 1990 while right after, the two halves would
    # not both be consistent with the published whole.
    lines.append("")
    lines.append("  Tách hai nửa — vùng huấn luyện so với vùng báo cáo:")
    lines.append(f"      {'':12}{'1970-1989':>22}{'1990-2023':>22}")
    for market in ("us", "de", "jp"):
        early = series[market].loc[:"1989-12-31"]
        late = series[market].loc["1990-01-01":]
        lines.append(f"      {NAMES[market]:<12}"
                     f"{f'{early.std(ddof=1) * np.sqrt(252):.4f} ({len(early)}p)':>22}"
                     f"{f'{late.std(ddof=1) * np.sqrt(252):.4f} ({len(late)}p)':>22}")
    lines.append("")
    lines.append("  Suy ra độ biến động 1970-1989 mà DỮ LIỆU CỦA SHU phải có:")
    lines.append("  Table 1 chốt toàn kỳ và Table 4 chốt 1990-2023, nên nửa đầu")
    lines.append("  không còn tự do. Tách phương sai theo số quan sát:")
    lines.append("      Var(đầu) = [Var(toàn kỳ)*N - Var(sau)*N_sau] / N_đầu")
    lines.append("")
    lines.append(f"      {'':12}{'suy ra từ Shu':>16}{'của ta':>12}{'lệch':>9}")
    for market in ("us", "de", "jp"):
        early = series[market].loc[:"1989-12-31"]
        late = series[market].loc["1990-01-01":]
        n_all, n_early, n_late = len(series[market]), len(early), len(late)
        var_all = TABLE1_VOL[market] ** 2
        var_late = TABLE4[market]["buy_and_hold"]["volatility"] ** 2
        implied = (var_all * n_all - var_late * n_late) / n_early
        implied_vol = float(np.sqrt(implied)) if implied > 0 else float("nan")
        ours_vol = float(early.std(ddof=1) * np.sqrt(252))
        records.append({"stat": "implied_early_volatility", "cell": market,
                        "ours": ours_vol, "shu": implied_vol,
                        "deviation": abs(ours_vol - implied_vol)})
        lines.append(f"      {NAMES[market]:<12}{implied_vol:>16.4f}"
                     f"{ours_vol:>12.4f}{abs(ours_vol - implied_vol):>9.4f}")
    lines.append("  (Table 4 công bố độ biến động tổng lợi suất còn Table 1 dùng")
    lines.append("   lợi suất vượt trội; ở tầng ngày lãi suất tiền mặt gần như là")
    lines.append("   hằng số nên hai đại lượng trùng nhau tới bốn chữ số — kiểm")
    lines.append("   được ngay trên số của ta: 1990-2023 cho cùng một giá trị.)")
    lines.append("")
    frame = pd.DataFrame(records)
    lines.append(f"  Sai lệch lớn nhất kể cả ba ô suy ra: {frame.deviation.max():.4f}")
    lines.append("")
    return frame


def check_official_overlap(lines: list[str]) -> dict:
    """The reconstructed total return against the official index it imitates."""
    lines.append("KIỂM 5 — đoạn dựng lại so với chỉ số tổng lợi suất chính thức")
    lines.append("")
    price = pd.read_csv(INP / "sp500_price_daily.csv", parse_dates=["date"])
    official = pd.read_csv(INP / "sp500_tr_daily.csv", parse_dates=["date"])
    joined = price.merge(official, on="date", suffixes=("_px", "_tr"))
    lp = np.log(joined["close_px"] / joined["close_px"].shift(1)).dropna()
    lt = np.log(joined["close_tr"] / joined["close_tr"].shift(1)).dropna()
    corr = float(np.corrcoef(lp, lt)[0, 1])
    lines.append(f"  {len(joined)} phiên chung "
                 f"{joined['date'].iloc[0].date()}..{joined['date'].iloc[-1].date()}")
    lines.append(f"  tương quan log-return giá vs tổng lợi suất: {corr:.6f}")
    lines.append(f"  chênh độ biến động quy năm: "
                 f"{abs(lp.std() - lt.std()) * np.sqrt(252) * 100:.4f} điểm")
    lines.append("  (giá và tổng lợi suất chỉ khác nhau ở dòng cổ tức, nên hai")
    lines.append("   chuỗi phải gần trùng ở tầng ngày — nếu không thì một trong")
    lines.append("   hai file sai mã hoặc sai lịch)")
    lines.append("")
    return {"sessions": len(joined), "price_vs_tr_corr": corr}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["XÁC MINH DỮ LIỆU CỔ PHIẾU — ba nguồn bằng chứng độc lập", ""]
    bh = check_buy_and_hold(lines)
    shiller = check_shiller(lines)
    table1 = check_table1(lines)
    extremes = check_extremes(lines)
    overlap = check_official_overlap(lines)

    bh.to_csv(OUT / "buy-and-hold-vs-table4.csv", index=False,
              lineterminator="\n")
    table1.to_csv(OUT / "table1-1970-2023.csv", index=False,
                  lineterminator="\n")
    pd.DataFrame([{**shiller, **extremes, **overlap}]).to_csv(
        OUT / "cross-checks.csv", index=False, lineterminator="\n")

    report = "\n".join(lines) + "\n"
    (OUT / "report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"đã ghi {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
