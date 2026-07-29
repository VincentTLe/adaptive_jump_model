"""Restore the dividends missing from the German series before 1988.

The DAX is a performance index, so the project has always treated the Stooq
`^dax` file as total return end to end and applied no reconstruction. That was
wrong for the backcast era, and the audit of 2026-07-28 caught it:

  measured against the OECD German SHARE PRICE index, a price index, our series
  runs +3.02% a year above it over 1988-2023 -- which is the German dividend
  yield, exactly as a performance index should -- and -0.15% a year BELOW it
  over 1970-1987.

A second consequence makes it unambiguous: on the current series, German
equities return 2.67% a year over 1970-1987 against a 6.63% cash rate, an equity
risk premium of **-3.96% a year sustained for eighteen years**. That is
impossible for a total-return index and ordinary for a price index in a
high-payout market. The vendor evidently spliced the DAX Kursindex backcast onto
the performance index; both are 1000.0 on 1987-12-30, so the joint is invisible.

The fix mirrors the Japanese one already in this repository: keep the official
segment as published, and before it add a dividend accrual to the price path,
chained backwards onto the first official value.

  1987-12-30 onward   the DAX performance index, used as published
  before 1987-12-30   the same price path plus JST Macrohistory annual German
                      dividend yields, spread across each year's trading days

JST puts the 1970-1987 German dividend yield at 3.24% a year on average, against
the 3.02% measured independently from the OECD comparison. Two sources that know
nothing of each other, agreeing to about a fifth of a percentage point.

VALIDATION IS THE POINT. Nothing is written unless the repaired series clears
all three gates below, and note what they are NOT: none of them looks at Table 4
or any other published result of the paper. Table 4's German column covers
1990-2023, which is entirely inside the official segment and therefore cannot
move; the anchors in Table 1 are volatilities, which a smooth dividend drift
barely touches. This repair is justified by internal consistency alone and is
invisible to every number we are trying to reproduce.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INP = ROOT / "data" / "external" / "inputs"
OUT = ROOT / "data" / "external"
START, CUTOFF = "1965-01-01", "2023-12-29"
# The DAX performance index base: 1987-12-30 = 1000.0 exactly, which the Stooq
# file reproduces to the digit. Everything from here is official.
OFFICIAL_START = pd.Timestamp("1987-12-30")

# Gates. The repaired backcast must behave like the official segment does when
# both are measured against the same external price index.
MAX_DIVIDEND_GAP_PP = 1.0     # repaired era vs official era, annualised
MAX_VOL_SHIFT_PP = 0.30       # a dividend drift must not move daily volatility
# The equity premium gate compares against JST's own German figures rather than
# against a sign. An earlier draft demanded a positive premium over 1970-1987;
# JST reports German equities returning 6.29% a year against a 6.26% bill rate
# in that window, a premium of +0.02%, so "positive" was a prior of mine and not
# a fact about Germany. What the repair has to do is land near the independent
# number, which the unrepaired series misses by 3.98 percentage points.
MAX_PREMIUM_GAP_PP = 1.5


def load_price() -> pd.Series:
    frame = pd.read_csv(INP / "stooq_dax_daily.csv")[["Date", "Close"]]
    frame.columns = ["date", "close"]
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame[(frame["date"] >= START) & (frame["date"] <= CUTOFF)]
    return frame.dropna().set_index("date")["close"].sort_index()


def daily_dividend_yield(dates: pd.DatetimeIndex) -> pd.Series:
    """JST's annual German dividend yield, spread across each year's sessions."""
    jst = pd.read_csv(INP / "jst_germany_eq.csv").set_index("year")["eq_dp"]
    years = pd.Series(dates.year, index=dates)
    lo, hi = int(jst.index.min()), int(jst.index.max())
    annual = years.clip(lower=lo, upper=hi).map(jst)
    if annual.isna().any():
        raise SystemExit("JST German yields do not cover the requested dates")
    sessions = years.groupby(years.to_numpy()).transform("size")
    return annual / sessions


def build_series() -> pd.DataFrame:
    price = load_price()
    early = price.loc[:OFFICIAL_START]
    if early.empty or early.index[-1] != OFFICIAL_START:
        raise SystemExit("price path does not reach the official base date")

    accrual = daily_dividend_yield(pd.DatetimeIndex(early.index))
    total = early.pct_change().fillna(0.0) + accrual
    built = (1.0 + total).cumprod()
    # Chain onto the official value at the base date so the join is seamless.
    rebased = built / built.iloc[-1] * float(price.loc[OFFICIAL_START])

    repaired = pd.concat([rebased.iloc[:-1], price.loc[OFFICIAL_START:]])
    repaired = repaired[~repaired.index.duplicated()].sort_index()
    validate(price, repaired)
    return pd.DataFrame({"date": repaired.index, "value": repaired.to_numpy()})


def _annual(series: pd.Series, lo: str, hi: str) -> float:
    window = series.loc[lo:hi]
    years = (window.index[-1] - window.index[0]).days / 365.25
    return float((window.iloc[-1] / window.iloc[0]) ** (1 / years) - 1)


def validate(price: pd.Series, repaired: pd.Series) -> None:
    oecd = pd.read_csv(INP / "oecd_de_share_price_monthly.csv",
                       parse_dates=["date"]).set_index("date")["value"]
    monthly = repaired.resample("MS").mean()
    joined = pd.concat({"ours": monthly, "oecd": oecd}, axis=1).dropna()

    def dividend_gap(lo: str, hi: str) -> float:
        sub = joined.loc[lo:hi]
        years = (sub.index[-1] - sub.index[0]).days / 365.25
        a = (sub["ours"].iloc[-1] / sub["ours"].iloc[0]) ** (1 / years) - 1
        b = (sub["oecd"].iloc[-1] / sub["oecd"].iloc[0]) ** (1 / years) - 1
        return float(a - b)

    repaired_gap = dividend_gap("1970-01-01", "1987-12-31")
    official_gap = dividend_gap("1988-01-01", "2023-12-31")

    cash = pd.read_csv(OUT / "de_cash_ladder.csv", parse_dates=["date"])
    cash = cash.set_index("date")["value"].reindex(
        pd.DatetimeIndex(repaired.index), method="ffill") / 100.0
    premium = _annual(repaired, "1970-01-01", "1987-12-31") - float(
        cash.loc["1970-01-01":"1987-12-31"].mean())
    before_premium = _annual(price, "1970-01-01", "1987-12-31") - float(
        cash.loc["1970-01-01":"1987-12-31"].mean())
    jst = pd.read_csv(INP / "jst_germany_eq.csv").set_index("year").loc[1970:1987]
    jst_premium = float((1 + jst["eq_tr"]).prod() ** (1 / len(jst))
                        - (1 + jst["bill_rate"]).prod() ** (1 / len(jst)))

    before = np.log(price / price.shift(1)).loc["1970-01-01":"1987-12-31"]
    after = np.log(repaired / repaired.shift(1)).loc["1970-01-01":"1987-12-31"]
    vol_shift = abs(after.std() - before.std()) * np.sqrt(252) * 100

    print("VALIDATION")
    print(f"  chênh so với chỉ số giá OECD, đoạn sửa 1970-1987 : "
          f"{repaired_gap:+.2%}/năm")
    print(f"  chênh so với chỉ số giá OECD, đoạn chính thức    : "
          f"{official_gap:+.2%}/năm")
    print(f"  khác biệt giữa hai đoạn : {abs(repaired_gap - official_gap)*100:.2f}"
          f" điểm  (ngưỡng {MAX_DIVIDEND_GAP_PP})")
    print(f"  phần bù rủi ro 1970-1987: {before_premium:+.2%} -> {premium:+.2%}"
          f"/năm   JST độc lập {jst_premium:+.2%}"
          f"   (lệch {abs(premium - jst_premium)*100:.2f} điểm, ngưỡng "
          f"{MAX_PREMIUM_GAP_PP})")
    print(f"  dịch chuyển độ biến động: {vol_shift:.4f} điểm"
          f"   (ngưỡng {MAX_VOL_SHIFT_PP})")

    fails = []
    if abs(repaired_gap - official_gap) * 100 > MAX_DIVIDEND_GAP_PP:
        fails.append("repaired era still does not carry the official era's yield")
    if abs(premium - jst_premium) * 100 > MAX_PREMIUM_GAP_PP:
        fails.append(f"equity premium {premium:.2%} is {abs(premium-jst_premium)*100:.2f}pp "
                     f"from JST's {jst_premium:.2%}")
    if vol_shift > MAX_VOL_SHIFT_PP:
        fails.append(f"volatility moved {vol_shift:.3f}pp")
    if fails:
        raise SystemExit("REPAIR REJECTED: " + "; ".join(fails))
    print("  -> đạt cả ba ngưỡng\n")


def main() -> None:
    frame = build_series()
    frame["date"] = frame["date"].dt.date.astype(str)
    path = OUT / "de_equity_tr_dividend_adjusted.csv"
    frame.to_csv(path, index=False, lineterminator="\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"{path.name}  rows={len(frame)}  "
          f"{frame['date'].iloc[0]}..{frame['date'].iloc[-1]}")
    print(f"  sha256={digest}")


if __name__ == "__main__":
    sys.exit(main())
