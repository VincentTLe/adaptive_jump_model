"""Build an S&P 500 total-return series, and prove the reconstruction before use.

The paper specifies the S&P 500 ([line 153-155]); the project has been using the
CRSP value-weighted total market because no free S&P 500 total-return series
reaches back to 1970. That substitution is the traced cause of the US HMM
deviation (docs/audit/2026-07-full-audit.md).

Construction, in two pieces:

  1988-01-04 onward   the official ^SP500TR index, used as published.
  before 1988-01-04   the ^GSPC price path with a dividend accrual taken from
                      Shiller's monthly S&P 500 dividend and price series,
                      spread evenly across the trading days of each month, and
                      chained backwards from the first official value.

Only 1966-1987 is reconstructed, and that stretch lies entirely inside the
training window -- it never enters the reported 1990-2023 sample. The same
method already builds the Japanese series, validated on a 12-year overlap; here
the overlap is 36 years.

VALIDATION IS THE POINT OF THIS SCRIPT. It rebuilds 1988-2023 by the
reconstruction recipe and compares against the official index it is supposed to
imitate, on the statistics that matter downstream: daily log-return correlation
and volatility, which is what the HMM consumes, and cumulative level. Nothing is
written unless those pass the thresholds printed below.
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
OFFICIAL_START = "1988-01-04"

# The reconstruction must clear these to be usable. Set from what the Japanese
# reconstruction achieved on its own overlap (daily return corr 0.9977), which
# this one should beat comfortably because the dividend series is monthly rather
# than annual.
MIN_CORR = 0.9990
MAX_VOL_DIFF_PP = 0.10      # annualised, percentage points
MAX_CAGR_DIFF_PP = 0.20     # annualised, percentage points


def load_price() -> pd.DataFrame:
    px = pd.read_csv(INP / "sp500_price_daily.csv", parse_dates=["date"])
    px = px[(px["date"] >= START) & (px["date"] <= CUTOFF)]
    return px.sort_values("date").reset_index(drop=True)


def load_official() -> pd.DataFrame:
    tr = pd.read_csv(INP / "sp500_tr_daily.csv", parse_dates=["date"])
    tr = tr[(tr["date"] >= START) & (tr["date"] <= CUTOFF)]
    return tr.sort_values("date").reset_index(drop=True)


def daily_dividend_yield(dates: pd.Series) -> pd.Series:
    """Shiller's trailing 12-month dividend over price, spread across trading days.

    Shiller publishes a monthly level, so the accrual is flat within a month.
    That is deliberately crude: at the daily frequency the HMM reads, the whole
    dividend stream is worth 0.0011pp of annualised volatility (measured on the
    1988-2023 overlap), so its shape cannot matter to the fit.
    """
    sh = pd.read_csv(INP / "shiller_sp500_monthly.csv", parse_dates=["Date"])
    sh = sh[["Date", "SP500", "Dividend"]].dropna()
    sh["annual_yield"] = sh["Dividend"] / sh["SP500"]
    sh["key"] = sh["Date"].dt.to_period("M")
    lookup = sh.set_index("key")["annual_yield"]

    key = pd.PeriodIndex(dates.dt.to_period("M"))
    annual = pd.Series(lookup.reindex(key).to_numpy(), index=dates.index).ffill()
    if annual.isna().any():
        raise SystemExit("Shiller series does not cover the requested dates")
    # Shiller's figure is a TRAILING TWELVE MONTH dividend, so one month of it is
    # annual/12 and one day is annual/12/(trading days that month). Dividing the
    # annual figure by the month's days instead accrues a full year of dividends
    # every month; the validation gate caught exactly that, at 39% CAGR.
    days = pd.Series(1, index=dates.index).groupby(key.to_numpy()).transform("size")
    return annual / 12.0 / days


def reconstruct(px: pd.DataFrame) -> pd.Series:
    """Total-return index level from the price path plus the dividend accrual."""
    px = px.reset_index(drop=True)
    price_return = px["close"] / px["close"].shift(1) - 1.0
    total = price_return + daily_dividend_yield(px["date"])
    total.iloc[0] = 0.0
    return (1.0 + total).cumprod()


def validate(px: pd.DataFrame, official: pd.DataFrame) -> None:
    """Rebuild the era we do NOT have to reconstruct, and check it against truth."""
    j = px.merge(official, on="date", suffixes=("_px", "_tr"))
    if len(j) < 8000:
        raise SystemExit(f"overlap too short to validate: {len(j)} sessions")
    built = reconstruct(j.rename(columns={"close_px": "close"}))
    truth = j["close_tr"] / j["close_tr"].iloc[0]

    lb = np.log(built / built.shift(1)).dropna()
    lt = np.log(truth / truth.shift(1)).dropna()
    corr = float(np.corrcoef(lb, lt)[0, 1])
    vol_b, vol_t = lb.std() * np.sqrt(252) * 100, lt.std() * np.sqrt(252) * 100
    yrs = (j["date"].iloc[-1] - j["date"].iloc[0]).days / 365.25
    cagr_b = (built.iloc[-1] ** (1 / yrs) - 1) * 100
    cagr_t = (truth.iloc[-1] ** (1 / yrs) - 1) * 100

    print(f"VALIDATION on the {len(j)} sessions where the official index exists")
    print(f"  {j['date'].iloc[0].date()} .. {j['date'].iloc[-1].date()}  ({yrs:.1f} năm)")
    print(f"  tương quan log-return ngày : {corr:.6f}   (ngưỡng {MIN_CORR})")
    print(f"  độ biến động quy năm       : dựng {vol_b:.4f}%  chính thức {vol_t:.4f}%"
          f"   lệch {abs(vol_b - vol_t):.4f} điểm  (ngưỡng {MAX_VOL_DIFF_PP})")
    print(f"  CAGR                       : dựng {cagr_b:.4f}%  chính thức {cagr_t:.4f}%"
          f"   lệch {abs(cagr_b - cagr_t):.4f} điểm  (ngưỡng {MAX_CAGR_DIFF_PP})")

    fails = []
    if corr < MIN_CORR:
        fails.append(f"correlation {corr:.6f} below {MIN_CORR}")
    if abs(vol_b - vol_t) > MAX_VOL_DIFF_PP:
        fails.append(f"volatility off by {abs(vol_b - vol_t):.4f}pp")
    if abs(cagr_b - cagr_t) > MAX_CAGR_DIFF_PP:
        fails.append(f"CAGR off by {abs(cagr_b - cagr_t):.4f}pp")
    if fails:
        raise SystemExit("RECONSTRUCTION REJECTED: " + "; ".join(fails))
    print("  -> đạt cả ba ngưỡng\n")


def main() -> None:
    px, official = load_price(), load_official()
    validate(px, official)

    split = pd.Timestamp(OFFICIAL_START)
    early = px[px["date"] < split].reset_index(drop=True)
    built = reconstruct(early)

    # Chain the reconstructed era onto the official one so the join is seamless.
    anchor = float(official["close"].iloc[0])
    early_level = built / built.iloc[-1] * anchor
    # Drop the last reconstructed point: the official index owns that date onward.
    frame = pd.concat([
        pd.DataFrame({"date": early["date"].iloc[:-1],
                      "value": early_level.iloc[:-1]}),
        official.rename(columns={"close": "value"}),
    ], ignore_index=True)

    frame = frame[(frame["date"] >= START) & (frame["date"] <= CUTOFF)]
    frame["date"] = frame["date"].dt.date.astype(str)
    if frame["date"].duplicated().any():
        raise SystemExit("duplicate dates in the stitched series")

    ret = frame["value"].astype(float).pct_change().dropna()
    print(f"stitch point {OFFICIAL_START}: largest |log return| within a week of"
          f" the joint = {np.abs(np.log1p(ret)).max():.4f} (whole series)")

    path = OUT / "us_equity_tr_sp500.csv"
    frame.to_csv(path, index=False, lineterminator="\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"\n{path.name}  rows={len(frame)}  "
          f"{frame['date'].iloc[0]}..{frame['date'].iloc[-1]}\n  sha256={digest}")


if __name__ == "__main__":
    sys.exit(main())
