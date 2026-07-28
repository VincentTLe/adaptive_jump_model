"""Does the S&P 500 series reproduce Shu's fitted regime volatilities?

The audit traced the US HMM deviation to the equity series: on 1987-10-19 the
S&P 500 fell 20.47% and our CRSP total-market proxy fell 17.41%, and every
3000-day window containing that day fits a high-volatility regime about 8
percentage points below Shu's. Figure 2 of the paper publishes those fitted
volatilities, extracted losslessly from the PDF vectors.

This is the decisive test of that diagnosis, and it is cheap: refit the HMM on
the real S&P 500 price series for the same windows and see whether the fitted
high-state volatility moves onto Figure 2's curve.

Prediction, written before running: our high-state volatility for windows ending
1990-1998 should rise from about 34-36% toward Shu's 42-44%, and the windows
after 1999-08-31 -- which already agree to 0.12pp -- should not move.

The HMM consumes daily log returns, where dividends are immaterial: over the
9,070 sessions the price and total-return series share, their daily log-return
volatility differs by 0.0011pp with correlation 0.99960. So the price series is
the correct input here and no reconstruction is needed for this test.
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from multiprocessing import get_context  # noqa: E402

from adaptive_jump.config import load_config  # noqa: E402
from adaptive_jump.models import best_hmm_terminal_fit  # noqa: E402

RUN = ROOT / ("artifacts/fixed-baselines/"
              "fixed-baselines-7b95ec50dece-6bd27647967d-13641890668f")
CFG = load_config(ROOT / "research-expanding-v8-5.toml")
ANN = np.sqrt(252.0)

# Figure 2, high-volatility regime, annualised %, read from the PDF vectors.
SHU_HIGH = {1988: 42.28, 1989: 43.43, 1990: 43.88, 1991: 43.20, 1992: 43.52,
            1993: 41.60, 1994: 42.07, 1995: 44.00, 1996: 42.75, 1997: 42.32,
            1998: 41.79, 1999: 32.52, 2000: 21.31, 2002: 21.92, 2008: 25.72,
            2020: 35.60}
SHU_LOW = {1990: 12.99, 1995: 11.38, 1998: 11.39, 2000: 10.00, 2002: 9.68,
           2008: 11.78, 2020: 10.42}


def fit(x: np.ndarray) -> tuple[float, float]:
    f = best_hmm_terminal_fit(pd.Series(x), CFG.model_protocol, CFG.hmm_protocol)
    lo, hi = f.variances
    # annualised, in PERCENT, to match Figure 2's axis
    return float(np.sqrt(lo) * ANN * 100), float(np.sqrt(hi) * ANN * 100)


def windows(series: pd.Series, dates: pd.DatetimeIndex, targets: list[str]):
    w = CFG.model_protocol.fit_window
    out = []
    for t in targets:
        pos = dates.searchsorted(pd.Timestamp(t), side="right") - 1
        if pos >= w - 1:
            out.append((dates[pos], series.iloc[pos - w + 1: pos + 1].to_numpy()))
    return out


def main() -> None:
    targets = ["1990-06-29", "1992-06-30", "1995-06-30", "1998-07-17",
               "1999-06-30", "2000-09-29", "2002-04-15", "2008-10-10",
               "2020-03-23"]

    # Ours: CRSP total market, exactly the series the sealed run used.
    cur = pd.read_csv(RUN / "us" / "features.csv", parse_dates=["date"])
    cur = cur.dropna(subset=["equity_log"]).reset_index(drop=True)
    a = windows(cur["equity_log"], pd.DatetimeIndex(cur["date"]), targets)

    # The paper's series: real S&P 500 daily closes.
    spx = pd.read_csv(ROOT / "data/external/inputs/sp500_price_daily.csv",
                      parse_dates=["date"])
    spx = spx[spx["date"] <= CFG.replication_cutoff.isoformat()]
    spx["log"] = np.log(spx["close"] / spx["close"].shift(1))
    spx = spx.dropna(subset=["log"]).reset_index(drop=True)
    b = windows(spx["log"], pd.DatetimeIndex(spx["date"]), targets)

    with ProcessPoolExecutor(max_workers=9,
                             mp_context=get_context("forkserver")) as ex:
        fa = list(ex.map(fit, [x for _, x in a]))
        fb = list(ex.map(fit, [x for _, x in b]))

    print(f"{'ngày khớp':<12}{'1987 trong cửa sổ':>18}"
          f"{'  vol CAO: CRSP':>16}{'S&P 500':>9}{'Shu (Hình 2)':>14}"
          f"{'  vol THẤP: CRSP':>17}{'S&P 500':>9}{'Shu':>7}")
    cut = pd.Timestamp("1999-08-31")
    for (da, _), (lo_a, hi_a), (db, _), (lo_b, hi_b) in zip(a, fa, b, fb):
        year = da.year
        shu_hi = SHU_HIGH.get(year, float("nan"))
        shu_lo = SHU_LOW.get(year, float("nan"))
        flag = "có" if da <= cut else "không"
        print(f"{str(da.date()):<12}{flag:>18}"
              f"{hi_a:>16.2f}{hi_b:>9.2f}{shu_hi:>14.2f}"
              f"{lo_a:>17.2f}{lo_b:>9.2f}{shu_lo:>7.2f}")

    pre = [(hi_a, hi_b, SHU_HIGH.get(da.year))
           for (da, _), (_, hi_a), (_, hi_b) in zip(a, [f for f in fa], fb)
           if da <= cut and SHU_HIGH.get(da.year)]
    if pre:
        ea = np.mean([abs(x - s) for x, _, s in pre])
        eb = np.mean([abs(y - s) for _, y, s in pre])
        print(f"\nCửa sổ chứa 1987 — sai số tuyệt đối trung bình so với Hình 2:")
        print(f"  CRSP (hiện tại): {ea:.2f} điểm")
        print(f"  S&P 500        : {eb:.2f} điểm")


if __name__ == "__main__":
    main()
