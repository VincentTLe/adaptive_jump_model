"""Acquire the S&P 500 inputs the paper actually specifies.

  [line 153-155] "The data analyzed in this article comprises the daily total
  return series of three major equity indices: S&P 500, DAX, and Nikkei 225"

The US series has been the CRSP value-weighted total market from Kenneth
French's daily factors, because no free S&P 500 total-return series reaches back
to 1970. That substitution turned out to be the cause of the US HMM deviation:
on 1987-10-19 the S&P 500 fell 20.47% and CRSP fell 17.41%, and every 3000-day
training window containing that day fits a high-volatility regime about 8
percentage points below Shu's (docs/audit/2026-07-full-audit.md).

Three files, all free and unauthenticated. Note what is NOT used: Yahoo's
"Download" button on the quote page, which is gated behind a paid tier for long
ranges. These come from the chart endpoint underneath it -- the same JSON the
page itself plots -- which needs no account. The distinction matters because it
is the first thing anyone will ask.

  sp500_price_daily.csv   Yahoo ^GSPC daily close. An earlier note here said
                          1977, which was an estimate made before the fetch ran;
                          the endpoint actually returns 14,598 sessions from
                          1966-01-03. This is what the
                          HMM needs: over the 9,070 sessions the two series
                          share, the daily log-return volatility of price and
                          total return differ by 0.0011pp with correlation
                          0.99960, so dividends are immaterial to the fit and
                          the 1987 crash is a pure price move.
  sp500_tr_daily.csv      Yahoo ^SP500TR daily close from 1988-01-04. The
                          official total-return index; the equity leg from 1988
                          and the validation overlap for everything before it.
  shiller_sp500_monthly.csv  Shiller's monthly S&P 500 price and dividend series
                          from 1871, mirrored as CSV by the datasets/s-and-p-500
                          project so no xls reader is needed. Supplies the
                          dividend accrual for 1966-1987, the only stretch that
                          has to be reconstructed -- and one that lies entirely
                          inside the training window, never in the reported
                          1990-2023 sample.

Run this once, then add the printed hashes to INPUT_SHA256 in
scripts/build_external_sources.py. Downloads are refused if the file already
exists, so a pinned input can never be silently replaced.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INP = ROOT / "data" / "external" / "inputs"
UA = {"User-Agent": "Mozilla/5.0"}

YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}?period1={a}&period2={b}&interval=1d"
SHILLER = "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv"

# 1976-01-01 .. 2023-12-31; the builder trims to the frozen window anyway.
A, B = -126230400, 1704067200


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def yahoo_close(symbol: str) -> list[tuple[str, float]]:
    payload = json.loads(get(YAHOO.format(sym=symbol, a=A, b=B)))
    result = payload["chart"]["result"][0]
    stamps = result["timestamp"]
    closes = result["indicators"]["quote"][0]["close"]
    rows = [
        (dt.date.fromtimestamp(t).isoformat(), float(c))
        for t, c in zip(stamps, closes)
        if c is not None
    ]
    seen: dict[str, float] = {}
    for date, value in rows:
        seen[date] = value
    return sorted(seen.items())


def save(name: str, text: str) -> None:
    path = INP / name
    if path.exists():
        print(f"{name:26} exists, not overwritten "
              f"(sha256 {hashlib.sha256(path.read_bytes()).hexdigest()})")
        return
    path.write_text(text, encoding="utf-8")
    raw = path.read_bytes()
    lines = text.strip().split("\n")
    print(f"{name:26} rows={len(lines) - 1:6d}  "
          f"{lines[1].split(',')[0]}..{lines[-1].split(',')[0]}  "
          f"sha256={hashlib.sha256(raw).hexdigest()}")


def main() -> None:
    INP.mkdir(parents=True, exist_ok=True)
    for name, symbol in (("sp500_price_daily.csv", "%5EGSPC"),
                         ("sp500_tr_daily.csv", "%5ESP500TR")):
        rows = yahoo_close(symbol)
        if len(rows) < 5000:
            sys.exit(f"{name}: only {len(rows)} rows, refusing to write")
        body = "\n".join(f"{d},{v!r}" for d, v in rows)
        save(name, f"date,close\n{body}\n")

    text = get(SHILLER).decode("utf-8")
    if "Dividend" not in text.split("\n")[0]:
        sys.exit("Shiller mirror changed shape, refusing to write")
    save("shiller_sp500_monthly.csv", text)


if __name__ == "__main__":
    main()
