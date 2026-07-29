# Data provenance

Every canonical series below is rebuilt deterministically by
`scripts/build_external_sources.py` from sha256-pinned raw inputs in
`data/external/inputs/`, and every output hash is pinned again in the frozen
research contract (`research-expanding-v9-3.toml`). The builder refuses to run
if any input hash has moved.

## What the paper uses, and why we cannot

> [line 150-157] "The data analyzed in this article comprises the daily total return series of three major equity indices: S&P 500, DAX, and Nikkei 225, representing the US, Germany, and Japan, respectively. These data are sourced from the Bloomberg Terminal5 . For the risk-free rates, we use the 3-month Treasury Bill Yield from each corresponding country, sourced from the Global Financial Data (GFD) database. All data spans from the start of 1970 to the end of 2023."

Both sources are subscription products. Every series here is a free
reconstruction of one of them, and each is classified in the contract as
`documented_proxy_candidate` rather than a match.

## How early the data has to start

The paper says 1970, but 1970 is not a free choice — it is what the procedure
forces. Section 3.4.2 fits on a 3000-trading-day window and Section 3.4.3 adds
an 8-year validation window before the first out-of-sample day, so the series
must begin roughly twenty years before the reported period starts:

> [line 713-715] "Since our data begin in 1970, with training windows spanning 12 years and validation windows 8 years, the out-of-sample testing period begins in 1990."

A series that starts late therefore does not shorten the sample at the front —
it deletes the beginning of the reported 1990-2023 period. Required anchor is
about **1970-02**; all six of our series clear it by four to sixteen years.

| series | starts | vs the paper's 1970 | vs the required anchor |
|---|---|---|---|
| us equity (S&P 500 TR) | 1966-01-03 | +4.0 yr | +4.1 yr |
| us cash (DTB3) | 1954-01-04 | +16.0 yr | +16.1 yr |
| de equity (DAX) | 1965-01-04 | +5.0 yr | +5.1 yr |
| de cash (ladder) | 1965-01-01 | +5.0 yr | +5.1 yr |
| jp equity (N225 TR) | 1965-01-05 | +5.0 yr | +5.1 yr |
| jp cash (ladder) | 1965-01-01 | +5.0 yr | +5.1 yr |

The binding constraint is **not history length. It is total return.** Free daily
history is easy to find; free daily *total-return* history is not, and the paper
is explicit that it uses total-return series. Two of the three markets need a
dividend reconstruction over part of the span, and both reconstructions sit
entirely inside the training window, never inside the reported period.

## Equity

### United States — S&P 500 total return

| segment | source | note |
|---|---|---|
| 1988-01-04 onward | `^SP500TR`, the official S&P 500 Total Return index | used as published |
| 1966-01-03 .. 1988-01-03 | `^GSPC` price path + Shiller monthly dividends | reconstructed, chained onto the first official value |

- `^SP500TR` — https://finance.yahoo.com/quote/%5ESP500TR/history/ (9,070 sessions)
- `^GSPC` — https://finance.yahoo.com/quote/%5EGSPC/history/ (14,598 sessions from 1966)
- Shiller dividends — https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv

Validated by rebuilding 1988-2023 with the reconstruction recipe and comparing
against the official index it imitates, over their 9,070 shared sessions: daily
log-return correlation **0.999603**, annualised volatility off by **0.0027 pp**,
CAGR off by **0.0837 pp**. `scripts/build_sp500_tr.py` raises rather than
returns if any of the three thresholds fails.

**This series replaced the CRSP value-weighted total market on 2026-07-28.** The
substitution was the traced cause of the US HMM deviation: on 1987-10-19 the
S&P 500 fell 20.47% and CRSP fell 17.41%, so every 3000-day window containing
that day fitted a high-volatility regime about 8 pp below the values Figure 2 of
the paper publishes. See `docs/audit/2026-07-full-audit.md`.

### Germany — DAX performance index

- Stooq `^dax` — https://stooq.com/q/d/?s=%5Edax (16,815 sessions from 1959-09-28)

The DAX is a performance index, so dividends are already inside it and no
reconstruction is needed. Before 1988 the series carries the Stehle academic
backcast; the file hits exactly **1000.0 on 1987-12-30**, the official DAX base
date and base value, which is the signature of that lineage.

Yahoo's `^GDAXI` starts 1987-12-30 and is therefore **not usable on its own** —
it misses the entire training and validation history the procedure requires.
Cross-checks: correlation 1.0000 against `^GDAXI` after 2000, and monthly
correlation 0.979-0.985 against the independent OECD MEI share-price index
before 1988. Known limitation: pre-Xetra fixings 1988-1999 differ from Yahoo
closes intraday (daily correlation 0.82-0.90) while monthly levels agree.

### Japan — Nikkei 225 total return

| segment | source | note |
|---|---|---|
| 2011-12-19 onward | official Nikkei 225 Total Return | used as published |
| 2020-07-09 .. 2022-05-31 | `^N225` price path + calibrated accrual | bridges a hole in the mirror; endpoint error 2e-16 |
| 1965 .. 2011-12-18 | `^N225` price path + JST annual dividend yields | reconstructed, anchored at the first official value |

- Nikkei 225 TR — https://indexes.nikkei.co.jp/en/nkave/index/profile?idx=nk225tr
- `^N225` — https://finance.yahoo.com/quote/%5EN225/history/ (14,508 sessions from 1965-01-05)
- JST Macrohistory — https://www.macrohistory.net/database/

Validated on the 2012-2023 overlap: daily return correlation **0.9977**, implied
dividend yields within **0.3 pp** of the JST series.

Known limitation, and the reason the contract requests a 1969-05-01 start rather
than 1970-01-01: the Tokyo exchange traded Saturdays until January 1989 and our
`^N225` series contains no Saturday sessions, so a 3000-session window spans
about eighteen months more calendar time than the paper's. Starting literally at
1970-01-01 pushes the first out-of-sample day to 1990-09-17 in Japan and throws
away the first nine months of the reported period.

## Risk-free rates

The paper uses each country's 3-month Treasury bill yield from Global Financial
Data. Only the US has a free daily equivalent covering the whole span; Germany
and Japan need documented ladders, and those ladders are the largest remaining
data-side approximation in the study.

### United States — direct, no substitution

- `DTB3`, 3-month Treasury bill secondary market rate, discount basis —
  https://fred.stlouisfed.org/series/DTB3 — daily, from 1954-01-04.

Fetched live rather than stored, with a 1-day availability lag and a 10-day
staleness limit enforced by the pipeline.

### Germany — three segments, monthly

| span | series | source |
|---|---|---|
| .. 1975-06 | OECD MEI 3-month interbank rate | https://fred.stlouisfed.org/series/IR3TIB01DEM156N |
| 1975-07 .. 2007-08 | IMF IFS Germany Treasury bill rate | https://fred.stlouisfed.org/series/INTGSTDEM193N |
| 2007-09 .. | ECB euro-area AAA 3-month spot yield | https://data.ecb.europa.eu/data/datasets/YC |

The IMF German bill series simply ends in 2007-08, which forces the third
segment. Splice quality measured on the 2004-2007 overlap: **-0.09 pp +- 0.18**.
The first segment is an interbank rate rather than a bill rate and carries a
credit spread, but it only touches the 1970-75 warm-up, never the reported
period.

### Japan — two segments, monthly

| span | series | source |
|---|---|---|
| .. 2017-06 | IMF IFS Japan Treasury bill rate | https://fred.stlouisfed.org/series/INTGSTJPM193N |
| 2017-07 .. | BoJ 3-month uncollateralised call rate | https://www.stat-search.boj.or.jp/ |

The IMF Japanese bill series ends in 2017-06. Overlap agreement across 28 years:
correlation **0.986**, with the call rate averaging **+0.50 pp** above the bill
rate — a documented level caveat. In the negative-rate era the joint delta is
about zero, and Japanese short rates sit near zero throughout the affected span,
so the effect on the strategy is small.

Both ladders are monthly, held constant within the month, and made available to
the model with a two-month-start lag.

## What is irreducible

1. **Bloomberg's exact index vendor and close convention** for the three equity
   series. Bounded rather than closed: results are reported at trading delays of
   1, 5 and 10 days, and the delay-10 column shows how much a mis-timed close
   could matter.
2. **GFD's exact bill definitions** for Germany and Japan. Approximated by the
   ladders above, with every splice measured.
3. **The Japanese pre-2012 total return.** No free official series exists.
   Reconstructed and validated on a 12-year overlap.
4. **The German pre-1988 daily path.** The Stehle backcast has been checked
   against OECD monthly data; no independent *daily* source exists publicly.

## Refreshing the data: what actually works

Checked 2026-07-28, from this machine, after the morning's fetch had succeeded:

| host | result |
|---|---|
| `query1`/`query2.finance.yahoo.com/v8/finance/chart/...` | **429 Too Many Requests** — both hosts, caret encoded or not |
| `finance.yahoo.com/quote/<sym>/history/` | 404 to a plain client |
| `stooq.com/q/d/l/?s=...` | 200, but the body is a JavaScript proof-of-work challenge, not a CSV |
| `fred.stlouisfed.org/graph/fredgraph.csv?id=...` | timed out 2026-07-28; **200 on 2026-07-29**, 226,513 bytes in 2.9s |
| `raw.githubusercontent.com` (Shiller) | 200 |
| `data-api.ecb.europa.eu` | 200 |

So the Yahoo chart endpoint is open in the sense that it needs no login, but it
is rate-limited hard enough that it cannot be treated as a dependable feed, and
Stooq's CSV link now requires a real browser. That is why `stooq_dax_daily.csv`
is recorded in the contract as manually downloaded on 2026-07-25.

**None of this can break a run.** Fetched inputs are stored under
`data/external/inputs/` with pinned hashes, and the one series the pipeline
still fetches live — `DTB3` — is captured into the run's acquisition manifest
(`data/raw/<config>-<timestamp>/us_cash.csv`) and verified against it rather
than re-downloaded. The v9 manifest already holds DTB3 for 1969-05-01..2023-12-29,
14,262 rows.

If a series ever does need refreshing, download it once in a browser, drop it in
`data/external/inputs/`, and update the `INPUT_SHA256` pin in
`scripts/build_external_sources.py`. The builder refuses to run on unpinned
bytes, which is the property that makes browser downloads acceptable here.


## The US bill rate, checked against an independent manual download

FRED refused this machine on 2026-07-28 and answered normally on 2026-07-29, so
the v9.3 bundle was acquired through the pipeline rather than by hand. The owner
separately downloaded the same series from a browser, which turns a convenience
into a real check: an automated fetch and a human one, made independently,
should agree byte for byte or one of them is wrong.

```
manual download            sha256 62106f6db8dcade6dc70bdd75ae89dc08720e6fdef7e012bb193aae7d8e74471
data/raw/shu-replication-expanding-v9-3-20260729T081133Z/us_cash.csv
                           sha256 62106f6db8dcade6dc70bdd75ae89dc08720e6fdef7e012bb193aae7d8e74471
```

Identical, so the manual copy was deleted rather than kept as a second source of
truth: it is already stored under `data/raw/<run>/us_cash.csv` with its hash
recorded in that run's manifest.

One note for anyone repeating the comparison. DTB3 carries **605 blank rows**
over 1969-2023 — US market holidays, the first three being 1969-05-30,
1969-07-04 and 1969-07-21. pandas reads those as NaN, and `(a == b).all()` is
therefore `False` on two byte-identical files. Use `a.equals(b)`, or compare the
hashes, which is what settles it here.

The exact request the pipeline makes:

```
https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3&cosd=1969-05-01&coed=2023-12-31
```
