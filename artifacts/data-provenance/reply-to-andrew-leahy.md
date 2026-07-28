**Subject:** Re: Data sources — what we're using, and where the free feeds fall short

---

Dear Andrew,

Thank you — and you're right about the paywall. The "Download to CSV" button
requires a session cookie, but Yahoo's chart endpoint underneath it is open and
returns JSON without any authentication:

```
https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC?period1=-1325583000&period2=1785262403&interval=1d
```

Swap the symbol and it works for anything Yahoo carries. It needs a browser
User-Agent header, and it returns timestamps plus arrays of open/high/low/close,
which is a few lines to flatten. That is exactly how we pull `^GSPC`, `^SP500TR`
and `^N225` — the script is `scripts/fetch_sp500_inputs.py` in our repository.

So the history is obtainable. The constraint that actually bites us is a
different one, and it is worth stating clearly because it shaped every choice
below.

## The binding constraint is total return, not history length

Shu, Yu and Mulvey specify total-return series, and buy the data:

> "The data analyzed in this article comprises the daily total return series of
> three major equity indices: S&P 500, DAX, and Nikkei 225... These data are
> sourced from the Bloomberg Terminal. For the risk-free rates, we use the
> 3-month Treasury Bill Yield from each corresponding country, sourced from the
> Global Financial Data (GFD) database. All data spans from the start of 1970 to
> the end of 2023."

Both are subscription products, so everything we have is a free reconstruction.
Of the three links you sent, `^GSPC` and `^N225` are **price** indices — they
exclude dividends, which over 1970-2023 is most of the equity return. `^GDAXI`
is a performance index and does include dividends, but Yahoo only carries it
from 1987-12-30.

That date matters more than it looks. The method needs a 3000-trading-day
training window plus an 8-year validation window *before* the first tradeable
day, which is why the paper's data starts in 1970 and its results start in 1990.
A series that begins late does not shorten the sample at the front — it deletes
the beginning of the reported 1990-2023 period. So `^GDAXI` from 1987 would cost
us most of the 1990s in Germany.

## What we ended up using

**United States — S&P 500 total return.** The official `^SP500TR` index from
1988-01-04 as published, and before that the `^GSPC` price path plus a dividend
accrual taken from Robert Shiller's monthly S&P 500 dividend series, chained
backwards onto the first official value. Only 1966-1987 is reconstructed and it
sits entirely inside the training window.

We validate rather than assume: we rebuild 1988-2023 *by the reconstruction
recipe* and compare against the official index it is imitating, over their 9,070
shared sessions. Daily log-return correlation 0.999603, annualised volatility off
by 0.0027pp, CAGR off by 0.0837pp. The builder raises an error rather than
returning a series if any threshold fails — which caught a real bug: an early
version divided the trailing-twelve-month dividend by the days in the month
instead of the year, accruing a full year of dividends every month, and it
produced a 39% CAGR. Daily correlation and volatility both looked fine; only the
level check found it.

**Germany — DAX performance index, from Stooq** (`https://stooq.com/q/d/?s=%5Edax`),
which reaches back to 1959 and carries the Stehle academic backcast before 1988.
No dividend reconstruction is needed because the DAX is already a performance
index. The file hits exactly 1000.0 on 1987-12-30 — the official DAX base date
and base value — which is the signature of that lineage. We cross-checked it
against `^GDAXI` (correlation 1.0000 after 2000) and against the independent
OECD monthly share-price index for the 1970s (correlation 0.979-0.985).

**Japan — Nikkei 225 total return.** Official Nikkei 225 TR from 2011-12-19;
before that, the `^N225` price path plus annual dividend yields from the
Jordà-Schularick-Taylor Macrohistory Database, anchored at the first official
value. Validated on the 2012-2023 overlap: daily return correlation 0.9977, and
implied dividend yields within 0.3pp of the JST series.

One quirk worth knowing: the Tokyo exchange traded Saturdays until January 1989,
and the free `^N225` series contains no Saturday sessions. A 3000-session window
therefore spans about eighteen months more calendar time than the paper's, so
starting literally at 1970-01-01 pushes Japan's first out-of-sample day to
September 1990 and throws away nine months of the reported period. We start the
series in 1965 and request 1969-05-01 to compensate, and we document that as a
deviation rather than hiding it.

## The T-bill substitutions

This is the part I would most like your opinion on, because it is where we are
furthest from the paper. The paper uses each country's 3-month Treasury bill
yield from GFD. Only the US has a free daily equivalent covering the span.

**US — no substitution.** FRED `DTB3`, 3-month Treasury bill secondary market
rate, daily from 1954. https://fred.stlouisfed.org/series/DTB3

**Germany — a three-segment ladder, monthly**, because the IMF's German bill
series simply ends in August 2007:

| span | series |
|---|---|
| to 1975-06 | OECD 3-month interbank rate (FRED `IR3TIB01DEM156N`) |
| 1975-07 to 2007-08 | IMF IFS Germany Treasury bill rate (FRED `INTGSTDEM193N`) |
| 2007-09 onward | ECB euro-area AAA 3-month spot yield |

Splice quality measured on the 2004-2007 overlap: −0.09pp ± 0.18. The first
segment is an interbank rate and carries a credit spread, but it only touches
the 1970-75 warm-up and never the reported period.

**Japan — two segments, monthly**, because the IMF's Japanese bill series ends
in June 2017:

| span | series |
|---|---|
| to 2017-06 | IMF IFS Japan Treasury bill rate (FRED `INTGSTJPM193N`) |
| 2017-07 onward | Bank of Japan 3-month uncollateralised call rate |

Across their 28-year overlap the two correlate 0.986, with the call rate
averaging 0.50pp above the bill rate — a level difference we document. In the
negative-rate era the gap is about zero, and Japanese short rates sit near zero
throughout the affected span.

Both ladders are monthly, held flat within the month, and released to the model
with a two-month lag so nothing is used before it could have been known.

## Where the replication currently stands

We have been working on why our HMM baseline did not match the paper's Table 4,
and it turned out to be two separate defects, both found by testing our numbers
against published quantities we had not previously used.

**First, the index.** We had been using the CRSP value-weighted total market as
a US proxy, on the reasoning that buy-and-hold matched. It did not survive
scrutiny. On 1987-10-19 the S&P 500 fell 20.47% and CRSP fell 17.41%, so every
3000-day window containing that day fitted a high-volatility regime about 8
percentage points below the values the paper's Figure 2 publishes — and the
model then turned defensive across 1998-2002 where theirs stayed invested.
Switching to the real S&P 500 moved all eight US metrics toward the paper and
none away. As an independent check, our fixed-window persistence curve now
reproduces the paper's Table 3 to a mean error of 1.9%, against 7.0% on the CRSP
series; Table 3 involves no trading, no selection and no metric definition, so it
tests the state sequence alone.

**Second, and more interesting, the drawdown definition.** The figures in the
paper are matplotlib vector drawings, not bitmaps, so the shaded bear regions
can be read out exactly. We recovered the authors' own daily position paths from
Figures 5 and 6 — four of them — and each panel's printed annotation (bear share
and number of regime shifts) validates the extraction independently: 27.8% and
96 shifts for the US HMM, and so on for the three JM panels.

Running *their* positions on *our* returns reproduces Table 4's turnover to 0.002
and its regime-shift count exactly, and still missed the drawdown by 0.059. A
discrepancy that survives substituting the authors' own regime calls for ours is
not a modelling discrepancy. It is the definition. Two published facts pin it
down: their buy-and-hold drawdowns are only reproducible with the equity leg at
total return, and the caption of Figure 5 states the strategy curve is *flat*
while fully invested in the risk-free asset — so the drawdown path pays nothing
while in cash. Across ten cells, that basis cuts the mean absolute drawdown error
from 0.033 to 0.007.

All three markets now sit at seven of eight Table 4 metrics within tolerance,
against four of eight for the US when we started.

**What is left is turnover, and we are reporting it as unidentified rather than
closing it.** The paper describes a cross-validation procedure for the smoothing
window but never publishes the candidate set it searches. We swept eight
defensible sets: turnover spans 1.30-2.91 in the US, 1.82-2.43 in Germany and
2.75-4.69 in Japan, so the spread from that one unstated choice is several times
the discrepancy we are investigating. One set puts the US inside tolerance on all
eight metrics, and we deliberately did not adopt it — it does not help Japan and
makes Germany worse, and choosing it would be fitting to the answer rather than
replicating.

## Attached

`shu-replication-data-2026-07-28.zip` contains:

- `canonical/` — the six series the model actually reads, as `date,value` CSVs;
- `raw-inputs/` — every raw file they are built from, so the construction can be
  re-run or checked independently;
- `docs/data-provenance.md` — source, URL, span, construction and known
  limitation for each series;
- `docs/hmm-status-vs-table4.txt` — where every metric currently stands against
  the paper, in all three markets.

Everything is rebuilt deterministically from sha256-pinned inputs, and the
hashes are frozen in the research contract, so the builder refuses to run if any
input file has changed underneath us.

Two questions I would value your view on. First, whether the German and Japanese
rate ladders are an acceptable stand-in for GFD's bill series, or whether the
segment boundaries are doing more damage than we think. Second, whether it is
worth asking the authors directly for their candidate grid for the smoothing
parameter — it is the one remaining thing we cannot pin down from the paper, and
it is the only metric still out.

Best regards,
Tan
