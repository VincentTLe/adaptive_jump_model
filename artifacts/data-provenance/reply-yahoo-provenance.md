**Subject:** Re: how we got 1970s data out of Yahoo without paying

---

Dear Andrew,

Good question, and worth answering carefully because it is the first thing
anyone should ask.

## Short answer

We never used Yahoo's **Download** button. That is the part they charge for.

The chart you see on a Yahoo quote page is drawn from a separate address that
serves the raw numbers as text. That address is open — no account, no payment —
and it returns the whole history, not just the recent part. You can paste this
into a browser and see it yourself:

```
https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC?period1=-1325583000&period2=1785262403&interval=1d
```

That is where our US files came from. For the Japanese index we used `yfinance`,
a free Python package that reads the same address.

What we actually have:

| file | symbol | first day | last day | rows |
|---|---|---|---|---|
| US shares, price | `^GSPC` | 1966-01-03 | 2023-12-29 | 14,598 |
| US shares, with dividends | `^SP500TR` | 1988-01-04 | 2023-12-29 | 9,070 |
| Japanese shares, price | `^N225` | 1965-01-05 | 2023-12-29 | 14,508 |

One caveat worth knowing: that address is rate-limited hard. When I re-tested it
last week we were refused for a whole day. It works for an occasional download;
it is not something to build a pipeline on. That is why we download once and
then lock the file down.

## The longer answer, which I think matters more

I would rather not have you take Yahoo's word for it, and we don't either. The
question "did Yahoo really give us correct 1970s data" can be settled without
trusting Yahoo at all, by checking our files against sources that have nothing
to do with them. We do this routinely, and it is how we caught two real errors
in other parts of our data.

**For the US, two independent academic sources agree with our file.**

*Robert Shiller's monthly S&P 500 series* (Yale, compiled from a completely
different pipeline, going back to 1871). His monthly figure is the average of
the daily closes, so our daily file must reproduce it when averaged. Across
**696 months from 1966 to 2023**, correlation is **0.99999328** and the median
disagreement is **0.0005%**.

*Kenneth French's daily US market returns* (Dartmouth, built from the CRSP
academic database). Comparing daily returns:

| period | correlation with our file |
|---|---|
| 1970-1989 | 0.9867 (5,054 days) |
| 1990-2023 | 0.9927 (8,565 days) |

These are two different indices, so they should not agree perfectly — but a
fabricated or misdated file could not agree this closely with either.

**For Japan, the index publisher itself confirms our figures.** Nikkei launched
its dividend-inclusive index in 2012 but calculated it backwards to a starting
point of **28 December 1979, at 6,569.47**. Our Japanese series is built from the
Yahoo price data plus dividends, anchored in 2011 and chained *backwards* — so
that 1979 value is a prediction our construction never saw. We land at
**6,470.24**, which is 1.5% away after 32 years, or about 0.05% a year.

**And the paper itself checks our 1970s data.** Shu et al. publish nine
statistics describing all three markets over **1970-2023** — exactly the stretch
you are asking about. We reproduce all nine, the worst off by 0.0035. Our
Japanese volatility over that period comes out at 20.54% against their 20.5%.

**Finally, a check that catches misdated data specifically.** The ten biggest
daily falls and ten biggest daily gains in our US file all land on October 1987,
the 2008 crisis, the 2020 Covid crash, or the 1997-98 Asian crisis. The single
worst is 19 October 1987 at −20.47%, which is the historical record. A file that
was scaled correctly but shifted in time would pass every statistical check and
fail this one.

## If you would still prefer a different source

That is easy to accommodate, and honestly the checks above are what make it
easy: if we swapped Yahoo for another provider, those same comparisons would
tell us within minutes whether anything had changed. Stooq and MacroTrends both
carry long histories for these indices for free, and the official S&P and Nikkei
sites publish their own. Say the word and I will re-source and re-run the
comparison rather than argue about it.

I should also correct something in my previous note: I had written that the free
S&P 500 price history starts in 1977. That was an estimate I made before running
the download; it actually returns data from January 1966. I have fixed it in our
records.

Best regards,
Tan
