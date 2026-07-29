**Subject:** Re: S&P 500 / DAX / Nikkei data — how to get it free, and where the replication stands

---

Dear Andrew,

Thank you for the links. Answering the scraping question first, then a progress
update, since the two turned out to be connected.

# 1. How to get the data without paying

You are right that "Download to CSV" is the paid feature. But the chart you see
on the page is drawn from a *different* address that serves the raw numbers as
text, and that one is open — no account, no payment, and it returns the full
history rather than a recent slice.

Here is the S&P 500 back to 1966, which you can paste straight into a browser:

```
https://query1.finance.yahoo.com/v8/finance/chart/^GSPC?period1=-126230400&period2=1704067200&interval=1d
```

Three things to know about it:

**The two numbers are dates in "Unix time"** — seconds counted from 1 January
1970. So 1704067200 is 1 Jan 2024. The useful part is that dates *before* 1970
are simply **negative numbers**, which is how you reach back into the 1960s:

| date you want | number to use |
|---|---|
| 1 Jan 1966 | −126230400 |
| 1 Jan 1970 | 0 |
| 1 Jan 2024 | 1704067200 |

(Your own link used −1325583000, which is 30 December 1927.)

**The `^` sometimes needs writing as `%5E`** depending on the browser, so
`^GSPC` becomes `%5EGSPC`.

**What comes back** is a block of text containing two long lists: one of dates,
one of closing prices, lined up position by position. It is a couple of lines of
code to turn into a spreadsheet, and the free Python package `yfinance` does it
for you if you would rather not.

The same address works for your other two symbols — `^N225` and `^GDAXI`.

**One warning.** It refuses you after a handful of requests; when I re-tested
last week we were locked out for a full day. It is fine for downloading a file
once, and no good as a live feed. Stooq (the German source we use) has gone
further and now makes you solve a puzzle in a browser before releasing a file at
all. So for a fixed research dataset, the sensible approach is to download each
file once by hand and then freeze it — which is what we do. Each file has a
fingerprint stored, and the program refuses to run if any file has changed.

# 2. What your three links can and cannot give us

This is where it connects to the research. Two of the three are not usable on
their own, for reasons worth stating:

**`^GSPC` and `^N225` are price-only.** They track share prices but exclude
dividends. Over fifty years dividends are most of what you earn from shares, and
the paper we are replicating explicitly uses dividend-inclusive series. So for
those two markets we have to add the dividends back ourselves, from separate
sources.

**`^GDAXI` does include dividends, but only starts on 30 December 1987.** That
sounds like a small problem and is actually a large one. The method needs about
twenty years of history *before* the first day it can trade, so its results
start in 1990 only because its data starts in 1970. A source that begins in 1987
does not shorten the warm-up — it deletes the first decade of the results. We
use Stooq's German series instead, which reaches back to 1959.

# 3. Where the replication stands

Before extending the method we have to show we can reproduce what the authors
published. Their main table reports 8 numbers per market across 3 markets. **We
now match 7 of the 8 in every market** (the US column was 4 of 8 when this round
began). Their published figure is in bold:

| | S&P 500 | DAX | Nikkei 225 |
|---|---|---|---|
| Annual return | 8.50% vs **8.5%** | 6.78% vs **6.4%** | 2.24% vs **2.5%** |
| Risk (volatility) | 11.31% vs **11.3%** | 14.00% vs **14.0%** | 15.98% vs **16.0%** |
| Return per unit of risk | 0.547 vs **0.54** | 0.367 vs **0.35** | 0.177 vs **0.19** |
| Worst peak-to-trough loss | −29.2% vs **−28.9%** | −43.9% vs **−40.5%** | −51.0% vs **−48.6%** |
| Return per unit of worst loss | 0.212 vs **0.21** | 0.117 vs **0.12** | 0.056 vs **0.06** |
| Typical bad-day loss | −1.79% vs **−1.8%** | −2.20% vs **−2.2%** | −2.51% vs **−2.5%** |
| Share of time invested | 72.4% vs **72%** | 73.0% vs **73%** | 68.0% vs **68%** |
| **How often it trades** | 171% vs **141%** | 226% vs **246%** | 314% vs **290%** |

# 4. How we know the data is right

Since everything we use is a free substitute for data the authors bought, none
of it deserves trust just because it loaded. Four checks, none of which relies
on us being clever:

**The paper's own "do nothing" benchmark.** For each market it reports what you
would have earned simply holding the shares from 1990 to 2023 — six numbers per
market, with no model in them, so they test the data and nothing else. We
reproduce all eighteen; fourteen match to the last digit printed. The largest
miss is Japan's worst-loss figure, −78.1% against their −79.1%.

**The paper also publishes statistics reaching back to 1970**, which matters
because the benchmark above only covers 1990 onward while the models learn from
the twenty years before that. Nine numbers describing all three markets over
1970–2023; we reproduce all nine, the worst off by 0.0035.

**Two independent academic sources agree with our US file.** Robert Shiller's
monthly S&P 500 series at Yale, built from an entirely different pipeline, agrees
with ours across **696 months from 1966** at a median disagreement of **0.0005%**.
Kenneth French's daily US market data at Dartmouth agrees at a correlation of
0.987 over 1970–1989 and 0.993 over 1990–2023.

**For Japan, the index publisher confirms us.** Nikkei launched its
dividend-inclusive index in 2012 but calculated it backwards to a starting value
of 6,569.47 on 28 December 1979. Our Japanese series is anchored in 2011 and
built *backwards*, so that 1979 figure is something our construction never saw.
We land at 6,470.24 — 1.5% away after 32 years, about 0.05% a year.

There is also a check aimed specifically at data that is scaled correctly but
dated wrongly: the ten biggest daily falls and ten biggest daily gains in our US
file all land on October 1987, the 2008 crisis, the 2020 Covid crash, or the
1997–98 Asian crisis, with the worst being 19 October 1987 at −20.47%.

# 5. Two real mistakes we found in our own data

Both had the same shape — fine over the period the paper reports on, wrong over
the earlier period the models learn from. That is a bad place for an error to
hide, which is why we went looking.

**The US: we had the wrong index.** We had been using a broad "whole US market"
index as a stand-in for the S&P 500 because the summary statistics matched. But
on 19 October 1987 the S&P 500 fell 20.5% while the broad market fell 17.4%.
That single day sits inside every training window for the following twelve
years, and it made our model believe turbulent markets were far calmer than the
authors' model did — so our strategy hid in cash through 1998–2002 while theirs
stayed invested. Using the real S&P 500 moved all eight US numbers toward the
paper and none away.

**Germany: our data was missing its dividends before 1988.** The German index is
supposed to include them. We compared it against an independent German index
that deliberately excludes dividends: ours ran 3% a year higher after 1988,
exactly as it should, and 0.15% a year *lower* before 1988 — so the dividends
were simply absent from the older stretch. The giveaway was that on the old
data, German shares lost to a bank deposit by 4% a year for eighteen straight
years, which does not happen. We added the dividends back from an independent
source and checked the repair three ways before adopting it.

Neither fix can be accused of being tuned to produce the right answer: the
period the paper reports on was untouched by both.

# 6. The one number that does not match — and why nobody could match it

The remaining disagreement is **how often the strategy trades**, and it is the
same cell in all three markets.

The method has one setting: how many days of evidence to require before
believing the market has changed state. A short setting trades often, a long one
trades rarely. The paper never says which values it considered — only that the
setting is chosen automatically each month, by picking whichever performed best
over the previous eight years.

Two problems, and the second is the real one.

**First, we do not know the menu they chose from.** We tried eight sensible
menus; trading frequency ranges from about 130% to 290% in the US alone — far
wider than the gap we are trying to explain. So this number is not pinned down
by what was published.

**Second, the rule they use cannot really choose.** I tested it directly: split
each eight-year window in half, ask which setting wins on the first half and
which wins on the second, and see how often the two agree. A rule detecting
something real would usually agree.

They agree **17% of the time. With six options on the menu, guessing gives 17%.**
The same result in all three markets.

I then checked that the test itself works, by giving it two strategies where one
is genuinely and consistently better: it found the winner **100% of the time**,
even for a very small advantage. So the test can see a real difference — the
settings simply do not have one.

In plain terms: two researchers with the same data, the same code and the same
menu would make different choices in roughly a third of the months, purely by
chance. This is not something we got wrong, and not something we could resolve
by writing to the authors.

# 7. What is next, and two questions

The model above is the simpler of the two in the paper — the benchmark. The
paper's actual contribution is a second model, and that is what we want to
extend. It uses the *same* automatic-selection rule, so I expect the same
instability there and want to be careful not to mistake it for a modelling
error.

Two things I would value your view on:

1. For Germany and Japan, no free source gives one consistent interest-rate
   series back to 1970, so we stitched two or three together. Japan is the
   shakiest — the country barely had a Treasury bill market before 1986, and
   different providers report quite different numbers for the same years. Is
   stitching acceptable here, or would you handle it differently?

2. Is the finding in §6 — that the paper's automatic setting-selection is not
   reproducible even in principle — worth writing up as a result in its own
   right, or should it stay as a limitation inside the replication?

I have attached an archive with every data file we use, every raw file they are
built from so the construction can be checked independently, and a longer
technical version of this note.

Best regards,
Tan
