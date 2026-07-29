**Subject:** Progress update — we can now reproduce the paper's results, and what we found along the way

---

Dear Andrew,

An update on where the project stands. I have kept this short and plain; there
is a longer technical version in the attachment if you want the details.

## The short version

Before we extend the Shu, Yu & Mulvey (2024) method, we have to show we can
reproduce what they published. Their main results table has 8 numbers per
market, across 3 markets. **We now match 7 of the 8 in every market.** We match
none of them by luck: every number is produced by code that runs end to end from
raw data files.

Along the way we found **two real mistakes in our own data** — both invisible in
the period the paper reports on, both wrong in the earlier period the models
learn from. Both are now fixed.

The one number that still does not match is *how often the strategy trades*. I
spent a good while on it and I now believe **it cannot be matched by anyone**,
for reasons I explain at the end. That turned out to be the most interesting
thing we found.

## 1. The data — and why it was harder than expected

The authors buy their data from Bloomberg and Global Financial Data. We cannot,
so everything we use is a free substitute that we then have to prove is good
enough.

The hard part was not finding long history. It was finding history **that
includes dividends**. Over 50 years dividends are most of what you earn from
shares, and free sources usually give you share prices only. Two of our three
markets needed us to add the dividends back ourselves.

Here is every file we use and where it comes from:

| What | Where to get it |
|---|---|
| US shares, with dividends (1988 on) | https://finance.yahoo.com/quote/%5ESP500TR/history/ |
| US shares, price only (1966 on) | https://finance.yahoo.com/quote/%5EGSPC/history/ |
| US dividend history (Robert Shiller) | https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv |
| German shares (1959 on) | https://stooq.com/q/d/?s=%5Edax |
| Japanese shares, price only (1965 on) | https://finance.yahoo.com/quote/%5EN225/history/ |
| Japanese shares, with dividends (2011 on) | https://www.investing.com/indices/nikkei-225-total-return-historical-data |
| Dividend yields for Germany and Japan | https://www.macrohistory.net/app/download/9834512569/JSTdatasetR6.xlsx |
| US interest rate | https://fred.stlouisfed.org/series/DTB3 |
| German interest rate (1975-2007) | https://fred.stlouisfed.org/series/INTGSTDEM193N |
| German interest rate (before 1975) | https://fred.stlouisfed.org/series/IR3TIB01DEM156N |
| German interest rate (2007 on) | https://data.ecb.europa.eu/data/datasets/YC |
| Japanese interest rate (to 2017) | https://fred.stlouisfed.org/series/INTGSTJPM193N |
| Japanese interest rate (2017 on) | https://www.stat-search.boj.or.jp/ |

Two extra files we use only to *check* the others, not to run anything:

| What | Where |
|---|---|
| An independent German share index | https://api.db.nomics.world/v22/series/OECD/MEI/DEU.SPASTT01.IXOB.M?observations=1 |
| An independent Japanese share index (MSCI) | https://app2.msci.com/products/index-data-search/ |

**A practical note on your scraping question.** Yahoo does have an open data
address behind the download button, but it blocks you after a few requests — we
got refused for a whole day. Stooq now makes you solve a puzzle in a browser
before it will hand over a file. And the US Federal Reserve site refuses our
machine entirely. So for a fixed research dataset, the sensible thing is to
download each file once by hand and then lock it down, which is what we do: each
file has a fingerprint recorded, and the program refuses to run if a file has
changed. Nothing is downloaded while an experiment is running.

## 2. How we know the data is right

Three checks, none of which relies on us being clever:

**The paper publishes a "do nothing" benchmark.** For each market it reports what
you would have earned simply holding the shares from 1990 to 2023 — six numbers
per market. There is no model in those numbers, so they test the data and only
the data. We reproduce all eighteen. Fourteen match to the last digit the paper prints.
Of the four that do not, one is a rounding hair in the US (−55.25% against
−55.2%) and three are Japanese — the largest being Japan's worst-loss figure,
where we get −78.1% against their −79.1%. Japan is our weakest market and §7
says why.

**The paper also publishes statistics going back to 1970.** This matters because
the "do nothing" check only covers 1990 onward, and the models learn from the
twenty years before that. The paper prints nine numbers describing all three
markets over 1970-2023. We reproduce all nine, the worst off by 0.0035.

**The worst and best days should be the famous ones.** All twenty of the ten
biggest daily gains and ten biggest daily falls in our US data land on October
1987, the 2008 crisis, the 2020 Covid crash, or the 1997-98 Asian crisis. The
single worst is 19 October 1987 at −20.47%, which is the historical record. A
data file that is scaled correctly but dated wrongly would fail this while
passing everything else.

## 3. Two mistakes we found in our own data

Both were the same shape: they looked fine over the period the paper reports on,
and were wrong over the earlier period the models learn from. That is a nasty
place for an error to hide, and it is why we went looking.

**The US: we had the wrong index.** We had been using a broad "whole US market"
index as a stand-in for the S&P 500, because the summary statistics matched. But
on 19 October 1987 the S&P 500 fell 20.5% while the broad market fell 17.4%.
That one day sits inside every twelve-year training window for the next twelve
years, and it made our model think turbulent markets were much calmer than the
paper's model did. The consequence was that our strategy hid in cash through
1998-2002 while theirs stayed invested. Using the real S&P 500 moved all eight
US numbers toward the paper and none away.

**Germany: our data was missing its dividends before 1988.** The German index is
supposed to include dividends. We checked it against an independent German share
index that deliberately excludes them: ours ran 3% a year higher after 1988,
exactly as it should — and ran 0.15% a year *lower* before 1988. So the
dividends were simply absent from the older stretch. The giveaway was that on
the old data, German shares *lost* to a bank deposit by 4% a year for eighteen
straight years, which does not happen. We added the dividends back using an
independent source, and checked the repair three different ways before using it.

Neither fix can be accused of being "tuned to get the right answer": the period
the paper reports on was untouched by both.

## 4. Where we stand

Their published number is on the right of each pair. "Within tolerance" means
the difference is small enough to have been agreed in advance.

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
| **Within tolerance** | **7 of 8** | **7 of 8** | **7 of 8** |

For context, the US column was 4 of 8 when we started this round.

One thing worth mentioning: the paper never says exactly how it measures the
"worst peak-to-trough loss", and for this kind of strategy there are two
reasonable readings that differ by six percentage points. We worked out which
one they used from two clues in their own paper — one of their figure captions
says the strategy's line goes *flat* while it is sitting in cash, which settles
it. That change alone took the US from 5 of 8 to 7 of 8.

## 5. The one number that does not match — and why nobody could match it

The remaining disagreement is **how often the strategy trades**, and it is the
same cell in all three markets.

Here is the situation. The method has one setting: how many days of evidence to
require before believing the market has changed state. A short setting trades
often, a long one trades rarely. The paper does not tell us which values it
considered — it only says the setting is chosen automatically, month by month,
by picking whichever one performed best over the previous eight years.

Two problems, and the second is the real one.

**First**, we do not know the menu they chose from. We tried eight sensible
menus; the trading frequency ranges from about 130% to 290% in the US alone.
That range is much wider than the gap we are trying to explain, so this number
simply is not pinned down by what they published.

**Second — and this surprised me — the rule they use to choose cannot really
choose.** I tested it directly: split each eight-year window in half, ask which
setting wins on the first half, ask which wins on the second half, and see how
often the two agree. If the rule were detecting something real, the two halves
would usually agree.

They agree **17% of the time. With six options, pure guessing gives 17%.**
The same result in all three markets.

I checked that the test itself works: when I gave it two strategies where one is
genuinely and consistently better, it identified the winner **100% of the time**,
even for a very small advantage. So the test can see a real difference. The
settings simply do not have one.

What this means in plain terms: two researchers with the same data, the same
code and the same menu of settings would make different choices in roughly a
third of the months, purely by chance. So this number is not something we got
wrong, and not something we could fix by asking the authors — it is not
reproducible by construction.

I think this may be worth reporting in its own right, and it is one of the two
questions I would like your view on.

## 6. What is next

The model above is the simpler of the two in the paper — it is the benchmark.
The paper's actual contribution is a second model, and that is the one we intend
to extend. It uses the *same* automatic-choice rule described in §5, so I expect
the same instability to show up there, and I want to be careful not to mistake
it for a modelling error when it does.

## 7. Two questions for you

1. For Germany and Japan, no free source gives a single consistent interest-rate
   series back to 1970, so we stitched two or three sources together. Japan is
   the shakiest: the country barely had a Treasury bill market before 1986, so
   different providers report quite different numbers for the same years. Is
   stitching acceptable here, or would you want that handled differently?

2. Is the §5 finding — that the paper's automatic setting-selection is not
   reproducible even in principle — worth writing up as a result on its own, or
   should it stay as a limitation inside the replication?

## Attached

`shu-replication-data-<date>.zip` contains every data file we use, every raw file
they are built from so you can rebuild them yourself, and the detailed technical
version of this report with the full tables.

Best regards,
Tan
