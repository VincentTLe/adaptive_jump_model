# Free parameters the paper never fixes

Shu, Yu and Mulvey (2024), arXiv:2402.05272v3. Line numbers refer to
`pdftotext -layout 2402.05272v3.pdf`, 1167 lines.

Every row here is a knob **we** had to set because the paper does not. Results
are conditional on these settings. Read this file before proposing a change to
any knob listed in it, and never search a row for the setting that best matches
the paper's numbers — see CLAUDE.md, "Free parameters the paper never fixes".

Status legend: **open** = still our choice, alternatives materially move the
numbers · **bounded** = alternatives measured, spread known · **closed** = the
paper does pin it after all, row kept so nobody re-opens it.

---

## 1. Feature standardisation geometry — OPEN, largest known lever

**What the paper says.** Only that the features arriving at the model are
standardised:

> [line 397] "In our application, given an observation sequence of D standardized features"

Section 3.4.1 (line 494 onward) defines the three features and their halflives
and says nothing further about scaling. Searched: `clip`, `winsor`,
`standardi[sz]`, `normali[sz]`, `preprocess`, `scale`, `outlier`, `robust`.

**What the paper does NOT say.** Whether the scaler is fitted on the whole
history, on the training window, or per refit; whether features are clipped or
winsorised at all. `DataClipperStd` / clipping at three sigma appears in the
authors' GitHub example notebook for the NASDAQ data set. **It is not in the
paper.** Do not cite it as if it were.

The one adjacent statement is about raw returns, not features, and sits in the
Data section:

> [line 169-170] "Despite a few extreme returns during these events, we do not process outlier values to minimize manual intervention."

**What we chose.** Causal expanding full-history standardisation anchored at the
sample start, `min_observations = 63`, then `_IdentityScaler` into the jump
model (`src/adaptive_jump/features.py:124-145`, config key `standardizer =
"expanding_full_history_ddof1"`).

**Consequence, measured.** The fit window is therefore not centred or unit
scaled. On the v8.4 run, features entering each JM refit have

| market | `dd_10` std | `sortino_20` std | `sortino_60` std | anisotropy |
|---|---|---|---|---|
| us | 1.220 | 0.769 | 0.656 | 1.86x |
| de | 1.242 | 0.927 | 0.892 | 1.39x |
| jp | 1.020 | 0.854 | 0.824 | 1.24x |

with means +0.196 to +0.307 on `dd_10`. The jump model minimises
`0.5*||x - theta||^2`, an isotropic distance, so in the US fit `dd_10` carries
about 3.5x the weight of `sortino_60`. The distortion differs by market, which
is why the deviation from the paper differs by market.

**Spread across alternatives** (Japan JM Sharpe, paper reports 0.31). Judge these
by how far the point estimate sits from the paper, not by whether a confidence
interval covers it — see "Why the confidence interval proves nothing" below:

| variant | jp JM Sharpe | side effect |
|---|---|---|
| expanding, anchored (current) | 0.157 / 0.169 | us and de closest to paper |
| per-refit clip 3 sigma + StandardScaler | 0.219 | degrades us 0.788 -> 0.460, de 0.361 -> 0.310 |
| the same with lambda fixed at 35 | 0.310 | not a selectable spec; leverage still 43% vs 75% |
| cold start 1970 | 0.260 | |
| anchor 1970 | 0.263 | |

Ledger conclusion, already established: no single preprocessing variant
reproduces Table 3 and Table 4 at the same time. Treat this row as bounded
evidence, not as an unsolved bug to keep re-litigating.

---

## 2. Jump-penalty candidate grid — OPEN

**What the paper says.** That the penalty is chosen monthly by validation
Sharpe over an eight-year lookback:

> [line 704-711] "we use a time-series cross-validation approach, updating the optimal jump penalty monthly... We then select the value that yields the highest Sharpe ratio during this validation period"

Table 3 (line 643) exercises lambda in {0, 5, 15, 35, 70, 150} and the text
calls 50 to 100 "a typical value" (line 638-639).

**What the paper does NOT say.** The candidate grid actually used for selection.
Table 3 is an illustration of persistence, not a stated search space.

**What we chose.** `lambda_grid = [0, 5, 15, 35, 70, 150]`, matching Table 3.

**Consequence.** On v8.4 the Japanese JM parks at the top of that grid in 30.4%
of months (`boundaries.csv`), i.e. the selector wants more persistence than the
grid offers. Widening was measured on the HMM side and moved the optimum rather
than bracketing it; the same test has not been run for lambda.

---

## 3. HMM smoothing grid — OPEN

**What the paper says.** The same monthly validation-Sharpe procedure selects
the HMM smoothing window:

> [line 715-716] "We employ the same method to optimally select the smoothing hyperparameter k for HMMs."

Table 3 exercises k in {0, 2, 4, 8, 20}.

**What we chose.** `smoothing_grid = [0, 2, 4, 8, 20]`.

**Consequence, measured.** Boundary gates fail because the selector parks at
k = 20 (jp delay-1 39.5%, us delay-10 39.0%). Extending the grid was measured
across six shapes: extensions that clear the gate on one market break it on
another, and the long-tail grids that clear all three move the HMM Sharpe far
from the paper (de +0.043 -> -0.240). Grid choice is therefore a live free
parameter with a known, large spread. Rejected on a priori grounds: the paper's
own values are the only non-arbitrary choice available.

---

## 4. Risk-free instrument for Germany and Japan — BOUNDED

**What the paper says.**

> [line 155-157] "For the risk-free rates, we use the 3-month Treasury Bill Yield from each corresponding country, sourced from the Global Financial Data (GFD) database."

**What the paper does NOT say.** How GFD constructs those series back to 1970,
which matters because German Bubills and Japanese short bills did not trade
across the whole window. Their series is almost certainly itself a chain.

**What we chose.** Documented ladders: Germany OECD 3M interbank -> IMF IFS
T-bill -> ECB 3M AAA; Japan IMF IFS T-bill -> BoJ 3M call after 2017-06
(`scripts/build_external_sources.py`, splice deltas recorded in the config).

**Consequence, measured.** Swapping the US 1-month for the 3-month bill moves
features by 0.01 sigma, flips the sign of the signal on 0.57% of days, and
shifts every Sharpe by 0.02-0.03 in the same direction. A level effect, not an
ordering effect.

---

## 5. Comparison sample across models and delays — CLOSED in config, was open

**What the paper says.** Nothing about how rows are aligned across models and
delays when computing Table 4.

**What we chose.** `comparison_sample =
per_market_all_delays_intersection_of_complete_metric_rows`
(`config.lock.toml:129`).

**Consequence, measured.** The Japanese buy-and-hold Sharpe is 0.193 under this
frozen rule and 0.189 under a per-model sample. The audit ledger quoted 0.189,
i.e. a convention the run itself does not use. Any figure quoted from a run must
name its convention.

---

## Why the confidence interval proves nothing here

A bootstrap confidence interval on our own Sharpe was used, repeatedly, as if
covering the paper's value were evidence of replication. It is not, and the
reason is arithmetic rather than a defect in the resampling.

The standard error of a Sharpe ratio is about `sqrt((1 + SR^2 / 2) / T)` with
`T` in years, so a 95% interval spans roughly:

| years | SR = 0.2 | SR = 0.5 | SR = 0.7 |
|---|---|---|---|
| 20 | 0.885 | 0.930 | 0.978 |
| **34** | **0.679** | **0.713** | **0.750** |
| 100 | 0.396 | 0.416 | 0.437 |

The paper's window is 34 years, so any interval we compute is about 0.70 wide.
Our measured widths, 0.646 to 0.681, match that. An interval that wide covers
essentially every value anyone might propose: on the US cell the buy-and-hold
interval also contains the paper's HMM *and* JM figures, so it cannot even
reject "buy-and-hold equals the jump model". Reaching a width of 0.05 would take
roughly 6,900 years of data.

There is also no valid significance test available. The paper reports one number
per cell with no standard error, computed on licensed data we do not hold, so
there is no sampling distribution to test against and no null hypothesis to
reject. "Their value is inside our interval" is not a test result.

What to report instead:

- **Closeness of the point estimate**, against a tolerance stated in advance.
  The owner's standing tolerance is 0.05 in absolute Sharpe, tightening to 0.03.
- **The spread across our own free choices**, which is what the rows above
  measure. That spread is the honest uncertainty statement for a replication:
  it says how much of the gap we could produce ourselves by setting an
  unspecified knob differently.
- **Paired differences within our own run** (JM minus HMM on the same days),
  where the market noise cancels. Even these stay wide: on v8.3 the US
  difference was +0.099 with an interval of [-0.073, +0.278], so the ordering
  is a statement about point estimates and must be written that way.

## Maintenance

Add a row whenever a decision is made that the paper does not force. Record the
quote that shows the paper is silent, the choice, and the measured spread. A row
without a measured spread is an admission that the sensitivity is unknown.
