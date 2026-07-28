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

**Spread across alternatives** (Japan JM Sharpe, paper reports 0.31). Every row
names the run it came from, because the same recipe moves as other settings
change and stale figures were previously carried forward under the label
"current":

| variant | run | jp JM Sharpe | side effect |
|---|---|---|---|
| expanding, anchored 1969-05 | **v8.4 (current)** | **0.197** | us JM 0.662 vs 0.68, de JM 0.391 vs 0.44 |
| expanding, anchored 1970 | v8.3 | 0.260 | out-of-sample window started 1990-08, not 1990-01 |
| expanding, min_obs 250 | v8.1 / v8.2 | 0.157 / 0.169 | superseded windows |
| per-refit clip 3 sigma + StandardScaler | v8.2 arm | 0.219 | degrades us 0.788 -> 0.460, de 0.361 -> 0.310 |
| the same with lambda fixed at 35 | v8.2 arm | 0.310 | not a selectable spec; leverage 43% vs 75% |
| cold start 1970 | v8.2 arm | 0.260 | |

Only the first row describes the current pipeline. Quote it, or name the run.

Ledger conclusion, already established: no single preprocessing variant
reproduces Table 3 and Table 4 at the same time. Treat this row as bounded
evidence, not as an unsolved bug to keep re-litigating.

---

## 2. Jump-penalty candidate grid — OPEN

**What the paper says.** That the penalty is chosen monthly by validation
Sharpe over an eight-year lookback:

> [line 704-711] "we use a time-series cross-validation approach, updating the optimal jump penalty monthly... We then select the value λ̂ that yields the highest Sharpe ratio during this validation period"

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

**What the paper does NOT say.** The candidate set. Section 3.4.3 describes the
selection procedure and never lists what it selects from. Table 3 exercises k in
{0, 2, 4, 8, 20} (line 643), but that table is an illustration of how
persistence responds to k, not a declaration of the search space:

> [line 646] "Table 3: Average number of shifts per year in the online inferred regime sequence from 1982 to 2023,"

The one value the paper does name is the literature default it inherits from
Bulla et al. (2011), the same paper cited at line 387 as the source of the
median filter:

> [line 390] "Originally, k was set at 6; in our approach, it is selected from a range of candidate values automatically via a cross-validation framework."

**What we chose.** v8 through v8.4: `smoothing_grid = [0, 2, 4, 8, 20]`, copied
from Table 3. **v8.5 onward: `[0, 2, 4, 6, 8, 20]`.** Copying Table 3 dropped
k = 6, so the candidate range excluded the only value the paper names — our
construction error, not a property of the paper. The grid as a whole stays a
free parameter; only the omission of 6 was a defect.

The justification for adding 6 is that the paper names it. It is **not** that it
improves agreement with Table 4 — see CLAUDE.md on never searching an
unspecified knob for the setting that best matches the target. The direction of
the effect was measured after the decision, not before it.

**Consequence, measured.** Under the Table-3 grid the boundary gates fail
because the selector parks at k = 20 (jp delay-1 39.5%, us delay-10 39.0%).
Extending the grid was separately measured across six shapes: extensions that
clear the gate on one market break it on another, and the long-tail grids that
clear all three move the HMM Sharpe far from the paper (de +0.043 -> -0.240).
Grid choice is therefore a live free parameter with a known, large spread; the
v8.5 change is the one edit to it that has an a priori justification.

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

## Do not report confidence intervals in this project

Standing instruction from the owner, and it is the right call. Intervals were
used here as if covering the paper's value were evidence of replication, which
it never was: the paper publishes one number per cell with no standard error, on
licensed data we do not hold, so there is no sampling distribution to test
against and no hypothesis to reject. On top of that, at this sample length an
interval on a Sharpe ratio is wide enough that the buy-and-hold cell covered the
paper's jump-model figure as well as its own — a test that cannot separate the
three models it is asked about is not a test.

The arithmetic that was published here to make that point was itself wrong: it
applied the Lo (2002) standard error with an annualised Sharpe and a sample size
counted in years, mixing two frequencies. It is retracted along with the rest.

Report instead:

- **Closeness of the point estimate** to the paper, against a tolerance fixed in
  advance. The owner's standing tolerance is 0.05 absolute Sharpe, tightening to
  0.03. Count model cells separately from buy-and-hold: buy-and-hold contains no
  model, so it measures the data, not the replication.
- **All eight rows of Table 4, not the Sharpe row alone.** Turnover is the
  paper's own headline property of the jump model ("as low as 44%"), and it is
  where the current run diverges most.
- **The spread across our own free choices**, which is what the rows above
  measure. That spread is the honest uncertainty statement for a replication: it
  says how much of the gap we could produce ourselves by setting an unspecified
  knob differently.

## Maintenance

Add a row whenever a decision is made that the paper does not force. Record the
quote that shows the paper is silent, the choice, and the measured spread. A row
without a measured spread is an admission that the sensitivity is unknown.
