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

**How much of Table 4 this row now owns (2026-07-28).** After the S&P 500
substitution and the drawdown basis of row 8, turnover is the *only* Table 4
metric outside tolerance in any of the three markets, and this grid is the only
free parameter it depends on. The dependence is not in the smoother and not in
the metric:

- our fixed-k persistence curve reproduces Table 3 to a mean 1.9% on the paper's
  own index, so the smoother and the online state sequence are right;
- Shu's own position path, read off Figure 6 and applied to our returns, gives
  turnover 1.4123 against the published 1.410 and exactly 96 regime shifts, so
  the turnover definition and the trading accounting are right;
- of 128 / 152 / 208 signal flips, only 3 / 3 / 2 are manufactured by switching
  candidate at a month boundary, so the composition layer is not the cause.

Inverting the published turnover through our own fixed-k curve puts Shu's
effective k near 13.4 (us), 3.6 (de) and 6.4 (jp) — no single value, and one of
them lands in the gap our grid leaves between 8 and 20. Our v9 US picks are k20
33%, k8 29%, k6 21%, k0 9%, k4 6%; the 9% of months at k = 0, where no filter is
applied at all, contribute about 23% of all our trading.

**This row is therefore recorded as UNIDENTIFIED, not as an open gap to be
closed.** The paper publishes a selection procedure and no candidate set, and
the turnover row is reachable from within our grid — so a set that reproduces
141% certainly exists. Finding it by search is the move CLAUDE.md forbids. Any
future change here needs a justification that could have been written before the
number was known, as adding k = 6 did.

**The spread, measured.** Eight candidate sets, none adopted, delay 1, all three
markets (artifacts/hmm-residual/08-grid-identification/):

| market | turnover across the eight sets | Table 4 | inside? |
|---|---|---|---|
| S&P 500 | 1.295 .. 2.913 | 1.410 | yes |
| DAX | 1.816 .. 2.432 | 2.460 | 0.028 above; bracketed by the fixed-k curve, so reachable |
| Nikkei | 2.751 .. 4.686 | 2.900 | yes |

The spread is several times the deviation under investigation in every market,
so this row carries no information about replication quality in either
direction.

**A set that gets the US to 8/8, and why it was not taken.** Dropping k = 0 —
`{2, 4, 6, 8, 20}` — puts the S&P 500 inside tolerance on all eight metrics
(turnover 1.442 against 1.410, total deviation 0.088). There is a real a priori
argument for it: the paper says it *applies* a median filter of window k, and a
window of zero applies none, exactly as Table 3 lists it beside lambda = 0 while
calling that column "equivalent to k-means clustering" — a reference point, not
a candidate. It is nonetheless **not adopted**, on two grounds. The argument was
noticed only after measuring that the 9% of months selecting k = 0 drive about
23% of our trading, so its ordering is contaminated. And it does not generalise:
Japan is unchanged and Germany gets slightly worse. A rule that works in one
market out of three is a fit, not a rule.

**No set reproduces Table 4 in all three markets at once.** Scored on all eight
metrics rather than turnover alone, the best any of the eight achieves is 8/8
(us), 7/8 (de) and 7/8 (jp) — and by three different sets. Turnover also trades
against Sharpe: on the US, `dense_wide` reaches turnover 1.295 with Sharpe
0.598, `dense_small` reaches Sharpe 0.541 with turnover 1.501. There is a
frontier here, and the paper does not say where on it to stand.

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

## 7. Turnover definition — CLOSED, the paper states it

Kept as a row so nobody re-opens it. The Table 4 caption (lines 747-753) names
"portfolio turnover" without defining it, and this project carried
`0.5 * sum|d weight|` annualised as an assumption for months. It is not an
assumption. One page later the paper gives the identity in words and numbers:

> [line 781-783] "turnover of the JM-guided 0/1 strategy applied to the S&P 500 is as low as 44%, meaning that on average, the portfolio manager buys and sells 44% of total allocation (a combined 88% trading) each year"

44% one way, 88% combined, denominator the entire allocation. `backtest.py:194`
computes exactly this. The phrase "of total allocation" independently kills the
leverage-scaled readings.

Confirmed a second time, from four figure annotations that state a raw shift
count and a bear share for cells Table 4 also reports — lines 903, 829, 851,
873. Converting each count through the sample length and halving reproduces the
printed turnover to 0.002/0.001/0.007/0.013, and one minus the bear share
reproduces the printed leverage to within a point.

Those annotations are also **targets in their own right**, and counts are
sharper than ratios. Against v8.5, HMM: us 128 shifts against the published 96;
de 151 against 167 implied; jp 208 against 197 implied. The US bear share is
29.1% against 27.8% — the same exposure budget, a third more shifts inside it.

---

## 8. Drawdown basis — CLOSED, two published facts pin it down

**What the paper says.** Nothing directly. Table 4's caption defines the row as
"maximum drawdown ("MDD")" and stops there, and the return row above it is
labelled

> [line 747] "annualized performance metric: compound annual growth rate ("Return", including the risk-free rate),"

which tells us the *return* row credits the cash leg, and says nothing about
which path the drawdown is read from. For a 0/1 strategy those are different
paths, and the difference is large: on the US HMM it is 5.9 percentage points.

**What settles it.** Two published facts, neither of which is a fit:

1. Table 4's buy-and-hold drawdowns (-55.2% / -72.7% / -79.1%) are reproduced to
   0.001 / 0.000 / 0.012 with the equity leg at total return, and missed by
   0.045 / 0.028 / 0.031 on any excess-return path. So the invested leg is total
   return.
2. The caption of Figure 5 says the shading marks days

   > [line 899-900] "when the JM-guided 0/1 strategy is fully invested in the risk-free asset, leading to a flat yellow curve."

   So the plotted strategy path is flat in cash: the cash leg contributes
   nothing.

Total return when invested plus nothing when in cash is a single basis, and it
is forced by those two statements rather than chosen to fit anything.

**What we do.** v9.1 onward: `[metrics] maximum_drawdown =
"risky_leg_wealth_flat_in_cash"`. Configs written before the field existed
default to `total_wealth`, so their sealed runs keep replaying to the numbers
they recorded.

**Why buy-and-hold could not settle it alone.** A portfolio that is never in
cash cannot be told apart by what the cash leg earns; its two columns agree to
every digit. The only cells that can decide are ones where the paper's own
positions are known, which is why Figures 5 and 6 were extracted.

**Consequence, measured.** Across ten cells — three buy-and-hold controls, our
three HMM paths, and four using Shu's own published positions — the mean
absolute drawdown error falls from 0.0330 to 0.0072 and the mean absolute Calmar
error from 0.0262 to 0.0055. The US HMM drawdown moves from -23.21% to -29.24%
against the published -28.9%, and its Calmar from 0.2666 to 0.2117 against 0.21.
Full table in docs/audit/2026-07-full-audit.md and
artifacts/hmm-residual/06-mdd-convention/.

**Left unresolved.** Whether the drawdown path carries the 10bp trading cost.
Including it gives mean errors 0.0116 on MDD and 0.0030 on Calmar — better on
one row, worse on the other, and below what Table 4's printed precision can
separate. Recorded as unresolvable rather than decided.

---

## 9. The Japanese risk-free rate before 1990 — OPEN, measured, not acted on

**What the paper says.** The instrument is specified; the vendor's construction
is not:

> [line 155-157] "For the risk-free rates, we use the 3-month Treasury Bill Yield from each corresponding country, sourced from the Global Financial Data (GFD) database."

**The problem.** Japan had essentially no Treasury bill market before 1986, so
"the 3-month Treasury Bill Yield" for 1970-1986 is whatever the vendor decided
to splice. Ours is the IMF IFS Japan Treasury bill rate; the one independent
series in the repository, JST Macrohistory's `bill_rate`, sits systematically
above it:

| period | ours | JST | gap |
|---|---|---|---|
| 1970-1979 | 5.25% | 7.27% | **-2.01pp** |
| 1980-1989 | 4.29% | 6.27% | **-1.98pp** |
| 1990-1999 | 2.05% | 2.74% | -0.70pp |
| 2000-2020 | 0.10% | 0.07% | +0.03pp |

Correlation 0.9723 — the shape agrees, the level does not. JST's 1974 peak of
12.5% matches the Japanese call rate; ours looks like an administered rate.
Neither is wrong; they are different instruments, and the paper's source made a
third choice we cannot see.

**Consequence, measured.** A cash rate that is too low inflates excess returns,
and Japan is the only market with a buy-and-hold cell outside 0.01 against Table
4. Substituting JST's rate moves the Japanese buy-and-hold Sharpe from 0.1306 to
0.1228 against the published 0.12 — deviation 0.0106 down to 0.0028.

**Not acted on.** That number was obtained by comparing against the target, so
switching on the strength of it is fitting to the answer. The row is recorded as
open and bounded: the Japanese risk-free level carries about a 2pp ambiguity
before 1990 and 0.7pp through the 1990s, and Japanese Sharpe figures inherit
roughly 0.008 of uncertainty from it. Any future change needs a justification
that could have been written before the comparison was run.

**Germany is not exposed the same way.** Its first ladder segment is an interbank
rate, which carries a credit spread, but it ends in 1975-06 and so only touches
the warm-up, never a training window that produces a reported signal.

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

---

## 6. Sample start, and the Japanese Saturday gap — DEVIATION, measured

**What the paper says.** This one is specified, which is why the row is a
deviation rather than a free parameter:

> [line 157] "All data spans from the start of 1970 to the end of 2023."

**What we do.** `requested_sample_start = "1969-05-01"` (v8.4, v8.5) — eight
months earlier than the paper's stated start.

**Why.** Not fidelity. Compensation for a data gap. The Tokyo exchange traded
Saturdays until January 1989 and our `^N225` series contains zero Saturday
sessions, so our 3000-session training window spans about eighteen months more
calendar time than Shu's. Starting literally at 1970-01-01 pushes the realised
out-of-sample start to 1990-03-15 (us), 1990-06-19 (de) and 1990-09-17 (jp),
discarding the first nine months of 1990 in Japan.

Earlier config comments described this as restoring "the paper's 1990-01-02
anchor". The paper names no such date; every mention of 1990 in it is a year.
That description is withdrawn (docs/audit/2026-07-full-audit.md).

**Consequence, measured.** On the overlapping days, the HMM is exactly
invariant to this choice — 0 differing states out of 10,619 / 10,605 / 10,282 —
because it fits the last 3000 log returns before each day and that set does not
depend on where the series began. The jump model is not: 3.71% (us), 1.10% (de)
and 15.84% (jp) of state cells differ, through the expanding standardiser
anchor of row 1 above.

Against buy-and-hold, which contains no model and so tests the window itself:
us 0.486 against 0.497, de 0.298 against 0.305, jp **0.138 against 0.193**,
with Shu at 0.48 / 0.30 / 0.12. The backdated window reproduces the reported
period; the literal one does not.

**Open.** The choice is not free for the jump model. On the shared window it
costs the US JM 0.088 Sharpe and moves its turnover from 0.636 to 1.006 against
Shu's 0.44 — away from the paper on the row the paper treats as the jump model's
identifying property. Reopen before the next JM freeze; do not inherit silently.
