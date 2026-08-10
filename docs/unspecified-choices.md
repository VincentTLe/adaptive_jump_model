# Free parameters the paper never fixes

Shu, Yu and Mulvey (2024), arXiv:2402.05272v3. Line numbers refer to
`pdftotext -layout data/external/inputs/2402.05272v3.pdf`, 1168 lines when
split on newlines only (str.splitlines also breaks on the form feeds and
shifts every number).

Every row here is a knob **we** had to set because the paper does not. Results
are conditional on these settings. Read this file before proposing a change to
any knob listed in it, and never search a row for the setting that best matches
the paper's numbers — see CLAUDE.md, "Free parameters the paper never fixes".

Status legend: **open** = still our choice, alternatives materially move the
numbers · **bounded** = alternatives measured, spread known · **closed** = the
paper does pin it after all, row kept so nobody re-opens it.

---

## 1. Feature standardisation geometry — CLOSED as bracketed-but-unidentifiable; largest measured lever

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
model (`src/adaptive_jump/features.py:124-138`, config key `standardizer =
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
| per-refit clip 3 sigma + StandardScaler ⚠ | v8.2 arm | 0.219 | degrades us 0.788 -> 0.460, de 0.361 -> 0.310 |
| the same with lambda fixed at 35 ⚠ | v8.2 arm | 0.310 | not a selectable spec; leverage 43% vs 75% |
| cold start 1970 | v8.2 arm | 0.260 | |

⚠ **The two clip-3-sigma rows are historical measurements only and are withdrawn
from interpretation.** `DataClipperStd` / clipping at three sigma is example code
from the authors' GitHub for a different data set, it is nowhere in the paper,
and by standing owner instruction (2026-07-31, registry `AMENDED` on
`jm-standardizer-geometry-002`) it must not be proposed, run, or cited as an
author-method candidate at all. The numbers stay so nobody re-measures them.

The v8.4 row is **two contract generations stale**: the current sealed baseline
is v10 (`fixed-baselines-36ca1ace131c-…`), where the fixed JM reads us 0.683,
de 0.398, jp 0.291. Quote the run, never the word "current".

**Axis status.** This row is no longer open. `jm-standardizer-geometry-002/-003`
exhausted the author-artifact geometry family — no cadence reproduces the
paper's Table-3 persistence curve (V0 2/6, V1 1/6, V3 3/6 against a rule
requiring 5/6) — and the registry closes the axis with
`no_further_geometry_variants = true`. The honest label is
**bracketed but unidentifiable from public information**, not "open".

**And it is not an independent choice.** The registry records
(`CORRECTION`, `expanding_full_history_ddof1`) that this geometry was identified
partly by which recipe reproduced Table-4 economics — a published number. It is
therefore target-conditioned in the same way the lambda grid is, must be
disclosed wherever the grid's circularity is disclosed, and **no result may be
described as independent of it**.

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

**Update 2026-07-30, after the sealed runs and an external source sweep.**

*Identification, sharpened.* Measured on the sealed v9.3/v9.4 states across
eight defensible candidate sets: no single set puts turnover inside tolerance in
more than one market, and the two that manage one demand opposite ends — the US
is matched only by 4|8|12|16|20 (which makes the US column 8/8) and Germany only
by 2|4 (likewise 8/8 for Germany); Japan by none, though its 2.90 is attainable.
One shared candidate set cannot satisfy both, so no grid choice reproduces the
row. This matches the effective-k inversion above (13.4 / 3.6 / 6.4).

*The grids are unpublished everywhere, now checked at the source.* The author's
`jumpmodels` package (PyPI 0.1.1, sole release) has no CV code, no grid, no HMM
and no median filter; the GitHub repo's only example hard-codes λ = 50 on
Nasdaq-100 data; arXiv v1 published a λ grid ({10, 22, 50, 100, 220, 500, 1000})
for a *different* protocol and dataset and v3 withdrew it; the companion paper
gives only endpoints (0-100, log-spaced). Externally verified 2026-07-29/30,
three-vote adversarial check per claim.

*Where our excess concentrates.* Episode-level comparison against Shu's own
Figure 6 path: 13 extra US bear episodes, 10 of them five sessions or shorter,
mostly post-2000 (so not attributable to the reconstructed pre-1988 segment),
7 of 11 post-2000 flickers in months where the CV picked k = 0. Removing the
k = 0 excess arithmetically (1.795 → 1.528) agrees with the no-zero candidate
set measured independently (1.530) — and still leaves the US outside tolerance,
so this refines the attribution without changing the verdict. There is a
defensible a-priori reading that k = 0 is not a *filter window* candidate at
all; it is recorded, not adopted, because adopting it for its effect on Table 4
would be fitting.

*External echo.* Li, Chen, Tao & Ji (2025), citing the paper, select λ by
information criteria from {0, 5, 10, 25, 50, 100} and land on the upper endpoint
for all twelve assets — the same top-of-grid concentration our runs show, under
a different rule on different data.

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

**And the rule chooses unstably across sub-samples, even with the set fixed.**
Splitting each 8-year validation window in half and comparing the argmax of each
half gives agreement of 17.1% (us), 15.1% (de) and 17.4% (jp).

> **Corrected 2026-08-06.** This paragraph used to compare those rates against
> "a chance rate of 16.7%" and to conclude that the rule "is not reproducible in
> principle". Both were retracted by
> `docs/audit/2026-07-29-codex-review-verdicts.md` §5 and the wording was never
> propagated here. 16.7% is `1/k`, the agreement rate for *uniform* draws, and
> the rule's choices are far from uniform (us: k20 33%, k8 29%, k6 21%, k0 9%,
> k4 6%, k2 1%). Computed from the two halves' own argmax distributions, the
> independence baseline is **20.2% (us), 20.8% (de), 22.7% (jp)** — so observed
> agreement is *below* independence, not at chance. The audit's instruction is
> explicit: "17.1% against chance 16.7%" must not be quoted again.

The selection rule is deterministic: identical data, code and grid always select
the same window, so nothing here makes it irreproducible. What is unstable is
the *estimate* — which candidate wins depends on which sample it is scored on.

Two limits of the instrument belong with the number. A deterministic control —
identical return paths, one handicapped by a constant 0.1 of Sharpe — agrees
100% of the time, but the realistic controls do not: invested-versus-cash, two
strategies that differ enormously, agree only 77% (us), 63% (de) and 46% (jp).
And two halves of a validation window are consecutive periods, not two draws
from one distribution, so if the best k genuinely drifts then disagreement is
signal rather than noise. The design cannot separate those.

What survives without any baseline argument: the median winning margin is
0.030-0.047 of Sharpe, about a third of months are decided by less than 0.02,
and the choice flips between consecutive months 5.5%-7.2% of the time even
though those two windows share seven eighths of their data. Turnover is
therefore unidentified twice over: the candidate set is unpublished, and a
Sharpe-ranked rule cannot pin down a quantity that Table 3 shows moving from 8.5
to 2.0 shifts per year across the same candidates.

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
(`scripts/data/build_external_sources.py`, splice deltas recorded in the config).

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

## 8. Drawdown basis — REOPENED 2026-07-29, the "pin" does not hold

> **This row was wrong and is retracted.** Fact 2 below misreads the caption it
> quotes. Figure 5's y-axis is *cumulative excess return* —
>
> > [line 786] "alongside the curves of cumulative excess returns from the three"
>
> — and a cumulative **excess** return curve is flat while in cash for the
> trivial reason that excess return is zero there. The flatness is implied by the
> axis label alone and carries no information about which path a drawdown is read
> from. Fact 1 stands but discriminates nothing relevant: buy-and-hold is never
> in cash, so `total_wealth`, `risky_leg_wealth_flat_in_cash` and its
> cost-retaining variant agree to every digit on all three control rows.
>
> With the pin gone, the only discriminating evidence is the four cells driven by
> Shu's own published positions — and there the two published rows disagree about
> the winner (MDD prefers the cost-free flat path at 0.0072 against 0.0116;
> Calmar prefers the cost-retaining one at 0.0030 against 0.0055). The selection
> in `scripts/probe_mdd_convention.py` is "whichever convention minimises error"
> against Table 4, which is exactly the search over an unspecified knob that
> `CLAUDE.md` forbids.
>
> A further objection to the adopted basis, independent of the fitting: it drops
> the 10bp trading cost from the drawdown path while the Return row charges it,
> so one Table 4 column would report return net of cost and drawdown gross of it.
>
> **What we do now.** `total_wealth` is the a-priori default again — the
> conventional reading of a portfolio drawdown, and what v8.x used. The
> flat-in-cash bases are reported as alternatives, not as the faithful choice.
>
> **What the retraction costs: nothing.** Rescored under both bases at the 0.05
> tolerance, every cell in all three markets gives the same verdict, and all
> three markets are 7/8 either way. See
> `artifacts/hmm-residual/01-status/hmm-vs-table4-v9-3.txt` and
> `docs/audit/2026-07-29-codex-review-verdicts.md` §2.

## 8a. The superseded argument, kept for the record

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

**What we did, v9.1 through v9.3** (superseded; v9.4 and v10 are back on
`total_wealth`): `[metrics] maximum_drawdown =
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

**Consequence, measured — and NOT separately identified.** Our annualised mean
excess return for Japanese buy-and-hold is 0.0305 against the 0.0281 implied by
Table 4's Sharpe times its volatility: a gap of 0.245pp/year. Substituting JST's
rate removes 0.18pp of it and moves the buy-and-hold Sharpe from 0.1306 to
0.1228 against the published 0.12. But our equity CAGR is also 0.20pp above the
published 0.8%, and the two candidates sum to more than the gap, so neither can
be named as the cause. An earlier draft of this row called the rate "the leading
explanation"; that is withdrawn.

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

## 10. hmmlearn estimation constants — BOUNDED by structural identity, spread unmeasured

**What the paper says.** The stack and the restart rule, and nothing finer:

> [line 371] "we execute the algorithm ten times from"

different k-means++-derived initial values, retaining the highest
log-likelihood; and

> [line 382] "On the training window ending at day t, we use the Viterbi algorithm to decode the hidden"

state sequence. It does not name min_covar, priors, iteration cap or
convergence tolerance.

**What we chose (sealed v9.4).** min_covar 1e-3 (the hmmlearn default),
covars_prior 0.0, transmat_prior 1.0, n_iter 1000, tol 1e-6 with a symmetric
strict-convergence gate.

**The author-code point of comparison (not the paper).** Shu's 2023 simulation
codebase (`continuous-jump-model`, see the 2026-07-30 round-2 audit note) is
structurally identical — same init_params set, same 10× k-means++ restart rule,
same Viterbi `.predict()` decoding — and differs only in five constants:
min_covar 1e-6, covars_prior 1e-6, transmat_prior 1+1e-5, n_iter 500, tol 1e-4.
Both parameterizations are near-MLE (zero or ~1e-6 pseudo-counts), so a priori
they should reach the same local optima under our stricter convergence gate.

**Not adopted, spread unmeasured.** These are example-code defaults from a
different study, exactly the class of artifact CLAUDE.md forbids promoting into
paper claims, and switching to them now — after knowing our turnover sits above
the target — would be searching a knob near the answer. The row exists so the
axis is on the books: if a spread measurement is ever wanted, it needs its own
frozen question first. Until then this row's honest status is: sensitivity
unknown, prior expectation small.

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

---

## 11. Which two calendar months anchor the semiannual JM refit — OPEN, spread unmeasured

**What the paper says.** The cadence, and only the cadence:

> [line 544] "In our JM implementation, the optimal model parameters Θ̂ are updated every six months by"

Figure 3's caption repeats it for the illustration:

> [line 612] "a 3000-day training window that moves forward every six months. Each date on the x-axis represents the"

**What the paper does NOT say.** Which two months. "Every six months" fixes the
interval, not the phase. Nothing in the paper names a refit date, and Figure 4's
in-sample window is described only as "ending at the end of 2020".

**What we chose.** `refit_months = [1, 7]` — January and July
(`config.py:_jm_protocol`, frozen to that literal in every contract). In the
sealed v10 run this lands on the first trading day of each January and July,
plus one bootstrap fit at the first eligible terminal day.

**Consequence.** Unmeasured. Six phases are possible ([1,7] through [6,12]);
each shifts every refit date by up to five months, which moves the parameters
Θ̂ that every subsequent day's online inference is conditioned on. The
mechanism is the same one row 1 measures for standardization geometry, so a
material spread cannot be ruled out a priori.

**Not measured here on purpose.** Sweeping the phase means running the
walk-forward six times and comparing against numbers we already know, which is
exactly the search this file exists to forbid. If the sensitivity is wanted it
needs its own frozen question, written before the six runs, that says what a
large spread would mean.

---

## 12. Sharpe denominator — CLOSED by the paper's own caption, deviation measured and immaterial

**What the paper says.** Table 4's caption defines the reported statistic:

> [line 751] "Sharpe ratio (average excess return over volatility)"

and the selection objective is the same statistic:

> [line 710] "We then select the value λ̂ that yields the highest Sharpe ratio"

**What the paper does NOT say.** Whose volatility — the strategy's, or the
excess series'. For a 0/1 strategy that sits in cash part of the time these are
different series, so the caption admits two readings.

**What we chose.** `mean(strategy − cash) / std(strategy)`, i.e. excess return
in the numerator and **total strategy** volatility in the denominator
(`backtest.py:annualized_excess_sharpe`, pinned as
`sharpe_denominator = "strategy_return_volatility"`). The textbook Sharpe uses
`std(strategy − cash)` in both places. The caption's wording — "average excess
return over volatility", not "over excess volatility" — reads slightly toward
ours, but it does not settle it.

**Consequence, measured.** Recomputed both conventions on all nine delay-1 rows
of the sealed v10 baseline (`fixed-baselines-36ca1ace131c-…`): the two
definitions differ by at most **0.000037** of Sharpe and 0.000023 on average
(largest cell: jp fixed_jm, 0.290783 against 0.290746). That is three orders of
magnitude below the 0.05 reporting tolerance, because daily cash returns are
tiny and almost constant. The row is recorded for completeness, not as a live
uncertainty.


---

## 13. EWM warm-up rows inside the expanding standardiser — OPEN, measured, undisclosed until 2026-08-06

**What the paper says.** Nothing. The features are EWM statistics and the paper
never states a burn-in:

> [line 504] "expectation is based on exponentially decaying weights over historical periods"

**What we chose.** `make_features` emits features from the very first rows
(`min_periods=0`, `burn_in_observations = 0`), so a "60-day-halflife" Sortino
exists from four observations — US rows 3-7 read 2.6, 3.3, 2.7, 1.6, 2.1
against a mature p1/p99 of -0.32/+0.62. `standardize_expanding` then drops the
first 63 rows from its *output* but keeps them in its *statistics* forever.

**Consequence, measured (2026-08-06 audit).** At US row 3063 the expanding std
of `sortino_60` is +43.4% and its mean +8.9% versus a 63-row burn-in. In the
fit window ending 1990-01-04 the per-feature stds are [1.50, 0.79, 0.63]
(anisotropy 2.39x) against [1.50, 0.89, 0.86] (1.75x) with a burn-in — so part
of the anisotropy row 1 attributes to the expanding geometry is actually
warm-up contamination. Over the OOS sample the US `sortino_60` z-score moves by
mean 0.109 sigma, max 0.612 sigma (jp 0.039/0.220, de 0.011/0.052).

**Bounded where tested.** Refitting the JM on that window under both variants
changes 0 of 3000 in-sample states (us, lambda 21.54) and 3 of 3000 (jp,
lambda 10); centroids and objective move (us 3097.5 -> 3821.6). A full
walk-forward under a feature burn-in has not been run and would need its own
frozen question; do not run it casually against known targets.

---

## 14. Daily risk-free conversion — OPEN, bounded, undisclosed until 2026-08-06

**What the paper says.** The instrument only:

> [line 155-157] "For the risk-free rates, we use the 3-month Treasury Bill Yield from each corresponding country, sourced from the Global Financial Data (GFD) database."

No day count, no compounding rule (`scripts/audit/check_paper_claims.py` machine-checks
that 252/365/360/day-count never appear in the body).

**What we chose.** `annual_percent / 100 / 252`, simple and uncompounded,
accruing on trading days only (`features.py:81-83`).

**Bounds, measured.** Compounded daily conversion would credit cash ~13 bp/yr
less at the sample-mean 4.47% yield. Actual/365 accrual moves Monday excess
returns by ~2 bp (cancels over a year). Separately, the US source DTB3 is a
discount-basis quote while the paper says "Yield": a bond-equivalent conversion
would raise the mean risk-free by ~14.4 bp/yr and lower every excess return by
the same. All are level effects far below the 0.05 Sharpe tolerance; recorded
so the axis is on the books.

---

## 15. The paper's printed DD formula omits its square root — CLOSED against the printed text

**What the paper prints.**

> [line 501] "Downside deviation, calculated as E R2 1{R<0} where R denotes the excess return and the"

— literally the second lower partial moment, no radical.

**What the code does.** `features.py:100-107` takes the square root.

**Why the code is right and the printed formula is the typo.** Four independent
checks, none of them our fit: (1) the paper itself says scaling DD by sqrt(2)
puts it on the volatility scale [line 614-615], which is true only with the
root (a second moment would need x2); (2) measured on the sealed US frame,
sqrt(2)*sqrt(252)*dd_10 has mean 15.1% against realised annualised excess vol
18.1% — the caption's claim holds numerically, while the un-rooted version
gives 1.7%; (3) Figure 3's DD panel is drawn on a 20%-80% axis, which fits the
rooted centroids and cannot fit 1.7-5%; (4) Shu's dissertation states the root
explicitly (docs/audit/2026-07-30-jm-deep-research.md:19). **Do not "fix" the
code to match line 501.**
