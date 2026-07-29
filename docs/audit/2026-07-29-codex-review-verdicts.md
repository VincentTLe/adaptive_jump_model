# Verdicts on the external (Codex) review, 2026-07-29

An external review raised ten findings against this repository. Each is checked
below against the code and the data, with the evidence to reproduce it. The
review's own numbers reproduce almost everywhere; where it overstates, that is
said, and where it understates, the stronger version is recorded.

Two findings (#1, #11) are blockers. #11 is not in the review: it was found while
checking #8, and it is the most consequential item on this page.

| # | claim | verdict |
|---|---|---|
| 1 | S&P 500 splice deletes a session | **CONFIRMED** — fixed, v9.3 |
| 2 | MDD/Calmar convention was fitted to Table 4 | **CONFIRMED** — retract |
| 3 | 2024-26 holdout is not selection-clean | **CONFIRMED** — relabel |
| 4 | holdout turnover is 2x too low; artifact incomplete | **CONFIRMED** |
| 5 | selection-noise conclusion unproven | **CONFIRMED** — retract wording |
| 6 | separation preregistration unprovable | **CONFIRMED with a nuance** |
| 7 | legacy cold archive lost | **CONFIRMED** |
| 8 | no full sealed v9 replication | **CONFIRMED** |
| 9 | replication underidentified | **CONFIRMED, already documented** |
| 10 | hygiene: ruff, DOIs, claim-checker scope | **CONFIRMED** |
| 11 | every HMM cell fails the project's own boundary gate | **NEW — blocker** |
| 12 | cache scripts skip the contract's sample-start trim | **NEW** |

---

## 1. The S&P 500 splice deleted 1988-01-04 — CONFIRMED, fixed

`scripts/build_sp500_tr.py` reconstructed the pre-official era with
`px["date"] < split`, so the reconstruction ended on 1987-12-31 — the session
*before* the official index begins. It then anchored that 12-31 level to the
official index's first close (1988-01-04) and dropped the row.

Consequence, measured:

```
price return 1987-12-31 : -0.3147%
price return 1988-01-04 : +3.5859%
compounded 12-30..01-04 : +3.2599%
what the shipped file had: -0.3008%     (i.e. 12-31's return, on 01-04's row)
```

The 1988-01-04 session was deleted outright; the file carried 14,597 rows where
the union of its inputs has 14,598. The defective file is the one pinned by
v9, v9.1 and v9.2 (sha `548ff7ca…`). Every rolling 3000-session window from
1988-01-04 to late 1999 read it, which covers the first decade of the reported
1990-2023 sample.

The review's diagnosis of the mechanism ("xóa nhầm 1987-12-31") is right. Its
`+3.2599%` is the price-only compounding; with the dividend accrual the correct
figure is `+3.29%`.

**Fix.** `stitch()` now reconstructs *through* the splice date and hands over to
the official index the day after, so the splice date is one both series cover.
Verified surgical: exactly one row restored and exactly one return corrected,
with 14,596 of 14,597 daily log returns bit-identical to the old file.

**Scope check — the other two builders do NOT have this bug.** Both take an
inclusive slice, so the row they drop is one the official series supplies:

- `scripts/build_de_total_return.py:91` `price.loc[:OFFICIAL_START]`, and it
  asserts `early.index[-1] == OFFICIAL_START`;
- `scripts/build_external_sources.py:155` `price.loc[:first_official]`.

**Status.** Data rebuilt (sha `5fcdf04f…`), contract `research-expanding-v9-3.toml`
written, US HMM refitted under it by `scripts/cache_v9_3_us_hmm.py`, which
carries three guards: the sealed v8.5 metric row reproduces through the same
code path (drift 3.6e-14); the file-loading path reproduces the stored v9
feature frame (max gap 1.0e-16 over 13,786 x 13); and the trim reproduces what
`acquire` wrote, with the *only* differences being the two splice rows.

That third guard caught a confound that would otherwise have invalidated the
comparison: `acquire` trims each series to `requested_sample_start` at fetch
time, so loading `data/external/` directly would have fed the HMM 811 extra
sessions from 1966-1969 and confounded the fix with a longer history.

**Result of the refit.** The correction moves the US column *away* from Shu,
which is the direction that makes a fix credible rather than suspicious:

```
                       v9 (defective)   v9.3 (repaired)   Shu
sharpe                     0.5471           0.5255        0.54
cagr                       0.0850           0.0820        0.085
turnover                   1.7065           1.7947        1.410
regime shifts                 116              122          ~96
```

The US column stays 7/8, because turnover was already the failing cell and its
deviation merely widens from 0.296 to 0.385.

## 2. The drawdown convention was chosen by fitting — CONFIRMED, retract

The justification given for `risky_leg_wealth_flat_in_cash` was that Table 4's
buy-and-hold row plus the caption of Figure 5 pin it down without fitting. That
justification does not survive reading the caption in context.

The caption is real — [line 899] "invested in the risk-free asset, leading to a flat yellow curve" — but [line 786] states the axis: "the curves of cumulative excess returns". A cumulative
**excess** return curve is flat while in cash for the trivial reason that excess
return is zero there. The flatness is implied by the axis label and carries no
information about the drawdown path.

What the published evidence can and cannot do:

- buy-and-hold is never in cash, so conventions A (total wealth), D (flat in
  cash) and E (flat in cash, costs retained) are **identical to the last digit**
  on all three control rows. The controls eliminate B and C and nothing else.
- the only discriminating evidence is the four cells driven by Shu's own
  published positions, and there the two published rows disagree:

```
MDD    mean abs error:  A 0.0330   D 0.0072   E 0.0116   -> D wins
Calmar mean abs error:  A 0.0262   D 0.0055   E 0.0030   -> E wins
```

`scripts/probe_mdd_convention.py` selects "whichever convention minimises error"
against Table 4. That is a search over an unspecified knob for the setting that
best matches the target paper — the thing `CLAUDE.md` forbids by name.

A point the review did not make, which makes D worse: **D drops the transaction
cost from the drawdown path while the Return row charges it**, so a single
Table 4 column would be reporting return net of costs and drawdown gross of them.

**Consequence.** The v9.1 change, which is what moved the US column from 5/8 to
7/8, is not a faithfulness fix. The drawdown basis returns to
`docs/unspecified-choices.md` as an open knob whose a-priori default is
`total_wealth`, with the spread across A/D/E reported as a limitation.

## 3. The 2024-2026 window is not selection-clean — CONFIRMED

`AGENTS.md:132` already states it correctly: *"A separate authorized source audit
already inspected public candidate series through July 2026, so those dates are
not untouched confirmation data."* `artifacts/data-source-audit/20260712T012740Z/audit.json`
carries dates through 2026-07-12.

The labels elsewhere contradict that:

- `research/holdout-2026-001.toml:4` `claim_class = "CONFIRMATORY_HOLDOUT"`,
  `:5` `stage = "ONE_SHOT_UNTOUCHED_WINDOW"`
- `paper/manuscript.tex:57, 995, 1015, 1070` and `README.md:38` "selection-clean"

The window is model-untouched and P&L-untouched. It is not selection-independent.
Every such label must be demoted to a post-selection exploratory readout.

The manuscript's own limitations paragraph (`:1110`) already says no untouched
confirmation sample supports a performance claim, so the defect is confined to
the labels, not the conclusion.

## 4. Holdout turnover is exactly half, artifact is not confirmatory-grade — CONFIRMED

`src/adaptive_jump/holdout_runner.py:_metric_row` calls `performance_metrics`
with `periods_per_year` and `volatility_ddof` only. It never passes
`turnover_scale`, so the function default 0.5 applies, while
`research-holdout-2026.toml:122` declares `mean_one_way_turnover_times_252`
(scale 1.0). Verified by recomputation from the stored trades:

```
us dd_only full: scale0.5 = 0.4046   contract scale1.0 = 0.8092
                 recorded in artifact = 0.404624   <- the 0.5 value
```

Every turnover figure in the holdout artifact is a factor of two too low. Sharpe
and the 0/3 conclusion do not depend on turnover and are unaffected.

The artifact holds `summary.json`, `holdout-metrics.csv`, a PNG and four CSVs per
market. There is no `run.json`, no `inventory.json`, no `config.lock.toml`, no
`data-manifest.json` — all four of which the sealed baseline runs carry. It does
not meet the bar its own claim class asserts.

Two additions the review did not make:

- the bootstrap "never positive" wording is wrong but trivially so: the positive
  draws are at 6.7e-16 and 1.3e-15, i.e. floating-point zero. Restate as "never
  positive beyond floating-point zero".
- **the window has almost no discriminating power.** In the holdout, all four
  German models post an identical Sharpe of 0.904079 with zero turnover, and
  three of four Japanese models an identical 1.270104. No model left equities on
  any day. That is why the German bootstrap gap is 1e-16.

## 5. "The selection rule reads noise and is irreproducible" — CONFIRMED, retract

Two separate errors.

**The chance baseline was understated.** The probe uses `1/len(grid)` = 16.7%.
For two draws from a non-uniform choice distribution the agreement rate under
independence is the cross-marginal product, not the uniform rate. Recomputed
from the half-window argmax distributions:

```
market  observed   1/k     independence baseline
us        17.1%   16.7%          20.2%
de        15.1%   16.7%          20.8%
jp        17.4%   16.7%          22.7%
```

The review's 20.15 / 20.78 / 22.65 reproduce. So agreement is not "at chance" —
it is **below** the independence baseline in all three markets. That does not
rescue the rule, but "17.1% against chance 16.7%" was a coincidence of the wrong
baseline and must not be quoted again.

**The reproducibility claim is a logic error.** `docs/audit/2026-07-full-audit.md:1637`
reads *"two researchers with identical data, identical code and identical grids
would select different smoothing windows in any month whose margin falls inside
the noise"*. The selection rule is deterministic; identical inputs give identical
output every time. The correct statement is about **sampling reliability**: the
choice is unstable with respect to which sample it is estimated on. The same
wrong sentence reached the advisor email and must be corrected there too.

**The design cannot separate no-signal from nonstationarity**, as the review
says. Two halves of a four-year window are two different periods, not two draws
from one distribution; if the best k genuinely drifts, disagreement is signal.
The probe's own realistic controls already demonstrate the confound — invested
vs cash agrees only 77/63/46% — and that was recorded at the time. It should be
stated as a limitation of the instrument, not as a property of the rule.

## 6. Separation preregistration — CONFIRMED, with a nuance

`research/separation-turnover-001.toml:4` declares `frozen_at_utc = 09:05:00Z`
while the artifact directory is stamped `…20260722T083551Z`. A freeze declared
half an hour after the run finished is not a freeze.

The nuance the review misses: `research/experiment_registry.jsonl:91` carries a
`FROZEN` entry at `08:32:46Z`, *before* the artifact at 08:35:51Z and before
completion at 08:40:34Z. The self-reported chronology is therefore internally
consistent, and only the TOML's own field is wrong.

But the review's stronger point stands: `git log` shows the spec first appearing
in `f015b30` — *"Complete the frozen separation-turnover exploratory as not
supported"* — the same commit as the result. Nothing independent of the author's
own timestamps corroborates the ordering.

The rho values recompute and the descriptive `not_supported` verdict is sound.
The per-month permutation ignores serial dependence in switch counts, so the
p-value is not a valid time-series inference.

## 7. The only copy of the legacy archive is gone — CONFIRMED

`.agent/session-log.jsonl:111` (2026-07-22T08:41:38Z) records
*"data/legacy-cold/legacy-minute-data.tar.zst (5.2G minute-era raw+processed
compressed to 1.1G; originals removed after tar+zstd verification)"*.

`data/legacy-cold/` does not exist, and `find /home/tle -name
legacy-minute-data.tar.zst` returns nothing. `.gitignore:72` lists
`data/legacy-cold/`, so the archive was never in version control.

Originals deleted, single copy, ignored by git, now absent. Who removed it
afterwards is not determinable from here, but the design was unsound before
anything was lost.

## 8. There is no full sealed v9 replication — CONFIRMED

Both v9 caches say so themselves:

- `artifacts/hmm-residual/v9-us-hmm/run.json` — *"HMM arm only, US only, v9
  config; NOT a sealed run"*
- `artifacts/hmm-residual/v9-2-de-hmm/run.json` — *"HMM arm only, Germany only,
  v9.2; NOT a sealed run"*

and neither v9.1 nor v9.2 has an acquisition manifest at all: the German cache
loads its two legs from disk with a hash check, because the full acquisition
path fetches the US bill rate live from FRED, which refuses this machine. The
v9.3 US refit inherits the same limitation and records it in its `run.json`.

The v9-series numbers are therefore per-market HMM-arm caches and cannot be
assembled into a replication result. `paper/manuscript.tex` still describes v7.

## 9. The replication remains underidentified — CONFIRMED, already documented

The paper fixes neither the scaler recipe, the EWM flags, the warm-up, the
grids, the refit anchor, the priors, the tie rule, nor exact data identifiers.
`docs/unspecified-choices.md` exists for this and `CLAUDE.md` states the rule.
Nothing new, and correctly characterised by the review.

## 10. Hygiene — CONFIRMED

- `ruff check scripts`: 59 errors (the review counted 57; two are from today's edits)
- `ruff format --check src tests`: 3 files would be reformatted
  (`src/adaptive_jump/features.py`, `tests/test_audit_hardening.py`,
  `tests/test_expanding_variant.py`)
- `paper/manuscript.tex:1226` `doi:10.10/s10479-024-06035-z` and `:1286`
  `doi:10.11/s41260-024-00376-x` are truncated. The real prefixes are `10.1007/`
  and `10.1057/`. Neither DOI as printed resolves.
- `scripts/check_paper_claims.py:56-63` scans `docs/unspecified-choices.md`,
  `docs/audit/2026-07-full-audit.md`, `CLAUDE.md` and `research*.toml`. It does
  **not** scan `paper/manuscript.tex` or `README.md` — the two documents a reader
  actually reads.
- 46 generated artifacts are force-tracked through `.gitignore` negations.

## 11. Every HMM cell fails the project's own upper-boundary gate — NEW, blocker

Found while checking #8; not in the review.

`upper_boundary_month_fraction_limit = 0.05` is part of the frozen selection
protocol. It counts the months in which the selection picks the *largest*
candidate on the grid; above 5%, the grid is binding and the optimum lies
outside it. The sealed v8.5 run enforces it and **fails**:
`status = "boundary_failed"`, `metrics_opened = false`, conclusion *"grid
expansion required before OOS metrics"*.

```
                     months at k=20   fraction   gate
v8.5  us hmm d1          14/408         3.4%     pass
v8.5  de hmm d1          22/408         5.4%     FAIL
v8.5  jp hmm d1         161/408        39.5%     FAIL
v9.3  us hmm d1          90/408        22.1%     FAIL
```

**A correction to how this was first measured.** The figures first written here
(33.0% for v9, 9.6% for v9.2, 21.5% for v9.3) counted *every* decision month.
The gate does not: `walkforward.boundary_diagnostic` restricts to OOS months,
`decision_dates >= oos_start`, which is why the sealed run's denominator is 408
and a naive count gives 418 or 468. Every number above is now produced by that
function, and it reproduces the sealed run's three rows exactly (14/408, 22/408,
161/408) as a guard before being trusted on anything new.

Two things follow.

**The reported cells were opened against the gate.** The German and Japanese HMM
columns come from a run whose own status forbids opening OOS metrics, and the
US v9 cache — the one whose 7/8 was reported — sits at 33.0%, six times the
limit and ten times the sealed v8.5 US figure it replaced. Substituting the S&P
500 for CRSP fixed the tail defect and simultaneously pushed the selection hard
against the ceiling of the grid.

**The turnover conclusion was premature.** The stated finding was that turnover
is unidentifiable at two levels. But a ceiling that binds in a third to two
fifths of all months means the rule wants *more* smoothing than the grid offers,
which forces *more* trading than its own objective would choose. That is the
direction of the US deviation (1.71 against 1.41) and of the Japanese one (3.14
against 2.90). Germany runs the other way (2.26 against 2.46) and is not
explained by it.

Extending the grid until the boundary stops binding is **not** fitting to
Table 4: the stopping criterion is the project's own pre-registered gate, which
is defined without reference to the paper's numbers. That test has never been
run, and it must be, before turnover can be called unidentified.

**After the v9.3 repair the gate still fails**, at 22.1% for the US against a
5% limit — so the ceiling is a property of the grid, not of the defective
series:

```
                        months at top of grid   fraction   gate
us   v9.3 hmm d1               90/408            22.1%     FAIL
de   v8.5 hmm d1               22/408             5.4%     FAIL
jp   v8.5 hmm d1              161/408            39.5%     FAIL
```

Every cell in `artifacts/hmm-residual/01-status/hmm-vs-table4-v9-3.txt` is
scored under this binding ceiling. A cell inside tolerance under a constraint
that binds is not a settled cell, and the 7/8 in each market should be read with
that attached.


## 12. The cache scripts skip the contract's sample-start trim — NEW

Same family as #1, found while putting all three markets on one contract.

`data.py` hands `sample_start` and `replication_cutoff` to `fetch_source`, so a
canonical processed file arrives already trimmed to the study window. Every
script that loads `data/external/` directly — which all the v9 caches do,
because the acquisition path fetches the US bill rate from FRED and this machine
is refused — skips that step and feeds the model whatever the file contains.

The US script was caught by its own guard 3 and trims explicitly. The German one
does not, and neither did the v9.2 cache that produced the reported DAX column:

```
contract requested_sample_start = 1969-05-01
us  v9.3 frame  13,787 rows  1969-05-02..2023-12-29   (trimmed)
de  v9.2 frame  14,851 rows  1965-01-05..2023-12-29   (NOT trimmed)
```

Four extra years of German history enter the fit. The visible consequence is the
decision-month count: the v9.2 German arm decided **468** months against the
sealed v8.5 run's **416**, because an earlier first-complete date moves the OOS
start earlier.

Whether it moves the reported metrics is a separate question and is answered
empirically rather than assumed: the states at any date depend only on the
trailing 3000 observations, and by 1990 that window no longer reaches 1969, so
the scored 1990-2023 cells may be untouched even though the decision history is
not. The refit under v9.3 carries a guard that reproduces the stored v9.2
German column, which settles it either way.

What is *not* untouched is the boundary fraction of §11, whose denominator is
every decision month. The German figure of 9.6% (45/468) and the sealed 5.3%
(22/416) are not measured over the same months, and neither should be quoted
without saying which.


## Two mistakes made while fixing the above, recorded because they were nearly costly

**A silent fallback printed a mislabelled row.** `hmm_status_v9_3.py` read each
market's path from its cache and fell through to the sealed run when the cache
was absent. After the repository cleanup removed the v9.2 German states, the
table went on printing a DAX row labelled "v9.2" that was in fact v8.5 — a
boundary fraction of 5.3% where v9.2's own number was different. The label and
the data disagreed and nothing said so. The fallback is now an error: a row that
cannot be built from the source it claims to describe is not printed at all.

**The boundary gate was measured the wrong way.** The fractions first reported
in §11 counted every decision month. `walkforward.boundary_diagnostic` counts
OOS months only, which is why the sealed run's denominator is 408 while a naive
count gives 418 or 468. All figures now come from that function, and it is
checked against the sealed run's own three rows (14/408, 22/408, 161/408) before
being trusted anywhere new. The conclusion is unchanged — the US sits at 22.1%
against a 5% limit — but the numbers were not the gate's numbers.

**A guard fired for the right reason and was believed too quickly.** Refitting
Germany under v9.3 tripped its guard with "v9.3 and v9.2 disagree, worst
maximum_drawdown -0.4020 against -0.4385". Both are the same position path: one
read on `total_wealth`, the other on the flat-in-cash basis v9.2 declared. The
comparison, not the contracts, was wrong. The guard did its job — it refused to
write a cache on a false premise — and the episode is the argument for having
written it.
