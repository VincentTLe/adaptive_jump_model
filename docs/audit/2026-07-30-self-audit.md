# Self-audit, 2026-07-30 — re-checking my own work, Codex's review, and what the deep research adds

Requested by the owner: re-examine everything done since the external review.
Method: re-run what can be re-run, diff my written claims against the artifacts,
and fold in the 85 externally-verified claims from the deep-research sweep
(3-vote adversarial verification per claim; synthesis stage failed on a spend
limit, raw verified claims retained; sources: PyPI/GitHub of the author's
package, arXiv v1 and v2406.09578, Bulla et al. 2011, Nikkei factsheets, GFD's
own index guide, Yahoo chart API metadata, citing papers).

## A. What re-verifies

| check | result |
|---|---|
| sealed v9.3 run replay | difference 0.0, `boundary_failed` as recorded |
| sealed v9.4 run replay | difference 0.0, `boundary_failed` as recorded |
| test suite | 341 passed |
| paper-claim checker | 27 quotations verified, 5 absence claims hold |
| paper's data-source sentence | [line 155] "These data are sourced from the Bloomberg Terminal" — equities Bloomberg, risk-free GFD, exactly as our provenance doc assumed |

## B. Two errors of mine found by this audit, both in yesterday's episode probe

**1. "27.95% — inside the printed precision" was false.** The printed half-digit
band around Figure 6's 27.8% is [27.75, 27.85]; ours is 27.95%, which is 0.15pp
outside. Close, not printed-equal. The docstring said otherwise and has been
corrected in place, with the correction noted rather than silently rewritten.

**2. "Signature of a noisier underlying series" was premature.** Dating the 16
short bear episodes kills the tidy version of that story: only 4 fall before
2000, when the rolling fit window still contains our reconstructed pre-1988
data; 12 fall after, when every window holds only the official ^SP500TR series.
The sharper attribution:

```
7 of the 11 post-2000 flickers occur in months where the CV selected k=0
k=0 is live on  9.5% of days   but carries  23.0% of all US turnover
counterfactual: k=0 days at the average intensity of the rest
                1.795 -> 1.528       (Shu: 1.41)
grid probe's no-zero candidate set, computed independently yesterday: 1.530
```

The two counterfactuals agree to 0.002, which is the kind of agreement that
makes an attribution trustworthy. (Honesty note: the 9%-of-months / 23%-of-
turnover figure itself is not new — the knob ledger recorded it on 2026-07-28.
What is new today is the recomputation on the sealed v9.4 run landing on the
same numbers, the episode-level link to Shu's own Figure 6 path, and the timing
evidence that the flickers post-date the reconstructed data.) So the proximate amplifier of the US excess is
**months in which the selection chooses no smoothing at all**, not raw data
noise. Two things follow, and neither is a licence to change the grid:

- there is a defensible *a-priori textual* argument that k=0 is not a candidate
  for a *filter window* — [line 387] "we apply a median filter" — with k=0 meaning
  not applying one — and the old grid probe already recorded that reading as its
  `filtered` set *before* today's flicker evidence existed;
- but even under that reading the US lands at 1.53 against 1.41, still outside
  tolerance, and Germany still demands the opposite end of the grid. The
  contradictory-constraints conclusion (§15 of the verdicts doc) is unchanged.
  Dropping k=0 *because it helps* would be fitting; we do not adopt it.

## C. What the deep research settles (85 verified claims, the ones that matter)

**C1. The candidate grids are unpublished anywhere, now checked at the source.**
The author's `jumpmodels` package (PyPI 0.1.1, the only release, 2024-10-04)
contains no CV code, no hyperparameter grid, no HMM, no median filter — only the
three JM estimators. The GitHub repo's sole example is a Nasdaq-100 demo on
Yahoo data with hard-coded λ = 50 (and 600 for the continuous variant); the repo
has no replication artifact for the JAM paper. The JAM paper itself publishes no
grid. Verdict: our "the k grid is our construction" stance is now backed by an
exhaustive primary-source search, not just by reading the paper.

**C2. The paper's own history shows the grid changed and was then withheld.**
arXiv v1 of 2402.05272 used different data (SPX/INDU/NDX from 1960, no DAX or
Nikkei), published an explicit λ grid {10, 22, 50, 100, 220, 500, 1000}, chose
on a single split, and footnoted a *rejection* of periodic re-selection. v3
switched to monthly CV, added DAX/Nikkei/1970, and published no grid at all. The
companion paper (arXiv 2406.09578) gives only endpoints — λ from 0.0 to 100.0,
log-spaced, count unstated — and re-selects biannually over 5 years, a different
protocol again. No version of the lineage publishes the v3 candidate sets.

**C3. Our median-filter implementation is confirmed against Bulla.** Bulla et
al. (2011) define the filter on discrete predicted state labels as
floor(median(last k)), k = 6 *trading days*, and with even k a 3-3 tie floors to
the low-variance state. Our rolling mean with strict `> 0.5` resolves ties the
same direction (tie → invested), matching both Bulla's floor and Shu's word
"exceeds". Bulla's inputs are k separate one-step-ahead predictions from rolling
refits — our daily-refit terminal-state design, not the tail of one decoded
path. Bulla published no k grid and explicitly declined to optimise k. Nothing
to change; one prior worry (odd/even median conventions) is retired.

**C4. Japan: even the paper's own series must be a construction before 1980.**
The official Nikkei 225 TR index (Bloomberg NKYTR) was first published
2012-12-03 and back-calculated only to 1979-12-28 (base 6569.47); Nikkei sells
the full daily history and gives away only 3 years. The paper starts Japanese
data in 1970 from Bloomberg — so whatever Shu et al. used for 1970-1979, it is
not the official TR index, because none exists. Two further vendor-specific
artifacts we cannot reproduce: estimated dividends on ex-date trued up the next
business day, and a gross-vs-net fork (NKYTR vs NKYNTR, ~40-52bp/yr). Our
reconstruction caveat was, if anything, understated: the target itself is a
construction there.

**C5. GFD could not have been the daily equity source.** GFD's own index guide:
country-level total-return families are monthly frequency and USD-denominated.
Consistent with the paper's sentence (equities Bloomberg, GFD only for T-bills).
Our provenance doc already had this split right.

**C6. External echo of the upper-endpoint concentration.** Li, Chen, Tao & Ji
(2025), citing the paper, select λ from {0, 5, 10, 25, 50, 100} by information
criteria and land on the **upper endpoint for all 12 assets**. Same pathology we
measured (§11/§14), different selection rule, different data — evidence that
top-of-grid concentration is a property of this model family's selection
problem, not of our implementation. (Their table carries oddly patterned digits,
flagged by the verifier; treat as an echo, not a benchmark.)

**C7. For the JM later (out of scope now, logged so it is not lost):** the
author's example pipeline hard-codes halflives [5, 20, 60], downside deviation
with negatives floored at zero (not dropped), clip at 3σ then standardise, and a
single global RANDOM_STATE = 0. Per CLAUDE.md, this is the author's example, not
the paper; it may inform JM-side hypotheses but must never be cited as "the
paper does X".

## D. Codex's review, re-scored after everything

All ten findings were confirmed on day one and nothing since has weakened any of
them. Where the day's own work stands after this audit:

| item | status after self-audit |
|---|---|
| S&P splice fix (v9.3/v9.4) | holds; two independent full runs agree to 0.00e+00 on six metrics |
| MDD basis retraction | holds; costs nothing at tolerance in either sealed run |
| selection-noise correction (§5) | holds; baseline 20-23% unchanged |
| boundary gate demotion (§14) | holds; now with the Li et al. echo for its descriptive half |
| grid-ceiling refutation (§13) | holds |
| contradictory-grids result (§15) | holds; strengthened by C1/C2 (no grid published anywhere) |
| episode-shape probe | arithmetic holds; **interpretation corrected** (B2), precision claim corrected (B1) |

## E. Honest bottom line

The replication is as closed as public data allows: 7/8 cells in all three
markets at delay 1 under the honest drawdown basis, on sealed, replayable runs.
The one open cell now has a mechanism (k=0 months amplify state noise into
trades), a proof that no grid choice closes it everywhere, an external echo of
its central pathology, and a data-side residual (Bloomberg vintages, Nikkei's
unpublishable pre-1980 series) that is not closable from here. The remaining
work on it is wording, not computation.
