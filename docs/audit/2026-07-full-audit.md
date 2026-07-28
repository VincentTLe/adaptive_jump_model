# Full audit — code, data, methodology vs Shu (2026-07-25/26)

Scope: every result-critical module, every data series, and a fresh close-read of
arXiv 2402.05272v3. Findings carry severity (blocker / major / minor / note),
reproducible evidence, and an affects-sealed-numbers flag. Fixes that would
change sealed numbers are NOT applied inside the audit; they stop and await
approval.

Status: IN PROGRESS. Part B (data) largely complete; Part A (code) and Part C
(paper) running as a multi-agent review with adversarial verification.

---

## Part B — Data audit

### B1. Coverage matrix (canonical, shu-replication-expanding-v8-2)

| series | rows | span | calendar gaps > 7d |
|---|---|---|---|
| us_equity | 14,850 | 1965-01-04..2023-12-29 | 0 (max 7d) |
| us_cash (DTB3, daily) | 14,737 | 1965-01-04..2023-12-29 | 0 |
| de_equity | 14,852 | 1965-01-04..2023-12-29 | 0 (max 6d) |
| de_cash (monthly ladder) | 708 | 1965-01..2023-12 | monthly by construction |
| jp_equity | 14,507 | 1965-01-05..2023-12-29 | 3 (New-Year closures 1985/86, Golden Week 2019 — benign) |
| jp_cash (monthly ladder) | 700 | 1965-01..2023-12 | 7 single/double-month source holes, see B1-F1 |

Coverage verdict: every phase requirement is met — warm-up 1965–70, training
1970–90, OOS 1990–2023, delay robustness. Per-market OOS starts recomputed from
the real series: US 1990-01-02, DE 1990-01-02, JP 1990-01-04.

**B1-F1 (note, cleared).** jp_cash source holes: 1985-06, 1986-06, 2018-08,
2019-02/03, 2019-06, 2019-12, 2020-12 (IMF and BoJ months genuinely absent at
source). Verified against the sealed JP frame: the 120-day staleness window
absorbs every hole — 0 missing feature rows after warm-up; the only NaN-cash
days are the 35 warm-up head days of 1965 (DE: 39; US: 0). No fix needed;
documented as a data limitation.

### B2. Integrity & provenance

- sha256 of all five `data/external/*.csv` recomputed → **all match** the pins
  in research-expanding-v8-2.toml.
- **B2-F1 (major, provenance — fix planned).** `data/external/inputs/*` (10
  files) are pinned NOWHERE. Hashes recorded now for the ledger:

| input | sha256 (first 20) | bytes |
|---|---|---|
| ff_us_daily.csv (Kenneth French, downloaded 2026-07-25) | f051e37d30c129359c68 | 1,208,053 |
| stooq_dax_daily.csv (Stooq ^dax, manual download by owner 2026-07-25) | a0ea6e8edcae145d00d0 | 752,609 |
| n225_price_daily.csv (Yahoo ^N225 via yfinance 2026-07-25) | 93c5b6fcbc0ebfc2487e | 385,792 |
| n225tr_official_merged.csv (Investing.com N225TR mirror, 2 manual downloads by owner, merged) | 1c2977502af445abeccb | 61,629 |
| jst_japan_eq.csv (JST Macrohistory R6 slice, 1964–2020) | 519c9c856772aeed7354 | 4,919 |
| INTGSTJPM193N.csv (FRED/IMF Japan T-bill) | f068a59109a996b265b6 | 17,220 |
| INTGSTDEM193N.csv (FRED/IMF Germany T-bill) | 8693c22c905abe575a0d | 6,600 |
| IR3TIB01DEM156N.csv (FRED/OECD DE interbank 3M) | 2b9d26f8dfc56280b1c1 | 24,089 |
| ecb_3m_aaa.csv (ECB SDW yield-curve 3M AAA) | 4b96c8b50d4ccad6bc88 | 3,130,120 |
| jp_equity_tr_full.csv (intermediate; superseded by in-builder construction) | f03965755b8789c3b878 | 366,458 |

  Planned fix (Part D): builder verifies input hashes against a pinned manifest;
  this table becomes the provenance ledger of record.
- **B2-F2 (minor — fix planned).** Cross-generation dependency: the builder
  reads `data/processed/shu-proxy-replication-v6-.../jp_cash.csv` (an OLD run's
  processed output) as the BoJ-call input for the JP cash ladder tail. Promote
  that series to `data/external/inputs/` with a pinned hash.

### B5. Table 1 anchor (new, never used before this audit)

Daily excess-return statistics 1970–2023 computed from our canonical series
(12,588 common days) vs the paper:

| pair | ours | paper |
|---|---|---|
| corr US–DE | 0.462 | 0.44 |
| corr US–JP | 0.138 | 0.12 |
| corr DE–JP | 0.259 | (Table 1 — pending exact transcription) |
| ann. variance US / DE / JP | 0.0286 / 0.0400 / 0.0425 | (pending) |

Joint validation of all three series at DAILY granularity within +0.02 of the
published values. Full-matrix comparison lands when the Table 1 transcription
from the close-read completes.

### B4. "What is still missing" register (answering the owner's question)

1. JP total return before 2012: reconstruction (N225 price + JST annual
   yields), no free official series exists. Method validated on the 2012–2023
   overlap (daily corr 0.9977; implied yields within 0.3pp of JST). Claims
   standing on it: JP rows of the replication tables. Caveat required in the
   paper: yes (drafted).
2. JP TR mirror hole 2020-07..2022-05: bridged with both official edges matched
   exactly (endpoint error 2e-16). Irreducible without paid Nikkei data.
3. DE equity before 1988: Stehle backcast lineage (single lineage worldwide);
   validated monthly vs OECD MEI (corr 0.9846 in the 1970s ≈ modern control
   0.9895). No DAILY independent check exists publicly — irreducible.
4. DE cash 1965–1975-06: interbank, not T-bill (T-bill series starts 1975-07).
   Touches warm-up/training only.
5. JP cash 2017-07..2023: BoJ call splice (NIRP era, joint delta ≈ 0).
6. Vendor close-time microstructure (Bloomberg 17:30 vs public fixings):
   irreducible; bounded by the delay-10 result (all Table-5 cells matched or
   exceeded at two-week delay).
7. jp_cash source holes (B1-F1): absorbed by staleness policy.

Conclusion: data is SUFFICIENT for every claim in the current paper plan; the
irreducible gaps are documented limitations, not blockers.

---

## Part A — Code audit (complete)

Six result-critical modules reviewed against a ~130-surface checklist; every
blocker/major candidate independently verified. **Zero blockers. Zero findings
that affect sealed numbers.** Confirmed findings and dispositions:

| id | sev | finding | disposition |
|---|---|---|---|
| 6.1 | major | `_verify_manifest` set-comparison cannot detect a duplicated (market,kind,source_id) manifest entry | **FIXED**: source-count check added (cli.py) + regression test |
| CFG-WS40 | minor | 40-char substantive-documentation gates accepted 40 spaces | **FIXED**: `.strip()` before length (config.py) + test |
| 5.20 | minor | dirty-tree provenance scope omitted `research-expanding-v8*.toml` and `scripts/` | **FIXED**: scope widened (data.py) |
| 3.2-latent | minor | `apply_signal` label-aligned Series inputs silently misalign (bit twice in scratchpad) | **FIXED**: positional conversion via `np.asarray` + property test |
| 3.7 | minor | `charge_initial_allocation` inert on the replication path | **FIXED**: threaded through walkforward (behavior unchanged; config value false) |
| 2.25 | minor | `_IdentityScaler` writes mean=0/scale=1 into jm-refits audit trail | documented (sealed records left as-is; values are truthful for the in-window transform) |
| B2-F1/F2 | major/minor | builder inputs unpinned + cross-generation dependency on an old run's output | **FIXED**: `INPUT_SHA256` pins (11 inputs) + `verify_inputs()` gate; BoJ call series promoted into inputs; rebuilt outputs byte-identical (5/5 hashes unchanged) |
| WF-1..6, 1.9, 1.11, 1.14, 3.15, 3.19, 2.18/19, 2.21/22, ART-*, CFG-*, CLI-* | minor/note | selection-calendar min_periods dependence, tie frequency, staleness reference point, EWM warm-up noise, CAGR observation-count convention (~6bp JP), ES interpolation, symmetric HMM convergence monitor, inventory scope boundary, etc. | documented; none affects sealed numbers; candidates for future hardening |

Cleared as CORRECT after adversarial checks (highlights): t+2 delay semantics
exactly per paper §3.1; all in-repo `apply_signal` callers positionally safe;
Sharpe/Calmar/turnover/leverage definitions conform to the Table 4 caption
word-for-word; B&H benchmark sample fairness (identical metric dates across
models); decision-timing equivalent to the paper's t/t+2 convention; refit
calendar = first trading days of Jan/Jul + one bootstrap fit (JP: 1978-05-26);
`_IdentityScaler x observation_loss_scale` exact (sqrt(3) verified); builder
constructions reproduce independently (US TR max rel diff 3e-16; JP stitch
points continuous, max |log-ret| 0.022).

## Part A7 — Independent recomputation of every reported headline number

Five fresh implementations (agents barred from reading the original analysis
code) reproduced from sealed artifacts:

- 9-cell Sharpe table + bootstrap CIs: **all 19 rows match** (points to 0.005,
  CI endpoints within RNG tolerance 0.02); "all 9 Shu values inside our 95%
  CIs" re-confirmed 9/9.
- Delay 1/5/10 table (18 selected cells + 6 fixed-candidate cells): **all
  match** to 3-4 decimals; DE fixed_jm delay-10 = 0.2937 (rounds to Shu's 0.29).
- Bear-share-by-era (15 rows): **all match**; the Figure-5 "shifted forward by
  2 days" convention moves every era share by at most 0.07pp — immaterial.
- Follow-CV US: composed monthly picks agree with the sealed pipeline **408/408
  months**; composed daily signal identical on all 8,565 OOS days; Sharpe
  0.7848/0.7877 vs reported 0.79; turnover 32.37% vs 32.4%; 22 shifts exact.
- Fixed-lambda sweeps: JP Sharpes reproduce to 0.0002.

**Corrections recorded (reported numbers whose provenance needed tightening):**

1. *Table-3 shift levels.* The sealed v8.2 US states give 3.67/1.10/0.76/0.43/
   0.33/0.19 shifts/yr for lambda 0..150 — FEWER than both Shu's Table 3
   (9.7..0.4) and the session's probe variant (which used a different warm-up).
   The expanding-standardizer identification rests on Table-4 economics,
   Figure-5 profiles and the 2020-exit signature — NOT on matching Table 3's
   absolute levels, and the ledger now says so explicitly.
2. *JP HMM small-k cell.* The reported "0.18 vs Shu 0.19" was computed on
   v8.1 artifacts (OOS 1991-03). On v8.2 (full window, OOS 1990-01) the same
   spec gives ~0.15 — the honest statement is "small-k range closes the JP HMM
   gap from 0.16 (wide grid) to ~0.04, not to 0.01". The small>>wide grid
   ranking conclusion is unchanged (verified on both runs).
3. *Run mix.* Reported US/DE numbers come from the v8.1 sealed run, JP from
   v8.2 (deliberate — JP window fix — now explicit).
4. *US k-range identification Sharpes.* A second, fully independent recompute
   of the k-grid table (2026-07-26, fresh implementation validated to 1e-16
   against the sealed candidate-returns accounting) reproduces DE exactly
   (0.25/0.25/0.14) and US turnovers exactly, but puts the US Sharpes at
   0.55/0.56/0.55/0.56 for grids A/B/C/wide — about 0.02 below the previously
   reported 0.57/0.58/0.58. Grid B (single candidate k=6, no selection step)
   isolates the discrepancy to the earlier report's US scoring, not to
   selection logic. The ranking conclusion (small-k >> wide on DE/JP, US flat
   across grids and near Shu's 0.54/141 with grid A at 0.55/135) is unchanged
   and re-confirmed on all three markets.

## Part C — Paper close-read (complete)

129 claims transcribed into the register (51 from title/§1/§2 + Table 1, 48
from §3 + Tables 2-3 + Figures 2-4 + footnotes 10-14, 30 from §4-5 + Tables
4-5 + Figures 5-6). Notable confirmations against our implementation: the
paper has NO footnotes after §3.4.3 and no appendix; the delay wording, the
l=3000 lookback, daily HMM refit, and Table 4/5 metric captions all map to
audited code surfaces (see Part A cleared list). Table 1 anchor comparison in
B5. The register lives in the audit workflow transcripts
(`subagents/workflows/wf_6e931ef8-0eb/journal.jsonl`).

## Part D — Fixes applied (all non-numeric; sealed replay re-verified)

- cli.py: manifest duplicate/missing source detection.
- config.py: whitespace-proof substantive-documentation gates.
- data.py: provenance dirty-tree scope covers all root research TOMLs + scripts.
- backtest.py: positional signal conversion in `apply_signal`.
- walkforward.py: `charge_initial_allocation` threaded end to end.
- scripts/build_external_sources.py: 11 input sha256 pins + `verify_inputs()`
  gate; cross-generation dependency removed (BoJ series promoted to inputs);
  rebuilt outputs byte-identical to the config pins (5/5).
- tests/test_audit_hardening.py: 7 new regression tests (manifest dup,
  whitespace gates, localfile sha/path gates, positional apply_signal,
  prepare_market expanding branch, v8-2 config anchor).
- Full suite after changes: 444 passed; the single failure is the pre-existing
  monitor-environment test (fails on a clean tree identically). Sealed runs
  re-verified clean via `verify_run`.

## Addendum (2026-07-26) — agent re-verification round

The four audit pieces originally completed solo (after the spend-limit
interruption) were re-run by independent agents once quota returned.

**models.py full review (agent, independent):** zero blockers, zero findings
affecting sealed numbers. Three minors, all per-spec but now documented:
(1) *median-tie direction* — `mean > 0.5` sends exact ties to risky; ties are
frequent with the even-k grid (55-180 OOS days per market/delay would flip
under `>=`); frozen and deterministic, flagged for a sensitivity run.
(2) *min_periods=1 sensitivity* — no OOS-governing decision ever had
under-windowed days for its SELECTED candidate; a full-window counterfactual
flips selections only at us 1989-12 (return-neutral), de 1994 d10
(return-neutral) and jp 1995-05 d10 (22 differing days, +12.3pp cumulative) —
the JP delay-10 robustness cell is the one spec-sensitive output. The rule is
also load-bearing: requiring full windows would delay first selection to
~1995. (3) *HMM resume validation* weaker than the JM path — fixed in
f727da7 (+ regression test); not triggered in sealed runs (verified clean).
Notable cleared items: online inference provably equals the paper's l=3000
prefix-DP; variance relabeling never ambiguous (min hi/lo ratio 3.3, zero
plausible label swaps); the symmetric convergence monitor is *stricter* than
stock hmmlearn (stock would accept stop-on-decrease fits; ours vetoes them,
and accepted terminals are stable at tol=1e-9).

**k-grid identification recompute (agent, fresh implementation):** correction
4 above; ranking conclusion re-confirmed on all three markets.

**Residual-cell investigation (completed 2026-07-26):**

*DE HMM — RESOLVED (two stacked causes).* (1) hmmlearn's `covars_prior=0.01`
materially distorts fits on decimal returns (low-state variance +7-40%,
high-state up to +97% in early windows; probe reproduces sealed fits exactly
13/13 and shows the `covars_prior=0` and percent-scaled variants agree to 5
decimals, isolating the prior — `min_covar` inert). (2) The non-paper wide
smoothing grid (k to 2560) wrecks the selection layer; the paper's own grid is
{0,2,4,8,20} (Table 3). Full-path refit under percent scaling + small grid:
**DE follow-CV 0.353 / turnover 214% vs Shu 0.35 / 246%**, with Shu's
fast-decaying delay shape (0.35→0.28→0.18). Cross-checked by the independent
fixed-k frontier (k=4: 0.327/202%/76.9% lev). US survives the same fix
(small-grid 0.59 vs Shu 0.54, usual overshoot zone); JP improves slightly
(0.166, turnover 314% vs Shu 290%). Sealed states differ on 5.5% of DE days;
state changes 316→350.

*JP JM — mechanism isolated, Shu bracketed, exact recipe unidentified.* New
Table-3 anchor: our sealed JM paths flip 2.1-2.6x LESS than Shu's published
shifts/yr at every lambda INCLUDING lambda=0 (penalty-irrelevant), while our
HMM k-grid flip rates match Shu — localizing the discrepancy to JM feature
standardization geometry (one-shot expanding windows enter fits off-center
+0.68 sigma and std-compressed 0.73-0.82). The selection layer is exonerated
(agent replica = sealed choices 100% in all three markets). Intervention arms
re-running the author library's own per-refit-window recipe (clip +-3sigma +
StandardScaler frozen at refit): flip rates move to ~1.3x Shu (from 0.4x) —
crossing Shu, so the true recipe lies inside this family; JP fixed
lambda=35 hits **0.310 / 68% turnover / 45 shifts vs Shu 0.31 / 72% / 48
shifts** (leverage still 43% vs 75%); JP selected 0.169→0.219; the arm exits
correctly during the 1990-94 Nikkei collapse (lambda=5 era Sharpe -0.51→
+0.03). But the same recipe DEGRADES the matched markets (US selected
0.788→0.460, DE 0.361→0.310) — no single preprocessing variant reproduces
Table 3 and Table 4 simultaneously; Shu's exact standardization remains the
one unpublished degree of freedom, with our variants bracketing their JP JM
row: 0.157/0.169 (expanding) — 0.219 clip / 0.310 fixed-λ35 — 0.260
cold-start-1970 — 0.263 anchor-1970 — vs Shu 0.31 (all inside the 95% CI).
Also eliminated: validation-metric raw-vs-excess (raw scores 0.148 vs 0.157,
choices identical post-2000); literal-1970 cold-start warm-up (+0.03 on the
matched window 0.260 vs 0.233; the sealed 0.157 vs 0.233 delta is the
Jan-Aug-1990 crash months).

## v8.3 outcome (2026-07-27, run fixed-baselines-353f037e328e-113bf877f679-3e77b13b7bc0)

The identified spec (expanding standardization anchored at 1970, min_obs 63;
HMM covars_prior=0 with the paper's own smoothing grid {0,2,4,8,20}) was
frozen (3e77b13) and run sealed end to end. Verify replays clean. Exploratory
delay-1 readout vs Shu Table 4, with 95% moving-block bootstrap CIs (block 21):

- us: JM 0.756 / 64% / 76.4% (Shu 0.68/44/80) · HMM 0.662 / 186% / 70.7%
  (0.54/141/72) · B&H 0.500 (0.48)
- de: JM 0.416 / 113% / 73.1% (Shu 0.44/170/84) · HMM 0.400 / 216% / 73.3%
  (0.35/246/73) · B&H 0.301 (0.30)
- jp: JM 0.260 / 159% / 64.5% (Shu 0.31/72/75) · HMM 0.226 / 325% / 68.2%
  (0.19/290/74) · B&H 0.189 (0.12; OOS starts 1990-08-31 so the Jan-Aug 1990
  crash months fall outside the window — flatters all JP cells vs Shu's
  1990-01 anchor; relative ordering unaffected)

**JM > HMM > B&H reproduces on all three markets — first time in the
project.** Shu's value sits inside the 95% CI in 9/9 cells. The JP fixed-JM
lambda-150 boundary concentration is cured (3.7% of months vs 32% under
v8.2); 8/9 fixed-JM boundary gates pass (DE d10 marginal at 5.9%). All
remaining gate failures are HMM k=20 upper-boundary concentration — the CV
parks at the top of the paper's own grid (us d10 39%, jp d1 39%) — recorded
as a finding about the paper's grid design; run status is therefore
boundary_failed and official metrics remain sealed per protocol (readout
above is exploratory, computed from the sealed selected-signal files).

## Verdict

The audit found NO defect that changes any sealed or reported number. All
material findings are provenance/hardening issues, now fixed with tests, plus
three provenance corrections to previously reported numbers (recorded above).
Data coverage is sufficient for every claim in the paper plan; irreducible
gaps are documented limitations.

## Correction: the 95% CI was used as evidence when it carries none (2026-07-27)

Two statements above lean on a bootstrap confidence interval as though covering
Shu's value supported replication: line 265 ("all inside the 95% CI") and line
288 ("Shu's value sits inside the 95% CI in 9/9 cells"). Both are arithmetically
true and evidentially empty. Retract the inference, keep the numbers.

The standard error of a Sharpe ratio over T years is about
`sqrt((1 + SR^2/2)/T)`, so at the paper's 34-year window any 95% interval is
roughly 0.70 wide; our measured widths were 0.646-0.681, exactly as expected.
An independent recomputation (block bootstrap, block in {5,21,63}, seeds 1-5,
2000 resamples) confirmed the coverage claim and then showed why it is vacuous:
every cell interval also contains the paper's values for the other two models in
that market, so the buy-and-hold interval cannot reject "buy-and-hold equals the
jump model". Reaching a 0.05-wide interval would need on the order of 6,900
years of data.

There is also no valid significance test to be had. The paper gives one value
per cell with no standard error, on licensed data we do not hold; there is no
sampling distribution to test against.

The same recomputation established what the intervals do say, which is less
comfortable than what was claimed: JM minus HMM contains zero in both markets
tested (us +0.099, CI [-0.073, +0.278], p2 = 0.270; de +0.018, CI [-0.178,
+0.221], p2 = 0.859), zero inside the interval in 15/15 bootstrap
configurations. JM minus buy-and-hold clears zero only for the US and only
marginally (p2 = 0.036, unstable across seeds). So "JM > HMM > B&H reproduces on
all three markets" is a statement about the ordering of point estimates and must
be written that way, never as a significant result.

Replication should therefore be judged by point-estimate closeness against a
tolerance fixed in advance (the owner's standing tolerance: 0.05 absolute
Sharpe, tightening to 0.03), together with the spread produced by the free
parameters the paper never fixes. Those parameters and their measured spreads
are now registered in docs/unspecified-choices.md.

## Correction to the correction, and the v8.4 tally (2026-07-27, later)

The retraction above is right that interval coverage was never evidence, but the
arithmetic it published to prove the point is itself wrong and is withdrawn. The
Lo (2002) standard error requires the Sharpe ratio and the sample size to be at
the same frequency; the table mixed an annualised Sharpe with a sample counted
in years. Corrected, the width is near 0.672 for every cell rather than varying
0.679 to 0.750 with the Sharpe, and the "6,900 years" figure should read roughly
6,150. The reported measured widths of 0.646 to 0.681 could not be traced to any
script or artifact in the repository; an independent recomputation gave 0.642 to
0.713 on v8.3, so the upper bound was wrong as well.

Per the owner's instruction, this project no longer reports confidence
intervals. The reasoning is in docs/unspecified-choices.md. Replication is
judged by point-estimate closeness against a tolerance fixed in advance, across
all eight rows of Table 4 rather than the Sharpe row alone.

The v8.4 tally, stated the way it should have been stated the first time
(delay-1, metrics-exploratory.csv, run 93de627bb755-d65262092c1a-17e1984c1817):

| tolerance | all nine cells | model cells only | model cells also passing their delay-1 boundary gate |
|---|---|---|---|
| 0.05 | 7/9 | 4/6 | **1/6** |
| 0.03 | 5/9 | 2/6 | 1/6 |

Three of the seven passing cells are buy-and-hold, which contains no model and
therefore measures the data rather than the replication. Of the four model cells
inside 0.05, three sit on cells whose own boundary gate failed (de fixed_jm
0.0515, de hmm 0.0637, jp hmm 0.3946). Only us fixed_jm both meets the tolerance
and clears its gate. Reporting "7/9" without that decomposition repeated the
inflation this audit was convened to stop.

Turnover, which the paper treats as the identifying property of the jump model,
does not reproduce: us 103.0% against 44%, jp 140.6% against 72%, de 105.5%
against 170%. The us jump-model Sharpe matches to 0.018 while trading 2.3 times
as much, so that cell agrees on the headline number and disagrees on the
behaviour underneath it.

## HMM anatomy on v8.4, and predictions for v8.5 (2026-07-27)

Written before the v8.5 run finished, so the predictions below cannot be
rewritten to fit whatever it produces.

### The HMM matches the paper at the model layer

Reported turnover is `0.5 * sum|dposition|` annualised (`backtest.py:124`), and
a 0/1 strategy moves by exactly 1 at each shift, so turnover is half the shifts
per year. That makes Table 4's turnover row directly comparable with Table 3's
shifts-per-year row, which is the only place the paper publishes how its model
responds to the smoothing window.

Fixed-k shifts per year on the US series, 1982-2023, against Table 3 (line 644):

| k | ours | Shu | ratio |
|---|---|---|---|
| 0 | 9.62 | 8.5 | 1.13x |
| 2 | 7.29 | 6.6 | 1.10x |
| 4 | 5.05 | 4.9 | 1.03x |
| 8 | 3.24 | 3.2 | 1.01x |
| 20 | 2.14 | 2.0 | 1.07x |

With the selector switched off, our smoothed state sequence flips at Shu's rate
across the whole grid. The selected paths agree too: turnover 184.2% against
141% (us), 222.7% against 246% (de), 316.7% against 290% (jp) — 1.31x, 0.91x,
1.09x, scattered around one rather than biased.

This is the opposite of the jump model, whose fixed-lambda flip rates sit at
0.43-0.93x of Table 3 and whose selected turnover misses by 2.3x. Whatever
drives the JM deviation, it is not shared by the HMM, even though both models
consume the same three features from the same frames.

An arithmetic correction, caught here rather than in a report: turnover is half
the shifts, so shifts are twice the turnover. Inverting that once turned a 1.31x
gap into a printed "5.23x" before it was checked against Table 3.

### The deviation that remains is the width of the k grid

Out-of-sample Sharpe with k held fixed for the whole period, no selection:

| market | k=0 | k=2 | k=4 | k=8 | k=20 | selected | Shu |
|---|---|---|---|---|---|---|---|
| us | 0.594 | 0.524 | 0.580 | **0.676** | 0.595 | 0.638 | 0.54 |
| de | 0.406 | 0.388 | **0.462** | 0.384 | 0.268 | 0.393 | 0.35 |
| jp | 0.095 | 0.192 | 0.202 | 0.197 | **0.213** | 0.182 | 0.19 |

Shu's published HMM Sharpe falls inside our own fixed-k range on all three
markets. The grid alone spans 0.15 (us), 0.19 (de) and 0.12 (jp) — every one of
those wider than the deviation being investigated. So the HMM deviation is not
evidence of a defect; it is the size of an unspecified knob, and the paper never
publishes the candidate set (see docs/unspecified-choices.md #3).

Two further facts about the selector, both from sealed artifacts:

- The selected path scores **below** the best fixed k in all three markets
  (0.638 vs 0.676, 0.393 vs 0.462, 0.182 vs 0.213). Re-picking k monthly costs
  more than it earns against simply holding the ex-post best k.
- The monthly decision is often close: the winner beats the runner-up by less
  than 0.02 Sharpe in 16.5% (us), 27.2% (de) and 19.6% (jp) of months.

The modal pick coincides with the ex-post best fixed k in all three markets
(87.6%, 52.2%, 40.3%, against 20% for a uniform pick). Three markets is far too
few to call that skill rather than coincidence, and it is recorded as an
observation, not a claim.

### Predictions for v8.5

v8.5 adds k=6 and changes nothing else (`tests/test_audit_hardening.py`
asserts the two contracts differ in that field alone; the acquisition manifests
are byte-identical on all six canonical series).

1. **The boundary gate will still fail on the same cells.** k=6 is an interior
   value and cannot relieve concentration at the top of the grid. Expect jp
   delay-1 (39.5% at k=20) and us delay-10 (39.0%) to fail again.
2. **The us HMM Sharpe will fall.** k=6 sits between k=4 (0.580) and k=8
   (0.676); cross-validation currently parks on k=8 in 87.6% of months, so any
   months that move to 6 pull the result toward the middle — and toward Shu's
   0.54, which is below all of it.
3. **de will fall to roughly 0.354**, from an earlier unsealed probe that put
   the deviation at +0.004 against Shu's 0.35.
4. **jp will barely move.** Its picks concentrate at k=20 (40.3%) and k=4
   (29.0%); k=6 lands between two cells that score 0.202 and 0.197.

If (2) and (3) hold, they are not a success. The grid was corrected because the
paper names k=6 at line 390, and the direction of the effect was predicted from
the fixed-k table above rather than discovered by trying it. A change that
improved agreement for any other reason would have to be reported as tuning.

## The sample-start choice: what it does, and a claim withdrawn (2026-07-27)

### Withdrawn: "the paper's 1990-01-02 anchor"

The v8.4 and v8.5 contracts justify `requested_sample_start = 1969-05-01` by
saying it puts every market "back on the paper's anchor", naming 1990-01-02.
**The paper gives no such date.** It says:

> [line 157] "All data spans from the start of 1970 to the end of 2023."
> [line 713-715] "Since our data begin in 1970, with training windows spanning 12 years and validation windows 8 years, the out-of-sample testing period begins in 1990."

and Table 4 reports "from 1990 to 2023". Every mention of 1990 in the paper is a
year, never a date. So 1969-05-01 does not restore a paper anchor; it **breaks a
specified item** (the 1970 data start) to reach a date we invented.

Both config comments are wrong on this point. They are left unedited because
both have been run and their text is part of what was frozen; the correction
lives here and in docs/unspecified-choices.md.

### Why the literal 1970 start does not reach 1990

Sessions per year, 1970-1989, and the 3000th session counted from 1970-01-01:

| market | sessions/yr | Saturdays | 3000th session | + 8 years |
|---|---|---|---|---|
| us | 252.7 | 0 | 1981-11-13 | 1989-11-13 |
| de | 250.2 | 0 | 1981-12-29 | 1989-12-29 |
| jp | 246.8 | 0 | 1982-03-08 | 1990-03-08 |

The US and Germany clear the eight-year mark before 1990 on raw sessions, so
their out-of-sample start falls back to the requested 1990-01-01; feature warm-up
then pushes the realised starts to 1990-03-15 and 1990-06-19. Japan does not
clear it at all, because the Tokyo exchange traded Saturdays until January 1989
and our `^N225` series contains **zero** Saturday sessions. Missing roughly 25
sessions a year for nineteen years makes our 3000-session window span about
eighteen months more calendar time than Shu's.

So the backdated start is **compensation for a data gap**, not fidelity to the
paper. It should be described that way wherever it is described at all.

### What the choice actually changes

v8.3 and v8.4 differ in this one field, so the two sealed runs isolate it.
Comparing raw states on the overlapping days:

| market | HMM state days compared | HMM states differing | JM state cells differing |
|---|---|---|---|
| us | 10,619 | **0** | 3.71% |
| de | 10,605 | **0** | 1.10% |
| jp | 10,282 | **0** | 15.84% |

The HMM is **exactly invariant**. It fits the last 3000 log returns before day t,
and that set does not depend on where the series began once 3000 observations
precede t. For the HMM the sample start is purely a reporting-window choice.

The jump model is not invariant, because its features pass through an expanding
full-history standardiser anchored at the sample start (unspecified-choices.md
#1). Moving the anchor moves every feature value forever.

### The decision

Keep 1969-05-01. Buy-and-hold contains no model, so it tests the window against
the paper's own published benchmark rather than any tuned knob:

| B&H Sharpe | 1970-01-01 | 1969-05-01 | Shu |
|---|---|---|---|
| us | 0.497 | 0.486 | 0.48 |
| de | 0.305 | 0.298 | 0.30 |
| jp | 0.193 | **0.138** | 0.12 |

The backdated window reproduces the period Shu report; the literal one does not,
decisively so in Japan, where it discards the first nine months of 1990 and with
them the opening of the Nikkei collapse.

Recorded as a **deviation from line 157**, not as replication. And it is not free
for the jump model: on the shared window it costs the US JM 0.088 Sharpe while
moving its turnover from 0.636 to 1.006 against Shu's 0.44 — the wrong direction
on the row the paper treats as the jump model's identifying property. That
trade-off must be reopened when the jump model is next frozen, and this entry
exists so it is not silently inherited.

### Framing correction from the owner

The purpose of this replication is to extend the **jump model**. The HMM and
buy-and-hold are comparison baselines, as is the paper's own JM. The target is
therefore agreement with Table 4, not good performance: a cell that beats the
paper is as wrong as one that trails it, and several of ours currently beat it
(us HMM +0.098, us MDD -19.7% against -28.9%, us Calmar 0.355 against 0.21).

## v8.5 sealed outcome, and the anchors we had never read (2026-07-28)

Run `fixed-baselines-7b95ec50dece-6bd27647967d-13641890668f`, 105 minutes,
status `boundary_failed`. One field changed from v8.4: the smoothing grid gains
k = 6. HMM readout, delay 1, against Table 4:

| metric | us v8.4 | us v8.5 | de v8.4 | de v8.5 | jp v8.4 | jp v8.5 |
|---|---|---|---|---|---|---|
| Sharpe | 0.098 | **0.074** | 0.043 | **0.018** | 0.008 | 0.013 |
| Return | 0.009 | 0.007 | 0.008 | 0.004 | 0.002 | 0.003 |
| Volatility | 0.003 | 0.002 | 0.000 | 0.000 | 0.000 | 0.000 |
| MDD | 0.092 | 0.092 | 0.030 | **0.003** | 0.031 | 0.031 |
| Calmar | 0.145 | 0.134 | 0.027 | **0.008** | 0.004 | 0.005 |
| ES 5% | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Turnover | 0.444 | 0.473 | 0.292 | **0.234** | 0.395 | **0.244** |
| Leverage | 0.011 | 0.011 | 0.000 | 0.000 | 0.000 | 0.000 |

(cells are absolute deviations). Within 0.05: us 4/8, de 7/8, jp 7/8. Within
0.03: us 4/8, de 7/8, jp 6/8. Germany improves on seven metrics and degrades on
none. The US gains on Sharpe and loses on turnover. Japan trades Sharpe for
turnover.

### The predictions, scored

Logged before the run finished, four claims:

1. **The gates fail on the same cells.** Correct, and exactly: us delay-10
   38.97% and jp delay-1 39.46%, unchanged to the digit from v8.4. k = 6 is
   interior and cannot relieve concentration at the top of the grid.
2. **The us Sharpe falls.** Correct, 0.638 to 0.614.
3. **Germany lands near 0.354.** **Wrong.** It landed at 0.368. The 0.354 came
   from an unsealed probe and did not reproduce; direction right, magnitude out
   by 0.014.
4. **Japan barely moves.** Correct, 0.182 to 0.177.

The sealed numbers also reproduce an independent preview computed through
`select_monthly_candidate` on v8.4's raw states to four decimals (0.6139 /
0.3681 / 0.1771), confirming that the smoothing grid never enters the HMM fit.

### Turnover: CLOSED by the paper's own words

The Table 4 caption does not define turnover, and this project has carried
`0.5 * sum|d weight|` as an assumption ever since. It is not an assumption. One
page later the paper states the identity numerically:

> [line 781-783] "turnover of the JM-guided 0/1 strategy applied to the S&P 500 is as low as 44%, meaning that on average, the portfolio manager buys and sells 44% of total allocation (a combined 88% trading) each year"

44% one-way, 88% combined, denominator the whole allocation. That is our
implementation exactly (`backtest.py:194`). Every alternative convention is
dead, including the leverage-scaled family, which "of total allocation" rules
out independently.

### Four published anchors nobody had used

Figures 5 and 6 annotate four cells with a bear share and a **raw count of
regime shifts**:

> [line 903] "Percentage of Bear Regimes Online Inferred by HMMs for the S&P 500: 27.8%, Number of Regime Shifts: 96"
> [line 829] "Percentage of Bear Regimes Online Inferred by JMs for the S&P 500: 19.7%, Number of Regime Shifts: 30"
> [line 851] "Percentage of Bear Regimes Online Inferred by JMs for the DAX: 15.7%, Number of Regime Shifts: 116"
> [line 873] "Percentage of Bear Regimes Online Inferred by JMs for the Nikkei 225: 25.3%, Number of Regime Shifts: 48"

Converted through our sample lengths, these reproduce Table 4 independently:

| cell | shifts/yr | half that | Table 4 turnover | 1 - bear | Table 4 leverage |
|---|---|---|---|---|---|
| S&P HMM | 2.825 | 141.2% | 141% | 72.2% | 72% |
| S&P JM | 0.883 | 44.1% | 44% | 80.3% | 80% |
| DAX JM | 3.413 | 170.7% | 170% | 84.3% | 84% |
| N225 JM | 1.414 | 70.7% | 72% | 74.7% | 75% |

Four confirmations of the turnover identity from the paper alone, and four new
targets that are counts rather than ratios. Ours against them, HMM, v8.5:

| market | our shifts | Shu | ratio |
|---|---|---|---|
| us | 128 | 96 (published) | 1.33 |
| de | 151 | 167 (from turnover) | 0.90 |
| jp | 208 | 197 (from turnover) | 1.06 |

The US bear share is 29.1% against 27.8%. **Same exposure budget, a third more
shifts inside it.** That is the shape of the US deviation: not more time in
cash, more chopping within the same time.

### What two rounds of parallel investigation eliminated

Round 1 (seven lines, each attacked by an independent refuter) and round 2 (six
more) eliminated: the data, the reporting window, the training-data start, the
metric definitions, the selection spec against Section 3.4.3, the unspecified
HMM fitting choices, the index substitution, a pure persistence account, a
validation-surface bias, and every alternative turnover convention.

Two findings survived their refuters only in part, and both are recorded as
open rather than concluded:

- The MDD gap is concentrated rather than spread, but the refuter showed the
  "index fall is a ceiling" argument false — the sealed delay-10 arm reaches
  -26.40% over a span in which the index *rose* 16.14%, so whipsaw can exceed
  the index. It also showed the 2000-02 route reproduces Shu's Sharpe exactly
  (0.540) while the 2020 route cannot reach the target at all.
- Inverting Shu's implied flip rates through our own per-market flip curves puts
  their behaviour at mean k ~ 13 on the US where we sit at 8, k ~ 3.6 on the DAX
  where we sit at 4, and k ~ 7 on the Nikkei where we sit at 10.8 while still
  flipping more, because Japan alone carries +0.86 shifts/yr of selection churn.
  Germany is therefore effectively matched; the US picks too small a k; Japan
  picks about right and pays for switching. Three markets, three situations —
  which is why the turnover error changes sign and why no single correction
  fixes it.

A claim NOT adopted: a pixel reconstruction of Figure 6 put Shu's own US HMM
drawdown near -22.3% against the printed -28.9%, implying the paper disagrees
with itself. Table 4 is internally consistent on that cell (0.54 x 11.3 = 6.10,
0.21 x 28.9 = 6.07), and a max statistic read off a rasterised curve is biased
shallow. Recorded as unverified. The refutation pass that would have tested it
died on the account spend limit, along with three other agents.

### The drawdown is computed on the total-return path (2026-07-28)

Tested because the excess reading would have explained the entire US HMM gap on
its own. It does not survive: buy-and-hold contains no model, so its drawdown
tests the definition directly, and the total-return reading matches Shu on all
three markets while the excess readings do not.

| cell | total return (ours) | excess, geometric | excess, arithmetic | Shu |
|---|---|---|---|---|
| us B&H | **-54.57%** | -58.98% | -30.29% | -55.2% |
| de B&H | **-72.68%** | -75.54% | -57.63% | -72.7% |
| jp B&H | **-77.33%** | -81.70% | -126.18% | -79.1% |
| us HMM | -19.72% | -21.82% | -23.44% | -28.9% |
| de HMM | **-40.20%** | -47.37% | -32.43% | -40.5% |
| jp HMM | **-51.73%** | -55.67% | -70.76% | -48.6% |

Summed absolute deviation over all nine cells: total return **0.237**, excess
geometric 0.450, excess arithmetic 1.651. The excess reading improves exactly
one cell -- us HMM, from 0.092 to 0.055 -- and degrades the other eight. That is
the signature of fitting to the answer, not of finding the definition.

`backtest.py:165-167` computes the drawdown on `cumprod(1 + strategy_return)`,
where `strategy_return` earns the cash rate while out of the market. Confirmed
correct.

**A consequence worth keeping.** On all nine cells the total-return drawdown is
strictly shallower than the excess drawdown, which is mechanical: the risk-free
rate adds positive drift while the strategy sits in cash. Figures 5 and 6 plot
CUMULATIVE EXCESS RETURN, so any drawdown read off those figures is the excess
statistic, not the one Table 4 reports. If a reconstruction of Figure 6 gives an
excess drawdown near -22%, then Shu's total-return drawdown must be shallower
than that, and could not be the printed -28.9%. Our own excess drawdown on the
same cell is -21.82%.

That argument stands or falls on whether the figure can be read accurately at
all, which is being tested by calibrating the extraction against Figure 5's
three JM curves, whose drawdowns Table 4 prints (-26.6%, -39.4%, -45.3%). A
method that cannot reproduce those has no standing to adjudicate a 6-point
dispute about Figure 6.

## The US HMM deviation, traced (2026-07-28)

Four rounds of parallel investigation eliminated eighteen hypotheses. The cause
is the one deviation the project had recorded from the beginning and dismissed
as harmless.

### Retraction first

An earlier reconstruction of Figure 6 put Shu's own drawdown near -22.3%,
implying the paper contradicted its own Table 4. **That is withdrawn.** The
figures are vector (`pdfimages -list` returns zero bitmaps on all 22 pages;
Figures 5 and 6 are matplotlib Form XObjects), so the geometry is exact. Three
independent inversions give -28.854%, -28.874% and -28.769%, all with peak
1998-07-17 and trough 2002-04-15, against a printed -28.9%.

The -22.3% was a convention error, and it was reproduced: reading the plotted
cumulative-excess curve as a wealth index gives -22.81% on that cell, and errs
by 20.8pp on average across the nine cells whose answer Table 4 prints (S&P
buy-and-hold -30.77% against a printed -55.2%). The plotted curve is an
arithmetic cumulative SUM of daily simple excess returns, so recovering a
drawdown requires differencing it, adding a risk-free series back and
compounding.

Two corroborations need no injected risk-free series at all: terminal plotted
excess 209.65% over 33.988 years = 6.168%/yr against Table 4's Sharpe x Vol =
6.102 (rounding band [6.019, 6.186]); and inverting the printed Calmar gives
|MDD| in [27.99%, 30.17%], which excludes -22.3% outright. The bear shading
recovers 27.80% of days and 96 shifts against the 27.8% and 96 printed at line
903.

The excess-versus-total question was also settled by measurement rather than
argument: on our own sealed US path the two conventions differ by 2.10pp, and
the excess reading is DEEPER, not shallower. It cannot manufacture a 9.18pp gap.

### The cause

The equity series. Recorded as a deviation since the first contract:

> [line 153-155] "The data analyzed in this article comprises the daily total return series of three major equity indices: S&P 500, DAX, and Nikkei 225"

Ours is the CRSP value-weighted total market from Kenneth French's daily
factors, because no free S&P 500 total-return series reaches back to 1970. It
was treated as harmless because buy-and-hold matches on every metric.

Buy-and-hold only tests 1990-2023. The difference is in 1987.

| 1987-10-19 | simple | log |
|---|---|---|
| S&P 500 (Shu) | -20.47% | -0.2290 |
| CRSP total market (ours) | **-17.41%** | **-0.1913** |

Twenty percent smaller in log magnitude, on the single most extreme day in the
whole training window; dropping that one day moves the 1978-1990 window's
annualised standard deviation from 14.81% to 13.79%.

The consequence is structural, switching on and off with whether that day is
inside the rolling window. Figure 2 publishes the fitted regime parameters, and
extracted losslessly from the vectors it splits exactly there:

| windows | n | share | our high-state vol minus Shu's |
|---|---|---|---|
| containing 1987-10-19 (fit dates to 1999-08-31) | 3000 | 28.3% | **-7.97pp** |
| 2009-2010 | 504 | 4.8% | +4.85pp |
| all others | 7084 | **66.9%** | **+0.12pp**, corr 0.995 |

Year by year through the divergent stretch (Shu / ours, annualised): 1990
43.88/34.65, 1995 44.00/36.69, 1998 41.79/34.12, then 2000 21.31/21.62 once
1987 leaves the window. Moment-matching their 1990 parameters implies their high
state holds about 2.7% of training days against our 8.3%.

A lower, broader high-volatility state is entered more readily. So:

| | ours | Shu |
|---|---|---|
| 1990-07-16..1990-10-11 drawdown | -19.72% | -19.08% |
| 1998-07-17..2002-04-15 drawdown | **-16.15%** | **-28.85%** |
| cash share, 1998 to the dot-com peak | 48.9% | less |
| cash share, peak to trough | **88.5%** | long into the 2002 bottom |

The two models agree in 1990 and diverge entirely across 1998-2002, which is
where the whole 9.18pp lives. Taking Figure 6's position path verbatim and
applying it to OUR returns reproduces seven of eight Table 4 cells within
rounding, with the trough at 2002-04-11 — so the data and the metrics are fine
and only the regime calls differ. The two position paths agree on 94.5% of days.

### What was eliminated on the way, and what it cost to be sure

The local-optimum explanation was tested and killed: on eight windows across the
divergent stretch, the protocol's ten seeds, forty single-pass k-means++ starts,
and a sweep over 21 mean pairs spanning +-6 sigma ALL converge to the identical
log-likelihood, volatilities and state shares. Our fit is the global optimum;
Shu's 43.88% is not a higher-likelihood solution on our data. That is what
forced the conclusion back onto the input series.

Also eliminated across the four rounds: the reporting window, the training-data
start, all three metric definitions, the selection spec, every rival turnover
convention, every smoother micro-convention, switching churn, validation-surface
bias, and a pure persistence account.

### Why this hid for so long

Three layers. Buy-and-hold matches, but only tests 1990-2023 while the defect is
in 1987. Round 1's index probe held the SIGNAL fixed and measured only the
return channel; the real channel is the refit, where a different index gives
different fitted parameters and only then a different signal — its own refuter
noted the refit channel was "only partially measured". And the error is
structural rather than random, switching with window membership, so it vanishes
from any full-sample average.

It also explains why Germany and Japan match on 7 of 8: those two use the
paper's own indices. Only the US is substituted.

### Consequence

This is a SPECIFIED-AND-WE-DEVIATE defect, not a free parameter. Fixing it is
replication. Free daily S&P 500 price is available from 1977 (Yahoo ^GSPC,
11,850 sessions, verified: 1987-10-19 = -20.47%), the official ^SP500TR from
1988 (9,070 sessions), and Shiller's monthly dividends from 1871. For the HMM
the price series suffices: over the 9,070 overlapping sessions the daily
log-return volatility of price and total return differ by 0.0011pp with
correlation 0.99960.
