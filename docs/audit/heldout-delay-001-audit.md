# Independent audit — heldout-delay-001

**Auditor:** separate agent; did not write `scripts/score_grid.py`,
`scripts/run_heldout_delay.py`, `scripts/_shu_table5.py` or
`research/heldout-delay-001.toml`, and did not share their author's context.
**Date:** 2026-08-01. **Commit under audit:** `e517497` on
`cleanup/research-protocol` (committed at 17:02:23 while this audit was in
progress; the bytes audited hash to the committed bytes — see §Restoration).

---

## VERDICT

**The numbers are real; the conclusion is not warranted.** All fifty-four cells
in `artifacts/heldout-delay/01-table5/cells.csv` reproduce **bit-identically**
(max absolute difference **0.0** across all 54 cells and all 54 relative
deviations, zero grade disagreements) under an independent recomputation written
from the library without importing either audited script, and the Table-5
transcription in `scripts/_shu_table5.py` is correct figure-for-figure against
`data/external/inputs/shu_paper.txt` lines 961–975, with the JM block read from
the correct three columns and the delay-1 column correctly duplicating Table 4.
The transposition hypothesis is **ruled out**. What fails is the premise, not
the arithmetic: **delays 5 and 10 are not held out for arm A.** Arm A's grids
were drawn from admissible sets that were *defined* by requiring every Table-5
delay-5 and delay-10 cell to fall within absolute 0.05 of the published value
(`scripts/probe_jm_grid_exhaustive2.py:261`, `scripts/probe_jm_per_market_grids.py:42`,
`scripts/probe_jm_best_13of14.py:49`), and the v10 reseal that adopted them was
*gated* on the same six cells (`scripts/gate_v10_reseal.py:89`). Arm B's grids
were selected on delay-1 Table-4 deviation alone. The comparison is therefore
between an arm that was screened on the "held-out" set and an arm that was not,
and all five of arm B's cells that fail the abs-0.05 screen are exactly the
cells that produce its D and F grades — arm A had 18/18 held-out cells inside
that screen **by construction**, arm B 13/18. The decision rule's inference
("if B's held-out worst cell is not better than A's, the searches measured
delay-1 noise") does not survive this: the observed ordering is what selection
alone predicts. Separately, the stated mechanism is **contradicted by
measurement** — Germany's arm B at delay 10 puts *less* weight on λ=847.69
(15.8% of months, down from 31.1% at delay 1) and *more* switches (169, up from
110), i.e. it is less persistent at delay 10, not "extremely persistent". The
reportable claim that survives is narrow: *arm A reproduces Shu's Table 5 at
delays 5 and 10 to B/B/C, on cells that were themselves an acceptance criterion
for arm A, and arm B does not.*

---

## FINDINGS, ranked by severity

### F-1 (CRITICAL) — the held-out set was an admission criterion for arm A

`scripts/_shu_table5.py:22-27` states: *"Every lambda grid this project has
searched was selected to minimise deviation on Table 4 … The delay-5 and
delay-10 columns above were never in any objective function."*
`research/heldout-delay-001.toml:9` asks about cells *"which no search in this
project has ever seen"*. Both statements are false.

Evidence chain:

| Where | What it does |
|---|---|
| `scripts/probe_jm_grid_exhaustive2.py:49-57` | hardcodes the same Table-5 JM cells at delays 5 and 10 |
| `scripts/probe_jm_grid_exhaustive2.py:261-273` | `pass_mask()` — for delay 5 and 10 a grid *passes* iff every Table-5 cell is within `TOL = 0.05` (line 44) |
| `artifacts/jm-residual/08-exhaustive-nine-arms/arms.csv` | the resulting pass counts: `us-d5 947,069`, `us-d10 2,369,049`, `de-d5 1,192,873`, `de-d10 1,102,081`, `jp-d5 168,384`, `jp-d10 513,280` |
| `scripts/probe_jm_per_market_grids.py:37,42` | intersects the delay-1, delay-5 and delay-10 pass sets → `per-market-intersections.csv`: **us 36,657**, de 0, jp 0 |
| `scripts/probe_jm_best_13of14.py:3-5,49` | for DE/JP, "best available" ≡ 7/8 Table-4 cells at delay 1 **AND all three Table-5 cells at delays 5 and 10** → DE 366 grids, JP 2,948 grids |
| `artifacts/jm-residual/09-per-market-grids/per-market-examples.csv` | contains `us,3,0|21.5443|70` — arm A's US grid |
| `artifacts/jm-residual/09-per-market-grids/best-13of14-examples.csv` | first rows `de,2,150|500` and `jp,2,10|220` — arm A's DE and JP grids |
| `research/experiment_registry.jsonl:117` | "*DE 366 grids, JP 2,948 grids (7/8 at d1 + **full Table-5 at d5 and d10**)*" |
| `research/experiment_registry.jsonl:118` | reseal spec: "*Grids were found by searching 6,474,511 menu subsets against **Table 4/5***" |
| `scripts/gate_v10_reseal.py:25,89-96` | gate 3b re-checks the adopted grids at delays 5 and 10 against `TABLE5_JM` at tolerance 0.05 and **fails the reseal** if any cell misses |
| `research/experiment_registry.jsonl:121` | adoption record: "*us 8/8 d1 + **3/3 d5 + 3/3 d10***, de 7/8 d1 + 3/3 + 3/3, jp 7/8 d1 + 3/3 + 3/3*" |
| `research/experiment_registry.jsonl:138` | "*There was NO STATED RULE for choosing among them: I took the first row of the -009 examples file*" — the choice **within** the screened set was arbitrary, but the screen itself was not |

By contrast arm B is genuinely delay-1-only: `scripts/probe_minimax_grid.py:50`
uses `arm_key(market, 1)` and `scripts/probe_dense_exhaustive.py:46` sets
`COST, DELAY = 10.0, 1`; neither ever evaluates delay 5 or 10.

Quantified consequence, recomputed by this audit:

```
HELD-OUT CELLS INSIDE THE ABSOLUTE 0.05 SCREEN THAT ARM A'S SELECTION IMPOSED
  arm A : 18 / 18      <- by construction; a miss would have failed gate 3b
  arm B : 13 / 18
arm B's five cells outside the screen:
  us d5 calmar  0.279 vs 0.39  (abs 0.111, rel 28.5%)   <- arm B's US worst cell
  de d10 sharpe 0.480 vs 0.29  (abs 0.190, rel 65.4%)
  de d10 calmar 0.230 vs 0.11  (abs 0.120, rel 109.1%)  <- arm B's DE worst cell
  jp d5  sharpe 0.150 vs 0.27  (abs 0.120, rel 44.6%)
  jp d10 sharpe 0.173 vs 0.24  (abs 0.067, rel 27.7%)
```

Every cell that drives arm B's C/F/D headline is a cell arm A was *forbidden to
fail*. This is exactly AGENTS.md §8's forbidden shortcut, "repeatedly viewed OOS
data is an untouched holdout".

*Reproduce:*
```bash
sed -n '261,273p' scripts/probe_jm_grid_exhaustive2.py
sed -n '85,96p'   scripts/gate_v10_reseal.py
cat artifacts/jm-residual/08-exhaustive-nine-arms/arms.csv
cat artifacts/jm-residual/09-per-market-grids/best-13of14-examples.csv
cat artifacts/jm-residual/09-per-market-grids/dejp-13of14-validation.txt
unset VIRTUAL_ENV && uv run python scripts/audit_heldout_recompute.py
```

*Weakest counter-argument, stated fairly:* the abs-0.05 screen does not
mechanically force a good **relative** grade. For jp d10 calmar (target 0.07) it
permits 71%, and for jp d10 cagr (target 0.034) it permits 147%. Arm A landed at
14.6% and 22.3%, far better than the screen requires. So the screen does not
explain all of arm A's performance — but it does explain the *comparison*, which
is the only thing the decision rule rests on.

### F-2 (CRITICAL) — arm A is labelled "pre-specified default"; it is a search winner

`research/heldout-delay-001.toml:50-54` calls arm A "the pre-specified default
that AGENTS.md 7.3 requires to be reported alongside any search winner". Global
`~/AGENTS.md:139-149` (§7.3) means by *pre-specified default* a setting chosen
without reference to the answer key. Arm A is the argmin/first-row of a
6,474,511-subset search against Tables 4 **and** 5
(`research/experiment_registry.jsonl:117-118`), chosen from the screened set by
no stated rule at all (`:138`). §7.3's remedy — "evaluate the selected setting on
cells the search never saw" — is not satisfied by *either* arm here: for arm A no
such cells remain in Table 5, and for arm B they exist but are being compared
against a screened opponent.

### F-3 (HIGH) — the mechanism asserted in the registry and commit message is false

The completion event (`research/experiment_registry.jsonl`, event
`2026-08-01T22:19:00Z`) and commit `e517497` both explain arm B's Germany
failure as *"an extremely persistent state sequence"* driven by λ=847.69. Direct
measurement of the monthly cross-validation choices contradicts this:

| market | arm | delay | share of months on the largest λ | switches | days risky |
|---|---|---|---|---|---|
| de | A | 1 / 5 / 10 | 0.299 / 0.284 / 0.362 | 18 / 18 / 18 | .850 / .856 / .869 |
| de | B | 1 / 5 / 10 | **0.311 / 0.267 / 0.158** | **110 / 102 / 169** | .804 / .774 / .740 |
| jp | B | 1 / 5 / 10 | 0.144 / 0.161 / 0.137 | 50 / 82 / 96 | .661 / .647 / .640 |
| us | A | 1 / 5 / 10 | 0.296 / 0.080 / 0.561 | 30 / 80 / 24 | .789 / .783 / .795 |

At delay 10 German arm B puts the **least** weight on λ=847.69 of any delay and
trades the **most** (169 switches against 110 at delay 1). It is the *least*
persistent configuration in its own row, not the most. Arm A's German grid, with
18 switches at every delay, is the rigid one. The high delay-10 Sharpe (0.480)
comes from the cross-validation rotating onto the *small* penalties (λ∈{0,1.875}
take 245 of 412 months at delay 10 against 78 at delay 1), not from persistence.
Whatever is happening, the published explanation is wrong and must be withdrawn.

*Reproduce:* the probe used is preserved at
`/tmp/claude-1017/-home-tle/69649cec-6fd3-40f9-9e01-42dd56f3559f/scratchpad/selection_mix.py`;
the same numbers follow from `select_monthly_candidate(...).choices["selected"]`
per arm and delay.

### F-4 (HIGH) — nothing checked the target table, the grading scale, or the held-out flag

Five faults pass every check that existed: **F11** (`us` delay-5 and delay-10
targets swapped → arm A's US grade silently drops B→C), **F09** (grade bands
reversed → arm B's Japan becomes an A), **F07** (delay-1 target used at every
delay), **F12** (`held_out` flag inverted → the published conclusion inverts,
arm B grading A/B/B) and **F10** (inert here, see F-6). The self-test only
compares the *pipeline* against the sealed run; the target table, the grading
scale and the held-out definition are never compared with anything. Given that
this experiment's entire content is "our number versus their printed number",
this was the single largest uncovered surface. **Closed by this audit** —
`tests/test_heldout_delay_audit.py::test_table5_transcription_matches_the_paper_text`
re-parses lines 961–975 of the paper and asserts the JM block cell by cell,
`::test_table5_delay_1_column_duplicates_table_4` checks the in-sample claim, and
`::test_grade_bands_are_the_scale_they_cite` pins the bands.

### F-5 (HIGH) — the scorer's own regression suite was red when the result was published

Adding the `delay` parameter to `score()` broke three tests in
`tests/test_score_grid_audit.py` — the stubs at lines 130–151 and 206–216 take
`(market, penalties, states_csv=None)` and now die of
`TypeError: … unexpected keyword argument 'delay'`. Those three are precisely
the NaN-blindness regression, the states-csv-exercise regression and the
reported-headline regression that a previous audit installed. The result was
committed with them failing.

*Reproduce (at `e517497`, before this audit's repair):*
`git stash && uv run python -m pytest tests/test_score_grid_audit.py -q`
→ `3 failed, 9 passed`. **Repaired** by widening the three stub signatures; the
module now passes 12/12.

### F-6 (MEDIUM) — the sealed-window lookup is keyed by delay but is untestable on real data

`scripts/score_grid.py:110-115` reads the evaluation window from the sealed row
for *that delay*. Every real sealed window is identical (`us`/`de`
1990-01-02→2023-12-29, `jp` 1990-01-19→2023-12-29 at all three delays), so fault
**F10** — pinning that lookup to delay 1 — changes nothing and passes both the
self-test and the driver. The code is correct; it is simply unexercised, and a
future run with delay-dependent windows would inherit an untested path.
**Closed** by `::test_the_sealed_window_row_is_keyed_by_the_delay`, which builds
a synthetic sealed run whose three delay rows start a year apart.

### F-7 (MEDIUM) — the delay hazard is closed, but only by an indirect assertion

The stated hazard (a delay that keys both the answer and its own known answer) is
genuinely closed: faults **F01** (delay pinned to 1 in the cross-validation),
**F02** (pinned in the execution) and **F03** (argument ignored entirely) are all
caught by the self-test, and I confirm the mechanism is the delay-keyed sealed
row plus the "three delays must disagree" assertion. But the self-test never
asserts *which* call site received the delay; it infers it from the answer
moving. **F01 and F02 produce different wrong answers and are indistinguishable
from each other in the failure message.** Closed by
`::test_delay_reaches_the_cross_validation` and `::test_delay_reaches_the_execution`,
which record the `delay_trading_days` kwarg at each call site separately.

My own recomputation applies the delay in both places
(`scripts/audit_heldout_recompute.py:92,101`), matching the Table 5 caption at
paper line 979–980. **What changes if only one does** — measured end to end, not
inferred:

| protocol | A: us / de / jp | B: us / de / jp |
|---|---|---|
| both (as published) | 5.9 B / 13.0 B / 22.3 C | 28.5 C / 109.1 F / 55.3 D |
| CV pinned to delay 1 (F01) | 23.6 C / 58.3 D / 15.0 B | 22.3 C / 69.2 F / 56.3 D |
| execution pinned to delay 1 (F02) | 57.3 D / 19.0 B / 19.4 B | 26.1 C / 31.0 C / 66.6 F |
| delay ignored entirely (F03) | 17.7 B / 51.4 D / 35.0 C | 18.6 B / 49.7 D / 66.4 F |

The protocol is **decisive**, not cosmetic: getting it wrong in either direction
moves arm A by up to 45 percentage points and flips several grade bands, and
under F03 arm A is no longer better than arm B in Germany at all. The published
run applies it correctly — this is a statement about how much the self-test is
load-bearing, not a defect in the result.

### F-8 (MEDIUM) — arm B's German and Japanese state matrices have no known-answer coverage

`self_test()` exercises the `states_csv` path only for the US
(`scripts/score_grid.py:192-219`), on a file that is bit-identical to the sealed
states on the shared penalties. Arm B's DE and JP numbers flow through
`artifacts/dense-menu/01-search/states-{de,jp}.csv`, which share **only two**
columns (λ=0 and λ=1000) with the 29-λ union caches and share **none** with the
sealed grids. I verified those two shared columns agree exactly on overlapping
dates for both markets, which is reassuring but is a two-column check on
48-column files. Arm B's worst cells (de d10) come through this uncovered path.

*Reproduce:* the column cross-check is four lines of pandas over the two files;
see the shared-column comparison in this audit's log.

### F-9 (LOW) — a stray zero-byte file `tol` was committed

`git show --stat e517497` lists `tol | 0`. It is a shell-redirect artifact, not
an artifact of the experiment, and it is now in history.

### F-10 (LOW) — the driver duplicates the arms' provenance

`scripts/run_heldout_delay.py:37-41` hardcodes the per-market states files while
the grids come from the frozen spec. A grid/menu mismatch is therefore
expressible without touching the spec. It is caught in practice — faults
**F04/F05/F06** all die at `resolve_columns` — but only because the two menus
happen to be disjoint. Recording the states path *in the spec* would make the
binding explicit.

### Non-findings (checked, clean)

* **Transcription.** `TABLE5_JM` matches paper lines 963–965 (S&P 500),
  968–970 (DAX), 973–975 (Nikkei 225) on all 27 figures; JM is the last three
  columns of seven in every row, row order Return/Sharpe/Calmar is correct, and
  `TABLE5_JM[m][1] == TABLE4[m]["fixed_jm"]` for all three markets and all three
  shared metrics. The docstring's line range (961–975) and caption line (979) are
  both accurate.
* **Evaluation window.** Identical across arms *and* delays. Forcing both arms
  onto the intersection of complete rows across both arms and all three delays
  changes nothing: sample sizes are already equal at 8565 (us), 8602 (de), 8336
  (jp) for every arm and delay, and every graded conclusion is unchanged
  (`rel_common` column of `artifacts/heldout-delay/02-audit/audit-cells.csv`).
* **Spec vs run.** `sha256(research/heldout-delay-001.toml)` =
  `1253cf9a552a9c755205db4bfd19580e1c55ebb183710a4e77e3d48507742b4d`, matching
  both registry rows. Arms, grids, markets, cells, delays, grade bands and the
  decision rule as run all match the frozen bytes; arm A equals
  `research-calibrated-v10.toml`'s own `lambda_grid` per market. No drift.
* **Grade-band arithmetic.** All 54 grades recomputed independently; zero
  disagreements.

---

## Independent recomputation

Written in `scripts/audit_heldout_recompute.py`, from
`select_monthly_candidate` / `apply_signal` / `performance_metrics` and
`research-calibrated-v10.toml`, importing neither `score_grid.py` nor
`run_heldout_delay.py`, with the Table-5 targets re-read from the paper by hand.
Output: `artifacts/heldout-delay/02-audit/audit-cells.csv`.

**Held-out headline (worst of six cells per arm and market):**

| arm | market | claimed | audited (sealed window) | audited (common window) | grade |
|---|---|---|---|---|---|
| A_sealed_default | us | 0.05867297 | 0.05867297 | 0.05867297 | B |
| A_sealed_default | de | 0.13011351 | 0.13011351 | 0.13011351 | B |
| A_sealed_default | jp | 0.22251482 | 0.22251482 | 0.22251482 | C |
| B_minimax_search | us | 0.28480423 | 0.28480423 | 0.28480423 | C |
| B_minimax_search | de | 1.09103250 | 1.09103250 | 1.09103250 | F |
| B_minimax_search | jp | 0.55269709 | 0.55269709 | 0.55269709 | D |

**Held-out cells, claimed vs audited** (delay-1 rows omitted; all reproduce
identically too):

| arm | market | d | cell | Shu | claimed | audited | Δ |
|---|---|---|---|---|---|---|---|
| A | us | 5 | cagr | 0.114 | 0.11105247 | 0.11105247 | 0 |
| A | us | 5 | sharpe | 0.71 | 0.69808428 | 0.69808428 | 0 |
| A | us | 5 | calmar | 0.39 | 0.36711754 | 0.36711754 | 0 |
| A | us | 10 | cagr | 0.117 | 0.11396685 | 0.11396685 | 0 |
| A | us | 10 | sharpe | 0.70 | 0.68569546 | 0.68569546 | 0 |
| A | us | 10 | calmar | 0.28 | 0.29417386 | 0.29417386 | 0 |
| A | de | 5 | cagr | 0.075 | 0.07437894 | 0.07437894 | 0 |
| A | de | 5 | sharpe | 0.38 | 0.35766292 | 0.35766292 | 0 |
| A | de | 5 | calmar | 0.13 | 0.14691476 | 0.14691476 | 0 |
| A | de | 10 | cagr | 0.059 | 0.06593174 | 0.06593174 | 0 |
| A | de | 10 | sharpe | 0.29 | 0.30957596 | 0.30957596 | 0 |
| A | de | 10 | calmar | 0.11 | 0.10653685 | 0.10653685 | 0 |
| A | jp | 5 | cagr | 0.040 | 0.03678055 | 0.03678055 | 0 |
| A | jp | 5 | sharpe | 0.27 | 0.26439994 | 0.26439994 | 0 |
| A | jp | 5 | calmar | 0.09 | 0.07297789 | 0.07297789 | 0 |
| A | jp | 10 | cagr | 0.034 | 0.02643450 | 0.02643450 | 0 |
| A | jp | 10 | sharpe | 0.24 | 0.20038915 | 0.20038915 | 0 |
| A | jp | 10 | calmar | 0.07 | 0.05975080 | 0.05975080 | 0 |
| B | us | 5 | cagr | 0.114 | 0.11214190 | 0.11214190 | 0 |
| B | us | 5 | sharpe | 0.71 | 0.67740513 | 0.67740513 | 0 |
| B | us | 5 | calmar | 0.39 | 0.27892635 | 0.27892635 | 0 |
| B | us | 10 | cagr | 0.117 | 0.11471453 | 0.11471453 | 0 |
| B | us | 10 | sharpe | 0.70 | 0.68060190 | 0.68060190 | 0 |
| B | us | 10 | calmar | 0.28 | 0.27116568 | 0.27116568 | 0 |
| B | de | 5 | cagr | 0.075 | 0.07894543 | 0.07894543 | 0 |
| B | de | 5 | sharpe | 0.38 | 0.40519347 | 0.40519347 | 0 |
| B | de | 5 | calmar | 0.13 | 0.13799031 | 0.13799031 | 0 |
| B | de | 10 | cagr | 0.059 | 0.09083712 | 0.09083712 | 0 |
| B | de | 10 | sharpe | 0.29 | 0.47974470 | 0.47974470 | 0 |
| B | de | 10 | calmar | 0.11 | 0.23001358 | 0.23001358 | 0 |
| B | jp | 5 | cagr | 0.040 | 0.01789212 | 0.01789212 | 0 |
| B | jp | 5 | sharpe | 0.27 | 0.14950974 | 0.14950974 | 0 |
| B | jp | 5 | calmar | 0.09 | 0.04032269 | 0.04032269 | 0 |
| B | jp | 10 | cagr | 0.034 | 0.02185654 | 0.02185654 | 0 |
| B | jp | 10 | sharpe | 0.24 | 0.17347081 | 0.17347081 | 0 |
| B | jp | 10 | calmar | 0.07 | 0.04653799 | 0.04653799 | 0 |

**Largest disagreement: 0.0.** Over all 54 cells (including the 18 in-sample
delay-1 cells), max |claimed − audited| = 0.0, max |claimed relative deviation −
audited relative deviation| = 0.0, grade disagreements = 0.

---

## Fault-injection matrix

Each fault is a **single edit** applied in an isolated shadow tree
(`scratchpad/mkfault.py`) that symlinks `src/`, `research/`, `data/` and the
config, keeps a **real copy** of every script, and keeps
`artifacts/heldout-delay/` real so a faulted driver writes inside the shadow.
*(Harness note, disclosed because it is the same class of error this audit
looks for: the first build symlinked the unpatched scripts, and
`Path(__file__).resolve()` followed the symlink back into the real repository —
so eight runs silently executed unfaulted code and rewrote the real
`cells.csv`. The rewritten file was verified bit-identical to my independent
recomputation, and the whole matrix was rebuilt with real copies and rerun.
The table below is from the corrected run.)*

Legend: ✅ = caught (non-zero exit / explicit failure); ❌ = passed silently.

Held-out headline the faulted driver printed, as `us / de / jp` worst cell and
grade. The published (control) row is `A 5.9 B / 13.0 B / 22.3 C` and
`B 28.5 C / 109.1 F / 55.3 D`.

| # | Fault (one edit) | `--self-test` | driver exit | driver headline | Verdict |
|---|---|---|---|---|---|
| F00 | control, no edit | ✅ pass (3.44e-14) | 0 | reproduces the published summary exactly | baseline |
| F01 | `delay`→`1` in `select_monthly_candidate` (CV blind to delay) | ✅ **FAIL** `us d5 cagr 0.110815 vs sealed 0.111052` | 0 | A 23.6 C / 58.3 D / 15.0 B · B 22.3 C / 69.2 F / 56.3 D | caught by self-test only |
| F02 | `delay`→`1` in `apply_signal` (execution blind to delay) | ✅ **FAIL** `us d5 cagr 0.112772` | 0 | A 57.3 D / 19.0 B / 19.4 B · B 26.1 C / 31.0 C / 66.6 F | caught by self-test only |
| F03 | `delay = DELAY` at top of `score()` (argument ignored) | ✅ **FAIL** `us d5 cagr 0.111390` | 0 | A 17.7 B / 51.4 D / 35.0 C · B 18.6 B / 49.7 D / 66.4 F | caught by self-test only |
| F04 | the two arms' grids swapped in the driver | ❌ pass (driver-only fault) | **1** | `penalty 15.0 is not a column of us: jm-states.csv` | caught by the matcher |
| F05 | arm B pointed at the sealed states file | ❌ pass | **1** | same refusal | caught by the matcher |
| F06 | `states` argument dropped on the `score()` call | ❌ pass | **1** | same refusal | caught by the matcher |
| F07 | `TABLE5_JM[market][1]` used as the target at every delay | ❌ pass | 0 | A 11.2 B / 40.8 D / 50.2 D · B 17.8 B / 27.8 C / 66.4 F | **SILENT** |
| F08 | `resolve_columns` snaps to the nearest column | ✅ **FAIL** "the matcher accepted a penalty that is not a column" | 0 | identical to control (both menus contain the exact values) | caught by self-test only |
| F09 | grade bands reversed (A↔D, B↔C) | ❌ pass | 0 | same deviations, grades become A **C** / **C** / **B**, B **F** / **B** / **A** | **SILENT** |
| F10 | sealed window row pinned to delay 1 | ❌ pass | 0 | identical to control — the fault has no effect on this data (F-6) | **SILENT** |
| F11 | Table-5 `us` delay-5/delay-10 targets swapped | ❌ pass | 0 | A **31.1 C** / 13.0 B / 22.3 C · B **30.5 C** / 109.1 F / 55.3 D | **SILENT** |
| F12 | `held_out` flag inverted (delay 1 graded as held out) | ❌ pass | 0 | A 0.5 **A** / 9.5 B / 21.4 C · B 0.6 **A** / 8.5 B / **7.6 B** — the opposite conclusion | **SILENT** |
| F13 | window filter `&`→`\|` (metrics over the whole path) | ✅ **FAIL** `us d1 cagr 0.113506 vs sealed 0.111390` | 0 | A 5.3 B / 11.5 B / 22.3 C · B 28.0 C / 101.9 F / 57.3 D | caught by self-test only |

**Nine of thirteen faults produce a complete, plausible, exit-0 driver run, and
five pass everything (F07, F09, F10, F11, F12).** Four of those five live on the
*target and grading* side, which neither the self-test nor the driver ever
checks; the fifth (F10) is inert on this data. F12 is the most alarming: a
one-word edit turns the published conclusion into its opposite — arm B grades
A/B/B and beats arm A in Japan — and nothing complains. F11 shows that a
transposed transcription of exactly the kind this audit was asked to rule out
would have produced a plausible, unchallenged number (arm A's US grade drops from
B to C).

The scorer's self-test is strong on the half it covers (5/5 pipeline faults
caught, including both one-sided delay faults and the nearest-match snap) and
blind on the other half. The driver contributes exactly one guard — the column
matcher, which catches all three grid/menu faults.

All four are now caught by `tests/test_heldout_delay_audit.py`:
F07 and F12 by `::test_delay_1_can_never_enter_the_held_out_set` plus
`::test_the_driver_runs_the_arms_the_spec_declares`, F09 by
`::test_grade_bands_are_the_scale_they_cite`, F11 by
`::test_table5_transcription_matches_the_paper_text`.

---

## Tests added

`tests/test_heldout_delay_audit.py` — **15 tests, all passing**
(`uv run python -m pytest tests/test_heldout_delay_audit.py -q` → `15 passed`):

1. `test_table5_transcription_matches_the_paper_text` — re-parses lines 961–975
   of `shu_paper.txt`, asserts seven figures per row and that the JM block is the
   last three, cell by cell against `TABLE5_JM`. **Catches F11.**
2. `test_table5_delay_1_column_duplicates_table_4` — the docstring's in-sample
   claim, checked against `_shu_table4.TABLE4`.
3. `test_delay_1_can_never_enter_the_held_out_set` — **catches F12/F07.**
4. `test_grade_bands_are_the_scale_they_cite` — bands, ordering, boundary
   behaviour, NaN and inf. **Catches F09.**
5–7. `test_delay_reaches_the_cross_validation[1,5,10]` — **catches F01/F03.**
8–10. `test_delay_reaches_the_execution[1,5,10]` — **catches F02/F03.**
11. `test_the_sealed_window_row_is_keyed_by_the_delay` — synthetic sealed run
    with three different window starts. **Catches F10.**
12. `test_an_unpublished_delay_is_refused`.
13. `test_frozen_spec_matches_its_registry_hash`.
14. `test_the_driver_runs_the_arms_the_spec_declares` — and forbids the driver
    from hardcoding any penalty. **Catches F04.**
15. `test_arm_a_is_the_sealed_configs_own_grid`.

`tests/test_score_grid_audit.py` — three stubs widened to the new `score()`
signature so the previously-red regressions run again; **12 passing**.
Combined: `27 passed in 64.70s`.

**Tests deliberately NOT added:** nothing that would score a grid against the
delay-5 or delay-10 cells inside the test suite. Decision rule §3 forbids
consuming the held-out set, and a test that asserts a held-out number is a
search of budget one.

---

## What should change

1. **Retract the framing.** Delays 5 and 10 are in-sample for arm A. The correct
   statement of the result is: *arm B, selected on delay 1 only, does not
   reproduce Table 5 at delays 5 and 10 (us C, de F, jp D). Arm A does (B/B/C),
   but arm A was screened on exactly those cells, so this is not evidence that
   arm A generalises.* The comparison between the arms carries no information
   about generalisation.
2. **Correct or withdraw the persistence mechanism** in the registry event and
   in `e517497`'s message (F-3). The measurement says the opposite.
3. **The honest held-out test is still available**, but it must use an arm
   never screened on Table 5 — e.g. a literature-named grid from
   `probe_jm_grid_exhaustive2.NAMED_001` or the companion log-spaced default —
   as the pre-specified default that §7.3 actually asks for. That is a new
   frozen question, and it should be frozen before any such number is computed.
4. **Fix `_shu_table5.py:22-27`**, which is the load-bearing false statement.

---

## Restoration

No protected file was edited: fault injection used shadow copies, never
in-place edits.

```
$ git status --short
?? docs/audit/heldout-delay-001-audit.md
?? scripts/audit_heldout_recompute.py
?? tests/test_heldout_delay_audit.py
 M tests/test_score_grid_audit.py

$ git diff --stat -- scripts/score_grid.py scripts/run_heldout_delay.py \
      scripts/_shu_table5.py research/heldout-delay-001.toml
(no output — all four unchanged)

$ sha256sum scripts/score_grid.py scripts/_shu_table5.py \
      scripts/run_heldout_delay.py research/heldout-delay-001.toml
54caeeae58fd13ed6f9331f706101beaf53a23127ffea21386aab6297835db59  scripts/score_grid.py
8d13df8afdd5891e6796f7949bc8c2c78e46c6bcc49b98539f7421e4725068b0  scripts/_shu_table5.py
0aa8c39d6357fe9dac801531023a898f94cf32307084a2bcb61e6f7fe1b4d680  scripts/run_heldout_delay.py
1253cf9a552a9c755205db4bfd19580e1c55ebb183710a4e77e3d48507742b4d  research/heldout-delay-001.toml
```

`artifacts/heldout-delay/01-table5/cells.csv` was rewritten by the leaked
harness runs described above and verified bit-identical to the independent
recomputation afterwards (max |Δ| = 0.0 over 54 cells).
