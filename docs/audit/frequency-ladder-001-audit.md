# Independent audit — frequency-ladder-001

Auditor: an agent that wrote none of `research/frequency-ladder-001.toml`,
`scripts/run_frequency_ladder.py` or `scripts/diagnose_penalty_frequency_map.py`.
Everything below was recomputed from the raw artifacts, not read and agreed
with. Date of audit: 2026-08-01/02. Branch: `cleanup/research-protocol`.

## VERDICT

**The numbers are real, the menu is causal in the narrow sense claimed, and the
German improvement is genuine but is smaller and less exclusive than the
headline makes it sound.** All 48 committed cells reproduce to 9.4e-17 through
an evaluation I wrote from the library without touching the runner or the
scorer; the menu re-derives exactly from the refit objectives in all three
markets; the recorded `objective` really is the penalised objective (58
independent refits reproduce it exactly); no window dated on or after
1990-01-01 can move the menu (poisoning every post-cutoff objective with noise
leaves all three menus bit-identical); the contributing windows all end
1989-07-03 or earlier, before the 1990-01-02 evaluation start; and the ladder
run's state machinery is bit-identical to the sealed v10 baseline, so the German
comparison changes the menu and nothing else. **But three things in the claim do
not survive.** First, the spec's own audit trail is wrong where it matters: it
asserts that no window with a decreasing objective is dated before 1990, and
one is — Japanese, 1989-07-03, whose mathematically impossible negative jump
rate is what makes it contribute λ=75.0 to Japan's slowest rung; refitting that
window with six times as many restarts lowers its objective at two penalties by
1.1% and 1.8% and turns the whole region linear, so the map underlying the
Japanese menu is not converged, in the market that grades F. Second, "specified a priori in regime
frequency … the only free choice" is an overstatement: the *targets* are a
priori, but the *values* are quantised to midpoints of a union penalty grid
assembled from the paper's own published Table-3 penalties and the authors'
withdrawn arXiv-v1 grid — the two sources the owner forbade — and the lattice is
so coarse that any top rung between 5 and 10 shifts/year yields the identical
menu, so the ladder is not what determines the numbers. Third, the inversion is
not "exact": the finite difference is a chord slope, i.e. an interval-average
jump count, and the derived penalties systematically undershoot their targets by
up to 32%, so the menu that was run is uniformly slower than the ladder it
claims to implement. On the German claim specifically: 1.230 against 1.700 is a
real, like-for-like improvement over 0.264, but the F→C grade jump is
arm-conditional (arm L gives Germany grade D), and arm M was reported as the
headline even though the spec's own pre-registered condition for preferring it
("if arm M beats arm L everywhere") is false — arm M is worse in Japan. A
sensitivity sweep over six defensible ladders confirms the split: the
*qualitative* finding is robust (Germany improves in **all twelve** ladder/arm
combinations, from 0.264 to somewhere in 1.14–1.70; Japan is F in all twelve;
the US never beats D), but the *headline* is not — Germany's worst cell spans
25.9% (C) to 65.4% (F). The result should be reported as *"a menu derived from a
frequency target, quantised onto a pre-existing penalty lattice, reliably fixes
the direction of Germany's turnover error and reliably costs the US and Japan,
with a grade that depends on which rungs you chose"*, not as a clean a-priori
derivation that measures a grade.

---

## FINDINGS, ranked by severity

### F-1 (HIGH) The frozen spec asserts a fact about its own inputs that is false, and the error sets Japan's slowest rung

`research/frequency-ladder-001.toml:110-112`:

> "5. The five training windows with a decreasing objective (registry
> penalty-frequency-map-001) are a known defect of the fit, not of this
> construction. **None of them is before 1990, so none contributes to the
> menu.**"

`artifacts/penalty-frequency/01-map/suboptimal-fits-jp.csv` lists
`1989-07-03`, which is before the cutoff and does contribute. The registry
entry the spec cites *names that date itself* ("JP 1989-07-03, 1990-01-04 and
2020-07-01 at lambda 70"), so the claim contradicts its own source.

On that window the objective **falls** from 3005.140 at λ=70 to 2972.256 at
λ=80, a drop of 1.094%. The implied rate is **−0.276 shifts/year**, which is
impossible: L(λ) is a minimum of affine functions with non-negative slopes. A
negative rate satisfies `per_year <= target` for *every* rung, so the window
contributes the midpoint 75.0 at the 0.25/yr rung purely because the fit is
stuck in a worse local optimum.

**I repaired the window to see how much the defect matters.** Refitting the same
3000 observations with `n_init=60` instead of the protocol's 10:

| λ | recorded (n_init=10) | repaired (n_init=60) | shortfall | jumps |
|---|---|---|---|---|
| 60 | 2965.140 | **2932.256** | 32.88 (1.12%) | 2 |
| 70 | 3005.140 | **2952.256** | 52.88 (1.79%) | 2 |
| 80 | 2972.256 | 2972.256 | 0 | 2 |
| 100 | 3012.256 | 3012.256 | 0 | 2 |

Repaired, all three intervals give exactly 0.168 shifts/year, the constant a
2-jump solution implies — the true curve is linear here and the recorded one is
not. **Two of the four penalties were suboptimal, and only one of them made the
objective decrease.** The monotonicity check in
`scripts/diagnose_penalty_frequency_map.py:41` therefore detects a *lower bound*
on the damage, exactly as its own docstring says: "5 of 256 windows" is a floor,
not a count, and the map is built on fits that `n_init=10` does not reliably
converge.

The consequences for the menu, measured two ways:

| Japan, arm L | 8/yr | 4/yr | 2/yr | 1/yr | 0.5/yr | 0.25/yr |
|---|---|---|---|---|---|---|
| as run (17 windows) | 0.55 | 2.829 | 8.598 | 17.5 | **45.0** | **75.0** |
| dropping 1989-07-03 (16 windows) | 0.55 | 2.829 | 8.598 | 17.5 | **47.949** | **82.5** |

and, for the window itself, the repaired rate on (60, 70) is 0.168, already at
or below the 0.25 target, so a converged fit would have had that window
contribute **at most 65** rather than 75 at the slowest rung — the opposite
direction from dropping it. I am not claiming to know the repaired median
without repairing every window; I am claiming the recorded map is not a
faithful map, in a market whose result is F, and that the spec's assertion to
the contrary is false.

Both rungs the two treatments disagree about are the two the arm-M truncation is
defined around.

```
unset VIRTUAL_ENV && uv run python -m pytest tests/test_frequency_ladder_audit.py \
  -k no_contributing_window_has_a_decreasing_objective -rx
```
(pinned as a strict `xfail` so that repairing the fit, or correcting the spec,
turns into a loud failure.)

### F-2 (HIGH) The ladder is not the only free choice: the menu is quantised onto a lattice built from the paper's published penalties and the authors' withdrawn grid

`scripts/run_frequency_ladder.py:53` takes the **midpoint of two consecutive
penalties of the union grid**. That grid is not a property of the frequency
framing; it is the 29-value union assembled by
`scripts/probe_jm_grid_identification.py:47-57` out of eight named candidate
sets, two of which are exactly the sources the owner ruled out:

* `table3_sealed = (0, 5, 15, 35, 70, 150)` — the paper's own published jump
  penalties (`data/external/inputs/shu_paper.txt:643`);
* `v1_author_withdrawn = (10, 22, 50, 100, 220, 500, 1000)` — the authors'
  arXiv-v1 candidate set.

Every rung of every arm is a midpoint of two union values, or (where the median
falls between two windows' answers) the mean of two such midpoints — verified
for all eighteen rungs. **At least eight of the eighteen** are bracketed
directly by a paper-published or author-supplied penalty:

| rung | bracketed by | source of the bracket |
|---|---|---|
| US 37.5 | 35, 40 | **35 = paper Table 3** |
| US 65 | 60, 70 | **70 = paper Table 3** |
| US 23.5 | 22, 25 | **22 = authors' v1** |
| DE 23.5 | 22, 25 | **22 = authors' v1** |
| DE 185 | 150, 220 | **150 = paper Table 3, 220 = authors' v1** |
| JP 17.5 | 15, 20 | **15 = paper Table 3** |
| JP 45 | 40, 50 | **50 = authors' v1** |
| JP 75 | 70, 80 | **70 = paper Table 3** |

(and three more indirectly: US 5.460 is a mean of midpoints straddling
**5 = paper Table 3**; DE and JP 8.598 are bracketed above by
**10 = authors' v1**.)

And the lattice is coarse enough that the ladder barely determines the menu at
all. Sweeping only the top rung:

| top rung target | US λ | DE λ | JP λ |
|---|---|---|---|
| 20 | 0.05 | 0.05 | 0.05 |
| 11 | 0.30 | 0.55 | 0.55 |
| **10 … 6** | **0.55** | **0.55** | **0.55** |
| 5 | 1.465 | 1.465 | 1.465 |

Any a-priori belief between 6 and 10 shifts/year gives the identical first rung
in all three markets. Fault **F11** (change the spec's ladder from 8.0 to 10.0)
ran to completion, passed the re-derivation gate, and produced numerically
identical results — a change to the one declared free choice was undetectable.

This is *not* evaluation-period lookahead. It is a violation of the constraint
the spec sets itself at lines 11-16 ("no searching to the answer, and no using
their published values"): the answer is snapped onto a grid that contains their
published values.

```
unset VIRTUAL_ENV && uv run python -m pytest tests/test_frequency_ladder_audit.py \
  -k every_menu_value_is_a_midpoint
```

### F-3 (HIGH) The inversion is a biased interval approximation, not exact, and the midpoint rule's stated justification is false

`research/frequency-ladder-001.toml:29-34` ("The inversion is exact and free")
and `:50` ("the representative penalty for an interval is its midpoint,
**because the jump count is constant across the interval**").

The mathematics of the first claim is right for the exact minimiser: L(λ) =
min over (S, θ) of a family of affine functions of λ with slopes jumps(S) ≥ 0,
hence concave, piecewise linear, non-decreasing, with slope = jumps at the
optimum. I verified the implementation reads the right column:
`jumpmodels/jump.py:88` defines `val_` as the **penalised** objective, and 58
refits I ran independently reproduce `union-refits.csv` exactly (worst
difference 0 to printed precision on all 29 penalties of two German windows).

But `np.diff(obj) / np.diff(lam)` at `run_frequency_ladder.py:54` is a **chord**
slope, which for a concave function is the interval-*average* jump count, not
the jump count at the midpoint. The stated justification is false by
measurement: on the German window `1981-09-24` the interval the 8/yr rung
selects is (0.1, 1.0), across which the jump count falls **111 → 61**, a factor
1.8.

The chord is the *average* jump count over the interval, ∫jumps(λ)dλ / Δλ.
Because jumps(λ) falls convexly, that average is dominated by the
fast-switching left end and exceeds the count at the midpoint, so the derived
penalty systematically **over-smooths**. Measured by refitting every derived
penalty on every pre-1990 window (312 refits, states decoded, label changes
counted):

| ladder rung | US realised | DE realised | JP realised |
|---|---|---|---|
| 8 /yr | **5.46** (−32%) | **6.22** (−22%) | **5.80** (−27%) |
| 4 /yr | 3.86 (−3%) | 3.19 (−20%) | 3.28 (−18%) |
| 2 /yr | 1.85 (−7%) | 1.68 (−16%) | 1.76 (−12%) |
| 1 /yr | 1.01 (+1%) | 0.92 (−8%) | 1.01 (+1%) |
| 0.5 /yr | 0.50 (0%) | 0.42 (−16%) | 0.42 (−16%) |
| 0.25 /yr | 0.25 (0%) | 0.17 (−32%) | 0.17 (−32%) |

(median in-sample shifts/year over the pre-1990 windows;
`artifacts/audit/frequency-ladder-001/realised-frequency.csv`)

Every rung realises at or below its target. The menu that was run is uniformly
slower than the ladder it claims to implement. The direction is *against* the
German claim rather than for it — Germany fails by trading too little — so this
is not a finding that the German number is inflated. It is a finding that
"the menu implements 8 to 0.25 shifts per year" is not what was implemented,
and the residual German miss (27.6%) is partly this bias rather than a fact
about the frequency framing.

```
unset VIRTUAL_ENV && uv run python -m pytest tests/test_frequency_ladder_audit.py \
  -k penalised_objective
```

### F-4 (MEDIUM-HIGH) The re-derivation gate covers arm L only — and the headline number comes from arm M

`scripts/run_frequency_ladder.py:77-85` re-derives the menu and compares it
against `arms["L_full_ladder"][market]`. Arm M is consumed at line 111 with no
check of any kind, and nothing anywhere asserts the relation the spec declares
for it ("the ladder truncated at 0.5 shifts per year", i.e. M == L[:5]).

Fault **F7** — feed arm L's menu while labelling it arm M, one line — ran to
completion (rc=0) and wrote a `summary.csv` in which arm M carries arm L's
numbers. Nothing refused, nothing warned. Fault **F10** — silently drop one
more rung from arm M in the spec — also passed, and changed Germany's headline
from 27.6% to 20.8% and the US from 47.1% to 67.2%.

```
unset VIRTUAL_ENV && uv run python -m pytest tests/test_frequency_ladder_audit.py \
  -k "every_arm_is_the_menu or arm_m_is_arm_l_truncated"
```

### F-5 (MEDIUM) The arm that supplies the headline was chosen after the numbers

Declaring both arms up front is right, and the spec does it
(`research/frequency-ladder-001.toml:54-68`). But it never says which arm is
primary, and the only pre-registered statement that would have made arm M the
story is **false**:

> `:126` "If arm M beats arm L everywhere, the finding is about the selector
> being allowed to opt out …"

Arm M does not beat arm L everywhere: Japan is 116.2% under M against 78.4%
under L. The registry outcome nonetheless reports arm M's grades as the
headline. Under arm L, Germany is 55.4% grade **D**, not 27.6% grade **C**, so
"Germany moves from grade F to grade C" is arm-conditional.

The German **turnover** claim is not arm-conditional and survives: 1.260 (arm L)
and 1.230 (arm M) against Shu's 1.700, versus 0.264 under the calibrated v10
menu. Report the turnover, qualify the grade.

### F-6 (MEDIUM) The spec's grading table, cell list and cutoff prose are decorative

The runner imports `CELLS` from `scripts/score_grid.py:31` and `grade` from
`scripts/_shu_table5.py:72`, and hardcodes the cutoff at
`scripts/run_frequency_ladder.py:36`. The spec's `[grading]`, `protocol.cells`
and `derivation.aggregation` are never compared with any of them. Faults
**F12** (widen band C from 40% to 80%), **F13** (drop turnover from the spec's
cell list) and **F14** (change the cutoff prose from 1990 to 2010) all ran to
completion with the committed numbers unchanged.

Note the asymmetry: changing the *runner's* `CUTOFF` constant is caught (F1,
F2), because it moves the derived menu; changing the *spec's statement of the
cutoff* is not. The gate binds code to code, never code to spec.

```
unset VIRTUAL_ENV && uv run python -m pytest tests/test_frequency_ladder_audit.py \
  -k spec_declares_the_cells
```

### F-7 (LOW-MEDIUM) The freeze is evidenced only by self-reported timestamps, one of which is internally inconsistent

`git log -- research/frequency-ladder-001.toml` returns a single commit,
`e0f2533`, which contains the spec, the runner, the artifacts and the registry
together. There is no cryptographic evidence of ordering; only the registry
timestamps and file mtimes.

Those disagree. The spec records `amended_at_utc = "2026-08-02T02:41:00Z"` and
the registry event agrees, but the file's mtime is **02:28:27Z**, and two of
the three states files were already written after that (`states-us.csv`
02:32:57Z, `states-de.csv` 02:37:07Z). The recorded amendment time therefore
postdates results the amendment claims to precede.

The filesystem ordering *does* support the substantive claim — the spec was last
written at 02:28:27Z, the first result artifact at 02:32:57Z, so the amendment
really did precede every result — but the record is the only evidence there is,
and it should be right.

**The content of the amendment, by contrast, checks out completely.** The spec
on disk hashes to
`4dc4832522933731020133ed4f34e27309951389625e6cc6cc6895865a71890a`, exactly the
`spec_sha256` of the `SPEC_AMENDED_BEFORE_RESULTS` event. And the pre-amendment
file is recoverable from the amendment's own description: delete the two lines
the amendment added (`amended_at_utc`, `amendment`), substitute the four
rounded transcriptions back (1.465348864441625→1.47, 5.459611390906074→5.46,
2.829145724599095→2.83, 8.59842836500576→8.60), and the reconstruction hashes
to
`8ad7564b780b5d34d1932d73f106038d6b1fb36ced659c2813f64d5df468826b` — bit for
bit the `FROZEN_BEFORE_RESULTS` hash. Had a single other byte changed — a
ladder rung, the cutoff, a grading band, a decision rule — the hash could not
match. **The amendment changed exactly what it says it changed and nothing
else.** This is now pinned by
`test_the_amendment_changed_only_what_it_says_it_changed`.

`scripts/run_frequency_ladder.py` was last modified at 02:26:29Z, i.e. after
the 02:23:56Z freeze, with no history to say what changed.

### F-8 (LOW) The ladder's length is justified by a statistic measured on the evaluation period, and the statistic is unsourced

`research/frequency-ladder-001.toml:49`:

> "Six values because the monthly selector cannot resolve neighbours — measured
> **median winner-minus-runner-up is 0.0075 Sharpe** — so a finer menu adds
> variance, not information."

`grep -rn "0.0075"` finds that number nowhere in the repository outside this
line. The nearest published statistic is
`docs/audit/2026-07-full-audit.md:437`, a winner/runner-up margin computed over
the monthly selection decisions — which run from about 1989 to 2023, i.e. the
evaluation period. So the ladder's *length*, unlike its rungs, was set with a
number measured on data the menu is supposed not to have seen, contradicting
`:25-27` ("That statement can be written down before any data is touched").

The channel is weak — it fixes a count, not a value — and the sensitivity table
below shows an 11-rung ladder does not rescue any market. It is recorded so the
"only free choice" claim is not read as stronger than it is.

### F-9 (LOW) The map script and the runner disagree about which windows define the map, and the registry quotes the map script

`scripts/diagnose_penalty_frequency_map.py:146` medians over **all** windows;
`scripts/run_frequency_ladder.py:43` medians over pre-1990 only. The published
map artifact records no cutoff at all, and the registry entry for
`penalty-frequency-map-001` quotes the all-window medians ("at 0.25, US 65, DE
185, JP 185") — which are not the menu. Japan's 0.25 rung is 75, not 185.

Measured difference:

| market | pre-1990 (the menu) | all windows (the registry's map) |
|---|---|---|
| us | 0.55, 1.465, 5.460, 23.5, 37.5, 65.0 | 0.30, 1.465, 4.185, 17.5, 37.5, 65.0 |
| de | 0.55, 2.829, 8.598, 23.5, 90.0, 185.0 | 0.55, 2.829, 8.598, 23.5, 75.0, 185.0 |
| jp | 0.55, 2.829, 8.598, 17.5, 45.0, 75.0 | 0.55, 1.465, 6.098, 25.9, 65.0, 185.0 |

The restriction binds hardest on Japan and moves its top rung by 2.5×. That is
the causality constraint doing real work, and costing the experiment its worst
market — which is the honest direction, and worth stating rather than leaving
implicit.

---

## What I checked and found sound

* **All 48 cells reproduce.** Worst absolute disagreement between my
  independent evaluation and `artifacts/frequency-ladder/01-run/cells.csv`:
  **9.37e-17**. Table below.
* **The menu re-derives exactly** from `union-refits.csv` in all three markets,
  to `rtol=1e-12`, using an inversion I wrote from the spec's prose.
* **The recorded objective is the penalised objective.** 58 independent refits
  (two German pre-1990 windows × 29 penalties) reproduce `union-refits.csv`
  exactly, and the chord slopes are bracketed by the endpoint jump counts, as
  concavity requires.
* **No post-cutoff window can move the menu.** Replacing every objective on a
  window dated ≥ 1990-01-01 with uniform noise in ±1e6 leaves all three menus
  bit-identical. 18 (us) / 17 (de) / 17 (jp) pre-1990 windows exist and all
  carry 3000 observations.
* **The contributing windows end before the evaluation window opens.** Latest
  `training_end` among contributors: 1989-07-03; evaluation starts 1990-01-02
  (us, de) and 1990-01-19 (jp).
* **The German comparison is like for like.** `features.csv` is byte-identical
  between the v9.4 baseline the refits came from and the v10 baseline the cells
  were scored on (md5 `7c8bf645cf19ab44438ae11b9d2702fb` for de, and likewise
  for us and jp); both runs use `research-calibrated-v10.toml`, delay 1, 10 bps,
  window 1990-01-02..2023-12-29, n=8602. Refitting Germany at λ=150 through the
  runner's own `fixed_jm_states` call reproduces the sealed
  `jm-states.csv` column on **all 10,671 non-NaN cells, zero differences,
  identical NaN mask and identical index**. The only thing that differs between
  the two German runs is the candidate menu.
* **Selection and execution are causal.** `_score_decision`
  (`src/adaptive_jump/walkforward.py:470`) scores each candidate on returns
  strictly inside `(decision_date − 8y, decision_date]`; `apply_signal` applies
  a one-day delay. This is the sealed protocol, shared by both German runs.
* **The Table-4 targets are the paper's.** My own transcription from
  `data/external/inputs/shu_paper.txt:738-745` matches `scripts/_shu_table4.py`
  and `cells.csv` in all 24 target cells (DAX JM turnover `170%` → 1.70 at
  line 744, S&P `44%`, Nikkei `72%`).
* **The refit objectives, though produced under a different config, are valid
  inputs.** `union-refits.csv` was written by
  `scripts/probe_jm_grid_identification.py:163` under
  `research-expanding-v9-4.toml`, not under the
  `research-calibrated-v10.toml` the spec names as canonical, and the spec does
  not record this. I checked every field that could matter: `ModelProtocol`
  (n_states 2, fit_window 3000, standardizer `expanding_full_history_ddof1`,
  min obs 63) and the JM fit hyperparameters (n_init 10, random_state 0,
  max_iter 1000, tol 1e-8, refit_months (1,7)) are **identical** between the
  two configs, and the `features.csv` files are byte-identical across the two
  baseline directories. The only difference is
  `SelectionProtocol.boundary_fraction_limit` (0.05 in v9.4, 1.0 in v10), which
  does not enter the derivation. Worth recording in the spec's `[sources]`
  block; not a defect in the result. (The v10 relaxation to 1.0 does mean the
  selector is unconstrained about sitting on a menu endpoint, which is what
  makes the arm L / arm M contrast meaningful in the first place — a property
  of the sealed baseline, shared by both German runs.)

---

## Independent recomputation (all 48 cells)

Built from `adaptive_jump.walkforward.select_monthly_candidate`,
`adaptive_jump.backtest.apply_signal` and `performance_metrics`, config
`research-calibrated-v10.toml`, features
`artifacts/fixed-baselines/fixed-baselines-36ca1ace131c-.../{market}/features.csv`,
states `artifacts/frequency-ladder/01-run/states-{market}.csv`. **Neither
`scripts/score_grid.py` nor `scripts/run_frequency_ladder.py` was imported.**

| arm | mkt | cell | auditor | runner | Shu | auditor rel dev | runner rel dev | abs gap |
|---|---|---|---|---|---|---|---|---|
| L_full_ladder | de | cagr | 0.054611 | 0.054611 | 0.086 | 36.50% | 36.50% | 6.2e-17 |
| L_full_ladder | de | volatility | 0.154405 | 0.154405 | 0.164 | 5.85% | 5.85% | 5.6e-17 |
| L_full_ladder | de | sharpe | 0.266060 | 0.266060 | 0.440 | 39.53% | 39.53% | 5.6e-17 |
| L_full_ladder | de | maximum_drawdown | -0.512032 | -0.512032 | -0.394 | 29.96% | 29.96% | 0 |
| L_full_ladder | de | calmar | 0.080231 | 0.080231 | 0.180 | 55.43% | 55.43% | 2.8e-17 |
| L_full_ladder | de | expected_shortfall_5pct | -0.024277 | -0.024277 | -0.025 | 2.89% | 2.89% | 5.2e-17 |
| L_full_ladder | de | turnover | 1.259707 | 1.259707 | 1.700 | 25.90% | 25.90% | 0 |
| L_full_ladder | de | leverage | 0.749477 | 0.749477 | 0.840 | 10.78% | 10.78% | 0 |
| L_full_ladder | jp | cagr | 0.020047 | 0.020047 | 0.047 | 57.35% | 57.35% | 5.9e-17 |
| L_full_ladder | jp | volatility | 0.151782 | 0.151782 | 0.171 | 11.24% | 11.24% | 0 |
| L_full_ladder | jp | sharpe | 0.162924 | 0.162924 | 0.310 | 47.44% | 47.44% | 0 |
| L_full_ladder | jp | maximum_drawdown | -0.504435 | -0.504435 | -0.453 | 11.35% | 11.35% | 0 |
| L_full_ladder | jp | calmar | 0.049023 | 0.049023 | 0.120 | 59.15% | 59.15% | 7.6e-17 |
| L_full_ladder | jp | expected_shortfall_5pct | -0.024632 | -0.024632 | -0.026 | 5.26% | 5.26% | 5.2e-17 |
| L_full_ladder | jp | turnover | 1.284789 | 1.284789 | 0.720 | 78.44% | 78.44% | 0 |
| L_full_ladder | jp | leverage | 0.587452 | 0.587452 | 0.750 | 21.67% | 21.67% | 0 |
| L_full_ladder | us | cagr | 0.107460 | 0.107460 | 0.112 | 4.05% | 4.05% | 2.8e-17 |
| L_full_ladder | us | volatility | 0.125735 | 0.125735 | 0.131 | 4.02% | 4.02% | 0 |
| L_full_ladder | us | sharpe | 0.667567 | 0.667567 | 0.680 | 1.83% | 1.83% | 0 |
| L_full_ladder | us | maximum_drawdown | -0.267116 | -0.267116 | -0.266 | 0.42% | 0.42% | 0 |
| L_full_ladder | us | calmar | 0.314233 | 0.314233 | 0.330 | 4.78% | 4.78% | 0 |
| L_full_ladder | us | expected_shortfall_5pct | -0.019573 | -0.019573 | -0.020 | 2.14% | 2.14% | 2.8e-17 |
| L_full_ladder | us | turnover | 0.706130 | 0.706130 | 0.440 | 60.48% | 60.48% | 0 |
| L_full_ladder | us | leverage | 0.785756 | 0.785756 | 0.800 | 1.78% | 1.78% | 0 |
| M_no_freeze | de | cagr | 0.066666 | 0.066666 | 0.086 | 22.48% | 22.48% | 0 |
| M_no_freeze | de | volatility | 0.147846 | 0.147846 | 0.164 | 9.85% | 9.85% | 5.6e-17 |
| M_no_freeze | de | sharpe | 0.347963 | 0.347963 | 0.440 | 20.92% | 20.92% | 5.6e-17 |
| M_no_freeze | de | maximum_drawdown | -0.369711 | -0.369711 | -0.394 | 6.16% | 6.16% | 5.6e-17 |
| M_no_freeze | de | calmar | 0.139149 | 0.139149 | 0.180 | 22.69% | 22.69% | 5.6e-17 |
| M_no_freeze | de | expected_shortfall_5pct | -0.023380 | -0.023380 | -0.025 | 6.48% | 6.48% | 4.2e-17 |
| **M_no_freeze** | **de** | **turnover** | **1.230412** | **1.230412** | **1.700** | **27.62%** | **27.62%** | **0** |
| M_no_freeze | de | leverage | 0.737387 | 0.737387 | 0.840 | 12.22% | 12.22% | 0 |
| M_no_freeze | jp | cagr | 0.017923 | 0.017923 | 0.047 | 61.87% | 61.87% | 9.4e-17 |
| M_no_freeze | jp | volatility | 0.143439 | 0.143439 | 0.171 | 16.12% | 16.12% | 2.8e-17 |
| M_no_freeze | jp | sharpe | 0.149272 | 0.149272 | 0.310 | 51.85% | 51.85% | 0 |
| M_no_freeze | jp | maximum_drawdown | -0.390394 | -0.390394 | -0.453 | 13.82% | 13.82% | 0 |
| M_no_freeze | jp | calmar | 0.054846 | 0.054846 | 0.120 | 54.30% | 54.30% | 7.6e-17 |
| M_no_freeze | jp | expected_shortfall_5pct | -0.023452 | -0.023452 | -0.026 | 9.80% | 9.80% | 5.6e-17 |
| M_no_freeze | jp | turnover | 1.556862 | 1.556862 | 0.720 | 116.23% | 116.23% | 0 |
| M_no_freeze | jp | leverage | 0.552063 | 0.552063 | 0.750 | 26.39% | 26.39% | 0 |
| M_no_freeze | us | cagr | 0.116757 | 0.116757 | 0.112 | 4.25% | 4.25% | 5.6e-17 |
| M_no_freeze | us | volatility | 0.121918 | 0.121918 | 0.131 | 6.93% | 6.93% | 8.3e-17 |
| M_no_freeze | us | sharpe | 0.753063 | 0.753063 | 0.680 | 10.74% | 10.74% | 0 |
| M_no_freeze | us | maximum_drawdown | -0.194678 | -0.194678 | -0.266 | 26.81% | 26.81% | 5.6e-17 |
| M_no_freeze | us | calmar | 0.471610 | 0.471610 | 0.330 | 42.91% | 42.91% | 0 |
| M_no_freeze | us | expected_shortfall_5pct | -0.018868 | -0.018868 | -0.020 | 5.66% | 5.66% | 6.6e-17 |
| M_no_freeze | us | turnover | 0.647285 | 0.647285 | 0.440 | 47.11% | 47.11% | 0 |
| M_no_freeze | us | leverage | 0.780269 | 0.780269 | 0.800 | 2.47% | 2.47% | 0 |

**Largest disagreement: `M_no_freeze / jp / cagr`, 9.37e-17 absolute
(0.017923050684516495 vs 0.017923050684516398), i.e. floating-point summation
order. There is no substantive disagreement anywhere in the table.**

Reproduce:
```
unset VIRTUAL_ENV && uv run python -m pytest tests/test_frequency_ladder_audit.py \
  -k headline_german_cells
```

---

## Ladder sensitivity (SENSITIVITY ANALYSIS ONLY)

**This table exists to measure how load-bearing the one free choice is. It must
not be used to pick a better ladder.** `decision_rule` item 4 of the frozen spec
forbids adjusting the ladder after seeing the result, and that prohibition
covers this table: the numbers below are a property of the experiment's
fragility, not a menu of candidates.

Six ladders — the frozen one and five alternatives a person could have written
down with the same kind of argument — each derived through the *same* pre-1990
inversion, fitted (26 extra penalties across the three markets) and scored
through the same pipeline. Each is shown full and truncated one rung, the same
L/M contrast the spec declares.

The first row is the control: it reproduces the committed run exactly, so the
harness is faithful.

| ladder | arm | US worst | DE worst | JP worst | DE turnover |
|---|---|---|---|---|---|
| **frozen 8 → 0.25** | full (L) | 60.5% F | 55.4% D | 78.4% F | 1.260 |
| **frozen 8 → 0.25** | truncated (M) | **47.1% D** | **27.6% C** | **116.2% F** | **1.230** |
| anchor 12 → 0.375 | full | 60.5% F | 62.6% F | 137.2% F | 1.318 |
| anchor 12 → 0.375 | truncated | 73.9% F | 60.3% F | 267.4% F | 1.318 |
| anchor 6 → 0.1875 | full | 60.5% F | 54.8% D | 74.9% F | 1.143 |
| anchor 6 → 0.1875 | truncated | 60.5% F | 62.6% F | 61.6% F | 1.318 |
| span 12 → 1 | full | 42.4% D | 27.6% C | 275.8% F | 1.230 |
| span 12 → 1 | truncated | 87.2% F | 28.3% C | 275.8% F | **1.699** |
| span 6 → 0.5 | full | 60.5% F | 65.4% F | 133.0% F | 1.260 |
| span 6 → 0.5 | truncated | 73.9% F | 60.3% F | 128.8% F | 1.318 |
| √2 steps, 8 → 0.25, 11 rungs | full | 53.8% D | 47.7% D | 70.0% F | 1.260 |
| √2 steps, 8 → 0.25, 11 rungs | truncated | 53.8% D | 25.9% C | 74.2% F | 1.260 |

`artifacts/audit/frequency-ladder-001/ladder-sensitivity.csv`,
`ladder-menus.csv`.

**What is robust.** The *qualitative* finding survives the free choice
completely. Germany has the best worst-cell in 11 of the 12 rows (the exception
is `span 6 → 0.5, full`, where the US is 60.5% F against Germany's 65.4% F — both
failing). German turnover improves from the v10 menu's 0.264 in **every single
row** (range 1.14 to 1.70 against Shu's 1.70). Japan is **F in all twelve rows**.
The US never reaches better than D. So "a behaviourally-specified menu fixes
Germany, costs the US, and cannot rescue Japan" is a property of the
construction, not of the particular ladder.

**What is not robust.** The headline number is not. Germany's worst-cell
deviation spans **25.9% (C) to 65.4% (F)** across six defensible ladders, and
grades C in four rows, D in three and F in five. "Germany moves from grade F to
grade C" is therefore a statement about `8, 4, 2, 1, 0.5, 0.25` **truncated**,
not about frequency-specified menus in general. The US spans 42.4% to 87.2%; the
truncation that helps Germany and the US under the frozen ladder *hurts* the US
under three of the five alternatives.

**A trap, stated explicitly so nobody falls into it.** The row `span 12 → 1,
truncated` gives German turnover **1.699** against Shu's published **1.700** — a
0.06% match on the cell this project has never reproduced. **That number must
not be adopted, quoted as a result, or used to motivate a follow-up.** It was
produced by an auditor sweeping the one free choice after seeing the answer,
which is precisely the search `decision_rule` item 4 forbids and precisely the
overfitting this repository exists to detect. Its only legitimate use is the one
it is put to here: as evidence that German turnover is a *smooth and highly
sensitive function of the ladder*, so a good German number is weak evidence for
the frequency framing. If anything, its existence strengthens the case that the
frozen ladder's 27.6% should be read as one draw from a wide distribution, not
as a measurement.

---

## Fault-injection matrix

Fourteen single-edit faults plus a control, each applied to a **copy** of the
runner or of the frozen spec in a scratch directory. Nothing under audit was
modified (see the `git status` / `git diff` section). The control copy
reproduces `artifacts/frequency-ladder/01-run/summary.csv` exactly, so the
harness is faithful. The harness itself is committed at
`artifacts/audit/frequency-ladder-001/fault_injection.py`.

| # | fault (one edit) | target | caught? | by what |
|---|---|---|---|---|
| — | control (no edit) | — | n/a | reproduces the committed summary.csv byte for byte |
| F1 | cutoff moved to 2000-01-01 | runner:36 | **CAUGHT** | re-derivation gate: derives `[0.05]*6` for us |
| F2 | cutoff filter removed entirely | runner:43 | **CAUGHT** | re-derivation gate: derives `[0.30, 1.465, 4.185, 17.5, …]` |
| F3 | mean instead of median across windows | runner:60 | **CAUGHT** | re-derivation gate |
| F4 | interval left edge instead of midpoint | runner:53 | **CAUGHT** | re-derivation gate: derives `[0.1, 1.0, 4.821, 22.0, …]` |
| F5 | `<=` inverted to `>=` in the target search | runner:56 | **CAUGHT** | re-derivation gate: derives `[0.05]*6` |
| F6 | DE and JP menus swapped at the point of use | runner:111 | **CAUGHT** | `score_grid.resolve_columns`: "penalty 17.5 is not a column of de" |
| **F7** | **arm L's menu fed while labelled arm M** | runner:111 | **SILENT** | nothing — arm M is never re-derived (F-4) |
| **F8** | **re-derivation gate dropped** | runner:80 | **SILENT** | nothing — no test asserts the gate exists or fires |
| F9 | sealed v10 states used instead of the ladder's | runner:92 | **CAUGHT** | `resolve_columns`: "penalty 0.55 is not a column of us: jm-states.csv" |
| **F10** | **arm M truncated one rung further in the spec** | spec:66-67 | **SILENT** | nothing — de goes 27.6%→20.8%, us 47.1%→67.2% |
| **F11** | **spec ladder changed 8.0 → 10.0** | spec:48 | **SILENT** | nothing — identical menu, identical numbers (F-2) |
| **F12** | **grading band C widened 40% → 80%** | spec:81 | **SILENT** | nothing — the grader is `_shu_table5.GRADE_BANDS` |
| **F13** | **turnover dropped from the spec's cell list** | spec:75 | **SILENT** | nothing — the cell list is `score_grid.CELLS` |
| **F14** | **cutoff prose changed 1990 → 2010 in the spec** | spec:51 | **SILENT** | nothing — the cutoff is a runner constant |

Reproduce (from a scratch copy, so nothing writes into a committed artifact —
see `artifacts/audit/frequency-ladder-001/README.md`):
```
SCRATCH=$(mktemp -d) && cp artifacts/audit/frequency-ladder-001/*.py "$SCRATCH"/
unset VIRTUAL_ENV && uv run python "$SCRATCH"/fault_injection.py
```

**Seven of fourteen faults passed silently, and all seven sit in the same blind
spot: the gate binds the runner to arm L of the spec, and nothing binds
anything to arm M, to the spec's declared protocol, or to the gate's own
existence.** The new tests fail on F7, F8, F10, F11 (via the midpoint-lattice
test, which documents why F11 is undetectable), F12, F13 and F14.

---

## Tests added

`tests/test_frequency_ladder_audit.py`, 16 tests. Run:

```
unset VIRTUAL_ENV && uv run python -m pytest tests/test_frequency_ladder_audit.py -q -rx
```

**Result: 15 passed, 1 xfailed (strict), 0 failed, ~24 s.**

| test | pins | catches |
|---|---|---|
| `frozen_spec_matches_its_registry_hash` | spec on disk == last hashed registry row | spec edited after the run |
| `the_amendment_is_recorded_before_the_result_event` | registry event order and timestamps | a freeze recorded after a result |
| `the_amendment_changed_only_what_it_says_it_changed` | pre-amendment spec reconstructed → freeze hash | any other byte changing during the amendment |
| `the_spec_declares_the_cells_delay_cost_and_bands_the_code_uses` | spec `[grading]`, `protocol.cells`, delay, cost vs the code | **F12, F13** |
| `every_arm_is_the_menu_the_refit_objectives_derive` | **all** arms re-derived, not just arm L | a hand-typed arm-M menu |
| `arm_m_is_arm_l_truncated_at_the_rung_the_spec_names` | M == L[:5] and the ladder is the declared one | **F10, F11** |
| `no_window_dated_on_or_after_the_cutoff_can_move_the_menu` | causality, by poisoning every post-cutoff objective | a cutoff that silently stops binding |
| `contributing_windows_end_before_the_evaluation_period` | latest contributing `training_end` < evaluation start | a window overlapping the test set |
| `the_cutoff_is_the_start_of_the_evaluation_period` | runner `CUTOFF` ≤ earliest sealed `start` | a cutoff drifting into the test set |
| `the_spec_prose_names_the_cutoff_the_runner_uses` | spec prose date == runner constant | **F14** |
| `the_runner_refuses_a_menu_that_does_not_re_derive` | the gate exists and fires before any fit | **F8** |
| `every_menu_value_is_a_midpoint_of_the_union_penalty_grid` | the menu is quantised onto the union lattice (F-2) | a menu that stops being derived from the artifact |
| `the_recorded_objective_is_penalised_and_its_slope_brackets_jumps` | `val_` == L(λ); chord slope bracketed by jump counts; jump count **not** constant across the interval (F-3) | a changed objective column or a broken concavity assumption |
| `no_contributing_window_has_a_decreasing_objective` | **strict xfail** — pins F-1 | a repaired fit, or a corrected spec, turns this into a loud failure |
| `the_headline_german_cells_reproduce_through_the_library` | DE / arm M recomputed independently == `cells.csv` | **F7**, and any drift in the committed cells |
| `the_german_comparison_changes_only_the_menu` | index/NaN-mask parity with the sealed states; menus disjoint | a German comparison that quietly changed something else |

The five spec-level silent faults were each replayed against these tests to
confirm they now fail:
`artifacts/audit/frequency-ladder-001/verify_tests_catch_faults.py`.

```
F10_armM_truncated_further       arm_m_is_arm_l_truncated…            CAUGHT
F11_ladder_changed               arm_m_is_arm_l_truncated…            CAUGHT
F12_grading_bands_widened        the_spec_declares_the_cells…         CAUGHT
F13_turnover_dropped_from_cells  the_spec_declares_the_cells…         CAUGHT
F14_cutoff_prose_changed         the_spec_prose_names_the_cutoff…     CAUGHT
```

**Not pinned, deliberately:** the F-2 lattice problem and the F-3 inversion bias
are *design* findings, not regressions. The tests record them (the midpoint
test, and the `jumps[0] > jumps[-1]` assertion) so that a reader cannot restate
"the ladder is the only free choice" or "the inversion is exact" without
tripping over a test that says otherwise, but neither can be fixed by a test.

---

## Nothing under audit was modified

Every fault was applied to a **copy** in a scratch directory. The spec, the
runner, the map script, the scorer, the sealed configs and every committed
artifact of the run are untouched.

```
$ git status --short
 M .gitignore
 M docs/audit/heldout-delay-001-audit.md      <- NOT MINE, see below
?? artifacts/audit/
?? docs/audit/frequency-ladder-001-audit.md
?? tests/test_frequency_ladder_audit.py

$ git diff --stat
 .gitignore                            | 4 ++++
 docs/audit/heldout-delay-001-audit.md | 4 ++--
 2 files changed, 6 insertions(+), 2 deletions(-)

$ git diff -- research/ scripts/ src/ artifacts/frequency-ladder \
      artifacts/jm-residual artifacts/penalty-frequency research-calibrated-v10.toml
(empty — nothing under audit changed)

$ sha256sum research/frequency-ladder-001.toml scripts/run_frequency_ladder.py \
      scripts/diagnose_penalty_frequency_map.py scripts/score_grid.py
4dc4832522933731020133ed4f34e27309951389625e6cc6cc6895865a71890a  research/frequency-ladder-001.toml
d274b141dd8d4647c93daa6cc64a9532c5e1bb1747e025851114301b8067313f  scripts/run_frequency_ladder.py
8888ed02861e8a672479d70e2247e1cf291bea08d8ebe9eee0fba4646286ed48  scripts/diagnose_penalty_frequency_map.py
54caeeae58fd13ed6f9331f706101beaf53a23127ffea21386aab6297835db59  scripts/score_grid.py

$ md5sum artifacts/frequency-ladder/01-run/*.csv
ab998be51296f9510629585b61882ef4  cells.csv
01359216df16d235a80b821ed9ef6139  states-de.csv
fc7a1ba7b200d85bcafef247c0fa329d  states-jp.csv
e68b2886fcde158990fb24b7dbfd25f9  states-us.csv
32c3430b9fb41dde82ddd1babf12c46a  summary.csv
```

The spec still hashes to its `SPEC_AMENDED_BEFORE_RESULTS` registry hash, so
neither the audit nor anything else has touched it.

What I did add:

* `docs/audit/frequency-ladder-001-audit.md` — this report (new file);
* `tests/test_frequency_ladder_audit.py` — 16 regression tests (new file);
* `artifacts/audit/frequency-ladder-001/` — the recomputations, the fault
  harness and the evidence files this report cites (new directory);
* `.gitignore` — one negation, `!artifacts/audit/` plus a re-exclusion of the
  large refitted state matrices, following the existing `artifacts/jm-residual`
  precedent, so the small evidence tables are committable.

### One thing I found and did not touch

`docs/audit/heldout-delay-001-audit.md` — the **previous** audit's report, not
part of this audit — is modified in the working tree and the modification is
text corruption, not content:

```
-**The numbers are real; the conclusion is not warranted.** All fifty-four cells
+**The numbers are real; the conclusion is not warranted.** All fifduplicatingty-four cells
...
-the correct three columns and the delay-1 column correctly duplicating Table 4.
+the correct three columns and the delay-1 column correctly  Table 4.
```

The word `duplicating` has been cut from one sentence and spliced into the
middle of `fifty-four` in another — the signature of a botched find-and-replace.
Its mtime (22:06:34) falls inside this audit's session, but I never opened that
file; no command I ran writes to `docs/`, and my own writes are the two new
files listed above. I have left it exactly as found rather than quietly
reverting someone else's working tree. **Recommended:**
`git checkout -- docs/audit/heldout-delay-001-audit.md`, after confirming
nobody intended those bytes.
