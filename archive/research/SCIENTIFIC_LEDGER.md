# Scientific Development Ledger

This ledger records the mathematical ideas, frozen experiments, corrections,
and evidence produced in this repository. It distinguishes a proposed theory
from an implemented mechanism and both from market-performance evidence.
Corrections are appended or explicitly marked; historical claims are not
silently upgraded. Generated artifacts remain ignored and are identified by
run ID and inventory hash.

## Current mathematical baseline

The canonical fixed Jump Model (JM) uses two fitted centers and solves

\[
\min_{\Theta,s}\sum_t \tfrac12\lVert x_t-\theta_{s_t}\rVert^2
+\lambda\sum_{t\ge1}\mathbf 1\{s_t\ne s_{t-1}\}.
\]

The active proxy protocol uses v7 features `DD10`, `Sortino20`, and
`Sortino60`. `DD10` is downside deviation, not drawdown:
`sqrt(EWM mean(min(excess return, 0)^2))` with half-life 10. The protocol uses
a 3,000-observation training prefix; `StandardScaler`; Jan/Jul refits; raw
lambdas `[0, 5, 15, 35, 70, 150, 300, 600, 1200]`; monthly trailing-eight-year
Sharpe selection; a one-trading-day execution delay, where a signal after `t`
first earns at `t+2`; and 10 bps one-way cost. The sample is capped at
2023-12-31.

## Mathematical developments

### Exact time-varying-transition decoder

The repository generalized the discrete JM decoder to

\[
J(s\mid\hat\Theta)=\sum_t L_t(s_t)
+\sum_{t\ge1}C_t(s_{t-1},s_t),
\]

where every arrival day may have its own transition matrix. Dynamic
programming remains exact in \(O(TK^2)\). A constant off-diagonal sequence
exactly nests the reference fixed JM. Brute-force, prefix, limiting-case, and
toy-path oracles are in `tests/test_tv_jump.py`.

### Evidence-adaptive arrival penalty

`adaptive-confidence-001` introduced

\[
C_t(i,j)=\lambda_0\exp\!\left[
-\beta\tanh\!\left(\frac{[L_t(i)-L_t(j)]_+}{q_{\rm train}}\right)
\right],\quad i\ne j,
\]

with zero diagonal. Its established properties are:

- \(\beta=0\) exactly recovers fixed JM.
- Without destination loss advantage, the transition still costs
  \(\lambda_0\).
- For positive evidence,
  \(\lambda_0e^{-\beta}\le C_t(i,j)<\lambda_0\).
- `log(2)` and `log(4)` therefore cap the discount at one half and one
  quarter of the fixed penalty.
- Scaling every loss and \(q_{\rm train}\) by the same positive constant
  leaves the penalty unchanged.

The market study reused fixed-v7 scalers and fitted centers and changed only
online decoding. It did **not** jointly fit adaptive centers, learn beta, or
prove a latency/false-switch theorem.

### Binary value-difference recursion and same-day amplification

For two states, define

\[
g_t=L_t(1)-L_t(0),\qquad
a_t=C_t(0,1),\qquad b_t=C_t(1,0),
\]

and let \(d_t=V_t(1)-V_t(0)\) be the dynamic-programming value
difference. The exact recursion is

\[
d_t=g_t+\operatorname{clip}(d_{t-1},-b_t,a_t).
\]

Thus fixed JM is an evidence accumulator with a hysteresis interval. In the
arrival-adaptive model, current evidence enters once through \(g_t\) and
again by moving the active barrier. When the evidence-supported barrier is
active,

\[
\left|\frac{\partial d_t}{\partial g_t}\right|
=1+\frac{\beta C_t}{q_{\rm train}}
\operatorname{sech}^2(|g_t|/q_{\rm train})>1.
\]

This establishes a local one-observation amplification mechanism; it is not a
probabilistic false-switch theorem. It explains why the arrival formula can
react too strongly to an isolated gap and motivated a lagged-evidence
alternative.

### Lagged-evidence transition penalty

The frozen candidate in `lagged-evidence-mechanism-001` is

\[
C_t^{\rm lag}(i,j)=\lambda_0\exp\!\left[
-\beta\tanh\!\left(
\frac{[L_{t-1}(i)-L_{t-1}(j)]_+}{q_{\rm train}}
\right)\right],\quad i\ne j,
\]

with zero diagonal and the fixed-lambda matrix at \(t=0\). Its objective is

\[
J_{\rm lag}(s)=\sum_tL_t(s_t)
+\sum_{t\ge1}C_t^{\rm lag}(s_{t-1},s_t).
\]

Established code-level properties are:

- \(\beta=0\) is bit-exact fixed JM.
- \(\lambda_0e^{-\beta}\le C_t^{\rm lag}(i,j)\le\lambda_0\).
- Common positive scaling of all losses and \(q_{\rm train}\) leaves the
  penalty unchanged.
- The additive objective remains exactly solvable in \(O(TK^2)\).
- Brute-force objective, prefix-invariance, direction, and toy-path tests pass.
- On locked toys, arrival switches on an isolated shock and alternating noise,
  lagged does not; on a persistent shift, arrival/fixed/lagged first switch at
  indices `3/6/4`, so lagging retains part of the latency gain.

The first frozen real-market mechanism run was interrupted and registered as
`INVALID_IMPLEMENTATION` before any market artifact, event table, summary, or
conclusion existed. It did not contain future look-ahead, but it violated its
own performance-free protocol by reading return columns and reconstructing
fits instead of consuming only the sealed scaler, centers, and
\(q_{\rm train}\). It also passed the mechanical prerequisite as a literal
Boolean and had a verifier that did not replay candidate states or event
counts. No state-path or mechanism conclusion from that partial execution is
accepted. The correction retains the same mathematical formula and decision
inequalities, replaces refitting with sealed-parameter generation, derives the
mechanical Boolean from executable evidence, aligns boundary-switch counting,
and requires a full independent replay before refreeze and rerun.

The implementation is causal. However, on Jan/Jul refit dates the previous-row
loss is recomputed under scaler and centers fit through the current row.
Therefore “lagged-evidence” is precise, while strict
\(\mathcal F_{t-1}\)-predictability is not claimed. Changing activation to
old centers would be a different protocol and would break exact beta-zero
nesting with the sealed v7 parent.

The corrected frozen run
`lagged-evidence-6f964f5724b2-26cbca8871be-d173ca32c86f` subsequently completed
and was independently reconstructed across US, DE, and JP without reading any
choice, trade, return, or performance file. At `beta=log(4)`, arrival versus
lagged candidate-path whipsaws were `6→2` in US, `6→3` in DE, and `5→1` in JP;
pooled whipsaws were therefore `17→6`. JP candidate-path switches fell
`266→258`, while `11` persistent events still switched before fixed JM. Only
`log(4)` passed the frozen rule. This is accepted performance-free mechanism
support, not evidence of improved Sharpe, drawdown, turnover, or profit.

A limited primary-source literature scan found adjacent work on arbitrary
transition matrices, state-specific penalties, time-varying penalties, and
lagged-observable transition probabilities, but not this exact bounded
prior-loss-gap regularizer. The safe description is a candidate mathematical
contribution within statistical JM, not a global novelty claim.

### Binary directed-cost identity

For constant binary transition costs \(a=C(0,1)\) and \(b=C(1,0)\),

\[
aN_{01}+bN_{10}
=\tfrac12(a+b)N_{\rm switch}
+\tfrac12(a-b)(s_T-s_0).
\]

Therefore a constant zero-diagonal asymmetric matrix is a symmetric switching
penalty plus a boundary bias, not independent bull/bear duration control. With
time-varying directed costs the per-transition decomposition still holds, but
the antisymmetric term need not telescope. The active model is not a
semi-Markov duration model.

### Reliability-gated extension (diagnostic not supportive)

The open theory is that evidence discounts should be trusted only when fitted
states are distinguishable on information available at the refit. The frozen
training-prefix reliability statistic in `adaptive-separation-001` is

\[
D=\lVert\mu_0-\mu_1\rVert,
\qquad
\rho_k=\operatorname{median}_{u\in A_k}\lVert z_u-\mu_k\rVert,
\qquad
R_{\rm train}=\frac{D}{D+\rho_0+\rho_1},
\]

where \(A_k\) contains training rows strictly nearer center \(k\). It is
bounded, label-symmetric, and invariant to translation, orthogonal coordinate
changes, and common positive distance scaling. The candidate next model is

\[
C_t(i,j)=\lambda_0\exp[-\beta R_{\rm train}
\tanh([L_t(i)-L_t(j)]_+/q_{\rm train})].
\]

The completed `adaptive-separation-001` diagnostic did not justify this
gate: all leave-one-market-out fits were invalid under the frozen optimizer
criterion, and the descriptive reliability ordering did not separate
whipsaws from persistent events consistently. It is closed as the next model
unless new performance-free evidence supplies a different reliability
statistic and a separately frozen study.

## Experiment history

### 2026-06-23 — minute prototype (archived)

- Tried the duration mapping \(\lambda=\log(d-1)\) and hand-set additive or
  multiplicative adaptive penalties: noise raised the cost and shocks lowered
  it.
- Exact time-varying DP and synthetic paths worked.
- One-shot real minute paths were 98–99.8% identical and transaction costs
  overwhelmed the strategies. Loss and penalty scales were not calibrated to
  the later daily protocol.
- Status: useful prototype, no market evidence. Sources remain in Git history;
  relevant commits include `3f7426e`, `1efbfb1`,
  `6712c2d`, and `fe73275`.

### 2026-07-08 — approximate daily baseline and P1 (archived)

- P0 attempted the paper features on public daily data but differed in
  fit/validation cadence, risk-free rate, HMM comparability, and data
  reproducibility. Its reported Sharpe values are not a replication result.
- P1 tried
  \(\lambda_t=\min\{\exp(b_0+b_1z_t),5000\}\), with standardized clipped
  DD10 and \(b_1=0\) nesting fixed JM. Historical deltas were
  `-0.08/-0.14/+0.02` for US/DE/JP; a 0.10 complexity rent selected the fixed
  case in most blocks.
- Status: historical null on a superseded protocol, not current evidence.
  Sources remain in Git history; commits include `c90abad`,
  `87e28a0`, `98facb5`, and `d7bdba8`.

### 2026-07-08/10 — P2 asymmetric exit/re-entry (withdrawn)

- P2a used different `0→1` and `1→0` costs around a common lambda.
- The binary identity above showed that the intended state-specific
  persistence interpretation was false.
- P2b also mapped a duration log difference onto an incompatible penalty
  scale.
- Status: the historical numbers describe threshold/boundary shifts only; the
  persistence claim was withdrawn. See archived commits `e058361` and
  `2b5cbad`.

### 2026-07-12 — `fixed-baselines-001-v7` (complete)

- Frozen the canonical causal proxy baseline described above.
- All 18 grid-boundary checks passed and 27 metric rows were independently
  reproduced, but fixed JM failed the directional replication gate in all
  three markets.
- Status: **proxy non-replication**, not a refutation of Shu et al., because
  the exact long paper sample and source definitions were unavailable.
- Artifact:
  `fixed-baselines-8adb330565d6-3636939b525d-e9614112b234`.

### 2026-07-14 — 4,000-row window sensitivity (complete, boundary failed)

- Changed only the JM fit window from 3,000 to 4,000 observations.
- The upper-lambda check failed in 8 of 9 market/delay rows, so metrics and
  bootstrap results stayed sealed.
- Status: candidate-domain coverage failure; Sharpe improvement or harm is
  unknown. Artifact:
  `jm-window-cd9ac0b9d7a6-3636939b525d-6c19911401ad`.

### 2026-07-15 — Table-3 grid attribution (withdrawn before run)

- Proposed restricting candidates to illustrative paper values to test grid
  attribution.
- No model was fit and no output was produced.
- Status: withdrawn before run. Its immutable identity is the registry event
  with spec hash `0e7edadc7c09...`. The later fixed-baseline audit reused
  `research/hyperparameter-grid-attribution.toml`, so that current filename
  must not be treated as the withdrawn Table-3 contract.

### 2026-07-15/16 — persistence-calibrated candidate search (complete)

- Developed a pre-OOS behavior-only domain search using occupancy,
  transition counts, duplicate-path collapse, and log-spaced switch rates.
- It selected nine-candidate JM and HMM grids without opening outer Sharpe or
  P&L.
- Status: completed calibration procedure, not a model or performance result.

### 2026-07-16 — behavior-calibrated grid evaluation (boundary failed)

- Evaluated whether candidate-domain calibration alone repaired v7.
- Upper-edge selection failed 16 of 18 locked checks; metrics remained sealed.
- Status: domain still under-covered. Artifact:
  `grid-eval-684fb4d81a9a-3636939b525d-9c81579e9de4`.

### 2026-07-17 — `adaptive-confidence-001` (complete)

- The first `q_train` definition based on pairwise loss gaps was withdrawn
  before adaptive states or metrics when empty fitted states made it
  undefined.
- The corrected scale is the raw MAD of every finite state-loss entry on the
  exact 3,000-row training prefix, with no epsilon or future-data fallback.
- Beta scenarios were exactly `0`, `log(2)`, and `log(4)` on the unchanged v7
  grid. Beta zero matched parent candidate states, choices, signals, trades,
  and metrics exactly.
- US switches fell `21→18/19`, but Sharpe fell by `0.06/0.07`.
- DE switches fell `32→24/14`, turnover fell by `0.50/1.12`, and Sharpe
  rose by `0.11/0.11`; drawdown deltas were only floating-point negatives
  of order \(10^{-16}\).
- JP switches rose `13→23/27`; the outcomes were mixed across Sharpe and worse
  on drawdown/turnover/switching.
- The exact frozen decision rule classified the study `not_supported`; the
  penalty mechanism itself was operational. This is development-sample
  evidence and supports no performance claim.
- Artifact:
  `adaptive-confidence-1b0c327b2db4-3636939b525d-864d671cf973`; source
  commits `edfc616` through `1f522ab`.

### Proposed `adaptive-confidence-002` tolerance correction

- Treating \(|\Delta\mathrm{MDD}|\le10^{-9}\) as zero would reclassify the DE
  rows and the study label to `mixed` without changing any state, trade, return,
  or metric.
- Status: recorded but not run. It is a reporting-rule correction, not a
  mathematical or performance contribution.

### 2026-07-17 — `adaptive-separation-001` (complete, inconclusive)

- Tests whether causal training-prefix reliability predicts reversal of an
  arrival-discount-attributable fixed-lambda candidate switch over 20 emitted
  signal days.
- Defines an event by exact terminal DP predecessor plus an arrival-only
  ablation: restoring only day \(t\)'s off-diagonal penalty to fixed lambda
  must return the terminal state to its source.
- Uses leave-one-market-out prediction and never reads returns, Sharpe, MDD,
  positions, choices, or selected-path performance.
- Its first source lock was withdrawn before computation because it placed
  fixed-v7 `features.csv` in the adaptive artifact. The corrected lock binds
  features and adaptive state/refit files to their separate sealed parents.
- A second pre-result lock made `log_discount = log(lambda/C)`, coefficient
  tolerance, exact-tie handling, full-rank, and gradient validity explicit.
- The first 56-event run was invalidated: concrete event inspection exposed
  dates before the registered v7 outer samples. The corrected lock added only
  starts US 2007-12-04, DE 2008-01-03, and JP 2009-05-07. That correction is
  explicitly post-run; no formula, horizon, estimator, or decision rule moved.
- The corrected run admitted 42 exact events: US 14 (6 whipsaws), DE 15 (6),
  and JP 13 (5). All had valid reliability geometry and no exact DP tie.
- Audit maxima were \(4.44\times10^{-16}\) for the penalty formula,
  \(2.22\times10^{-16}\) for log discount, \(9.10\times10^{-13}\) for
  fixed objective, and exactly zero for reconstructed \(q_{train}\).
- All leave-one-market-out fits failed the locked \(10^{-9}\) gradient
  criterion (US \(1.99\times10^{-9}\), DE \(2.58\times10^{-9}\), JP
  \(1.61\times10^{-8}\)); the official result is `inconclusive`.
- The mechanism explanation was not borne out descriptively. Median valid
  refit reliability was DE `0.55` and JP `0.56`, while DE whipsaw
  events were slightly more separated than persistent events
  (`0.57` vs `0.56`). JP had only a small difference in the proposed
  direction (`0.56` vs `0.57`).
- Status: completed mechanism diagnostic; this reliability gate is not
  justified and was not sent to a P&L test. Artifact:
  `adaptive-separation-813f66912526-26cbca8871be-fefc608b9081`.

### 2026-07-17 — `fixed-baseline-assumption-audit-001` (complete)

- A clean detached-worktree rerun at `1f522ab` repeated the original fixed-v7
  JM/HMM pipeline from the canonical through-2023 manifest. Every scientific
  file was byte-identical to the sealed parent, so the original proxy result is
  reproducible.
- Shu et al. v3 does not disclose the complete JM-lambda or HMM-smoothing
  cross-validation grids. Table 3 shows illustrative fixed values
  `{0,5,15,35,70,150}` for JM and `{0,2,4,8,20}` for HMM; historical v1 used
  JM `{10,22,50,100,220,500,1000}` under a materially different design. The
  official package accepts user-supplied penalties and contains no paper
  backtest or HMM calibration pipeline.
- The audit fit only the seven historical-v1 JM lambdas missing from v7. It
  reused every overlapping parent state and did not refit the HMM performance
  path. A performance-free HMM check compared internal KMeans `n_init=10`
  with a literal one-start-per-outer-seed interpretation on 95 windows:
  terminal state changed in `0/95`, while the winning seed changed in `54/95`.
  This is sampled terminal-state stability, not full-path equivalence.
- The turnover display bug was isolated. Paper turnover is
  \(0.5\times252\times\operatorname{mean}|\Delta position|\); combined
  annualized traded notional is exactly twice that. The old display reported
  the second quantity as turnover. Trading cost and P&L were already correct
  at \(0.001|\Delta position|\), so the fix changes reporting only.
- Locally added candidates were binding in primary-delay OOS monthly choices:
  about `41%/47%/58%` of JM choices and `55%/50%/87%` of HMM choices in
  US/DE/JP lay outside the Table-3-visible sets.

Primary-delay fixed-JM evidence on the matched core sample is:

| Market | Expanded control: Sharpe / MDD / turnover / cash / switches | Table-3-visible delta: Sharpe / MDD improvement / turnover / cash / switches |
| --- | --- | --- |
| US | `0.57 / -0.34 / 0.66 / 0.21 / 21` | `+0.01 / 0 / 0 / +0.06 / 0` |
| DE | `0.19 / -0.39 / 1.00 / 0.18 / 32` | `-0.13 / +0.02 / +0.56 / +0.14 / +18` |
| JP | `0.33 / -0.32 / 0.46 / 0.28 / 13` | `-0.32 / -0.02 / +2.43 / +0.36 / +69` |

- Historical-v1 and source-union JM grids were mixed: they raised DE Sharpe
  by `0.11` and `0.06` while lowering US and JP Sharpe. Restricting to
  Table-3-visible values therefore does not rescue the fixed baseline and is
  especially harmful in JP.
- The 5% upper-boundary rule is local, absent from the paper, and does not
  enter states, trades, costs, or returns. It failed `94/216` descriptive
  rows in this audit and sealed no metric.
- Full-window versus partial-window startup on the source-union HMM grid
  changed no selected path or metric. All `7,995` multi-candidate score ties
  were exact floating-point ties, not tolerance-only ties. Higher tie-breaking
  changed the US fixed-JM path in one core family, so tie semantics are a real
  numerical-protocol sensitivity but not a selected winner.
- Every overall local market gate remained false. One DE component
  (`fixed-JM Sharpe > HMM Sharpe`) flipped under the `table3_both` cell, so the
  frozen label is `core_grid_sensitive`: the local conclusion depends on
  tested candidate sets, while the actual final-v3 grids remain
  underidentified. This is exploratory development evidence, not a paper
  replication or performance claim.
- Concrete timelines were independently reconstructed from score to active
  candidate, state, signal, t+2 position, turnover, and 10-bps cost for all 33
  pair/market events. Two intermediate runs were invalidated before completion
  registration for non-governing timeline provenance and missing diagnostic
  evidence; the final artifact is self-contained and independently replayed.
- Artifact:
  `fixed-baseline-assumption-audit-79c94852c8fd-3636939b525d-4cc8cdbccd14`.

### 2026-07-18 — `endpoint-grid-audit-001` (complete)

- Tested one source-derived endpoint per fixed-model family after exact
  current-code reproduction of the sealed base behavior: JM
  `362.04` (next candidate `512` invalid) and HMM `1249` (next
  candidate `1250` invalid). No wider grid or winner search was permitted.
- Two pre-result executions were invalidated before accounting or metrics. The
  first corrected an optional parent metadata field absent from the historical
  schema. The second restored the default pandas parser used by the sealed
  witness producer after a round-trip parser changed `32,199` US input cells by
  at most `2.22e-16`. Neither correction changed a scientific rule or spec hash.
- Run `endpoint-grid-audit-05e9d08f619b-77b30ef98fa0-24ca06c297e8`
  passed the US smoke, three-market exact base selection-behavior parity, all
  formula/accounting checks, and independent verification twice. It accessed
  no post-2023 data and authorizes no performance or paper-replication claim.
- At primary delay 1, JM endpoint-minus-base changes were: US Sharpe
  `-0.07`, MDD `0`, turnover `+0.12`, cash `-0.01`, switches `+4`;
  DE Sharpe `+0.08`, MDD `-0.09`, turnover `-0.12`, cash
  `-0.04`, switches `-4`; JP all five deltas `0`. The HMM endpoint changed
  none of the five primary metrics in any market.
- Concrete causal traces linked US choice/signal `2022-01-31` to t+2
  position/trade `2022-02-02`, and DE choice/signal `2019-01-31` to t+2
  position/trade `2019-02-04`. JP changed nine monthly choices at the primary
  delay but their selected state/signal paths were identical, so no position or
  trade changed.
- Cell D failed the locked three-market rescue. US passed all conditions; DE
  failed JM-versus-buy-and-hold Sharpe; JP failed all three. Primary JM endpoint
  selection remained above the descriptive 5% rate in US/DE/JP at
  `6.70%/28.50%/5.08%`, so the finite JM optimum is still unidentified.
- Conclusion: candidate-grid truncation is a real and material source of model
  sensitivity, but this predefined endpoint extension does not explain or
  rescue the three-market proxy non-replication. The local 5% rule only reports
  unresolved truncation; it never changed or censored a metric.

### 2026-07-18 — reconciled grid inventory

The exact candidate sets already tested are now recorded in one place:

- Paper-v3 Table-3-visible values only: JM
  `[0, 5, 15, 35, 70, 150]`; HMM `[0, 2, 4, 8, 20]`. These are
  illustrations, not the undisclosed full final-v3 validation grids.
- Canonical repo v7: JM `[0, 5, 15, 35, 70, 150, 300, 600, 1200]`; HMM
  `[0, 2, 4, 6, 8, 10, 20, 40, 80, 160, 320, 640, 1280, 2560]`.
- Historical-v1 JM: `[10, 22, 50, 100, 220, 500, 1000]`.
- Source-union audit: JM
  `[0, 5, 10, 15, 22, 35, 50, 70, 100, 150, 220, 500, 1000]`; HMM
  `[0, 2, 4, 6, 8, 20]`.
- Performance-free behavior domain: JM `0` plus
  `2^(j/2), j=-8,...,22` (32 candidates); `0` through
  `362.04` were globally valid and `512` onward failed the
  occupancy/transition rule. HMM tested every integer `k=0,...,2560`; `0..1249`
  were globally valid and `1250+` failed. HMM candidates are smoothings of one
  raw HMM path, not separate Gaussian-HMM fits.
- Behavior-selected base grids: JM
  `[0, 0.35, 1, 5.66, 16, 32, 64,
  181.02, 256]`; HMM `[0, 3, 9, 32, 54, 114, 166, 402, 1115]`.
  The endpoint audit added only JM `362.04` and HMM `1249`.
- Adaptive mechanism scenarios used the unchanged v7 lambda grid and exactly
  `beta=[0, log(2), log(4)]`; the lagged P&L readout carried forward only the
  performance-free-selected `beta=log(4)`.

No tested grid recovered the paper ordering across all three proxy markets.
The complete final-v3 paper grids remain unidentified.

### 2026-07-18 — `lagged-evidence-performance-001` (complete)

- The mathematical rule replaces arrival evidence by the previous observation:
  `C_t(i,j)=lambda*exp(-beta*tanh([L_(t-1)(i)-L_(t-1)(j)]_+/q_train))` off the
  diagonal, with a zero diagonal. On Jan/Jul refit dates, that previous loss is
  recomputed under parameters fitted through `t`; the sealed rule is causal at
  `t`, but is not strictly `F_(t-1)`-measurable on those dates.
- The extension preserves exact dynamic programming. For `beta>=0`,
  `lambda*exp(-beta) <= C_t(i,j) <= lambda`; `beta=0` is exactly fixed JM; and
  for a positive loss gap the penalty derivative is non-positive. For any path at one fixed candidate lambda
  with `N` switches, its adaptive objective lies between the fixed objective
  minus `N*lambda*(1-exp(-beta))` and the fixed objective. These are algebraic
  properties, not a delay, whipsaw, or profit theorem.
- The study reused sealed v7 features, scalers, centers and candidate states,
  selected lambda separately each month over the previous eight calendar years,
  used the exact parent t+2 sample, 10-bps one-way costs, and paper turnover
  `0.5*252*mean(abs(position change))`. It accessed no post-2023 data.
- Run `lagged-pnl-bad599271e2d-643dd3e6d96f-be70588256b2` passed source locks,
  beta-zero parity, US smoke, full replay of choices/signals/positions/trades,
  accounting identities, cutoff checks, artifact allowlists, plus completion-time and separate CLI
  source replays. One earlier result artifact was invalidated before inspection
  when the verifier's optional error reporter attempted to subtract booleans;
  the reproduced implementation-only fix changed the run identity.

| Market | Delta Sharpe | Delta MDD | Delta turnover | Delta cash | Switches fixed→lagged |
| --- | ---: | ---: | ---: | ---: | ---: |
| US | `+0.02` | `+0.00` | `-0.19` | `+0.04` | `21→15` |
| DE | `+0.17` | `0` within `1e-9` | `-0.75` | `-0.05` | `32→8` |
| JP | `+0.08` | `+0.00` | `+0.70` | `+0.00` | `13→33` |

- The frozen primary rule was `supported`: mean Delta Sharpe was `+0.09`
  and all three market deltas were positive. The readout improved the net-Sharpe/
  switching combination in US and DE, but not uniformly: JP turnover and switches
  increased substantially.
- It did not restore the paper's all-market ordering on the proxy. Lagged Sharpe
  was `0.587` in US versus proxy HMM `0.654`; `0.338` in DE versus proxy B&H
  `0.290` and HMM `0.008`; and `0.413` in JP versus proxy B&H `0.545`. It was the
  strongest local comparator only in DE.
- The adaptive upper lambda was selected in `8.81%`, `45.31%`, and `30.11%`
  of US/DE/JP OOS months. The descriptive 5% rule never gated a metric, but this
  concentration means the finite optimum remains unidentified. The positive
  readout is conditional on the frozen grid and repeatedly inspected development
  sample; it is not an out-of-sample performance or paper-replication claim.

### 2026-07-18 — `lagged-selection-attribution-001` (complete)

- This was frozen only after the lagged P&L result was known, so it is a
  post-result mechanical diagnostic. It replayed the exact parent states and
  monthly choices without refitting, regenerating states, rerunning CV,
  expanding the grid, or reading post-2023 data.
- The four cells were fixed states/fixed choices (`FF`), fixed states/lagged
  choices (`FL`), lagged states/fixed choices (`LF`), and lagged states/lagged
  choices (`LL`). `FF` and `LL` reproduced the parent trades exactly.
- For each nonlinear metric `M`, the two-factor interaction was
  `M_LL-M_LF-M_FL+M_FF`. Shapley allocations averaged the path and choice
  marginal changes across both possible baselines and exactly summed to
  `M_LL-M_FF`; they are accounting identities, not causal estimands.
- Equal-market mean Delta Sharpe was `+0.09`. The direct path effect at
  fixed choices was `-0.14`, the direct choice effect at fixed paths was
  `+0.02`, and interaction was `+0.21`. Shapley allocation was
  `-0.03` to candidate-state family and `+0.13` to monthly choices.
- Market Sharpe Shapley allocations (path / choice) were US
  `-0.10 / +0.12`, DE `+0.01 / +0.16`, and JP
  `-0.02 / +0.10`. Thus the positive parent readout is not evidence
  that lagged candidate paths alone were uniformly better.
- Mean turnover Shapley allocation was `+0.18` to paths and `-0.26` to
  choices. JP was adverse on both axes (`+0.28 / +0.42`), so its
  higher turnover cannot be assigned solely to either path or selector.
- The verifier reconstructed source schedules, all four signal/trade paths,
  t+2 positions, 10-bps costs, paper turnover, and both Shapley sides. A
  separate replay passed with maximum error exactly `0.0`. Concrete traces
  retain choice→state→signal→position→signed-trade transitions.
- The first completed implementation was invalidated after independent audit
  found that `NaN != NaN` mislabeled each cell's uninitialized first row as a
  trade example. Metrics and attribution were unaffected. The corrected run
  requires both signed trades to be finite and has a direct regression test.
- Run:
  `lagged-attribution-73a5995c487e-52854fc3c22a-197915169632`.
  Result: `diagnostic_complete`; no cell winner, performance claim, causal
  claim, or paper-replication claim is authorized.

### Mathematical queue after the lagged readout

- The frozen 2x2 diagnostic is complete. Its large interaction and adverse
  direct path effect make a selector-free theorem or mechanism test more
  important than another same-sample P&L sweep.
- First validation priority: untouched markets or genuinely prospective data;
  more optimization on US/DE/JP through 2023 cannot create confirmation.
- If a new transition model is later justified, the ranked candidates are:
  robustly cap the signed loss gap, require two-day evidence confirmation, then
  add a soft semi-Markov reversal surcharge. A diagonal feature metric is a
  lower-priority geometry test. None has been run or selected by P&L.

### 2026-07-18 — `balanced-lagged-mechanism-001` (complete, not supported)

- Frozen post-result candidate, motivated by the lagged upper-lambda
  concentration and the large path/choice interaction: for \(i\ne j\),
  \(C_t(i,j)=\lambda_0\bigl[1-(1-e^{-\beta})\tanh((L_{t-1}(i)-L_{t-1}(j))/q_{\rm train})\bigr]\),
  zero diagonal, fixed matrix at \(t=0\). The pair-sum identity
  \(C_t(i,j)+C_t(j,i)=2\lambda_0\) preserves the binary DP
  hysteresis-interval width exactly, while signed lagged evidence still tilts
  transition direction. \(\beta\in\{0,\log 4\}\) with \(\beta=0\) bit-exact
  fixed JM; only `log4` is decision-bearing and there was no new beta search.
- Three pre-run freezes were withdrawn before any market path: a stale
  timestamp with an under-specified matched-category algorithm, an `h=20`
  exposure credit hole, and a 65-hex transcription typo in the parent
  inventory hash (corrected to the sealed value already recorded by both
  accepted lagged specs). The final freeze is
  `a7d9914ca1a8ab8660cd262c1f759c2e6b25972062536dc151492c8b92ff4cfc`; no
  scientific rule changed across the last refreeze.
- Two response-independent implementation corrections preceded the first
  passing US smoke: the parent `run.json` is metadata outside the sealed
  inventory (matching every prior study's source lock), and the
  stale-vs-current refit probe now skips lambdas whose terminal loss row
  lacks a sealed center, because a missing center saturates the signed
  evidence at \(\pm 1\) and makes the arrival penalty parameter-independent
  by construction. At the genuine second US refit, 6 of 8 event lambdas were
  informative and all 6 were distinct (minimum distance `2.36e-3`).
- US smoke passed: parent lagged parity and beta-zero parity exact over 558
  cells each, formula, pair sum, and bounds exact through the second refit,
  and prefix plus future-mutation invariance held.
- Run `balanced-lagged-a7d9914ca1a8-643dd3e6d96f-17961bfd667f` completed with
  33 own events and 13 matched anchors, passed pre- and post-completion
  verification, and passed a separate CLI replay that reconstructed all 108
  candidate paths across the three markets.
- Result `not_supported`. Passed: mechanics, both nontriviality conditions,
  anchor coverage after the response-independent `t+40` same-refit filter,
  market switch guard, matched market whipsaw, latency by market, pooled
  latency retention `0.875`, and both lock-in guards with zero own or matched
  unconfirmed-persistent responses. Failed: own-market whipsaw (JP 2 versus
  1), own pooled whipsaw (7 versus 6, not strictly lower), and matched pooled
  whipsaw (5 versus 5, not strictly lower).
- Interpretation stays mechanism-level: preserving the fixed pair-average
  transition scale retained most early confirmations and produced no lock-in,
  but it did not reduce discount-attributable reversals on this repeatedly
  inspected sample. No P&L, provider access, post-2023 data, or monthly
  selection was touched, and no P&L study is authorized either way.

## Claims that remain open

- No theorem yet bounds detection delay or false-switch probability for the
  adaptive penalty.
- Switch counts are not true whipsaw labels; the 20-day reversal outcome is an
  explicit diagnostic convention, not latent-regime ground truth.
- Beta is scenario-fixed, not estimated.
- State centers are fixed-v7 centers in the adaptive market study.
- Only three development markets are available, with correlated lambdas,
  betas, refits, and overlapping training windows.
- The through-2023 sample has been inspected repeatedly and is not an
  untouched holdout for future model claims.
- Novelty relative to the full literature has not been established.
- A reliability gate must earn a separate frozen model experiment before its
  profitability can be evaluated.

### 2026-07-21 — `balanced-lagged-performance-001` (complete, not supported)

- The mechanism study itself authorized no P&L. The owner later explicitly
  requested a separate full readout, so this study was frozen before any
  balanced choice, trade, or aggregate metric was opened. It is post-result
  exploratory evidence on the repeatedly inspected through-2023 sample.
- It compared fixed, one-sided lagged log4, and pair-balanced log4 under
  separate monthly eight-year lambda selection, the unchanged
  `[0, 5, 15, 35, 70, 150, 300, 600, 1200]` grid, t+2 execution, 10-bps
  one-way costs, and paper turnover. `beta=log(4)` was inherited and remains
  uncalibrated for profit.

| Market | Model | Sharpe | MDD | Turnover | Cash fraction | Switches |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| US | fixed | `0.57` | `-0.34` | `0.65` | `0.21` | `21` |
| US | lagged log4 | `0.59` | `-0.34` | `0.47` | `0.25` | `15` |
| US | balanced log4 | `0.62` | `-0.34` | `0.40` | `0.26` | `13` |
| DE | fixed | `0.17` | `-0.39` | `0.99` | `0.18` | `32` |
| DE | lagged log4 | `0.34` | `-0.39` | `0.25` | `0.13` | `8` |
| DE | balanced log4 | `0.34` | `-0.39` | `0.50` | `0.17` | `16` |
| JP | fixed | `0.33` | `-0.32` | `0.46` | `0.28` | `13` |
| JP | lagged log4 | `0.41` | `-0.32` | `1.16` | `0.28` | `33` |
| JP | balanced log4 | `0.24` | `-0.41` | `1.37` | `0.23` | `39` |

| Market | Balanced minus lagged: Delta Sharpe | Delta MDD | Delta turnover | Delta cash | Delta switches |
| --- | ---: | ---: | ---: | ---: | ---: |
| US | `+0.03` | `0` within `1e-9` | `-0.06` | `+0.01` | `-2` |
| DE | `-0.00` | `0` within `1e-9` | `+0.25` | `+0.03` | `+8` |
| JP | `-0.17` | `-0.10` | `+0.21` | `-0.05` | `+6` |

| Market | Balanced minus fixed: Delta Sharpe | Delta MDD | Delta turnover | Delta cash | Delta switches |
| --- | ---: | ---: | ---: | ---: | ---: |
| US | `+0.05` | `+0.00` | `-0.25` | `+0.05` | `-8` |
| DE | `+0.17` | `0` within `1e-9` | `-0.50` | `-0.01` | `-16` |
| JP | `-0.08` | `-0.09` | `+0.91` | `-0.05` | `+26` |

- Frozen decision: balanced-minus-lagged mean Delta Sharpe was `-0.05`
  with `1/3` positive markets, so the primary contrast failed.
  Balanced-minus-fixed mean Delta Sharpe was `+0.04` with `2/3` positive
  markets and passed separately. Both were required; result `not_supported`.
- On the balanced-readout aligned decision months, upper-candidate
  `lambda=1200` selection was fixed `2.07%/3.65%/4.55%`
  (`4/193`, `7/192`, `8/176`), lagged `8.81%/45.31%/30.11%`, and balanced
  `9.84%/45.31%/38.64%` in US/DE/JP. The diagnostic was descriptive only;
  adaptive finite optima remain unidentified.
- US smoke opened no metric and checked `12,138` t+2/cost cells, `388` oracle
  choice rows, and `8,092` oracle trade rows. The corrected run completed
  parallel US/DE/JP execution and passed completion-time plus separate CLI
  replay. The latter reconstructed `9` metric rows, `15,228` selection-surface
  rows, and `35,073` timeline rows with maximum error `0.0` and exact fixed/
  lagged oracle parity. Focused tests: `89 passed`; full suite: `530 passed`.
- Concrete active-lambda traces were reconstructed from sealed scalers and
  centers for US 2010-05-28→2010-06-02, DE 2008-01-28→2008-01-30, and JP
  2009-05-11→2009-05-13. In each, prior-day loss generated the exact
  pair-balanced arrival penalty, selected state, `signal=1-state`, t+2
  position, signed trade, 10-bps cost, and net return.
- The first run ending `ceed18fc5288` was invalidated after static audit found
  that its source lock named only three used feature columns although pandas
  physically loaded thirteen. The corrected lock records all thirteen loaded
  columns separately from the three used columns. Scientific CSVs, trades,
  and the decision are byte-identical between runs; only source/
  implementation/run metadata and inventory differ.
- Accepted run:
  `balanced-pnl-3ae665413a01-4e747110ba1c-def64a60db4c`.
  Pair balancing helped the selected US strategy but increased activity in DE
  and was adverse in JP. This is a factual development-sample readout, not an
  OOS, holdout, alpha, robustness, significance, profitability, stable-profit,
  or paper-replication claim.

### 2026-07-21 — Economic objective correction and manuscript start

- Re-reading Shu, Yu, and Mulvey and their public `jump-models` repository
  clarified the central target. The paper does not merely compare clustering
  outputs: it asks whether a persistent regime signal improves a 0/1
  market-or-cash strategy over both buy-and-hold and HMM after realistic delay
  and costs. The public repository is a generic JM library and does not contain
  the paper's HMM, data, monthly validation, accounting pipeline, or final
  candidate grids.
- The project-wide primary gap is now
  `G_m = Sharpe_JM,m - max(Sharpe_BuyHold,m, Sharpe_HMM,m)`. Mathematical
  improvement over fixed JM and behavior such as latency or whipsaw are useful
  intermediate evidence, but they are not economic model success.
- Independent recomputation from 18 accepted trade CSVs reconfirmed identical
  within-market dates, no duplicate or post-2023 rows, signal at `t` earning
  `t+2`, exact 10-bps costs, and paper turnover
  `0.5*252*mean(abs(position change))`.
- Under the corrected target, fixed JM and arrival-adaptive JM beat both
  benchmarks in `0/3` markets. Lagged-evidence and pair-balanced JM each beat
  both only in DE (`1/3`). The best tested gaps are US balanced
  `-0.04` versus HMM, DE lagged `+0.05`, and JP lagged
  `-0.13` versus buy-and-hold. No experiment result or historical
  lifecycle label was changed; only its relationship to the final objective
  was clarified.
- The mathematical sequence remains: fixed symmetric transition cost;
  arrival-day evidence discount; lagged-evidence discount to avoid current-day
  double use; and pair-balanced signed lagged evidence satisfying
  `C_t(i,j)+C_t(j,i)=2*lambda0`. Each exactly nested fixed JM at `beta=0` and was
  checked by formula, objective, toy path, brute force, prefix invariance, and
  concrete trade timelines.
- The next proposed mathematical hypothesis adds a past-only predictive loss
  for the return earned at `t+2` to the JM clustering objective. Its simple
  principle is that a useful regime should be both feature-coherent and
  economically different. This hypothesis has not been implemented or run;
  its parameters and evaluation evidence must be frozen separately.
- README and the advisor HTML were reduced to the economic question, one
  five-step workflow, the model equations, one benchmark result table, and the
  honest conclusion. A self-contained paper draft was started at
  `paper/manuscript.tex`. This was a documentation and objective-reconciliation
  task, not a new experiment, so no registry row was added.

### 2026-07-21 — Timing terminology clarification

- An older ledger entry used the phrase “two-day signal delay.” The canonical
  protocol is a one-trading-day execution delay: a signal formed after day
  `t` first earns the return at `t+2`. This clarification changes no
  position, trade, cost, metric, or experiment result.

### 2026-07-21 — Balanced P&L file-provenance correction

- A final response-independent static audit found that the accepted run ending
  `def64a60db4c` called 43 explicitly locked scientific inputs
  `files_read_sha256`, while `verify_inventory` also integrity-hashed all 232
  immutable entries across the four source inventories.
- That ambiguity did not affect any state, choice, trade, metric, or decision,
  but it made the file-access provenance incomplete. The run was marked
  `INVALID_IMPLEMENTATION`, and the same spec was refrozen for a metadata-only
  correction after every numerical result was already known.
- Source-lock schema 2 records `43` explicitly locked inputs and `232`
  integrity-hashed inventory entries in separate maps with matching counts.
  The direct regression test verifies source namespacing and complete entry
  retention.
- The new run passed completion-time verification and a separate CLI replay:
  `9` metric rows, `15,228` selection-surface rows, `35,073` timeline rows,
  maximum error `0.0`, and exact fixed/lagged oracle parity.
- Accepted run: `balanced-pnl-3ae665413a01-4e747110ba1c-eaae6444a9a5`.

### 2026-07-21 — Frozen simple-JM challenger suite

- Scientific question: can one prespecified simple causal JM variant beat both
  same-sample buy-and-hold and Gaussian HMM in US, DE, and JP under the
  canonical through-2023 `t+2`, 10-bps protocol? The primary gap was
  `G_m(v) = Sharpe_v,m - max(Sharpe_BH,m, Sharpe_HMM,m)`; the same variant had
  to satisfy `G_m>0` in all three markets.
- Five definitions were frozen together before stage-A performance was opened.
  The execution order was static lambda 50, DD-only, then confirmation,
  return-aware, and robust L1. The raw nine-value lambda grid, 3,000-row
  window, Jan/Jul refits, trailing-eight-year monthly selection, dates, delay,
  cost, and comparators did not change after results.
- Static lambda 50 solved the ordinary squared-loss JM with a constant
  illustrative paper-visible penalty and no monthly lambda reselection.
- DD-only used
  `sum_t 0.5*(z_DD,t-theta_s,t)^2 + lambda*sum_t I(s_t != s_t-1)`.
  It retained the canonical four-field complete-row calendar, so the three US
  startup rows with DD and excess return but missing Sortino were not admitted.
- Two-observation confirmation post-processed the sealed selected fixed state:
  initialize at the first finite state; thereafter accept `s_t` only when
  `s_t=s_(t-1)`, otherwise retain the previously accepted state. This is causal
  and adds the smallest nontrivial confirmation latency.
- Return-aware JM fitted
  `0.5*||z_t-theta_s||^2 + gamma*0.5*m_t*(y_(t+2)-mu_s)^2` plus the fixed
  switch cost. `m_t=1` only when the full-calendar `t+2` target had matured by
  the refit cutoff. Target mean and population scale used only matured targets
  in the past-only window. Gamma zero routed exactly through every sealed fixed
  candidate state, choice, signal, position, cost, and return; gamma one was
  the only new fit. Online decoding used features only.
- Robust L1 replaced squared feature loss by raw cityblock loss
  `||z_t-theta_s||_1`; its M-step used componentwise medians and its fixed-cost
  path remained an exact dynamic program.
- Verification covered formulas, stored objectives, toy confirmation paths,
  brute-force DP equivalence, randomized fit recomputation, prefix invariance,
  canonical row masking, matured-target exclusion, exact discrete gamma-zero
  routing, `t+2`, 10-bps accounting, paper turnover, and the 2023 cutoff.
  Focused tests were `63 passed` after adding the CSV route regression.
- Accepted run:
  `simple-jm-suite-2d3d2a779b13-93b7ba818774-20260721T124416567215Z`.
  Independent replay reconstructed `24` common-sample metric rows and `45`
  concrete traces. Maximum metric difference was `2.37e-14`, below `1e-12`.
- No variant met the cross-market rule. Gaps in US/DE/JP were static lambda 50
  `-0.07/-0.13/-0.36`; DD-only
  `+0.25/-0.06/-0.12`; confirmed 2d
  `-0.03/-0.14/-0.25`; return-aware
  `-0.09/-0.14/-0.22`; and robust L1
  `-0.28/-0.12/-0.18`.
- DD-only was the useful result. Relative to fixed JM, Sharpe changed
  `+0.34/+0.06/+0.09` in US/DE/JP. It beat both controls in US
  with Sharpe `0.91`, MDD `-0.19`, turnover `0.34`, cash fraction
  `0.14`, and `11` switches. It remained below buy-and-hold in DE and JP.
  Its MDD worsened by `0.04` in DE and `0.11` in JP; JP turnover rose
  by `0.21` and switches rose `13->19`.
- Confirmation behaved as intended mechanically: fixed-JM one-day excursions
  were delayed/removed and switches changed `21->19`, `32->26`, and `13->13`.
  It improved Sharpe only in US and passed the economic target in `0/3`.
- Return-aware gamma one was almost behaviorally null: it differed from fixed
  on two US signal days and zero JP signal days; DE changed more but switches
  rose `32->36`. Robust L1 changed paths materially but increased switches in
  US/JP and worsened MDD in all three. Both passed `0/3`.
- DD-only is scale-confounded. Three standardized squared feature coordinates
  contribute a different observation-loss scale than one coordinate while the
  same raw lambda grid is held fixed. DE selected lambda zero in `123/193`
  months; JP selected the upper `1200` in `32/176` post-initial OOS months
  (`18.18%`). The result therefore does not yet prove that Sortino features are
  noise.
- The next minimal diagnostic hypothesis is one scale-matched DD control:
  `L_DD,scaled = 3 * 0.5*(z_DD-theta)^2`, with exactly the same raw lambda grid
  and all other protocol fields unchanged. It asks whether the DD result
  survives after separating feature removal from effective regularization.
  Its outcome is not known and must be frozen before performance.
- This is repeatedly inspected exploratory development evidence. It supports
  no alpha, profitability, robustness, paper-replication, holdout, or
  generalization claim.

### 2026-07-21 — Corrected simple-suite evidence artifact

- A response-independent final audit found that the first complete run ending
  `20260721T124416567215Z` had valid deterministic metrics but did not satisfy
  the full frozen evidence contract. Static lambda 50, DD-only, and confirmed
  2d supplied no point losses in `27/45` concrete traces; DD refit centers were
  absent; the real-US smoke compared only two overlapping rows; and a later
  verifier-only edit meant the exact executed runner was no longer available.
- No candidate state, choice, signal, position, trade, return, metric, or
  decision was changed in response. The correction records centers and active
  state count, replays each fixed-JM trace refit and terminal online DP,
  strengthens each US prefix comparison to `128` rows, and locks all six
  implementation files including the canonical fixed-JM code.
- Corrected accepted run:
  `simple-jm-suite-2d3d2a779b13-fe3e1f0b6d22-20260721T135918520651Z`.
  It locks `31` explicitly read scientific inputs and integrity-checks all
  `153` source-inventory entries. Its `24` metric rows are numerically identical
  to the first run in Sharpe, MDD, turnover, cash, switches, and primary gap.
- All `45/45` traces now retain fit date, objective, scaler, centers, point loss
  for each reachable state, transition penalty, terminal cumulative DP values,
  raw state, post-filter state, signal, t+2 position, turnover, cost, gross
  return, and net return. A collapsed state's NaN-center loss is represented as
  positive infinity, matching the upstream DP's unreachable-state rule.
- One-state collapse is not rare. DD-only collapsed in `95/132/124`
  fit-by-lambda rows in US/DE/JP and monthly CV selected a collapsed fit in
  `0/43/75` of `194/193/177` choice months. Return-aware counts were
  `63/85/93` fit rows and `24/53/78` selected months. Robust-L1 counts were
  `75/91/98` and `62/83/87`. These fits were retained, not excluded after
  results.
- The route and P&L remain mathematically determined, but a one-state candidate
  cannot be interpreted as two recovered regimes. For DD-only, this strengthens
  the existing loss-scale confound and makes the prespecified three-times-loss
  control the minimal next diagnostic.
- Completion-time and separate CLI verification both passed: `24` metric rows,
  `45` traces, `9` collapse rows, and maximum metric replay difference
  `2.37e-14`. The focused simple-JM test set passed `81` tests before this run.
  No post-2023 model or P&L data was accessed, and no performance, alpha,
  profitability, holdout, replication, robustness, or generalization claim is
  authorized.

### 2026-07-21 — Final provenance-complete simple-suite artifact

- After the intermediate corrected run was complete, full repository tests
  exposed that adding refit diagnostics directly to the canonical fixed-JM
  record schema broke checkpoint/parallel contracts. Static audit also found
  that its six-file implementation lock omitted result-affecting helpers.
- The compatibility correction restored the canonical default refit schema and
  made centers/state-collapse diagnostics opt-in only for DD-only. The lock was
  expanded to `artifacts.py`, `backtest.py`, `config.py`, `walkforward.py`,
  `pyproject.toml`, and `uv.lock`, for twelve result/environment files total.
- Final accepted run:
  `simple-jm-suite-2d3d2a779b13-544237a59943-20260721T145043479851Z`.
  It locks `31` explicitly read scientific inputs, all `153` upstream inventory
  entries, and the exact twelve-file implementation/environment bundle.
- The built-in verifier passes `24` metric rows, `45` complete loss-to-trade
  traces, and `9` fit-degeneracy rows; maximum metric replay error is
  `2.3647750424515834e-14`. The decision remains `not_supported`: DD-only passes
  US only and every other frozen challenger passes `0/3` markets.
- Independent audit validated all `115` immutable artifact entries, all `1,692`
  selected-fit mappings, all traces, accounting, cutoff, and gamma-zero route.
  Maximum loss/DP errors were `1.78e-15`/`9.09e-13`; accounting error was zero,
  and the latest scientific date was `2023-12-29`.
- All scientific outputs are unchanged: `summary.csv` and every trade, state,
  refit, trace, smoke, and source-lock file are byte-identical to the
  intermediate corrected run. Only run metadata, the implementation lock, and
  inventory differ, so this correction cannot strengthen the economic result.
- Focused simple-JM tests passed `82`; the full repository suite passed `613`.
  No post-2023 model or P&L row was accessed. This repeatedly inspected
  exploratory result supports no alpha, profitability, performance, holdout,
  paper-replication, robustness, or generalization claim.

### 2026-07-22 — `dd-loss-scale-001` (complete, not supported)

- Question: did DD-only improve because removing Sortino features helped, or
  because reducing three observation-loss coordinates to one made the unchanged
  lambda grid effectively stronger?
- The frozen control used
  `Q_3 = 3 * 0.5 * (z_DD - theta)^2 = 3 Q_1`. It kept the raw grid
  `[0, 5, 15, 35, 70, 150, 300, 600, 1200]`, complete-row mask,
  3,000-observation window, Jan/Jul refits, monthly trailing-eight-year
  online-state Sharpe selection, one-day trading delay, 10-bps cost, paper
  turnover, comparators, and through-2023 sample unchanged.
- The objective tolerance was mechanically scaled from `1e-8` to `3e-8`, so
  its value relative to `Q_3` remained unchanged. Formula equality, every toy
  path, lambda-third path equivalence, DP/brute-force equivalence, US
  prefix-invariance, monthly selection, concrete loss-to-trade traces, `t+2`
  accounting, costs, turnover, source identities, and cutoff all passed.
- Completed run:
  `dd-loss-scale-e1e84ddbbdda-65ccb507abba-20260722T045053128156Z`.
  Frozen spec hash:
  `e1e84ddbbdda749f832a2d5a17e3fd8d8d064b6315f74f7c68dec9826d3f7feb`.
  Implementation hash:
  `65ccb507abbad7fd6878c0cfc09d8ba82f9914ad1f60d28f9f8315373a370759`;
  producing HEAD:
  `ba61b06e644824500d5351473e6631d24d7f9da9`.
- Completion-time and separate CLI replay verified `15` metric rows, `9`
  complete traces, and `3` degeneracy rows. Maximum metric replay error was
  `5.218048215738236e-15`. US smoke preserved all `128` prefix rows after
  appending `128` future rows. No post-2023 model or P&L row was accessed.

| Market | Scaled Sharpe | G | MDD | Turnover | Cash | Switches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| US | `0.88` | `+0.23` | `-0.20` | `0.47` | `0.14` | `15` |
| DE | `0.20` | `-0.09` | `-0.45` | `0.50` | `0.11` | `16` |
| JP | `0.43` | `-0.12` | `-0.35` | `0.81` | `0.14` | `23` |

- Scaled-minus-ordinary-DD Sharpe was
  `-0.02/-0.03/+0.00`; MDD delta
  `-0.00/-0.02/+0.09`; turnover delta
  `+0.12/-0.31/+0.14`; cash delta
  `+0.01/-0.00/+0.05`; and switch delta `+4/-10/+4` in
  US/DE/JP. A positive MDD delta means a less-negative drawdown.
- Raw lambda moved upward in `73.7%/81.3%/66.1%` of paired months. The upper
  endpoint was selected in `0%/22.9%/29.0%`, and monthly validation selected
  one-state fits in `0/194`, `44/193`, and `63/177` US/DE/JP months.
  Thus loss scaling produced the intended regularization response, but the
  finite grid binds and two-state interpretation weakens in DE/JP.
- Scaled DD retained a positive Sharpe advantage over fixed JM in all three
  markets, weakening a pure loss-scale explanation for the original DD result.
  It does not prove that Sortino features are harmful. The official decision is
  `not_supported` because scaled DD beats both economic controls only in US.
  This repeatedly inspected exploratory result supports no performance, alpha,
  profitability, robustness, holdout, paper-replication, or generalization
  claim.

### 2026-07-30/31 — `jm-grid-identification-001` (complete)

- The Section-15 identification question asked of the JM row on sealed v9.4,
  frozen before any new lambda was fitted; 29-lambda union pass replayed the
  sealed states exactly (parity gate, three markets).
- **Registered interpretation drawn: the paper's own Table-3 persistence curve
  does not reproduce** — 1/6 lambdas within the printed half-unit, systematic
  ratio ~0.75, including at lambda = 0 where the lambda scale plays no role.
  The feature/standardization geometry is thereby isolated as the dominant
  unresolved axis, and the grid table is conditional on our geometry.
- Conditional on that geometry: no pre-named grid (eight, all sourced) reaches
  8/8 in any market; the registered turnover-alone criterion is met by no grid
  in US or DE; DAX's published turnover 1.70 lies above the entire tested
  spread. US CV-selected path: 34 shifts / 21.9% bear vs the authors' printed
  30 / 19.7%.
- Adversarially verified (3 independent auditors): recomputation bit-for-bit
  PASS; two process findings (post-hoc glyph threshold, missing artifacts and
  completion event) remediated with numbers unchanged.
- Next frozen question, registered in advance: `jm-standardizer-geometry-002`.
- Artifact: `artifacts/jm-residual/01-grid-identification/`. This exploratory
  result supports no performance, replication, or grid-choice claim.

### 2026-07-31 — `jm-standardizer-geometry-002` / `-003` (complete; axis closed)

- Mechanism-first revisit of the standardizer axis on sealed v9.4 (premise
  changes named in the -002 spec; prior v8-era rows were CRSP-era). Probe
  gates: src-vs-probe-local V1 exact in all three markets.
- **Owner correction recorded:** the clip-3σ variant was withdrawn from
  interpretation (registry AMENDED event) — example-code recipes are not
  author-method candidates, per the standing CLAUDE.md rule the variant list
  should never have included it. No conclusion depended on it (clip was
  immaterial where measured). -003's justification reframed to cadence-family
  enumeration, without example-code appeal.
- **Finding: the paper's Table-3 persistence curve is bracketed by the
  exhausted cadence family and matched by no member** — λ=0 shift rate:
  expanding 7.24, per-refit 15.07, frozen-initial 5.88, published 9.7;
  registered I1 rule failed by all (2/6, 1/6, 3/6). The axis is dominant
  (2x swing) yet unidentifiable from public information.
- Descriptives adopted nowhere: frozen-initial gives the closest US anchor
  (32 shifts / 19.4% bear vs printed 30 / 19.7%) and best conditional
  Table-4 counts (6/4/5 of 8). Registered closure: JM residual attribution
  moves to data/vintage differences, mirroring the HMM turnover closure;
  further geometry variants barred without a new primary source.
- Artifacts: `artifacts/jm-residual/02-standardizer-geometry/`,
  `artifacts/jm-residual/03-frozen-initial-scaler/`. Exploratory; no
  performance, replication, or adoption claim.

### 2026-07-31 — `jm-effective-lambda-inversion-004` (complete; JM row closed)

- The owner-requested inversion, done by READING the authors' choices from
  their own Figure 5 (lossless vector extraction, re-validated against each
  panel's printed annotations) rather than searching our knobs.
- **QA: their paths on our data reproduce their table — 8/8 us, 8/8 de,
  7/8 jp (23/24 cells); turnover deviations 0.001/0.030/0.005.** Accounting,
  costs, conventions and the reconstructed data are thereby proven at table
  level; the entire JM residual is the unpublished state sequences.
- QB/QC: the authors' US path is ~a single fixed lambda 35 under our sealed
  geometry (95.7% daily agreement); DE/JP are not expressible by any
  constant lambda under any tested geometry (89.5%/81.6%) though per-month
  ceilings reach 96-99%. Effective lambdas are descriptive only — grid
  candidacy from these results is forbidden by the frozen spec.
- Together with -001/-002/-003 this closes the JM row the way the HMM
  turnover row closed: mechanism characterized, accounting proven, residual
  localized to unpublished choices on unavailable vintages. Artifact:
  `artifacts/jm-residual/04-effective-lambda-inversion/`. Exploratory; no
  performance, replication, or adoption claim.

### 2026-07-31 — `jm-inferred-grid-005` / `-006` (complete; owner-instructed inference)

- The owner instructed reverse-engineering the unpublished lambda grid. Done
  as estimation FROM the authors' own Figure-5 monthly choices (never as a
  search of our knobs against their table): derivation rules frozen before
  the histograms were looked at; validation replays registered as
  conditional diagnostics; adoption owner-gated.
- -005 (all months): supports flooded by tiny lambdas — the registered tie
  rule's artifact (quiet months are uninformative and first-argmax hands
  them to the smallest lambda). US conditional replay still 7/8 with Sharpe
  0.680 vs 0.68.
- -006 (shift months only, ties to largest; estimator family declared
  exhausted): **US support concentrates in the 10-100 band with mass near
  35** — matching -004's constant-lambda finding from the independent
  daily-agreement direction; only 23/408 US months informative. Conditional
  replays: US lands turnover in tolerance (0.412 vs 0.44) or Sharpe in
  tolerance (0.724 vs 0.68) depending on the grid variant, never both — the
  documented CV-instability mechanism; DE 4/8, JP 3/8, so the -004
  expressibility bound is the final word for DE/JP.
- Best available answer to "what was their grid": through our lambda family,
  their US selections live in ~[10, 100] centered near 35; DE/JP cannot be
  resolved because their sequences are outside our expressible space.
  Descriptive only; artifacts in `artifacts/jm-residual/05-inferred-grid/`
  and `06-inferred-grid-shift-months/`. Exploratory; no adoption.

### 2026-07-31 — `jm-grid-exhaustive-007`/`-008` (complete; the existence map)

- Owner-instructed target-conditioned exhaustive search over all 6,474,511
  subsets (sizes 2-8) of the 29-lambda sourced union, nine arms (3 markets x
  delays 1/5/10; Table-5 cells for the long delays, delay inside the CV per
  the Table-5 caption). Honesty label in both specs: solutions are
  calibration artifacts, never the authors' grid; nothing adopted.
- **Delay 1: the three-market common grid DOES NOT EXIST** — Germany and
  Japan are individually unreachable (0 passing grids each) even with full
  calibration freedom; the US row is comfortably reachable (109,400 grids).
  The -004 expressibility bound reappears at the table level.
- **Delays 5/10: common grids are abundant** — 71,349 (d5), 454,762 (d10),
  7,107 surviving both, clustering around {small lambda, ~20-22, 220}; zero
  survive all nine arms (forced by the d1 zeros).
- Machinery verified at machine precision (nine parity gates, worst 3.33e-14)
  and every headline example re-validated through the real unaccelerated
  pipeline: 24/24 PASS. Artifacts:
  `artifacts/jm-residual/08-exhaustive-nine-arms/`. Exploratory; no
  performance, replication, or adoption claim.

### 2026-07-31 — `jm-per-market-grid-009` (complete; per-market grids and the blocking cells)

- Owner-instructed: per-market grids meeting Shu at delays 1, 5 AND 10, with
  every nonexistence statement scoped to the 29-lambda menu and quantified by
  a blocking-cell frontier instead of asserted.
- **US: 36,657 grids satisfy all fourteen published cells**; three winners
  validated through the real pipeline 9/9 PASS (example {0, 21.5, 70}, worst
  delay-1 deviation 0.011 across all eight Table-4 cells).
- **DE/JP: best available is 13/14** (366 and 2,948 grids respectively; one
  example each validated 6/6 PASS). The d1 miss is pinned by the frontier:
  DE turnover is a joint constraint (range reaches 4.63 but only 1.465 when
  the other seven cells hold); JP leverage has a lattice ceiling of 0.737 vs
  the 0.75 target (0.636 given the rest) — state-sequence shape, not menu
  narrowness, within the tested menu.
- Artifacts: `artifacts/jm-residual/09-per-market-grids/`. Calibration
  artifacts only; reseal is an owner decision.

### 2026-08-01/2026-08-07 — `grid-selection-rule-001` (complete; v11 reseal PROPOSED, not performed)

- Frozen 2026-08-01, owner-approved: among each market's admissible grids
  (the 13/14 or 14/14 sets from -009), adopt whichever has the highest
  daily agreement with the AUTHORS' own Figure-5 state sequence, instead of
  "the first row of an examples file" (how v10 was actually chosen). No
  strategy metric enters the selection. A same-day partial run found DE's
  complete 366-grid enumeration ranked the adopted grid dead last; US/JP
  were truncated to the top 200 rows per work-chunk (a bug) and left
  incomplete for five days.
- Finished 2026-08-07: fixed the truncation bug and completed all three
  markets. **US**: winner `{0,0.1,20,220}` agreement 0.9609 (3-way tie) vs
  adopted `{0,21.544,70}` 0.9476 — differs, but adopted sits in the upper
  part of the `[0.9353,0.9609]` spread. **DE**: winner
  `{0.1,1,10,21.544,26.827,40,100,500}` agreement 0.8951 (unique) vs
  adopted `{150,500}` **0.8585 — the exact minimum of the
  `[0.8585,0.8951]` spread: the adopted German grid ranks dead last, 366th
  of 366**, on a complete enumeration. **JP**: winner
  `{1.931,20,25,26.827,40,51.795,220}` agreement 0.8516 (2-way tie) vs
  adopted `{10,220}` 0.8152 — differs, roughly 58% up the
  `[0.7645,0.8516]` spread (below median, not worst).
- Independently CONFIRMED by an agent that read none of the three
  producing scripts: full from-scratch recomputation of all 366 DE grids
  (own `select_monthly_candidate` call, own scoring code) reproduced the
  dead-last finding to exact float64 equality; US/JP winner+adopted
  spot-checked exact; all three adopted grids' admissible-set membership
  independently reconfirmed. Receipt:
  `docs/audit/2026-08-07-grid-selection-rule-001-receipt.md`.
- Prerequisite infrastructure repair, same session: the -008/-009 binary
  caches (`*.npz`/`*.npy`) had not survived on disk between sessions —
  the same class of loss as the deleted v9.4 run directory. Rebuilt by
  substituting the v10 run directory for the missing v9.4-hash one
  (features.csv proven byte-identical by reseal gate 2; the rebuilt us-d1
  cache reproduces the sealed v9.4 anchor to float precision) and fixing
  one stale parity check that compared against v9.4's own now-missing
  `metrics-exploratory.csv` for delay 5/10 (replaced with a fast-vs-real-
  code-path check needing no external file). All three scientifically
  load-bearing rebuilt outputs (`arms.csv`, `intersections.csv`,
  `solutions.csv`) verified byte-identical to the previously sealed,
  committed versions via `git diff` before anything downstream was
  trusted.
- Per the spec's own consequence rule: the winner differs from the
  adopted grid in all three markets, so **a v11 reseal is PROPOSED to the
  owner, not performed automatically**. Every mechanism result scored
  against the v10 DE/JP baseline since 2026-08-01 (the five original
  candidates named in the -001 CORRECTION, plus
  `adaptive-separation-001` and `jm-disagreement-anatomy-010`'s DE/JP
  legs) must be rerun against any new baseline before "mechanism X fails
  in market Y" is restated; `lagged-capguard-001` is US-only and
  unaffected. Artifacts: `artifacts/grid-selection-rule/01-rule/`.

## 2026-08-09 — resampling intervals are descriptive, not gates

**Retraction (mine, owner-caught).** I wrote that the return-sampling
bootstrap CI half-width, about ±0.03 Sharpe, was a "binding noise floor"
that DA-JM "must clear". Withdrawn. A CI half-width moves with the
confidence level, block length, bootstrap procedure, sample length and
period, and the stationarity assumption — change 95% to 90%, or add ten
years of data, and the "floor" shrinks with no model changing. It is
*estimated conditional sampling uncertainty for one comparison under one
bootstrap design*. Also withdrawn: writing the bootstrap replicate
fraction as "P(Δ > 0)", which reads as a posterior probability.

Not withdrawn: Sharpe-difference inference is a studied problem (Ledoit &
Wolf 2008 studentized time-series bootstrap; Politis & Romano 1994). The
tool is valid; promoting it to a decision gate was the error. This repeats
a defect already corrected once — see `docs/audit/2026-07-full-audit.md`
(§ "the 95% CI was used as evidence when it carries none").

**Two different ±0.03 exist in this repo — do not conflate them.**
`research/lagged-capguard-001.toml` records a *5-seed optimizer spread* of
about ±0.03 Sharpe (a spread of point estimates across initialization
seeds). The 2026-08-09 receipt records a *bootstrap CI half-width* of about
±0.03 (a sampling-uncertainty interval). Same number, different quantity,
different meaning. Neither is a threshold.

**Precedent that is frozen, not endorsed.** `research/ajm-ext-001.toml`
gates on `paired_block_bootstrap_95pct_lower_bound > 0`. That contract is
hash-pinned to a closed experiment and its bootstrap was never run
(`docs/audit/2026-08-06-ajm-ext-d1-receipt.md`). It is not a template. The
correct in-repo model is `research/holdout-2026-001.toml`: "95 percent
interval; reported, never gating".

**Replacement hierarchy** (registry `da-jm-evaluation-hierarchy-2026-08-09`,
written into `docs/theory/da-jm-formalization.md` item 6): effect size →
cross-market transport → mechanism consistency → robustness → external
transport → resampling evidence last and descriptive only. Neither
"interval contains 0" nor "interval excludes 0" decides success.

## 2026-08-09 — the Sharpe I bootstrapped was not the Sharpe this repo reports

**Defect (mine), found and fixed in the same patch.**
`scripts/studentized_sharpe_difference.py` re-derives the Sharpe from
uncentered moments so the difference is a smooth function of means (the
Ledoit–Wolf construction). I wrote the denominator as
`sqrt(second_moment − mean²)` — the **population** standard deviation — while
every other Sharpe in this repository comes from `performance_metrics` with
`volatility_ddof=1`, the **sample** standard deviation. The two are different
estimators, so the Δ I reported was a number no other artifact here could
reproduce. Nothing flagged it: the intervals looked reasonable, and a
plausible-looking number is exactly what this failure mode produces.

The fix is exact matching, not a renaming. By the Bessel identity
`(1/(T−1))Σ(r−m)² = (T/(T−1))(g − m²)`, the whole correction is one constant,
`sqrt((T−1)/T)`, applied to Δ and to `grad f`. Because it multiplies `Δ̂`,
`Δ*`, `s(Δ̂)` and `s(Δ*)` alike and every resample has length T, **the
studentized statistic and both p-values are invariant under multiplication by
a positive nonzero constant** — and `sqrt((T−1)/T) > 0` for every `T > 1`, so
the condition holds here. Only Δ and the
endpoints move, by 5.8e-5 relative (us 0.004391656 → 0.004391400). So the
published reading never changed. That is the uncomfortable part: an estimator
mismatch that happens to be immaterial here would have been material for any
statistic that does not divide it back out, and I had no test that would have
caught either case.

There is now a hard gate in `main()` (`SystemExit` if the moment form and
`performance_metrics` disagree by more than 1e-12) and a regression suite that
pins the estimand on each arm separately, not only on the difference — a
difference can match by cancelling two equal errors. Fault injection confirms
the test can fail.

**Not independently verified.** Written and checked by the same agent, against
the separate-verifier rule, because the session was scoped to a wording patch
with no new experiment. Owed before any DA-JM decision cites these intervals.

**Review correction to the above (owner-caught, same day, before merge).** The
fix had two defects of its own, and both are the same kind of mistake as the
original: a correction applied wider than the thing it corrects.

I scaled *both* estimators by `sqrt((T−1)/T)`. Only `repo` should carry it.
`lw_excess` uses the Ledoit & Wolf Eq. (1)–(2) *functional form* (with the
repo's `sqrt(252)` applied afterwards for reporting — their equation is
unannualized, so "verbatim" would be too strong), and its denominators *are*
the population sds. Scaling it left the script with two Bessel-corrected
estimators and no canonical one, so the "cross-check against the published
estimator" was checking the repo convention against itself. Scaling is now
estimator-specific and the zero-cash collapse is asserted as a *relationship*,
`grad_repo[:4] = ddof_scale(T) · grad_lw`, not as equality.

And one of the tests I wrote to prevent exactly this class of error was itself
vacuous: `assert x == approx(0.0, abs=1e-9) or x != 0.0` is true for every
finite value. The cash derivative is `sqrt(252)·sqrt((T−1)/T)·(1/σ₂ − 1/σ₁)`,
which vanishes only when the two arms have equal volatility; on the fixture it
is **+2.59**. The tautology was tolerating a coordinate three orders of
magnitude from the zero it pretended to allow. The lesson worth keeping is that
a test containing `or` between "is zero" and "is not zero" asserts nothing, and
that writing tests to pin an estimand does not help if the assertions are not
themselves falsifiable. Every new test here was fault-injected before it was
believed.

The regeneration is measurable: the 15 `repo` rows are bit-identical, the 3
`lw_excess` rows move by exactly `1/sqrt((T−1)/T)` to 12 decimals, and every
p-value is unchanged to the last digit — the invariance claim confirmed rather
than argued.

**Independent verification (separate agent) — found defects, now fixed.** The
verification I recorded as owed came back with two things worth keeping.

The first is a defect I introduced *while fixing the previous one*. The report
told the reader that the canonical Ledoit–Wolf Δ is "larger than the repo Δ by
`1/ddof_scale(T)`, by construction". That is the **zero-cash** identity, printed
next to data with a nonzero cash leg. Its own numbers refute it: the claim is
off by 4–21×, and for Japan the canonical Δ is *smaller*, the opposite of what
it asserts. The real cause is that the two estimators use different
denominators — `sd_ddof1(strategy)` versus `sd_pop(strategy − cash)` — and the
ddof constant is the smaller part of the gap. Three commits in a row, the same
failure mode: stating a clean relationship that holds in a special case as if it
held in general.

The second is that my regression suite was much weaker than it looked. A
mutation battery killed only **7 of 18** deliberate faults. Survivors included:
`lw_excess` dropping the cash subtraction entirely, the confidence quantile
becoming the median, the p-value tail flipped, the equal-tailed endpoints
swapped, the HAC variance divided by *T−1*, the block sum normalised by *b*
instead of √*b*, and the circular bootstrap silently becoming a moving block —
i.e. nearly every quantity that sets the width of the interval we publish.
Eight assertions were vacuous, one spectacularly so: it defined
`population := corrected/ddof_scale` and then asserted their ratio equals
`ddof_scale`, which is true for *any* estimator whatsoever.

The rebuilt suite kills **12 of 12**. `prewhitened_psi`'s VAR(1) recoloring
remains untested and is recorded as a known gap rather than quietly left.

The lesson is not "write more tests". It is that a test suite's value is not
visible from reading it — every assertion here looked purposeful — and the only
cheap way to find out is to break the code on purpose and see whether anything
notices.

**Fourth instance, same pattern (owner-caught in PR review).** The report said
"the two agree exactly only when the cash leg is zero". Also false: at zero cash
they still differ by `ddof_scale(T)` — which is what the test on that very line
asserts. So the prose contradicted the test sitting next to it. Fixed at the
source that generates the report, and every "Eq. (1)–(2) verbatim" softened to
"functional form, with the repo's annualization applied for reporting" (their
equation is unannualized).

Four occurrences is a pattern, not four accidents, so the response is a rule
rather than another fix: `AGENTS.md` now carries an **Anti-overclaim rules**
section and the principle *AI confidence is not evidence*. The operative one
here is the first: if a result was shown under a condition — cash = 0, beta = 1,
one market, one seed — that condition must appear in the conclusion.

**Fifth instance — the correction itself overcorrected (owner-caught, same
review).** Fixing the above, I wrote "the two estimators are NEVER exactly
equal". That swaps one absolute for another. The defensible statement is that
they are **not identical functions**; two non-identical functions can still
agree numerically in special cases, and here they do — at `Δ_lw = 0` the
zero-cash relation gives `Δ_repo = 0` too. Two further scope slips in the same
sentence: "invariant to any constant rescaling" should be *positive nonzero*
constant (at `c = 0` the studentized ratio is undefined; a negative `c` flips
the signed draws and the equal-tailed endpoints), and the DA-JM novelty claim
asserted that no duration/semi-Markov/hazard modification *exists* in the SJM
lineage — a literature search cannot establish non-existence, and Deep
Statistical Jump Models already notes its generic state loss can penalize
staying in a state too long.

The pattern underneath all five is one habit: reaching for the strongest
sentence the evidence almost supports. The fix is not more caution in tone but
naming the scope in the sentence itself.
