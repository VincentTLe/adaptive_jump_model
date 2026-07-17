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
`Sortino60`; a 3,000-observation training prefix; `StandardScaler`; Jan/Jul
refits; raw lambdas `[0, 5, 15, 35, 70, 150, 300, 600, 1200]`; monthly
trailing-eight-year Sharpe selection; two-day signal delay; and 10 bps
one-way cost. The sample is capped at 2023-12-31.

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

### Proposed reliability-gated extension

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

That gate is only a hypothesis. It must not be performance-tested unless the
mechanism diagnostic first supports the predictive value of
\(R_{\rm train}\), and a separate model study is frozen.

## Experiment history

### 2026-06-23 — minute prototype (archived)

- Tried the duration mapping \(\lambda=\log(d-1)\) and hand-set additive or
  multiplicative adaptive penalties: noise raised the cost and shocks lowered
  it.
- Exact time-varying DP and synthetic paths worked.
- One-shot real minute paths were 98–99.8% identical and transaction costs
  overwhelmed the strategies. Loss and penalty scales were not calibrated to
  the later daily protocol.
- Status: useful prototype, no market evidence. Sources are under
  `archive/legacy-minute/`; relevant commits include `3f7426e`, `1efbfb1`,
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
  Sources are under `archive/pre-audit-daily/`; commits include `c90abad`,
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
- Status: withdrawn before run. See
  `research/hyperparameter-grid-attribution.toml`.

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
- US switches fell `21→18/19`, but Sharpe fell by `0.0574/0.0681`.
- DE switches fell `32→24/14`, turnover fell by `0.4967/1.1175`, and Sharpe
  rose by `0.1054/0.1106`; drawdown deltas were only floating-point negatives
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
  refit reliability was DE `0.5463` and JP `0.5578`, while DE whipsaw
  events were slightly more separated than persistent events
  (`0.5675` vs `0.5626`). JP had only a small difference in the proposed
  direction (`0.5632` vs `0.5659`).
- Status: completed mechanism diagnostic; this reliability gate is not
  justified and was not sent to a P&L test. Artifact:
  `adaptive-separation-813f66912526-26cbca8871be-fefc608b9081`.

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
