# Duration-Aware Jump Model (DA-JM) — formalization

Status: **math only, no code, no frozen experiment spec yet.**

## Revision history

- **v1 (2026-08-08).** Original formalization: total-cost parameterization
  `phi_k(d) = -log q_k(d)` replacing the constant lambda, with a
  `pi = sigmoid(lambda)` back-out. Independently verified as internally
  correct mathematics (receipt
  `docs/audit/2026-08-08-da-jm-formalization-receipt.md`).
- **v2 (2026-08-09, this version).** The sigmoid back-out is **retracted**
  (internally correct but numerically dead at this repo's lambda scales —
  see Section 7's warning box) and the model is reparameterized as an
  **excess (LLR-vs-geometric) duration cost added on top of the untouched
  classic-lambda machinery**, per the adversarially verified fact-finding
  (registry `da-jm-open-questions-factfinding-2026-08-09`) and the owner
  decisions of 2026-08-09 (D_max=504 with hazard-level geometric tail;
  left-censored first in-window segment; restricted-mean anchors computed
  from v12; beta roles 2.0 primary / 0.5 adversarial / 1.0 identity). The
  v1 receipt covers the v1 math; every v2-specific claim below cites the
  fact-finding NOTE that verified it.

Notation follows this repo's TOML-spec style (`sum_t`, `argmin`, plain
ASCII).

---

## 1. Baseline: the classic constant-lambda JM (untouched)

```
J_JM(theta, s_1:T) = sum_t L(x_t, theta_{s_t})
                   + lambda * sum_t=2..T I(s_t != s_{t-1})
```

K=2 states, coordinate descent (DP E-step / cluster-mean M-step). In this
repo lambda is never a single constant: each market has a calibrated grid
and monthly trailing-CV selection among per-lambda candidate paths. **DA-JM
leaves all of that machinery byte-identical** — the calibrated grids, the
CV rule, the online decode, the costs. This is the load-bearing change from
v1, which replaced lambda and therefore had to translate it into a
probability scale (the step that failed numerically).

Already-published neighbors (novelty sweep + CJM PDF read, registry
`da-jm-novelty-sweep-2026-08-08`): CJM's state-pair matrix `Lambda[i][j]`
is zero-diagonal and strictly first-order; no Mulvey-lab paper implements
any duration/hazard penalty; no found paper embeds a duration cost in the
SJM's penalized-DP framework. A second novelty pass over 2025-2026
preprints and forward citations is running; strong novelty claims wait for
it.

## 2. The DA-JM objective: excess duration cost

```
J_DA = J_JM + J_duration(beta)
```

Segments j = 1..J with state z_j and length d_j. For each market m and
state k, fix a **geometric reference** q_G with scale pi*_{m,k} and a
**discrete-Weibull duration model** q_{W,beta} with scale pi_{beta,m,k}
(both anchored to the same observed duration statistic — Section 5). The
duration term charges each segment the log-likelihood ratio of the
reference against the duration model:

```
J_duration = sum_j Delta_phi_{z_j}(d_j),
Delta_phi_k(d) = log[ q_{G,pi*_k}(d) / q_{W,beta,pi_beta,k}(d) ]
```

with the censoring and cap conventions of Sections 4-6. Immediate
consequences (all adversarially verified in the fact-finding NOTE):

- **beta = 1 gives an objective identity, not just path equivalence.**
  At beta=1 the Weibull family IS the geometric family and the anchor
  equation is the same equation, so pi_{1,k} = pi*_k and
  Delta_phi_k(d) = 0 for every d, every k, every anchor value:
  J_DA == J_JM term by term. This is strictly stronger than v1's
  reduction theorem and needs no symmetric-state assumption.
- Negative partial costs occur (Delta_u < 0 at young ages for beta > 1,
  etc.) and are harmless: the augmented DP is min-sum on a finite
  time-layered DAG — verified against brute-force path enumeration on 300
  random problems, 0 mismatches.

## 3. Discrete-Weibull hazard machinery (unchanged from v1, verified)

```
q(d)  = pi^((d-1)^beta) - pi^(d^beta),          d = 1, 2, ...
S(d)  = pi^((d-1)^beta)                          [survival]
h(d)  = 1 - pi^(d^beta - (d-1)^beta)             [hazard]
u(j)  = -log(1 - h(j)) = (j^beta - (j-1)^beta) * (-log pi)   [stay cost]
v(d)  = -log h(d)                                [terminate cost]
-log q(d) = sum_{j=1}^{d-1} u(j) + v(d)          [telescoping identity]
```

At beta=1: h constant (memoryless), u constant, v constant. All checked
two independent ways in the v1 receipt.

The excess decomposes the same way:

```
Delta_u_k(j) = u_{W,beta}(j) - u_G(j)
Delta_v_k(d) = v_{W,beta}(d) - v_G(d)
Delta_phi_k(d) = sum_{j=1}^{d-1} Delta_u_k(j) + Delta_v_k(d)
```

(verified to <2e-15 at d in {1, 5, 74, 500}).

## 4. D_max and the hazard-level geometric tail (decided: D_max = 504)

Owner decision 2026-08-09, sharpening the earlier cap proposal: the cap is
defined **at the hazard level**, not by clipping Delta_phi:

```
j <= D_max:  Delta_u_k(j) as in Section 3
j >  D_max:  Delta_u_k(j) = 0
d <= D_max:  Delta_v_k(d) as in Section 3
d >  D_max:  Delta_v_k(d) = 0
```

Equivalently: the duration model's hazard follows the Weibull for ages up
to 504 trading days (2 years) and **reverts exactly to the geometric
reference's hazard beyond** — a spliced distribution that is still a
proper duration distribution (hazards in (0,1), geometric tail sums to 1).
A segment that survives past 504 keeps the excess it accumulated in the
memory zone and accrues nothing further.

This one convention simultaneously removes both failure modes the
adversarial check found in the uncapped form: the unbounded fragmentation
pressure on real multi-year segments at beta > 1 (+14 to +183 nats), and
the unbounded never-switch subsidy at beta < 1. Total |excess| per segment
is bounded by the memory-zone accumulation (order of a few nats at the
anchors measured).

Precedents: Durland & McCurdy (1994) freeze their hazard beyond a memory
cap tau (tau = 9 quarters — chosen there by in-sample likelihood search, a
selection method this repo forbids; we set D_max a priori like Lam
1997/2004's 40 quarters); Langrock-Zucchini HSMM-as-HMM approximations use
exact-duration-up-to-N plus a geometric tail — literally this device.
Rationale for 504 specifically: only 3-6 segments per market exceed 504
days on the sealed canonical paths — too few to identify hazard shape
beyond it; and the failure mode DA-JM targets (short re-entries) lives at
young ages. If the cap binds often it is reported as a limitation.

## 5. Anchors: restricted mean, interior segments, per (market, state), from v12

Owner decision 2026-08-09 (replacing v1's sigmoid back-out AND the interim
full-mean proposal):

```
mu504_{m,k} = mean( min(D_i, 504) )
```

over the **interior** segments i of market m, state k, of the **v12**
canonical monthly-selected fixed-JM delay-1 state path — excluding the
first segment (left-censored: the regime may predate the OOS window) and
the final segment (right-censored: still running at sample end). Then for
every beta arm, solve the scale so the restricted mean matches:

```
E_{pi}[ min(D, 504) ] = sum_{d=1}^{504} S_pi(d) = mu504_{m,k}
```

once for the geometric reference (giving pi*_{m,k}) and once per beta
(giving pi_{beta,m,k}).

Verified properties (fact-finding NOTE):

- **Well-posed**: E[min(D,504)] depends only on hazards at ages < 504 —
  entirely inside the memory zone, independent of the tail convention —
  and is strictly increasing in pi, so the solve has a unique root for any
  target in (1, 504).
- **Identity-safe**: at beta=1 the anchor equation is the geometric
  reference's own equation, so pi_1 = pi* exactly and Section 2's
  objective identity holds. Implementation requirement: **special-case
  beta=1 to Delta == 0 exactly** — a root-solver residual (~1e-15) is
  enough to flip exact DP ties, and the identity gate is bit-for-bit.
- **Re-anchoring per beta is load-bearing, not cosmetic**: at a shared pi,
  q(1) = 1 - pi identically in beta — short segments would not be
  discriminated at all. Mean-matched re-anchoring is what makes beta > 1
  genuinely surcharge short segments (at anchor 74: Delta_phi(1) = +1.16
  nats at beta=1.25, +2.30 at beta=1.5) while subsidizing near-anchor
  lengths — the mass-to-the-middle reshaping.
- **Robustness of the anchor itself**: the restricted mean caps the
  influence of the few multi-year runs that dominate a full mean (US bull:
  full mean 464.5 vs median 97 over only 15 segments) and is consistent
  with the memory zone: the statistic never looks past the age range where
  the model has memory.
- Anchors are computed from sealed v12 artifacts (deterministic, never
  searched) and are disclosed as in-sample-flavored development anchors.
  **They must be recomputed from v12, not v11** — v12 changes DE's grid
  and therefore DE's canonical path.

## 6. Interpretation of beta (excess form)

`Delta_u_k(j)` is the excess marginal cost of staying one more day at age
j, relative to the geometric reference with the same restricted-mean
persistence:

- beta = 1: zero everywhere (the identity).
- beta > 1 (mean-matched): negative at young ages up to a crossover
  (subsidize staying while the regime is young), positive past it
  (pressure to leave old regimes); terminating very young segments carries
  a surcharge (Delta_phi(1) > 0). Net effect: segments are pushed toward
  the anchor scale — few 1-5-day flickers, fewer indefinitely-old regimes
  inside the memory zone. This is the direction the August-2022
  lagged-capguard autopsy motivates ("re-enter mid-chop, get run over" =
  a short young segment that should have been suppressed) — hence
  **beta = 2.0 is the preregistered PRIMARY arm**.
- beta < 1 (mean-matched): the reverse reshaping (heavier mass at both
  very short and very long durations) — the direction the daily
  latent-state HSMM literature reports (Bulla & Bulla 2006: NB shape
  0.02-0.33 in both states, effective Weibull beta ~0.4-0.6). Hence
  **beta = 0.5 is the ADVERSARIAL / opposite-direction control**, not a
  co-equal candidate: if 2.0 fails and 0.5 wins, the primary hypothesis is
  REFUTED and 0.5 seeds a new frozen hypothesis for a later experiment —
  the paper does not silently become "the beta=0.5 model".

Convexity argument for the monotonicity of `j^beta - (j-1)^beta` (v1
receipt): second difference positive for beta>1, negative for beta<1;
e.g. beta=2 gives 2j-1 (increasing), beta=0.5 gives 1, 0.414, 0.318, ...
(decreasing).

## 7. What was retracted, and the augmented DP

**RETRACTED (v1 Section 7): `pi = sigmoid(lambda)`.** The idea was to
reuse the calibrated lambda as logit(pi). Empirically fatal at this repo's
lambda scales (fact-finding NOTE, all numbers verified):

- lambda >= ~37: `sigmoid(lambda)` rounds to exactly 1.0 in float64 →
  hazard 0, v = +inf — the model can never switch at all. Most calibrated
  lambdas (40-1000) hit this.
- lambda = 20 (no overflow): the beta modulation of the stay costs
  accumulates to ~0.019 nats over 3000 days — inert — while the one
  surviving effect (a v-term modulation ~ (beta-1)*log d) has a PERVERSE
  sign: it rewards terminating old segments. A log-space implementation
  would "fix" the overflow and silently ship that perverse residual —
  the trap is documented here so the retraction is understood as "wrong
  scale anchor", not "numerical bug to patch".
- Root cause: implied durations at calibrated lambdas are astronomical
  (lambda=20 → E[D] ~ 4.9e8 days; lambda=220 → 3.5e95) against observed
  segment means of 130-1190 days. **In the JM, durations are loss-driven,
  not penalty-driven** — lambda is a loss-scale smoother, not a duration
  prior, and cannot be converted into one.

**Augmented DP (unchanged in structure from v1, costs now the excess):**
state (k, d), d saturating at D_max (absorbing age bucket, zero excess
inside it):

```
V_t(k,d) = L(x_t, theta_k) +
    { V_t-1(k, d-1) + Delta_u_k(d-1)                          stay, d >= 2
    { min_{k' != k} [ min_d' V_t-1(k',d') + Delta_v_k'(d') ] + lambda
                                                              switch in, d = 1
```

(the constant lambda rides along exactly as in the classic DP; Delta terms
vanish at beta=1 leaving the classic recursion). Right-censoring is exact:
the final open segment accumulates only its Delta_u sum =
-log[S_beta/S_G](d) — the textbook censored-likelihood contribution.
Left-censoring (owner decision): the FIRST segment of each trailing
window charges no Delta at all (its age is unknown; assigning it age 1
would misattribute the young-age cost schedule precisely where beta
acts). Complexity O(T*K*D_max) with the switch-in min memoized once per
source state per timestep.

**M-step invariance (v1 receipt, unchanged):** Delta_phi has zero
theta-dependence, so for any fixed path the M-step is identical to classic
JM's for any beta; combined with the beta=1 objective identity, induction
over coordinate-descent iterations gives full-fit bit-for-bit reproduction
at beta=1 — the identity gate.

**Integration facts (fact-finding NOTE, file:line verified):** candidate
states are produced by a DAILY fresh forward-DP decode of the trailing
3000-row window with frozen centroids — there is no persistent duration
counter anywhere, so no reset-vs-carry question exists; the left-censor
convention above is the whole boundary story. Scenario arms keep candidate
columns == the lambda grid, so the monthly CV machinery needs zero API
change. The lambda-monotonicity gate's argument (pointwise min of affine,
nondecreasing-in-lambda path objectives) still holds at fixed beta because
Delta_phi is lambda-independent per path; cross-beta comparisons need
their own gate. The augmented DP needs a custom fit loop (JumpModel.fit
would treat every (k,d) meta-state as a cluster; precedent:
simple_jm_fitting's custom E/M loops).

## 8. Design decisions (owner, 2026-08-09) and what remains open

Decided (frozen intent; the experiment spec will restate them verbatim):

1. **D_max = 504** trading days, hazard-level geometric tail (Section 4).
2. **Left-censor** the first in-window segment (Section 7).
3. **No duration state across refits** — architecture fact, nothing to
   decide (Section 7).
4. **Anchors**: restricted mean to 504, interior segments only, per
   (market, state), computed from **v12**, re-anchored per beta
   (Section 5).
5. **beta roles**: 2.0 PRIMARY (preregistered, motivated by the
   August-2022 autopsy BEFORE DA-JM existed), 0.5 ADVERSARIAL control,
   1.0 IDENTITY gate. No winner selection between 0.5 and 2.0 on the
   evaluation sample.
6. **Success criterion** (spec-level, recorded here for completeness):
   primary is Delta_m = Sharpe_DA - Sharpe_fixedJM per market at delay 1.
   *Directional support*: Delta > 0 in US, DE, JP. *Statistical support*:
   paired moving-block bootstrap on the two daily return streams (same
   block indices both streams), 95% one-sided lower bound > 0 in all
   three markets (intersection-union test — no extra multiplicity
   correction needed for the all-three claim). Delays 5/10 are robustness
   reporting and cannot rescue a primary failure. A separate *economic*
   tier is Sharpe_DA > max(JM, HMM, B&H). No hard minimum Sharpe delta is
   set; "statistically supported" and "economically material" are kept as
   distinct labels.
7. **Gates before any real-data P&L** (mechanism gates only, never
   profitability evidence): beta=1 full-fit bit-for-bit identity;
   synthetic DGP recovery (planted beta=2 duration structure must be
   recovered better than by memoryless JM, per a metric frozen in the
   spec; planted geometric must show NO improvement); flat-loss
   adversarial case (excess costs must not induce periodic switching to
   harvest negative Delta_phi); brute-force DP parity on small problems.

Still open (to pin in the spec, none block the doc):

- Bootstrap block length and the exact synthetic-DGP recovery metric.
- Per-state beta: explicitly deferred (one new parameter only).
- v12 anchors: numeric values await the v12 seal (gated on the
  n_init=180 convergence stress test, registry
  `v12-de-ninit180-stress-gate`).

## 9. What this document is not

No frozen experiment spec, no code, no anchors computed yet. Order fixed
by the owner 2026-08-09: v12 stress gate → v12 reseal → (parallel: lambda50
donor rebuild) → this doc → second novelty pass completes → freeze DA-JM
spec → implement + gates → P&L.
