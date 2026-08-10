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
  from the approved canonical baseline, to be selected before any anchor is
  computed; beta roles 2.0 primary / 0.5 adversarial / 1.0 identity). The
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

Already-published neighbors (two independent novelty passes, registry
`da-jm-novelty-sweep-2026-08-08` and
`da-jm-novelty-second-pass-2026-08-09`): CJM's state-pair matrix
`Lambda[i][j]` is zero-diagonal and strictly first-order; no Mulvey-lab
paper implements any duration/hazard penalty; across 51 forward citations
and every 2025-2026 SJM-lineage paper (penalty form verified in text) the
constant per-transition lambda is unmodified.

**Frozen novelty wording (owner instruction, 2026-08-09).** The only
claim permitted is the narrow intersection one:

> To our knowledge, no prior work incorporates explicit regime-duration /
> hazard dependence into the penalized dynamic-programming objective of
> the Statistical Jump Model family.

Explicitly forbidden phrasings: "we are the first duration-aware regime
model" (false — Sichel 1991, Durland & McCurdy 1994, Bulla & Bulla 2006);
"no duration-aware Jump Model exists" (too strong for the coverage
achieved); "DA-JM is novel" (too vague). The novelty is the intersection
SJM penalized clustering + explicit duration hazard + augmented DP, and
because 9 citing works were unreachable and Google Scholar is not
crawlable, the manuscript must say "to our knowledge", never a categorical
non-existence claim.

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

## 5. Anchors: restricted mean, interior segments, per (market, state)

Owner decision 2026-08-09 (replacing v1's sigmoid back-out AND the interim
full-mean proposal):

```
mu504_{m,k} = mean( min(D_i, 504) )
```

over the **interior** segments i of market m, state k, of the canonical
monthly-selected fixed-JM delay-1 state path of the **baseline explicitly
approved and frozen for the eventual DA-JM experiment** (see the note at the
end of this section) — excluding the
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
- Anchors are computed from sealed artifacts (deterministic, never searched)
  and are disclosed as in-sample-flavored development anchors.

**Which baseline the anchors come from is NOT yet decided, and no anchor may
be computed until it is.** Earlier versions of this document said the anchors
come from v12. That is withdrawn: **v12 failed** its own pre-frozen
convergence gate (`v12-de-ninit180-stress-gate`, 2026-08-09) and was stopped,
it is never retroactively rescued, and no v12 artifact exists to anchor to.
The anchors come from the canonical baseline that is explicitly approved and
frozen for the DA-JM experiment itself — which today is undecided, because the
currently canonical v11-ninit60 is known defective on DE (its grid fails the
admissibility rule that selected it) and its replacement protocol (working
name v13) requires an estimand-terms optimizer-fidelity requirement frozen
before it runs. Since the choice of baseline changes each market's canonical
path, and DE's most of all, computing anchors before the baseline is selected
would silently fix them to a path that may not be the one the experiment uses.
The anchor computation is therefore blocked on that decision, not merely
pending.

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
   (market, state), re-anchored per beta (Section 5). The baseline they are
   computed FROM is deliberately not decided here — see "Still open" below;
   it is the baseline explicitly approved and frozen for the DA-JM
   experiment, and **no anchor is computed until that baseline is selected**.
5. **beta roles**: 2.0 PRIMARY (preregistered, motivated by the
   August-2022 autopsy BEFORE DA-JM existed), 0.5 ADVERSARIAL control,
   1.0 IDENTITY gate. No winner selection between 0.5 and 2.0 on the
   evaluation sample.
6. **Evaluation hierarchy** (owner decision 2026-08-09, registry
   `da-jm-evaluation-hierarchy-2026-08-09`). This REPLACES the earlier
   two-tier "directional + statistical support" criterion, whose
   statistical tier required a 95% one-sided bootstrap lower bound > 0 in
   all three markets. **That gate is withdrawn**: a resampling interval is
   not permitted to decide success or failure here, in either direction.
   Evidence is weighed in this order, strongest first:

   1. **Effect size.** Report `Delta_m = Sharpe_DA,m - Sharpe_fixedJM,m`
      per market at delay 1, against a SINGLE primary comparator (the
      canonical fixed JM), as the raw number — never hidden behind
      significance. A vector like (+0.12, +0.09, +0.07) is interesting and
      (+0.003, +0.002, +0.001) is not, and no interval is needed to tell
      them apart. **No fixed threshold** (in particular no ±0.03) is
      imposed.
   2. **Cross-market transport.** The primary criterion: Delta > 0 in US,
      DE and JP for the same frozen variant, with non-trivial magnitude.
   3. **Mechanism consistency.** If the claim is duration dependence, the
      model must demonstrably change short segments, hazards, whipsaws and
      the relevant transition episodes in the PREREGISTERED direction. A
      Sharpe increase whose duration mechanism does not behave as the
      theory predicts is not support.
   4. **Robustness.** Delays 1/5/10, transaction costs, the prespecified
      baseline family (item 6b), the optimizer seed families, subperiods.
   5. **External transport** — the gold standard. Freeze DA-JM completely,
      then apply it to markets or periods never used to conceive it.
      Delta > 0 on untouched markets outweighs any interval computed on
      development US/DE/JP.
   6. **Resampling evidence — LAST, and descriptive only.** Reported as a
      description of uncertainty, never as a guillotine: neither "interval
      contains 0" nor "interval excludes 0" decides anything. If reported
      it must use the **studentized** Sharpe-difference procedure (Ledoit &
      Wolf 2008), not a raw percentile interval, with the **block length
      pinned before any P&L** — trying several and keeping the nicest is
      forbidden. Label it **paired resampling evidence on repeatedly
      inspected development data**; resampling does not undo selection
      history (White 2000).

   **Episode-level reporting is mandatory** for any challenger that acts
   only at transitions — which DA-JM plausibly is. Alongside daily returns,
   report per-divergence-episode `Delta_R_e` with win/loss counts, median,
   distribution, **concentration** (what share of the net comes from the
   largest one, three and five episodes), regime context, and
   crisis-versus-normal split. The confirmed_2d study
   (`artifacts/confirmed2d-episodes/`) is the worked example of why: its
   aggregate Sharpe advantage dissolves into a handful of single days once
   viewed this way.
6b. **Baseline-uncertainty robustness** (owner addition, 2026-08-09).
   Because the exact Shu JM is demonstrably underidentified from public
   information, "DA-JM beats OUR calibrated reconstruction" invites the
   obvious reviewer question. The primary comparator stays single, but the
   spec additionally reports Delta (DA structure vs fixed structure) under
   a **small prespecified baseline family**: the canonical calibrated
   grid, the Shu arXiv-v1 disclosed grid, and the Table-3 illustrative
   grid (labelled as illustrative, never as a disclosed production grid).
   **No winner is selected among them** — the point is whether the DA
   structure helps under each plausible reconstruction, not which
   reconstruction flatters it.
7. **Gates before any real-data P&L** (mechanism gates only, never
   profitability evidence):
   - beta=1 full-fit bit-for-bit identity with the classic JM;
   - **synthetic recovery over three DGP classes** (owner correction,
     2026-08-09 — planting beta=2 alone is a home game, since the data
     would come from DA-JM's own assumed family):
     **(A) geometric / memoryless null** — DA-JM must show NO artificial
     advantage over classic JM;
     **(B) discrete-Weibull, beta=2, correctly specified** — DA-JM must
     recover the duration mechanism better than memoryless JM on a metric
     frozen in the spec;
     **(C) out-of-family duration** (e.g. negative-binomial / explicit
     HSMM sojourns) — DA-JM need not recover the exact parameters, but
     must respond in the correct DIRECTION when genuine duration
     dependence is present;
   - flat-loss adversarial case: excess costs must not induce periodic
     switching purely to harvest negative Delta_phi;
   - brute-force DP parity on small problems.

Still open (to pin in the spec, none block the doc):

- Bootstrap block length (must be fixed before any P&L — a practice
  requirement, not a gate; the resampling itself is descriptive) and
  the exact synthetic-recovery metric for classes A/B/C.
- Per-state beta: explicitly deferred (one new parameter only).
- **Which baseline the anchors are computed from — the blocking one.**
  **v12 FAILED and is not a candidate.** Its pre-frozen convergence gate
  stopped it (registry `v12-de-ninit180-stress-gate`, 2026-08-09 — 254/255
  DE fits identical at n_init=180, one window improved; the rule required
  objectives AND states to match), that record stands permanently, and it is
  never retroactively rescued. Nothing in this document may say the anchors
  come from v12.

  The optimizer-fidelity characterization is now COMPLETE (verdict
  PROPAGATING, with the paired L4 delta the level that applies to a
  challenger). Its result is a statement about the CRITERION — objective
  identity across restart depths is too strong for an estimand-fidelity
  policy — and not a rehabilitation of any grid. The currently canonical
  v11-ninit60 remains known defective on DE, and its replacement protocol
  (working name v13) requires an estimand-terms fidelity requirement frozen
  BEFORE it runs; that has not happened, so **no approved DA-JM baseline
  exists yet**. Anchors come from the canonical baseline explicitly approved
  and frozen for this experiment, and cannot be computed before that
  selection is made.

## 9. What this document is not

No frozen experiment spec, no code, no anchors computed yet.

Order as revised by the owner after the v12 gate failed (2026-08-09):
v12 stress gate (**DONE — FAILED, v12 stopped, never rescued**) →
optimizer-fidelity characterization with independent seeds (**DONE —
PROPAGATING, paired L4 verified**) → second novelty pass (**DONE**) →
**decide the canonical baseline for this experiment**, with no further
n_init escalation under any outcome and with any replacement protocol's
fidelity requirement frozen in estimand terms before it runs (**OPEN — this
is the blocker**) → compute anchors from that approved baseline → freeze the
DA-JM spec → implement + mechanism gates → only then P&L.

No anchors have been computed. No spec is frozen. No code exists.
