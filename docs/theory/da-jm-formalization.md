# Duration-Aware Jump Model (DA-JM) — formalization

Status: **math only, no code, no frozen experiment spec yet.** This is step 3
("formalize DA-JM") of the owner's plan, following the literature-novelty
sweep (registry NOTE `da-jm-novelty-sweep-2026-08-08`). Nothing here has been
implemented or run. Everything below is derived by hand; every claim that
looked "obvious" at first pass and turned out to need a careful check is
flagged as such, because two of them did not survive the check as originally
stated.

Notation follows this repo's own TOML-spec style (`sum_t`, `argmin`, plain
ASCII), not LaTeX, to match `research/*.toml`.

---

## 1. Baseline: the classic constant-lambda JM

For a single market, K=2 states, T observations:

```
J(theta, s_1:T) = sum_t=1..T  L(x_t, theta_{s_t})
                 + lambda * sum_t=2..T  I(s_t != s_{t-1})
```

Fit by coordinate descent: fix theta, solve for s via exact DP (first-order,
`dp_tv` in this repo, `tv_jump.py:168`); fix s, update theta by cluster mean.
The DP recursion for a *constant* penalty matrix `Lambda` (this repo already
generalizes `dp_tv` to a *time-varying* `penalty_seq[t]`, but the base case
here is `penalty_seq[t] = Lambda` for all t):

```
V_t(k) = L(x_t, theta_k) + min_k'  [ V_t-1(k') + Lambda[k'][k] ]
```

Classic JM: `Lambda[k][k] = 0`, `Lambda[k'][k] = lambda` for `k' != k` (single
scalar, symmetric, zero diagonal).

**Already-published generalization** (found during the novelty sweep, not new
here): CJM (`paper/ssrn-4556048.pdf`, Sec 3.3) already allows a full
state-pair matrix `Lambda[i][j] >= 0`, still zero-diagonal, still strictly
first-order (`(s_t-1, s_t)` only — confirmed by direct PDF read, no duration
term). This matters below: part of the beta=1 reduction result lands *inside*
this already-known family, not beyond it. (Citation note: this fact is
sourced from the earlier literature workflow's PDF read, not re-verified by
the math-only independent check on this document — the math check confirmed
everything derived *from* it, not the citation itself.)

---

## 2. Duration-augmented objective

Partition the path into segments `j = 1..J`, each with state `z_j`, start
`a_j`, end `b_j`, length `d_j = b_j - a_j + 1`, consecutive segments differing
(`z_j != z_j+1`):

```
J_DA(theta, segments) = sum_j [ sum_t=a_j..b_j  L(x_t, theta_{z_j})  +  phi_{z_j}(d_j) ]
```

`phi_k(d)` is a state-specific duration cost, charged once per completed
segment, not once per timestep. This is the standard explicit-duration /
semi-Markov move (Yu 2010 survey; Guedon 2003) — nothing new in this
paragraph either. The new part is what `phi_k` is and how it plugs into
*this* DP-based (not EM-based) model, section 4 below.

---

## 3. Discrete-Weibull duration family

```
q_k(d) = pi_k^((d-1)^beta_k)  -  pi_k^(d^beta_k),     d = 1, 2, 3, ...
```

`pi_k` in (0,1) is a per-state scale parameter, `beta_k > 0` the duration-
dependence shape. (Standard "Type I discrete Weibull," Nakagawa & Osaki 1975
— citing as borrowed, per the novelty sweep's recommendation, not claiming it
as new.)

Survival function telescopes cleanly:

```
S_k(d) = P(D_k >= d) = sum_{d'>=d} q_k(d') = pi_k^((d-1)^beta_k)
```

Hazard:

```
h_k(d) = q_k(d) / S_k(d) = 1 - pi_k^(d^beta_k - (d-1)^beta_k)
```

Checked at beta_k=1: `h_k(d) = 1 - pi_k`, constant in d — confirms the
memoryless/geometric claim at the hazard level, not just at the pmf level
(the pmf-level check alone, `q_k(d) = pi_k^(d-1)(1-pi_k)`, is the easy part;
the hazard staying constant is the part that actually matters for the DP
below).

---

## 4. Augmented-state DP

State at time t: `(k_t, d_t)` — regime and consecutive days in it. Stay:
`(k,d) -> (k,d+1)`. Switch: `(k,d) -> (k',1)`.

Decompose `phi_k(d) = -log q_k(d)` into per-step increments via the hazard
(this is the standard HSMM-Viterbi trick, not new):

```
-log q_k(d) = sum_{j=1..d-1} u_k(j)  +  v_k(d)
  where  u_k(j) = -log(1 - h_k(j))     ["survive past age j" cost]
         v_k(d) = -log(h_k(d))         ["terminate exactly at age d" cost]
```

Closed form from the discrete-Weibull hazard:

```
u_k(j) = (j^beta_k - (j-1)^beta_k) * lambda_k,     lambda_k := -log(pi_k)
```

(`v_k(d)` has no equally clean closed form for beta_k != 1; not needed in
closed form for the DP itself, only for the reduction proof below.)

DP recursion:

```
V_t(k,d) = L(x_t, theta_k) +
    { V_t-1(k, d-1) + u_k(d-1)                              if d >= 2 (stay)
    { min_{k' != k} [ min_d' V_t-1(k', d') + v_k'(d') ]      if d == 1 (switch in)

V_1(k,1) = L(x_1, theta_k)     [no duration cost charged at the first obs —
                                 a boundary convention, see open question #2]
```

**Right-censoring falls out for free — and charges exactly the correct
amount, not just "nothing extra."** The final active segment (from its start
through t=T, length d) never has `v_k(d)` charged on it, because `v` is only
paid at the moment a segment *ends* (switches out) — true by construction of
the recursion, not an assumption. What it *does* accumulate is the `u_k(j)`
stay-costs for `j=1..d-1`, and Section 4's telescoping identity shows these
sum to exactly `-log S_k(d)`. That is precisely the textbook right-censored
survival-likelihood contribution (`P(D>=d)`, not the density `q_k(d)`) — the
statistically correct treatment of an unfinished segment, exactly, not just
approximately reasonable. (Independently confirmed.)

Complexity: `O(T * K * D_max)` if, for each timestep and source state `k'`,
`min_d' [V_t-1(k',d') + v_k'(d')]` is memoized once and reused across all `K`
destination states `k`; `O(T * K^2 * D_max)` if recomputed per destination.
Either is tractable; the memoized form is the one to implement. `D_max` is a
cap that must be pinned before any code runs (open question #1).

---

## 5. Reduction theorem (beta_k = 1)

This is the part that did **not** survive the first pass unchanged — the
naive "beta=1 gives back the JM" claim is directionally right but the
argument needs two real steps, and the literal target it reduces to is
**not** always the single-scalar-lambda classic JM.

**Step 1 — general (state-asymmetric) pi_1, pi_2, K=2.**

At beta_k=1: `u_k(j) = lambda_k` (constant in j, `lambda_k = -log(pi_k)`),
`v_k(d) = mu_k` (constant in d, `mu_k = -log(1-pi_k)`). Both duration terms
drop out of the DP, so `d` can be marginalized: define
`V~_t(k) = min_d V_t(k,d)`. The augmented DP collapses to:

```
V~_t(k) = L(x_t, theta_k) + min( V~_t-1(k) + lambda_k,
                                  V~_t-1(k') + mu_k' )     [k' = the other state]
```

This is a first-order DP with penalty matrix `Lambda[k][k] = lambda_k`
(nonzero diagonal), `Lambda[k'][k] = mu_k'` for `k' != k`. **This is not the
classic JM** (nonzero diagonal, and CJM's own Lambda_ij convention fixes the
diagonal at zero) — it lands in the *already-published* state-pair-penalty
family (Sec 1 above) but strictly outside its zero-diagonal convention, for
general `pi_1 != pi_2`.

**Step 2 — further special case pi_1 = pi_2 = pi (symmetric).**

Now `lambda_1 = lambda_2 = lambda` (shared), and the recursion is:

```
V~_t(k) = L(x_t, theta_k) + min( V~_t-1(k) + lambda, V~_t-1(k') + mu )
```

Factor the shared additive constant out of the min (`min(A+c, B+d) = c +
min(A, B+d-c)`, valid for any c):

```
V~_t(k) = L(x_t, theta_k) + lambda + min( V~_t-1(k), V~_t-1(k') + (mu - lambda) )
```

`lambda` is now added identically at *every* step regardless of the state
chosen — it contributes a path-independent constant `(T-1)*lambda` to the
total objective and cannot change the argmin. Dropping it:

```
V~_t(k) = L(x_t, theta_k) + min( V~_t-1(k), V~_t-1(k') + lambda_effective )
    where   lambda_effective = mu - lambda = log(pi) - log(1-pi) = logit(pi)
```

**This is exactly the classic single-scalar constant-lambda JM**, with

```
lambda_effective = logit(pi) = log( pi / (1 - pi) )
```

**Checked, not assumed:** `lambda_effective > 0` (a normal, sensible
switching penalty) requires `pi > 0.5`, i.e. expected geometric duration
`E[D] = 1/(1-pi) > 2` days. A state whose fitted duration distribution has
`E[D] < 2` days would, under this reduction, produce a *negative* effective
lambda (switching *rewarded*) — which is the correct behavior for a
short-lived, alternating-by-construction state, not a bug, but worth stating
plainly since it is not the intuition one gets from staring at the classic
JM alone.

**Statement of "JM ⊂ Duration-JM":** the correct form of this claim is
*path-equivalence*, not objective-value equality (the two objectives differ
by an additive path-independent constant) — for every theta, the beta=1,
symmetric-pi slice of Duration-JM and the classic JM (at
`lambda=logit(pi)`) have the *same* argmin state sequence. That is the
rigorous sense in which the classic JM is recovered as a special case.

---

## 6. Interpretation of beta != 1

`u_k(j) = (j^beta_k - (j-1)^beta_k) * lambda_k` is the marginal cost of
staying one more day at age j.

- `beta_k = 1`: constant marginal cost (memoryless, Section 5).
- `beta_k > 1`: `j^beta_k-(j-1)^beta_k` increasing in j — staying gets
  *more* expensive as the regime ages ("aging fragility": positive duration
  dependence, switch-pressure grows with regime age).
- `beta_k < 1`: decreasing in j — staying gets *cheaper* as the regime ages
  ("seasoning": negative duration dependence, entrenched regimes get more
  entrenched).

Why (convexity of `x^beta`, not just asserted): the second difference of
`f(j) = j^beta - (j-1)^beta` is positive for `beta>1` (convex power) and
negative for `0<beta<1` (concave power), so `f` is increasing in the first
case, decreasing in the second — e.g. `beta=2`: `f(j)=2j-1`, exactly
increasing; `beta=0.5`: `f(1,2,3,...) = 1, 0.414, 0.318, ...`, decreasing.
`beta=1`: `f(j)=1` for every j, the constant case of Section 5.

This is exactly the shape of question Sichel (1991) asked about NBER
expansion/contraction durations with a Weibull hazard — the closest prior
art found in the novelty sweep, cited here as the source of the
interpretation, not as something DA-JM discovers.

---

## 7. Parameter discipline (owner's step 5: add only beta)

Per the instruction to add exactly one new parameter, not 3-4 per state:

- `beta` shared across both states within a market (not per-state) as the
  *first* cut — matches "don't give bull/bear their own 3-4 parameters yet."
  A per-state `beta_k` is a natural follow-up, explicitly deferred.
- `pi` (equivalently `lambda`) is **not** a new parameter — back it out from
  the *already-sealed, already-calibrated* lambda via
  `pi = sigmoid(lambda) = 1/(1+exp(-lambda))`, the inverse of Section 5's
  `lambda_effective = logit(pi)`. This only holds rigorously in the
  symmetric-state reduction (Step 2) — see open question #4 for why the
  calibrated grids (multi-value, monthly-CV-selected, not a single scalar
  lambda) make this back-out non-trivial for DE/JP specifically.
- **Falsifiable gate for the eventual frozen spec:** DA-JM at `beta=1`, fed
  the *same* fixed lambda as one held-fixed comparator arm (e.g.
  `static_lambda50`, or any single month's CV-selected lambda held constant
  for that month), must reproduce the classic-JM state path **bit-for-bit**
  across a *full coordinate-descent fit*, not just a single DP step. This is
  a real regression test, not a soft sanity check. Section 5 only proves
  single-step path equivalence at a fixed theta; closing the gap to a full
  fit needs one more fact, proved here rather than left as an assumption:
  `J_DA`'s theta-dependence lives entirely in `L(x_t, theta_{z_j})` —
  `phi_k(d)` does not depend on theta at all. So for any *fixed* path, the
  theta-minimizing M-step (e.g. cluster mean) is identical in form and value
  to classic JM's M-step, for *any* beta, not only beta=1. Combined with
  Section 5's per-step path equivalence at beta=1, induction over
  coordinate-descent iterations (E-step path-equivalent ⟹ same points
  assigned to each state ⟹ M-step identical ⟹ same theta going into the next
  E-step ⟹ ...) gives the full bit-for-bit result, not just a plausible
  expectation of it. (Gap identified and closed during independent
  verification, 2026-08-08 — see the receipt.)

---

## 8. Open questions to pin before any frozen spec (not yet decided)

1. **D_max cap.** Needs a concrete number before code exists (e.g. capped at
   the 3000-observation fit window, or some smaller multiple of the longest
   observed segment in the sealed baselines — DE's slowest grid runs ~430
   days/segment on average per the -008/-009 work). Whatever is chosen must
   be justified on a priori grounds (project rule: never tune a knob against
   the fit outcome), then reported as a limitation if it turns out to bind.
2. **t=1 boundary convention.** Charging nothing for the very first
   observation (mirrors classic JM) vs. charging a `v_k(1)`-style "start"
   cost. Almost certainly immaterial for T~8600 days, but should be stated,
   not silently assumed.
3. **Duration state across monthly refits.** This repo's whole pipeline
   refits monthly (Jan/Jul) on a rolling window and reselects lambda by
   trailing CV. Does the duration counter `d` reset at each refit boundary,
   or carry through from the previous month's terminal state? This is a real
   design choice with no obvious default — carrying `d` through is more
   faithful to "the regime doesn't know a refit happened," resetting is
   simpler to implement and audit. Needs an explicit decision, not a default.
4. **Back-out of pi_k for the multi-value calibrated grids.** Section 7's
   `pi = sigmoid(lambda)` back-out is clean for a single scalar lambda
   (`static_lambda50`, or US's simpler grids). DE/JP's calibrated grids are
   multi-value CV-selected schedules, not one constant — there is no single
   "the" lambda to invert. Needs a stated convention (e.g. invert the
   *modal* monthly-selected lambda, or run the beta=1 gate per-month against
   whichever lambda CV picked that month) before the falsifiable gate in
   Section 7 can even be run on DE/JP.
5. **How is beta itself chosen?** Inner-CV like lambda currently is (same
   monthly-trailing-Sharpe selection machinery, extended to a 2-D grid over
   `(lambda, beta)`), or fixed a priori on theoretical grounds and reported
   as a limitation? The owner's plan does not yet say; this determines
   whether DA-JM inherits the exact same selection-rule machinery already in
   this repo (cheap) or needs a new one (real design work).
6. ~~M-step (centroid update) interaction~~ — **closed on paper, no longer
   open.** Proved in Section 7: `phi_k(d)` carries no theta-dependence, so
   the M-step is identical in form/value to classic JM's for any beta, given
   a fixed path. Still worth a direct code-level check once implemented (an
   assertion, not a re-derivation), but this is no longer an open design
   question.

---

## 9. What this document does *not* do

No frozen experiment spec, no code, no lambda/beta values chosen, no market
run. Per the owner's plan this is step 3 only; step 4 (the DP derivation) is
folded into Section 4 above since the two are inseparable in practice. Next
step per the owner's own plan is a decision on open questions 1-5, then a
frozen `research/*.toml` spec with its own experiment id, before any code.
