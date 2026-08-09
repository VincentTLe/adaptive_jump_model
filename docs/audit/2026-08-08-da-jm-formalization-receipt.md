# DA-JM formalization — independent verification receipt (2026-08-08)

Scope: full mathematical recomputation audit of
`docs/theory/da-jm-formalization.md`, performed by a separate agent that did
not write the document, from first principles (rederived every claim, not
just checked internal consistency). No code exists for DA-JM; this is a
math-only check of a pre-freeze formalization document.

## Verdict: CONFIRMED, with one gap identified and closed

### Confirmed by independent rederivation

1. **Discrete-Weibull survival/hazard closed forms** (Sec 3): `S_k(d) =
   pi_k^((d-1)^beta_k)` (telescoping sum), `h_k(d) = 1 -
   pi_k^(d^beta_k-(d-1)^beta_k)`, constant at beta_k=1. Exact match.
2. **Hazard-decomposition identity** (Sec 4): `-log q_k(d) = sum_{j=1}^{d-1}
   u_k(j) + v_k(d)`. Verified two independent ways (product-telescoping and
   direct factoring); both agree, including the `d=1` boundary case.
3. **Closed form `u_k(j) = (j^beta_k-(j-1)^beta_k)*lambda_k`**: direct
   substitution from the hazard formula, exact.
4. **Reduction theorem** (Sec 5, the highest-risk part):
   - beta=1 collapses the augmented DP to a first-order DP with **nonzero
     diagonal** `Lambda[k][k]=lambda_k` — correctly identified as landing in
     the state-pair-penalty family, not the zero-diagonal classic JM, for
     general (asymmetric) `pi_1 != pi_2`. Confirmed by full re-derivation
     (induction on `V~_t(k) = min_d V_t(k,d)`), not just accepted.
   - Under the further symmetric assumption `pi_1=pi_2=pi`, the
     constant-factoring identity `min(A+c,B+d)=c+min(A,B+d-c)` correctly
     yields `lambda_effective = logit(pi)`. Confirmed generally (proved for
     any real c, not just this instance).
   - `lambda_effective>0 <=> pi>0.5 <=> E[D]>2 days`: confirmed both
     directions independently (rederived `E[D]=1/(1-pi)` from the geometric
     pmf directly, not assumed).
   - Path-independent-constant argument: the verifier proved by induction
     that `V~_t(k) = V~~_t(k) + (t-1)*lambda` holds for **all** t (not just
     asserted for the endpoint), making the "same argmin" claim rigorous.
5. **Right-censoring**: confirmed true by construction of the recursion, and
   *strengthened* — the open final segment's accumulated `u_k` cost sums to
   exactly `-log S_k(d)`, the textbook right-censored survival-likelihood
   contribution, not merely "nothing extra charged." Document updated to
   state this explicitly.

### Gap identified and closed

Section 7 (now corrected) claimed the beta=1 bit-for-bit reproduction gate
was proved by Section 5. Section 5 only proves single-DP-step path
equivalence at a *fixed* theta — a full coordinate-descent fit alternates
E-step (path DP) and M-step (centroid update) across iterations, and nothing
in the original Section 5 addressed the M-step. The verifier proved the
missing piece: `J_DA`'s theta-dependence lives entirely in `L(x_t,
theta_{z_j})`, `phi_k(d)` has zero theta-dependence, so for any fixed path
the M-step is identical in form/value to classic JM's M-step, for *any*
beta. Combined with Section 5's per-step path equivalence at beta=1,
induction over coordinate-descent iterations gives the full bit-for-bit
result. Section 7 and Section 8 (open question #6, now closed) both updated
to state this explicitly rather than leave it as an assumption.

### Minor notes, not corrections

- Complexity is `O(T*K*D_max)` only if `min_d'[V_t-1(k',d')+v_k'(d')]` is
  memoized once per source state per timestep; `O(T*K^2*D_max)` if
  recomputed per destination. Document now states the memoized form is the
  one to implement.
- Section 6's beta!=1 monotonicity claim (aging fragility / seasoning) was
  narrative in the original; the verifier supplied the convexity argument
  (second difference of `j^beta-(j-1)^beta` sign-checked at beta=2 and
  beta=0.5). Document now shows this work.
- Section 1's claim that CJM's state-pair matrix is zero-diagonal and
  strictly first-order is sourced from the earlier literature workflow's PDF
  read (`da-jm-novelty-sweep-2026-08-08`), not re-verified by this math-only
  pass — flagged as a citation dependency, not a gap in the math itself.

## Bottom line

Every mathematical claim in `docs/theory/da-jm-formalization.md` checks out
under independent, from-scratch rederivation. One real (but small, and now
closed) gap existed between the single-step proof and the full-pipeline
claim; the document has been corrected to close it rather than merely noted
as a caveat. No code has been written; the six open design questions in
Section 8 (five remain open, #6 now closed) still need explicit decisions
before any frozen experiment spec.
