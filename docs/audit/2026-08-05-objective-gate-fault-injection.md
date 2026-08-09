# Objective-gate fault injection — independent verifier report (2026-08-05)

Scope: `_require_nondecreasing_jm_objectives` in `src/adaptive_jump/models.py`,
after its conversion from consecutive-pair to running-max comparison. The gate
was written by one agent and fault-injected by a separate agent that did not
write it, per the standing separate-audit rule and the
`independent_scorer_fault_injection` health gate of `research/ajm-ext-001.toml`.

Contract under test: a total decrease from the running maximum beyond one
solver tolerance (`jm_protocol.tol * observation_loss_scale`) must raise;
anything less must pass.

## Verdict: CERTIFY

Probe: monkeypatched `adaptive_jump.models.JumpModel` (same pattern as
`_patch_jump_objectives` in `tests/test_models.py`), driving the real
`fit_fixed_jm_window`.

| Case | Result |
| --- | --- |
| single-lambda grid | no raise, no crash |
| two exactly equal objectives | passes |
| 10 rungs each dropping 0.75 tol | raises at rung 2, naming the running-max lambda (0), not the adjacent one |
| dip, full recovery to a new max, dip again (each 0.9 tol) | passes; max advance proven — the same sequence with a final 1.5 tol dip raises naming the new max |
| rise then drop 1.5 tol from the new max | raises naming the new max's lambda — `best_penalty` updates |
| dip 0.75 tol then cumulative drift 1.5 tol | raises, skipping the intermediate lambda |
| NaN / inf objective | rejected upstream by the pre-existing non-finite check (`models.py`, inside the fit loop, before the gate) |
| loss-scale coupling (tol 1e-8, scale 4.0) | 2 tol drop passes at scale 4, raises at scale 1; 6 tol drop raises at both — pins `absolute_tolerance = tol * observation_loss_scale` |

Additional adversarial cases consistent with the contract: 0.81 tol total
drift over 9 rungs passes; reversed configured grid order still gates in
sorted-lambda order; negative objectives raise correctly; a drop of exactly
one tolerance passes (strict `<`).

## Documented boundary, not a defect

With objectives near 1e10 and tol 1e-8, the `32*ulp(scale)` floor dominates
(~6.1e-5), so a 1000-tol drop passes. Only reachable when |objective| is
above ~1e6 with tol 1e-8; real objectives here are O(100). This is the spec's
own float-noise floor. Cosmetic: `%g` formatting prints both values as "90"
for sub-tolerance differences; the named lambdas are correct.

## Suite state at certification

`pytest tests/test_models.py -q`: 35 passed. Full suite: 563 passed,
1 pre-existing documented xfail (frequency-ladder). `ruff check` clean on the
changed files; two E501 in `tests/test_heldout_delay_audit.py` are
pre-existing on the committed baseline (verified via stash).
