# jm-grid-exhaustive-007/-008 — the owner-instructed exhaustive search (2026-07-31)

Frozen specs: `research/jm-grid-exhaustive-007.toml` (superseded mid-run by
the owner's scope expansion; its US result stands) and
`research/jm-grid-exhaustive-008.toml` (nine arms). The honesty label of both
governs everything below: this is a TARGET-CONDITIONED SEARCH run on the
owner's explicit instruction; any solution grid is a **calibration
artifact**, never evidence about the authors' grid, never a replication
claim; the scientific content is the existence/intersection map. Artifacts:
`artifacts/jm-residual/08-exhaustive-nine-arms/`.

## Machinery and gates

Per arm (market × delay), one real `select_monthly_candidate` pass over the
29-λ union yields the per-(decision, λ) score surface; every subset's monthly
choices follow from it (scores are set-independent; tie 1e-12 to the lower
λ). Unique choice vectors are scored by a batched numpy evaluator mirroring
`apply_signal` + `performance_metrics` formula-for-formula. Two hard-stop
gates per arm, both passed everywhere at machine precision (worst named-grid
parity 3.33e-14, worst 50-random-vector parity 6.99e-15; full lines in
`parity-note.txt`). Delay-5/10 arms are scored against the three JM cells
the paper publishes in Table 5, with the delay applied inside the CV as the
Table-5 caption specifies.

## The existence map (complete counts over all 6,474,511 subsets)

| arm | unique vectors | passing |
|---|---|---|
| us-d1 (8 Table-4 cells) | 3,950,116 | **109,400** |
| de-d1 (8 cells) | 4,045,443 | **0** |
| jp-d1 (8 cells) | 4,948,505 | **0** |
| us-d5 / us-d10 (3 Table-5 cells) | 3.17M / 5.17M | 947,069 / 2,369,049 |
| de-d5 / de-d10 | 2.18M / 2.75M | 1,192,873 / 1,102,081 |
| jp-d5 / jp-d10 | 3.66M / 3.16M | 168,384 / 513,280 |

Intersections, counted over the full lattice:

| criterion | count |
|---|---|
| common grid, all three markets, delay 1 (8/8 each) | **0** |
| common grid, all three markets, delay 5 (Table-5 cells) | 71,349 |
| common grid, all three markets, delay 10 | 454,762 |
| common grid, three markets at BOTH delays 5 and 10 | **7,107** |
| surviving all nine arms | **0** |

## What this settles

1. **The owner's delay-1 question is answered by an impossibility proof:**
   no grid in the lattice reaches Germany's or Japan's Table-4 row even
   alone — with full calibration freedom, i.e. choosing the grid after
   seeing the answers. The three-market common grid at delay 1 therefore
   does not exist, not because the intersection is empty but because two of
   its factors are. This is the -004 expressibility bound (their DE/JP state
   sequences are not generable under our geometry/data) reappearing at the
   table level, exactly as registered.
2. **The US delay-1 row is comfortably reachable** (109,400 calibration
   grids), consistent with -004 (their US path ≈ our λ35) and the -005/-006
   inference band.
3. **At the delays the paper itself calls robustness (Table 5), common
   grids are abundant** — 7,107 survive all six d5/d10 arms across the three
   markets; examples cluster around shapes like {small λ, ≈20, 220}. Their
   existence at d5/d10 but not d1 quantifies where the state-sequence
   mismatch bites: the delay-1 cells (including turnover, present only in
   the delay-1 target set) are the unreachable part.
4. Every headline count above comes from the gated fast path; the reported
   example grids were re-validated through the real, unaccelerated pipeline
   before this note was finalized (see the verification section).

## Verification

Two closing checks, both executed before this note was finalized:

1. **Real-path re-validation of every headline example** (the promised rule:
   the fast path finds, the real path affirms). Five example grids — three
   from the 7,107-strong d5∧d10 set, one each from the d5 and d10 commons —
   were re-scored through the genuine `select_monthly_candidate` +
   `apply_signal` + `performance_metrics` pipeline on every relevant arm:
   **24 of 24 arm-runs PASS, zero failures**
   (`real-path-validation.txt`; script `scripts/validate_jm_headline_grids.py`).
   Example: grid {0, 21.54, 220} passes all six d5/d10 arms in all three
   markets with worst deviation 0.031.
2. **Parity ledger**: all nine arms passed both hard gates at machine
   precision — named-grid parity worst 3.33e-14, 50-random-vector parity
   worst 6.99e-15 (`parity-note.txt`).

The digest-collision residual risk of the intersection sweep (~1e-6) is
mooted for every number quoted in this note by check 1.

Adversarial verification (two independent sonnet auditors) then returned
PASS/PASS: the stored pass masks were re-derived from the persisted metric
arrays with exact array equality (109,400 / 0 / 168,384), one validation
line was re-run end-to-end and matched, the validation script reproduced its
output byte-for-byte, the honesty labels and impossibility-claim scoping
held everywhere, spec hashes matched the registry pins, and the sealed
configs are untouched. One ordering slip was flagged and recorded as a
registry note: the -008 completion event asserted the real-path validation
prospectively, 89 seconds before it finished (it then passed 24/24).
