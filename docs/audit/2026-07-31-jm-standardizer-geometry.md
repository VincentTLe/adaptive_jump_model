# jm-standardizer-geometry-002 / -003 — the geometry axis, opened and closed (2026-07-31)

Frozen specs: `research/jm-standardizer-geometry-002.toml`,
`research/jm-standardizer-geometry-003.toml` (both registered before their
runs; -003 declares its sequential-testing risk in the spec). Artifacts:
`artifacts/jm-residual/02-standardizer-geometry/`, `.../03-frozen-initial-scaler/`.
All EXPLORATORY; nothing adopted; the sealed config is unchanged.

## Owner correction, recorded first

-002 as frozen included a clip-3σ variant labeled "the author's example-code
recipe". The owner re-instructed — repeating a standing CLAUDE.md rule that
this session violated in spirit — that **DataClipperStd / 3σ clipping is
example code for a different dataset, is nowhere in the paper, and must not
be run as an author-method candidate at all**; provenance labels do not make
running it acceptable. That variant (V2) is **withdrawn from
interpretation** (registry AMENDED event). No conclusion depended on it: V2
tracked V1 pointwise, so the clip was immaterial where it was measured, and
nothing was adopted from any variant. -003's justification is likewise
reframed by registry entry: its a-priori ground is the exhaustive
enumeration of scaler-fitting cadences, not any example-code pattern.

## The question and the instruments

-001 ended with the registered finding that the paper's own Table-3
persistence curve does not reproduce under our sealed geometry (ratio ≈ 0.75
at every λ, including λ = 0 where the λ scale plays no role). The geometry
family testable a priori has exactly three members — the scaler can be fit
on the expanding history (V0, sealed), per refit window (V1), or once on the
first window and frozen (V3). Instruments, read in registered order: the
Table-3 curve (mechanism), the authors' printed US anchor (30 shifts /
19.7% bear), and the Table-4 JM cells (conditional).

Gate: V1 was computed twice — through `fixed_jm_states` and through the
probe-local loop — and agreed exactly in all three markets (64,338 / 64,026 /
62,052 cells; committed parity note).

## Results

**I1 — the paper's curve is bracketed by the family and matched by no member
(US, shifts/year, 1982–2023):**

| λ | Shu | V0 expanding | V1 per-refit | V3 frozen-initial |
|---|---|---|---|---|
| 0 | 9.7 | 7.24 | 15.07 | 5.88 |
| 5 | 2.7 | 2.00 | 4.17 | 2.21 |
| 15 | 1.7 | 1.33 | 2.40 | 1.36 |
| 35 | 0.8 | 0.57 | 1.31 | 0.69 |
| 70 | 0.5 | 0.48 | 0.88 | 0.55 |
| 150 | 0.4 | 0.29 | 0.43 | 0.38 |

Registered I1 rule (≥ 5 of 6 λ within 0.15): V0 2/6, V1 1/6, V3 3/6 — **no
member passes**. The axis is simultaneously *dominant* (V0→V1 swings the λ=0
rate by a factor of two, dwarfing every grid effect measured in -001) and
*unidentifiable* (the published curve lies between the cadences at low λ and
above V0 at high λ; no cadence tracks it).

**I2 — the US selected-path anchor, per variant:** V0 34 shifts / 21.9%
bear; V1 30 / 24.4%; V3 **32 / 19.4%** against the authors' printed 30 /
19.7% — descriptively the closest, adopted nowhere.

**I3 — Table-4 JM cells (conditional on geometry, tolerance 0.05):**

| variant | US | DE | JP |
|---|---|---|---|
| V0 sealed | 4/8 (0.75 / 0.50) | 3/8 (0.30 / 1.17) | 3/8 (0.20 / 1.41) |
| V1 per-refit | 5/8 (0.60 / 0.44) | 5/8 (0.35 / 1.03) | 3/8 (0.18 / 2.01) |
| V3 frozen-initial | 6/8 (0.78 / 0.47) | 4/8 (0.30 / 1.03) | 5/8 (0.24 / 1.33) |

(cells: within-tolerance count, then Sharpe / turnover.) No variant
dominates; V1 lands US turnover on 0.441 against the published 0.44 while
doubling JP turnover; V3 is the best overall count but overshoots US Sharpe.
Per the frozen decision rules these numbers ground no adoption and no
performance claim.

## Registered closure

Per -003's `registered_interpretations[2]` and `family_exhaustion` clause:
**the geometry axis closes as bracketed-but-unidentifiable from public
information.** The paper says "standardized" and nothing else; its own
persistence curve is inconsistent with every cadence that word can denote on
our data. Together with -001 (no λ grid reproduces Table 4's JM row under
any tested geometry) the JM residual attribution now moves to data/vintage
differences — the same terminus as the HMM turnover row, reached by the same
mechanism-first route. Further geometry variants are barred by the spec
(`no_further_geometry_variants = true`); any reopening requires a new
primary source (e.g. Chenyu Yu's dissertation when DataSpace posts it).

## Adversarial verification and remediation

Three independent auditors reviewed both experiments. The recomputation
auditor reproduced every curve value-for-value from the state artifacts,
confirmed V0's curve is identical to -001's row-for-row, and matched the V0
Table-4 rows to the sealed run's own metrics; it also independently
identified the sealed v9.4 run among three candidates by numeric match. Two
process findings came back and both were remediated by *executed checks*,
not notes:

1. **The -002 spec's V0 replay parity gate had never been executed** (the
   probe read the sealed states directly). Remediation: the sealed pipeline
   was recomputed through `fixed_jm_states` on the sealed features in all
   three markets and equals the sealed `jm-states.csv` exactly
   (64,338 / 64,026 / 62,052 cells; appended to the -002 parity note).
2. **The custom-scaler fit path lacked the fit-time gates** that
   `fit_fixed_jm_window` enforces (finite objective, binary states).
   Remediation: the gates were added to the probe, and V3 was re-run under
   them in all three markets — every refit passed, and the gated replay is
   cell-for-cell identical to the committed artifacts (a determinism replay
   as a side effect; -003 parity note).

The compliance auditor separately confirmed: -003's FROZEN registry event
precedes its results; the sequential-testing and family-exhaustion clauses
are declared in the spec; the drawn interpretation matches the registered
rule given the numbers; and the V2 withdrawal is recorded with no surviving
conclusion depending on V2 (its immateriality claim checks out in the
artifacts).
