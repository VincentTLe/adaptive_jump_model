# jm-grid-identification-001 — results (2026-07-30)

Frozen spec: `research/jm-grid-identification-001.toml` (registry event with
spec hash `f33da9b4…`, frozen before any new λ was fitted). Prior art:
`fixed-baseline-assumption-audit-001` (v7 sensitivity audit); changed premises
are listed in the spec. Artifacts:
`artifacts/jm-residual/01-grid-identification/`. All numbers below are
EXPLORATORY and adopt nothing.

## A. Control

The 29-λ union run reproduced the sealed v9.4 `jm-states.csv` exactly at the
six sealed λs in all three markets (64,338 / 64,026 / 62,052 cells, zero
differences) — so everything below shares the sealed machinery, and the
parallel `n_jobs` path was exercised end-to-end a second time.

## B. Published anchors

**B1. The paper's own Table-3 persistence curve does NOT reproduce, and the
miss is systematic.** Shifts/year, US, 1982-2023, fixed λ:

| λ | ours | Shu | ratio |
|---|---|---|---|
| 0 | 7.24 | 9.7 | 0.75 |
| 5 | 2.00 | 2.7 | 0.74 |
| 15 | 1.33 | 1.7 | 0.78 |
| 35 | 0.57 | 0.8 | 0.71 |
| 70 | 0.48 | 0.5 | 0.96 |
| 150 | 0.29 | 0.4 | 0.73 |

Our fixed-λ paths are *smoother than the paper's at every λ*, by a roughly
constant factor ≈ 0.75. By the repo's printed-precision convention (half the
last printed unit, 0.05) the curve reproduces at exactly **1 of 6** λs
(λ = 70); every other point misses by 0.11–2.46. The λ = 0 row is the sharp
one: with no penalty the λ scale plays no role, so a 7.24-vs-9.7 gap there
can only come from the feature/standardization geometry feeding k-means —
precisely the knob the paper leaves open (row 1 of
docs/unspecified-choices.md; the paper says only "standardized"). Contrast:
the HMM's fixed-k curve reproduces Table 3 to a mean 1.9% on the same data,
so counting, window and data are right; the JM-specific feature layer is
where the difference lives. Caveat recorded: Table 3's caption does not name
the market; US is assumed as in every neighboring figure.

Per the spec's registered interpretation, this finding OUTRANKS section C:
the standardizer/λ-scale geometry becomes the next frozen question, and the
grid table below must be read as conditional on our geometry.

**B2. The CV-selected US path is close to the authors' printed anchor.**
Ours 34 shifts / 21.9% bear vs their 30 / 19.7% (and 34/30 = 1.13 matches the
turnover ratio 0.50/0.44 = 1.14 — the whole US turnover deviation is four
extra flips). No published DE/JP anchors exist; ours: DE 80 shifts / 25.7%
bear, JP 93 shifts / 41.7% bear. The JP bear share is 17pp above what Table
4's leverage row implies (~25%), which restates the JP JM problem — the state
sequence is far too bearish — in state-sequence units.

## C. Grid identification: no grid reaches 8/8 anywhere

Eight pre-named grids (sources in the spec), monthly CV replay through the
sealed machinery, delay 1, 10bp, all eight Table-4 cells, tolerance 0.05:

| grid | US | DE | JP |
|---|---|---|---|
| table3_sealed | 4/8 | 3/8 | 3/8 |
| v1_author | 4/8 | **6/8** | 4/8 |
| li2025_citing | **7/8** | 4/8 | 3/8 |
| bocconi_wild | 5/8 | 3/8 | 3/8 |
| hackmd_wild | 5/8 | 4/8 | **5/8** |
| companion_log5 | 6/8 | 4/8 | 3/8 |
| companion_log9 | **7/8** | 4/8 | 4/8 |
| typical_range | 5/8 | 4/8 | 3/8 |

- No market reaches 8/8 under any tested grid, and the per-market best is
  achieved by three different grids (US: li2025/companion_log9; DE:
  v1_author; JP: hackmd_wild) — the HMM Section-15 contradictory-constraints
  shape again.
- **Germany's published turnover (1.70) lies ABOVE the entire tested spread**
  (0.498..1.172): no CV-selected path from any of the eight grids trades as
  much as the paper's DE JM. Under the B1 finding (our sequences are
  systematically smoother), this is the same geometry story, not a grid
  story. US and JP spreads bracket their targets (0.235..0.824 ∋ 0.44;
  0.589..1.859 ∋ 0.72).
- JP Sharpe never exceeds 0.223 (target 0.31) under any grid — consistent
  with the known standardizer lever (0.157..0.310 measured in the pre-v9
  era).

Under the spec's registered turnover-alone criterion
(`turnover_identified_in_market`): **no grid places the turnover cell inside
tolerance for the US or Germany**; only `v1_author_withdrawn` does for
Japan.

## Registered conclusion

The JM row is UNIDENTIFIED by λ-grid choice, like the HMM row in verdicts
§15 — but with a sharper attribution than the HMM case: the paper's own
Table-3 curve fails to reproduce at λ = 0, which isolates the
feature/standardization geometry (not the grid, not the λ scale alone) as
the dominant unresolved axis. Next frozen question, per the spec's
registered interpretation: the standardizer-geometry spread of the JM cells
on sealed v9.4 data (`jm-standardizer-geometry-002`, frozen before any
variant was computed). Nothing is adopted; the sealed config is unchanged.

## Adversarial verification and remediation

Three independent auditors (spec-vs-code, fitting-rule compliance,
independent recomputation) reviewed the probe and artifacts. The
recomputation auditor reproduced the Table-3 curve bit-for-bit, the US
anchors exactly, and matched the probe's `table3_sealed` US row to the
sealed run's own metrics at 3.3e-14. The process auditors returned FAIL on
four findings, all remediated before the registry completion event, none
changing a number: (1) the report's ≈ glyph used an unregistered 0.15
threshold that marked λ = 150 (deviation 0.11) as approximately reproduced —
replaced with the repo's printed-precision 0.05, under which the curve
reproduces at 1/6 λs; (2) the parity evidence lived only in stdout — now a
committed `parity-note.txt`; (3) the spec's registered turnover-alone
criterion was not surfaced — now in the report and above; (4) the registry
lacked a completion event recording which registered interpretation was
drawn — appended. Minor items (grid-key naming, metrics parameters passed
explicitly from the sealed config instead of library defaults) fixed in the
same pass.

The supporting sonnet sweep on feature conventions (30/30 agents, no
failures) additionally confirmed from the author's repo that the EWM uses
pandas defaults (`adjust=True`, matching our sealed `ewm_adjust = true`) and
that the DD formula matches ours operation-for-operation — leaving the
scaler geometry as the only structural difference between our pipeline and
the author-code recipe, which is exactly what -002 varies.
