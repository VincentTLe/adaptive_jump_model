# jm-inferred-grid-005 / -006 — the owner-instructed grid inference (2026-07-31)

The owner instructed: if the λ grid is unpublished, estimate it or reverse-
engineer it. Done the legitimate way — the estimand is the AUTHORS' hidden
candidate set, the estimator is the AUTHORS' own Figure-5 monthly choices
(-004's validated extraction), the derivation rules were frozen before the
histograms were looked at, and the replays are registered as conditional
diagnostics with adoption owner-gated. This is estimation from their
artifact, not a search of our knobs against their table; the distinction and
its limits are recorded in both specs.

## The two estimators (family declared exhausted)

**-005, all months:** weight each month by agreement×days, take the 90%
support. Result: supports flooded with tiny λs (0, 0.1, 1, …). Diagnosis,
recorded in the completion event: the registered tie rule inherited from
-004 hands *quiet* months (no shift in their path — 385 of 408 US months) to
the smallest of many perfectly-agreeing λs. Quiet months carry no
information about λ. The conditional replay was still informative: US union
7/8 with Sharpe 0.680 against the published 0.68.

**-006, shift months only, ties to the largest λ** (sequentially declared;
the opposite tie direction brackets the artifact): the US support becomes

> 10 | 13.9 | 15 | 26.8 | 30 | 35 | 40 | 60 | 80 | 100 | 1000

— concentrated in the 10–100 band with mass near 35, exactly where -004's
independent constant-λ finding (95.7% daily agreement at λ = 35) sits. Only
23 of 408 US months are informative (their path has 30 shifts): a small
sample by construction, the inherent limit of this inference.

## Conditional replays (our monthly CV over the inferred grids, V0 geometry)

| run | US | DE | JP |
|---|---|---|---|
| sealed baseline | 4/8 | 3/8 | 3/8 |
| -005 union | **7/8** (Sharpe 0.680 ≈ 0.68) | 3/8 | 3/8 |
| -006 union | 6/8 (Sharpe 0.724, dev 0.044) | 4/8 | 3/8 |
| -006 per-market | 5/8 (turnover 0.412 vs 0.44 ✓) | 4/8 | 3/8 |

The US pattern is the honest punchline: inferred grids land *either* the
Sharpe *or* the turnover cell, never both — the CV walks differently over
any candidate set, which is the same selection-instability mechanism
measured for the HMM (split-half agreement at chance). Germany and Japan do
not improve materially under any inferred grid, as the -004 expressibility
bound predicted: their sequences are outside what our geometry/data can
generate regardless of the candidate set.

## The answer to "what was their grid"

The best statement public information supports: **through our λ family, the
authors' US selections live in roughly [10, 100] with mass near 35; the
DE/JP grids cannot be resolved because their state sequences are not
expressible under any tested geometry on our data.** The estimate is
descriptive. Nothing is adopted; both specs bar it, and any future adoption
is an owner decision that must carry this two-estimator derivation chain in
full.

## Adversarial verification

An independent auditor recomputed both derivations from the raw artifacts
without reading the probes first: every support set matched byte-for-byte
(including the 23/385 informative/quiet US month split, which requires the
within-month diff reading — the naive whole-sequence diff gives 24/384 and
would have falsely failed the check), the us/union `within_tol = 7` was
re-derived to the digit, and per-λ weights reproduced `support.csv` exactly.
Compliance: FROZEN registry events precede all artifact mtimes; the -006
sequential/exhaustion declarations are present; both reports carry the
conditional/adoption-forbidden labels; live configs are unchanged; spec
hashes match the registry pins (specs unedited post-freeze); and the -006
registered success condition ("7/8 with turnover nearer 0.44") is correctly
reported as NOT met. One minor flag, accepted: -005's internal
`frozen_at_utc` reads 51 seconds before its file mtime — a clock read
followed by continued editing, causally coherent, unlike the -002/-004
future-dating already corrected in the registry.
