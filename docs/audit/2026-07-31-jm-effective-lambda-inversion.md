# jm-effective-lambda-inversion-004 — the authors' own paths close the JM row (2026-07-31)

Frozen spec: `research/jm-effective-lambda-inversion-004.toml` (registered
before any number was computed). This is the inversion the owner asked for,
done the legitimate way: instead of searching our knobs for what matches
their table, we read the authors' choices out of their own Figure 5 — the
JM bear shading is lossless matplotlib vector data, extracted in a prior
session and re-validated here against the annotations the authors printed
on each panel:

> [line 829] "Percentage of Bear Regimes Online Inferred by JMs for the S&P 500: 19.7%, Number of Regime Shifts: 30"

extraction: 30 shifts / 19.7% (exact); DAX printed 116 / 15.7%, extracted
114 / 15.6% (sub-pixel); Nikkei printed 48 / 25.3%, extracted 48 / 25.3%
(exact). The caption fixes the convention:

> [line 896] "bear regimes online inferred by JMs (shifted forward by 2 days)"

so the shading IS the traded position path, applied to returns with no
further shift, 10bp one-way — the same scoring that reproduced the HMM
Figure-6 turnover to 0.002. Artifacts:
`artifacts/jm-residual/04-effective-lambda-inversion/`.

## QA — their paths on our data reproduce their table: 23/24 cells

| market | within 0.05 | turnover (dev) | Sharpe (dev) |
|---|---|---|---|
| US | **8/8** | 0.441 (0.001) | 0.672 (0.008) |
| DE | **8/8** | 1.670 (0.030) | 0.437 (0.003) |
| JP | 7/8 | 0.725 (0.005) | 0.333 (0.023) |

This is the strongest statement the replication can make: **given the
authors' state sequences, our data, accounting, costs and conventions
reproduce Table 4's entire JM row almost exactly.** Every earlier axis
(grid, geometry) and this one now agree — the whole JM residual lives in
the state sequences, i.e. in the unpublished selection (grid) and geometry.
Notably Germany, whose published turnover no tested grid could even reach,
lands at 1.670 vs 1.70 under their own path — on our reconstructed DAX data.

## QB/QC — how much of their path our machinery can express

Daily agreement between their (un-shifted) regime sequence and our
per-lambda fixed online paths:

| market | best constant λ (V0 sealed) | agreement | per-month ceiling | months < 90% |
|---|---|---|---|---|
| US | 35 | 95.7% | 98.7% | 13/408 |
| DE | 26.8 | 89.5% | 98.0% | 24/408 |
| JP | 150 (top of union) | 81.6% | 98.8% | 12/408 |

(V1/V3 geometries do not improve the constant-λ match; full table in
`ceiling.csv`.) Reading, per the registered interpretations: the authors' US
path is *nearly expressible* by a single fixed λ ≈ 35 under our sealed
geometry; Germany and especially Japan are not expressible by any constant λ
under any tested geometry, even though month-wise cherry-picking reaches
96–99% — each month exists in our λ family, but no single λ (hence no
grid-with-CV under our geometry, whose selected path can only be a
month-wise composition of these) recovers their composition. The effective-λ
trajectory (`trajectory.csv`) is descriptive of their path through our lens;
per the spec it is **forbidden from seeding any grid or config**.

## Adversarial verification

Two independent auditors. The recomputation auditor rebuilt QA from the raw
position and return files without reading the probe: turnover matched to
full double precision (difference 0.0 in all three markets), Sharpe to
~4e-5, and the within-tolerance counts exactly — identifying the single JP
miss as maximum_drawdown (−0.399 vs −0.453, deviation 0.054, four
thousandths over the bar). The compliance auditor passed the no-candidacy
boundary, the hard-stop gates, and the shift convention, and caught one real
process defect: the frozen_at_utc fields hand-written into the -002/-003/-004
specs were estimated, not read from the clock, and are future-dated relative
to the actual writes. The freeze-before-results ordering is nonetheless
established by git (the -002 spec was committed before its probe ran) and by
filesystem mtimes (-004: spec 01:16:29Z < probe outputs 01:49:39Z <
completion 02:01:46Z), and a registry CORRECTION event now supersedes those
fields. Rule adopted: registry and spec timestamps always come from `date
-u`, never estimation.

## Registered closure

With QA at 23/24, the JM row closes the way the HMM turnover row closed,
but stronger: accounting proven at table level, expressibility quantified at
daily resolution, and the entire residual localized to the unpublished
state-sequence choices on vintage data we cannot see. The remaining
deviations under their own path (JP Sharpe +0.023; the one JP miss is
maximum_drawdown at deviation 0.054) measure
pure data differences and sit at or inside the tolerance everywhere but one
cell. Nothing is adopted; the sealed config is unchanged.
