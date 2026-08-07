# Replication atlas + disagreement anatomy (2026-08-07)

Owner instruction: stop reading the replication as tables of numbers — draw
WHERE our model switches regimes against the paper's own figures, and treat
the grid hypothesis visually instead of by another search. This note records
what was rendered, the gates, the one new frozen experiment, and the honesty
frame that travels with every figure. Commits `1652729` (metrics module),
`9f926ff` (atlas), `c9b5294` (freeze), `d154d78` (anatomy results).

## 1. What the atlas is

`scripts/render_replication_atlas.py` → `docs/atlas/replication-atlas.html`
(18 dark-theme figures, all embedded; machine-readable tables in
`artifacts/jm-residual/atlas/`). Rendering only: it fits nothing, adopts
nothing, and every agreement/effective-λ curve carries the -004 no-candidacy
sentence on the figure itself.

Inputs are sealed artifacts plus the lossless Figure-5/6 digitizations,
re-validated on load against the annotations the authors printed:

- [line 829] "Percentage of Bear Regimes Online Inferred by JMs for the S&P 500: 19.7%, Number of Regime Shifts: 30"
- [line 851] "Percentage of Bear Regimes Online Inferred by JMs for the DAX: 15.7%, Number of Regime Shifts: 116"
- [line 873] "Percentage of Bear Regimes Online Inferred by JMs for the Nikkei 225: 25.3%, Number of Regime Shifts: 48"
- [line 903] "Percentage of Bear Regimes Online Inferred by HMMs for the S&P 500: 27.8%, Number of Regime Shifts: 96"

Extraction recovers 30/0.197 (US, exact), 114 vs 116 printed (DE, sub-pixel
floor, caveat printed on every DE panel), 48/0.253 (JP, exact), 96/0.278
(US HMM, exact).

The shading convention is load-bearing and quoted, not assumed: the red
spans ARE the traded position, because the caption says the regimes are

- [line 899] "bear regimes online inferred by JMs (shifted forward by 2 days)"

so wealth applies the digitized position with no further shift, and regime
comparison un-shifts it (`(1-position).shift(-2)`), exactly the -004
convention. The comparison window is fixed by the paper's own design:

- [line 715] "testing period begins in 1990"

## 2. Gates (all passed before any figure was drawn)

1. `verify_run` on the v10 baseline
   `fixed-baselines-36ca1ace131c-ed7abd7daea3-f9f3e0a93736` — inventory and
   metric recompute clean.
2. Figure-5/6 annotation validation per panel (§1).
3. Union-cache parity: the 29-λ union states equal the sealed v10
   `jm-states.csv` at every sealed λ — 32,169 / 21,342 / 20,684 cells,
   zero NaN-mask and zero value differences.
4. v9.4 reconstruction anchor gate: the replication contract's CV path was
   REBUILT (the v9.4 run directory no longer exists on disk) from the v10
   features — byte-identical to v9.4 by reseal gate 2 — plus the Table-3
   grid columns of the union cache, and matches the sealed
   `selected-anchors.csv` exactly: us 34 shifts / 0.21868067717454753 /
   8565 days, de 80 / 0.256568239944199 / 8602, jp 93 / 0.4173256649892164 /
   8346. A mismatch stops the script; no knob may be adjusted to pass.
5. Ceiling reproduction: the per-λ daily-agreement recomputation reproduces
   the certified -004 `ceiling.csv` best-constant rows to ≤ 1e-12
   (us 35 / 95.7492%, de 26.826957952797247 / 89.5000%, jp 150 / 81.6155%).

One in-flight defect was self-caught and fixed before anything shipped: the
HMM comparison first read `hmm-states.csv` (the raw per-fit decode, 284
flips) instead of the CV-selected `hmm-delay-1/selected-signal.csv` (122
flips, matching episode-shape-13); the wrong series would have overstated
our HMM's excess flipping by 2.3×. The V2 clip-3σ curve is excluded from the
geometry overlay per the standing owner prohibition and the -002 AMENDED
event; the overlay shows V1 and V3 only.

## 3. New event-level descriptives (repeatedly inspected dev data)

Switch-level metrics (Harding–Pagan concordance; change-point F1 with a
±10-trading-day margin and greedy 1-1 matching; segmentation covering) are
new to the record — every earlier comparison was aggregate. Headlines, all
descriptive, basis = authors' un-shifted regimes:

| market | path | concordance | switch F1 | matched | note |
|---|---|---|---|---|---|
| us | v10 calibrated | 94.8% | 0.37 | 11/30 | shift counts 30/30, cells 8/8 — yet 2 of 3 switch events differ by >10d |
| us | best constant λ=35 | 95.7% | 0.46 | 10/30 | the -004 ceiling, now event-resolved |
| us | v9.4-recon | 94.1% | 0.31 | 10/30 | median lag +0.5d |
| de | v9.4-recon | 87.2% | 0.32 | 31/114 | composition gap, not timing (ours 80 switches) |
| jp | v9.4-recon | 79.8% | 0.33 | 23/48 | ours 93 switches, bear +16pp |
| us | HMM selected | 97.5% | 0.79 | 92/96 recall | occupancy matches; ours 122 flips vs their 96 |

Reading: **aggregate-cell agreement does not imply event agreement.** The
calibrated v10 US path that reproduces every published cell and the exact
shift count still shares only about a third of the individual switch dates
within ten trading days. Tables cannot show this; the ribbon and lag figures
do.

## 4. jm-disagreement-anatomy-010 — NOT SUPPORTED (frozen before results)

Spec `research/jm-disagreement-anatomy-010.toml` (sha256 `071be656…`,
FROZEN registry row precedes the probe; the probe refuses to run on hash
drift). Question: does the ours-vs-Fig5 disagreement concentrate in the
eras where our equity data rests on a different source than theirs (JP
reconstructed TR pre 2011-12-19; DE pre-Xetra fixings pre 2000), with the
US as placebo at the same cut dates? Threshold ratio 1.5, frozen.

Result: **NOT SUPPORTED — the concentration points the other way.**
r_DE = 0.574 (modern era 14.6% vs pre-Xetra 8.4%), r_JP = 0.764 (official
era 23.9% vs reconstructed 18.2%); US placebos 0.375 / 0.983. The sharpest
cell, descriptive: Japan in the official-N225TR era is **675 extra-bear
days, 0 timing days, switch-F1 0.00** — our modern-era Japanese bear
episodes share not a single day with any episode of theirs, precisely where
the data excuse is unavailable. Per the registered interpretation branch,
this weighs against the elimination-based data/vintage attribution recorded
at the -002/-003 closure and toward the unpublished selection/geometry
choices. The era≠cause confounder was disclosed at freeze; note the
confound would have produced the OPPOSITE pattern (more disagreement in the
bear-heavy reconstructed era), which strengthens the reading. Prior
evidence in the same direction, cited per the spec's own requirement:
episode-shape-13 already found the US HMM excess flickers do not
concentrate in the reconstructed pre-1988 era.

## 5. Honesty frame (mirrored on the HTML banner)

- The `jm-replication` claim is RETRACTED (registry); the v9.4 path drawn
  here is "the replication contract's CV path (reconstructed)", never a
  sealed replication result — its original run stopped `boundary_failed`.
- v10 is a CALIBRATED baseline: its grids were searched against the
  published cells, so agreement with those cells is by construction. On the
  wealth figures the v10 line is captioned as the ceiling of any
  grid-adjustment strategy — the residual gap to the authors' path is the
  part no grid can reach (DE: 0 of 4,045,443 delay-1 grids; JP: 0 of
  4,948,505; -008).
- All numbers are repeatedly inspected development data; no holdout claim;
  nothing here may seed a grid or config.

## 6. Verifier receipt

An independent verifier (agent that did not write the atlas code) must
recompute, from raw artifacts: the fig-5 validations, one full per-λ
agreement vector per market against `ceiling.csv` and
`concordance-by-lambda.csv` (≤ 1e-12), the v9.4 reconstruction anchors, the
v10 wealth identity (`selected-signal` through `apply_signal` vs sealed
trades ≤ 1e-12), the switch tables (independent implementation, exact
match), the decomposition identities, and the -010 readout arithmetic. The
receipt is appended below when signed; until then the HTML banner says the
numbers await certification.
