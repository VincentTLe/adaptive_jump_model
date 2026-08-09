# jm-per-market-grid-009 — per-market grids and the blocking cells (2026-07-31)

Frozen spec: `research/jm-per-market-grid-009.toml`. Owner instructions
incorporated: per-market grids meeting Shu at delays 1, 5 AND 10; every
nonexistence statement scoped to the 29-λ sourced menu (sizes 2–8) and
replaced by a measured blocking-cell frontier; small models for verification
agents. Artifacts: `artifacts/jm-residual/09-per-market-grids/`.

## Results

- **US: 36,657 grids satisfy all fourteen published cells** (eight Table-4
  cells at delay 1 plus the three Table-5 cells at each of delays 5 and 10).
  Three winners re-validated through the real pipeline: 9/9 PASS; example
  {0, 21.5, 70} has worst delay-1 deviation 0.011.
- **DE/JP: best available within the menu is 13/14** — DE 366 grids, JP
  2,948 grids (7/8 at delay 1 + full Table-5 at both long delays); one
  example each validated 6/6 PASS (DE {150, 500}, JP {10, 220}), and the
  delay-1 miss is exactly the frontier's blocking cell in both.
- **The blocking cells, measured over the whole lattice**: DE turnover is a
  joint constraint — the lattice's turnover range reaches 4.63, comfortably
  covering the 1.70 target, but among the 139,911 vectors that pass the
  other seven cells the maximum is 1.465; JP leverage has a lattice-wide
  ceiling of 0.737 against the 0.75 target, and 0.636 given the other seven
  cells. Within the tested menu, the emptiness is state-sequence shape, not
  λ coverage. A denser/real-valued λ extension for DE/JP remains on the
  table per the owner's suggestion, with this frontier as the prior.

## Adversarial verification (sonnet, per the owner's model instruction)

PASS on all three sections, zero discrepancies: the full 18-row frontier
table was independently re-derived from the raw result arrays with exact
agreement (139,911 / 1.4648 / 4.6287 / 0.7373 / 0.6360 / 139,917 / 2,668 all
to full precision); the US {0, 20, 60} grid was re-run end-to-end through
the real pipeline at all three delays and cleared all 14 cells; ordering,
menu scoping, no-adoption language, and required outputs all held — and the
auditor confirmed -009 did not repeat -008's prospective-completion defect.
Stated caveat, carried honestly: the Q1 intersection counts (36,657 / 0 / 0)
were verified by cross-reference against the artifact, not independently
re-derived by the auditor.

Grids remain calibration artifacts; reseal is the owner's decision.
