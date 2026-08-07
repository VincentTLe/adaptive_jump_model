# Task state (2026-08-07)

## Just completed

**lagged-capguard-001 — NOT SUPPORTED, certified.** The cap-guarded lagged
challenger (owner-directed, US only, both baseline grids, frozen spec
`1e6b03a7…`) made the worst-grid delta WORSE: min-over-grids ΔSharpe
capguard −0.0709 vs plain lagged −0.0638, and fails the −0.05 rent. R1
refuted the guard's premise on US (lagged's excess switches do not sit in
cap-binding months, share −0.125). Measured composition mechanism: zero
manufactured switch days — the damage is positioning mix (g2 capguard
0.6173 below both parents 0.6833/0.6768). Independent verifier CERTIFIED
7/7 at identity precision (`scripts/verify_lagged_capguard.py`, receipt in
the registry chain). Per the registered interpretation the cap-guard idea
is CLOSED for the lagged mechanism.

Context row, descriptive: plain lagged_log4 is itself non-positive vs fixed
on the resealed US baseline under both grids (−0.064 / −0.007) — the v7-era
US dev support does not reappear.

**Replication track — CLOSED with per-market labels** (2026-08-07 atlas,
verifier-certified 9/9; `docs/atlas/replication-atlas.html`): US ≈
replicated (30/30 shifts, Sharpe 0.683 vs 0.68, 95.7% daily concordance at
λ=35); DE/JP bounded-with-causes — their Fig-5 sequences are not generable
from public information (0 of 6.47M grids; geometry family exhausted), and
jm-disagreement-anatomy-010 (NOT SUPPORTED) moved the residual attribution
from data/vintage to the authors' unpublished selection/geometry. Reopen
only on a new primary source (Yu dissertation, DataSpace, re-check late
2026; author e-mail sent by owner, no reply).

## Next

No active experiment. The standing queue holds ONE motivated candidate:
**semi-Markov dwell cost** (nonzero-diagonal penalty; the 2026-07-11 audit
showed `dp_tv` already sums stay-penalties, so no solver change). It needs
its own frozen question and experiment id before any code runs. The owner
decides whether to spend it — note the record: every adaptive variant so
far has closed non-positive under its frozen rule, and US-only dev evidence
cannot confirm anything regardless of direction (no confirmation region
exists; AP-ex-Japan is burned).
