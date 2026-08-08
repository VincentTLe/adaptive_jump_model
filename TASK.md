# Task state (2026-08-07)

## Just completed

**Mulvey-lab literature sweep — closed, no rescue for DE/JP.** Read 7
companion papers from the same Princeton lab (2406.09578, 2410.14841,
plus 5 more acquired 2026-08-08: the CJM paper, Deep-SJM, Allocation-
Focused Regimes, Deep-Generative-Models, and Luo & Mulvey 2026 — the one
the owner's advisor specifically flagged for its page-13 grid disclosure).
None discloses the target paper's (2402.05272) actual grid; every paper
uses a different feature set, standardization silence, and validation
protocol (no lab house-style). The one real, disclosed, real-market grid
found (Luo & Mulvey's λ∈[1,100]-step-10) was tested directly against
Table 4 (same method as jm-grid-identification-001's other 8 named grids,
n_init raised 10→60 to clear a local optimum at λ=61→71 never fit before)
— result us 5/8, de 3/8, jp 3/8 within tolerance, worse than the existing
frontier grids on both markets. Confirms rather than contradicts: the
DE/JP block is state-sequence shape, not λ menu coverage. Registry:
jm-grid-identification-001 PROCESS_NOTE (FROZEN + EXPERIMENT_COMPLETE).

**grid-selection-rule-001 — complete, verified, v11 reseal PROPOSED (not
performed).** A frozen 2026-08-01 spec (owner-approved) that orders each
market's already-admissible grids (the 13/14 or 14/14 -009 sets) by daily
agreement with the AUTHORS' own Figure-5 state sequence — no strategy
metric involved — instead of "first row of an examples file" (how v10 was
actually chosen). Result, full and untruncated: US winner {0,0.1,20,220}
0.9609 vs adopted {0,21.5,70} 0.9476 (differs, upper part of spread); **DE
winner {0.1,1,10,21.5,26.8,40,100,500} 0.8951 vs adopted {150,500} 0.8585
— the adopted German grid ranks DEAD LAST, 366th of 366, on a complete
enumeration**; JP winner {1.93,20,25,26.8,40,51.8,220} 0.8516 vs adopted
{10,220} 0.8152 (differs, below median, not worst). Independently CONFIRMED
by full from-scratch recomputation of all 366 DE grids (receipt:
`docs/audit/2026-08-07-grid-selection-rule-001-receipt.md`). Required
rebuilding lost -008/-009 binary caches first (substituted v10's run
directory for the deleted v9.4-hash one; all load-bearing rebuilt outputs
verified byte-identical to the sealed originals before use).

**Decision pending from the owner:** per the spec's own consequence rule, a
v11 reseal is proposed, not automatic. If accepted, every mechanism result
scored against the v10 DE/JP baseline since 2026-08-01 — dd-only, static
lambda50, arrival beta=log2, scale-free penalty, feature-metric rotation,
`adaptive-separation-001`, and `jm-disagreement-anatomy-010`'s DE/JP legs —
must be rerun before "mechanism X fails in market Y" is restated.
`lagged-capguard-001` is US-only and unaffected either way.

**lagged-capguard-001 — NOT SUPPORTED, certified.** The cap-guarded lagged
challenger (owner-directed, US only, both baseline grids, frozen spec
`1e6b03a7…`) made the worst-grid delta WORSE: min-over-grids ΔSharpe
capguard −0.0709 vs plain lagged −0.0638, and fails the −0.05 rent. Visual
autopsy (owner-directed, redrawn in the paper's own Figure-5 grammar after a
style correction) localized the failure: lagged re-enters mid-way through
the August 2022 bear-market rally and rides the October leg down while
fixed stays in cash — the concrete motivation for a semi-Markov dwell-cost
candidate, not just an unmotivated variant. Independent verifier CERTIFIED
7/7. Per the registered interpretation the cap-guard idea is CLOSED for the
lagged mechanism.

**Replication track — CLOSED with per-market labels** (2026-08-07 atlas,
verifier-certified 9/9; `docs/atlas/replication-atlas.html`): US ≈
replicated (30/30 shifts, Sharpe 0.683 vs 0.68, 95.7% daily concordance at
λ=35); DE/JP bounded-with-causes — their Fig-5 sequences are not generable
from public information (0 of 6.47M grids; geometry family exhausted), and
jm-disagreement-anatomy-010 (NOT SUPPORTED) moved the residual attribution
from data/vintage to the authors' unpublished selection/geometry. Reopen
only on a new primary source (Yu dissertation, DataSpace, re-check late
2026; author e-mail sent by owner, no reply).

**Documentation audit — Table-3 grid mischaracterization, fixed.** Owner
correction: the Table-3 illustrative grid {0,5,15,35,70,150} is not Shu's
disclosed production/CV grid. An 8-agent audit found this caveat already
present in SCIENTIFIC_LEDGER.md/hyperparameter-grid-attribution.toml since
2026-07-17/18 but missing from this session's newer deliverables; added it
in 10 locations (labeling-only, no certified number changed). One edit
(research/ajm-ext-001.toml) was attempted and reverted — that contract is
hash-enforced against its certified run and even a comment breaks
`tests/test_ajm_ext_runner.py`; the clarification went into the receipt
doc instead.

## Next

**Owner decision needed:** accept the v11 reseal (adopt each market's
grid-selection-rule-001 winner) or keep v10 and treat the ranking as
descriptive only. Either way, the next actual experiment work is rerunning
the mechanism list above against whichever baseline is chosen — this is
mechanical (baselines + rerun harness both already exist), not new design.

After that queue clears, the standing motivated candidate is **semi-Markov
dwell cost** (nonzero-diagonal penalty; `dp_tv` already sums stay-penalties,
so no solver change) — sharpened by the lagged-capguard-001 autopsy into a
concrete target: penalize exactly the "re-enter mid-chop, get run over"
behavior it diagnosed. Needs its own frozen question and experiment id
before any code runs. An independent 8-agent survey of every other
past-proposed, not-yet-run idea (23 ranked candidates) is available on
request; several are cheaper than semi-Markov (already-frozen specs never
executed, or known bugs in prior runs awaiting a rerun).
