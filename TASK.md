# Task state (2026-08-07)

## Just completed

**baseline-reseal-v11 — SEALED and independently verified (owner chose
option (a): reseal + rerun).** `research-calibrated-v11.toml` adopts
grid-selection-rule-001's winning per-market grids (us [0,0.1,20,220]; de
[0.1,1,10,21.544,26.827,40,100,500]; jp [1.931,20,25,26.827,40,51.795,220]),
byte-identical to v10 otherwise. Run
`fixed-baselines-ef90298f32e5-5c822491f87a-82c4499ff4ac`, status complete,
directional gate passed, HMM identity gate exact. Table 4/5 pass counts
UNCHANGED from v10 (us 14/14, de 13/14, jp 13/14) but DE/JP blocking-cell
deviations measurably tighten (DE turnover 1.4363→0.9090, JP leverage
0.1498→0.1206) under a grid chosen purely by state-path agreement with the
authors' Figure-5, zero strategy metric in the selection — an encouraging
independent cross-check, not a replication claim. CONFIRMED by a separate
verifying agent (receipt
`docs/audit/2026-08-08-baseline-reseal-v11-receipt.md`): config hash,
HMM-identity, full raw-feature pipeline replay for US/JP (diff ≤3.3e-14),
`adaptive-jump verify` all exact. v10's grids remain whitelisted in
config.py so the sealed v10 run stays independently reloadable.

**NEXT REQUIRED (per grid-selection-rule-001's own consequence rule, not
yet started):** rerun dd-only, static lambda50, arrival beta=log2,
scale-free penalty, feature-metric rotation, adaptive-separation-001, and
jm-disagreement-anatomy-010's DE/JP legs against v11 before restating any
"mechanism X fails in market Y" verdict. Several of these have known bugs
flagged in the 2026-08-07 idea survey that should be fixed during the
rerun, not carried forward: scale-free-penalty-001 never ran the mechanism
it named (needs a fresh spec, can't patch the old id); feature-metric-
rotation-001 has a tautological causality falsifier and a single-reference
gap-normalization bug; the simple-jm-suite return-aware/Robust-L1 variants
were fit through an unintended double-standardization scaler bug. This is
the largest remaining block of work from the 2026-08-07/08 sessions.

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

(Owner chose the reseal 2026-08-08 — see baseline-reseal-v11 above.
`lagged-capguard-001` below is US-only and unaffected either way.)

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

**In progress:** rerun dd-only, static lambda50, arrival beta=log2,
scale-free penalty, feature-metric rotation, adaptive-separation-001, and
jm-disagreement-anatomy-010's DE/JP legs against v11 — mechanical (v11
baseline + rerun harness both already exist), except where a known bug
(see baseline-reseal-v11 above) needs fixing first.

After that queue clears, the standing motivated candidate is **duration-aware
(semi-Markov) dwell cost** — sharpened by the lagged-capguard-001 autopsy into
a concrete target: penalize exactly the "re-enter mid-chop, get run over"
behavior it diagnosed.

CORRECTION (2026-08-08, before any code ran): the line above previously said
"nonzero-diagonal penalty; `dp_tv` already sums stay-penalties, so no solver
change." That is true only for a constant marginal stay-cost — verified by
hand: a discrete-Weibull duration cost phi_k(d) = -log q_k(d) is affine in d
exactly at shape beta=1 (the geometric/memoryless case), where the marginal
cost per extra day is the constant -log(q_k) — and a constant marginal cost is
exactly what `dp_tv`'s existing nonzero-diagonal `penalty_seq[t][k,k]` already
sums. That constant-hazard case recovers the *original* constant-lambda JM,
not a semi-Markov extension. Genuine duration-dependence (beta != 1, hazard
that changes with regime age) needs an augmented-state DP, `V_t(k,d)` instead
of `V_t(k)` (state (k,d): stay -> (k,d+1), switch -> (k',1)), which the
current solver does not have. Cost is still cheap (O(T*K*D_max), no
combinatorial blowup), but "no solver change" was wrong and is retracted.
Needs its own frozen question and experiment id before any code runs; formal
DP derivation comes first, per owner request 2026-08-08.

**Literature novelty check done (2026-08-08, registry NOTE
`da-jm-novelty-sweep-2026-08-08`):** duration-dependent/semi-Markov regime
switching is NOT novel in general — settled since the 1990s (Sichel 1991
Weibull-hazard NBER cycles; Durland & McCurdy 1994; Diebold-Lee-Weinbach 1994;
Bulla & Bulla 2006 for finance, the exact paper the CJM authors themselves
cite for their own duration-misspecification stress test). Confirmed by
direct PDF read (not inference) that none of the three checked Mulvey-lab
papers (Continuous JM `ssrn-4556048`, state-aware/MoE JM `ssrn-5817083`,
allocation-focused regimes `ssrn-5235747`) implement any duration/hazard
penalty — all are strictly first-order, function of `(s_{t-1},s_t)` only.
No paper found (two independent search angles) combining explicit-duration/
hazard cost with the SJM/JM-lineage's penalized-DP framework specifically —
this is the one real gap. Defensible framing: do NOT claim
duration-dependence itself is novel (cite Sichel 1991, Bulla & Bulla 2006 as
prior art); the correctly-scoped claim is narrower — embedding a duration/
hazard cost inside the SJM's penalized-DP objective, which needs the
augmented-state DP `V_t(k,d)` above, has not been done. Caveat: single search
pass per angle; a second independent pass on very recent 2025–2026 SJM
preprints is warranted before fully certifying, per this project's
separate-audit-agent convention — not yet done.

An independent 8-agent survey of every other past-proposed, not-yet-run idea
(23 ranked candidates) is available on request; several are cheaper than
duration-aware dwell cost (already-frozen specs never executed, or known bugs
in prior runs awaiting a rerun).
