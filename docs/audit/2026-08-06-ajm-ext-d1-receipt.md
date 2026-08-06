# AJM-EXT-001 D1 transport run — independent verification receipt (2026-08-06)

Scope: full recomputation audit of the sealed transport run
`ajm-ext-e331b96d662c-e524087d0978`
(`artifacts/ajm-ext-001/ajm-ext-e331b96d662c-e524087d0978`), performed by a
separate agent that wrote none of the runner, arm, gate, or scorer code. Every
number below was recomputed from the sealed files or the pinned raw archives;
no stored aggregate was trusted. This receipt is the first authorized reading
of the run's gate verdict per the contract's health-gate ordering and the
`sealed_until_receipt` marker in `gate.json`.

Identity:

| Field | Value |
| --- | --- |
| Run id | `ajm-ext-e331b96d662c-e524087d0978` |
| Contract | `research/ajm-ext-001.toml` |
| Contract sha256 (recomputed from bytes) | `e331b96d662ca703a3fa5140d4ba9d92544c9eed553eb36df1525fafbc0b6a49` |
| Registry pin | FROZEN event 2026-08-05T23:20:47Z, `experiment_registry.jsonl` line 169, same hash |
| Git sha | `e524087d0978f086f2e7e4ea74066a427ecaab38` (= repo HEAD at verification, clean tree) |
| Data lock | `research/ajm-ext-001-data.lock.toml` |
| claim_class / scientific_claim_allowed | `TRANSPORT_SCREEN` / `false` (both confirmed in `run.json`) |
| Run window | started 2026-08-06T11:10:37Z, finished 2026-08-06T11:48:37Z |

All recomputation used the repo's own `.venv` interpreter; random sampling in
checks 5b and 6 used seed 20260806, fixed before any sampled result was seen.
CSVs were parsed with `float_precision="round_trip"` (see the incident note
below — the one apparent discrepancy of this audit was caused by pandas'
default imprecise float parser, not by the run).

## 1. Integrity — PASS

Recomputed sha256 for all 303 files in the run directory (everything except
`inventory.json` and `run.json`, the same convention as
`artifacts._inventory_files`) and compared against `inventory.json`:
**0 modified, 0 missing, 0 extra**. `run.json` carries the pinned contract
sha, `claim_class = TRANSPORT_SCREEN`, `scientific_claim_allowed = false`, and
a run id consistent with `ajm-ext-{contract_sha[:12]}-{git_sha[:12]}` and the
directory name. The one declared constant deviation (HMM `n_iter` 1000 →
10000, capacity not acceptance, tol untouched at 1e-6) is covered by the
registry PROCESS_NOTE of 2026-08-06 recorded before any metric of this run was
readable, and `run.json`'s `hmm_constants` match it.

## 2. Data — PASS

Each region's `frame.csv` re-derived from the hash-pinned zips via
`ajm_ext_sources.load_region_frame` (which itself refuses on any archive-hash
or contract-hash drift) and compared **byte-for-byte** against the sealed
file: identical for all three regions (8740 rows each, 1990-07-02 →
2023-12-29; sha256 `f687cec5…`, `1a1a06af…`, `f4c54ab0…` for North America /
Europe / Japan).

Confirmation-region seal: a full path-and-content scan of the run directory
for `asia`, `pacific`, `ex japan`, `ex_japan`, `confirmation` found **zero
hits**. In `data/external/fama-french/` the Asia-Pacific ex Japan zip is
present, hashes exactly to the lock value (`325bb893…`), and has **no
extracted CSV** beside it (only the three transport CSVs are extracted). The
confirmation region appears nowhere in the run.

## 3. Accounting — PASS

All 84 trade files (3 regions × [4 specs × 2 models × 3 delays + HMM × 3
delays + buy-and-hold]) checked row by row:

- `gross = position·equity + (1−position)·cash`: max abs deviation **0.0**;
- `cost = one_way_turnover × 10 bps`: max abs deviation **0.0**;
- `strategy = gross − cost`: max abs deviation **0.0**;
- turnover independently recomputed from the position path
  (`|Δposition|`, ffill, zero first allocation): max abs deviation **0.0**;
- delay law `position[t] == signal[t−(delay+1)]`: holds on every row of every
  file, NaN masks included;
- positions binary 0/1 wherever defined; buy-and-hold exactly
  position = signal = 1, turnover = cost = 0, strategy = equity.

All identities are bitwise, far inside the 1e-15 requirement.

## 4. Metrics — PASS

Every one of the 84 rows of `metrics.csv` recomputed from its sealed trade
file with `backtest.performance_metrics` at the frozen defaults (total-wealth
drawdown basis, paper turnover ×0.5, 252, ddof 1, ES 5%): `start`, `end`,
`observations` exact on all rows, and the **maximum absolute difference
across all 84 × 8 float cells is 0.0** (bitwise equality). No extra or
missing metric rows: the 84 recomputed keys map one-to-one onto the CSV.

## 5. Challenger faithfulness — PASS

Structural, all 12 spec directories:

- λ = 0 nesting: in all 6 `shu_v3_table3` directories (the grid containing
  0.0) the challenger λ=0 column equals the fixed λ=0 column on **0 differing
  days**;
- fixed and challenger state tables share identical date indexes, identical
  populated-cell masks (5676–5739 fully populated days of 8740), and binary
  values;
- `q-train.csv` strictly positive everywhere (global minimum 0.2148…), and
  complete: 44 fit dates × the full λ grid in every directory, matching the
  refit tables.

Independent partial replay through public APIs (no refitting against sealed
parameters — a fresh end-to-end rebuild): seed 20260806 selected
**Fama-French Japan / shu_v3_table3|trailing_3000_ddof0**; `build_arm` was
rerun on the first 3300 rows of the pinned region frame with the same protocol
objects (`JMProtocol((0,5,15,35,70,150), n_init 10, seed 0, max_iter 1000,
tol 1e-8, refits Jan/Jul)`, β = ln 4). Result: 298 populated decode days
(2002-01-02 → 2003-02-21), **1788/1788 fixed state cells and 1788/1788
challenger state cells identical** to the sealed files on the overlapping
dates; 3/3 refit dates with scaler means/scales and objectives at max abs
diff **0.0**; 18/18 q_train values at max abs diff **0.0**. The engine's
internal β=0-nests-fixed equality gate executed inside this replay and did
not raise.

Descriptive observation (not a defect): 334 of 3432 refit rows carry
`collapsed_to_one_state = True`, confined to the top of the penalty grids
(λ ∈ {500, 1000} in `shu_v1`, plus λ = 220 for Japan expanding and λ = 150 in
one Japan `shu_v3_table3` spec). Single-state collapse under extreme jump
penalties is the expected model behavior; the engine records it
diagnostically and the inherited protocol treats it as a handled case, not an
error.

## 6. Selection — PASS

Monthly selection recomputed from the sealed state tables via
`walkforward.select_monthly_candidate` under the contract protocol (8y
validation, min 252 valid returns, tie 1e-12 toward lower λ, boundary limit
1.0 descriptive, cost 10 bps) for five sampled combinations (three random,
one targeted, one HMM control; seed 20260806):

| Combination | Months | Choices | Surface | Trades | Metric row |
| --- | --- | --- | --- | --- | --- |
| Japan / shu_v3_table3\|causal_expanding / challenger / d5 | 166 | equal | equal | 0.0 | 0.0 |
| Japan / shu_v1\|causal_expanding / challenger / d10 | 166 | equal | equal | 0.0 | 0.0 |
| North America / shu_v1\|trailing / challenger / d10 | 168 | equal | equal | 0.0 | 0.0 |
| Japan / shu_v3_table3\|causal_expanding / challenger / d1 (targeted) | 166 | equal | equal | 0.0 | 0.0 |
| Japan / control / hmm / d1 (smoothed states rebuilt from sealed HMM path) | 169 | equal | equal | 0.0 | 0.0 |

For every sampled combination the full chain — choices, eligibility surface,
traded path (all columns), and the corresponding `metrics.csv` row — was
re-derived and matched **bitwise** (max abs diff 0.0).

## 7. Gate — PASS (recomputation agrees with `gate.json` exactly)

The transport verdict recomputed from `metrics.csv` delay-1 rows alone, with
independent arithmetic (not the gate module): per region per spec
Δ = challenger Sharpe − fixed Sharpe; estimand = min over the 4 specs; pass
requires ≥ 2 positive regions including Fama-French Europe and no region
whose challenger has both strictly deeper MDD and strictly higher turnover
than its paired fixed leg in **every** spec.

Recomputed gate summary (first authorized reading):

| Region | Estimand (min Δ Sharpe) | Binding spec | Positive | Guardrail breached |
| --- | --- | --- | --- | --- |
| Fama-French Europe | −0.16731 | shu_v3_table3\|trailing_3000_ddof0 | no | no |
| Fama-French Japan | −0.16239 | shu_v3_table3\|causal_expanding_ddof1_min63 | no | no |
| Fama-French North America | −0.17639 | shu_v3_table3\|causal_expanding_ddof1_min63 | no | no |

Positive regions: **none** (0 of the required ≥ 2). Required region
(Fama-French Europe) positive: **no**. Guardrail breach: **none**.
**Transport gate: FAILED.** Every field of `gate.json` (passed,
required_region_positive, positive_regions, and per-region region / estimand
/ binding_spec / positive / guardrail_breached) equals the recomputation,
estimands to bitwise precision. For context, the per-spec deltas are mixed —
7 of 12 region×spec pairs are positive — but the frozen min-over-specs
estimand is negative in all three regions, driven by the `shu_v3_table3`
grids.

Consequences under the frozen contract: `[confirmation_rule].opens_only_if`
is not met — the Fama-French Asia-Pacific ex Japan confirmation region
**stays sealed**; per `[stopping]`, failure ends ajm-ext-001 with no tuning
and retrying on confirmation data. The confirmation-stage bootstrap was
correctly never run and no bootstrap output exists in the artifact.

## 8. Boundary descriptions — PASS

`boundaries.csv` recomputed in full (not only the sampled selections): all
**72 rows** re-derived from the sealed choices files against the top of each
λ grid; `months` exact everywhere, `top_fraction` max abs diff **0.0**.

## Incident log — one verifier-side false alarm, resolved

The first metrics pass flagged `expected_shortfall_5pct` on Japan /
shu_v3_table3|causal_expanding / challenger / delay-1 (stored −0.0227278 vs
recomputed −0.0224656, diff 2.6e-4). Root cause was in the **verifier's own
tooling**: pandas `read_csv` at its default `float_precision` parses the
sealed value `-0.014600000000000002` (2021-03-31, an entry-day return of
−0.0136… minus 10 bps) to exactly −0.0146, one ulp off, which flips the tie
structure at the 5% quantile (7 apparent ties instead of 6, ES tail of 186
days instead of the true 180). Two independent closures: (a) re-parsing with
`float_precision="round_trip"` makes the stored value bitwise exact, as it
does every other cell; (b) check 6 re-derived this exact row in memory from
the sealed states without any CSV round-trip and matched the stored ES
bitwise. The sealed artifact was correct; no run file was modified at any
point.

## Not verified, stated for scope

- The 44-fit-date full-period JM fits were not independently refit beyond the
  3-fit-date prefix replay of check 5; full-period fixed-leg fidelity rests on
  the replayed mechanism, deterministic solver constants (`random_state` 0,
  `n_init` 10), and the sealed inventory.
- The HMM EM fits themselves were not re-run (10 seeds × 44 windows); the HMM
  control was verified from its sealed state path forward (smoothing →
  selection → trading → metrics, all bitwise).
- Process gates that live outside the artifact were checked as records, not
  re-executed: the registry FROZEN pin, the n_iter PROCESS_NOTE, and the
  independent objective-gate fault-injection audit of 2026-08-05
  (`docs/audit/2026-08-05-objective-gate-fault-injection.md`, verdict
  CERTIFY, written by an agent that did not write the gate) all predate this
  run's metrics.

None of these reservations touches a frozen requirement of the verification
order; all eight ordered checks passed.

## Verdict

Every ordered check (integrity, data, accounting, metrics, challenger
faithfulness including an independent partial replay, selection, gate,
boundaries) was recomputed and passed, almost everywhere at bitwise
precision. The artifact is internally consistent, reproducible from the
pinned raw data, and its stored gate verdict is exactly what the sealed
metric rows imply: the transport gate failed — zero positive regions, the
required Europe region negative, guardrail clean — so the confirmation region
remains sealed and ajm-ext-001 ends per its stopping rule.

VERDICT: CERTIFIED



---

## Correction addendum — 2026-08-06, post-certification re-audit

Two prose defects in this receipt, found by a later 9-agent re-audit with
adversarial verification. Neither touches any gated quantity: the inventory,
accounting, metric recomputation, and gate table above were independently
re-reproduced bitwise by a second verifier with its own seed.

1. Line "7 of 12 region x spec pairs are positive" is wrong: the correct count
   is **8 of 12** (miscounted cell: Japan shu_v3_table3|trailing, +0.0209).
   The error is conservative (understates the challenger) and was copied into
   the registry completion event, which now carries a CORRECTION.
2. This receipt did not state the OOS-window consequence: all transport P&L
   spans 2010-02/2010-04..2023-12 and excludes the 2008 crisis that sat inside
   the US/DE development windows. Quantified afterward from archived dev
   paths, the era shift explains ~1% of the dev-to-FF delta gap — but the
   omission itself was a disclosure defect, recorded in the registry.
