# Reseal to a calibrated baseline (v10), and the extension reruns it unblocks

Date: 2026-07-31 (UTC timestamps in `research/experiment_registry.jsonl`;
every timestamp in this note came from `date -u`, none is estimated).
Owner authorization: "reseal là cái gì? làm đi. chạy lại tất cả các thí
nghiệm trên dữ liệu mới luôn, cùng với các lưới lambda mà chúng ta có được".

## 1. What a reseal is, and what this one adopts

Every result in this project comes from a *sealed run*: a frozen contract
(TOML) plus a hashed data manifest plus a git commit, replayable to a
difference of 0.0. The sealed JM baseline until today used the Table-3 grid
`[0, 5, 15, 35, 70, 150]` and did not reproduce Table 4. Reseal = adopt a new
contract, rerun the whole pipeline under it, seal the output, and label the
result honestly.

`research-calibrated-v10.toml` (sha256 `36ca1ace…`) is that contract. It is
byte-identical to `research-expanding-v9-4.toml` except for three things:

| Change | Value | Why |
|---|---|---|
| `[jm].lambda_grid` + per-market `jm_lambda_grid` | us `[0, 21.544346900318832, 70]`, de `[150, 500]`, jp `[10, 220]` | the calibration grids from `jm-per-market-grid-009` |
| `[study].claim_label` / `claim_class` | `calibrated baseline` / `CALIBRATED_BASELINE` | the grids were searched against the published targets |
| `[selection].upper_boundary_month_fraction_limit` | `1.0` (report-only) | a 2-3 value grid concentrates at its upper edge by construction |

**The honesty constraint is enforced in code, not in prose.** `config.py`
whitelists the three calibration grids verbatim and makes them loadable *only*
under `claim_label = "calibrated baseline"`; the relaxed boundary limit is
likewise unavailable to a replication contract; and `directional_gate` refuses
to phrase a calibrated run's conclusion as replication. Two-way coupling is
pinned by `tests/test_calibrated_contract.py`, including a test that the
existing replication contract is unaffected in every respect.

Why the grids are calibration artifacts: they were found by scoring 6,474,511
subsets of a 29-value sourced menu against the 14 published Table-4/Table-5
cells. Agreement with those cells is therefore *by construction*. Their value
is not evidential — it is that they give the extension studies a fixed,
documented comparator, so an extension-minus-baseline delta is internally
valid with both sides on the same frozen grids.

## 2. Gates, declared before running

Registered in the FROZEN row of `baseline-reseal-v10` before any computation:

1. every canonical data series must be byte-identical to v9.4;
2. the HMM row must reproduce the v9.4 run exactly (its protocol is untouched);
3. `fixed_jm` must equal the `-009` real-path validation: bit-identical state
   columns, and the measured pass pattern.

All three passed on the adopted artifact (`scripts/gate_v10_reseal.py`):

- **Gate 1** — 6/6 canonical series identical, including a freshly downloaded
  FRED `DTB3`.
- **Gate 2** — `features.csv`, `hmm-states.csv`, `hmm-fits.csv`,
  `hmm-candidates.csv` byte-identical per market; `hmm` and `buy_and_hold`
  metric rows equal to 1e-12 on equal windows.
- **Gate 3** — all seven sealed λ state columns bit-identical to the `-009`
  union cache; sealed metrics reproduce the `-009` pattern exactly:
  us 8/8 at delay 1 plus 3/3 at delay 5 and 3/3 at delay 10; de 7/8 (turnover,
  deviation 1.436) plus 3/3 and 3/3; jp 7/8 (leverage, deviation 0.150) plus
  3/3 and 3/3.

Independent `verify_run` on the adopted artifact: 125-file hash inventory
intact, all 27 metric rows recomputed from the sealed trades with a maximum
absolute difference of 3.797e-14.

Adopted run: `fixed-baselines-36ca1ace131c-ed7abd7daea3-f9f3e0a93736`. It is
the **first `fixed-baselines` run in this project to reach `status: complete`**
— every earlier one stopped at `boundary_failed` and therefore never wrote
`trades/`, `metrics.csv` or `claim.json`. That is also why the extension
studies had been impossible to rerun: they all read `trades/`.

## 3. Two process defects, both self-caught

**A discarded run.** The first v10 run passed every gate, but its `claim.json`
carried the hardcoded string "directional proxy replication" — replication
wording on a calibrated artifact. The run was discarded *before adoption*, the
wording was fixed in `directional_gate`, and the baseline was regenerated at
the fixed commit. Registry: `PROCESS_NOTE`.

**A concurrent writer.** While regenerating, a `pgrep` pattern of mine used
`\|` inside an ERE — where it is a literal, matching nothing — so a live
launcher looked dead and a second process was started on the same run
directory. Both ran the same commit on the same inputs, the pipeline is
deterministic, and the final artifact was verified with no writer alive and
re-passed every gate with identical values. Recorded anyway, with two rules:
never `\|` in `pgrep`/`pkill`, and long runs launch only under harness-tracked
supervision. Registry: `PROCESS_NOTE`.

## 4. What the reseal unblocked, and the first result

`simple-jm-suite-002` (frozen hash `4275edf0…`) re-asks the five `-001`
challenger questions against the adopted baseline. The `-001` answers came
from the proxy era: Yahoo price proxies including a dividend-free Nikkei,
evaluation windows starting 2007-2009, the 9-value historical grid, and a
baseline that did not reproduce Shu's ordering in any market. Same questions,
replication-grade data, calibrated comparator.

Sharpe at delay 1, and the two gaps that matter:

| Market | Model | Sharpe | vs stronger control | vs fixed JM |
|---|---|---|---|---|
| us | buy & hold / HMM / **fixed JM** | 0.483 / 0.526 / **0.683** | — / — / +0.158 | — |
| us | static λ50 | 0.775 | +0.250 | +0.092 |
| us | DD-only | 0.766 | +0.240 | +0.083 |
| us | confirmed 2d | 0.653 | +0.127 | −0.030 |
| us | return-aware / robust L1 | 0.402 / 0.326 | −0.124 / −0.200 | −0.281 / −0.358 |
| de | buy & hold / HMM / **fixed JM** | 0.298 / 0.367 / **0.398** | — / — / +0.031 | — |
| de | confirmed 2d | 0.407 | +0.040 | +0.009 |
| de | DD-only / static λ50 | 0.342 / 0.334 | −0.025 / −0.033 | −0.057 / −0.064 |
| jp | buy & hold / HMM / **fixed JM** | 0.138 / 0.177 / **0.291** | — / — / +0.114 | — |
| jp | robust L1 | 0.303 | +0.126 | +0.013 |
| jp | return-aware / confirmed 2d | 0.261 / 0.259 | +0.084 / +0.082 | −0.030 / −0.032 |
| jp | static λ50 / DD-only | 0.160 / 0.117 | −0.017 / −0.060 | −0.131 / −0.174 |

The frozen decision rule is `G_m(v) = Sharpe_v − max(Sharpe_BH, Sharpe_HMM) > 0`
in all three markets. **One variant satisfies it: two-observation confirmation.**
The recorded conclusion is "supported".

**Three caveats that must travel with that sentence.**

1. *The bar moved under the criterion.* In `-001` the JM baseline did not beat
   the controls, so clearing them was the hard part. Here the calibrated
   `fixed_jm` already clears them in all three markets (+0.158, +0.031,
   +0.114). The criterion was written against a weak baseline and is easier
   now for reasons that have nothing to do with the challenger.
2. *Against the model it modifies, confirmation is not an improvement.*
   Confirmed-2d is −0.030 in the US and −0.032 in Japan, and +0.009 in
   Germany. It passes the frozen rule by degrading the baseline less than the
   gap to the controls. Reading "confirmation wins" off the decision file
   would be wrong.
3. *One-state fits remain material outside the US.* Among selected months,
   collapsed one-state fits: us 0% for all three fitted variants; de 26.5%
   (DD-only), 12.4% (return-aware), 46.6% (robust L1); jp 28.9%, 0%, 4.4%.
   German challenger results in particular describe a frequently one-state
   rule, not a recovered two-regime structure.

The US picture is the one that echoes the old write-up, now against a much
stronger baseline: static λ50 (+0.092 over the baseline, 12 switches, turnover
0.177) and DD-only (+0.083) both beat the calibrated JM, and neither survives
in Germany or Japan.

Every number above is repeatedly-inspected through-2023 development evidence.
No holdout exists. No performance, profitability, or generalization claim
attaches to any of it.

## 5. Extension harnesses restored

The arrival / lagged / pair-balanced runners had been deleted in `4d244ae`.
They are restored from `ba1c1e7` and generalized off their v7 literals: source
run ids, experiment ids and manifest digests now come from the frozen spec and
are checked against each source `run.json`, and every per-market λ assumption
resolves through `config.jm_protocol_for`.

**The adversarial review of that restoration is incomplete.** The workflow
launched to attack it lost seven of ten agents to a spend limit and returned
nothing. I verified five points by hand before merging and record exactly
that scope:

- `expected_lambdas = {0.0, *positives}` → `set(full per-market grid)`:
  identical wherever the grid contains 0, and required for de `(150, 500)` and
  jp `(10, 220)`, which have no zero λ;
- `market="us"` → `market=inputs.market` in `run_locked_smoke`: the only call
  site loads `us`, so behaviour is unchanged;
- the two removed `SPEC_SHA256` byte pins are covered by the registry lock —
  I ran the test that proves a one-byte spec change raises `registry lock`;
- restored tests are not weaker (assertions 35→37, 7→7, 26→29, 25→25);
- `EVALUATION_STARTS` and the 20-day whipsaw horizon are unchanged and still
  validated by the spec loader.

**Not verified by me:** the penalty-sequence math was not re-derived
independently, and the restored tests were not diffed line by line against
their originals. That work is outstanding.

One code fix was needed to run any of them: `verify_source_identity` demanded
an `experiment_id` in the source `run.json`, which a `fixed-baselines` run does
not carry — that field is a study-run convention. Identity now falls back to
`(run_id, config_sha256, status)` and *refuses* when no contract hash is
supplied, so the run id is never the only anchor
(`tests/test_study_sources.py`).

## 6. State

Adopted and sealed: the v10 calibrated baseline. Complete: `simple-jm-suite-002`.
Running or pending at the time of writing: `dd-loss-scale-002` (frozen
`ad7c0aae…`), `adaptive-confidence-002` (frozen `42a7a705…`, US smoke passed
with 60/60 β=0 state cells exact and prefix invariance), then the lagged
mechanism/performance/attribution chain and the balanced chain, each of which
needs its own frozen spec pinned to the run before it.
