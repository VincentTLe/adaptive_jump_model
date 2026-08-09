# Resurrected adaptive-transition-cost harnesses (-002)

Three retired experiment families are restored from `ba1c1e7` (deleted by
`4d244ae`) and generalized so they rerun against the **calibrated-v10** baseline
instead of the v7 proxy baseline. This is a rerun harness, not a redesign: the
DP math, the β scenarios `{0, log 2, log 4}`, the timing conventions, the
whipsaw / persistence / confirmed-early definitions, the decision gates and every
output schema are byte-for-byte the same rules as -001.

Nothing under `artifacts/`, `research/*.toml` or
`research/experiment_registry.jsonl` was touched. No experiment was run.

## Worktree base — read this first

The worktree was created from **`origin/main` (311753c)**, which is the PR-#5
merge and does *not* contain the calibrated-v10 contract. Every premise of the
task (`config.jm_protocol_for`, `CALIBRATED_JM_GRIDS`, `claim_label = "calibrated
baseline"`, `research-calibrated-v10.toml`) lives on **`cleanup/research-protocol`
(f9f3e0a)**, the branch the task names. The agent branch was therefore reset to
`cleanup/research-protocol` before any work. `311753c` is untouched and still
reachable from `origin/main`; no user branch was modified.

## 1. Restored files

### New shared modules

| File | Purpose |
| --- | --- |
| `src/adaptive_jump/study_grids.py` | Resolve one λ grid per market from `config.jm_protocol_for(market)`; derive the positive (event) subset; parse and compare a spec's per-market grid table. |
| `src/adaptive_jump/study_sources.py` | Parse a spec's source table (`experiment_id` / `artifact_subdir` / `run_id` / `run_inventory_sha256`) and verify it against the source run's own `run.json`. |

### Arrival / confidence family

`confidence_spec.py`, `confidence_model.py`, `confidence_evaluation.py`,
`confidence_runner.py`

### Lagged family

`lagged_study.py`, `lagged_model.py`, `lagged_mechanics.py`, `lagged_sources.py`,
`lagged_analysis.py`, `lagged_runner.py`, `lagged_verifier.py` (mechanism);
`lagged_performance.py` (P&L); `lagged_attribution.py` (2×2 attribution)

### Balanced family

`balanced_model.py`, `balanced_mechanics.py`, `balanced_sources.py`,
`balanced_events.py`, `balanced_replay.py`, `balanced_analysis.py`,
`balanced_smoke.py`, `balanced_smoke_replay.py`, `balanced_decision_replay.py`,
`balanced_event_replay.py`, `balanced_runner.py`, `balanced_verifier.py`
(mechanism); `balanced_performance.py` (P&L)

### Shared core (restored **trimmed**)

`separation_analysis.py`, `separation_study.py` — only the pieces the three
families import: `MarketInputs`, `load_market_inputs`, `_read_candidates`,
`_decode_parameters`, `_refit_for_date`, `terminal_decision`,
`arrival_ablation_state`, `SeparationStudyError`. adaptive-separation-001's own
spec loader and logistic prediction machinery are **not** restored (see §4).
`separation_evaluation.py` and `separation_runner.py` are not restored at all.

### Tests

`tests/conftest.py` (new) plus the twelve restored test modules:
`test_confidence_study.py`, `test_lagged_study.py`, `test_lagged_model.py`,
`test_lagged_mechanics.py`, `test_lagged_runner.py`, `test_lagged_verifier.py`,
`test_lagged_performance.py`, `test_lagged_attribution.py`,
`test_balanced_model.py`, `test_balanced_analysis.py`, `test_balanced_runner.py`,
`test_balanced_verifier.py`.

### Entry points

The standalone `main()`s were restored as-is — no `cli.py` change was needed:

```
python -m adaptive_jump.confidence_runner    --config <contract> [--spec ...] [--smoke]
python -m adaptive_jump.lagged_runner        --config <contract> [--spec ...] [--smoke] [--verify RUN]
python -m adaptive_jump.lagged_performance   --config <contract> [--spec ...]
python -m adaptive_jump.lagged_attribution   --config <contract> [--spec ...]
python -m adaptive_jump.balanced_runner      --config <contract> [--spec ...] [--smoke] [--verify RUN]
python -m adaptive_jump.balanced_performance --config <contract> [--spec ...]
```

Each `--spec` now defaults to `research/<EXPERIMENT_ID>.toml`, i.e. the -002 file.

## 2. Every generalization made

### (a) Study identity: literal → module constant

Each loader pinned its own id inline. It is now one `EXPERIMENT_ID` constant per
family, bumped to `-002`, exactly as `simple_jm_suite.py` was bumped in `1d0c638`.

| Constant | file:line |
| --- | --- |
| `adaptive-confidence-002` | `src/adaptive_jump/confidence_spec.py:31` |
| `lagged-evidence-mechanism-002` | `src/adaptive_jump/lagged_study.py:36` |
| `lagged-evidence-performance-002` | `src/adaptive_jump/lagged_performance.py:66` |
| `lagged-selection-attribution-002` | `src/adaptive_jump/lagged_attribution.py:63` |
| `balanced-lagged-mechanism-002` | `src/adaptive_jump/balanced_model.py:41` |
| `balanced-lagged-performance-002` | `src/adaptive_jump/balanced_performance.py:61` |

Consumers of those constants: `confidence_runner.py:259,377`,
`lagged_runner.py:339`, `lagged_verifier.py:37`, `lagged_performance.py:439,884,1101`,
`lagged_attribution.py:36,165,211,582,856`, `balanced_verifier.py:41`,
`balanced_runner.py:349`, `balanced_performance.py:294,1504,1671,1889`.

### (b) Sources: literals → spec fields validated against `run.json`

Deleted literals — the v7 experiment id, the v7 data-manifest digest, the pinned
upstream spec hashes, and the hardcoded artifact directory names:

| Was | Now | file:line |
| --- | --- | --- |
| `fixed.get("experiment_id") != "fixed-baselines-001-v7"` | `read_source_reference(fixed, ...)` | `lagged_study.py:202`, `balanced_model.py:208`, `lagged_performance.py:156`, `balanced_performance.py:217` |
| `data_manifest_sha256 != "3636939b…"` | spec field, non-empty string; digest checked against the run's `data-manifest.json` | `lagged_study.py:223`, `lagged_sources.py:93` |
| `arrival.get("spec_sha256") != "1b0c327b…"` | spec field, non-empty string | `lagged_study.py:225` |
| `parent.get("spec_sha256") != "6f964f57…"` | spec field, non-empty string | `balanced_model.py:247` |
| `root / artifact_root / "adaptive-confidence-001" / run_id` | `spec.arrival.directory(root, artifact_root)` | `lagged_sources.py:78` |
| `root / artifact_root / "lagged-evidence-mechanism-001" / run_id` | `spec.parent.directory(...)` | `balanced_sources.py:68` |
| the same three literal directories | `spec.<source>.directory(...)` | `lagged_performance.py:262`, `balanced_performance.py:477` |
| `_assert_run_metadata(..., experiment_id="lagged-evidence-mechanism-001")` etc. | `experiment_id=spec.<source>.experiment_id` | `balanced_performance.py:527,542,565` |
| `metadata.get("experiment_id") != "lagged-evidence-performance-001"` | `!= PARENT_EXPERIMENT_ID` | `lagged_attribution.py:229` |

New identity gate, applied before any evidence is read: the source run's
`run.json` must declare the spec's `experiment_id`, the spec's `run_id`, status
`complete`, and (where applicable) the same `config_sha256` —
`study_sources.verify_source_identity`, called from `confidence_runner.py:58`,
`lagged_sources.py:79,86`, `lagged_performance.py:267`,
`balanced_sources.py:71,78`, `balanced_performance.py:487`.

### (c) Per-market λ grids

`spec.lambdas` / `spec.event_lambdas` change type from `tuple[float, ...]` to
`dict[str, tuple[float, ...]]`, with `spec.lambdas_for(market)` /
`spec.event_lambdas_for(market)` accessors:
`confidence_spec.py:62`, `lagged_study.py:120,124`, `lagged_performance.py:123`,
`lagged_attribution.py:104`, `balanced_model.py:122,126`,
`balanced_performance.py:123`.

Grid validation is now a per-market table equal to the contract's own grids
(`grids_equal(lambdas, market_grids(config, markets))`) and the event grid is the
market's positive lambdas (`positive_grids`):
`confidence_spec.py:121`, `lagged_study.py:233-234`, `lagged_performance.py:190`,
`lagged_attribution.py:147`, `balanced_model.py:245-246`,
`balanced_performance.py:256`.

Deleted module-level tuples: `LAMBDAS` / `POSITIVE_LAMBDAS` from
`lagged_study.py`, `lagged_performance.py`, `balanced_model.py`,
`balanced_performance.py`, `separation_study.py`.

Resolution sites (candidate columns, refit λ₀ sets, event loops, cell counts):
`confidence_model.py:139-146,152-166,225,263`, `confidence_runner.py:94,194`,
`lagged_model.py:234,241`, `lagged_analysis.py:53,72-74,88,110,159,178`,
`lagged_mechanics.py:83,117,191,286,290`, `lagged_runner.py:53,140`,
`lagged_sources.py:134,138`, `lagged_verifier.py:86,382,424`,
`lagged_performance.py:311,336,591,596`, `lagged_attribution.py:271`,
`balanced_model.py:357,366,439,446,472`, `balanced_sources.py:130-141`,
`balanced_analysis.py:73,118,164,296`, `balanced_events.py:83,108`,
`balanced_replay.py:115,130,166,207`, `balanced_decision_replay.py:56`,
`balanced_event_replay.py:176,201,382`, `balanced_smoke.py:105,113,115,127,171,227,375,397,406`,
`balanced_smoke_replay.py:66,100,107,109,121,159,208,325,347,356`,
`balanced_runner.py:79`, `balanced_verifier.py:103-104,125,247,257,565`,
`balanced_performance.py:597,742,752,761,801,1283,1443`.

Functions that gained a `market` parameter because they resolve a grid:
`separation_analysis._read_candidates`, `confidence_model._parent_states`
(argument already per-market, now supplied per market),
`lagged_analysis._input_spec`, `lagged_verifier._read_state_table`,
`lagged_performance._read_states`, `balanced_model._read_candidates`,
`balanced_analysis.matched_response`, `balanced_event_replay.matched_response`,
`balanced_smoke.balanced_penalty_checks`, `balanced_smoke_replay._penalty_checks`,
`balanced_verifier._read_state`, `balanced_verifier._smoke_coverage_exact`,
`balanced_performance._select_paths`.

### (d) Contract file: `"research.toml"` → the contract the run loaded

The implementation locks hashed `root / "research.toml"` and the verifiers
reloaded it as "the canonical config". Under a calibrated rerun the contract is
`research-calibrated-v10.toml`, so both are now driven by the loaded config:

* `implementation_lock(root, spec, config_path)` — `lagged_sources.py:164`,
  `balanced_sources.py:173`, `balanced_performance.py:710`; and
  `_implementation_sha(root, spec, config_path)` — `lagged_performance.py:714`,
  `lagged_attribution.py:601`.
* `run.json` gains `"config_path"` (basename only) — `lagged_runner.py:256`,
  `balanced_runner.py:267`, `confidence_runner.py:313`,
  `lagged_performance.py:1032`, `lagged_attribution.py:794`,
  `balanced_performance.py:1812`.
* Verifiers resolve the contract from that field — `lagged_verifier.py:340`,
  `balanced_verifier.py:478`; `CANONICAL_CONFIG` is gone.
* `run.json` also records the source experiment ids, so a sealed run states its
  own lineage.

`_lock_key(path, root)` (`lagged_sources.py:160`, `balanced_sources.py:169`,
`balanced_performance.py:706`, `lagged_performance.py:710`,
`lagged_attribution.py:597`) labels a locked file by its repository path, falling back
to its basename when the file is outside the repo (spec files in `tmp_path`).

### (e) `models._fit_fixed_jm` → `models.fit_fixed_jm_window`

`confidence_model.py:181`. The function was renamed upstream and made
loss-scale-aware; with `observation_loss_scale = 1.0` and the default
`sklearn_standard_scaler_ddof0` standardizer the code path is numerically
identical to the -001 one (same `StandardScaler`, same `tol`, same `sort_by`).

### (f) Per-market fit protocol

`config.jm_protocol` → `config.jm_protocol_for(market)` wherever a JM fit or a
TV-model is constructed: `confidence_model.py:143,177,180,195-198`,
`lagged_model.py:241`. Refit months, `n_init`, `random_state`, `max_iter` and
`tol` are contract-wide and unchanged; only `lambda_grid` differs by market.

## 3. What a -002 spec must contain

Common to all six: `schema_version = 1`, `experiment_id = "<family>-002"`,
`claim_class = "EXPLORATORY"`, the family's evidence-lane flags unchanged from
-001, and `[storage] artifact_subdir = "<family>-002"`. Every **source table**
now requires four fields:

```toml
experiment_id        = "<the id the source run's run.json carries>"
artifact_subdir      = "<directory under artifact_root holding that run>"
run_id               = "<the run directory name>"
run_inventory_sha256 = "<sha256 of that run's inventory.json>"
```

Every **grid** is a per-market table whose values must equal
`config.jm_protocol_for(market).lambda_grid` (and, for event grids, its positive
subset):

```toml
[candidates]
raw_lambda_grid   = { us = [0.0, 21.544346900318832, 70.0], de = [150.0, 500.0], jp = [10.0, 220.0] }
event_lambda_grid = { us = [21.544346900318832, 70.0],      de = [150.0, 500.0], jp = [10.0, 220.0] }
```

Per family, on top of the -001 fields:

**`adaptive-confidence-002.toml`** — `[parent]` = the four source fields plus
`config_sha256` (must equal the loaded contract), `data_manifest_sha256`,
`data_cutoff = "2023-12-31"`; `[candidates] raw_lambda_grid` as a per-market
table; `[comparison] markets`; `[penalty] beta`, `q_train`,
`missing_center_loss`, `q_train_fallback` unchanged; `[controls]` unchanged.

**`lagged-evidence-mechanism-002.toml`** — `[fixed_source]` and
`[arrival_source]` = the four source fields (fixed also carries `config_sha256`
and `data_manifest_sha256`, arrival also carries `spec_sha256`);
`[candidates] raw_lambda_grid` + `event_lambda_grid` as per-market tables;
`[candidates] fitted_parameter_source = "sealed <arrival experiment_id>
refits-and-scales.csv; no model refit"`; `[scope] evaluation_start` still
`{us = 2007-12-04, de = 2008-01-03, jp = 2009-05-07}`; `[verification]`,
`[execution]`, `[decision]` unchanged.

**`lagged-evidence-performance-002.toml`** — `[fixed_source]`,
`[arrival_source]`, `[lagged_source]` = the four source fields (+ `config_sha256`
/ `data_manifest_sha256` on fixed, `spec_sha256` on arrival and lagged);
`[model] raw_lambda_grid` per-market; `[model] refit_boundary_convention` with
`<lagged experiment_id>` substituted for the -001 name; `[protocol]`,
`[decision]`, `[verification]`, `[storage]` unchanged.

**`lagged-selection-attribution-002.toml`** — `[parent]` = run id, inventory,
`choices_sha256`, `spec_sha256`, `implementation_sha256`, `config_sha256`,
`data_cutoff`; `[state_sources] raw_lambda_grid` per-market plus the two
inherited inventory hashes; `[storage] artifact_subdir =
"lagged-selection-attribution-002"`; the parent contract read from disk is
`research/lagged-evidence-performance-002.toml`.

**`balanced-lagged-mechanism-002.toml`** — `[fixed_source]` and
`[parent_lagged_source]` = the four source fields (+ `config_sha256` /
`data_manifest_sha256` on fixed, `spec_sha256` on parent);
`[candidates] raw_lambda_grid` + `event_lambda_grid` per-market;
`[storage] artifact_subdir = "balanced-lagged-mechanism-002"`; the toy contract,
matched windows and decision fields unchanged. There is **no** implementation
byte-pin any more — the FROZEN registry row carries the spec hash.

**`balanced-lagged-performance-002.toml`** — `[fixed_source]`,
`[lagged_source]`, `[balanced_source]`, `[lagged_performance_oracle]` = the four
source fields plus their existing hash fields (`balanced` keeps
`run_json_sha256` and the per-market `candidate_sha256` table); `[model]
raw_lambda_grid` per-market; `[model] fitted_parameters = "sealed <fixed
experiment_id> scalers and centers; no refit or state regeneration in the P&L
runner"`; `[protocol] outer_start` unchanged; `[storage] artifact_subdir =
"balanced-lagged-performance-002"`. No implementation byte-pin.

Each -002 spec also needs a FROZEN row in `research/experiment_registry.jsonl`
carrying its `frozen_spec_hash`; the registry lock is unchanged and is now the
*only* place a spec's bytes are pinned.

## 4. Behavior I was forced to change, and why

1. **The two `SPEC_SHA256` byte pins are removed**
   (`balanced_model.py`, `balanced_performance.py`). A -002 spec has a different
   digest by construction, so keeping the pin would mean every rerun requires an
   implementation edit — the exact coupling this task exists to remove. The
   registry lock (`balanced_sources._registry_lock`,
   `balanced_performance._registry_lock`) already requires the latest FROZEN row's
   `frozen_spec_hash` to equal `spec.sha256`, which is the same guarantee held in
   the place that owns freezing. Regression case:
   `tests/test_balanced_model.py::test_registry_lock_rejects_any_byte_change`.

2. **`separation_study.py` / `separation_analysis.py` are restored trimmed.**
   The three families import six symbols from them; the rest belongs to
   adaptive-separation-001, which is not in scope and whose `load_separation_spec`
   carried a hardcoded v7 config digest and the `fixed-baselines-001-v7` /
   `adaptive-confidence-001` ids. Restoring dead code that violates deliverable
   (a) would have been worse than not restoring it. Filenames are kept so
   `implementation_lock`'s hashed file list stays meaningful; the docstrings say
   what the modules now are.

3. **`load_market_inputs` reads the market's full ordered grid** instead of
   `["date", "0.0", *positive_lambdas]`. Identical for the v7 grid, but the DE
   grid `(150, 500)` and the JP grid `(10, 220)` contain no zero lambda, so the
   literal `"0.0"` column would have been a hard read error. Correspondingly
   `expected_lambdas` becomes `set(spec.lambdas)` rather than
   `{0.0, *spec.lambdas}`.

4. **The `λ₀ = 0` β-invariance assertion is conditional**
   (`confidence_model.py:263`). At `λ₀ = 0` every transition penalty is `0`
   whatever β discounts it, so that column must be β-invariant — but only markets
   whose calibrated grid contains zero have such a column. The check still fires
   for `us`; `de` and `jp` have nothing to check.

5. **`run.json` gains `config_path` and the source experiment ids.** A schema
   addition, not a change to any recorded number. Without it the verifiers cannot
   know which contract to reload, and the previous behaviour (assume
   `research.toml`) is silently wrong under a calibrated rerun.

6. **`lagged_study.EVALUATION_STARTS` / `balanced_model.EVALUATION_STARTS` /
   `balanced_performance.OUTER_START` stay pinned as module constants.** These
   are the per-market first out-of-sample decision dates. They are fixed by the
   sample and the 3000-observation fit window, not by the candidate grid, so the
   move to per-market grids does not touch them — and leaving them free would
   make the window that the whipsaw counts are measured over a tunable knob.

7. **`_lock_key`** replaces a bare `path.relative_to(root)` in the five
   implementation locks, falling back to the basename for files outside the repo.
   Only the *label* changes, and only for paths a real run never produces; the
   hashed content is unchanged.

## 5. Status

* `uv run pytest tests/ -q --ignore=tests/test_handoff.py` → **480 passed,
  4 skipped** (the four skips are pre-existing: data files not built in this
  checkout).
* `uv run ruff check src tests` → **All checks passed!**
* No experiment executed; no `research/*.toml`, `experiment_registry.jsonl` or
  `artifacts/` file added, changed or removed.

## 6. What is NOT done here

* No `-002` spec TOML is written (out of scope). Until one exists, every loader
  raises on the archived `-001` files by design — their id, grid shape and source
  tables no longer match. §3 is the checklist.
* No registry rows are added.
* Verifying an archived `-001` run now requires checking out the pre-restoration
  commit, the same rule `1d0c638` recorded for `simple-jm-suite-001`.
