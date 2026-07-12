# AGENTS.md

## 1. Purpose

This repository supports serious mathematical-finance research on adaptive
jump models for daily market-regime detection. The intraday/minute stack under
`archive/` is frozen provenance, not an active product. Reactivating intraday
work requires an explicit user-approved task that replaces the active scope;
never maintain daily and intraday research stacks in parallel.

When present and branch-relevant, `TASK.md` defines the current frequency,
hypothesis, protocol, allowed files, and deliverables. If no active `TASK.md`
exists, the latest explicit user request is the task contract for review and
setup work only. Do not start a conclusion-bearing experiment until its frozen
contract exists in `TASK.md` or a versioned machine-readable config.

The objective is not to build a large software product. Prefer small, explicit,
testable research modules. Mathematical correctness, causal evaluation,
reproducibility, and honest claims take priority over feature count, runtime
convenience, or presentation.

The owner is learning the theory and does not want autonomous research drift.
After each approved milestone, stop for review before starting the next model,
data acquisition, long run, dependency group, dashboard feature, or claim.
Explain assumptions and tradeoffs in plain language before implementation.

When a task authorizes a learning deliverable, create or update a tracked HTML
lesson under `docs/learning/`. It MUST:

- assume no prior mathematical-finance or software knowledge;
- introduce intuition and a tiny hand-worked numerical example before notation;
- define every symbol, unit, index, and state convention;
- connect the equation to the exact code path and resulting artifact;
- separate intuition, exact mathematics, empirical evidence, and limitations;
- state what the result does and does not prove;
- use hand-built HTML/CSS visuals rather than Mermaid;
- render correctly in a real browser on desktop and mobile.

Learning HTML is authored documentation, not an experiment result. Never copy
rounded result values into it manually; source any empirical value from a
verified run artifact.

In this file:

- **MUST / MUST NOT** are non-negotiable.
- **SHOULD / SHOULD NOT** require a documented reason to deviate.
- **MAY** is optional.

## 2. Instruction Precedence and Sources of Truth

Follow instructions in this order:

1. platform, safety, and security constraints;
2. account-level and ancestor-directory `AGENTS.md` files, which this project
   file may tighten but never loosen;
3. the user's latest explicit request;
4. the active and branch-relevant `TASK.md`;
5. this project `AGENTS.md`;
6. agent-specific files such as `CLAUDE.md`;
7. `STATUS.md`, session logs, reports, commit messages, and other documentation.

A lower-precedence document cannot override constraints or expand authorization
imposed by a higher-precedence instruction. An explicit user request may grant
an approval that a rule requires, narrow scope, replace a stale task, or
downgrade a claim to EXPLORATORY / ENGINEERING. It does not silently waive
account-level constraints or scientific evidence gates for a conclusion-bearing
claim.

Git state and the checked-out files are authoritative for what code exists.
`TASK.md` is authoritative only for the task it actually describes. Session
logs, `STATUS.md`, PR text, reports, and commit messages are context and claims
to verify, not proof.

If `TASK.md`, `STATUS.md`, the latest handoff, and the current branch disagree:

- do not silently choose one;
- identify the branch/ref and exact HEAD being inspected;
- state which document appears stale;
- follow the current user request and ask only if the conflict would materially
  change scope, mathematics, data, or external state.

Never assume `main` is the requested review surface when the user names a PR,
branch, commit, worktree, or artifact.

An active `TASK.md` SHOULD identify `task_id`, `status`, and `target_branch` or
`target_ref`. If these are missing or conflict with the current target, treat
the file as context until reconciled with the user's request. `STATUS.md` is a
derived summary; update it only when requested or task-authorized and only
after a verified milestone.

## 3. Request Type and Authorization

Classify the request before acting:

- **Review / diagnose / explain / report:** read-only unless changes are also
  requested.
- **Implement / fix / refactor:** change only the smallest coherent scope
  required by the request, including necessary tests and task-authorized docs.
- **Run an experiment:** generate only the data and artifacts authorized by the
  active task.
- **Publish:** commit, push, PR, release, upload, or other remote mutation
  requires explicit authorization.

Do one task at a time. Do not turn a review into an implementation, and do not
add adjacent modules merely because they may be useful later.

If `TASK.md` defines allowed files, treat that list as a write boundary. Reading
other relevant repository files for analysis is allowed. Procedural handoff
files are covered by Section 22.

For small, reversible implementation details within scope, make a reasonable
assumption and record it. Ask before proceeding when ambiguity would change:

- the research hypothesis or claim class;
- the mathematical objective or state semantics;
- data source, universe, frequency, or sample boundaries;
- public APIs, file structure, dependencies, or authorized files;
- external state, paid services, or destructive operations.

## 4. Start-of-Work Orientation Gate

Before repository work:

1. run `git status --short --branch`;
2. record repo root, branch/worktree, exact HEAD, and dirty status;
3. confirm whether the target is `main`, a PR, a branch, a commit, or the
   current working tree;
4. read `TASK.md` and the latest valid `.agent/session-log.jsonl` entry, if
   present;
5. inspect the relevant code, tests, config, data manifest, and existing diff;
6. state the single active task and the intended verification.

Do not modify the repository merely to perform orientation. A detached HEAD is
not an error, but MUST be reported before any commit or publication.

Preserve pre-existing user changes. If dirty changes overlap the requested
files, inspect them and stop before any action that could overwrite, discard,
or reinterpret them.

For a non-trivial task, maintain a short plan and one overall objective.
Bounded independent steps may run in parallel only with disjoint file and
output ownership.

## 5. Scope and Change Discipline

Make the smallest coherent change that satisfies the task. Necessary tests,
run manifests, and directly affected documentation are part of that change;
unrelated cleanup is not.

Without explicit authorization, MUST NOT:

- rename or remove public functions, classes, CLI flags, or output columns;
- reorganize directories or rename/delete files;
- rewrite unrelated code or generated reports;
- change the hypothesis, primary metric, success rule, or evaluation protocol;
- add, remove, or upgrade dependencies;
- introduce data downloads, brokerage behavior, live trading, or remote
  services;
- modify files outside the active task's write boundary;
- use destructive Git commands.

If an out-of-scope defect is discovered, report it as a risk or proposed next
task. Fix it only when it directly blocks the authorized task and the user or
`TASK.md` permits the expansion.

Phase-specific permissions, module ordering, symbols, grids, run sizes, and
deliverables belong in `TASK.md`, not in this project policy file.

## 6. Research Claim Classes

An experiment or artifact is **conclusion-bearing** when it is used to support,
reject, rank, or headline a research hypothesis or economic claim. Every claim
or experiment component MUST declare a class. A mixed report declares one
primary class and maps each secondary claim to its own class:

- **REPRODUCTION:** has no unresolved outcome-relevant deviation from a named
  source's data definition, sample period, features, preprocessing, estimator,
  tuning, online inference, costs, delay, and metrics. Treat a protocol
  difference as outcome-relevant unless equivalence is demonstrated.
- **REPLICATION:** tests the same research hypothesis with deliberate,
  documented data or protocol differences.
- **EXTENSION:** tests a new model or hypothesis against validated baselines.
- **EXPLORATORY:** develops hypotheses or protocol; results may guide later
  choices but are not confirmatory.
- **ENGINEERING / SMOKE:** validates wiring, runtime, or artifacts and supports
  no scientific conclusion.

A close match in one metric, market, period, or table cell is not reproduction.
A reproduction MUST include a source-versus-implementation protocol table and
compare every predeclared primary result.

Claim class describes intent and protocol, not whether the result succeeded.
Report reproduction outcome separately as MATCHED, NOT_MATCHED, or
INCONCLUSIVE.

Use precise conclusion language:

- Prefer "no consistent improvement was detected for this model, data, and
  protocol."
- Do not infer equivalence, a universal null, or a false hypothesis from failure
  to outperform.
- A null or equivalence claim requires a predeclared equivalence margin,
  appropriate uncertainty, and adequate power.
- Correlated variants on the same outer sample are not independent experiments.
- An isolated selected win is exploratory unless multiplicity was handled.

Restricted terms:

- **reproduced:** requires the reproduction gate and predeclared tolerances
  across primary results; "near-exact reproduction" is prohibited -- use
  REPRODUCTION or a qualified REPLICATION;
- **calibrated:** requires a stated calibration procedure and diagnostics;
- **optimal / CV-optimal:** means only optimal over the declared search domain,
  with no unresolved boundary issue;
- **robust:** requires the predeclared robustness suite;
- **statistically significant / null:** requires an inferential procedure;
- **economically meaningful:** requires a causal backtest with comparator
  parity and cost/delay sensitivity;
- **alpha / tradable / production-ready:** requires separate evidence and is
  never implied by a research backtest.

## 7. Frozen Experiment Contract

Before a conclusion-bearing empirical run, record a frozen specification in
`TASK.md`, a design document, or a machine-readable config. Use `N/A` with a
reason for genuinely inapplicable fields. Algorithm-only work instead requires
the objective, assumptions, identifiability analysis, and test plan in Section
8. The empirical specification MUST define:

1. experiment ID, claim class, hypothesis, and primary estimand;
2. exact mathematical objective, units, indices, transition direction, and
   state-label convention;
3. data source, instruments, frequency, fields, cutoff, and sample dates;
4. train, inner-validation, outer-test, and locked-confirmation boundaries;
5. feature timing, preprocessing, warm-up, and missing-data rules;
6. fitting, refitting, decoder initialization/history, and inference mode;
7. baseline and challenger pipelines;
8. complete grids, caps, tie-breaks, seeds/restarts, and stopping rules;
9. selection objective, primary metrics, uncertainty method, and multiplicity
   handling;
10. signal timing, execution delay, transaction costs, risk-free return, and
    accounting conventions;
11. success, failure, and inconclusive criteria; outcome-dependent early
    stopping is forbidden unless a formal sequential design was frozen;
12. expected runtime, artifacts, and whether the run is development or
    confirmatory.

Before a locked holdout is opened, the specification MUST be committed or
content-hashed. A post-result change to result-affecting code, a feature, grid,
threshold, window, metric, market set, state mapping, or success rule creates a
new experiment ID. Even a valid bug fix consumes the viewed holdout for the old
claim; subsequent evidence on that sample is EXPLORATORY.

Do not silently promote a smoke, partial, interrupted, or development run to a
full or confirmatory result.

## 8. Mathematical Semantics and Identifiability Gate

Before empirical testing of a new parameter, penalty, transition rule, or model:

1. write the exact objective and define every term, unit, index, sign, and
   source-to-destination convention;
2. derive the mapping from theory or cited model to the implemented scale;
3. show that the new parameter changes path ranking beyond a constant, endpoint
   term, label swap, or reparameterization;
4. verify dimensional consistency, limiting cases, state permutations, and the
   nested baseline;
5. add deterministic, invariance, directionality, and brute-force oracle tests
   where feasible;
6. establish structural identifiability analytically, then run synthetic
   recovery tests for finite-sample estimability and implementation behavior;
7. distinguish joint model fitting from a decode-only modification.

For the two-state case, explicitly check:

`c01 * N01 + c10 * N10
 = 0.5 * (c01 + c10) * N_switch
 + 0.5 * (c01 - c10) * (state_T - state_0).`

Therefore, a two-state zero-diagonal directed switch-cost matrix is a symmetric
switch penalty plus a boundary term. It MUST NOT be described as identifying
different bull/bear dwell-time hazards or state-specific persistence.

A theory-derived value requires the derivation and assumptions, not only a
citation. The code MUST test scale and sign. Do not convert a difference into a
ratio, exponentiate a log-odds quantity, or discard transition normalization
terms without a derivation showing that the transformation is valid.

A claim about state-specific transition hazards requires a row-normalized
transition likelihood including stay costs, or an explicit semi-Markov/duration
model, plus a synthetic implied-duration recovery test. A decode-only change
MUST NOT be described as a jointly fitted asymmetric model.

A brute-force test proving that dynamic programming solves the supplied
objective validates the algorithm, not the scientific semantics of that
objective. Both require separate tests.

If a parameter is not identifiable under the proposed objective or
data-generating process, it cannot support an economic interpretation or a null
conclusion.

## 9. Result-Affecting Choices and Parameter Provenance

Every result-affecting choice MUST appear in a parameter ledger with:

`name, value_or_grid, units, role, provenance, selected_on, bounds,
sensitivity, experiment_id`.

A versioned machine-readable config or manifest may serve as this ledger when
it contains all required provenance fields; do not maintain a duplicate manual
table merely for compliance.

This includes model coefficients and also clipping, caps, window lengths,
cadence, minimum runs, grid bounds, tie-breaks, thresholds, seeds/restarts,
label mapping, delay, costs, and risk-free assumptions.

Use one provenance category:

- **ESTIMATED:** learned from training data by a stated likelihood/objective;
- **INNER_CV:** selected only inside the training/validation structure;
- **THEORY:** derived with assumptions, scale, and citation;
- **SOURCE_FIXED:** copied for reproduction fidelity; this is not evidence that
  the value is calibrated for an extension;
- **PREREGISTERED:** a protocol/scenario choice fixed before outer results and
  covered by sensitivity;
- **SCENARIO:** an economic assumption evaluated as a sensitivity, not tuned;
- **NUMERICAL_GUARDRAIL:** demonstrated non-binding;
- **UNCALIBRATED:** none of the above.

An UNCALIBRATED choice may appear in exploratory work but MUST NOT support a
confirmatory headline claim. If a numerical cap or fallback binds, it is no
longer merely numerical and MUST be calibrated, expanded, or treated as a
protocol limitation.

Model coefficients, transition costs, and statistical decision thresholds
cannot become calibrated merely by being PREREGISTERED. They require
ESTIMATED, INNER_CV, THEORY, or justified SOURCE_FIXED provenance as applicable.

No hand-set "complexity rent," Sharpe threshold, noise threshold, improvement
margin, minimum duration, or transition cost may be presented as statistically
or theoretically calibrated without the corresponding derivation or
calibration.

## 10. Causal Walk-Forward Evaluation

For a state, prediction, signal, or position at decision time `t`, every
result-affecting quantity MUST be computable from information available at or
before `t`.

Record and test the event-time tuple:

`observation_available_at -> decision_at -> execution_at -> earned_return_interval`.

Index alignment MUST rule out same-close or same-bar execution using information
that was not yet available.

- Fit scalers, imputers, clipping, feature transforms, labels, centers,
  penalties, state mappings, and hyperparameters using past data only.
- When any result-affecting model or hyperparameter selection occurs, use an
  inner time-series selection loop and a distinct outer evaluation loop.
- Apply the same timing, cost, delay, cash-return, and accounting conventions
  during validation and evaluation.
- Purge or embargo boundaries when features or targets overlap them.
- Specify refit cadence and whether decoder state/history is carried, warmed
  from past observations, or reset. A reset is a material protocol choice.
- For outputs claimed to be online or filtered, test prefix invariance:
  appending future observations MUST NOT change already emitted states.
- Separate offline segmentation, filtered inference, smoothed/Viterbi paths,
  and real-time detection. Never call them interchangeable.
- Do not use future-state smoothing, full-sample relabeling, or hindsight for an
  online claim.

State labels are permutation-invariant. Any state-to-bull/bear or
state-to-position mapping MUST be learned on training data and then frozen for
the associated outer period. Never choose it using outer returns.

Pointwise causal OOS is not the same as an untouched confirmatory holdout.

## 11. Holdout Governance

Maintain separate development and locked-confirmation samples for every
confirmatory empirical REPLICATION or EXTENSION. Without an untouched sample,
the claim remains EXPLORATORY.

- Lock the confirmation period before viewing its results.
- Freeze implementation, grids, primary metrics, and decision rules before
  opening it.
- If its results influence any feature, coefficient, threshold, grid, window,
  market choice, post-hoc hypothesis, selected primary outcome, or success
  criterion, it becomes development data.
- A new post-hoc claim requires a new untouched period, a prospectively defined
  eligible-market universe run exhaustively, or prospective evaluation. Never
  select a new confirmation market after seeing its outcome.
- Record unsuccessful and abandoned variants; do not preserve only the winning
  research path.

Repeatedly inspecting an outer sample while changing the method does not cause
look-ahead inside each run, but it destroys confirmatory status. Label it
honestly.

## 12. Baselines, Selection Fairness, and Search Boundaries

Validate the baseline before evaluating an extension.

Compare baseline and challenger as complete selection pipelines:

- same outer dates and eligible observations;
- same return, cost, delay, risk-free, and accounting definitions;
- same preprocessing information set;
- adequate convergence checks for each estimator, with all search and compute
  differences disclosed;
- clearly disclosed feature or inference differences;
- selection performed without outer-test information.

"Baseline nested in the grid" is true only when `beta=0` or the corresponding
case uses the same implementation, objective, preprocessing, seeds/restarts,
tie-breaks, and scoring procedure.

The same validation dates do not imply the same search budget. When one family
searches more candidates or was iterated more often, use nested time-series
selection or an untouched outer comparison of the complete pipelines. Merely
disclosing the advantage keeps the result EXPLORATORY; it is not CLAIM_READY.

Record all evaluated trials and the search trace; for a finite grid, record the
full candidate surface. Flag every selected boundary value. A headline value is
not CV-optimal until grid expansion or a plateau test is completed on
development data. Until resolved, call it "best in the searched grid."
Expanding a grid after outer results creates a new experiment ID and consumes
that outer sample for confirmation.

## 13. Uncertainty, Multiplicity, and Negative Results

Separate and report:

1. optimizer/initialization variability;
2. time-series sampling variability;
3. model-selection variability;
4. period, market, and endpoint sensitivity;
5. provider, revision, and implementation uncertainty.

Repeated seeds estimate only item 1. A five-seed range is not a Sharpe noise
floor and cannot define a universal discovery threshold.

For economic comparisons:

- use paired net-return or paired outer-fold differences;
- use a predeclared time-series-appropriate uncertainty method, such as block
  bootstrap or HAC when its assumptions are defensible;
- report point estimates and intervals, not only rounded rankings;
- report the number of markets, metrics, periods, models, and variants searched;
- address multiplicity when selecting headline results;
- report cost, delay, turnover, drawdown, and endpoint sensitivity.

Overlapping folds, correlated markets, repeated endpoints, and reused samples
are not independent observations. Define the dependence/cluster unit for the
estimand, resample at that unit, and report effective sample size or its best
defensible analogue.

State whether uncertainty is conditional on an already selected model. To
include model-selection uncertainty, the resampling or repeated-split procedure
MUST refit and reselect the complete pipeline; bootstrapping only final OOS
returns estimates conditional performance.

Do not call a threshold "BIC-like" unless it is actually derived on the
corresponding likelihood and penalty scale. Do not count rented/free variants
of the same search as independent null experiments.

## 14. Paper Reproduction and Literature Use

For a named-paper reproduction:

- use the exact named/versioned paper as the primary source and check its
  official appendix, code, data documentation, and errata;
- cite the exact section, equation, table, and implementation reference;
- create a protocol-difference table covering provider, price versus total
  return, risk-free series, dates, frequency, features, warm-up, fit/validation/
  test windows, refit cadence, inference, smoothing, delay, costs, and metrics;
- implement the live model-selection chain, not merely a superficially similar
  walk-forward loop;
- compare the full predeclared table or pattern, not a favorable cell.

If a material component differs, call the work a REPLICATION or pattern
comparison, not an exact reproduction.

Before claiming novelty, search and cite the relevant prior art. Existing
implementations, state-specific transition models, semi-Markov models, and
time-varying-transition models MUST be distinguished from the proposed
contribution.

## 15. Data Integrity and Provenance

`data/raw/` is immutable. Never delete, overwrite, reformat, repair, or write
derived columns into raw files.

Store:

- derived datasets and caches under `data/processed/`;
- immutable run outputs under `artifacts/<run_id>/`, including figures, tables,
  dashboards, manifests, full-precision evidence, and narrative reports;
- authored learning material under `docs/learning/`;
- temporary debug output under `artifacts/debug/<debug_id>/` and label it
  `DEBUG_ONLY`.

`data/`, `artifacts/`, and generated `reports/` content are local ignored
runtime state and MUST NOT be committed. Frozen specifications, source code,
schemas, acquisition procedures, and the experiment registry are tracked
project metadata and MUST live outside ignored output directories.

Never silently fall back from missing real data to synthetic data, a shorter
sample, a different symbol, or stale cached output.

Every conclusion-bearing dataset MUST have a manifest recording:

- vendor/source and retrieval cutoff;
- symbol, field, frequency, and row count;
- price-return versus total-return status and adjustment convention;
- risk-free series and excess-return convention;
- timezone, daylight-saving, session/calendar, and timestamp semantics;
- missingness, filtering, exclusions, and corporate-action handling;
- first/last usable date and warm-up loss;
- content hash or equivalent immutable identifier.

External data downloads require explicit authorization. Any authorized
acquisition MUST be scripted and record query parameters and retrieval time.
Existing files in `data/raw/` remain immutable; adding a new write-once raw file
requires explicit task authorization. Never commit credentials, API keys,
paid/licensed data, or large raw datasets.

If licensing prevents committing data, commit the acquisition/validation
procedure and manifest. A clean checkout with the documented permitted data
must be able to reproduce the reported inputs.

Distinguish:

- **computational reproducibility:** rerun from an immutable snapshot/hash;
- **source reproducibility:** repeat an API/vendor query that may later revise;
- **conceptual replication:** retest the method when the original snapshot
  cannot legally or technically be preserved.

A downloader alone does not guarantee bitwise reproducibility.

For intraday data, additionally document bar-close availability, exchange
timezone/session and DST rules, quote-versus-trade source, clock alignment,
stale/crossed quote handling, bid-ask bounce, halts, auctions, latency, spread
and market-impact assumptions, and the prohibition on same-bar execution. For
multi-asset universes, address constituent history, delistings, and survivorship.

## 16. Backtesting Rules

Backtesting requires authorization in the current user request or `TASK.md`.
It is required before claiming that regimes are economically meaningful.

Every backtest MUST:

- use signals available by the declared decision timestamp;
- apply the declared execution delay before returns are earned;
- map states to positions using training data only;
- compare all models on identical outer dates;
- state whether returns are arithmetic or log and how they are annualized;
- include cash/risk-free return, one-way cost, slippage assumptions, and initial
  allocation conventions;
- define turnover units, trades, exposure, leverage, and position bounds;
- preserve aligned OOS dates, raw returns, states, signals, positions, costs,
  and net returns so metrics can be recomputed;
- for a full conclusion-bearing run, evaluate predeclared cost and delay
  sensitivity and an economically relevant effect-size threshold under
  plausible friction scenarios.

Validate accounting identities and test the first position, transitions,
block boundaries, delayed execution, cash periods, and final equity.

No brokerage integration, live orders, or production deployment is authorized
by a backtest task.

## 17. Quick, Full, and Confirmatory Execution

Quick/smoke mode is for debugging only. Mark its artifacts `DEBUG_ONLY` and
keep them separate from full outputs.

Full mode MUST execute the frozen protocol over the authorized data, grids,
seeds/restarts, and sensitivities. MUST NOT silently:

- downsample or shorten the period;
- shrink grids, restarts, or seed counts;
- skip models, markets, folds, or sensitivity cases;
- reuse stale incompatible artifacts;
- replace real data with synthetic data.

Before a long run, state expected scope and approximate runtime. A requested
full run is authorization to perform that documented local run; do not reduce
it merely to save computation.

Use checkpoints for expensive runs. An incomplete run may atomically update or
append checkpoints under its run ID. Once marked complete, that run is
immutable and any rerun receives a new ID. If interrupted or blocked, label the
result PARTIAL and report exactly what completed.

A full runner MUST record the expected versus completed cross-product of
markets, folds, seeds/restarts, and sensitivity cases, and exit nonzero when a
required cell is missing.

## 18. Coding and Dependency Rules

- Prefer small, explicit functions and modules.
- Keep core logic in `src/adaptive_jump/`; notebooks and runners stay thin.
- Add type hints and docstrings to public functions.
- Pass random generators/seeds explicitly.
- Validate inputs and raise specific errors.
- No silent fallback or broad `except Exception: pass`.
- Keep fitting, decoding, evaluation, and reporting separable.
- Preserve backward compatibility unless an API change is authorized.
- Do not hand-edit generated artifacts when a source generator exists.
- Inspect the final diff for unrelated formatting and generated-file churn.

The standard library and dependencies already locked by the project are
allowed. Ask before adding a package, changing a pin, or introducing a new
runtime/service. Building a temporary environment from the existing lock is
allowed. Package availability does not authorize data downloads.

Keep dependency roles explicit in `pyproject.toml` and its lockfile:

- **core:** numerical research and the canonical experiment runner;
- **dev:** tests, lint, and packaging checks;
- **viz:** static or interactive plotting used by verified reports;
- **dashboard:** the local experiment-control and observability application;
- **audit-backtest:** independent backtest/metrics libraries used only for
  parity checks against the canonical accounting engine.

Do not install every anticipated library into core. Add an optional dependency
only in the task that exercises it, pin/lock it, test its real path, and explain
why the existing stack is insufficient. A dashboard MUST call the same frozen
runner/config used by the CLI; it MUST NOT contain hidden model defaults,
separate research logic, or editable confirmatory outcomes. A third-party
backtest is an audit comparator, not ground truth: align dates, positions,
execution timing, costs, cash returns, and finalization rules before comparing
results.

Explain why the current stack is insufficient before requesting a dependency.
Do not add one merely for convenience when the existing stack is adequate.

## 19. Testing and Verification Gates

Behavior-changing implementation work requires tests. Documentation-only work
does not require invented unit tests, but MUST receive diff/format/consistency
checks.

For mathematical code, include as applicable:

- deterministic small examples;
- edge and invalid-input cases;
- brute-force oracle comparisons;
- numerical invariants and scale/sign tests;
- permutation and state-label tests;
- baseline nesting after the predeclared canonical relabeling, or by a
  permutation-invariant path/objective comparison;
- directionality and identifiability tests;
- synthetic parameter/path recovery.

For time-series and backtest code, include:

- chronology and split-boundary tests;
- prefix-invariance tests for online inference;
- train-only preprocessing and label-mapping tests;
- delay, cost, turnover, cash, and equity accounting tests;
- block reset/carry-state tests.

For runners, add a small integration test that exercises the real config schema
and expected artifacts. Unit tests do not validate reported market metrics, and
quick runs do not validate full-mode claims.

Run focused tests first, then the relevant suite and lint. Never claim a command
passed unless it was run in the current worktree. Record exact commands,
counts, failures, skips, warnings, and anything not run.

Do not weaken tests to make an implementation pass. A regression test for a bug
SHOULD fail before the fix. Passing tests and lint demonstrate implementation
health, not validation of a research hypothesis.

Use three distinct completion states:

- **CODE_COMPLETE:** implementation and code-level tests pass;
- **EXPERIMENT_COMPLETE:** the frozen full protocol ran and artifacts are
  complete;
- **CLAIM_READY:** uncertainty, robustness, provenance, and claim gates pass.

Do not collapse these into a single "done."

## 20. Reproducible Runs and Artifacts

Every conclusion-bearing run MUST have an experiment/run ID and a
machine-readable manifest containing:

- claim class, hypothesis, primary metric, and config;
- exact Git HEAD and clean/dirty status;
- exact command and all resolved defaults;
- Python and dependency versions;
- data manifest IDs/hashes and exact date/fold boundaries;
- grids, tie-breaks, seeds/restarts, and state mapping;
- start/end time, runtime, completion status, and warnings;
- generated artifact paths.

Maintain a tracked append-only experiment registry at the path defined by
`TASK.md` (default: `research/experiment_registry.jsonl`). It is research
metadata, not a generated report, and MUST NOT live under ignored `artifacts/`
or `reports/`. Each entry records at least:

`experiment_id, parent_id, frozen_spec_hash, claim_class, outcome, status,
outer_sample_ids_accessed, result_driven_changes, timestamp`.

The registry is the audit trail for holdout reuse and correlated research
attempts. It does not replace the per-run manifest.

Save machine-readable, full-precision evidence:

- all candidate/CV scores and selected paths;
- per-fold and per-market metrics;
- aligned OOS states, signals, positions, returns, costs, and net returns where
  licensing/size permits;
- formulas or code paths used to derive aggregate metrics.

Reports and plots MUST be generated from these artifacts, not by manually
copying rounded values. HTML, PNG, or a headline CSV without a reproducible
generation path is not sufficient evidence.

A confirmatory run and any CLAIM_READY result require a clean committed tracked
source/config tree. Authorized ignored data and artifact files may exist, but
their manifests and hashes MUST identify the exact inputs. Any uncommitted or
untracked source, config, acquisition code, or learning/report generator makes
the run dirty. A dirty-tree exploratory run must archive the full tracked patch
and identify or hash every untracked source/config input; a patch hash alone is
insufficient.

Report generation MUST fail closed rather than mix artifacts when experiment
ID, config hash, data hash, or Git HEAD is inconsistent.

Do not overwrite a prior conclusion-bearing run. A changed config gets a new
run ID. If artifacts are too large or licensed, store hashes, schemas, and exact
regeneration instructions.

## 21. Git, Multi-Agent Work, and External State

Run `git status` before and after work. Preserve unrelated and pre-existing
changes.

Do not reset, clean, restore/checkout over file changes, stash, amend, rebase,
commit, push, or open/update a PR unless explicitly authorized. Switching to a
branch/ref explicitly named by the user is allowed only after confirming that
the worktree is safe and recording the resulting HEAD.

When a commit is authorized:

- stage only in-scope files;
- inspect the staged diff;
- exclude secrets and unauthorized data/artifacts;
- report the exact commit SHA.

For multi-agent work:

- the primary/top-level agent owns the plan, repository edits, verification,
  and handoff;
- delegate bounded, independent tasks;
- only one writer may own a file or output directory at a time;
- subagents should return findings, not append session logs;
- never run parallel experiments that write the same paths;
- the primary agent MUST verify child-agent claims and inspect the combined
  diff.

Local completion does not imply permission to publish or mutate remote state.

## 22. Session Handoff

`.agent/session-log.jsonl` is append-only handoff context, not the source of
truth for code or results.

Only the primary/top-level agent appends a handoff entry. Append exactly one
entry when the session materially changes the repository or runs a
conclusion-bearing/expensive experiment. Skip log writes for read-only reviews,
no-edit requests, and subagent-only analysis.

New entries MUST use the canonical fields and types:

`ts: string, agent: string, model: string, goal: string, files: string[],
verification: string[], commit: string, next: string, notes: string`.

Legacy entries may omit `model`, use `next_step`, or store `verification` as a
string. Preserve them as append-only history and make the renderer read both
forms; never rewrite old lines merely to normalize schema.

Record branch, starting/ending HEAD, dirty status, experiment/run IDs, partial
runs, and blockers inside `notes` until the handoff schema is explicitly
upgraded.

- Use `commit: "uncommitted"` when no commit was created and record HEAD in
  `notes`.
- Record only commands and results from the current session.
- Never fabricate a commit, test result, runtime, or artifact.
- Never include secrets, tokens, private data, or signed URLs.

After appending, run:

`bash .agent/handoff.sh '<one-line-json-entry>'`

The helper MUST validate JSON, required fields, and field types before append.
Malformed log lines MUST fail loudly with a line number; the renderer MUST NOT
silently skip audit-trail corruption. Do not edit `session-log.html` manually.
If handoff files are outside an active task's allowed-file list, this section
grants a narrow procedural exception only for the two handoff files.

Ordering is: verify and inspect the diff; perform any explicitly authorized
code commit; append/render the handoff; inspect final status; report. Handoff
files normally remain uncommitted after recording a commit SHA unless the user
separately authorizes a metadata commit; disclose that dirty state.

## 23. Completion and Reporting

A research task may be CODE_COMPLETE, EXPERIMENT_COMPLETE, or CLAIM_READY.
Report which state was reached.

For every repository-changing or experiment-running top-level task, report:

1. status: complete, partial, or blocked;
2. completion state;
3. files changed, or `none`;
4. what changed;
5. tests added, or why none were needed;
6. exact commands and results;
7. generated artifacts and key findings;
8. protocol deviations and unverified items;
9. limitations, remaining identification risks, and smallest logical next step.

Before reporting completion:

- inspect `git diff` and `git status`;
- run `git diff --check`;
- verify no unrelated or unauthorized changes;
- ensure the conclusion is no stronger than the design and evidence;
- do not claim reproducibility when inputs, config, or commands are missing.

A negative or inconclusive result may complete a research task if the authorized
protocol was executed fully and reported honestly. A positive result is not
CLAIM_READY until the same evidence gates pass.

For a short read-only review, provide concise evidence-backed findings and
state that no files or external state were changed; the full nine-item template
is optional.

## 24. Explicit Anti-Patterns

The following are forbidden:

- one close Sharpe value => "near-exact reproduction";
- seed spread => sampling noise floor;
- a hand-entered threshold => complexity penalty or BIC;
- same validation dates => same model-selection budget;
- a nested parameter value with different restarts/procedure => exact baseline
  nesting;
- zero-diagonal directed costs => state-specific duration asymmetry;
- block-reset endpoint effects => economic transition asymmetry;
- repeated grid-boundary selections => calibrated optimum;
- passing unit tests => validated research hypothesis;
- HTML/PNG/rounded CSV without manifest => reproducible experiment;
- repeatedly viewed OOS data => untouched holdout;
- no improvement in one implementation => a general model class is disproved;
- quick, partial, stale, or incompatible artifacts => full result.
