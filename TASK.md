# Task: Hyperparameter Grid Attribution

## Identity

- `task_id`: `hyperparameter-grid-attribution-001`
- `status`: `FROZEN`
- `target_branch`: `cleanup/research-protocol`
- `starting_ref`: `77d9195b709cb2877614c7c542c1f31365f1db99`
- `parent_experiment`: `fixed-baselines-001-v7`
- `claim_class`: `EXPLORATORY`
- `analysis_label`: `PROTOCOL ATTRIBUTION`
- `scientific_claims`: forbidden
- `data_downloads`: forbidden
- `post_2023_access`: forbidden
- `model_refitting`: forbidden
- `adaptive_experiment`: forbidden

The owner approved this task on 2026-07-15 after the v7 proxy
non-replication and its grid-boundary diagnostics had already been observed.
This is therefore exploratory attribution, not preregistered replication
evidence.

## Question

Did the project's decision to enlarge the JM and HMM candidate grids materially
change their relative out-of-sample performance under the otherwise frozen v7
protocol?

The task does not attempt to discover the authors' actual cross-validation
grids. Shu, Yu, and Mulvey (2024), arXiv:2402.05272v3, does not disclose those
complete grids.

## Grid Provenance

The paper's Table 3 reports regime-shift frequencies for fixed illustrative
hyperparameters:

- JM lambda: `[0, 5, 15, 35, 70, 150]`;
- HMM smoothing length k: `[0, 2, 4, 8, 20]`.

Section 3.3 separately states that HMM smoothing originally used `k = 6` before
the paper's cross-validation procedure. Section 3.4.3 says lambda and k are
chosen from ranges of candidates, but does not list the complete ranges.

The following values are project choices, not paper-disclosed values:

- JM `[300, 600, 1200]`: geometric upper-bound expansion;
- HMM `10`: interpolation between illustrated values;
- HMM `[40, 80, 160, 320, 640, 1280, 2560]`: geometric upper-bound expansion;
- the rule requiring expansion when the upper boundary is selected in more
  than 5% of out-of-sample months.

Accordingly, the restricted branch is always named **Table 3 illustrated
grid**, never paper grid, original grid, or author grid.

## Frozen Inputs

The parent is the sealed v7 artifact:

`artifacts/fixed-baselines/fixed-baselines-8adb330565d6-3636939b525d-e9614112b234`

- config SHA-256:
  `8adb330565d64f8ed6edd986f0422dbba72585eda4efd34b0c1b41b95450d81b`;
- data manifest SHA-256:
  `3636939b525d604c5c4180d7e3abb6192b53b81a068f009ad6ca83a945e53a84`;
- inventory SHA-256:
  `08e42044c25ef80e92f4b565034652a6b87fe94de0e9122eee1a418395239d55`;
- parent artifact Git SHA:
  `e9614112b234abcff26f33c446a0a692bf31c262`;
- data cutoff: `2023-12-31`;
- US outer sample: `2007-12-04..2023-12-29`;
- Germany outer sample: `2008-01-03..2023-12-29`;
- Japan outer sample: `2009-05-07..2023-12-29`.

Reuse only the parent's hash-verified precomputed candidate state paths. Do not
download data, rebuild features, refit JM/HMM models, or access any later date.

## Frozen Design

Run the existing monthly cross-validation, signal timing, transaction-cost
accounting, and metric definitions for four cells:

1. expanded JM and expanded HMM: v7 control;
2. Table 3 illustrated JM and expanded HMM;
3. expanded JM and Table 3 illustrated HMM;
4. Table 3 illustrated JM and Table 3 illustrated HMM.

The expanded grids are exactly those in `research.toml`:

- JM `[0, 5, 15, 35, 70, 150, 300, 600, 1200]`;
- HMM `[0, 2, 4, 6, 8, 10, 20, 40, 80, 160, 320, 640, 1280, 2560]`.

The control cell must reproduce the parent's monthly choices, trading paths,
and metrics exactly within the canonical verifier's numerical tolerance. Any
control mismatch stops the experiment before attribution results are opened.

All four cells retain the v7 contract: trailing eight-calendar-year online
validation, monthly selection, annualized excess Sharpe objective, at least 252
valid validation returns, numerical ties resolved toward less smoothing,
terminal online states, signal-to-return offset of two trading rows, local cash
return, 10 bps per one-way trade, and delays 1, 5, and 10.

## Estimands

Primary, separately for each market at delay 1:

`(Sharpe_JM - Sharpe_HMM)_table3_both -
 (Sharpe_JM - Sharpe_HMM)_expanded_control`

Secondary:

- restricted-minus-expanded Sharpe for JM and HMM separately;
- maximum drawdown, turnover, leverage, and switch count;
- the same comparisons at delays 5 and 10;
- selected-candidate frequencies and upper-bound diagnostics.

The 5% diagnostic is reported but does not seal metrics in this study, because
the candidate boundary is the factor being investigated. No grid may be
expanded or edited after results are viewed under this experiment ID.

## Uncertainty And Interpretation

Use paired stationary block bootstrap with 10,000 draws, seed `20260715`, mean
block length 60 trading days, sensitivity lengths 20 and 120, and 95%
confidence intervals. Bootstrap paired daily strategy-excess-return
differences on each cell's common comparison sample.

Classify the descriptive primary sign pattern as:

- `CONSISTENT_EXPLANATION`: positive in all three markets;
- `MIXED_EXPLANATION`: positive in one or two markets;
- `NOT_SUPPORTED`: positive in no markets.

Report confidence intervals separately. These labels mean only whether grid
restriction is consistent with explaining part of the observed JM-versus-HMM
gap. They do not establish the authors' grid, reproduce the paper, validate the
5% rule, or support an adaptive model.

## Artifacts And Runtime

Write ignored runtime output under
`artifacts/hyperparameter-grid-attribution/<run_id>/`. It must include the
frozen spec, parent identity, code identity, candidate choices, boundary
diagnostics, trading paths, metrics, bootstrap draws or resumable state,
inventory, verifier receipt, and an English HTML report.

Expected runtime is 5-15 minutes on the current host because model fitting is
reused. This is an estimate, not an acceptance criterion.

## Acceptance

1. Frozen spec and registry identity agree byte-for-byte by SHA-256.
2. Parent artifact and all reused inputs pass canonical hash verification.
3. No fit, data-provider, post-2023, or adaptive code path is invoked.
4. Expanded-expanded control reproduces v7 before attribution is opened.
5. All four cells use identical non-grid protocol and common metric samples.
6. Recalculation and inventory verification pass independently.
7. A real local-monitor subprocess reaches a terminal verified state.
8. The English report passes desktop and mobile Chromium checks with no page
   error, horizontal overflow, blank primary chart, or contradictory label.
9. No generated artifact or runtime state is tracked by Git.

## Write Boundary

Authorized tracked paths are:

- `TASK.md` and the archive copy of the completed monitor task;
- `research/hyperparameter-grid-attribution.toml`;
- append-only `research/experiment_registry.jsonl`;
- the minimum existing CLI, monitor registry, research, verifier, reporting,
  and focused test files required to run this exact study;
- at most two small new research modules if existing modules cannot express the
  comparison without mixing responsibilities;
- the two procedural `.agent/` handoff files.

Do not modify dependencies, `research.toml`, market data, parent artifacts,
learning material, baseline model mathematics, features, state labels, costs,
delays, metric definitions, or the frozen v7 conclusion.

## Commit Sequence

1. Freeze this contract, machine-readable spec, and registry row; stop.
2. Implement the smallest runner, verifier, report, and tests; stop.
3. Run through the local monitor and verify the generated artifact; stop.
4. Record the exact exploratory result and browser acceptance; close the task.

Each commit remains below approximately 400 changed lines and 15 files and is
pushed to `origin/cleanup/research-protocol` before the next milestone.
