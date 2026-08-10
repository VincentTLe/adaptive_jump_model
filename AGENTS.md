# AGENTS.md - adaptive_jump_model

This repository studies daily market-regime models based on Shu, Yu, and Mulvey
(2024), arXiv:2402.05272. Account-level instructions also apply.

**MUST** and **MUST NOT** are non-negotiable. **SHOULD** requires a recorded
reason to deviate.

**AI confidence is not evidence.** Fluent, assured prose is not a substitute
for a check that was actually run. If the check was not run, say so.

## 0. Scientific objective

**The unit of progress is a scientific question answered.**

> Can a causal Jump-Model-guided market/cash strategy identify persistent
> unfavorable regimes well enough to achieve higher net risk-adjusted
> performance than both buy-and-hold and Gaussian HMM under the same protocol?

There are two distinct levels of evidence. A mathematical extension has local
value when it changes the fixed-JM objective in a verified, causal, identifiable
way or improves a declared mechanism metric. It has economic value only when
the resulting strategy beats the benchmarks under the same protocol.

**The primary estimand for a structural challenger is the paired delta against
the fixed JM.** For market `m` and prespecified variant `v`:

`Delta_m(v) = Sharpe_v,m - Sharpe_fixedJM,m`.

The fixed JM is the arm the extension claims to improve, so it is the arm the
extension must beat. The cross-market target is `Delta_m(v) > 0` for the same
frozen `v` in every declared market, with non-trivial magnitude. Pairing is not
cosmetic: a perturbation that moves both arms together contributes a common
component that cancels in the difference, which is why the paired delta is
measured directly rather than inferred from each arm's separate spread.

The **stronger economic criterion**, which a variant must also satisfy before it
is claimed to be economically useful, is:

`Sharpe_v,m > max(Sharpe_fixedJM,m, Sharpe_HMM,m, Sharpe_BuyHold,m)`.

**Beating HMM and buy-and-hold while losing to the fixed JM is NOT evidence that
an extension improved the JM.** It only shows the JM family beats those
benchmarks, which is already the established baseline result. Any variant that
loses to fixed JM in a market has failed in that market regardless of how it
scores against HMM or B&H, and must be reported that way.

Maximum drawdown, turnover, cash fraction, and switch count are secondary
guardrails. Exact numbers reported by Shu, Yu, and Mulvey are context, not a
replication target, because this repository uses a later public proxy sample.

The canonical baseline for downstream work is whichever fixed-baselines config
is currently sealed and recorded as canonical in `TASK.md` and the registry —
today v11-ninit60 (`5b12efa2…`), with its known DE defect disclosed. A sealed
baseline is a fixed comparator, not permanent proof, and not a reason to forbid
new exploratory work. Data limitations are reported separately from model
behavior.

Priority is:

1. the owner's latest explicit request;
2. the smallest experiment that moves the question above;
3. the smallest code or data fix that unblocks that experiment;
4. infrastructure, presentation, and process only when explicitly requested.

Do not start new monitor features, textbook work, governance documents,
frameworks, or broad refactors unless the current request asks for them or they
are the smallest blocker to observing or validating the active experiment.

Every handoff `goal` starts by saying which scientific question moved. When
the owner explicitly requested infrastructure, say that no scientific question
moved instead of disguising engineering work as research progress.

## Anti-overclaim rules

These exist because the same failure has now happened several times: a
relationship is proved or tested under one special condition, and the prose
then silently drops the condition. Example on record — `Delta_repo =
ddof_scale(T) * Delta_lw` holds at a ZERO cash leg; it was written into a
report as if it held on the real data, where it is wrong by 4-21x and has the
wrong sign in one market.

1. **Special cases are not general results.** If a result was shown only under
   a condition — cash = 0, beta = 1, one market, one seed, equal volatility,
   one synthetic example — that condition MUST appear in the conclusion.
   Before generalizing, ask: *what changes when this condition is removed?*

2. **No absolute claims without scope.** Before writing *exactly, always,
   never, identical, equivalent, guarantees, proves, impossible, by
   construction, must, cannot*, ask: *under exactly what assumptions is this
   true?* If assumptions exist, state them inside the claim.

3. **Separate three kinds of statement.** FACT — directly observed from code
   or data. DERIVATION — follows mathematically under stated assumptions.
   INTERPRETATION — what we think it means. Never present an interpretation as
   a fact, and never present a conditional derivation as unconditional.

4. **For every load-bearing claim, ask for the simplest counterexample** that
   would make it false, and test that counterexample when practical.

5. **Important tests need a negative case.** Ask: *if this relationship were
   broken in a plausible way, would this test fail?* If not, the test does not
   validate the relationship. An assertion that cannot fail is not evidence.

6. **Confidence comes from evidence, not from stronger wording.** Prefer:
   checked directly against source; derived under these assumptions; matched
   an independent implementation; passed these adversarial cases; observed
   only on this market/sample; not independently verified. Do not replace
   missing evidence with confident prose.

7. **Every important result must answer "what could make this conclusion
   wrong?"** Keep the answer concrete.

8. **Prefer a narrow correct statement over a broad impressive one.**

## 1. Authorization and orientation

Instruction precedence is: platform and safety rules, account-level
`AGENTS.md`, the owner's latest explicit request, an active branch-relevant
`TASK.md`, this file, then other documentation. Git state and actual artifacts
are authoritative; reports and handoffs are claims to verify.

At session start, run `git status --short --branch`, read the latest handoff
and current task if present, then inspect the relevant code, config, data
manifest, and existing diff. Preserve all pre-existing changes.

An approved exploratory task authorizes its complete implementation-run-debug-
verify loop. Do not pause after every milestone or commit. Ask only when a
decision would change the hypothesis, mathematical semantics, data/sample,
holdout status, public API, dependencies, external/paid state, or require a
destructive action. A prohibition in a closed `TASK.md` does not govern a new
task.

## 2. Two research lanes

### Exploratory - default

Purpose: learn quickly whether an idea deserves stronger testing.

Before inspecting its result, record a compact specification in the existing
task/config/run metadata:

- question and reason the idea might work;
- baseline, challenger, and exact mathematical difference;
- candidate values or search domain, units, and provenance;
- development sample, selection score, costs, and delay;
- each "useful regime behavior" objective as a named metric, direction, and
  comparison rule (for example, lower turnover or maximum drawdown within an
  advance-set Sharpe tolerance);
- results that would support, weaken, or falsify the idea.

There are no arbitrary line, file, commit, or runtime quotas. Use the smallest
coherent implementation, prefer configuration over a duplicate runner, reuse
the canonical pipeline when its semantics match, and state expected runtime.
Complexity is justified by the scientific question, not by a numeric code
budget.

If an exploratory implementation grows beyond roughly 300 changed lines or one
working day, pause once to tell the owner what is growing and why before
continuing. This is a warning, not a cap or a reason to split coherent work.

Exploratory work normally does not need an independent verifier, HTML report,
monitor integration, bootstrap inference, or a new governance document. Use
focused tests and simple machine-readable output. A result used to choose the
next research direction MUST retain enough evidence to inspect it: code/config
identity, data identity, candidate scores, chosen parameters, and aligned
states, signals, delayed positions, trades, costs, and returns where applicable.

Smoke and debugging runs do not need a registry entry. Add one when a run opens
or reuses an outer/holdout sample, changes future model selection, or will be
cited as research evidence. Exploratory results are preliminary findings, never
headline claims.

### Confirmatory - promotion only

Promote only a specific survivor of exploratory work. Before running, freeze
and hash the full contract in Section 6. Require immutable artifacts, a
verifier, uncertainty on paired net-return differences, honest holdout status,
a registry entry, and a generated report. At most one confirmatory study is
active at a time.

## 3. Non-negotiable scientific rules

1. **Causality.** Every quantity used at decision time `t` is computable from
   information available at or before `t`. Fit preprocessing, labels,
   state-to-position mappings, and hyperparameters on past data only. Online
   outputs are prefix-invariant: adding future observations cannot change an
   already emitted state.
2. **Execution timing.** Under the project protocol, a signal formed at the end
   of day `t` earns the return at `t+2`. Tests must make observation,
   decision, execution, and earned-return dates explicit.
3. **Frictions.** Apply the declared one-way cost, normally 10 bps, and the same
   delay in validation and evaluation. No cost-free or delay-free headline.
4. **Holdout.** The development cutoff remains 2023-12-31. No model or P&L
   experiment may use post-2023 rows without explicit authorization. A separate
   authorized source audit already inspected public candidate series through
   July 2026, so those dates are not untouched confirmation data. Once an
   outcome influences a choice, that sample is development data.
5. **Data integrity.** `data/raw/` is immutable. Never silently substitute
   synthetic, shorter, stale, price-only, total-return, or risk-free data.
   Conclusion-bearing data has source, field, cutoff, coverage, and hash
   provenance. Runtime data, artifacts, and generated reports remain ignored.
6. **Parameter provenance.** Label every result-affecting value as ESTIMATED,
   INNER_CV, THEORY, SOURCE_FIXED, PREREGISTERED, SCENARIO, NUMERICAL_GUARDRAIL,
   or UNCALIBRATED. An uncalibrated value is allowed for exploration but cannot
   support a promoted claim. If a guardrail binds, it becomes a model/protocol
   choice and must be treated as such.
7. **Mathematical identity.** Before testing a new penalty or transition rule,
   write the exact objective, units, indices, sign, and transition direction;
   show that the new parameter changes path ranking; verify the nested baseline
   and limiting cases; and add a brute-force oracle where feasible. For two
   states, always check:
   `c01*N01 + c10*N10 = 0.5*(c01+c10)*N_switch
   + 0.5*(c01-c10)*(s_T-s_0)`.
   A zero-diagonal directed switch cost is therefore a symmetric switch penalty
   plus a boundary term, not evidence of state-specific persistence. Duration
   claims need stay costs or an explicit duration/semi-Markov model and a
   synthetic recovery test.
8. **Honest claims.** Report negative and inconclusive results. Failure to beat
   a baseline does not disprove a model class. One matching number does not
   reproduce a paper. Correlated variants are not independent experiments.
   Words such as reproduced, calibrated, robust, statistically significant,
   CV-optimal, economically meaningful, alpha, tradable, and production-ready
   require the corresponding confirmatory evidence.

## 4. Verify strategy semantics before performance

For every new or changed strategy/model path, verify in this order:

1. the natural-language idea and mathematical objective agree;
2. deterministic toy cases produce the expected state path;
3. the real program executes the intended pipeline;
4. emitted state, signal, delayed position, buy/sell transition, turnover, cost,
   and earned return agree on sampled dates;
5. behavior remains causal under prefix and future-data tests;
6. only then interpret Sharpe, drawdown, or another performance metric.

Tests generated by an AI and an LLM judge can help, but neither is sole evidence
of financial semantics. Inspect concrete timelines and trades. A scientific
gate may lock aggregate P&L or claims, but MUST NOT hide diagnostic inputs,
features, states, selected parameters, signals, positions, or trades needed to
check what the model actually did.

Keep three states separate:

- **CODE_COMPLETE:** implementation tests pass;
- **EXPERIMENT_COMPLETE:** the declared run finished with complete artifacts;
- **CLAIM_READY:** uncertainty, provenance, holdout, and claim gates pass.

## 5. Parameter search and boundaries

Inspect objective scale and units on development data before choosing a search
range. Coarse-to-fine or parallel search is allowed, but every adaptively
inspected sample is development data and every evaluated candidate is retained.
Freeze the final search domain before confirmatory evaluation.

Always-risky and always-cash are explicit comparator strategies, not two values
of `lambda` or HMM smoothing `k`. Sending a switching penalty toward infinity
discourages switching; which constant path wins still depends on data loss,
initial conditions, and boundary conventions. Never label that limit as both
constant comparators.

If exploratory selection concentrates on the largest finite candidate, expand
the range at most once for that baseline. Persistent concentration is a valid
finding that the objective is monotone or flat over the tested range; report it
and move on. Do not build another calibration framework merely to make a
boundary rule pass. Boundary concentration blocks a claim of a finite optimum,
not all exploratory comparison.

## 6. Confirmatory contract

Freeze and content-hash: experiment ID and hypothesis; primary estimand and
metrics; mathematical objective; data manifest and sample boundaries; train,
validation, and outer split; refit cadence; features and preprocessing; complete
search domains, tie-breaks, seeds, and stopping rules; selection objective;
costs, delay, risk-free and accounting rules; uncertainty and multiplicity
method; success/failure/inconclusive criteria; expected runtime and artifacts.

Compare complete selection pipelines on identical dates, information sets, and
frictions. A post-result change to a result-affecting choice creates a new
experiment ID and makes the viewed sample development data.

## 7. Code, tests, artifacts, and handoff

- Prefer clear direct code. Line count is not a quality metric. Core research
  logic belongs in `src/adaptive_jump/`; CLI and experiment runners stay thin.
- Add an abstraction only when it removes real duplication or clarifies a
  scientific responsibility. Do not create parallel model stacks.
- New dependencies, public API changes, data downloads, and remote services
  require approval.
- Use focused tests for exploratory changes. Run the full relevant suite before
  a confirmatory run, before releasing a claim/demo, and when shared core
  behavior changes. UI changes additionally require real browser acceptance.
- Generated outputs stay untracked. Sealed artifacts are immutable and reports
  are generated from full-precision machine-readable evidence.
- Run `git status` before and after work. Stage only in-scope files. Follow the
  owner's current commit/push permission and never add AI attribution.
- Commit messages use the established prefix taxonomy:
  `research:` (experiments, specs, results, registry), `model:` (model/math
  code), `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`, `build:`,
  optionally scoped like `fix(monitor):`. Imperative mood, lower-case prefix,
  one concern per commit. Example: `research: freeze expanding-standardizer
  replication spec`.
- Append one handoff after a meaningful repository change or scientific or
  expensive run using:
  `bash .agent/handoff.sh '<one-line-json-entry>'`
  Never rewrite earlier entries or fabricate verification. Handoff files
  normally remain dirty after the code commit and must be disclosed.

## 8. Forbidden shortcuts

- ceremony completed means research progressed;
- infrastructure completed means an experiment succeeded;
- another grid expansion means the boundary issue is solved;
- passing tests means the hypothesis is validated;
- a high Sharpe proves the strategy logic is correct;
- a hand-entered threshold is calibrated;
- equal dates imply equal model-selection budgets;
- zero-diagonal directed costs imply duration asymmetry;
- repeatedly viewed OOS data is an untouched holdout;
- rounded CSV, HTML, or PNG without provenance is reproducible evidence;
- a partial, reduced, stale, or incompatible run is a full result;
- one failed implementation disproves the broader model class.
