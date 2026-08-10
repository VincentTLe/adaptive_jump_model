# AGENTS.md — adaptive_jump_model

Durable behavior rules for AI agents in this repository. Account-level
instructions also apply. **Current research state lives in `CURRENT.md`, not
here.** History lives in `research/SCIENTIFIC_LEDGER.md` and the registry.

**MUST** / **MUST NOT** are non-negotiable. **SHOULD** needs a recorded reason
to deviate.

## 0. What counts as progress

**The unit of progress is a scientific question answered** — not infrastructure,
documents, or dashboards. Infrastructure and presentation work happen only when
explicitly requested, or when they are the smallest blocker to the active
experiment. **Complexity is a research cost:** use the smallest coherent
implementation, prefer configuration over a duplicate runner, and build no
speculative features, parallel model stacks, or new governance frameworks.

The question this repository studies:

> Can we improve the Shu-style Statistical Jump Model for causal equity/cash
> market timing?

**The primary estimand for a structural challenger is the paired delta against
the fixed JM**: `Delta_m(v) = Sharpe_v,m − Sharpe_fixedJM,m` for market `m` and
prespecified variant `v`, measured under identical conditions. The fixed JM is
the arm an extension claims to improve, so it is the arm the extension must
beat. Beating HMM and buy-and-hold while losing to the fixed JM is **not**
evidence that an extension improved the JM. Before calling a variant
economically useful, also require
`Sharpe_v,m > max(Sharpe_fixedJM,m, Sharpe_HMM,m, Sharpe_BuyHold,m)`.

The canonical baseline is whichever sealed config `CURRENT.md` and the registry
record as canonical, with its known defects disclosed. A sealed baseline is a
fixed comparator, not permanent proof.

## OWNER CHECK

Before implementing any new model or experiment, first present:

    Question
    Baseline
    One thing being changed
    Why it may help
    New parameters
    What stays identical
    What result would support the idea
    What result would make us stop

Before a large implementation or a new mathematical mechanism, explain the
design first and obtain owner approval. The owner does not need every
implementation detail, but **MUST understand every result-affecting assumption,
parameter choice, and scientific claim.**

## 1. Honest claiming

**AI confidence is not evidence.** Assured prose is not a substitute for a check
that was actually run. If the check was not run, say so.

1. **Special cases are not general results.** A condition the result was shown
   under — one market, one seed, `beta = 1`, zero cash — MUST appear in the
   conclusion.
2. **No absolute claim without scope.** Before *exactly, always, never,
   identical, equivalent, guarantees, proves, impossible, must, cannot*, put the
   governing assumptions inside the claim.
3. **Separate FACT (observed from code or data) / DERIVATION (follows under
   stated assumptions) / INTERPRETATION (what we think it means).** Never
   present an interpretation as a fact or a conditional derivation as
   unconditional.
4. **Important tests need a plausible negative case.** If the relationship were
   broken realistically, would this test fail? An assertion that cannot fail is
   not evidence. Name the simplest counterexample for each load-bearing claim
   and test it when practical.
5. **Confidence comes from evidence, not stronger wording.** Say which: checked
   against source; derived under these assumptions; matched an independent
   implementation; observed only on this sample; not independently verified.
6. **Prefer a narrow correct claim over a broad impressive one.** Every
   important result answers "what could make this conclusion wrong?".
7. A literature search shows what was not found; it cannot prove non-existence.

## 2. Non-negotiable scientific rules

1. **Causality / no future-data leakage.** Every quantity used at decision time
   `t` is computable from information at or before `t` — including
   preprocessing, labels, state-to-position mappings, and hyperparameters.
   Online outputs are prefix-invariant: adding future observations cannot change
   an already emitted state.
2. **Execution timing.** A signal formed at the end of day `t` earns the return
   at `t+2`. Tests make observation, decision, execution, and return dates
   explicit.
3. **Frictions.** Apply the declared one-way cost (normally 10 bps) and the same
   delay in validation and evaluation. No cost-free or delay-free headline.
4. **Holdout.** Development cutoff 2023-12-31. **No model or P&L experiment may
   use post-2023 rows without explicit owner authorization.** An authorized
   source audit already inspected public candidate series through July 2026, so
   those dates are not untouched confirmation data. Once an outcome influences a
   choice, that sample is development data.
5. **Data integrity.** `data/raw/` is immutable; never silently substitute a
   different series. Conclusion-bearing data carries source, field, cutoff,
   coverage, and hash provenance.
6. **Parameter provenance.** Label every result-affecting value ESTIMATED,
   INNER_CV, THEORY, SOURCE_FIXED, PREREGISTERED, SCENARIO,
   NUMERICAL_GUARDRAIL, or UNCALIBRATED. An uncalibrated value may explore but
   cannot support a promoted claim.
7. **Mathematical identity first.** Before testing a new penalty or transition
   rule, write the exact objective, units, indices, sign, and transition
   direction; show the new parameter changes path ranking; verify the nested
   baseline and limiting cases; add a brute-force oracle where feasible. A
   zero-diagonal directed switch cost is a symmetric switch penalty plus a
   boundary term, not evidence of state-specific persistence.
8. **Never search over a knob the source paper leaves unspecified for the value
   that best matches the paper's numbers** — that is fitting to the answer.
   Choose a priori, report the spread across alternatives as a limitation, and
   keep `docs/unspecified-choices.md` current.
9. **Report negative and inconclusive results.** One matching number does not
   reproduce a paper; correlated variants are not independent experiments.

## 3. Experiment discipline

- **One experiment = one primary scientific idea.** Do not bundle mechanisms and
  then attribute the outcome to whichever one is convenient.
- **Do not silently change a frozen experiment protocol.** Any post-result change
  to a result-affecting choice creates a new experiment ID, and the viewed
  sample becomes development data.
- **Do not rescue a failed experiment through post-result tuning.** A gate frozen
  before a run stands after it, including when a later finding suggests the gate
  was too strict. Record the corrected reading; do not revive the result.
- Keep **CODE_COMPLETE** (tests pass), **EXPERIMENT_COMPLETE** (declared run
  finished with complete artifacts), and **CLAIM_READY** (uncertainty,
  provenance, holdout gates pass) separate.
- Exploratory is the default lane and needs a compact pre-registered spec:
  question, baseline, exact mathematical difference, search domain, sample,
  costs, delay, and what would falsify it. Confirmatory promotion additionally
  requires a frozen hashed contract, immutable artifacts, an independent
  verifier, and a registry entry; at most one is active at a time.
- **A result must be checkable by someone who did not write the code.** Retain
  code/config identity, data identity, candidate scores, chosen parameters, and
  the aligned states, signals, positions, trades, costs, and returns.
- **Verify semantics before performance**, in this order: idea and objective
  agree → toy cases give the expected path → the real program runs the intended
  pipeline → state, signal, delayed position, turnover, cost, and return agree
  on sampled dates → causality holds under prefix tests → only then interpret
  Sharpe. AI-written tests are not by themselves evidence of correct financial
  semantics.

## 4. Project language: English only

All persistent project material is English: `README.md`, `CURRENT.md`,
`TASK.md`, this file, `CLAUDE.md`, everything under `docs/` and `research/`, the
manuscript, specs, receipts, the ledger, registry entries, generated reports,
comments, docstrings, test names, CLI messages, commit messages, and
agent-written PR and issue text. Do not translate third-party sources, quoted
paper text, raw data, proper names, or literal strings whose original form is
provenance. Do not mass-translate historical files, and never edit an
append-only registry entry over language — append a correction.

## 5. Working rules

- Precedence: platform and safety rules, account-level `AGENTS.md`, the owner's
  latest explicit request, this file, then other documentation. Git state and
  artifacts are authoritative; reports and handoffs are claims to verify.
- At session start: `git status --short --branch`, read `CURRENT.md` and the
  latest handoff, inspect the relevant code and diff. Preserve pre-existing
  changes.
- An approved task authorizes its full implement-run-debug-verify loop; do not
  pause after every milestone. Ask when a decision would change the hypothesis,
  mathematical semantics, data or sample, holdout status, public API,
  dependencies, or external state, or requires a destructive action.
- **Subagents are not default behavior.** Spawn one when the owner asks, or when
  a result needs an independent check by an agent that did not write the code.
- New dependencies, public API changes, data downloads, and remote services
  require approval.
- Core research logic in `src/adaptive_jump/`; CLI and runners stay thin.
  Generated outputs stay untracked; sealed artifacts are immutable.
- Stage only in-scope files. **Never add AI attribution to commits or PRs.**
  Commit prefixes `research:`, `model:`, `feat:`, `fix:`, `docs:`, `test:`,
  `refactor:`, `chore:`, `build:` — imperative mood, one concern per commit.
- Append one handoff after a meaningful change or expensive run:
  `bash .agent/handoff.sh '<one-line-json-entry>'`. Never rewrite earlier
  entries or fabricate verification.
