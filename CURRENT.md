# Current Research State

Last updated 2026-08-09. This is the human-facing summary of where the project
stands. Read it on its own — you should not need `research/TASK.md`, the
registry, or any audit receipt to understand it.

## 1. Question

Can we improve the Shu-style Statistical Jump Model for causal equity/cash
market timing?

A Jump Model reads daily market features and decides which of two regimes the
market is in. The strategy holds equities in the favorable regime and cash in
the unfavorable one. *Causal* means every decision uses only information
available on or before that day. We test on US, German, and Japanese index
proxies under one shared protocol: same features, same refit schedule, a one-day
execution delay, and a 10 bps one-way trading cost.

The open question is whether a structural change to the model beats the plain
fixed Jump Model under that identical protocol. It is not a claim that we have
found such an improvement.

## 2. Baseline

The comparator is the **sealed v11-ninit60 fixed Jump Model baseline**. *Sealed*
means its configuration and fitted results are frozen and stored, so every
challenger is compared against exactly the same numbers rather than a freshly
refit target.

A challenger is judged by the *paired difference* in each market: the
challenger's Sharpe minus the fixed JM's Sharpe under identical conditions.
Beating buy-and-hold or the HMM while losing to the fixed JM is not evidence
that the JM was improved.

**Known defect, disclosed:** the German leg of this baseline is not clean. Its
selected grid fails the admissibility rule that was supposed to select it, and
some German fits are not fully converged at the sealed optimizer setting. It
remains the working comparator because everything downstream is already measured
against it, and swapping comparators mid-study would make past and future
results incomparable.

This baseline is not globally optimal, not mathematically final, and not
permanent. No replacement exists, and none is being built here.

## 3. What We Know

- **The Shu-style fixed JM replication is good enough to serve as the main
  comparator.** The US leg reproduces the published behavior closely; the German
  and Japanese legs are bounded with documented causes and are not claimed as
  exact reproductions.
- **The optimizer can land in different local solutions.** Different random
  starting points sometimes give different fitted parameters and a few different
  daily states. This is a property of the method, not a bug — and it means a
  small difference between two arms can reflect optimizer sensitivity rather
  than the change being tested.
- **So challengers are judged by the paired challenger-minus-fixed-JM
  difference**, measured directly under identical conditions rather than
  inferred from each arm separately, because movement common to both arms can
  cancel in the difference. In the one place this was measured — the
  optimizer-fidelity diagnostic on confirmed_2d — the paired deltas were indeed
  much tighter than the fixed JM's own raw Sharpe spread. That is an observed
  result for that variant on that sample, not a guarantee that paired deltas
  will be tighter for future challengers.
- **Generic duration-dependent state regularization is not claimed as novel.**
  The published Deep Statistical Jump Model paper already notes its state-loss
  framework can penalize staying in one regime too long, and duration dependence
  in regime models is decades old. Any novelty claim here must be narrower.

## 4. What Failed

Four directions are closed. None is reopened, and none is rescued by later
findings.

- **Lagged-evidence adaptive lambda.** The switching penalty was made to depend
  on the previous day's evidence. It lost to the paired fixed JM in all three
  transport regions and was stopped by its own pre-registered rule. NOT
  SUPPORTED.
- **Lagged cap guard.** An attempt to repair the above by falling back to the
  fixed path in months where the fixed model's selection hit the top of its
  grid. It made the worst case worse, not better. NOT SUPPORTED.
- **confirmed_2d.** A variant requiring two consecutive days of agreement before
  switching, and the only variant with cross-market support under the study's
  own rule — so it was examined episode by episode. Its whole aggregate
  advantage turned out to be a small residual of two large offsetting tails,
  concentrated in one to five single days, and partly a transaction-cost timing
  artifact. NOT SUPPORTED as a mechanism.
- **v12 stress gate.** A proposed re-sealed German baseline. Its convergence
  gate, frozen before the run, required both objectives and states to match a
  reference; one of 255 objectives did not. v12 is permanently stopped. Later
  evidence that its state paths were in fact identical does not revive it —
  rescuing a failed experiment after seeing the result is exactly what a frozen
  gate exists to prevent.

One framing was also retracted: an earlier write-up treated a bootstrap
confidence interval as a minimum effect size that future models "must clear". It
is not. It estimates sampling uncertainty under one particular resampling
design, and it is reported last and descriptively, never as a gate.

## 5. Active Idea

**Duration-Aware Jump Model (DA-JM).**

The standard Jump Model knows *which* regime the market is in. DA-JM asks
whether *how long the regime has already lasted* should also affect how easily
the model switches out of it — a two-day-old calm regime and a two-year-old calm
regime may not deserve the same resistance to switching.

Status, explicitly:

- Theory and proposal only.
- **Not implemented.** Nothing in the source tree computes a duration-aware
  objective.
- The parameterization is **not frozen**; the open design questions have
  proposals but no owner decision.
- **No DA-JM experiment has been run.** No experiment ID, no result.
- **Novelty is not settled.** A literature search cannot prove non-existence,
  and the generic form of this idea is already in print (Section 3).

## 6. Next Step

One active step: **an owner design review of DA-JM, before any implementation.**

The review must put the design in front of the owner in plain terms — what
single thing changes relative to the fixed JM, which new parameters that
introduces and where their values come from, what stays identical, and which
outcomes would support the idea versus stop it. Choosing the DA-JM parameters
belongs to that review, not to the agent preparing it. Implementation starts
only after the owner approves.

## 7. Not Doing

- **No v13** and no replacement baseline.
- **No rescue tuning** of a failed experiment after its results are known.
- **No new universal bootstrap or noise-floor gate** — resampling evidence stays
  descriptive and last.
- **No DA-JM implementation before owner approval.**
- **No broad infrastructure expansion**, dashboards, or new governance documents.

---

If you need to check any of the above: `research/TASK.md` holds the detailed
state, `research/experiment_registry.jsonl` the append-only experiment history,
`research/SCIENTIFIC_LEDGER.md` the evidence and corrections, and `docs/audit/`
the verification receipts.
