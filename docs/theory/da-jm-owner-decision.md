# DA-JM owner decision: one causal experiment

**Status:** Proposal for owner decision. This document does not authorize implementation, a P&L run, a new data download, or holdout access.

## Decision requested

Approve, revise, or reject the experiment below before any model code is written.

- [ ] Approve as written.
- [ ] Revise the choices identified in an owner comment.
- [ ] Reject and stop DA-JM work.

## Owner check

**Question**

Can an age-dependent jump penalty improve out-of-sample regime detection relative to a fixed-penalty jump model, without paying for fewer short episodes with mechanically longer detection delays?

**Baseline**

Use the fixed JM with one shared, source-based lambda grid for every market:

`[10, 22, 50, 100, 220, 500, 1000]`

Use `n_init = 60`. The grid was disclosed in Shu et al.'s withdrawn arXiv v1 and is therefore an a-priori source choice; it is not claimed to be the authors' current production grid. The calibrated v11 grids may appear only as a labeled sensitivity analysis because they were recovered market by market from evaluation-period outputs.

**One thing being changed**

Replace the constant excess transition cost with an age-dependent excess cost. Solve the resulting problem exactly with an augmented `(state, age)` dynamic program. All signal construction, fitting, selection, execution, and evaluation choices remain shared between the baseline and challenger.

**Why it may help**

A constant jump penalty uses the same evidence threshold for a one-day reversal and for leaving a persistent regime. An age-dependent cost can express the narrower hypothesis that very short episodes deserve more resistance while established regimes should not receive the same extra protection.

**New parameters**

- Duration exponent `beta`: `2.0` is the primary hypothesis, `1.0` is the identity check, and `0.5` is an adversarial direction.
- Maximum tracked age `D_max = 504` trading days.
- Duration anchors are re-estimated only at the existing January/July refits.

No other new tuning parameter is permitted. These three beta values and `D_max` are frozen before external confirmation.

**What stays identical**

- Same market data, features, standardization, state count, and signal timing.
- Same 3,000-day rolling fit window and January/July refit schedule.
- Same monthly lambda-selection rule and `t+2` primary execution delay.
- Same 10 bps transaction cost and evaluation metrics.
- Same lambda candidate is used by both arms within every paired comparison.
- No post-2023 model result or P&L without separate owner authorization.

Duration anchors must be causal. At each refit, estimate them only from uncensored segments in the training window available at that date, then freeze them until the next refit. Never estimate anchors from the full 1990--2023 selected path and feed them back into earlier decisions.

**What result would support the hypothesis**

After all mechanism gates pass and the complete specification is frozen, `beta = 2.0` must improve paired net Sharpe over the shared fixed-JM baseline on untouched external markets. The improvement must be economically non-trivial, present under the declared delay/cost checks, consistent with the intended reduction in short false episodes, and not driven by one market or a few episodes.

**What result would stop the hypothesis**

Stop without rescue tuning if any identity or exactness gate fails; if `beta = 2.0` does not transport to untouched markets; if its gain is concentrated in one market or a few episodes; or if only the adversarial `beta = 0.5` direction wins. A win by `beta = 0.5` rejects this primary hypothesis and may motivate a separately approved future question.

## Required mechanism gates before any P&L comparison

1. `beta = 1.0` reproduces the fixed-JM fitted state path bit for bit.
2. The augmented dynamic program agrees with brute-force enumeration on small cases.
3. Flat loss produces no arbitrary churn.
4. Synthetic tests separate the intended cases: geometric durations, Weibull-like duration dependence, and an out-of-family process.

Failure at any gate stops the experiment and is treated as an implementation or identification failure, not an invitation to tune.

## Development and confirmation boundary

US, Germany, and Japan through 2023 may be used only for implementation checks and exploratory mechanism plots because they have already influenced the project. They cannot provide the confirmatory result.

Before acquiring or using any candidate external market, freeze the code, this specification, the primary metric, and the stopping rule. Candidate confirmation markets are the UK, Canada, and Australia, subject to a separate owner decision on data acquisition and provenance. Results from those markets are read once for the declared confirmation.

## Smallest next action after approval

Implement only the augmented dynamic program and the four mechanism gates. Return the gate evidence to the owner before requesting authorization for any market-level P&L run.
