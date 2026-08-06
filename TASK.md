# Active Task: AJM-EXT-001

## Goal

Evaluate one frozen adaptive Jump Model against a bounded family of Shu-style
fixed-JM baselines on public data that was not used to match Shu's tables.
The goal is an external replication/extension, not an exact reconstruction of
Shu's unpublished implementation choices.

## Frozen Question

Does `lagged_evidence_log4` beat every paired fixed-JM specification and the
stronger of buy-and-hold and HMM on independent regional data, after the same
`t+2` execution timing and 10-bps one-way cost?

Contract: `research/ajm-ext-001.toml`.

## Outcome: NOT SUPPORTED — experiment complete (2026-08-06)

The transport gate failed 0/3 and AJM-EXT-001 ends under its frozen stopping
rule. Run `ajm-ext-e331b96d662c-e524087d0978` was independently verified and
CERTIFIED before the verdict was first read
(`docs/audit/2026-08-06-ajm-ext-d1-receipt.md`): inventory, frames, trades,
metrics, beta-zero nesting, a partial replay, and the gate arithmetic all
reproduce exactly.

Estimand (min over the four paired specs, delay 1): Europe −0.167, Japan
−0.162, North America −0.176. Corrected reading (registry CORRECTION,
2026-08-06, after an owner objection and a 9-agent re-audit): the challenger
beats its paired fixed JM on **8 of 12** cells and the mean-over-specs delta is
~0.00 (EU +0.035, NA +0.026, JP −0.037) — the −0.17 headlines are ~68% the
min-over-four-specs construction, ~31% transport decay, ~1% era. Paired-delta
CIs are ±0.25–0.33 at this n, so the honest label is spec-fragility of a
statistically weak effect. The era objection was quantified and refuted (dev
deltas hold on the FF-overlap era: +0.091 → +0.088), but the 2010+/no-GFC OOS
window was an undisclosed design consequence, now recorded. The whipsaw story
holds at the JP (4.8×) and EU (2.9×) binding cells but not at NA's (1.4×,
fixed never at the cap); "guardrail clean" holds under the implemented
every-spec reading only.

Consequences, per the contract: the confirmation region (Fama-French
Asia-Pacific ex Japan) was never opened, stays sealed, and is burned for
successor experiments. `lagged_evidence_log4` is a development-supported
mechanism that did not transport.

## Next

No active task. Any new candidate (capped gap, two-day confirmation,
semi-Markov dwell cost, or a grid-robust variant of the lagged mechanism
motivated by the short-grid failure mode) needs its own frozen question and a
new experiment id before any code runs.
