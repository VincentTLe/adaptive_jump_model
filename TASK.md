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
−0.162, North America −0.176. The challenger beats its paired fixed JM on 7 of
12 spec-region cells, but every region has at least one paired specification
that degrades by ~0.17 Sharpe — always the short Table-3 grid, where
discounting the switching penalty pushes selection toward whipsaw. Guardrail
clean everywhere.

Consequences, per the contract: the confirmation region (Fama-French
Asia-Pacific ex Japan) was never opened, stays sealed, and is burned for
successor experiments. `lagged_evidence_log4` is a development-supported
mechanism that did not transport.

## Next

No active task. Any new candidate (capped gap, two-day confirmation,
semi-Markov dwell cost, or a grid-robust variant of the lagged mechanism
motivated by the short-grid failure mode) needs its own frozen question and a
new experiment id before any code runs.
