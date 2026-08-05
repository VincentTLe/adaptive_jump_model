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

## Current Phase: Health Gates

Completed:

- archived and hash-verified the pre-cleanup workspace and full Git history;
- reduced live artifacts to the four replay dependencies plus small audit
  evidence;
- drafted the contract: one challenger, four baseline specifications, data
  roles, pass/fail rules, and a no-retry confirmation rule;
- confirmed existing beta-zero nesting, DP/brute-force parity, shared
  preprocessing, and future-mutation prefix tests (`55 passed`);
- added a canonical fail-fast gate for a decrease in the fitted JM objective as
  lambda increases — impossible at the global optimum, so a firing certifies a
  suboptimal local fit. The first version compared adjacent pairs only, letting
  sub-tolerance decreases accumulate across the grid; the independent verifier
  caught this and the gate now compares against the running maximum
  (`35 model tests passed`);
- a separate agent that did not write the gate fault-injected it: CERTIFIED,
  `docs/audit/2026-08-05-objective-gate-fault-injection.md`;
- deleted the 26 source modules and 13 test files of the closed studies
  (balanced, confidence, separation, holdout runner, lagged study machinery),
  keeping the challenger's dependencies. Parity proof: all three sealed runs
  verify bit-identically before and after (max metric difference <= 3.8e-14),
  and the remaining suite is green. Deleted files stay recoverable from git
  history and the 2026-08-05 cold archive; `RUNNABLE_SPECS` was trimmed to the
  two specs the code can still execute.

- registered the freeze (2026-08-05T23:20:47Z): the contract is committed and a
  registry event pins `frozen_spec_hash`
  `e331b96d662ca703a3fa5140d4ba9d92544c9eed553eb36df1525fafbc0b6a49`; the
  2023-12-31 cutoff was kept by owner decision, recorded in the contract.

Still required before any new P&L is interpreted:

1. The external runner must replay states -> monthly selection -> trades ->
   metrics, with an independently produced verification receipt.
2. Official public regional data metadata and hashes must be frozen. Downloading
   it requires owner approval.

## Data Roles

- `D0 development`: every existing US/DE/JP proxy, Shu table value, grid search,
  and artifact. Burned; useful only for diagnosis.
- `D1 transport`: Fama-French North America, Europe, and Japan. Used once to
  decide whether the frozen challenger is transportable.
- `D2 confirmation`: Fama-French Asia-Pacific ex Japan. Unopened until D1 passes
  the frozen gate; one opening, no tuning and retrying.

## Stop Rule

If D1 fails, stop AJM-EXT-001. If D1 passes but D2 is negative or inconclusive,
stop AJM-EXT-001. A new feature, beta, grid, or decision rule requires a new
experiment ID and cannot reuse D2 as a holdout.

## Not Active

Exact Shu-v3 grid hunting, calibrated-v10 agreement, the invalidated
`return_aware`/`robust_l1` artifacts, and the frequency-ladder result are not
active research paths.
