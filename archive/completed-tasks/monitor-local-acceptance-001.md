# Task: Local Monitor Operational Acceptance

## Identity

- `task_id`: `monitor-local-acceptance-001`
- `status`: `LOCAL_OPERATIONAL_ACCEPTED`
- `target_branch`: `cleanup/research-protocol`
- `starting_ref`: `94b86312d9b528f08ea0d0cb7febe928f5432270`
- `parent_task`: `live-research-monitor-001`
- `primary_class`: `ENGINEERING / SMOKE`
- `scientific_claims`: forbidden
- `data_downloads`: forbidden
- `permitted_data_access`: existing frozen proxy inputs through 2023 only
- `post_2023_access`: forbidden
- `adaptive_experiment`: forbidden
- `cloudflare_acceptance`: out of scope

The owner approved this task on 2026-07-14. It tests whether the local monitor
can control one real canonical subprocess and preserve its evidence. It does not
reopen the completed v7 scientific experiment or create new market evidence.

## Objective

Establish or reject **local operational acceptance** for the monitor. A passing
result must show, through the real loopback server and canonical CLI subprocess,
that the owner can enqueue, observe, cancel, interrupt, restart, resume, replay,
and finish one frozen engineering replay whose artifact passes the canonical
verifier and whose deterministic research outputs match the direct-CLI
reference.

This task does not establish remote operational acceptance. Cloudflare Tunnel
and Access still require the owner's real hostname, account, audience, tunnel
credential, and exact owner/viewer email addresses.

## Observed Outcome

Local operational acceptance completed on 2026-07-14. This was an engineering
replay of already observed through-2023 inputs, not a scientific experiment.

- Chromium enqueued and canceled job `686af0348a5a424c810ad35f51171699`.
  Its atomic US-HMM checkpoint remained valid at 850 completed fits.
- Chromium enqueued job `cb450ec8d00d4cae9bb8df7f150b66b2`, which resumed
  at fit 851, survived a monitor shutdown at fit 2,250, and resumed as attempt
  2 without restarting the completed prefix.
- The append-only journal ended with 43,887 unique contiguous events.
  `artifact_verified` at sequence 43,886 preceded the successful
  `process_finished` event at sequence 43,887.
- The independent verifier accepted 125 inventoried files, 18 boundary rows,
  and 27 metric rows with maximum recomputation difference
  `7.327471962526033e-15`.
- All 125 non-checkpoint path/hash entries matched the frozen direct-CLI
  reference. The reference's six Git-bound checkpoint files were the only
  excluded entries.
- Production Chromium passed Live, Queue, Replay, Compare, and Evidence at
  1440x900 and 390x844, plus a JavaScript-disabled fallback. One mobile Replay
  overflow found during acceptance was fixed in commit `7fe5eb6` and the full
  browser matrix then passed with no page errors or horizontal overflow.

The replay reproduced the existing proxy non-replication exactly; it did not
create a new scientific result. Real Cloudflare Tunnel, Access policy, OTP,
owner/viewer routing, and ordinary-browser remote access remain unaccepted.

## Frozen Replay

- Queue study ID: `monitor-local-acceptance-001`.
- Canonical CLI study: `replication`.
- Inputs: the existing hash-checked proxy manifest and canonical files capped at
  2023-12-31 by `research.toml`.
- No provider request or data acquisition command may run.
- Expected runtime: approximately 60-100 minutes on the current host, including
  controlled cancellation and restart overhead.
- The replay may calculate already observed v7 states, trades, and metrics only
  to test engineering parity. They must not be interpreted as new evidence.

The direct-CLI reference is the sealed run:

`artifacts/fixed-baselines/fixed-baselines-8adb330565d6-3636939b525d-e9614112b234`

Its frozen `inventory.json` SHA-256 is:

`08e42044c25ef80e92f4b565034652a6b87fe94de0e9122eee1a418395239d55`

The new run ID and mutable `run.json` will differ because they include the new
Git revision and lifecycle timestamps. Each market's `checkpoint.json` and
`checkpoint.pkl` will also differ because checkpoint identity is bound to that
Git revision. Excluding only those six checkpoint files, all inventory paths
and SHA-256 values must match the direct-CLI reference. Any other mismatch fails
acceptance and requires attribution; it must not be repaired by changing the
model, data, protocol, metric, gate, or reference artifact.

## Verified Gap To Repair First

At the starting ref, `ResearchWorker` marks a child `succeeded` when it exits
with code zero, while `adaptive-jump run` does not invoke the canonical verifier
before returning zero. The Evidence view verifies only its separately registered
sealed artifacts. Therefore a new job is not yet canonically verified merely
because the queue says `succeeded`.

Before the replay:

1. `adaptive-jump run` MUST call the canonical verifier on its returned artifact
   before printing the path or returning zero.
2. Verification failure MUST return a nonzero CLI status and MUST NOT emit an
   `artifact_verified` event.
3. Verification success MUST emit one decision-visible `artifact_verified`
   event containing only run identity and terminal status, never metrics,
   returns, wealth, Sharpe, bootstrap output, or conclusion text.
4. The worker may mark the job `succeeded` only after that verified CLI process
   exits zero.

## Local Authentication Harness

The real `adaptive-jump monitor --config research.toml` command, production API,
SQLite queue, event journal, worker, event pipe, and canonical child command must
be used. Dependency-injected preview services do not satisfy this task.

Because real Cloudflare credentials are unavailable, local acceptance MAY use
an ephemeral HTTPS JWKS endpoint and a short-lived RS256 owner assertion. The
production origin must validate algorithm, key, issuer, audience, expiry, and
exact owner email. Playwright may attach the assertion header to loopback
requests. Keys, certificates, tokens, and CSRF values must stay under ignored
runtime or temporary storage and must not be committed.

This harness validates the origin authentication path but is not evidence that
Cloudflare Tunnel, Access policy, OTP, owner/viewer routing, or remote networking
works.

## Acceptance Sequence

1. Confirm clean Git identity, reference artifact verification, data hashes, and
   an empty production queue catalog before the new FROZEN registry row.
2. Start the real monitor on `127.0.0.1` with the ephemeral local issuer. Confirm
   no non-loopback listener is created.
3. In real Chromium, authenticate as the configured owner and enqueue only
   `monitor-local-acceptance-001`.
4. Wait for an identity-bound HMM checkpoint and visible model/resource events.
   Cancel the active job through the UI and confirm the complete process group
   exits while the last atomic checkpoint remains readable.
5. Enqueue the same study again. Confirm it reuses the exact checkpoint identity
   and advances beyond the saved work rather than starting from zero.
6. Stop the monitor during active work. The job must become `interrupted`, no
   child may remain alive, and queue/event history must survive restart.
7. Restart the real monitor, resume the interrupted job through the UI, reconnect
   SSE, and confirm event sequence remains monotonic without duplicates.
8. Let the replay finish. The journal must contain `artifact_verified` before
   `process_finished`, and the queue may then become `succeeded` with exit code
   zero.
9. Run the canonical verifier independently on the new artifact. Excluding the
   six identity-bound checkpoint files, its inventory paths and SHA-256 values
   must match the frozen direct-CLI reference.
10. Exercise replay at desktop 1440x900 and mobile 390x844. Require no page
    errors, incoherent overlap, horizontal overflow, blank primary chart, or
    unauthorized outcome event.
11. Confirm runtime state remains ignored, no process is orphaned, and no
    generated artifact, token, certificate, SQLite file, log, PID, or event
    journal is tracked.

## Outcome Rules

- `LOCAL_OPERATIONAL_ACCEPTED`: every acceptance item passes and the canonical
  artifact matches every non-checkpoint reference inventory entry exactly.
- `LOCAL_OPERATIONAL_FAILED`: any required lifecycle, security, verifier,
  parity, browser, or process-ownership check fails.
- `INCONCLUSIVE`: an external interruption prevents completion before a
  pass/fail observation. Resume may continue only under the unchanged task and
  exact code/data identity.

No outcome may be described as a replication result, model improvement,
scientific confirmation, remote acceptance, Cloudflare acceptance, production
trading readiness, or evidence for an adaptive model.

Bootstrap remains code-tested rather than production-observed because the
completed JM-4000 study correctly stops at its boundary gate before bootstrap.
This task must not bypass that gate merely to exercise a dashboard stage.

## Write Boundary

Authorized tracked paths are:

- `TASK.md` and `archive/completed-tasks/live-research-monitor-001.md`;
- append-only `research/experiment_registry.jsonl`;
- `README.md` and authored English files under `docs/monitor/`;
- `src/adaptive_jump/cli.py` and focused monitor queue/event code only where
  required by this contract;
- focused tests under `tests/`;
- the two procedural `.agent/` handoff files.

One small acceptance helper may be added under `scripts/` only if manual shell
orchestration cannot preserve exact, reviewable evidence. It may not implement
research logic or an authentication bypass.

Do not modify dependencies, `research.toml`, frozen study TOML, market data,
sealed artifacts, reports, learning textbook, archive provenance other than the
named task copy, model mathematics, features, grids, seeds, state semantics,
costs, delays, metrics, gates, or scientific claims.

Generated replay output and operational telemetry remain ignored and must not
be committed.

## Commit And Verification Discipline

1. Freeze this contract and registry row in one documentation-only commit.
2. Repair the canonical verification contract and register the smoke study in
   one reviewed code commit of at most about 400 changed lines and 15 files.
3. Run the acceptance sequence without changing code or frozen inputs.
4. Record the exact pass, failure, or inconclusive result in a final small
   documentation and append-only registry commit.

Run focused tests first, then full pytest, Ruff, lock/package checks, clean
archive installation, paper hash, archive immutability, protected-path checks,
and both frozen-run verifiers. The real replay starts only after all pre-run
checks pass. Stop after each commit for owner review.
