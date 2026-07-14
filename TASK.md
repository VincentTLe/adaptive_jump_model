# Task: Live Research Monitor

## Identity

- `task_id`: `live-research-monitor-001`
- `status`: `active`
- `target_branch`: `cleanup/research-protocol`
- `starting_ref`: `d525c0d5f979a970e21735e31b9bb4c5043f4b8d`
- `primary_class`: `ENGINEERING / SMOKE`
- `scientific_runs`: forbidden
- `data_downloads`: forbidden
- `post_2023_access`: forbidden
- `adaptive_experiment`: forbidden

The owner approved the implementation plan and its structural dependencies on
2026-07-13. This task adds operational observability around the canonical daily
research package. It does not create scientific evidence or change any frozen
experiment.

## Objective And Success Criteria

Build one English-language control center that lets the owner queue, observe,
cancel, resume, replay, and verify registered frozen studies. The advisor may
inspect the same protocol-safe evidence but cannot mutate any job or queue.

Success requires all of the following:

- one active heavy worker with a persistent, reorderable queue;
- append-only runtime events with reconnectable server-sent streaming;
- exact checkpoint/resume for long JM, HMM, selection, and bootstrap stages;
- live features, candidate states, selected lambda, signal, delayed position,
  CV surface, boundary status, resource usage, and provisional ETA;
- server-side outcome locks until the canonical verifier and research gate
  authorize metrics;
- read-only replay of the completed v7 and JM-4000 artifacts;
- authenticated remote access through Cloudflare Tunnel and Access;
- real-browser desktop/mobile acceptance and clean-clone setup instructions.

## Locked Architecture

- Add `adaptive-jump monitor --config research.toml`; keep every existing CLI
  command and public output compatible.
- Use FastAPI and Uvicorn for a loopback-only origin, vanilla ES modules for
  the browser, Apache ECharts vendored for offline charts, SQLite for control
  state, append-only JSONL for events, and SSE plus REST for communication.
- Launch scientific work only through the canonical CLI in a subprocess. The
  monitor must not duplicate model, selection, backtest, metric, or verifier
  logic.
- Keep runtime state under ignored `artifacts/.monitor/`. It is operational
  telemetry, never scientific evidence and never an input to a claim.
- Accept only code-registered study IDs whose latest registry state is
  `FROZEN`. Never accept arbitrary commands, paths, config edits, or uploads.
- Bind only `127.0.0.1`. Cloudflare Tunnel is the sole remote ingress.
- Validate the Cloudflare Access assertion signature, issuer, audience,
  algorithm, expiry, and email at the origin. Enforce owner/viewer roles on
  every API route, not only in the interface.
- Use exact-email OTP access by default. Secrets, audience IDs, tunnel
  credentials, and role email addresses remain outside Git.
- Keep runtime history indefinitely. No delete API or delete control is in
  scope.

## Approved Dependencies

The following additions are authorized and must be exact-lockfile pinned:

- runtime optional group: FastAPI 0.139.0, Uvicorn 0.51.0, PyJWT with its
  cryptography extra 2.13.0, and psutil 7.2.2;
- development group: Playwright 1.61.0;
- browser asset: Apache ECharts 6.1.0, served locally with its license, source
  URL, release identity, and SHA-256 manifest;
- deployment binary: `cloudflared`, installed outside the package from an
  official pinned release after its SHA-256 is checked.

Do not add MLflow, Aim, Prefect, a frontend framework, a Node build system,
Docker, Redis, a second database, or a second research package.

## Scientific And Security Boundaries

- Existing sealed runs remain immutable and are replayed read-only. They must
  not be rerun under the new instrumentation code SHA.
- Observer and checkpoint hooks default to no-op. Differential tests must show
  that uninterrupted result-bearing outputs are unchanged.
- Features, states, validation choices, signal, scheduled position, and gate
  diagnostics may be shown during a frozen run. Realized OOS returns, wealth,
  Sharpe, metrics, bootstrap claims, and claim text remain backend-locked until
  `metrics_opened=true` and canonical verification succeeds.
- A boundary failure is a valid terminal research state, not a software error.
- Missing, expired, wrongly signed, wrong-issuer, or wrong-audience Access
  assertions fail closed. Viewer mutations return `403`; locked outcomes
  return `423`.
- Origin checks, signed CSRF tokens, same-origin CSP, no CORS, path allowlists,
  and an append-only mutation audit are required before remote deployment.

## Write Boundary

Authorized tracked paths are:

- `TASK.md`, the completed-task archive, `README.md`, `.gitignore`,
  `pyproject.toml`, and `uv.lock`;
- `src/adaptive_jump/cli.py`, `models.py`, `walkforward.py`, `inference.py`,
  `window_runner.py`, `window_study.py`, and a single new
  `src/adaptive_jump/monitor/` package;
- focused monitor/checkpoint tests under `tests/`;
- authored English monitor documentation under `docs/monitor/` and deployment
  templates under `deploy/`;
- the two procedural `.agent/` handoff files under the standing exception.

Do not modify `research.toml`, frozen study TOML, experiment registry, market
data, sealed artifacts, reports, learning textbook, archive provenance, model
mathematics, grids, state semantics, costs, delays, metrics, gates, or claims.

## Milestones And Verification

1. Freeze this contract and lock the approved setup dependencies.
2. Add the event contract and no-op observer with differential tests.
3. Add identity-bound checkpoints and exact resume tests.
4. Add the one-worker queue, cancellation, recovery, and resource telemetry.
5. Add authenticated REST/SSE APIs and server-side scientific locks.
6. Build the live, replay, comparison, and evidence interface.
7. Add Cloudflare/systemd deployment and run full acceptance.

Each milestone is one reviewed commit of at most about 400 changed lines and 15
files. Run focused tests first, then full pytest, Ruff, lock, package, archive,
paper-hash, and frozen-run verification where relevant. UI milestones require
real Chromium checks at 1440x900 and 390x844. No scientific run, data fetch,
extension, or claim is authorized by this task.
