# Task: Fixed JM/HMM Endpoint-Grid Audit

## Identity

- `task_id`: `endpoint-grid-audit-001`
- `status`: `FROZEN / NOT RUN`
- `target_branch`: `cleanup/research-protocol`
- `parent_experiment`: `fixed-baselines-001-v7`
- `frozen_spec`: `research/endpoint-grid-audit.toml`
- `frozen_spec_sha256`:
  `05e9d08f619b0bd0ca2fc49cb508b754e38743ee312034c3223db52bb42dbfa7`
- `claim_class`: `EXPLORATORY`
- `data_cutoff`: `2023-12-31`
- `adaptive_model_access`: forbidden
- `performance_claim`: forbidden
- `monitor_changes`: forbidden

Accepted prior results are summarized in `research/STATUS.md`; mathematical and
experimental history is retained in `research/SCIENTIFIC_LEDGER.md`.

## Scientific Question

Were the behavior-calibrated nine-point JM and HMM grids truncated too early,
and can adding exactly their last globally valid eligible endpoint explain the
fixed-v7 proxy non-replication?

This is a one-shot sensitivity audit. It does not recover the paper's
undisclosed final grid, search for the best endpoint, or claim replication or
profitability.

## Why These Endpoints

The endpoints are not chosen from market returns after the fact. They are
derived from the sealed pre-OOS calibration as the maximum candidate satisfying
both `globally_valid=True` and `eligible=True`:

- JM endpoint: `2^(17/2) = 362.03867196751236`; the next candidate `512`
  was globally invalid.
- HMM endpoint: `1249`; the next candidate `1250` was globally invalid.

The base grids remain the previously calibrated nine-candidate grids. The audit
adds one endpoint to each family and forbids every further expansion.

## Five Paths and Four Logical Cells

Only five unique paths are fit:

- `B&H`
- `J0`: base fixed-JM grid
- `J1`: `J0` plus the single derived JM endpoint
- `K0`: base HMM smoothing grid
- `K1`: `K0` plus the single derived HMM endpoint

The four cells `A=(J0,K0)`, `B=(J1,K0)`, `C=(J0,K1)`, and
`D=(J1,K1)` are compositions of those paths; they are not four separately
fit models.

## Causal and Accounting Protocol

- Reuse v7 features, 3,000-observation fit window, fitted protocol, and causal
  through-2023 timeline.
- Recompute all base and endpoint candidates under the current audited code.
- Require exact selection-behavior parity with the sealed base witness in all
  three markets before constructing accounting or metrics.
- Use signal at `t` for the position at the second subsequent observation.
- Charge 10 bps on full one-way position changes.
- Report paper turnover:
  `0.5 * 252 * mean(abs(position change))`.
- Compare all five paths on one per-market intersection for delays `1, 5, 10`.
- Run the performance-free US smoke first, then US/DE/JP with three
  `forkserver` workers and one numerical thread each.

The local 5% upper-boundary rule is descriptive only. It cannot stop, hide, or
change a metric.

## Frozen Decision

At primary delay 1, cell D is a three-market rescue only if, in every market:

1. Sharpe(`J1`) is strictly greater than Sharpe(`K1`).
2. Sharpe(`J1`) is strictly greater than Sharpe(`B&H`).
3. Absolute MDD(`J1`) is strictly smaller than absolute MDD(`B&H`).

An absolute MDD difference of at most `1e-9` is neutral, not an improvement.
The audit also reports endpoint-minus-base changes in Sharpe, MDD, turnover,
cash fraction, and switch count, plus concrete
choice→signal→t+2-position→trade dates.

If an endpoint remains the most-selected candidate above the descriptive 5%
rate, the finite optimum is unidentified and the study stops without expanding
the grid.
