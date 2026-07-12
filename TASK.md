# Task: Freeze And Implement Causal Features

## Identity

- `task_id`: `causal-features-001`
- `status`: `active`
- `target_branch`: `cleanup/research-protocol`
- `starting_ref`: `279208e`
- `primary_class`: `ENGINEERING / SMOKE`
- `target_study_class`: `REPLICATION`
- `claim_status`: no model-performance or investment claim allowed

The owner approved continuous execution of the existing plan. This task may
proceed through small verified commits without another approval stop.

## Inputs

- Frozen six-source bundle in `research.toml`
- Protocol config ID/hash: `shu-proxy-replication-v2` /
  `1963d093164b7b6bd52d31ea9f5744d1d1628905f19f5ac71b107557c29ba497`
- Accepted acquisition manifest:
  `data/raw/shu-proxy-replication-v1-20260712T071245Z/manifest.json`
- Manifest SHA-256:
  `438c28426fffcbd1b57cb2c2439d4efd319e650cac977d98c339b1e362596634`
- Replication cutoff: `2023-12-31`; extension remains disabled

The prior acquisition proves source availability, but MUST be rerun after this
contract commit so the accepted feature input carries the v2 config hash.

Changing the return, timing, missing-data, feature, delay, or cost rules below
requires a new config ID/hash before any model run.

## Definitions

For an equity index level `P_t` observed at the end of trading date `t`:

```text
equity_simple_t = P_t / P_(t-1) - 1
equity_log_t    = log(P_t / P_(t-1))
```

An annualized cash yield `y_s` is quoted in percent. After its causal
availability date is reached, its daily simple cash return is:

```text
cash_t   = (y_s / 100) / 252
excess_t = equity_simple_t - cash_t
```

This conversion treats all three heterogeneous source quotes as annualized
simple-yield proxies. It is an explicit paper deviation: the paper does not
publish quote-basis conversion, day count, or release alignment.

## Causal Cash Alignment

- US daily `DTB3`: observation `s` becomes usable on `s + 1 calendar day`.
- German/Japanese monthly averages: month `s` becomes usable at the start of
  the second following month. Example: January is first usable on 1 March.
- Alignment is backward-as-of only; a later source value can never affect an
  earlier equity date.
- US observations may be carried at most 10 calendar days.
- Monthly observations may be carried at most 120 calendar days.
- Beyond the staleness limit, cash and excess returns are missing.
- Negative yields are retained. No interpolation, averaging, clipping, or
  zero substitution is allowed.

This as-of carry is a protocol-authorized transformation, not acquisition
forward-fill. Raw and canonical source observations remain unchanged.

## Equity Missing Data

- Non-positive index levels fail.
- Missing index levels are removed before return calculation; no price is
  imputed.
- A return is measured between consecutive valid source observations and is
  dated at the later observation.
- The elapsed calendar-day gap is recorded so compressed multi-day returns are
  visible in validation artifacts.

## Paper Features

For excess return `R_t`, pandas EWM uses `adjust=True`, `ignore_na=False`,
observation-based halflife, and no custom burn-in:

```text
negative_t = min(R_t, 0)
DD_h(t)    = sqrt(EWM_h(negative_t^2))
Sortino_h  = EWM_h(R_t) / DD_h(t)
```

- Features: `DD_10`, `Sortino_20`, `Sortino_60`.
- Each Sortino numerator and denominator uses its own stated halflife.
- A zero denominator yields missing, never infinity or an epsilon-adjusted
  value.
- No clipping, winsorization, outlier removal, annualization, or feature
  scaling occurs here.
- Trailing-3,000 standardization belongs to the model protocol, not this task.

## Effective OOS Eligibility

The model cannot train before complete excess-return features exist. For each
market:

1. find the 3,000th complete feature row;
2. add eight complete calendar years for online validation;
3. choose the first later complete feature date not before `1990-01-01`.

The computed date is descriptive eligibility, not an experiment result. It
must not be moved earlier or harmonized across markets by backfill/splicing.

## Backtest Accounting Contract

- Signal convention: `1 = risky equity`, `0 = cash`.
- A signal identified at end of date `t` with delay `d` is first applied to
  return observation `t + d + 1`; primary `d = 1`, hence `t + 2`.
- Strategy simple return is
  `w_t * equity_simple_t + (1 - w_t) * cash_t - cost_t`.
- One-way cost is `0.001 * abs(w_t - w_(t-1))` for 10 bps.
- Establishing the first executable allocation is cost-free, matching the
  paper's zero-turnover buy-and-hold convention.
- Missing signal/return/cash produces no strategy observation; no default
  position is invented.
- Delay sensitivities 5 and 10 reuse the same timeline formula later.

## Write Boundary

- `TASK.md`, `research.toml`
- `src/adaptive_jump/config.py`, `src/adaptive_jump/features.py`,
  `src/adaptive_jump/backtest.py`
- focused tests under `tests/`
- `docs/learning/03-returns-features.html`
- ignored `artifacts/causal-features/**`
- procedural `.agent/session-log.jsonl` and `.agent/session-log.html`

No JM/HMM fit, hyperparameter selection, strategy metric, extension data,
dashboard, or scientific claim is allowed.

## Acceptance Criteria

- config v2 freezes every rule above and loader rejects unsafe changes;
- unit tests cover yield conversion, causal availability, staleness, negative
  yields, missing prices, return dating, exact EWM formulas, zero DD, delay,
  cost, and first-trade handling;
- leakage tests show future rate/equity changes cannot alter past outputs;
- real-data smoke emits per-market coverage and effective OOS eligibility with
  source/config/code hashes but no performance metric;
- all tests and Ruff checks pass;
- beginner HTML renders correctly in Chromium desktop/mobile.

## Completion

Record artifact hashes and verification, then continue to a separately frozen
fixed-JM/HMM walk-forward protocol under continuous-execution approval.
