# Task: Free-Data Parity Audit

## Identity

- `task_id`: `data-parity-001`
- `status`: `active`
- `target_branch`: `cleanup/research-protocol`
- `starting_ref`: `9a985d8341e1396417228087870b6912f102b232`
- `primary_class`: `ENGINEERING / SMOKE`
- `claim_status`: no scientific, model-performance, or investment claim allowed

This task is authorized by the owner's request to reproduce Shu, Yu, and
Mulvey (2024) using the best verifiable free sources before extending the
sample. It audits data definitions and availability only.

## Source Paper

- Title: *Downside Risk Reduction Using Regime-Switching Signals: A
  Statistical Jump Model Approach*
- Version: arXiv `2402.05272v3`, 17 September 2024
- URL: <https://arxiv.org/abs/2402.05272>
- Local PDF SHA-256:
  `141e48bd5ccaefe5d2c276a3c8772716b583b1df91517109d54a2452f4cb3af1`

Paper-specified data contract:

- daily total-return series for the S&P 500, DAX, and Nikkei 225;
- each equity index denominated in its local currency;
- corresponding local three-month Treasury Bill Yield;
- equity source Bloomberg Terminal and rate source Global Financial Data;
- source sample from the start of 1970 through the end of 2023;
- no outlier processing;
- each market studied independently.

The paper does **not** publish exact Bloomberg/GFD series identifiers, yield
conversion conventions, holiday alignment rules, or downloadable experiment
code. Do not infer those details silently.

## Objective

For each of the six required series, identify and verify the best free source
candidate, measure definition and coverage parity against the paper, and
recommend one of:

1. `definition-parity candidate`;
2. `documented proxy candidate`;
3. `no acceptable free candidate found`.

The audit must separately assess the paper period and whether the same series
can continue to the latest completed local trading session. It must not join
different index definitions merely to obtain longer history.

## Source Priority

1. official index-provider or government/central-bank series;
2. OECD or FRED distribution of an identified official series;
3. Yahoo Finance;
4. another public source such as Stooq, only with a documented definition.

All sources must be free, require no private credentials, and permit the local
research use performed here. Search snippets are discovery aids, not evidence;
open the provider documentation and exercise the actual download path.

## Audit Fields

Record for every candidate:

- provider, canonical URL, series ID/ticker, and retrieval URL or API;
- price index versus total-return index, gross/net treatment, local currency;
- rate instrument, tenor, quote convention, frequency, and units;
- timezone/session meaning and publication lag when documented;
- first/last valid observation, row count, missing values, duplicate dates;
- download timestamp and SHA-256 of the raw response;
- relevant usage/licensing statement;
- overlap agreement with at least one independent source when possible;
- every deviation from the paper and its likely consequence.

Do not treat matching names as matching definitions. Do not forward-fill a
market series, convert a price index into a total-return index without a
verified dividend source, or substitute an interbank rate for a Treasury bill
without labeling it as a proxy.

## Allowed Work

- read provider documentation and public primary sources;
- download candidate data into ignored `data/raw/` or temporary directories;
- write generated audit evidence under ignored
  `artifacts/data-source-audit/<run_id>/`;
- create `docs/learning/01-data-parity.html` after the evidence exists;
- update this task's status and completion notes.

The learning page may explain index/rate definitions and proxy labels, but it
must not contain manually copied empirical result values.

## Write Boundary

- `TASK.md`
- `docs/learning/01-data-parity.html`
- ignored `data/**`
- ignored `artifacts/**`
- procedural `.agent/session-log.jsonl` and `.agent/session-log.html`

No package source, test, dependency, README, or research protocol config may be
changed in this task. A later approved task will implement the downloader and
freeze `research.toml` after the source decision.

## Acceptance Criteria

- all six paper series have an evidence-backed status;
- at least two independent candidates are checked where two exist;
- actual coverage and data quality are measured from downloaded responses;
- paper-period and extension suitability are reported separately;
- exact versus proxy status is explicit and no incompatible series are joined;
- the artifact contains machine-readable findings and raw-response hashes;
- the beginner HTML renders in Chromium at desktop and mobile widths;
- no model fitting, backtest, OOS metric, or scientific conclusion is produced.

## Stop Conditions

Stop and report rather than substituting silently when an exact series is
paywalled, its definition cannot be verified, its terms prohibit the intended
use, or free candidates conflict materially. Source selection requires owner
review before `research.toml` is frozen or downloader code is written.
