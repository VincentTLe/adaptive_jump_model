# Current Research Status

Last reconciled: 2026-07-17. The append-only experiment registry remains the
authority for lifecycle status; this page is the short human-readable view.

## Bottom line

The fixed v7 proxy pipeline is reproducible and its accounting is now audited,
but it does not reproduce the paper's three-market result. No mathematical
extension has yet earned a stable-profit claim. The lagged-evidence transition
penalty passed its performance-free mechanism rule; no new market P&L has been
opened.

## Accepted evidence

| Role | Accepted experiment / run | Current conclusion |
| --- | --- | --- |
| Baseline | `fixed-baselines-001-v7` / `fixed-baselines-8adb330565d6-3636939b525d-e9614112b234` | Valid proxy non-replication |
| Fixed-model audit | `fixed-baseline-assumption-audit-001` / `fixed-baseline-assumption-audit-79c94852c8fd-3636939b525d-4cc8cdbccd14` | Pipeline reproducible; grids are binding; disclosed values do not rescue all markets |
| Mathematical challenger | `adaptive-confidence-001` / `adaptive-confidence-1b0c327b2db4-3636939b525d-864d671cf973` | Mechanism operational; frozen performance rule not supported |
| Mechanism diagnostic | `adaptive-separation-001` / `adaptive-separation-813f66912526-26cbca8871be-fefc608b9081` | Inconclusive; global separation gate not justified |
| Lagged mechanism | `lagged-evidence-mechanism-001` / `lagged-evidence-6f964f5724b2-26cbca8871be-d173ca32c86f` | Performance-free mechanism rule supported at `beta=log(4)`; no P&L claim |

Invalidated runs remain preserved for provenance, but they are not accepted
evidence. In particular, the `2207...` and `d6fe...` fixed-audit runs and the
`f505...` separation run must not be used for conclusions.

## Paper versus current proxy

The values below are Sharpe ratios at the primary one-day signal delay. They
are not like-for-like replications: the paper reports 1990--2023 on its exact
sources, while the free proxy outer samples begin in 2007--2009.

| Market | Paper B&H / HMM / JM | Proxy B&H / HMM / JM |
| --- | --- | --- |
| US | `0.48 / 0.54 / 0.68` | `0.513 / 0.654 / 0.570` |
| DE | `0.30 / 0.35 / 0.44` | `0.290 / 0.008 / 0.166` |
| JP | `0.12 / 0.19 / 0.31` | `0.545 / 0.399 / 0.329` |

The proxy therefore misses the paper's central ordering: fixed JM is not the
best of B&H, HMM, and JM in any of the three markets. JP also demonstrates a
materially different target sample: proxy B&H Sharpe is about `0.545`, versus
`0.12` in the paper.

## What can cause which failure

| Repo choice | Can change state/P&L? | What the audit says |
| --- | --- | --- |
| JM lambda candidate grid | Yes | Strongly binding and a plausible contributor, but tested paper-visible/historical grids do not rescue all three markets; the true final paper grid remains unknown |
| HMM smoothing grid | Yes, for the HMM comparator | Can change the JM-vs-HMM comparison, but it cannot change fixed-JM standalone returns |
| 5% upper-boundary guardrail | No | Can stop or label a study as boundary-failed; it cannot make Sharpe, drawdown, or trades worse |
| Three-market directional gate | No | It is a stricter repo classification inspired by the paper's reported ordering; it cannot alter any metric |

The first two choices remain genuine underidentification. The last two can
explain why the repository says `FAIL`, but not why the underlying proxy
strategy earned different returns from the paper.

Concrete evidence makes the distinction visible. Canonical v7 passed all
18/18 boundary checks; at the primary delay the upper-JM candidate was
selected in only 2.06% of US, 3.63% of DE, and 4.52% of JP months,
all below the local 5% rule.

| Fixed-JM grid sensitivity versus v7, primary-delay Sharpe | US | DE | JP |
| --- | ---: | ---: | ---: |
| Values visible in paper Table 3 | +0.011 | -0.129 | -0.316 |
| Historical repo v1 | -0.053 | +0.112 | -0.109 |
| Union of source-visible values | -0.042 | +0.065 | -0.192 |

The Table-3 JM grid also changed JP from 13 to 82 switches and from about
0.330 to 0.014 Sharpe. This proves that the JM grid is influential, but
the signs across markets show that none of these tested alternatives recovers
the paper's three-market result.

## What the audit established

- The sealed fixed-v7 result was replayed exactly; `670/670` audit checks and
  the independent scientific replay passed.
- The old turnover display was exactly two times the paper convention. The
  corrected statistic is `0.5 * 252 * mean(abs(position change))`. Costs,
  returns, drawdowns, and Sharpe were already correct.
- The paper does not disclose its complete final cross-validation grids.
  Locally added candidates are selected often, so grid choice matters.
- Restricting candidates to values visible in the paper changes paths but does
  not recover a stable result across US, DE, and JP; it is especially harmful
  to JP.
- The local 5% upper-boundary rule is not from the paper and is descriptive
  only. It does not enter the model or P&L.

## Why the paper result was not reproduced

The evidence supports an underidentified proxy non-replication, not one single
code failure:

1. Exact paper sources and the 1970 warm-up are unavailable. US/DE proxy data
   cannot produce the paper's 1990 start; the tested outer periods are much
   shorter and economically different.
2. Some instrument definitions differ, including the JP price-index proxy and
   DE/JP cash-rate proxies.
3. Complete final-v3 JM and HMM candidate grids are undisclosed.
4. Grid selection is path-discrete and strongly binding, but no tested
   source-grounded grid works across all three markets.
5. The first adaptive penalty used today's loss both as today's state loss and
   as today's barrier discount. That same-day double use can amplify a one-day
   shock and is consistent with the JP whipsaw result.

Items 1--4 concern the fixed-model paper gap. Item 5 motivates the next
mathematical model; it does not explain the fixed-JM non-replication.

## Active next decision

The corrected Lagged-Evidence JM passed exact formula, fixed-JM nesting,
brute-force objective, prefix-invariance, toy-path, and independent-replay
checks. At `beta=log(4)`, pooled candidate-path whipsaws fell `17→6`, every
market was non-worse, JP candidate-path switches fell `266→258`, and `11`
confirmed-early events remained. The result concerns state paths only.

The next fixed-model question is whether the behavior-calibrated JM and HMM
grids were truncated at their upper endpoints. This must be frozen as one
predefined endpoint test; the local 5% boundary remains descriptive and cannot
change or censor metrics. See `TASK.md` and `SCIENTIFIC_LEDGER.md` for the full
history.
