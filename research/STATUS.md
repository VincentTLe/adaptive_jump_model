# Current Research Status

Last reconciled: 2026-07-18. The append-only experiment registry remains the
authority for lifecycle status; this page is the short human-readable view.

## Bottom line

The fixed v7 proxy pipeline is reproducible and its accounting is now audited,
but it does not reproduce the paper's three-market result. No mathematical
extension has earned a stable-profit claim. The lagged-evidence transition
penalty passed its performance-free mechanism rule and its frozen development-
sample P&L rule, but adaptive upper-grid concentration and repeated use of the
same three markets prevent a confirmation claim. The fixed-model endpoint audit
confirmed that the JM grid is binding, but its single source-derived extension
did not rescue the paper ordering. Post-result attribution shows that lagged
P&L differences cannot be read as a standalone improvement in the state path:
the monthly selector response and path-choice interaction are material. A
pair-balanced variant that preserves the fixed pair-average transition scale
kept most early confirmations without creating lock-in but did not reduce own
or matched whipsaws, so it is not supported.

## Accepted evidence

| Role | Accepted experiment / run | Current conclusion |
| --- | --- | --- |
| Baseline | `fixed-baselines-001-v7` / `fixed-baselines-8adb330565d6-3636939b525d-e9614112b234` | Valid proxy non-replication |
| Fixed-model audit | `fixed-baseline-assumption-audit-001` / `fixed-baseline-assumption-audit-79c94852c8fd-3636939b525d-4cc8cdbccd14` | Pipeline reproducible; grids are binding; disclosed values do not rescue all markets |
| Mathematical challenger | `adaptive-confidence-001` / `adaptive-confidence-1b0c327b2db4-3636939b525d-864d671cf973` | Mechanism operational; frozen performance rule not supported |
| Mechanism diagnostic | `adaptive-separation-001` / `adaptive-separation-813f66912526-26cbca8871be-fefc608b9081` | Inconclusive; global separation gate not justified |
| Lagged mechanism | `lagged-evidence-mechanism-001` / `lagged-evidence-6f964f5724b2-26cbca8871be-d173ca32c86f` | Performance-free mechanism rule supported at `beta=log(4)`; no P&L claim |
| Lagged P&L | `lagged-evidence-performance-001` / `lagged-pnl-bad599271e2d-643dd3e6d96f-be70588256b2` | Frozen development rule supported; Sharpe delta positive in 3/3 markets, but JP whipsaw and adaptive endpoint concentration remain |
| Lagged attribution | `lagged-selection-attribution-001` / `lagged-attribution-73a5995c487e-52854fc3c22a-197915169632` | Post-result mechanical diagnostic complete; choice schedule dominates mean Sharpe Shapley allocation, interaction is large, and no causal/performance claim is allowed |
| Endpoint-grid audit | `endpoint-grid-audit-001` / `endpoint-grid-audit-05e9d08f619b-77b30ef98fa0-24ca06c297e8` | JM endpoint is binding but does not rescue all markets; HMM endpoint is null at the primary delay |
| Balanced mechanism | `balanced-lagged-mechanism-001` / `balanced-lagged-a7d9914ca1a8-643dd3e6d96f-17961bfd667f` | Pair balance preserved latency (0.875 retention) with zero lock-in but did not reduce whipsaws; not supported |

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
| JM lambda candidate grid | Yes | Strongly binding and a plausible contributor, but neither tested paper-visible/historical grids nor the last globally valid eligible endpoint rescue all three markets; the true final paper grid remains unknown |
| HMM smoothing grid | Yes, for the HMM comparator | Its added eligible endpoint changes no primary-delay metric; it cannot change fixed-JM standalone returns |
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

The one-shot endpoint test sharpened that conclusion. It added only JM
`362.03867196751236` and HMM `1249`, both derived from sealed pre-OOS
calibration. Primary-delay endpoint-minus-base changes were:

| Market | JM Delta Sharpe | JM Delta MDD | JM Delta turnover | JM Delta cash | JM Delta switches | HMM metric change |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| US | `-0.0739` | `0` | `+0.1248` | `-0.0097` | `+4` | none |
| DE | `+0.0790` | `-0.0892` | `-0.1244` | `-0.0435` | `-4` | none |
| JP | `0` | `0` | `0` | `0` | `0` | none |

The JM endpoint was selected in `6.70%`, `28.50%`, and `5.08%` of primary
US/DE/JP choice months. Thus grid truncation is real, but the tested endpoint
helps DE on Sharpe, turnover, and switch count while worsening its MDD; it
hurts US on Sharpe, turnover, and switches; and leaves the JP primary path
unchanged. The frozen three-market rescue failed and the finite JM optimum
remains unidentified; no metric was hidden by the descriptive 5% rule.

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
- The source-derived endpoint audit passed exact all-market base parity. Its JM
  extension changed primary paths in US and DE but not JP; its HMM extension
  changed no primary metric and the frozen three-market rescue failed.

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

## Latest model decision and next step

Lagged-Evidence JM at `beta=log(4)` passed formula, beta-zero nesting,
brute-force objective, prefix-invariance, toy paths, t+2 accounting, plus completion-time and separate CLI
source replays. Lagged-minus-fixed Sharpe was `+0.0169/+0.1714/
+0.0839` in US/DE/JP, for a market-equal mean of `+0.0908`. US switches fell
`21→15` and DE `32→8`; JP instead rose `13→33`. MDD improved slightly in US
and JP and was equal within `1e-9` in DE.

Lagged JM was the strongest local B&H/HMM/JM comparator only in DE; US HMM
Sharpe remained higher (`0.654` versus `0.587`) and JP B&H remained higher
(`0.545` versus `0.413`). It therefore did not recover the paper ordering.
This satisfies the frozen exploratory rule, not a stable-profit claim. The
upper lambda was selected in `8.81%/45.31%/30.11%` of adaptive US/DE/JP months,
so the finite optimum is unidentified and the result is conditional on the
frozen grid. The same through-2023 markets have been inspected repeatedly.

The frozen post-result 2x2 attribution is complete. Equal-market mean Sharpe
attribution was total `+0.090769`, path Shapley `-0.034835`, and choice-schedule
Shapley `+0.125603`. The corresponding direct effects at the fixed counterpart
were `-0.137373` and `+0.023065`, with interaction `+0.205076`. For turnover,
path and choice Shapley effects were `+0.181900` and `-0.258276`; in JP both
were positive. These are mechanical allocations of nonlinear metrics, not
causal effects.

The evidence therefore does not support the simple story that lagging the loss
made the state path intrinsically better. Much of the readout came through how
monthly validation changed lambda choices and through interaction. The next
confirmation priority is untouched markets or genuinely prospective data.
Robust-gap, two-day-confirmation, semi-Markov, and feature-metric variants
remain mathematical hypotheses; they must not be swept and winner-selected on
the repeatedly inspected US/DE/JP sample. See `TASK.md` and
`SCIENTIFIC_LEDGER.md` for exact grids, equations, and evidence status.
