# Current Research Status

Last reconciled: 2026-07-22. The append-only experiment registry remains the
authority for lifecycle status; this page is the short human-readable view.

## Bottom line

The primary economic target is for a JM-guided strategy to beat both
same-sample buy-and-hold and the canonical Gaussian HMM after the same delay
and costs. For market m:

`G_m = Sharpe_JM,m - max(Sharpe_BuyHold,m, Sharpe_HMM,m)`.

No tested JM currently achieves `G_m > 0` in all three markets. The strongest
observed candidate by market is DD-only in US, lagged-evidence JM in DE, and
scaled DD in JP. Their gaps are `+0.25 / +0.05 / -0.12`. This
ex-post envelope passes in US and DE, but it mixes different models and is not
a deployable cross-market rule; JP still loses buy-and-hold.

Fresh data through 2026 has been added. Every evaluation below is walk-forward
causal: each monthly decision uses only trailing data, so the whole
2008/2009--2026 span is already out-of-sample per decision. Two distinct axes
matter. Walk-forward leakage: none, on any window. Selection bias: DD-only was
chosen from several variants after inspecting the through-2023 sample, so only
the 2024-01-02 to 2026-06-30 window is free of that choice.

On the **full walk-forward through 2026**, DD-only still beats both controls in
the US (`0.89` vs the stronger control `0.63`, gap `+0.258`) and loses in
DE and JP -- `1/3`, unchanged from development. Adding 2.5 new years barely
moved the US number (`0.91` through 2023 to `0.89` through 2026).

On the **isolated 2024-2026 window** the frozen binary rule returns
`not_supported`, `0/3`: US B&H `1.05` vs DD-only `0.78`, DE all four
`0.90` (the German JM never left equity), JP B&H `1.27` vs DD-only `1.17`.
This window is short (~620 days), the paired bootstrap intervals include zero,
and it was a broad bull that penalizes any cash rotation. It therefore
**fails to confirm** the US edge on selection-independent data but **does not
refute** the full walk-forward result. It is weak evidence, not a clean
negative. The sample is now spent, and the lagged-log4 batch remains available
under the same frozen contract.

The five-variant simple suite found that DD-only improved fixed-JM Sharpe in
all three markets and beat both controls in US. The completed loss-scale
control multiplied DD observation loss by three and left data, grid, timing,
cost, and selection unchanged. Its Sharpe stayed above fixed JM in all three
markets, but it also passed only US. This weakens a pure loss-scale explanation
without proving that the Sortino features are harmful.

The shared pipeline and accounting are internally consistent. Current source
reproduced the accepted simple result and can run both the simple suite and the
scale control. Independent checks confirmed identical market dates, a one-day
trading delay (decision at t, trade at the close of t+1, first return/cost row
at t+2), 10-bps costs, no post-2023 rows, and the paper turnover convention.
The mathematical contribution is a verified family of causal time-varying JM
decoders. The economic contribution remains incomplete, and no stable-profit
or generalization claim is authorized.

## Accepted evidence

| Role | Accepted experiment / run | Current conclusion |
| --- | --- | --- |
| Baseline | `fixed-baselines-001-v7` / `fixed-baselines-8adb330565d6-3636939b525d-e9614112b234` | Valid causal proxy pipeline; fixed JM beats both benchmarks in `0/3` markets |
| Fixed-model audit | `fixed-baseline-assumption-audit-001` / `fixed-baseline-assumption-audit-79c94852c8fd-3636939b525d-4cc8cdbccd14` | Pipeline reproducible; grids are binding; disclosed values do not rescue all markets |
| Mathematical challenger | `adaptive-confidence-001` / `adaptive-confidence-1b0c327b2db4-3636939b525d-864d671cf973` | Mechanism operational; beats both economic benchmarks in `0/3` markets |
| Mechanism diagnostic | `adaptive-separation-001` / `adaptive-separation-813f66912526-26cbca8871be-fefc608b9081` | Inconclusive; global separation gate not justified |
| Lagged mechanism | `lagged-evidence-mechanism-001` / `lagged-evidence-6f964f5724b2-26cbca8871be-d173ca32c86f` | Performance-free mechanism rule supported at `beta=log(4)`; no P&L claim |
| Lagged P&L | `lagged-evidence-performance-001` / `lagged-pnl-bad599271e2d-643dd3e6d96f-be70588256b2` | Incremental rule versus fixed supported; beats both economic benchmarks only in DE (`1/3`) |
| Lagged attribution | `lagged-selection-attribution-001` / `lagged-attribution-73a5995c487e-52854fc3c22a-197915169632` | Post-result mechanical diagnostic complete; choice schedule dominates mean Sharpe Shapley allocation, interaction is large, and no causal/performance claim is allowed |
| Endpoint-grid audit | `endpoint-grid-audit-001` / `endpoint-grid-audit-05e9d08f619b-77b30ef98fa0-24ca06c297e8` | JM endpoint is binding but does not rescue all markets; HMM endpoint is null at the primary delay |
| Balanced mechanism | `balanced-lagged-mechanism-001` / `balanced-lagged-a7d9914ca1a8-643dd3e6d96f-17961bfd667f` | Pair balance preserved latency (0.875 retention) with zero lock-in but did not reduce whipsaws; not supported |
| Balanced P&L | `balanced-lagged-performance-001` / `balanced-pnl-3ae665413a01-4e747110ba1c-eaae6444a9a5` | Not supported versus lagged; beats both economic benchmarks only in DE (`1/3`) |
| Simple challengers | `simple-jm-suite-001` / `simple-jm-suite-2d3d2a779b13-544237a59943-20260721T145043479851Z` | No cross-market winner; DD-only beats both controls in US and improves fixed-JM Sharpe in all three, but passes only `1/3` and is loss-scale confounded |
| DD loss-scale control | `dd-loss-scale-001` / `dd-loss-scale-e1e84ddbbdda-65ccb507abba-20260722T045053128156Z` | Mechanism verified; scaled DD beats both controls only in US (`1/3`), so the result is `not_supported` |
| Separation-turnover diagnostic | `separation-turnover-001` / `separation-turnover-8674ff4d9470-20260722T083551Z` | Not supported; decision-time DD center separation associates positively (US `+0.035`, DE `+0.320`, JP `+0.155`) with next-month switches, JP flips negative once collapsed one-state months are excluded, so no separation gate is justified |
| One-shot 2024-2026 holdout | `holdout-2026-001` / `holdout-20260722T111757Z` | Frozen rule returns `not_supported` on the isolated 2024-2026 window (`0/3`: US B&H `1.05` vs DD-only `0.78`; DE all four `0.90`; JP B&H `1.27` vs DD-only `1.17`). But the full walk-forward through 2026 still has DD-only beating both controls in the US (`1/3`, unchanged). The window is short and bull-dominated with bootstrap intervals spanning zero, so it fails to confirm the edge on selection-independent data without refuting the 18-year result |

Invalidated runs remain preserved for provenance, but they are not accepted
evidence. In particular, the `2207...` and `d6fe...` fixed-audit runs and the
`f505...` separation run must not be used for conclusions. Balanced P&L runs
ending `ceed18fc5288` and `def64a60db4c` are invalidated for access-provenance
metadata only. The first understated the thirteen feature columns physically
loaded. The second ambiguously labeled 43 explicitly locked files as all files
read even though inventory verification integrity-hashed 232 entries. Every
scientific CSV, trade, decision, smoke, and frozen lock is byte-identical across
the corrected runs. Only `eaae6444a9a5` is accepted.
The first simple-suite run ending `20260721T124416567215Z` is likewise retained
for provenance but is not the accepted artifact: 27 of 45 traces lacked point
losses, its real-data smoke compared only two overlapping rows, and its exact
executed runner source was no longer available after a verifier-only edit. Its
metrics were valid. The intermediate `20260721T135918520651Z` run reproduces
every metric and supplies the missing evidence, but it is also superseded: full
tests exposed that its added refit diagnostics changed the canonical checkpoint
schema, and its implementation lock omitted four result-affecting helpers plus
the environment lockfiles. The accepted `20260721T145043479851Z` run restores
the canonical schema, keeps diagnostics local to DD-only, and locks all twelve
result or environment files. Its summary and every scientific path are
byte-identical to the intermediate run.


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
18/18 boundary checks. Using its sealed decision-month denominators, the
upper-JM candidate was selected in 2.06% of US (4/194), 3.63% of DE (7/193),
and 4.52% of JP (8/177) months at the primary delay; all were below the local
5% rule.

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
`362.04` and HMM `1249`, both derived from sealed pre-OOS
calibration. Primary-delay endpoint-minus-base changes were:

| Market | JM Delta Sharpe | JM Delta MDD | JM Delta turnover | JM Delta cash | JM Delta switches | HMM metric change |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| US | `-0.07` | `0` | `+0.12` | `-0.01` | `+4` | none |
| DE | `+0.08` | `-0.09` | `-0.12` | `-0.04` | `-4` | none |
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

## Latest simple-suite decision

No frozen simple challenger wins in all three markets. DD-only is the only
variant that passes any market, and it passes US only:

| Variant | US G | DE G | JP G | Passes |
| --- | ---: | ---: | ---: | ---: |
| Static lambda 50 | `-0.07` | `-0.13` | `-0.36` | `0/3` |
| DD-only | `+0.25` | `-0.06` | `-0.12` | `1/3` |
| Confirmed 2d | `-0.03` | `-0.14` | `-0.25` | `0/3` |
| Return-aware | `-0.09` | `-0.14` | `-0.22` | `0/3` |
| Robust L1 | `-0.28` | `-0.12` | `-0.18` | `0/3` |

DD-only Sharpe was `0.91/0.23/0.42` in US/DE/JP, improving
canonical fixed JM by `+0.34/+0.06/+0.09`. US MDD improved from
`-0.34` to `-0.19`, turnover fell `0.65->0.34`, cash fell
`0.21->0.14`, and switches fell `21->11`. In DE, MDD worsened
`-0.39->-0.43` despite lower turnover and six fewer switches. In JP,
MDD worsened `-0.32->-0.43`, turnover rose `0.46->0.67`,
cash fell `0.28->0.09`, and switches rose `13->19`.

The mechanisms did what their formulas said, but not what the economic target
needed. Confirmation removed fixed-JM one-day excursions and cut switches in
US/DE, yet passed `0/3`. Return-aware gamma 1 changed only two US signal days
and zero JP signal days; it was economically neutral or worse. Robust L1
changed paths materially but increased activity in US/JP and passed `0/3`.

### DD loss-scale control

The frozen control used
`L_DD,scaled = 3 * 0.5 * (z_DD - theta)^2` with the same raw grid and every
other protocol field unchanged. It completed with result `not_supported`:

| Market | Scaled Sharpe | G | MDD | Turnover | Cash | Switches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| US | `0.88` | `+0.23` | `-0.20` | `0.47` | `0.14` | `15` |
| DE | `0.20` | `-0.09` | `-0.45` | `0.50` | `0.11` | `16` |
| JP | `0.43` | `-0.12` | `-0.35` | `0.81` | `0.14` | `23` |

Scaled minus ordinary DD-only was:

| Market | Delta Sharpe | Delta MDD | Delta turnover | Delta cash | Delta switches |
| --- | ---: | ---: | ---: | ---: | ---: |
| US | `-0.02` | `-0.00` | `+0.12` | `+0.01` | `+4` |
| DE | `-0.03` | `-0.02` | `-0.31` | `-0.00` | `-10` |
| JP | `+0.00` | `+0.09` | `+0.14` | `+0.05` | `+4` |

A positive MDD delta means a less-negative drawdown. Formula, toy-path,
lambda-third path, brute-force, prefix, timing, cost, turnover, and concrete
trace checks passed. Raw lambda moved upward in `73.7%/81.3%/66.1%` of paired
US/DE/JP months, as expected when observation loss triples. The upper endpoint
was selected in `0%/22.9%/29.0%`, and selected one-state fits occurred in
`0/194`, `44/193`, and `63/177` months. Thus the mechanism behaved as intended
mathematically, but finite-grid binding and state collapse remain material in
DE and JP.

Fit collapse is also material. DD-only occupied only one training state in
`95/132/124` fit-by-lambda rows in US/DE/JP, and monthly CV selected such a fit
in `0/43/75` months. Return-aware counts were `63/85/93` fitted rows and
`24/53/78` selected months; robust-L1 counts were `75/91/98` and `62/83/87`.
The DP makes an unavailable state's loss infinite, so states, trades, and P&L
remain deterministic. Economically, however, a selected collapsed candidate
is a one-state rule, not evidence of two recovered regimes. These fits were
reported and retained; none was removed after seeing performance.


The final accepted run independently replayed `24` metric rows, all `45`
complete loss-to-trade traces, and `9` collapse summaries, with maximum metric
difference `2.37e-14` under the frozen `1e-12` tolerance. It locked `31`
explicit scientific inputs, all `153` upstream inventory entries, and `12`
result or environment files; the US smoke compared `128` unchanged prefix rows
for each fitted variant. Independent audit and all `613` repository tests
passed. No post-2023 row was accessed. This remains repeatedly inspected
exploratory development evidence, not a performance or generalization claim.

## Earlier adaptive model evidence

Lagged-Evidence JM at `beta=log(4)` passed formula, beta-zero nesting,
brute-force objective, prefix-invariance, toy paths, t+2 accounting, plus completion-time and separate CLI
source replays. Lagged-minus-fixed Sharpe was `+0.02/+0.17/
+0.08` in US/DE/JP, for a market-equal mean of `+0.09`. US switches fell
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
attribution was total `+0.09`, path Shapley `-0.03`, and choice-schedule
Shapley `+0.13`. The corresponding direct effects at the fixed counterpart
were `-0.14` and `+0.02`, with interaction `+0.21`. For turnover,
path and choice Shapley effects were `+0.18` and `-0.26`; in JP both
were positive. These are mechanical allocations of nonlinear metrics, not
causal effects.

The subsequent pair-balanced P&L readout did not support the challenger.
Balanced-minus-lagged Sharpe was `+0.03/-0.00/-0.17` in US/DE/JP,
for an equal-market mean of `-0.05` and only one positive market. US MDD
was equal within `1e-9`, turnover fell `0.06`, cash rose `0.01`, and
switches fell `15→13`. DE MDD was equal within `1e-9`, but turnover rose
`0.25`, cash rose `0.03`, and switches rose `8→16`. JP MDD worsened
`0.10`, turnover rose `0.21`, cash fell `0.05`, and switches
rose `33→39`.

Balanced-minus-fixed passed separately with mean Delta Sharpe `+0.04`
and `2/3` positive markets, but the frozen decision required both contrasts.
Upper-candidate selection for balanced log4 was `9.84%/45.31%/38.64%`, so the
finite adaptive optimum remains unidentified. The corrected run passed exact
source/accounting replay with maximum artifact error `0.0`; its numerical
outputs are byte-identical to the first run invalidated solely for inaccurate
access-provenance metadata.

The evidence therefore does not support the simple story that lagging the loss
made the state path intrinsically better. Much of the readout came through how
monthly validation changed lambda choices and through interaction. The next
confirmation priority is untouched markets or genuinely prospective data.
Robust-gap, two-day-confirmation, semi-Markov, and feature-metric variants
remain mathematical hypotheses; they must not be swept and winner-selected on
the repeatedly inspected US/DE/JP sample. See `TASK.md` and
`SCIENTIFIC_LEDGER.md` for exact grids, equations, and evidence status.
