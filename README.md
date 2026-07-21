# Adaptive Jump Model

This project asks one practical question:

> Can a causal Jump-Model strategy beat both buy-and-hold and a Gaussian HMM
> on the same market sample after execution delay and trading costs?

That is also the economic question behind Shu, Yu, and Mulvey's study. Their
Jump Model is not useful merely because it clusters days. Its states must lead
to a better market-or-cash strategy than the two benchmarks.

## The experiment in one minute

1. Compute equity log return for HMM and three downside-risk features for JM:
   DD-10, Sortino-20, and Sortino-60.
2. Fit two-state Gaussian HMM and Jump Model candidates using past data only.
3. Each month, select JM lambda and HMM smoothing k from the preceding eight
   years.
4. Map the favorable state to 100% equity and the unfavorable state to 100%
   local cash.
5. A signal formed after day `t` first earns the return at `t+2`; charge
   10 bps whenever the position changes.
6. Compare net Sharpe on identical dates.

For market m and a prespecified JM variant v, the primary score is

$$
G_m(v) =
\operatorname{Sharpe}_{v,m}
- \max(\operatorname{Sharpe}_{B\&H,m},
        \operatorname{Sharpe}_{HMM,m}).
$$

Variant v succeeds in a market only when G_m(v) > 0. The cross-market target
requires the same prespecified v to pass every market. Maximum drawdown,
turnover, cash fraction, and switch count describe the risk and trading cost of
achieving that Sharpe; they do not replace the benchmark test.

## Models

**Buy and hold.** Always hold the equity index.

**Gaussian HMM.** Infer a latent two-state Markov process from univariate
equity log returns, then hold equity in the favorable state and cash in the
unfavorable state.

**Fixed Jump Model.** Cluster observations around two centers while charging a
constant cost λ whenever the state changes:

$$
\min_{\Theta,s}
\sum_t \tfrac12\lVert x_t-\theta_{s_t}\rVert^2
+\lambda\sum_{t\ge1}\mathbf 1\{s_t\ne s_{t-1}\}.
$$

**Evidence-adaptive Jump Models.** Keep the same observation loss but let the
switch cost respond to evidence. The first version used the current day's loss
gap. The stronger version uses the previous observation's gap, evaluated with
parameters available at the current decision date:

$$
C_t(i,j)=
\lambda_0\exp\!\left[
-\beta\tanh\!\left(
\frac{[L_{t-1}(i)-L_{t-1}(j)]_+}{q_{\rm train}}
\right)\right],\qquad i\ne j.
$$

Here q_train is the positive raw median absolute deviation of state losses on
the 3,000-row training prefix. Arrival and lagged rules used beta in
{0, log(2), log(4)}; the later pair-balanced rule used {0, log(4)}. In every
case beta=0 exactly reproduces fixed JM. Beta was scenario-fixed, not learned
to maximize profit.

Pair-balanced uses a signed lagged gap: it discounts one direction while
increasing the opposite direction so $C_t(i,j)+C_t(j,i)=2\lambda_0$. This
preserves the pair-average transition cost.

## Current result

These are the accepted net Sharpe ratios on the public proxy sample through
2023. Every model in a market uses the same dates, `t+2` execution, and 10-bps
one-way costs.

| Market | Buy & Hold | HMM | Fixed JM | Arrival adaptive | Lagged adaptive | Pair-balanced | Best JM gap G_m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| US | 0.512964 | **0.653725** | 0.569865 | 0.501745 | 0.586777 | 0.616326 | **-0.037398** |
| DE | 0.289638 | 0.007599 | 0.166440 | 0.277070 | **0.337888** | 0.335543 | **+0.048251** |
| JP | **0.544589** | 0.398539 | 0.329270 | 0.385254 | 0.413215 | 0.244542 | **-0.131375** |

The per-market best observed JM beats both benchmarks only in Germany. This
upper envelope uses Pair-balanced in US and Lagged in DE/JP; it is not one
universal model or deployable selection rule:

| Market | Best observed JM | MDD | Turnover | Cash fraction | Switches | Beats both? |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| US | Pair-balanced | -0.337905 | 0.404844 | 0.261740 | 13 | No: HMM is higher |
| DE | Lagged | -0.387794 | 0.248337 | 0.134270 | 8 | **Yes** |
| JP | Lagged | -0.317989 | 1.159509 | 0.279699 | 33 | No: buy-and-hold is higher |

**Bottom line:** the mathematical mechanisms are implemented and verified, but
no tested JM wins against both benchmarks in all three markets. Lagging the
evidence is an observed development-sample improvement in Germany, not yet a cross-market
trading result.

Turnover follows the paper convention:
0.5 × 252 × mean(|change in position|). For a 0/1 strategy, it annualizes
position changes and counts a complete equity-to-cash-to-equity round trip as
one unit. Transaction cost is calculated separately from the unhalved one-way
position change, so correcting the old turnover display did not change costs,
returns, or Sharpe.

## What the work has established

- The fixed-JM, HMM, signal, `t+2` position, trade, cost, and return pipeline
  is causally aligned and independently replayed.
- Fixed JM does not beat the strongest benchmark in US, DE, or JP on this
  proxy.
- Current-day adaptive discounting beats both benchmarks in 0/3 markets.
- Lagged discounting beats both in DE, beats HMM but not buy-and-hold in JP,
  and loses HMM in US.
- Pair balancing comes closest in US but is poor in JP.
- The complete grids used by the paper are not published and grid choice
  matters, but the source-grounded alternatives tested here did not produce a
  three-market winner.

The contribution so far is therefore a verified family of causal,
time-varying-transition Jump Models and a precise map of where they help or
fail. Benchmark outperformance would show that the inferred state is
economically useful; it would not prove that a latent state is the market's
unique “true regime.”

## Relation to the Shu paper

Shu, Yu, and Mulvey report that fixed JM beats both HMM and buy-and-hold in US,
Germany, and Japan over 1990--2023. This repository uses later public proxy
series whose usable test periods begin in 2007--2009, so their exact Sharpe
values are context, not our target. Our target is the same *within-sample
ordering* against B&H and HMM.

The authors' public [jump-models repository](https://github.com/Yizhan-Oliver-Shu/jump-models)
contains the generic JM library and examples, not the paper's full data,
HMM comparison, monthly validation pipeline, or final candidate grids.

## Reproduce

The frozen research sample stops at 2023-12-31.
No reported model or P&L experiment used later rows. A separate source audit
inspected public candidate series through July 2026, so those dates are not
untouched holdout evidence.

```bash
uv python install 3.12.3
uv sync --locked --extra data
.venv/bin/python -m pytest -q

.venv/bin/adaptive-jump fetch --config research.toml
.venv/bin/adaptive-jump run --study replication --config research.toml
.venv/bin/adaptive-jump verify --run artifacts/fixed-baselines/<run_id>

tectonic paper/manuscript.tex --outdir artifacts/paper
```

## Local monitor

The monitor uses the same locked environment:

```bash
uv sync --locked --extra data --extra monitor
.venv/bin/adaptive-jump monitor --config research.toml
```

Local use requires no authentication environment variables. It displays a
separately launched `adaptive-jump run`; it does not start or alter an
experiment. Remote deployment is documented in
[`docs/monitor/deployment.md`](docs/monitor/deployment.md). `requirements.txt`
is a compatibility export, not the dependency source; use `pyproject.toml` and
`uv.lock` for the locked environment.

Raw data and generated runs belong in ignored `data/` and `artifacts/`.
Only `src/adaptive_jump/` is active source; `archive/` is frozen history.

## Read next

- [Simple advisor brief](docs/research-workflow-comparison.html)
- [Paper source](paper/manuscript.tex) — compiles to the ignored local file
  `artifacts/paper/manuscript.pdf`
- [Current evidence](research/STATUS.md)
- [Mathematical and experimental history](research/SCIENTIFIC_LEDGER.md)
- [Current task](TASK.md)
- [Frozen core protocol](research.toml)
- [Original paper](2402.05272v3.pdf)
