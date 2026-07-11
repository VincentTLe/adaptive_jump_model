# B — Adaptive Jump Penalty on Daily Data (design spec)

- **Date:** 2026-07-07
- **Status:** design approved; P0 starting
- **Owner:** Claude (implementation) / Vincent (research direction)
- **Skill trail:** brainstorming → this spec → (writing-plans) → implementation

## Context

Target paper: Shu, Yu & Mulvey, *Downside Risk Reduction Using Regime-Switching
Signals: A Statistical Jump Model Approach* (arXiv 2402.05272v3, Aug 2024).
Reference code: `jumpmodels` (github Yizhan-Oliver-Shu/jump-models, Apache-2.0).

The current repo drifted from the paper: it uses tick/minute microstructure data,
microstructure features, and a hand-set jump penalty (λ≈3.37). This spec rebuilds
cleanly on the paper's actual setup (daily data, return-based features, CV-selected
λ) and then adds the project's contribution — an adaptive/time-varying penalty
`λ_t` (lane **B**). Lane **A** (asymmetric penalty matrix) is deferred to a fast
follow-on that reuses this same foundation.

## Goal (ambition = theory + empirical)

Establish, **above the evaluation noise floor**, whether an adaptive
`λ_t = f_β(z_t)` (β *learned*, never hand-set) beats the CV-optimal **fixed** λ of
the paper's JM, on daily equity indices, under a realistic backtest (cost + delay).
Either outcome is informative: a win is a contribution; a null strongly motivates
lane A.

## Hard constraints

- **Reproduce-first:** no contribution work until the paper baseline is reproduced
  (qualitatively) with `jumpmodels`.
- **No hand-set result-affecting coefficients** (repo AGENTS.md rule): every λ / β
  is CV-selected, MLE-fit, or theory-derived, with provenance recorded.
- **Reproduction bar = qualitative** (agreed): on Yahoo daily, JM > HMM > B&H on
  Sharpe/MDD with much lower turnover than HMM, and bear regimes align with known
  crises (2000, 2008, 2020). Exact Table-4 numbers not required (vendor/period differ).

## New dependencies (both implied by approved decisions)

- `jumpmodels` — the reference JM engine (approved).
- `yfinance` — fetch Yahoo daily data.

## Phases

### P0 — Foundation / reproduction  *(STARTING)*
1. Env: install `jumpmodels`, `yfinance` into `/tmp/adaptive_jump_model_venv`; run
   one of their example notebooks to confirm code+environment match the paper.
2. Data: Yahoo daily for `^GSPC`, `^GDAXI`, `^N225` + risk-free (`^IRX` or constant);
   longest available window per index.
3. Features (paper Table 2): EWM Downside Deviation (hl 10), EWM Sortino (hl 20, 60),
   standardized — reuse `jumpmodels` preprocessing utilities if shipped.
4. Models: 2-state JM (`jumpmodels`) + 2-state Gaussian HMM (`hmmlearn`), online
   inference with a 1-day trading delay.
5. λ selection: CV-Sharpe of the 0/1 strategy over a validation window (paper §3.4.3)
   → CV-optimal **fixed** λ baseline.
6. Backtest: 0/1 strategy (bull→100% index, bear→100% risk-free), 10bps one-way cost,
   1-day delay; metrics Return, Vol, Sharpe, MDD, Calmar, turnover, ES.
7. Noise floor: measure eval variability (seeds / window perturbation) so later
   deltas are judged against it.

**P0 output:** a table reproducing the JM > HMM > B&H pattern + the fixed-λ baseline
+ the noise floor. (This also *is* the "simplify" step — it retires the tick /
microstructure / old-backtest sprawl.)

### P1 — Adaptive layer (lane B)
- `λ_t = f_β(z_t)`, β learned by CV-Sharpe (or TVTP likelihood). The DP handles a
  time-varying `λ_t` unchanged (O(TK²)).
- **Open question (resolve after P0):** the conditioning signal `z_t` on daily data.
  Candidates: slow realized-volatility; the JM's own assignment margin / ambiguity;
  the paper's features.
- Compare to the P0 fixed-λ baseline against the noise floor. **Timebox:** if no lift
  above the noise floor after a bounded effort, record it as a null result and pivot
  to lane A on this same foundation.

### P2 — Verification
- Cost/delay sensitivity, delay robustness, adaptive-vs-fixed vs noise floor,
  multi-market.

## Risks & mitigations
- *Adaptive ≈ fixed null* (seen before) → measure noise floor, timebox, pivot to A.
- *Yahoo vs Bloomberg data differences* → qualitative reproduction bar.
- *Multi-agent git* (Codex active on `fix/backtest-trade-audit`) → P0 code goes on its
  own branch; decide before writing source modules.
