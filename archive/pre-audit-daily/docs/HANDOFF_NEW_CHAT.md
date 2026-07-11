# Handoff — for the next implementer (Codex/ChatGPT)

Date: 2026-07-11
Branch: `p0-daily-baseline`, PR #2 (draft) → `main`
Written by: the Claude session that pushed P2b and then verified the 2026-07-11 external audit.
This file replaces the stale 2026-06-30 handoff (which described the pre-pivot minute pipeline).

## One-paragraph state

The branch pivots the project from the legacy tick/minute pipeline to a daily walk-forward
setting inspired by Shu–Yu–Mulvey (arXiv 2402.05272): CV-selected jump penalty, 10 bps costs,
t+2 delay, train-only scaling, OOS 2005-01 → 2024-07. P0 (baseline) shows JM > {return-only
HMM, B&H} on OOS Sharpe in 3/3 markets (0.67/0.54/0.43). P1 (covariate λ_t) is a narrow null.
**P2 (asymmetric penalty) is withdrawn: the design was mathematically degenerate** (see below).
An external audit on 2026-07-11 was verified claim-by-claim against this repo and accepted;
PR #2's body is the corrected account. Suite: 107 tests + ruff clean.

## Accepted audit findings (all verified against this repo — do not re-litigate, build on them)

1. **P0 is an approximate Yahoo replication, not a reproduction.** Yahoo close returns
   (not Bloomberg total-return), rf=0, OOS 2005–2024 (`OOS_START` in
   `scripts/p0_daily_baseline.py`), 2520d fit / 5y val / 126d reselection (paper: 3000d / 8y /
   monthly), per-block decoder reset. S&P Sharpe 0.67 vs paper 0.68 agrees, but CAGR/MDD/turnover
   do not (8.5%/−33.9%/~127% vs 11.2%/−26.6%/44%), and HMM < B&H on DAX/Nikkei breaks the paper's
   JM > HMM > B&H ordering. The HMM comparator fits `GaussianHMM` on 1-D returns — it is NOT
   feature-matched.
2. **The fixed-λ baseline is not yet provably CV-optimal.** The grid caps at 1000 and the cap is
   selected in 14/40 (S&P), 20/40 (DAX), 19/39 (N225) blocks, with ties broken toward larger λ.
3. **The 5-seed spread (≤0.03) is optimizer-initialization sensitivity, not a significance
   threshold.** No Sharpe delta on this branch has statistical inference behind it. The
   "0.05 noise rent" language in older commit messages is withdrawn.
4. **P2a/P2b do not test asymmetric persistence.** For any binary path, transitions alternate:
   `a·N01 + b·N10 = (a+b)/2 · N_switch + (a−b)/2 · (s_T − s_0)`.
   A zero-diagonal 2×2 penalty is exactly a symmetric penalty plus a constant endpoint/threshold
   bias — it cannot encode different bull vs bear persistence. The per-block decoder reset turns
   the endpoint term into a recurring boundary artifact. P2b additionally applies
   δ = ln(d_bull−1) − ln(d_bear−1) (a penalty *difference*, ~0.2) as a log-*ratio* around λ̄,
   producing cost gaps ≈ λ̄·δ (e.g. 11.07 at λ̄=50) — inconsistent with its own justification.
   The committed P2 numbers describe a threshold-shift model; their commit-message conclusions
   (e058361, 2b5cbad) are withdrawn.
5. **Reproducibility blocker:** `data/processed/yahoo_cache/{GSPC,GDAXI,N225}.csv` is required
   (`load_returns` raises without it) but no cache-builder is committed and nothing in the repo
   imports yfinance. A clean clone cannot re-run P0–P2. There is no data manifest/hash/cutoff.

## What is trustworthy on this branch

- `src/adaptive_jump/tv_jump.py` — exact time-varying-penalty DP. **Key fact for the redesign:**
  `dp_tv` minimizes `Σ_t L(t,s_t) + Σ_{t≥1} penalty_seq[t][s_{t−1}, s_t]` — the (T,K,K) matrix is
  applied at every step **including stays**, so nonzero diagonals (dwell costs) already work and
  are brute-force oracle-tested. A correct asymmetric/semi-Markov design needs no solver change.
- 107 unit tests + ruff (venv recipe below). Tests cover solvers/models, not the pipeline scripts.
- Committed artifacts under `reports/p0|p1|p2/` match the numbers quoted here and in PR #2.
- `.agent/session-log.jsonl` — append-only cross-agent session log (one JSON line per session;
  regenerate the HTML with `python3 .agent/render_log.py .agent`).

Do NOT trust: interpretations in commit messages e058361/2b5cbad (superseded by PR #2 body);
root `reports/` dashboard + `demo_summary.md`, `STATUS.md`, `TASK.md` (all legacy minute-era).

## Prioritized next work (auditor-endorsed; acceptance criteria included)

1. **Deterministic data acquisition.** `scripts/build_yahoo_cache.py`: yfinance download with a
   fixed end cutoff, writes the three cache CSVs + a manifest (source, tickers, cutoff, SHA-256
   per file). README with exact commands for cache → P0 → P1. Accept: clean clone reproduces the
   committed P0 metrics from the manifest data, or documents divergence caused by Yahoo revisions.
2. **λ-grid plateau check.** Extend the grid (e.g. ×{2000, 4000, 8000}); rerun P0. Accept: either
   selections leave the boundary / metrics stable (baseline stands) or the baseline numbers move
   (then all deltas must be re-based).
3. **Statistical inference.** Paired moving-block bootstrap on daily strategy-return differences
   (challenger − fixed) for every comparison; keep post-2024-07 data (never downloaded so far) as
   a genuinely untouched holdout. Accept: every reported delta ships with a CI.
4. **Asymmetry done right (P3).** Normalized transition-cost matrix λ_ij = −log P(i,j) with
   nonzero diagonal (2 free hazards; nests fixed-λ), or explicit semi-Markov/dwell-time costs;
   chain the online decoder state across blocks instead of resetting each 126d. Check Aydınhan
   et al. 2024 (state-specific jump penalties) for novelty overlap BEFORE building. Accept:
   a 10-line identity/sanity argument that the parameterization actually spans
   different-persistence hypotheses (the check P2 skipped), then the walk-forward protocol.
5. **Docs cleanup.** Mark `STATUS.md`/`TASK.md`/root `reports/` as legacy or rewrite; add README
   + minimal CI (pytest + ruff).

## Environment

```bash
python3 -m venv /tmp/adaptive_jump_model_venv
/tmp/adaptive_jump_model_venv/bin/pip install -r requirements.txt joblib
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. /tmp/adaptive_jump_model_venv/bin/python -m pytest -q -p no:cacheprovider   # 107 passed
/tmp/adaptive_jump_model_venv/bin/python -m ruff check --no-cache src scripts tests
# pipelines (need data/processed/yahoo_cache/ until step 1 above is done):
PYTHONPATH=src:. /tmp/adaptive_jump_model_venv/bin/python scripts/p0_daily_baseline.py   # ~8-10 min
PYTHONPATH=src:. /tmp/adaptive_jump_model_venv/bin/python scripts/p1_adaptive_lambda.py  # ~4 min
PYTHONPATH=src:. /tmp/adaptive_jump_model_venv/bin/python scripts/p2_asymmetric.py       # ~2 min
```

## Non-negotiable rules (repo AGENTS.md)

- No hand-set result-affecting coefficients: CV/walk-forward-selected, MLE-fit, or theory-derived
  only; placeholders must be labelled UNCALIBRATED.
- Do not modify or commit `data/raw`; do not commit `data/processed`.
- Do not claim success without tests and exact command outputs; state what is verified vs inferred.
- Append one line to `.agent/session-log.jsonl` at session end; never rewrite git history.
