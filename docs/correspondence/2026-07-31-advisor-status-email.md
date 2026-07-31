# Email to advisor before the 2026-07-31 meeting

---

Dear Professor,

Ahead of this afternoon's meeting, here is a full status summary of my replication of Shu, Yu & Mulvey (2024), "Downside risk reduction using regime-switching signals: a statistical jump model approach" (Journal of Asset Management; arXiv:2402.05272): the data reconstruction, the HMM results, the current state of the JM, and the central obstacle — the unpublished λ candidate grid.

## 1. Data replacement (the paper uses Bloomberg + GFD; I rebuilt everything from free public sources)

- **S&P 500 Total Return**: the official ^SP500TR series from 1988, spliced onto a reconstructed pre-1988 segment; one splice bug (a deleted 1988-01-04 session) was caught and fixed, with a regression test guarding it.
- **DAX**: natively a total-return index; the pre-1988 segment follows the Stehle backcast lineage, cross-checked against OECD monthly data.
- **Nikkei 225 TR**: the official TR index only exists from 1979-12-28 (and was only published from 2012) — which means the authors' own 1970-79 segment must itself be a constructed series; I built ours from JST Macrohistory + the official series + a bridge over the 2020-22 gap.
- **Risk-free rates**: US 3-month T-bill (FRED DTB3, verified byte-identical against a manual download), a Bundesbank-based scale for Germany, an IMF/BoJ scale for Japan.
- Every run is "sealed": config + data manifest + git commit are hashed; replays reproduce with exactly 0.0 deviation.

## 2. HMM baseline — effectively a successful replication

On the sealed run: **7 of 8 Table-4 cells within the 0.05 tolerance in ALL THREE markets** (delay 1). Several cells are near-exact (German volatility is off by 0.00004). The single failing cell is turnover (ours 1.79/2.26/3.14 vs their 1.41/2.46/2.90), and the cause has been traced to the root: (i) the smoothing parameter k is itself selected from an unpublished candidate grid; (ii) validation Sharpe is nearly flat in k — a fact recorded in the very literature they cite (Nystrup's 2018 thesis) — so the hyperparameter selection is intrinsically unstable in turnover, as a matter of principle; (iii) the authors' own conference slides print "96 regime shifts, 27.8% bear days" for the US — I get 122 shifts / 27.95%: state occupancy matches, and the entire excess is short-lived flips.

## 3. JM — the part that does not yet match, and evidence that my system is not the cause

The decisive test: **the regime shading in the paper's Figure 5 is lossless vector data — I extracted the exact daily state sequences the authors ran** (they match the numbers printed on each panel: 30/116/48 shifts, 19.7%/15.7%/25.3% bear). Applying THEIR sequences to MY data **reproduces 23 of 24 cells of Table 4's JM row** (turnover deviations 0.001/0.030/0.005; Sharpe deviations 0.008/0.003/0.023). So my data, cost accounting, and trading conventions are all correct — the entire residual sits in *regenerating those state sequences myself*, which depends on two things the paper does not publish: the λ candidate grid and the exact feature-standardization recipe.

My own independent run currently reaches 4/8 (US), 3/8 (DE), 3/8 (JP) — and I have run a chain of frozen experiments (question + pass criteria registered before running, with adversarial audits by independent agents) to characterize this gap.

## 4. Why the λ grid matters so much (short mechanism explanation)

The JM is essentially temporal k-means with a switching penalty: each day is assigned to a bull/bear state so as to minimize "feature distance + λ × (number of state changes)". **λ is the persistence knob of the state sequence**: λ=0 gives ~9.7 regime shifts per year, λ=150 only ~0.4 (the paper's own Table 3). The strategy then re-selects λ̂ at the end of each month **from a candidate grid**, by Sharpe ratio on an 8-year validation window. The grid therefore defines the "menu of persistence levels" the selection procedure is allowed to use — and because validation Sharpe is nearly flat across candidates, the selection is very sensitive to grid membership. On the SAME data with the SAME code, changing only the grid moves my German Sharpe from 0.13 to 0.49 and my US turnover from 0.24 to 0.82. The grid is not an implementation detail — it is a result-governing parameter, and the paper says only that λ is chosen from a set of candidate values, without ever naming the set.

## 5. Other groups' λ grids (nobody outside the author team knows the real one)

- **The authors themselves, arXiv v1**: once published the grid {10, 22, 50, 100, 220, 500, 1000} — then WITHDREW it in v3 (the protocol changed) and never published the new grid. Shu's 2025 Princeton PhD thesis (I have the full text) likewise says only "a list of candidate jump penalties".
- **Li, Chen, Tao & Ji 2025** (Mathematics 13(17):2837, cites the paper): chose their own grid {0, 5, 10, 25, 50, 100}, selected by information criteria — and λ piles up at the upper boundary for all 12 assets.
- **Bocconi student group** (B&S Capital Markets): chose {0, 5, ..., 150}, S&P only, a different window — not comparable to Table 4.
- **Community notebook (HackMD)**: chose {0, 0.1, 1, 10, 100}, on the DJIA.
- I machine-scanned **all 14 citing papers** on Semantic Scholar: **no independent replication of the three-market protocol has published results matching Table 4** — every independent implementation had to invent its own grid, because the real one exists nowhere in public.

## 6. Reverse-engineering and exhaustive search — latest results (last night / this morning)

- Estimating the grid from the authors' own monthly choices (read out of Figure 5): their US grid lies around the range **[10, 100], with mass near ~35**.
- **Exhaustive sweep of 6,474,511 grid combinations** (every subset of size 2-8 of 29 sourced λ values), nine arms (3 markets × delays 1/5/10; the long delays scored against Table 5), with the evaluator verified at machine precision and every named result re-scored through the original, unaccelerated pipeline:
  - **US: grids matching Shu on ALL 14 cells exist** (8 Table-4 cells at delay 1, plus 3 Table-5 cells at each of delays 5 and 10). Example {0, 21.5, 70}: worst deviation 0.011 at delay 1, 0.023 at delay 5, 0.014 at delay 10. There are 36,657 such grids.
  - **DE/JP: the best attainable is 13 of 14 cells** — DE has 366 grids and JP has 2,948 reaching 7/8 at delay 1 plus all six Table-5 cells at BOTH delay 5 and delay 10 (revalidated through the original pipeline: DE {150, 500} and JP {10, 220} each PASS 6/6 arms; the single failing cell is exactly the one the frontier analysis predicts — turnover for DE, leverage for JP). No grid in the 29-value menu reaches all 8 delay-1 cells, and the frontier analysis identifies the blocking cell precisely: for DE it is turnover (any path that keeps the other 7 cells trades at most 1.47 vs the target 1.70 — a joint constraint, not missing λ coverage, since the unconditional turnover span reaches 4.6); for JP it is leverage (ceiling 0.74 vs target 0.75 across the whole lattice, and only 0.64 once the other 7 cells pass — my series is intrinsically more "bearish" than theirs). I stress the scope of these statements: *within the current menu*; the next step is to try a denser / real-valued λ menu for DE/JP to rule out menu narrowness, although the frontier evidence points at the shape of the state sequences (driven by the standardization geometry — the paper says only "standardized") rather than at λ coverage.
- One honesty note on the winning grids: they are small (2-3 values), and that is NOT evidence that they are the authors' grid. Having searched 6.47 million subsets against 14 published target numbers, finding matches is expected by construction — which is precisely why every such grid is labeled a **calibration artifact** in the frozen specs, never "the authors' grid". Their value is different: they give a fixed, honest, fully documented baseline for the extension phase, and extension-vs-baseline deltas remain internally valid because both sides share the same frozen grid.

## 7. Status of the JM extension experiments — previously run, but on the earlier proxy pipeline

A correction to what I reported before: the extension experiments are **not unrun**. In the earlier phase of this project (the public-proxy era, before the current replication-grade rebuild), I ran two complete frozen extension studies and wrote them up as working papers (in `paper/split-v1/`):

- **"One Risk Measure, Fewer Trades"** — a frozen five-challenger suite: a DD-only JM (single downside-deviation feature), static λ=50, two-observation confirmation, a return-aware training loss, and a robust L1 loss, plus a prespecified 3× loss-scale control. Headline: on the US proxy, the DD-only JM raised net Sharpe from 0.570 to 0.908, cut maximum drawdown from -33.9% to -19.4%, halved position changes (21 → 11), and beat both buy-and-hold and the HMM; the result did not extend to Germany or Japan.
- **"When Should a Jump Model Switch?"** — evidence-adaptive transition costs with exact dynamic programming (arrival-day, lagged, and pair-balanced rules, all nesting the fixed JM exactly at β=0). Headline: lagged evidence (β=log4) cut discount-attributable whipsaws from 17 to 6 pooled across markets and improved Sharpe over the fixed JM in all three markets (+0.017 US, +0.171 DE, +0.084 JP), but beat both economic benchmarks only in Germany and made Japan far more active (13 → 33 switches); pair-balancing was not an overall improvement.

Why these must be rerun rather than cited as-is: they used **different data** (Yahoo price proxies — including a dividend-free Nikkei — with evaluation windows starting Dec 2007 / Jan 2008 / May 2009, not 1990-2023), a **different λ grid** ({0, 5, 15, 35, 70, 150, 300, 600, 1200}), and a baseline JM that did not reproduce Shu's ordering in any market. Their conclusions are therefore about the proxy setting, not about Shu's setting. Rerunning them against the current replication-grade data requires a sealed baseline first — which is the pending decision: whether to "reseal" the JM baseline with the calibrated grids (US has a full 14/14 grid; DE/JP would use the best-13/14 grids with the blocking cell documented), labeled transparently as calibration rather than as the authors' grid, so the extension results have a clean reference point.

## Reference links

- Paper (v3): https://arxiv.org/abs/2402.05272 · v1 with the later-withdrawn grid: https://arxiv.org/abs/2402.05272v1
- Shu's PhD thesis (full text, free): https://dataspace.princeton.edu/handle/88435/dsp01g158bm716
- Shu's Wolfe Research slides (printing the 96-shift / 30-shift figures): https://drive.google.com/file/d/1-8a9GzfyDELUIq0rq7NF2iqmyMikCmGr/view
- Official package (contains no grid / no CV code): https://github.com/Yizhan-Oliver-Shu/jump-models
- Li et al. 2025: https://www.mdpi.com/2227-7390/13/17/2837
- Bocconi Students Capital Markets: https://www.bscapitalmarkets.com/statistical-jump-models-for-regime-switching.html
- HackMD notebook (DJIA): https://hackmd.io/@e41406/HkUKkKTpR

All experiments, the registry, and the audit trail live on branch `cleanup/research-protocol` (registry: `research/experiment_registry.jsonl`; audit notes: `docs/audit/`). I am happy to walk through any item in detail at the meeting.

Best regards,
Tan
