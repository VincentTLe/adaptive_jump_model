# JM deep-research (2026-07-30) — the dissertation, the package pins, and what they fix

Context: the JM phase is opening. Two workflow sweeps ran (verifier panels
again died on a spend limit; every claim retained below was verified by hand
against the downloaded primary source — the same caveat and the same stronger
manual check as the round-2 note). Sources archived in-repo with hashes.

## J1. Shu's 2025 Princeton dissertation is public, and it pins the JM protocol — but still not the grid

"Modeling Regime Changes in Financial Markets Using Statistical Jump Models:
Methodology and Applications" (Princeton ORFE, advisor Mulvey; DataSpace
handle 88435/dsp01g158bm716, behind an ALTCHA wall, fetched via headless
browser). Chapter 3 "is based on the article by Shu, Yu, and Mulvey 2024a" —
the paper under replication. Hand-verified:

- **Features, fixed.** Table 3.2 and the surrounding text: "an exponentially
  weighted moving (EWM) downside deviation (DD) with a halflife of 10 trading
  days, and EWM Sortino ratios with halflives of 20 and 60 days", identical
  across all three indices; DD defined as sqrt of the EWM mean of R²·1{R<0}.
  This matches our `dd_10`, `sortino_20`, `sortino_60` exactly, and supersedes
  the example-notebook halflives [5, 20, 60] (different dataset) as the
  authoritative recipe for this paper.
- **Standardization, still unfixed.** The features are only ever called
  "standardized" — the dissertation adds nothing to the paper. Row 1 of
  docs/unspecified-choices.md stands unchanged.
- **CV protocol, confirmed word-for-word with the paper.** Monthly λ
  re-selection maximizing the 0/1 strategy's validation Sharpe on
  online-inferred regimes over an 8-year lookback, 10bp one-way (footnote 11),
  one-day delay, λ̂ applied at t+2, 12-year training + 8-year validation from
  1970 ⇒ OOS 1990-2023. Footnote 12 explicitly recommends selecting on
  online-inferred (not in-sample) regimes. All already implemented.
- **The λ candidate grid is still unpublished.** The Chapter-3 algorithm input
  reads "A list of candidate jump penalties." — no values, anywhere in the
  dissertation. The 0-100 log-spaced grid belongs to Chapter 4 (the companion
  paper's different, biannual protocol). The grid's status is therefore
  identical to the HMM k grid: procedure published, candidate set not.
- **Single λ per candidate, in fit AND online inference.** Chapter 3's online
  inference minimizes objective (3.1) — which carries the candidate's own λ —
  with parameters frozen at Θ̂ over an l = 3000 lookback, taking the last
  state. No separate out-of-sample penalty exists in the JAM protocol.
- **Table 3.4 = Table 4**, digit for digit (hand-checked: JM Sharpe
  0.68/0.44/0.31, JM turnover 44%/170%/72%, leverage 80%/84%/75%).

Archived: `data/external/inputs/shu-princeton-dissertation-2025.pdf`, sha256
`af6301c4b626217eb46d29d55efb78b1494078eb00c2b68ec5a3e9c72cd3759d`.

## J2. The April-2024 deck: λ_is vs λ_oos is a real author idea — but not this paper's protocol

Shu's QWAFAFEW × NEW deck (2024-04-22) states "In-sample and out-of-sample
optimal jump penalties need not be the same." and its Nasdaq demo uses
λ_is = 100.0 with λ_oos = 20.0. Recorded as a **hypothesis input for future JM
extensions only**: Chapter 3 (J1) uses a single λ, so this must never be
imported into the replication. The deck also restates the frozen-Θ,
last-state online-inference convention — a second primary source for it.

Archived: `data/external/inputs/shu-qwafafew-2024-04-22-slides.pdf`, sha256
`4be6671a6ddb19e85c5c698f3e665bc6834b9c996dcb68cecdc279d14c084e0c`.

(Housekeeping: an extraction agent flagged a "contradiction" over the Wolfe
slides' JM anchors — it had fetched the *October-2023 QES webcast* deck, which
indeed has no anchors. The anchors (19.7% bear, 30 shifts) are printed in the
*October-2024 8th Annual conference* deck archived on 2026-07-30 with hash.
Two different decks; no contradiction.)

## J3. jumpmodels package conventions our replication inherits (fidelity by construction)

From the package source (github.com/Yizhan-Oliver-Shu/jump-models, the
`jumpmodels` PyPI package we call): per-observation loss is 0.5 × squared
Euclidean distance (any λ is only meaningful under this scale); constant
off-diagonal penalty matrix; state labels sorted by cumulative return
(`sort_by="cumret"`); defaults n_components=2, n_init=10 k-means++ restarts
from one seeded stream, max_iter=1000, tol=1e-8, RANDOM_STATE=0; coordinate
descent stops on repeated clustering / tolerance / max_iter; a restart wins
only with a strictly lower objective AND a permutation-distinct clustering.
We call this package directly with n_init=10, random_state=0, max_iter=1000,
tol=1e-8 (config `[jm]`), so these conventions hold in our runs by
construction rather than by reimplementation.

## J4. Speed facts that shaped the parallelization

- The package's online-inference DP is a pure-Python loop over the window's
  rows (the author documents that Numba cannot easily accelerate it) — this
  is why one day of inference costs ~54ms against ~0.2ms of scaling.
- Bemporad et al. (2018) Algorithm 3 gives an *exact* recursive online
  inference — but its arrival-cost recursion assumes a fixed window start.
  Ours slides (l = 3000, both endpoints move), which breaks the anchor, so no
  published exact warm-start applies to this protocol as specified.
- The dissertation itself points at "Nystrup et al. (2020a) … a
  computationally efficient method for performing the above calculations for
  a range of consecutive days" — recorded as a lead for a future *algorithmic*
  speedup; not pursued now because the parallel day-fan-out already reaches
  the machine's core count without touching any numerical path.

**Result of the parallelization** (`fixed_jm_states(..., n_jobs=N)`, same
schedule and numerics as the serial loop, refits and day-inference fanned over
processes exactly like the existing HMM `n_jobs`): golden verification
recomputed all three markets from the sealed v9.4 feature frames with
n_jobs=30 and diffed against the sealed `jm-states.csv` artifacts —
**0 differing cells** (us 0/64,338; de 0/64,026; jp 0/62,052, NaN masks
identical), at 1.4 min per market against ~19 min serial, a ~13× wall-clock
reduction with bit-identical output. Unit tests additionally pin
parallel-equals-serial on states, refits, and every checkpoint snapshot, and
resume-from-checkpoint under n_jobs>1.

## J4a. Third sweep (same day): the source space is now mined out

A follow-up sweep chased the four remaining angles. This one got partial
external verification before the spend limit hit (7 claims at 3-0/2-0);
the rest were checked by hand where it mattered. What it settled:

- **Single-λ is now conclusively the JAM protocol**, from four independent
  directions: the v3 text ("select the value λ̂ that yields the highest Sharpe
  ratio … and use this value for the following month", verified 2-0), the
  dissertation (J1), the package's Nasdaq demo (same instance refit at λ=50
  then `predict_online`, no split anywhere, 2-0), and the repo's entire issue
  tracker (9 issues, zero mention of a second penalty or any grid, 3-0). The
  April-deck λ_is/λ_oos idea (J2) originates in the *slides'* Nasdaq
  illustration, not in the package demo and not in any protocol text.
- **Two-cadence protocol confirmed at the source** (3-0): Θ̂ refits every six
  months on the 3000-day window while λ re-selects monthly — exactly our
  `refit_months = [1, 7]` + monthly selection, already sealed.
- **No third-party replication of the three-market protocol exists.** The
  complete Semantic Scholar citation record (14 papers as of 2026-07-30) was
  machine-scanned: zero hits for DAX/Nikkei/replication/turnover angles. The
  two independent backtests found chose their own grids and windows — a
  Bocconi student group (S&P only, grid {0,5,10,20,30,40,50,60,80,100,150},
  λ=80 picked on OOS Sharpe, i.e. look-ahead; Sharpe 0.93 vs 0.77 B&H on
  2016-2025) and a HackMD DJIA notebook (grid [0, 0.1, 1, 10, 100]). Neither
  bears on the paper's grid; both illustrate that every implementer invents
  one.
- **The one remaining primary-source hope for the λ grid is not yet
  retrievable**: co-author Chenyu Yu defended a Princeton ORFE PhD on
  2025-11-05 ("From Hybrid to End-to-End: Frameworks for Decision-Focused
  Financial Regime Modeling", committee includes Mulvey). The ORFE DataSpace
  collection was browsed newest-first on 2026-07-30: the dissertation is not
  deposited yet (collection currently ends at the 2025 cycle). **Re-check
  DataSpace around late 2026.** Two further Yu papers (SSRN 5235747, JPM 2026;
  SSRN 5817083 "Deep Statistical Jump Models") are Cloudflare-gated and are
  factor-allocation work — at abstract level they document a *different*,
  performance-driven λ-selection philosophy, not the JAM CV protocol.

Bottom line of the three sweeps: every author artifact that exists has been
opened; the λ grid is published in none of them; the protocol around it is
now pinned at the primary source in every other respect and matches what we
sealed. The grid remains OUR construction by necessity, exactly like the HMM
k grid — and the two third-party grids found in the wild are evidence that
this is the field's condition, not this repository's defect.

## J5. Where the sealed v9.4 JM stands against Table 4 (the investigation's starting line)

From the sealed run's exploratory metrics, delay 1, `total_wealth` basis —
these are the numbers the JM investigation starts from, NOT a claim of
replication:

| cell | US | DE | JP |
|---|---|---|---|
| within 0.05 | 4/8 | 3/8 | 3/8 |
| Sharpe | 0.754 vs 0.68 | 0.302 vs 0.44 | 0.197 vs 0.31 |
| turnover | 0.500 vs 0.44 | 1.172 vs 1.70 | 1.406 vs 0.72 |

Notable structure: the US JM *overshoots* Sharpe by roughly what Germany
undershoots; Japanese turnover is double the target while German turnover is
two-thirds of it. Each candidate explanation (grid, standardizer geometry,
λ selection instability) needs its own frozen question before any code
changes — the standing rules apply unchanged.
