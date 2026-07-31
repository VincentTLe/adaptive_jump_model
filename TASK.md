# Active Task: None in progress

## Last completed: HMM baseline closed on sealed v9.4 (2026-07-30)

The HMM replication is as closed as public data allows: 7/8 Table-4 cells inside
the 0.05 tolerance in all three markets at delay 1, on sealed runs that replay
at metric difference 0.0 (`fixed-baselines-34e51cd7…` and its v9.3 twin, which
agree to 0.00e+00 on every metric the drawdown basis cannot touch).

The one open cell, turnover, is closed as a question rather than as a number:

- no single smoothing-candidate set puts it inside tolerance in more than one
  market, and the US and Germany demand opposite ends of the grid — so no grid
  choice reproduces the row (`docs/audit/2026-07-29-codex-review-verdicts.md`
  §15);
- the proximate amplifier is months where the CV selects k = 0 (9.5% of days,
  23% of US turnover), and the residual points at data we cannot obtain: the
  candidate grids are published nowhere (primary-source sweep, 2026-07-29), and
  the official Nikkei TR series does not exist before 1979-12-28 even for the
  paper's own authors;
- Shu's own Figure-6 position path on our returns reproduces their turnover to
  0.002, so the accounting is right and the difference lives in the state
  sequence;
- (round 2, 2026-07-30) the authors' Wolfe Research slides print the
  state-sequence target directly — HMM: 96 shifts, 27.8% bear; JM: 30 shifts,
  19.7% bear — our sealed path has 122 shifts on a 27.95% bear share, and the
  lineage itself (Nystrup 2018, Paper D) published why Sharpe-CV cannot pin
  turnover down: Sharpe is nearly flat across filter windows while turnover is
  not (`docs/audit/2026-07-30-deep-research-round2.md`).

Three data defects were found and fixed along the way (CRSP-for-S&P, missing
German dividends pre-1988, a splice that deleted the 1988-01-04 session), each
now guarded by a test or a build gate.

## Earlier milestone kept for context: holdout-2026-001

The 2024-2026 window returned `not_supported` (0/3) for DD-only against the
stronger control, while the full walk-forward US edge persisted; the window is
short, bull-only, and post-selection (see `research/SCIENTIFIC_LEDGER.md` and
`artifacts/holdout-2026-001.tar.zst`). Its labels were demoted from
"selection-clean" to post-selection readout on 2026-07-29.

## JM baseline closed (2026-07-31), four frozen experiments

The JM row is closed the way the HMM turnover row closed, with stronger
evidence: given the authors' own Figure-5 state sequences, our data and
accounting reproduce Table 4's JM row at 8/8 (us), 8/8 (de), 7/8 (jp) —
turnover deviations 0.001/0.030/0.005; the one miss is the JP drawdown at
0.054. What cannot be regenerated from public information is the state
sequence itself: no published-source lambda grid works
(`jm-grid-identification-001`), the paper's own persistence curve is
bracketed by the exhausted standardizer-cadence family and matched by no
member (`-002`/`-003`; clip-3sigma variant withdrawn per owner — example
code is never an author-method candidate), and their DE/JP sequences are not
expressible by any constant lambda under any tested geometry even though
per-month pieces are (`-004`, effective-lambda inversion; descriptive only,
grid candidacy forbidden). Sổ: `docs/audit/2026-07-30-jm-deep-research.md`,
`2026-07-30-jm-grid-identification.md`, `2026-07-31-jm-standardizer-geometry.md`,
`2026-07-31-jm-effective-lambda-inversion.md`. Reopen only on a new primary
source (Chenyu Yu dissertation, when DataSpace posts it — optional, nothing
blocks on it).

## Next: the extension research

The project's actual goal. Candidate extensions already named in the ledger
(each needs its own frozen question before any code): capped gap, two-day
confirmation, semi-Markov dwell cost, a regime-stratified or longer holdout —
compared against OUR sealed baselines, which is now an honest yardstick. The
k=0 concentration finding and the author-example pipeline notes
(`docs/audit/2026-07-30-self-audit.md` §C7) are recorded inputs to that
design, not licences to fit. Paper writing is the other open milestone
("paper xử lý sau").

## Standing rules

Walk-forward causal everywhere; no confidence intervals; never search an
unspecified knob for the value that best matches the paper; quote the paper
with `[line N] "…"` and run `scripts/check_paper_claims.py`.
