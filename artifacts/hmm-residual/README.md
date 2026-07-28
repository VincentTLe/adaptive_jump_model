# HMM residual investigation, 2026-07-28

Why this directory exists: the v9 readout closed most of the US HMM deviation
but left drawdown, Calmar and turnover open, and the answers had to survive
being asked again a week later. Everything a later question could need lives
here rather than in a scratch directory.

Heavy run output stays out of git (`.gitignore` re-includes only the reports,
the extracted paths and the per-cell tables — 588 KB). The rest rebuilds from
the scripts named below.

| directory | what it holds | rebuilt by |
|---|---|---|
| `v9-us-hmm/` | the v9 US HMM fitted once: states, fits, smoothed candidates, CV surface, monthly choices, selected signal, position path, metrics | `scripts/cache_v9_us_hmm.py` |
| `01-status/` | all three markets against Table 4, under both drawdown bases | `scripts/hmm_status_table.py` |
| `02-turnover-anatomy/` | the fixed-k turnover curve per market, the selector's picks, and the flips manufactured by switching candidate | `scripts/probe_turnover_anatomy.py` |
| `03-table3-anchor/` | shifts per year at fixed k, 1982-2023, against Table 3 | `scripts/probe_table3_anchor.py` |
| `04-figure6-path/` | Shu's own traded positions, read off Figures 5 and 6 | `scripts/extract_paper_positions.py` |
| `05-mdd-anatomy/` | Shu's positions run on our returns; drawdown episodes; day-level disagreement | `scripts/probe_mdd_anatomy.py` |
| `06-mdd-convention/` | four drawdown conventions against ten published cells | `scripts/probe_mdd_convention.py` |
| `07-table5-delays/` | Table 5's nine Calmar cells at delays 1, 5 and 10, as an out-of-sample check on the drawdown basis | `scripts/probe_table5_delays.py` |
| `08-grid-identification/` | eight candidate sets, none adopted: how far grid choice alone moves turnover | `scripts/probe_grid_identification.py` |

`v9-us-hmm/` is a cache, not a sealed run: it fits the HMM alone and borrows
v8.5's comparison sample so the two are scored over identical days. A sealed run
derives its own sample across the jump model too. `run.json` says so, and
nothing here may be cited as a sealed result.

What the directory establishes, in order:

1. Turnover is the only Table 4 metric outside tolerance in all three markets,
   and not with a common sign — we trade more than Shu in the US and Japan and
   less in Germany (`01-status/`).
2. On the S&P 500 our fixed-k persistence curve reproduces Table 3 to a mean
   error of 1.9%, against 7.0% on the CRSP series we had been using
   (`03-table3-anchor/`). The smoother and the state sequence are right.
3. Shu's own position path on our returns gives turnover 1.4123 against the
   published 1.410 and exactly 96 regime shifts (`05-mdd-anatomy/`), so our
   turnover metric is right too and the deviation is entirely in which
   smoothing window the monthly cross-validation picks -- from a candidate set
   the paper never specifies.
4. The same substitution does NOT close the drawdown, which means the drawdown
   gap was never a modelling gap. Table 4's buy-and-hold row and the caption of
   Figure 5 pin the drawdown to a wealth path that is flat while in cash; across
   ten cells that basis cuts the mean drawdown error from 0.0330 to 0.0072 and
   the mean Calmar error from 0.0262 to 0.0055 (`06-mdd-convention/`).
5. On that basis all three markets reach 7/8 within 0.05, each failing on
   turnover alone (`01-status/`), and the turnover row is not identified by the
   paper: across eight candidate sets it spans 1.295-2.913 (us), 1.816-2.432
   (de) and 2.751-4.686 (jp), several times the deviation under investigation
   (`08-grid-identification/`). No set reproduces Table 4 in all three markets
   at once.
