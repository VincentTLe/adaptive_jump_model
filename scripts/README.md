# Scripts

Around sixty programs live here. **They are not a menu of things to run.** Most
are one-off programs written for a specific study that has since finished; they
are kept so that study's sealed numbers stay reproducible, not because anyone
should re-run them today.

Nothing here is part of the fixed-JM pipeline. That is
[`src/adaptive_jump/`](../src/adaptive_jump/README.md); the normal way to run
the baseline is the `adaptive-jump` CLI, not a script.

## Before you run anything

Ask two questions first:

1. **Which frozen contract does this script belong to?** Experiment runners read
   a `research/*.toml`. Find that file and read its scope.
2. **Is that experiment still open?** Check [`../CURRENT.md`](../CURRENT.md) and
   `research/experiment_registry.jsonl`. Several directions here are closed and
   explicitly not reopened — re-running one does not reopen it.

Scripts that write into `artifacts/` produce sealed evidence. Do not launch one
casually.

## Families, by name prefix

- **`build_*`, `fetch_*`** — data reconstruction and source acquisition
  (`build_sp500_tr.py`, `build_de_total_return.py`,
  `build_external_sources.py`, `fetch_*`). These built the pinned files under
  `data/external/`, whose hashes the config already records. Re-running them is
  a data-provenance action, not a research action.
- **`audit_*`, `verify_*`, `check_*`, `validate_*`, `gate_*`** — independent
  verification: recompute a sealed result, check a published claim, or run a
  frozen gate. The safest family to read.
- **`run_*`** — runners for frozen experiments (`run_simple_jm_suite_003.py`,
  `run_frequency_ladder.py`, `run_lagged_capguard.py`, ...). Each pairs with a
  `research/*.toml`. `run_fast.sh` is a convenience wrapper, not an experiment.
- **`probe_*`, `diagnose_*`** — research diagnostics: grid searches, episode
  anatomy, disagreement analysis. Largely historical. Many were written to
  answer a question that has since been answered, and several document
  *negative* results.
- **`stress_*`** — higher-`n_init` robustness checks on a sealed config.
- **`optimizer_fidelity_*`, `studentized_sharpe_difference.py`,
  `plot_paired_stability.py`** — the recent measurement tools behind the
  optimizer-nonuniqueness and paired-delta work described in `CURRENT.md`.
- **`render_replication_atlas.py`, `sealed_v9_table.py`, `score_grid.py`,
  `freeze_002_spec.py`, `_shu_table4.py`, `_shu_table5.py`** — presentation and
  reference helpers. The `_shu_table*.py` files hold the published Shu et al.
  tables transcribed once, so every probe compares against the same numbers.

Nothing above is a complete listing, and a prefix is a hint rather than a
guarantee. When in doubt, read the script's own module docstring — all 61
Python scripts here have one, and it names the study.
