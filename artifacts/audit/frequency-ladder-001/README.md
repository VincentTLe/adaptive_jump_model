# Evidence for docs/audit/frequency-ladder-001-audit.md

Produced by the independent auditor of `frequency-ladder-001`. Nothing here was
written by the runner, the map script or the scorer under audit, and nothing
here modifies them.

| file | what it is |
|---|---|
| `recomputed-cells.csv` | all 48 cells recomputed from the library, joined to the committed `artifacts/frequency-ladder/01-run/cells.csv`; `abs_gap` is the disagreement (max 9.4e-17) |
| `realised-frequency.csv` | 312 refits: the in-sample jump count actually produced by each derived penalty on each pre-1990 window, i.e. the measured bias of the inversion (F-3) |
| `refit-parity-de.csv` | 58 refits of two German pre-1990 windows at all 29 union penalties, against the recorded objectives — proves `union-refits.csv` records the penalised objective |
| `jp-1989-07-03-refit-repair.txt` | the defective Japanese contributing window refitted with `n_init=60`; two penalties were suboptimal, only one of them detectably (F-1) |
| `ladder-sensitivity.csv`, `ladder-menus.csv` | six ladders × two arms × three markets. **SENSITIVITY ANALYSIS ONLY** — see the warning in the report; these must not be used to select a ladder |
| `states-{us,de,jp}.csv` | the ladder run's states plus the 26 extra penalties the alternative ladders needed. Regenerable; gitignored |
| `audit_recompute.py` | independent re-derivation of the menu and independent scoring of the cells |
| `audit_realised_frequency.py` | produced `realised-frequency.csv` |
| `audit_ladder_sensitivity.py` | produced `ladder-sensitivity.csv` |
| `fault_injection.py` | the fault harness: 14 single-edit faults on **copies** of the runner and the frozen spec |
| `verify_tests_catch_faults.py` | replays the silent spec-level faults against the new tests to confirm they now fail |

## Running them again

These were run from a scratch directory, and three of them write their outputs
to the paths listed inside them. To re-run, copy to a scratch directory first so
that nothing writes into a committed artifact:

```
SCRATCH=$(mktemp -d)
cp artifacts/audit/frequency-ladder-001/*.py "$SCRATCH"/
unset VIRTUAL_ENV && uv run python "$SCRATCH"/audit_recompute.py
unset VIRTUAL_ENV && uv run python "$SCRATCH"/fault_injection.py
```

`fault_injection.py` creates one directory per fault beside itself, copies the
runner and the frozen spec into it, applies a single edit and runs it. The
`control` directory must reproduce
`artifacts/frequency-ladder/01-run/summary.csv` byte for byte; if it does not,
the harness is not faithful and its verdicts mean nothing.

The regression tests that pin the findings need no scratch directory:

```
unset VIRTUAL_ENV && uv run python -m pytest tests/test_frequency_ladder_audit.py -q -rx
```
