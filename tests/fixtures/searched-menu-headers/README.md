# Header lines of the searched lambda menus

`tests/test_score_grid_audit.py` asserts one property of the penalty menus that
the grid searches actually ran over: no two of their columns are close enough to
be confused by the scorer's column matcher. That property lives entirely in the
column names, i.e. in the first line of each file.

The menus themselves are multi-megabyte daily state matrices that `.gitignore`
keeps out of the repository, so a clean checkout does not have them and the test
had no way to run. Each file here is the **first line, byte for byte**, of the
corresponding local artifact, copied with `head -n 1`. Nothing was computed,
rounded or invented, and no model was refitted to produce them.

| File here | Copied from | Produced by |
| --- | --- | --- |
| `union-states.csv` | `artifacts/jm-residual/01-grid-identification/us/union-states.csv` | `jm-grid-identification-001` |
| `states-de.csv` | `artifacts/dense-menu/01-search/states-de.csv` | `dense-menu-exhaustive-001` |
| `states-jp.csv` | `artifacts/dense-menu/01-search/states-jp.csv` | `dense-menu-exhaustive-001` |

These are stand-ins, not a second source of truth. On any machine that does have
the artifact, the test re-checks that the copy is still identical to the
artifact's first line, so a regenerated menu cannot leave a stale header here
silently passing.
