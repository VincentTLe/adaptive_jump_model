"""Table 5 of Shu, Yu and Mulvey (2024) - the trading-delay robustness table.

Transcribed from data/external/inputs/shu_paper.txt, lines 961-975. Only the
left margin is removed; every figure is as printed:

             S&P 500              HMM                        JM
                 B&H    delay = 1     5     10        1        5      10
Return         10.2%         8.5%  8.6%   8.3%    11.2%    11.4%   11.7%
Sharpe          0.48         0.54  0.55   0.51     0.68     0.71    0.70
Calmar          0.16         0.21  0.28   0.25     0.33     0.39    0.28
                 DAX              HMM                        JM
                 B&H    delay = 1     5     10        1        5      10
Return          6.8%         6.4%  5.1%   3.1%     8.6%     7.5%    5.9%
Sharpe          0.30         0.35  0.25   0.12     0.44     0.38    0.29
Calmar          0.09         0.12  0.09   0.04     0.18     0.13    0.11
          Nikkei 225              HMM                        JM
                 B&H    delay = 1     5     10        1        5      10
Return          0.8%         2.5%  1.2%   0.5%     4.7%     4.0%    3.4%
Sharpe          0.12         0.19  0.11   0.07     0.31     0.27    0.24
Calmar          0.04         0.06  0.03   0.02     0.12     0.09    0.07

WHY THIS FILE EXISTS. The delay-1 column of Table 5 duplicates Table 4 and is
therefore IN sample. HELD_OUT below contains delays 5 and 10 only, and nothing
may add delay 1 to it.

WARNING — DELAYS 5 AND 10 ARE NO LONGER HELD OUT FOR ANYTHING DESCENDED FROM
THE v10 RESEAL. This docstring used to claim that "the delay-5 and delay-10
columns were never in any objective function", so that scoring a selected grid
here turned it into a testable prediction. That is false and was retracted by
docs/audit/heldout-delay-001-audit.md, which named this exact docstring as "the
load-bearing false statement". The evidence:

  scripts/probe_jm_grid_exhaustive2.pass_mask()  screens candidate grids
      directly against the Table-5 JM cells at delays 5 and 10, tolerance 0.05
  scripts/gate_v10_reseal.py:89-96               fails the reseal unless every
      delay-5 and delay-10 cell is inside 0.05 in every market

Any grid drawn from those searches is 18/18 inside this screen BY CONSTRUCTION,
so scoring it here is a selection statistic, not a prediction. Registry event:
heldout-delay-001 / PARTIALLY_RETRACTED_AFTER_AUDIT.

An honest held-out test needs an arm that was never screened on Table 5 — a
literature-named grid, for instance — and its own frozen question, written
before any such number is computed.

The caption states the protocol that makes the comparison fair, at line 979:
"A corresponding trading delay is also applied in the cross-validation when
selecting the optimal smoothing hyperparameter." The grid is held fixed; the
cross-validation is re-run at the delay being tested.
"""

from __future__ import annotations

# market -> delay -> metric. Metric keys match adaptive_jump.backtest.
TABLE5_JM: dict[str, dict[int, dict[str, float]]] = {
    "us": {
        1: dict(cagr=.112, sharpe=.68, calmar=.33),
        5: dict(cagr=.114, sharpe=.71, calmar=.39),
        10: dict(cagr=.117, sharpe=.70, calmar=.28),
    },
    "de": {
        1: dict(cagr=.086, sharpe=.44, calmar=.18),
        5: dict(cagr=.075, sharpe=.38, calmar=.13),
        10: dict(cagr=.059, sharpe=.29, calmar=.11),
    },
    "jp": {
        1: dict(cagr=.047, sharpe=.31, calmar=.12),
        5: dict(cagr=.040, sharpe=.27, calmar=.09),
        10: dict(cagr=.034, sharpe=.24, calmar=.07),
    },
}

# The delays no grid search has ever seen. Delay 1 is excluded on purpose.
HELD_OUT: tuple[int, ...] = (5, 10)

# AGENTS.md section 7.1. Frozen before the first held-out number was computed.
GRADE_BANDS: tuple[tuple[str, float], ...] = (
    ("A", 0.02),
    ("B", 0.20),
    ("C", 0.40),
    ("D", 0.60),
)


def grade(relative_deviation: float) -> str:
    """A/B/C/D/F for one cell. A non-finite deviation is an F, never an A."""
    if not (relative_deviation >= 0.0):  # NaN reaches here and must not pass
        return "F"
    for letter, ceiling in GRADE_BANDS:
        if relative_deviation <= ceiling:
            return letter
    return "F"
