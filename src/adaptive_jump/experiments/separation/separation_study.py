"""Shared terminal-decision core for the evidence-penalty studies.

This module was the mathematical core of adaptive-separation-001. Only the
pieces the arrival, lagged and balanced harnesses import survive here --
the exact terminal state, its exact last-step predecessor, and the local
arrival ablation. The separation study's own frozen-spec loader and its
logistic prediction machinery are gone with the study; nothing here pins an
experiment id, a run id or a lambda grid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from adaptive_jump.tv_jump import dp_tv


class SeparationStudyError(ValueError):
    """Raised when a shared decision invariant fails."""


@dataclass(frozen=True)
class TerminalDecision:
    state: int
    predecessor: int
    state_margin: float
    predecessor_margin: float
    state_tied: bool
    predecessor_tied: bool


def _argmin_with_margin(values: np.ndarray) -> tuple[int, float, bool]:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or len(vector) < 2 or not np.isfinite(vector).any():
        raise SeparationStudyError("terminal decision vector is invalid")
    minimum = float(np.nanmin(vector))
    state = int(np.nanargmin(vector))
    tied = int(np.count_nonzero(vector == minimum)) > 1
    ordered = np.sort(vector[np.isfinite(vector)])
    margin = float(ordered[1] - ordered[0]) if len(ordered) > 1 else math.inf
    return state, margin, tied


def terminal_decision(loss: np.ndarray, penalty_seq: np.ndarray) -> TerminalDecision:
    """Return the terminal online state and its exact last-step predecessor."""
    losses = np.asarray(loss, dtype=float)
    penalties = np.asarray(penalty_seq, dtype=float)
    if losses.ndim != 2 or len(losses) < 2:
        raise SeparationStudyError("terminal attribution needs at least two rows")
    values = dp_tv(losses, penalties, return_value_mx=True)
    state, state_margin, state_tied = _argmin_with_margin(values[-1])
    predecessor_cost = values[-2] + penalties[-1, :, state]
    predecessor, predecessor_margin, predecessor_tied = _argmin_with_margin(
        predecessor_cost
    )
    return TerminalDecision(
        state=state,
        predecessor=predecessor,
        state_margin=state_margin,
        predecessor_margin=predecessor_margin,
        state_tied=state_tied,
        predecessor_tied=predecessor_tied,
    )


def arrival_ablation_state(
    loss: np.ndarray, penalty_seq: np.ndarray, lambda0: float
) -> int:
    """Reset only the final arrival costs to fixed lambda and emit the state."""
    losses = np.asarray(loss, dtype=float)
    penalties = np.asarray(penalty_seq, dtype=float)
    if penalties.ndim != 3 or penalties.shape[:2] != losses.shape:
        raise SeparationStudyError("ablation loss and penalty shapes differ")
    if not np.isfinite(lambda0) or lambda0 < 0:
        raise SeparationStudyError("ablation lambda must be finite and nonnegative")
    ablated = penalties.copy()
    n_states = losses.shape[1]
    ablated[-1] = lambda0 * (1.0 - np.eye(n_states))
    return int(dp_tv(losses, ablated, return_value_mx=True)[-1].argmin())
