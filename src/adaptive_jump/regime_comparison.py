"""Pure metrics for comparing two binary regime paths on a shared calendar.

Conventions, shared with the digitized Figure-5 artifacts: state 1.0 = bear;
switch direction +1 = entering bear; lag = our switch position minus theirs
on the joint trading calendar, so positive lag means we switch later. The
daily agreement rate is the Harding-Pagan concordance index; switch F1 with
a tolerance margin and segmentation covering follow van den Burg & Williams
(2020). Everything here is I/O-free and fits nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class RegimeComparisonError(ValueError):
    """Raised when a path or an era partition violates the comparison contract."""


def _validated(path: pd.Series) -> pd.Series:
    values = path.dropna()
    if not values.isin([0.0, 1.0]).all():
        raise RegimeComparisonError("path values must be 0, 1, or missing")
    if not values.index.is_monotonic_increasing or not values.index.is_unique:
        raise RegimeComparisonError("path index must be sorted and unique")
    return values.astype(float)


def _aligned(theirs: pd.Series, ours: pd.Series) -> tuple[pd.Series, pd.Series]:
    a = _validated(theirs)
    b = _validated(ours)
    joint = a.index.intersection(b.index)
    if len(joint) == 0:
        raise RegimeComparisonError("no shared dates to compare")
    return a.loc[joint], b.loc[joint]


def _segments(values: pd.Series) -> list[np.ndarray]:
    """Maximal constant-regime runs, as arrays of joint-calendar positions."""
    array = values.to_numpy()
    boundaries = np.flatnonzero(np.diff(array) != 0.0) + 1
    return np.split(np.arange(len(array)), boundaries)


def concordance(theirs: pd.Series, ours: pd.Series) -> float:
    """Harding-Pagan concordance: share of shared dates in the same regime."""
    a, b = _aligned(theirs, ours)
    return float((a.to_numpy() == b.to_numpy()).mean())


def switches(path: pd.Series) -> pd.DataFrame:
    """Regime flips as (date, direction); +1 enters bear, -1 exits."""
    values = _validated(path)
    if values.empty:
        return pd.DataFrame(columns=["date", "direction"])
    flips = values.diff().dropna()
    events = flips[flips != 0.0]
    return pd.DataFrame(
        {"date": events.index, "direction": events.astype(int).to_numpy()}
    ).reset_index(drop=True)


def match_switches(
    theirs: pd.Series, ours: pd.Series, *, margin: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Greedy 1-1 pairing of same-direction switches within ``margin`` days.

    Candidate pairs are sorted by (|lag|, their date, our date) and accepted
    only while both endpoints are unused, so an exact pair always beats a
    distant one and no switch is matched twice.
    """
    a, b = _aligned(theirs, ours)
    position = {date: pos for pos, date in enumerate(a.index)}
    their_events = switches(a)
    our_events = switches(b)

    candidates = []
    for i, their in their_events.iterrows():
        for j, our in our_events.iterrows():
            if their["direction"] != our["direction"]:
                continue
            lag = position[our["date"]] - position[their["date"]]
            if abs(lag) <= margin:
                candidates.append((abs(lag), their["date"], our["date"], i, j, lag))
    candidates.sort(key=lambda entry: (entry[0], entry[1], entry[2]))

    used_theirs: set[int] = set()
    used_ours: set[int] = set()
    rows = []
    for _, their_date, our_date, i, j, lag in candidates:
        if i in used_theirs or j in used_ours:
            continue
        used_theirs.add(i)
        used_ours.add(j)
        rows.append(
            {
                "their_date": their_date,
                "our_date": our_date,
                "direction": int(their_events.loc[i, "direction"]),
                "lag": int(lag),
            }
        )
    matches = pd.DataFrame(rows, columns=["their_date", "our_date", "direction", "lag"])
    if not matches.empty:
        matches = matches.sort_values("their_date").reset_index(drop=True)
    unmatched_theirs = their_events[~their_events.index.isin(used_theirs)].reset_index(
        drop=True
    )
    unmatched_ours = our_events[~our_events.index.isin(used_ours)].reset_index(
        drop=True
    )
    return matches, unmatched_theirs, unmatched_ours


def switch_f1(theirs: pd.Series, ours: pd.Series, *, margin: int) -> dict[str, float]:
    """Change-point precision/recall/F1 under the matching above."""
    matches, unmatched_theirs, unmatched_ours = match_switches(
        theirs, ours, margin=margin
    )
    n_theirs = len(matches) + len(unmatched_theirs)
    n_ours = len(matches) + len(unmatched_ours)
    if n_theirs == 0 and n_ours == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    precision = len(matches) / n_ours if n_ours else 0.0
    recall = len(matches) / n_theirs if n_theirs else 0.0
    denominator = precision + recall
    f1 = 0.0 if denominator == 0.0 else 2.0 * precision * recall / denominator
    return {"precision": precision, "recall": recall, "f1": f1}


def covering(gt: pd.Series, pred: pd.Series) -> float:
    """Segmentation covering: length-weighted best Jaccard per gt segment."""
    a, b = _aligned(gt, pred)
    score = 0.0
    pred_segments = _segments(b)
    for segment in _segments(a):
        start, stop = segment[0], segment[-1]
        best = 0.0
        for other in pred_segments:
            intersection = min(stop, other[-1]) - max(start, other[0]) + 1
            if intersection <= 0:
                continue
            union = max(stop, other[-1]) - min(start, other[0]) + 1
            best = max(best, intersection / union)
        score += len(segment) * best
    return score / len(a)


def disagreement_decomposition(theirs: pd.Series, ours: pd.Series) -> pd.DataFrame:
    """Classify every disagreement day as timing, missing, or extra.

    A disagreement day is bear in exactly one path. It is ``timing`` when its
    bear episode overlaps some bear episode of the other path (a shifted or
    truncated boundary), ``missing`` when their episode has no counterpart in
    ours, and ``extra`` when our episode has no counterpart in theirs.
    """
    a, b = _aligned(theirs, ours)

    def bear_episodes(values: pd.Series) -> list[np.ndarray]:
        return [
            segment for segment in _segments(values) if values.iloc[segment[0]] == 1.0
        ]

    def overlaps(segment: np.ndarray, episodes: list[np.ndarray]) -> bool:
        return any(
            segment[0] <= other[-1] and other[0] <= segment[-1] for other in episodes
        )

    their_episodes = bear_episodes(a)
    our_episodes = bear_episodes(b)
    rows = []
    for pos in np.flatnonzero(a.to_numpy() != b.to_numpy()):
        if a.iloc[pos] == 1.0:
            episode = next(e for e in their_episodes if e[0] <= pos <= e[-1])
            kind = "timing" if overlaps(episode, our_episodes) else "missing"
        else:
            episode = next(e for e in our_episodes if e[0] <= pos <= e[-1])
            kind = "timing" if overlaps(episode, their_episodes) else "extra"
        rows.append({"date": a.index[pos], "kind": kind})
    return pd.DataFrame(rows, columns=["date", "kind"])


def era_slices(
    index: pd.DatetimeIndex, eras: list[tuple[str, pd.Timestamp, pd.Timestamp]]
) -> list[tuple[str, pd.DatetimeIndex]]:
    """Partition ``index`` by inclusive era bounds; must cover exactly once."""
    if len(index) == 0:
        raise RegimeComparisonError("empty index")
    assigned = np.zeros(len(index), dtype=int)
    result = []
    for name, start, end in eras:
        mask = (index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))
        assigned += np.asarray(mask, dtype=int)
        result.append((name, index[mask]))
    if (assigned > 1).any():
        first = index[assigned > 1][0]
        raise RegimeComparisonError(f"era boundaries overlap at {first.date()}")
    if (assigned == 0).any():
        first = index[assigned == 0][0]
        raise RegimeComparisonError(f"uncovered date {first.date()} in era partition")
    return result
