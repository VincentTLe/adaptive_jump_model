from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

import adaptive_jump.calibration as calibration
from adaptive_jump.config import load_config

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "research/persistence-calibrated-search.toml"
CONFIG = load_config(ROOT / "research.toml")
MARKETS = ("us", "de", "jp")


def _rules():
    return replace(
        calibration.load_calibration_rules(SPEC, CONFIG),
        exclusive_ends=dict.fromkeys(MARKETS, date(2020, 1, 13)),
        minimum_state_fraction=0.2,
        minimum_transitions=2,
        maximum_candidates=3,
        minimum_budget=3,
    )


def _paths(future: int = 0):
    dates = pd.bdate_range("2020-01-01", periods=10)
    patterns = (
        [0, 1, 0, 1, 0, 1, 0, 1],
        [0, 0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 0, 0],
    )
    patterns = (*patterns, patterns[-1], [0] * 8)
    patterns = [states + [future] * 2 for states in patterns]
    result = {}
    candidates_by_model = {
        "fixed_jm": (0, 1, 2, 3, 4),
        "hmm": (0, 2, 4, 6, 8),
    }
    for model, candidates in candidates_by_model.items():
        frame = pd.DataFrame(dict(zip(candidates, patterns, strict=True)), index=dates)
        result[model] = {market: frame.copy() for market in MARKETS}
    return result


def test_frozen_contract_and_calibration_guards() -> None:
    rules = calibration.load_calibration_rules(SPEC, CONFIG)
    expected = "83beedafca3781d708f0a5ed74bd19127998e441f4242843c07127bcc90487b3"
    assert rules.sha256 == expected

    result = calibration.calibrate_paths(_paths(), _rules())
    assert result.grids == {
        "fixed_jm": (0.0, 1.0, 2.0),
        "hmm": (0.0, 2.0, 4.0),
    }
    rows = result.candidate_diagnostics
    rows = rows.set_index(["model", "candidate"])
    assert rows.loc[("fixed_jm", 3.0), "duplicate_of"] == 2
    assert not bool(rows.loc[("hmm", 8.0), "globally_valid"])

    changed = calibration.calibrate_paths(_paths(1), _rules())
    pd.testing.assert_frame_equal(result.market_diagnostics, changed.market_diagnostics)
    assert result.grids == changed.grids

    paths = _paths()
    for market in MARKETS:
        paths["hmm"][market].loc[:, 4:] = 0
    with pytest.raises(calibration.CalibrationError, match="minimum common"):
        calibration.calibrate_paths(paths, _rules())


def test_jm_expands_until_three_invalid_candidates() -> None:
    rules = replace(_rules(), jm_initial_j_min=0, jm_initial_j_max=2, jm_hard_j_max=5)
    values = {calibration.jm_penalty(j): True for j in range(3)}
    assert calibration.next_jm_index(values, rules) == 3
    values.update({calibration.jm_penalty(j): False for j in (3, 4)})
    assert calibration.next_jm_index(values, rules) == 5
    values[calibration.jm_penalty(5)] = False
    assert calibration.next_jm_index(values, rules) is None
