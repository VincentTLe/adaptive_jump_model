from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adaptive_jump.confidence_evaluation import (
    _add_deltas,
    _full_path,
    _selected_timeline,
)
from adaptive_jump.confidence_model import StateEvidence, _assert_beta_zero_states
from adaptive_jump.confidence_runner import _conclusion
from adaptive_jump.confidence_spec import (
    BETAS,
    ConfidenceStudyError,
    load_confidence_spec,
)
from adaptive_jump.config import load_config
from adaptive_jump.walkforward import SelectionResult

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "research/adaptive-confidence-001.toml"
REGISTRY = ROOT / "research/experiment_registry.jsonl"


def test_confidence_spec_is_bound_to_v7_and_registry() -> None:
    config = load_config(ROOT / "research.toml")
    spec = load_confidence_spec(SPEC, config)
    records = [
        json.loads(line)
        for line in REGISTRY.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["experiment_id"] == spec.experiment_id
    ]

    assert records[-1]["frozen_spec_hash"] == spec.sha256
    assert records[-1]["status"] in {"FROZEN", "EXPERIMENT_COMPLETE"}
    assert spec.betas == (0.0, np.log(2.0), np.log(4.0))
    assert spec.lambdas == config.jm_protocol.lambda_grid
    assert spec.data_cutoff.isoformat() == "2023-12-31"


def test_confidence_spec_rejects_post_2023_access(tmp_path: Path) -> None:
    changed = tmp_path / "study.toml"
    changed.write_text(
        SPEC.read_text(encoding="utf-8").replace(
            "post_2023_access = false", "post_2023_access = true"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfidenceStudyError, match="evidence lane"):
        load_confidence_spec(changed, load_config(ROOT / "research.toml"))


def test_beta_zero_state_oracle_is_exact() -> None:
    dates = pd.date_range("2020-01-01", periods=3)
    generated = pd.DataFrame(
        {0.0: [np.nan, 0.0, 1.0], 5.0: [np.nan, 1.0, 1.0]},
        index=dates,
    )
    _assert_beta_zero_states(generated, generated.copy(), market="toy")

    changed = generated.copy()
    changed.loc[dates[-1], 5.0] = 0.0
    with pytest.raises(ConfidenceStudyError, match="beta0 state mismatch"):
        _assert_beta_zero_states(changed, generated, market="toy")


def test_tradeoff_rule_uses_all_frozen_inequalities() -> None:
    metrics = pd.DataFrame(
        {
            "market": ["us", "us", "us"],
            "beta": BETAS,
            "sharpe": [1.0, 1.1, 0.9],
            "maximum_drawdown": [-0.20, -0.19, -0.18],
            "turnover": [10.0, 9.0, 8.0],
            "cash_fraction": [0.3, 0.31, 0.32],
            "switch_count": [20, 18, 17],
        }
    )

    result = _add_deltas(metrics)

    assert not bool(result.loc[result["beta"] == 0.0, "reduced_tradeoff"].iloc[0])
    assert bool(result.loc[result["beta"] == BETAS[1], "reduced_tradeoff"].iloc[0])
    assert not bool(result.loc[result["beta"] == BETAS[2], "reduced_tradeoff"].iloc[0])
    assert result.loc[result["beta"] == BETAS[1], "delta_sharpe"].iloc[
        0
    ] == pytest.approx(0.1)


def test_selected_timeline_maps_signal_to_t_plus_2_trade() -> None:
    config = load_config(ROOT / "research.toml")
    dates = pd.bdate_range("2020-01-01", periods=6, name="date")
    frame = pd.DataFrame(
        {
            "date": dates,
            "equity_simple": [0.01, 0.02, -0.01, -0.02, 0.03, 0.01],
            "cash_return": [0.0001] * 6,
        }
    )
    signal = pd.Series([1.0, 1.0, 0.0, 0.0, 1.0, 1.0], index=dates)
    selection = SelectionResult(
        signal=signal,
        choices=pd.DataFrame({"decision_date": [dates[0]], "selected": [5.0]}),
        surface=pd.DataFrame(),
        candidate_returns=pd.DataFrame(index=dates),
    )
    values = pd.DataFrame({5.0: [1.0] * 6}, index=dates)
    evidence = StateEvidence(
        states={beta: values.copy() for beta in BETAS},
        loss0=pd.DataFrame({5.0: [3.0] * 6}, index=dates),
        loss1=pd.DataFrame({5.0: [1.0] * 6}, index=dates),
        q_train=values.copy(),
        c01={BETAS[1]: pd.DataFrame({5.0: [2.0] * 6}, index=dates)},
        c10={BETAS[1]: pd.DataFrame({5.0: [5.0] * 6}, index=dates)},
        refits=pd.DataFrame(),
    )
    full_path = _full_path(frame, selection, config)

    timeline = _selected_timeline(
        frame,
        evidence,
        selection,
        full_path,
        BETAS[1],
        config,
        "toy",
    )
    switch = timeline.loc[timeline["signal_date"] == dates[2]].iloc[0]

    assert switch["previous_emitted_state"] == 0
    assert switch["state"] == 1
    assert switch["arrival_loss_advantage"] == 2.0
    assert switch["emitted_transition_penalty"] == 2.0
    assert pd.Timestamp(switch["execution_date"]) == dates[4]
    assert switch["position"] == 0.0
    assert switch["one_way_turnover"] == 1.0
    assert switch["transaction_cost"] == pytest.approx(0.001)


def test_study_conclusion_requires_one_beta_across_all_markets() -> None:
    summary = pd.DataFrame(
        [
            {
                "market": market,
                "beta_label": label,
                "reduced_tradeoff": label == "log2",
            }
            for market in ("us", "de", "jp")
            for label in ("0", "log2", "log4")
        ]
    )
    mechanisms = {
        market: {
            label: {
                "selected_state_differences": 1,
                "evidence_discounted_switches": 1,
            }
            for label in ("log2", "log4")
        }
        for market in ("us", "de", "jp")
    }

    conclusion = _conclusion(summary, mechanisms)

    assert conclusion["tradeoff_result"] == "supported"
    assert conclusion["markets_reduced_by_beta"] == {"log2": 3, "log4": 0}
    assert conclusion["mechanism_operational"] is True
    assert conclusion["performance_claim_allowed"] is False
