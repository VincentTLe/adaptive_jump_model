from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import calibrated_config, calibrated_spec_text

from adaptive_jump.lagged_analysis import (
    _extract_events,
    _path_behavior,
)
from adaptive_jump.lagged_study import (
    LaggedStudyError,
    beta_label,
    classify_mechanism,
    load_lagged_spec,
    summarize_mechanism,
)
from adaptive_jump.models import FEATURE_COLUMNS
from adaptive_jump.separation_analysis import MarketInputs
from adaptive_jump.tv_jump import (
    dp_tv,
    evidence_penalty_seq,
    lagged_evidence_penalty_seq,
    lam_to_penalty_seq,
    loss_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "lagged-evidence-mechanism-001.toml"


def _spec_text() -> str:
    return calibrated_spec_text(SOURCE, calibrated_config())


def _write_spec(tmp_path: Path) -> Path:
    path = tmp_path / "lagged-evidence-mechanism-002.toml"
    path.write_text(_spec_text(), encoding="utf-8")
    return path


def _loaded_spec(tmp_path: Path):
    return load_lagged_spec(_write_spec(tmp_path), calibrated_config())


def test_lagged_spec_binds_sources_grids_and_cutoff(tmp_path: Path) -> None:
    config = calibrated_config()
    spec = _loaded_spec(tmp_path)

    assert spec.data_cutoff.isoformat() == "2023-12-31"
    assert spec.betas == (0.0, np.log(2.0), np.log(4.0))
    for market in spec.markets:
        grid = config.jm_protocol_for(market).lambda_grid
        assert spec.lambdas_for(market) == grid
        # the event grid is the market's positive lambdas
        assert spec.event_lambdas_for(market) == tuple(
            value for value in grid if value > 0.0
        )
    assert len({spec.lambdas_for(m) for m in spec.markets}) == 3
    # both sealed sources are named by the spec, never by a literal
    assert str(spec.fixed.artifact_subdir) == "fixed-baselines"
    assert spec.arrival.experiment_id.startswith("adaptive-confidence-")
    assert spec.fixed_allowed_files == ("features.csv", "jm-states.csv")
    assert spec.performance_files_forbidden == (
        "summary.csv",
        "selected-timeline.csv",
        "choices.csv",
        "conclusion.json",
        "trades.csv",
    )
    assert not (set(spec.arrival_allowed_files) & set(spec.performance_files_forbidden))


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("post_2023_access = false", "post_2023_access = true"),
        (
            "monthly_performance_selection_allowed = false",
            "monthly_performance_selection_allowed = true",
        ),
        ('data_cutoff = "2023-12-31"', 'data_cutoff = "2024-01-01"'),
        (
            "lambda_selected_by_performance = false",
            "lambda_selected_by_performance = true",
        ),
        (
            'information_claim = "causal lagged-loss evidence; not strictly '
            'F_(t-1)-predictable on a refit date"',
            'information_claim = "strictly predictable"',
        ),
    ],
)
def test_lagged_spec_rejects_evidence_lane_or_control_changes(
    tmp_path: Path, old: str, new: str
) -> None:
    changed = tmp_path / "study.toml"
    changed.write_text(
        _spec_text().replace(old, new),
        encoding="utf-8",
    )

    with pytest.raises(LaggedStudyError):
        load_lagged_spec(changed, calibrated_config())


def _toy_candidates(
    features: pd.DataFrame,
    dates: pd.DatetimeIndex,
    builder,
    *,
    beta: float,
    lambda0: float,
    fit_window: int,
) -> pd.Series:
    centers = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    result = pd.Series(np.nan, index=dates)
    for terminal in range(fit_window - 1, len(dates)):
        raw = features.iloc[terminal - fit_window + 1 : terminal + 1].to_numpy()
        losses = loss_matrix(raw, centers)
        penalties = builder(losses, lambda0, beta, 1.0)
        result.iloc[terminal] = int(
            dp_tv(losses, penalties, return_value_mx=True)[-1].argmin()
        )
    return result


def _toy_inputs(rule: str, tmp_path: Path):
    config = calibrated_config()
    base = load_lagged_spec(_write_spec(tmp_path), config)
    dates = pd.bdate_range("2020-01-01", periods=9, name="date")
    loss_gaps = np.array([-8.0] * 5 + [2.0, -2.0, -8.0, -8.0])
    x_value = (loss_gaps + 2.0) / 2.0
    features = pd.DataFrame(
        {
            FEATURE_COLUMNS[0]: x_value,
            FEATURE_COLUMNS[1]: np.zeros(len(dates)),
            FEATURE_COLUMNS[2]: np.zeros(len(dates)),
        },
        index=dates,
    )
    beta = math.log(4.0)
    lambda0 = 4.0

    fixed = _toy_candidates(
        features,
        dates,
        lambda loss, lam, _beta, _q: lam_to_penalty_seq(np.full(len(loss), lam), 2),
        beta=0.0,
        lambda0=lambda0,
        fit_window=5,
    )
    builder = evidence_penalty_seq if rule == "arrival" else lagged_evidence_penalty_seq
    model = _toy_candidates(
        features,
        dates,
        builder,
        beta=beta,
        lambda0=lambda0,
        fit_window=5,
    )
    candidates = {
        0.0: pd.DataFrame({lambda0: fixed}, index=dates),
        beta: pd.DataFrame({lambda0: model}, index=dates),
    }
    refits = pd.DataFrame(
        [
            {
                "market": "us",
                "fit_date": dates[4],
                "training_start": dates[0],
                "training_end": dates[4],
                "lambda0": lambda0,
                "q_train": 1.0,
                "fixed_objective": 0.0,
                "scaler_mean": json.dumps([0.0, 0.0, 0.0]),
                "scaler_scale": json.dumps([1.0, 1.0, 1.0]),
                "centers": json.dumps([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            }
        ]
    )
    inputs = MarketInputs(
        market="us",
        features=features,
        model_dates=dates,
        candidates=candidates,
        refits=refits,
    )
    spec = replace(
        base,
        markets=("us",),
        evaluation_starts={"us": dates[4].date()},
        betas=(0.0, beta),
        event_betas=(beta,),
        lambdas={"us": (lambda0,)},
        event_lambdas={"us": (lambda0,)},
        rules=("arrival", "lagged"),
        fit_window=5,
        horizon=2,
    )
    return inputs, candidates, spec


def test_arrival_event_is_locally_attributed_but_lagged_filters_the_shock(
    tmp_path: Path,
) -> None:
    arrival_inputs, arrival_candidates, spec = _toy_inputs("arrival", tmp_path)
    lagged_inputs, lagged_candidates, _ = _toy_inputs("lagged", tmp_path)

    arrival, arrival_audit = _extract_events(
        arrival_inputs, arrival_candidates, "arrival", spec
    )
    lagged, lagged_audit = _extract_events(
        lagged_inputs, lagged_candidates, "lagged", spec
    )

    assert len(arrival) == 1
    event = arrival.iloc[0]
    assert bool(event["whipsaw_20"])
    assert not bool(event["persistent_20"])
    assert event["terminal_predecessor"] == event["source_state"] == 0
    assert event["destination_state"] == 1
    assert event["ablated_state"] == 0
    assert pd.Timestamp(event["evidence_date"]) == pd.Timestamp(event["signal_date"])
    assert event["transition_penalty"] < event["lambda0"]
    assert arrival_audit["ablation_attributable"] == 1

    assert lagged.empty
    assert lagged_audit["admitted_events"] == 0


def test_path_behavior_counts_switches_and_differences_without_selection(
    tmp_path: Path,
) -> None:
    inputs, candidates, spec = _toy_inputs("arrival", tmp_path)

    rows = _path_behavior(inputs, candidates, "arrival", spec)

    assert len(rows) == 1
    lambda0 = spec.event_lambdas_for("us")[0]
    expected = candidates[spec.event_betas[0]][lambda0].dropna()
    fixed = candidates[0.0][lambda0].reindex(expected.index)
    assert rows[0]["switch_count"] == int(
        np.count_nonzero(np.diff(expected.to_numpy()))
    )
    assert rows[0]["state_differences_vs_fixed"] == int((expected != fixed).sum())


def test_path_behavior_counts_transition_into_evaluation_start(
    tmp_path: Path,
) -> None:
    inputs, candidates, spec = _toy_inputs("arrival", tmp_path)
    beta = spec.event_betas[0]
    lambda0 = spec.event_lambdas_for("us")[0]
    model = candidates[beta][lambda0].copy()
    valid = model.dropna().index
    model.loc[valid] = [0, 1, 1, 1, 1]
    candidates[beta][lambda0] = model
    start = valid[1]
    spec = replace(spec, evaluation_starts={"us": start.date()})

    rows = _path_behavior(inputs, candidates, "arrival", spec)

    assert rows[0]["start"] == start
    assert rows[0]["observations"] == 4
    assert rows[0]["switch_count"] == 1


def _decision_summary() -> pd.DataFrame:
    records = []
    for beta in ("log2", "log4"):
        for market in ("us", "de", "jp"):
            for rule in ("arrival", "lagged"):
                records.append(
                    {
                        "market": market,
                        "beta_label": beta,
                        "rule": rule,
                        "whipsaw_count": 2 if rule == "arrival" else 1,
                        "confirmed_early_count": (
                            1 if rule == "lagged" and market == "us" else 0
                        ),
                        "switch_count": (
                            10
                            if rule == "arrival" and market == "jp"
                            else 8
                            if rule == "lagged" and market == "jp"
                            else 5
                        ),
                        "state_differences_vs_fixed": 1 if rule == "lagged" else 0,
                    }
                )
    return pd.DataFrame.from_records(records)


def test_mechanism_gate_requires_every_frozen_condition_and_prefers_log2(
    tmp_path: Path,
) -> None:
    spec = _loaded_spec(tmp_path)
    passing = _decision_summary()

    conclusion = classify_mechanism(passing, spec, mechanical_prerequisites_passed=True)

    assert conclusion["result"] == "supported"
    assert conclusion["advancing_beta_labels"] == ["log2", "log4"]
    assert conclusion["selected_beta_label"] == "log2"

    failures = []
    market_worse = passing.copy()
    market_worse.loc[
        (market_worse["beta_label"] == "log2")
        & (market_worse["market"] == "us")
        & (market_worse["rule"] == "lagged"),
        "whipsaw_count",
    ] = 3
    failures.append(market_worse)

    pooled_equal = passing.copy()
    pooled_equal.loc[
        (pooled_equal["beta_label"] == "log2") & (pooled_equal["rule"] == "lagged"),
        "whipsaw_count",
    ] = 2
    failures.append(pooled_equal)

    jp_equal = passing.copy()
    jp_equal.loc[
        (jp_equal["beta_label"] == "log2")
        & (jp_equal["market"] == "jp")
        & (jp_equal["rule"] == "lagged"),
        "switch_count",
    ] = 10
    failures.append(jp_equal)

    no_latency = passing.copy()
    no_latency.loc[
        (no_latency["beta_label"] == "log2") & (no_latency["rule"] == "lagged"),
        "confirmed_early_count",
    ] = 0
    failures.append(no_latency)

    trivial = passing.copy()
    trivial.loc[
        (trivial["beta_label"] == "log2") & (trivial["rule"] == "lagged"),
        "state_differences_vs_fixed",
    ] = 0
    failures.append(trivial)

    for failed in failures:
        result = classify_mechanism(failed, spec, mechanical_prerequisites_passed=True)
        assert not result["by_beta"]["log2"]["advances"]


def test_mechanism_gate_fails_when_mechanical_prerequisites_fail(
    tmp_path: Path,
) -> None:
    spec = _loaded_spec(tmp_path)

    conclusion = classify_mechanism(
        _decision_summary(),
        spec,
        mechanical_prerequisites_passed=False,
    )

    assert conclusion["result"] == "not_supported"
    assert conclusion["selected_beta_label"] is None


def test_summary_counts_events_and_complete_path_rows(tmp_path: Path) -> None:
    spec = _loaded_spec(tmp_path)
    events = pd.DataFrame(
        [
            {
                "market": market,
                "rule": rule,
                "beta_label": beta_label(beta),
                "whipsaw_20": rule == "arrival",
                "persistent_20": rule == "lagged",
                "confirmed_early": rule == "lagged",
            }
            for market in spec.markets
            for beta in spec.event_betas
            for rule in spec.rules
        ]
    )
    behavior = pd.DataFrame(
        [
            {
                "market": market,
                "rule": rule,
                "beta_label": beta_label(beta),
                "switch_count": 2,
                "state_differences_vs_fixed": int(rule == "lagged"),
                "lambda0": lambda0,
            }
            for market in spec.markets
            for beta in spec.event_betas
            for rule in spec.rules
            for lambda0 in spec.event_lambdas_for(market)
        ]
    )

    summary = summarize_mechanism(events, behavior, spec)

    assert len(summary) == 12
    assert (summary.loc[summary["rule"] == "arrival", "whipsaw_count"] == 1).all()
    assert (
        summary.loc[summary["rule"] == "lagged", "confirmed_early_count"] == 1
    ).all()
    assert (
        summary.loc[summary["rule"] == "lagged", "state_differences_vs_fixed"]
        == len(spec.event_lambdas_for("us"))
    ).all()

    with pytest.raises(LaggedStudyError, match="path coverage changed"):
        summarize_mechanism(events, behavior.iloc[:-1], spec)
