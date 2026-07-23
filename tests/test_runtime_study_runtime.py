from datetime import date

import numpy as np
import pandas as pd

from adaptive_jump.inference import BootstrapProgress
from adaptive_jump.runtime.events import ResearchEvent
from adaptive_jump.runtime.study_runtime import (
    baseline_selection_recorder,
    bootstrap_recorder,
    emit_boundary_rows,
    emit_selected_signal,
    model_observer,
)
from adaptive_jump.walkforward import SelectionProgress, SelectionResult


def test_model_observer_adds_only_date_matched_finite_features() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02", "2026-01-05"]),
            "dd_10": [0.2, np.nan],
            "sortino_20": [1.5, 2.0],
            "sortino_60": [0.8, 1.0],
            "excess_return": [-0.01, 0.02],
        }
    )
    events = []
    observer = model_observer(events.append, "us", "fixed_jm", frame)
    assert observer is not None

    observer(
        ResearchEvent(
            "terminal_state", "fixed_jm", date=date(2026, 1, 2), payload={"state": 1}
        )
    )

    assert events[0].market == "us" and events[0].model == "fixed_jm"
    assert events[0].payload["state"] == 1
    assert events[0].payload["features"] == {
        "dd_10": 0.2,
        "sortino_20": 1.5,
        "sortino_60": 0.8,
        "excess_return": -0.01,
    }


def test_selection_recorder_saves_then_emits_latest_cv_snapshot() -> None:
    progress = SelectionProgress(
        choices=pd.DataFrame(
            {"decision_date": [pd.Timestamp("2026-01-30")], "selected": [5.0]}
        ),
        surface=pd.DataFrame(
            {
                "decision_date": [pd.Timestamp("2026-01-30")] * 2,
                "candidate": [0.0, 5.0],
                "valid_returns": [2000, 2000],
                "sharpe": [np.nan, 0.7],
                "eligible": [False, True],
            }
        ),
    )
    order = []
    events = []

    def save(model, delay, value):
        order.append((model, delay, value))

    recorder = baseline_selection_recorder(save, events.append, "us")
    recorder("fixed_jm", 1, progress)

    assert order == [("fixed_jm", 1, progress)]
    assert events[0].kind == "selection_checkpoint"
    assert events[0].payload["selected_candidate"] == 5.0
    assert events[0].payload["cv_surface"][0]["sharpe"] is None
    assert (events[0].market, events[0].model, events[0].delay) == (
        "us",
        "fixed_jm",
        1,
    )


def test_boundaries_and_bootstrap_expose_no_outcome_values() -> None:
    events = []
    boundaries = pd.DataFrame(
        [
            {
                "model": "fixed_jm",
                "delay": np.int64(1),
                "upper_candidate": 1200.0,
                "selected_months": np.int64(2),
                "total_months": np.int64(50),
                "fraction": 0.04,
                "limit": 0.05,
                "passed": np.bool_(True),
            }
        ]
    )
    emit_boundary_rows(events.append, boundaries, "us")
    saved = []
    progress = BootstrapProgress(
        np.array([99.0, -99.0]), np.random.default_rng(2).bit_generator.state
    )
    recorder = bootstrap_recorder(
        lambda block, value: saved.append((block, value)), events.append, 10_000
    )

    recorder(60, progress)

    assert events[0].payload["passed"] is True
    assert saved == [(60, progress)]
    assert events[1].completed == 2 and events[1].total == 10_000
    assert events[1].payload == {"mean_block_length": 60}
    assert "draw" not in str(events[1].payload)


def test_selected_signal_reports_only_the_precomputed_future_position() -> None:
    events = []
    dates = pd.bdate_range("2023-12-27", periods=3)
    selection = SelectionResult(
        signal=pd.Series([0.0, 1.0, 1.0], index=dates),
        choices=pd.DataFrame({"decision_date": [dates[1]], "selected": [35.0]}),
        surface=pd.DataFrame(),
        candidate_returns=pd.DataFrame(),
    )

    emit_selected_signal(events.append, selection, "fixed_jm", 1, "us")

    event = events[0]
    assert (event.kind, event.market, event.model, event.delay) == (
        "selected_signal",
        "us",
        "fixed_jm",
        1,
    )
    assert event.date == dates[-1].date()
    assert event.payload == {
        "decision_date": dates[1].date().isoformat(),
        "selected_candidate": 35.0,
        "signal": 1,
        "scheduled_position": 1,
        "effective_return_offset": 2,
    }
