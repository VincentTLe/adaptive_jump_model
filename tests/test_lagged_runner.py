from __future__ import annotations

import inspect
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from adaptive_jump import lagged_runner, separation_analysis
from adaptive_jump.models import FEATURE_COLUMNS


def _smoke_checks() -> dict[str, bool]:
    return {
        "sealed_arrival_exact": True,
        "beta_zero_exact": True,
        "prefix_invariant": True,
        "future_mutation_effect_present": True,
        "refit_convention_numeric": True,
        "lagged_discounts_present": True,
        "performance_files_accessed": False,
        "return_columns_accessed": False,
        "post_2023_accessed": False,
    }


def test_mechanical_gate_uses_computed_result_and_replays() -> None:
    smoke = _smoke_checks()
    replay = {
        "us": {
            "sealed_arrival_exact": True,
            "beta_zero_exact": True,
            "sealed_arrival_state_cells_checked": 12,
            "beta_zero_state_cells_checked": 8,
            "return_columns_accessed": False,
        }
    }

    failed = lagged_runner._mechanical_checks({"passed": False}, smoke, replay)
    assert not failed["passed"]

    passed = lagged_runner._mechanical_checks({"passed": True}, smoke, replay)
    assert passed["passed"]
    replay["us"]["sealed_arrival_exact"] = False
    assert not lagged_runner._mechanical_checks({"passed": True}, smoke, replay)[
        "passed"
    ]


def test_locked_input_loader_requests_no_return_or_refit_file(monkeypatch) -> None:
    dates = pd.bdate_range("2020-01-01", periods=2, name="date")
    fixed = pd.DataFrame({0.0: [0, 0], 5.0: [1, 1]}, index=dates)
    features = pd.DataFrame(0.0, index=dates, columns=FEATURE_COLUMNS)
    inputs = SimpleNamespace(features=features, candidates={0.0: fixed})
    spec = SimpleNamespace(
        markets=("us",),
        lambdas=(0.0, 5.0),
        event_lambdas=(5.0,),
        data_cutoff=date(2020, 1, 31),
        fit_window=2,
    )
    seen: dict[str, object] = {}

    def fake_load(
        market, feature_path, arrival_dir, loader_spec, *, include_fixed_objective
    ):
        seen["feature_path"] = Path(feature_path)
        seen["arrival_dir"] = Path(arrival_dir)
        assert market == "us"
        assert loader_spec.lambdas == (5.0,)
        assert include_fixed_objective is False
        return inputs

    def fake_read(path, *, usecols):
        seen["state_path"] = Path(path)
        seen["usecols"] = usecols
        return fixed.rename_axis("date").reset_index()

    monkeypatch.setattr(lagged_runner, "load_market_inputs", fake_load)
    monkeypatch.setattr(lagged_runner.pd, "read_csv", fake_read)
    loaded, actual = lagged_runner._load_inputs(
        "us", Path("fixed/us"), Path("arrival/us"), spec
    )

    assert loaded is inputs
    pd.testing.assert_frame_equal(actual, fixed, check_freq=False)
    assert seen["feature_path"] == Path("fixed/us/features.csv")
    assert seen["state_path"] == Path("fixed/us/jm-states.csv")
    assert seen["usecols"] == ["date", "0.0", "5.0"]


def test_shared_loader_excludes_fixed_objective_when_requested(monkeypatch) -> None:
    dates = pd.bdate_range("2020-01-01", periods=3)
    feature_frame = pd.DataFrame({"date": dates})
    for column in FEATURE_COLUMNS:
        feature_frame[column] = 0.0
    candidate_frame = pd.DataFrame(
        {
            "date": dates,
            "0.0": [float("nan"), 0.0, 0.0],
            "5.0": [float("nan"), 1.0, 1.0],
        }
    )
    refit_frame = pd.DataFrame(
        [
            {
                "market": "us",
                "fit_date": dates[1],
                "training_start": dates[0],
                "training_end": dates[1],
                "lambda0": lambda0,
                "q_train": 1.0,
                "scaler_mean": "[0.0, 0.0, 0.0]",
                "scaler_scale": "[1.0, 1.0, 1.0]",
                "centers": "[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]",
            }
            for lambda0 in (0.0, 5.0)
        ]
    )
    seen_refit_columns: list[str] = []

    def fake_read(path, *, usecols):
        name = Path(path).name
        if name == "features.csv":
            source = feature_frame
        elif name == "refits-and-scales.csv":
            seen_refit_columns.extend(usecols)
            source = refit_frame
        else:
            source = candidate_frame
        return source.loc[:, usecols].copy()

    monkeypatch.setattr(separation_analysis.pd, "read_csv", fake_read)
    spec = SimpleNamespace(
        markets=("us",),
        lambdas=(5.0,),
        data_cutoff=date(2020, 1, 31),
        fit_window=2,
    )
    inputs = separation_analysis.load_market_inputs(
        "us",
        Path("features.csv"),
        Path("arrival"),
        spec,
        include_fixed_objective=False,
    )

    assert "fixed_objective" not in seen_refit_columns
    assert "fixed_objective" not in inputs.refits
    assert inputs.model_dates.equals(pd.DatetimeIndex(dates))


def test_terminal_limited_adapter_keeps_mutated_future_rows(monkeypatch) -> None:
    dates = pd.bdate_range("2020-01-01", periods=5, name="date")
    features = pd.DataFrame(0.0, index=dates, columns=FEATURE_COLUMNS)
    mutated = features.copy()
    mutated.loc[dates[3] :, FEATURE_COLUMNS] = 999.0
    fixed = pd.DataFrame({0.0: 0.0}, index=dates)
    refits = pd.DataFrame({"fit_date": [dates[1]]})
    inputs = SimpleNamespace(features=features, refits=refits, model_dates=dates)
    spec = SimpleNamespace(fit_window=2)
    config = object()
    sentinel = object()
    seen: dict[str, object] = {}

    def fake_generate(
        feature_frame, fixed_states, sealed_refits, actual_config, actual_spec, **kwargs
    ):
        seen.update(
            feature_frame=feature_frame,
            fixed_states=fixed_states,
            sealed_refits=sealed_refits,
            config=actual_config,
            spec=actual_spec,
            kwargs=kwargs,
        )
        return sentinel

    monkeypatch.setattr(lagged_runner, "generate_locked_candidates", fake_generate)
    result = lagged_runner._generate_locked(
        "us", inputs, fixed, config, spec, terminal_limit=2, features=mutated
    )

    assert result is sentinel
    assert len(seen["feature_frame"]) == len(features)
    assert seen["feature_frame"].iloc[-1][FEATURE_COLUMNS[0]] == 999.0
    assert seen["fixed_states"] is fixed
    assert seen["sealed_refits"] is refits
    assert seen["kwargs"]["terminal_limit"] == 2


def test_run_us_smoke_delegates_to_locked_mechanics(monkeypatch) -> None:
    config = SimpleNamespace(path=Path("/repo/research.toml"))
    spec = object()
    inputs = object()
    fixed = object()
    sources = SimpleNamespace(
        fixed_markets={"us": Path("fixed/us")},
        arrival_markets={"us": Path("arrival/us")},
    )
    expected = {"status": "passed"}

    monkeypatch.setattr(
        lagged_runner, "verify_source_inputs", lambda root, cfg, study: sources
    )
    monkeypatch.setattr(
        lagged_runner,
        "_load_inputs",
        lambda market, fixed_dir, arrival_dir, study: (inputs, fixed),
    )
    monkeypatch.setattr(
        lagged_runner,
        "run_locked_smoke",
        lambda actual_inputs, actual_fixed, cfg, study, builders: (
            expected
            if (actual_inputs, actual_fixed, cfg, study, builders)
            == (inputs, fixed, config, spec, lagged_runner.BUILDERS)
            else None
        ),
    )

    assert lagged_runner.run_us_smoke(config, spec) is expected


def test_runner_has_no_return_or_refit_generation_path() -> None:
    source = inspect.getsource(lagged_runner)
    for forbidden in (
        "_load_parent_frame",
        "generate_adaptive_states",
        "_complete_model_frame",
        "jm-refits.csv",
        "excess_return",
        "mechanical_prerequisites_passed=True",
    ):
        assert forbidden not in source
    assert "prefix_terminal_limit = 20" not in source
    assert "run_locked_smoke(inputs, fixed, config, spec, BUILDERS)" in source
    assert 'mechanical_prerequisites_passed=mechanics["passed"]' in source
    assert len(source.splitlines()) < 500
