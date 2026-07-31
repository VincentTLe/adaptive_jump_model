from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from adaptive_jump import lagged_performance as performance
from adaptive_jump.config import load_config
from adaptive_jump.walkforward import SelectionResult

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "research/lagged-evidence-performance-001.toml"
REGISTRY = ROOT / "research/experiment_registry.jsonl"


def test_frozen_lagged_performance_spec_matches_registry() -> None:
    config = load_config(ROOT / "research.toml")
    spec = performance.load_lagged_performance_spec(SPEC, config)
    rows = [
        json.loads(line)
        for line in REGISTRY.read_text(encoding="utf-8").splitlines()
        if json.loads(line).get("experiment_id") == spec.experiment_id
    ]

    assert rows[-1]["status"] == "EXPERIMENT_COMPLETE"
    assert rows[-1]["frozen_spec_hash"] == spec.sha256
    assert spec.cutoff.isoformat() == "2023-12-31"
    assert spec.markets == performance.MARKETS
    assert spec.lambdas == performance.LAMBDAS
    assert spec.beta == pytest.approx(performance.BETA)
    assert spec.artifact_subdir == Path("lagged-evidence-performance-001")


def test_metric_rows_use_explicit_half_turnover_and_deltas() -> None:
    config = load_config(ROOT / "research.toml")
    dates = pd.bdate_range("2020-01-02", periods=4)

    def path(position: list[float], turnover: list[float]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": dates,
                "cash_return": [0.0] * 4,
                "position": position,
                "one_way_turnover": turnover,
                "strategy_return": [0.01, -0.01, 0.02, 0.0],
            }
        )

    fixed = performance._metric_row(
        "us", "fixed", path([1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]), config
    )
    lagged = performance._metric_row(
        "us",
        "lagged_log4",
        path([1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
        config,
    )
    result = performance._add_deltas(pd.DataFrame([fixed, lagged])).set_index("model")

    assert performance.TURNOVER_SCALE == 0.5
    assert result.loc["fixed", "turnover"] == pytest.approx(0.5 * 252 * 2 / 4)
    assert result.loc["lagged_log4", "turnover"] == pytest.approx(0.5 * 252 * 1 / 4)
    assert result.loc["lagged_log4", "delta_turnover"] == pytest.approx(-31.5)
    assert result.loc["fixed", "delta_turnover"] == 0.0


@pytest.mark.parametrize(
    ("deltas", "expected"),
    [
        ((0.20, 0.10, -0.05), "supported"),
        ((0.70, -0.10, -0.10), "not_supported"),
        ((0.10, 0.10, -0.30), "not_supported"),
    ],
)
def test_primary_decision_uses_mean_and_two_positive_markets(
    deltas: tuple[float, float, float], expected: str
) -> None:
    rows = []
    for market, delta in zip(performance.MARKETS, deltas, strict=True):
        rows.extend(
            [
                {"market": market, "model": "fixed", "delta_sharpe": 0.0},
                {
                    "market": market,
                    "model": "arrival_log4",
                    "delta_sharpe": 100.0,
                },
                {
                    "market": market,
                    "model": "lagged_log4",
                    "delta_sharpe": delta,
                },
            ]
        )

    decision = performance._decision(pd.DataFrame(rows))

    assert decision["result"] == expected
    assert decision["positive_market_count"] == sum(value > 0 for value in deltas)
    assert decision["primary_mean_delta_sharpe"] == pytest.approx(sum(deltas) / 3)
    assert decision["arrival_control_mean_delta_sharpe"] == 100.0


@pytest.mark.parametrize("failure", ["schema", "cutoff"])
def test_candidate_states_reject_schema_or_post_2023_rows(
    tmp_path: Path, failure: str
) -> None:
    spec = performance.load_lagged_performance_spec(
        SPEC, load_config(ROOT / "research.toml")
    )
    columns = spec.lambdas if failure == "cutoff" else spec.lambdas[:-1]
    when = "2024-01-02" if failure == "cutoff" else "2023-12-29"
    path = tmp_path / "states.csv"
    pd.DataFrame({"date": [when], **{str(value): [0.0] for value in columns}}).to_csv(
        path, index=False
    )

    message = (
        "candidate-state values changed" if failure == "cutoff" else "schema changed"
    )
    with pytest.raises(performance.LaggedPerformanceError, match=message):
        performance._read_states(path, spec)


def test_change_trace_links_signal_to_t_plus_2_position_and_trade() -> None:
    config = load_config(ROOT / "research.toml")
    dates = pd.bdate_range("2020-01-02", periods=7, name="date")
    frame = pd.DataFrame(
        {
            "date": dates,
            "equity_simple": [0.01, 0.00, -0.02, 0.01, -0.01, 0.02, 0.00],
            "cash_return": [0.0] * len(dates),
        }
    )

    def selection(signal: list[float], selected: float) -> SelectionResult:
        return SelectionResult(
            signal=pd.Series(signal, index=dates, name="selected_signal"),
            choices=pd.DataFrame({"decision_date": [dates[0]], "selected": [selected]}),
            surface=pd.DataFrame(),
            candidate_returns=pd.DataFrame(index=dates),
        )

    selections = {
        "fixed": selection([1.0] * len(dates), 5.0),
        "arrival_log4": selection([1.0] * len(dates), 5.0),
        "lagged_log4": selection([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 15.0),
    }
    full = {
        model: performance._full_path(frame, selected, config)
        for model, selected in selections.items()
    }
    aligned = {
        model: path.dropna(subset=["position"]).reset_index(drop=True)
        for model, path in full.items()
    }

    traces = performance._change_traces(
        "us", frame, selections, full, aligned
    ).set_index("event_type")
    trade = traces.loc["trade"]

    assert set(traces.index) == {"choice", "state", "position", "trade"}
    assert trade["challenger"] == "lagged_log4"
    assert pd.Timestamp(trade["signal_date"]) == dates[2]
    assert pd.Timestamp(trade["execution_date"]) == dates[4]
    assert trade["offset_observations"] == 2
    assert trade["fixed_position"] == 1.0
    assert trade["challenger_position"] == 0.0
    assert trade["fixed_turnover"] == 0.0
    assert trade["challenger_turnover"] == 1.0
    assert trade["challenger_cost"] == pytest.approx(0.001)


def _decision_rows() -> list[dict[str, object]]:
    return [
        {"market": market, "model": model, "delta_sharpe": 0.1}
        for market in performance.MARKETS
        for model in performance.MODELS
    ]


@pytest.mark.parametrize("failure", ["nan", "duplicate", "missing"])
def test_primary_decision_rejects_incomplete_or_non_finite_market_mean(
    failure: str,
) -> None:
    rows = _decision_rows()
    if failure == "nan":
        rows[-1]["delta_sharpe"] = float("nan")
    elif failure == "duplicate":
        rows[-1] = rows[-2].copy()
    else:
        rows.pop()

    with pytest.raises(performance.LaggedPerformanceError):
        performance._decision(pd.DataFrame(rows))


def test_frame_replay_comparison_rejects_resealed_value_change() -> None:
    expected = pd.DataFrame({"date": ["2023-12-29"], "position": [1.0]})
    assert performance._assert_frame_close(expected.copy(), expected, "toy") == 0.0
    boundary = pd.DataFrame({"model": ["fixed"], "passed": [True]})
    assert performance._assert_frame_close(boundary.copy(), boundary, "boundary") == 0.0

    changed = expected.copy()
    changed.loc[0, "position"] = 0.0
    with pytest.raises(performance.LaggedPerformanceError, match="stored values"):
        performance._assert_frame_close(changed, expected, "toy")


def test_verifier_artifact_allowlist_has_exact_three_by_three_trade_coverage() -> None:
    config = load_config(ROOT / "research.toml")
    spec = performance.load_lagged_performance_spec(SPEC, config)
    files = performance._expected_artifact_files(spec)

    assert len(files) == 31
    assert {
        f"{market}/trades/{model}.csv"
        for market in performance.MARKETS
        for model in performance.MODELS
    }.issubset(files)
    assert {"summary.csv", "boundaries.csv", "study.lock.toml"}.issubset(files)
