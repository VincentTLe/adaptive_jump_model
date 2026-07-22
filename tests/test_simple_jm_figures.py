from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import adaptive_jump.simple_jm_figures as figures
from adaptive_jump.artifacts import (
    TRADE_COLUMNS,
    read_json,
    write_inventory,
    write_json,
)


def _trade_frame(dates: pd.DatetimeIndex, signal: list[float]) -> pd.DataFrame:
    signal_values = np.asarray(signal, dtype=float)
    position = np.r_[signal_values[0], signal_values[0], signal_values[:-2]]
    equity = np.linspace(-0.012, 0.014, len(dates))
    cash = np.full(len(dates), 0.0001)
    turnover = np.r_[0.0, np.abs(np.diff(position))]
    cost = turnover * 0.001
    gross = position * equity + (1.0 - position) * cash
    return pd.DataFrame(
        {
            "date": dates,
            "equity_simple": equity,
            "cash_return": cash,
            "signal": signal_values,
            "position": position,
            "gross_return": gross,
            "one_way_turnover": turnover,
            "transaction_cost": cost,
            "strategy_return": gross - cost,
        },
        columns=TRADE_COLUMNS,
    )


def _buy_and_hold(dates: pd.DatetimeIndex) -> pd.DataFrame:
    frame = _trade_frame(dates, [1.0] * len(dates))
    frame["position"] = 1.0
    frame["gross_return"] = frame["equity_simple"]
    frame["one_way_turnover"] = 0.0
    frame["transaction_cost"] = 0.0
    frame["strategy_return"] = frame["equity_simple"]
    return frame


def _write_run(tmp_path: Path, start: str = "2020-02-18") -> Path:
    run = tmp_path / "simple-jm-suite-test-run"
    dates = pd.bdate_range(start, periods=14)
    # Four decisions at rows 1, 3, 5, 7 first affect recorded positions,
    # costs, and returns on the t+2 accounting rows 3, 5, 7, 9.
    signal = [1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    rows = []
    for market in figures.MARKETS:
        for model in figures.WEALTH_MODELS:
            frame = (
                _buy_and_hold(dates)
                if model == "buy_and_hold"
                else _trade_frame(dates, signal)
            )
            target = run / market / model
            target.mkdir(parents=True, exist_ok=True)
            frame.to_csv(target / "trades.csv", index=False, float_format="%.17g")
            rows.append(
                {
                    "market": market,
                    "model": model,
                    "start": str(dates[0].date()),
                    "end": str(dates[-1].date()),
                    "observations": len(frame),
                    "cash_fraction": float(1.0 - frame["position"].mean()),
                    "switch_count": int((frame["one_way_turnover"] > 0).sum()),
                }
            )
    pd.DataFrame(rows).to_csv(run / "summary.csv", index=False)
    write_json(
        run / "run.json",
        {
            "schema_version": 1,
            "study_kind": "simple-jm-suite-001",
            "status": "complete",
            "run_id": run.name,
        },
    )
    write_inventory(run)
    return run


def test_load_figure_run_keeps_signal_and_delayed_position_distinct(
    tmp_path: Path,
) -> None:
    run = figures.load_figure_run(_write_run(tmp_path))
    path = run.paths["us"]["dd_only"]

    assert path.loc[1, "signal"] == 0.0
    assert path.loc[1, "position"] == 1.0
    np.testing.assert_array_equal(
        path["position"].iloc[2:].to_numpy(),
        path["signal"].iloc[:-2].to_numpy(),
    )
    assert figures.TRADING_DELAY_DAYS == 1
    assert figures.RETURN_ACCOUNTING_OFFSET == 2


def test_load_rejects_post_2023_data(tmp_path: Path) -> None:
    with pytest.raises(figures.SimpleJMFigureError, match="post-2023"):
        figures.load_figure_run(_write_run(tmp_path, "2024-01-02"))


def test_load_rejects_wrong_transaction_cost(tmp_path: Path) -> None:
    run = _write_run(tmp_path)
    path = run / "us" / "dd_only" / "trades.csv"
    frame = pd.read_csv(path)
    row = frame.index[frame["one_way_turnover"].gt(0)][0]
    frame.loc[row, "transaction_cost"] = 0.0
    frame.to_csv(path, index=False, float_format="%.17g")
    write_inventory(run)

    with pytest.raises(figures.SimpleJMFigureError, match="trade accounting mismatch"):
        figures.load_figure_run(run)


def test_load_rejects_inventory_mutation(tmp_path: Path) -> None:
    run = _write_run(tmp_path)
    summary_path = run / "summary.csv"
    summary = pd.read_csv(summary_path)
    summary.loc[0, "switch_count"] += 1
    summary.to_csv(summary_path, index=False)

    with pytest.raises(figures.SimpleJMFigureError, match="inventory mismatch"):
        figures.load_figure_run(run)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 2, "run metadata"),
        ("study_kind", "different-study", "run metadata"),
        ("status", "running", "run metadata"),
        ("run_id", "different-run", "directory and identity"),
    ],
)
def test_load_rejects_invalid_run_identity(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    run = _write_run(tmp_path)
    metadata_path = run / "run.json"
    metadata = read_json(metadata_path)
    metadata[field] = value
    write_json(metadata_path, metadata)

    with pytest.raises(figures.SimpleJMFigureError, match=message):
        figures.load_figure_run(run)


def test_indexed_wealth_uses_compounded_simple_returns() -> None:
    observed = figures._indexed_wealth(pd.Series([0.10, -0.05, 0.02]))

    np.testing.assert_allclose(observed, [110.0, 104.5, 106.59])


def test_render_writes_two_figures_in_png_svg_and_pdf(tmp_path: Path) -> None:
    run = _write_run(tmp_path)

    outputs = figures.render_figures(run, tmp_path / "reports")

    expected_stems = {
        "us-causal-regimes",
        "shu-style-net-wealth",
    }
    assert len(outputs) == 6
    assert {path.stem for path in outputs} == expected_stems
    assert {path.suffix for path in outputs} == {".png", ".svg", ".pdf"}
    assert all(path.parent == tmp_path / "reports" / run.name for path in outputs)
    assert all(path.stat().st_size > 1000 for path in outputs)
    assert all(run not in path.parents for path in outputs)


def test_render_rejects_output_inside_the_sealed_run(tmp_path: Path) -> None:
    run = _write_run(tmp_path)

    with pytest.raises(figures.SimpleJMFigureError, match="outside the sealed run"):
        figures.render_figures(run, run)
