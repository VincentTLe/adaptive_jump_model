"""Tests for independent replay verification of sealed simple-JM runs."""

import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import adaptive_jump.simple_jm_suite as suite
import adaptive_jump.simple_jm_verifier as verifier
from adaptive_jump.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _valid_trace(**overrides: object) -> pd.DataFrame:
    row: dict[str, object] = {
        "market": "us",
        "variant": "dd_only",
        "trace_number": 1,
        "signal_date": "2023-01-03",
        "trade_date": "2023-01-05",
        "signal_row": 7,
        "trade_row": 9,
        "loss_family": "squared feature loss used online",
        "loss_state_0": 0.2,
        "loss_state_1": 0.8,
        "transition_penalty": 0.5,
        "terminal_value_state_0": 1.2,
        "terminal_value_state_1": 1.8,
        "active_state_count": 2,
        "collapsed_to_one_state": False,
        "raw_state": 0.0,
        "state": 0.0,
        "signal": 1.0,
        "position": 1.0,
        "one_way_turnover": 1.0,
        "transaction_cost": 0.001,
        "gross_return": 0.02,
        "strategy_return": 0.019,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_trace_trade_rows_are_linked_to_verified_accounting(tmp_path: Path) -> None:
    dates = pd.bdate_range("2023-01-02", periods=6)
    signal = np.asarray([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    position = np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 1.0])
    turnover = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    equity = np.full(6, 0.02)
    cash = np.zeros(6)
    gross = position * equity
    cost = turnover * 0.001
    trades = pd.DataFrame(
        {
            "date": dates,
            "equity_simple": equity,
            "cash_return": cash,
            "signal": signal,
            "position": position,
            "gross_return": gross,
            "one_way_turnover": turnover,
            "transaction_cost": cost,
            "strategy_return": gross - cost,
        }
    )
    target = tmp_path / "us" / "dd_only"
    target.mkdir(parents=True)
    trades.to_csv(target / "trades.csv", index=False, float_format="%.17g")
    selected = trades.iloc[3]
    trace = _valid_trace(
        trade_date=selected["date"],
        position=selected["position"],
        one_way_turnover=selected["one_way_turnover"],
        transaction_cost=selected["transaction_cost"],
        gross_return=selected["gross_return"],
        strategy_return=selected["strategy_return"],
    )

    verifier._verify_trace_trade_rows(tmp_path, trace)

    trace.loc[0, "strategy_return"] += 0.01
    with pytest.raises(suite.SimpleJMSuiteError, match="strategy_return"):
        verifier._verify_trace_trade_rows(tmp_path, trace)


def test_replay_scaled_selector_uses_choices_signal_and_t_plus_2(
    tmp_path: Path,
) -> None:
    config = load_config(ROOT / "research.toml")
    dates = pd.bdate_range("2010-01-04", "2019-03-29", name="date")
    row = np.arange(len(dates))
    features = pd.DataFrame(
        {
            "date": dates,
            "equity_simple": 0.0002 + 0.01 * np.sin(row / 13),
            "cash_return": np.zeros(len(dates)),
        }
    )
    states = pd.DataFrame(
        {
            penalty: ((row // (20 + number)) % 2).astype(float)
            for number, penalty in enumerate(config.jm_protocol.lambda_grid)
        },
        index=dates,
    )
    selection = suite.select_monthly_candidate(
        features,
        states,
        config.selection_protocol,
        delay_trading_days=1,
        one_way_cost_bps=10,
        periods_per_year=252,
        volatility_ddof=1,
    )
    trades = suite.apply_signal(
        features,
        selection.signal.reset_index(drop=True),
        delay_trading_days=1,
        one_way_cost_bps=10,
    )
    complete = trades.loc[:, suite.METRIC_REQUIRED].notna().all(axis=1)
    trades = trades.loc[complete, suite.TRADE_COLUMNS].reset_index(drop=True)
    target = tmp_path / "us" / suite.SCALED_DD_VARIANT
    target.mkdir(parents=True)
    states.reset_index().to_csv(target / "candidate-states.csv", index=False)
    selection.choices.to_csv(target / "choices.csv", index=False)
    selection.signal.rename("selected_signal").reset_index().to_csv(
        target / "selected-signal.csv", index=False
    )
    pd.DataFrame(columns=suite.REFIT_COLUMNS).to_csv(target / "refits.csv", index=False)
    suite.write_json(target / "boundary.json", {})
    trades.to_csv(target / "trades.csv", index=False, float_format="%.17g")
    stored = suite.read_trade_path(target / "trades.csv", 1, 10)
    source = suite.LossScaleMarketSource("us", features, {})

    verifier._replay_scaled_selector(
        tmp_path,
        "us",
        source,
        stored["date"],
        stored,
        config,
    )

    signal = pd.read_csv(target / "selected-signal.csv")
    changed = signal["selected_signal"].notna().idxmax()
    signal.loc[changed, "selected_signal"] = (
        1.0 - signal.loc[changed, "selected_signal"]
    )
    signal.to_csv(target / "selected-signal.csv", index=False)
    with pytest.raises(suite.SimpleJMSuiteError, match="selector replay changed"):
        verifier._replay_scaled_selector(
            tmp_path,
            "us",
            source,
            stored["date"],
            stored,
            config,
        )


def test_implementation_source_requires_one_complete_historical_snapshot(
    tmp_path: Path,
) -> None:
    _git(tmp_path, "init", "-q")
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.write_text("first-old\n", encoding="utf-8")
    second.write_text("second-old\n", encoding="utf-8")
    _git(tmp_path, "add", "first.py", "second.py")
    _git(
        tmp_path,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        "first",
    )
    first_commit = _git(tmp_path, "rev-parse", "HEAD")
    old_mapping = {
        "first.py": hashlib.sha256(first.read_bytes()).hexdigest(),
        "second.py": hashlib.sha256(second.read_bytes()).hexdigest(),
    }

    first.write_text("first-new\n", encoding="utf-8")
    second.write_text("second-new\n", encoding="utf-8")
    _git(tmp_path, "add", "first.py", "second.py")
    _git(
        tmp_path,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        "second",
    )
    mixed_mapping = {
        "first.py": old_mapping["first.py"],
        "second.py": hashlib.sha256(second.read_bytes()).hexdigest(),
    }

    assert verifier._implementation_source_commit(tmp_path, old_mapping) == first_commit
    with pytest.raises(suite.SimpleJMSuiteError, match="no single Git commit"):
        verifier._implementation_source_commit(tmp_path, mixed_mapping)


def test_implementation_source_accepts_matching_non_git_export(tmp_path: Path) -> None:
    source = tmp_path / "runner.py"
    source.write_text("exact source\n", encoding="utf-8")
    mapping = {"runner.py": hashlib.sha256(source.read_bytes()).hexdigest()}

    assert verifier._implementation_source_commit(tmp_path, mapping) is None
