import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import adaptive_jump.simple_jm_suite as suite
from adaptive_jump.artifacts import TRADE_COLUMNS, ArtifactError
from adaptive_jump.config import load_config
from adaptive_jump.walkforward import SelectionResult


def test_implementation_hashes_cover_result_code_and_environment_lock() -> None:
    hashes = suite._implementation_hashes(ROOT)

    assert {
        "src/adaptive_jump/artifacts.py",
        "src/adaptive_jump/backtest.py",
        "src/adaptive_jump/config.py",
        "src/adaptive_jump/models.py",
        "src/adaptive_jump/simple_jm_controls.py",
        "src/adaptive_jump/simple_jm_fitting.py",
        "src/adaptive_jump/simple_jm_l1.py",
        "src/adaptive_jump/simple_jm_return.py",
        "src/adaptive_jump/simple_jm_suite.py",
        "src/adaptive_jump/walkforward.py",
        "pyproject.toml",
        "uv.lock",
    } <= hashes.keys()


ROOT = Path(__file__).resolve().parents[1]


def test_runner_error_follows_artifact_error_contract() -> None:
    assert issubclass(suite.SimpleJMSuiteError, ArtifactError)


def _trade_frame(
    dates: pd.DatetimeIndex,
    *,
    strategy_return: list[float] | None = None,
    position: list[float] | None = None,
    turnover: list[float] | None = None,
) -> pd.DataFrame:
    rows = len(dates)
    strategy = strategy_return or [0.01, -0.02, 0.03, 0.0][:rows]
    positions = position or [1.0] * rows
    one_way_turnover = turnover or [0.0] * rows
    return pd.DataFrame(
        {
            "date": dates,
            "equity_simple": np.linspace(0.01, 0.02, rows),
            "cash_return": np.zeros(rows),
            "signal": positions,
            "position": positions,
            "gross_return": strategy,
            "one_way_turnover": one_way_turnover,
            "transaction_cost": np.zeros(rows),
            "strategy_return": strategy,
        },
        columns=TRADE_COLUMNS,
    )


def test_align_paths_uses_complete_common_date_intersection() -> None:
    dates = pd.bdate_range("2023-01-02", periods=4)
    first = _trade_frame(dates)
    second = _trade_frame(dates).iloc[1:].copy()
    second.loc[second.index[1], "strategy_return"] = np.nan

    aligned = suite._align_paths({"first": first, "second": second})

    expected = pd.Series([dates[1], dates[3]], name="date")
    for path in aligned.values():
        pd.testing.assert_series_equal(path["date"], expected)
        assert tuple(path.columns) == TRADE_COLUMNS


def test_align_paths_rejects_different_market_returns() -> None:
    dates = pd.bdate_range("2023-01-02", periods=4)
    first = _trade_frame(dates)
    second = _trade_frame(dates)
    second.loc[1, "equity_simple"] += 0.001

    with pytest.raises(suite.SimpleJMSuiteError, match="different market returns"):
        suite._align_paths({"first": first, "second": second})


def test_trade_route_equal_tolerates_only_csv_scale_continuous_noise() -> None:
    dates = pd.bdate_range("2023-01-02", periods=4)
    reference = _trade_frame(
        dates,
        position=[1.0, 0.0, 0.0, 1.0],
        turnover=[0.0, 1.0, 0.0, 1.0],
    )
    round_tripped = reference.copy()
    continuous = (
        "equity_simple",
        "cash_return",
        "gross_return",
        "transaction_cost",
        "strategy_return",
    )
    round_tripped.loc[1, list(continuous)] += 5e-16

    assert suite._trade_route_equal(reference, round_tripped)

    changed = reference.copy()
    changed.loc[1, "strategy_return"] += 1e-10
    assert not suite._trade_route_equal(reference, changed)


@pytest.mark.parametrize("column", ["date", "signal", "position", "one_way_turnover"])
def test_trade_route_equal_requires_exact_discrete_route(column: str) -> None:
    dates = pd.bdate_range("2023-01-02", periods=4)
    reference = _trade_frame(
        dates,
        position=[1.0, 0.0, 0.0, 1.0],
        turnover=[0.0, 1.0, 0.0, 1.0],
    )
    changed = reference.copy()
    if column == "date":
        changed.loc[1, column] += pd.Timedelta(days=1)
    else:
        changed.loc[1, column] = 1.0 - changed.loc[1, column]

    assert not suite._trade_route_equal(reference, changed)


def test_metric_rows_use_paper_turnover_and_count_switches() -> None:
    dates = pd.bdate_range("2023-01-02", periods=4)
    path = _trade_frame(
        dates,
        position=[1.0, 0.0, 0.0, 1.0],
        turnover=[0.0, 1.0, 0.0, 1.0],
    )
    paths = {
        "buy_and_hold": path,
        "hmm": path,
        "fixed_jm": path,
        "static_lambda50": path,
    }

    rows = suite._metric_rows("us", paths, load_config(ROOT / "research.toml"))
    static = next(row for row in rows if row["model"] == "static_lambda50")

    assert suite.PAPER_TURNOVER_SCALE == 0.5
    assert static["turnover"] == pytest.approx(0.5 * (2 / 4) * 252)
    assert static["switch_count"] == 2
    assert static["cash_fraction"] == pytest.approx(0.5)


def test_decision_requires_strict_positive_gap_in_every_market(monkeypatch) -> None:
    stronger_control = 0.7
    challenger_sharpes = {
        "static_lambda50": {market: 0.8 for market in suite.MARKETS},
        "dd_only": {"us": 0.8, "de": 0.8, "jp": stronger_control},
        "confirmed_2d": {market: 0.6 for market in suite.MARKETS},
        "return_aware": {market: 0.65 for market in suite.MARKETS},
        "robust_l1": {market: 0.4 for market in suite.MARKETS},
    }

    def fake_metrics(path, **_kwargs):
        return {
            "sharpe": float(path["strategy_return"].iloc[0]),
            "leverage": 1.0,
        }

    monkeypatch.setattr(suite, "performance_metrics", fake_metrics)
    config = load_config(ROOT / "research.toml")
    dates = pd.bdate_range("2023-01-02", periods=4)
    rows = []
    for market in suite.MARKETS:
        sharpes = {
            "buy_and_hold": 0.5,
            "hmm": stronger_control,
            "fixed_jm": 0.75,
            **{
                variant: values[market]
                for variant, values in challenger_sharpes.items()
            },
        }
        paths = {
            model: _trade_frame(dates, strategy_return=[sharpe] * len(dates))
            for model, sharpe in sharpes.items()
        }
        rows.extend(suite._metric_rows(market, paths, config))

    summary = pd.DataFrame.from_records(rows)
    decision = suite._decision(summary)
    by_variant = {row["variant"]: row for row in decision["variants"]}

    jp_dd = summary.loc[
        (summary["market"] == "jp") & (summary["model"] == "dd_only")
    ].iloc[0]
    assert jp_dd["gap_vs_stronger_control"] == pytest.approx(0.0)
    assert bool(jp_dd["market_pass"]) is False
    assert by_variant["dd_only"]["cross_market_support"] is False
    assert by_variant["static_lambda50"]["cross_market_support"] is True
    assert decision["supported_variants"] == ["static_lambda50"]
    assert decision["conclusion"] == "supported"


def test_math_contract_receipt_covers_nested_and_brute_force_checks() -> None:
    assert suite._verify_math_contracts() == {
        "l1_formula": True,
        "l1_brute_force": True,
        "return_gamma_zero_exact": True,
        "return_formula": True,
        "return_brute_force": True,
    }


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
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_validate_traces_accepts_complete_loss_to_t_plus_2_chain() -> None:
    suite._validate_traces(_valid_trace())


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("loss_state_0", np.nan, "loss"),
        ("loss_state_1", np.inf, "loss"),
        ("transition_penalty", np.nan, "penalty"),
        ("transition_penalty", np.inf, "penalty"),
        ("raw_state", 1.0, "raw state"),
        ("state", 1.0, "state"),
        ("signal", 0.0, "signal"),
        ("position", 0.0, "position"),
        ("trade_row", 8, r"t\+2"),
    ],
)
def test_validate_traces_rejects_incomplete_or_inconsistent_evidence(
    column: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(suite.SimpleJMSuiteError, match=message):
        suite._validate_traces(_valid_trace(**{column: value}))


def test_validate_traces_allows_confirmed_state_to_differ_from_raw_state() -> None:
    trace = _valid_trace(
        variant="confirmed_2d",
        terminal_value_state_0=1.8,
        terminal_value_state_1=1.2,
        raw_state=1.0,
        state=0.0,
        signal=1.0,
        position=1.0,
    )

    suite._validate_traces(trace)


def test_validate_traces_allows_only_unreachable_collapsed_loss_inf() -> None:
    collapsed = _valid_trace(
        loss_state_1=np.inf,
        terminal_value_state_1=np.inf,
        active_state_count=1,
        collapsed_to_one_state=True,
    )

    suite._validate_traces(collapsed)

    reachable_inf = collapsed.copy()
    reachable_inf.loc[0, "loss_state_0"] = np.inf
    with pytest.raises(suite.SimpleJMSuiteError, match="reachable.*loss"):
        suite._validate_traces(reachable_inf)


def _write_suite_contract(repo: Path, *, registered_hash: str | None = None) -> Path:
    research = repo / "research"
    research.mkdir()
    (repo / "canonical").mkdir()
    (repo / "lambda50").mkdir()
    spec = research / "simple-jm-suite-001.toml"
    spec.write_text(
        """schema_version = 1
experiment_id = "simple-jm-suite-001"
status = "FROZEN_BEFORE_RESULTS"
claim_class = "EXPLORATORY"

[sources]
canonical_run_root = "canonical"
lambda50_run_root = "lambda50"
markets = ["us", "de", "jp"]
cutoff = "2023-12-31"
post_2023_access = false

[variants.static_lambda50]
[variants.dd_only]
[variants.confirmed_2d]
[variants.return_aware]
[variants.robust_l1]
""",
        encoding="utf-8",
    )
    digest = hashlib.sha256(spec.read_bytes()).hexdigest()
    registry = {
        "experiment_id": suite.EXPERIMENT_ID,
        "status": "FROZEN",
        "frozen_spec_hash": registered_hash or digest,
    }
    (research / "experiment_registry.jsonl").write_text(
        json.dumps(registry) + "\n", encoding="utf-8"
    )
    return spec


def test_load_simple_jm_spec_accepts_matching_frozen_registry(tmp_path: Path) -> None:
    spec_path = _write_suite_contract(tmp_path)
    config = replace(
        load_config(ROOT / "research.toml"), path=tmp_path / "research.toml"
    )

    loaded = suite.load_simple_jm_spec(config, spec_path)

    assert loaded.sha256 == hashlib.sha256(spec_path.read_bytes()).hexdigest()
    assert loaded.canonical_root == (tmp_path / "canonical").resolve()
    assert loaded.lambda50_root == (tmp_path / "lambda50").resolve()


def test_load_simple_jm_spec_rejects_registry_hash_mismatch(tmp_path: Path) -> None:
    spec_path = _write_suite_contract(tmp_path, registered_hash="0" * 64)
    config = replace(
        load_config(ROOT / "research.toml"), path=tmp_path / "research.toml"
    )

    with pytest.raises(
        suite.SimpleJMSuiteError, match="no matching pre-result FROZEN registry row"
    ):
        suite.load_simple_jm_spec(config, spec_path)


def test_runner_events_use_existing_contract_without_stdout(capsys) -> None:
    events = []
    suite._emit_stage(events.append, "stage_started", "dd_only", completed=0, total=3)
    suite._emit_stage(events.append, "stage_completed", "dd_only", completed=3, total=3)
    dates = pd.bdate_range("2023-01-02", periods=2)
    signal = pd.Series([0.0, 1.0], index=dates, name="selected_signal")
    selection = SelectionResult(
        signal=signal,
        choices=pd.DataFrame({"decision_date": [dates[-1]], "selected": [35.0]}),
        surface=pd.DataFrame(),
        candidate_returns=pd.DataFrame(),
    )
    output = suite.VariantOutput(
        market="us",
        variant="dd_only",
        states=pd.DataFrame(),
        refits=pd.DataFrame(),
        selection=selection,
        selected_state=1.0 - signal,
        signal=signal,
        full_trades=pd.DataFrame(),
        boundary={
            "upper_candidate": 1200.0,
            "selected_months": 1,
            "total_months": 2,
            "fraction": 0.5,
            "limit": 0.05,
            "passed": False,
            "descriptive_only": True,
        },
    )

    suite._emit_variant_events(events.append, {("us", "dd_only"): output})

    assert [event.kind for event in events] == [
        "stage_started",
        "stage_completed",
        "selected_signal",
        "boundary_diagnostic",
    ]
    assert events[2].market == events[3].market == "us"
    assert events[2].model == events[3].model == "dd_only"
    assert capsys.readouterr().out == ""


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


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

    assert suite._implementation_source_commit(tmp_path, old_mapping) == first_commit
    with pytest.raises(suite.SimpleJMSuiteError, match="no single Git commit"):
        suite._implementation_source_commit(tmp_path, mixed_mapping)


def test_implementation_source_accepts_matching_non_git_export(tmp_path: Path) -> None:
    source = tmp_path / "runner.py"
    source.write_text("exact source\n", encoding="utf-8")
    mapping = {"runner.py": hashlib.sha256(source.read_bytes()).hexdigest()}

    assert suite._implementation_source_commit(tmp_path, mapping) is None
