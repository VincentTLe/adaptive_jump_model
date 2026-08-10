"""Independent audit regressions for ``scripts/score_grid.py``.

Every test here pins a property whose violation has already produced a wrong
number somewhere in this project: a scorer that matches the wrong penalty
column, a known-answer test that cannot fail, a comparison that reads NaN as
agreement, and a timing convention that drifts away from t + 2.

Tests marked ``xfail(strict=True)`` describe a defect that is present today.
Repairing the scorer turns them into XPASS, which fails the suite and forces
the marker to be removed together with the fix. Nothing here refits a model:
the four self-test regressions stub :func:`score_grid.score` out entirely, and
the pipeline regressions run on synthetic frames.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adaptive_jump.backtest import apply_signal
from adaptive_jump.config import SelectionProtocol, load_config
from adaptive_jump.walkforward import WalkForwardError, select_monthly_candidate

ROOT = Path(__file__).resolve().parents[1]
UNION_STATES = ROOT / "artifacts/jm-residual/01-grid-identification/us/union-states.csv"
HEADERS = Path(__file__).parent / "fixtures/searched-menu-headers"
# (searched menu, tracked copy of its header line). The menus are multi-megabyte
# state matrices .gitignore keeps local; only their column names are under test,
# so the header copy stands in on a checkout that does not have them. See that
# directory's README for how the copies were made.
MENUS = (
    (UNION_STATES, HEADERS / "union-states.csv"),
    (ROOT / "artifacts/dense-menu/01-search/states-de.csv", HEADERS / "states-de.csv"),
    (ROOT / "artifacts/dense-menu/01-search/states-jp.csv", HEADERS / "states-jp.csv"),
)


def _require(*paths: Path) -> None:
    """Skip, naming the file, when a local-only run artifact is absent.

    These artifacts are large regenerable run outputs that .gitignore keeps out
    of the repository, so a clean checkout does not have them. The audit only
    means anything against the real artifact, so it says which one is missing
    instead of dying on FileNotFoundError or passing on nothing.
    """
    for path in paths:
        if not path.exists():
            pytest.skip(f"{path.relative_to(ROOT)} not built in this checkout")


@pytest.fixture(scope="module")
def scorer():
    """Import ``scripts/score_grid.py`` without executing ``main()``."""
    path = ROOT / "scripts/score_grid.py"
    if str(ROOT / "scripts") not in sys.path:
        # main() imports _shu_table4 before it puts scripts/ on the path, which
        # only works because argv[0]'s directory is already there under
        # `python scripts/score_grid.py`. Importing the module has to supply it.
        sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("score_grid_under_audit", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sealed_cells(module, market: str, delay: int) -> dict[str, float]:
    reported = pd.read_csv(module.BASE / "metrics.csv")
    row = reported[
        (reported.market == market)
        & (reported.model == "fixed_jm")
        & (reported.delay == delay)
    ].iloc[0]
    return {cell: float(row[cell]) for cell in module.CELLS}


def _states_csv(tmp_path: Path, columns: dict[str, float]) -> Path:
    index = pd.bdate_range("2020-01-02", periods=5)
    frame = pd.DataFrame({name: value for name, value in columns.items()}, index=index)
    frame.index.name = "date"
    path = tmp_path / "states.csv"
    frame.to_csv(path)
    return path


# --------------------------------------------------------------------------
# 1. the wrong penalty column
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("columns", "penalty"),
    [
        ({"1.0": 0.0, "1.1": 1.0}, 1.05),  # nearest-neighbour snapping
        ({"1.0": 0.0, "1.1": 1.0}, 2.0),  # absent from the menu
        ({"1.0": 0.0, "1.0000000005": 1.0}, 1.0),  # ambiguous within tolerance
    ],
)
def test_penalty_must_match_exactly_one_column(scorer, tmp_path, columns, penalty):
    """A penalty that is not unambiguously a column must stop the run.

    Snapping to the nearest column is the failure that matters: the searched
    menus are log spaced, so a mistyped lambda lands close enough to a
    neighbour to look plausible and score a different model.
    """
    _require(scorer.BASE / "us/features.csv")
    source = _states_csv(tmp_path, columns)
    with pytest.raises(SystemExit) as raised:
        scorer.score("us", [penalty], source)
    assert "not a column" in str(raised.value)


def test_a_penalty_present_in_the_menu_is_accepted(scorer, tmp_path):
    """Control: matching succeeds, so the guard above is not vacuous.

    The duplicated penalty is rejected further down the pipeline, by the
    uniqueness check inside the selection code, not by the column matcher.
    """
    _require(scorer.BASE / "us/features.csv")
    source = _states_csv(tmp_path, {"1.0": 0.0, "1.1": 1.0})
    with pytest.raises(WalkForwardError, match="candidate values must be unique"):
        scorer.score("us", [1.0, 1.0], source)


def test_searched_menus_are_far_wider_apart_than_the_matching_tolerance():
    """No two menu columns may be confusable at the matcher's tolerance."""
    for menu, header in MENUS:
        # Where the menu itself is present, the tracked copy must still BE its
        # header, so a regenerated menu cannot leave a stale copy passing here.
        if menu.exists():
            with menu.open(encoding="utf-8") as handle:
                assert next(handle) == header.read_text(encoding="utf-8"), (
                    f"{header.name} is stale: it is no longer the first line of "
                    f"{menu.relative_to(ROOT)}"
                )
        names = pd.read_csv(header, nrows=0).columns[1:]
        values = sorted(float(name) for name in names)
        gaps = [
            abs(a - b) / max(abs(b), 1e-9)
            for a, b in zip(values[1:], values[:-1], strict=True)
        ]
        assert min(gaps) > 1e-3, f"{menu.name}: columns are only {min(gaps):.2e} apart"
        assert not any(
            np.isclose(a, b, rtol=1e-9, atol=1e-9)
            for a, b in zip(values[1:], values[:-1], strict=True)
        )


# --------------------------------------------------------------------------
# 2. a known-answer test that cannot fail
# --------------------------------------------------------------------------


def test_self_test_rejects_a_result_that_is_all_nan(scorer, monkeypatch):
    _require(scorer.BASE / "metrics.csv")
    # The stubs below take ``delay`` because score() gained that parameter for
    # heldout-delay-001. Without it these three regressions died of TypeError
    # instead of testing anything, which is how a guard rail stops guarding.
    monkeypatch.setattr(
        scorer,
        "score",
        lambda market, penalties, states_csv=None, delay=1: dict.fromkeys(
            scorer.CELLS, math.nan
        ),
    )
    with pytest.raises(SystemExit):
        scorer.self_test()


def test_self_test_exercises_the_states_csv_argument(scorer, monkeypatch):
    # self_test() only reaches the states_csv path when the union menu is on
    # disk, so without it this test would assert that a branch it never entered
    # ran. The header copy cannot stand in: the gate is on the real file.
    _require(scorer.BASE / "metrics.csv", UNION_STATES)
    seen: list[object] = []

    def recorder(market, penalties, states_csv=None, delay=1):
        seen.append(states_csv)
        return _sealed_cells(scorer, market, delay)

    monkeypatch.setattr(scorer, "score", recorder)
    scorer.self_test()
    assert any(source is not None for source in seen)


# --------------------------------------------------------------------------
# 3. a timing convention that drifts from t + 2
# --------------------------------------------------------------------------


def test_apply_signal_acts_two_rows_after_the_signal():
    """The frozen convention: decide on day t, hold for the return of t + 2."""
    dates = pd.bdate_range("2020-01-02", periods=6)
    returns = pd.DataFrame(
        {"date": dates, "equity_simple": 0.01, "cash_return": 0.0001}
    )
    signal = pd.Series([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    path = apply_signal(returns, signal, delay_trading_days=1, one_way_cost_bps=10)

    assert path["position"].tolist()[:2] == [pytest.approx(np.nan, nan_ok=True)] * 2
    assert path["position"].tolist()[2:] == [1.0, 1.0, 0.0, 0.0]
    # the switch is charged on the row where the position actually changes
    assert path["one_way_turnover"].tolist()[2:] == [0.0, 0.0, 1.0, 0.0]
    assert path["transaction_cost"].iloc[4] == pytest.approx(10 / 10_000)
    assert path["strategy_return"].iloc[4] == pytest.approx(0.0001 - 10 / 10_000)


def test_scorer_constants_match_the_frozen_contract(scorer):
    """The scorer hardcodes delay, cost and drawdown basis; pin them to v10."""
    config = load_config(ROOT / "configs/baselines/legacy/research-calibrated-v10.toml")
    backtest = config.document["backtest"]
    assert scorer.DELAY == backtest["primary_delay_trading_days"]
    assert scorer.DELAY + 1 == backtest["signal_to_return_offset"]
    assert scorer.COST == float(backtest["one_way_cost_bps"])
    source = (ROOT / "scripts/score_grid.py").read_text(encoding="utf-8")
    basis = config.metrics_protocol.drawdown_basis
    assert f'drawdown_basis="{basis}"' in source


def test_self_test_rejects_a_changed_delay(scorer, monkeypatch):
    monkeypatch.setattr(scorer, "DELAY", 5)
    monkeypatch.setattr(
        scorer,
        "score",
        lambda market, penalties, states_csv=None: _sealed_cells(
            scorer, market, scorer.DELAY
        ),
    )
    with pytest.raises(SystemExit):
        scorer.self_test()


# --------------------------------------------------------------------------
# 4. NaN read as agreement in the reported headline
# --------------------------------------------------------------------------


def test_reported_worst_deviation_is_not_blind_to_nan(scorer, monkeypatch, capsys):
    monkeypatch.setattr(
        scorer,
        "score",
        lambda market, penalties, states_csv=None, delay=1: dict.fromkeys(
            (*scorer.CELLS, "cagr", "sharpe", "calmar", "switches"), math.nan
        ),
    )
    monkeypatch.setattr(sys, "argv", ["score_grid.py", "us", "15,70"])
    scorer.main()
    assert "worst relative deviation 0.0%" not in capsys.readouterr().out


# --------------------------------------------------------------------------
# 5. the sample the metrics are measured on
# --------------------------------------------------------------------------


def test_sample_start_moves_when_one_candidate_column_starts_late():
    """A grid is only comparable to Table 4 on Table 4's window.

    Monthly selection cannot start until every candidate has a state, so one
    late column silently shortens the measured sample. A scorer that clips to
    a fixed window and reports no start, end or observation count cannot show
    that this has happened, which is why the sample has to be asserted.
    """
    dates = pd.bdate_range("2015-01-01", periods=900)
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(
        {
            "date": dates,
            "equity_simple": rng.normal(0.0004, 0.01, len(dates)),
            "cash_return": 0.00005,
        }
    )
    states = pd.DataFrame(
        {
            0.0: (np.arange(len(dates)) // 7 % 2).astype(float),
            5.0: (np.arange(len(dates)) // 40 % 2).astype(float),
        },
        index=dates,
    )
    late = states.copy()
    late.iloc[:200, 1] = np.nan
    protocol = SelectionProtocol(1, 20, 1e-12, 1.0)

    def first_signal(frame: pd.DataFrame) -> pd.Timestamp:
        selection = select_monthly_candidate(
            returns,
            frame,
            protocol,
            delay_trading_days=1,
            one_way_cost_bps=10.0,
        )
        return selection.signal.first_valid_index()

    complete, truncated = first_signal(states), first_signal(late)
    assert truncated > complete
    assert (truncated - complete).days > 200
