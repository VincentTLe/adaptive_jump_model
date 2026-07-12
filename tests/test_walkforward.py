from datetime import date

import numpy as np
import pandas as pd
import pytest

from adaptive_jump.config import SelectionProtocol
from adaptive_jump.walkforward import (
    WalkForwardError,
    boundary_diagnostic,
    select_monthly_candidate,
)


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2020-01-02", "2022-03-31")
    equity = np.where(np.arange(len(dates)) % 3 == 0, -0.002, 0.004)
    returns = pd.DataFrame(
        {"date": dates, "equity_simple": equity, "cash_return": 0.0001}
    )
    states = pd.DataFrame({0.0: 0.0, 5.0: 1.0}, index=dates)
    return returns, states


def _selection() -> SelectionProtocol:
    return SelectionProtocol(1, 20, 1e-12, 0.05)


def test_monthly_selection_waits_for_full_calendar_history() -> None:
    returns, states = _inputs()

    result = select_monthly_candidate(
        returns,
        states,
        _selection(),
        delay_trading_days=1,
        one_way_cost_bps=10,
    )

    first_decision = result.choices.iloc[0]["decision_date"]
    assert first_decision == pd.Timestamp("2021-01-29")
    assert result.choices["selected"].eq(0.0).all()
    assert result.signal.loc[first_decision] == 1.0
    assert result.signal.loc[: first_decision - pd.Timedelta(days=1)].isna().all()


def test_numerical_tie_selects_lower_candidate() -> None:
    returns, states = _inputs()
    states[5.0] = states[0.0]

    result = select_monthly_candidate(
        returns,
        states,
        _selection(),
        delay_trading_days=1,
        one_way_cost_bps=10,
    )

    assert result.choices["selected"].eq(0.0).all()


def test_unsorted_candidate_columns_keep_their_state_paths() -> None:
    returns, states = _inputs()

    result = select_monthly_candidate(
        returns,
        states[[5.0, 0.0]],
        _selection(),
        delay_trading_days=1,
        one_way_cost_bps=10,
    )

    assert result.choices["selected"].eq(0.0).all()


def test_candidate_validation_path_uses_frozen_delay_and_cost() -> None:
    returns, states = _inputs()

    result = select_monthly_candidate(
        returns,
        states,
        _selection(),
        delay_trading_days=1,
        one_way_cost_bps=10,
    )

    assert result.candidate_returns[0.0].first_valid_index() == returns.loc[2, "date"]
    assert result.candidate_returns.loc[returns.loc[2, "date"], 0.0] == pytest.approx(
        returns.loc[2, "equity_simple"]
    )


def test_future_changes_do_not_change_past_selection() -> None:
    returns, states = _inputs()
    changed_returns = returns.copy()
    changed_returns.loc[changed_returns.index[-10] :, "equity_simple"] = -0.5

    before = select_monthly_candidate(
        returns, states, _selection(), delay_trading_days=1, one_way_cost_bps=10
    )
    after = select_monthly_candidate(
        changed_returns,
        states,
        _selection(),
        delay_trading_days=1,
        one_way_cost_bps=10,
    )

    cutoff = changed_returns.loc[changed_returns.index[-11], "date"]
    pd.testing.assert_frame_equal(
        before.choices.loc[before.choices["decision_date"] <= cutoff].reset_index(
            drop=True
        ),
        after.choices.loc[after.choices["decision_date"] <= cutoff].reset_index(
            drop=True
        ),
    )
    pd.testing.assert_series_equal(
        before.signal.loc[:cutoff], after.signal.loc[:cutoff]
    )


def test_boundary_frequency_uses_only_oos_months() -> None:
    choices = pd.DataFrame(
        {
            "decision_date": pd.to_datetime(["2020-12-31", "2021-01-29", "2021-02-26"]),
            "selected": [10.0, 10.0, 0.0],
        }
    )

    diagnostic = boundary_diagnostic(
        choices, (0.0, 10.0), oos_start=date(2021, 1, 1), fraction_limit=0.5
    )

    assert diagnostic.selected_months == 1
    assert diagnostic.total_months == 2
    assert diagnostic.fraction == 0.5
    assert diagnostic.passed


def test_boundary_check_rejects_missing_oos_choices() -> None:
    choices = pd.DataFrame(
        {"decision_date": pd.to_datetime(["2020-12-31"]), "selected": [0.0]}
    )

    with pytest.raises(WalkForwardError, match="no OOS monthly choices"):
        boundary_diagnostic(
            choices, (0.0, 10.0), oos_start=date(2021, 1, 1), fraction_limit=0.05
        )
