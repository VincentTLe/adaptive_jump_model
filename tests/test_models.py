import numpy as np
import pandas as pd
import pytest
from jumpmodels.jump import JumpModel

from adaptive_jump.config import JMProtocol, ModelProtocol
from adaptive_jump.models import ModelError, fixed_jm_states, terminal_online_state


def _protocols(fit_window: int = 6) -> tuple[ModelProtocol, JMProtocol]:
    model = ModelProtocol(2, fit_window, 0, 1)
    jm = JMProtocol((5.0,), 4, 0, 100, 1e-8, (1, 7))
    return model, jm


def _frame(periods: int = 14) -> pd.DataFrame:
    dates = pd.bdate_range("2020-12-21", periods=periods)
    regime = np.where(np.arange(periods) < periods // 2, 1.0, -1.0)
    return pd.DataFrame(
        {
            "date": dates,
            "dd_10": -regime + np.arange(periods) * 0.01,
            "sortino_20": regime,
            "sortino_60": regime * 0.5,
            "excess_return": regime * 0.01,
        }
    )


def test_terminal_state_matches_upstream_online_dp() -> None:
    features = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    returns = np.array([0.02, 0.01, -0.01, -0.02])
    model = JumpModel(n_components=2, jump_penalty=5, n_init=4, random_state=0)
    model.fit(features, ret_ser=returns, sort_by="cumret")

    expected = int(np.asarray(model.predict_online(features))[-1])

    assert terminal_online_state(model, features) == expected


def test_fixed_jm_refits_first_eligible_then_first_january_date() -> None:
    model, jm = _protocols()

    result = fixed_jm_states(_frame(), model, jm)

    fit_dates = result.refits["fit_date"].drop_duplicates().tolist()
    assert fit_dates == [pd.Timestamp("2020-12-28"), pd.Timestamp("2021-01-01")]
    assert result.refits["observations"].eq(6).all()
    assert result.states.loc[:"2020-12-25"].isna().all().all()
    assert result.states.loc["2020-12-28":, 5.0].isin([0.0, 1.0]).all()


def test_fixed_jm_does_not_change_past_states_when_future_changes() -> None:
    model, jm = _protocols()
    original = _frame()
    changed = original.copy()
    changed.loc[changed.index[-2] :, ["dd_10", "sortino_20", "sortino_60"]] *= 100

    before = fixed_jm_states(original, model, jm).states
    after = fixed_jm_states(changed, model, jm).states

    pd.testing.assert_series_equal(before.iloc[:-2, 0], after.iloc[:-2, 0])


def test_fixed_jm_uses_cumulative_return_state_order() -> None:
    model, jm = _protocols()

    result = fixed_jm_states(_frame(), model, jm)
    first_fit = result.refits.iloc[0]

    assert first_fit["lambda"] == 5.0
    assert np.isfinite(first_fit["objective"])


def test_rejects_nonfinite_or_malformed_model_inputs() -> None:
    model, jm = _protocols()
    malformed = _frame().drop(columns="dd_10")
    with pytest.raises(ModelError, match="missing model columns"):
        fixed_jm_states(malformed, model, jm)

    nonfinite = _frame()
    nonfinite.loc[3, "dd_10"] = np.inf
    with pytest.raises(ModelError, match="must be finite"):
        fixed_jm_states(nonfinite, model, jm)
