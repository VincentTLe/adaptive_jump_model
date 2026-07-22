import numpy as np
import pandas as pd
import pytest
from jumpmodels.jump import JumpModel

from adaptive_jump.config import HMMProtocol, JMProtocol, ModelProtocol
from adaptive_jump.models import (
    FixedJMFit,
    FixedJMResult,
    HMMResult,
    ModelError,
    best_hmm_terminal_fit,
    fit_fixed_jm_window,
    fixed_jm_states,
    hmm_states,
    smoothed_hmm_states,
    terminal_online_state,
)


def _protocols(fit_window: int = 6) -> tuple[ModelProtocol, JMProtocol]:
    model = ModelProtocol(2, fit_window, 0, 1)
    jm = JMProtocol((5.0,), 4, 0, 100, 1e-8, (1, 7))
    return model, jm


def _hmm_protocol(seeds: tuple[int, ...] = (0, 1, 2)) -> HMMProtocol:
    return HMMProtocol((0, 2), seeds, 0.001, 100, 1e-6)


class _FakeHMM:
    def __init__(self, *, random_state: int, **_: object) -> None:
        self.seed = random_state
        self.delta = 1e-3 if random_state == 0 else 5e-7
        self.covars_ = np.array([[[9.0]], [[1.0]]])

    def fit(self, _: np.ndarray) -> "_FakeHMM":
        self.monitor_.report(0.0)
        self.monitor_.report(self.delta)
        return self

    def score(self, _: np.ndarray) -> float:
        return float(self.seed)

    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.zeros(len(values), dtype=int)


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


def test_fixed_jm_observer_is_output_neutral() -> None:
    model, jm = _protocols()
    events = []

    baseline = fixed_jm_states(_frame(), model, jm)
    observed = fixed_jm_states(_frame(), model, jm, observer=events.append)

    pd.testing.assert_frame_equal(observed.states, baseline.states)
    pd.testing.assert_frame_equal(observed.refits, baseline.refits)
    assert events[0].kind == "stage_started"
    assert events[-1].kind == "stage_completed"
    terminals = [event for event in events if event.kind == "terminal_state"]
    assert len(terminals) == 9
    assert terminals[-1].completed == terminals[-1].total == 9
    assert terminals[-1].payload["states"] == [{"candidate": 5.0, "state": 1}]


def test_fixed_jm_default_loss_scale_is_exactly_unchanged() -> None:
    model, jm = _protocols()

    implicit = fixed_jm_states(_frame(), model, jm, include_fit_diagnostics=True)
    explicit = fixed_jm_states(
        _frame(),
        model,
        jm,
        observation_loss_scale=1.0,
        include_fit_diagnostics=True,
    )

    pd.testing.assert_frame_equal(explicit.states, implicit.states, check_exact=True)
    assert explicit.refits.to_csv(index=False) == implicit.refits.to_csv(index=False)


def test_scale_three_matches_one_third_lambda_in_fit_and_online_path() -> None:
    model, _ = _protocols()
    frame = _frame()
    base_protocol = JMProtocol((5.0 / 3.0,), 4, 0, 100, 1e-8, (1, 7))
    scaled_protocol = JMProtocol((5.0,), 4, 0, 100, 1e-8, (1, 7))
    window = frame.iloc[: model.fit_window]

    base_fit = fit_fixed_jm_window(
        window,
        model,
        base_protocol,
        feature_columns=("dd_10",),
    )
    scaled_fit = fit_fixed_jm_window(
        window,
        model,
        scaled_protocol,
        feature_columns=("dd_10",),
        observation_loss_scale=3.0,
    )
    base_path = fixed_jm_states(
        frame,
        model,
        base_protocol,
        feature_columns=("dd_10",),
    )
    scaled_path = fixed_jm_states(
        frame,
        model,
        scaled_protocol,
        feature_columns=("dd_10",),
        observation_loss_scale=3.0,
    )

    assert isinstance(scaled_fit, FixedJMFit)
    assert scaled_fit.observation_loss_scale == 3.0
    np.testing.assert_array_equal(
        scaled_fit.models[5.0].labels_,
        base_fit.models[5.0 / 3.0].labels_,
    )
    np.testing.assert_allclose(
        scaled_fit.models[5.0].centers_ / np.sqrt(3.0),
        base_fit.models[5.0 / 3.0].centers_,
        rtol=0,
        atol=1e-15,
    )
    assert scaled_fit.models[5.0].val_ / 3.0 == pytest.approx(
        base_fit.models[5.0 / 3.0].val_,
        rel=0,
        abs=1e-15,
    )
    pd.testing.assert_series_equal(
        scaled_path.states[5.0],
        base_path.states[5.0 / 3.0],
        check_names=False,
        check_exact=True,
    )


@pytest.mark.parametrize("scale", [True, 0.0, -1.0, np.inf, np.nan])
def test_fixed_jm_rejects_invalid_observation_loss_scale(scale: float) -> None:
    model, jm = _protocols()

    with pytest.raises(ModelError, match="loss scale"):
        fixed_jm_states(_frame(), model, jm, observation_loss_scale=scale)


def test_scaled_fixed_jm_rejects_checkpoint_resume() -> None:
    model, jm = _protocols()
    frame = _frame()
    initial = fixed_jm_states(frame, model, jm)

    with pytest.raises(ModelError, match="checkpoint resume"):
        fixed_jm_states(
            frame,
            model,
            jm,
            observation_loss_scale=3.0,
            initial=initial,
        )


def test_fixed_jm_resumes_exactly_from_causal_checkpoint() -> None:
    model, jm = _protocols()
    frame = _frame()
    captured: list[FixedJMResult] = []

    def interrupt(result: FixedJMResult) -> None:
        captured.append(result)
        raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        fixed_jm_states(frame, model, jm, checkpoint_every=3, progress=interrupt)
    events = []
    resumed_checkpoints = []
    resumed = fixed_jm_states(
        frame,
        model,
        jm,
        initial=captured[0],
        progress=resumed_checkpoints.append,
        observer=events.append,
    )
    uninterrupted = fixed_jm_states(frame, model, jm)

    pd.testing.assert_frame_equal(resumed.states, uninterrupted.states)
    pd.testing.assert_frame_equal(resumed.refits, uninterrupted.refits)
    pd.testing.assert_frame_equal(resumed_checkpoints[-1].states, resumed.states)
    assert events[0].completed == 3
    assert sum(event.kind == "terminal_state" for event in events) == 6
    assert resumed.refits["fit_date"].value_counts().eq(1).all()


def test_fixed_jm_accepts_empty_checkpoint() -> None:
    model, jm = _protocols()
    frame = _frame()
    uninterrupted = fixed_jm_states(frame, model, jm)
    empty = FixedJMResult(
        pd.DataFrame(
            pd.NA,
            index=uninterrupted.states.index,
            columns=(5,),
        ),
        uninterrupted.refits.iloc[:0].copy(),
    )

    resumed = fixed_jm_states(frame, model, jm, initial=empty)
    already_complete = fixed_jm_states(frame, model, jm, initial=uninterrupted)

    pd.testing.assert_frame_equal(resumed.states, uninterrupted.states)
    pd.testing.assert_frame_equal(resumed.refits, uninterrupted.refits)
    pd.testing.assert_frame_equal(already_complete.states, uninterrupted.states)
    pd.testing.assert_frame_equal(already_complete.refits, uninterrupted.refits)


def test_fixed_jm_rejects_invalid_checkpoint_prefix() -> None:
    model, jm = _protocols()
    frame = _frame()
    complete = fixed_jm_states(frame, model, jm)
    gapped_states = complete.states.copy()
    gapped_states.loc[pd.Timestamp("2020-12-30"), 5.0] = np.nan
    two_jm = JMProtocol((0.0, 5.0), 4, 0, 100, 1e-8, (1, 7))
    partial_states = fixed_jm_states(frame, model, two_jm).states
    partial_states.loc[pd.Timestamp("2020-12-30"), 0.0] = np.nan

    with pytest.raises(ModelError, match="contiguous causal prefix"):
        fixed_jm_states(
            frame, model, jm, initial=FixedJMResult(gapped_states, complete.refits)
        )
    with pytest.raises(ModelError, match="partial candidate row"):
        fixed_jm_states(
            frame,
            model,
            two_jm,
            initial=FixedJMResult(partial_states, complete.refits),
        )
    with pytest.raises(ModelError, match="refit dates violate the prefix"):
        fixed_jm_states(
            frame,
            model,
            jm,
            initial=FixedJMResult(complete.states, complete.refits.iloc[:1]),
        )
    with pytest.raises(ModelError, match="refits are incomplete"):
        fixed_jm_states(
            frame,
            model,
            jm,
            initial=FixedJMResult(
                complete.states, complete.refits.drop(columns="objective")
            ),
        )


def test_rejects_nonfinite_or_malformed_model_inputs() -> None:
    model, jm = _protocols()
    malformed = _frame().drop(columns="dd_10")
    with pytest.raises(ModelError, match="missing model columns"):
        fixed_jm_states(malformed, model, jm)

    nonfinite = _frame()
    nonfinite.loc[3, "dd_10"] = np.inf
    with pytest.raises(ModelError, match="must be finite"):
        fixed_jm_states(nonfinite, model, jm)
    with pytest.raises(ModelError, match="checkpoint interval must be positive"):
        fixed_jm_states(_frame(), model, jm, checkpoint_every=0)


def test_hmm_rejects_bad_restart_and_selects_highest_likelihood(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kmeans_calls: list[dict[str, object]] = []
    hmm_calls: list[dict[str, object]] = []

    class RecordingKMeans:
        def __init__(self, **kwargs: object) -> None:
            kmeans_calls.append(dict(kwargs))

        def fit(self, _: np.ndarray) -> "RecordingKMeans":
            self.cluster_centers_ = np.array([[-0.25], [0.25]])
            return self

    class RecordingHMM(_FakeHMM):
        def __init__(self, **kwargs: object) -> None:
            hmm_calls.append(dict(kwargs))
            super().__init__(**kwargs)

        def fit(self, values: np.ndarray) -> "RecordingHMM":
            np.testing.assert_array_equal(self.means_, [[-0.25], [0.25]])
            return super().fit(values)

    monkeypatch.setattr("adaptive_jump.models.KMeans", RecordingKMeans)
    monkeypatch.setattr("adaptive_jump.models.GaussianHMM", RecordingHMM)
    protocol = HMMProtocol(
        (0, 2),
        (0, 1, 2),
        0.001,
        100,
        1e-6,
        kmeans_n_init=1,
        covars_prior=0.0,
    )
    model = ModelProtocol(2, 4, 0, 1)

    fit = best_hmm_terminal_fit(pd.Series([0.1, -0.1, 0.2, -0.2]), model, protocol)

    assert fit.seed == 2
    assert fit.log_likelihood == 2.0
    assert fit.terminal_state == 1
    assert fit.variances == (1.0, 9.0)
    assert fit.accepted_starts == 2
    assert fit.failed_starts[0].startswith(
        "seed=0: ModelError: strict convergence failed"
    )

    assert kmeans_calls == [
        {
            "n_clusters": 2,
            "init": "k-means++",
            "n_init": 1,
            "random_state": seed,
        }
        for seed in (0, 1, 2)
    ]
    explicit = {
        "startprob_prior": 1.0,
        "transmat_prior": 1.0,
        "means_prior": 0.0,
        "means_weight": 0.0,
        "covars_prior": 0.0,
        "covars_weight": 1.0,
        "verbose": False,
        "params": "stmc",
        "init_params": "stc",
        "implementation": "log",
    }
    assert [call["random_state"] for call in hmm_calls] == [0, 1, 2]
    for call in hmm_calls:
        assert {key: call[key] for key in explicit} == explicit


def test_real_hmm_labels_low_and_high_conditional_variance() -> None:
    rng = np.random.default_rng(7)
    returns = np.r_[rng.normal(0, 0.005, 120), rng.normal(0, 0.03, 120)]
    model = ModelProtocol(2, len(returns), 0, 1)
    protocol = HMMProtocol((0, 2), (0, 1, 2), 0.001, 500, 1e-6)

    fit = best_hmm_terminal_fit(pd.Series(returns), model, protocol)

    assert fit.terminal_state == 1
    assert fit.variances[0] < fit.variances[1]
    assert 1 <= fit.accepted_starts <= 3


@pytest.mark.parametrize("delta", [-0.1, 0.1])
def test_hmm_rejects_monitor_false_positive_convergence(
    monkeypatch: pytest.MonkeyPatch, delta: float
) -> None:
    class MisreportedConvergence(_FakeHMM):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)
            self.delta = delta

    monkeypatch.setattr("adaptive_jump.models.GaussianHMM", MisreportedConvergence)
    model = ModelProtocol(2, 4, 0, 1)

    with pytest.raises(ModelError, match="all HMM restarts failed"):
        best_hmm_terminal_fit(pd.Series([0.1, -0.1, 0.2, -0.2]), model, _hmm_protocol())


def test_hmm_accepts_small_negative_delta_within_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NumericalOscillation(_FakeHMM):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)
            self.delta = -5e-7

    monkeypatch.setattr("adaptive_jump.models.GaussianHMM", NumericalOscillation)
    model = ModelProtocol(2, 4, 0, 1)

    fit = best_hmm_terminal_fit(
        pd.Series([0.1, -0.1, 0.2, -0.2]), model, _hmm_protocol()
    )

    assert fit.accepted_starts == 3


def test_hmm_rejects_max_iteration_without_tolerance_convergence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class HitsIterationLimit(_FakeHMM):
        def fit(self, _: np.ndarray) -> "HitsIterationLimit":
            for iteration in range(self.monitor_.n_iter):
                self.monitor_.report(float(iteration))
            return self

    monkeypatch.setattr("adaptive_jump.models.GaussianHMM", HitsIterationLimit)
    model = ModelProtocol(2, 4, 0, 1)

    with pytest.raises(ModelError, match="all HMM restarts failed"):
        best_hmm_terminal_fit(pd.Series([0.1, -0.1, 0.2, -0.2]), model, _hmm_protocol())


def test_hmm_daily_fit_is_causal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("adaptive_jump.models.GaussianHMM", _FakeHMM)
    model = ModelProtocol(2, 4, 0, 1)
    frame = _frame(9).rename(columns={"excess_return": "equity_log"})
    changed = frame.copy()
    changed.loc[changed.index[-1], "equity_log"] = -99.0

    before = hmm_states(frame, model, _hmm_protocol()).states
    after = hmm_states(changed, model, _hmm_protocol()).states

    pd.testing.assert_series_equal(before.iloc[:-1], after.iloc[:-1])


def test_hmm_observer_is_output_neutral(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("adaptive_jump.models.GaussianHMM", _FakeHMM)
    model = ModelProtocol(2, 4, 0, 1)
    frame = _frame(9).rename(columns={"excess_return": "equity_log"})
    events = []

    baseline = hmm_states(frame, model, _hmm_protocol())
    observed = hmm_states(frame, model, _hmm_protocol(), observer=events.append)

    pd.testing.assert_series_equal(observed.states, baseline.states)
    pd.testing.assert_frame_equal(observed.fits, baseline.fits)
    assert [events[0].kind, events[-1].kind] == ["stage_started", "stage_completed"]
    terminals = [event for event in events if event.kind == "terminal_state"]
    assert len(terminals) == 6
    assert terminals[-1].completed == terminals[-1].total == 6
    assert terminals[-1].payload["state"] == 1


def test_hmm_resumes_from_contiguous_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("adaptive_jump.models.GaussianHMM", _FakeHMM)
    model = ModelProtocol(2, 4, 0, 1)
    frame = _frame(10).rename(columns={"excess_return": "equity_log"})
    captured = {}

    def stop_after_first(result: HMMResult) -> None:
        captured["result"] = result
        raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        hmm_states(
            frame,
            model,
            _hmm_protocol(),
            checkpoint_every=2,
            progress=stop_after_first,
        )
    resumed = hmm_states(frame, model, _hmm_protocol(), initial=captured["result"])
    complete = hmm_states(frame, model, _hmm_protocol())

    pd.testing.assert_series_equal(resumed.states, complete.states)
    pd.testing.assert_frame_equal(resumed.fits, complete.fits)


def test_parallel_hmm_matches_sequential_results() -> None:
    rng = np.random.default_rng(11)
    returns = np.r_[rng.normal(0, 0.005, 120), rng.normal(0, 0.03, 122)]
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-02", periods=len(returns)),
            "equity_log": returns,
        }
    )
    model = ModelProtocol(2, 240, 0, 1)
    protocol = HMMProtocol((0, 2), (0, 1), 0.001, 500, 1e-6)

    sequential = hmm_states(frame, model, protocol, n_jobs=1)
    parallel = hmm_states(frame, model, protocol, n_jobs=2)

    pd.testing.assert_series_equal(sequential.states, parallel.states)
    pd.testing.assert_frame_equal(sequential.fits, parallel.fits)


def test_hmm_majority_filter_uses_strict_half_threshold() -> None:
    states = pd.Series([np.nan, 0.0, 1.0, 1.0, 0.0])

    candidates = smoothed_hmm_states(states, (0, 2, 4))

    full = smoothed_hmm_states(states, (0, 2, 4), require_full_window=True)

    assert full[0].equals(candidates[0])
    assert full[2].first_valid_index() == 2
    assert full[4].first_valid_index() == 4
    assert full[4].iloc[:4].isna().all()
    assert np.isnan(candidates[0].iloc[0])
    assert candidates[0].iloc[1] == 0.0
    assert candidates[2].iloc[2] == 0.0
    assert candidates[2].iloc[3] == 1.0
    assert candidates[4].iloc[4] == 0.0


def test_hmm_raises_when_every_restart_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    class NeverConverges(_FakeHMM):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)
            self.delta = 1.0

    monkeypatch.setattr("adaptive_jump.models.GaussianHMM", NeverConverges)
    model = ModelProtocol(2, 4, 0, 1)

    with pytest.raises(ModelError, match="all HMM restarts failed"):
        best_hmm_terminal_fit(pd.Series([0.1, -0.1, 0.2, -0.2]), model, _hmm_protocol())
