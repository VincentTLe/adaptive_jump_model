import numpy as np
import pytest

from adaptive_jump.jump_model import fit_jump_model


def test_jump_model_fits_persistent_two_block_data():
    features = np.r_[np.zeros((20, 1)), np.ones((20, 1)) * 5.0]

    result = fit_jump_model(features, switch_penalty=1.0, n_states=2, n_init=4, random_state=11)

    assert len(result.states) == len(features)
    assert result.centers.shape == (2, 1)
    assert result.n_switches == 1
    assert len(set(result.states[:20])) == 1
    assert len(set(result.states[20:])) == 1
    assert result.states[19] != result.states[20]
    assert np.isfinite(result.total_cost)


def test_high_penalty_reduces_switches_relative_to_low_penalty():
    features = np.array([[0.0], [5.0], [0.0], [5.0], [0.0], [5.0]])

    low = fit_jump_model(features, switch_penalty=0.0, n_states=2, n_init=3, random_state=1)
    high = fit_jump_model(features, switch_penalty=100.0, n_states=2, n_init=3, random_state=1)

    assert high.n_switches <= low.n_switches


def test_jump_model_accepts_vector_penalty():
    features = np.r_[np.zeros((5, 1)), np.ones((5, 1)) * 4.0]
    penalty = np.ones(len(features))

    result = fit_jump_model(features, switch_penalty=penalty, n_states=2, n_init=3, random_state=3)

    assert len(result.states) == len(features)
    assert isinstance(result.switch_penalty, np.ndarray)


def test_jump_model_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="2-D"):
        fit_jump_model(np.array([1.0, 2.0]), switch_penalty=1.0)
    with pytest.raises(ValueError, match="finite"):
        fit_jump_model(np.array([[1.0], [np.nan]]), switch_penalty=1.0)
    with pytest.raises(ValueError, match="nonnegative"):
        fit_jump_model(np.array([[1.0], [2.0]]), switch_penalty=-1.0)
    with pytest.raises(ValueError, match="length"):
        fit_jump_model(np.array([[1.0], [2.0]]), switch_penalty=np.array([1.0]))
