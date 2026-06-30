import numpy as np
import pytest

from adaptive_jump.hmm import fit_gaussian_hmm


def test_hmm_fits_simple_two_regime_gaussian_data():
    rng = np.random.default_rng(123)
    low = rng.normal(0.0, 0.1, size=120)
    high = rng.normal(0.0, 1.0, size=120)
    x = np.r_[low, high]

    result = fit_gaussian_hmm(x, n_states=2, n_init=4, max_iter=50, random_state=7)

    assert len(result.states) == len(x)
    assert result.means.shape == (2,)
    assert result.variances.shape == (2,)
    assert result.startprob.shape == (2,)
    assert result.transmat.shape == (2, 2)
    assert np.allclose(result.transmat.sum(axis=1), 1.0)
    assert (result.variances > 0).all()
    assert result.variances[0] <= result.variances[1]
    assert set(result.states.tolist()) <= {0, 1}
    assert np.isfinite(result.loglik)


def test_hmm_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="1-D"):
        fit_gaussian_hmm(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="finite"):
        fit_gaussian_hmm(np.array([0.0, np.nan]))
    with pytest.raises(ValueError, match="n_states"):
        fit_gaussian_hmm(np.array([0.0, 1.0]), n_states=0)
    with pytest.raises(ValueError, match="n_init"):
        fit_gaussian_hmm(np.array([0.0, 1.0]), n_init=0)
