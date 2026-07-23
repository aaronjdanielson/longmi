"""Tests for the StatsmodelsGLM adapter."""

import numpy as np
import pandas as pd
import pytest

statsmodels = pytest.importorskip("statsmodels")
import statsmodels.api as sm  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402

from longmi.analysis import StatsmodelsGLM  # noqa: E402


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(71)
    n = 300
    x = rng.standard_normal(n)
    wave = rng.choice([1, 2, 3], size=n)
    y = rng.poisson(np.exp(0.5 + 0.3 * x))
    return pd.DataFrame({"y": y.astype(float), "x": x, "wave": wave})


def test_matches_direct_call(frame):
    adapter = StatsmodelsGLM("y ~ x", family="poisson")
    ours = adapter.fit(frame)
    direct = smf.glm("y ~ x", data=frame, family=sm.families.Poisson()).fit()
    assert ours.names == tuple(direct.params.index)
    np.testing.assert_array_equal(ours.estimates, direct.params.to_numpy())
    direct_cov = direct.cov_params().to_numpy()
    np.testing.assert_array_equal(
        ours.covariance, 0.5 * (direct_cov + direct_cov.T)
    )
    assert ours.dfcom == float(direct.df_resid)


def test_subset_and_dfcom_opt_out(frame):
    adapter = StatsmodelsGLM(
        "y ~ x",
        family="poisson",
        subset=lambda f: f["wave"] == 3,
        use_dfcom=False,
    )
    ours = adapter.fit(frame)
    sub = frame[frame["wave"] == 3]
    direct = smf.glm("y ~ x", data=sub, family=sm.families.Poisson()).fit()
    np.testing.assert_array_equal(ours.estimates, direct.params.to_numpy())
    assert ours.dfcom is None
    assert ours.metadata["n_obs"] == len(sub)


def test_unknown_family_rejected(frame):
    with pytest.raises(ValueError, match="family"):
        StatsmodelsGLM("y ~ x", family="tweedie").fit(frame)
