"""Bias and coverage of the NB GLMM imputer + GEE + Rubin under MAR.

Data are generated from the imputer's model class (NB random intercept,
log link, treatment-by-wave effect); the substantive analysis is the
marginal Poisson GEE (exchangeable, robust). The target is the pooled
GEE treat:wave3 coefficient, whose truth is obtained from the same GEE
fit to a single very large complete dataset (marginal and conditional
coefficients differ under a log link with random intercepts, so the GEE
estimand is calibrated on complete data rather than set to the
conditional parameter).

Run with: pytest -m slow tests/simulation -q  (LONGMI_SIM_REPS scales it)
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from longmi import LongitudinalData, pool_rubin  # noqa: E402
from longmi.analysis import StatsmodelsGEE  # noqa: E402
from longmi.impute import NegativeBinomialImputer  # noqa: E402

from harness import n_reps, run_study  # noqa: E402

pytestmark = pytest.mark.slow

WAVES = (1, 2, 3)
KAPPA, TAU = 2.0, 0.5
N = 150
M = 8
TARGET = "treat:C(wave)[T.3]"

ANALYSIS = StatsmodelsGEE(
    "y ~ treat * C(wave)",
    groups="pid",
    family="poisson",
    cov_struct="exchangeable",
)


def simulate(rng, n):
    treat = (rng.uniform(size=n) < 0.5).astype(float)
    b = TAU * rng.standard_normal(n)
    rows = []
    for i in range(n):
        for w in WAVES:
            lin = (
                1.0
                + 0.2 * (w == 2)
                + 0.4 * (w == 3)
                + 0.3 * treat[i]
                + 0.4 * treat[i] * (w == 3)
                + b[i]
            )
            lam = rng.gamma(KAPPA, np.exp(lin) / KAPPA)
            rows.append((i, w, float(rng.poisson(lam)), treat[i]))
    return pd.DataFrame(rows, columns=["pid", "wave", "y", "treat"])


def impose_mar(frame, rng):
    out = frame.copy()
    wide = frame.pivot(index="pid", columns="wave", values="y")
    dropped = pd.Series(False, index=wide.index)
    for j in (2, 3):
        keep = 1.0 / (1.0 + np.exp(-(2.3 - 0.25 * wide[j - 1])))
        dropped |= rng.uniform(size=len(wide)) > keep
        out.loc[
            out["pid"].isin(wide.index[dropped]) & (out["wave"] >= j), "y"
        ] = np.nan
        wide.loc[dropped, j] = wide.loc[dropped, j - 1]
    return out


def marginal_truth() -> float:
    """GEE estimand calibrated on one very large complete dataset."""
    frame = simulate(np.random.default_rng(424242), 60_000)
    fit = ANALYSIS.fit(frame)
    return float(fit.estimates[fit.names.index(TARGET)])


def one_replicate(rng):
    data = LongitudinalData(
        impose_mar(simulate(rng, N), rng),
        id_col="pid",
        time_col="wave",
        outcome_col="y",
        predictor_cols=("treat",),
        outcome_type="count",
        times=WAVES,
    )
    fit = NegativeBinomialImputer(time_interactions=("treat",)).fit(data)
    collection = fit.impute(M, random_state=rng)
    pooled = pool_rubin(collection.analyze(ANALYSIS))
    j = pooled.names.index(TARGET)
    lo, hi = pooled.conf_int(0.95)[j]
    return float(pooled.qbar[j]), float(lo), float(hi)


class TestNegBinImputerFrequentistValidity:
    def test_mar_bias_and_coverage(self):
        truth = marginal_truth()
        summary = run_study(one_replicate, truth, n_reps(60), seed=903)
        print(f"\nnegbin mar: {summary}")
        summary.assert_unbiased()
        summary.assert_nominal_coverage()
