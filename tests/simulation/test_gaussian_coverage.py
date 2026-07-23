"""Bias and coverage of the joint Gaussian imputer under MCAR and MAR.

Data are generated from the imputer's own model class (correct
specification, A5); the target is the wave-3 treatment effect estimated by
OLS with HC1 variance on each completed dataset and pooled with Rubin's
rules. Under A1-A8 the pooled estimator should be unbiased with
approximately nominal 95% coverage.

Run with: pytest -m slow tests/simulation -q  (LONGMI_SIM_REPS scales it)
"""

import numpy as np
import pandas as pd
import pytest

from longmi import AnalysisEstimate, CallableAnalysis, LongitudinalData, pool_rubin
from longmi.impute import JointGaussianImputer

from harness import n_reps, run_study

pytestmark = pytest.mark.slow

WAVES = (1, 2, 3)
TRUE_SIGMA = np.array(
    [[1.00, 0.50, 0.30], [0.50, 1.20, 0.60], [0.30, 0.60, 1.50]]
)
TRUE_B = np.array([[2.0, 2.5, 3.0], [0.0, 0.6, 1.2]])
TRUTH = TRUE_B[1, 2]  # wave-3 treatment effect
N = 200
M = 10


def simulate(rng):
    treat = (rng.uniform(size=N) < 0.5).astype(float)
    x = np.column_stack([np.ones(N), treat])
    y = x @ TRUE_B + rng.multivariate_normal(np.zeros(3), TRUE_SIGMA, size=N)
    frame = pd.DataFrame(
        {
            "pid": np.repeat(np.arange(N), 3),
            "wave": np.tile(WAVES, N),
            "y": y.ravel(),
            "treat": np.repeat(treat, 3),
        }
    )
    return frame


def impose_mcar(frame, rng):
    out = frame.copy()
    drop = (out["wave"] > 1) & (rng.uniform(size=len(out)) < 0.25)
    out.loc[drop, "y"] = np.nan
    return out


def impose_mar(frame, rng):
    out = frame.copy()
    wide = frame.pivot(index="pid", columns="wave", values="y")
    dropped = pd.Series(False, index=wide.index)
    for j in (2, 3):
        keep = 1.0 / (1.0 + np.exp(-(2.6 - 0.45 * wide[j - 1])))
        dropped |= rng.uniform(size=len(wide)) > keep
        out.loc[
            out["pid"].isin(wide.index[dropped]) & (out["wave"] >= j), "y"
        ] = np.nan
        wide.loc[dropped, j] = wide.loc[dropped, j - 1]
    return out


def wave3_treatment_effect(frame) -> AnalysisEstimate:
    sub = frame[frame["wave"] == 3]
    x = np.column_stack([np.ones(len(sub)), sub["treat"].to_numpy()])
    y = sub["y"].to_numpy()
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    resid = y - x @ beta
    scale = len(y) / (len(y) - 2)
    cov = scale * xtx_inv @ (x * (resid**2)[:, None]).T @ x @ xtx_inv
    return AnalysisEstimate(
        names=("intercept", "treat"),
        estimates=beta,
        covariance=cov,
        dfcom=float(len(y) - 2),
    )


def make_replicate(impose):
    imputer = JointGaussianImputer(burn_in=200, thin=20)

    def one_replicate(rng):
        data = LongitudinalData(
            impose(simulate(rng), rng),
            id_col="pid",
            time_col="wave",
            outcome_col="y",
            predictor_cols=("treat",),
            times=WAVES,
        )
        collection = imputer.fit(data).impute(M, random_state=rng)
        pooled = pool_rubin(
            collection.analyze(CallableAnalysis(wave3_treatment_effect))
        )
        j = pooled.names.index("treat")
        lo, hi = pooled.conf_int(0.95)[j]
        return float(pooled.qbar[j]), float(lo), float(hi)

    return one_replicate


class TestGaussianImputerFrequentistValidity:
    def test_mcar_bias_and_coverage(self):
        summary = run_study(
            make_replicate(impose_mcar), TRUTH, n_reps(100), seed=901
        )
        print(f"\ngaussian mcar: {summary}")
        summary.assert_unbiased()
        summary.assert_nominal_coverage()

    def test_mar_bias_and_coverage(self):
        summary = run_study(
            make_replicate(impose_mar), TRUTH, n_reps(100), seed=902
        )
        print(f"\ngaussian mar: {summary}")
        summary.assert_unbiased()
        summary.assert_nominal_coverage()
