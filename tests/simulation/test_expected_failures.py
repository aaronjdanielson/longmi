"""Expected-failure demonstrations: longmi must NOT rescue an analysis
whose validity conditions are deliberately violated.

Each study breaks exactly one assumption and asserts that the pooled
result is *materially biased* (and, where applicable, undercovers). These
are statistical failures, not bugs; a package built on explicit validity
conditions should demonstrate both directions.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from longmi import (  # noqa: E402
    AnalysisEstimate,
    CallableAnalysis,
    LongitudinalData,
    pool_rubin,
)
from longmi.analysis import StatsmodelsGEE  # noqa: E402
from longmi.impute import JointGaussianImputer, NegativeBinomialImputer  # noqa: E402

from harness import RepOutcome, n_reps, run_study  # noqa: E402

pytestmark = pytest.mark.slow

WAVES = (1, 2, 3)


# ---------------------------------------------------------------------------
# Failure 1 (A8): the imputation model omits the exposure-by-time
# interaction that both the data-generating process and the analysis
# contain. The pooled interaction coefficient must be attenuated.
# ---------------------------------------------------------------------------

KAPPA, TAU = 2.0, 0.5
INTERACTION = 0.5
N = 150
M = 8
TARGET = "treat:C(wave)[T.3]"

ANALYSIS = StatsmodelsGEE(
    "y ~ treat * C(wave)",
    groups="pid",
    family="poisson",
    cov_struct="exchangeable",
)


def simulate_counts(rng, n):
    treat = (rng.uniform(size=n) < 0.5).astype(float)
    b = TAU * rng.standard_normal(n)
    rows = []
    for i in range(n):
        for w in WAVES:
            lin = (
                1.0 + 0.2 * (w == 2) + 0.4 * (w == 3)
                + 0.3 * treat[i] + INTERACTION * treat[i] * (w == 3) + b[i]
            )
            lam = rng.gamma(KAPPA, np.exp(lin) / KAPPA)
            rows.append((i, w, float(rng.poisson(lam)), treat[i]))
    return pd.DataFrame(rows, columns=["pid", "wave", "y", "treat"])


def impose_mar_counts(frame, rng):
    out = frame.copy()
    wide = frame.pivot(index="pid", columns="wave", values="y")
    dropped = pd.Series(False, index=wide.index)
    for j in (2, 3):
        keep = 1.0 / (1.0 + np.exp(-(1.8 - 0.25 * wide[j - 1])))
        dropped |= rng.uniform(size=len(wide)) > keep
        out.loc[
            out["pid"].isin(wide.index[dropped]) & (out["wave"] >= j), "y"
        ] = np.nan
        wide.loc[dropped, j] = wide.loc[dropped, j - 1]
    return out


def marginal_interaction_truth() -> float:
    frame = simulate_counts(np.random.default_rng(515151), 60_000)
    fit = ANALYSIS.fit(frame)
    return float(fit.estimates[fit.names.index(TARGET)])


def count_data(frame):
    return LongitudinalData(
        frame,
        id_col="pid",
        time_col="wave",
        outcome_col="y",
        predictor_cols=("treat",),
        outcome_type="count",
        times=WAVES,
    )


class TestOmittedInteractionIsNotRescued:
    def test_uncongenial_imputation_attenuates_the_interaction(self):
        def one_replicate(rng):
            data = count_data(impose_mar_counts(simulate_counts(rng, N), rng))
            # deliberately uncongenial: no treat-by-wave terms (A8 violated)
            fit = NegativeBinomialImputer(time_interactions=()).fit(data)
            pooled = pool_rubin(fit.impute(M, random_state=rng).analyze(ANALYSIS))
            j = pooled.names.index(TARGET)
            lo, hi = pooled.conf_int(0.95)[j]
            return RepOutcome(
                float(pooled.qbar[j]), float(lo), float(hi),
                se=float(pooled.se[j]), fmi=float(pooled.fmi[j]),
            )

        truth = marginal_interaction_truth()
        summary = run_study(one_replicate, truth, n_reps(40), seed=907)
        print(f"\nnegbin omitted-interaction (expected failure): {summary}")
        summary.assert_low_failure_rate(0.10)
        summary.assert_materially_biased()
        assert summary.bias < 0  # attenuation toward no interaction


# ---------------------------------------------------------------------------
# Failure 2 (A2): MNAR generation analyzed under MAR. Retention depends on
# the current, about-to-be-hidden count; MAR imputation conditioned on the
# observed history cannot recover the wave-3 mean.
# ---------------------------------------------------------------------------


def impose_mnar_counts(frame, rng):
    out = frame.copy()
    wide = frame.pivot(index="pid", columns="wave", values="y")
    dropped = pd.Series(False, index=wide.index)
    for j in (2, 3):
        keep = 1.0 / (1.0 + np.exp(-(2.2 - 0.35 * wide[j])))  # current value!
        dropped |= rng.uniform(size=len(wide)) > keep
        out.loc[
            out["pid"].isin(wide.index[dropped]) & (out["wave"] >= j), "y"
        ] = np.nan
    return out


def wave3_mean(frame) -> AnalysisEstimate:
    y = frame.loc[frame["wave"] == 3, "y"].to_numpy()
    return AnalysisEstimate(
        names=("mean_w3",),
        estimates=[y.mean()],
        covariance=[[y.var(ddof=1) / len(y)]],
        dfcom=float(len(y) - 1),
    )


class TestMnarIsNotRescuedByMarImputation:
    def test_wave3_mean_remains_biased(self):
        # analytic truth: E[Y_w3] = mean over arms of exp(lin + tau^2/2)
        lin0, lin1 = 1.0 + 0.4, 1.0 + 0.4 + 0.3 + INTERACTION
        truth = 0.5 * (np.exp(lin0 + TAU**2 / 2) + np.exp(lin1 + TAU**2 / 2))

        def one_replicate(rng):
            data = count_data(impose_mnar_counts(simulate_counts(rng, N), rng))
            fit = NegativeBinomialImputer(time_interactions=("treat",)).fit(data)
            pooled = pool_rubin(
                fit.impute(M, random_state=rng).analyze(
                    CallableAnalysis(wave3_mean)
                )
            )
            lo, hi = pooled.conf_int(0.95)[0]
            return RepOutcome(
                float(pooled.qbar[0]), float(lo), float(hi),
                se=float(pooled.se[0]), fmi=float(pooled.fmi[0]),
            )

        summary = run_study(one_replicate, truth, n_reps(40), seed=908)
        print(f"\nnegbin MNAR-under-MAR (expected failure): {summary}")
        summary.assert_low_failure_rate(0.10)
        summary.assert_materially_biased()
        summary.assert_undercovers()
        assert summary.bias < 0  # high counts were preferentially hidden


# ---------------------------------------------------------------------------
# Failure 3 (A5/A8): the Gaussian imputation model omits a covariate that
# drives both the outcome and the dropout. Conditioning on too little
# turns MAR into effective MNAR for the model actually used.
# ---------------------------------------------------------------------------


class TestOmittedAuxiliaryIsNotRescued:
    def test_gaussian_wave3_mean_biased_without_the_auxiliary(self):
        n = 200
        truth = 3.0  # E[Y_3] = 3.0 + 1.0 * E[x2] with E[x2] = 0

        def one_replicate(rng):
            x2 = rng.standard_normal(n)
            y = (
                np.column_stack([np.full(n, 2.0), np.full(n, 2.5), np.full(n, 3.0)])
                + 1.0 * x2[:, None]
                + rng.multivariate_normal(
                    np.zeros(3),
                    np.array([[1.0, 0.5, 0.3], [0.5, 1.2, 0.6], [0.3, 0.6, 1.5]]),
                    size=n,
                )
            )
            frame = pd.DataFrame(
                {
                    "pid": np.repeat(np.arange(n), 3),
                    "wave": np.tile(WAVES, n),
                    "y": y.ravel(),
                    "x2": np.repeat(x2, 3),
                }
            )
            # dropout depends on x2 (MAR given x2; MNAR without it)
            p_drop = 1.0 / (1.0 + np.exp(-(1.2 * x2 - 0.4)))
            gone = rng.uniform(size=n) < p_drop
            frame.loc[
                frame["pid"].isin(np.flatnonzero(gone)) & (frame["wave"] > 1),
                "y",
            ] = np.nan

            data = LongitudinalData(
                frame,
                id_col="pid",
                time_col="wave",
                outcome_col="y",
                predictor_cols=(),  # x2 deliberately omitted (A5/A8 violated)
                times=WAVES,
            )
            fit = JointGaussianImputer(burn_in=200, thin=20).fit(data)
            pooled = pool_rubin(
                fit.impute(8, random_state=rng).analyze(
                    CallableAnalysis(wave3_mean)
                )
            )
            lo, hi = pooled.conf_int(0.95)[0]
            return RepOutcome(
                float(pooled.qbar[0]), float(lo), float(hi),
                se=float(pooled.se[0]), fmi=float(pooled.fmi[0]),
            )

        summary = run_study(one_replicate, truth, n_reps(40), seed=909)
        print(f"\ngaussian omitted-auxiliary (expected failure): {summary}")
        summary.assert_low_failure_rate(0.10)
        summary.assert_materially_biased()
        assert summary.bias < 0  # high-x2 (high-outcome) dropouts imputed too low