"""Bernoulli imputer + logistic GEE + Rubin: end-to-end validation.

The GEE target is the population solution beta* of the marginal
estimating equation — computed from one very large complete synthetic
population, never the conditional alpha (they differ under a logit link
with random intercepts). Validated scenarios must pass the standard
gates; deliberately violated assumptions must fail visibly.

Run with: pytest -m slow tests/simulation -q
"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from longmi import LongitudinalData, pool_rubin  # noqa: E402
from longmi.analysis import StatsmodelsGEE  # noqa: E402
from longmi.impute import BernoulliImputer  # noqa: E402

from harness import RepOutcome, n_reps, run_study  # noqa: E402

pytestmark = pytest.mark.slow

WAVES = (1, 2, 3)
TAU = 0.8
N, M = 250, 8
TARGET = "treat:C(wave)[T.3]"
ANALYSIS = StatsmodelsGEE("y ~ treat * C(wave)", groups="pid",
                          family="binomial", cov_struct="exchangeable")


def simulate(rng, n, interaction=0.7):
    treat = (rng.uniform(size=n) < 0.5).astype(float)
    b = TAU * rng.standard_normal(n)
    rows = []
    for i in range(n):
        for w in WAVES:
            logit = (-0.5 + 0.3 * (w == 2) + 0.4 * (w == 3) + 0.5 * treat[i]
                     + interaction * treat[i] * (w == 3) + b[i])
            rows.append((i, w, float(rng.uniform() < 1/(1+np.exp(-logit))),
                         treat[i]))
    return pd.DataFrame(rows, columns=["pid", "wave", "y", "treat"])


def impose_mar(frame, rng):
    out = frame.copy()
    y1 = frame[frame["wave"] == 1].set_index("pid")["y"]
    p_miss = 0.12 + 0.25 * out["pid"].map(y1)
    out.loc[(out["wave"] > 1) & (rng.uniform(size=len(out)) < p_miss), "y"] = np.nan
    return out


def impose_mcar(frame, rng):
    out = frame.copy()
    out.loc[(out["wave"] > 1) & (rng.uniform(size=len(out)) < 0.25), "y"] = np.nan
    return out


def impose_mnar(frame, rng):
    out = frame.copy()   # missingness depends on the current, hidden value
    p_miss = 0.05 + 0.45 * out["y"]
    out.loc[(out["wave"] > 1) & (rng.uniform(size=len(out)) < p_miss), "y"] = np.nan
    return out


def marginal_truth(interaction=0.7):
    frame = simulate(np.random.default_rng(626262), 60_000, interaction)
    e = ANALYSIS.fit(frame)
    return float(e.estimates[e.names.index(TARGET)])


def make_replicate(impose, interactions=("treat",), interaction=0.7):
    def one(rng):
        d = LongitudinalData(
            impose(simulate(rng, N, interaction), rng),
            id_col="pid", time_col="wave", outcome_col="y",
            predictor_cols=("treat",), outcome_type="binary", times=WAVES)
        fit = BernoulliImputer(time_interactions=interactions).fit(d)
        pooled = pool_rubin(fit.impute(M, random_state=rng).analyze(ANALYSIS))
        j = pooled.names.index(TARGET)
        lo, hi = pooled.conf_int(0.95)[j]
        return RepOutcome(float(pooled.qbar[j]), float(lo), float(hi),
                          se=float(pooled.se[j]), fmi=float(pooled.fmi[j]))
    return one


def check_validated(s):
    s.assert_low_failure_rate()
    s.assert_unbiased()
    s.assert_nominal_coverage()
    s.assert_se_calibrated()


class TestBernoulliMarginalGEE:
    def test_mcar(self):
        s = run_study(make_replicate(impose_mcar), marginal_truth(),
                      n_reps(40), seed=1001)
        print(f"\nbernoulli mcar: {s}")
        check_validated(s)

    def test_mar(self):
        s = run_study(make_replicate(impose_mar), marginal_truth(),
                      n_reps(40), seed=1002)
        print(f"\nbernoulli mar: {s}")
        check_validated(s)

    def test_omitted_interaction_expected_failure(self):
        # strong interaction + heavy MAR so the uncongenial imputation
        # model's attenuation is visible above Monte Carlo noise
        def impose_heavy(frame, rng):
            out = frame.copy()
            y1 = frame[frame["wave"] == 1].set_index("pid")["y"]
            p_miss = 0.30 + 0.35 * out["pid"].map(y1)
            out.loc[(out["wave"] > 1)
                    & (rng.uniform(size=len(out)) < p_miss), "y"] = np.nan
            return out

        s = run_study(make_replicate(impose_heavy, interactions=(),
                                     interaction=1.5),
                      marginal_truth(1.5), n_reps(40), seed=1003)
        print(f"\nbernoulli omitted interaction: {s}")
        s.assert_materially_biased()
        assert s.bias < 0

    def test_mnar_under_mar_expected_failure(self):
        # target: wave-3 marginal prevalence (analytic-free: large-pop truth)
        frame = simulate(np.random.default_rng(636363), 60_000)
        truth = frame.loc[frame["wave"] == 3, "y"].mean()
        from longmi import AnalysisEstimate, CallableAnalysis

        def w3(f):
            y = f.loc[f["wave"] == 3, "y"].to_numpy()
            return AnalysisEstimate(names=("prev",), estimates=[y.mean()],
                                    covariance=[[y.var(ddof=1)/len(y)]],
                                    dfcom=float(len(y)-1))

        def one(rng):
            d = LongitudinalData(
                impose_mnar(simulate(rng, N), rng),
                id_col="pid", time_col="wave", outcome_col="y",
                predictor_cols=("treat",), outcome_type="binary", times=WAVES)
            fit = BernoulliImputer(time_interactions=("treat",)).fit(d)
            pooled = pool_rubin(fit.impute(M, random_state=rng)
                                .analyze(CallableAnalysis(w3)))
            lo, hi = pooled.conf_int(0.95)[0]
            return RepOutcome(float(pooled.qbar[0]), float(lo), float(hi),
                              se=float(pooled.se[0]), fmi=float(pooled.fmi[0]))
        s = run_study(one, truth, n_reps(40), seed=1004)
        print(f"\nbernoulli MNAR-under-MAR: {s}")
        s.assert_materially_biased()
        s.assert_undercovers()
        assert s.bias < 0  # events preferentially hidden

    def test_delta_curve_monotone(self):
        from longmi import AnalysisEstimate, CallableAnalysis, DeltaAdjustment

        rng = np.random.default_rng(1005)
        d = LongitudinalData(
            impose_mar(simulate(rng, 300), rng),
            id_col="pid", time_col="wave", outcome_col="y",
            predictor_cols=("treat",), outcome_type="binary", times=WAVES)
        fit = BernoulliImputer(time_interactions=("treat",)).fit(d)

        def w3(f):
            y = f.loc[f["wave"] == 3, "y"].to_numpy()
            return AnalysisEstimate(names=("prev",), estimates=[y.mean()],
                                    covariance=[[y.var(ddof=1)/len(y)]],
                                    dfcom=float(len(y)-1))
        curve = []
        for dl in (-1.0, -0.5, 0.0, 0.5, 1.0):
            sc = None if dl == 0 else DeltaAdjustment(dl)
            cm = fit.impute(8, random_state=1006, delta=sc)
            curve.append(pool_rubin(cm.analyze(CallableAnalysis(w3))).qbar[0])
        print("\nbernoulli delta curve:", [f"{v:.3f}" for v in curve])
        assert all(a < b for a, b in zip(curve, curve[1:]))
