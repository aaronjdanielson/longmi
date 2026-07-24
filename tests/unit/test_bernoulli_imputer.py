"""Tests for the Bernoulli random-intercept imputer (PR 3)."""

import numpy as np
import pandas as pd
import pytest

from longmi import DeltaAdjustment, DeltaScenario, LongitudinalData
from longmi.impute import BernoulliImputer

WAVES = (1, 2, 3)


def simulate(n=200, seed=71, prev=0.0):
    rng = np.random.default_rng(seed)
    treat = (rng.uniform(size=n) < 0.5).astype(float)
    b = 0.8 * rng.standard_normal(n)
    rows = []
    for i in range(n):
        for w in WAVES:
            logit = prev - 0.3 + 0.3 * (w == 3) + 0.8 * treat[i] + b[i]
            y = float(rng.uniform() < 1 / (1 + np.exp(-logit)))
            rows.append((i, w, y, treat[i]))
    return pd.DataFrame(rows, columns=["pid", "wave", "y", "treat"])


def impose_mar(frame, seed=73):
    rng = np.random.default_rng(seed)
    out = frame.copy()
    y1 = frame[frame["wave"] == 1].set_index("pid")["y"]
    p_miss = 0.15 + 0.2 * out["pid"].map(y1)
    holes = (out["wave"] > 1) & (rng.uniform(size=len(out)) < p_miss)
    out.loc[holes, "y"] = np.nan
    return out


def make_data(frame):
    return LongitudinalData(frame, id_col="pid", time_col="wave",
                            outcome_col="y", predictor_cols=("treat",),
                            outcome_type="binary", times=WAVES)


@pytest.fixture(scope="module")
def mar_data():
    return make_data(impose_mar(simulate()))


@pytest.fixture(scope="module")
def fit(mar_data):
    return BernoulliImputer(time_interactions=("treat",)).fit(mar_data)


class TestInvariants:
    def test_draws_are_binary_and_preserve_observed(self, mar_data, fit):
        cm = fit.impute(4, random_state=81)
        obs = ~mar_data.missing_mask
        for f in cm:
            y = f["y"].to_numpy()
            assert set(np.unique(y)) <= {0.0, 1.0}
            np.testing.assert_array_equal(
                f.loc[obs, "y"].to_numpy(), mar_data.frame.loc[obs, "y"].to_numpy())

    def test_reproducible_and_variable(self, fit, mar_data):
        a = fit.impute(3, random_state=5)
        b = fit.impute(3, random_state=5)
        mask = mar_data.missing_mask.to_numpy()
        for fa, fb in zip(a, b):
            np.testing.assert_array_equal(fa["y"].to_numpy(), fb["y"].to_numpy())
        draws = np.stack([f["y"].to_numpy()[mask] for f in a])
        assert (draws.std(axis=0) > 0).mean() > 0.1

    def test_declaration(self, fit):
        d = fit.declaration
        assert d.supported_outcome_types == ("binary",)
        assert d.analysis_nested_in_imputation_model is False
        assert "conditional" in d.congeniality_status


class TestSeparationSafeguards:
    def test_all_zero_and_all_one_refused(self):
        f = simulate(n=40)
        for value in (0.0, 1.0):
            f2 = f.copy(); f2["y"] = value
            d = make_data(f2)
            with pytest.raises(ValueError, match="not\\s+identified"):
                BernoulliImputer().fit(d)

    def test_rank_deficient_observed_design_refused(self):
        f = impose_mar(simulate(n=60))
        f["dup"] = f["treat"]
        d = LongitudinalData(f, id_col="pid", time_col="wave", outcome_col="y",
                             predictor_cols=("treat", "dup"),
                             outcome_type="binary", times=WAVES)
        with pytest.raises(ValueError, match="rank deficient"):
            BernoulliImputer().fit(d)

    def test_nonbinary_outcome_type_refused(self):
        f = simulate(n=40)
        d = LongitudinalData(f, id_col="pid", time_col="wave", outcome_col="y",
                             predictor_cols=("treat",),
                             outcome_type="continuous", times=WAVES)
        with pytest.raises(ValueError, match="binary outcomes only"):
            BernoulliImputer().fit(d)


class TestDeltaLogitScale:
    def test_outcome_scale_refused(self):
        with pytest.raises(ValueError, match="logit"):
            BernoulliImputer(delta=DeltaAdjustment(1.0, scale="outcome"))

    def test_delta_shifts_imputed_prevalence_directionally(self, fit, mar_data):
        base = fit.impute(6, random_state=7)
        up = fit.impute(6, random_state=7, delta=DeltaAdjustment(1.5))
        down = fit.impute(6, random_state=7, delta=DeltaAdjustment(-1.5))
        mask = mar_data.missing_mask.to_numpy()
        mean = lambda cm: np.mean([f["y"].to_numpy()[mask].mean() for f in cm])
        assert mean(down) < mean(base) < mean(up)

    def test_targeted_scenario_applies_only_where_specified(self, fit, mar_data):
        sc = DeltaScenario(adjustments=(
            DeltaAdjustment(3.0, where={"treat": 1.0}),
        ))
        cm = fit.impute(4, random_state=11, delta=sc)
        rd = cm.metadata["realized_deltas"]
        frame = mar_data.frame
        mask = mar_data.missing_mask.to_numpy()
        treated_missing = frame.loc[mask, "treat"].to_numpy() == 1.0
        assert (rd[treated_missing] == 3.0).all()
        assert (rd[~treated_missing] == 0.0).all()


class TestStatisticalRecovery:
    def test_pooled_wave3_prevalence_covers_complete_truth(self):
        from longmi import AnalysisEstimate, CallableAnalysis, pool_rubin

        complete = simulate(n=300, seed=91)
        data = make_data(impose_mar(complete, seed=93))
        fit = BernoulliImputer(time_interactions=("treat",)).fit(data)
        cm = fit.impute(10, random_state=95)

        def w3(f):
            y = f.loc[f["wave"] == 3, "y"].to_numpy()
            return AnalysisEstimate(names=("prev",), estimates=[y.mean()],
                                    covariance=[[y.var(ddof=1) / len(y)]],
                                    dfcom=float(len(y) - 1))
        pooled = pool_rubin(cm.analyze(CallableAnalysis(w3)),
                            validity=cm.declaration)
        truth = complete.loc[complete["wave"] == 3, "y"].mean()
        lo, hi = pooled.conf_int(0.95)[0]
        assert lo < truth < hi
