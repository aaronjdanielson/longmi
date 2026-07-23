"""Tests for the negative-binomial GLMM imputer.

Mechanical invariants are exact; statistical checks use a fixed seed on
data simulated from the imputer's own model class (NB random-intercept,
log link) so the truth is known.
"""

import numpy as np
import pandas as pd
import pytest

from longmi import (
    AnalysisEstimate,
    CallableAnalysis,
    DeltaAdjustment,
    LongitudinalData,
    pool_rubin,
)
from longmi.impute import NegativeBinomialImputer

WAVES = (1, 2, 3)
TRUE = {
    "intercept": 1.0,
    "wave2": 0.2,
    "wave3": 0.4,
    "treat": 0.3,
    "treat_wave3": 0.4,
    "kappa": 2.0,
    "tau": 0.5,
}


def simulate(n=150, seed=23):
    rng = np.random.default_rng(seed)
    treat = (rng.uniform(size=n) < 0.5).astype(float)
    b = TRUE["tau"] * rng.standard_normal(n)
    rows = []
    for i in range(n):
        for w in WAVES:
            eta = (
                TRUE["intercept"]
                + TRUE["wave2"] * (w == 2)
                + TRUE["wave3"] * (w == 3)
                + TRUE["treat"] * treat[i]
                + TRUE["treat_wave3"] * treat[i] * (w == 3)
                + b[i]
            )
            lam = rng.gamma(TRUE["kappa"], np.exp(eta) / TRUE["kappa"])
            rows.append((i, w, float(rng.poisson(lam)), treat[i]))
    return pd.DataFrame(rows, columns=["pid", "wave", "y", "treat"])


def impose_mar(frame, seed=29):
    """Monotone dropout depending on the previous observed count."""
    rng = np.random.default_rng(seed)
    out = frame.copy()
    wide = frame.pivot(index="pid", columns="wave", values="y")
    dropped = pd.Series(False, index=wide.index)
    for j in (2, 3):
        keep_prob = 1.0 / (1.0 + np.exp(-(2.3 - 0.25 * wide[j - 1])))
        dropped |= rng.uniform(size=len(wide)) > keep_prob
        gone = wide.index[dropped]
        out.loc[out["pid"].isin(gone) & (out["wave"] >= j), "y"] = np.nan
        wide.loc[dropped, j] = wide.loc[dropped, j - 1]
    return out


def make_data(frame):
    return LongitudinalData(
        frame,
        id_col="pid",
        time_col="wave",
        outcome_col="y",
        predictor_cols=("treat",),
        outcome_type="count",
        times=WAVES,
    )


@pytest.fixture(scope="module")
def complete_frame():
    return simulate()


@pytest.fixture(scope="module")
def mar_data(complete_frame):
    data = make_data(impose_mar(complete_frame))
    assert 0.10 < data.n_missing / data.n_rows < 0.45
    return data


@pytest.fixture(scope="module")
def collection(mar_data):
    imputer = NegativeBinomialImputer(time_interactions=("treat",))
    return imputer.impute(mar_data, m=8, random_state=np.random.default_rng(31))


class TestInvariants:
    def test_completion_and_count_support(self, mar_data, collection):
        assert collection.m == 8
        observed = ~mar_data.missing_mask
        for frame in collection:
            y = frame["y"].to_numpy()
            assert np.all(np.isfinite(y))
            assert np.all(y >= 0)
            np.testing.assert_array_equal(y, np.floor(y))  # integers
            np.testing.assert_array_equal(
                frame.loc[observed, "y"].to_numpy(),
                mar_data.frame.loc[observed, "y"].to_numpy(),
            )

    def test_reproducible_given_seed(self, mar_data):
        imputer = NegativeBinomialImputer(time_interactions=("treat",))
        a = imputer.impute(mar_data, m=3, random_state=np.random.default_rng(5))
        b = imputer.impute(mar_data, m=3, random_state=np.random.default_rng(5))
        for fa, fb in zip(a, b):
            np.testing.assert_array_equal(fa["y"].to_numpy(), fb["y"].to_numpy())

    def test_imputations_differ_across_m(self, mar_data, collection):
        mask = mar_data.missing_mask.to_numpy()
        draws = np.stack([f["y"].to_numpy()[mask] for f in collection])
        assert (np.ptp(draws, axis=0) > 0).mean() > 0.5

    def test_declaration_carried(self, collection):
        decl = collection.declaration
        assert decl.supported_outcome_types == ("count",)
        assert decl.parameter_uncertainty_propagated is True
        assert "wave" in collection.metadata["design_terms"][1]


class TestGuards:
    def test_continuous_outcome_rejected(self, complete_frame):
        frame = complete_frame.copy()
        frame["y"] = frame["y"] + 0.5
        data = LongitudinalData(
            frame,
            id_col="pid",
            time_col="wave",
            outcome_col="y",
            predictor_cols=("treat",),
            outcome_type="continuous",
        )
        with pytest.raises(ValueError, match="count outcomes only"):
            NegativeBinomialImputer().impute(data, 2, np.random.default_rng(0))

    def test_outcome_scale_delta_rejected(self):
        with pytest.raises(ValueError, match="linear-predictor scale only"):
            NegativeBinomialImputer(
                delta=DeltaAdjustment(delta=1.0, scale="outcome")
            )

    def test_unknown_interaction_column_rejected(self, mar_data):
        imputer = NegativeBinomialImputer(time_interactions=("nope",))
        with pytest.raises(ValueError, match="not predictor columns"):
            imputer.impute(mar_data, 2, np.random.default_rng(0))


def wave3_mean(frame) -> AnalysisEstimate:
    y = frame.loc[frame["wave"] == 3, "y"].to_numpy()
    return AnalysisEstimate(
        names=("mean_w3",),
        estimates=[y.mean()],
        covariance=[[y.var(ddof=1) / len(y)]],
        dfcom=float(len(y) - 1),
    )


class TestStatisticalRecovery:
    def test_mi_recovers_wave3_mean_under_mar(
        self, complete_frame, mar_data, collection
    ):
        pooled = pool_rubin(
            collection.analyze(CallableAnalysis(wave3_mean)),
            validity=collection.declaration,
        )
        complete_mean = complete_frame.loc[
            complete_frame["wave"] == 3, "y"
        ].mean()
        lo, hi = pooled.conf_int(0.95)[0]
        assert lo < complete_mean < hi
        # dropout follows high counts, so available-case means are too low
        available_mean = mar_data.frame.loc[
            mar_data.frame["wave"] == 3, "y"
        ].mean()
        assert available_mean < pooled.qbar[0]

    def test_linear_predictor_delta_scales_imputed_means(self, mar_data):
        rng_a, rng_b = np.random.default_rng(9), np.random.default_rng(9)
        base = NegativeBinomialImputer(time_interactions=("treat",)).impute(
            mar_data, 6, rng_a
        )
        delta = DeltaAdjustment(delta=np.log(2.0), label="doubled means")
        shifted = NegativeBinomialImputer(
            time_interactions=("treat",), delta=delta
        ).impute(mar_data, 6, rng_b)
        mask = mar_data.missing_mask.to_numpy()
        base_mean = np.mean([f["y"].to_numpy()[mask].mean() for f in base])
        shifted_mean = np.mean([f["y"].to_numpy()[mask].mean() for f in shifted])
        assert shifted_mean / base_mean == pytest.approx(2.0, rel=0.15)
        assert shifted.declaration.missingness_assumption.startswith("MNAR")