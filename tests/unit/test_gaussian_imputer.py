"""Tests for the joint Gaussian reference imputer.

Mechanical invariants are exact; statistical checks use a fixed seed on
data simulated from the imputer's own model class (multivariate normal
with participant-level predictors) so the truth is known.
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
from longmi.impute import JointGaussianImputer

WAVES = (1, 2, 3)
TRUE_SIGMA = np.array(
    [
        [1.00, 0.50, 0.30],
        [0.50, 1.20, 0.60],
        [0.30, 0.60, 1.50],
    ]
)
# wave-specific intercept and treatment effect: columns are waves
TRUE_B = np.array(
    [
        [2.0, 2.5, 3.0],  # intercept per wave
        [0.0, 0.6, 1.2],  # treatment effect per wave (grows with time)
    ]
)


def simulate(n=240, seed=11):
    rng = np.random.default_rng(seed)
    treat = (rng.uniform(size=n) < 0.5).astype(float)
    x = np.column_stack([np.ones(n), treat])
    y = x @ TRUE_B + rng.multivariate_normal(np.zeros(3), TRUE_SIGMA, size=n)
    frame = pd.DataFrame(
        {
            "pid": np.repeat(np.arange(n), 3),
            "wave": np.tile(WAVES, n),
            "y": y.ravel(),
            "treat": np.repeat(treat, 3),
        }
    )
    return frame, y


def impose_mar(frame, seed=13):
    """Sequential MAR dropout: high previous outcome -> more dropout."""
    rng = np.random.default_rng(seed)
    out = frame.copy()
    wide = frame.pivot(index="pid", columns="wave", values="y")
    dropped = pd.Series(False, index=wide.index)
    for j in (2, 3):
        keep_prob = 1.0 / (1.0 + np.exp(-(2.6 - 0.45 * wide[j - 1])))
        dropped |= rng.uniform(size=len(wide)) > keep_prob
        gone = wide.index[dropped]
        out.loc[out["pid"].isin(gone) & (out["wave"] >= j), "y"] = np.nan
        wide.loc[dropped, j] = wide.loc[dropped, j - 1]  # carry for next step
    return out


def make_data(frame):
    return LongitudinalData(
        frame,
        id_col="pid",
        time_col="wave",
        outcome_col="y",
        predictor_cols=("treat",),
        outcome_type="continuous",
        times=WAVES,
    )


@pytest.fixture(scope="module")
def mar_data():
    frame, y_true = simulate()
    masked = impose_mar(frame)
    data = make_data(masked)
    assert 0.10 < data.n_missing / data.n_rows < 0.45
    return data, y_true


@pytest.fixture(scope="module")
def collection(mar_data):
    data, _ = mar_data
    imputer = JointGaussianImputer(burn_in=300, thin=30)
    return imputer.impute(data, m=10, random_state=np.random.default_rng(7))


class TestInvariants:
    def test_collection_certifies_completion(self, mar_data, collection):
        data, _ = mar_data
        assert collection.m == 10
        for frame in collection:
            assert frame["y"].notna().all()
            observed = ~data.missing_mask
            np.testing.assert_array_equal(
                frame.loc[observed, "y"].to_numpy(),
                data.frame.loc[observed, "y"].to_numpy(),
            )

    def test_reproducible_given_seed(self, mar_data):
        data, _ = mar_data
        imputer = JointGaussianImputer(burn_in=50, thin=5)
        a = imputer.impute(data, m=3, random_state=np.random.default_rng(42))
        b = imputer.impute(data, m=3, random_state=np.random.default_rng(42))
        for fa, fb in zip(a, b):
            np.testing.assert_array_equal(fa["y"].to_numpy(), fb["y"].to_numpy())

    def test_imputations_differ_across_m(self, mar_data, collection):
        data, _ = mar_data
        mask = data.missing_mask.to_numpy()
        draws = np.stack([f["y"].to_numpy()[mask] for f in collection])
        assert np.std(draws, axis=0).min() > 0

    def test_declaration(self):
        decl = JointGaussianImputer().declaration
        assert decl.parameter_uncertainty_propagated is True
        assert decl.outcome_uncertainty_propagated is True
        assert decl.supported_outcome_types == ("continuous",)
        assert decl.missingness_assumption == "MAR"


class TestGuards:
    def test_count_outcome_rejected(self):
        frame, _ = simulate(n=60)
        frame["y"] = np.round(np.abs(frame["y"]))
        data = LongitudinalData(
            frame,
            id_col="pid",
            time_col="wave",
            outcome_col="y",
            predictor_cols=("treat",),
            outcome_type="count",
        )
        with pytest.raises(ValueError, match="continuous outcomes only"):
            JointGaussianImputer().impute(data, 2, np.random.default_rng(0))

    def test_incomplete_row_grid_rejected_at_construction(self):
        frame, _ = simulate(n=60)
        frame = frame.drop(index=frame.index[1])  # participant 0 loses wave 2
        with pytest.raises(ValueError, match="grid is incomplete"):
            make_data(frame)

    def test_undeclared_times_require_explicit_opt_in(self):
        frame, _ = simulate(n=60)
        data = LongitudinalData(  # no times declared
            frame,
            id_col="pid",
            time_col="wave",
            outcome_col="y",
            predictor_cols=("treat",),
        )
        with pytest.raises(ValueError, match="declare the design grid"):
            JointGaussianImputer().impute(data, 2, np.random.default_rng(0))
        # explicit opt-in works
        JointGaussianImputer(
            burn_in=5, thin=2, allow_undeclared_times=True
        ).impute(data, 2, np.random.default_rng(0))

    def test_incomplete_row_grid_rejected_by_imputer_without_declared_times(self):
        frame, _ = simulate(n=60)
        frame = frame.drop(index=frame.index[1])
        data = LongitudinalData(  # no times declared: constructor cannot know
            frame,
            id_col="pid",
            time_col="wave",
            outcome_col="y",
            predictor_cols=("treat",),
        )
        imputer = JointGaussianImputer(allow_undeclared_times=True)
        with pytest.raises(ValueError, match="every design wave"):
            imputer.impute(data, 2, np.random.default_rng(0))

    def test_time_varying_predictor_rejected(self):
        frame, _ = simulate(n=60)
        frame = frame.copy()
        frame["treat"] = frame["treat"] + 0.01 * frame["wave"]
        data = make_data(frame)
        with pytest.raises(ValueError, match="varies within participant"):
            JointGaussianImputer().impute(data, 2, np.random.default_rng(0))

    def test_single_imputation_rejected(self, mar_data):
        data, _ = mar_data
        with pytest.raises(ValueError, match="m must be >= 2"):
            JointGaussianImputer().impute(data, 1, np.random.default_rng(0))


def wave3_treatment_effect(frame) -> AnalysisEstimate:
    """OLS of the wave-3 outcome on treatment, HC1 robust variance."""
    sub = frame[frame["wave"] == 3]
    x = np.column_stack([np.ones(len(sub)), sub["treat"].to_numpy()])
    y = sub["y"].to_numpy()
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    resid = y - x @ beta
    scale = len(y) / (len(y) - x.shape[1])
    meat = (x * (resid**2)[:, None]).T @ x
    cov = scale * xtx_inv @ meat @ xtx_inv
    return AnalysisEstimate(
        names=("intercept", "treat"),
        estimates=beta,
        covariance=cov,
        dfcom=float(len(y) - 2),
    )


class TestStatisticalRecovery:
    def test_mi_recovers_truth_under_mar(self, mar_data, collection):
        """Pooled wave-3 treatment effect close to truth; available-case is
        the biased comparator (dropout depends on the outcome history)."""
        _, _ = mar_data
        pooled = pool_rubin(
            collection.analyze(CallableAnalysis(wave3_treatment_effect)),
            validity=collection.declaration,
        )
        idx = pooled.names.index("treat")
        truth = TRUE_B[1, 2]
        assert pooled.qbar[idx] == pytest.approx(truth, abs=3 * pooled.se[idx])
        lo, hi = pooled.conf_int(0.95)[idx]
        assert lo < truth < hi

    def test_pooled_se_exceeds_single_imputation_se(self, collection):
        ests = collection.analyze(CallableAnalysis(wave3_treatment_effect))
        pooled = pool_rubin(ests)
        idx = pooled.names.index("treat")
        single_se = np.sqrt(ests[0].covariance[idx, idx])
        assert pooled.se[idx] > single_se
        assert pooled.fmi[idx] > 0.05  # real missing information present

    def test_delta_adjustment_shifts_imputations(self, mar_data):
        data, _ = mar_data
        rng_a, rng_b = np.random.default_rng(3), np.random.default_rng(3)
        base = JointGaussianImputer(burn_in=50, thin=5).impute(data, 4, rng_a)
        delta = DeltaAdjustment(delta=2.0, scale="outcome", label="shift +2")
        shifted = JointGaussianImputer(burn_in=50, thin=5, delta=delta).impute(
            data, 4, rng_b
        )
        mask = data.missing_mask.to_numpy()
        for fa, fb in zip(base, shifted):
            np.testing.assert_allclose(
                fb["y"].to_numpy()[mask] - fa["y"].to_numpy()[mask],
                2.0,
                rtol=1e-12,
            )


class TestCategoricalIdAlignment:
    def test_ordered_categorical_ids_align_wide_and_long(self):
        """Regression for the a2 alignment bug: ordered categorical IDs
        whose category order differs from label order must not misassign
        imputations."""
        import pandas as pd

        rng = np.random.default_rng(99)
        ids = [f"id{k}" for k in range(8)]
        cat_order = ["id5", "id1", "id7", "id0", "id3", "id6", "id2", "id4"]
        rows = []
        for k, pid in enumerate(ids):
            for w in (1, 2, 3):
                rows.append((pid, w, float(rng.normal()), float(k % 2)))
        frame = pd.DataFrame(rows, columns=["pid", "wave", "y", "treat"])
        frame["pid"] = pd.Categorical(frame["pid"], categories=cat_order,
                                      ordered=True)
        frame.loc[1, "y"] = np.nan   # id0 wave 2 in original order
        frame.loc[9, "y"] = np.nan   # id3 wave 1
        data = LongitudinalData(
            frame, id_col="pid", time_col="wave", outcome_col="y",
            predictor_cols=("treat",), times=(1, 2, 3),
        )
        imputer = JointGaussianImputer(burn_in=5, thin=2,
                                       allow_undeclared_times=True)
        y, x, waves = imputer._wide_arrays(data)
        np.testing.assert_array_equal(
            np.isnan(y).ravel(), data.missing_mask.to_numpy()
        )
        # end to end: fit + impute must run and preserve observed values
        collection = imputer.fit(data).impute(2, random_state=1)
        obs = ~data.missing_mask
        for f in collection:
            np.testing.assert_array_equal(
                f.loc[obs, "y"].to_numpy(), data.frame.loc[obs, "y"].to_numpy()
            )
