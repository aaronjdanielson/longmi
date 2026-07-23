"""Tests for the fit-object API, numerical diagnostics, and provenance."""

import numpy as np
import pandas as pd
import pytest

from longmi import DeltaAdjustment, LongitudinalData
from longmi.diagnostics import (
    GaussianChainDiagnostics,
    NegativeBinomialFitDiagnostics,
)
from longmi.impute import (
    JointGaussianFit,
    JointGaussianImputer,
    NegativeBinomialFit,
    NegativeBinomialImputer,
)

WAVES = ("baseline", "month_3", "month_12")  # lexicographic order differs


def gaussian_frame(n=120, seed=3):
    rng = np.random.default_rng(seed)
    treat = (rng.uniform(size=n) < 0.5).astype(float)
    means = {"baseline": 1.0, "month_3": 2.0, "month_12": 4.0}
    rows = []
    for i in range(n):
        for w in WAVES:
            y = means[w] + 0.5 * treat[i] + rng.standard_normal()
            if w != "baseline" and rng.uniform() < 0.25:
                y = np.nan
            rows.append((i, w, y, treat[i]))
    return pd.DataFrame(rows, columns=["pid", "wave", "y", "treat"])


def gaussian_data():
    return LongitudinalData(
        gaussian_frame(),
        id_col="pid",
        time_col="wave",
        outcome_col="y",
        predictor_cols=("treat",),
        times=WAVES,
    )


def count_frame(n=120, seed=17):
    rng = np.random.default_rng(seed)
    treat = (rng.uniform(size=n) < 0.5).astype(float)
    rows = []
    for i in range(n):
        b = 0.4 * rng.standard_normal()
        for j, w in enumerate((1, 2, 3)):
            eta = 1.0 + 0.2 * j + 0.3 * treat[i] + b
            y = float(rng.poisson(rng.gamma(2.0, np.exp(eta) / 2.0)))
            if w > 1 and rng.uniform() < 0.25:
                y = np.nan
            rows.append((i, w, y, treat[i]))
    return pd.DataFrame(rows, columns=["pid", "wave", "y", "treat"])


def count_data():
    return LongitudinalData(
        count_frame(),
        id_col="pid",
        time_col="wave",
        outcome_col="y",
        predictor_cols=("treat",),
        outcome_type="count",
        times=(1, 2, 3),
    )


class TestDeclaredWaveOrder:
    def test_row_sorting_uses_declared_order(self):
        data = gaussian_data()
        first_rows = data.frame.groupby("pid").head(3)
        # every participant's rows appear in declared order, not
        # lexicographic ("month_12" < "month_3" alphabetically)
        per_pid = first_rows.groupby("pid")["wave"].apply(tuple)
        assert all(waves == WAVES for waves in per_pid)

    def test_gaussian_waves_follow_declared_order(self):
        fit = JointGaussianImputer(burn_in=10, thin=2).fit(gaussian_data())
        assert fit.model_specification["waves"] == list(WAVES)

    def test_negbin_reference_wave_is_declared_first(self):
        fit = NegativeBinomialImputer().fit(count_data())
        terms = fit.model_specification["design_terms"]
        assert terms[0] == "intercept"
        assert "wave[2]" in terms and "wave[3]" in terms
        assert "wave[1]" not in terms  # declared first wave is the reference


class TestFitReuse:
    def test_gaussian_fit_scenarios_share_randomness(self):
        fit = JointGaussianImputer(burn_in=20, thin=3).fit(gaussian_data())
        assert isinstance(fit, JointGaussianFit)
        mar = fit.impute(3, random_state=11)
        mnar = fit.impute(
            3, random_state=11, delta=DeltaAdjustment(delta=1.5, scale="outcome")
        )
        mask = mar.source.missing_mask.to_numpy()
        for fa, fb in zip(mar, mnar):
            np.testing.assert_allclose(
                fb["y"].to_numpy()[mask] - fa["y"].to_numpy()[mask],
                1.5,
                rtol=1e-12,
            )
        assert mar.declaration.missingness_assumption == "MAR"
        assert mnar.declaration.missingness_assumption.startswith("MNAR")

    def test_negbin_fit_reused_without_refitting(self):
        imputer = NegativeBinomialImputer(time_interactions=("treat",))
        fit = imputer.fit(count_data())
        assert isinstance(fit, NegativeBinomialFit)
        a = fit.impute(2, random_state=1)
        b = fit.impute(2, random_state=2)
        assert a.m == b.m == 2
        # same fingerprint and fit diagnostics, different draws
        assert a.metadata["data_fingerprint"] == b.metadata["data_fingerprint"]
        assert a.metadata["fit_diagnostics"] is b.metadata["fit_diagnostics"]

    def test_negbin_impute_time_delta_scale_checked(self):
        fit = NegativeBinomialImputer().fit(count_data())
        with pytest.raises(ValueError, match="linear-predictor scale only"):
            fit.impute(2, random_state=0, delta=DeltaAdjustment(1.0, scale="outcome"))


class TestSeedHandling:
    def test_int_seed_equals_generator_and_is_recorded(self):
        imputer = JointGaussianImputer(burn_in=10, thin=2)
        a = imputer.fit(gaussian_data()).impute(2, random_state=99)
        b = imputer.fit(gaussian_data()).impute(
            2, random_state=np.random.default_rng(99)
        )
        for fa, fb in zip(a, b):
            np.testing.assert_array_equal(fa["y"].to_numpy(), fb["y"].to_numpy())
        assert a.metadata["random_state"]["seed"] == 99
        assert b.metadata["random_state"]["seed"] is None
        assert a.metadata["random_state"]["bit_generator"] == "PCG64"
        assert "longmi_version" in a.metadata["random_state"]

    def test_invalid_random_state_rejected(self):
        fit = JointGaussianImputer(burn_in=5, thin=2).fit(gaussian_data())
        with pytest.raises(TypeError, match="random_state"):
            fit.impute(2, random_state="not-a-seed")


class TestDiagnostics:
    def test_gaussian_chain_diagnostics(self):
        fit = JointGaussianImputer(burn_in=100, thin=20).fit(gaussian_data())
        collection = fit.impute(4, random_state=5)
        diag = collection.metadata["diagnostics"]
        assert isinstance(diag, GaussianChainDiagnostics)
        assert diag.n_sweeps == 100 + 3 * 20  # burn_in + (m-1) * thin
        assert -1.0 <= diag.trace_lag1_autocorrelation <= 1.0
        assert diag.trace_ess > 0
        assert fit.diagnostics is diag

    def test_negbin_fit_diagnostics(self):
        fit = NegativeBinomialImputer(time_interactions=("treat",)).fit(count_data())
        diag = fit.diagnostics
        assert isinstance(diag, NegativeBinomialFitDiagnostics)
        assert diag.optimizer_success or diag.gradient_norm < 1e-3 * max(
            1.0, abs(diag.final_objective)
        )
        assert np.isfinite(diag.final_objective)
        assert min(diag.hessian_eigenvalues) > 0  # identified model
        assert diag.n_quad == 40

    def test_negbin_run_metadata_reports_grid_and_draw_ranges(self):
        fit = NegativeBinomialImputer().fit(count_data())
        collection = fit.impute(3, random_state=7)
        meta = collection.metadata
        assert meta["grid_max_boundary_mass"] <= 1e-8
        kappa_lo, kappa_hi = meta["kappa_draw_range"]
        tau_lo, tau_hi = meta["tau_draw_range"]
        assert 0 < kappa_lo <= kappa_hi
        assert 0 < tau_lo <= tau_hi


class TestPredictorFiniteness:
    def test_nonfinite_predictor_rejected(self):
        frame = gaussian_frame()
        frame.loc[0, "treat"] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            LongitudinalData(
                frame,
                id_col="pid",
                time_col="wave",
                outcome_col="y",
                predictor_cols=("treat",),
                times=WAVES,
            )
