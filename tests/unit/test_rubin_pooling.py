"""Unit tests for Rubin pooling against hand calculations.

Formulas: docs/algorithms/rubin_pooling.md (Rubin 1987; Barnard-Rubin 1999).
"""

import numpy as np
import pytest

from longmi import AnalysisEstimate, pool_rubin


def scalar_estimate(q, u, dfcom=None):
    return AnalysisEstimate(
        names=("beta",), estimates=[q], covariance=[[u]], dfcom=dfcom
    )


class TestScalarHandCalculation:
    """M=3, Q = 1, 2, 3, U = 1 each: every quantity is computable by hand."""

    @pytest.fixture()
    def result(self):
        return pool_rubin([scalar_estimate(q, 1.0) for q in (1.0, 2.0, 3.0)])

    def test_point_estimate(self, result):
        assert result.qbar[0] == pytest.approx(2.0)

    def test_within_variance(self, result):
        assert result.ubar[0, 0] == pytest.approx(1.0)

    def test_between_variance(self, result):
        # B = ((1-2)^2 + 0 + (3-2)^2) / (3-1) = 1
        assert result.b[0, 0] == pytest.approx(1.0)

    def test_total_variance(self, result):
        # T = 1 + (1 + 1/3) * 1 = 7/3
        assert result.t[0, 0] == pytest.approx(7.0 / 3.0)

    def test_riv_lambda(self, result):
        assert result.riv[0] == pytest.approx(4.0 / 3.0)
        assert result.lambda_[0] == pytest.approx((4.0 / 3.0) / (7.0 / 3.0))

    def test_rubin_df(self, result):
        lam = (4.0 / 3.0) / (7.0 / 3.0)
        assert result.df[0] == pytest.approx((3 - 1) / lam**2)

    def test_fmi(self, result):
        lam = (4.0 / 3.0) / (7.0 / 3.0)
        df = 2.0 / lam**2
        riv = 4.0 / 3.0
        assert result.fmi[0] == pytest.approx((riv + 2.0 / (df + 3.0)) / (1.0 + riv))

    def test_confidence_interval_uses_t_reference(self, result):
        from scipy import stats

        lo, hi = result.conf_int(0.95)[0]
        crit = stats.t.ppf(0.975, result.df[0])
        se = np.sqrt(7.0 / 3.0)
        assert lo == pytest.approx(2.0 - crit * se)
        assert hi == pytest.approx(2.0 + crit * se)


class TestBarnardRubinDf:
    def test_matches_formula(self):
        qs, us, dfcom = (0.1, 0.3, 0.2), (0.09, 0.11, 0.10), 48.0
        result = pool_rubin([scalar_estimate(q, u, dfcom=dfcom) for q, u in zip(qs, us)])
        m = 3
        qbar = np.mean(qs)
        b = np.sum((np.array(qs) - qbar) ** 2) / (m - 1)
        ubar = np.mean(us)
        t = ubar + (1 + 1 / m) * b
        lam = (1 + 1 / m) * b / t
        df_old = (m - 1) / lam**2
        df_obs = (dfcom + 1) / (dfcom + 3) * dfcom * (1 - lam)
        expected = df_old * df_obs / (df_old + df_obs)
        assert result.df[0] == pytest.approx(expected, rel=1e-12)
        # Barnard-Rubin df never exceeds either component
        assert result.df[0] < df_old
        assert result.df[0] < df_obs

    def test_inconsistent_dfcom_rejected(self):
        ests = [scalar_estimate(1.0, 1.0, dfcom=10.0), scalar_estimate(2.0, 1.0, dfcom=20.0)]
        with pytest.raises(ValueError, match="dfcom"):
            pool_rubin(ests)


class TestVectorPooling:
    @pytest.fixture()
    def result(self):
        names = ("intercept", "slope")
        qs = [np.array([1.0, 0.5]), np.array([1.2, 0.7]), np.array([0.8, 0.6])]
        us = [
            np.array([[0.10, 0.01], [0.01, 0.05]]),
            np.array([[0.12, 0.02], [0.02, 0.06]]),
            np.array([[0.11, 0.015], [0.015, 0.055]]),
        ]
        ests = [
            AnalysisEstimate(names=names, estimates=q, covariance=u)
            for q, u in zip(qs, us)
        ]
        return pool_rubin(ests), qs, us

    def test_full_covariance_hand_calculation(self, result):
        pooled, qs, us = result
        m = 3
        qbar = np.mean(qs, axis=0)
        dev = np.array(qs) - qbar
        b = dev.T @ dev / (m - 1)
        t = np.mean(us, axis=0) + (1 + 1 / m) * b
        np.testing.assert_allclose(pooled.qbar, qbar, rtol=1e-14)
        np.testing.assert_allclose(pooled.b, b, rtol=1e-14)
        np.testing.assert_allclose(pooled.t, t, rtol=1e-14)

    def test_t_is_symmetric_psd(self, result):
        pooled, _, _ = result
        np.testing.assert_allclose(pooled.t, pooled.t.T)
        assert np.all(np.linalg.eigvalsh(pooled.t) > 0)

    def test_summary_table_alignment(self, result):
        pooled, _, _ = result
        table = pooled.summary()
        assert list(table.index) == ["intercept", "slope"]
        np.testing.assert_allclose(table["se"].to_numpy(), np.sqrt(np.diag(pooled.t)))


class TestGuards:
    def test_single_imputation_rejected(self):
        with pytest.raises(ValueError, match="at least 2"):
            pool_rubin([scalar_estimate(1.0, 1.0)])

    def test_term_order_mismatch_rejected(self):
        a = AnalysisEstimate(
            names=("a", "b"), estimates=[1.0, 2.0], covariance=np.eye(2)
        )
        b = AnalysisEstimate(
            names=("b", "a"), estimates=[2.0, 1.0], covariance=np.eye(2)
        )
        with pytest.raises(ValueError, match="term ordering"):
            pool_rubin([a, b])

    def test_zero_between_variance(self):
        # matches current mice: lambda = 0 gives infinite df, fmi = 0
        result = pool_rubin([scalar_estimate(2.0, 1.5) for _ in range(4)])
        assert result.b[0, 0] == 0.0
        assert result.t[0, 0] == pytest.approx(1.5)
        assert result.riv[0] == 0.0
        assert result.lambda_[0] == 0.0
        assert result.fmi[0] == 0.0
        assert np.isinf(result.df[0])
        # normal reference when df is infinite
        lo, hi = result.conf_int(0.95)[0]
        assert hi - lo == pytest.approx(2 * 1.959963984540054 * np.sqrt(1.5), rel=1e-9)

    def test_zero_between_variance_with_finite_dfcom(self):
        # lambda = 0 boundary of Barnard-Rubin: df equals the observed-data df
        dfcom = 48.0
        result = pool_rubin(
            [scalar_estimate(2.0, 1.5, dfcom=dfcom) for _ in range(4)]
        )
        df_obs = (dfcom + 1) / (dfcom + 3) * dfcom
        assert result.df[0] == pytest.approx(df_obs, rel=1e-12)
        assert result.fmi[0] == pytest.approx(2.0 / (df_obs + 3.0), rel=1e-12)

    def test_pooling_is_invariant_to_imputation_order(self):
        qs = [np.array([1.0, 0.5]), np.array([1.2, 0.7]), np.array([0.8, 0.6])]
        us = [np.eye(2) * 0.1, np.eye(2) * 0.12, np.eye(2) * 0.11]
        ests = [
            AnalysisEstimate(names=("a", "b"), estimates=q, covariance=u)
            for q, u in zip(qs, us)
        ]
        forward = pool_rubin(ests)
        reverse = pool_rubin(ests[::-1])
        np.testing.assert_allclose(forward.qbar, reverse.qbar, rtol=1e-14)
        np.testing.assert_allclose(forward.t, reverse.t, rtol=1e-14)
        np.testing.assert_allclose(forward.df, reverse.df, rtol=1e-14)

    def test_indefinite_covariance_is_rejected(self):
        # symmetric, nonnegative diagonal, but eigenvalues (-1, 3)
        with pytest.raises(ValueError, match="positive semidefinite"):
            AnalysisEstimate(
                names=("a", "b"),
                estimates=[0.0, 0.0],
                covariance=[[1.0, 2.0], [2.0, 1.0]],
            )

    def test_large_sample_mode_ignores_dfcom(self):
        ests = [scalar_estimate(q, 1.0, dfcom=30.0) for q in (1.0, 2.0, 3.0)]
        result = pool_rubin(ests, df_method="large_sample")
        lam = result.lambda_[0]
        assert result.df[0] == pytest.approx((3 - 1) / lam**2)

    def test_unknown_df_method_rejected(self):
        ests = [scalar_estimate(q, 1.0) for q in (1.0, 2.0)]
        with pytest.raises(ValueError, match="df_method"):
            pool_rubin(ests, df_method="nope")

    def test_asymmetric_covariance_rejected(self):
        with pytest.raises(ValueError, match="symmetric"):
            AnalysisEstimate(
                names=("a", "b"),
                estimates=[0.0, 0.0],
                covariance=[[1.0, 0.5], [0.1, 1.0]],
            )

    def test_nonfinite_estimate_rejected(self):
        with pytest.raises(ValueError, match="non-finite"):
            AnalysisEstimate(names=("a",), estimates=[np.nan], covariance=[[1.0]])


class TestValidityReport:
    def test_report_renders_declared_and_undeclared_fields(self):
        result = pool_rubin(
            [scalar_estimate(q, 1.0) for q in (1.0, 2.0)],
            validity={
                "missingness_assumption": "MAR",
                "sampling_unit": "participant",
                "parameter_uncertainty_propagated": True,
                "observed_outcomes_preserved": True,
                "congeniality_status": "conditionally supported",
            },
        )
        report = result.validity_report()
        # provenance tags distinguish assertions from enforced properties
        assert "Missingness assumption: MAR [declared]" in report
        assert "MAR empirically testable: No [declared]" in report
        assert "Parameter uncertainty propagated: Yes [verified by backend]" in report
        assert "Observed outcomes preserved: Yes [verified]" in report
        assert "Pooling method: Rubin [verified]" in report
        # undeclared favorable claims must not default to Yes, and get no tag
        assert "Longitudinal dependence modeled: not declared" in report

    def test_accepts_declaration_object(self):
        from longmi import ValidityDeclaration

        decl = ValidityDeclaration(
            missingness_assumption="MCAR", pooling_method="Rubin"
        )
        result = pool_rubin(
            [scalar_estimate(q, 1.0) for q in (1.0, 2.0)], validity=decl
        )
        assert "Missingness assumption: MCAR [declared]" in result.validity_report()


class TestResultIntegrity:
    def test_pooled_result_validates_shapes(self):
        from longmi import RubinPooledResult

        good = pool_rubin([scalar_estimate(q, 1.0) for q in (1.0, 2.0, 3.0)])
        with pytest.raises(ValueError, match="m must be >= 2"):
            RubinPooledResult(
                names=good.names, m=1, qbar=good.qbar, ubar=good.ubar,
                b=good.b, t=good.t, riv=good.riv, lambda_=good.lambda_,
                fmi=good.fmi, df=good.df,
            )
        with pytest.raises(ValueError, match="shape"):
            RubinPooledResult(
                names=good.names, m=good.m, qbar=[1.0, 2.0], ubar=good.ubar,
                b=good.b, t=good.t, riv=good.riv, lambda_=good.lambda_,
                fmi=good.fmi, df=good.df,
            )
        with pytest.raises(ValueError, match="t must equal"):
            RubinPooledResult(
                names=good.names, m=good.m, qbar=good.qbar, ubar=good.ubar,
                b=good.b, t=good.t * 2.0, riv=good.riv, lambda_=good.lambda_,
                fmi=good.fmi, df=good.df,
            )

    def test_result_arrays_are_immutable(self):
        result = pool_rubin([scalar_estimate(q, 1.0) for q in (1.0, 2.0, 3.0)])
        with pytest.raises(ValueError, match="read-only"):
            result.qbar[0] = 999.0
        with pytest.raises(ValueError, match="read-only"):
            result.t[0, 0] = 999.0

    def test_estimate_arrays_are_immutable(self):
        est = scalar_estimate(1.0, 1.0)
        with pytest.raises(ValueError, match="read-only"):
            est.estimates[0] = 999.0
        with pytest.raises(ValueError, match="read-only"):
            est.covariance[0, 0] = 999.0
        with pytest.raises(TypeError):
            est.metadata["new"] = 1
