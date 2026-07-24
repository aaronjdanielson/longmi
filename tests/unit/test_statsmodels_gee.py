"""Tests for the StatsmodelsGEE adapter.

The adapter must be a transparent wrapper: fitting through longmi must
reproduce a direct statsmodels call exactly — same terms, same order,
same coefficients, same robust covariance. The fixture deliberately
mirrors the awkward features of the motivating registry analysis:
case-sensitive string participant IDs, unbalanced panels dominated by
baseline-only participants, a categorical cohort with a non-default
reference level, and an exposure-by-time interaction.
"""

import numpy as np
import pandas as pd
import pytest

statsmodels = pytest.importorskip("statsmodels")
import statsmodels.api as sm  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402

from longmi import CompletedDatasetCollection, LongitudinalData, pool_rubin  # noqa: E402
from longmi.analysis import StatsmodelsGEE  # noqa: E402
from longmi.impute import NegativeBinomialImputer  # noqa: E402

FORMULA = (
    "y ~ exposed * time_months + C(cohort, Treatment(reference='V2'))"
)


def registry_frame(n=400, seed=41):
    """Unbalanced count panel with case-sensitive string IDs."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        # 'r12' and 'R12' style IDs must stay distinct participants
        pid = ("r" if i % 2 else "R") + str(i // 2)
        exposed = int(rng.uniform() < 0.7)
        cohort = "V1" if rng.uniform() < 0.4 else "V2"
        b = 0.4 * rng.standard_normal()
        n_waves = rng.choice([1, 1, 2, 4])  # mostly baseline-only
        for t in (0.0, 3.0, 6.0, 12.0)[: int(n_waves)]:
            eta = 1.2 + 0.02 * t + 0.25 * exposed - 0.01 * exposed * t + b
            y = rng.poisson(rng.gamma(2.0, np.exp(eta) / 2.0))
            rows.append((pid, t, float(y), exposed, cohort))
    return pd.DataFrame(
        rows, columns=["pid", "time_months", "y", "exposed", "cohort"]
    )


@pytest.fixture(scope="module")
def frame():
    return registry_frame()


class TestExactStatsmodelsAgreement:
    def test_matches_direct_call_poisson_exchangeable(self, frame):
        adapter = StatsmodelsGEE(
            FORMULA,
            groups="pid",
            family="poisson",
            cov_struct="exchangeable",
        )
        ours = adapter.fit(frame)

        # the adapter sorts by cluster (stable); compare on the same order
        sorted_frame = frame.sort_values("pid", kind="mergesort").reset_index(
            drop=True
        )
        direct = smf.gee(
            FORMULA,
            groups="pid",
            data=sorted_frame,
            family=sm.families.Poisson(),
            cov_struct=sm.cov_struct.Exchangeable(),
        ).fit(cov_type="robust")

        assert ours.names == tuple(direct.params.index)
        np.testing.assert_array_equal(ours.estimates, direct.params.to_numpy())
        # AnalysisEstimate symmetrizes 0.5 * (U + U.T); statsmodels' raw
        # sandwich is asymmetric only in the last ulps
        direct_cov = direct.cov_params().to_numpy()
        np.testing.assert_array_equal(
            ours.covariance, 0.5 * (direct_cov + direct_cov.T)
        )
        np.testing.assert_allclose(ours.covariance, direct_cov, rtol=1e-12)
        assert ours.dfcom is None
        assert ours.metadata["n_clusters"] == frame["pid"].nunique()

    def test_longitudinal_data_round_trip_does_not_change_the_fit(self, frame):
        """Wrapping in LongitudinalData (validation, sorting, float outcome)
        must not perturb the GEE in any way."""
        # categorical predictors must be indicator-encoded (0.1 contract)
        encoded = frame.assign(cohort_v1=(frame["cohort"] == "V1").astype(float))
        data = LongitudinalData(
            encoded,
            id_col="pid",
            time_col="time_months",
            outcome_col="y",
            predictor_cols=("exposed", "cohort_v1"),
            outcome_type="count",
        )
        adapter = StatsmodelsGEE(
            "y ~ exposed * time_months + cohort_v1",
            groups="pid", family="poisson", cov_struct="exchangeable"
        )
        via_longmi = adapter.fit(data.frame)
        sorted_direct = adapter.fit(
            encoded.sort_values(["pid", "time_months"]).reset_index(drop=True)
        )
        np.testing.assert_array_equal(
            via_longmi.estimates, sorted_direct.estimates
        )
        np.testing.assert_array_equal(
            via_longmi.covariance, sorted_direct.covariance
        )

    def test_case_sensitive_ids_are_distinct_clusters(self, frame):
        adapter = StatsmodelsGEE(FORMULA, groups="pid", family="poisson")
        result = adapter.fit(frame)
        upper_merged = frame.assign(pid=frame["pid"].str.upper())
        merged = adapter.fit(upper_merged)
        assert result.metadata["n_clusters"] > merged.metadata["n_clusters"]


class TestGuards:
    def test_cov_struct_instance_rejected(self):
        with pytest.raises(ValueError, match="not an instance"):
            StatsmodelsGEE(
                "y ~ x",
                groups="pid",
                cov_struct=sm.cov_struct.Exchangeable(),
            )._cov_struct(sm)

    def test_unknown_family_rejected(self, frame):
        adapter = StatsmodelsGEE(FORMULA, groups="pid", family="tweedie")
        with pytest.raises(ValueError, match="family"):
            adapter.fit(frame)

    def test_cov_struct_class_accepted_fresh_per_fit(self, frame):
        adapter = StatsmodelsGEE(
            FORMULA,
            groups="pid",
            family="poisson",
            cov_struct=sm.cov_struct.Exchangeable,
        )
        a = adapter.fit(frame)
        b = adapter.fit(frame)
        np.testing.assert_array_equal(a.estimates, b.estimates)


class TestEndToEndWithImputation:
    def test_nb_imputation_then_gee_then_rubin(self):
        """The full pipeline the package exists for: incomplete counts ->
        NB imputation -> marginal GEE per completed dataset -> Rubin."""
        rng = np.random.default_rng(53)
        n = 120
        rows = []
        for i in range(n):
            exposed = int(rng.uniform() < 0.5)
            b = 0.3 * rng.standard_normal()
            for t in (0.0, 3.0, 6.0):
                eta = 1.0 + 0.03 * t + 0.2 * exposed + b
                y = float(rng.poisson(rng.gamma(2.0, np.exp(eta) / 2.0)))
                rows.append((i, t, y, exposed))
        frame = pd.DataFrame(rows, columns=["pid", "time_months", "y", "exposed"])
        # MAR dropout on previous outcome
        wide = frame.pivot(index="pid", columns="time_months", values="y")
        dropped = pd.Series(False, index=wide.index)
        for j, t in enumerate((3.0, 6.0), start=1):
            prev = wide[wide.columns[j - 1]]
            keep = 1.0 / (1.0 + np.exp(-(2.5 - 0.3 * prev)))
            dropped |= rng.uniform(size=len(wide)) > keep
            frame.loc[
                frame["pid"].isin(wide.index[dropped])
                & (frame["time_months"] >= t),
                "y",
            ] = np.nan

        data = LongitudinalData(
            frame,
            id_col="pid",
            time_col="time_months",
            outcome_col="y",
            predictor_cols=("exposed",),
            outcome_type="count",
            times=(0.0, 3.0, 6.0),
        )
        fit = NegativeBinomialImputer(time_interactions=("exposed",)).fit(data)
        collection = fit.impute(5, random_state=61)
        adapter = StatsmodelsGEE(
            "y ~ exposed * time_months",
            groups="pid",
            family="poisson",
            cov_struct="exchangeable",
        )
        pooled = pool_rubin(
            collection.analyze(adapter), validity=collection.declaration
        )
        assert pooled.names[0] == "Intercept"
        assert "exposed:time_months" in pooled.names
        assert np.all(pooled.se > 0)
        assert "Missingness assumption: MAR [declared]" in pooled.validity_report()