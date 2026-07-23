"""Delta-adjustment response curve: sensitivity output must move
coherently with delta.

On one fixed masked dataset, the same fitted NB imputation model is run
across a grid of link-scale deltas with a common seed (shared randomness,
so the curve is the delta effect, not Monte Carlo noise). The pooled
wave-3 mean must be strictly increasing in delta, bracket the MAR result,
and sit near the MAR result scaled by exp(delta) on the imputed share.
"""

import numpy as np
import pytest

pytest.importorskip("statsmodels")

from longmi import (  # noqa: E402
    AnalysisEstimate,
    CallableAnalysis,
    DeltaAdjustment,
    pool_rubin,
)
from longmi.impute import NegativeBinomialImputer  # noqa: E402

from test_expected_failures import (  # noqa: E402
    count_data,
    impose_mar_counts,
    simulate_counts,
)

pytestmark = pytest.mark.slow

DELTAS = [np.log(0.7), np.log(0.85), 0.0, np.log(1.15), np.log(1.4)]


def wave3_mean(frame) -> AnalysisEstimate:
    y = frame.loc[frame["wave"] == 3, "y"].to_numpy()
    return AnalysisEstimate(
        names=("mean_w3",),
        estimates=[y.mean()],
        covariance=[[y.var(ddof=1) / len(y)]],
        dfcom=float(len(y) - 1),
    )


class TestDeltaResponseCurve:
    def test_pooled_estimate_moves_monotonically_with_delta(self):
        rng = np.random.default_rng(910)
        data = count_data(impose_mar_counts(simulate_counts(rng, 200), rng))
        fit = NegativeBinomialImputer(time_interactions=("treat",)).fit(data)

        estimates = {}
        for delta in DELTAS:
            scenario = None if delta == 0.0 else DeltaAdjustment(delta)
            collection = fit.impute(10, random_state=911, delta=scenario)
            pooled = pool_rubin(
                collection.analyze(CallableAnalysis(wave3_mean)),
                validity=collection.declaration,
            )
            estimates[delta] = float(pooled.qbar[0])
            if delta != 0.0:
                assert collection.declaration.missingness_assumption.startswith(
                    "MNAR"
                )

        curve = [estimates[d] for d in DELTAS]
        print("\ndelta response (wave-3 mean):")
        for d, e in zip(DELTAS, curve):
            print(f"  delta={d:+.3f} (x{np.exp(d):.2f}) -> {e:.4f}")

        # strictly increasing in delta
        assert all(a < b for a, b in zip(curve, curve[1:])), curve
        # MAR sits strictly inside the bracketing scenarios
        assert curve[0] < estimates[0.0] < curve[-1]

        # magnitude sanity: shifts should scale with the imputed share of
        # wave-3 cells (delta only moves imputed values)
        mask = data.missing_mask.to_numpy()
        wave3 = (data.frame["wave"] == 3).to_numpy()
        imputed_share = mask[wave3].mean()
        assert imputed_share > 0.2
        spread = curve[-1] - curve[0]
        full_scale_spread = estimates[0.0] * (np.exp(DELTAS[-1]) - np.exp(DELTAS[0]))
        ratio = spread / (imputed_share * full_scale_spread)
        assert 0.5 < ratio < 2.0, (spread, imputed_share, full_scale_spread)