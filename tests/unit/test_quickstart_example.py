"""Executable version of the documentation quickstart.

Mirrors docs/getting_started/quickstart.md step for step (with a small
synthetic dataset and small M so it stays fast). If this test breaks, the
quickstart is broken — update both together.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from longmi import DeltaAdjustment, LongitudinalData, pool_rubin  # noqa: E402
from longmi.analysis import StatsmodelsGEE  # noqa: E402
from longmi.impute import NegativeBinomialImputer  # noqa: E402


def seizures_long() -> pd.DataFrame:
    rng = np.random.default_rng(88)
    rows = []
    for i in range(100):
        treatment = int(rng.uniform() < 0.5)
        baseline = int(rng.poisson(8.0)) + 1
        age = int(rng.integers(20, 60))
        b = 0.3 * rng.standard_normal()
        drop = False
        for period in (1, 2, 3, 4):
            eta = 0.4 + 0.5 * np.log(baseline) - 0.05 * period * treatment + b
            y = float(rng.poisson(rng.gamma(2.0, np.exp(eta) / 2.0)))
            drop = drop or (period > 1 and rng.uniform() < 0.12)
            rows.append(
                (f"s{i}", period, np.nan if drop else y, treatment, baseline, age)
            )
    return pd.DataFrame(
        rows,
        columns=["subject", "period", "seizures", "treatment",
                 "baseline_seizures", "age"],
    )


def test_quickstart_runs_end_to_end():
    frame = seizures_long()

    data = LongitudinalData(
        frame,
        id_col="subject",
        time_col="period",
        outcome_col="seizures",
        predictor_cols=("treatment", "baseline_seizures", "age"),
        outcome_type="count",
        times=(1, 2, 3, 4),
    )
    assert data.n_missing > 0

    imputer = NegativeBinomialImputer(time_interactions=("treatment",))
    fit = imputer.fit(data)
    assert fit.diagnostics.gradient_norm < 1e-3 * max(
        1.0, abs(fit.diagnostics.final_objective)
    ) or fit.diagnostics.optimizer_success

    completed = fit.impute(m=5, random_state=20260723)

    analysis = StatsmodelsGEE(
        "seizures ~ treatment * period + baseline_seizures + age",
        groups="subject",
        family="poisson",
        cov_struct="exchangeable",
    )
    result = pool_rubin(
        completed.analyze(analysis), validity=completed.declaration
    )

    table = result.summary()
    assert "treatment:period" in table.index
    assert (table["se"] > 0).all()
    report = result.validity_report()
    assert "Missingness assumption: MAR [declared]" in report
    assert "Observed outcomes preserved: Yes [verified]" in report

    # MNAR sensitivity reusing the same fit and seed
    lower = fit.impute(
        m=5, random_state=20260723, delta=DeltaAdjustment(np.log(0.8))
    )
    result_lower = pool_rubin(lower.analyze(analysis), validity=lower.declaration)
    assert "MNAR(delta=" in result_lower.validity_report()