"""The PR-2 refactor (shared GLMM machinery) must not change NB results.

Golden values were captured from the pre-refactor implementation with a
fixed seed; the refactored code must reproduce them to near machine
precision (identical RNG consumption order is part of the contract).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from longmi import LongitudinalData
from longmi.impute import NegativeBinomialImputer

GOLDEN = json.loads((Path(__file__).parent / "nb_refactor_golden.json").read_text())


def test_nb_results_unchanged_by_refactor():
    rng = np.random.default_rng(23); n = 80
    treat = (rng.uniform(size=n) < .5).astype(float)
    b = .5 * rng.standard_normal(n)
    rows = []
    for i in range(n):
        for w in (1, 2, 3):
            lin = 1 + .2*(w==2) + .4*(w==3) + .3*treat[i] + b[i]
            rows.append((i, w, float(rng.poisson(rng.gamma(2., np.exp(lin)/2.))), treat[i]))
    f = pd.DataFrame(rows, columns=["pid","wave","y","treat"])
    f.loc[(f["wave"]>1) & (rng.uniform(size=len(f))<.25), "y"] = np.nan
    d = LongitudinalData(f, id_col="pid", time_col="wave", outcome_col="y",
                         predictor_cols=("treat",), outcome_type="count",
                         times=(1,2,3))
    fit = NegativeBinomialImputer(time_interactions=("treat",)).fit(d)
    np.testing.assert_allclose(fit.theta_hat, GOLDEN["theta_hat"], rtol=1e-10)
    np.testing.assert_allclose(np.diag(fit.theta_cov), GOLDEN["cov_diag"], rtol=1e-8)
    cm = fit.impute(3, random_state=77)
    mask = d.missing_mask.to_numpy()
    np.testing.assert_array_equal(cm[0]["y"].to_numpy()[mask][:12], GOLDEN["draws_m1"])
    np.testing.assert_array_equal(cm[2]["y"].to_numpy()[mask][:12], GOLDEN["draws_m3"])
