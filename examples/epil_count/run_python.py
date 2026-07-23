"""Python side of the epil worked example.

Fits the substantive Poisson GEE

    log E(y_ij) = b0 + bT*treat + bP*period + bTP*treat:period
                  + bB*log(1 + base) + bA*age

with exchangeable working correlation clustered by subject, on

- the complete upstream data (the benchmark),
- the available cases under the shared MAR mask
  (validation/masks/epil_mar_seed_20260723.csv), and
- M = 20 completed datasets from longmi's negative-binomial GLMM imputer
  (categorical wave, treat-by-wave interactions so the analysis's
  treat:period term is nested — A8), pooled with Rubin's rules,

then runs a delta-adjusted MNAR sensitivity analysis (the same fitted
imputation model, means of the imputed counts shifted by exp(delta)).

Because the source data are complete, the complete-data GEE is the truth
the missing-data methods are trying to recover: available-case shows the
cost of ignoring the dropout mechanism; MI should land nearer the
benchmark with honestly wider uncertainty.

Writes machine-readable results to results_python.csv with
language-neutral term names, for comparison against the R reference
(run_reference.R -> results_r.csv, complete/available-case analyses) by
compare_results.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from missingness import apply_mask, mask_path
from prepare_data import load_epil, normalized_hash

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from longmi import LongitudinalData  # noqa: E402
from longmi.analysis import StatsmodelsGEE  # noqa: E402
from longmi.impute import NegativeBinomialImputer  # noqa: E402
from longmi.pooling import pool_rubin  # noqa: E402
from longmi.scenarios import DeltaAdjustment  # noqa: E402

HERE = Path(__file__).resolve().parent

FORMULA = "y ~ treat * period + lbase1 + age"
M_IMPUTATIONS = 20
MI_SEED = 20260723

# statsmodels term name -> language-neutral name (order fixed by the formula)
TERMS = {
    "Intercept": "intercept",
    "treat": "treat",
    "period": "period",
    "treat:period": "treat_period",
    "lbase1": "lbase1",
    "age": "age",
}


def fit_gee(frame: pd.DataFrame, analysis: str) -> pd.DataFrame:
    model = smf.gee(
        FORMULA,
        groups="subject",
        data=frame,
        family=sm.families.Poisson(),
        cov_struct=sm.cov_struct.Exchangeable(),
    )
    fit = model.fit()  # robust (sandwich) covariance is the default
    if list(fit.params.index) != list(TERMS):
        raise RuntimeError(f"unexpected term ordering: {list(fit.params.index)}")
    return pd.DataFrame(
        {
            "analysis": analysis,
            "term": list(TERMS.values()),
            "estimate": fit.params.to_numpy(),
            "robust_se": fit.bse.to_numpy(),
            "n_rows": len(frame),
            "n_subjects": frame["subject"].nunique(),
        }
    )


def fit_mi(observed: pd.DataFrame, analysis: str, delta: float | None) -> pd.DataFrame:
    """NB imputation -> GEE per completed dataset -> Rubin pooling."""
    data = LongitudinalData(
        observed,
        id_col="subject",
        time_col="period",
        outcome_col="y",
        predictor_cols=("treat", "lbase1", "age"),
        outcome_type="count",
        times=(1, 2, 3, 4),
    )
    imputer = NegativeBinomialImputer(time_interactions=("treat",))
    fit = fit_mi.cache.get("fit")
    if fit is None:
        fit = imputer.fit(data)
        fit_mi.cache["fit"] = fit
        diag = fit.diagnostics
        print(
            f"NB imputation model: optimizer ok "
            f"(grad {diag.gradient_norm:.2e}), "
            f"min Hessian eigenvalue {min(diag.hessian_eigenvalues):.3g}"
        )
    scenario = None if delta is None else DeltaAdjustment(
        delta=delta, label=f"means x {np.exp(delta):.2f}"
    )
    collection = fit.impute(M_IMPUTATIONS, random_state=MI_SEED, delta=scenario)
    adapter = StatsmodelsGEE(
        FORMULA, groups="subject", family="poisson", cov_struct="exchangeable"
    )
    pooled = pool_rubin(collection.analyze(adapter), validity=collection.declaration)
    name_map = dict(zip(pooled.names, TERMS.values()))
    if set(name_map) != set(TERMS):
        raise RuntimeError(f"unexpected MI term ordering: {pooled.names}")
    return pd.DataFrame(
        {
            "analysis": analysis,
            "term": [TERMS[n] for n in pooled.names],
            "estimate": pooled.qbar,
            "robust_se": pooled.se,
            "n_rows": len(observed),
            "n_subjects": observed["subject"].nunique(),
        }
    )


fit_mi.cache = {}


def main() -> None:
    epil = load_epil()
    print(f"data: {len(epil)} rows, sha256 {normalized_hash(epil)}")

    mask = pd.read_csv(mask_path("mar_monotone"))
    observed = apply_mask(epil, mask)
    available = observed.dropna(subset=["y"]).reset_index(drop=True)
    print(
        f"MAR mask: {int(mask['observed'].sum())} observed cells, "
        f"{len(available)} available-case rows"
    )

    results = pd.concat(
        [
            fit_gee(epil, "complete"),
            fit_gee(available, "available_case"),
            fit_mi(observed, "mi_rubin", delta=None),
            # MNAR sensitivity: imputed means shifted down/up 20%
            fit_mi(observed, "mi_delta_low", delta=float(np.log(0.8))),
            fit_mi(observed, "mi_delta_high", delta=float(np.log(1.25))),
        ],
        ignore_index=True,
    )
    out = HERE / "results_python.csv"
    results.to_csv(out, index=False)
    print(f"wrote {out}")
    print(results.to_string(index=False))

    # the headline contrast: treat:period across analyses
    headline = results[results["term"] == "treat_period"][
        ["analysis", "estimate", "robust_se"]
    ]
    print("\ntreat:period (truth = complete-data row):")
    print(headline.to_string(index=False))


if __name__ == "__main__":
    main()
