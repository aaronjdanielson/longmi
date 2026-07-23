"""Python side of the epil first validation task.

Fits the substantive Poisson GEE

    log E(y_ij) = b0 + bT*treat + bP*period + bTP*treat:period
                  + bB*log(1 + base) + bA*age

with exchangeable working correlation clustered by subject, on

- the complete upstream data (the benchmark), and
- the available cases under the shared MAR mask
  (validation/masks/epil_mar_seed_20260723.csv),

and writes machine-readable results to results_python.csv with
language-neutral term names, for comparison against the R reference
(run_reference.R -> results_r.csv) by compare_results.py.

No imputation happens here yet: this task establishes that both languages
load identical data, share one missingness mask, and agree on the
complete-data and available-case analyses — the end-to-end baseline the MI
engine will be validated against.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from missingness import apply_mask, mask_path
from prepare_data import load_epil, normalized_hash

HERE = Path(__file__).resolve().parent

FORMULA = "y ~ treat * period + lbase1 + age"

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
        [fit_gee(epil, "complete"), fit_gee(available, "available_case")],
        ignore_index=True,
    )
    out = HERE / "results_python.csv"
    results.to_csv(out, index=False)
    print(f"wrote {out}")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
