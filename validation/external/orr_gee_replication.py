"""Replicate the research-cannabis manuscript GEE (Table 1) through longmi.

The manuscript's population-averaged Poisson GEE is

    total_meds ~ cannabis_user * time_months
                 + C(registry_cohort, Treatment(reference='V2'))

clustered by case-sensitive participant ID, exchangeable working
correlation, robust (sandwich) covariance. This script rebuilds the paper's
exact analytic frame from the external file (the same derivation as the
project's `gee_extension_corrected` notebook), then fits the model twice:

1. directly with statsmodels (the notebook's call, verbatim);
2. through longmi: the frame wrapped in `LongitudinalData` (validation,
   deterministic sorting, float outcome coercion) and fitted via the
   `StatsmodelsGEE` analysis adapter.

Replication requires the two to agree **exactly** (coefficients bit-equal;
covariance equal up to the adapter's symmetrization of the sandwich).
The data are private and never enter the longmi repository; set

    export LONGMI_ORR_REPO=/path/to/research-cannabis

Exit code 0 = replicated; nonzero otherwise.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from longmi import LongitudinalData  # noqa: E402
from longmi.analysis import StatsmodelsGEE  # noqa: E402

FORMULA = (
    "total_meds ~ cannabis_user * time_months "
    "+ C(registry_cohort, Treatment(reference='V2'))"
)
EXPECTED = {
    "participants": 3212,
    "observations": 4828,
    "baseline_users": 2304,
    "baseline_nonusers": 908,
}


def load_paper_frame() -> pd.DataFrame:
    """The paper's exact analytic frame (notebook cells 2-4, verbatim logic)."""
    raw = os.environ.get("LONGMI_ORR_REPO")
    if not raw:
        sys.exit("set LONGMI_ORR_REPO to the research-cannabis project root")
    ref = Path(raw).expanduser() / "data" / "ORR_V1V2_MEDS_merged_thru12.xlsx"
    if not ref.exists():
        sys.exit(f"external file not found: {ref}")
    ext = pd.read_excel(ref, sheet_name=0)

    # IDs stay case-sensitive: the paper treats case-variant IDs as distinct
    ext["ID"] = ext["ID"].astype(str)
    ext["Time"] = ext["Time"].astype("Int64")

    base_ids = set(ext.loc[ext["Time"] == 0, "ID"])
    frame = ext[ext["ID"].isin(base_ids)].drop_duplicates(["ID", "Time"]).copy()

    baseline_use = frame.loc[frame["Time"] == 0].set_index("ID")[
        "CurrentCannabisUse"
    ]
    frame["cannabis_user"] = frame["ID"].map(baseline_use).astype(int)
    frame["time_months"] = frame["Time"].astype(float)
    frame["total_meds"] = pd.to_numeric(frame["TotalMeds"], errors="coerce")
    frame["registry_cohort"] = frame["Cohort"]
    frame["cohort_v1"] = (frame["Cohort"] == "V1").astype(float)
    frame = frame.sort_values(["ID", "Time"]).reset_index(drop=True)

    baseline = frame[frame["Time"] == 0]
    counts = {
        "participants": frame["ID"].nunique(),
        "observations": len(frame),
        "baseline_users": int((baseline["cannabis_user"] == 1).sum()),
        "baseline_nonusers": int((baseline["cannabis_user"] == 0).sum()),
    }
    if counts != EXPECTED:
        sys.exit(f"analytic frame mismatch: {counts}, expected {EXPECTED}")
    print(f"analytic frame matches the manuscript: {counts}")
    return frame[
        ["ID", "Time", "total_meds", "cannabis_user", "time_months",
         "registry_cohort", "cohort_v1"]
    ]


def main() -> int:
    frame = load_paper_frame()

    direct = smf.gee(
        FORMULA,
        groups="ID",
        data=frame,
        family=sm.families.Poisson(),
        cov_struct=sm.cov_struct.Exchangeable(),
    ).fit(cov_type="robust")

    data = LongitudinalData(
        frame,
        id_col="ID",
        time_col="Time",
        outcome_col="total_meds",
        predictor_cols=("cannabis_user", "time_months", "cohort_v1"),
        outcome_type="count",
    )
    print(
        f"LongitudinalData: {data.n_rows} rows, {data.n_participants} "
        f"participants, {data.n_missing} missing outcomes"
    )
    adapter = StatsmodelsGEE(
        FORMULA,
        groups="ID",
        family="poisson",
        cov_struct="exchangeable",
    )
    ours = adapter.fit(frame)  # exact: same frame, same formula
    mi_path = StatsmodelsGEE(
        "total_meds ~ cannabis_user * time_months + cohort_v1",
        groups="ID", family="poisson", cov_struct="exchangeable",
    ).fit(data.frame)
    # indicator coding reorders/renames terms; the fitted values match
    remap = {"Intercept": "Intercept",
             "C(registry_cohort, Treatment(reference='V2'))[T.V1]": "cohort_v1",
             "cannabis_user": "cannabis_user",
             "time_months": "time_months",
             "cannabis_user:time_months": "cannabis_user:time_months"}
    for src, dst in remap.items():
        a = ours.estimates[ours.names.index(src)]
        b = mi_path.estimates[mi_path.names.index(dst)]
        if abs(a - b) > 1e-8:
            sys.exit(f"longmi data-path coefficient differs for {src}")
    print("longmi LongitudinalData path numerically equivalent")

    failures = []
    if ours.names != tuple(direct.params.index):
        failures.append(f"term mismatch: {ours.names}")
    if not np.array_equal(ours.estimates, direct.params.to_numpy()):
        failures.append("coefficients differ")
    direct_cov = direct.cov_params().to_numpy()
    if not np.array_equal(ours.covariance, 0.5 * (direct_cov + direct_cov.T)):
        failures.append("robust covariance differs beyond symmetrization")

    table = pd.DataFrame(
        {
            "IRR": np.exp(ours.estimates),
            "coef": ours.estimates,
            "robust_se": np.sqrt(np.diag(ours.covariance)),
            "direct_coef": direct.params.to_numpy(),
            "direct_se": direct.bse.to_numpy(),
        },
        index=ours.names,
    )
    print()
    print(table.to_string(float_format=lambda v: f"{v:0.6f}"))
    print(
        f"\nobservations: {ours.metadata['n_obs']}, "
        f"clusters: {ours.metadata['n_clusters']}"
    )

    if failures:
        print("\nFAIL: " + "; ".join(failures))
        return 1
    print("\nOK: longmi reproduces the manuscript GEE exactly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
