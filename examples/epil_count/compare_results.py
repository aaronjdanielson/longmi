"""Compare the Python and R results for the epil first validation task.

Deterministic bookkeeping (row counts, subjects, term sets) must match
exactly. Coefficients and robust standard errors come from two independent
GEE implementations (statsmodels vs geepack) with different convergence
criteria and scale/working-correlation estimation details, so they are
compared at an absolute tolerance of 2e-3 — cross-implementation
statistical agreement, not the 1e-12 deterministic parity required of
Rubin pooling.

Run after run_python.py and run_reference.R:
    python compare_results.py
Exits nonzero on any disagreement.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
TOLERANCE = 2e-3


def load(name: str) -> pd.DataFrame:
    path = HERE / name
    if not path.exists():
        sys.exit(f"{path.name} not found; run run_python.py / run_reference.R first")
    return pd.read_csv(path)


def main() -> int:
    py = load("results_python.csv")
    r = load("results_r.csv")

    merged = py.merge(
        r, on=["analysis", "term"], suffixes=("_py", "_r"), validate="one_to_one"
    )
    if len(merged) != len(py) or len(merged) != len(r):
        sys.exit("term sets differ between Python and R results")

    failures = []
    for col in ("n_rows", "n_subjects"):
        bad = merged[merged[f"{col}_py"] != merged[f"{col}_r"]]
        if not bad.empty:
            failures.append(f"{col} differs:\n{bad}")

    merged["d_estimate"] = (merged["estimate_py"] - merged["estimate_r"]).abs()
    merged["d_robust_se"] = (merged["robust_se_py"] - merged["robust_se_r"]).abs()

    report = merged[
        ["analysis", "term", "estimate_py", "estimate_r", "d_estimate",
         "robust_se_py", "robust_se_r", "d_robust_se"]
    ]
    print(report.to_string(index=False, float_format=lambda v: f"{v:0.6f}"))
    print(
        f"\nmax |estimate diff| = {merged['d_estimate'].max():.2e}, "
        f"max |robust SE diff| = {merged['d_robust_se'].max():.2e}, "
        f"tolerance = {TOLERANCE:.0e}"
    )

    for col in ("d_estimate", "d_robust_se"):
        bad = merged[merged[col] > TOLERANCE]
        if not bad.empty:
            failures.append(
                f"{col} exceeds {TOLERANCE}:\n"
                f"{bad[['analysis', 'term', col]].to_string(index=False)}"
            )

    if failures:
        print("\nFAIL")
        for f in failures:
            print(f)
        return 1
    print("\nOK: Python and R agree within tolerance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
