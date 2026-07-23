"""Exact cross-language parity: longmi.pool_rubin vs R mice::pool.scalar.

The fixtures here are byte-identical to those in
validation/r/rubin_reference.R; that script (run with mice, rule
"rubin1987") produced validation/r/rubin_reference.csv, which is committed.
Pooling is deterministic, so agreement is required at relative tolerance
1e-12 for every reported quantity.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from longmi import AnalysisEstimate, pool_rubin

REFERENCE_CSV = (
    Path(__file__).resolve().parents[2] / "validation" / "r" / "rubin_reference.csv"
)

RTOL = 1e-12

# Fixtures — keep in exact sync with validation/r/rubin_reference.R.
QA = [1.5, 2.0, 2.5, 1.8, 2.2]
UA = [0.36, 0.44, 0.40, 0.38, 0.42]
QC = [0.10, 0.30, 0.20]
UC = [0.09, 0.11, 0.10]
QE = [0.5, 0.5005, 0.4995]  # lambda ~ 8e-6: exercises mice's 1e-4 clamp
UE = [0.04, 0.04, 0.04]
QF = [0.25, 0.25, 0.25, 0.25]  # zero between-imputation variance
UF = [0.01, 0.012, 0.011, 0.009]
QD = np.array(
    [
        [0.90, 0.42, -0.15],
        [1.10, 0.38, -0.09],
        [0.95, 0.45, -0.20],
        [1.05, 0.40, -0.12],
        [1.00, 0.35, -0.14],
    ]
)
UD_DIAG = np.array(
    [
        [0.040, 0.0110, 0.0050],
        [0.044, 0.0100, 0.0055],
        [0.042, 0.0115, 0.0048],
        [0.041, 0.0105, 0.0052],
        [0.043, 0.0095, 0.0051],
    ]
)
# Off-diagonal entries exercise the multivariate path; mice pools per
# parameter from the diagonal, so the per-parameter results must not depend
# on these.
UD_OFFDIAG = np.array([[0.0, 0.004, -0.002], [0.004, 0.0, 0.001], [-0.002, 0.001, 0.0]])


def scalar_case(qs, us, dfcom=None):
    return pool_rubin(
        [
            AnalysisEstimate(
                names=("beta",), estimates=[q], covariance=[[u]], dfcom=dfcom
            )
            for q, u in zip(qs, us)
        ]
    )


def vector_case(dfcom):
    names = ("beta0", "beta1", "beta2")
    ests = []
    for q_row, u_diag in zip(QD, UD_DIAG):
        cov = UD_OFFDIAG + np.diag(u_diag)
        ests.append(
            AnalysisEstimate(names=names, estimates=q_row, covariance=cov, dfcom=dfcom)
        )
    return pool_rubin(ests)


@pytest.fixture(scope="module")
def reference():
    if not REFERENCE_CSV.exists():
        pytest.skip(
            "R reference values not generated; run "
            "`Rscript validation/r/rubin_reference.R`"
        )
    table = pd.read_csv(REFERENCE_CSV)
    return {(row["case"], row["param"]): row for _, row in table.iterrows()}


@pytest.fixture(scope="module")
def pooled():
    return {
        ("A", "beta"): (scalar_case(QA, UA, dfcom=None), 0),
        ("B", "beta"): (scalar_case(QA, UA, dfcom=96.0), 0),
        ("C", "beta"): (scalar_case(QC, UC, dfcom=48.0), 0),
        ("D", "beta0"): (vector_case(194.0), 0),
        ("D", "beta1"): (vector_case(194.0), 1),
        ("D", "beta2"): (vector_case(194.0), 2),
        ("E", "beta"): (scalar_case(QE, UE, dfcom=48.0), 0),
        ("F", "beta"): (scalar_case(QF, UF, dfcom=None), 0),
    }


CASES = [("A", "beta"), ("B", "beta"), ("C", "beta"),
         ("D", "beta0"), ("D", "beta1"), ("D", "beta2"),
         ("E", "beta"), ("F", "beta")]
QUANTITIES = ["qbar", "ubar", "b", "t", "riv", "lambda", "fmi", "df"]


@pytest.mark.parametrize("case", CASES, ids=["-".join(c) for c in CASES])
@pytest.mark.parametrize("quantity", QUANTITIES)
def test_matches_mice(reference, pooled, case, quantity):
    row = reference[case]
    result, j = pooled[case]
    ours = {
        "qbar": result.qbar[j],
        "ubar": result.ubar[j, j],
        "b": result.b[j, j],
        "t": result.t[j, j],
        "riv": result.riv[j],
        "lambda": result.lambda_[j],
        "fmi": result.fmi[j],
        "df": result.df[j],
    }[quantity]
    assert ours == pytest.approx(float(row[quantity]), rel=RTOL)


def test_reference_metadata(reference):
    # dfcom bookkeeping: case A is the large-sample reference
    assert pd.isna(reference[("A", "beta")]["dfcom"])
    assert float(reference[("B", "beta")]["dfcom"]) == 96.0
    assert float(reference[("C", "beta")]["dfcom"]) == 48.0
    assert float(reference[("D", "beta0")]["dfcom"]) == 194.0
