"""Load and normalize the MASS::epil seizure-count data from upstream.

The dataset is never redistributed inside longmi (MASS is GPL-licensed);
Python loads it from the Rdatasets mirror via statsmodels and R loads it
from the installed MASS package. Both sides normalize to the same schema
and record row/column counts plus a SHA-256 hash of the normalized data so
the harness can prove they analyzed identical inputs.

Normalized schema (sorted by subject, period):
    subject : int, 1..59
    period  : int, 1..4
    y       : int, seizure count in the two-week period
    treat   : int, 0 = placebo, 1 = progabide
    base    : int, baseline 8-week seizure count
    age     : int, years
    lbase1  : float, log(1 + base)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

COLUMNS = ["subject", "period", "y", "treat", "base", "age", "lbase1"]


def load_epil() -> pd.DataFrame:
    """Load MASS::epil from the Rdatasets mirror and normalize it."""
    import statsmodels.api as sm

    raw = sm.datasets.get_rdataset("epil", package="MASS", cache=True).data
    frame = pd.DataFrame(
        {
            "subject": raw["subject"].astype(int),
            "period": raw["period"].astype(int),
            "y": raw["y"].astype(int),
            "treat": (raw["trt"].astype(str) == "progabide").astype(int),
            "base": raw["base"].astype(int),
            "age": raw["age"].astype(int),
        }
    )
    frame["lbase1"] = np.log1p(frame["base"])
    frame = frame.sort_values(["subject", "period"]).reset_index(drop=True)
    if frame.shape[0] != 236 or frame["subject"].nunique() != 59:
        raise RuntimeError(
            f"unexpected epil shape: {frame.shape[0]} rows, "
            f"{frame['subject'].nunique()} subjects (expected 236 rows, 59 subjects)"
        )
    return frame[COLUMNS]


def normalized_hash(frame: pd.DataFrame) -> str:
    """SHA-256 of the integer core of the normalized data.

    Uses only the integer columns rendered as CSV text, so R can reproduce
    the identical byte stream without floating-point formatting concerns.
    """
    core = frame[["subject", "period", "y", "treat", "base", "age"]]
    text = core.to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def write_provenance(frame: pd.DataFrame, path: Path) -> dict:
    record = {
        "dataset": "epil",
        "upstream_package": "MASS",
        "upstream_license": "GPL-2 | GPL-3",
        "source": "Rdatasets mirror via statsmodels.datasets.get_rdataset",
        "citation": (
            "Thall PF, Vail SC (1990). Some covariance models for longitudinal "
            "count data with overdispersion. Biometrics 46, 657-671; distributed "
            "in Venables WN, Ripley BD, MASS."
        ),
        "erratum": (
            "MASS 7.3-65 corrected y in row 31 (subject 8, period 3) from 21 "
            "to 23; the R reference requires MASS >= 7.3-65 to match this "
            "mirror."
        ),
        "n_rows": int(frame.shape[0]),
        "n_subjects": int(frame["subject"].nunique()),
        "columns": list(frame.columns),
        "sha256_normalized_core": normalized_hash(frame),
    }
    path.write_text(json.dumps(record, indent=2) + "\n")
    return record


if __name__ == "__main__":
    epil = load_epil()
    record = write_provenance(epil, HERE / "epil_provenance.json")
    print(json.dumps(record, indent=2))
