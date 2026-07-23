"""Reproducible missingness mechanisms for the epil example.

The complete upstream data are the benchmark; missingness is imposed here
and shared across languages as a mask CSV (subject, period, observed) under
``validation/masks/`` — the mask contains no copied outcome data, so it can
live in the repository. Python and R both join the same mask to their own
upstream copy of the data, which sidesteps cross-language random-number
reproducibility entirely.

Mechanisms
----------
``mcar_light``
    Each post-baseline cell is missing independently with probability 0.10,
    regardless of any data (MCAR; non-monotone).

``mar_monotone``
    Sequential MAR dropout. Everyone is observed at period 1. Conditional
    on being retained through period j-1, the retention probability is

        logit(lambda_ij) = gamma0_j + gamma1 * log(1 + base_i)
                           + gamma2 * age_i + gamma3 * treat_i
                           + gamma4 * log(1 + y_{i,j-1})

    which depends only on fully observed baseline information, treatment,
    wave, and the previously *observed* outcome — sequential MAR. Dropout
    is monotone: once out, always out.

``mnar_stress_test``
    Same sequential form, but retention additionally depends on the
    current-period outcome y_ij — the value being hidden. Ordinary MAR
    imputation is not guaranteed to recover the truth here; the example
    uses it to demonstrate exactly that.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_data import load_epil

HERE = Path(__file__).resolve().parent
MASKS_DIR = HERE.parent.parent / "validation" / "masks"

SEED_TAG = "20260723"

# Calibrated to give roughly 7% / 16% / 26% cumulative dropout by
# periods 2 / 3 / 4 under mar_monotone (see README for realized rates).
MAR_GAMMA = {
    "gamma0": {2: 4.1, 3: 3.8, 4: 3.7},
    "gamma1": -0.20,  # higher baseline seizure count -> more dropout
    "gamma2": -0.01,  # older -> slightly more dropout
    "gamma3": -0.30,  # active arm -> more dropout
    "gamma4": -0.25,  # higher previous count -> more dropout
}
# The MNAR stress test replaces the previous-outcome term with the
# current, about-to-be-hidden outcome.
MNAR_GAMMA = {**MAR_GAMMA, "gamma4": 0.0, "gamma5": -0.45}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def mcar_light(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    mask = frame[["subject", "period"]].copy()
    missing = (frame["period"] > 1) & (rng.uniform(size=len(frame)) < 0.10)
    mask["observed"] = (~missing).astype(int)
    return mask


def _sequential_dropout(
    frame: pd.DataFrame,
    rng: np.random.Generator,
    gamma: dict,
    use_current_outcome: bool,
) -> pd.DataFrame:
    wide = frame.pivot(index="subject", columns="period", values="y")
    meta = frame.drop_duplicates("subject").set_index("subject")
    observed = pd.DataFrame(
        1, index=wide.index, columns=wide.columns, dtype=int
    )
    for j in (2, 3, 4):
        retained = observed[j - 1] == 1
        eta = (
            gamma["gamma0"][j]
            + gamma["gamma1"] * np.log1p(meta["base"])
            + gamma["gamma2"] * meta["age"]
            + gamma["gamma3"] * meta["treat"]
            + gamma["gamma4"] * np.log1p(wide[j - 1])
        )
        if use_current_outcome:
            eta = eta + gamma["gamma5"] * np.log1p(wide[j])
        keep = rng.uniform(size=len(wide)) < _sigmoid(eta.to_numpy())
        observed[j] = (retained & keep).astype(int)
    long = observed.reset_index().melt(
        id_vars="subject", var_name="period", value_name="observed"
    )
    return long.sort_values(["subject", "period"]).reset_index(drop=True)


MECHANISMS = {
    "mcar_light": lambda frame, rng: mcar_light(frame, rng),
    "mar_monotone": lambda frame, rng: _sequential_dropout(
        frame, rng, MAR_GAMMA, use_current_outcome=False
    ),
    "mnar_stress_test": lambda frame, rng: _sequential_dropout(
        frame, rng, MNAR_GAMMA, use_current_outcome=True
    ),
}

MASK_FILES = {
    "mcar_light": f"epil_mcar_light_seed_{SEED_TAG}.csv",
    "mar_monotone": f"epil_mar_seed_{SEED_TAG}.csv",
    "mnar_stress_test": f"epil_mnar_stress_seed_{SEED_TAG}.csv",
}


def mask_path(mechanism: str) -> Path:
    return MASKS_DIR / MASK_FILES[mechanism]


def generate(mechanism: str, seed: int) -> pd.DataFrame:
    frame = load_epil()
    rng = np.random.default_rng(seed)
    mask = MECHANISMS[mechanism](frame, rng)
    mask = mask.astype({"subject": int, "period": int, "observed": int})
    if (mask.loc[mask["period"] == 1, "observed"] != 1).any():
        raise AssertionError("period 1 must be fully observed")
    return mask


def apply_mask(frame: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """Return the observed-data frame: y set to NA where observed == 0."""
    merged = frame.merge(mask, on=["subject", "period"], validate="one_to_one")
    out = merged.copy()
    out["y"] = out["y"].astype(float).where(out["observed"] == 1)
    return out.drop(columns="observed")


def retention_summary(mask: pd.DataFrame) -> pd.Series:
    return mask.groupby("period")["observed"].mean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()
    MASKS_DIR.mkdir(parents=True, exist_ok=True)
    for mechanism in MECHANISMS:
        mask = generate(mechanism, args.seed)
        path = mask_path(mechanism)
        mask.to_csv(path, index=False)
        rates = retention_summary(mask)
        print(f"{mechanism}: wrote {path.name}")
        print(
            "  retention by period: "
            + ", ".join(f"{p}: {r:.3f}" for p, r in rates.items())
        )
