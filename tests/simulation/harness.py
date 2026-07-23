"""Reusable simulation harness for frequentist validation.

Each study defines ``one_replicate(rng) -> (estimate, ci_low, ci_high)``
for a scalar target parameter with known truth; the harness repeats it,
then summarizes bias, RMSE, empirical coverage, and interval width with
Monte Carlo standard errors. Replication counts default to modest values
(these tests are marked ``slow``); set ``LONGMI_SIM_REPS`` to scale a full
run, e.g.::

    LONGMI_SIM_REPS=500 pytest -m slow tests/simulation -q
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import numpy as np


def n_reps(default: int) -> int:
    value = os.environ.get("LONGMI_SIM_REPS")
    return int(value) if value else default


@dataclass(frozen=True)
class StudySummary:
    truth: float
    n_reps: int
    bias: float
    bias_mc_se: float
    rmse: float
    coverage: float
    coverage_mc_se: float
    mean_ci_width: float

    def __str__(self) -> str:
        return (
            f"reps={self.n_reps} truth={self.truth:+.4f} "
            f"bias={self.bias:+.4f} (mc se {self.bias_mc_se:.4f}) "
            f"rmse={self.rmse:.4f} coverage={self.coverage:.3f} "
            f"(mc se {self.coverage_mc_se:.3f}) "
            f"width={self.mean_ci_width:.4f}"
        )

    def assert_unbiased(self, z: float = 4.0) -> None:
        """Bias indistinguishable from zero at z Monte Carlo SEs."""
        assert abs(self.bias) < z * self.bias_mc_se, (
            f"bias {self.bias:+.4f} exceeds {z} x MC SE "
            f"{self.bias_mc_se:.4f}"
        )

    def assert_nominal_coverage(
        self, level: float = 0.95, z: float = 3.0, floor: float = 0.85
    ) -> None:
        # band from the *nominal* binomial SE: the observed-proportion SE
        # degenerates to zero at 0%/100% coverage. Mild overcoverage is
        # expected from t-reference MI intervals at moderate M and is
        # tolerated by the upper side of the band.
        nominal_se = float(np.sqrt(level * (1 - level) / self.n_reps))
        band = max(0.03, z * nominal_se)
        assert self.coverage > floor, f"coverage {self.coverage:.3f} below floor"
        assert abs(self.coverage - level) < band, (
            f"coverage {self.coverage:.3f} outside {level} +/- {band:.3f}"
        )


def run_study(
    one_replicate: Callable[[np.random.Generator], tuple[float, float, float]],
    truth: float,
    reps: int,
    seed: int,
) -> StudySummary:
    root = np.random.SeedSequence(seed)
    estimates = np.empty(reps)
    covered = np.empty(reps, dtype=bool)
    widths = np.empty(reps)
    for r, child in enumerate(root.spawn(reps)):
        estimate, lo, hi = one_replicate(np.random.default_rng(child))
        estimates[r] = estimate
        covered[r] = lo < truth < hi
        widths[r] = hi - lo
    errors = estimates - truth
    coverage = float(covered.mean())
    return StudySummary(
        truth=truth,
        n_reps=reps,
        bias=float(errors.mean()),
        bias_mc_se=float(errors.std(ddof=1) / np.sqrt(reps)),
        rmse=float(np.sqrt(np.mean(errors**2))),
        coverage=coverage,
        coverage_mc_se=float(np.sqrt(coverage * (1 - coverage) / reps)),
        mean_ci_width=float(widths.mean()),
    )
