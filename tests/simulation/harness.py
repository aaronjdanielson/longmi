"""Reusable simulation harness for frequentist validation.

Each study defines ``one_replicate(rng) -> RepOutcome`` for a scalar target
parameter with known truth; the harness repeats it and summarizes bias,
RMSE, empirical SD, mean reported SE, coverage, interval width, fraction
of missing information, and the **numerical failure rate** (replicates
that raised). Replication counts default to modest smoke values (these
tests are marked ``slow``); scale a full archived run with::

    LONGMI_SIM_REPS=1000 pytest -m slow tests/simulation -q

Acceptance criteria are pre-specified in the assertion helpers, not chosen
after inspecting output. Three outcomes are distinguished:

- **validated** — `assert_unbiased`, `assert_nominal_coverage`,
  `assert_se_calibrated`, `assert_low_failure_rate` all pass under the
  declared assumptions;
- **expected failure** — `assert_materially_biased` /
  `assert_undercovers` pass under a deliberately violated assumption
  (bias here is informative, not a bug);
- **numerical failure** — replicates raised; tracked separately because
  optimizer failure is an engineering issue, not a statistical finding.

Gates are Monte-Carlo-aware: with S replicates the MC standard error of
coverage c-hat is sqrt(c(1-c)/S), so at S = 60 an observed 0.94 is noise
while 0.86 is evidence; the standardized-bias gate widens to 4 MC SEs of
the bias when S is too small to resolve the nominal 0.10 threshold.
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
class RepOutcome:
    """One replicate's result for the scalar target parameter."""

    estimate: float
    ci_low: float
    ci_high: float
    se: float = float("nan")
    fmi: float = float("nan")


@dataclass(frozen=True)
class StudySummary:
    truth: float
    n_reps: int
    n_failures: int
    bias: float
    bias_mc_se: float
    rmse: float
    sd_empirical: float
    mean_se: float
    coverage: float
    mean_ci_width: float
    mean_fmi: float

    @property
    def failure_rate(self) -> float:
        return self.n_failures / self.n_reps

    @property
    def se_ratio(self) -> float:
        """Mean reported SE over empirical SD; ~1 when SEs are calibrated."""
        return self.mean_se / self.sd_empirical

    @property
    def standardized_bias(self) -> float:
        return self.bias / self.sd_empirical

    def __str__(self) -> str:
        return (
            f"reps={self.n_reps} fail={self.n_failures} "
            f"truth={self.truth:+.4f} bias={self.bias:+.4f} "
            f"(mc se {self.bias_mc_se:.4f}, std {self.standardized_bias:+.3f}) "
            f"rmse={self.rmse:.4f} sd={self.sd_empirical:.4f} "
            f"se_ratio={self.se_ratio:.3f} coverage={self.coverage:.3f} "
            f"width={self.mean_ci_width:.4f} fmi={self.mean_fmi:.3f}"
        )

    # -- validated-scenario gates (pre-specified) --------------------------

    def assert_low_failure_rate(self, tolerance: float = 0.05) -> None:
        """Numerical failures are an engineering issue; smoke gate 5%,
        archived-run gate 1% (pass tolerance=0.01 at large S)."""
        assert self.failure_rate <= tolerance, (
            f"numerical failure rate {self.failure_rate:.3f} exceeds "
            f"{tolerance}"
        )

    def assert_unbiased(self, standardized: float = 0.10, z: float = 4.0) -> None:
        """|bias| below max(0.10 x empirical SD, z x MC SE of the bias)."""
        limit = max(standardized * self.sd_empirical, z * self.bias_mc_se)
        assert abs(self.bias) < limit, (
            f"bias {self.bias:+.4f} exceeds gate {limit:.4f} "
            f"(standardized {self.standardized_bias:+.3f})"
        )

    def assert_nominal_coverage(
        self, level: float = 0.95, z: float = 3.0, floor: float = 0.85
    ) -> None:
        # band from the *nominal* binomial SE: the observed-proportion SE
        # degenerates to zero at 0%/100% coverage. Mild overcoverage is
        # expected from t-reference MI intervals at moderate M.
        nominal_se = float(np.sqrt(level * (1 - level) / self.n_reps))
        band = max(0.03, z * nominal_se)
        assert self.coverage > floor, f"coverage {self.coverage:.3f} below floor"
        assert abs(self.coverage - level) < band, (
            f"coverage {self.coverage:.3f} outside {level} +/- {band:.3f}"
        )

    def assert_se_calibrated(self, low: float = 0.80, high: float = 1.30) -> None:
        """Mean reported SE within [low, high] x empirical SD (skipped when
        the study did not record SEs)."""
        if np.isnan(self.mean_se):
            return
        assert low < self.se_ratio < high, (
            f"SE ratio {self.se_ratio:.3f} outside [{low}, {high}]"
        )

    # -- expected-failure gates (pre-specified) -----------------------------

    def assert_materially_biased(
        self, standardized: float = 0.25, z: float = 4.0
    ) -> None:
        """The deliberately violated assumption must visibly bite: |bias|
        above z MC SEs *and* above `standardized` x empirical SD."""
        assert abs(self.bias) > z * self.bias_mc_se, (
            f"bias {self.bias:+.4f} not distinguishable from zero "
            f"(mc se {self.bias_mc_se:.4f}) — expected failure did not occur"
        )
        assert abs(self.standardized_bias) > standardized, (
            f"standardized bias {self.standardized_bias:+.3f} below "
            f"{standardized} — expected failure did not occur"
        )

    def assert_undercovers(self, max_coverage: float = 0.90) -> None:
        assert self.coverage < max_coverage, (
            f"coverage {self.coverage:.3f} not below {max_coverage} — "
            "expected failure did not occur"
        )


def run_study(
    one_replicate: Callable[[np.random.Generator], RepOutcome],
    truth: float,
    reps: int,
    seed: int,
) -> StudySummary:
    root = np.random.SeedSequence(seed)
    estimates, covered, widths, ses, fmis = [], [], [], [], []
    failures = 0
    for child in root.spawn(reps):
        try:
            out = one_replicate(np.random.default_rng(child))
        except Exception:
            failures += 1
            continue
        estimates.append(out.estimate)
        covered.append(out.ci_low < truth < out.ci_high)
        widths.append(out.ci_high - out.ci_low)
        ses.append(out.se)
        fmis.append(out.fmi)
    if len(estimates) < max(10, reps // 2):
        raise AssertionError(
            f"only {len(estimates)}/{reps} replicates succeeded; "
            "study is numerically broken"
        )
    est = np.asarray(estimates)
    errors = est - truth
    return StudySummary(
        truth=truth,
        n_reps=reps,
        n_failures=failures,
        bias=float(errors.mean()),
        bias_mc_se=float(errors.std(ddof=1) / np.sqrt(len(errors))),
        rmse=float(np.sqrt(np.mean(errors**2))),
        sd_empirical=float(est.std(ddof=1)),
        mean_se=float(np.nanmean(ses)) if ses else float("nan"),
        coverage=float(np.mean(covered)),
        mean_ci_width=float(np.mean(widths)),
        mean_fmi=float(np.nanmean(fmis)) if fmis else float("nan"),
    )
