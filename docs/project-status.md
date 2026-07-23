# Project status

The canonical feature-maturity table. Other pages (README, examples,
algorithm docs) should defer to this file rather than restate status;
detailed exact-vs-approximate computation claims live only in the
algorithm pages.

Version: `0.1.0.dev0` (not on PyPI). Public API is **experimental** —
expected to stabilize after the simulation grid below is complete.

## Feature maturity

| Feature | Implementation | Unit tests | Numerical diagnostics | R parity | Simulation bias/coverage | API status |
| --- | --- | --- | --- | --- | --- | --- |
| `LongitudinalData` / completion invariants | complete | complete | n/a | n/a | n/a | experimental |
| `pool_rubin` (Rubin + Barnard–Rubin) | complete | complete | n/a | **complete** — bit-compatible with `mice::pool.scalar`, verified at 1e-12 incl. boundary cases | n/a | experimental |
| `JointGaussianImputer` | complete | complete | chain autocorrelation/ESS per run | pending | **initial study passing** (MCAR & MAR, wave-3 treatment effect; see below) | experimental |
| `NegativeBinomialImputer` | complete | complete | optimizer, curvature, quadrature, grid boundary mass | pending | **initial study passing** (MAR, GEE interaction target; see below) | experimental |
| `StatsmodelsGEE` adapter | complete | complete — exact agreement with direct statsmodels fits, incl. the motivating registry analysis (private data, external harness) | n/a | n/a | covered via NB study | experimental |
| `StatsmodelsGLM` adapter | complete | complete | n/a | n/a | pending | experimental |
| Delta-adjusted MNAR sensitivity | complete (scalar delta; group/wave-specific deltas planned) | complete | n/a | pending | pending | experimental |
| Bootstrap-then-impute pooling | planned (0.2) | — | — | — | — | — |
| Eligibility indicators (structural missingness) | planned | — | — | — | — | — |

## Simulation evidence to date

Seeded studies in `tests/simulation/` (`pytest -m slow`), correct-model
specification, 60 replications (scale with `LONGMI_SIM_REPS`):

| Study | Bias (MC SE) | 95% CI coverage | Notes |
| --- | --- | --- | --- |
| Gaussian imputer, MCAR 25%, wave-3 treatment effect | −0.006 (0.022) | 0.983 | mild overcoverage expected from t-reference at M = 10 |
| Gaussian imputer, sequential MAR, same target | −0.042 (0.028) | 0.967 | |
| NB imputer + Poisson GEE + Rubin, sequential MAR, treat×wave-3 | −0.045 (0.029) | 0.967 | truth calibrated on a large complete dataset (marginal estimand) |

Not yet demonstrated: misspecified-model failure modes, intermittent
missingness, wider grids over sample size / missingness fraction /
correlation / overdispersion, delta-shift response curves, R parity for
the imputation backends. **The backends should not be described as
statistically validated beyond this table.**

## Cross-language validation

- Rubin pooling: exact parity with `mice` (complete).
- epil example: Python (statsmodels) vs R (geepack) complete-data and
  available-case GEE agree within 2e-3 (cross-implementation tolerance);
  the MI arm currently runs on the Python side only.
- CATS tutorial parity: external checkout pinned and verified; the
  statistical comparison is in progress.
