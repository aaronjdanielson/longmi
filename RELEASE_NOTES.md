# longmi 0.1.0a1 — public alpha

First public pre-release. Usable, documented, and initially validated; the
API may still change. See [docs/project-status.md](docs/project-status.md)
for the canonical maturity table and the archived release-scale simulation
evidence in `validation/releases/0.1.0a1/`.

## Implemented

One incomplete longitudinal response: validated data contracts
(`LongitudinalData`), two posterior-predictive imputers (joint Gaussian;
negative-binomial GLMM with verified-convergence fitting and numerical
diagnostics), reusable fit objects with scenario reuse, statsmodels
GEE/GLM analysis adapters, Rubin pooling bit-compatible with
`mice::pool.scalar`, scalar delta-adjusted MNAR sensitivity, and
provenance-tagged validity reports.

## Validation evidence

- Exact R parity for Rubin pooling (1e-12, boundary cases included).
- Release-scale seeded simulations (500 replicates/scenario): correct-model
  MCAR/monotone-MAR/intermittent-MAR unbiased with near-nominal coverage;
  high-missingness stress holds coverage with FMI ≈ 0.6.
- Expected-failure demonstrations: omitted exposure-by-time interaction
  (std. bias ≈ −1.5, masked by conservative Rubin SEs — nominal coverage
  alone is not evidence of a good estimator); MNAR analyzed under MAR
  (std. bias ≈ −7.4, coverage 0.02, on metrics conditional on the 98% of
  replicates that fit; 2% numerical-failure rate reported separately);
  omitted auxiliary (std. bias ≈ −1.1, coverage 0.80).
- End-to-end public example (`examples/epil_count`): complete-data
  benchmark, available-case, MI, IPW-GEE, single-realization MNAR stress
  test, delta-response curve, and a cross-method R MI comparison
  (statistical agreement, not backend parity).

## Limits and non-claims

- One incomplete longitudinal outcome; predictors must be fully observed.
- A complete participant-wave grid; every supplied row is treated as
  eligible for imputation — structural ineligibility (death, withdrawal,
  administrative censoring, competing events) must be handled before
  constructing `LongitudinalData`.
- Gaussian or negative-binomial outcome models; scalar delta sensitivity.
- Standard Rubin pooling, subject to the documented compatibility
  conditions (GEE caveats in `docs/theory/gee_after_imputation.md`).
- No claim of robustness to arbitrary MNAR mechanisms; no recovery of
  information that was never observed; MAR and congeniality are declared
  assumptions, never verified by the software.
- Pre-1.0: interfaces may change without deprecation.

## Deferred to 0.2

Bootstrap-then-impute pooling; group/wave-specific delta matrices;
eligibility indicators; multi-chain Gaussian diagnostics; backend-level R
parity; wider simulation grids.
