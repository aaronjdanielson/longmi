# longmi 0.2.0a1

Binary longitudinal outcomes.

* `outcome_type="binary"`: strict {0,1} validation, boolean
  normalization, binary completion guarantees.
* `BernoulliImputer` / `BernoulliFit`: logistic random-intercept
  imputation on the shared GLMM machinery (Gauss-Hermite ML with
  verified convergence, curvature validation, adaptive-grid intercept
  draws). Outcomes are Bernoulli draws, never rounded probabilities.
  Separation safeguards: all-0/all-1, rank-deficiency, and curvature
  refusals; quasi-separation warnings.
* Targeted delta rules: `DeltaAdjustment(where=..., times=...)` and
  `DeltaScenario` (overlaps rejected; realized per-row deltas recorded).
  Wired in the Bernoulli backend; logit scale only — exp(delta)
  multiplies conditional odds, never probabilities.
* Shared `_glmm` machinery extracted from the NB backend
  (refactor-invariance verified by golden test).
* Congeniality stated honestly: the imputer is conditional; marginal
  logistic GEE is a different estimand. The MI-to-GEE workflow is
  validated by simulation against the marginal GEE target (MCAR/MAR
  validated; omitted-interaction and MNAR-under-MAR fail as expected;
  delta curve monotone) and agrees with R mice(logreg) within 0.10
  pooled SEs on a shared benchmark (validation/binary/).

Not supported: binomial counts (successes of n), incomplete predictors,
random slopes, outcome-scale delta for binary data, automatic separation
resolution, or any claim of universal congeniality with marginal GEE.

# longmi 0.1.0a3

Correctness and hardening release; no new statistical methods.

* **Fixed: silent Gaussian imputation misalignment with ordered
  categorical participant IDs** whose category order differs from label
  order. The wide representation now retains the validated long-frame
  participant order exactly, with a mechanical wide/long mask invariant
  and a regression test. (0.1.0a2 users: use numeric or ordinary string
  IDs with the joint Gaussian imputer.)
* Predictors must now be genuinely numeric and finite at data
  construction (booleans become 0/1; categoricals must be
  indicator-encoded); empty frames are rejected.
* Covariance validation: negative diagonals always rejected; positive
  semidefiniteness judged relative to the matrix's own scale (tiny
  absolute matrices no longer pass on the unit-scale tolerance).
* StatsmodelsGEE sorts by cluster (and time when given) internally and
  requires an explicit time column for autoregressive correlation.
* Gaussian backend: design rank check, fit-time initialization
  validation, eigenvalue-based conditional-covariance draws with a
  materially-indefinite guard.
* NB backend: observed-data existence and rank checks; bounded rejection
  sampling for numerically invalid parameter draws (counts reported in
  run metadata, never silent clipping).

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
