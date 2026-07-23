# Negative-binomial GLMM imputer — algorithm as implemented

## At a glance

| Item | Negative-binomial imputer |
| --- | --- |
| Status | Implemented; see [project status](../project-status.md) for validation maturity |
| Outcome | Nonnegative longitudinal count |
| Missingness | MAR; delta-adjusted MNAR sensitivity (link scale) |
| Dependence | Participant random intercept |
| Likelihood integration | Gauss–Hermite quadrature (numerical) |
| Parameter draw | Large-sample normal approximation (declared) |
| Random-effect draw | Adaptive numerical grid approximation (controlled) |
| Outcome draw | Gamma–Poisson, conditional on sampled quantities |
| Main assumption | Correct conditional count model incl. the analysis's interactions (A5, A8) |
| Do not use when | Structural zero inflation; random slopes needed |

Implementation:
[src/longmi/impute/negbin.py](https://github.com/aaronjdanielson/longmi/blob/main/src/longmi/impute/negbin.py).
Theory: Proposition 2 of
[mathematical_foundations.md](../theory/mathematical_foundations.md).

## Model

For participant $i$ at wave $j$:

$$
Y_{ij} \mid b_i \sim \mathrm{NegBin}(\mu_{ij}, \kappa),
\qquad
\log \mu_{ij} = x_{ij}^\top \beta + b_i,
\qquad
b_i \sim N(0, \tau^2),
$$

with $\mathrm{Var}(Y \mid b) = \mu + \mu^2/\kappa$ and design
$x_{ij}$ = intercept + categorical wave effects + predictors + the
declared predictor-by-wave interactions (`time_interactions`; congeniality
requires the analysis's exposure-by-time interaction to be listed — A8).
Time-varying predictors are supported (the design is row-level).

## Fitting and draws

The backend separates fitting from generation:
`NegativeBinomialImputer.fit(data)` returns a `NegativeBinomialFit`
(diagnostics, declaration, model specification, data fingerprint), and
`fit.impute(m, random_state, delta=...)` generates completed datasets —
so MAR and delta scenarios reuse one fitted model.

1. **ML fit with verified convergence.** The marginal likelihood
   integrates $b_i$ by Gauss–Hermite quadrature (1-D, `n_quad` nodes);
   $\theta = (\beta, \log\kappa, \log\tau)$ maximized by BFGS. The fit is
   **refused** unless the optimizer reports success or the gradient
   inf-norm is small ($< 10^{-3} \max(1, |\ell|)$) — a failed optimizer
   never quietly generates imputations. Outcome, message, iterations,
   objective, and gradient norm are retained in
   `NegativeBinomialFitDiagnostics`.
2. **Tolerance-aware curvature validation.** The observed-information
   covariance $\widehat H^{-1}$ is validated by eigendecomposition: with
   $\varepsilon = 10^{-8}$, eigenvalues below $-\varepsilon\lambda_{\max}$
   raise (materially indefinite — possible nonidentification or a failed
   optimum); tiny negatives above that threshold are repaired to the
   tolerance and the repair is recorded (`covariance_repaired`,
   `covariance_min_eigenvalue`).
3. **Parameter draw (A6, declared approximation).** Per imputation,
   $\theta^{(m)} \sim N(\widehat\theta, \widehat H^{-1})$ — the
   large-sample normal approximation to the posterior, on a scale where
   $\kappa, \tau$ stay positive. This is an *approximation* to
   Proposition 2's exact posterior draw (Rubin 1987 sec. 4.3), and the
   backend's `ValidityDeclaration` says so.
4. **Random-intercept draw (controlled numerical approximation).**
   $b_i$ is sampled from a **numerically normalized grid approximation**
   to $p(b_i \mid y_i^{\mathrm{obs}}, \theta^{(m)})$ by inverse-CDF: the
   grid starts at $\pm 8\tau$ (≈25 points per prior sd) and **expands
   adaptively** until the probability mass in the outermost cells falls
   below $10^{-8}$; if the mass cannot be contained the run raises. The
   realized maximum boundary mass and expansion count are reported in the
   run metadata. Participants with no observed outcomes draw from the
   $N(0, \tau^2)$ prior.
5. **Outcome draw.** Gamma–Poisson:
   $\Lambda \sim \mathrm{Gamma}(\kappa, \text{rate} = \kappa/\mu)$,
   $Y \sim \mathrm{Poisson}(\Lambda)$ — nonnegative integers by
   construction, re-verified by `completed_with`.
6. **Delta adjustment** is supported on the linear-predictor scale only
   ($\mu$ multiplied by $e^\delta$ before the draw); an outcome-scale
   shift of a count is refused as a non-model-based transformation.

Wave order follows the declared design order (`times=`), which the backend
requires unless constructed with `allow_undeclared_times=True` — absent
rows cannot be imputed. Run metadata also records the seed / bit
generator / package version and the ranges of sampled $\kappa$ and
$\tau$; quadrature sensitivity is checked by refitting with a different
`n_quad` (recorded in the diagnostics).

## Validation

Unit tests
([tests/unit/test_negbin_imputer.py](https://github.com/aaronjdanielson/longmi/blob/main/tests/unit/test_negbin_imputer.py)):
count support and preservation invariants, seeded reproducibility,
between-imputation variability, guards, declaration carriage; fixed-seed
statistical checks — pooled recovery of the wave-3 mean under
outcome-dependent monotone dropout (with available-case biased low), and
imputed means scaling by $e^\delta$ under linear-predictor delta.
Simulation-grid coverage and comparison with R implementations arrive with
the simulation suite.
