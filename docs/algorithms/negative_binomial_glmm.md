# Negative-binomial GLMM imputer — algorithm as implemented

Implementation:
[src/longmi/impute/negbin.py](../../src/longmi/impute/negbin.py).
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

1. **ML fit.** The marginal likelihood integrates $b_i$ by Gauss–Hermite
   quadrature (1-D, `n_quad` nodes); $\theta = (\beta, \log\kappa,
   \log\tau)$ maximized by BFGS; observed-information covariance from a
   central-difference Hessian.
2. **Parameter draw (A6, declared approximation).** Per imputation,
   $\theta^{(m)} \sim N(\widehat\theta, \widehat H^{-1})$ — the
   large-sample normal approximation to the posterior, on a scale where
   $\kappa, \tau$ stay positive. This is an *approximation* to
   Proposition 2's exact posterior draw (Rubin 1987 sec. 4.3), and the
   backend's `ValidityDeclaration` says so.
3. **Random-intercept draw (exact given $\theta^{(m)}$).**
   $b_i \sim p(b_i \mid y_i^{\mathrm{obs}}, \theta^{(m)})$ by numerical
   inverse-CDF on a fine grid ($\pm 8\tau$, 401 points); participants with
   no observed outcomes draw from the $N(0, \tau^2)$ prior.
4. **Outcome draw.** Gamma–Poisson:
   $\Lambda \sim \mathrm{Gamma}(\kappa, \text{rate} = \kappa/\mu)$,
   $Y \sim \mathrm{Poisson}(\Lambda)$ — nonnegative integers by
   construction, re-verified by `completed_with`.
5. **Delta adjustment** is supported on the linear-predictor scale only
   ($\mu$ multiplied by $e^\delta$ before the draw); an outcome-scale
   shift of a count is refused as a non-model-based transformation.

## Validation

Unit tests
([tests/unit/test_negbin_imputer.py](../../tests/unit/test_negbin_imputer.py)):
count support and preservation invariants, seeded reproducibility,
between-imputation variability, guards, declaration carriage; fixed-seed
statistical checks — pooled recovery of the wave-3 mean under
outcome-dependent monotone dropout (with available-case biased low), and
imputed means scaling by $e^\delta$ under linear-predictor delta.
Simulation-grid coverage and comparison with R implementations arrive with
the simulation suite.
