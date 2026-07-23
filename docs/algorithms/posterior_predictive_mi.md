# Posterior-predictive multiple imputation — algorithm contract

Theory and citations: Proposition 2 of
[mathematical_foundations.md](../theory/mathematical_foundations.md)
(Rubin 1978, 1987; Tanner & Wong 1987; Schafer 1997). Interfaces:
[src/longmi/impute/base.py](https://github.com/aaronjdanielson/longmi/blob/main/src/longmi/impute/base.py) and
[src/longmi/contracts.py](https://github.com/aaronjdanielson/longmi/blob/main/src/longmi/contracts.py).

Two concrete backends implement this contract — the joint Gaussian
reference imputer ([joint_gaussian_imputer.md](joint_gaussian_imputer.md))
and the negative-binomial GLMM imputer
([negative_binomial_glmm.md](negative_binomial_glmm.md)); this file fixes
the contract they satisfy.

## Algorithm

Given validated `LongitudinalData` and a requested number of imputations
$M$, for $m = 1, \dots, M$:

1. **Parameter draw (A6).** Draw
   $\theta^{(m)} \sim p(\theta \mid Y^{\mathrm{obs}}, X)$ — the
   observed-data posterior (or a documented large-sample approximation to
   it, which must be declared in the backend's `ValidityDeclaration`).
2. **Outcome draw (A6).** Draw
   $Y^{\mathrm{mis},(m)} \sim
    p(Y^{\mathrm{mis}} \mid Y^{\mathrm{obs}}, X, \theta^{(m)})$,
   respecting the within-participant dependence structure (A8).
3. **Optional delta adjustment.** If a
   [`DeltaAdjustment`](https://github.com/aaronjdanielson/longmi/blob/main/src/longmi/scenarios.py) scenario is active,
   shift the draw on the declared scale before completion; count outcomes
   remain nonnegative integers.
4. **Completion.** Call `LongitudinalData.completed_with(...)`, which fills
   exactly the missing cells and re-validates: observed values preserved
   bit-for-bit, no missing outcome left, count constraints satisfied.

Return the $M$ frames as a `CompletedDatasetCollection` — its construction
is the certificate that the mechanical invariants hold.

## Requirements on every backend

- Both stages of randomness are present: a backend that fixes
  $\theta = \widehat\theta$ and samples only outcomes is **not** a valid
  `longmi` imputer (it yields improper, overconfident imputations).
- All randomness flows through the single `numpy.random.Generator` passed to
  `impute` — no hidden global state; runs are reproducible given the seed.
- Count outcomes are drawn as counts (e.g. the gamma–Poisson representation
  of the negative binomial), never rounded Gaussians:

$$
\Lambda_{ij}^{(m)} \sim
\operatorname{Gamma}\!\left(\kappa^{(m)},
  \frac{\kappa^{(m)}}{\mu_{ij}^{(m)}}\right),
\qquad
Y_{ij}^{\mathrm{mis},(m)} \sim \operatorname{Poisson}(\Lambda_{ij}^{(m)}).
$$

- The backend's `ValidityDeclaration` states: supported outcome types, the
  assumed missingness mechanism, whether parameter uncertainty is
  propagated, whether the analysis model is nested in the imputation model,
  known congeniality conditions, and the recommended pooling method.
- The imputation model contains every analysis variable, the
  exposure-by-time interaction, the relevant time functional form,
  within-participant dependence, and auxiliaries used to explain
  missingness (A8); categorical wave effects with exposure-by-wave
  interactions are the safe default, since they nest a linear-time analysis.

## Validation obligations

Stochastic imputations are never compared draw-for-draw across languages.
Backends are validated by: unit tests of the mechanical invariants;
simulation under known data-generating processes (bias, coverage, widening
with missing information, deliberate misspecification, delta-shift
response); and statistical agreement with trusted R implementations on
pooled estimates, standard errors, and imputed-value distributions.
