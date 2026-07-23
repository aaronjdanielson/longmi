# GEE after multiple imputation

The motivating application analyzes each completed dataset with a marginal
GEE model. This file records why that combination needs a qualification, what
`longmi` claims, and what it refuses to claim.

## The workflow

For completed dataset $m$, the GEE estimator solves

$$
\sum_i D_{im}^{\top} V_{im}^{-1}
\left(Y_i^{(m)} - \mu_i(\beta)\right) = 0,
$$

yielding $\widehat\beta^{(m)}$ and a robust (sandwich) complete-data
covariance $U^{(m)}$. The coefficient vectors and covariance matrices are
pooled with the multivariate Rubin rules
([../algorithms/rubin_pooling.md](../algorithms/rubin_pooling.md)).

MI followed by GEE is established practice in longitudinal research
(Beunckens, Sotto & Molenberghs 2008; Lipsitz, Fitzmaurice & Weiss 2020).

## The qualification

A GEE estimator is an estimating-equation (method-of-moments) estimator, not
a likelihood-based one. The relevant results:

- **Meng (1994)** formalizes *congeniality*: whether the imputation and
  analysis procedures can be embedded in a coherent common Bayesian model.
  Without it, Rubin's variance can be too large or too small.
- **Robins & Wang (2000)** derive variance estimators for imputation
  estimators that remain consistent when the imputation and analysis models
  are incompatible or misspecified — directly relevant to GEE.
- **Yang & Kim (2016)** show Rubin's variance estimator can have asymptotic
  bias when the complete-data estimator is method-of-moments and
  congeniality is absent, and propose an over-imputation variance estimator.

`longmi` therefore does **not** claim:

> any imputation model + any GEE + Rubin pooling is automatically valid.

The defensible claim is:

> **Standard Rubin pooling is supported when imputations are proper, the
> imputation model is compatible or congenial with the GEE estimand, and the
> complete-data sandwich covariance is valid.**

## What every imputation backend must declare

```text
supported outcome types
assumed missingness mechanism
whether parameter uncertainty is propagated
whether the analysis model is nested in the imputation model
known congeniality conditions
recommended pooling method
```

This is the `ValidityDeclaration` contract in
[../../src/longmi/contracts.py](../../src/longmi/contracts.py), surfaced on
pooled results via `validity_report()`.

For the motivating count-outcome model, the imputer must contain at least
the exposure $G_i$, time $t_j$, their interaction $G_i t_j$, and baseline
covariates $C_i$ — and preferably categorical wave effects with the
corresponding exposure-by-wave interactions, since a categorical-wave
imputation model contains the linear-time analysis as a restricted special
case (A8).

## Robust fallback: bootstrap-then-impute

When congeniality is uncertain, the planned robust inference mode (0.2) is
bootstrap-then-impute (Bartlett & Hughes 2020):

1. resample participants (the sampling unit, per A1);
2. refit the imputation model within the bootstrap sample;
3. generate imputations inside the bootstrap sample;
4. refit the complete-data analysis;
5. obtain inference from the bootstrap distribution.

Bartlett & Hughes find that the order matters: impute-then-bootstrap is
generally not valid under uncongeniality or misspecification, whereas
appropriate bootstrap-then-impute procedures can be. Nuisance models are
refit within each replicate; estimated quantities are never held fixed
across replicates.
