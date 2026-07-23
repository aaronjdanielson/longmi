# Assumptions A1–A8

Every method in `longmi` is valid only under stated assumptions. This file is
the canonical enumeration; imputers and pooled results reference these labels
in their `ValidityDeclaration`. None of A2, A5, A8 is fully verifiable from
observed data — the package distinguishes what it *enforces*, what it
*declares*, and what must be *argued substantively*.

## A1. Independent clusters

$$
(Y_i, R_i, X_i) \perp (Y_k, R_k, X_k), \qquad i \neq k.
$$

Dependence **within** participants is allowed and must be explicitly modeled.
The participant (or cluster) is the sampling unit for all resampling-based
inference.

*Practical check:* confirm the id column identifies the independent unit; if
participants are nested in sites, the site may be the correct unit.

## A2. Missing at random (MAR)

$$
p_\psi(R \mid Y^{\mathrm{obs}}, Y^{\mathrm{mis}}, X)
= p_\psi(R \mid Y^{\mathrm{obs}}, X).
$$

Missingness may depend on observed baseline and longitudinal outcomes, but
not additionally on the missing outcomes after conditioning on the observed
data. MCAR is the stronger special case $p(R \mid Y, X) = p(R)$.

*Practical check:* MAR is **not empirically testable** against MNAR from the
observed data alone. The package therefore treats MNAR sensitivity analysis
(delta adjustment) as the appropriate response to doubt about A2, not a
hypothesis test.

## A3. Distinct parameters

The response-model parameter $\theta$ and missingness-model parameter $\psi$
are variation independent: the joint parameter space is
$\Theta \times \Psi$, and the prior or likelihood imposes no shared
restrictions that would break ignorability.

*Practical check:* rarely violated in applied settings; violated when, e.g.,
a shared random effect drives both outcome and dropout (a shared-parameter
MNAR model).

## A4. Positivity

For histories represented in the target population,

$$
P(R_{ij} = 1 \mid H_{i,j-1}) > 0.
$$

No observed type of participant can have a structurally zero probability of
follow-up.

*Practical check:* inspect follow-up rates within strata of predictors and
observed history; structural zeros (e.g., a site that never administered
wave 3) must be handled by design, not imputation.

## A5. Correct imputation model

$p_\theta(Y^{\mathrm{mis}} \mid Y^{\mathrm{obs}}, X)$ is correctly specified
for the estimand of interest — or at least sufficiently rich that the
relevant conditional means, interactions, and variances are correctly
represented.

*Practical check:* posterior-predictive checks on observed margins;
simulation under known data-generating processes (the simulation suite
deliberately includes misspecified imputation models to demonstrate
failure).

## A6. Proper parameter and outcome draws

Each imputation includes uncertainty in **both** the model parameters and
the missing outcomes (Proposition 2). Fixing parameters at point estimates
produces improper, overconfident imputations.

*Enforced:* every imputer's `ValidityDeclaration` must state
`parameter_uncertainty_propagated`; the reference imputers draw
$\theta^{(m)}$ per imputation.

## A7. Valid complete-data estimator

The estimator applied to each completed dataset is consistent and
asymptotically normal when complete data are observed, with a valid
covariance estimator (for GEE, the robust sandwich covariance).

*Practical check:* the analysis adapter must return the covariance actually
recommended for the complete-data method, not a model-based shortcut.

## A8. Congeniality / compatibility

The imputation model preserves the relationships needed by the analysis
model. In particular it must include:

- every analysis variable;
- the exposure-by-time interaction;
- the relevant time functional form;
- clustering / within-participant dependence;
- auxiliary variables used to explain missingness.

A categorical-wave imputation model contains a linear-time analysis as a
restricted special case, so imputing with categorical wave effects (and the
corresponding exposure-by-wave interactions) is the safe default.

**Special care with GEE:** standard Rubin variance is not universally robust
to uncongeniality when the completed-data estimator is method-of-moments
(Meng 1994; Yang & Kim 2016). See
[gee_after_imputation.md](gee_after_imputation.md) for the qualification and
robust alternatives.
