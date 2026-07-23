# Mathematical foundations and validity conditions

`longmi` implements established missing-data theory; it does not introduce a
new proof of multiple imputation. This chapter states the framework, the
assumptions (detailed in [assumptions.md](assumptions.md)), and four cited
propositions. Each proposition records what is proved, under which
assumptions, and where the original proof lives. Full citations are in
[REFERENCES.md](https://github.com/aaronjdanielson/longmi/blob/main/REFERENCES.md).

## What these propositions establish

1. Under MAR and parameter distinctness, likelihood-based outcome
   inference may ignore the missingness model *(Proposition 1)*.
2. Drawing parameters and then missing outcomes generates
   posterior-predictive imputations *(Proposition 2)*.
3. Repeated completed-data estimates can be pooled under
   proper-imputation and compatibility conditions *(Proposition 3)*.
4. GEE and other estimating-equation analyses need additional care,
   because Rubin variance is not universally robust to uncongeniality
   *(Proposition 4)*.

Plain-language background is in
[why impute the response?](../explanation/why_impute_the_response.md) and
[MCAR, MAR, and MNAR](../explanation/mcar_mar_mnar.md).

## 1. Full and observed data

Participant $i = 1, \dots, n$ has longitudinal response vector

$$
Y_i = (Y_{i1}, \dots, Y_{iJ})^\top
$$

at known observation times $t_1, \dots, t_J$, and fully observed predictors
$X_i$. The response indicator is

$$
R_{ij} = \mathbf{1}\{Y_{ij} \text{ is observed}\},
$$

and the response partitions as $Y = (Y^{\mathrm{obs}}, Y^{\mathrm{mis}})$.

Two models are in play:

- the longitudinal response model $p_\theta(Y \mid X)$, parameterized by
  $\theta$;
- the missingness model $p_\psi(R \mid Y, X)$, parameterized by $\psi$.

The estimand $Q$ is a functional of the response model (in the motivating
application, coefficients of a marginal count-outcome regression).

## 2. Assumptions

Stated in full in [assumptions.md](assumptions.md); in brief:

- **A1 (independent clusters)** — $(Y_i, R_i, X_i)$ are independent across
  $i$; dependence *within* a participant is allowed and explicitly modeled.
- **A2 (missing at random)** —
  $p_\psi(R \mid Y^{\mathrm{obs}}, Y^{\mathrm{mis}}, X)
   = p_\psi(R \mid Y^{\mathrm{obs}}, X)$.
- **A3 (distinct parameters)** — $\theta$ and $\psi$ are variation
  independent.
- **A4 (positivity)** — observed histories have nonzero follow-up
  probability.
- **A5 (correct imputation model)** —
  $p_\theta(Y^{\mathrm{mis}} \mid Y^{\mathrm{obs}}, X)$ is correctly
  specified for the estimand of interest.
- **A6 (proper parameter and outcome draws)** — each imputation propagates
  uncertainty in both the parameters and the missing outcomes.
- **A7 (valid complete-data estimator)** — the completed-data estimator is
  consistent and asymptotically normal with a valid covariance estimator.
- **A8 (congeniality / compatibility)** — the imputation model preserves the
  relationships needed by the analysis model.

## 3. Proposition 1 — Ignorability under MAR

**Claim.** Under A2 (MAR) and A3 (distinctness), the missingness mechanism
may be ignored for likelihood-based or Bayesian inference about $\theta$.

**Source.** Rubin DB. Inference and Missing Data. *Biometrika.*
1976;63(3):581–592.

**Derivation.** The observed-data likelihood is

$$
L(\theta, \psi)
= p(Y^{\mathrm{obs}}, R \mid X, \theta, \psi)
= \int
    p_\theta(Y^{\mathrm{obs}}, Y^{\mathrm{mis}} \mid X)\,
    p_\psi(R \mid Y^{\mathrm{obs}}, Y^{\mathrm{mis}}, X)
  \, dY^{\mathrm{mis}}.
$$

Under MAR the missingness density does not depend on $Y^{\mathrm{mis}}$, so
it moves outside the integral:

$$
L(\theta, \psi)
= p_\psi(R \mid Y^{\mathrm{obs}}, X)
  \int p_\theta(Y^{\mathrm{obs}}, Y^{\mathrm{mis}} \mid X)\, dY^{\mathrm{mis}}
= p_\psi(R \mid Y^{\mathrm{obs}}, X)\;
  p_\theta(Y^{\mathrm{obs}} \mid X).
$$

If $\theta$ and $\psi$ are distinct (A3), the factorization implies inference
about $\theta$ can be based on $p_\theta(Y^{\mathrm{obs}} \mid X)$ alone,
without fitting the missingness model.

**What this does and does not establish.** This is *ignorability under MAR*,
not a proof that MAR is true. The theorem establishes the consequence of MAR
if it is assumed; MAR itself is not empirically testable from the observed
data (see A2 in [assumptions.md](assumptions.md)).

## 4. Proposition 2 — Posterior-predictive imputation

**Claim.** Drawing parameters from their observed-data posterior and then
missing responses from their conditional distribution generates draws from
the posterior-predictive distribution
$p(Y^{\mathrm{mis}} \mid Y^{\mathrm{obs}}, X)$.

**Sources.** Rubin 1978; Rubin 1987; Tanner & Wong 1987; Schafer 1997.

**Derivation.** For imputation $m$, draw

$$
\theta^{(m)} \sim p(\theta \mid Y^{\mathrm{obs}}, X),
\qquad
Y^{\mathrm{mis},(m)} \sim
p\!\left(Y^{\mathrm{mis}} \mid Y^{\mathrm{obs}}, X, \theta^{(m)}\right).
$$

Marginalizing over $\theta$,

$$
p(Y^{\mathrm{mis}} \mid Y^{\mathrm{obs}}, X)
= \int
    p(Y^{\mathrm{mis}} \mid Y^{\mathrm{obs}}, X, \theta)\,
    p(\theta \mid Y^{\mathrm{obs}}, X)
  \, d\theta,
$$

which is ordinary Bayesian marginalization: the two-stage scheme samples
exactly this mixture.

**Consequence for implementation.** Every `longmi` imputer must draw *both*
model parameters and missing outcomes (A6). Fixing
$\theta = \widehat\theta$ and sampling only outcomes omits parameter
uncertainty and generally yields overconfident inference.

**Terminological care.** The displayed integral proves that the two-stage
algorithm generates *posterior-predictive draws*. "Proper imputation" has a
more technical repeated-sampling definition in Rubin (1987, ch. 4).
Posterior-predictive draws from a correctly specified Bayesian model are the
canonical route to proper MI, but the two statements are distinguished
throughout this documentation.

**Count outcomes.** For a negative-binomial response model, one
posterior-predictive draw uses the gamma–Poisson representation:

$$
\Lambda_{ij}^{(m)} \sim
\operatorname{Gamma}\!\left(\kappa^{(m)},\,
  \frac{\kappa^{(m)}}{\mu_{ij}^{(m)}}\right),
\qquad
Y_{ij}^{\mathrm{mis},(m)} \sim \operatorname{Poisson}(\Lambda_{ij}^{(m)}),
$$

which guarantees $Y_{ij}^{\mathrm{mis},(m)} \in \{0, 1, 2, \dots\}$.

## 5. Proposition 3 — Repeated-imputation inference

**Claim.** Under proper imputation, valid complete-data inference, regularity
conditions, and congeniality, completed-data estimates may be pooled with
Rubin's repeated-imputation rules; the pooled estimator is consistent for
$Q_0$ as $n \to \infty$.

**Sources.** Rubin 1987; Meng 1994; Wang & Robins 1998.

**Point estimate.** For completed dataset $m$, compute
$\widehat Q^{(m)}$; the MI point estimate is

$$
\overline Q_M = \frac{1}{M} \sum_{m=1}^{M} \widehat Q^{(m)}.
$$

Conditional on the observed data, the completed datasets are
posterior-predictive draws, so by the conditional law of large numbers

$$
\overline Q_M
\xrightarrow{\ \text{a.s.}\ }
E\!\left[\widehat Q(Y^{\mathrm{obs}}, Y^{\mathrm{mis}}, X)
  \mid Y^{\mathrm{obs}}, X\right]
\qquad (M \to \infty).
$$

That much is elementary Monte Carlo. The stronger claim — that under
identification, correct imputation-model specification (A5), regularity, a
consistent complete-data estimator (A7), and compatibility (A8),

$$
\overline Q_M \xrightarrow{\ p\ } Q_0 \qquad (n \to \infty)
$$

— is the subject of Wang & Robins (1998), who derive the asymptotic behavior
and variance structure of parametric multiple-imputation estimators,
including procedures beyond Rubin's original class of proper imputations.

**Variance.** With within-imputation variance
$U^{(m)} = \widehat{\operatorname{Var}}(\widehat Q^{(m)} \mid
Y^{\mathrm{complete},(m)})$, define

$$
\overline U = \frac{1}{M} \sum_{m=1}^{M} U^{(m)},
\qquad
B = \frac{1}{M-1} \sum_{m=1}^{M}
  \left(\widehat Q^{(m)} - \overline Q\right)
  \left(\widehat Q^{(m)} - \overline Q\right)^{\!\top},
$$

$$
\boxed{\,T = \overline U + \left(1 + \tfrac{1}{M}\right) B.\,}
$$

The motivation is the law of total variance,

$$
\operatorname{Var}(Q \mid Y^{\mathrm{obs}})
= E\!\left[\operatorname{Var}(Q \mid Y^{\mathrm{complete}})
    \mid Y^{\mathrm{obs}}\right]
+ \operatorname{Var}\!\left[E(Q \mid Y^{\mathrm{complete}})
    \mid Y^{\mathrm{obs}}\right],
$$

with $\overline U$ estimating the first term, $B$ the second, and $B/M$
accounting for the Monte Carlo error in $\overline Q_M$ itself.

**This one-line decomposition is motivation, not a complete proof.** Exact
repeated-imputation validity additionally requires proper imputations (A6),
valid complete-data inference (A7), and congeniality between the imputation
and analysis procedures (A8; Meng 1994). The same formula applies with
covariance matrices for vector $Q \in \mathbb{R}^p$. Degrees of freedom and
fraction of missing information follow Barnard & Rubin (1999); the exact
formulas as implemented are in
[../algorithms/rubin_pooling.md](../algorithms/rubin_pooling.md).

## 6. Proposition 4 — Estimating-equation qualification

**Claim.** When the substantive analysis is a method-of-moments or GEE
estimator, standard Rubin variance requires additional compatibility
conditions; robust alternatives include Robins–Wang variance,
over-imputation, or bootstrap-then-impute.

**Sources.** Robins & Wang 2000; Yang & Kim 2016; Bartlett & Hughes 2020;
see also Meng 1994.

A GEE estimator solves $\sum_i U_i(\beta) = 0$ and is not derived from a
likelihood. Meng (1994) formalizes *congeniality* — whether the imputation
and analysis procedures can be viewed as arising from a coherent common
Bayesian model — and shows Rubin's variance can be conservative or
anticonservative without it. Robins & Wang (2000) derive variance estimators
that remain consistent when the imputation and analysis models are
incompatible or misspecified. Yang & Kim (2016) show Rubin's variance
estimator can be asymptotically biased when the complete-data estimator is
method-of-moments and congeniality fails.

`longmi` therefore does **not** claim that any imputation model plus any GEE
plus Rubin pooling is automatically valid. The defensible claim, stated in
[gee_after_imputation.md](gee_after_imputation.md), is:

> Standard Rubin pooling is supported when imputations are proper, the
> imputation model is compatible or congenial with the GEE estimand, and the
> complete-data sandwich covariance is valid.

Bootstrap-then-impute (Bartlett & Hughes 2020) is the planned robust
fallback: resample participants, refit the imputation model within each
replicate, impute, refit the analysis, and take inference from the bootstrap
distribution. The interface anticipates this mode now; implementation is
scheduled for 0.2.

## 7. What the software enforces

Proofs establish validity **under stated assumptions**; the package enforces
what is mechanically checkable and *declares* the rest:

- checkable, and enforced by `LongitudinalData` /
  `CompletedDatasetCollection`: observed values never overwritten; every
  eligible missing value imputed; count outcomes remain nonnegative
  integers; predictors fully observed; identical term ordering across
  completed-data fits;
- declared, via each imputer's `ValidityDeclaration` and surfaced by
  `result.validity_report()`: the assumed missingness mechanism, whether
  parameter and outcome uncertainty are propagated, congeniality status,
  and the recommended pooling method;
- assessed empirically, via the simulation suite (bias, coverage, widening
  with missing information, deliberate misspecification) and cross-language
  parity with R.
