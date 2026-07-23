# Joint Gaussian reference imputer — algorithm as implemented

## At a glance

| Item | Joint Gaussian imputer |
| --- | --- |
| Status | Implemented; see [project status](../project-status.md) for validation maturity |
| Outcome | Continuous, fixed design waves |
| Missingness | MAR; delta-adjusted MNAR sensitivity |
| Dependence | Unstructured covariance across waves |
| Mean model | Wave-saturated (every predictor × every wave) |
| Parameter draw | Exact conjugate data augmentation (Jeffreys prior) |
| Outcome draw | Conditional multivariate normal |
| Main assumption | Approximately Gaussian outcome; participant-level predictors (A5, A8) |
| Do not use when | Count/bounded/strongly non-Gaussian outcomes; time-varying predictors |

Implementation:
[src/longmi/impute/gaussian.py](https://github.com/aaronjdanielson/longmi/blob/main/src/longmi/impute/gaussian.py).
Theory: Proposition 2 of
[mathematical_foundations.md](../theory/mathematical_foundations.md);
sources: Schafer (1997, the NORM algorithm with regressors); Tanner & Wong
(1987) for data augmentation.

## Model

Participants $i = 1, \dots, n$ observed at the $J$ design waves; predictors
$x_i \in \mathbb{R}^K$ (intercept added automatically) constant within
participant. The response vector follows the multivariate regression

$$
Y_i = B^\top x_i + e_i, \qquad e_i \sim N_J(0, \Sigma),
$$

with $B \in \mathbb{R}^{K \times J}$ (a separate coefficient for every
predictor at every wave) and unstructured $\Sigma$. The mean model is fully
saturated in time, so exposure-by-wave interactions are always present
(A8) and any analysis linear in these predictors and time is nested.

## Prior and exact conditionals

Jeffreys prior $p(B, \Sigma) \propto |\Sigma|^{-(J+1)/2}$. Given complete
data $(Y, X)$:

$$
\widehat B = (X^\top X)^{-1} X^\top Y,
\qquad
S = (Y - X\widehat B)^\top (Y - X\widehat B),
$$

$$
\Sigma \mid Y \sim \mathrm{InvWishart}(n - K,\ S),
\qquad
B \mid \Sigma, Y \sim \mathrm{MatrixNormal}\!\left(\widehat B,\
(X^\top X)^{-1},\ \Sigma\right).
$$

The matrix-normal draw is $B = \widehat B + A Z L^\top$ with
$AA^\top = (X^\top X)^{-1}$, $LL^\top = \Sigma$, $Z$ a $K \times J$ matrix
of standard normals.

## Data augmentation

1. **Initialize** missing cells by per-wave OLS prediction from $X$.
2. **Sweep** (Tanner–Wong): P-step draws $(\Sigma, B)$ from the exact
   conditionals above given the current completed data; I-step redraws
   every missing cell from the conditional multivariate normal
   $Y_i^{\mathrm{mis}} \mid Y_i^{\mathrm{obs}}, B, \Sigma$ (rows with no
   observed waves draw from the unconditional $N(B^\top x_i, \Sigma)$
   marginal on their missing coordinates).
3. **Keep** a completed dataset after `burn_in` sweeps, then after every
   further `thin` sweeps, until $M$ imputations exist. Both parameter and
   outcome uncertainty are propagated every sweep (A6).
4. An optional `DeltaAdjustment` shifts kept draws (identity link:
   linear-predictor and outcome scales coincide) for MNAR sensitivity.
5. Each kept matrix is passed to `LongitudinalData.completed_with`, which
   re-validates preservation of observed values and complete filling.

The backend separates fitting from generation:
`JointGaussianImputer.fit(data)` returns a `JointGaussianFit` (validated
design, declaration, model specification, data fingerprint), and
`fit.impute(m, random_state, delta=...)` runs a fresh chain — scenario
runs with equal seeds share identical underlying draws, so a delta
scenario differs from MAR only by the deterministic shift.

Imputations come from a single chain separated by `thin` sweeps after
`burn_in`. Every run reports `GaussianChainDiagnostics` — per-sweep lag-1
autocorrelation and a crude single-chain ESS of the `log det(Sigma)`
trace, plus the autocorrelation at kept-imputation spacing — so the
adequacy of `burn_in`/`thin` is inspected from the run, not asserted from
the defaults (multi-chain R-hat/ESS arrive with the simulation suite).
All randomness flows through an integer seed or the caller's
`numpy.random.Generator`; the seed, bit generator, and package version
are recorded in the run metadata. Wave order follows the declared design
order (`times=`), which the backend requires unless constructed with
`allow_undeclared_times=True`.

## Requirements and refusals

- continuous outcome only;
- complete participant-by-wave row grid (missing outcomes allowed, missing
  rows not);
- participant-level numeric predictors (time-varying predictors need the
  GLMM backends);
- $n - K \ge J$, else the $\Sigma$ posterior is improper — refused;
- $M \ge 2$.

## Validation

Unit tests
([tests/unit/test_gaussian_imputer.py](https://github.com/aaronjdanielson/longmi/blob/main/tests/unit/test_gaussian_imputer.py)):
mechanical invariants (preservation, reproducibility by seed,
between-imputation variability, guards) and fixed-seed statistical checks —
pooled recovery of a wave-specific treatment effect under sequential MAR
dropout, pooled SE exceeding single-imputation SE with nonzero FMI, and the
exact delta-shift property. Simulation-grid coverage and cross-language
statistical comparison arrive with the simulation suite (PR 5).
