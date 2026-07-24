# Bernoulli random-intercept imputer — algorithm as implemented

## At a glance

| Item | Bernoulli imputer |
| --- | --- |
| Status | Implemented; see [project status](../project-status.md) |
| Outcome | Binary (0/1) longitudinal |
| Missingness | MAR; logit-scale delta MNAR sensitivity (targeted rules supported) |
| Dependence | Participant random intercept |
| Likelihood | `y*eta - logaddexp(0, eta)`, Gauss–Hermite integration |
| Parameter draw | Large-sample normal approximation (declared) |
| Random-effect draw | Adaptive numerical grid approximation (controlled) |
| Outcome draw | Bernoulli(expit(eta + b)) — never rounded probabilities |
| Main caveat | Conditional (subject-specific) imputer vs marginal logistic GEE — evaluated by simulation, not assumed congenial |
| Do not use when | Complete/quasi separation; binomial counts (successes of n); random slopes needed |

Implementation: `src/longmi/impute/bernoulli.py`, built on the shared
GLMM machinery (`_glmm.py`) — the same verified-convergence ML,
curvature validation, and adaptive-grid draws as the NB backend.

## Model

$$
Y_{ij}\mid b_i \sim \mathrm{Bernoulli}(p_{ij}),\quad
\mathrm{logit}(p_{ij}) = x_{ij}^\top\alpha + b_i,\quad
b_i \sim N(0,\tau^2),
$$

with design intercept + categorical waves + predictors + declared
predictor-by-wave interactions (A8).

## Separation and identifiability safeguards

Fits are refused when: no outcomes observed; all observed outcomes 0 or
all 1 (marginal complete separation); observed design rank deficient;
optimization or curvature validation fails. A maximum absolute
coefficient above 15 on the logit scale triggers a quasi-separation
warning — near-deterministic imputations are surfaced, never silent.
Diagnostics record observed events/nonevents, optimizer facts, curvature
spectrum, grid boundary mass, and rejected parameter draws.

## Delta sensitivity (logit scale only)

$$\mathrm{logit}\,p^{\delta}_{ij} = \mathrm{logit}\,p^{\mathrm{MAR}}_{ij} + \delta_{ij}$$

so $e^{\delta}$ multiplies the **conditional odds**, never the
probability; outcome-scale shifts are refused. Targeted rules
(`DeltaAdjustment(where=..., times=...)`, `DeltaScenario`) are supported:
rules apply only to missing responses, overlaps are rejected, and the
realized per-row delta vector is recorded in run metadata for audit.

## Conditional vs marginal — the congeniality caveat

The imputer models subject-specific (conditional) log-odds $\alpha$; a
marginal logistic GEE targets population-averaged $\beta$, and under a
logit link with random intercepts $\alpha \neq \beta$ (no closed-form
collapse). The declaration therefore states
`analysis_nested_in_imputation_model = False`, and the MI-to-GEE
workflow is validated **by simulation against the marginal target**
$\beta^\*$ (the population solution of the GEE estimating equation,
computed from a large complete synthetic population) — see
`tests/simulation/test_bernoulli_coverage.py`. Acceptable performance in
evaluated scenarios does not establish universal congeniality.
