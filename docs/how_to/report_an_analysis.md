# Report an analysis

What a manuscript's methods section (or supplement) should state about a
longmi analysis. Everything below is recorded in the result objects — no
reconstruction from memory.

## Checklist

- [ ] Number and pattern of missing outcomes (monotone vs intermittent;
      retention by wave) — from `LongitudinalData` / your mask summary.
- [ ] Assumed missingness mechanism, stated as an assumption
      (`result.validity_report()`).
- [ ] Imputation model: backend, variables, wave terms, interactions,
      auxiliaries, dependence structure —
      `completed.metadata["model_specification"]`.
- [ ] Number of imputations M and how parameters were drawn (exact
      conjugate vs large-sample approximation — the backend's
      declaration `notes` say which).
- [ ] Diagnostics: optimizer/curvature outcome (NB), chain
      autocorrelation (Gaussian), and the per-parameter `fmi`.
- [ ] Analysis model and covariance estimator (e.g. Poisson GEE,
      exchangeable working correlation, robust sandwich).
- [ ] Pooling rule and degrees-of-freedom method (Rubin's rules;
      Barnard–Rubin, `mice`-compatible).
- [ ] Delta values examined for MNAR sensitivity and their substantive
      justification.
- [ ] Software version and random seed —
      `completed.metadata["random_state"]`.

## Paragraph template

> Missing longitudinal outcomes were multiply imputed using longmi
> version {X} (seed {S}). The imputation model was a {backend: e.g.
> negative-binomial generalized linear mixed model with a participant
> random intercept}, including {variables}, categorical wave effects, and
> {exposure}-by-wave interactions, so the substantive model's
> {exposure}-by-time term was represented in the imputation model. We
> generated M = {M} completed datasets under the missing-at-random
> assumption; parameter uncertainty was propagated by {exact conjugate
> data augmentation / draws from the large-sample normal approximation to
> the posterior}. Each completed dataset was analyzed with {analysis:
> e.g. a population-averaged Poisson GEE with exchangeable working
> correlation and robust standard errors, clustered by participant}, and
> estimates were combined using Rubin's rules with Barnard–Rubin degrees
> of freedom. The fraction of missing information for the primary
> contrast was {fmi}. Because missing-at-random is untestable, we
> repeated the analysis with delta-adjusted imputations shifting imputed
> {means} by {factors}; conclusions {were / were not} materially altered.

## One honesty rule

Do not describe the analysis as having "verified" MAR or congeniality —
the validity report deliberately tags these `[declared]`. State them as
assumptions and point to the sensitivity analysis.
