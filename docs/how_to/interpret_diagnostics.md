# Interpret diagnostics

longmi refuses to impute from a demonstrably failed fit (optimizer
failure, materially indefinite curvature). Everything softer is reported
for you to judge. This page says what each diagnostic measures and what to
do when it is concerning.

## Negative-binomial backend (`fit.diagnostics`)

| Diagnostic | Measures | Reassuring | Concerning | Action |
| --- | --- | --- | --- | --- |
| `optimizer_success`, `gradient_norm` | Whether the marginal-likelihood fit reached an acceptable optimum | Success, small gradient | (You will not see this: a failed fit raises) | Rescale predictors; simplify the design; more quadrature nodes |
| `hessian_eigenvalues` (min) | Local identifiability / numerical stability | All clearly positive | Near zero: weakly identified parameter (often τ with few repeat observations) | Reconsider the random-effect structure; more data per participant |
| `covariance_repaired`, `covariance_min_eigenvalue` | Whether roundoff-level negative eigenvalues were clipped | `False`, or `True` with a tiny magnitude | (Materially indefinite raises instead) | — |
| `n_quad` + refit sensitivity | Quadrature adequacy | Estimates stable across e.g. `n_quad=25/40/60` | Material parameter movement | Increase nodes; suspect a very wide random-effect distribution |
| `grid_max_boundary_mass` (run metadata) | Containment of the random-intercept posterior grid | ≤ 1e-8 (enforced) | (Exceeding tolerance after expansion raises) | Inspect the fit; τ draws may be extreme |
| `kappa_draw_range`, `tau_draw_range` (run metadata) | Stability of the parameter draws | Narrow, plausible ranges | Orders-of-magnitude spread → the normal approximation is strained | More data; fewer parameters; treat results cautiously |

## Joint Gaussian backend (`metadata["diagnostics"]`)

| Diagnostic | Measures | Reassuring | Concerning | Action |
| --- | --- | --- | --- | --- |
| `trace_lag1_autocorrelation` | Per-sweep mixing of the data-augmentation chain (log det Σ trace) | Low/moderate | Near 1: sticky chain | Increase `thin` and `burn_in` |
| `trace_ess` | Crude single-chain effective sample size | Large relative to sweeps | Small | Same |
| `kept_lag1_autocorrelation` | Dependence *between kept imputations* | Near zero — imputations effectively independent | Clearly positive | Increase `thin`; this is the one that directly affects pooling |

These are single-chain heuristics; multi-chain R-hat/ESS arrive with the
simulation suite. Judge the defaults from the run's numbers, not from the
defaults themselves.

## Pooled result (`summary()`)

| Diagnostic | Measures | Concerning | Action |
| --- | --- | --- | --- |
| `fmi` | Fraction of missing information per parameter | > ~0.5: the conclusion leans heavily on the imputation model | Increase M (more imputations); add auxiliaries to the imputation model; widen the delta-sensitivity range and report it prominently |
| `df` | Barnard–Rubin degrees of freedom | Very small: unstable variance estimates | Increase M |
| MI SE vs single-fit SE | Honesty of the extra width | MI SE *smaller* than a complete-data fit's would indicate a bug — report it | — |
