# Choosing an imputer

| Feature | `JointGaussianImputer` | `NegativeBinomialImputer` |
| --- | --- | --- |
| Outcome | Continuous | Nonnegative count |
| Longitudinal structure | Fixed waves, unstructured covariance | Random participant intercept |
| Time in the mean model | Wave-saturated (every predictor × every wave) | Categorical wave + declared interactions |
| Predictors | Participant-level (constant within participant), numeric | Row-level (time-varying allowed) |
| Parameter uncertainty | Exact conjugate data augmentation (Jeffreys prior) | Large-sample normal approximation, N(θ̂, H⁻¹) |
| Within-person dependence | Unstructured Σ | Random intercept, N(0, τ²) |
| MNAR sensitivity | Delta on outcome/linear-predictor scale (identity link: same thing) | Delta on linear-predictor scale only |
| Strongest use case | Approximately continuous repeated outcomes | Overdispersed repeated counts (medication counts, seizure counts) |
| Avoid when | Strongly non-Gaussian, bounded, or count outcomes | Zero inflation, structural zeros, random slopes needed |
| Diagnostics | Single-chain autocorrelation / ESS per run | Optimizer, curvature, quadrature, grid boundary mass |
| Validation status | See [project status](../project-status.md) | See [project status](../project-status.md) |

## Rules of thumb

- **Count outcome → NB imputer.** Drawing counts as counts (gamma–Poisson)
  beats rounding Gaussians; support constraints are enforced, not patched.
- **Continuous outcome on a fixed wave grid → joint Gaussian.** Its
  wave-saturated mean and unstructured covariance make it hard for a
  linear analysis model to be uncongenial with it (A8).
- **Whichever you choose, put the analysis's exposure-by-time interaction
  in the imputation model** — automatic for the Gaussian backend; for the
  NB backend list the exposure in `time_interactions=`.
- **Neither backend imputes covariates.** Predictors must be complete
  ([preparing data](../getting_started/preparing_data.md)).

Both imputers are declared `experimental` until the simulation grid in
[project status](../project-status.md) says otherwise; treat agreement
between them (where both apply) as reassurance, not proof.
