# longmi

Multiple imputation for one incomplete longitudinal response, with the validity
conditions stated in the software rather than buried in a paper.

`longmi` implements established missing-data theory — Rubin (1976, 1987),
Barnard–Rubin (1999), Meng (1994), Wang–Robins (1998) — it does not introduce a
new proof of multiple imputation. Its contribution is:

- an independent, tested Python implementation;
- explicit declaration of the assumptions under which each method is valid;
- safeguards against incompatible imputation and analysis specifications;
- longitudinal count-outcome support;
- Python–R numerical validation;
- simulation evidence for bias and coverage (planned);
- robust alternatives when ordinary Rubin pooling is questionable (planned).

## Scope of 0.1

One incomplete longitudinal response, with:

- independent participants or clusters;
- known observation times;
- fully observed predictors;
- MCAR or MAR missingness (delta-adjusted MNAR sensitivity planned);
- posterior-predictive multiple imputation;
- continuous and count outcomes;
- arbitrary complete-data analyses through an adapter;
- Rubin pooling with Barnard–Rubin degrees of freedom.

Out of scope for 0.1: imputation of incomplete covariates or exposures,
fully conditional specification, wide-format methods.

## Current status

The deterministic core is implemented and validated:

- `LongitudinalData` — strict validation of long-format incomplete data
  ([src/longmi/data.py](src/longmi/data.py));
- `CompletedDatasetCollection` — completed datasets with observed-value
  preservation enforced;
- `AnalysisEstimate` / `RubinPooledResult` — typed containers
  ([src/longmi/results.py](src/longmi/results.py));
- `pool_rubin()` — scalar and multivariate Rubin rules, Barnard–Rubin
  small-sample degrees of freedom, fraction of missing information
  ([src/longmi/pooling/rubin.py](src/longmi/pooling/rubin.py));
- exact numerical parity with R `mice::pool.scalar`
  ([validation/r/rubin_reference.R](validation/r/rubin_reference.R),
  [tests/cross_language/](tests/cross_language/)).

Two imputation backends are implemented:

- `JointGaussianImputer` — posterior-predictive joint Gaussian imputation
  for fixed-wave data via exact conjugate data augmentation (Schafer 1997),
  wave-saturated mean model, unstructured covariance, delta-adjustment
  support ([src/longmi/impute/gaussian.py](src/longmi/impute/gaussian.py),
  [docs/algorithms/joint_gaussian_imputer.md](docs/algorithms/joint_gaussian_imputer.md));
- `NegativeBinomialImputer` — NB random-intercept imputation for
  longitudinal counts: Gauss–Hermite ML, declared large-sample posterior
  approximation for parameter draws, exact conditional random-intercept
  draws, gamma–Poisson outcome draws, linear-predictor delta adjustment
  ([src/longmi/impute/negbin.py](src/longmi/impute/negbin.py),
  [docs/algorithms/negative_binomial_glmm.md](docs/algorithms/negative_binomial_glmm.md)).

Backends separate fitting from generation:

```python
fit = imputer.fit(data)                       # validate, estimate, diagnose
mar = fit.impute(m=100, random_state=1)       # MAR
mnar = fit.impute(m=100, random_state=1,      # MNAR scenario, shared
                  delta=DeltaAdjustment(...)) # randomness with MAR run
```

Each fit exposes `diagnostics` (optimizer/curvature checks for the NB
backend; single-chain autocorrelation/ESS for the Gaussian backend),
`declaration`, `model_specification`, and `data_fingerprint`; run metadata
records the seed, bit generator, and package version. A failed optimizer
or materially indefinite curvature refuses to impute rather than warn.

Analyses plug in through `AnalysisModel.fit`; wrap plain functions with
`CallableAnalysis`. Next milestones: statsmodels GEE/GLM adapters, the MI
arm of the epil example, then the simulation and cross-language
statistical validation suite — the backends are not claimed statistically
validated until that bias/coverage evidence exists.

## Examples and validation data

- [examples/epil_count/](examples/epil_count/) — the primary worked example
  (`MASS::epil` seizure counts, loaded from upstream in both languages —
  never redistributed — with shared missingness masks; complete-data and
  available-case GEE currently agree across Python/statsmodels and
  R/geepack within 2e-3);
- [examples/cats_tutorial_parity/](examples/cats_tutorial_parity/) —
  pinned external checkout of the published tutorial's R code and simulated
  CATS data as a methodological oracle;
- [tests/fixtures/](tests/fixtures/) — package-owned synthetic data for
  fast deterministic tests.

## Mathematical documentation

- [docs/theory/mathematical_foundations.md](docs/theory/mathematical_foundations.md)
  — foundations and validity conditions, as four cited propositions;
- [docs/theory/assumptions.md](docs/theory/assumptions.md) — assumptions A1–A8;
- [docs/theory/gee_after_imputation.md](docs/theory/gee_after_imputation.md)
  — congeniality and the estimating-equation qualification;
- [docs/algorithms/rubin_pooling.md](docs/algorithms/rubin_pooling.md) — the
  pooling algorithm as implemented;
- [docs/algorithms/posterior_predictive_mi.md](docs/algorithms/posterior_predictive_mi.md)
  — the imputation algorithm contract.

## Design principle

Every result should be able to state the conditions under which it is valid:

```python
result.validity_report()
```

renders the missingness assumption, whether parameter and outcome uncertainty
were propagated, whether observed outcomes were preserved, the congeniality
status, and the pooling method — as part of the result object, not only prose.

## Development

```bash
pip install -e ".[dev]"
pytest
```

Cross-language reference values are regenerated with:

```bash
Rscript validation/r/rubin_reference.R
```

See [REFERENCES.md](REFERENCES.md) for the citation list.
