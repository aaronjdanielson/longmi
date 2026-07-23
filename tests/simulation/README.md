# Simulation tests

Frequentist performance under known data-generating processes. These are
seeded pytest modules marked `slow` and excluded from the default run:

```bash
pytest -m slow tests/simulation -q          # default replication counts
LONGMI_SIM_REPS=500 pytest -m slow tests/simulation -q   # full run
```

Implemented (via [harness.py](harness.py): bias with Monte Carlo SE, RMSE,
empirical 95% coverage, interval width):

- `test_gaussian_coverage.py` — joint Gaussian imputer under MCAR and
  sequential MAR; target = wave-3 treatment effect (OLS/HC1 per completed
  dataset, Rubin pooling).
- `test_negbin_coverage.py` — NB GLMM imputer + Poisson GEE
  (exchangeable, robust) + Rubin under sequential MAR; the GEE estimand's
  truth is calibrated on one very large complete dataset (marginal vs
  conditional attenuation under the log link).

Planned extensions: misspecified-imputation-model failure demonstrations,
MNAR delta-shift response curves, intermittent (non-monotone) missingness,
and wider grids over sample size, missingness fraction, correlation, and
overdispersion.
