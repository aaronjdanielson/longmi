# Worked example: epileptic seizure counts

The primary longmi vignette, built on `MASS::epil` (Thall & Vail 1990): a
real, public, overdispersed longitudinal count outcome whose complete
values are known — so every missing-data method in the package can be
checked against the truth it is trying to recover. Implementation lives in
[examples/epil_count/](../../examples/epil_count/).

## The narrative

1. **Load complete seizure-count data** from upstream in both Python and R,
   normalized to a shared schema and proven identical by hash. *(done)*
2. **Fit the complete-data GEE** — Poisson, exchangeable working
   correlation, robust SEs, model
   `log E(y) = b0 + bT*treat + bP*period + bTP*treat:period +
   bB*log(1+base) + bA*age`; the central contrast is `H0: bTP = 0`.
   *(done, Python and R agree within 2e-3)*
3. **Create monotone MAR dropout** with a sequential logistic mechanism
   depending only on observed history; masks are shared CSVs so both
   languages see the identical pattern. *(done)*
4. **Show the cost of available-case GEE** relative to the complete-data
   benchmark. *(done — compare `results_python.csv` analyses)*
5. **Impute** missing counts with a negative-binomial longitudinal model
   (posterior-predictive, gamma–Poisson draws). *(awaits the NB imputer)*
6. **Refit the GEE** on each completed dataset via the analysis adapter.
7. **Pool with Rubin's rules** (`pool_rubin`, Barnard–Rubin df).
8. **Compare** MI against complete-data, available-case, and IPW estimates.
9. **Delta-adjusted MNAR sensitivity analysis**, plus the
   `mnar_stress_test` mask demonstrating where MAR imputation fails.
10. **Reproduce the workflow in R** and compare pooled results.

## Why epil

- Public and loadable from authoritative upstreams in both languages — no
  data redistribution (MASS is GPL; see the licensing notes in
  [examples/epil_count/README.md](../../examples/epil_count/README.md),
  including the MASS 7.3-65 row-31 erratum).
- Complete source data: induced missingness means the hidden values are
  known, which the motivating application (ORR medication counts) cannot
  offer.
- Already the count-GEE example in the statsmodels documentation, so the
  Python starting point is independently documented.
