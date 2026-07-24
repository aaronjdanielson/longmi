# Worked example: epileptic seizure counts

The primary longmi vignette, built on `MASS::epil` (Thall & Vail 1990): a
real, public, overdispersed longitudinal count outcome whose complete
values are known — so every missing-data method in the package can be
checked against the truth it is trying to recover. Implementation lives in
[examples/epil_count/](https://github.com/aaronjdanielson/longmi/blob/main/examples/epil_count/).

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
5. **Impute** missing counts with the negative-binomial GLMM imputer
   (categorical wave, treat-by-wave interactions, gamma–Poisson draws;
   M = 20, seed 20260723). *(done)*
6. **Refit the GEE** on each completed dataset via `StatsmodelsGEE`.
   *(done)*
7. **Pool with Rubin's rules** (`pool_rubin`, validity carried from the
   imputer's declaration). *(done)*
8. **Compare** MI against complete-data, available-case, and IPW-GEE
   estimates (IPW: stabilized sequential-MAR weights, respondent-level
   bootstrap re-estimating the weights per replicate). *(done — see
   `comparison_table.csv`)*
9. **Delta-adjusted MNAR sensitivity analysis** — a seven-point response
   curve (`delta_response.csv`, means x0.7...x1.4). The deltas are
   *sensitivity scenarios, not estimated corrections*: the curve answers
   "how would the estimate move if missing outcomes were systematically
   shifted relative to their MAR predictions?", never "which delta is
   true?". Plus the **MNAR stress test in a single induced-missingness
   realization** (`mi_under_mnar`): MAR imputation applied to
   MNAR-generated missingness. In this single dataset the headline
   interaction changed only modestly under the MNAR mask — a single
   realization is not a validation study; the release-scale simulations
   (see [project status](../project-status.md)) demonstrate the
   systematic bias and lost coverage that arise when MAR is imposed under
   the evaluated MNAR mechanism. The epil example demonstrates *how to
   perform and interpret* the stress test. *(done)*
10. **Reproduce the MI workflow in R** (`run_mi_reference.R`: wide-format
    mice PMM + geepack + mice::pool on the same mask) — a *cross-method
    statistical comparison*, not backend parity: pooled estimates agree
    within 0.5 pooled SEs, SE ratios 0.88-1.21. *(done)*

## Why epil

- Public and loadable from authoritative upstreams in both languages — no
  data redistribution (MASS is GPL; see the licensing notes in
  [examples/epil_count/README.md](https://github.com/aaronjdanielson/longmi/blob/main/examples/epil_count/README.md),
  including the MASS 7.3-65 row-31 erratum).
- Complete source data: induced missingness means the hidden values are
  known, which the motivating application (ORR medication counts) cannot
  offer.
- Already the count-GEE example in the statsmodels documentation, so the
  Python starting point is independently documented.
