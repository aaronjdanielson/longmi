# Simulation tests

Frequentist performance under known data-generating processes. Seeded
pytest modules marked `slow`, excluded from the default run:

```bash
pytest -m slow tests/simulation -q                       # smoke counts
LONGMI_SIM_REPS=1000 pytest -m slow tests/simulation -q  # archived run
```

[harness.py](harness.py) reports bias (with Monte Carlo SE and
standardized bias), RMSE, empirical SD, mean reported SE (calibration
ratio), empirical 95% coverage, interval width, mean fraction of missing
information, and the **numerical failure rate** — with pre-specified,
Monte-Carlo-aware acceptance gates. Three outcomes are distinguished:
**validated**, **expected failure** (a deliberately violated assumption
visibly bites), and **numerical failure** (an engineering issue, tracked
separately).

## The validation matrix

| Scenario | File | Outcome class |
| --- | --- | --- |
| Correct model, MCAR (Gaussian) | `test_gaussian_coverage.py` | validated |
| Correct model, monotone MAR (Gaussian, NB+GEE) | `test_gaussian_coverage.py`, `test_negbin_coverage.py` | validated |
| Correct model, intermittent (non-monotone) MAR (both) | same two files | validated |
| High-missingness stress, ORR-like attrition (NB) | `test_negbin_coverage.py` | validated, elevated FMI |
| Imputation omits the exposure-by-time interaction (NB; A8) | `test_expected_failures.py` | expected failure (attenuation) |
| MNAR generation analyzed under MAR (NB; A2) | `test_expected_failures.py` | expected failure (bias + undercoverage) |
| Imputation omits a covariate driving outcome and dropout (Gaussian; A5/A8) | `test_expected_failures.py` | expected failure |
| Delta-adjustment response curve (NB) | `test_delta_response.py` | monotone, brackets MAR, magnitude scales with imputed share |

GEE-target truths are calibrated on one very large complete dataset
(marginal vs conditional attenuation under the log link); mean targets use
analytic values.

Remaining extensions (tracked in [project status](../../docs/project-status.md)):
wider grids over sample size, missingness fraction, correlation, and
overdispersion; misspecified dependence structures; and the archived
1000-replicate release report.
