# epil_count — the primary worked example

Longitudinal seizure counts (`MASS::epil`: 59 participants x 4 periods,
Thall & Vail 1990) with reproducible, package-imposed missingness. The
source data are complete, so the complete-data analysis is a benchmark the
missing-data methods can actually be checked against — the values hidden by
our induced dropout are known.

## Data and licensing

The dataset is **never copied into this repository** (MASS is GPL-2/GPL-3).
Each language loads it from its own upstream:

- Python: `statsmodels.datasets.get_rdataset("epil", package="MASS")`
  (Rdatasets mirror, cached locally);
- R: `data("epil", package = "MASS")`, requiring **MASS >= 7.3-65** — that
  version corrected `y` in row 31 (subject 8, period 3) from 21 to 23, and
  older copies silently disagree with the mirror.

Both sides normalize to the same schema and must produce the same SHA-256
over the integer core ([epil_provenance.json](epil_provenance.json)); the R
script refuses to run on a hash mismatch.

## Missingness

[missingness.py](missingness.py) defines three mechanisms and writes shared
masks (`subject, period, observed` — no outcome data) to
`validation/masks/`, so Python and R join the identical pattern instead of
relying on cross-language random-number reproducibility:

- `mcar_light` — 10% independent post-baseline missingness;
- `mar_monotone` — sequential MAR dropout via
  `logit(retention) = gamma0_j + gamma1*log(1+base) + gamma2*age +
  gamma3*treat + gamma4*log(1+y_prev)`; realized retention by period
  1.000 / 0.932 / 0.814 / 0.763 (seed 20260723);
- `mnar_stress_test` — retention additionally depends on the current,
  about-to-be-hidden outcome; realized retention 1.000 / 0.881 / 0.780 /
  0.678. Used to demonstrate that MAR imputation is not guaranteed to
  recover the truth under MNAR.

## Substantive model

Poisson GEE with exchangeable working correlation, clustered by subject,
robust (sandwich) covariance:

    log E(y_ij) = b0 + bT*treat + bP*period + bTP*treat:period
                  + bB*log(1 + base) + bA*age

The central contrast is `treat_period` (`H0: bTP = 0`) — do seizure
trajectories differ by treatment?

## First validation task (implemented)

```bash
python prepare_data.py      # provenance + hash
python missingness.py       # regenerate shared masks
python run_python.py        # complete-data + available-case GEE (statsmodels)
Rscript run_reference.R     # the same, independently, with geepack
python compare_results.py   # exact bookkeeping, 2e-3 tolerance on estimates
```

Current agreement: max |estimate difference| 5.1e-4, max |robust SE
difference| 4.1e-4 across both analyses (different GEE implementations;
deterministic quantities match exactly). `results_python.csv` /
`results_r.csv` are regenerable outputs.

## Planned continuation (with the imputation engine)

Impute the MAR-masked counts with a negative-binomial longitudinal model,
refit the GEE per completed dataset, pool with Rubin's rules, compare MI
against the complete-data benchmark, available-case, and IPW estimates, and
close with a delta-adjusted MNAR sensitivity analysis — the full narrative
in [docs/examples/epilepsy_counts.md](../../docs/examples/epilepsy_counts.md).
