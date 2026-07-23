# cats_tutorial_parity — published-method validation oracle

Methodological parity between longmi and the R implementation published
with "Multiple imputation for longitudinal data: A tutorial" (*Statistics in
Medicine*, doi:10.1002/sim.10274). The tutorial's repository provides a
simulated dataset resembling the CATS cohort (`CATS_dataL.csv`) plus the
authors' full R syntax, letting us run the authors' code and longmi on the
same longitudinal structure, missingness patterns, imputation
specification, substantive model, and pooling procedure.

## Setup

The repository is an **external pinned checkout** — it carries no explicit
license, so it is used locally and never redistributed here.

```bash
export LONGMI_TUTORIAL_REPO="/absolute/path/to/Longitudinal_multiple_imputation_tutorial"
python external_repo.py   # verifies required files and the pinned commit
```

The pinned commit lives in
[validation/external_repositories.toml](../../validation/external_repositories.toml).

## Status

Scaffolding only. The parity runs (`run_python.py`, `run_external_r.py`,
`compare_results.py`) arrive with the first longmi imputation backends;
comparisons will be statistical (pooled estimates, standard errors,
fraction of missing information, imputed-margin summaries), never
draw-for-draw.
