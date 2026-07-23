# External validation: the longitudinal MI tutorial (simulated CATS)

The R (and Stata) syntax published with "Multiple imputation for
longitudinal data: A tutorial" (*Statistics in Medicine*,
doi:10.1002/sim.10274) is longmi's external methodological oracle. The
actual CATS participant data are not publicly available; the authors
provide a simulated dataset designed to resemble CATS (`CATS_dataL.csv`)
together with the script that generated it (`Simulation_of_data.R`).

Role in longmi's validation strategy:

| Example | Purpose |
| --- | --- |
| `epil` | Real public longitudinal count data; main user tutorial |
| simulated CATS | Methodological parity with the published R tutorial |
| longmi synthetic fixture | Fast deterministic unit and CI tests |

The tutorial repository is treated as an **external pinned checkout**
(no explicit upstream license → nothing is copied or redistributed):
located via `LONGMI_TUTORIAL_REPO`, pinned by commit in
[validation/external_repositories.toml](https://github.com/aaronjdanielson/longmi/blob/main/validation/external_repositories.toml),
verified by
[examples/cats_tutorial_parity/external_repo.py](https://github.com/aaronjdanielson/longmi/blob/main/examples/cats_tutorial_parity/external_repo.py).

**Status:** the external checkout and validation contract (pinning,
required-file checks) are implemented. Cross-language statistical parity
for the imputation distributions and pooled analyses — authors' R
implementation vs longmi on the same data, missingness, imputation
specification, substantive model, and pooling — remains in progress.
Comparisons will be statistical (pooled estimates, standard errors,
fractions of missing information, imputed margins), never draw-for-draw.
