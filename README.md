# longmi

longmi creates multiple plausible versions of missing repeated outcomes,
fits the same analysis to each version, and combines the estimates and
uncertainty using Rubin's rules.

It is designed for one incomplete longitudinal outcome and makes the
assumptions behind every result explicit — as part of the result object,
not only in a paper.

```text
Incomplete longitudinal data
            ↓
Fit an imputation model
            ↓
Generate M completed datasets
            ↓
Fit the same analysis M times
            ↓
Pool estimates and uncertainty
            ↓
Report the result and its validity conditions
```

## Installation

```bash
pip install -e ".[analysis]"     # from a checkout; not yet on PyPI
```

## Quickstart

```python
from longmi import LongitudinalData, pool_rubin
from longmi.analysis import StatsmodelsGEE
from longmi.impute import NegativeBinomialImputer

data = LongitudinalData(
    frame,                        # long format, NaN where the outcome is missing
    id_col="subject",
    time_col="period",
    outcome_col="seizures",
    predictor_cols=("treatment", "baseline_seizures", "age"),
    outcome_type="count",
    times=(1, 2, 3, 4),           # declared design grid
)

# treatment-by-wave terms keep the analysis's interaction represented
# in the imputation model (assumption A8)
imputer = NegativeBinomialImputer(time_interactions=("treatment",))

fit = imputer.fit(data)           # validate, estimate, diagnose once
completed = fit.impute(m=50, random_state=20260723)

analysis = StatsmodelsGEE(
    "seizures ~ treatment * period + baseline_seizures + age",
    groups="subject",
    family="poisson",
    cov_struct="exchangeable",
)

result = pool_rubin(completed.analyze(analysis), validity=completed.declaration)
print(result.summary())
print(result.validity_report())
```

The same fitted model reruns under a missing-not-at-random scenario:

```python
from longmi import DeltaAdjustment
import numpy as np

shifted = fit.impute(m=50, random_state=20260723,
                     delta=DeltaAdjustment(np.log(0.8)))  # means x 0.8
```

## What is implemented

- **Data contract** — `LongitudinalData` validates the participant-wave
  grid, preserves observed outcomes bit-for-bit through completion, and
  enforces count support ([src/longmi/data.py](src/longmi/data.py)).
- **Imputers** — `JointGaussianImputer` (continuous outcomes; exact
  conjugate data augmentation, wave-saturated mean, unstructured
  covariance) and `NegativeBinomialImputer` (longitudinal counts; NB
  random intercept, Gauss–Hermite ML with verified convergence,
  large-sample posterior-approximation parameter draws, adaptive-grid
  random-intercept draws, gamma–Poisson outcome draws). Both support
  delta-adjusted MNAR sensitivity analysis and report numerical
  diagnostics; a failed fit refuses to impute.
- **Analyses** — `StatsmodelsGEE` (marginal GEE, robust sandwich,
  verified to reproduce direct statsmodels fits exactly),
  `StatsmodelsGLM`, and `CallableAnalysis` for custom estimators.
- **Pooling** — `pool_rubin`, multivariate, bit-compatible with
  `mice::pool.scalar` (verified at 1e-12 against R), Barnard–Rubin
  degrees of freedom, fraction of missing information, and
  `validity_report()` distinguishing verified properties from declared
  assumptions.

## Validation status

Deterministic components are cross-validated against R (`mice`);
imputation backends have unit, invariant, and numerical-diagnostic tests
plus a seeded simulation suite for bias and confidence-interval coverage
(`pytest -m slow tests/simulation`). See
[docs/project-status.md](docs/project-status.md) for the canonical
per-feature maturity table — the backends are not claimed statistically
validated beyond what that table states.

## Documentation

- [docs/index.md](docs/index.md) — documentation home (build with
  `mkdocs serve` after `pip install -e ".[docs]"`);
- [Why impute the response?](docs/explanation/why_impute_the_response.md)
  and [MCAR, MAR, MNAR](docs/explanation/mcar_mar_mnar.md) — concepts;
- [Choosing an imputer](docs/how_to/choose_an_imputer.md),
  [interpreting diagnostics](docs/how_to/interpret_diagnostics.md),
  [reporting an analysis](docs/how_to/report_an_analysis.md) — task guides;
- [Mathematical foundations and validity conditions](docs/theory/mathematical_foundations.md),
  [assumptions A1–A8](docs/theory/assumptions.md),
  [GEE after imputation](docs/theory/gee_after_imputation.md) — theory;
- [algorithm specifications](docs/algorithms/) — what the software
  actually computes, approximations included;
- [worked examples](docs/examples/) — the epil count example and the
  external validation oracles.

## Development

```bash
pip install -e ".[dev]"
pytest                       # fast suite
pytest -m slow tests/simulation -q   # simulation studies
Rscript validation/r/rubin_reference.R   # regenerate R pooling reference
```

longmi implements established missing-data theory (Rubin 1976, 1987;
Barnard–Rubin 1999; Meng 1994; Wang–Robins 1998); it does not introduce a
new proof of multiple imputation. See [REFERENCES.md](REFERENCES.md) for
sources and [CITATION.cff](CITATION.cff) to cite the software. MIT
license.
