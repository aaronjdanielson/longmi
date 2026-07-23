# Quickstart

An end-to-end MAR analysis of an incomplete longitudinal count outcome:
fit an imputation model once, generate completed datasets, fit the same
GEE to each, pool with Rubin's rules, and read the validity report. This
example is executable (it is run as a test:
`tests/unit/test_quickstart_example.py`).

```python
import numpy as np
import pandas as pd

from longmi import DeltaAdjustment, LongitudinalData, pool_rubin
from longmi.analysis import StatsmodelsGEE
from longmi.impute import NegativeBinomialImputer

# Long format: one row per participant-period; NaN where the outcome
# was not measured. Predictors must be fully observed.
frame = pd.read_csv("seizures_long.csv")   # subject, period, seizures, ...

data = LongitudinalData(
    frame,
    id_col="subject",
    time_col="period",
    outcome_col="seizures",
    predictor_cols=("treatment", "baseline_seizures", "age"),
    outcome_type="count",
    times=(1, 2, 3, 4),        # the declared design grid
)

# The analysis below contains treatment * period, so the imputation
# model must represent that relationship too (assumption A8):
imputer = NegativeBinomialImputer(time_interactions=("treatment",))

fit = imputer.fit(data)        # validates, estimates, runs diagnostics
print(fit.diagnostics)         # refuse-to-impute checks already passed

completed = fit.impute(m=50, random_state=20260723)

analysis = StatsmodelsGEE(
    "seizures ~ treatment * period + baseline_seizures + age",
    groups="subject",
    family="poisson",
    cov_struct="exchangeable",
)

result = pool_rubin(completed.analyze(analysis), validity=completed.declaration)

print(result.summary())          # estimates, SEs, CIs, fmi, df per term
print(result.validity_report())  # assumptions: verified vs declared
```

## MNAR sensitivity in two lines

MAR is an assumption, not a finding
([why?](../explanation/mcar_mar_mnar.md)). Rerun the *same fitted model*
under "dropouts' means are 20% lower than MAR predicts":

```python
lower = fit.impute(m=50, random_state=20260723,
                   delta=DeltaAdjustment(np.log(0.8)))
result_lower = pool_rubin(lower.analyze(analysis), validity=lower.declaration)
```

With equal seeds the two runs share randomness, so differences between
scenarios are the delta's effect, not Monte Carlo noise. See
[running delta sensitivity](../how_to/run_delta_sensitivity.md).

## What just happened

1. `LongitudinalData` validated the participant-wave grid and recorded
   which outcomes are missing (observed values can never be overwritten).
2. `fit()` estimated the NB random-intercept imputation model and refused
   to continue if the optimizer failed or the curvature was indefinite.
3. `impute()` drew model parameters *and* missing outcomes 50 times —
   both sources of uncertainty, per Proposition 2 of the
   [mathematical foundations](../theory/mathematical_foundations.md).
4. The GEE ran once per completed dataset with identical term ordering.
5. `pool_rubin` combined the 50 fits: total variance
   `T = Ubar + (1 + 1/M) B`, Barnard–Rubin degrees of freedom, fraction
   of missing information.

Next: [understanding the output](understanding_the_output.md).
