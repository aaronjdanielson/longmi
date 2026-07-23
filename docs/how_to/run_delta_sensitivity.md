# Run MNAR delta sensitivity

MAR is not testable ([why](../explanation/mcar_mar_mnar.md)), so doubt
about it is addressed by *delta adjustment*: impute under MAR, shift the
imputed values by a specified departure, and see how the pooled inference
moves.

## Pattern

```python
import numpy as np
from longmi import DeltaAdjustment, pool_rubin

fit = imputer.fit(data)          # fit ONCE

scenarios = {
    "mar":        None,
    "means_x0.8": DeltaAdjustment(np.log(0.8), label="dropouts 20% lower"),
    "means_x1.25": DeltaAdjustment(np.log(1.25), label="dropouts 25% higher"),
}

results = {}
for name, delta in scenarios.items():
    completed = fit.impute(m=50, random_state=20260723, delta=delta)
    results[name] = pool_rubin(
        completed.analyze(analysis), validity=completed.declaration
    )
```

Three details that matter:

- **Reuse the fit.** Scenarios share one fitted imputation model; nothing
  is re-estimated, so runs are cheap and comparable.
- **Reuse the seed.** With equal `random_state`, scenario runs share the
  same underlying draws — differences between scenarios are the delta's
  effect, with Monte Carlo noise cancelled in the comparison.
- **The scale is the link scale.** For log-link count models,
  `delta = log(r)` multiplies the imputed means by `r` *before* the
  counts are drawn (the model-based pattern-mixture adjustment).
  Outcome-scale shifts are only available for continuous responses; for
  counts they are refused as a non-model-based transformation.

## Choosing deltas

Anchor them substantively: "dropouts take 20% fewer medications than MAR
predicts" is a defensible sentence; an arbitrary ±0.3 is not. Run a
symmetric range (e.g. ×0.8, ×0.9, MAR, ×1.1, ×1.25) and report where —
if anywhere — the conclusion changes sign or loses significance. The
result's declaration records the scenario
(`Missingness assumption: MNAR(delta=...)`,
`MNAR sensitivity performed: Yes`).

A worked example (epil seizure counts, deltas ×0.8 / ×1.25) lives in
[examples/epilepsy_counts.md](../examples/epilepsy_counts.md).
