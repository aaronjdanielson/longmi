# Understanding the output

`pool_rubin` returns a `RubinPooledResult`. Its two most important
methods:

## `result.summary()`

One row per model term:

| Column | Meaning |
| --- | --- |
| `estimate` | Pooled point estimate: the mean of the M completed-data estimates. |
| `se` | Pooled standard error `sqrt(T)`, where `T = Ubar + (1 + 1/M) B` combines average within-imputation variance and between-imputation variance. |
| `ci_lower`, `ci_upper` | t-reference interval with the Barnard–Rubin degrees of freedom. |
| `riv` | Relative increase in variance due to missingness. |
| `lambda` | Proportion of total variance attributable to missingness. |
| `fmi` | Fraction of missing information for this parameter. High values (say > 0.5) mean the conclusion leans heavily on the imputation model: increase M, strengthen the imputation model with auxiliaries, and emphasize sensitivity analysis. |
| `df` | Barnard–Rubin degrees of freedom (bit-compatible with `mice`). |

Two properties worth internalizing:

- **MI standard errors should be wider than a single filled-in dataset's**
  — the extra width *is* the missing information. A "more precise" result
  from single imputation is not more accurate, it is overconfident.
- **The full covariance matrices** (`result.t`, `result.ubar`,
  `result.b`) are retained, so linear contrasts of pooled coefficients
  are available — not only per-term SEs.

## `result.validity_report()`

Renders the conditions under which the result is valid, tagged by
provenance:

```text
Missingness assumption: MAR [declared]
MAR empirically testable: No [declared]
Independent sampling unit: participant [declared]
Parameter uncertainty propagated: Yes [verified by backend]
Outcome uncertainty propagated: Yes [verified by backend]
Observed outcomes preserved: Yes [verified]
...
Congeniality status: conditionally supported: ... [declared]
Pooling method: Rubin [verified]
```

- **[verified]** — mechanically enforced by longmi's construction checks.
- **[verified by backend]** — enforced by the producing imputer's
  implementation.
- **[declared]** — an assumption. The software cannot prove MAR or
  congeniality; the report exists precisely so those declarations are
  visible next to the numbers. `not declared` means the producer made no
  claim — favorable values are never defaulted.

## Run metadata

`completed.metadata` on the `CompletedDatasetCollection` records the model
specification, data fingerprint, seed / bit generator / package version,
the delta scenario, and backend diagnostics — everything needed for the
[reporting checklist](../how_to/report_an_analysis.md).
