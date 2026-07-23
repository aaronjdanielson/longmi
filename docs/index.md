# longmi

longmi creates multiple plausible versions of missing repeated outcomes,
fits the same analysis to each version, and combines the estimates and
uncertainty using Rubin's rules. It handles one incomplete longitudinal
outcome and makes the assumptions behind every result explicit — as part
of the result object, not only in a paper.

```text
Incomplete longitudinal data
 → fit an imputation model
 → generate M completed datasets
 → fit the same analysis M times
 → pool estimates and uncertainty
 → report the result and its validity conditions
```

## Where to start

**Applied researcher** — *"How do I analyze my incomplete longitudinal
outcome?"*
Start with the [quickstart](getting_started/quickstart.md), then
[preparing data](getting_started/preparing_data.md),
[choosing an imputer](how_to/choose_an_imputer.md), and
[understanding the output](getting_started/understanding_the_output.md).
When you write it up, use the
[reporting guide](how_to/report_an_analysis.md).

**Statistician** — *"Under exactly what assumptions is this valid?"*
Start with [why impute the response?](explanation/why_impute_the_response.md),
then the
[mathematical foundations and validity conditions](theory/mathematical_foundations.md)
(four cited propositions), [assumptions A1–A8](theory/assumptions.md), and
the [estimating-equation qualification](theory/gee_after_imputation.md).
The [algorithm pages](algorithms/posterior_predictive_mi.md) state exactly
what is computed, approximations included.

**Contributor** — *"How do I implement or validate another backend?"*
Read the [posterior-predictive MI contract](algorithms/posterior_predictive_mi.md)
and the two backend pages as worked precedents; validation obligations
(unit invariants, simulation bias/coverage, cross-language parity) are in
[project status](project-status.md).

## The package's stance

longmi implements established missing-data theory — Rubin (1976, 1987),
Barnard–Rubin (1999), Meng (1994), Wang–Robins (1998) — it does not
introduce a new proof of multiple imputation. Its contribution is an
independent, tested implementation whose validity conditions are declared
in software: every pooled result can render a
`validity_report()` distinguishing what was **verified** by construction
from what the analyst **declared**.

Feature maturity is tracked in one place:
[project status](project-status.md).
