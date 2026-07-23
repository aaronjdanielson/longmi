# MCAR, MAR, and MNAR

Three assumptions about *why* outcomes are missing, in increasing order of
danger. Formal statements live in [assumptions.md](../theory/assumptions.md)
(A2); this page is the plain-language version.

## MCAR — missing completely at random

Missingness is unrelated to anything: a coin flip. Under MCAR even the
available-case analysis is unbiased (just less precise).

> After a lab mix-up, a random 10% of month-3 samples were lost.

## MAR — missing at random

After conditioning on **observed** data — baseline covariates, treatment,
earlier outcomes — missingness does not additionally depend on the value
that is missing.

> Participants with high seizure counts *at the previous visit* are more
> likely to skip the next one. Dropout depends on data we have.

This is the assumption longmi's imputers work under: an imputation model
that conditions on the observed history restores valid inference.

## MNAR — missing not at random

Missingness depends on the missing value itself, even given everything
observed.

> Participants skip a visit *because of how they feel that day* — the very
> outcome that would have been recorded.

No analysis of the observed data alone can fix this without extra
assumptions.

## The uncomfortable fact

**MAR versus MNAR is not testable from the observed data.** The data you
would need to run the test are exactly the data that are missing. Any
software that claims to have "verified MAR" is overreaching — which is why
longmi's `validity_report()` renders the mechanism as `[declared]`, never
`[verified]`.

## What to do about it

1. **Make MAR as plausible as possible**: condition the imputation model
   on everything that predicts both the outcome and dropout (auxiliary
   variables included).
2. **Stress-test the assumption** with
   [delta adjustment](../how_to/run_delta_sensitivity.md): impute under
   MAR, then shift the imputed values ("suppose dropouts were 20% worse
   than MAR predicts") and see whether conclusions survive. A range of
   deltas traces how far reality would have to deviate from MAR before
   the conclusion changes.
3. **Report** the assumed mechanism, the deltas examined, and the
   fraction of missing information — the
   [reporting guide](../how_to/report_an_analysis.md) has the checklist.
