# Why impute the response?

longmi does not replace each missing outcome with a single best guess. It
draws several plausible values from a model for the missing outcome,
refits the analysis for each completed dataset, and uses variation across
those analyses to represent uncertainty about what was not observed.

That distinction is the whole method:

$$
\text{single deterministic filling} \neq \text{proper multiple imputation}.
$$

Filling each hole once — with a mean, a carried-forward value, a
regression prediction — produces a dataset that *looks* complete, so
standard errors computed from it behave as if nothing had been missing.
The result is overconfidence, sometimes severe. Multiple imputation keeps
the analysis honest by making the imputations disagree with each other
exactly as much as the observed data leave the missing values uncertain.

## What the machinery guarantees — and what it cannot

**Guaranteed by construction** (longmi refuses otherwise):

- observed responses are never changed — imputation only fills cells that
  were actually missing;
- both sources of uncertainty are propagated: the model parameters are
  drawn anew for every imputation, then the missing outcomes are drawn
  given those parameters (Proposition 2 of the
  [mathematical foundations](../theory/mathematical_foundations.md));
- count outcomes stay nonnegative integers; the analysis sees the same
  term ordering in every completed dataset.

**Not guaranteed by any machinery:**

- the result remains **conditional on the imputation model**. If that
  model misses a relationship the analysis needs (an interaction, a
  nonlinear time trend), the pooled estimate inherits the distortion —
  assumption A8;
- **MAR is an assumption, not an empirical finding.** No test of the
  observed data can distinguish MAR from MNAR
  ([MCAR, MAR, and MNAR](mcar_mar_mnar.md)); the honest response to doubt
  is [delta-adjusted sensitivity analysis](../how_to/run_delta_sensitivity.md),
  not a p-value;
- with **heavy attrition**, MI can be formally valid yet strongly
  model-dependent: the fraction of missing information (`fmi` in the
  output) tells you how much of your conclusion rests on the model rather
  than the data. Report it.

## Why not just analyze the available cases?

Available-case GEE is valid under MCAR — missingness unrelated to
anything. Under the far more common MAR (dropout related to *observed*
history, e.g. sicker participants leave), available-case analyses are
biased, and their apparent precision is misleading. An imputation model
that conditions on the observed history restores validity under MAR — at
the price of the model dependence described above, which longmi insists
on surfacing rather than hiding.
