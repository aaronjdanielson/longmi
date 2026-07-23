# Preparing data

`LongitudinalData` expects **long format**: one row per participant-time
pair, `NaN` in the outcome column where the response was not measured.

## The rules it enforces

- **One row per (id, time).** Duplicates are rejected with examples.
- **A complete design grid when `times` is declared** (the default): if a
  participant has no row at a scheduled wave, construction fails — an
  absent row is *not* a row with a missing response, and no imputer can
  fill a cell that does not exist. Add rows with `NaN` outcomes first:

  ```python
  full_grid = pd.MultiIndex.from_product(
      [frame["subject"].unique(), (1, 2, 3, 4)], names=["subject", "period"]
  )
  frame = (frame.set_index(["subject", "period"])
                .reindex(full_grid).reset_index())
  # then restore participant-level predictors with a groupby ffill/bfill
  ```

- **Fully observed, finite predictors.** Imputing incomplete covariates
  is outside the 0.1 scope; missing or non-finite predictor values are
  rejected.
- **Count support.** With `outcome_type="count"`, observed outcomes must
  be nonnegative integers, and every completed dataset is re-checked.
- **Declared wave order wins.** Rows are sorted by `(id, time)` using the
  order you declare in `times` — `("baseline", "month_3", "month_12")`
  stays in that order, never alphabetized.

## What longmi 0.1 does *not* handle

- incomplete predictors or exposures;
- time-varying predictors in the joint Gaussian imputer (participant-level
  only; the NB backend accepts time-varying predictors);
- structural ineligibility (death, withdrawal, administrative censoring):
  currently every participant is assumed eligible at every declared wave.
  If some cells are structurally impossible rather than unmeasured, do
  not add them as `NaN` rows — that would ask the imputer to fill them.
  An eligibility indicator is planned.
