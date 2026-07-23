"""Strictly validated containers for incomplete longitudinal data.

`longmi` 0.1 handles one incomplete longitudinal response with independent
participants (A1), known observation times, and fully observed predictors.
This module enforces every mechanically checkable invariant at construction
time so that imputers and analyses can rely on them:

- one row per (id, time) pair;
- predictors fully observed;
- outcome missingness explicitly tracked;
- count outcomes are nonnegative integers wherever observed;
- completed datasets never overwrite observed values and leave no missing
  outcome behind.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = ["LongitudinalData", "CompletedDatasetCollection"]

_OUTCOME_TYPES = ("continuous", "count")


def _check_count_values(values: pd.Series, what: str) -> None:
    arr = values.to_numpy(dtype=float)
    if np.any(arr < 0):
        raise ValueError(f"{what}: count outcome has negative values")
    if not np.array_equal(arr, np.floor(arr)):
        raise ValueError(f"{what}: count outcome has non-integer values")


class LongitudinalData:
    """Long-format longitudinal data with an incomplete outcome.

    Parameters
    ----------
    frame:
        Long-format data, one row per participant-time observation. Rows for
        which the outcome was not measured carry ``NaN`` in ``outcome_col``.
    id_col, time_col, outcome_col:
        Column names for the participant (cluster) identifier — the
        independent sampling unit (A1) — the known observation time, and the
        response.
    predictor_cols:
        Fully observed predictor columns (exposure, covariates, auxiliaries).
        Missingness in any predictor is an error: imputing incomplete
        predictors is outside the 0.1 scope.
    outcome_type:
        ``"continuous"`` or ``"count"``. Count outcomes must be nonnegative
        integers wherever observed, and completed datasets must keep them so.
    times:
        Optional explicit design times. When given, every observed time must
        belong to this set.

    The stored frame is sorted by ``(id, time)`` with a fresh integer index,
    making row order — and hence imputation-value alignment and analysis term
    ordering — deterministic.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        id_col: str,
        time_col: str,
        outcome_col: str,
        predictor_cols: Sequence[str] = (),
        outcome_type: str = "continuous",
        times: Sequence[Any] | None = None,
    ) -> None:
        if outcome_type not in _OUTCOME_TYPES:
            raise ValueError(
                f"outcome_type must be one of {_OUTCOME_TYPES}, got {outcome_type!r}"
            )
        predictor_cols = tuple(predictor_cols)
        required = [id_col, time_col, outcome_col, *predictor_cols]
        if len(set(required)) != len(required):
            raise ValueError("id/time/outcome/predictor columns must be distinct")
        missing_cols = [c for c in required if c not in frame.columns]
        if missing_cols:
            raise ValueError(f"frame is missing required columns: {missing_cols}")

        data = frame.loc[:, required].copy()

        if data[id_col].isna().any():
            raise ValueError(f"id column {id_col!r} contains missing values")
        if data[time_col].isna().any():
            raise ValueError(f"time column {time_col!r} contains missing values")

        duplicated = data.duplicated(subset=[id_col, time_col])
        if duplicated.any():
            dupes = (
                data.loc[duplicated, [id_col, time_col]]
                .head(5)
                .to_records(index=False)
                .tolist()
            )
            raise ValueError(
                f"duplicate (id, time) rows, e.g. {dupes}; expected one row "
                "per participant-time"
            )

        if times is not None:
            allowed = set(times)
            observed_times = set(data[time_col].unique().tolist())
            unknown = observed_times - allowed
            if unknown:
                raise ValueError(
                    f"times {sorted(unknown, key=str)} not in the declared "
                    f"design times {sorted(allowed, key=str)}"
                )

        for col in predictor_cols:
            if data[col].isna().any():
                raise ValueError(
                    f"predictor column {col!r} contains missing values; "
                    "longmi 0.1 requires fully observed predictors"
                )

        outcome = pd.to_numeric(data[outcome_col], errors="raise")
        data[outcome_col] = outcome.astype(float)
        observed = data[outcome_col].notna()
        if outcome_type == "count" and observed.any():
            _check_count_values(data.loc[observed, outcome_col], "observed data")

        data = data.sort_values([id_col, time_col], kind="mergesort").reset_index(
            drop=True
        )

        self._frame = data
        self.id_col = id_col
        self.time_col = time_col
        self.outcome_col = outcome_col
        self.predictor_cols = predictor_cols
        self.outcome_type = outcome_type
        self.times = tuple(times) if times is not None else None

    # -- basic views ------------------------------------------------------

    @property
    def frame(self) -> pd.DataFrame:
        """A defensive copy of the validated, deterministically ordered data."""
        return self._frame.copy()

    @property
    def missing_mask(self) -> pd.Series:
        """Boolean mask, aligned to ``frame``, True where the outcome is missing."""
        return self._frame[self.outcome_col].isna()

    @property
    def n_rows(self) -> int:
        return len(self._frame)

    @property
    def n_participants(self) -> int:
        return self._frame[self.id_col].nunique()

    @property
    def n_missing(self) -> int:
        return int(self.missing_mask.sum())

    @property
    def is_complete(self) -> bool:
        return self.n_missing == 0

    def observed_times(self) -> tuple[Any, ...]:
        return tuple(sorted(self._frame[self.time_col].unique().tolist(), key=str))

    # -- completion -------------------------------------------------------

    def completed_with(self, imputed_values: Any) -> pd.DataFrame:
        """Return a completed dataset, filling exactly the missing outcomes.

        ``imputed_values`` must have length ``n_missing`` and aligns with the
        missing rows in stored (id, time) order. Observed values are
        untouched by construction, and the result is re-checked with
        :meth:`check_completed` so the invariant cannot silently erode.
        """
        values = np.asarray(imputed_values, dtype=float)
        if values.shape != (self.n_missing,):
            raise ValueError(
                f"expected {self.n_missing} imputed values, got shape {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("imputed values must be finite")

        completed = self._frame.copy()
        completed.loc[self.missing_mask, self.outcome_col] = values
        self.check_completed(completed)
        return completed

    def check_completed(self, completed: pd.DataFrame) -> None:
        """Validate that ``completed`` is a legal completion of this data.

        Raises ``ValueError`` unless: same rows in the same order; id, time
        and predictors unchanged; every outcome present; observed outcomes
        bit-for-bit preserved; count constraints satisfied everywhere.
        """
        required = [self.id_col, self.time_col, self.outcome_col, *self.predictor_cols]
        missing_cols = [c for c in required if c not in completed.columns]
        if missing_cols:
            raise ValueError(f"completed dataset lacks columns: {missing_cols}")
        if len(completed) != self.n_rows:
            raise ValueError(
                f"completed dataset has {len(completed)} rows, expected {self.n_rows}"
            )

        fixed = [self.id_col, self.time_col, *self.predictor_cols]
        for col in fixed:
            if not completed[col].reset_index(drop=True).equals(
                self._frame[col].reset_index(drop=True)
            ):
                raise ValueError(
                    f"completed dataset altered column {col!r}; id, time and "
                    "predictors must be unchanged"
                )

        outcome = pd.to_numeric(
            completed[self.outcome_col], errors="raise"
        ).reset_index(drop=True)
        if outcome.isna().any():
            n = int(outcome.isna().sum())
            raise ValueError(f"completed dataset still has {n} missing outcomes")

        observed = ~self.missing_mask.to_numpy()
        original = self._frame[self.outcome_col].to_numpy()
        if not np.array_equal(outcome.to_numpy()[observed], original[observed]):
            raise ValueError(
                "completed dataset overwrote observed outcome values; "
                "observed values must be preserved exactly"
            )

        if self.outcome_type == "count":
            _check_count_values(outcome, "completed data")


class CompletedDatasetCollection:
    """``M`` completed datasets tied to their source :class:`LongitudinalData`.

    Construction validates every completed frame via
    :meth:`LongitudinalData.check_completed`, so holding a collection is
    itself the certificate that observed values were preserved and every
    eligible missing value was imputed.
    """

    def __init__(
        self,
        source: LongitudinalData,
        frames: Sequence[pd.DataFrame],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        frames = tuple(f.reset_index(drop=True) for f in frames)
        if len(frames) == 0:
            raise ValueError("at least one completed dataset is required")
        for k, frame in enumerate(frames, start=1):
            try:
                source.check_completed(frame)
            except ValueError as exc:
                raise ValueError(f"completed dataset {k}: {exc}") from exc
        self.source = source
        self._frames = frames
        self.metadata = dict(metadata or {})

    @property
    def m(self) -> int:
        return len(self._frames)

    def __len__(self) -> int:
        return self.m

    def __iter__(self) -> Iterator[pd.DataFrame]:
        return (f.copy() for f in self._frames)

    def __getitem__(self, index: int) -> pd.DataFrame:
        return self._frames[index].copy()

    def analyze(self, fit: Callable[[pd.DataFrame], Any]) -> list[Any]:
        """Apply a complete-data analysis to each completed dataset in order.

        ``fit`` is an :class:`longmi.contracts.AnalysisModel`-style callable
        returning an :class:`longmi.results.AnalysisEstimate`; the list is
        ready for :func:`longmi.pooling.pool_rubin`.
        """
        return [fit(frame) for frame in self]
