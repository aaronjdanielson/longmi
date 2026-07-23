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

from typing import TYPE_CHECKING, Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from .results import AnalysisEstimate

if TYPE_CHECKING:
    from .contracts import AnalysisModel, ValidityDeclaration

__all__ = ["LongitudinalData", "CompletedDatasetCollection"]

_OUTCOME_TYPES = ("continuous", "count")


def _check_outcome_values(values: pd.Series, outcome_type: str, what: str) -> None:
    arr = values.to_numpy(dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{what}: outcome contains non-finite values")
    if outcome_type == "count":
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
        belong to this set, and (with ``require_complete_grid``, the default)
        every participant must have a row at every design time — an absent
        row is not the same thing as a row with a missing response, and no
        imputer can fill a row that does not exist. Add rows with ``NaN``
        outcomes before imputation. 0.1 assumes every participant is
        eligible at every declared time; distinguishing structural
        ineligibility (an ``eligible`` column) is future work.
    require_complete_grid:
        When ``times`` is declared, enforce the complete participant-by-wave
        row grid described above. Opting out is explicit.

    The stored frame is sorted by ``(id, time)`` with a fresh integer index
    — using the *declared* design order of ``times`` when given, natural
    sort otherwise — making row order (and hence imputation-value alignment
    and analysis term ordering) deterministic.
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
        require_complete_grid: bool = True,
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
            times = tuple(times)
            if len(set(times)) != len(times):
                raise ValueError("declared design times must be unique")
            allowed = set(times)
            observed_times = set(data[time_col].unique().tolist())
            unknown = observed_times - allowed
            if unknown:
                raise ValueError(
                    f"times {sorted(unknown, key=str)} not in the declared "
                    f"design times {sorted(allowed, key=str)}"
                )
            if require_complete_grid:
                ids = pd.unique(data[id_col])
                expected = pd.MultiIndex.from_product(
                    [ids, times], names=[id_col, time_col]
                )
                actual = pd.MultiIndex.from_frame(data[[id_col, time_col]])
                missing_pairs = expected.difference(actual)
                if len(missing_pairs):
                    examples = missing_pairs.tolist()[:5]
                    raise ValueError(
                        "the longitudinal grid is incomplete: missing "
                        f"participant-time rows include {examples}; add rows "
                        "with NaN outcomes before imputation (an absent row "
                        "is not a row with a missing response)"
                    )

        for col in predictor_cols:
            if data[col].isna().any():
                raise ValueError(
                    f"predictor column {col!r} contains missing values; "
                    "longmi 0.1 requires fully observed predictors"
                )
            if pd.api.types.is_numeric_dtype(data[col]) and not np.all(
                np.isfinite(data[col].to_numpy(dtype=float))
            ):
                raise ValueError(
                    f"predictor column {col!r} contains non-finite values"
                )

        outcome = pd.to_numeric(data[outcome_col], errors="raise")
        data[outcome_col] = outcome.astype(float)
        observed = data[outcome_col].notna()
        if observed.any():
            _check_outcome_values(
                data.loc[observed, outcome_col], outcome_type, "observed data"
            )

        # declared design order takes precedence over natural sorting
        if times is not None:
            order = {t: k for k, t in enumerate(times)}
            data = (
                data.assign(_longmi_time_order=data[time_col].map(order))
                .sort_values([id_col, "_longmi_time_order"], kind="mergesort")
                .drop(columns="_longmi_time_order")
                .reset_index(drop=True)
            )
        else:
            data = data.sort_values(
                [id_col, time_col], kind="mergesort"
            ).reset_index(drop=True)

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
        """Times present in the data, in design order when declared."""
        if self.times is not None:
            present = set(self._frame[self.time_col].unique().tolist())
            return tuple(t for t in self.times if t in present)
        return tuple(
            self._frame[self.time_col].drop_duplicates().sort_values().tolist()
        )

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

        _check_outcome_values(outcome, self.outcome_type, "completed data")


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
        declaration: "ValidityDeclaration | None" = None,
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
        # the producing imputer's validity declaration, carried so pooling
        # can attach it without the caller reconstructing core metadata
        self.declaration = declaration

    @property
    def m(self) -> int:
        return len(self._frames)

    def __len__(self) -> int:
        return self.m

    def __iter__(self) -> Iterator[pd.DataFrame]:
        return (f.copy() for f in self._frames)

    def __getitem__(self, index: int) -> pd.DataFrame:
        return self._frames[index].copy()

    def analyze(self, model: "AnalysisModel") -> list[AnalysisEstimate]:
        """Fit a complete-data analysis to each completed dataset in order.

        ``model`` is an :class:`longmi.contracts.AnalysisModel`: its
        ``fit(frame)`` is called once per completed dataset and must return
        an :class:`AnalysisEstimate`. Wrap a plain function with
        :class:`longmi.analysis.CallableAnalysis`. The list is ready for
        :func:`longmi.pooling.pool_rubin`.
        """
        estimates: list[AnalysisEstimate] = []
        for index, frame in enumerate(self, start=1):
            estimate = model.fit(frame)
            if not isinstance(estimate, AnalysisEstimate):
                raise TypeError(
                    f"analysis of completed dataset {index} returned "
                    f"{type(estimate).__name__}, expected AnalysisEstimate"
                )
            estimates.append(estimate)
        return estimates
