"""Base classes and shared machinery for imputation backends.

Backends separate model fitting from imputation generation:

    fit = imputer.fit(data)          # validate, estimate, diagnose once
    completed = fit.impute(m=100, random_state=rng)          # MAR
    shifted = fit.impute(m=100, random_state=rng2,
                         delta=DeltaAdjustment(...))         # MNAR scenario

so sensitivity scenarios reuse the fitted model (and, with equal seeds,
the same underlying randomness — reducing Monte Carlo noise in scenario
differences). ``imputer.impute(data, m, random_state)`` remains as the
one-shot convenience. Fit objects expose ``diagnostics``, ``declaration``,
``model_specification`` and ``data_fingerprint``.
"""

from __future__ import annotations

import abc
import hashlib
from typing import Any

import numpy as np

from .. import __version__ as _longmi_version
from ..contracts import ValidityDeclaration
from ..data import CompletedDatasetCollection, LongitudinalData
from ..scenarios import DeltaAdjustment

__all__ = ["BaseImputer", "BaseFit"]


def normalize_random_state(
    random_state: int | np.random.Generator,
) -> tuple[np.random.Generator, dict[str, Any]]:
    """Accept an integer seed or a Generator; return it with provenance."""
    if isinstance(random_state, (int, np.integer)):
        rng = np.random.default_rng(int(random_state))
        record: dict[str, Any] = {"seed": int(random_state)}
    elif isinstance(random_state, np.random.Generator):
        rng = random_state
        record = {"seed": None}
    else:
        raise TypeError(
            "random_state must be an int seed or numpy.random.Generator, "
            f"got {type(random_state).__name__}"
        )
    record["bit_generator"] = type(rng.bit_generator).__name__
    record["longmi_version"] = _longmi_version
    return rng, record


def data_fingerprint(data: LongitudinalData) -> str:
    """SHA-256 over the validated frame, for result provenance."""
    text = data.frame.to_csv(index=False, float_format="%.17g")
    return hashlib.sha256(text.encode()).hexdigest()


class BaseFit(abc.ABC):
    """A fitted imputation model, reusable across scenarios and draws."""

    declaration: ValidityDeclaration
    model_specification: dict[str, Any]
    data_fingerprint: str
    diagnostics: Any

    @abc.abstractmethod
    def impute(
        self,
        m: int,
        random_state: int | np.random.Generator,
        *,
        delta: DeltaAdjustment | None = None,
    ) -> CompletedDatasetCollection:
        """Produce ``m`` completed datasets; ``delta`` overrides the
        imputer-level default scenario for this run."""


class BaseImputer(abc.ABC):
    """Base class for posterior-predictive imputation backends.

    Subclasses implement :meth:`fit`; each imputation must draw both model
    parameters and missing outcomes (A6) and return a
    :class:`CompletedDatasetCollection`, whose construction certifies
    observed-value preservation and complete filling.
    """

    @property
    @abc.abstractmethod
    def declaration(self) -> ValidityDeclaration:
        """The validity conditions this backend claims (see contracts)."""

    @abc.abstractmethod
    def fit(self, data: LongitudinalData) -> BaseFit:
        """Validate the data, fit the imputation model, run diagnostics."""

    def impute(
        self,
        data: LongitudinalData,
        m: int,
        random_state: int | np.random.Generator,
    ) -> CompletedDatasetCollection:
        """One-shot convenience: ``fit(data).impute(m, random_state)``."""
        return self.fit(data).impute(m, random_state)
