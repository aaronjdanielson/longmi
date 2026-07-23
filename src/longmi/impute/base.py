"""Abstract base for imputation backends.

Concrete imputers (joint Gaussian reference, negative-binomial GLMM) arrive
in later releases; this base fixes the contract they must satisfy
(Proposition 2 and assumptions A5–A6, A8).
"""

from __future__ import annotations

import abc

import numpy as np

from ..contracts import ValidityDeclaration
from ..data import CompletedDatasetCollection, LongitudinalData

__all__ = ["BaseImputer"]


class BaseImputer(abc.ABC):
    """Base class for posterior-predictive imputation backends.

    Subclasses must draw both model parameters and missing outcomes for each
    imputation (A6) and return a :class:`CompletedDatasetCollection`, whose
    construction certifies observed-value preservation and complete filling.
    """

    @property
    @abc.abstractmethod
    def declaration(self) -> ValidityDeclaration:
        """The validity conditions this backend claims (see contracts)."""

    @abc.abstractmethod
    def impute(
        self,
        data: LongitudinalData,
        m: int,
        random_state: np.random.Generator,
    ) -> CompletedDatasetCollection:
        """Produce ``m`` completed datasets from posterior-predictive draws."""
