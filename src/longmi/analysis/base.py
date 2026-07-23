"""Abstract base for complete-data analysis adapters.

Adapters (``StatsmodelsGEE``, ``StatsmodelsGLM``, ``CallableAnalysis``)
arrive in a later release; this base fixes the contract: an adapter returns
only ``(Q_hat, U, metadata)`` as an :class:`AnalysisEstimate`, with a valid
complete-data covariance (A7) and identical term ordering for every
completed dataset.
"""

from __future__ import annotations

import abc

import pandas as pd

from ..results import AnalysisEstimate

__all__ = ["BaseAnalysis"]


class BaseAnalysis(abc.ABC):
    """Base class for complete-data analysis adapters."""

    @abc.abstractmethod
    def fit(self, frame: pd.DataFrame) -> AnalysisEstimate:
        """Fit the substantive model to one completed dataset."""
