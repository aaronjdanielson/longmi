"""Analysis adapters.

The single supported convention is :meth:`AnalysisModel.fit`;
``CallableAnalysis`` wraps a plain function into it so the package never
grows two competing calling conventions. Statsmodels adapters
(``StatsmodelsGEE``, ``StatsmodelsGLM``) arrive in a later release.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from ..results import AnalysisEstimate
from .base import BaseAnalysis

__all__ = ["CallableAnalysis"]


class CallableAnalysis(BaseAnalysis):
    """Adapt a plain function ``frame -> AnalysisEstimate`` to the
    :class:`AnalysisModel` interface.

    The wrapped function must return an :class:`AnalysisEstimate` with
    identical parameter naming and ordering for every completed dataset
    (checked again at pooling time).
    """

    def __init__(
        self,
        func: Callable[[pd.DataFrame], AnalysisEstimate],
        *,
        name: str | None = None,
    ) -> None:
        if not callable(func):
            raise TypeError("func must be callable")
        self.func = func
        self.name = name or getattr(func, "__name__", "callable")

    def fit(self, frame: pd.DataFrame) -> AnalysisEstimate:
        estimate = self.func(frame)
        if not isinstance(estimate, AnalysisEstimate):
            raise TypeError(
                f"analysis {self.name!r} returned "
                f"{type(estimate).__name__}, expected AnalysisEstimate"
            )
        return estimate
