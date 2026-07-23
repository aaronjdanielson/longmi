"""Sensitivity-analysis scenario specifications.

MAR is not empirically testable (A2), so doubt about it is addressed by
delta-adjusted MNAR sensitivity analysis: impute under MAR, then shift the
imputed values (or the linear predictor generating them) by a specified
delta and observe how the pooled inference moves. This module defines the
specification objects only; imputers consume them.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["DeltaAdjustment"]

_SCALES = ("linear_predictor", "outcome")


@dataclass(frozen=True)
class DeltaAdjustment:
    """A single delta-adjustment scenario.

    Attributes
    ----------
    delta:
        The shift applied to imputed values. ``0.0`` reproduces the MAR
        analysis.
    scale:
        ``"linear_predictor"`` shifts on the model's link scale (for a log
        link, ``delta = log(ratio)`` multiplies the imputed mean by
        ``ratio``); ``"outcome"`` shifts the drawn outcome directly (count
        outcomes are truncated at zero and rounded by the imputer).
    label:
        Short name used in reports, e.g. ``"dropout 20% higher"``.
    """

    delta: float
    scale: str = "linear_predictor"
    label: str | None = None

    def __post_init__(self) -> None:
        if self.scale not in _SCALES:
            raise ValueError(f"scale must be one of {_SCALES}, got {self.scale!r}")
        if not float("-inf") < float(self.delta) < float("inf"):
            raise ValueError("delta must be finite")
