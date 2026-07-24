"""Sensitivity-analysis scenario specifications.

MAR is not empirically testable (A2), so doubt about it is addressed by
delta-adjusted MNAR sensitivity analysis: impute under MAR, then shift
imputed values (or the linear predictor generating them) by specified
deltas. Deltas are sensitivity scenarios, never estimated corrections.

Rules may be *targeted*: ``where`` restricts a rule to rows matching
predictor values, ``times`` to specific waves — supporting generic
group-by-wave surfaces delta[g, j] without encoding any specific study.
``DeltaScenario`` combines several non-overlapping rules. Adjustments
apply only to missing responses; every rule and the realized per-row
delta vector are recorded for audit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

if TYPE_CHECKING:
    from .data import LongitudinalData

__all__ = ["DeltaAdjustment", "DeltaScenario"]

_SCALES = ("linear_predictor", "outcome")


@dataclass(frozen=True)
class DeltaAdjustment:
    """A single delta-adjustment rule.

    Attributes
    ----------
    delta:
        The shift applied to targeted missing values. ``0.0`` reproduces
        the MAR analysis.
    scale:
        ``"linear_predictor"`` shifts the model link scale *before* the
        missing value is drawn (log link: ``delta = log(ratio)``
        multiplies the imputed mean by ``ratio``; logit link: ``exp(delta)``
        multiplies the conditional odds, never the probability) — the
        model-based pattern-mixture mechanism and the only scale count and
        binary imputers accept. ``"outcome"`` is a deterministic post-draw
        shift, supported for continuous responses only.
    where:
        Optional mapping ``{predictor_column: required_value}`` restricting
        the rule to matching rows.
    times:
        Optional collection of waves the rule applies to.
    label:
        Short name used in reports.
    """

    delta: float
    scale: str = "linear_predictor"
    where: Mapping[str, Any] | None = None
    times: tuple[Any, ...] | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        if self.scale not in _SCALES:
            raise ValueError(f"scale must be one of {_SCALES}, got {self.scale!r}")
        if not np.isscalar(self.delta) or not np.isfinite(self.delta):
            raise ValueError("delta must be one finite numeric scalar")
        if self.where is not None:
            object.__setattr__(self, "where", dict(self.where))
        if self.times is not None:
            object.__setattr__(self, "times", tuple(self.times))

    @property
    def is_targeted(self) -> bool:
        return self.where is not None or self.times is not None

    def _match_mask(self, data: "LongitudinalData") -> np.ndarray:
        frame = data.frame
        mask = np.ones(len(frame), dtype=bool)
        if self.where is not None:
            for col, value in self.where.items():
                if col not in data.predictor_cols:
                    raise ValueError(
                        f"delta rule references {col!r}, which is not a "
                        "declared predictor column"
                    )
                col_mask = (frame[col] == value).to_numpy()
                if not col_mask.any():
                    raise ValueError(
                        f"delta rule matches no rows: {col!r} == {value!r} "
                        "never occurs"
                    )
                mask &= col_mask
        if self.times is not None:
            known = set(
                data.times if data.times is not None else data.observed_times()
            )
            unknown = set(self.times) - known
            if unknown:
                raise ValueError(
                    f"delta rule references unknown times {sorted(unknown, key=str)}"
                )
            mask &= frame[data.time_col].isin(self.times).to_numpy()
        return mask


@dataclass(frozen=True)
class DeltaScenario:
    """Several non-overlapping targeted delta rules applied together."""

    adjustments: tuple[DeltaAdjustment, ...]
    label: str | None = None

    def __post_init__(self) -> None:
        adjustments = tuple(self.adjustments)
        if not adjustments:
            raise ValueError("DeltaScenario needs at least one adjustment")
        if not all(isinstance(a, DeltaAdjustment) for a in adjustments):
            raise TypeError("adjustments must be DeltaAdjustment objects")
        scales = {a.scale for a in adjustments}
        if len(scales) > 1:
            raise ValueError("all adjustments in a scenario must share a scale")
        object.__setattr__(self, "adjustments", adjustments)

    @property
    def scale(self) -> str:
        return self.adjustments[0].scale


def realized_deltas(
    scenario: "DeltaAdjustment | DeltaScenario | None",
    data: "LongitudinalData",
) -> np.ndarray:
    """Per-missing-row delta vector for a scenario (audit + application).

    Returns an array aligned with the missing rows of ``data`` in stored
    order. Overlapping rules (two rules targeting the same missing row)
    are rejected as ambiguous. Adjustments apply only to missing
    responses by construction.
    """
    n_missing = int(data.missing_mask.sum())
    out = np.zeros(n_missing)
    if scenario is None:
        return out
    rules = (
        (scenario,) if isinstance(scenario, DeltaAdjustment)
        else scenario.adjustments
    )
    missing = data.missing_mask.to_numpy()
    touched = np.zeros(n_missing, dtype=bool)
    for rule in rules:
        rule_rows = rule._match_mask(data)[missing]
        overlap = touched & rule_rows
        if overlap.any():
            raise ValueError(
                "overlapping delta rules target the same missing rows; "
                "make the rules disjoint"
            )
        touched |= rule_rows
        out[rule_rows] += rule.delta
    return out
