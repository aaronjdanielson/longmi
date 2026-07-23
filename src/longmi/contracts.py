"""Interfaces and validity declarations.

Every imputation backend must state, as data, the conditions under which its
output supports standard Rubin pooling (see
``docs/theory/gee_after_imputation.md``). Nothing here performs computation;
these are the contracts the rest of the package is written against.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Mapping, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    from .data import CompletedDatasetCollection, LongitudinalData
    from .results import AnalysisEstimate

__all__ = ["ValidityDeclaration", "Imputer", "AnalysisModel"]

_NOT_DECLARED = "not declared"


def _render(value: Any) -> str:
    if value is None:
        return _NOT_DECLARED
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (tuple, list)):
        return ", ".join(str(v) for v in value) if value else _NOT_DECLARED
    return str(value)


@dataclass(frozen=True)
class ValidityDeclaration:
    """Machine-readable statement of an inference's validity conditions.

    ``None`` always means "not declared" — a producer must opt in to every
    favorable claim explicitly. Labels A1–A8 refer to
    ``docs/theory/assumptions.md``.
    """

    missingness_assumption: str | None = None  # "MCAR" | "MAR" | "MNAR(delta)"
    mar_empirically_testable: bool | None = False
    sampling_unit: str | None = None  # A1, e.g. "participant"
    parameter_uncertainty_propagated: bool | None = None  # A6
    outcome_uncertainty_propagated: bool | None = None  # A6
    observed_outcomes_preserved: bool | None = None
    analysis_terms_in_imputation_model: bool | None = None  # A8
    longitudinal_dependence_modeled: bool | None = None  # A8
    analysis_nested_in_imputation_model: bool | None = None  # A8
    congeniality_status: str | None = None  # e.g. "conditionally supported"
    pooling_method: str | None = None  # e.g. "Rubin"
    mnar_sensitivity_performed: bool | None = None
    supported_outcome_types: tuple[str, ...] = ()
    notes: str | None = None

    # Provenance of each field in longmi's pipeline: "verified" claims are
    # mechanically enforced (CompletedDatasetCollection construction, the
    # pooling call itself); "verified by backend" are enforced by the
    # producing imputer's implementation; "declared" are analyst or backend
    # assertions that longmi cannot prove (MAR is not testable; congeniality
    # is argued, not proved). Rendered so the report never overstates.
    _PROVENANCE = {
        "missingness_assumption": "declared",
        "mar_empirically_testable": "declared",
        "sampling_unit": "declared",
        "parameter_uncertainty_propagated": "verified by backend",
        "outcome_uncertainty_propagated": "verified by backend",
        "observed_outcomes_preserved": "verified",
        "analysis_terms_in_imputation_model": "declared",
        "longitudinal_dependence_modeled": "declared",
        "analysis_nested_in_imputation_model": "declared",
        "congeniality_status": "declared",
        "pooling_method": "verified",
        "mnar_sensitivity_performed": "declared",
        "supported_outcome_types": "declared",
    }

    _LABELS = {
        "missingness_assumption": "Missingness assumption",
        "mar_empirically_testable": "MAR empirically testable",
        "sampling_unit": "Independent sampling unit",
        "parameter_uncertainty_propagated": "Parameter uncertainty propagated",
        "outcome_uncertainty_propagated": "Outcome uncertainty propagated",
        "observed_outcomes_preserved": "Observed outcomes preserved",
        "analysis_terms_in_imputation_model": "Analysis terms included in imputation model",
        "longitudinal_dependence_modeled": "Longitudinal dependence modeled",
        "analysis_nested_in_imputation_model": "Analysis model nested in imputation model",
        "congeniality_status": "Congeniality status",
        "pooling_method": "Pooling method",
        "mnar_sensitivity_performed": "MNAR sensitivity performed",
        "supported_outcome_types": "Supported outcome types",
        "notes": "Notes",
    }

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ValidityDeclaration":
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown validity fields: {sorted(unknown)}")
        prepared = dict(values)
        if "supported_outcome_types" in prepared:
            prepared["supported_outcome_types"] = tuple(
                prepared["supported_outcome_types"]
            )
        return cls(**prepared)

    def as_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def report(self) -> str:
        lines = []
        for name, label in self._LABELS.items():
            value = getattr(self, name)
            if name == "notes" and value is None:
                continue
            rendered = _render(value)
            tag = self._PROVENANCE.get(name)
            if tag and rendered != _NOT_DECLARED:
                rendered = f"{rendered} [{tag}]"
            lines.append(f"{label}: {rendered}")
        return "\n".join(lines)


@runtime_checkable
class Imputer(Protocol):
    """An imputation backend.

    ``impute`` must return a
    :class:`longmi.data.CompletedDatasetCollection`, which enforces that
    observed values are preserved, every eligible missing value is filled,
    and outcome-type constraints hold. ``declaration`` states the validity
    conditions (Propositions 2–4).
    """

    @property
    def declaration(self) -> "ValidityDeclaration": ...

    def impute(
        self,
        data: "LongitudinalData",
        m: int,
        random_state: "np.random.Generator",
    ) -> "CompletedDatasetCollection": ...


@runtime_checkable
class AnalysisModel(Protocol):
    """A complete-data analysis adapter.

    ``fit`` receives one completed long-format dataset and returns only
    ``(Q_hat, U, metadata)`` as an :class:`longmi.results.AnalysisEstimate`,
    with identical parameter naming and ordering for every completed dataset.
    """

    def fit(self, frame: "pd.DataFrame") -> "AnalysisEstimate": ...
