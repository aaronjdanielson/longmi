"""longmi — multiple imputation for incomplete longitudinal outcomes.

Implements established missing-data theory (Rubin 1976, 1987; Barnard–Rubin
1999) with the validity conditions stated as part of the software. See
``docs/theory/mathematical_foundations.md``.
"""

from .analysis import CallableAnalysis
from .contracts import AnalysisModel, Imputer, ValidityDeclaration
from .data import CompletedDatasetCollection, LongitudinalData
from .pooling import pool_rubin
from .results import AnalysisEstimate, RubinPooledResult
from .scenarios import DeltaAdjustment

try:
    from importlib.metadata import PackageNotFoundError, version as _version
    __version__ = _version("longmi")
except PackageNotFoundError:  # running from a checkout without install
    __version__ = "0+unknown"

__all__ = [
    "AnalysisEstimate",
    "AnalysisModel",
    "CallableAnalysis",
    "CompletedDatasetCollection",
    "DeltaAdjustment",
    "Imputer",
    "LongitudinalData",
    "RubinPooledResult",
    "ValidityDeclaration",
    "pool_rubin",
    "__version__",
]
