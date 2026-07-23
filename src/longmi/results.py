"""Typed result containers for completed-data estimates and pooled inference.

The mathematical definitions implemented here are stated in
``docs/theory/mathematical_foundations.md`` (Proposition 3) and
``docs/algorithms/rubin_pooling.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["AnalysisEstimate", "RubinPooledResult"]

_SYMMETRY_RTOL = 1e-8
_PSD_RTOL = 1e-10


def _freeze(arr: np.ndarray) -> np.ndarray:
    """Own a copy and make it read-only: frozen dataclasses do not prevent
    in-place mutation of array contents."""
    out = np.array(arr, dtype=float, copy=True)
    out.setflags(write=False)
    return out


def _as_vector(values: Any, p: int, what: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape != (p,):
        raise ValueError(f"{what} must have shape ({p},), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{what} contains non-finite values")
    return _freeze(arr)


def _as_covariance(values: Any, p: int, what: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape != (p, p):
        raise ValueError(f"{what} must have shape ({p}, {p}), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{what} contains non-finite values")
    scale = np.max(np.abs(arr)) or 1.0
    if not np.allclose(arr, arr.T, rtol=0.0, atol=_SYMMETRY_RTOL * scale):
        raise ValueError(f"{what} is not symmetric")
    sym = 0.5 * (arr + arr.T)
    eig_scale = max(1.0, float(np.linalg.norm(sym, ord=2)))
    min_eigenvalue = float(np.linalg.eigvalsh(sym).min())
    if min_eigenvalue < -_PSD_RTOL * eig_scale:
        raise ValueError(
            f"{what} is not positive semidefinite; "
            f"minimum eigenvalue={min_eigenvalue:.6g}"
        )
    return _freeze(sym)


@dataclass(frozen=True)
class AnalysisEstimate:
    """One completed-data fit: ``(Q_hat, U, metadata)``.

    Attributes
    ----------
    names:
        Parameter names, in the order of ``estimates``. Pooling requires the
        exact same ordering across all completed-data fits (A8 bookkeeping).
    estimates:
        Point estimates ``Q_hat`` of shape ``(p,)``.
    covariance:
        Complete-data covariance ``U`` of shape ``(p, p)``. For GEE this must
        be the robust sandwich covariance (A7).
    dfcom:
        Complete-data degrees of freedom, if a finite-sample reference is
        appropriate; ``None`` means the large-sample (infinite) reference.
    metadata:
        Free-form provenance (model description, software, options).
    """

    names: tuple[str, ...]
    estimates: np.ndarray
    covariance: np.ndarray
    dfcom: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = tuple(str(n) for n in self.names)
        if len(names) == 0:
            raise ValueError("names must be non-empty")
        if len(set(names)) != len(names):
            raise ValueError("names must be unique")
        p = len(names)
        object.__setattr__(self, "names", names)
        object.__setattr__(
            self, "estimates", _as_vector(self.estimates, p, "estimates")
        )
        object.__setattr__(
            self, "covariance", _as_covariance(self.covariance, p, "covariance")
        )
        if self.dfcom is not None:
            dfcom = float(self.dfcom)
            if not dfcom > 0:
                raise ValueError("dfcom must be positive")
            object.__setattr__(self, "dfcom", dfcom)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def p(self) -> int:
        return len(self.names)


@dataclass(frozen=True)
class RubinPooledResult:
    """Pooled repeated-imputation inference (Rubin 1987; Barnard–Rubin 1999).

    All per-parameter arrays are aligned with ``names``. Matrices ``ubar``,
    ``b`` and ``t`` are full ``(p, p)`` covariance matrices; the per-parameter
    quantities (``riv``, ``lambda_``, ``fmi``, ``df``) are computed from their
    diagonals exactly as in ``mice::pool.scalar`` (rule ``rubin1987``).
    """

    names: tuple[str, ...]
    m: int
    qbar: np.ndarray
    ubar: np.ndarray
    b: np.ndarray
    t: np.ndarray
    riv: np.ndarray
    lambda_: np.ndarray
    fmi: np.ndarray
    df: np.ndarray
    dfcom: float | None = None
    validity: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        names = tuple(str(n) for n in self.names)
        if len(names) == 0 or len(set(names)) != len(names):
            raise ValueError("names must be non-empty and unique")
        object.__setattr__(self, "names", names)
        p = len(names)
        if int(self.m) < 2:
            raise ValueError(f"m must be >= 2, got {self.m}")
        object.__setattr__(self, "m", int(self.m))

        object.__setattr__(self, "qbar", _as_vector(self.qbar, p, "qbar"))
        for what in ("ubar", "b", "t"):
            object.__setattr__(
                self, what, _as_covariance(getattr(self, what), p, what)
            )
        scale = max(1.0, float(np.max(np.abs(self.t))))
        expected_t = self.ubar + (1.0 + 1.0 / self.m) * self.b
        if not np.allclose(self.t, expected_t, rtol=1e-10, atol=1e-12 * scale):
            raise ValueError("t must equal ubar + (1 + 1/m) b")

        for what, low, high in (
            ("riv", 0.0, np.inf),
            ("lambda_", 0.0, 1.0),
            ("fmi", 0.0, 1.0),
        ):
            arr = _as_vector(getattr(self, what), p, what)
            if np.any(arr < low) or np.any(arr > high):
                raise ValueError(f"{what} must lie in [{low}, {high}]")
            object.__setattr__(self, what, arr)

        df = np.asarray(self.df, dtype=float)
        if df.shape != (p,):
            raise ValueError(f"df must have shape ({p},), got {df.shape}")
        if np.any(np.isnan(df)) or np.any(df <= 0):
            raise ValueError("df must be positive (infinity allowed)")
        object.__setattr__(self, "df", _freeze(df))

        if self.dfcom is not None and not float(self.dfcom) > 0:
            raise ValueError("dfcom must be positive")
        if self.validity is not None:
            object.__setattr__(
                self, "validity", MappingProxyType(dict(self.validity))
            )

    @property
    def p(self) -> int:
        return len(self.names)

    @property
    def se(self) -> np.ndarray:
        """Pooled standard errors ``sqrt(diag(T))``."""
        return np.sqrt(np.diag(self.t))

    def conf_int(self, level: float = 0.95) -> np.ndarray:
        """Per-parameter confidence intervals on the t reference with ``df``.

        Infinite degrees of freedom fall back to the normal reference.
        Returns an array of shape ``(p, 2)``.
        """
        if not 0.0 < level < 1.0:
            raise ValueError("level must be in (0, 1)")
        alpha = 1.0 - level
        crit = np.where(
            np.isfinite(self.df),
            stats.t.ppf(1.0 - alpha / 2.0, np.where(np.isfinite(self.df), self.df, 1.0)),
            stats.norm.ppf(1.0 - alpha / 2.0),
        )
        half = crit * self.se
        return np.column_stack([self.qbar - half, self.qbar + half])

    def summary(self, level: float = 0.95) -> pd.DataFrame:
        """Per-parameter summary table."""
        ci = self.conf_int(level)
        return pd.DataFrame(
            {
                "estimate": self.qbar,
                "se": self.se,
                "ci_lower": ci[:, 0],
                "ci_upper": ci[:, 1],
                "riv": self.riv,
                "lambda": self.lambda_,
                "fmi": self.fmi,
                "df": self.df,
            },
            index=pd.Index(self.names, name="parameter"),
        )

    def validity_report(self) -> str:
        """Render the validity declaration attached to this result.

        Fields never silently default to a favorable value: anything the
        producing imputer or workflow did not declare renders as
        ``not declared``.
        """
        from .contracts import ValidityDeclaration

        declared = dict(self.validity or {})
        declared.setdefault("pooling_method", "Rubin")
        return ValidityDeclaration.from_mapping(declared).report()
