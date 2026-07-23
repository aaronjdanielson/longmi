"""Rubin's repeated-imputation combining rules.

Implements the pooled point estimate, total variance
``T = Ubar + (1 + 1/M) B``, relative increase in variance, fraction of
missing information, and degrees of freedom. Sources: Rubin (1987);
Barnard & Rubin (1999). Derivations and exact formulas:
``docs/algorithms/rubin_pooling.md``.

Degrees-of-freedom methods
--------------------------
``"barnard_rubin"`` (default)
    Barnard-Rubin (1999), computed with the same algebraic rearrangement
    ``mice::pool.scalar`` (rule ``"rubin1987"``, mice >= 3.x) uses, so the
    two agree bit-for-bit — including the ``lambda = 0`` boundary: zero
    between-imputation variance yields infinite df (normal reference) and
    ``fmi = 0`` when ``dfcom`` is infinite, and the observed-data df
    ``dfobs`` with finite ``dfcom``. (Historical note: mice versions before
    the rearrangement clamped ``lambda`` below 1e-4; current mice and
    longmi do not.)
``"large_sample"``
    Rubin (1987) large-sample df ``(M - 1) / lambda**2`` even when
    ``dfcom`` is provided (no Barnard-Rubin adjustment).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ..contracts import ValidityDeclaration
from ..results import AnalysisEstimate, RubinPooledResult

__all__ = ["pool_rubin"]

_DF_METHODS = ("barnard_rubin", "large_sample")


def _common_dfcom(estimates: Sequence[AnalysisEstimate]) -> float | None:
    values = {est.dfcom for est in estimates}
    if len(values) > 1:
        raise ValueError(
            f"inconsistent dfcom across completed-data fits: {sorted(values, key=str)}"
        )
    return values.pop()


def _degrees_of_freedom(
    m: int, lam: np.ndarray, dfcom: float | None, df_method: str
) -> np.ndarray:
    with np.errstate(divide="ignore"):
        df_old = np.where(lam > 0, (m - 1) / np.square(lam), np.inf)
    if df_method == "large_sample" or dfcom is None or not np.isfinite(dfcom):
        return df_old
    # Barnard-Rubin via mice's rearrangement, well-defined at lambda = 0
    # (equal to df_old * df_obs / (df_old + df_obs) wherever both exist)
    tmp = (1.0 - lam) * (1.0 + dfcom) * dfcom
    return (m - 1) * tmp / ((dfcom + 3.0) * (m - 1) + np.square(lam) * tmp)


def pool_rubin(
    estimates: Sequence[AnalysisEstimate],
    *,
    df_method: str = "barnard_rubin",
    validity: Mapping[str, Any] | ValidityDeclaration | None = None,
) -> RubinPooledResult:
    """Pool completed-data estimates with Rubin's rules.

    Parameters
    ----------
    estimates:
        One :class:`AnalysisEstimate` per completed dataset, ``M >= 2``. All
        must share the identical parameter name ordering and the same
        ``dfcom``; a permuted ordering is an error, never silently
        realigned. The result is invariant to the order of the list.
    df_method:
        Degrees-of-freedom rule; see the module docstring. The default,
        ``"barnard_rubin"``, is bit-compatible with ``mice::pool.scalar``
        (mice >= 3.x).
    validity:
        Validity-declaration fields to attach for ``validity_report()`` —
        either a mapping or a :class:`ValidityDeclaration` (e.g. carried on
        ``CompletedDatasetCollection.declaration`` from the imputer).

    Notes
    -----
    Between-imputation variance uses the ``M - 1`` denominator. ``fmi`` is
    ``(riv + 2/(df + 3)) / (1 + riv)`` with the reported df (its
    ``df = inf`` limit is ``riv / (1 + riv)``).
    """
    if df_method not in _DF_METHODS:
        raise ValueError(f"df_method must be one of {_DF_METHODS}, got {df_method!r}")
    m = len(estimates)
    if m < 2:
        raise ValueError(f"Rubin pooling requires at least 2 imputations, got {m}")

    first = estimates[0]
    for k, est in enumerate(estimates[1:], start=2):
        if est.names != first.names:
            raise ValueError(
                "parameter names/ordering differ between completed-data fits "
                f"(fit 1: {first.names}, fit {k}: {est.names}); "
                "refusing to pool — completed-data fits must use identical "
                "term ordering"
            )
    dfcom = _common_dfcom(estimates)
    names = first.names

    q = np.stack([est.estimates for est in estimates])  # (m, p)
    u = np.stack([est.covariance for est in estimates])  # (m, p, p)

    qbar = q.mean(axis=0)
    ubar = u.mean(axis=0)
    dev = q - qbar
    b = dev.T @ dev / (m - 1)
    t = ubar + (1.0 + 1.0 / m) * b

    b_diag = np.diag(b)
    ubar_diag = np.diag(ubar)
    t_diag = np.diag(t)
    if np.any(ubar_diag <= 0):
        bad = [names[j] for j in np.flatnonzero(ubar_diag <= 0)]
        raise ValueError(
            f"non-positive within-imputation variance for parameters {bad}"
        )

    extra = (1.0 + 1.0 / m) * b_diag
    riv = extra / ubar_diag
    lam = extra / t_diag

    df = _degrees_of_freedom(m, lam, dfcom, df_method)
    with np.errstate(invalid="ignore"):
        fmi = np.where(
            np.isfinite(df),
            (riv + 2.0 / (df + 3.0)) / (1.0 + riv),
            riv / (1.0 + riv),
        )

    if isinstance(validity, ValidityDeclaration):
        validity = validity.as_dict()
    return RubinPooledResult(
        names=names,
        m=m,
        qbar=qbar,
        ubar=ubar,
        b=b,
        t=t,
        riv=riv,
        lambda_=lam,
        fmi=fmi,
        df=df,
        dfcom=dfcom,
        validity=dict(validity) if validity is not None else None,
    )
