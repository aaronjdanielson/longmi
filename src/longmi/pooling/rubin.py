"""Rubin's repeated-imputation combining rules.

Implements the pooled point estimate, total variance
``T = Ubar + (1 + 1/M) B``, relative increase in variance, fraction of
missing information, and Barnard–Rubin small-sample degrees of freedom —
numerically matching ``mice::pool.scalar`` (rule ``"rubin1987"``) per
parameter. Sources: Rubin (1987); Barnard & Rubin (1999). Derivations and
the exact formulas are in ``docs/algorithms/rubin_pooling.md``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ..results import AnalysisEstimate, RubinPooledResult

__all__ = ["pool_rubin"]


def _common_dfcom(estimates: Sequence[AnalysisEstimate]) -> float | None:
    values = {est.dfcom for est in estimates}
    if len(values) > 1:
        raise ValueError(
            f"inconsistent dfcom across completed-data fits: {sorted(values, key=str)}"
        )
    return values.pop()


def pool_rubin(
    estimates: Sequence[AnalysisEstimate],
    *,
    validity: Mapping[str, Any] | None = None,
) -> RubinPooledResult:
    """Pool completed-data estimates with Rubin's rules.

    Parameters
    ----------
    estimates:
        One :class:`AnalysisEstimate` per completed dataset, ``M >= 2``. All
        must share the identical parameter name ordering and the same
        ``dfcom``; a permuted ordering is an error, never silently realigned.
    validity:
        Optional validity-declaration fields (see
        :class:`longmi.contracts.ValidityDeclaration`) to attach to the
        result for ``validity_report()``.

    Returns
    -------
    RubinPooledResult

    Notes
    -----
    Between-imputation variance uses the ``M - 1`` denominator. Degrees of
    freedom follow Barnard & Rubin (1999) when ``dfcom`` is finite and
    Rubin's ``(M - 1) / lambda**2`` otherwise; a parameter with zero
    between-imputation variance has ``riv = lambda = fmi = 0`` and infinite
    degrees of freedom. ``mice`` additionally clamps ``lambda`` below 1e-4
    when computing Barnard–Rubin degrees of freedom; `longmi` does not, so
    agreement with ``mice`` is exact only for ``lambda >= 1e-4`` (any
    realistic amount of missing information).
    """
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
    p = first.p

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

    with np.errstate(divide="ignore"):
        df_old = np.where(lam > 0, (m - 1) / np.square(lam), np.inf)
    if dfcom is not None and np.isfinite(dfcom):
        df_obs = (dfcom + 1.0) / (dfcom + 3.0) * dfcom * (1.0 - lam)
        with np.errstate(invalid="ignore"):
            df = np.where(
                np.isfinite(df_old),
                df_old * df_obs / (df_old + df_obs),
                df_obs,
            )
    else:
        df = df_old

    with np.errstate(invalid="ignore"):
        fmi = np.where(
            np.isfinite(df),
            (riv + 2.0 / (df + 3.0)) / (1.0 + riv),
            riv / (1.0 + riv),
        )

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
