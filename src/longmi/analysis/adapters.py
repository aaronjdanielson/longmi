"""Analysis adapters.

The single supported convention is :meth:`AnalysisModel.fit`;
``CallableAnalysis`` wraps a plain function into it so the package never
grows two competing calling conventions. ``StatsmodelsGEE`` adapts a
marginal GEE (statsmodels) — the estimating-equation caveats of
``docs/theory/gee_after_imputation.md`` apply to pooling its output.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import pandas as pd

from ..results import AnalysisEstimate
from .base import BaseAnalysis

__all__ = ["CallableAnalysis", "StatsmodelsGEE", "StatsmodelsGLM"]


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


_GEE_FAMILIES = ("gaussian", "poisson", "binomial", "gamma")
_GEE_COV_STRUCTS = ("independence", "exchangeable", "autoregressive")


class StatsmodelsGLM(BaseAnalysis):
    """GLM via ``statsmodels`` formulas, for analyses of independent rows
    (e.g. a single-wave cross-section of each completed dataset).

    Parameters
    ----------
    formula:
        Patsy formula.
    family:
        One of ``"gaussian" | "poisson" | "binomial" | "gamma"``, or a
        ``statsmodels`` family instance.
    cov_type:
        Covariance estimator (default ``"nonrobust"``, the model-based
        covariance; pass e.g. ``"HC1"`` for heteroskedasticity-robust —
        but for repeated measures use :class:`StatsmodelsGEE`, which
        accounts for clustering).
    subset:
        Optional boolean-returning callable applied to the completed frame
        before fitting (e.g. ``lambda f: f["wave"] == 3``).
    use_dfcom:
        Attach the residual degrees of freedom as ``dfcom`` so pooling
        applies the Barnard-Rubin small-sample reference (default True).
    """

    def __init__(
        self,
        formula: str,
        *,
        family: Any = "gaussian",
        cov_type: str = "nonrobust",
        subset: Callable[[pd.DataFrame], "pd.Series"] | None = None,
        use_dfcom: bool = True,
        fit_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.formula = formula
        self.family = family
        self.cov_type = cov_type
        self.subset = subset
        self.use_dfcom = use_dfcom
        self.fit_kwargs = dict(fit_kwargs or {})

    def _statsmodels(self):
        try:
            import statsmodels.api as sm
            import statsmodels.formula.api as smf
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "StatsmodelsGLM requires statsmodels; "
                "install with `pip install longmi[analysis]`"
            ) from exc
        return sm, smf

    def _family(self, sm):
        if isinstance(self.family, str):
            if self.family not in _GEE_FAMILIES:
                raise ValueError(
                    f"family must be one of {_GEE_FAMILIES} or a statsmodels "
                    f"family instance, got {self.family!r}"
                )
            return {
                "gaussian": sm.families.Gaussian,
                "poisson": sm.families.Poisson,
                "binomial": sm.families.Binomial,
                "gamma": sm.families.Gamma,
            }[self.family]()
        return self.family

    def fit(self, frame: pd.DataFrame) -> AnalysisEstimate:
        sm, smf = self._statsmodels()
        if self.subset is not None:
            frame = frame.loc[self.subset(frame)]
        model = smf.glm(self.formula, data=frame, family=self._family(sm))
        result = model.fit(cov_type=self.cov_type, **self.fit_kwargs)
        return AnalysisEstimate(
            names=tuple(result.params.index),
            estimates=result.params.to_numpy(),
            covariance=result.cov_params().to_numpy(),
            dfcom=float(result.df_resid) if self.use_dfcom else None,
            metadata={
                "adapter": "StatsmodelsGLM",
                "formula": self.formula,
                "family": type(self._family(sm)).__name__,
                "cov_type": self.cov_type,
                "n_obs": int(result.nobs),
            },
        )


class StatsmodelsGEE(BaseAnalysis):
    """Marginal GEE via ``statsmodels`` formulas.

    Fits ``statsmodels.formula.api.gee`` on each completed dataset and
    returns the coefficient vector with its **robust (sandwich)
    covariance** (A7) — the default ``cov_type="robust"`` — under the term
    naming and ordering the formula produces, identically for every
    completed dataset.

    Parameters
    ----------
    formula:
        Patsy formula, e.g.
        ``"y ~ treat * time + C(cohort, Treatment(reference='V2'))"``.
    groups:
        Cluster (participant) column — the independent sampling unit (A1).
    family:
        One of ``"gaussian" | "poisson" | "binomial" | "gamma"``, or a
        ``statsmodels`` family instance.
    cov_struct:
        Working correlation: ``"independence" | "exchangeable" |
        "autoregressive"``, or a ``statsmodels`` covariance-structure
        *class* (a fresh instance is constructed per fit — the objects are
        stateful and must never be shared across completed datasets).
    time:
        Optional within-cluster time column (needed by some structures).
    cov_type:
        Covariance estimator passed to ``fit`` (default ``"robust"``).
    fit_kwargs:
        Extra keyword arguments for ``model.fit`` (e.g. ``maxiter``).

    Notes
    -----
    GEE is method-of-moments: Rubin pooling of its output is supported
    under the congeniality conditions of
    ``docs/theory/gee_after_imputation.md``, not unconditionally. ``dfcom``
    is left ``None`` (large-sample reference), matching GEE's asymptotic
    inference. Requires the optional ``statsmodels`` dependency
    (``pip install longmi[analysis]``).
    """

    def __init__(
        self,
        formula: str,
        groups: str,
        *,
        family: Any = "gaussian",
        cov_struct: Any = "independence",
        time: str | None = None,
        cov_type: str = "robust",
        fit_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.formula = formula
        self.groups = groups
        self.family = family
        self.cov_struct = cov_struct
        self.time = time
        self.cov_type = cov_type
        self.fit_kwargs = dict(fit_kwargs or {})

    def _statsmodels(self):
        try:
            import statsmodels.api as sm
            import statsmodels.formula.api as smf
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "StatsmodelsGEE requires statsmodels; "
                "install with `pip install longmi[analysis]`"
            ) from exc
        return sm, smf

    def _family(self, sm):
        if isinstance(self.family, str):
            if self.family not in _GEE_FAMILIES:
                raise ValueError(
                    f"family must be one of {_GEE_FAMILIES} or a statsmodels "
                    f"family instance, got {self.family!r}"
                )
            return {
                "gaussian": sm.families.Gaussian,
                "poisson": sm.families.Poisson,
                "binomial": sm.families.Binomial,
                "gamma": sm.families.Gamma,
            }[self.family]()
        return self.family

    def _cov_struct(self, sm):
        # always a fresh instance: statsmodels covariance structures carry
        # fitted state and must not leak between completed datasets
        if isinstance(self.cov_struct, str):
            if self.cov_struct not in _GEE_COV_STRUCTS:
                raise ValueError(
                    f"cov_struct must be one of {_GEE_COV_STRUCTS} or a "
                    f"statsmodels covariance-structure class, "
                    f"got {self.cov_struct!r}"
                )
            return {
                "independence": sm.cov_struct.Independence,
                "exchangeable": sm.cov_struct.Exchangeable,
                "autoregressive": sm.cov_struct.Autoregressive,
            }[self.cov_struct]()
        if isinstance(self.cov_struct, type):
            return self.cov_struct()
        raise ValueError(
            "pass cov_struct as a name or a class, not an instance — "
            "instances carry fitted state across completed datasets"
        )

    def fit(self, frame: pd.DataFrame) -> AnalysisEstimate:
        sm, smf = self._statsmodels()
        model = smf.gee(
            self.formula,
            groups=self.groups,
            data=frame,
            family=self._family(sm),
            cov_struct=self._cov_struct(sm),
            time=None if self.time is None else frame[self.time],
        )
        result = model.fit(cov_type=self.cov_type, **self.fit_kwargs)
        return AnalysisEstimate(
            names=tuple(result.params.index),
            estimates=result.params.to_numpy(),
            covariance=result.cov_params().to_numpy(),
            dfcom=None,  # GEE inference is large-sample
            metadata={
                "adapter": "StatsmodelsGEE",
                "formula": self.formula,
                "family": type(self._family(sm)).__name__,
                "cov_struct": type(self._cov_struct(sm)).__name__,
                "cov_type": self.cov_type,
                "n_obs": int(result.nobs),
                "n_clusters": len(model.group_labels),
            },
        )
