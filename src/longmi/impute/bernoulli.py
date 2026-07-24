"""Bernoulli random-intercept imputer for longitudinal binary outcomes.

Model for participant i at wave j:

    Y_ij | b_i ~ Bernoulli(p_ij),  logit(p_ij) = x_ij' alpha + b_i,
    b_i ~ N(0, tau^2)

Built on the shared GLMM machinery (Gauss-Hermite ML with verified
convergence, tolerance-aware curvature validation, large-sample normal
parameter draws — a declared approximation — and adaptive-grid
random-intercept draws). Missing outcomes are drawn as Bernoulli
variables, never rounded probabilities. Delta adjustment acts on the
LOGIT scale only: exp(delta) multiplies the conditional odds, never the
probability; targeted rules (where/times) are supported.

Separation safeguards: fits are refused when no outcomes are observed,
when all observed outcomes are 0 or all are 1, when the observed design
is rank deficient, or when optimization/curvature checks fail. Very
large coefficients are surfaced in diagnostics rather than silently
producing near-deterministic imputations.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..contracts import ValidityDeclaration
from ..data import CompletedDatasetCollection, LongitudinalData
from ..diagnostics import NegativeBinomialFitDiagnostics as _FitDiagnostics
from ..scenarios import DeltaAdjustment, DeltaScenario, realized_deltas
from . import _glmm
from .base import BaseFit, BaseImputer, data_fingerprint, normalize_random_state

__all__ = ["BernoulliImputer", "BernoulliFit"]

_MAX_ABS_COEF_WARN = 15.0  # |logit| beyond this is near-deterministic


def _bernoulli_loglik(y: np.ndarray, eta: np.ndarray, extra) -> np.ndarray:
    """y * eta - log(1 + exp(eta)), overflow-safe."""
    return y * eta - np.logaddexp(0.0, eta)


def _check_delta(delta):
    if delta is None:
        return None
    scale = delta.scale if not isinstance(delta, DeltaScenario) else delta.scale
    if scale != "linear_predictor":
        raise ValueError(
            "binary imputation supports delta adjustment on the logit "
            "(linear-predictor) scale only: exp(delta) multiplies the "
            "conditional odds, never the probability"
        )
    return delta


class BernoulliImputer(BaseImputer):
    """Posterior-predictive Bernoulli random-intercept imputation.

    Parameters mirror :class:`NegativeBinomialImputer`:
    ``time_interactions`` (list the analysis exposure at minimum — A8),
    ``n_quad``, a default ``delta`` (logit scale; targeted rules and
    :class:`DeltaScenario` supported), and ``allow_undeclared_times``.
    """

    def __init__(
        self,
        *,
        time_interactions: Sequence[str] = (),
        n_quad: int = 25,
        delta=None,
        allow_undeclared_times: bool = False,
    ) -> None:
        if n_quad < 10:
            raise ValueError("n_quad must be at least 10")
        self.time_interactions = tuple(time_interactions)
        self.n_quad = n_quad
        self.delta = _check_delta(delta)
        self.allow_undeclared_times = allow_undeclared_times

    @property
    def declaration(self) -> ValidityDeclaration:
        return self._declaration(self.delta)

    def _declaration(self, delta) -> ValidityDeclaration:
        return ValidityDeclaration(
            missingness_assumption=(
                "MAR" if delta is None else "MNAR(delta)"
            ),
            mar_empirically_testable=False,
            sampling_unit="participant",
            parameter_uncertainty_propagated=True,
            outcome_uncertainty_propagated=True,
            observed_outcomes_preserved=True,
            analysis_terms_in_imputation_model=None,
            longitudinal_dependence_modeled=True,
            analysis_nested_in_imputation_model=False,
            congeniality_status=(
                "conditionally supported: the imputer is subject-specific "
                "(conditional logit) while marginal logistic GEE targets a "
                "different coefficient scale; the MI-to-GEE workflow is "
                "evaluated by simulation, not assumed congenial"
            ),
            pooling_method="Rubin",
            mnar_sensitivity_performed=delta is not None,
            supported_outcome_types=("binary",),
            notes=(
                "Parameter draws use the large-sample normal approximation "
                "N(theta_hat, H^-1) on (alpha, log tau). Random intercepts "
                "use the adaptive-grid conditional approximation; outcomes "
                "are Bernoulli draws from expit(eta + b), never rounded "
                "probabilities. Delta multiplies conditional odds."
            ),
        )

    def fit(self, data: LongitudinalData) -> "BernoulliFit":
        if data.outcome_type != "binary":
            raise ValueError(
                "BernoulliImputer supports binary outcomes only; "
                f"got outcome_type={data.outcome_type!r}"
            )
        unknown = set(self.time_interactions) - set(data.predictor_cols)
        if unknown:
            raise ValueError(
                f"time_interactions {sorted(unknown)} are not predictor columns"
            )
        if data.times is None and not self.allow_undeclared_times:
            raise ValueError(
                "declare the design grid (LongitudinalData(times=...)) or "
                "construct the imputer with allow_undeclared_times=True"
            )
        x, names, id_index, waves = _glmm.build_design(
            data, self.time_interactions
        )
        frame = data.frame
        y = frame[data.outcome_col].to_numpy(dtype=float)
        observed = ~data.missing_mask.to_numpy()
        n_ids = int(id_index.max()) + 1
        y_filled = np.where(observed, y, 0.0)

        # separation and identifiability safeguards
        if not observed.any():
            raise ValueError(
                "Bernoulli imputation requires at least one observed outcome"
            )
        n_events = int(y_filled[observed].sum())
        n_obs = int(observed.sum())
        if n_events == 0 or n_events == n_obs:
            raise ValueError(
                "all observed outcomes are "
                f"{'1' if n_events else '0'}; the logistic model is not "
                "identified (complete separation at the margin)"
            )
        if np.linalg.matrix_rank(x[observed]) < x.shape[1]:
            raise ValueError(
                "the observed-data imputation design is rank deficient; "
                "one or more fixed effects are not identified"
            )

        nodes, weights = np.polynomial.hermite.hermgauss(self.n_quad)
        # stable start: working-logit least squares
        p_bar = np.clip((y_filled[observed].mean()), 0.05, 0.95)
        z0 = np.log(p_bar / (1 - p_bar))
        alpha0, *_ = np.linalg.lstsq(
            x[observed],
            np.where(y_filled[observed] > 0, z0 + 1.0, z0 - 1.0),
            rcond=None,
        )
        theta0 = np.concatenate([alpha0, [np.log(0.5)]])

        def negll(theta):
            return _glmm.mixture_negll(
                theta, y_filled, x, id_index, observed, n_ids, nodes,
                weights, row_loglik=_bernoulli_loglik, n_extra=0,
            )

        theta_hat, cov, info = _glmm.fit_marginal_ml(
            negll, theta0, "Bernoulli GLMM"
        )
        diagnostics = _FitDiagnostics(n_quad=self.n_quad, **info)
        return BernoulliFit(
            self, data, x, names, id_index, waves, y_filled, observed,
            ~observed, n_ids, theta_hat, cov, diagnostics,
            n_events=n_events, n_obs=n_obs,
        )


class BernoulliFit(BaseFit):
    """Fitted Bernoulli GLMM: reusable across scenarios and runs."""

    def __init__(self, imputer, data, x, names, id_index, waves, y_filled,
                 observed, missing, n_ids, theta_hat, theta_cov, diagnostics,
                 n_events, n_obs):
        self._imputer = imputer
        self._data = data
        self._x = x
        self._id_index = id_index
        self._y_filled = y_filled
        self._observed = observed
        self._missing = missing
        self._n_ids = n_ids
        self.theta_hat = theta_hat
        self.theta_cov = theta_cov
        self._chol = np.linalg.cholesky(theta_cov)
        self.diagnostics = diagnostics
        self.declaration = imputer.declaration
        self.data_fingerprint = data_fingerprint(data)
        max_coef = float(np.max(np.abs(theta_hat[:-1])))
        self.model_specification: dict[str, Any] = {
            "backend": "BernoulliImputer",
            "design_terms": list(names),
            "waves": list(waves),
            "time_interactions": list(imputer.time_interactions),
            "random_effects": "participant intercept",
            "n_quad": imputer.n_quad,
            "observed_events": n_events,
            "observed_nonevents": n_obs - n_events,
            "max_abs_coefficient": max_coef,
            "parameter_draws": "large-sample normal approximation",
        }
        if max_coef > _MAX_ABS_COEF_WARN:
            import warnings

            warnings.warn(
                f"maximum |coefficient| = {max_coef:.1f} exceeds "
                f"{_MAX_ABS_COEF_WARN}: probable quasi-separation; "
                "imputations may be near-deterministic",
                stacklevel=2,
            )

    def impute(
        self,
        m: int,
        random_state,
        *,
        delta=None,
    ) -> CompletedDatasetCollection:
        if m < 2:
            raise ValueError("m must be >= 2 for multiple imputation")
        imp = self._imputer
        delta = _check_delta(delta) if delta is not None else imp.delta
        row_deltas = realized_deltas(delta, self._data)
        rng, rng_record = normalize_random_state(random_state)
        x, id_index, missing = self._x, self._id_index, self._missing
        p = x.shape[1]

        completed_frames = []
        max_boundary = 0.0
        rejected = 0
        for _ in range(m):
            for _attempt in range(50):
                theta = self.theta_hat + self._chol @ rng.standard_normal(
                    len(self.theta_hat)
                )
                tau = float(np.exp(theta[-1]))
                if np.isfinite(tau) and tau > 0 and np.isfinite(
                    x @ theta[:p]
                ).all():
                    break
                rejected += 1
            else:
                raise RuntimeError(
                    "could not generate a numerically valid parameter draw"
                )
            alpha = theta[:p]
            eta = x @ alpha
            b, boundary, _exp = _glmm.draw_intercepts_grid(
                self._y_filled, eta, self._observed, id_index, self._n_ids,
                tau, rng, row_loglik=_bernoulli_loglik, extra=(),
            )
            max_boundary = max(max_boundary, boundary)
            from scipy.special import expit

            logit = eta[missing] + b[id_index[missing]] + row_deltas
            prob = expit(logit)
            draws = (rng.uniform(size=missing.sum()) < prob).astype(float)
            completed_frames.append(self._data.completed_with(draws))

        return CompletedDatasetCollection(
            self._data,
            completed_frames,
            declaration=imp._declaration(delta),
            metadata={
                "imputer": "BernoulliImputer",
                "model_specification": dict(self.model_specification),
                "data_fingerprint": self.data_fingerprint,
                "random_state": rng_record,
                "delta": delta,
                "realized_deltas": row_deltas,
                "fit_diagnostics": self.diagnostics,
                "grid_max_boundary_mass": max_boundary,
                "rejected_parameter_draws": rejected,
            },
        )
