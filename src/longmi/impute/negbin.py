"""Negative-binomial GLMM imputer for longitudinal counts.

Model for participant i at wave j:

    Y_ij | b_i ~ NegBin(mean mu_ij, size kappa)
    log mu_ij = x_ij' beta + b_i,        b_i ~ N(0, tau^2)

with design x_ij = intercept + categorical wave effects + predictors
(+ declared predictor-by-wave interactions). The negative binomial uses the
size parameterization: Var(Y | b) = mu + mu^2 / kappa. Wave order follows
the declared design order, never a re-sort.

Fitting and draws
-----------------
``fit(data)`` maximizes the marginal likelihood (random intercept
integrated by 1-D Gauss-Hermite quadrature) over theta = (beta, log kappa,
log tau) with BFGS, verifies the optimizer outcome (success flag or small
gradient norm — a failed optimizer never quietly generates imputations),
and inverts a central-difference Hessian with tolerance-aware validation:
tiny negative eigenvalues attributable to roundoff are repaired and
recorded; materially indefinite curvature raises. All of this lands in
``NegativeBinomialFitDiagnostics``.

Per imputation, parameters are drawn from the **large-sample normal
approximation to the posterior**, theta^(m) ~ N(theta_hat, H^-1) — a
declared approximation to Proposition 2's exact posterior draw (Rubin 1987
sec. 4.3). Random intercepts are then sampled from a **numerically
normalized grid approximation** to their conditional posterior
p(b_i | y_i^obs, theta^(m)): the grid starts at +/- 8 tau and expands
adaptively until the probability mass in the outermost cells falls below
1e-8, with the realized boundary mass reported in the run metadata
(participants with no observed outcomes draw from N(0, tau^2)). Missing
counts are drawn by the gamma-Poisson representation

    Lambda ~ Gamma(kappa, rate = kappa / mu),   Y ~ Poisson(Lambda),

which guarantees nonnegative-integer imputations. Delta adjustment is
supported on the linear-predictor scale only (mu multiplied by
exp(delta) before the draw) — the model-based pattern-mixture mechanism.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import optimize, special

from ..contracts import ValidityDeclaration
from ..data import CompletedDatasetCollection, LongitudinalData
from ..diagnostics import NegativeBinomialFitDiagnostics
from ..scenarios import DeltaAdjustment
from .base import BaseFit, BaseImputer, data_fingerprint, normalize_random_state

__all__ = ["NegativeBinomialImputer", "NegativeBinomialFit"]

_GH_NODES = 40
_B_POINTS_PER_TAU = 25  # grid density: ~25 points per prior sd
_B_START_RANGE = 8.0  # initial half-width in units of tau
_B_BOUNDARY_TOL = 1e-8
_B_MAX_EXPANSIONS = 8
_GRAD_TOL = 1e-3  # accepted gradient inf-norm when BFGS reports failure
_COV_EIG_RTOL = 1e-8  # repairable-vs-materially-indefinite threshold


def _nb_loglik(y: np.ndarray, log_mu: np.ndarray, kappa: float) -> np.ndarray:
    """Elementwise log NegBin(y | mean exp(log_mu), size kappa).

    Written via logaddexp so extreme linear predictors explored by the
    optimizer cannot overflow exp(log_mu).
    """
    log_kappa_plus_mu = np.logaddexp(np.log(kappa), log_mu)
    return (
        special.gammaln(y + kappa)
        - special.gammaln(kappa)
        - special.gammaln(y + 1.0)
        + kappa * (np.log(kappa) - log_kappa_plus_mu)
        + y * (log_mu - log_kappa_plus_mu)
    )


def _check_delta(delta: DeltaAdjustment | None) -> DeltaAdjustment | None:
    if delta is not None and delta.scale != "linear_predictor":
        raise ValueError(
            "count imputation supports delta adjustment on the "
            "linear-predictor scale only (an outcome-scale shift of a "
            "count is a post-draw transformation, not a model-based "
            "adjustment)"
        )
    return delta


class NegativeBinomialImputer(BaseImputer):
    """Posterior-predictive NB random-intercept imputation for count outcomes.

    Parameters
    ----------
    time_interactions:
        Predictor columns whose wave interactions enter the imputation
        model. Congeniality (A8) requires the analysis model's
        exposure-by-time interaction to be represented here: list the
        exposure column at minimum.
    n_quad:
        Gauss-Hermite nodes for the random-intercept integral. Refit with a
        different value to check quadrature sensitivity; the used value is
        recorded in the fit diagnostics.
    delta:
        Default :class:`DeltaAdjustment` scenario (linear-predictor scale
        only); ``fit(...).impute(..., delta=...)`` overrides per run.
    allow_undeclared_times:
        Backends imputing longitudinal dropout need the declared design
        grid (``LongitudinalData(times=...)``) — absent rows cannot be
        imputed. Pass ``True`` to accept data without declared times.
    """

    def __init__(
        self,
        *,
        time_interactions: Sequence[str] = (),
        n_quad: int = _GH_NODES,
        delta: DeltaAdjustment | None = None,
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

    def _declaration(self, delta: DeltaAdjustment | None) -> ValidityDeclaration:
        return ValidityDeclaration(
            missingness_assumption=(
                "MAR" if delta is None else f"MNAR(delta={delta.delta})"
            ),
            mar_empirically_testable=False,
            sampling_unit="participant",
            parameter_uncertainty_propagated=True,
            outcome_uncertainty_propagated=True,
            observed_outcomes_preserved=True,
            analysis_terms_in_imputation_model=None,  # depends on the analysis
            longitudinal_dependence_modeled=True,  # random participant intercept
            analysis_nested_in_imputation_model=None,
            congeniality_status=(
                "conditionally supported: categorical wave effects plus the "
                "declared predictor-by-wave interactions; analyses needing "
                "other interactions must extend time_interactions"
            ),
            pooling_method="Rubin",
            mnar_sensitivity_performed=delta is not None,
            supported_outcome_types=("count",),
            notes=(
                "Parameter draws use the large-sample normal approximation "
                "to the posterior, N(theta_hat, H^-1) on (beta, log kappa, "
                "log tau) — a declared approximation to an exact posterior "
                "draw. Random intercepts are sampled from a numerically "
                "normalized, adaptively expanded grid approximation to "
                "their conditional posterior; outcomes are gamma-Poisson "
                "draws given theta^(m) and b_i."
            ),
        )

    # -- design -----------------------------------------------------------

    def _design(self, data: LongitudinalData):
        if data.outcome_type != "count":
            raise ValueError(
                "NegativeBinomialImputer supports count outcomes only; "
                f"got outcome_type={data.outcome_type!r}"
            )
        unknown = set(self.time_interactions) - set(data.predictor_cols)
        if unknown:
            raise ValueError(
                f"time_interactions {sorted(unknown)} are not predictor columns"
            )
        if data.times is None and not self.allow_undeclared_times:
            raise ValueError(
                "declare the design grid (LongitudinalData(times=...)) so "
                "absent rows are caught before imputation, or construct the "
                "imputer with allow_undeclared_times=True to treat the "
                "observed times as the full design"
            )
        frame = data.frame
        # declared design order takes precedence over natural sorting
        waves = list(data.times) if data.times is not None else list(
            data.observed_times()
        )
        if len(waves) < 2:
            raise ValueError("need at least 2 waves")

        cols = [np.ones(len(frame))]
        names = ["intercept"]
        for w in waves[1:]:
            cols.append((frame[data.time_col] == w).to_numpy(dtype=float))
            names.append(f"wave[{w}]")
        for col in data.predictor_cols:
            values = pd.to_numeric(frame[col], errors="raise").to_numpy(dtype=float)
            cols.append(values)
            names.append(col)
            if col in self.time_interactions:
                for w in waves[1:]:
                    cols.append(
                        values * (frame[data.time_col] == w).to_numpy(dtype=float)
                    )
                    names.append(f"{col}:wave[{w}]")
        x = np.column_stack(cols)
        if np.linalg.matrix_rank(x) < x.shape[1]:
            raise ValueError("imputation design matrix is rank deficient")

        ids = frame[data.id_col].to_numpy()
        _, id_index = np.unique(ids, return_inverse=True)
        return x, names, id_index, waves

    # -- likelihood -------------------------------------------------------

    def _neg_loglik(self, theta, y, x, id_index, observed, n_ids, nodes, weights):
        p = x.shape[1]
        beta = theta[:p]
        kappa = np.exp(theta[p])
        tau = np.exp(theta[p + 1])
        eta = x @ beta
        total = np.zeros((n_ids, len(nodes)))
        for k, z in enumerate(nodes):
            b = np.sqrt(2.0) * tau * z
            ll = _nb_loglik(y[observed], eta[observed] + b, kappa)
            total[:, k] = np.bincount(id_index[observed], ll, minlength=n_ids)
        log_w = np.log(weights / np.sqrt(np.pi))
        return -float(special.logsumexp(total + log_w, axis=1).sum())

    def _numerical_hessian(self, theta, args, eps=1e-4):
        d = len(theta)
        hessian = np.empty((d, d))
        f = lambda t: self._neg_loglik(t, *args)
        for a in range(d):
            for b in range(a, d):
                pp = theta.copy(); pp[a] += eps; pp[b] += eps
                pm = theta.copy(); pm[a] += eps; pm[b] -= eps
                mp = theta.copy(); mp[a] -= eps; mp[b] += eps
                mm = theta.copy(); mm[a] -= eps; mm[b] -= eps
                hessian[a, b] = hessian[b, a] = (
                    f(pp) - f(pm) - f(mp) + f(mm)
                ) / (4 * eps * eps)
        return hessian

    def fit(self, data: LongitudinalData) -> "NegativeBinomialFit":
        x, names, id_index, waves = self._design(data)
        frame = data.frame
        y = frame[data.outcome_col].to_numpy(dtype=float)
        observed = ~data.missing_mask.to_numpy()
        missing = ~observed
        n_ids = int(id_index.max()) + 1
        y_filled = np.where(observed, y, 0.0)

        nodes, weights = np.polynomial.hermite.hermgauss(self.n_quad)
        obs_y = y_filled[observed]
        pseudo = np.log(obs_y + 0.5)
        beta0, *_ = np.linalg.lstsq(x[observed], pseudo, rcond=None)
        theta0 = np.concatenate([beta0, [np.log(2.0), np.log(0.5)]])
        args = (y_filled, x, id_index, observed, n_ids, nodes, weights)
        result = optimize.minimize(
            self._neg_loglik, theta0, args=args, method="BFGS",
            options={"gtol": 1e-6, "maxiter": 500},
        )
        gradient_norm = float(np.linalg.norm(result.jac, ord=np.inf))
        objective_ok = np.isfinite(result.fun) and np.all(np.isfinite(result.x))
        converged = objective_ok and (
            result.success
            or gradient_norm < _GRAD_TOL * max(1.0, abs(float(result.fun)))
        )
        if not converged:
            raise RuntimeError(
                "NB GLMM fit did not converge: "
                f"success={result.success}, message={result.message!r}, "
                f"gradient inf-norm={gradient_norm:.3g}; refusing to "
                "generate imputations from a failed optimum"
            )

        hessian = self._numerical_hessian(result.x, args)
        cov = np.linalg.inv(hessian)
        cov = 0.5 * (cov + cov.T)
        eigval, eigvec = np.linalg.eigh(cov)
        cov_min_eig = float(eigval.min())
        scale = max(1.0, float(np.max(np.abs(eigval))))
        tolerance = _COV_EIG_RTOL * scale
        if cov_min_eig < -tolerance:
            raise RuntimeError(
                "parameter covariance is materially indefinite "
                f"(min eigenvalue {cov_min_eig:.6g} vs tolerance "
                f"{-tolerance:.6g}); the model may be unidentified or "
                "unconverged"
            )
        repaired = bool(cov_min_eig < tolerance)
        if repaired:
            eigval = np.maximum(eigval, tolerance)
            cov = eigvec @ np.diag(eigval) @ eigvec.T

        diagnostics = NegativeBinomialFitDiagnostics(
            optimizer_success=bool(result.success),
            optimizer_message=str(result.message),
            n_iterations=int(result.nit),
            final_objective=float(result.fun),
            gradient_norm=gradient_norm,
            hessian_eigenvalues=tuple(
                float(v) for v in np.linalg.eigvalsh(hessian)
            ),
            covariance_repaired=repaired,
            covariance_min_eigenvalue=cov_min_eig,
            n_quad=self.n_quad,
        )
        return NegativeBinomialFit(
            self, data, x, names, id_index, waves,
            y_filled, observed, missing, n_ids,
            result.x, cov, diagnostics,
        )


class NegativeBinomialFit(BaseFit):
    """Fitted NB GLMM: reusable across scenarios and imputation runs."""

    def __init__(
        self, imputer, data, x, names, id_index, waves,
        y_filled, observed, missing, n_ids, theta_hat, theta_cov, diagnostics,
    ):
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
        self.model_specification: dict[str, Any] = {
            "backend": "NegativeBinomialImputer",
            "design_terms": list(names),
            "waves": list(waves),
            "time_interactions": list(imputer.time_interactions),
            "random_effects": "participant intercept",
            "n_quad": imputer.n_quad,
            "parameter_draws": "large-sample normal approximation N(theta_hat, H^-1)",
        }

    def _draw_intercepts(self, eta, kappa, tau, rng):
        """Sample b_i from a normalized grid approximation to its
        conditional posterior, expanding the grid until boundary mass is
        below tolerance. Returns (draws, boundary_mass, expansions)."""
        observed, id_index, n_ids = self._observed, self._id_index, self._n_ids
        y, half = self._y_filled, _B_START_RANGE * tau
        expansions = 0
        while True:
            n_points = max(int(2 * half / tau * _B_POINTS_PER_TAU) | 1, 51)
            grid = np.linspace(-half, half, n_points)
            log_post = np.zeros((n_ids, n_points))
            for k, b in enumerate(grid):
                ll = _nb_loglik(y[observed], eta[observed] + b, kappa)
                log_post[:, k] = np.bincount(
                    id_index[observed], ll, minlength=n_ids
                )
            log_post += -0.5 * (grid / tau) ** 2  # N(0, tau^2) prior kernel
            log_post -= log_post.max(axis=1, keepdims=True)
            mass = np.exp(log_post)
            mass /= mass.sum(axis=1, keepdims=True)
            boundary = float((mass[:, 0] + mass[:, -1]).max())
            if boundary <= _B_BOUNDARY_TOL or expansions >= _B_MAX_EXPANSIONS:
                break
            half *= 1.5
            expansions += 1
        if boundary > _B_BOUNDARY_TOL:
            raise RuntimeError(
                "random-intercept grid boundary mass "
                f"{boundary:.3g} still exceeds {_B_BOUNDARY_TOL:g} after "
                f"{expansions} expansions; the conditional posterior is "
                "not contained — inspect the fit"
            )
        cdf = np.cumsum(mass, axis=1)
        cdf /= cdf[:, -1:]
        u = rng.uniform(size=n_ids)
        picks = (cdf < u[:, None]).sum(axis=1)
        b = grid[np.minimum(picks, len(grid) - 1)]
        has_obs = np.bincount(id_index[observed], minlength=n_ids) > 0
        b[~has_obs] = tau * rng.standard_normal(int((~has_obs).sum()))
        return b, boundary, expansions

    def impute(
        self,
        m: int,
        random_state: int | np.random.Generator,
        *,
        delta: DeltaAdjustment | None = None,
    ) -> CompletedDatasetCollection:
        if m < 2:
            raise ValueError("m must be >= 2 for multiple imputation")
        imp = self._imputer
        delta = _check_delta(delta) if delta is not None else imp.delta
        rng, rng_record = normalize_random_state(random_state)
        x, id_index, missing = self._x, self._id_index, self._missing
        p = x.shape[1]

        completed_frames = []
        max_boundary_mass = 0.0
        total_expansions = 0
        kappa_range = [np.inf, -np.inf]
        tau_range = [np.inf, -np.inf]
        for _ in range(m):
            theta = self.theta_hat + self._chol @ rng.standard_normal(
                len(self.theta_hat)
            )
            beta = theta[:p]
            kappa = float(np.exp(theta[p]))
            tau = float(np.exp(theta[p + 1]))
            kappa_range = [min(kappa_range[0], kappa), max(kappa_range[1], kappa)]
            tau_range = [min(tau_range[0], tau), max(tau_range[1], tau)]
            eta = x @ beta
            b, boundary, expansions = self._draw_intercepts(eta, kappa, tau, rng)
            max_boundary_mass = max(max_boundary_mass, boundary)
            total_expansions += expansions
            log_mu = eta[missing] + b[id_index[missing]]
            if delta is not None:
                log_mu = log_mu + delta.delta
            lam = rng.gamma(shape=kappa, scale=np.exp(log_mu) / kappa)
            draws = rng.poisson(lam).astype(float)
            completed_frames.append(self._data.completed_with(draws))

        return CompletedDatasetCollection(
            self._data,
            completed_frames,
            declaration=imp._declaration(delta),
            metadata={
                "imputer": "NegativeBinomialImputer",
                "model_specification": dict(self.model_specification),
                "data_fingerprint": self.data_fingerprint,
                "random_state": rng_record,
                "delta": None if delta is None else delta.delta,
                "fit_diagnostics": self.diagnostics,
                "grid_max_boundary_mass": max_boundary_mass,
                "grid_expansions": total_expansions,
                "kappa_draw_range": tuple(kappa_range),
                "tau_draw_range": tuple(tau_range),
            },
        )
