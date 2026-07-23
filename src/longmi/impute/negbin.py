"""Negative-binomial GLMM imputer for longitudinal counts.

Model for participant i at wave j:

    Y_ij | b_i ~ NegBin(mean mu_ij, size kappa)
    log mu_ij = x_ij' beta + b_i,        b_i ~ N(0, tau^2)

with design x_ij = intercept + categorical wave effects + predictors
(+ declared predictor-by-wave interactions). The negative binomial uses the
size parameterization: Var(Y | b) = mu + mu^2 / kappa.

Fitting and draws
-----------------
The marginal likelihood integrates the random intercept by Gauss-Hermite
quadrature (1-D, exact in the limit of nodes). theta = (beta, log kappa,
log tau) is estimated by ML; per imputation, parameters are drawn from the
**large-sample normal approximation to the posterior**,

    theta^(m) ~ N(theta_hat, H^-1),

a declared approximation to Proposition 2's exact posterior draw (Rubin
1987 sec. 4.3 sanctions large-sample draws; the ValidityDeclaration says
so). Then per participant the random intercept is drawn from its exact
conditional posterior p(b_i | y_i^obs, theta^(m)) by numerical
inverse-CDF on a fine grid (participants with no observed outcomes draw
from N(0, tau^2)), and missing counts are drawn by the gamma-Poisson
representation

    Lambda ~ Gamma(kappa, rate = kappa / mu),   Y ~ Poisson(Lambda),

which guarantees nonnegative-integer imputations. Delta adjustment is
supported on the linear-predictor scale only (mu multiplied by
exp(delta) before the draw) — the model-based pattern-mixture mechanism.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import optimize, special

from ..contracts import ValidityDeclaration
from ..data import CompletedDatasetCollection, LongitudinalData
from ..scenarios import DeltaAdjustment
from .base import BaseImputer

__all__ = ["NegativeBinomialImputer"]

_GH_NODES = 40
_B_GRID = 401
_B_RANGE = 8.0  # grid half-width in units of tau


def _nb_loglik(y: np.ndarray, log_mu: np.ndarray, kappa: float) -> np.ndarray:
    """Elementwise log NegBin(y | mean exp(log_mu), size kappa)."""
    mu = np.exp(log_mu)
    return (
        special.gammaln(y + kappa)
        - special.gammaln(kappa)
        - special.gammaln(y + 1.0)
        + kappa * np.log(kappa / (kappa + mu))
        + y * (log_mu - np.log(kappa + mu))
    )


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
        Gauss-Hermite nodes for the random-intercept integral.
    delta:
        Optional :class:`DeltaAdjustment` for MNAR sensitivity.
    """

    def __init__(
        self,
        *,
        time_interactions: Sequence[str] = (),
        n_quad: int = _GH_NODES,
        delta: DeltaAdjustment | None = None,
    ) -> None:
        if n_quad < 10:
            raise ValueError("n_quad must be at least 10")
        if delta is not None and delta.scale != "linear_predictor":
            raise ValueError(
                "count imputation supports delta adjustment on the "
                "linear-predictor scale only (an outcome-scale shift of a "
                "count is a post-draw transformation, not a model-based "
                "adjustment)"
            )
        self.time_interactions = tuple(time_interactions)
        self.n_quad = n_quad
        self.delta = delta

    @property
    def declaration(self) -> ValidityDeclaration:
        return ValidityDeclaration(
            missingness_assumption=(
                "MAR" if self.delta is None else f"MNAR(delta={self.delta.delta})"
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
            mnar_sensitivity_performed=self.delta is not None,
            supported_outcome_types=("count",),
            notes=(
                "Parameter draws use the large-sample normal approximation "
                "to the posterior, N(theta_hat, H^-1) on (beta, log kappa, "
                "log tau) — a declared approximation to an exact posterior "
                "draw. Random intercepts and outcomes are exact conditional "
                "draws given theta^(m)."
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
        frame = data.frame
        waves = sorted(
            data.times
            if data.times is not None
            else frame[data.time_col].unique().tolist()
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
        return x, names, id_index

    # -- likelihood -------------------------------------------------------

    def _neg_loglik(self, theta, y, x, id_index, observed, n_ids, nodes, weights):
        p = x.shape[1]
        beta = theta[:p]
        kappa = np.exp(theta[p])
        tau = np.exp(theta[p + 1])
        eta = x @ beta
        # per-node participant log-likelihood contributions
        total = np.zeros((n_ids, len(nodes)))
        for k, z in enumerate(nodes):
            b = np.sqrt(2.0) * tau * z
            ll = _nb_loglik(y[observed], eta[observed] + b, kappa)
            total[:, k] = np.bincount(id_index[observed], ll, minlength=n_ids)
        log_w = np.log(weights / np.sqrt(np.pi))
        return -float(special.logsumexp(total + log_w, axis=1).sum())

    def _fit(self, y, x, id_index, observed, n_ids):
        nodes, weights = np.polynomial.hermite.hermgauss(self.n_quad)
        obs_y = y[observed]
        # start: Poisson-ish GLM via log(y+0.5) least squares, moderate kappa/tau
        pseudo = np.log(obs_y + 0.5)
        beta0, *_ = np.linalg.lstsq(x[observed], pseudo, rcond=None)
        theta0 = np.concatenate([beta0, [np.log(2.0), np.log(0.5)]])
        args = (y, x, id_index, observed, n_ids, nodes, weights)
        result = optimize.minimize(
            self._neg_loglik, theta0, args=args, method="BFGS",
            options={"gtol": 1e-6, "maxiter": 500},
        )
        if not np.all(np.isfinite(result.x)):
            raise RuntimeError("NB GLMM fit failed to converge")
        hessian = self._numerical_hessian(result.x, args)
        cov = np.linalg.inv(hessian)
        # guard: symmetrize and repair tiny negative eigenvalues
        cov = 0.5 * (cov + cov.T)
        eigval, eigvec = np.linalg.eigh(cov)
        if eigval.min() <= 0:
            eigval = np.clip(eigval, 1e-10, None)
            cov = eigvec @ np.diag(eigval) @ eigvec.T
        return result.x, cov, (nodes, weights)

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

    # -- conditional draws ------------------------------------------------

    @staticmethod
    def _draw_intercepts(y, eta, id_index, observed, n_ids, kappa, tau, rng):
        """Exact draw from p(b_i | y_i^obs, theta) by grid inverse-CDF."""
        grid = np.linspace(-_B_RANGE * tau, _B_RANGE * tau, _B_GRID)
        log_post = np.zeros((n_ids, _B_GRID))
        for k, b in enumerate(grid):
            ll = _nb_loglik(y[observed], eta[observed] + b, kappa)
            log_post[:, k] = np.bincount(id_index[observed], ll, minlength=n_ids)
        log_post += -0.5 * (grid / tau) ** 2  # N(0, tau^2) prior (unnormalized)
        log_post -= log_post.max(axis=1, keepdims=True)
        cdf = np.cumsum(np.exp(log_post), axis=1)
        cdf /= cdf[:, -1:]
        u = rng.uniform(size=n_ids)
        picks = (cdf < u[:, None]).sum(axis=1)
        b = grid[np.minimum(picks, _B_GRID - 1)]
        # participants with no observed outcomes: prior draw
        has_obs = np.bincount(id_index[observed], minlength=n_ids) > 0
        b[~has_obs] = tau * rng.standard_normal((~has_obs).sum())
        return b

    def impute(
        self,
        data: LongitudinalData,
        m: int,
        random_state: np.random.Generator,
    ) -> CompletedDatasetCollection:
        if m < 2:
            raise ValueError("m must be >= 2 for multiple imputation")
        rng = random_state
        x, names, id_index = self._design(data)
        frame = data.frame
        y = frame[data.outcome_col].to_numpy(dtype=float)
        observed = ~data.missing_mask.to_numpy()
        missing = ~observed
        n_ids = int(id_index.max()) + 1
        y_filled = np.where(observed, y, 0.0)

        theta_hat, theta_cov, _ = self._fit(y_filled, x, id_index, observed, n_ids)
        chol = np.linalg.cholesky(theta_cov)
        p = x.shape[1]

        completed_frames = []
        for _ in range(m):
            theta = theta_hat + chol @ rng.standard_normal(len(theta_hat))
            beta = theta[:p]
            kappa = float(np.exp(theta[p]))
            tau = float(np.exp(theta[p + 1]))
            eta = x @ beta
            b = self._draw_intercepts(
                y_filled, eta, id_index, observed, n_ids, kappa, tau, rng
            )
            log_mu = eta[missing] + b[id_index[missing]]
            if self.delta is not None:
                log_mu = log_mu + self.delta.delta
            lam = rng.gamma(shape=kappa, scale=np.exp(log_mu) / kappa)
            draws = rng.poisson(lam).astype(float)
            completed_frames.append(data.completed_with(draws))

        return CompletedDatasetCollection(
            data,
            completed_frames,
            declaration=self.declaration,
            metadata={
                "imputer": "NegativeBinomialImputer",
                "design_terms": names,
                "time_interactions": self.time_interactions,
                "n_quad": self.n_quad,
                "delta": None if self.delta is None else self.delta.delta,
            },
        )
