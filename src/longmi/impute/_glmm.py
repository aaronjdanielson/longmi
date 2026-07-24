"""Shared random-intercept GLMM machinery (internal).

Used by the negative-binomial and Bernoulli backends. Everything here is
family-agnostic: a family supplies ``n_extra`` (raw extra parameters
between beta and log tau in theta) and ``row_loglik(y, eta_plus_b,
extra_raw)``. The theta layout convention is ``(beta, *extra, log_tau)``.
Extraction is behavior-preserving: RNG consumption order and arithmetic
match the pre-refactor NB implementation (see
tests/unit/test_glmm_refactor_regression.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, special

B_POINTS_PER_TAU = 25
B_START_RANGE = 8.0
B_BOUNDARY_TOL = 1e-8
B_MAX_EXPANSIONS = 8
GRAD_TOL = 1e-3
COV_EIG_RTOL = 1e-8


def build_design(data, time_interactions):
    """Fixed-effect design: intercept + categorical waves + predictors +
    declared predictor-by-wave interactions. Declared wave order."""
    frame = data.frame
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
        if col in time_interactions:
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


def mixture_negll(theta, y, x, id_index, observed, n_ids, nodes, weights,
                  row_loglik, n_extra):
    """Negative marginal log likelihood, random intercept integrated by GH."""
    p = x.shape[1]
    beta = theta[:p]
    extra = theta[p:p + n_extra]
    tau = np.exp(theta[p + n_extra])
    eta = x @ beta
    total = np.zeros((n_ids, len(nodes)))
    for k, z in enumerate(nodes):
        b = np.sqrt(2.0) * tau * z
        ll = row_loglik(y[observed], eta[observed] + b, extra)
        total[:, k] = np.bincount(id_index[observed], ll, minlength=n_ids)
    log_w = np.log(weights / np.sqrt(np.pi))
    return -float(special.logsumexp(total + log_w, axis=1).sum())


def numerical_hessian(f, theta, eps=1e-4):
    d = len(theta)
    hessian = np.empty((d, d))
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


def fit_marginal_ml(negll, theta0, backend_name):
    """BFGS with verified convergence + tolerance-aware curvature.

    Returns (theta_hat, covariance, info) where info carries the
    optimizer/curvature facts backends put into their diagnostics.
    """
    result = optimize.minimize(
        negll, theta0, method="BFGS", options={"gtol": 1e-6, "maxiter": 500}
    )
    gradient_norm = float(np.linalg.norm(result.jac, ord=np.inf))
    objective_ok = np.isfinite(result.fun) and np.all(np.isfinite(result.x))
    converged = objective_ok and (
        result.success
        or gradient_norm < GRAD_TOL * max(1.0, abs(float(result.fun)))
    )
    if not converged:
        raise RuntimeError(
            f"{backend_name} fit did not converge: "
            f"success={result.success}, message={result.message!r}, "
            f"gradient inf-norm={gradient_norm:.3g}; refusing to "
            "generate imputations from a failed optimum"
        )
    hessian = numerical_hessian(negll, result.x)
    try:
        cov = np.linalg.inv(hessian)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("observed information is singular") from exc
    cov = 0.5 * (cov + cov.T)
    eigval, eigvec = np.linalg.eigh(cov)
    cov_min_eig = float(eigval.min())
    scale = max(1.0, float(np.max(np.abs(eigval))))
    tolerance = COV_EIG_RTOL * scale
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
    info = {
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "n_iterations": int(result.nit),
        "final_objective": float(result.fun),
        "gradient_norm": gradient_norm,
        "hessian_eigenvalues": tuple(
            float(v) for v in np.linalg.eigvalsh(hessian)
        ),
        "covariance_repaired": repaired,
        "covariance_min_eigenvalue": cov_min_eig,
    }
    return result.x, cov, info


def draw_intercepts_grid(y, eta, observed, id_index, n_ids, tau, rng,
                         row_loglik, extra):
    """Sample b_i from a normalized adaptive-grid approximation to its
    conditional posterior; boundary-mass controlled. Returns
    (draws, boundary_mass, expansions)."""
    half = B_START_RANGE * tau
    expansions = 0
    while True:
        n_points = max(int(2 * half / tau * B_POINTS_PER_TAU) | 1, 51)
        grid = np.linspace(-half, half, n_points)
        log_post = np.zeros((n_ids, n_points))
        for k, b in enumerate(grid):
            ll = row_loglik(y[observed], eta[observed] + b, extra)
            log_post[:, k] = np.bincount(
                id_index[observed], ll, minlength=n_ids
            )
        log_post += -0.5 * (grid / tau) ** 2
        log_post -= log_post.max(axis=1, keepdims=True)
        mass = np.exp(log_post)
        mass /= mass.sum(axis=1, keepdims=True)
        boundary = float((mass[:, 0] + mass[:, -1]).max())
        if boundary <= B_BOUNDARY_TOL or expansions >= B_MAX_EXPANSIONS:
            break
        half *= 1.5
        expansions += 1
    if boundary > B_BOUNDARY_TOL:
        raise RuntimeError(
            "random-intercept grid boundary mass "
            f"{boundary:.3g} still exceeds {B_BOUNDARY_TOL:g} after "
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
