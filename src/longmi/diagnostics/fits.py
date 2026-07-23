"""Diagnostic records attached to imputation fits and runs.

These are plain data: the numerical facts a user needs to judge whether a
fit and its draws can be trusted. Backends refuse to impute from fits whose
diagnostics indicate failure (optimizer non-convergence, materially
indefinite curvature); everything softer is reported here rather than
hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["NegativeBinomialFitDiagnostics", "GaussianChainDiagnostics"]


@dataclass(frozen=True)
class NegativeBinomialFitDiagnostics:
    """Numerical diagnostics for a :class:`NegativeBinomialFit`.

    Attributes
    ----------
    optimizer_success, optimizer_message, n_iterations, final_objective,
    gradient_norm:
        BFGS outcome for the marginal-likelihood maximization; the fit is
        refused unless the optimizer succeeded or the gradient norm is
        small.
    hessian_eigenvalues:
        Eigenvalues of the observed-information matrix (ascending).
    covariance_repaired, covariance_min_eigenvalue:
        Whether tiny negative eigenvalues of the parameter covariance were
        clipped, and the pre-repair minimum. Materially indefinite
        curvature raises instead of being repaired.
    n_quad:
        Gauss-Hermite nodes used.
    """

    optimizer_success: bool
    optimizer_message: str
    n_iterations: int
    final_objective: float
    gradient_norm: float
    hessian_eigenvalues: tuple[float, ...]
    covariance_repaired: bool
    covariance_min_eigenvalue: float
    n_quad: int


@dataclass(frozen=True)
class GaussianChainDiagnostics:
    """Single-chain diagnostics for one :meth:`JointGaussianFit.impute` run.

    The data-augmentation sampler is one chain; kept imputations are
    separated by ``thin`` sweeps after ``burn_in``. The trace statistic is
    ``log det(Sigma)`` per sweep.

    Attributes
    ----------
    n_sweeps, burn_in, thin, m:
        Chain geometry for this run.
    trace_lag1_autocorrelation:
        Lag-1 autocorrelation of the post-burn-in trace (per sweep).
    trace_ess:
        Crude effective sample size of the post-burn-in trace,
        ``n * (1 - rho) / (1 + rho)`` with the lag-1 autocorrelation —
        a single-chain heuristic, not a substitute for multi-chain R-hat.
    kept_lag1_autocorrelation:
        Lag-1 autocorrelation of the trace at the kept-imputation spacing;
        near zero indicates the kept imputations are effectively
        independent draws.
    """

    n_sweeps: int
    burn_in: int
    thin: int
    m: int
    trace_lag1_autocorrelation: float
    trace_ess: float
    kept_lag1_autocorrelation: float
    notes: str = field(
        default=(
            "single-chain heuristics; multi-chain R-hat/ESS are planned "
            "with the simulation suite"
        )
    )
