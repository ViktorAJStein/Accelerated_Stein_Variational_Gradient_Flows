from __future__ import annotations

import math

import numpy as np
from scipy.stats import gaussian_kde

from .targets import GaussianTarget, Target


FLOOR = 1e-12


def metric_name(target: Target, kernel: str) -> str:
    """Only the Gaussian/bilinear experiment adopts the parametric diagnostic."""
    if kernel == "bilinear" and isinstance(target, GaussianTarget):
        return "gaussian_moment_kl"
    return "historical_metric"


def gaussian_moment_kl(x: np.ndarray, target: Target) -> float:
    """KL(N(mu_hat, Sigma_hat) || target), with empirical covariance divisor N.

    This is a Gaussian plug-in energy, not KL of the atomic empirical measure.
    Numerically rank-deficient fits give +inf (standard SVD machine-precision
    rank threshold); no ridge, eigenvalue clipping or density floor is added.
    """
    if not isinstance(target, GaussianTarget):
        raise ValueError("Gaussian target required, not just known moments")
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or min(x.shape) == 0 or not np.all(np.isfinite(x)):
        raise ValueError("particles must be a finite nonempty (N,d) array")
    n, dimension = x.shape
    mean = np.asarray(target.mean, dtype=float)
    covariance = np.asarray(target.covariance, dtype=float)
    if (mean.shape != (dimension,) or covariance.shape != (dimension, dimension)
            or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance))
            or not np.array_equal(covariance, covariance.T)):
        raise ValueError("target mean/covariance must be finite and dimensionally consistent")
    cholesky = np.linalg.cholesky(covariance)  # rejects non-SPD targets
    if n <= dimension:
        return float("inf")
    empirical_mean = x.mean(axis=0)
    centered = x - empirical_mean
    whitened = np.linalg.solve(cholesky, centered.T).T
    # Singular values avoid squaring the condition number before resolving rank.
    singular_values = np.linalg.svd(whitened, compute_uv=False)
    rank_threshold = max(whitened.shape) * np.finfo(float).eps * singular_values[0]
    if singular_values[-1] <= rank_threshold:
        return float("inf")
    eigenvalues = singular_values**2 / n
    if not np.all(np.isfinite(eigenvalues)):
        raise FloatingPointError("nonfinite fitted covariance")
    delta = eigenvalues - 1.0
    terms = eigenvalues - 1.0 - np.log(eigenvalues)
    near = np.abs(delta) < 1e-4
    z = delta[near]
    terms[near] = z**2 * (0.5 + z * (-1/3 + z * (1/4 - z/5)))
    difference = np.linalg.solve(cholesky, empirical_mean - mean)
    value = float(0.5 * (terms.sum() + difference @ difference))
    if not np.isfinite(value):
        raise FloatingPointError("nonfinite Gaussian plug-in energy")
    return value


def historical_kde_score(x: np.ndarray, target: Target) -> float:
    """Reproduce the paper code's same-sample, target-floored diagnostic."""
    kde = gaussian_kde(x.T)
    q = kde.evaluate(x.T)
    log_unnormalized = target.log_unnormalized(x)
    floored_target = np.log(np.exp(log_unnormalized) + FLOOR)
    offset = (
        target.log_z
        if target.historical_plot_offset is None
        else target.historical_plot_offset
    )
    return float(np.mean(np.log(q + FLOOR) - floored_target) + offset)


def independent_kde_kl(
    x: np.ndarray, target: Target, *, samples: int, seed: int
) -> tuple[float, float]:
    """Estimate the genuine KL of the fitted KDE using independent KDE draws."""
    kde = gaussian_kde(x.T)
    draws = kde.resample(samples, seed=np.random.default_rng(seed)).T
    log_ratio = kde.logpdf(draws.T) - target.log_density(draws)
    return (
        float(np.mean(log_ratio)),
        float(np.std(log_ratio, ddof=1) / math.sqrt(samples)),
    )
