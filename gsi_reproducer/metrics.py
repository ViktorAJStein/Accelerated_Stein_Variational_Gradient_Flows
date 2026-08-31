from __future__ import annotations

import math

import numpy as np
from scipy.stats import gaussian_kde

from .targets import Target


FLOOR = 1e-12


def gaussian_moment_kl(x: np.ndarray, target: Target) -> float:
    if target.mean is None or target.covariance is None:
        raise ValueError("target moments required")
    empirical_mean = x.mean(axis=0)
    empirical_covariance = np.cov(x, rowvar=False)
    target_precision = np.linalg.inv(target.covariance)
    _, logdet_empirical = np.linalg.slogdet(empirical_covariance)
    _, logdet_target = np.linalg.slogdet(target.covariance)
    difference = target.mean - empirical_mean
    dimension = x.shape[1]
    return float(
        0.5
        * (
            np.trace(target_precision @ empirical_covariance)
            - dimension
            + logdet_target
            - logdet_empirical
            + difference @ target_precision @ difference
        )
    )


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
