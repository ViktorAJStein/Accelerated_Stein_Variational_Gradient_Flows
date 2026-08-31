from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Target:
    name: str
    mean: np.ndarray | None
    covariance: np.ndarray | None
    limits: tuple[float, float, float, float]
    log_z: float
    historical_plot_offset: float | None = None

    def score(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def log_unnormalized(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def log_density(self, x: np.ndarray) -> np.ndarray:
        return self.log_unnormalized(x) - self.log_z


@dataclass(frozen=True)
class GaussianTarget(Target):
    def score(self, x: np.ndarray) -> np.ndarray:
        assert self.mean is not None and self.covariance is not None
        return -(x - self.mean) @ np.linalg.inv(self.covariance)

    def log_unnormalized(self, x: np.ndarray) -> np.ndarray:
        assert self.mean is not None and self.covariance is not None
        difference = x - self.mean
        precision = np.linalg.inv(self.covariance)
        quadratic = np.einsum("ni,ij,nj->n", difference, precision, difference)
        dimension = x.shape[1]
        _, logdet = np.linalg.slogdet(self.covariance)
        return -0.5 * (quadratic + dimension * np.log(2.0 * np.pi) + logdet)


@dataclass(frozen=True)
class NonLipschitzTarget(Target):
    def score(self, x: np.ndarray) -> np.ndarray:
        with np.errstate(over="ignore", invalid="ignore"):
            return -(x**3)

    def log_unnormalized(self, x: np.ndarray) -> np.ndarray:
        with np.errstate(over="ignore", invalid="ignore"):
            return -0.25 * np.sum(x**4, axis=1)


@dataclass(frozen=True)
class BananasTarget(Target):
    def score(self, x: np.ndarray) -> np.ndarray:
        radius = np.linalg.norm(x, axis=1)
        if np.any(radius == 0.0):
            raise FloatingPointError("double-bananas score is singular at the origin")
        radial = 4.0 * x * (1.0 - 3.0 / radius[:, None])
        exponential = np.exp(np.clip(24.0 * x[:, 0], -700.0, 700.0))
        horizontal = 4.0 * (
            exponential * (x[:, 0] - 3.0) + x[:, 0] + 3.0
        ) / (exponential + 1.0)
        return -radial - np.column_stack((horizontal, np.zeros(len(x))))

    def log_unnormalized(self, x: np.ndarray) -> np.ndarray:
        radial = -2.0 * (np.linalg.norm(x, axis=1) - 3.0) ** 2
        components = np.logaddexp(
            -2.0 * (x[:, 0] - 3.0) ** 2,
            -2.0 * (x[:, 0] + 3.0) ** 2,
        )
        return radial + components


CODE_GAUSSIAN_COVARIANCE = np.array([[3.0, -2.0], [-2.0, 3.0]])
GAUSSIAN = GaussianTarget(
    name="gaussian_bilinear",
    mean=np.zeros(2),
    covariance=CODE_GAUSSIAN_COVARIANCE,
    limits=(-8.0, 8.0, -8.0, 8.0),
    log_z=0.0,
    historical_plot_offset=0.0,
)
NON_LIPSCHITZ = NonLipschitzTarget(
    name="non_lipschitz",
    mean=np.zeros(2),
    covariance=(2.0 * math.gamma(0.75) / math.gamma(0.25)) * np.eye(2),
    limits=(-8.0, 8.0, -3.0, 3.0),
    log_z=float(np.log(math.gamma(0.25) ** 2 / 2.0)),
    historical_plot_offset=float(math.gamma(0.25) ** 2 / 2.0),
)
BANANAS = BananasTarget(
    name="double_bananas",
    mean=None,
    covariance=None,
    limits=(-8.0, 8.0, -5.0, 5.0),
    log_z=2.1165420987117627,
    historical_plot_offset=2.1165420987117627,
)
ANISOTROPIC = GaussianTarget(
    name="anisotropic_gaussian",
    mean=np.ones(2),
    covariance=np.diag([10.0, 0.05]),
    limits=(-12.0, 14.0, -4.0, 4.0),
    log_z=0.0,
    historical_plot_offset=0.0,
)

TARGETS = {
    target.name: target
    for target in (GAUSSIAN, NON_LIPSCHITZ, BANANAS, ANISOTROPIC)
}
