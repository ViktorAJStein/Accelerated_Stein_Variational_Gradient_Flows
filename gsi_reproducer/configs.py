from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    target_name: str
    kernel: str
    initial_mean: tuple[float, float]
    initial_covariance: tuple[tuple[float, float], tuple[float, float]]
    particles: int = 500
    steps: int = 1000
    seed: int = 42
    asvgd_increment: float = 0.31622776601683794  # sqrt(paper tau=0.1)
    base_dt: float = 0.1
    bandwidth: float = 0.1
    regularization: float = 0.1
    beta: float | None = None
    restart_mode: str = "none"
    asvgd_variant: str = "intended"
    update_order: str = "momentum_first"
    bilinear_matrix: tuple[tuple[float, float], tuple[float, float]] | None = None
    metric_stride: int = 1
    trajectory_stride: int = 1

    def initial_covariance_array(self) -> np.ndarray:
        return np.asarray(self.initial_covariance, dtype=float)

    def bilinear_matrix_array(self) -> np.ndarray | None:
        if self.bilinear_matrix is None:
            return None
        return np.asarray(self.bilinear_matrix, dtype=float)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


PAPER_CONFIGS = {
    "gaussian_bilinear": ExperimentConfig(
        name="gaussian_bilinear",
        target_name="gaussian_bilinear",
        kernel="bilinear",
        initial_mean=(1.0, 1.0),
        initial_covariance=((3.0, 2.0), (2.0, 3.0)),
        asvgd_increment=0.1,
        base_dt=0.01,
        bilinear_matrix=((1.0, 0.0), (0.0, 1.0)),
    ),
    "non_lipschitz": ExperimentConfig(
        name="non_lipschitz",
        target_name="non_lipschitz",
        kernel="rbf",
        initial_mean=(5.0, 0.0),
        initial_covariance=((1.0, 0.0), (0.0, 1.0)),
    ),
    "double_bananas": ExperimentConfig(
        name="double_bananas",
        target_name="double_bananas",
        kernel="rbf",
        initial_mean=(7.0, 0.0),
        initial_covariance=((1.0, 0.0), (0.0, 1.0)),
        beta=0.985,
        restart_mode="constant_beta",
    ),
    "anisotropic_gaussian": ExperimentConfig(
        name="anisotropic_gaussian",
        target_name="anisotropic_gaussian",
        kernel="rbf",
        initial_mean=(0.0, 0.0),
        initial_covariance=((1.0, 0.0), (0.0, 1.0)),
    ),
}


def with_profile(config: ExperimentConfig, profile: str) -> ExperimentConfig:
    values = config.to_dict()
    if profile == "paper":
        # Literal reported tau=0.1 reproduces the published SVGD/ULA time scale.
        # Adaptive restarts remain disabled, matching source defaults and visible acceleration.
        if config.name != "double_bananas":
            values["restart_mode"] = "none"
    elif profile == "surviving-code-steps":
        values.update(asvgd_increment=0.1, base_dt=0.01)
        if config.name != "double_bananas":
            values["restart_mode"] = "none"
    elif profile == "literal-paper-restarts":
        if config.name != "double_bananas":
            values["restart_mode"] = "paper"
    elif profile == "legacy-public":
        values.update(
            asvgd_variant="specialized_public",
            restart_mode="none",
            update_order="momentum_first",
            asvgd_increment=0.1,
            base_dt=0.01,
        )
        if config.name != "double_bananas":
            values["beta"] = None
    else:
        raise ValueError(f"unknown profile: {profile}")
    return ExperimentConfig(**values)
