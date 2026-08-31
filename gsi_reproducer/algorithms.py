from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .configs import ExperimentConfig
from .metrics import historical_kde_score
from .targets import Target


METHODS = ("ASVGD", "SVGD", "ULA", "ULD", "MALA")


@dataclass
class RunResult:
    config: ExperimentConfig
    metric_iterations: np.ndarray
    historical_metrics: dict[str, np.ndarray]
    trajectory_iterations: np.ndarray
    trajectories: dict[str, np.ndarray]
    final_states: dict[str, np.ndarray]
    restart_counts: dict[str, int]
    failure_iterations: dict[str, int | None]


def rbf_kernel(x: np.ndarray, sigma: float) -> np.ndarray:
    differences = x[:, None, :] - x[None, :, :]
    return np.exp(-np.sum(differences**2, axis=2) / (2.0 * sigma**2))


def _historical_metric(x: np.ndarray, target: Target, kernel: str) -> float:
    del kernel
    return historical_kde_score(x, target)


def _asvgd_force(
    x: np.ndarray,
    y: np.ndarray,
    target: Target,
    config: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(x)
    score = target.score(x)
    if config.kernel == "bilinear":
        matrix = config.bilinear_matrix_array()
        assert matrix is not None
        kernel = x @ matrix @ x.T + np.ones((n, n))
        velocity = n * np.linalg.solve(
            kernel + n * config.regularization * np.eye(n), y
        )
        trace_term = np.trace(velocity.T @ kernel @ velocity) / n**2
        force = kernel @ score / n + (1.0 + trace_term) * (x @ matrix)
        return force, kernel, velocity

    kernel = rbf_kernel(x, config.bandwidth)
    velocity = n * np.linalg.solve(
        kernel + n * config.regularization * np.eye(n), y
    )
    gram = velocity @ velocity.T
    weights = n * kernel + kernel @ (gram * kernel) - kernel * (kernel @ gram)
    laplacian = np.diag(weights.sum(axis=1)) - weights
    interaction = laplacian @ x / (n**2 * config.bandwidth**2)
    if config.asvgd_variant == "intended":
        force = kernel @ score / n + interaction
    elif config.asvgd_variant == "specialized_public":
        force = score / n + interaction
    elif config.asvgd_variant == "historical_half":
        force = score / n + 0.5 * interaction
    else:
        raise ValueError(f"unknown ASVGD variant: {config.asvgd_variant}")
    return force, kernel, velocity


def _paper_alpha(
    x: np.ndarray,
    previous_x: np.ndarray | None,
    previous_previous_x: np.ndarray | None,
    age: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    restarts = 0
    if previous_x is not None and previous_previous_x is not None:
        current_speed = np.linalg.norm(x - previous_x, axis=1)
        previous_speed = np.linalg.norm(previous_x - previous_previous_x, axis=1)
        mask = current_speed < previous_speed
        age = age.copy()
        age[mask] = 0.0
        restarts = int(mask.sum())
    return age / (age + 3.0), age, restarts


def run_experiment(
    config: ExperimentConfig,
    target: Target,
    *,
    progress: bool = True,
) -> RunResult:
    n = config.particles
    dimension = 2
    rng = np.random.RandomState(config.seed)
    initial = rng.multivariate_normal(
        np.asarray(config.initial_mean), config.initial_covariance_array(), size=n
    )
    states = {method: initial.copy() for method in METHODS}
    active = {method: True for method in METHODS}
    failure_iterations: dict[str, int | None] = {method: None for method in METHODS}
    asvgd_momentum = np.zeros((n, dimension))
    uld_velocity = np.zeros((n, dimension))
    age = np.zeros(n)
    previous_x: np.ndarray | None = None
    previous_previous_x: np.ndarray | None = None
    restart_counts = {"speed_particle_events": 0, "gradient_events": 0}

    metric_iterations = np.arange(0, config.steps + 1, config.metric_stride)
    if metric_iterations[-1] != config.steps:
        metric_iterations = np.append(metric_iterations, config.steps)
    metric_lookup = {int(k): i for i, k in enumerate(metric_iterations)}
    historical_metrics = {
        method: np.full(len(metric_iterations), np.nan) for method in METHODS
    }

    trajectory_iterations = np.arange(0, config.steps + 1, config.trajectory_stride)
    if trajectory_iterations[-1] != config.steps:
        trajectory_iterations = np.append(trajectory_iterations, config.steps)
    trajectory_lookup = {int(k): i for i, k in enumerate(trajectory_iterations)}
    trajectories = {
        method: np.empty((len(trajectory_iterations), n, dimension))
        for method in METHODS
    }

    for iteration in range(config.steps + 1):
        if iteration in metric_lookup:
            metric_index = metric_lookup[iteration]
            for method in METHODS:
                if active[method]:
                    try:
                        historical_metrics[method][metric_index] = _historical_metric(
                            states[method], target, config.kernel
                        )
                    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                        historical_metrics[method][metric_index] = np.nan
                elif target.historical_plot_offset is not None:
                    # Original arrays were zero-initialized. After a caught metric
                    # failure, plotting added the target offset to untouched zeros.
                    historical_metrics[method][metric_index] = target.historical_plot_offset
        if iteration in trajectory_lookup:
            trajectory_index = trajectory_lookup[iteration]
            for method in METHODS:
                trajectories[method][trajectory_index] = states[method]
        if iteration == config.steps:
            break
        if progress and (iteration % max(1, config.steps // 10) == 0):
            print(f"  {config.name}: iteration {iteration}/{config.steps}", flush=True)

        current_x = states["ASVGD"]
        if config.update_order == "position_first":
            current_x = current_x + config.asvgd_increment * asvgd_momentum
            states["ASVGD"] = current_x

        force, kernel, velocity = _asvgd_force(
            current_x, asvgd_momentum, target, config
        )
        if config.beta is not None:
            alpha = np.full(n, config.beta)
        elif config.restart_mode == "paper":
            alpha, age, speed_events = _paper_alpha(
                current_x, previous_x, previous_previous_x, age
            )
            restart_counts["speed_particle_events"] += speed_events
            if config.kernel == "rbf":
                score = target.score(current_x)
                b = -score + current_x
                phi = np.sum((b @ velocity.T) * kernel)
                phi -= np.sum(kernel.sum(axis=0) * np.sum(current_x * velocity, axis=1))
                if phi < 0.0:
                    alpha[:] = 0.0
                    age[:] = 0.0
                    restart_counts["gradient_events"] += 1
        elif config.restart_mode == "none":
            alpha = np.full(n, iteration / (iteration + 3.0))
        else:
            raise ValueError(f"unknown restart mode: {config.restart_mode}")

        asvgd_momentum = alpha[:, None] * asvgd_momentum + config.asvgd_increment * force
        if config.update_order == "momentum_first":
            states["ASVGD"] = current_x + config.asvgd_increment * asvgd_momentum
        elif config.update_order != "position_first":
            raise ValueError(f"unknown update order: {config.update_order}")
        previous_previous_x = None if previous_x is None else previous_x.copy()
        previous_x = current_x.copy()
        age += 1.0

        x_svgd = states["SVGD"]
        score_svgd = target.score(x_svgd)
        if config.kernel == "rbf":
            kernel_svgd = rbf_kernel(x_svgd, config.bandwidth)
            laplacian = np.diag(kernel_svgd.sum(axis=1)) - kernel_svgd
            svgd_velocity = (
                laplacian @ x_svgd / config.bandwidth**2
                + kernel_svgd @ score_svgd
            ) / n
        else:
            matrix = config.bilinear_matrix_array()
            assert matrix is not None
            kernel_svgd = x_svgd @ matrix @ x_svgd.T + np.ones((n, n))
            svgd_velocity = x_svgd @ matrix + kernel_svgd @ score_svgd / n
        svgd_candidate = x_svgd + config.base_dt * svgd_velocity
        if np.all(np.isfinite(svgd_candidate)):
            states["SVGD"] = svgd_candidate
        else:
            active["SVGD"] = False
            failure_iterations["SVGD"] = iteration + 1

        noise_scale = np.sqrt(2.0 * config.base_dt)
        x_ula = states["ULA"]
        if active["ULA"]:
            with np.errstate(over="ignore", invalid="ignore"):
                ula_candidate = (
                    x_ula
                    + config.base_dt * target.score(x_ula)
                    + noise_scale * rng.randn(n, dimension)
                )
            if np.all(np.isfinite(ula_candidate)):
                states["ULA"] = ula_candidate
            else:
                active["ULA"] = False
                failure_iterations["ULA"] = iteration + 1

        if active["ULD"]:
            x_uld = states["ULD"] + config.base_dt * uld_velocity
            with np.errstate(over="ignore", invalid="ignore"):
                velocity_candidate = (
                    (1.0 - config.base_dt) * uld_velocity
                    + config.base_dt * target.score(x_uld)
                    + noise_scale * rng.randn(n, dimension)
                )
            if np.all(np.isfinite(x_uld)) and np.all(np.isfinite(velocity_candidate)):
                states["ULD"] = x_uld
                uld_velocity = velocity_candidate
            else:
                active["ULD"] = False
                failure_iterations["ULD"] = iteration + 1

        if active["MALA"]:
            x_mala = states["MALA"]
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                score_mala = target.score(x_mala)
                proposal_mean = x_mala + config.base_dt * score_mala
                proposal = proposal_mean + noise_scale * rng.randn(n, dimension)
                backward_mean = proposal + config.base_dt * target.score(proposal)
                log_target_ratio = target.log_density(proposal) - target.log_density(x_mala)
                log_proposal_ratio = (
                    -np.sum((x_mala - backward_mean) ** 2, axis=1)
                    + np.sum((proposal - proposal_mean) ** 2, axis=1)
                ) / (4.0 * config.base_dt)
                log_acceptance = np.minimum(0.0, log_target_ratio + log_proposal_ratio)
            accept = (
                np.all(np.isfinite(proposal), axis=1)
                & np.isfinite(log_acceptance)
                & (np.log(rng.rand(n)) < log_acceptance)
            )
            x_mala = x_mala.copy()
            x_mala[accept] = proposal[accept]
            states["MALA"] = x_mala

        if not np.all(np.isfinite(states["ASVGD"])):
            raise FloatingPointError(
                f"ASVGD became non-finite at iteration {iteration + 1}"
            )

    return RunResult(
        config=config,
        metric_iterations=metric_iterations,
        historical_metrics=historical_metrics,
        trajectory_iterations=trajectory_iterations,
        trajectories=trajectories,
        final_states={method: state.copy() for method, state in states.items()},
        restart_counts=restart_counts,
        failure_iterations=failure_iterations,
    )
