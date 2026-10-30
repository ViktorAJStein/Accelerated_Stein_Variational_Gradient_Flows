#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

import matplotlib
import numpy as np
import scipy

from gsi_reproducer.algorithms import METHODS, RunResult, run_experiment
from gsi_reproducer.configs import PAPER_CONFIGS, ExperimentConfig, with_profile
from gsi_reproducer.metrics import independent_kde_kl
from gsi_reproducer.plotting import plot_metrics, plot_trajectories
from gsi_reproducer.targets import CODE_GAUSSIAN_COVARIANCE, TARGETS


def save_result(
    result: RunResult,
    target,
    output_dir: Path,
    *,
    save_trajectories: bool,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(result.config.to_dict(), indent=2) + "\n"
    )
    environment = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
    }
    (output_dir / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")

    arrays: dict[str, np.ndarray] = {
        "metric_iterations": result.metric_iterations,
        "trajectory_iterations": result.trajectory_iterations,
    }
    for method in METHODS:
        prefix = "historical" if result.metric_name == "historical_metric" else result.metric_name
        arrays[f"{prefix}_{method}"] = result.metrics[method]
        arrays[f"final_{method}"] = result.final_states[method]
        arrays[f"initial_{method}"] = result.trajectories[method][0]
        if save_trajectories:
            arrays[f"trajectory_{method}"] = result.trajectories[method]
    np.savez_compressed(output_dir / "results.npz", **arrays)

    summary: dict[str, object] = {
        "experiment": result.config.name,
        "paper_scale": result.config.particles == 500 and result.config.steps == 1000,
        "metric_name": result.metric_name,
        result.metric_name: {},
        "restart_counts": result.restart_counts,
        "failure_iterations": result.failure_iterations,
    }
    metric_summary = summary[result.metric_name]
    independent_summary = {}
    if result.metric_name == "historical_metric":
        summary["independent_final_kde_kl"] = independent_summary
    else:
        summary["covariance_divisor"] = "N"
        summary["interpretation"] = "KL of moment-fitted Gaussian, not atomic empirical measure"
    assert isinstance(metric_summary, dict)
    for index, method in enumerate(METHODS):
        metric = result.metrics[method]
        metric_summary[method] = {
            "initial": float(metric[0]),
            "final": float(metric[-1]),
            "minimum": float(np.nanmin(metric)),
            "minimum_iteration": int(result.metric_iterations[np.nanargmin(metric)]),
        }
        if result.metric_name != "historical_metric":
            continue  # No KDE fitting or auxiliary Monte Carlo for Gaussian/bilinear.
        if result.failure_iterations[method] is None:
            estimate, standard_error = independent_kde_kl(
                result.final_states[method],
                target,
                samples=10_000,
                seed=result.config.seed + 1000 + index,
            )
            independent_summary[method] = {
                "estimate": estimate,
                "standard_error": standard_error,
                "samples": 10_000,
            }
        else:
            independent_summary[method] = {
                "status": "skipped_after_numerical_failure",
                "failure_iteration": result.failure_iterations[method],
            }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def run_one(
    config: ExperimentConfig,
    output_root: Path,
    *,
    progress: bool,
    save_trajectories: bool,
) -> dict[str, object]:
    target = TARGETS[config.target_name]
    print(f"Running {config.name}: N={config.particles}, steps={config.steps}")
    result = run_experiment(config, target, progress=progress)
    experiment_dir = output_root / config.name
    summary = save_result(
        result, target, experiment_dir, save_trajectories=save_trajectories
    )

    second_result = None
    if config.name == "gaussian_bilinear":
        inverse = np.linalg.inv(CODE_GAUSSIAN_COVARIANCE)
        second_config = replace(
            config,
            name="gaussian_bilinear_Qinv",
            bilinear_matrix=tuple(tuple(float(value) for value in row) for row in inverse),
        )
        print("Running Gaussian comparison with A=Q^{-1}")
        second_result = run_experiment(second_config, target, progress=progress)
        second_summary = save_result(
            second_result,
            target,
            output_root / second_config.name,
            save_trajectories=save_trajectories,
        )
        summary["Qinv_comparison"] = second_summary
        (experiment_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    plot_metrics(
        result,
        target,
        experiment_dir,
        second_bilinear_result=second_result,
    )
    plot_trajectories(result, target, experiment_dir)
    print(f"Wrote {experiment_dir}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the reconstructed GSI ASVGD toy experiments."
    )
    parser.add_argument(
        "--experiment",
        choices=("all", *PAPER_CONFIGS.keys()),
        default="all",
    )
    parser.add_argument(
        "--profile",
        choices=(
            "paper",
            "surviving-code-steps",
            "literal-paper-restarts",
            "legacy-public",
        ),
        default="paper",
    )
    parser.add_argument("--output", type=Path, default=Path("results/paper"))
    parser.add_argument("--quick", action="store_true", help="N=80, 80 steps; debugging only")
    parser.add_argument("--particles", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--save-trajectories", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    names = list(PAPER_CONFIGS) if args.experiment == "all" else [args.experiment]
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "profile": args.profile,
        "quick_debug_only": args.quick,
        "experiments": names,
        "summaries": {},
    }
    if args.quick:
        print("WARNING: --quick is a debugging run, not a paper reproduction.")
    for name in names:
        config = with_profile(PAPER_CONFIGS[name], args.profile)
        overrides: dict[str, object] = {}
        if args.quick:
            overrides.update(particles=80, steps=80, trajectory_stride=1)
        if args.particles is not None:
            overrides["particles"] = args.particles
        if args.steps is not None:
            overrides["steps"] = args.steps
        if args.seed is not None:
            overrides["seed"] = args.seed
        config = replace(config, **overrides)
        manifest["summaries"][name] = run_one(
            config,
            args.output,
            progress=not args.quiet,
            save_trajectories=args.save_trajectories,
        )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
