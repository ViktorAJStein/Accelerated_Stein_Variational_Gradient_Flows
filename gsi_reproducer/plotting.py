from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from .algorithms import METHODS, RunResult
from .targets import Target


COLORS = {
    "ASVGD": "tab:blue",
    "SVGD": "tab:orange",
    "ULA": "tab:green",
    "ULD": "tab:red",
    "MALA": "tab:purple",
}


def _save(figure: plt.Figure, output_base: Path) -> None:
    for suffix in ("png", "pdf"):
        figure.savefig(output_base.with_suffix(f".{suffix}"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_metrics(
    result: RunResult,
    target: Target,
    output_dir: Path,
    *,
    second_bilinear_result: RunResult | None = None,
) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.5))
    parametric = result.metric_name == "gaussian_moment_kl"
    x = result.metric_iterations
    for method in METHODS:
        values = result.metrics[method]
        visible = values if parametric else np.where(values > 0.0, values, np.nan)
        label = method
        if second_bilinear_result is not None and method in ("ASVGD", "SVGD"):
            label = rf"{method}, $A=I$"
        axis.plot(x, visible, label=label, color=COLORS[method], linewidth=2)
    if second_bilinear_result is not None:
        for method, linestyle in (("ASVGD", "--"), ("SVGD", "--")):
            values = second_bilinear_result.metrics[method]
            axis.plot(
                second_bilinear_result.metric_iterations,
                values if parametric else np.where(values > 0.0, values, np.nan),
                label=rf"{method}, $A=Q^{{-1}}$",
                color="black" if method == "ASVGD" else "gray",
                linestyle=linestyle,
                linewidth=2,
            )
    collections = [result.metrics]
    if second_bilinear_result is not None:
        collections.append(second_bilinear_result.metrics)
    has_zero = parametric and any(np.any(v == 0.0) for c in collections for v in c.values())
    if has_zero:
        axis.set_yscale("symlog", linthresh=1e-14)
        axis.set_ylim(bottom=0.0)
    else:
        axis.set_yscale("log")
    axis.set_xlabel("Iterations")
    axis.set_ylabel("Gaussian plug-in KL" if parametric else "Historical MC/KDE score")
    scale = "paper-scale" if result.config.particles == 500 and result.config.steps == 1000 else "debug"
    axis.set_title(f"{target.name}: {scale} reconstruction")
    axis.grid(which="both", alpha=0.35)
    axis.legend()
    hidden = 0 if parametric else sum(int(np.sum(values <= 0.0)) for values in result.metrics.values())
    if hidden:
        axis.text(
            0.01,
            0.01,
            f"{hidden} nonpositive values are not representable on the log axis",
            transform=axis.transAxes,
            fontsize=8,
        )
    if has_zero:
        axis.text(0.01, 0.01, "Linear axis below 1e-14; no energy floor", transform=axis.transAxes, fontsize=8)
    _save(figure, output_dir / result.metric_name)


def _target_contours(axis: plt.Axes, target: Target) -> None:
    xmin, xmax, ymin, ymax = target.limits
    xx, yy = np.mgrid[xmin:xmax:180j, ymin:ymax:180j]
    points = np.column_stack((xx.ravel(), yy.ravel()))
    log_values = target.log_density(points).reshape(xx.shape)
    finite = log_values[np.isfinite(log_values)]
    levels = np.linspace(np.quantile(finite, 0.55), np.quantile(finite, 0.98), 7)
    axis.contour(xx, yy, log_values, levels=levels, colors="black", linewidths=1.0)
    axis.set_xlim(xmin, xmax)
    axis.set_ylim(ymin, ymax)
    axis.set_aspect("equal", adjustable="box")
    axis.set_axis_off()


def _trajectory_panel(axis: plt.Axes, history: np.ndarray, target: Target, title: str) -> None:
    # At most 201 temporal points are needed to display the paths clearly.
    temporal_indices = np.linspace(
        0, len(history) - 1, min(201, len(history)), dtype=int
    )
    displayed = history[temporal_indices]
    segments = np.stack((displayed[:-1], displayed[1:]), axis=2)
    segments = segments.transpose(1, 0, 2, 3).reshape(-1, 2, 2)
    collection = LineCollection(segments, colors=(0.35, 0.5, 0.9, 0.12), linewidths=0.6)
    axis.add_collection(collection)
    axis.scatter(
        displayed[0, :, 0],
        displayed[0, :, 1],
        s=9,
        marker="o",
        color=(59 / 255, 76 / 255, 192 / 255),
        zorder=3,
    )
    axis.scatter(
        displayed[-1, :, 0],
        displayed[-1, :, 1],
        s=9,
        marker="s",
        color=(180 / 255, 4 / 255, 38 / 255),
        zorder=4,
    )
    _target_contours(axis, target)
    axis.set_title(title)


def plot_trajectories(result: RunResult, target: Target, output_dir: Path) -> None:
    figure, axes = plt.subplots(1, len(METHODS), figsize=(19, 4.2))
    for axis, method in zip(axes, METHODS, strict=True):
        _trajectory_panel(axis, result.trajectories[method], target, method)
    figure.suptitle(
        f"{target.name}: blue circles are initial; red squares are final",
        fontsize=12,
    )
    figure.tight_layout()
    _save(figure, output_dir / "trajectories")
