#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from gsi_reproducer.algorithms import METHODS, run_experiment
from gsi_reproducer.configs import PAPER_CONFIGS
from gsi_reproducer.targets import TARGETS


def main() -> int:
    rng = np.random.default_rng(20260825)
    score_errors = {}
    step = 1e-6
    for name, target in TARGETS.items():
        x = rng.normal(size=(5, 2))
        if name == "double_bananas":
            x[:, 0] += 1.0
        numerical = np.empty_like(x)
        for coordinate in range(2):
            direction = np.zeros_like(x)
            direction[:, coordinate] = step
            numerical[:, coordinate] = (
                target.log_density(x + direction) - target.log_density(x - direction)
            ) / (2.0 * step)
        score_errors[name] = float(np.max(np.abs(numerical - target.score(x))))

    smoke = {}
    for name, base_config in PAPER_CONFIGS.items():
        config = replace(
            base_config,
            particles=12,
            steps=3,
            metric_stride=1,
            trajectory_stride=1,
        )
        result = run_experiment(config, TARGETS[config.target_name], progress=False)
        smoke[name] = {
            method: bool(
                np.all(np.isfinite(result.final_states[method]))
                and np.all(np.isfinite(result.historical_metrics[method]))
            )
            for method in METHODS
        }

    summary = {
        "score_worst_errors": score_errors,
        "smoke": smoke,
        "status": "pass",
    }
    if max(score_errors.values()) > 1e-5:
        summary["status"] = "fail"
    if not all(all(methods.values()) for methods in smoke.values()):
        summary["status"] = "fail"
    Path("verification.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if summary["status"] != "pass":
        raise AssertionError(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
