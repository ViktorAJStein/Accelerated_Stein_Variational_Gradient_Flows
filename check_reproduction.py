#!/usr/bin/env python3
"""Compare a paper-scale run with the committed validation summaries."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from gsi_reproducer.metrics import gaussian_moment_kl, historical_kde_score, independent_kde_kl
from gsi_reproducer.targets import GAUSSIAN


EXPERIMENTS = (
    "gaussian_bilinear",
    "gaussian_bilinear_Qinv",
    "non_lipschitz",
    "double_bananas",
    "anisotropic_gaussian",
)
METHODS = ("ASVGD", "SVGD", "ULA", "ULD", "MALA")


def close(actual: float, expected: float, *, rtol: float, atol: float) -> bool:
    return math.isfinite(actual) and math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/paper"),
        help="Root produced by run.py --experiment all --profile paper",
    )
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENTS, default=EXPERIMENTS)
    args = parser.parse_args()

    failures: list[str] = []
    checked = 0
    for experiment in args.experiments:
        actual_path = args.results / experiment / "summary.json"
        expected_path = Path("validation/summaries") / f"{experiment}.json"
        if not actual_path.exists():
            failures.append(f"missing {actual_path}")
            continue
        actual = json.loads(actual_path.read_text())
        expected = json.loads(expected_path.read_text())

        if actual.get("metric_name") == "gaussian_moment_kl":
            # Keep the committed historical baselines and their tolerances intact.
            # Recompute their OLD diagnostics only here, from saved states; the
            # default Gaussian experiment itself performs no KDE/MC estimation.
            config = json.loads((actual_path.parent / "config.json").read_text())
            if actual.get("covariance_divisor") != "N":
                failures.append(f"{experiment}: expected covariance divisor N")
            actual["historical_metric"] = {}
            actual["independent_final_kde_kl"] = {}
            with np.load(actual_path.parent / "results.npz", allow_pickle=False) as arrays:
                for index, method in enumerate(METHODS):
                    actual["historical_metric"][method] = {}
                    for field in ("initial", "final"):
                        state = arrays[f"{field}_{method}"]
                        plug_in = gaussian_moment_kl(state, GAUSSIAN)
                        stored = actual["gaussian_moment_kl"][method][field]
                        checked += 1
                        if not close(stored, plug_in, rtol=1e-12, atol=1e-14):
                            failures.append(f"{experiment}/{method}: invalid {field} Gaussian energy")
                        actual["historical_metric"][method][field] = historical_kde_score(state, GAUSSIAN)
                    if actual["failure_iterations"][method] is None:
                        estimate, standard_error = independent_kde_kl(
                            arrays[f"final_{method}"], GAUSSIAN, samples=10_000,
                            seed=config["seed"]+1000+index)
                        actual["independent_final_kde_kl"][method] = {
                            "estimate": estimate, "standard_error": standard_error}
                    else:
                        actual["independent_final_kde_kl"][method] = {
                            "status": "skipped_after_numerical_failure",
                            "failure_iteration": actual["failure_iterations"][method]}

        if actual.get("paper_scale") is not True:
            failures.append(f"{experiment}: run is not marked paper_scale")
        if actual["failure_iterations"] != expected["failure_iterations"]:
            failures.append(
                f"{experiment}: failure iterations {actual['failure_iterations']} "
                f"!= {expected['failure_iterations']}"
            )

        for method in METHODS:
            for field in ("initial", "final"):
                got = float(actual["historical_metric"][method][field])
                want = float(expected["historical_metric"][method][field])
                checked += 1
                if not close(got, want, rtol=0.02, atol=0.02):
                    failures.append(
                        f"{experiment}/{method}: historical {field} {got:.8g} "
                        f"outside tolerance of {want:.8g}"
                    )

            got_independent = actual["independent_final_kde_kl"][method]
            want_independent = expected["independent_final_kde_kl"][method]
            if "estimate" in want_independent:
                checked += 1
                got = float(got_independent["estimate"])
                want = float(want_independent["estimate"])
                if not close(got, want, rtol=0.05, atol=0.02):
                    failures.append(
                        f"{experiment}/{method}: independent estimate {got:.8g} "
                        f"outside tolerance of {want:.8g}"
                    )
            elif got_independent != want_independent:
                failures.append(
                    f"{experiment}/{method}: independent failure record changed"
                )

    report = {"status": "fail" if failures else "pass", "checks": checked, "failures": failures}
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
