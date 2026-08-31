#!/usr/bin/env python3
"""Compare a paper-scale run with the committed validation summaries."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


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
    args = parser.parse_args()

    failures: list[str] = []
    checked = 0
    for experiment in EXPERIMENTS:
        actual_path = args.results / experiment / "summary.json"
        expected_path = Path("validation/summaries") / f"{experiment}.json"
        if not actual_path.exists():
            failures.append(f"missing {actual_path}")
            continue
        actual = json.loads(actual_path.read_text())
        expected = json.loads(expected_path.read_text())

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
