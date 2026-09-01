# GSI 2026 reproduction

Paper: Viktor Stein and Wuchen Li, *Accelerated Stein Variational Gradient Flow*, Geometric Science of Information 2026, pp. 80–90, DOI [10.1007/978-3-032-03921-7_9](https://doi.org/10.1007/978-3-032-03921-7_9).

## Validated reconstruction

From the repository root:

```bash
uv sync --frozen
uv run python verify.py
uv run python run.py --experiment all --profile paper --output results/gsi2026
uv run python check_reproduction.py --results results/gsi2026
```

See [`../../EXPECTED_RESULTS.md`](../../EXPECTED_RESULTS.md), [`../../REPRODUCIBILITY.md`](../../REPRODUCIBILITY.md), and [`../../PROVENANCE.md`](../../PROVENANCE.md).

## Recovered historical configurations

A later audit of the historical experiment archive found byte-identical copies of 19/20 published image files. Exact path-encoded settings are in [`../shared/toy_published_runs.json`](../shared/toy_published_runs.json).

Important distinctions:

- the non-Lipschitz and banana figures use horizontal initial means `(5,0)` and `(7,0)`, not the transposed values in the prose;
- the published Gaussian trajectory uses `A=4Q^{-1}`;
- the exact non-Gaussian image paths identify restart/damping choices, but their raw arrays and exact executed source revision are absent;
- the supported `paper` profile remains a validated reconstruction and does not claim bitwise trajectory identity.

Historical exploratory scripts are retained under [`../../legacy/`](../../legacy/README.md), not used by the command above.
