# Validated expected results

Validation environment: Python 3.14.4, NumPy 2.4.6, SciPy 1.18.1, Matplotlib 3.11.1. Command:

```bash
uv sync --frozen
uv run python verify.py
uv run python run.py --experiment all --profile paper --output results/paper --quiet
```

Full validation completed in approximately 90 seconds on the audit machine. Small floating-point and Monte Carlo differences across platforms are expected.

## ASVGD summary (`N=500`, 1000 updates, seed 42)

| Experiment | Historical initial | Historical final | Historical minimum (iteration) | Independent final `KL(KDE||target)` |
|---|---:|---:|---:|---:|
| Gaussian, `A=I` | 2.4648 | 0.04007 | 0.04007 (911) | 0.0270 +/- 0.0028 |
| Gaussian, `A=Q^{-1}` | 2.4648 | 0.04007 | 0.04007 (914) | 0.0274 +/- 0.0028 |
| Non-Lipschitz | 31.1350 | 4.6782 | 4.6762 (397) | 0.0049 +/- 0.0009 |
| Double bananas | 26.4498 | -0.7375 | -0.7448 (562) | 2.4669 +/- 0.0488 |
| Anisotropic Gaussian | 11.7405 | 0.0757 | 0.0661 (389) | 0.0382 +/- 0.0019 |

The historical and independent columns intentionally answer different questions. In particular, the negative banana value is a same-sample KDE artifact and is not a true negative KL.

## Expected numerical failure

For the non-Lipschitz target at the paper's reported base step `dt=0.1`, ULA overflows at iteration 7. This is recorded in `summary.json`; the historical plot then displays the zero-initialized-array offset `6.5725`, reproducing the paper-code artifact. The package continues the other methods instead of crashing.

## Visual checks

- Gaussian ASVGD curves rapidly settle near the KDE floor `0.04`, as in `reference_figures/KL_Gauss.png`.
- Non-Lipschitz ASVGD settles near `4.7` only because the historical plot adds `Z` instead of `log Z`; the corrected independent estimate is near zero.
- Banana SVGD finishes near `1.0`; ASVGD becomes nonpositive on the historical log plot around the middle of the run; ULA/MALA concentrate on one mode while ASVGD/ULD explore both.
- Anisotropic ULA remains large/unstable, while ASVGD and the corrected stochastic methods approach the narrow target geometry.

Validated plot copies are in `validation/`; regenerate them rather than treating PNG pixels as the numerical baseline.
