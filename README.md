# Accelerated Stein Variational Gradient Flows

Reference implementation and reproducible toy experiments for:

> Viktor Stein and Wuchen Li, **Accelerated Stein Variational Gradient Flow**, Geometric Science of Information 2026, pp. 80–90.
>
> [Springer](https://doi.org/10.1007/978-3-032-03921-7_9) · [arXiv:2503.23462](https://arxiv.org/abs/2503.23462)

## Reproduce the paper figures

The clean runner reproduces all four paper-scale toy studies with their published particle count and horizon:

1. Gaussian target with generalized bilinear kernels `A=I` and `A=Q^{-1}`;
2. non-Lipschitz quartic target;
3. double-bananas target;
4. anisotropic Gaussian target.

Requirements: Python 3.12–3.14 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/ViktorAJStein/Accelerated_Stein_Variational_Gradient_Flows.git
cd Accelerated_Stein_Variational_Gradient_Flows

uv sync --frozen
uv run python verify.py
uv run python run.py --experiment all --profile paper --output results/paper
uv run python check_reproduction.py --results results/paper
```

The full command uses `N=500`, 1000 updates, and seed 42. It takes roughly 1–3 minutes on a modern laptop and writes, for each experiment:

- `historical_metric.{png,pdf}` — the paper-code KDE diagnostic, including documented plotting artifacts;
- `trajectories.{png,pdf}` — initial and final particle states;
- `results.npz` — metric arrays and initial/final states;
- `summary.json` — settings, restart/failure counts, historical scores, and independent KDE-KL estimates.

Expected numerical summaries and validated plots are committed in [`EXPECTED_RESULTS.md`](EXPECTED_RESULTS.md) and [`validation/`](validation/). Published reference images are in [`reference_figures/`](reference_figures/).

### Run one experiment

```bash
uv run python run.py --experiment double_bananas --output results/bananas
```

Available names are:

```text
gaussian_bilinear
non_lipschitz
double_bananas
anisotropic_gaussian
all
```

### Quick smoke run

```bash
uv run python run.py --experiment all --quick --output results/quick
```

Quick mode (`N=80`, 80 updates) is for debugging only. It is not a paper reproduction: for example, its banana score and mode exploration differ dramatically from the full run.

## Reproducibility profiles

The historical paper text, surviving scripts, and published figures do not specify one completely consistent discretization. The runner therefore makes every consequential choice explicit:

| Profile | Purpose |
|---|---|
| `paper` | Recommended figure reconstruction using verified forces and figure-supported settings |
| `surviving-code-steps` | Uses the surviving scripts' `h=0.1`, `dt=0.01` uniformly |
| `literal-paper-restarts` | Diagnostic literal implementation of the declared restart condition |
| `legacy-public` | Historical public force/order behavior for comparison |

For example:

```bash
uv run python run.py --experiment non_lipschitz \
  --profile literal-paper-restarts --output results/restart-diagnostic
```

Please read [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) and [`PROVENANCE.md`](PROVENANCE.md) before interpreting small discrepancies. Exact historical non-Gaussian arrays and edited figure-generating source states are unavailable; this repository provides a validated **paper reconstruction**, not a claim of bitwise identity.

## Metrics

The original plots evaluate a Gaussian KDE on the same particles used to fit it. This quantity is biased, can be negative, and is not a trustworthy KL estimate. It is retained as `historical_metric` solely to compare with the paper.

`summary.json` also reports an explicitly labelled independent estimate of `KL(KDE || target)` using fresh KDE samples and correctly normalized targets. Historical and independent values should not be mixed.

## Verification

```bash
uv run python verify.py
```

The verification suite checks:

- analytic target scores against finite differences;
- corrected RBF force against direct finite-particle sums;
- regularization scaling;
- smoke execution of every target/method;
- finite outputs and expected output shapes.

A GitHub Actions workflow runs this verification and a quick end-to-end experiment on every push and pull request. Pushes to `main` and manually dispatched workflows also run all paper-scale experiments, check their summaries against tolerant committed baselines, and upload the regenerated plots.

## Legacy research scripts

The original exploratory files remain at repository root for provenance:

- `Particle_algo_Gaussian_kernel.py`;
- `Particle_algo_all_kernels.py`;
- `targets.py`, `adds.py`, and `plotting.py`;
- `bayesian_nn_ASVGD.py` and `data/`.

They may execute experiments at import time, rely on historical dependencies, and contain known discretization/metric inconsistencies. Use `run.py` and `gsi_reproducer/` for the reproducible paper experiments. The pinned toy environment does not attempt to modernize the separate legacy BNN stack.

## Citation

```bibtex
@InProceedings{SL2026,
  author    = {Stein, Viktor and Li, Wuchen},
  editor    = {Nielsen, Frank and Barbaresco, Fr{\'e}d{\'e}ric},
  title     = {Accelerated Stein Variational Gradient Flow},
  booktitle = {Geometric Science of Information},
  year      = {2026},
  publisher = {Springer Nature Switzerland},
  address   = {Cham},
  pages     = {80--90},
  isbn      = {978-3-032-03921-7},
  series    = {Lecture Notes in Computer Science},
  doi       = {10.1007/978-3-032-03921-7_9}
}
```

## Feedback

Questions and reproducibility reports are welcome through the GitHub issue tracker or [Viktor Stein's website](https://viktorajstein.github.io).

Licensed under the [MIT License](LICENSE).
