# GSI ASVGD toy-experiment reproducer

This is a clean, import-safe reconstruction of the toy experiments from **Accelerated Stein Variational Gradient Flow**. It is intended for reproducing the paper-scale qualitative behavior and for controlled experimentation.

## Important scope statement

The original arrays and exact edited source for every non-Gaussian PNG were not preserved. Therefore this package is a **paper reconstruction**, not a claim of bitwise historical reproduction. It uses:

- settings stated in the paper where consistent;
- settings verified from the surviving code;
- coordinate orientations measured from the published figures;
- the mathematically verified generic RBF acceleration formula.

The folder `reference_figures/` contains the published figures for visual comparison. `PROVENANCE.md` lists all known ambiguities.

## Why a quick run can look completely different

That is plausible and is not, by itself, evidence of an implementation error:

- paper figures use **500 particles and 1000 updates**;
- mode exploration in double bananas takes hundreds of updates;
- reducing particle count changes the empirical kernel operator, not just plot resolution;
- narrow bandwidth `sigma=0.1` is especially sensitive to particle spacing;
- KDE curves are unreliable at small particle counts;
- trajectory clouds with 20--80 particles cannot visually resemble 500-particle figures.

However, the earlier edited script also had genuine usage problems: importing it launched two unintended runs, the suggested command omitted the paper's banana damping and initialization, and saving failed if `results/` did not already exist. So either explanation—or both—is plausible.

## Installation

Install [uv](https://docs.astral.sh/uv/), then from this directory run:

```bash
uv sync --frozen
uv run python verify.py
```

No experiment runs during import.

## Reproduce all paper-scale toy experiments

```bash
uv run python run.py --experiment all --profile paper --output results/paper
```

This runs `N=500`, 1000 updates for:

1. Gaussian target with generalized bilinear kernel (`A=I` and `A=Q^{-1}`);
2. convex non-Lipschitz target;
3. double bananas with `beta=0.985`;
4. anisotropic Gaussian target.

The validated run takes roughly 1--3 minutes on a modern laptop because it deliberately evaluates the historical KDE score at every iteration.

Run one experiment:

```bash
uv run python run.py --experiment double_bananas --output results/bananas
```

## Quick debugging—not reproduction

```bash
uv run python run.py --experiment double_bananas --quick --output results/quick
```

The command prints a warning and records `paper_scale: false`. Do not compare this plot directly with the paper. In validation, this quick banana run ended at historical score `8.385` (80 updates), while the paper-scale run reached `-0.738` after 1000 updates and showed cross-mode transport.

Custom size:

```bash
uv run python run.py --experiment double_bananas \
  --particles 250 --steps 500 --seed 42 --output results/custom
```

## Restart and historical comparisons

The default reconstruction disables adaptive restarts for non-banana experiments. This matches the surviving source defaults and, at full scale, recovers the visible ASVGD acceleration. To test the restart policy literally declared in the paper:

```bash
uv run python run.py --experiment non_lipschitz \
  --profile literal-paper-restarts --output results/literal-restarts
```

The audited restart specification/code is inconsistent; in the corrected literal implementation the gradient condition triggers almost every step and suppresses acceleration. This profile is diagnostic rather than the recommended reproduction.

To use the literal surviving code's `max_time/subdiv=0.01` base step and `h=0.1` for every target:

```bash
uv run python run.py --experiment all \
  --profile surviving-code-steps --output results/code-steps
```

Historical public-code comparison:

```bash
uv run python run.py --experiment double_bananas \
  --profile legacy-public --output results/legacy
```

The legacy profile uses the specialized public RBF force (bare `score/N` rather than `K@score/N`) and disables restarts. It is included to expose provenance differences, not as the recommended algorithm.

## Outputs

Each experiment directory contains:

- `config.json`: complete resolved parameters;
- `environment.json`: Python/platform/package versions;
- `summary.json`: initial/final/minimum metrics and independent KDE-KL check;
- `results.npz`: metric arrays, initial particles, and final particles;
- `historical_metric.png` and `.pdf`;
- `trajectories.png` and `.pdf`.

Add `--save-trajectories` to put full checkpoint trajectories in `results.npz`; this substantially increases file size.

## Metric warning

The paper code's plotted quantity fits `scipy.stats.gaussian_kde` to the particles and evaluates it on those same particles. It also floors the target density at `1e-12`. For visual fidelity, `historical_metric.*` preserves two paper-code artifacts: the non-Lipschitz offset bug (`Z` was added instead of `log Z`) and the offset-only line left by a numerically failed, zero-initialized metric array. Numerical failures are explicitly recorded in `summary.json`. The score can become negative for other targets and disappear on a logarithmic axis, so it is retained only for historical comparison.

`summary.json` additionally reports an independently sampled estimate of

```text
KL(fitted KDE || normalized target)
```

with a Monte Carlo standard error. Use that field for interpretation, not the historical curve's nonnegativity.

## Default reconstruction choices

- Gaussian/bilinear figure: surviving-code scale `h=0.1`, `dt=0.01`;
- Gaussian-kernel figures: reported paper scale `tau=0.1`, hence ASVGD `h=sqrt(tau)` and base-method `dt=0.1`;
- the `surviving-code-steps` profile instead uses `h=0.1`, `dt=0.01` for all targets;
- RBF bandwidth `sigma=0.1`;
- continuum regularization `eps=0.1`, implemented as `K + N*eps*I`;
- intended generic RBF acceleration;
- momentum-first ordering, matching surviving figure-era code;
- adaptive restarts disabled for non-banana default runs, matching source defaults and visible acceleration;
- a separate `literal-paper-restarts` diagnostic profile;
- horizontal initial means `(5,0)` and `(7,0)`, as verified from figure pixels rather than the transposed coordinates in the caption.

See `PROVENANCE.md` before treating small numerical differences as failures.
