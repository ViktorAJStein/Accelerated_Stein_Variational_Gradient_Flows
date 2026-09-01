# Reproduction status for arXiv:2509.04008

Primary source: Viktor Stein and Wuchen Li, *Towards understanding Accelerated Stein Variational Gradient Flow -- Analysis of Generalized Bilinear Kernels for Gaussian target distributions*, submitted 4 September 2025, [arXiv:2509.04008](https://arxiv.org/abs/2509.04008).

Section 6 of the preprint cites this repository for its numerical experiments.

## Toy experiments

The preprint's 20 toy image files are byte-identical to the GSI 2026 image set. Therefore the supported toy runner and verification are shared:

```bash
uv sync --frozen
uv run python verify.py
uv run python run.py --experiment all --profile paper --output results/arxiv2509-toy
uv run python check_reproduction.py --results results/arxiv2509-toy
```

Recovered exact figure configurations are recorded once in [`../shared/toy_published_runs.json`](../shared/toy_published_runs.json). Read [`toy/README.md`](toy/README.md) for the boundary between exact image provenance and reconstructed dynamics.

## Bayesian neural networks

The preprint also reports BNN experiments that are **not yet validated by the root environment**. The historical script and data are retained under [`../../legacy/bnn/`](../../legacy/bnn/), but they use a separate old Theano stack and no original output arrays/logs have been recovered.

See [`bnn/README.md`](bnn/README.md). Do not cite the current legacy script as a complete reproduction of the BNN tables until that checklist passes.

## Release policy

A release/tag named `arxiv2509-repro-v1` should only be created after the BNN table values have been independently reproduced in a locked environment. Until then, the toy path is supported and the BNN path is explicitly pending.
