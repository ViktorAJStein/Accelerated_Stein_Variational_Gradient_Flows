# Accelerated Stein Variational Gradient Flows

These python scripts evaluate and plot the discretized accelerated Stein variational gradient flow (with Wasserstein-2 regularization) as described in the preprint [TODO].
Currently, only the Gaussian (or: RBF) kernel and the generalized bilinear kernel $K(x, y) := x^{\mathsf{T}} A y + 1$ are available, but more will be implemented soon.

The file targets.py contains a custom class ```target```, which specifies the target distribution (its density, score, normalization constant...) and allows you to define your own target distributions.
Auxiliary functions are collected in plotting.py, adds.py.

If you use this code please cite this preprint, preferably like this:
@unpublished{SL25,
 title={Accelerated Stein Variational Gradient Flow},
 author={Stein, Viktor and Li, Wuchen},
 note = {ArXiv preprint},
 volume = {arXiv:2503.TODO},
 year = {2025},
 month = {Mar},
 url = {https://arxiv.org/abs/2503.TODO}
 }

Feedback / Contact
---
This code is written and maintained by [Viktor Stein](viktorajstein.github.io). Any comments, feedback, questions and bug reports are welcome! Alternatively you can use the GitHub issue tracker.
