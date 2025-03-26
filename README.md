# Accelerated Stein Variational Gradient Flows

These python scripts evaluate and plot the discretized accelerated Stein variational gradient flow (with Wasserstein-2 regularization) as described in the preprint [TODO].
Currently, only the Gaussian (or: RBF) kernel and the generalized bilinear kernel $K(x, y) := x^{\mathsf{T}} A y + 1$ are available, but more will be implemented soon.

The file targets.py contains a custom class ```target```, which specifies the target distribution (its density, score, normalization constant...) and allows you to define your own target distributions.
Auxiliary functions are collected in plotting.py, adds.py.

If you use this code please cite this preprint, preferably like this:
```
@unpublished{SL25,
 title={Accelerated Stein Variational Gradient Flow},
 author={Stein, Viktor and Li, Wuchen},
 note = {ArXiv preprint},
 volume = {arXiv:2503.TODO},
 year = {2025},
 month = {Mar},
 url = {https://arxiv.org/abs/2503.TODO}
 }
```

Feedback / Contact
---
This code is written and maintained by [Viktor Stein](viktorajstein.github.io). Any comments, feedback, questions and bug reports are welcome! Alternatively you can use the GitHub issue tracker.

Required packages
---
* numpy
* matplotlib.pyplot
* scipy
* tqdm
* pingouin

Supported targets
---------------------------
* bananas (TODO, see [WL2020])
* GMM_scale: a Gaussian mixture, where one Gaussian has very high variance and the other very low.
* nonLip: $\pi(x) = \frac{2}{\Gamma\left(\frac{1}{4}\right)^2} \exp\left(-\frac{1}{4}(x_1^4 + x_2^4)\right)$.
* cauchy: $\pi(x) = \frac{1}{\pi} (1 + \| x \|_2^2)^{-2}$.
* skewed_Gaussian: a multivariate normal distibution with mean $[1, 1]$ and anisotropic variance $Q = \text{diag}(10, 0.05)$.
* GMM: the convex combination of two normal distributions with unit variance and means given by a vector $a$ and $- a$, currently set to $a = [1, 1]$: $\pi = \frac{1}{2} \mathcal{N}(a, \text{id}) + \frac{1}{2} \mathcal{N}(-a, \text{id})$
* U2, U3, U4: see [TODO].


--------------------------
Copyright (c) 2025 Viktor Stein

This software was written by Viktor Stein. It was developed at the Institute of Mathematics, TU Berlin.

This is free software. You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version. If not stated otherwise, this applies to all files contained in this package and its sub-directories.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
