# Accelerated Stein Variational Gradient Flows

These python scripts evaluate and plot the discretized accelerated Stein variational gradient flow (with Wasserstein-2 regularization) as described in the preprint [Accelerated Stein Variational Gradient Flow](https://arxiv.org/abs/2503.23462).
Currently, only the Gaussian (or: RBF) kernel and the generalized bilinear kernel $K(x, y) := x^{\mathsf{T}} A y + 1$ are available, but more will be implemented soon.

The file targets.py contains a custom class ```target```, which specifies the target distribution (its density, score, normalization constant...) and allows you to define your own target distributions.
Auxiliary functions are collected in ```plotting.py``` and ```adds.py```.

If you use this code please cite this preprint, preferably like this:
```
@inproceedings{SL2026,
    author = {Viktor Stein and Wuchen Li},
    title = {Accelerated {S}tein Variational Gradient Flow},
    booktitle = {Geometric Science of Information. 7th International Conference, GSI 2025, Saint-Malo, France, October 29–31, 2025, Proceedings},
    publisher = {Springer Nature Switzerland},
    year = {2026},
    series = {Lecture Notes in Computer Science},
    address= {Cham},
    pages = {80--90},
    volume = {16033},
    editor = {Frank Nielsen and Frédéric Barbaresco},
    doi = {https://doi.org/10.1007/978-3-032-03921-7_9},
    eprint = {https://link.springer.com/chapter/10.1007/978-3-032-03921-7_9}
}
```

Feedback / Contact
---
This code is written and maintained by [Viktor Stein](https://viktorajstein.github.io). Any comments, feedback, questions and bug reports are welcome! Alternatively you can use the GitHub issue tracker.

Required packages
---
* numpy=1.22.3
* matplotlib=3.5.3
* scipy=1.7.3
* tqdm=4.67.1
* pingouin=
* theano=1.0.5 (only for the BNN example)
* pandas=1.5.3 (only for the BNN example)

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
