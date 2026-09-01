# Legacy research code

These files are retained for provenance and historical comparison. They are **not** the supported reproduction interface.

## `toy/`

Original exploratory toy scripts and helper modules. Known limitations include import-time execution, implicit parameters, inconsistent force/discretization variants, and the biased same-sample KDE diagnostic. Use the repository-root `run.py` and `gsi_reproducer/` instead.

## `bnn/`

Historical Bayesian neural-network experiment and its UCI-style data files. This uses an older Theano/pandas stack that is deliberately not mixed into the locked toy environment. The machine-specific dataset path was replaced with a path relative to `legacy/bnn/`, and broken uploaded line continuations in the ASVGD evaluation expression were restored to their intended parenthesized form. No algorithmic BNN changes or new performance claims are made.

Run historical scripts from their own directory so local imports and output paths resolve there. Their results are not covered by the root verification or paper-toy regression checks.
