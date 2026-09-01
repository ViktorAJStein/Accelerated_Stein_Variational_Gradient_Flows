# Bayesian neural-network reproduction — pending

The historical implementation is [`../../../legacy/bnn/bayesian_nn_ASVGD.py`](../../../legacy/bnn/bayesian_nn_ASVGD.py), with datasets under `legacy/bnn/data/`.

It is retained for provenance but is not covered by the root `uv.lock` or CI.

## Required before supported status

- [ ] Verify dataset sources, redistribution terms, checksums, preprocessing, and exact train/test protocol.
- [ ] Select and lock a reproducible BNN environment (historical Theano container or a separately verified port).
- [ ] Record every seed and dataset split.
- [ ] Remove implicit output paths and add a parameterized CLI.
- [ ] Run a seconds-scale smoke test.
- [ ] Run one diagnostic dataset/configuration.
- [ ] Reproduce every reported dataset table at the paper's decision scale.
- [ ] Save raw per-run arrays, aggregate tables, logs, and runtime metadata.
- [ ] Add regression tolerances and CI smoke coverage.
- [ ] Document any discrepancy rather than silently changing metrics or preprocessing.

Do not create the `arxiv2509-repro-v1` tag until these checks are complete.
