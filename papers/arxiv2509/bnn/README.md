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

## Automatic paper-style summary tables

The legacy runner now writes `bnn_summary/` in its working directory after each
completed dataset, before interactive plotting. During the run these tables are
explicitly marked **partial**; after the final dataset, the complete cross-dataset
tables are also printed. The training, random seeds, metrics and timing boundaries
are unchanged. For every recorded iteration budget (including 2000 and 5000), rows
are datasets and columns are ASVGD/SVGD RMSE, test LL and elapsed seconds:

- `summary_<budget>.md` and `.tex`: three-decimal mean +/- population SD, `ddof=0`;
- `summary.md`: all budget tables;
- `summary.csv`: unrounded statistics, run counts and descriptive better-mean flags;
- `summary_metadata.json`: completion status, budgets, counts and actual model defaults/overrides.

LaTeX uses `booktabs` and captions below tables; it is not compiled automatically.
Lower RMSE/runtime and higher LL are bolded using unrounded means (both methods for
exact ties). Bold is **not** a significance test. No best-budget selection, run
exclusion or `nanmean` is used: nonfinite/incomplete arrays raise an explicit error.
Iteration counts are budgets, not a claim that early stopping never occurred.

You can export summaries from existing arrays without Theano or any new training:

```bash
uv run python legacy/bnn/bnn_summary.py --results /path/to/one/regime \
  --budgets 10 25 50 100 200 300 400 500 1000 1500 2000 5000 \
  --output results/bnn_tables
uv run python scripts/verify_bnn_summary.py
```

Supply budgets in their **stored array order**, and use separate output directories
for different regimes. Files must be named `results_{ASVGD,SVGD}_<dataset>.npy`,
with shape `(runs, budgets, 3)` and columns `(RMSE, LL, seconds)`. All detected
datasets are included unless `--datasets` explicitly selects a documented scope.
Saved-array export cannot infer training settings; its metadata says so.

This reporting feature does not establish BNN reproduction. In particular, the
current constructor defaults use different master step sizes (SVGD `1e-3`, ASVGD
`1e-6`) and 20 particles; they do **not** implement both manuscript regimes as
configured. The reporter records those actual defaults rather than labelling them
as the paper settings. A configuration/provenance repair and backend validation
remain separate work. The implementation is **Theano autodiff plus NumPy/SciPy**,
not a purely NumPy neural network. No Torch migration is included in this change.
