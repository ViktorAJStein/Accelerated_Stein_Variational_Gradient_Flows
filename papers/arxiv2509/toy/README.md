# arXiv:2509.04008 toy experiments

## Verified

- All 20 preprint toy images are byte-identical to the GSI 2026 images.
- The historical archive contains exact copies of 19; `KL_Gauss.png` has separate exact local provenance.
- The archive recovers path-encoded configurations for the four published figure families.
- Eleven Gaussian generalized-bilinear configurations retain raw `(1001,500,2)` particle arrays and `(1001,)` KDE-score arrays.

## Not recovered

- raw arrays for non-Lipschitz, banana, and anisotropic published runs;
- exact figure-generating source revisions;
- per-run environments and logs.

Run the shared supported reconstruction from the repository root. Configuration evidence is in [`../../shared/toy_published_runs.json`](../../shared/toy_published_runs.json). The committed reference images remain under [`../../../reference_figures/`](../../../reference_figures/).
