# Provenance and reconstruction choices

## Verified

- Paper scale: `N=500`, 1000 nominal updates, seed 42 in public code.
- Surviving code sets internal `tau=10/1000=0.01` and uses `sqrt(tau)=0.1` for ASVGD. The non-Gaussian published curves instead strongly identify the reported `tau=0.1`: it reproduces the banana SVGD endpoint near 1 and the anisotropic ULA instability. The Gaussian/bilinear run remains stable only at the surviving-code scale.
- Direct finite-particle sums verify the generic RBF force with `K@score/N` and interaction coefficient `1/(N^2 sigma^2)`.
- Continuum regularization `eps` implies `N(K+N eps I)^{-1}Y`.
- The published Gaussian KL image is byte-identical to archived `KL_comparison_6.png` and uses `A=Q^{-1}`.
- Its initial KL level `2.6` agrees with the code target covariance `[[3,-2],[-2,3]]`, not with interpreting this matrix as target precision.
- Pixel segmentation plus the public plotting convention shows positive horizontal initial displacement in the non-Lipschitz and banana figures. The caption's `(0,5)` and `(0,7)` are transposed relative to the images.
- Double bananas uses constant `beta=0.985` according to the paper.
- The non-Lipschitz normalizer in public code is wrong: it stores `Z` where plotting expects `log Z`. This package preserves that offset only in `historical_metric.*` for visual comparison; independent KDE-KL summaries use the correct normalized target.

## Historical ambiguities

- The exact non-Gaussian figure-generating source and arrays are missing.
- Surviving specialized code omits `K` from the RBF score term and historical versions halve the interaction. These formulas fail direct checks and do not produce the cross-mode banana transport visible in the paper.
- The article uses position-before-momentum; surviving code uses momentum-before-position. At constant damping they differ primarily by one saved position index, but arrays cannot match exactly.
- The article's restart counter pseudocode conflicts with its stated `alpha_k=k/(k+3)` schedule.
- Surviving restart code reads an uninitialized negative history index on its first speed check, applies gradient restart too late, and reverses the score sign in the Gaussian gradient condition.
- The historical same-sample KDE score is not a nonnegative KL estimator.

## Package policy

The default `paper` profile chooses mathematically verified RBF dynamics, figure-verified initial orientation, a correctly normalized independent metric, historical momentum-first ordering, target-specific time-scale evidence (`0.01` internally for bilinear Gaussian; reported `0.1` for Gaussian-kernel figures), and no adaptive restarts for the non-banana examples. Disabling restarts matches the surviving function defaults and is necessary to recover the visible full-scale acceleration; the corrected literal restart condition triggers almost every iteration and suppresses acceleration. The separate `literal-paper-restarts` profile exposes that discrepancy.

This gives an auditable working baseline, but it may not reproduce every raster pixel. The `legacy-public` profile is provided to expose implementation drift. Results from different profiles must not be mixed under one label.
