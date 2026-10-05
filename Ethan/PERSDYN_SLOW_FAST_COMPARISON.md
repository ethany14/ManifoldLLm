# PersDyn slow-fast and person-distance comparison

## Research question

Does a small hierarchical nonlinear model, with a fixed person-level representation from the first ten ratings and a rapidly changing deviation from that person's baseline, improve prediction of later self-rated personality states? Does an additional supervised person-distance loss improve either prediction or distance ordering?

This is an **exploratory proxy** for the proposed digital-personality method. It is **not** a variational autoencoder, a Riemannian metric, an LLM, or a test of independent behavior. The source is the [PersDyn study's open workbook](https://osf.io/t5v34/); provenance and cleaning are documented in `PERSDYN_PUBLIC_PILOT.md`.

## Exact comparison

We reused the previous 60-person dataset, 1,844 complete records, first-ten-observation calibration, and seed-42 participant split: 36 train people (742 prediction instances), 12 development people (255), and 12 test people (247). The test participants and their outcomes were already examined in the first-stage pilot, so these are development-stage results, not an untouched confirmatory test.

For each person, the mean of the **first ten** O/C/E/A/N states is held fixed as the slow input. A 5-to-4-dimensional nonlinear encoder produces `phi`. A separate 5-to-4-dimensional encoder represents the deviation of the previous state from the fixed baseline as fast state `z`. A shared decoder predicts the next state from `phi`, `z`, and time features. Both variants have identical parameters, initialization seeds (1, 2, 3), learning rate, maximum epochs, and early stopping on development-set MSE.

The geometry variant adds one loss only: pairwise distances between training participants' `phi` embeddings are encouraged to match pairwise distances between their *future mean self-ratings*. The weight is fixed at 0.002. No test-person future ratings are used in training. This is a Euclidean distance-alignment regularizer, **not** a learned curved manifold or geodesic model. It is a useful minimal ablation before implementing the proposed H-VAE and behavior-induced Riemannian metric.

## Results

| Model or reference | Test mean MAE (0-100) | Test person-distance Spearman |
|---|---:|---:|
| Prior pilot: running person mean | 12.549 | - |
| Prior pilot: Ridge with updated history features | **12.243** | - |
| Ridge with exactly the fixed-history model inputs | 12.647 | - |
| Slow-fast, no geometry; 3-seed mean | 12.988 +/- 0.016 | 0.344 |
| Same slow-fast network + person-distance loss; 3-seed mean | 12.986 +/- 0.016 | 0.357 |
| First-ten history mean, raw 5-d distance | - | **0.496** |
| First-ten history mean, train-fitted PCA 4-d distance | - | 0.389 |

The geometry loss changes distance ordering slightly, but its 0.002-point MAE difference is negligible. Both hierarchical variants are worse than both Ridge baselines on this split. The fixed-history Ridge uses **exactly the same input information** as the two neural variants; the earlier Ridge uses a continuously updated history mean and is an additional strong practical baseline. Even the geometry variant's 4-dimensional person-distance correlation is below a train-fitted 4-dimensional PCA representation of the first-ten history means. The across-seed standard deviation reflects model initialization only; it is not a confidence interval across participants or independent datasets.

We also reran the same protocol on participant split seeds 43-46, retaining model initialization seeds 1-3. Across five exploratory, overlapping person splits (42-46), mean test MAE was **12.113** for matched-input fixed-history Ridge, **12.652** for no-geometry slow-fast, and **12.652** for the geometry variant (difference -0.00024). Average test person-distance Spearman was 0.318 without geometry, 0.358 with geometry, and 0.412 for train-fitted 4-dimensional PCA of the first-ten means. Geometry improved prediction by tiny amounts on four splits and worsened it on one; it did not close the gap to the simple baseline. These split results are not independent replications, and the test people were inspected during exploratory model development.

## Interpretation and next decision

**No evidence here that the present nonlinear hierarchy or distance regularizer improves future-state prediction.** A more complex model is not justified by these results alone. However, this negative result does not rule out the full research proposal: only 60 people are available, `phi` is inferred from five self-rated dimensions rather than multimodal behavior, and the loss is a simple distance proxy rather than a behavior-induced geodesic. The dataset has no independent behavioral outcome or dialogue text, so it cannot validate an LLM-based digital personality.

The next useful step is *not* to tune the loss repeatedly against these same test participants. Seek an open longitudinal dataset with observed actions and context, set a fresh test before modeling, and report participant-level uncertainty. If the hierarchy does not clear a strong simple baseline there, defer the H-VAE/geometry expansion. If it does, compare Euclidean and decoder-induced geodesic variants under the same data and model capacity.

## Reproduce

After installing `Ethan/requirements.txt` in a writable environment, from `Ethan/` run:

```powershell
python .\run_persdyn_slow_fast.py
```

The script validates the original OSF workbook SHA-256 and writes `artifacts/persdyn_slow_fast/results.json`, including person IDs, run seeds, each run's metrics, and the distance diagnostics. To reproduce the additional splits, pass `--split-seed 43` (then 44-46) and a distinct `--output` path. The raw workbook remains in gitignored `data/persdyn/source/`.
