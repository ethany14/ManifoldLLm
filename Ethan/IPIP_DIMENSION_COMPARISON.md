# IPIP M1: fixed-data dimension comparison

## Design

Compare **8, 16 and 32 latent dimensions** using the same nonlinear manifold method, cleaning rules, response-case splits, item forms and training-only scaling as the completed 8D pilot. Training seeds are 42, 43 and 44. The original completed 8D seed42 run is reused; eight additional models are trained. The runner verifies exact equality of source-row split indices, form A/B and scaling arrays for every run. Test cases remain unscored.

Data randomness and model-training randomness are separate: `--data-seed 42` fixes the data protocol, while `--seed` controls model training. This corrects the earlier script interface in which changing `--seed` would also change the split and item forms.

Each run uses the same 30-epoch budget and best development hidden-B standardized MSE checkpoint rule. Hidden layers remain 256/128 for encoding and 128/256 for decoding. Only the posterior dimension and connected layer sizes change. Parameter counts are 300,092 (8D), 303,180 (16D) and 309,356 (32D); this is not an exactly parameter-matched capacity experiment.

The KL term remains a mean over latent coordinates, with weight 0.001, as in the original pilot. No KL schedule, variance bound, loss coefficient, missingness policy or decoder type is retuned by dimension. This is a controlled exploratory capacity comparison, not a comprehensive hyperparameter search or intrinsic-dimension estimate.

## Measures and interpretation

- Main development criterion: hidden-B item RMSE (1–5) and macro item R² given only form A.
- Personality-content retention: hidden-B facet/domain mean-score R². These are same-occasion questionnaire outcomes, not independent external validity.
- Split-form consistency: Euclidean and decoder-path matching in the same fixed 256-case development gallery.
- Geometry numerical behavior: local metric eigenvalues and median condition number at the same first 32 development cases.
- Posterior behavior: mean variance; near-zero/clamped variance remains an uncertainty limitation, even if prediction improves.
- At each dimension: matching-dimension PCA(A)+ridge(B). Raw 150D ridge and train-item means are reference baselines. The previously documented neural/ridge missing-target difference remains.

Across-seed sample standard deviations measure initialization/training variability, **not sampling confidence intervals**. Results on the repeatedly inspected development set are exploratory. Choosing a dimension here does not turn this comparison into an untouched test evaluation, or establish that Riemannian geometry rather than added capacity causes an improvement.

## Reproduction

```powershell
.\.venv\Scripts\python.exe Ethan\run_ipip_dimension_sweep.py
```

Output directory: `Ethan/artifacts/ipip/dimension_sweep_v1/`. It holds the fixed sweep protocol, per-run logs, new model directories, progress and final `comparison_results.json`. The runner resumes completed runs but refuses to overwrite incomplete or existing training directories. Participant-level products remain ignored by Git.

The study does not include 64D yet: start with 16D and 32D as agreed, assess whether improvements justify the additional dimensions, then decide whether a separate 64D run is warranted.

## Results

All nine dimension/seed combinations are complete (eight newly trained models plus the original 8D seed42 run). Exact equality of split indices, item forms and scaling arrays was verified. Test cases remain unscored.

### Hidden-item prediction and personality-score retention

Values for M1 are means over three training seeds; ± is sample standard deviation over those seeds, not a confidence interval. PCA is effectively deterministic on the fixed data and its repeated seed values are not independent data replications.

| Latent dimension | M1 hidden-item RMSE (1–5) | Matching-dimension PCA+ridge RMSE | M1 macro item R² | M1 hidden facet macro R² | M1 hidden domain macro R² |
|---|---:|---:|---:|---:|---:|
| 8 | 0.93738 ± 0.00057 | 0.94750 | 0.3355 | 0.5790 | 0.8291 |
| 16 | 0.91035 ± 0.00027 | 0.91914 | 0.3724 | 0.6278 | 0.8400 |
| 32 | **0.89771 ± 0.00029** | 0.90051 | **0.3899** | **0.6471** | **0.8428** |

The raw 150D A-input ridge reference achieves RMSE **0.88150** on the same hidden B answers. Thus increasing manifold dimension improves prediction but still does not outperform that higher-dimensional linear predictor. M1 outperforms matching-dimension PCA descriptively at each tested dimension. The nonlinear advantage is small at 32D (approximately 0.31% lower RMSE), while increasing from 8D to 16D reduces M1 RMSE by approximately 2.88%; increasing from 16D to 32D gives a further 1.39% reduction.

### Geometry and partial-questionnaire consistency

| Dimension | Coordinate matching, mean | Decoder-path matching, mean | Mean of per-seed median metric condition numbers | Mean posterior variance |
|---|---:|---:|---:|---:|
| 8 | 80.08% | 81.51% | 4.26 | 0.000395 |
| 16 | 91.15% | 94.14% | 5.77 | 0.000345 |
| 32 | 93.75% | 96.22% | 159.73 | 0.000336 |

Matching uses the same fixed 256-case development gallery (chance 0.39%), with each case's two nonoverlapping item forms. It is **not longitudinal stability**. Both readouts use the same geometry-trained model, so these columns do not isolate the benefit of geometry during training.

The 32D condition-number summary also varies substantially across seeds (SD approximately 95.97, versus 0.15 for 16D). This means the sampled decoder-induced metric becomes more anisotropic: some coordinate directions change decoded item predictions much less than others. It does **not** by itself prove failed numerical computation or an invalid manifold; sampled metrics remain positive, and the ridge supplies a floor. Still, it is a relevant engineering concern if path optimization or geometry-based updates are central to the implementation. A comparison of optimized path convergence has not yet been run.

All dimensions retain very small posterior variances, near the model's `logvar=-8` lower bound (variance approximately 0.000335). Dimension expansion **does not solve uncertainty calibration**. These models behave close to deterministic encoders and should not be sold as calibrated probabilistic personality estimates.

### Current implementation recommendation

**Use 16D as the provisional primary M1 implementation; retain 32D as the higher-capacity predictive comparison.** The method itself remains the selected nonlinear learned manifold:

`300 keyed questionnaire items + observation mask -> nonlinear encoder -> 16D person coordinate -> nonlinear decoder -> pullback Riemannian metric -> decoded-path similarity`.

Sixteen dimensions already capture substantially more hidden-item/facet information than 8D and give more consistent split-form coordinates, while sampled metric conditioning remains much better than 32D. Thirty-two dimensions are preferred **if hidden-item prediction alone is the criterion**. The recommendation for 16D is an explicit engineering tradeoff motivated by operational geometry, not a prespecified statistical decision rule or proof that 16D is intrinsically optimal. A 16D coordinate is not equivalent to 16 established personality traits.

If proceeding with a single reproducible working checkpoint, `artifacts/ipip/dimension_sweep_v1/dim16_seed42/model.pt` is available; seed42 is the original convention, not a seed chosen for having the best score. That model selects its checkpoint using development data, as do all other runs. The 32D mean selected epoch is 28.7 of 30, so more training could change its performance and conditioning; that would require a separately recorded comparison rather than revising this run silently.

Next, keep this split fixed and compare the selected 16D architecture with the same reconstruction/KL/form objectives but Euclidean or no contrastive geometry. Investigate the posterior-variance floor and, if optimized geodesics are required, test path discretization and optimization under the actual regularized metric. Independent behavioral or longitudinal evidence is still needed before calling the representation a complete digital personality. Reserve the test for the frozen final protocol.

### Saved evidence

Aggregate: `artifacts/ipip/dimension_sweep_v1/comparison_results.json`; per-run logs, checkpoints, protocol, exact training-source snapshots, posteriors and splits remain local and Git-ignored. The original 8D pilot was not overwritten. The scripts do not train or evaluate an LLM.
