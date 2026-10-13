# IPIP 16D M1: geometry-objective comparison

## Question and fixed design

With the selected 16-dimensional nonlinear encoder/decoder fixed, compare whether matching complementary questionnaire forms using decoder-induced path distances changes the representation relative to Euclidean matching or no contrastive matching. This isolates the **choice of training distance**, not nonlinear versus linear models or manifold existence versus nonexistence.

Conditions:

1. **Decoder path:** reconstruction + split-form reconstruction + KL + 0.02 decoder-path symmetric InfoNCE.
2. **Euclidean:** same, with coordinate-Euclidean distances substituted into the same normalized InfoNCE.
3. **None:** the same reconstruction, split-form reconstruction and KL; no contrastive term. It still learns a nonlinear decoder, so a pullback metric can be constructed and evaluated.

All conditions use the same 303,180-parameter 16D architecture, seeds 42/43/44, 30 epochs, AdamW settings, item dropout, mask handling, 150/150 facet-balanced forms and train-only scaling. The fixed split contains 14,335 training / 3,071 development / 3,073 reserved test response cases. Same-seed conditions initialize the same architecture and retain the same reconstruction-view forward calls. Path/Euclidean contrastive calculations themselves consume no random draws. A unit test verifies that the refactored decoder-path objective equals the previously executed objective exactly on fixed tensors.

The three completed decoder-path runs from `dimension_sweep_v1` are reused, not reselected. Six comparison models are trained. The runner checks exact split/form/scaling equality and exact saved training-source hashes. Every condition uses the same best-development-hidden-B-MSE checkpoint rule. This selects potentially different epochs and compares selected models under equal epoch budgets, not necessarily models after identical update counts.

## Geometry, readouts and limitations

The common decoder defines `G(phi)=J_g(phi)^T J_g(phi)+1e-4 I`. Decoder-path training sums four regularized decoded increments along straight coordinate paths. Matching uses eight segments. These are not optimized geodesics. Both Euclidean and path readouts are evaluated for **every** training condition on the same fixed 256-case development gallery.

Prediction scores use only form A inputs to predict hidden B responses. Facet/domain outcomes are means of hidden B answers, not externally measured behavior. Matching is consistency between two forms of the **same questionnaire occasion**, not longitudinal test-retest stability or real-world identity.

Results are exploratory development evidence after dimension selection on this same development set. Three-seed SD measures training variability, not sampling uncertainty; paired seed differences are descriptive and not a statistical significance test. The reserved test is not scored. Low posterior variance and ordinal-to-continuous MSE simplifications remain unchanged.

## Reproduction

```powershell
.\.venv\Scripts\python.exe Ethan\test_ipip_manifold.py -v
.\.venv\Scripts\python.exe Ethan\run_ipip_geometry_ablation.py
```

Local output: `Ethan/artifacts/ipip/geometry_ablation_v1/`, containing protocol, progress, per-run logs, new checkpoints and final `comparison_results.json`. Existing runs are not overwritten; participant-level outputs remain ignored by Git.

## Results

All nine condition/seed combinations completed: three existing path models and six new models. Split/form/scaling equality and saved source hashes passed verification. Test cases remain unscored.

### Prediction on hidden B items

Means over three seeds; ± is seed SD, not a confidence interval.

| Contrastive training distance | Hidden-item RMSE (1–5) | Macro item R² | Hidden facet macro R² | Hidden domain macro R² |
|---|---:|---:|---:|---:|
| Decoder path | **0.91035 ± 0.00027** | 0.3724 | 0.6278 | 0.8400 |
| Euclidean | 0.91065 ± 0.00036 | 0.3720 | 0.6273 | **0.8409** |
| None | 0.91181 ± 0.00017 | 0.3705 | 0.6253 | 0.8402 |

Path minus Euclidean average RMSE is approximately **-0.00031** (0.034% relative improvement), and path minus none is approximately **-0.00146** (0.161% relative improvement). These are very small descriptive differences. Path RMSE is lower in each of the three paired seeds, but that is not a sampling significance test and is insufficient for a claim of robust generalization superiority. Euclidean has the slightly higher hidden-domain R². No condition dominates every measure.

### Split-form matching and metric behavior

| Training condition | Coordinate-distance top-1 | Decoder-path top-1 | Mean per-seed median metric condition number |
|---|---:|---:|---:|
| Decoder path | 91.15% | 94.14% | 5.77 |
| Euclidean | **92.19%** | 94.14% | **4.35** |
| None | 90.89% | **94.40%** | 5.18 |

Gallery: 256 development cases; chance 0.39%. The two contrastive conditions have the **same mean path-readout accuracy**. None is slightly higher, by approximately 0.26 percentage points. Therefore the experiment does not show that adding decoder-path contrastive training improves this matching readout. The Euclidean-trained model also supports a decoder-induced metric and a useful path readout.

Path readout is higher than coordinate readout within each condition, but these are tiny-gallery, same-occasion development comparisons. They do not establish external psychological distance validity or a learned geodesic advantage. Lower metric condition number indicates less sampled anisotropy, not intrinsically more correct personality geometry.

Mean posterior variances are approximately 0.000345 (path), 0.000338 (Euclidean) and 0.000337 (none), near the lower-bound variance 0.000335. The uncertainty-calibration problem remains in every condition.

## Implementation conclusion: how to use manifold

Retain the **16D learned nonlinear encoder/decoder manifold** as the provisional core method:

1. Encode masked 300-item personality observations into a compact coordinate `phi`.
2. Train a nonlinear decoder `g(phi)` to predict observed personality evidence, including reconstruction from partial forms.
3. Define the operational local geometry using the decoder Jacobian, `G=J_g^T J_g+1e-4 I`, in explicitly standardized item units.
4. Use decoded path length for a geometry-aware similarity readout. If shortest-path behavior is needed, implement and verify an optimized/geodesic approximation separately; the current straight path is not that algorithm.
5. Add same-case split-form contrastive training **only as an optional objective**. Do not describe decoder-path InfoNCE as necessary for a manifold, or as empirically superior here.

The manifold construction is supplied by the learned decoder and its pullback metric; the training-distance choice is a separate design decision. All three conditions can support this construction. Defining a metric alone also does not prove that observed data lie on a psychologically meaningful smooth manifold; that requires further coverage, rank, validity and robustness evidence.

For continuity, the existing `dimension_sweep_v1/dim16_seed42/model.pt` remains the reference prototype, not a statistically proven winner. The new `geometry_ablation_v1/euclidean_seed42/model.pt` and `none_seed42/model.pt` are matched alternatives. Do not replace the reference checkpoint or promote one to a final model based on these repeatedly examined development differences alone.

The appropriate next implementation milestone is **operational geometry and inference**, rather than another assertion that manifold training is effective: verify path discretization/optimization, expose partial-answer coordinate inference with its feature schema and checkpoint, and separately address the near-deterministic posterior. Real behavioral/personality updating or language conditioning requires linked external/longitudinal evidence and additional training; none is established by questionnaire reconstruction.

## Method lineage and scope

- Kingma & Welling (2014), *Auto-Encoding Variational Bayes*: inspiration for Gaussian encoder/decoder training, adapted here with masking and auxiliary objectives. https://arxiv.org/abs/1312.6114
- Arvanitidis, Hansen & Hauberg (2018), *Latent Space Oddity*: inspiration for decoder-induced geometry. This implementation uses a regularized deterministic-mean pullback, not that paper's complete stochastic geometry. https://arxiv.org/abs/1710.11379
- Official IPIP data/scoring package: https://doi.org/10.23668/psycharchives.2448

The fixed-form InfoNCE combination is a project-specific adaptation, not a reproduction of either paper's personality experiment. Aggregate evidence is saved in `artifacts/ipip/geometry_ablation_v1/comparison_results.json`, including paired per-seed differences. Original source data and completed earlier runs were not modified.
