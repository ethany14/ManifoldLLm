# IPIP-NEO-300 M1 pilot: completed development results

This is the original single-seed 8D result. The completed [fixed-data 8/16/32D comparison](IPIP_DIMENSION_COMPARISON.md) adds three-seed dimension evidence and provisionally recommends 16D for the geometry-focused implementation. The original pilot and its files remain unchanged.

## What was implemented

We trained the selected learned low-dimensional manifold method on **300 keyed personality questionnaire items**. A masked nonlinear variational encoder produces an 8-dimensional personality coordinate, and a nonlinear decoder reconstructs item responses. Its Jacobian defines `G(phi)=J_g(phi)^T J_g(phi)+1e-4 I`. Decoder-path lengths participate in split-form contrastive training; this is not an Isomap-coordinate regression or a frozen-LLM adapter.

This answers the current implementation question: **how to represent multidimensional personality evidence as a learned coordinate with an operational decoder-induced geometry**. It does not yet implement language behavior, temporal personality updating or a validated digital agent.

## Run and split

- Johnson (2005) sample inside the supplied official package: 20,993 original response cases.
- After prespecified missingness/reliability screens and exact-vector deduplication: **20,479 cases**.
- Train 14,335; development 3,071; reserved test 3,073. Test cases were **not scored**.
- Seed 42, 30 epochs on the local RTX 5070 Ti Laptop GPU; best development checkpoint at **epoch 25**. Total script time approximately 56 seconds.
- Original ZIP unmodified. Source hashes, source-row indices, preprocessing, fixed 150/150 item forms and checkpoint saved locally.

The full protocol is in [IPIP_M1_PROTOCOL.md](IPIP_M1_PROTOCOL.md). These are response-case splits, not verified unique-person splits: exact duplicates are removed but repeated submissions with changed answers cannot be identified from this file.

## Predicting unobserved items

Each of 30 facets contributes five input A items and five hidden B items. On development cases, the encoder sees **only A**. Scores below refer to the hidden B answers. Facet/domain scores are means of observed B answers, not totals incorporating visible A answers.

| Method | Hidden item RMSE (1–5) | Macro item R² | Hidden facet macro R² | Hidden domain macro R² |
|---|---:|---:|---:|---:|
| Training-item mean | 1.1509 | -0.0003 | 0.0004 | 0.0004 |
| PCA 8D + ridge | 0.9475 | 0.3210 | 0.5601 | 0.8204 |
| **M1 nonlinear manifold 8D** | **0.9371** | **0.3358** | **0.5800** | **0.8305** |
| Raw A 150D + ridge | 0.8815 | 0.4118 | 0.6638 | 0.8509 |

M1 improves item RMSE by approximately **1.10% relative to the 8D PCA route**, but the unrestricted 150D ridge baseline predicts better. This supports a working compact nonlinear representation, not a claim of overall prediction superiority. Only one seed was run and no paired uncertainty interval was estimated. Ridge uses train-mean-imputed missing training targets, whereas M1 masks missing targets, so a strict comparison should align that treatment.

These R² scores measure **split-form prediction within one questionnaire occasion**. They are not external behavioral validity, causal personality structure, or demonstrated temporal stability.

## Geometry and split-form consistency

In a fixed gallery of the first 256 development cases, matching A-view to B-view posterior means gives:

| Readout on the same trained model | Top-1 split-form matching |
|---|---:|
| Chance reference | 0.39% |
| Coordinate Euclidean distance | 79.69% |
| Decoder-path distance, 8 segments | 80.86% |

The decoder-path readout makes three additional correct matches out of 256. This is a **small descriptive difference**, not evidence that geometry is superior: both readouts use one geometry-trained model and there is no matched geometry-free training ablation in this run. Shared response style and same-occasion effects may contribute. The gallery is an engineering check, not a real-world identity benchmark.

At 32 development coordinates, sampled metric eigenvalues range from **1.535 to 15.055**, with median condition number **4.266**. The mean Jacobian directional finite-difference discrepancy is approximately **0.00056**. This checks operational numerical geometry. Positive eigenvalues do not establish intrinsic curvature or psychological meaning; the ridge guarantees a positive floor.

Lengths follow straight latent paths, not optimized shortest geodesics. The metric is calculated from decoder means in standardized item units. It does not include the full stochastic generative-model metric.

## Important posterior limitation

Average posterior variances by dimension are approximately **0.00034–0.00046**; some reach the `logvar=-8` lower bound. Thus this run behaves close to a deterministic autoencoder, despite its Gaussian posterior and KL term. **Do not present its posterior uncertainty as calibrated.** A later uncertainty-focused experiment should fix a KL/variance protocol and test calibration; changing that setting here would be a separate run rather than silently replacing the result.

## Current conclusion and next step

The method is now implemented on an explicitly high-dimensional personality questionnaire: **300-dimensional personality evidence → masked nonlinear encoder → 8-dimensional coordinate → nonlinear decoder → decoder-induced Riemannian metric**. It can infer coordinates from partial answers and use decoded path lengths for similarity. No Isomap mapping or language model is required for this foundational stage.

Keep M1 as the selected implementation route on task-fit grounds, while stating that this first run does not establish optimal predictive accuracy or a complete digital personality. Next priorities are a matched 8D Euclidean/geometry-free training ablation on the same split, additional seeds, and a posterior/ordinal-decoder check. Use the reserved test only after freezing those decisions. A later digital-personality agent needs independent behavioral/longitudinal evidence and a trained behavioral or language decoder; questionnaire coherence alone is insufficient.

## Reproducibility and literature

Local aggregate source: `artifacts/ipip/m1_seed42/development_results.json`. Exact executed source is preserved as `training_source.py` and matches the SHA-256 recorded in `protocol.json`. The live script received only post-run scoring initialization, singleton-batch robustness and source-snapshot/logging changes; the original run and checkpoint were not overwritten.

- Kingma & Welling (2014), *Auto-Encoding Variational Bayes*. https://arxiv.org/abs/1312.6114
- Arvanitidis, Hansen & Hauberg (2018), *Latent Space Oddity: On the Curvature of Deep Generative Models*. https://arxiv.org/abs/1710.11379
- Johnson (2005), *Ascertaining the validity of individual protocols from web-based personality inventories*. https://doi.org/10.1016/j.jrp.2004.09.009
- Official data/scoring package. https://doi.org/10.23668/psycharchives.2448
