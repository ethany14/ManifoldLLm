# IPIP-NEO-300 learned personality manifold: first-run protocol

## Purpose and method

Implement a learned low-dimensional personality manifold from high-dimensional personality observations. This is the selected **M1 nonlinear variational encoder/decoder** method, adapted from the earlier longitudinal prototype to a cross-sectional questionnaire. There is no Isomap target, pretrained language encoder, LLM training, fast emotional state, or temporal transition in this run.

The masked encoder takes 300 standardized keyed item values and 300 observation indicators, passes through 256/128-unit SiLU layers, and returns an 8-dimensional Gaussian posterior. The decoder maps coordinates through 128/256-unit SiLU layers back to the 300 standardized items. Posterior means define the operational coordinate. Eight dimensions is a fixed pilot setting, not an intrinsic-dimension estimate or a claim that personality has eight traits.

The decoder induces the local metric:

`G(phi) = J_g(phi)^T J_g(phi) + 1e-4 I`.

Distances used in split-form contrastive training sum four decoded increments along a straight coordinate path, including the same ridge contribution in each segment. This approximates length under the regularized pullback metric; it is **not an optimized shortest geodesic**. Diagnostics use 8 segments for case matching and 32 for sampled pair lengths. Geometry is in training-standardized item units, not a validated psychological distance unit. The metric uses the deterministic decoder mean; it is not the full uncertainty-aware stochastic metric from the latent-geometry literature.

## Local source and cleaning

Source: Johnson (2005), `ipip20993.dat`, within `Kajonius_Johnson_2019_SUPPL.zip`. The script also reads `DAT20993.doc` and `IPIP-NEO-ItemKey.xls` directly from the nested archives and records SHA-256 hashes. The item key's **Full#**, not row order or Short#, identifies each of the 300 items and its facet. The DAT codebook states reverse-scored answers were already recoded: **do not reverse them again**. Sex, age, supplied domain/facet totals and reliability indices are not model inputs. Reliability indices are used only for filtering.

Rules fixed before fitting:

- Values 1–5 are observed; zeros, nonnumeric and out-of-range entries are missing.
- Keep at least 270 valid item responses and at least 8/10 responses in each of 30 facets.
- Keep supplied `RNDSPLIT >= 0.5` and `RNDCORR >= 0`. These are conservative exploratory engineering screens, not official diagnostic quality thresholds.
- Remove exact duplicate keyed response vectors, including missingness, before splitting. Distinct vectors may still be repeated submissions by one person; verified participant identity is unavailable.
- Seed 42 splits retained **response cases** 70/15/15 into train/development/reserved test. This is not guaranteed to be a unique-person split.
- Fit per-item mean and standard deviation on training cases only. Missing normalized values are zero-filled and accompanied by an observation mask. Missing targets do not enter the neural reconstruction loss.

Observed audit: 20,993 original cases, 20,511 eligible cases, 32 exact duplicates removed, **20,479 retained**. Split counts: **14,335 train / 3,071 development / 3,073 reserved test**. Retained item missingness is approximately 0.525%. These counts are produced by the script, not assumptions.

## Training and model selection

Seed 42, 30 epochs, batch 256, AdamW learning rate 0.001, weight decay 0.0001. Randomly hide 30% of observed inputs. Objective:

`masked reconstruction + 0.5 split-form reconstruction + 0.001 KL + 0.02 symmetric split-form decoder-path InfoNCE`.

Two complementary fixed forms each contain 5 randomly selected items from every facet (150 items per form). Each view reconstructs the questionnaire; the geometry objective aligns posterior means of the two views of the same response case against other cases within a 32-case subbatch. Distance normalization retains gradients. Contrastive temperature is 0.2. Partial views of the same questionnaire are not independent temporal observations; response style may contribute to matching.

This is a VAE-inspired masked-denoising model with auxiliary losses, **not a pure ELBO VAE**. Treating ordinal items as continuous MSE targets is an explicit pilot simplification. A cumulative-link/ordinal decoder is a later option, not silently assumed here.

The best checkpoint is selected solely by development standardized B-item MSE given A inputs. Only 150 A items are available to the encoder for this evaluation. The 150 B answers never enter the evaluation encoder, though train-case B answers are legitimate training targets. No model selection or evaluation uses the reserved test cases.

## Reported checks

1. Hidden-B item RMSE on the original 1–5 scale and macro item R².
2. Hidden-B facet and domain mean-score R², computed only from observed B answers. This measures split-form score prediction, **not independent external personality validity**.
3. Full-input reconstruction, explicitly labeled as non-independent information preservation.
4. Fixed 256-case development gallery A/B matching using coordinate distance and decoder-path distance. This is split-form consistency, not temporal stability or real-world identification.
5. Sampled metric eigenvalues, condition numbers and Jacobian finite-difference check. A positive metric partly follows from the ridge and does not establish curvature or psychological validity.
6. Fixed train-item-mean baseline; 8D PCA of A inputs plus ridge to B; raw A-input ridge to B (alpha 10). Ridge uses train-mean-imputed missing B training targets while neural training masks them, so strict model comparisons need a shared missing-target treatment later.

Single-seed exploratory results establish an implementation milestone. They cannot select a universally best method or establish human-like digital personality. Future evaluation needs other seeds, matched geometry-free/Euclidean objectives, dimension sensitivity, ordinal-decoder sensitivity, external behaviors and eventually consented linked longitudinal data.

## Reproduction and products

Run from repository root:

```powershell
.\.venv\Scripts\python.exe Ethan\test_ipip_manifold.py -v
.\.venv\Scripts\python.exe Ethan\run_ipip_manifold.py
```

Output: `Ethan/artifacts/ipip/m1_seed42/`. Contains protocol, data audit, item metadata, prepared arrays with source-row indices and split assignments, posterior arrays, training history, `model.pt` and `development_results.json`. These participant-level files are Git-ignored. Preserve the original archive; do not overwrite a completed run. To repeat, use a different `--output` and record the settings. GPU calculations request deterministic algorithms with warning-only fallback; bitwise reproducibility across systems is not guaranteed.

## Literature lineage

- Kingma & Welling (2014), *Auto-Encoding Variational Bayes*: Gaussian posterior and nonlinear generative encoder/decoder. https://arxiv.org/abs/1312.6114
- Arvanitidis, Hansen & Hauberg (2018), *Latent Space Oddity*: decoder-induced latent geometry. This implementation adapts the deterministic mean-decoder pullback and adds a split-form contrastive objective; it does not reproduce that paper's full stochastic metric. https://arxiv.org/abs/1710.11379
- Johnson (2005), *Ascertaining the validity of individual protocols from web-based personality inventories*, Journal of Research in Personality 39, 103–129. https://doi.org/10.1016/j.jrp.2004.09.009
- Official dataset and scoring package, PsychArchives. https://doi.org/10.23668/psycharchives.2448
