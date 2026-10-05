# Decoder-induced manifold distance: development-only engineering check

## Scope and model label

This is a **deterministic engineering proxy** for the proposal's B2-to-M1 transition, not the formal B2 or M1 hierarchical VAE in [the protocol](PROPOSAL_ALIGNED_M1_PROTOCOL.md). It uses Beck's self-reported next-prompt `studying` task and the existing person split. Only the 105 training people (2,680 pairs) and 35 development people (871 pairs) contribute to fitting, model selection, or reported scores. The 36 test people's scores are deliberately not calculated. The development set has itself been used previously, so this is exploratory.

Three seeds train the same nonlinear slow-person / fast-state predictor with a 29-dimensional state-reconstruction decoder. A four-dimensional person embedding `phi` feeds that decoder. At fixed zero fast state and centered time, the decoder's Jacobian supplies a local pull-back metric; short latent segments are measured as length of their decoded paths, and shortest paths over an 8-neighbor person graph approximate longer distances. The graph is built from training-person embeddings; each development person is attached to training neighbors using only the first-ten-prompt anchor. The geometric readout averages training-person outcome rates by these graph distances and mixes 20% of this rate with 80% of the unchanged base predictor.

The key control uses the **same trained network, graph topology, k, weights, and 20% mixture**, but substitutes ordinary Euclidean edge lengths. Thus this pilot tests whether the decoder-induced metric adds value beyond a generic neighbor readout. Training-person outcome rates are allowed; no development-person outcome rate is used as a predictor. It is not yet an external behavioral-similarity validation and has no language or independent action data.

## Development results

Mean Brier across seeds 1, 2, and 3; lower is better:

| Prediction rule | Development Brier |
|---|---:|
| Deterministic slow–fast predictor | 0.18578 |
| Same predictor + Euclidean graph neighbor readout | **0.18427** |
| Same predictor + decoder-metric graph shortest-path readout | 0.18436 |

The manifold readout changes person-level retrieved rates relative to Euclidean retrieval (mean absolute differences 0.028, 0.032, and 0.035 across the three seeds), so the geometry is computationally active. However, its Brier is slightly **worse** than the matched Euclidean graph readout on average. Seed-level geometry-minus-Euclidean Brier differences are +0.000795, −0.000255, and −0.000262. Median smallest Jacobian singular values for a 30-training-person sample are positive (0.054, 0.090, 0.077), with median condition numbers 4.6, 4.9, and 12.8; this is a limited local-rank check, not proof of globally smooth or nonzero-curvature personality geometry.

## Decision

The decoder metric can be implemented and used in prediction, but this pilot **does not show an advantage over Euclidean distance**. Do not label it a validated M1 or choose hyperparameters by repeatedly examining the same development/test people. The next substantive step is the proposal's actual matched probabilistic B1/B2/M1 architecture, ideally with new consented longitudinal data containing scored psychological anchors and independent behavioral tasks. On Beck, any further implementation remains an engineering study; a fresh cohort is needed for a confirmatory claim.

Reproduce from repository root:

```powershell
.\.venv\Scripts\python.exe Ethan\run_beck_behavior_pilot.py
.\.venv\Scripts\python.exe Ethan\run_beck_decoder_metric_dev.py
```

Machine-readable configuration, per-seed scores, and metric-rank diagnostics: `Ethan/artifacts/beck_behavior_pilot/decoder_metric_dev_results.json`. Source: [Beck & Jackson (2022)](https://doi.org/10.1177/09567976221093307), [openESM harmonization](https://openesmdata.org/datasets/0060_beck/). Geometry motivation: [Arvanitidis et al., *Latent Space Oddity*](https://arxiv.org/abs/1710.11379); this citation does not claim their paper validates a personality manifold.
