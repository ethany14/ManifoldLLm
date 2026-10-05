# Full EmoBank: Isomap-free decoder metric pilot

Status: **exploratory, development-only sentence-affect experiment**.
This complements, but cannot be pooled with, the Beck person-level behavior
pilot. It does not establish a digital personality manifold.

## Data and protocol

- Same full prepared EmoBank file and aligned frozen Qwen3-4B embeddings as the
  earlier full-data experiments: 8,062 official training sentences, 999 official
  development sentences. The official test split is **not scored** here because
  it was used repeatedly in earlier comparisons.
- A 16-dimensional nonlinear variational autoencoder reconstructs the 2,560-D
  standardized language features and predicts valence/arousal/dominance (VAD).
  Its objective is VAD MSE + 0.1 reconstruction MSE + 0.001 KL.
- No Isomap coordinate is generated or used. The decoder mean induces
  `G(z) = J_g(z)^T J_g(z)` locally. For evaluation, we approximate path length
  by summing decoded changes along five subdivisions of a straight latent
  segment. This is **not** an optimized global geodesic.
- The same trained VAE is evaluated with (a) its direct VAD head, (b) a
  Euclidean kNN readout, and (c) a decoded-path kNN readout. The two neighbor
  readouts use the same 32 Euclidean candidate neighbors and eight selected
  neighbors; only the distance/ranking changes. Model selection uses direct
  development VAD MSE, not a preferred neighbor result.

## Development results

Mean of per-seed VAD R² (higher is better):

| Seed | Direct VAD head | Euclidean kNN | Decoder-path kNN |
|---:|---:|---:|---:|
| 1 | 0.2827 | 0.2474 | 0.2500 |
| 2 | 0.2710 | 0.2570 | 0.2447 |
| 3 | 0.2762 | 0.2308 | 0.2320 |
| **Mean** | **0.2766** | **0.2450** | **0.2423** |

The decoder-path readout was slightly better than Euclidean in two seeds but
worse in one; its mean was lower. Five sampled training points per seed had
nonzero smallest decoder-Jacobian singular values, but this is only a local
rank diagnostic, **not** evidence of global curvature or affective validity.
These development scores should not be compared numerically with the earlier
*test*-set AE+VAD results as though they used the same endpoint.

## Interpretation

The experiment shows that a nonlinear decoder metric can be computed without
mapping to Isomap and can change out-of-sample VAD-neighbor predictions. It
does not improve mean VAD R² here. The VAD head outperforms both distance-based
readouts. Geometry is applied after VAE training, not optimized end-to-end.
EmoBank records are isolated sentences with no stable person ID or future
behavior, so this experiment cannot answer the personality question.

Run from repository root:

```powershell
.\.venv\Scripts\python.exe Ethan\run_emobank_decoder_metric_dev.py
```

Machine-readable per-seed metrics and local rank checks:
`Ethan/artifacts/emobank_full/decoder_metric_dev_results.json`.
