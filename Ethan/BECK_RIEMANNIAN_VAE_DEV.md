# Beck slow-fast VAE with decoder-induced Riemannian readout

Status: **exploratory, development-only**. This is a newly trained variational
model on the same Beck next-prompt studying task and the same saved participant
split as the earlier graph and decoder-metric pilots. It does not train on or
report the held-out test participants. The development people have been
inspected in prior work, so these numbers are not confirmatory.

## What changed

- A person-level posterior `q(phi | first-ten-prompt anchor)` and a prompt-level
  posterior `q(z | current-anchor deviation)` replace deterministic point
  encoders. One sampled `phi` is shared by all prompts of a person in each
  training step.
- The nonlinear decoder reconstructs 29 standardized situation, affect, and
  activity measurements; a separate head predicts next-prompt self-reported
  studying. Training uses binary cross-entropy + 0.05 reconstruction MSE +
  0.002 KL. Model selection uses **base-predictor development Brier only**.
- The decoder induces a local metric on the four-dimensional person coordinate:
  `G(phi) = J_phi g(phi, z=0, time=0)^T J_phi g(phi, z=0, time=0)`.
  Decoder-path lengths approximate graph-edge lengths. Shortest paths over a
  fixed kNN graph give the operational distance used in neighbor readout.
- Euclidean and Riemannian readouts use the **same trained checkpoint**, people,
  training-person rates, kNN topology, `k=8`, and fixed 0.2 mixture. Only
  edge lengths change. Three seeds are 1, 2, and 3.

The model does **not** target Isomap coordinates.

## Development results

105 train people / 2,680 prediction pairs; 35 development people / 871 pairs.
Brier is lower-is-better. Values below are the **mean of per-seed Brier scores**,
not the score of an ensemble.

| Readout | Mean development Brier |
|---|---:|
| Base slow-fast VAE | 0.184051 |
| Same VAE + Euclidean graph readout | 0.182853 |
| Same VAE + decoder-metric graph readout | 0.182691 |

The Riemannian-minus-Euclidean Brier differences were -0.000473, -0.000030,
and +0.000016 for seeds 1, 2, and 3. Thus the mean difference is small and
not directionally consistent. It is **not evidence that Riemannian personality
geometry beats Euclidean distance**. Sampled person-coordinate decoder
Jacobians were nondegenerate, but that check does not establish meaningful
curvature, construct validity, or external behavioral value.

## Limits and next experiment

The outcome is a self-report, not independently observed behavior. The graph
topology is selected by Euclidean kNN; only edge costs use the decoder.
The metric uses one fixed reference state/time and has not been stress-tested
across contexts. Geometry participates in the readout **after VAE training**;
this is not end-to-end geometry-aware training. Development data are reused.
We should next predefine several reference contexts and compare the metric's
stability, implement matched geometry-on/geometry-off training if warranted,
and then evaluate on a new untouched participant cohort with independently
assessed outcomes.

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe Ethan\run_beck_riemannian_vae.py
```

Exact per-seed scores, config, audit, and metric diagnostics are in
`Ethan/artifacts/beck_behavior_pilot/riemannian_vae_dev_results.json`.
