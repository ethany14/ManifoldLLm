# Geometry-in-training comparison: corrected normalization

This is a documented correction to
[the first geometry-training protocol](PSYCHARCHIVES_GEOMETRY_TRAINING_PROTOCOL.md),
not a confirmatory experiment. In the first run the mean off-diagonal
distance was detached before normalization. Although normalized logits
were finite, that implementation did **not** differentiate through the
normalizer and the decoder-path condition developed a much larger local
metric scale than either control. Its geometry-versus-Euclidean comparison
must be treated as a numerical pilot, not the method result.

Version 2 changes only this step: the contrastive logits divide pairwise
distances by the **differentiable** minibatch mean off-diagonal distance,
clamped below at `1e-6`. This makes uniform rescaling of all distances
cancel in both the forward loss and its gradient. The three conditions,
same-person positives, nonoverlapping five-prompt windows, four-segment
training path, contrastive weight 0.02, temperature 0.2, base losses,
seeds 1/2/3, fixed 30 epochs, split, and all development readouts remain
exactly as in version 1. No hyperparameter is selected by the new results.

Report sampled decoder-metric eigenvalue ranges and location variation
alongside the original valence, trait-probe, Euclidean matching, and
eight-segment decoder-path matching readouts. A decoder-path advantage
requires improvement beyond the Euclidean-identity control without
metric-scale inflation. Even then, identity contrastive training may learn
device/sensing fingerprints rather than psychological personality.

The development cohort has been inspected many times; these are engineering
diagnostics. The reserved test participants remain unopened. Licensed
source rows and individual coordinates are not exported.
