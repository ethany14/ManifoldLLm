# Development-only geometry-in-training comparison

This follow-up addresses a limitation of the previous analysis: the
decoder-induced manifold distance was computed only after training. This
protocol is fixed before running the comparison. Earlier analyses have
already used the development cohort, so the result remains exploratory.
The 69-person test set stays sealed.

## Matched conditions

Reuse the same PsychArchives source files, person split (318 train, 68
development), 22 cues, missingness handling, training-only preprocessing,
`SequenceManifold` architecture, optimizer, and three seeds (1, 2, 3).
Train each condition for exactly 30 epochs with minibatches of 32 people,
no development-based checkpoint selection. Keep all existing next-valence,
reconstruction, trait, transition, stability, and KL losses unchanged.

For each minibatch, independently encode the person's first five and last
five prompts (nonoverlapping, recurrent state reset). The diagonal of the
early-to-late distance matrix is the same person; off-diagonal entries are
other people in that minibatch. Compare:

1. `base`: no new identity loss.
2. `euclidean_identity`: symmetric InfoNCE identity loss using Euclidean
   distance between 4-D slow coordinates.
3. `decoder_path_identity`: the **same** symmetric InfoNCE loss, using the
   length of the straight coordinate path after nonlinear decoding with
   fast state fixed to zero. Approximate length with four equal segments.
   This path length is induced by the decoder's local pullback metric but
   is not an optimized geodesic.

Both active losses use weight **0.02**, temperature **0.2**, and divide
their minibatch distance matrix by its detached mean off-diagonal distance.
This normalization makes the contrastive logit scale comparable while
allowing gradients through the distances. No tuning on development scores.
The contrastive objective uses person identity, **not** personality trait
labels; an improvement in identity retrieval alone would not establish a
digital personality.

## Fixed readouts

Reuse the previous evaluation's metrics without changing thresholds:
next-questionnaire valence RMSE; per-trait and macro R² plus pooled RMSE
from a `Ridge(alpha=1)` probe fitted on train people's final slow states;
and first-five-to-last-five same-person rank-1 matching and within/between
distance ratios on development people. Report both Euclidean and eight-
segment decoder-path matching for **every** condition, regardless of its
training objective. Compare per-seed values, not only averages. Do not
write person IDs, rows, or coordinates into the aggregate result.

Success would require geometry-trained representations to improve more
than the Euclidean-identity control on both individual stability and trait
readouts without a material next-valence regression. If only identity
matching improves, the method may be learning a sensing/device fingerprint
rather than personality. No confirmatory claim is possible on this reused
development set.
