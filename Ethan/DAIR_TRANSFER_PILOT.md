# External emotion-transfer pilot: dair-ai/emotion

## Question and design

Does the 16-dimensional representation learned from EmoBank transfer to a
different emotion-labeling corpus, and does an affective local-geometry loss
improve that transfer? This is **not** a direct test of digital personality.

We used the official `split` configuration of
[dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion). A fixed,
label-stratified pilot sampled 800 training, 100 validation, and 100 test
examples (seed 42). Frozen `Qwen/Qwen3-4B` final-layer masked-mean embeddings
were extracted in the same way as for EmoBank. The representation models and
their preprocessing were fitted on EmoBank training data only. A logistic
regression probe was then fitted on the dair training labels; its regularization
was selected using dair validation macro-F1, and the selected classifier was
refitted on dair train + validation before one test evaluation. Thus this is
**linear-probe transfer**, not zero-shot classification.

The pilot CSV SHA-256 is
`9DB0E39CF5B9ED50E3F980C03D593F68B02A97ADAA41D65FAEFF04453BA623EA`.
The 1,000-row embedding array has shape `(1000, 2560)`. We found no exact
case-insensitive text overlap between EmoBank and the full dair corpus, nor
between the pilot train/test or validation/test subsets.

## Results

Test macro-F1 and accuracy, mean ± sample SD over EmoBank model seeds
42, 43, and 44. The fixed raw-Qwen and PCA results are repeated across seeds.

| Representation | Dimensions | Test macro-F1 | Test accuracy |
|---|---:|---:|---:|
| Raw Qwen | 2,560 | 0.423 ± 0.000 | 0.560 ± 0.000 |
| EmoBank PCA | 16 | 0.239 ± 0.000 | 0.450 ± 0.000 |
| AE + VAD, no geometry loss | 16 | 0.277 ± 0.037 | 0.507 ± 0.031 |
| AE + VAD, affective geometry loss | 16 | 0.258 ± 0.017 | 0.487 ± 0.015 |

The largest 16-dimensional AE result in an individual run was macro-F1 0.321
(seed 44, without geometry). The affective-geometry variant did not exceed it
on the same seed and did not exceed the no-geometry mean across three seeds.
Raw Qwen was strongest in all runs, although its dimensionality is much larger.

## Interpretation and next decision

This pilot does **not** support the claim that the current affective manifold
constraint improves external categorical-emotion classification. It does show
that some EmoBank-learned signal transfers through a 16-dimensional bottleneck,
but substantial information is lost relative to raw 2,560-dimensional Qwen.
The 100-example test set is small: it has only three `surprise` examples, and
all four methods received F1 = 0 for that class in seed 42. Differences between
16-dimensional methods should therefore be treated as preliminary.

The most informative next experiment is a larger, deduplicated dair test using
the same pretrained encoders, plus a fair same-dimensional control and multiple
seeds. For the *digital personality* research question, classification of
isolated emotion sentences remains insufficient: the team needs repeated
observations from the same people over time and an evaluation of stable
individual differences and state trajectories. A favorable emotion-transfer
result alone would not establish a digital personality manifold.

The per-seed CSV/JSON outputs are in `artifacts/dair_emotion/pilot_transfer*`.
Reproduction commands and source hashes are in `REPRODUCIBILITY.md`.
