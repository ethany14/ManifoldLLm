# Article Fidelity and Extension Register

This register separates source-paper equations, pre-existing adaptations, and new
research extensions. None of the variants is a complete reproduction of the paper.

Source: *Anchoring the Affective Manifold: Learning Canonical and Disentangled
Representations via Generative Cross-Modal Alignment*, Section 3, PDF pages 3-4
(printed pages 41607-41608): https://aclanthology.org/2026.acl-long.1929.pdf.
The equation mapping was checked against the supplied PDF. No claims of matching
the paper's results, training hyperparameters, or data are made.

## Retained Mathematical Forms

| Source | Current implementation | Verification |
| --- | --- | --- |
| Eq. 1-3 | Separate diagonal-Gaussian affect/private posterior heads; affect dimension 3 | `model.py`, `test_model.py` |
| Eq. 4 | Reparameterized sampling in training; means in evaluation | `manifold_losses.py`, equation and model tests |
| Eq. 5 | Squared feature reconstruction, summed across dimensions, averaged across examples | Equation tests and reconstruction gradient test |
| Section 3.5, ground-truth term | Squared VAD error on sampled affect coordinates | `training_loss` |
| Eq. 8-9 | Centered sample cross-covariance with denominator B-1, squared Frobenius norm | Equation tests |
| Eq. 10 | Analytic diagonal-Gaussian KL for both codes; beta applied once | Equation tests |
| Eq. 11 | Weighted sum through the original `total_objective` helper | Equation tests |

Factorization is a modeling assumption, not proof of learned independence.
Shared/private terminology is retained, but text-only data cannot demonstrate
modality invariance. The original `Alexander/manifold_losses.py` is unchanged.

## Existing Departures, Active in Every Variant

| ID | Where it differs | Reason and scientific consequence |
| --- | --- | --- |
| A1 | Section 3.2 pretrained backbones replaced with a randomly initialized embedding and GRU | Required no-LLM design. Features evolve during training; learned semantic quality differs. |
| A2 | Eq. 6-7 and Section 3.5 lexicon term are inactive, with weights and logged values zero | EmoBank has no paired audio/video or NRC-VAD pseudo-labels. This is neither cross-modal alignment nor dual anchoring. |
| A3 | Added next-token cross-entropy and a conditional text-output head | The paper reconstructs high-level features. Our additional language objective changes Eq. 11. Linear VAD conditioning is the baseline; G1 replaces that path. |
| A4 | Self-reconstruction target is `stopgrad(h)` | Prevents direct target-side shrinkage with a moving feature encoder, but changes gradient flow and does not guarantee against collapse. |
| A5 | Local engineering choices: EmoBank [-1,1] scores, 8D private code, linear posterior heads, clipped log variances, singleton covariance skipped, custom weights, posterior-mean validation | These are not claimed paper settings. Deterministic validation is not a sampled ELBO; the article specifies MLP posterior projections. |

## New Optional Training Extensions

### N1: Label-Neighbor Distance Matching

Location: `geometry.neighborhood_loss`, enabled only in named neighborhood variants.
For each mini-batch, select up to eight non-self nearest neighbors by human VAD.
Use posterior means for the predicted coordinates:

$$L_N=\frac1{|E|}\sum_{(i,j)\in E}
\left(\|\mu_{a,i}-\mu_{a,j}\|_2-\|y_i-y_j\|_2\right)^2.$$

Add `neighborhood_weight * L_N` to the existing objective (default weight 0.1).
This is our extension, not Eq. 7: there are no cross-modal reconstructions here.
It assumes Euclidean VAD label differences are useful targets and depends on batch
composition. It may reinforce annotation bias; it does not guarantee non-collapse.

### G1: Decoder-Based Causal Generation Conditioning

Location: `EmotionLanguageModel.forward`, `conditioning_mode="decoder"`.
Replace the baseline additive linear VAD conditioning with:

$$p_t=F_{\mathrm{priv},\mu}(h_t),\quad
\Delta h_t=D(a,p_t)-D(0,p_t),\quad
p(x_{t+1}|x_{\le t},a)=\operatorname{softmax}(W_o(h_t+\Delta h_t)+b_o).$$

Only causal prefix state h_t and externally supplied a enter this path; the
whole-sentence posterior is never fed into next-token predictions. Neutral-zero
centering isolates the affect intervention relative to that prefix. The decoder
now receives token-loss gradients as well as reconstruction gradients. Prefix
private codes differ in distribution from pooled sentence codes: this is an
experimental assumption, not a verified separation of content and emotion.

Both modes keep the same parameter layout for checkpoint compatibility and matched
initialization. The baseline's linear conditioning matrix is unused in G1.

## New Diagnostics, Not Training Losses

### D1: Fixed-Private Decoder Geometry

For fixed p, define f_p(a)=D(a,p), with a in a data-supported region U of R^3.
Its image is a candidate immersed emotional manifold, not automatically a globally
embedded or injective manifold:

$$J_p(a)=\partial f_p(a)/\partial a,\qquad G_p(a)=J_p(a)^T J_p(a).$$

G is positive definite only at full-column-rank points. We report singular values,
all metric entries, and numerical rank using `max(1e-6, 1e-4*sigma_max)` as threshold.
No positive diagonal is added to conceal rank loss. Analysis uses float64 copies
of weights, without updating checkpoints. Test cases use known affine and
rank-deficient decoders. Full rank does not establish semantic validity, absence
of self-intersections, or usefulness relative to random weights.

Repeat at the training mean private code and two empirical training private codes
at the 25th/75th percentiles of distance from that mean. Full-split affect standard
deviations are compared with training-label deviations. A ratio below 0.25 triggers
a heuristic low-spread warning, not a theorem or a criterion from the paper.

### D2: Sampled Paths and Emotional-Distance Checks

Edge cost approximates the decoded length of a straight coordinate segment:

$$\ell(a,b)\approx\sum_{r=1}^{R}
\|f_p(a+\tfrac rR(b-a))-f_p(a+\tfrac{r-1}R(b-a))\|_2.$$

Use a symmetrized k-neighbor graph in affect-coordinate space and SciPy shortest
paths with those edge costs. Graph edges use R=8 subdivisions. Displayed straight
paths use R=64. Graph paths are sample-constrained approximations, not solved
continuous geodesics, and need not beat a direct continuous path. Disconnected
pairs remain infinite and are counted, not silently repaired. Rank-deficient
regions make a Riemannian interpretation invalid even if graph distances exist.

The graph contains training and development coordinate subsamples without using
development labels to construct edges. It is a transductive diagnostic, not an
inductive prediction benchmark. Spearman correlations compare pairwise human VAD
distance with coordinate distance and graph distance on matched finite pairs.
Pairs share nodes; correlations are descriptive, not independent significance tests.
Label-neighborhood overlap is also reported. No graph setting is selected on test.

### D3: Decoder Slice and Support Visualization

Vary valence/arousal between training-posterior 5th/95th percentiles, fix dominance
at its training median, and fix private code at its training mean. This is a 2D
slice of a candidate 3D object, not the entire manifold. PCA is fitted only on
training GRU features for display; all metric/path calculations use full features.

A training-subsample nearest-neighbor threshold marks weak support (95th percentile
of leave-one-out nearest distances). Bounding boxes are not evidence of dense
support. Gray grid/crosses show weak support; path support is reported separately.
Projection can create or hide apparent intersections and curvature. A surface
plot is not evidence of successful emotional-manifold recovery.

## Named Experiments and Fair Comparisons

| Variant | Existing A1-A5 | N1 | G1 |
| --- | --- | --- | --- |
| baseline | yes | no | no |
| neighborhood | yes | yes | no |
| decoder | yes | no | yes |
| decoder_neighborhood | yes | yes | yes |

All use the same seed, official splits, train-only vocabulary, architecture sizes,
epoch budget, optimizer and base weights. Development checkpoint selection uses
the existing deterministic objective excluding N1 in every condition. Feature
reconstruction targets still evolve, so totals are not a universal quality metric;
inspect VAD errors, perplexity, collapse and geometry together. Multi-seed variation
and paired deltas are reported by `compare_manifold_variants.py`.

The runner defaults to no test evaluation. Use test only after fixing the protocol;
do not select variants or graph parameters on its results. Perplexity is conditional
on supplied VAD and includes unknown tokens, not unconditional language quality.

## What Remains Unverified

No new anti-collapse loss is added without diagnostic evidence. No guarantee of
independence, injectivity, true geodesics or emotional understanding is claimed.
Generated sentence emotion/content preservation still needs independent human
evaluation or an independently trained held-out evaluator. Decoder rank alone is
not a completion criterion. Repeated full-data runs, a random-decoder control,
and cross-dataset validation are needed before making a manifold-recovery claim.

Each new checkpoint/report records variant IDs, effective weights, seed, data hash,
source hashes, and this register's hash. Comparison runs archive the register and
the relevant source files, so later edits cannot silently change the recorded method.