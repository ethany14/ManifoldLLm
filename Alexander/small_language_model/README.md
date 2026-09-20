# Scratch-Trained Emotion Language Model

This folder contains a tiny causal GRU language model initialized with random
weights. PyTorch supplies neural-network operations,
not trained weights. Installing PyTorch requires package access; running the demo
after installation is entirely offline. Only Ethan's EmoBank CSV is read; none of
his code, Qwen embeddings, model weights, or manifold coordinates are used or changed.

## Run

Run these commands from the repository root, using a Python environment with
PyTorch installed:

```sh
python -m pip install -r Alexander/small_language_model/requirements.txt
python -m unittest Alexander.small_language_model.test_data Alexander.small_language_model.test_model Alexander.small_language_model.test_train -v
python -m Alexander.small_language_model.train --epochs 20
```

The full EmoBank CSV is the default. For a quick pipeline check:

```sh
python -m Alexander.small_language_model.train --csv Ethan/data/emobank/emobank_pilot_1000.csv --epochs 2 --output-dir Alexander/small_language_model/artifacts/pilot
```

The default output directory is `Alexander/small_language_model/artifacts/emobank`.
`model.pt` stores the best development checkpoint, architecture, vocabulary,
tokenization settings, loss weights, seed, and input CSV hash. `metrics.json`
records development history, final test metrics, a training-mean VAD baseline,
unknown-token/truncation rates, parameter count, and sample generation.
Use `--output-dir` to choose another directory. Older toy-model checkpoints are
not compatible with the new variational architecture; retrain them.

## Inputs and Outputs

The default input is `Ethan/data/emobank/emobank_manifold.csv`: 10,061 sentences
with human valence/arousal/dominance (VAD) ratings. Credit: Buechel and Hahn (2017),
EmoBank, CC BY-SA 4.0. Upstream source: https://github.com/JULIELab/EmoBank.
Ethan's preprocessing removes one missing-text record and transforms official
ratings using `(score - 3) / 2`. This loader validates [-1, 1] ratings but does
not normalize them a second time.

Official splits are retained: 8,062 train, 999 development, 1,000 test. The pilot
has 800/100/100. Duplicate IDs, missing text, invalid ratings and empty splits are
rejected. A local tokenizer lowercases text and separates words/punctuation;
only training text builds the vocabulary. Defaults keep at most 64 text tokens,
use a 4,000-token vocabulary cap including special tokens, and require frequency
at least two. Unseen words become `<unk>`; truncation and unknown rates are reported.
The language model input is `<bos>, word1, ...`; its target is `word1, ..., <eos>`.
Mini-batches are padded, and padding is excluded from loss and sentence pooling.

The model consumes token IDs and a requested VAD triple. It predicts the next-token
distribution at every position and estimates VAD from the sentence representation.
Generation begins with a start token and repeatedly selects a next token until the
end token or length limit. Only the requested emotion is supplied at generation
time, not a reference sentence or an external model's output.

## How Emotion Changes the Weights

Let h_t be the causal GRU state and a the requested VAD triple. Token probabilities
are computed as:

$$p(x_{t+1}\mid x_{\leq t},a)=\operatorname{softmax}(W_o(h_t+W_a a)+b_o).$$

The learned matrix W_a maps three emotion coordinates into the hidden state.
Changing a changes token probabilities, not the stored weights themselves.
Training updates W_a and the other weights by gradient descent.

A masked mean of GRU states feeds separate Gaussian affect (3D) and private (8D)
heads. We preserve and reuse `Alexander/manifold_losses.py`, including its
reparameterization, reconstruction, anchoring, covariance, KL and total objective.

$$q(z_a,z_p\mid h)=q(z_a\mid h)q(z_p\mid h),\qquad
z_k=\mu_k+\exp(\tfrac12\log\sigma_k^2)\odot\epsilon_k.$$

$$L_{\mathrm{self}}=\frac1B\sum_i\|\operatorname{stopgrad}(h_i)-D(z_{a,i},z_{p,i})\|_2^2.$$

$$L_{\mathrm{anchor}}=\frac{1}{B}\sum_i\|z_{a,i}-a_i\|_2^2,$$

$$C=\frac{(A-\bar A)^T(P-\bar P)}{B-1},\qquad L_{\mathrm{decorrelation}}=\|C\|_F^2,$$

$$L_{\mathrm{KL}}=\frac{1}{2B}\sum_{i,k,j}
\left(\mu_{k,ij}^2+\exp(\ell_{k,ij})-1-\ell_{k,ij}\right),
\quad \ell_k=\log\sigma_k^2,\ k\in\{a,p\}.$$

$$L=L_{\mathrm{next\ token}}+L_{\mathrm{self}}
+\lambda_{\mathrm{sup}}L_{\mathrm{anchor}}
+\lambda_{\mathrm{dec}}L_{\mathrm{decorrelation}}
+\lambda_{\mathrm{kl}}\beta L_{\mathrm{KL}}.$$

Here A and P contain sampled affect and private vectors during training. Defaults
are supervision=5, disentanglement=0.1, KL=0.01 and beta=1, configurable via CLI.
These are demonstration hyperparameters, not a claim to reproduce paper settings.
For singleton final batches covariance is skipped because it is undefined, while
all other losses still apply. Gaussian means are unbounded, unlike the old tanh
head; predicted VAD may exceed the label range. Log-variances are clamped to
[-10, 10] for numerical stability. Both codes feed the reconstruction decoder.

Adam updates the embedding, GRU, posterior heads and decoder using these losses;
language loss also updates the emotion-conditioning matrix and token head.
The sentence-level latent is never fed into the causal next-token path, avoiding
future-token leakage. Requested VAD is provided separately for conditional generation.

## Relationship to the Article

Article supplied for this project:
https://aclanthology.org/2026.acl-long.1929.pdf

This is a text-only adaptation, not a full reproduction of the paper:

| Paper component | Implementation |
| --- | --- |
| Eq. 1-4: factorized Gaussian codes and reparameterization | Separate affect/private mean and log-variance heads, stochastic sampling in training |
| Eq. 5: self-reconstruction | Decoder reconstructs the pooled scratch-GRU feature from both codes |
| Eq. 6-7: cross-modal reconstruction | Inactive: no aligned audio/video in EmoBank; logged as zero |
| Section 3.5: ground-truth VAD anchoring | Active, on sampled 3D affect code |
| Section 3.5: NRC-VAD lexicon anchoring | Inactive: no lexicon-derived pseudo-labels; logged as zero |
| Eq. 8-9: cross-covariance penalty | Active, shared/private centered sample covariance |
| Eq. 10: Gaussian KL with beta | Active for both codes; beta applied once |
| Eq. 11: weighted total | Reused for applicable terms, with added next-token loss |

The article uses pretrained RoBERTa/data2vec/TimeSformer features. Our GRU replaces
that feature extraction entirely. Because its features change during training,
we detach reconstruction targets to avoid direct target-side shrinkage; gradients
still flow through the encoder and latent codes. This is an adaptation and does
not guarantee against representation collapse. The next-token objective and
VAD-conditioned output layer are additions, not attributed to the article.

## Evaluation and Limits

Development `selection_total` (base objective excluding the optional neighborhood
term) selects the checkpoint and controls early stopping (`--patience 5`). Test
metrics are computed only after restoring that checkpoint, unless `--skip-test`
is set. The comparison runner skips test evaluation by default.
Evaluation uses posterior means, not Monte Carlo samples, so the reported total
is a deterministic proxy, not an estimated variational bound. Language losses
are weighted by non-padding token count; other losses by example count.
Covariance is a mini-batch statistic and therefore depends on batch composition.

Perplexity is conditional on each sentence's ground-truth VAD and includes unknown
tokens. Compare it only with runs sharing tokenization and conditioning settings.
VAD predictions depend only on text; requested VAD enters only the generation path.
MAE, RMSE and R-squared are reported for each VAD coordinate (R-squared is null
for constant labels). Test labels are not used to train the VAD predictor.

This is a small corpus for scratch language learning, so generation can remain
repetitive and VAD predictions can underperform the mean baseline. Decorrelation
does not prove independence, and reconstruction does not prove disentanglement.
The two-epoch pilot is a software smoke check, not a research result. Full claims
require full-data training, repeated seeds, and appropriate baseline comparisons.

## Manifold Research Variants

See [ARTICLE_DIFFERENCES.md](ARTICLE_DIFFERENCES.md) for the exact boundary between
paper mathematics, prior adaptations A1-A5, optional extensions N1/G1, and new
diagnostics D1-D3. The equations above describe the baseline, not the extensions.
All implementations remain scratch-trained and offline, with no existing LLMs.

| `--variant` | Generation conditioning | Additional training objective |
| --- | --- | --- |
| `baseline` (default) | Linear VAD offset | None |
| `neighborhood` | Linear VAD offset | Label-neighbor distance loss (N1) |
| `decoder` | Decoder difference using causal prefix private code (G1) | None |
| `decoder_neighborhood` | Decoder difference (G1) | N1 |

For N1, `--neighborhood-weight` defaults to 0.1. The original paper-derived losses
stay active. Version-2 checkpoints without `conditioning_mode` load as baseline;
new checkpoints store it explicitly. Use separate output folders for each variant.

```sh
python -m pip install -r Alexander/small_language_model/visualization/requirements.txt
python -m Alexander.small_language_model.train --variant decoder_neighborhood --epochs 20 --skip-test --output-dir Alexander/small_language_model/artifacts/decoder_neighborhood
python -m Alexander.small_language_model.visualization.diagnose_geometry --checkpoint Alexander/small_language_model/artifacts/decoder_neighborhood/model.pt --output-dir Alexander/small_language_model/manifold_results/decoder_neighborhood
```

The diagnostics compute the full decoder Jacobian and metric, rank sensitivity to
private codes, emotional spread and neighborhood agreement, and approximate graph
path distances. Images show a decoder-generated fixed-dominance slice projected
onto training-only PCA axes, sensitivity, and paths. Weak sampled-data support is
marked; neither a curved surface nor full rank establishes emotional validity.

Matched four-variant experiment, full dataset and repeated seeds:

```sh
python -m Alexander.small_language_model.compare_manifold_variants --epochs 20 --seeds 1 7 42 --run-dir Alexander/small_language_model/artifacts/manifold_full --report-dir Alexander/small_language_model/manifold_results/full
```

For a quick check, add `--csv Ethan/data/emobank/emobank_pilot_1000.csv --epochs 2`.
The runner refuses to overwrite existing experiment directories. It archives
source files and the differences register, saves checkpoint/data hashes and
effective weights, and produces per-seed CSV comparisons plus mean/SD and paired
baseline deltas in JSON. A one-seed SD is null, not an estimate of stability.
Test stays unevaluated unless `--evaluate-test` is explicitly supplied; do not use
it for tuning. This command can be computationally expensive on full data.

The included [pilot comparison](manifold_results/pilot/comparison.csv) is a
two-epoch, one-seed development smoke check. All four decoders had full local rank
at sampled points but low affect spread and weak emotional-distance correlations.
It does not demonstrate improved emotional geometry. Full-data multi-seed training,
random-decoder controls and independent generation evaluation remain to be run.

Tests after installing the visualization dependencies:

```sh
python -m unittest Alexander.small_language_model.test_data Alexander.small_language_model.test_model Alexander.small_language_model.test_train Alexander.small_language_model.test_geometry Alexander.small_language_model.test_experiments Alexander.small_language_model.visualization.test_plot_manifold -v
```

## Reload the Learned Model

```python
import torch
from Alexander.small_language_model.model import EmotionLanguageModel, generate

checkpoint = torch.load(
    "Alexander/small_language_model/artifacts/emobank/model.pt",
    map_location="cpu", weights_only=True,
)
model = EmotionLanguageModel(**checkpoint["config"])
model.load_state_dict(checkpoint["state_dict"])
print(generate(model, checkpoint["vocabulary"], [0.8, 0.6, 0.5]))
```