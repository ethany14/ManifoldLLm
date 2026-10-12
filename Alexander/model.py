"""Affective-manifold model built on the shared scratch-trained language backbone."""

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from small_language_model.core import GRUBackbone

from .geometry import decode_affect, neighborhood_loss
from .manifold_losses import (
    LossWeights,
    disentanglement_loss,
    gaussian_kl,
    reparameterize,
    squared_error,
    total_objective,
)


class EmotionLanguageModel(GRUBackbone):
    # Reading order: encode text -> infer coordinates -> decode features -> score losses.
    # The 3D affect space is a design choice aligned with VAD, not an estimated
    # intrinsic dimension of human emotion. See geometry.decoder_geometry for
    # the local-rank test that turns a decoder parameterization into candidate geometry.
    def __init__(self, vocabulary_size: int, embedding_dim: int = 32,
                 hidden_dim: int = 64, private_dim: int = 8,
                 conditioning_mode: str = "linear"):
        if conditioning_mode not in ("linear", "decoder"):
            raise ValueError("conditioning_mode must be linear or decoder")
        super().__init__(vocabulary_size, embedding_dim, hidden_dim)
        self.conditioning_mode = conditioning_mode
        # Two diagonal Gaussian posteriors divide the sentence representation into
        # affect (valence, arousal, dominance) and a private residual code. Separate
        # heads impose a factorized posterior, not statistical independence in data.
        self.affect_head = nn.Linear(hidden_dim, 3)
        self.private_head = nn.Linear(hidden_dim, private_dim)
        self.affect_log_variance = nn.Linear(hidden_dim, 3)
        self.private_log_variance = nn.Linear(hidden_dim, private_dim)
        # D(a, p) maps 3 + private_dim coordinates into hidden_dim features (64 by
        # default). Holding p fixed gives f_p(a) = D(a, p), our candidate affect
        # manifold parameterization. GELU permits a smooth nonlinear map, but the
        # learned map may be nearly flat, rank-deficient, or self-intersecting.
        # A 3D image is locally immersed only where df_p/da has rank 3; reconstruction
        # alone does not guarantee that condition or meaningful emotion geometry.
        self.decoder = nn.Sequential(
            nn.Linear(3 + private_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.emotion_condition = nn.Linear(3, hidden_dim, bias=False)
        self.token_head = nn.Linear(hidden_dim, vocabulary_size)

    def forward(self, tokens: Tensor, desired_vad: Tensor) -> dict[str, Tensor]:
        # 1. The shared GRU produces h_t from tokens up to time t, so language
        # prediction remains causal. hidden has shape [batch, time, hidden_dim].
        hidden = self.encode_tokens(tokens)
        # 2. Average non-padding token states to obtain one feature vector per
        # sentence. This pooled vector sees the whole sentence and is used only
        # for sentence-level posteriors/reconstruction, never to predict its tokens.
        mask = tokens.ne(0).unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        if self.conditioning_mode == "decoder":
            # 3a. Extension G1: D(requested_a, p_t) - D(0, p_t) changes the token
            # state relative to zero-VAD conditioning while holding prefix code p_t
            # fixed. p_t uses only h_t, avoiding future-token leakage. This routes
            # language gradients through the decoder; it is not a geodesic step.
            prefix_private = self.private_head(hidden)
            requested = desired_vad[:, None, :].expand(-1, hidden.shape[1], -1)
            offset = decode_affect(self.decoder, requested, prefix_private)
            offset = offset - decode_affect(self.decoder, torch.zeros_like(requested), prefix_private)
            conditioned = hidden + offset
        else:
            # 3b. Baseline: h_t + W_a * requested_a. This linear conditioning path
            # controls generation without using the reconstruction decoder.
            conditioned = hidden + self.emotion_condition(desired_vad).unsqueeze(1)
        # 4. Infer mu_a, mu_p and log(sigma^2) from text, not from requested VAD.
        # Thus changing the requested generation emotion cannot change the inferred
        # sentence affect. Clamping log-variance limits extreme sampling scales;
        # posterior means themselves are unbounded, even though labels are [-1, 1].
        affect_mean = self.affect_head(pooled)
        private_mean = self.private_head(pooled)
        affect_log_variance = self.affect_log_variance(pooled).clamp(-10, 10)
        private_log_variance = self.private_log_variance(pooled).clamp(-10, 10)
        # 5. z = mu + sigma * epsilon samples nearby latent codes in training while
        # preserving gradients to both posterior heads. Evaluation uses mu directly.
        # Sampling encourages training around codes, but does not ensure coverage
        # of all coordinates or rule out posterior collapse.
        affect_sample = reparameterize(affect_mean, affect_log_variance, self.training)
        private_sample = reparameterize(private_mean, private_log_variance, self.training)
        # 6. Decode both codes to reconstruct the sentence feature. Detaching the
        # target blocks the direct incentive to move that target toward the decoder;
        # gradients still reach the encoder through the inferred latent codes.
        # logits are unnormalized token scores; cross-entropy applies log-softmax.
        return {
            "logits": self.token_head(conditioned),
            "affect": affect_mean,
            "private": private_mean,
            "affect_sample": affect_sample,
            "private_sample": private_sample,
            "affect_log_variance": affect_log_variance,
            "private_log_variance": private_log_variance,
            "reconstruction": self.decoder(torch.cat([affect_sample, private_sample], dim=1)),
            "features": pooled.detach(),
        }


def training_loss(outputs: dict[str, Tensor], targets: Tensor,
                  vad: Tensor, weights: LossWeights | None = None,
                  neighborhood_weight: float = 0.0) -> dict[str, Tensor]:
    # 7. Preserve language information: mean negative log-probability of the true
    # next token, excluding padding ID 0. This is our added language objective,
    # not a paper-derived manifold loss.
    language = functional.cross_entropy(
        outputs["logits"].reshape(-1, outputs["logits"].shape[-1]),
        targets.reshape(-1), ignore_index=0,
    )
    weights = weights if weights is not None else LossWeights(cross=0.0, lexicon=0.0)
    zero = language.new_zeros(())
    # 8. Anchor affect axes to human VAD: mean ||z_a - y||^2. This fixes their
    # intended meaning/scale instead of allowing arbitrary latent rotations.
    # During sampling, expected squared error includes both mean error and variance;
    # its influence competes with the KL prior rather than perfectly enforcing labels.
    anchor = squared_error(outputs["affect_sample"], vad)
    # 9. Penalize batch cross-covariance between affect and private coordinates.
    # This discourages linear redundancy, not all dependence or semantic leakage.
    # A single example cannot define unbiased sample covariance, so skip it.
    decorrelation = (
        disentanglement_loss(outputs["affect_sample"], outputs["private_sample"])
        if len(vad) > 1 else zero
    )
    # 10. Make the decoder represent the encoder's features, so the coordinates
    # describe something about the input. The private code can still carry most
    # information; a low reconstruction error does not prove affect is being used.
    reconstruction = squared_error(outputs["reconstruction"], outputs["features"])
    # 11. Regularize both Gaussian posteriors toward N(0, I). This discourages
    # arbitrary mean/variance growth, but excessive KL can erase input information.
    kl = gaussian_kl(outputs["affect"], outputs["affect_log_variance"])
    kl = kl + gaussian_kl(outputs["private"], outputs["private_log_variance"])
    # Cross-modal reconstruction and lexicon supervision remain zero because this
    # text-only run has neither paired modalities nor lexicon-derived targets.
    terms = {
        "self": reconstruction, "cross": zero, "ground_truth": anchor,
        "lexicon": zero, "disentanglement": decorrelation, "kl": kl,
    }
    # 12. Optional extension N1: preserve human-label neighbor distances in the
    # posterior means. This shapes coordinate geometry, not decoder geodesics.
    local = neighborhood_loss(outputs["affect"], vad) if neighborhood_weight else zero
    # Optimizer steps use gradients of this weighted sum. No term explicitly
    # enforces full decoder rank, global injectivity, or a recovered emotion manifold.
    return {
        "total": language + total_objective(terms, weights) + neighborhood_weight * local,
        "neighborhood": local,
        "language": language,
        "anchor": anchor,
        "decorrelation": decorrelation,
        "reconstruction": reconstruction,
        "kl": kl,
        "cross": zero,
        "lexicon": zero,
    }


@torch.inference_mode()
def generate(model: EmotionLanguageModel, vocabulary: list[str],
             desired_vad: list[float], max_tokens: int = 16) -> str:
    model.eval()
    device = next(model.parameters()).device
    tokens = [vocabulary.index("<bos>")]
    condition = torch.tensor([desired_vad], dtype=torch.float32, device=device)
    for _ in range(max_tokens):
        result = model(torch.tensor([tokens], device=device), condition)
        scores = result["logits"][0, -1].clone()
        scores[:3] = -torch.inf
        next_token = int(scores.argmax())
        if next_token == vocabulary.index("<eos>"):
            break
        tokens.append(next_token)
    return " ".join(vocabulary[index] for index in tokens[1:])