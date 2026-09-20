"""Tiny causal GRU language model; no pretrained components or network access."""

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from .geometry import decode_affect, neighborhood_loss
from Alexander.manifold_losses import (
    LossWeights,
    disentanglement_loss,
    gaussian_kl,
    reparameterize,
    squared_error,
    total_objective,
)


class EmotionLanguageModel(nn.Module):
    def __init__(self, vocabulary_size: int, embedding_dim: int = 32,
                 hidden_dim: int = 64, private_dim: int = 8,
                 conditioning_mode: str = "linear"):
        super().__init__()
        if conditioning_mode not in ("linear", "decoder"):
            raise ValueError("conditioning_mode must be linear or decoder")
        self.conditioning_mode = conditioning_mode
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.recurrent = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.affect_head = nn.Linear(hidden_dim, 3)
        self.private_head = nn.Linear(hidden_dim, private_dim)
        self.affect_log_variance = nn.Linear(hidden_dim, 3)
        self.private_log_variance = nn.Linear(hidden_dim, private_dim)
        self.decoder = nn.Sequential(
            nn.Linear(3 + private_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.emotion_condition = nn.Linear(3, hidden_dim, bias=False)
        self.token_head = nn.Linear(hidden_dim, vocabulary_size)

    def forward(self, tokens: Tensor, desired_vad: Tensor) -> dict[str, Tensor]:
        hidden, _ = self.recurrent(self.embedding(tokens))
        mask = tokens.ne(0).unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        if self.conditioning_mode == "decoder":
            prefix_private = self.private_head(hidden)
            requested = desired_vad[:, None, :].expand(-1, hidden.shape[1], -1)
            offset = decode_affect(self.decoder, requested, prefix_private)
            offset = offset - decode_affect(self.decoder, torch.zeros_like(requested), prefix_private)
            conditioned = hidden + offset
        else:
            conditioned = hidden + self.emotion_condition(desired_vad).unsqueeze(1)
        affect_mean = self.affect_head(pooled)
        private_mean = self.private_head(pooled)
        affect_log_variance = self.affect_log_variance(pooled).clamp(-10, 10)
        private_log_variance = self.private_log_variance(pooled).clamp(-10, 10)
        affect_sample = reparameterize(affect_mean, affect_log_variance, self.training)
        private_sample = reparameterize(private_mean, private_log_variance, self.training)
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
    language = functional.cross_entropy(
        outputs["logits"].reshape(-1, outputs["logits"].shape[-1]),
        targets.reshape(-1), ignore_index=0,
    )
    weights = weights if weights is not None else LossWeights(cross=0.0, lexicon=0.0)
    zero = language.new_zeros(())
    anchor = squared_error(outputs["affect_sample"], vad)
    decorrelation = (
        disentanglement_loss(outputs["affect_sample"], outputs["private_sample"])
        if len(vad) > 1 else zero
    )
    reconstruction = squared_error(outputs["reconstruction"], outputs["features"])
    kl = gaussian_kl(outputs["affect"], outputs["affect_log_variance"])
    kl = kl + gaussian_kl(outputs["private"], outputs["private_log_variance"])
    terms = {
        "self": reconstruction, "cross": zero, "ground_truth": anchor,
        "lexicon": zero, "disentanglement": decorrelation, "kl": kl,
    }
    local = neighborhood_loss(outputs["affect"], vad) if neighborhood_weight else zero
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